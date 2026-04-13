"""
Main extraction pipeline: Pass 1 (fast scan) + Pass 2 (team-aware extraction).
"""

import cv2
import numpy as np
from tqdm import tqdm

from .helpers import (
    compute_player_sharpness, compute_pitch_green_ratio,
    smart_pitch_filter, filter_foreground_players, detect_persons,
    get_shot_type, fmt_timestamp,
)
from .calibration import extract_torso_crop, classify_person


# ── Default parameters ────────────────────────────────────────────────
# Each param has a comment explaining:
#   What it controls → Effect of ↑ (raise) / ↓ (lower)
DEFAULT_PARAMS = {
    # YOLO confidence threshold for person detection.
    # ↑ = fewer detections, only high-confidence → misses distant players
    # ↓ = more detections, includes uncertain → more false positives
    "PERSON_CONFIDENCE": 0.45,

    # Minimum number of persons in frame to keep it.
    # ↑ = only group shots → fewer frames
    # ↓ = accepts solo shots → more frames
    "MIN_PERSONS": 1,

    # Minimum (largest person bbox area / frame area) to consider frame.
    # Controls how "zoomed in" the camera must be.
    # ↑ = only extreme close-ups → fewer frames, higher logo detail
    # ↓ = accepts wider shots → more frames, smaller players
    "MIN_MAX_PERSON_AREA_RATIO": 0.015,

    # Sharpness threshold (normalized 0-1). Frames below this are flagged
    # as "motion blur" but still kept as candidates.
    # ↑ = stricter quality → only razor-sharp frames
    # ↓ = more tolerant → keeps slightly soft frames
    "MIN_SHARPNESS": 0.15,

    # Hard blur cutoff — frames below this are SKIPPED entirely.
    # This is the main "junk filter" for motion blur.
    # ↑ = removes more blurry frames → fewer candidates
    # ↓ = keeps more blurry frames → more candidates but noisier
    "MOTION_BLUR_SHARPNESS_MIN": 0.06,

    # Whether to check for green pitch in frame.
    # True = skip frames without visible grass (crowd shots, graphics)
    # False = disable pitch check entirely → more frames
    "ENABLE_PITCH_GREEN_FILTER": True,

    # Vertical position (0-1) to start looking for green pitch.
    # 0.40 = check bottom 60% of frame for grass.
    # ↑ = smaller ROI → less strict pitch check
    # ↓ = larger ROI → stricter pitch check
    "PITCH_ROI_Y_START": 0.40,

    # Minimum green ratio in pitch ROI to consider "on pitch".
    # ↑ = requires more visible grass → skips close-ups with little grass
    # ↓ = accepts less grass → keeps more close-ups (usually better for logos)
    "MIN_PITCH_GREEN_RATIO": 0.04,

    # Pass 1: check every Nth frame for zoom level.
    # ↑ = faster scan but may miss short zoom-ins
    # ↓ = slower scan but catches brief close-ups
    "SCAN_INTERVAL": 5,

    # Minimum consecutive quality frames to form a "segment".
    # ↑ = only sustained zoom-ins → fewer, longer segments
    # ↓ = catches brief zoom-ins too → more segments
    "MIN_SEGMENT_FRAMES": 2,

    # How many consecutive "bad" frames allowed before ending a segment.
    # ↑ = merges nearby segments → longer segments
    # ↓ = splits segments at any gap → more, shorter segments
    "SEGMENT_GAP_TOLERANCE": 3,
}


def pass1_fast_scan(video_path, yolo_model, device, params=None,
                    max_frames=None):
    """
    Pass 1: Fast scan to build zoom timeline and find quality segments.
    
    Scans every Nth frame, runs YOLO person detection, records
    max_person_area_ratio to identify when camera is zoomed into players.
    
    Returns: (segments, zoom_timeline, video_info)
    """
    p = {**DEFAULT_PARAMS, **(params or {})}

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_area = w * h

    scan_limit = max_frames if max_frames else total
    interval = p["SCAN_INTERVAL"]

    video_info = {"fps": fps, "total_frames": total, "width": w,
                  "height": h, "frame_area": frame_area}

    zoom_timeline = []
    batch_frames, batch_nums = [], []
    batch_size = 32

    pbar = tqdm(total=scan_limit, desc="Pass 1: Scanning zoom levels", unit="fr")
    frame_num = 0

    while True:
        ret, frame = cap.read()
        if not ret or frame_num >= scan_limit:
            break
        pbar.update(1)

        if frame_num % interval != 0:
            frame_num += 1
            continue

        batch_frames.append(frame)
        batch_nums.append(frame_num)

        if len(batch_frames) >= batch_size:
            _process_scan_batch(yolo_model, batch_frames, batch_nums,
                                zoom_timeline, frame_area, device, p)
            batch_frames, batch_nums = [], []

        frame_num += 1

    if batch_frames:
        _process_scan_batch(yolo_model, batch_frames, batch_nums,
                            zoom_timeline, frame_area, device, p)
    pbar.close()
    cap.release()

    # Find quality segments
    segments = _find_segments(zoom_timeline, p)

    return segments, zoom_timeline, video_info


def _process_scan_batch(model, frames, frame_nums, timeline,
                        frame_area, device, params):
    """Process a batch of frames for Pass 1 scan."""
    results = model.predict(frames, classes=[0], conf=params["PERSON_CONFIDENCE"],
                            device=device, verbose=False)
    for fn, res in zip(frame_nums, results):
        n_persons = len(res.boxes) if res.boxes is not None else 0
        max_ratio = 0.0
        if n_persons > 0:
            areas = []
            for box in res.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                areas.append((x2 - x1) * (y2 - y1) / frame_area)
            max_ratio = max(areas)
        timeline.append((fn, max_ratio, n_persons))


def _find_segments(zoom_timeline, params):
    """Find contiguous quality segments from zoom timeline."""
    min_ratio = params["MIN_MAX_PERSON_AREA_RATIO"]
    min_persons = params["MIN_PERSONS"]
    min_seg = params["MIN_SEGMENT_FRAMES"]

    segments = []
    current = []

    for i, (fn, ratio, n_p) in enumerate(zoom_timeline):
        if ratio >= min_ratio and n_p >= min_persons:
            current.append((fn, ratio, n_p))
        else:
            if current and i + 1 < len(zoom_timeline):
                next_fn, next_r, next_n = zoom_timeline[min(i + 1, len(zoom_timeline) - 1)]
                if next_r >= min_ratio and len(current) >= min_seg:
                    continue
            if len(current) >= min_seg:
                segments.append(current)
            current = []

    if len(current) >= min_seg:
        segments.append(current)

    return segments


def pass2_extract(video_path, segments, yolo_model, device,
                  calibration, overlay_mask, video_info, params=None,
                  max_frames=None):
    """
    Pass 2: Team-aware extraction from quality segments.
    
    For each segment:
    1. Read all frames
    2. Smart pitch filter (bypass for close-ups)
    3. YOLO detect + foreground filter
    4. Team classification via calibration
    5. Overlay-masked sharpness
    6. Team-aware scoring + categorization
    
    Returns: (candidates, stats)
    """
    p = {**DEFAULT_PARAMS, **(params or {})}
    fps = video_info["fps"]
    frame_area = video_info["frame_area"]
    frame_h = video_info["height"]
    frame_w = video_info["width"]
    total = video_info["total_frames"]
    scan_interval = p["SCAN_INTERVAL"]

    cap = cv2.VideoCapture(str(video_path))
    candidates = []
    stats = {
        "segments_processed": 0,
        "frames_analyzed": 0,
        "skipped_pitch": 0,
        "skipped_blurry": 0,
        "skipped_no_person": 0,
        "team_target": 0,
        "team_opponent": 0,
        "team_mixed": 0,
        "team_ambiguous": 0,
    }

    for seg_idx, segment in enumerate(tqdm(segments, desc="Pass 2: Team-aware extraction")):
        seg_start = segment[0][0]
        seg_end = segment[-1][0] + scan_interval
        seg_duration = (seg_end - seg_start) / fps

        cap.set(cv2.CAP_PROP_POS_FRAMES, seg_start)
        seg_candidates = []

        for fn in range(seg_start, min(seg_end, total)):
            ret, frame = cap.read()
            if not ret:
                break
            stats["frames_analyzed"] += 1

            # ── Pitch green (soft signal, NOT a hard filter) ──
            pitch_green = compute_pitch_green_ratio(frame, p["PITCH_ROI_Y_START"])

            # ── Person detection ──
            detections = detect_persons(yolo_model, frame, device,
                                        p["PERSON_CONFIDENCE"])

            if len(detections) < p["MIN_PERSONS"]:
                # Check pitch mask. No players + No green (e.g. locker room, sky) -> skip
                if p["ENABLE_PITCH_GREEN_FILTER"] and pitch_green < p["MIN_PITCH_GREEN_RATIO"]:
                    stats["skipped_pitch"] += 1
                    continue
                
                # Retrieve background image context (Null frame)
                sharpness = compute_player_sharpness(frame, [], overlay_mask)
                if sharpness < p["MOTION_BLUR_SHARPNESS_MIN"]:
                    stats["skipped_blurry"] += 1
                    continue
                    
                timestamp_sec = fn / fps
                seg_candidates.append({
                    "frame_num": fn,
                    "timestamp_sec": round(timestamp_sec, 2),
                    "timestamp_hms": fmt_timestamp(timestamp_sec),
                    "n_persons": 0,
                    "n_target": 0,
                    "n_opponent": 0,
                    "n_ambiguous": 0,
                    "team_dominance": 0.0,
                    "category": "background",
                    "shot_type": "wide",
                    "sharpness": round(sharpness, 4),
                    "is_motion_blur": sharpness < p["MIN_SHARPNESS"],
                    "max_person_area_ratio": 0.0,
                    "max_target_area_ratio": 0.0,
                    "target_coverage": 0.0,
                    "pitch_green_ratio": round(pitch_green, 4),
                    "score": round(sharpness * 0.3 + pitch_green * 0.2, 4),
                    "segment_idx": seg_idx,
                    "segment_duration": round(seg_duration, 1),
                })
                # Track as processed background frame, do not skip
                continue

            # ── Foreground filter (remove crowd) ──
            fg_detections = filter_foreground_players(detections, frame_h, frame_w)

            max_person_ratio = max(d["area"] / frame_area for d in fg_detections)
            if max_person_ratio < p["MIN_MAX_PERSON_AREA_RATIO"]:
                continue

            # ── Team classification ──
            n_target, n_opponent, n_ambiguous = 0, 0, 0
            for det in fg_detections:
                torso, status, torso_ov_mask = extract_torso_crop(
                    frame, det["bbox"], overlay_mask)
                if status == "ok" and torso is not None:
                    team, conf = classify_person(torso, calibration, torso_ov_mask)
                    det["team"] = team
                    det["team_conf"] = conf
                else:
                    det["team"] = "ambiguous"
                    det["team_conf"] = 0.0

                if det["team"] == "target":
                    n_target += 1
                elif det["team"] == "opponent":
                    n_opponent += 1
                else:
                    n_ambiguous += 1

            # ── Categorize frame ──
            n_total = n_target + n_opponent
            category, shot_type = _categorize_frame(
                n_target, n_opponent, n_ambiguous, max_person_ratio
            )

            # ── Soft pitch filter: skip only no-pitch + no-target frames ──
            # Frames with target players are ALWAYS kept regardless of pitch
            if p["ENABLE_PITCH_GREEN_FILTER"]:
                has_target = n_target > 0
                has_large_person = max_person_ratio > 0.06
                if (not has_target and not has_large_person
                        and pitch_green < p["MIN_PITCH_GREEN_RATIO"]):
                    stats["skipped_pitch"] += 1
                    continue

            # ── Sharpness (overlay-masked, prefer target team) ──
            target_dets = [d for d in fg_detections if d.get("team") == "target"]
            sharpness_dets = target_dets if target_dets else fg_detections
            sharpness = compute_player_sharpness(frame, sharpness_dets, overlay_mask)

            if sharpness < p["MOTION_BLUR_SHARPNESS_MIN"]:
                stats["skipped_blurry"] += 1
                continue

            is_motion_blur = sharpness < p["MIN_SHARPNESS"]

            # ── Team-aware scoring (pitch_green as soft bonus) ──
            score = _compute_team_score(
                sharpness, fg_detections, frame_area, category,
                n_target, n_opponent, max_person_ratio,
                pitch_green
            )

            # ── Target area metrics ──
            max_target_ratio = 0.0
            target_coverage = 0.0
            if target_dets:
                max_target_ratio = max(d["area"] / frame_area for d in target_dets)
                target_coverage = sum(d["area"] for d in target_dets) / frame_area

            team_dominance = n_target / max(n_total, 1)

            timestamp_sec = fn / fps
            seg_candidates.append({
                "frame_num": fn,
                "timestamp_sec": round(timestamp_sec, 2),
                "timestamp_hms": fmt_timestamp(timestamp_sec),
                "n_persons": len(fg_detections),
                "n_target": n_target,
                "n_opponent": n_opponent,
                "n_ambiguous": n_ambiguous,
                "team_dominance": round(team_dominance, 3),
                "category": category,
                "shot_type": shot_type,
                "sharpness": sharpness,
                "is_motion_blur": is_motion_blur,
                "max_person_area_ratio": round(max_person_ratio, 4),
                "max_target_area_ratio": round(max_target_ratio, 4),
                "target_coverage": round(target_coverage, 4),
                "pitch_green_ratio": round(pitch_green, 4),
                "score": round(score, 4),
                "segment_idx": seg_idx,
                "segment_duration": round(seg_duration, 1),
            })

        stats["segments_processed"] += 1

        # Add ALL qualifying frames from this segment to candidate pool.
        # Quota selection (downstream) will handle final count + diversity.
        candidates.extend(seg_candidates)

    cap.release()

    # Update stats
    stats["team_background"] = 0
    for c in candidates:
        cat = c["category"]
        if "target" in cat:
            stats["team_target"] += 1
        elif "opponent" in cat:
            stats["team_opponent"] += 1
        elif cat == "mixed":
            stats["team_mixed"] += 1
        elif cat == "background":
            stats["team_background"] += 1
        else:
            stats["team_ambiguous"] += 1

    candidates.sort(key=lambda x: x["timestamp_sec"])
    return candidates, stats


def _categorize_frame(n_target, n_opponent, n_ambiguous, max_person_ratio):
    """Categorize frame by team composition + shot type."""
    n_total = n_target + n_opponent
    shot_type = get_shot_type(max_person_ratio)

    if n_total == 0:
        return f"ambiguous_{shot_type}", shot_type

    target_ratio = n_target / n_total

    if target_ratio >= 0.5:
        return f"target_{shot_type}", shot_type
    elif target_ratio > 0:
        return "mixed", shot_type
    else:
        return f"opponent_{shot_type}", shot_type


def _compute_team_score(sharpness, detections, frame_area, category,
                        n_target, n_opponent, max_person_ratio,
                        pitch_green=0.0):
    """Compute team-aware quality score for frame selection.

    pitch_green acts as a soft bonus (0.05 weight) — frames with visible
    pitch get a small boost, but frames without pitch are NOT penalized
    if they have strong target player presence.
    """
    target_dets = [d for d in detections if d.get("team") == "target"]
    max_target_ratio = (max(d["area"] / frame_area for d in target_dets)
                        if target_dets else 0.0)

    # Soft pitch bonus: 0 to 0.05
    pitch_bonus = min(pitch_green / 0.10, 1.0) * 0.05

    if "target" in category:
        score = (
            sharpness * 0.25 +
            max_target_ratio * 0.25 +
            min(n_target / 4, 1.0) * 0.20 +
            max_person_ratio * 0.15 +
            0.10 +  # category bonus
            pitch_bonus
        )
    elif category == "mixed":
        score = (
            sharpness * 0.30 +
            max_person_ratio * 0.25 +
            min((n_target + n_opponent) / 5, 1.0) * 0.15 +
            (n_target / max(n_target + n_opponent, 1)) * 0.15 +
            0.05 +
            pitch_bonus
        )
    else:  # opponent or ambiguous
        score = (
            sharpness * 0.40 +
            max_person_ratio * 0.30 +
            min(len(detections) / 5, 1.0) * 0.15 +
            pitch_bonus
        )

    return round(score, 4)
