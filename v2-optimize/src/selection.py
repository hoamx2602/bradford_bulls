"""
Bradford Bulls v2 — Quota-based frame selection with tier-aware prioritization.

Key improvement over v1:
  - Candidates are sorted by sharpness tier (gold > silver > bronze) FIRST,
    then by score within each tier.
  - This ensures the training dataset is filled with the sharpest frames first.
"""

import numpy as np
from . import config


TIER_PRIORITY = {"gold": 0, "silver": 1, "bronze": 2}


def auto_target_frames(candidates, video_duration_sec,
                       max_candidate_ratio=0.8,
                       floor=100, ceiling=600):
    """
    Automatically compute how many frames to select.
    Non-linear scaling: short videos get higher density.
    """
    duration_min = video_duration_sec / 60.0

    if duration_min <= 15:
        frames_per_minute = 15
    elif duration_min <= 45:
        frames_per_minute = 8
    else:
        frames_per_minute = 5

    base = int(duration_min * frames_per_minute)
    pool_limit = int(len(candidates) * max_candidate_ratio)

    capped = min(base, pool_limit)
    result = max(capped, min(floor, pool_limit))
    result = min(result, ceiling)

    print(f"  Auto TARGET_FRAMES: {result}")
    print(f"    Video: {duration_min:.1f} min → Density: {frames_per_minute}/min → Base: {base}")
    print(f"    Candidates: {len(candidates)} (Cap {max_candidate_ratio*100:.0f}% = {pool_limit})")
    print(f"    Bounds: [{min(floor, len(candidates))}, {ceiling}]")

    return result


def select_by_quota(candidates, total_target, quota=None, min_time_gap=None):
    """
    Select frames using quota-based allocation with tier prioritization.

    Candidates are sorted by sharpness tier (gold first) then by score.
    This ensures the training dataset gets the sharpest frames possible.

    Args:
        candidates: List of candidate dicts from Pass 2
        total_target: Total number of frames to select
        quota: Dict mapping category → fraction
        min_time_gap: Minimum seconds between selected frames

    Returns:
        selected: List of selected candidate dicts
        selection_stats: Dict with per-category counts
    """
    if not candidates:
        return [], {}

    if min_time_gap is None:
        min_time_gap = config.MIN_TIME_GAP_SEC

    q = quota or config.DEFAULT_QUOTA

    # Group candidates by category
    by_category = {}
    for c in candidates:
        cat = c["category"]
        quota_key = _map_to_quota_key(cat)
        by_category.setdefault(quota_key, []).append(c)

    # Sort each group: tier first (gold > silver > bronze), then score
    for key in by_category:
        by_category[key].sort(key=lambda x: (
            TIER_PRIORITY.get(x.get("sharpness_tier", "bronze"), 2),
            -x["score"]
        ))

    # Allocate frames per category
    selected = []
    selection_stats = {}

    for cat_key, fraction in sorted(q.items(), key=lambda x: -x[1]):
        n_want = max(1, int(total_target * fraction))
        pool = by_category.get(cat_key, [])

        picked = _pick_with_diversity(pool, n_want, min_time_gap)
        selected.extend(picked)
        selection_stats[cat_key] = len(picked)

    # Fill remaining from highest-tier, highest-scored candidates
    if len(selected) < total_target:
        selected_nums = {c["frame_num"] for c in selected}
        remaining = [c for c in candidates if c["frame_num"] not in selected_nums]
        remaining.sort(key=lambda x: (
            TIER_PRIORITY.get(x.get("sharpness_tier", "bronze"), 2),
            -x["score"]
        ))

        for c in remaining:
            if len(selected) >= total_target:
                break
            if all(abs(c["timestamp_sec"] - s["timestamp_sec"]) >= min_time_gap / 2
                   for s in selected):
                selected.append(c)

    selected.sort(key=lambda x: x["timestamp_sec"])
    return selected, selection_stats


def _map_to_quota_key(category):
    """Map detailed category names to quota keys."""
    if category.startswith("ambiguous"):
        return "mixed"
    return category


def _pick_with_diversity(pool, n_want, min_gap_sec):
    """Pick up to n_want frames from pool, ensuring temporal diversity."""
    if not pool:
        return []

    picked = []
    for candidate in pool:
        if len(picked) >= n_want:
            break
        ts = candidate["timestamp_sec"]
        if all(abs(ts - p["timestamp_sec"]) >= min_gap_sec for p in picked):
            picked.append(candidate)

    return picked


def print_selection_summary(selected, selection_stats, total_candidates):
    """Print a summary of the quota selection results."""
    print(f"\n{'=' * 60}")
    print(f"  SELECTION RESULTS (v2 — Torso Sharpness + Tiers)")
    print(f"{'=' * 60}")
    print(f"  Total candidates:  {total_candidates}")
    print(f"  Selected frames:   {len(selected)}")

    # Tier breakdown
    tier_counts = {"gold": 0, "silver": 0, "bronze": 0}
    for c in selected:
        tier = c.get("sharpness_tier", "bronze")
        tier_counts[tier] = tier_counts.get(tier, 0) + 1

    print(f"\n  Sharpness tier breakdown:")
    tier_icons = {"gold": "🥇", "silver": "🥈", "bronze": "🥉"}
    for tier in ["gold", "silver", "bronze"]:
        count = tier_counts.get(tier, 0)
        pct = count / max(len(selected), 1) * 100
        print(f"    {tier_icons.get(tier, '')} {tier:10s}  {count:4d}  ({pct:5.1f}%)")

    print(f"\n  Per-category breakdown:")
    for cat, count in sorted(selection_stats.items()):
        pct = count / max(len(selected), 1) * 100
        bar = "█" * int(pct / 2)
        print(f"    {cat:25s}  {count:4d}  ({pct:5.1f}%)  {bar}")

    if selected:
        scores = [c["score"] for c in selected]
        sharp = [c.get("torso_sharpness", c["sharpness"]) for c in selected]
        print(f"\n  Score range:           {min(scores):.3f} — {max(scores):.3f}")
        print(f"  Torso sharpness range: {min(sharp):.3f} — {max(sharp):.3f}")
        print(f"  Time span:             {selected[0]['timestamp_hms']} — "
              f"{selected[-1]['timestamp_hms']}")
    print(f"{'=' * 60}")
