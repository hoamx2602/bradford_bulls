"""
Streamlit reviewer UI.

Annotator workflow:
  1. Launch with package directory as argument:
       streamlit run src/track_annotation/reviewer/app.py -- --package PATH

  2. For each track:
     - Inspect 3 keyframes (full + crop) side-by-side
     - Watch 2-second video clip
     - Reference 21 brand templates
     - Click brand button OR mark "skip" / "not visible"
     - (Optional) set position + visibility_quality

  3. Annotation appended to <package>/annotations.jsonl
     (re-runnable: existing annotations are loaded and editable)

Keyboard shortcuts (when supported):
  - SPACE / Enter : confirm and next
  - S            : skip
  - U            : unknown brand
  - ←/→          : navigate prev/next track

Run example:
    streamlit run src/track_annotation/reviewer/app.py -- \
        --package data/annotation_packages/match_2026_04_15
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import streamlit as st


# ============================================================
# CLI args (Streamlit-style: pass after --)
# ============================================================


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=str, required=True, help="Annotation package directory")
    # Strip Streamlit's own args
    argv = sys.argv[1:]
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    return parser.parse_args(argv)


# ============================================================
# Data access
# ============================================================


@st.cache_data
def load_manifest(package_dir: str) -> dict[str, Any]:
    return json.loads((Path(package_dir) / "manifest.json").read_text())


@st.cache_data
def list_tracks(package_dir: str) -> list[int]:
    tracks_root = Path(package_dir) / "tracks"
    return sorted(int(p.name.replace("track_", "")) for p in tracks_root.iterdir() if p.is_dir())


@st.cache_data
def load_track_meta(package_dir: str, track_id: int) -> dict[str, Any]:
    track_dir = Path(package_dir) / "tracks" / f"track_{track_id:05d}"
    return json.loads((track_dir / "meta.json").read_text())


def load_annotations(package_dir: str) -> dict[int, dict]:
    """Load existing annotations.jsonl into dict (last-write-wins per track)."""
    path = Path(package_dir) / "annotations.jsonl"
    out: dict[int, dict] = {}
    if not path.exists():
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                out[int(obj["track_id"])] = obj
            except (json.JSONDecodeError, KeyError):
                continue
    return out


def append_annotation(package_dir: str, annotation: dict) -> None:
    """Append one annotation row to annotations.jsonl."""
    path = Path(package_dir) / "annotations.jsonl"
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(annotation, ensure_ascii=False) + "\n")


def list_logo_templates(template_dir: str) -> list[Path]:
    p = Path(template_dir)
    if not p.exists():
        return []
    return sorted(
        list(p.glob("*.png")) + list(p.glob("*.jpg")) + list(p.glob("*.jpeg"))
    )


# ============================================================
# Streamlit UI
# ============================================================


def main():
    args = _parse_args()
    package_dir = str(Path(args.package).resolve())

    st.set_page_config(
        page_title="Bradford Bulls — Track Annotation Reviewer",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    manifest = load_manifest(package_dir)
    track_ids = list_tracks(package_dir)
    annotations = load_annotations(package_dir)

    # ---- Sidebar ----
    with st.sidebar:
        st.title("Track Reviewer")
        st.caption(f"{manifest['video']['filename']}")
        st.metric("Total tracks", len(track_ids))
        st.metric("Annotated", len(annotations))
        st.metric(
            "Remaining",
            len(track_ids) - len(annotations),
        )
        st.divider()

        if "track_idx" not in st.session_state:
            # Resume from first un-annotated track
            unannotated = [i for i, tid in enumerate(track_ids) if tid not in annotations]
            st.session_state.track_idx = unannotated[0] if unannotated else 0

        nav_col1, nav_col2 = st.columns(2)
        with nav_col1:
            if st.button("◀ Prev", use_container_width=True):
                st.session_state.track_idx = max(0, st.session_state.track_idx - 1)
                st.rerun()
        with nav_col2:
            if st.button("Next ▶", use_container_width=True):
                st.session_state.track_idx = min(
                    len(track_ids) - 1, st.session_state.track_idx + 1
                )
                st.rerun()

        jump_to = st.number_input(
            "Jump to track #",
            min_value=0,
            max_value=len(track_ids) - 1,
            value=st.session_state.track_idx,
            step=1,
        )
        if jump_to != st.session_state.track_idx:
            st.session_state.track_idx = jump_to
            st.rerun()

        st.divider()
        st.caption("Filters")
        only_unannotated = st.checkbox("Only show un-annotated", value=False)
        if only_unannotated:
            unannotated_ids = [tid for tid in track_ids if tid not in annotations]
            if unannotated_ids and track_ids[st.session_state.track_idx] in annotations:
                # Jump to first un-annotated
                st.session_state.track_idx = track_ids.index(unannotated_ids[0])
                st.rerun()

    # ---- Main panel ----
    if not track_ids:
        st.error("No tracks found in package.")
        return

    cur_tid = track_ids[st.session_state.track_idx]
    track_meta = load_track_meta(package_dir, cur_tid)
    track_dir = Path(package_dir) / "tracks" / f"track_{cur_tid:05d}"

    # Header
    st.title(f"Track #{cur_tid}")
    cols = st.columns(4)
    cols[0].metric("Frames", track_meta["num_frames"])
    cols[1].metric("Duration", f"{track_meta['duration_s']:.1f}s")
    cols[2].metric("Mean conf", f"{track_meta['mean_confidence']:.2f}")
    cols[3].metric("Mean area", f"{track_meta['mean_area_ratio']*100:.2f}%")

    if cur_tid in annotations:
        st.success(f"Already annotated: **{annotations[cur_tid].get('brand_id', '?')}** "
                   f"(position={annotations[cur_tid].get('position', '?')})")

    # Keyframes
    st.subheader("Keyframes (best evidence)")
    keyframes = track_meta.get("keyframes", [])
    if keyframes:
        cols = st.columns(len(keyframes))
        for col, kf in zip(cols, keyframes):
            with col:
                st.caption(
                    f"**{kf['strategy']}** · "
                    f"frame {kf['frame_idx']} · "
                    f"sharp={kf['sharpness']:.0f}"
                )
                if kf.get("crop"):
                    crop_path = track_dir / kf["crop"]
                    if crop_path.exists():
                        st.image(str(crop_path), caption="crop", use_container_width=True)
                if kf.get("full_frame"):
                    full_path = track_dir / kf["full_frame"]
                    if full_path.exists():
                        with st.expander("Full frame"):
                            st.image(str(full_path), use_container_width=True)
    else:
        st.info("No keyframes saved.")

    # Video clip
    clip_path = track_dir / "clip.mp4"
    if clip_path.exists():
        st.subheader("Video clip (2s)")
        st.video(str(clip_path), loop=True, autoplay=True)
    else:
        st.info("No video clip available for this track.")

    # Brand assignment
    st.subheader("Assign brand")
    brand_ids = manifest["logo_templates"]["brand_ids"]
    template_dir = manifest["logo_templates"]["dir"]
    templates = {p.stem: p for p in list_logo_templates(template_dir)}

    # Brand template grid (visual reference)
    with st.expander("Reference: 21 brand templates", expanded=False):
        ncols = 7
        for i in range(0, len(brand_ids), ncols):
            row = brand_ids[i:i + ncols]
            cols = st.columns(len(row))
            for col, bid in zip(cols, row):
                with col:
                    # Try to find a template image whose stem contains the brand_id
                    match = next(
                        (p for stem, p in templates.items() if bid in stem.lower()),
                        None,
                    )
                    if match:
                        st.image(str(match), caption=bid, use_container_width=True)
                    else:
                        st.caption(f"_{bid}_")

    # Brand selection
    existing = annotations.get(cur_tid, {})
    selected_brand = st.selectbox(
        "Brand",
        options=["(unassigned)", "unknown"] + brand_ids,
        index=(
            (["(unassigned)", "unknown"] + brand_ids).index(existing.get("brand_id"))
            if existing.get("brand_id") in (["(unassigned)", "unknown"] + brand_ids)
            else 0
        ),
    )
    selected_position = st.selectbox(
        "Position",
        options=[
            "chest_front", "chest_back",
            "sleeve_left", "sleeve_right",
            "short_left", "short_right",
            "collar", "other",
        ],
        index=0,
    )
    selected_quality = st.selectbox(
        "Visibility quality",
        options=["clear", "partial", "blurry", "occluded"],
        index=0,
    )
    is_target = st.checkbox("Is target team (Bradford Bulls)", value=True)

    save_col, skip_col, _ = st.columns([1, 1, 4])
    with save_col:
        if st.button("✅ Save & next", use_container_width=True, type="primary"):
            ann = {
                "track_id": cur_tid,
                "brand_id": selected_brand if selected_brand != "(unassigned)" else None,
                "position": selected_position,
                "visibility_quality": selected_quality,
                "is_target_team": is_target,
                "skip": False,
            }
            append_annotation(package_dir, ann)
            st.session_state.track_idx = min(len(track_ids) - 1, st.session_state.track_idx + 1)
            st.rerun()
    with skip_col:
        if st.button("⏭ Skip (not visible)", use_container_width=True):
            ann = {
                "track_id": cur_tid,
                "brand_id": None,
                "skip": True,
            }
            append_annotation(package_dir, ann)
            st.session_state.track_idx = min(len(track_ids) - 1, st.session_state.track_idx + 1)
            st.rerun()


if __name__ == "__main__":
    main()
