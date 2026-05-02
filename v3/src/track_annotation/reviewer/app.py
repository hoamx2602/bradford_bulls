"""
Streamlit reviewer UI — brand-aware version.

Brand vs Variant
----------------
The reviewer presents the annotator with the BRAND (e.g., "Aon") not raw
variant codes (e.g., "aon_red"). The variant_id is auto-derived from the
match's kit_context (loaded from manifest.json).

If a brand has multiple active variants for the same kit_context, a secondary
selector appears.

Annotation row written to annotations.jsonl:
    {
      "track_id": 42,
      "brand_id": "aon",            # master brand
      "variant_id": "aon_red",      # auto-derived from kit_context
      "position": "chest_front",
      "visibility_quality": "clear",
      "is_target_team": true,
      "skip": false,
      "kit_context": "home"         # snapshot from manifest at write time
    }

Run:
    streamlit run src/track_annotation/reviewer/app.py -- \\
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
# CLI args
# ============================================================


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=str, required=True)
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
    """Load existing annotations.jsonl (last-write-wins per track)."""
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
    path = Path(package_dir) / "annotations.jsonl"
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(annotation, ensure_ascii=False) + "\n")


def resolve_template_path(manifest: dict, template_path: str) -> Path | None:
    """Resolve a brand variant template_path against the templates dir."""
    base = Path(manifest["logo_templates"]["dir"])
    p = base / template_path
    return p if p.exists() else None


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

    kit_context = manifest.get("match_context", {}).get("kit_context", "any")
    active_brands = manifest["logo_templates"].get("active_brands", [])

    # ---- Sidebar ----
    with st.sidebar:
        st.title("Track Reviewer")
        st.caption(manifest["video"]["filename"])
        st.markdown(f"**Kit context**: `{kit_context}`")
        if manifest.get("match_context", {}).get("opponent"):
            st.caption(f"vs {manifest['match_context']['opponent']}")
        st.divider()

        st.metric("Total tracks", len(track_ids))
        st.metric("Annotated", len(annotations))
        st.metric("Remaining", len(track_ids) - len(annotations))
        st.metric("Active brands", len(active_brands))
        st.divider()

        if "track_idx" not in st.session_state:
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
        only_unannotated = st.checkbox("Only un-annotated", value=False)
        if only_unannotated:
            unannotated_ids = [tid for tid in track_ids if tid not in annotations]
            if unannotated_ids and track_ids[st.session_state.track_idx] in annotations:
                st.session_state.track_idx = track_ids.index(unannotated_ids[0])
                st.rerun()

    # ---- Main panel ----
    if not track_ids:
        st.error("No tracks found in package.")
        return

    cur_tid = track_ids[st.session_state.track_idx]
    track_meta = load_track_meta(package_dir, cur_tid)
    track_dir = Path(package_dir) / "tracks" / f"track_{cur_tid:05d}"

    st.title(f"Track #{cur_tid}")
    cols = st.columns(4)
    cols[0].metric("Frames", track_meta["num_frames"])
    cols[1].metric("Duration", f"{track_meta['duration_s']:.1f}s")
    cols[2].metric("Mean conf", f"{track_meta['mean_confidence']:.2f}")
    cols[3].metric("Mean area", f"{track_meta['mean_area_ratio']*100:.2f}%")

    if cur_tid in annotations:
        existing_ann = annotations[cur_tid]
        st.success(
            f"Already annotated: **{existing_ann.get('brand_id', '?')}** "
            f"(variant={existing_ann.get('variant_id', '?')}, "
            f"position={existing_ann.get('position', '?')})"
        )

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

    # ---- Brand assignment ----
    st.subheader(f"Assign brand  ·  kit_context = `{kit_context}`")

    # Brand template grid (only ACTIVE brands for this kit_context)
    with st.expander(
        f"Reference: {len(active_brands)} brands active for kit_context={kit_context!r}",
        expanded=False,
    ):
        ncols = 6
        for i in range(0, len(active_brands), ncols):
            row = active_brands[i:i + ncols]
            cols = st.columns(len(row))
            for col, brand in zip(cols, row):
                with col:
                    # Show first variant template (typically only 1 active per kit_context)
                    variants = brand.get("variants", [])
                    if variants:
                        tpl = resolve_template_path(manifest, variants[0]["template_path"])
                        if tpl:
                            st.image(str(tpl), caption=brand["display_name"], use_container_width=True)
                        else:
                            st.caption(f"_{brand['display_name']}_")
                            st.caption(f"(template missing: {variants[0]['template_path']})")
                    else:
                        st.caption(f"_{brand['display_name']}_")

    # Brand selection — by master brand_id, NOT variant_id
    brand_options = ["(unassigned)", "unknown"] + [b["id"] for b in active_brands]
    existing = annotations.get(cur_tid, {})
    default_idx = (
        brand_options.index(existing.get("brand_id"))
        if existing.get("brand_id") in brand_options
        else 0
    )
    selected_brand = st.selectbox(
        "Brand",
        options=brand_options,
        index=default_idx,
        format_func=lambda b: (
            "(unassigned)" if b == "(unassigned)"
            else "unknown" if b == "unknown"
            else next(
                (br["display_name"] for br in active_brands if br["id"] == b),
                b,
            )
        ),
    )

    # Auto-derive variant; show selector only if brand has >1 active variant
    selected_variant: str | None = None
    if selected_brand not in ("(unassigned)", "unknown"):
        brand_obj = next((b for b in active_brands if b["id"] == selected_brand), None)
        if brand_obj:
            active_variants = brand_obj.get("variants", [])
            if len(active_variants) == 1:
                selected_variant = active_variants[0]["id"]
                st.caption(f"Auto-derived variant: `{selected_variant}` "
                           f"(only active variant for kit_context={kit_context})")
            elif len(active_variants) > 1:
                selected_variant = st.selectbox(
                    "Variant (multiple active for this kit_context)",
                    options=[v["id"] for v in active_variants],
                )
            else:
                st.warning(f"Brand {selected_brand} has no active variant for kit_context={kit_context}")

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
                "variant_id": selected_variant,
                "position": selected_position,
                "visibility_quality": selected_quality,
                "is_target_team": is_target,
                "skip": False,
                "kit_context": kit_context,
            }
            append_annotation(package_dir, ann)
            st.session_state.track_idx = min(len(track_ids) - 1, st.session_state.track_idx + 1)
            st.rerun()
    with skip_col:
        if st.button("⏭ Skip (not visible)", use_container_width=True):
            ann = {
                "track_id": cur_tid,
                "brand_id": None,
                "variant_id": None,
                "skip": True,
                "kit_context": kit_context,
            }
            append_annotation(package_dir, ann)
            st.session_state.track_idx = min(len(track_ids) - 1, st.session_state.track_idx + 1)
            st.rerun()


if __name__ == "__main__":
    main()
