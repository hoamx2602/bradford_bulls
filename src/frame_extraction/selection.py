"""
Quota-based frame selection for balanced training datasets.

Ensures the final frame set has a controlled mix of:
- Target team dominant (65%)
- Mixed frames (20%)  
- Opponent dominant (15%)
- Crowd/graphics (0%)
"""

import numpy as np
from .helpers import compute_phash


# Default quotas — adjust for your use case
DEFAULT_QUOTA = {
    "target_closeup":   0.30,
    "target_medium":    0.25,
    "target_wide":      0.05,
    "mixed":            0.20,
    "opponent_closeup": 0.08,
    "opponent_medium":  0.07,
    "opponent_wide":    0.05,
}


def select_by_quota(candidates, total_target, quota=None, min_time_gap=1.5,
                    dedup_hash_thresh=8):
    """
    Select frames using quota-based allocation.
    
    Args:
        candidates: List of candidate dicts from Pass 2
        total_target: Total number of frames to select
        quota: Dict mapping category → fraction (must sum to 1.0)
        min_time_gap: Minimum seconds between selected frames (temporal diversity)
        dedup_hash_thresh: pHash distance threshold for deduplication
    
    Returns:
        selected: List of selected candidate dicts
        selection_stats: Dict with per-category counts
    """
    if not candidates:
        return [], {}

    q = quota or DEFAULT_QUOTA

    # Group candidates by category
    by_category = {}
    for c in candidates:
        cat = c["category"]
        # Map detailed categories to quota keys
        quota_key = _map_to_quota_key(cat)
        by_category.setdefault(quota_key, []).append(c)

    # Sort each group by score (descending)
    for key in by_category:
        by_category[key].sort(key=lambda x: x["score"], reverse=True)

    # Allocate frames per category
    selected = []
    selection_stats = {}

    for cat_key, fraction in sorted(q.items(), key=lambda x: -x[1]):
        n_want = max(1, int(total_target * fraction))
        pool = by_category.get(cat_key, [])

        picked = _pick_with_diversity(pool, n_want, min_time_gap)
        selected.extend(picked)
        selection_stats[cat_key] = len(picked)

    # If we didn't reach total_target, fill from highest-scored remaining
    if len(selected) < total_target:
        selected_nums = {c["frame_num"] for c in selected}
        remaining = [c for c in candidates if c["frame_num"] not in selected_nums]
        remaining.sort(key=lambda x: x["score"], reverse=True)

        for c in remaining:
            if len(selected) >= total_target:
                break
            # Check time gap
            if all(abs(c["timestamp_sec"] - s["timestamp_sec"]) >= min_time_gap / 2
                   for s in selected):
                selected.append(c)

    # Sort by timestamp
    selected.sort(key=lambda x: x["timestamp_sec"])

    return selected, selection_stats


def _map_to_quota_key(category):
    """Map detailed category names to quota keys."""
    # Categories from pipeline: target_closeup, target_medium, target_wide,
    #   mixed, opponent_closeup, opponent_medium, opponent_wide
    if category.startswith("ambiguous"):
        # Map ambiguous to mixed
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

        # Check temporal gap with already picked frames
        ts = candidate["timestamp_sec"]
        if all(abs(ts - p["timestamp_sec"]) >= min_gap_sec for p in picked):
            picked.append(candidate)

    return picked


def print_selection_summary(selected, selection_stats, total_candidates):
    """Print a summary of the quota selection results."""
    print(f"\n{'=' * 60}")
    print(f"  SELECTION RESULTS")
    print(f"{'=' * 60}")
    print(f"  Total candidates:  {total_candidates}")
    print(f"  Selected frames:   {len(selected)}")
    print(f"")
    print(f"  Per-category breakdown:")
    for cat, count in sorted(selection_stats.items()):
        pct = count / max(len(selected), 1) * 100
        bar = "█" * int(pct / 2)
        print(f"    {cat:25s}  {count:4d}  ({pct:5.1f}%)  {bar}")

    if selected:
        scores = [c["score"] for c in selected]
        sharp = [c["sharpness"] for c in selected]
        print(f"\n  Score range:       {min(scores):.3f} — {max(scores):.3f}")
        print(f"  Sharpness range:   {min(sharp):.3f} — {max(sharp):.3f}")
        print(f"  Time span:         {selected[0]['timestamp_hms']} — "
              f"{selected[-1]['timestamp_hms']}")
    print(f"{'=' * 60}")
