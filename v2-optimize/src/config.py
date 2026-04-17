"""
Bradford Bulls v2 — Configuration.

All tunable parameters in one place.
"""

# ═══════════════════════════════════════════════════════════════════════
# YOLO Person Detection
# ═══════════════════════════════════════════════════════════════════════
PERSON_MODEL = "yolo11l.pt"
PERSON_CONFIDENCE = 0.45
MIN_PERSONS = 1

# ═══════════════════════════════════════════════════════════════════════
# Sharpness Tiers (TORSO-BASED)
#
# Measured on the torso region (15%-65% of player height) where logos are.
# Combined Laplacian + Tenengrad score, normalized 0-1.
# ═══════════════════════════════════════════════════════════════════════
SHARPNESS_TIER_GOLD   = 0.20   # Logo clearly readable
SHARPNESS_TIER_SILVER = 0.12   # Slightly soft, logo still identifiable
SHARPNESS_TIER_BRONZE = 0.06   # Mild blur, logo partially readable
# Below BRONZE → skip frame entirely

# ═══════════════════════════════════════════════════════════════════════
# Person Size Filters
# ═══════════════════════════════════════════════════════════════════════
# Minimum (largest person bbox / frame area) to keep frame.
MIN_MAX_PERSON_AREA_RATIO = 0.015

# ═══════════════════════════════════════════════════════════════════════
# Pitch Detection
# ═══════════════════════════════════════════════════════════════════════
ENABLE_PITCH_GREEN_FILTER = True
PITCH_ROI_Y_START = 0.40
MIN_PITCH_GREEN_RATIO = 0.04

# ═══════════════════════════════════════════════════════════════════════
# Pass 1 Scan
# ═══════════════════════════════════════════════════════════════════════
SCAN_INTERVAL = 5             # Check every Nth frame in Pass 1
MIN_SEGMENT_FRAMES = 2        # Minimum consecutive quality frames
SEGMENT_GAP_TOLERANCE = 3     # Allowed bad frames within segment

# ═══════════════════════════════════════════════════════════════════════
# Selection Quotas
# ═══════════════════════════════════════════════════════════════════════
DEFAULT_QUOTA = {
    "target_closeup":   0.30,
    "target_medium":    0.20,
    "mixed":            0.15,
    "opponent_closeup": 0.08,
    "opponent_medium":  0.07,
    "target_wide":      0.05,
    "opponent_wide":    0.05,
    "background":       0.10,
}

# ═══════════════════════════════════════════════════════════════════════
# Temporal Diversity
# ═══════════════════════════════════════════════════════════════════════
MIN_TIME_GAP_SEC = 1.5         # Minimum seconds between selected frames

# ═══════════════════════════════════════════════════════════════════════
# Output
# ═══════════════════════════════════════════════════════════════════════
JPEG_QUALITY = 95

# ═══════════════════════════════════════════════════════════════════════
# Annotation Propagation (Phase 1b)
# ═══════════════════════════════════════════════════════════════════════
PROPAGATION_RADIUS = 3         # ±N frames from each annotated frame
PROPAGATION_MATCH_THRESHOLD = 0.5   # Template match confidence minimum
PROPAGATION_MIN_TRACK_RATIO = 0.5   # At least 50% bboxes must track successfully
