# Technical characteristics — Frame selection (training data)

This document records the practical technical insights behind frame selection for **logo-on-kit** training data, based on broadcast match footage.

Source notebook: `notebooks/colab_03_frame_extraction.ipynb`.

---

## Goals

- Build a frame set that is **useful for annotating kit sponsor logos**.
- Minimize wasted annotation on frames where **logos are not readable** (or not present on kits).
- Keep the pipeline **tunable** via simple thresholds so it generalizes across matches.

---

## Core insight 1 — Reject wide shots (camera too far)

**Problem**

- In wide shots, players occupy too few pixels.
- If you cannot clearly see the player, you cannot see sponsor logos (training signal is essentially zero).

**Heuristic used**

- Detect persons with YOLO.
- Compute:
  - `max_person_area_ratio = max(person_bbox_area) / frame_area`
- Reject frame if:
  - `max_person_area_ratio < MIN_MAX_PERSON_AREA_RATIO`

**Tuning**

- Increase `MIN_MAX_PERSON_AREA_RATIO` to be stricter (fewer far-away frames).
- Decrease it if you lose too many “good gameplay” frames.

---

## Core insight 2 — Reject stage / interview / non-pitch frames

**Problem**

- Some frames contain people (YOLO detects persons) but are not gameplay/pitch (stage, interviews, crowd shots).
- These are not useful for training “logo on kit during match” and create label noise or wasted annotation time.

**Heuristic used (pitch green filter)**

- Measure the fraction of “grass-green” pixels in a bottom-region ROI.
- Compute:
  - `pitch_green_ratio = green_pixels_in_bottom_roi / roi_pixels`
- Reject frame if:
  - `ENABLE_PITCH_GREEN_FILTER` is on **and**
  - `pitch_green_ratio < MIN_PITCH_GREEN_RATIO`

**Tuning**

- If the filter rejects too much (lighting, turf hue), reduce `MIN_PITCH_GREEN_RATIO`.
- If stage/interview frames still slip through, increase `MIN_PITCH_GREEN_RATIO`.
- ROI start is configurable (`PITCH_ROI_Y_START`) to focus on the bottom part of the broadcast frame where pitch typically appears.

---

## Core insight 3 — Allow a configurable % of “non-relevant team”

**Problem**

- For training Bradford Bulls kit sponsors, frames dominated by the opponent team (or non-kit scenes) are less valuable.
- However, hard-filtering can be risky if team detection is imperfect.

**Approach**

- Use a **quota** rather than an absolute filter:
  - Allow at most `MAX_NON_RELEVANT_PCT` of the final selected frames to be “non-relevant”.

**Implementation notes**

- This is only applied when `ENABLE_TEAM_RELEVANCE_FILTER = True`.
- In the notebook, team relevance is currently a **color heuristic**:
  - crop detected persons
  - compute `bulls_color_ratio` = max ratio of pixels matching configured `BULLS_HSV_RANGES`
  - `is_relevant_team = bulls_color_ratio >= BULLS_COLOR_MIN_RATIO`
- Selection enforces:
  - `non_relevant_selected <= ceil(TARGET_FRAMES * MAX_NON_RELEVANT_PCT)`

**Tuning**

- Start with `ENABLE_TEAM_RELEVANCE_FILTER = False` unless the HSV ranges are tuned for the Bulls kit in your match footage.
- After tuning, adjust:
  - `BULLS_HSV_RANGES`
  - `BULLS_COLOR_MIN_RATIO`
  - `MAX_NON_RELEVANT_PCT`

---

## Auditability (must-have for iteration)

To make threshold tuning tractable, the selection metadata is saved with per-frame diagnostics:

- `max_person_area_ratio` (wide-shot indicator)
- `pitch_green_ratio` (stage/not-on-pitch indicator)
- `bulls_color_ratio`, `is_relevant_team` (only meaningful when team relevance is enabled)

This data is written to `selected_frames_index.csv` so you can quickly inspect:

- which thresholds are too strict/too loose
- what kinds of frames are still slipping through

