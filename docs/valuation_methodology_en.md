# Sponsorship Valuation Methodology & Analytics Architecture

This document details how the system transforms raw Machine Learning data (AI Bounding Boxes) into actionable business metrics (Sponsorship Media Value) for Sponsors and Sports Clubs.

Unlike basic "frame counting" algorithms, enterprise-grade sports media analytics (e.g., Nielsen Sports, GumGum) deploy a sophisticated 4-Layer Filtering Pipeline. This structure guarantees that metrics are transparent, deduplicated, and accurately reflect true visual quality.

---

## Layer 1: Simultaneous Exposure Deduplication

**The Problem:** 
A close-up camera shot celebrating a goal might capture up to 4 players, all wearing jerseys emblazoned with the "AON" logo. A YOLO algorithm will output 4 independent Bounding Boxes. If the system simply sums the duration of these 4 boxes, it creates "Double Counting," resulting in an exposure time larger than the actual video duration.

**The Technical Solution:**
- **Event Aggregation:** The algorithm merges all duplicate detections occurring within the same millisecond (the same frame) into a **Single Exposure Event**. The time credited to the brand remains exactly the length of that broadcast segment.
- **Auxiliary Variable:** While consolidated, the system tracks the `Logo_Count`. This value is preserved as a weighted bonus for subsequent layers (as multiple concurrent logos command higher visual attention).

---

## Layer 2: Temporal Smoothing & Noise Reduction

**The Problem:** 
A player's dynamic movement (running, twisting, getting tackled) prevents AI from maintaining a 100% seamless detection. The return signal often flickers: `[Found] - [Found] - [Missed for 3 frames] - [Found]`. Clients cannot act upon a report flooded with disjointed "0.1-second micro-events."

**The Technical Solution:**
1. **Gap Bridging (Hysteresis):** 
   By implementing Object Tracking concepts, if a logo is temporarily obstructed and reappears within a tiny fraction of time (e.g., under 15 frames / 0.5 seconds), the system infers this as a physical obstruction rather than an exit. It automatically "bridges" them into one continuous exposure stripe.
2. **Minimum Viewability Threshold:**
   International broadcasting standards indicate that the human brain requires at least **1.5 to 2 seconds** to process and recognize a brand flashing on a screen. Any continuous exposure event falling short of this threshold is purged from the final financial report to maintain strict valuation integrity.

---

## Layer 3: Quality Indexing (The Real "Confidence")

**The Problem:** 
The "Confidence Score" returned by models like YOLO simply indicates "How certain the machine is that this is a logo"—it completely fails to capture the "Visibility" or prominence that a human television viewer perceives. This visibility is the crucial metric for tiering media value.

**The Technical Solution (Calculating QI):**
Every valid second of exposure passing through Layer 2 is graded from 0.0 to 1.0 based on a Quality Weight derived from 4 spatial factors:

1. **Size (Share of Voice):**
   + *Formula:* The ratio of the Bounding Box area divided by the total Video Screen Area.
   + A close-up cinematic shot provides exponentially higher quality and value than a wide tactical camera shot.
2. **Position & Prominence:**
   + A logo positioned dead-center on the screen receives a `1.0` multiplier. As the logo drifts toward the outer edges (peripheral areas rarely focused on by viewers), the multiplier drops to `0.5` or lower.
3. **Clarity & Lighting:**
   + At this stage, the AI's `Confidence Score` is merged with a Laplacian filter (Motion Blur detection) to act as a penalty coefficient. Heavily blurred or excessively dark exposures will face strict Quality deductions.
4. **Clutter (Brand Competition):**
   + Advanced metric: If a single screen simultaneously displays 5 different sponsor logos (on shirts, electronic boards, stadiums), viewer attention is fragmented. The Quality Index of each individual logo on screen undergoes a minor deduction to account for this competition.

---

## Layer 4: Reporting & Financial Conversion (Valuation Engine)

The final layer is the "revenue engine." Data surviving the above layers is transposed into a Dashboard using Marketing semantics rather than algorithmic jargon:

1. **Total On-Screen Time:** 
   The absolute raw seconds a brand was visible on the broadcast (deduplicated). e.g., 15 Minutes on screen.
2. **Total Clear Impact Events:** 
   The number of distinct, uninterrupted exposure events spanning over 2 seconds. e.g., 60 Events.
3. **100% Equivalent Time:** 
   The ultimate pinnacle metric. Derived by multiplying (Time per discrete Second) X (The QI score of that Second).
   e.g., Those raw 15 Minutes—often consisting of fragmented, blurry, or tiny wide-shots—will be mathematically compressed into an undeniable **"Equivalent of 5 Minutes 20 Seconds of Full-Screen, Crystal-Clear Domination."**
4. **Media Advertising Value Equivalence:**
   The final step calculates the 100% Equivalent Time `[5m 20s]` multiplied by standard `[TV Advertising Per-Second Rate]`. This delivers the paramount financial picture precisely tailored for club owners and investors during sponsorship renegotiations.
