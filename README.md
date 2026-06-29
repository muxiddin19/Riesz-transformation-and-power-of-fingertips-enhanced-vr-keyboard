# VR Keyboard — CVPR 2026 Implementation

**Real-Time Multimodal Fingertip Contact Detection via Depth and Motion Fusion**

Author: Mukhiddin Toshpulatov  
Institution: KAIST SpaceTop Research Center

---

## Overview

This project implements a vision-based virtual keyboard that detects fingertip contact with a physical keyboard surface using a combination of depth estimation and motion analysis — no physical key press required. The system runs in real time using an Intel RealSense D405 depth camera and fine-tuned Depth Anything V2 model.

---

## Paper Claims (Target)

| Metric | Value |
|---|---|
| Contact Detection Accuracy | 94.2% |
| Mean Absolute Error (MAE) | 3.2mm (after fine-tuning) |
| Words Per Minute (WPM) | 45.6 |
| Character Error Rate (CER) | 3.1% |
| F1-Score | 94.4% |
| False Positive Rate | 4.2% |

---

## Hardware Requirements

- **Camera:** Intel RealSense D405 (mandatory — uses hardware depth)
- **Connection:** USB 3.x port directly on motherboard (not a hub)
- **GPU:** CUDA-capable (tested on NVIDIA RTX 4060 Laptop)
- **OS:** Windows 10/11

---

## Software Dependencies

```bash
pip install pyrealsense2 mediapipe opencv-python torch numpy pynput pandas
```

Additionally requires:
- Depth Anything V2 source (metric depth variant)
- Fine-tuned checkpoint: `depth_anything_v2_vits_d405_finetuned.pth`

---

## Project Structure

```
project/
├── new.py                          # Main implementation file (v7)
├── riesz_visualizer.py             # Signal visualizer for depth/velocity
├── src/
│   ├── depth_model_manager.py      # DepthEstimator wrapper (ViT-B/S)
│   └── assets/
│       └── keyboard_annotations.json  # Key polygon coordinates
├── checkpoints/
│   └── depth_anything_v2_vits_d405_finetuned.pth
└── depth_velocity_log.csv          # Auto-saved tap data log
```

---

## Architecture

The system fuses three signals to detect contact:

```
RealSense D405
  ├── Color frame  →  MediaPipe Hands  →  Fingertip (x, y)
  └── Depth frame  →  Depth at fingertip  →  Distance from surface
                                                      │
                              ┌───────────────────────┤
                              │                       │
                     Velocity Analysis        Depth Hysteresis
                     (state machine)          (entry/exit gates)
                              │                       │
                              └──────────┬────────────┘
                                         │
                                 Tap Quality Score
                                 (compute_natural_tap_quality)
                                         │
                              ┌──────────┴────────────┐
                              │                       │
                       PATH A: velocity         PATH B: score
                       confirmed tap            confirmed tap
                              │                       │
                              └──────────┬────────────┘
                                         │
                                  Key Detection
                                  (polygon test)
                                         │
                                  Keyboard Output
```

---

## Detection Algorithm

### 1. Depth Smoothing

Raw depth from RealSense is smoothed over the last 8 frames using exponential moving average (α=0.35) with outlier rejection (jumps > 5cm discarded).

### 2. Hysteresis Gating

| Threshold | Value |
|---|---|
| Contact entry | < 18mm from surface |
| Contact exit | > 30mm from surface |

The hysteresis band prevents rapid toggling near the contact boundary.

### 3. Velocity State Machine

```
IDLE → APPROACHING → CONTACT → RETRACTING → IDLE
```

- **APPROACHING:** `vy > 15 px/s` sustained downward
- **CONTACT:** velocity drops by 50%+ from peak, peak ≥ 25 px/s
- **RETRACTING:** `vy < -8 px/s`

### 4. Tap Quality Score (`compute_natural_tap_quality`)

Combines four signals into a score in [0, 1]:

```python
score = (
    velocity_collapse * 0.50 +    # Did velocity stop suddenly?
    peak_is_recent    * 0.22 +    # Was there a recent peak > 11 px/s?
    depth_delta       * 0.18 +    # Did finger descend before stopping?
    lateral_stability * 0.10      # Was motion controlled?
) - dwell_penalty * 0.20          # Penalize hovering
```

### 5. Dual Trigger Paths

**Path A — Velocity confirmed:**
- State machine fired `CONTACT`
- Distance < 2.5cm
- Peak velocity ≥ 25 px/s

**Path B — Score confirmed:**
- Score ≥ 0.60
- `depth_ok = True` (within hysteresis zone)
- Distance < 2.5cm
- Peak velocity in recent history ≥ 18 px/s

### 6. Cooldown

15 frames (~500ms at 30fps) per finger after each tap, preventing double-triggers.

---

## Calibration

### Auto-Calibration (recommended)
Runs automatically on startup. Samples 9 depth points across the keyboard plane and computes a scale factor. Requires hands away from keyboard during startup.

### Manual Calibration
1. Click on the keyboard surface in the window
2. Press `C`
3. Enter the actual distance in cm when prompted

---

## Controls

| Key | Action |
|---|---|
| `C` | Manual calibration (click surface first) |
| `D` | Toggle debug mode (shows per-frame distances and scores) |
| `M` | Print performance metrics |
| `Q` / `Esc` | Quit |

---

## Configuration (Current Best Settings)

```python
VelocityBasedContactDetector(
    history_size=20,
    contact_entry_threshold_cm=1.8,   # 18mm entry
    contact_exit_threshold_cm=3.0,    # 30mm exit
    cooldown_frames=15,               # ~500ms
    min_peak_velocity=25.0,           # px/s
    velocity_threshold_approach=15.0, # px/s
    velocity_threshold_stop=6.0,      # px/s
    confidence_threshold=0.60
)

MIN_PEAK_FOR_SCORE_PATH = 18.0    # px/s — minimum velocity for score path
MIN_PEAK_FOR_VELOCITY_PATH = 25.0 # px/s — minimum velocity for state machine path
MAX_VELOCITY = 300.0              # px/s cap
```

---

## Tip Power Estimator

A secondary signal that estimates contact force by fusing:

| Component | Weight |
|---|---|
| Impact velocity | 45% |
| Brightness change at fingertip | 30% |
| Skin contact area (LAB color) | 25% |

Displayed as a gauge in the UI. Not currently used for tap gating but available for future integration.

---

## Debug Log Format

When debug mode is on (`D` key), each frame prints:

```
[DIST] 1.52 depth_ok=True vel=False
[TAPDBG] q=0.87 dist=1.52cm peak=66px/s dwell=0 reject=
[TAP] h0_f8 | score | score=0.87 | dist=1.5cm | peak=66px/s
[KEY SELECTED] s polygon_score=8.1
```

- `depth_ok` — whether finger is within hysteresis zone
- `vel` — whether velocity state machine fired this frame
- `q` — tap quality score
- `peak` — maximum recent velocity
- `dwell` — number of frames finger was stationary near surface
- `reject` — reason for rejection if any

---

## Current Performance (Observed)

| Metric | Observed |
|---|---|
| Contact Detection Accuracy | 100% (in testing sessions) |
| F1-Score | 100% (in testing sessions) |
| False Positive Rate | ~0% (after velocity gating) |
| WPM | ~14 (user still learning layout) |
| CER | 0% |

---

## Known Issues & Limitations

- **Auto-calibration fails if hand is over keyboard** during startup. Keep hands away and retry with `C`.
- **Two-hand depth interleaving** — with two hands tracked, depth readings for each finger alternate each frame. This is handled by per-finger smoothing but can cause occasional distance spikes.
- **Velocity cap at 300 px/s** — very fast taps hit the cap, losing true peak information. This slightly reduces score accuracy for aggressive typists.
- **Frame timeout** — RealSense D405 requires USB 3.x. If `Frame didn't arrive within 5000` error occurs, switch to a different USB port directly on the motherboard.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `Frame didn't arrive within 5000` | Use USB 3.x port, not a hub. Try opposite side of laptop. |
| Auto-calibration warns "unstable" | Remove hands from keyboard view during startup. |
| Taps not detected | Check `depth_ok` in debug — if always False, recalibrate manually with `C`. |
| Too many false positives | Increase `MIN_PEAK_FOR_SCORE_PATH` from 18 to 25. |
| Missed taps | Decrease `confidence_threshold` from 0.60 to 0.55, or lower `min_peak_velocity` to 20. |

---

## Development History

| Version | Key Change |
|---|---|
| v1 | Initial implementation — velocity state machine + depth hysteresis |
| v2 | Fixed `depth_ok` always False — smoothing lag issue |
| v3 | Added dual trigger paths (velocity + score) |
| v4 | Added `MIN_PEAK` velocity gate — eliminated `peak=0px/s` false positives |
| v5 | Tightened velocity path distance gate to 2.5cm — eliminated 3.8cm false positives |
| v6 | Raised `MAX_VELOCITY` to 300 px/s — captured fast tap peaks accurately |
| v7 (current) | Reduced dwell window and penalty — faster recovery between taps |
