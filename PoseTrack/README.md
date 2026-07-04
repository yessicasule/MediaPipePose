# DeepFusionPose
### Monocular Vision-Based Arm Tracking for Rehabilitation Robotics & Wearable Exoskeleton Calibration

---

## System Overview

A real-time pipeline that estimates human arm joint angles from a single RGB camera and drives a 4-avatar Unity Digital Twin, with future integration support for soft wearable exoskeletons.

```
Webcam
  │
  ▼
MediaPipe / MoveNet / PoseNet  ←── 3 pose frameworks compared
  │
  ▼
Joint Angle Estimator          ←── 2-link kinematic chain (shoulder + elbow)
  │
  ├──▶ Kalman / EMA / MA / SG filters
  ├──▶ Calibration module (arm_down → arm_forward → elbow_flex)
  ├──▶ Gesture Recognizer (WAVE / RAISE_ARM / REACH_FORWARD / ELBOW_FLEX / REST)
  │
  ▼
DeepFusionPose Model           ←── Transformer + CrossAttn + BiLSTM (Phase 4+5)
  │
  ├──▶ MC Dropout → uncertainty estimation (Phase 5)
  ├──▶ GAN temporal refinement (Phase 6)
  │
  ▼
UDP Streamer (port 9000)
  │   MP,pitch,roll,yaw,elbow
  │   MV,pitch,roll,yaw,elbow
  │   FU,pitch,roll,yaw,elbow,uncertainty
  │   GR,pitch,roll,yaw,elbow
  │
  ▼
Unity Digital Twin
  4 avatars (MediaPipe / MoveNet / Fusion / GAN) animate simultaneously
  Confidence badge on Fusion avatar
```

---

## Quick Start

### 1 — Install dependencies
```bash
pip install mediapipe opencv-python torch numpy scipy matplotlib
```
### Test Unity without a camera
```powershell
python scripts/mock_streamer.py --mode sinusoidal
```
Then in Unity: **MonoArm → Build Scene**, save, Press Play.

### Run live pipeline
```powershell
python scripts/run_demo.py --filter kalman
```

### Controls (run_demo.py)
| Key | Action |
|---|---|
| Q | Quit |
| F | Cycle filter (kalman → ma → sg) |
| C | Run calibration wizard |
| L | Toggle CSV logging |
| R | Reset filters |

## UDP Packet Format
```
S,<shoulder_flex>,<shoulder_abd>,<shoulder_rot>,<elbow_flex>\n
```
All values in degrees. Port 9000 (configurable).

## Architecture
```
Camera → MediaPipe Pose → coordinate_frame.py → angle_solver.py
       → angle_filter.py (Kalman/MA/SG) → calibration.py
       → udp_streamer.py → Unity UdpAngleReceiver → AvatarMuscleController
```

## Shoulder Angle Convention
- **Flexion (+)**: arm forward; **Extension (−)**: arm backward
- **Abduction (+)**: arm out to side; **Adduction (−)**: arm crossing body
- **Int. Rotation (+)**: palm inward; **Ext. Rotation (−)**: palm outward
- **Elbow Flexion**: 0° = straight, ~150° = fully bent

## Files Changed from Original 

### 3 — Run with trained models (after Kaggle training)
```bash
python scripts/run_demo.py \
  --fusion_ckpt outputs/models/fusion_best.pt \
  --gan_ckpt    outputs/models/gan_generator_best.pt \
  --scaler      outputs/models/fusion_scaler.json
```

### Demo Controls
| Key | Action |
|-----|--------|
| `Q` | Quit |
| `C` | Run calibration (3 reference poses) |
| `L` | Toggle angle logging (CSV + auto-plot on stop) |
| `S` | Save snapshot |
| `F` | Cycle filter: Kalman → EMA → Moving Average → Savitzky–Golay |

---

## Final Deliverables

| Deliverable | File(s) |
|-------------|---------|
| Real-time arm tracking app | `scripts/run_demo.py` |
| Joint angle estimation | `src/processing/joint_angle_estimator.py` |
| Filtering module (4 types) | `src/processing/angle_filter.py` |
| Gesture recognition | `src/processing/gesture_recognizer.py` |
| Unity avatar control | `Unity/UnityMedia/Assets/Scripts/` |
| Calibration module | `src/processing/calibration.py` |
| Data logging + visualization | `src/processing/angle_logger.py` + `scripts/plot_angles.py` |
| Latency benchmark | `scripts/benchmark_latency.py` |
| Exoskeleton reference stream | `src/streaming/exoskeleton_streamer.py` |

---

## Optional Extensions (All Implemented)

| Extension | Implementation |
|-----------|---------------|
| 2D-to-3D lifting model | `DeepFusionPose` — Transformer + BiLSTM trained on H3.6M 3D ground truth |
| Gesture recognition | `gesture_recognizer.py` — 5 gesture classes, rule-based, real-time |
| Future wearable exoskeleton | `exoskeleton_streamer.py` — APPLY/HOLD calibration packets on port 9001 |

---

## Performance Targets (Specification)

| Metric | Target | Status |
|--------|--------|--------|
| End-to-end latency | < 100 ms | Run `python scripts/benchmark_latency.py` |
| Frame rate | ≥ 20 fps | Live overlay in `run_demo.py` |
| Static pose variance | ≤ ±3–5° after filtering | Kalman filter achieves this |
| Continuous operation | ≥ 10 minutes | Tested via `run_demo.py` |

---

## Project Phases

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Data infrastructure & frame extraction | ✅ |
| 2 | H3.6M ground truth angle generation | ✅ |
| 3 | MediaPipe / MoveNet / PoseNet BiLSTM baselines | ✅ |
| 4 | DeepFusionPose (Transformer + CrossAttn + BiLSTM) | ✅ |
| 5 | Bayesian MC Dropout uncertainty | ✅ |
| 6 | GAN temporal refinement | ✅ |
| 7 | Occlusion robustness benchmark | ✅ |
| 8 | Unity Digital Twin (4-avatar) | ✅ |
| 9 | Exoskeleton calibration stream | ✅ |
| 10 | Ablation study (5 model configurations) | ✅ |

---

## Filters Implemented

| Filter | Class | Best for |
|--------|-------|---------|
| Kalman 2-State | `KalmanFilter2State` | Real-time, optimal for noisy angles (angle + velocity) |
| Exponential MA | `ExponentialMovingAverageFilter` | Low CPU, fast response |
| Moving Average | `MovingAverageFilter` | Simplest baseline |
| Savitzky–Golay | `SavitzkyGolayFilter` | Best smoothness, slight lag |

---

## Gesture Classes

| Gesture | Trigger condition |
|---------|-----------------|
| `REST` | All joints near neutral > 1.5 s |
| `RAISE_ARM` | shoulder_pitch > 70° sustained 0.8 s |
| `REACH_FORWARD` | shoulder_yaw > 45° sustained 0.6 s |
| `ELBOW_FLEX` | elbow_flexion < 50° sustained 0.5 s |
| `WAVE` | Elbow oscillates ≥ 3 cycles in 1.5 s |

---

## Unity Scene Setup

1. Open `Unity/UnityMedia` in Unity Editor
2. Menu → **PoseTrack → Build 4-Avatar Scene**
3. Press **Ctrl+S** to save
4. Press **▶ Play**
5. Run `scripts/run_demo.py` (or `scripts/mock_streamer.py` for testing)

---

## Communication Protocol

```
Port 9000 (avatar control):
  MP,<pitch>,<roll>,<yaw>,<elbow>              — MediaPipe raw
  MV,<pitch>,<roll>,<yaw>,<elbow>              — MoveNet raw
  FU,<pitch>,<roll>,<yaw>,<elbow>,<uncertainty>— Fusion + MC Dropout
  GR,<pitch>,<roll>,<yaw>,<elbow>              — GAN refined

Port 9001 (exoskeleton calibration):
  JSON per joint: {frame, timestamp, joint, angle, confidence, uncertainty, action}
  action = "APPLY" (confidence > 0.85) or "HOLD" (use last safe value)
```

---

## Calibration Procedure

1. In `run_demo.py`, press **C**
2. Follow 3 on-screen prompts:
   - Hold arm **down** at side → press SPACE
   - Extend arm **forward** horizontally → press SPACE
   - Bend elbow to **90°** → press SPACE
3. Calibration saved to `outputs/calibration.json`
4. Applied automatically to all subsequent angle estimates

---

## Data Logging

```bash
# During demo: press L to start/stop logging
# Log files saved to outputs/sessions/session_<timestamp>.csv

# Plot saved angle history
python scripts/plot_angles.py --csv outputs/sessions/session_<timestamp>.csv --comparison
```

---

## Latency Benchmark

```bash
python scripts/benchmark_latency.py --frames 300

# Output example:
# Component       Mean ms    P50 ms    P95 ms    Max ms
# capture            2.31      1.98      4.12     12.40
# mediapipe         28.54     27.21     36.80     52.10
# angles             0.41      0.38      0.89      2.10
# filter             0.03      0.02      0.08      0.22
# udp                0.12      0.10      0.28      1.20
# END-TO-END        31.41     30.12     42.80     66.10
# < 100 ms: 100.0% of frames pass
```

---

## References

Full bibliography with implementation notes → **[REFERENCES.md](REFERENCES.md)**

Key citations:

| ID | Source | Used for |
|----|--------|----------|
| R1 | Google — MediaPipe Pose | Primary pose framework |
| R2 | Google — MoveNet | Baseline comparison |
| R3 | Papandreou et al. — PoseNet | Baseline comparison |
| R4–R5 | Unity Technologies | Avatar rig, Quaternion rotations |
| R8 | Koritnik et al. | Two-link kinematic arm model |
| R9 | Biryukova et al. — J. Biomechanics 2000 | Shoulder angle decomposition |
| R10 | DergiPark filtering survey | Filter selection rationale |
| R11 | Van Biezen — Kalman lectures | `KalmanFilter2State` implementation |
| R12 | Savitzky–Golay tutorial | `SavitzkyGolayFilter` implementation |
| R13 | Vaswani et al. — Attention Is All You Need | Transformer encoder in Fusion model |
| R14 | Ionescu et al. — Human3.6M | Ground-truth training data |
| R15 | Gal & Ghahramani — MC Dropout | Bayesian uncertainty estimation |
| R16 | Goodfellow et al. — GAN | Temporal GAN refinement |
| R17 | Maciejasz et al. — Rehab Robotics Survey | Exoskeleton motivation |
| R18 | Polygerinos et al. — Soft Robotics | APPLY/HOLD gating logic |
