# MonoArm — Project Vision Document

> **Monocular Vision-Based Estimation of Human Arm Joint Angles for Real-Time Digital Avatar Control**

---

## 1. Project Identity

| Field | Value |
|---|---|
| **Project Name** | MonoArm |
| **Root Directory** | `D:\MonoArm` |
| **Python Module** | `D:\MonoArm\python\` |
| **Unity Project** | `D:\MonoArm\unity\` |
| **Language (Python)** | Python 3.10, venv |
| **Language (Unity)** | C# (Unity 6, Editor 6000.4.3f1) |
| **Target Hardware** | CPU-only student laptop, standard RGB webcam |
| **Conference Format** | IEEE (robotics/rehabilitation/biomechanics) |

---

## 2. Purpose & Motivation

Build a **real-time monocular vision-based system** that:

1. Captures video from a single RGB camera
2. Detects 2D human body keypoints using a pose estimation framework
3. Computes **anatomically consistent arm joint angles** (shoulder and elbow)
4. Transmits angles via UDP to Unity
5. Drives a humanoid avatar's arm in real-time

The system serves as a **foundational platform for future integration with a soft wearable arm exoskeleton**. Vision-derived joint angles will act as reference signals for calibrating and validating wearable exoskeleton sensors. The hardware exoskeleton project is separate and its details are currently unavailable.

### Application Domains
- Rehabilitation training
- Assistive arm support
- Human-machine interaction
- Teleoperation
- Digital human modeling

---

## 3. Arm & Joint Scope

### Target Arm
- **Right arm** initially
- Architecture must be designed so **adding the left arm is straightforward** (modular, parameterized by side)

### Joint Angles Estimated

| Joint | DOF | Angle | Description |
|---|---|---|---|
| **Shoulder** | 1 | Flexion-Extension | Arm forward/backward in sagittal plane |
| **Shoulder** | 2 | Abduction-Adduction | Arm out to side/in in coronal plane |
| **Shoulder** | 3 | Internal-External Rotation | Twisting of upper arm about its own longitudinal axis |
| **Elbow** | 1 | Flexion | Bending of forearm relative to upper arm |

**Total: 4 DOFs (3 shoulder + 1 elbow)**

### Angle Representation
- **Anatomically consistent** clinical angles, NOT generic Euler pitch/yaw/roll
- Shoulder flexion-extension, abduction-adduction: **reliably estimable from monocular 2D vision** — these are the primary outputs
- Internal-external rotation: **attempted using MediaPipe's pseudo-3D depth + forearm orientation as proxy**, with rigorous documentation of accuracy bounds and failure cases
- Elbow flexion: **highly reliable from vision**

### Kinematic Model
- Simplified **two-link kinematic chain**
- Upper arm: shoulder to elbow (rigid segment, constant length per user)
- Forearm: elbow to wrist (rigid segment, constant length per user)
- Small variations due to tracking noise are ignored
- Focus on **angle estimation**, not precise 3D joint position reconstruction
- Objective: consistent, smooth angle estimates suitable for real-time avatar animation and reference motion signals — **not medical-grade measurement accuracy**

---

## 4. Camera & Environment Constraints

| Parameter | Value |
|---|---|
| Camera type | Standard RGB webcam or laptop camera |
| Camera position | Fixed (desk/tripod) |
| Distance | 1–2 meters from user |
| View angle | **Both frontal and side/oblique** — system must handle both |
| User posture | **Both seated and standing** — system must handle both |
| Depth sensors | None (monocular only) |
| Markers/wearables | None required |

---

## 5. System Architecture

### 5.1 Subsystem Overview

```
┌─────────────────────┐    UDP (≥20-30 Hz)    ┌─────────────────────┐
│   Python Module      │ ──────────────────►  │   Unity Module       │
│                      │                       │                      │
│  • Camera Capture    │   Packet Format:      │  • UDP Receiver      │
│  • Pose Estimation   │   S,sh_flex,sh_abd,   │  • Angle Parser      │
│  • Angle Computation │     sh_rot,elb_flex\n │  • Smoothing Filter  │
│  • Temporal Filtering│                       │  • Avatar Rig Control│
│  • Data Logging      │                       │  • Calibration       │
│  • Real-time Plots   │                       │                      │
└─────────────────────┘                       └─────────────────────┘
```

### 5.2 Vision Module (Python)

**Responsibilities:**
1. Capture video frames from webcam
2. Run pose estimation to extract upper-body keypoints (shoulder, elbow, wrist, hip)
3. Compute anatomically consistent joint angles using vector geometry
4. Apply temporal filtering (moving average, Savitzky-Golay, Kalman)
5. Transmit filtered angles over UDP
6. Log all data to CSV files
7. Display real-time angle plots and annotated video feed

**Pose Estimation Frameworks to Evaluate:**
1. MediaPipe Pose (Google) — CPU-optimized, provides pseudo-3D landmarks
2. MoveNet (TensorFlow Hub) — Lightning variant for CPU
3. PoseNet (TensorFlow.js / TFLite) — lightweight

**Evaluation Criteria for Framework Selection:**
- Frame rate (FPS) on CPU
- Keypoint jitter (stability)
- Stability of computed elbow angles during static poses
- Robustness during arm motion
- Computational load (CPU %)
- Quality of depth estimates (if available)

**Outcome:** Select the single best framework and build the final system around it. Document the comparison in the paper.

### 5.3 Processing Module (within Python)

**Joint Angle Computation:**
- Vector geometry on 2D/pseudo-3D keypoints
- Anatomical coordinate frame definition relative to torso
- Shoulder angles via decomposition of upper arm orientation vector
- Elbow flexion via angle between upper arm and forearm vectors

**Temporal Filtering (implemented and compared):**
1. Moving Average filter
2. Savitzky-Golay filter
3. Kalman filter

**Filtering Target:** Static pose angle variance ≤ ±3–5 degrees after filtering.

### 5.4 Unity Module

**Responsibilities:**
1. Receive UDP packets containing joint angles
2. Parse angle data
3. Apply smoothing filter for frame-rate independent animation
4. Map angles to humanoid avatar rig using `Transform.localRotation` and `Quaternion.Euler`
5. Support calibration (reference poses → mapping parameters)
6. Visualize avatar arm motion in real-time

**Avatar:** Free humanoid model with proper Humanoid rig (Unity-chan, Mixamo, or equivalent)

**Unity Version:** Unity 6 (6000.4.3f1)

### 5.5 Communication Protocol

| Parameter | Value |
|---|---|
| Protocol | UDP |
| Update Rate | ≥ 20–30 Hz |
| Packet Terminator | Newline character `\n` |
| Format | `S,shoulder_flex,shoulder_abd,shoulder_rot,elbow_flex\n` |
| Units | Degrees |
| Coordinate Convention | Defined relative to Unity avatar coordinate frame during integration |

---

## 6. Calibration System

**Calibration Routine:**
1. User performs reference poses:
   - Arm down (anatomical position)
   - Arm forward (90° flexion)
   - Arm out to side (90° abduction)
   - Elbow flexed at 90°
2. System records angle values at each pose
3. Computes mapping parameters (offset, scale) between estimated angles and avatar joint limits
4. Parameters stored to file and applied during operation

**Future Use:** Calibration pipeline documented for use as reference signals for wearable exoskeleton joint sensor calibration.

---

## 7. Evaluation Strategy

### 7.1 Ground Truth Validation
- Use existing datasets with ground truth joint angles (e.g., **Human3.6M**, **CMU Panoptic**)
- Run pose estimator on dataset images
- Compute angles with our pipeline
- Compare against dataset ground truth angles
- **No model retraining** — evaluation of the angle computation pipeline only

### 7.2 Reference Pose Validation
- Known poses: arm straight (180°), arm bent at 90°, etc.
- Quantitative comparison of estimated vs. expected angles

### 7.3 Qualitative Assessment
- Smoothness of avatar motion
- Stability during static holds
- Robustness under moderate motion and partial occlusion
- System stability over 10+ minute continuous sessions

### 7.4 Filter Comparison
- Unfiltered vs. Moving Average vs. Savitzky-Golay vs. Kalman
- Metrics: variance reduction, lag introduced, tracking responsiveness

---

## 8. Performance Requirements

| Metric | Target |
|---|---|
| End-to-end latency | < 100 ms |
| Vision module FPS | ≥ 20 FPS |
| Avatar motion | Smooth, no visible jitter after filtering |
| Static pose stability | Within ±3–5 degrees |
| Continuous operation | ≥ 10 minutes without crash |
| Processing | CPU-only on student laptop |

---

## 9. Data Logging & Visualization

### Real-Time (During Operation)
- Live annotated video feed with keypoints overlay
- Real-time joint angle plots (matplotlib or OpenCV overlay)

### Post-Processing (After Sessions)
- CSV files with timestamped joint angle data
- Session metadata (framework used, filter settings, calibration params)
- Tools for generating publication-quality figures

---

## 10. Deliverables

1. **Real-time vision-based arm tracking application** (Python CLI)
2. **Pose framework comparison module** with evaluation scripts
3. **Joint angle estimation and filtering module** (with all three filters)
4. **Unity application** with avatar arm control
5. **Calibration module** (Python + Unity)
6. **Data generator application** (for testing Unity independently)
7. **Joint angle data logging and visualization tools**
8. **Ground truth validation pipeline** (against Human3.6M or similar)
9. **Technical documentation** — paper-ready, IEEE format, with full mathematical derivations
10. **Video recordings** demonstrating system operation

---

## 11. Paper Contribution & Framing

**Core Contribution:**
A monocular vision-based framework for estimating anatomically consistent arm joint angles in real time, with:
- Rigorous comparison of pose estimation frameworks on CPU
- Full mathematical derivation of angle computation from 2D/pseudo-3D keypoints
- Comparison of temporal filtering approaches
- Honest documentation of what monocular vision can and cannot reliably estimate (especially internal-external rotation)
- Demonstration via real-time avatar control
- Design for future integration with wearable exoskeleton sensor calibration

**Mathematical Content Required:**
- Kinematic chain model derivation
- Vector geometry for angle computation
- Anatomical coordinate frame definitions
- Filter equations (moving average, Savitzky-Golay, Kalman state-space formulation)
- Error metrics and statistical analysis

---

## 12. Optional Extensions (Not in Core Scope)

- 2D-to-3D lifting model for improved shoulder rotation estimation
- Gesture recognition from arm motion patterns
- VR/AR environment integration
- Direct wearable exoskeleton sensor fusion

---

## 13. Technical Constraints & Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Pose framework | Evaluate 3, pick 1 | Quality over complexity |
| Shoulder representation | Anatomical (flex/abd/rot) | Clinical relevance for rehab |
| Communication | UDP | Low latency for real-time |
| Processing | CPU-only | Student laptop constraint |
| Python version | 3.10 | Already available, compatible |
| Unity version | 6 (6000.4.3f1) | Already installed |
| Avatar | Free model | Budget constraint |
| Arms tracked | Right arm (left extensible) | Incremental scope |
| Camera views | Frontal + side | Robustness requirement |
| Paper format | IEEE conference | Standard for domain |
| Evaluation | Ground truth dataset + reference poses | No model retraining |

---

## 14. Known Limitations to Document

1. **Internal-external rotation** accuracy is fundamentally limited by monocular vision
2. **Depth ambiguity** in monocular camera — mitigated by pseudo-3D but not eliminated
3. **Simplified kinematic model** — not anatomically precise shoulder complex
4. **Partial occlusion** degrades tracking (arm behind body, self-occlusion)
5. **View-dependent accuracy** — different camera angles observe different rotational components
6. **Not medical-grade** — suitable for reference signals, not clinical measurement
