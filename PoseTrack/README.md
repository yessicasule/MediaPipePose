# PoseTrack — Pose Estimation Pipeline & Unity Avatar Integration

Real-time upper-body pose tracking with multi-framework benchmarking (MediaPipe, PoseNet, MoveNet) and live UDP streaming to a Unity humanoid avatar.

This README is organized by your 3-month milestones:
- Month 1: compare frameworks + choose one
- Month 2: compute joint angles reliably
- Month 3: stream angles into Unity and animate an avatar

---

## Project Structure

```
PoseTrack/
├── config/config.py              # Camera & MediaPipe settings
├── src/
│   ├── pose_tracker.py           # MediaPipe live tracker + angle overlay + UDP stream
│   ├── joint_angle_estimator.py  # Elbow flexion & shoulder angle computation
│   ├── angle_filter.py           # MovingAverage / Savitzky-Golay / Kalman filters
│   ├── udp_angle_streamer.py     # Background-threaded UDP sender
│   ├── calibration.py            # Reference-pose calibration routine
│   ├── synchronized_recorder.py  # Synchronized video + motion-capture recorder
│   ├── data_recorder.py          # CSV/JSON landmark data writer
│   └── video_recorder.py         # OpenCV video writer
├── benchmarks/
│   ├── run_mediapipe_on_frames.py
│   ├── run_movenet_on_frames.py
│   ├── posenet_tfjs/             # PoseNet via Node.js / TF.js
│   ├── run_all_benchmarks.py     # Unified benchmark runner
│   ├── visualize_benchmarks.py   # Comparison plots & dashboard
│   └── extract_frames.py
├── udp_angle_sender/
│   └── angle_generator.py        # Standalone simulated angle transmitter
├── Unity/PoseTrackReceiver/
│   ├── UdpAngleReceiver.cs       # UDP listener → ArmAngles struct
│   ├── ArmAngleController.cs     # Applies rotations to humanoid bones
│   ├── AngleSmoother.cs          # Exponential moving-average (angle-safe)
│   └── ArmRigSetup.cs            # Editor helper for bone assignment
├── run_complete_workflow.py      # End-to-end: record → analyze → benchmark → compare
├── run_benchmark_workflow.py     # Benchmark-only orchestrator
├── main.py                       # Quick-start pose tracking
└── outputs/
    ├── videos/  data/  plots/
    └── benchmarks/<session>/results/visualizations/
```

---

## Setup

### Python Environment

```bash
cd PoseTrack
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### MediaPipe Model Files

Download pose landmarker task files and place them in `models/`:

```
models/pose_landmarker_lite.task
models/pose_landmarker_full.task
models/pose_landmarker_heavy.task
```

Download from: https://developers.google.com/mediapipe/solutions/vision/pose_landmarker

### PoseNet (Node.js)

```bash
cd benchmarks/posenet_tfjs
npm install
```

Note: on Windows, PoseNet via pure TFJS is slower than native `tfjs-node`, but works without a native build toolchain.

---

## Usage

### 1 — Quick Pose Tracking (MediaPipe only)

```bash
python main.py
```

Press `SPACE` to start/stop recording. Press `q` to quit.
Joint angles (elbow flexion, shoulder elevation, horizontal) are displayed live on the video.

### 2 — Full Workflow: Record → Analyze → Benchmark → Compare

```bash
python run_complete_workflow.py
```

Guides you through 8 steps: system check, camera test, recording, analysis, report, frame extraction, benchmark, and comparison visualizations.

### 3 — Benchmark Existing Session (all three frameworks)

```bash
python run_benchmark_workflow.py --mode benchmark \
    --frames_dir outputs/benchmarks/<session>/frames \
    --session_name <session>
```

If you prefer running the benchmark steps manually:

```bash
python -m benchmarks.extract_frames --video outputs/videos/<video>.mp4 --out_dir benchmarks/frames/<session> --stride 2
python -m benchmarks.run_mediapipe_on_frames --frames_dir benchmarks/frames/<session> --out_json benchmarks/results/mediapipe.json
python -m benchmarks.run_movenet_on_frames --frames_dir benchmarks/frames/<session> --out_json benchmarks/results/movenet.json --model lightning
cd benchmarks/posenet_tfjs
node run_posenet_on_frames.mjs --frames_dir ../frames/<session> --out_json ../results/posenet.json
cd ../..
python -m benchmarks.evaluate_frameworks --results benchmarks/results/mediapipe.json benchmarks/results/movenet.json benchmarks/results/posenet.json --out_json benchmarks/results/evaluation.json
```

The evaluation JSON includes:
- FPS / latency
- elbow keypoint jitter (frame-to-frame mean step)
- elbow angle stability proxy (std dev during the first N frames)
- robustness proxy (fraction of frames with confident shoulder+elbow+wrist)

### 4 — Run Frameworks Individually

```bash
python -m benchmarks.run_mediapipe_on_frames \
    --frames_dir <frames_dir> --out_json results/mediapipe.json

python -m benchmarks.run_movenet_on_frames \
    --frames_dir <frames_dir> --out_json results/movenet.json

cd benchmarks/posenet_tfjs
node run_posenet_on_frames.mjs \
    --frames_dir ../../<frames_dir> --out_json ../results/posenet.json
```

### 5 — UDP Angle Sender (simulated data for Unity testing)

```bash
python udp_angle_sender/angle_generator.py --host 127.0.0.1 --port 9000 --hz 30
```

### 6 — Live Streaming to Unity (real angles from camera)

```python
from src.pose_tracker import PoseTracker

tracker = PoseTracker()
tracker.start_streaming(host="127.0.0.1", port=9000, hz=30)
tracker.run()
```

### 7 — Calibration

```python
import cv2, mediapipe as mp
from src.calibration import run_calibration

cap  = cv2.VideoCapture(0)
pose = mp.solutions.pose.Pose()
params = run_calibration(cap, pose)
print(params)
```

Saves calibration data to `outputs/data/calibration.json`.

---

## Benchmark Comparison

After running benchmarks, comparison charts are saved to `outputs/benchmarks/<session>/results/visualizations/`:

| File | Content |
|---|---|
| `latency_comparison.png` | Per-frame inference latency |
| `fps_latency_comparison.png` | FPS and mean/P90 latency bars |
| `accuracy_comparison.png` | Keypoint confidence scores |
| `benchmark_dashboard.png` | Full summary dashboard + radar chart |
| `benchmark_report.txt` | Text report with fastest/most accurate framework |

---

## Communication Protocol

All angle data is transmitted as UTF-8 UDP datagrams, one pose per packet:

```
S,shoulder_pitch,shoulder_yaw,shoulder_roll,elbow_flex\n
```

- All values in **degrees**
- Update rate: **≥ 30 Hz**
- Shoulder pitch: arm lift in sagittal plane (elevation)
- Shoulder yaw: horizontal rotation
- Shoulder roll: internal/external rotation
- Elbow flex: flexion angle (0 = straight, 90 = bent 90°)

---

## Unity Setup

### Prerequisites

- Unity 2022 LTS or later
- Project configured with **.NET Standard 2.1** or **.NET Framework 4.x** (for `System.Net.Sockets`)
- A Humanoid avatar configured in the Animator (**Avatar Definition** set to **Create From This Model**, Rig type: **Humanoid**)

### Step 1 — Import Scripts

Copy the `Unity/PoseTrackReceiver/` folder into `Assets/PoseTrackReceiver/` in your Unity project.

### Step 2 — Scene Setup

1. Select the **root GameObject** of your humanoid avatar.
2. Add Component → `UdpAngleReceiver`. Set **Listen Port** to `9000`.
3. Add Component → `ArmAngleController`.
4. In the `ArmAngleController` Inspector, click **"Auto-Find Humanoid Bones"** — this reads the Animator's Humanoid bone mapping and assigns `upperArmBone` (Left Upper Arm) and `lowerArmBone` (Left Lower Arm) automatically.
5. If you need the right arm, duplicate the component and change `upperArmBone` / `lowerArmBone` to the right-side bones.

### Step 3 — Coordinate Axis Adjustment

Unity's bone local axes differ per avatar. Use `shoulderAxisOffset` and `elbowAxisOffset` in the Inspector to rotate the neutral position so the arm looks natural at all-zero input angles.

Typical starting values (adjust to your avatar):

```
shoulderAxisOffset = (0, 0, -90)    // arm pointing down in T-pose
elbowAxisOffset    = (0, 0, 0)
```

### Step 4 — Firewall

On Windows, Unity must be allowed through the firewall to receive UDP on port 9000, or use `127.0.0.1` for local testing.

### Step 5 — Run

1. Start the UDP sender: `python udp_angle_sender/angle_generator.py`
   (or `python main.py` for live camera angles)
2. Press **Play** in Unity — the avatar arm will animate smoothly.

### Smoothing

`ArmAngleController.smoothing` controls the exponential moving average factor (0.01 = very smooth / slow, 1.0 = no smoothing / instantaneous).

---

## Filtering & Calibration Notes

Three filters are available in `src/angle_filter.py`:

| Filter | Class | When to use |
|---|---|---|
| Moving Average | `MovingAverageFilter(window=7)` | Simple, low-latency noise reduction |
| Savitzky-Golay | `SavitzkyGolayFilter(window=11, poly=3)` | Preserves peaks, good for fast motion |
| Kalman 1D | `KalmanFilter1D(process_noise, measurement_noise)` | Best static-pose stability (±3–5°) |

The live tracker uses Kalman by default. Tune `process_noise` (motion model trust) and `measurement_noise` (sensor trust) to balance lag vs. stability.

---

## Exoskeleton Calibration Reference

The joint angle outputs from `src/joint_angle_estimator.py` can serve as **reference signals** for calibrating wearable exoskeleton joint sensors:

1. Run `src/calibration.py` to record reference poses with the vision system.
2. Simultaneously record the exoskeleton encoder readings for the same poses.
3. Use the vision-derived angles as ground truth to compute offset and gain for each exoskeleton sensor axis.
4. Store the mapping in a calibration file alongside `calibration.json`.

This creates a traceable calibration chain: camera → vision angle → exoskeleton sensor.

---

## Dependencies

```bash
pip install mediapipe>=0.10.9 opencv-python>=4.8.1 numpy>=1.24 \
            pandas>=2.0 matplotlib>=3.7 psutil>=5.9 Pillow>=10.0 \
            tensorflow>=2.14 tensorflow-hub>=0.15
```

Node.js (for PoseNet): https://nodejs.org/
