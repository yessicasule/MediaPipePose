# PoseTrack — Monocular Vision-Based Arm Joint Angle Estimation

Real-time upper-body pose tracking from a single RGB camera. Computes shoulder and elbow joint angles, streams them over UDP to a Unity humanoid avatar, and provides a full benchmarking pipeline for comparing pose estimation frameworks (MediaPipe, MoveNet, PoseNet).

Designed as a vision-based reference layer for future soft wearable arm exoskeleton calibration and validation.

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────┐
│  VISION MODULE  (src/pose/, src/capture/)                    │
│                                                              │
│  RGB Camera  ──►  MediaPipe Pose  ──►  33 Body Landmarks    │
└────────────────────────────┬─────────────────────────────────┘
                             │ landmarks[11,12,13,14,15,16,23,24]
                             ▼
┌──────────────────────────────────────────────────────────────┐
│  PROCESSING MODULE  (src/processing/)                        │
│                                                              │
│  Vector Geometry  ──►  Raw Angles  ──►  Kalman / EMA Filter │
│                                                              │
│  Outputs (degrees):                                          │
│    shoulder_elevation   shoulder_yaw                         │
│    shoulder_roll        elbow_flexion                        │
└────────────────────────────┬─────────────────────────────────┘
                             │ UDP  "S,pitch,yaw,roll,elbow\n"
                             │ 127.0.0.1:9000  @30 Hz
                             ▼
┌──────────────────────────────────────────────────────────────┐
│  UNITY MODULE  (Unity/PoseTrackReceiver/)                    │
│                                                              │
│  UdpAngleReceiver  ──►  ArmAngleController                  │
│  (background thread)    (Quaternion.Euler on bones)         │
│                                                              │
│  PoseDebugUI  ──►  on-screen angle readout  (4 Hz refresh)  │
└──────────────────────────────────────────────────────────────┘
```

### Angle Definitions

| Angle | Joint | Description |
|---|---|---|
| `shoulder_elevation` | Shoulder (pitch) | Arm lift angle relative to torso, 0° = arm down |
| `shoulder_yaw` | Shoulder (yaw) | Horizontal rotation of upper arm |
| `shoulder_roll` | Shoulder (roll) | Axial rotation of upper arm |
| `elbow_flexion` | Elbow | Bend angle, 0° = fully extended |

All values transmitted and stored in degrees.

---

## Repository Structure

```
MediaPipePose/
└── PoseTrack/
    ├── config/
    │   └── config.py                  # paths, UDP settings, camera config
    ├── scripts/
    │   ├── run_live.py                # real-time camera → filter → UDP stream
    │   ├── run_capture_session.py     # simultaneous video record + angle log + stream
    │   ├── plot_angles.py             # post-session raw vs filtered angle plots
    │   └── data_generator.py         # simulated angles for Unity testing
    ├── src/
    │   ├── pose/
    │   │   ├── mediapipe_runner.py    # MediaPipe Pose wrapper
    │   │   ├── movenet_runner.py      # MoveNet (TF Hub) wrapper
    │   │   └── posenet_runner.py      # PoseNet wrapper
    │   ├── processing/
    │   │   ├── joint_angle_estimator.py  # vector geometry → angles
    │   │   ├── angle_filter.py           # MovingAverage, EMA, Kalman1D
    │   │   ├── calibration.py            # reference pose calibration
    │   │   └── angle_logger.py           # CSV logging + matplotlib plots
    │   ├── streaming/
    │   │   └── udp_streamer.py        # threaded UDP sender
    │   └── capture/
    │       └── video_recorder.py      # OpenCV VideoWriter wrapper
    ├── benchmarks/
    │   ├── extract_frames.py             # video → JPG frames
    │   ├── run_mediapipe_on_frames.py    # MediaPipe benchmark → JSON
    │   ├── run_movenet_on_frames.py      # MoveNet benchmark → JSON
    │   ├── run_all_benchmarks.py         # orchestrates all frameworks
    │   ├── render_comparison_video.py    # side-by-side skeleton video
    │   ├── visualize_benchmarks.py       # latency/accuracy plots
    │   └── posenet_tfjs/                 # Node.js PoseNet runner
    └── Unity/
        └── PoseTrackReceiver/
            ├── UdpAngleReceiver.cs    # background-thread UDP receiver
            ├── ArmAngleController.cs  # bone rotation driver
            ├── AngleSmoother.cs       # per-joint lerp smoother
            ├── ArmRigSetup.cs         # editor helper — auto-find humanoid bones
            └── PoseDebugUI.cs         # on-screen debug display (no Debug.Log)
```

---

## Installation

### Python

```bash
cd PoseTrack
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / Mac

pip install opencv-python mediapipe numpy tensorflow tensorflow-hub
pip install matplotlib pillow psutil   # for plots and benchmarks
```

### PoseNet (optional, Node.js)

```bash
cd PoseTrack/benchmarks/posenet_tfjs
npm install
```

---

## Python Pipeline

### A — Real-time live stream to Unity

```bash
cd PoseTrack
python scripts/run_live.py --filter kalman
```

Options:

| Flag | Default | Description |
|---|---|---|
| `--filter` | `kalman` | `kalman` or `ema` |
| `--camera` | `0` | Camera index |
| `--host` | `127.0.0.1` | Unity UDP IP |
| `--port` | `9000` | Unity UDP port |
| `--video path.mp4` | — | Use a video file instead of camera |

Press `q` to stop.

---

### B — Capture session (recommended for data collection)

Records video and joint angle data simultaneously, then auto-generates angle plots on exit.

```bash
python scripts/run_capture_session.py --session arm_test_01 --save_landmarks
```

Options:

| Flag | Default | Description |
|---|---|---|
| `--session` | `session_<unix>` | Name for the output folder |
| `--filter` | `kalman` | `kalman` or `ema` |
| `--save_landmarks` | off | Save all 33 MediaPipe keypoints per frame |
| `--no_stream` | off | Disable UDP (run without Unity open) |
| `--calib path.json` | — | Load a previously saved calibration |

**Keyboard controls during capture:**

| Key | Action |
|---|---|
| `s` | Toggle video recording on / off |
| `p` | Open / close live rolling angle plot window |
| `c` then `SPACE` | Cycle through calibration poses and capture |
| `d` | Toggle debug angle overlay on camera view |
| `q` / `ESC` | Stop session, save files, generate plots |

**Session output** — saved to `data/sessions/<session_name>/`:

```
raw.mp4                  raw camera recording
angles.csv               per-frame: frame_index, timestamp, elapsed_s,
                         inference_ms, pose_detected,
                         raw + filtered values for all 4 joints
landmarks.csv            (--save_landmarks) all 33 MediaPipe keypoints
<session>_angles.png     4-panel raw vs filtered plot, auto-generated
<session>_angles_comparison.png   all-joints overlay comparison
calibration.json         if calibration was performed
```

---

### C — Post-session angle plots

```bash
python scripts/plot_angles.py \
    --csv data/sessions/arm_test_01/angles.csv \
    --comparison
```

Produces:
- `*_angles.png` — 4-panel figure, one joint per row, raw (faint) and filtered (bold) overlaid, ±3° target band shown
- `*_angles_comparison.png` — all joints raw vs filtered on two stacked panels

---

### D — Calibration

Calibration maps the user's neutral arm position to the avatar's coordinate frame.

Run interactively during a capture session:
1. Press `c` — prompted to hold **arm_down** (arm hanging at side)
2. Press `SPACE` — captured
3. Press `c` again — prompted for **arm_forward** (arm horizontal in front)
4. Press `SPACE` — captured
5. Press `c` again — prompted for **elbow_flexed** (90° elbow bend)
6. Press `SPACE` — calibration finalised, saved to `calibration.json`

Reload in future sessions: `--calib data/sessions/arm_test_01/calibration.json`

---

### E — Simulated data generator (Unity testing without camera)

```bash
python scripts/data_generator.py --mode wave --hz 30
```

Modes: `wave` `circle` `static` `random`

---

## Benchmarking Pipeline

The benchmark workflow runs every framework on the **same video frames** for a fair comparison.

### Step 1 — Capture a session

```bash
python scripts/run_capture_session.py --session benchmark_01 --save_landmarks --no_stream
```

### Step 2 — Extract frames

```bash
python benchmarks/extract_frames.py \
    --video data/sessions/benchmark_01/raw.mp4 \
    --out_dir data/sessions/benchmark_01/frames
```

### Step 3 — Run each framework

```bash
python benchmarks/run_mediapipe_on_frames.py \
    --frames_dir data/sessions/benchmark_01/frames \
    --out_json   data/sessions/benchmark_01/results/mediapipe.json

python benchmarks/run_movenet_on_frames.py \
    --frames_dir data/sessions/benchmark_01/frames \
    --out_json   data/sessions/benchmark_01/results/movenet.json \
    --model      lightning
```

For PoseNet (requires Node.js):
```bash
node benchmarks/posenet_tfjs/run_posenet_on_frames.mjs \
    --frames_dir data/sessions/benchmark_01/frames \
    --out_json   data/sessions/benchmark_01/results/posenet.json
```

Or run all at once (Python frameworks only):
```bash
python benchmarks/run_all_benchmarks.py \
    --session_name benchmark_01 \
    --frames_dir   data/sessions/benchmark_01/frames
```

### Step 4 — Render side-by-side skeleton video

```bash
python benchmarks/render_comparison_video.py \
    --frames_dir data/sessions/benchmark_01/frames \
    --mediapipe  data/sessions/benchmark_01/results/mediapipe.json \
    --movenet    data/sessions/benchmark_01/results/movenet.json \
    --out        outputs/comparison_benchmark_01.mp4
```

Left panel shows MediaPipe skeleton (green), right panel shows MoveNet skeleton (red), with per-frame latency and confidence score in the header.

### Step 5 — Generate statistical plots

```bash
python benchmarks/visualize_benchmarks.py \
    --results_dir data/sessions/benchmark_01/results \
    --frames_dir  data/sessions/benchmark_01/frames \
    --output_dir  data/sessions/benchmark_01/plots
```

Outputs: latency comparison, FPS bar chart, keypoint score distributions, radar chart, benchmark report text file.

---

## Unity Setup

### Requirements

- Unity **2022.3 LTS** or later
- Universal Render Pipeline (URP) or Built-in Render Pipeline
- A humanoid-rigged avatar (Mixamo `.fbx` recommended)

---

### Step 1 — Create the Unity Project

1. Open Unity Hub → **New Project** → **3D (URP)** or **3D Core** → name it `PoseTrackAvatar`
2. Select Unity 2022.3 LTS as the editor version

---

### Step 2 — Import the C# Scripts

1. In the Project panel, create the folder `Assets/Scripts/PoseTrackReceiver/`
2. Copy all four files from `PoseTrack/Unity/PoseTrackReceiver/` into that folder:
   - `UdpAngleReceiver.cs`
   - `ArmAngleController.cs`
   - `AngleSmoother.cs`
   - `ArmRigSetup.cs`
   - `PoseDebugUI.cs`
3. Unity will compile automatically. Resolve any errors before continuing.

---

### Step 3 — Import the Avatar

1. Export your avatar from [Mixamo](https://www.mixamo.com):
   - Upload your character or choose one from the library
   - Download with **T-pose**, format **FBX for Unity**, skin **With Skin**
2. Drag the `.fbx` file into `Assets/Models/`
3. Select the model in the Project panel → **Inspector → Rig tab**:
   - Set **Animation Type** → `Humanoid`
   - Click **Configure** — Unity will map bones automatically
   - Verify that `LeftUpperArm` and `LeftLowerArm` (or Right, depending on which arm you track) show green ticks
   - Click **Done**, then **Apply**

> **Pixelation fix:** Select the avatar's texture files in `Assets/` → Inspector → set **Max Size** to `2048` or higher, **Compression** to `High Quality`, click **Apply**.

---

### Step 4 — Set Up the Scene

#### 4a — Place the avatar

1. Drag the imported avatar prefab from `Assets/Models/` into the **Hierarchy**
2. Set **Transform Position** to `(0, 0, 0)`, **Rotation** to `(0, 0, 0)`
3. Confirm the avatar is standing upright in the Scene view

#### 4b — Create the receiver GameObject

1. In the Hierarchy, right-click → **Create Empty** → name it `PoseReceiver`
2. With `PoseReceiver` selected, in the Inspector click **Add Component**:
   - Add **`UdpAngleReceiver`** — set `Listen Port` to `9000`
   - Add **`ArmAngleController`**

#### 4c — Assign bones

1. Select `PoseReceiver` in the Hierarchy
2. In the `ArmAngleController` component, click **Auto-Find Humanoid Bones**
   - This walks up the hierarchy to find the `Animator` on the avatar and fills in `Upper Arm Bone` and `Lower Arm Bone` automatically
   - If it fails, drag the bones manually: expand the avatar in the Hierarchy, locate `LeftUpperArm` and `LeftLowerArm` (under `Hips > Spine > ... > LeftShoulder`), drag them into the fields

> **Which arm?** The system currently tracks the left arm (MediaPipe landmark 11 = left shoulder). If your avatar faces you (mirrored), map to the right arm bones instead.

#### 4d — Tune the axis offsets

The coordinate conventions of MediaPipe and Unity may not align out of the box. Use the **Shoulder Axis Offset** and **Elbow Axis Offset** fields in `ArmAngleController` to add a constant rotation to each joint:

| Common correction | Field | Value |
|---|---|---|
| Arm points wrong direction at rest | `shoulderAxisOffset.x` | `±90` |
| Elbow bends backward | `elbowAxisOffset.x` | `±180` |
| Shoulder rotates on wrong axis | Swap pitch/yaw values in Python's `joint_angle_estimator.py` |

Start with `(0, 0, 0)` offsets and run a calibration session. Adjust until arm-down pose produces a neutral T-pose in Unity.

#### 4e — Set smoothing

In `ArmAngleController`, the `Smoothing` slider controls Unity-side lerp (separate from the Python Kalman filter):
- `0.15` — default, responsive
- `0.05` — very smooth, noticeable lag
- `0.30` — snappy, may show jitter if Python filter is insufficient

Both the Python filter and Unity smoother act in series. Generally keep Unity smoothing at `0.10–0.20` and tune the Kalman filter on the Python side.

---

### Step 5 — Add the Debug UI

This displays live angle values on screen **without using `Debug.Log`**, which would stall the render thread.

1. In the Hierarchy, right-click → **UI → Canvas**
   - **Render Mode**: Screen Space – Overlay
2. Right-click the Canvas → **UI → Legacy → Text** (or **TextMeshPro → Text** if TMP is installed)
   - Set **Font Size** `14`, **Color** white, position it in a corner
   - Expand the Rect Transform to fill the needed area
3. Create another **Empty** child of the Canvas → name it `DebugUI`
4. Add Component → **`PoseDebugUI`**:
   - **Receiver Obj** → drag `PoseReceiver` from the Hierarchy
   - **Label Text** → drag the Text object you created
   - **Refresh Hz** → `4` (refreshes 4 times per second — no render impact)

> If you use TextMeshPro: add `TMP_PRESENT` to **Project Settings → Player → Scripting Define Symbols**, then use the `labelTmp` field instead of `labelText`.

---

### Step 6 — Network Configuration

The Python app sends to `127.0.0.1:9000` by default (same machine). If Python and Unity run on **different machines**:

1. In `UdpAngleReceiver`, `listenPort` stays `9000` (receiving end listens on all interfaces)
2. In Python, pass `--host <Unity-machine-IP>`: 
   ```bash
   python scripts/run_live.py --host 192.168.1.50 --port 9000
   ```
3. On Windows, allow UDP 9000 through the firewall:
   - Windows Defender Firewall → Advanced Settings → Inbound Rules → New Rule → Port → UDP 9000

---

### Step 7 — Run the System

1. Start Unity → press **Play**
2. In a terminal:
   ```bash
   cd PoseTrack
   python scripts/run_live.py --filter kalman
   ```
3. Stand in front of the camera — the avatar arm should mirror your arm within ~1–2 seconds

For a full data-logging session:
```bash
python scripts/run_capture_session.py --session demo_01
```

---

## UDP Protocol

```
S,<shoulder_pitch>,<shoulder_yaw>,<shoulder_roll>,<elbow_flex>\n
```

- All values in degrees (float, 3 decimal places)
- Packet ends with `\n`
- Rate: 30 Hz (configurable via `Config.OUTPUT_VIDEO_FPS`)
- Transport: UDP IPv4, default port 9000

Example packet:
```
S,42.317,-8.124,3.501,67.890
```

---

## Performance Targets

| Metric | Target | How to check |
|---|---|---|
| End-to-end latency | < 100 ms | `inference_ms` column in `angles.csv` |
| Vision module FPS | ≥ 20 FPS | Shown in capture window overlay |
| Static pose variance | ±3–5° (filtered) | ±3° band in `plot_angles.py` output |
| Continuous runtime | ≥ 10 min | Session timer in capture overlay |

---

## Troubleshooting

### Camera will not open
- Windows: **Settings → Privacy → Camera** — ensure desktop app access is ON
- Close any app holding the camera (Teams, Zoom, OBS)
- Try `--camera 1` or `--camera 2`
- Use a saved video instead: `--video path/to/file.mp4`

### No data received in Unity
- Confirm Python output shows `Streaming to Unity at 127.0.0.1:9000`
- Check Windows Firewall — allow UDP port 9000 inbound
- Confirm `Listen Port` in `UdpAngleReceiver` matches `--port` in Python

### Avatar arm jerks or spins wildly
- The axis offset is wrong — adjust `shoulderAxisOffset` in `ArmAngleController`
- Run a calibration session to anchor the neutral pose
- Increase Unity `Smoothing` temporarily to diagnose whether it is an angle magnitude issue or a sign issue

### Avatar appears pixelated during motion
- Select avatar textures in Project panel → Inspector → increase **Max Size** to `2048`, set **Filter Mode** to `Bilinear` or `Trilinear`, **Apply**
- Check that the avatar's mesh **LOD** is not switching to a lower level — disable LOD Group component if present

### Unity render stutters when Python is running
- Never use `Debug.Log` inside `Update()`, `LateUpdate()`, or `FixedUpdate()` in any script — each call triggers a full editor console repaint
- All runtime diagnostics must go through `PoseDebugUI` (refreshes at 4 Hz via a timer, never logs to console)
- Confirm `ArmAngleController.Update()` contains only math operations and no string allocations

### High jitter after filtering
- Switch to Kalman filter: `--filter kalman`
- Improve lighting — shadows on joints confuse the pose estimator
- Ensure the full upper body (head to hips) is visible in frame
- Increase `measurement_noise` in `KalmanFilter1D` in `angle_filter.py` for smoother output at the cost of lag

---

## License

MIT
