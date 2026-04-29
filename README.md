# PoseTrack — Monocular Vision-Based Arm Joint Angle Estimation

Real-time upper-body pose tracking from a single RGB camera. Computes shoulder and elbow joint angles, streams them over UDP to a Unity humanoid avatar, and provides a full benchmarking pipeline for comparing pose estimation frameworks (MediaPipe, MoveNet, PoseNet).

---

## Installation

```bash
cd PoseTrack

python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

pip install opencv-python mediapipe numpy tensorflow tensorflow-hub
pip install matplotlib scipy pillow psutil
```

PoseNet benchmark only (requires Node.js):
```bash
cd benchmarks/posenet_tfjs
npm install
cd ../..
```

All commands below are run from the `PoseTrack/` directory.

---

## Unity Setup

The Unity project is already configured at `Unity/UnityMedia/`.

1. Open **Unity Hub** → Add project from disk → select `Unity/UnityMedia`
2. Open with **Unity 2022.3 LTS**
3. Open the scene `Assets/HumanoidScene1.unity`
4. Press **Play** — the UDP receiver starts automatically on port `9000`

**If you need to rebuild the scene from scratch**, see the full Unity setup guide in `Unity/UnityMedia/Assets/Scripts/` — the five C# scripts are `UdpAngleReceiver.cs`, `ArmAngleController.cs`, `AngleSmoother.cs`, `ArmRigSetup.cs`, and `PoseDebugUI.cs`.

**Bone assignment** — select `PoseReceiver` in the Hierarchy → `ArmAngleController` component → click **Auto-Find Humanoid Bones**. If it fails, drag `LeftArm` and `LeftForeArm` from the avatar hierarchy manually.

**Axis offsets** — if the arm points the wrong direction, adjust `Shoulder Axis Offset` and `Elbow Axis Offset` in `ArmAngleController`. Start at `(0,0,0)` and tune after running a calibration session.

**Avatar pixelation** — select avatar textures in the Project panel → Inspector → set Max Size to `2048`, Filter Mode to `Bilinear`, click Apply. Disable any LOD Group component on the avatar.

---

## Feature 1 — Real-Time Streaming to Unity

Reads the camera, estimates pose, filters angles, streams to Unity at 30 Hz.

```bash
python scripts/run_live.py
```

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--filter` | `kalman` | `kalman`, `ema`, `ma`, `sg` — see Filters section |
| `--camera` | `0` | Camera index. Try `1` or `2` if camera fails to open |
| `--host` | `127.0.0.1` | IP of the machine running Unity |
| `--port` | `9000` | UDP port |
| `--video path.mp4` | — | Use a recorded video file instead of live camera |

Press `q` to stop.

**Requires Unity to be in Play mode** to see the avatar move. Run without Unity using `--host` pointed at a closed port — the stream silently drops packets with no error.

---

## Feature 2 — Capture Session

Records video and logs joint angles frame-by-frame. Use this for data collection, analysis, and benchmarking. Also streams to Unity at the same time.

```bash
python scripts/run_capture_session.py --session arm_test_01
```

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--session` | `session_<unix>` | Name for the output folder |
| `--filter` | `kalman` | `kalman`, `ema`, `ma`, `sg` |
| `--save_landmarks` | off | Also save all 33 MediaPipe keypoints per frame |
| `--no_stream` | off | Disable UDP — run without Unity open |
| `--calib path.json` | — | Load a previously saved calibration file |
| `--camera` | `0` | Camera index |

**Keyboard controls:**

| Key | Action |
|-----|--------|
| `s` | Toggle video recording on / off |
| `p` | Open / close live rolling angle plot |
| `d` | Toggle debug angle overlay on camera view |
| `c` | Cycle to next calibration pose (see Feature 4) |
| `SPACE` | Capture the current calibration pose |
| `q` / `ESC` | Stop, save all files, generate plots |

**Output** — saved to `data/sessions/arm_test_01/`:

```
raw.mp4                          raw camera recording
angles.csv                       per-frame angles — frame_index, timestamp,
                                 elapsed_s, inference_ms, pose_detected,
                                 raw + filtered values for all 4 joints
landmarks.csv                    (--save_landmarks) all 33 MediaPipe keypoints
arm_test_01_angles.png           4-panel raw vs filtered plot, auto-generated on exit
arm_test_01_angles_comparison.png  all-joints overlay comparison plot
calibration.json                 saved if calibration was performed
```

---

## Feature 3 — Filters

Four filter options are available across all scripts that accept `--filter`:

| Option | Description | Best for |
|--------|-------------|----------|
| `kalman` | 1-D Kalman filter (default) | Real-time, balances lag vs smoothness |
| `ema` | Exponential moving average | Simple, low CPU |
| `ma` | Simple moving average, window = 7 | Uniform smoothing |
| `sg` | Savitzky-Golay, window = 11, order = 3 | Preserves motion peaks, requires `scipy` |

```bash
python scripts/run_live.py --filter sg
python scripts/run_capture_session.py --session test_01 --filter ema
```

To compare filters, run two separate capture sessions with different `--filter` values, then use `plot_angles.py` to overlay the resulting CSVs.

---

## Feature 4 — Calibration

Maps your neutral arm position to the avatar's coordinate frame. Run during any capture session.

1. Start a capture session
2. Stand with your arm hanging at your side
3. Press `c` — terminal shows `Hold 'arm_down' pose. Press SPACE to capture`
4. Press `SPACE`
5. Raise your arm horizontal in front of you
6. Press `c` — prompted for `arm_forward`
7. Press `SPACE`
8. Bend your elbow to 90°
9. Press `c` — prompted for `elbow_flexed`
10. Press `SPACE` — calibration finalised, saved to `data/sessions/arm_test_01/calibration.json`

Reload in future sessions:
```bash
python scripts/run_capture_session.py --session arm_test_02 --calib data/sessions/arm_test_01/calibration.json
```

---

## Feature 5 — Angle Plots

Auto-generated on capture session exit. Re-run manually at any time:

```bash
python scripts/plot_angles.py \
    --csv  data/sessions/arm_test_01/angles.csv \
    --comparison \
    --show
```

| Flag | Description |
|------|-------------|
| `--csv` | Path to `angles.csv` from a capture session |
| `--out path.png` | Custom output path (default: next to the CSV) |
| `--comparison` | Also save all-joints overlay comparison figure |
| `--show` | Open interactive window (requires display) |

Output:
- `*_angles.png` — 4-panel figure, one joint per row. Raw trace (faint) + filtered trace (solid) + ±3° target band around filtered mean
- `*_angles_comparison.png` — all four joints raw vs filtered on two stacked panels

---

## Feature 6 — Data Generator (Unity testing without a camera)

Sends synthetic angle values over UDP. Use this to test Unity bone assignment and axis offsets without a camera.

```bash
python scripts/data_generator.py --mode wave --hz 30
```

| Flag | Default | Description |
|------|---------|-------------|
| `--mode` | `wave` | `wave`, `circle`, `static`, `random` |
| `--hz` | `30` | Update rate |
| `--host` | `127.0.0.1` | Target IP |
| `--port` | `9000` | Target port |
| `--duration` | `0` (infinite) | Stop after N seconds |

---

## Feature 7 — Benchmark Pipeline

Runs MediaPipe, MoveNet, and PoseNet on the same recorded frames for a fair side-by-side comparison.

### Step 1 — Capture a session

```bash
python scripts/run_capture_session.py --session benchmark_01 --save_landmarks --no_stream
```

### Step 2 — Extract frames from the video

```bash
python benchmarks/extract_frames.py \
    --video   data/sessions/benchmark_01/raw.mp4 \
    --out_dir data/sessions/benchmark_01/frames
```

Optional flags: `--stride 2` (keep every 2nd frame), `--max_frames 300`

### Step 3 — Run framework benchmarks

**All at once (MediaPipe + MoveNet, skips PoseNet):**
```bash
python benchmarks/run_all_benchmarks.py \
    --session_name benchmark_01 \
    --frames_dir   data/sessions/benchmark_01/frames \
    --no-posenet
```

**Or individually:**
```bash
python benchmarks/run_mediapipe_on_frames.py \
    --frames_dir data/sessions/benchmark_01/frames \
    --out_json   data/sessions/benchmark_01/results/mediapipe.json

python benchmarks/run_movenet_on_frames.py \
    --frames_dir data/sessions/benchmark_01/frames \
    --out_json   data/sessions/benchmark_01/results/movenet.json \
    --model      lightning
```

**PoseNet (requires Node.js):**
```bash
node benchmarks/posenet_tfjs/run_posenet_on_frames.mjs \
    --frames_dir data/sessions/benchmark_01/frames \
    --out_json   data/sessions/benchmark_01/results/posenet.json
```

### Step 4 — Evaluate frameworks

Computes per-framework: FPS, latency, keypoint jitter (XY and angle-domain), static pose stability, and arm detection robustness.

```bash
python benchmarks/evaluate_frameworks.py \
    --results data/sessions/benchmark_01/results/mediapipe.json \
              data/sessions/benchmark_01/results/movenet.json \
    --out_json data/sessions/benchmark_01/results/evaluation.json \
    --min_score 0.3 \
    --static_first_n 60
```

### Step 5 — Visualise results

```bash
python benchmarks/visualize_benchmarks.py \
    --results_dir data/sessions/benchmark_01/results \
    --frames_dir  data/sessions/benchmark_01/frames \
    --output_dir  data/sessions/benchmark_01/plots
```

Output in `data/sessions/benchmark_01/plots/`:

```
latency_comparison.png       per-frame latency traces + histogram
fps_latency_comparison.png   FPS bar chart + mean vs P90 latency
accuracy_comparison.png      keypoint confidence scores per frame + box plot
benchmark_dashboard.png      full summary dashboard with radar chart
benchmark_report.txt         plain-text summary table
```

### Step 6 — Render side-by-side skeleton video

```bash
python benchmarks/render_comparison_video.py \
    --frames_dir data/sessions/benchmark_01/frames \
    --mediapipe  data/sessions/benchmark_01/results/mediapipe.json \
    --movenet    data/sessions/benchmark_01/results/movenet.json \
    --out        outputs/comparison_benchmark_01.mp4
```

Left panel: MediaPipe skeleton. Right panel: MoveNet skeleton. Per-frame latency and confidence shown in the header.

---

## UDP Protocol

```
S,<shoulder_pitch>,<shoulder_yaw>,<shoulder_roll>,<elbow_flex>\n
```

All values in degrees, 3 decimal places, newline-terminated, sent at 30 Hz to `127.0.0.1:9000` by default.

Example: `S,42.317,-8.124,3.501,67.890`

For two machines: pass `--host <Unity-machine-IP>` on the Python side, and open UDP port 9000 inbound on the Unity machine's firewall.

---

## Performance Targets

| Metric | Target | Where to check |
|--------|--------|----------------|
| End-to-end latency | < 100 ms | `inference_ms` column in `angles.csv` |
| Vision module FPS | ≥ 20 FPS | Shown in capture window overlay (`d` key) |
| Static pose variance | ±3–5° filtered | ±3° band in angle plots |
| Continuous runtime | ≥ 10 min | Session elapsed time in capture overlay |

---

## Troubleshooting

**Camera opens but pipeline exits immediately**
MSMF on Windows sometimes needs a moment to warm up. The scripts already handle this with a 10-frame drain and 500 ms delay. If it still fails, try `--camera 1` or `--camera 2`.

**Camera won't open at all**
- Settings → Privacy → Camera — ensure desktop app access is ON
- Close any app holding the camera (Teams, Zoom, OBS)
- Use a recorded video: `--video path/to/file.mp4`

**No data received in Unity / Debug UI stuck on "Waiting for pose data..."**
- Confirm Python terminal shows `Streaming to Unity at 127.0.0.1:9000`
- Confirm Unity is in Play mode
- Check Windows Firewall — allow UDP port 9000 inbound
- Confirm `Listen Port` in `UdpAngleReceiver` matches `--port` in Python

**Avatar arm spins wildly or snaps to extreme angles**
- The axis offset is wrong — adjust `Shoulder Axis Offset` in `ArmAngleController`
- Set Unity `Smoothing` to `0.05` temporarily to slow things down and diagnose
- Run a calibration session to anchor the neutral arm-down pose

**Avatar appears pixelated during motion**
- Select avatar textures → Inspector → Max Size `2048`, Filter Mode `Bilinear`, Apply
- Disable any LOD Group component on the avatar

**High jitter even with filtering**
- Use `--filter kalman` (default)
- Improve lighting — shadows on elbow/shoulder confuse MediaPipe
- Keep full upper body (head to hips) in frame
- Increase `measurement_noise` in `KalmanFilter1D` in `src/processing/angle_filter.py` for more smoothing at the cost of lag
- SG filter (`--filter sg`) may help if motion has sharp peaks — requires `pip install scipy`

**`evaluate_frameworks.py` or `visualize_benchmarks.py` crash**
- Run Step 3 (framework benchmarks) before Step 4/5 — both scripts require the result JSONs to exist
- For `sg` filter: `pip install scipy`
