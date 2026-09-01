# MonoArm — Deliverables Index

Maps each deliverable in the project specification to the code that implements
it. "Present" means the code is in the repository and exercised by the test
suite or by a documented command; artefacts that must be produced by running the
system on hardware with a camera are marked as such rather than claimed.

---

## Required deliverables

| # | Deliverable | State | Where |
|---|---|---|---|
| 1 | Real-time vision-based arm tracking application | Present | `scripts/run_web.py` (browser dashboard), `scripts/run_demo.py` (OpenCV window) |
| 2 | Joint angle estimation and filtering module | Present | `src/processing/angle_solver.py`, `coordinate_frame.py`, `angle_filter.py` |
| 3 | Unity application with avatar arm control | Present | `Unity/UnityMedia/Assets/Scripts/` |
| 4 | Calibration module | Present | `src/processing/calibration.py`, dashboard calibration panel |
| 5 | Joint angle data logging and visualisation tools | Present | `src/processing/angle_logger.py`, `scripts/plot_angles.py`, dashboard session browser |
| 6 | Technical documentation | Present | `docs/technical_report.md`, `docs/web_pipeline.md`, `README.md` |
| 7 | Video recordings of system operation | **Produced by running the system** | `scripts/record_demo.py`, `scripts/run_capture_session.py` — requires a camera; recordings are not stored in the repository |

---

## Optional extensions from the specification

| Extension | State | Where |
|---|---|---|
| Gesture recognition from arm motion | Present | `src/processing/gesture_recognizer.py` — five classes, reported on every dashboard frame |
| Exoskeleton calibration/validation interface | Present | `src/streaming/exoskeleton_streamer.py` — per-joint JSON with `APPLY`/`HOLD` gating on port 9001 |
| Pretrained 2D-to-3D lifting for better shoulder angles | **Research code, no trained weights** | `src/models/fusion_network.py`, `baseline_models.py`, `gan_refinery.py` — training must be run separately; not used by the live pipeline |
| VR / AR integration | Not implemented | — |

---

## Web dashboard (front-end + back-end)

| File | Description |
|---|---|
| `webapp/server.py` | FastAPI app: REST control plane, WebSocket data plane, session browser, MJPEG preview |
| `webapp/pipeline.py` | per-frame orchestration: pose → angles → all filters → calibration → UDP → log, with per-stage timing and a derivation trace |
| `webapp/metrics.py` | rolling latency, throughput, angle stability and keypoint jitter statistics |
| `webapp/explain.py` | description of every reported quantity, served to the UI at `/api/explain` |
| `webapp/sources.py` | browser / server-camera / recorded-video frame sources |
| `webapp/static/index.html`, `app.js`, `style.css` | the dashboard, with no external assets: sidebar layout, light/dark themes, expandable per-card explanations, hover-readable charts, figure gallery |
| `scripts/run_web.py` | launcher |

---

## Core application

| File | Description |
|---|---|
| `scripts/run_web.py` | **primary entry point** — dashboard and Unity stream together |
| `scripts/run_demo.py` | OpenCV desktop version of the live pipeline |
| `src/main.py` | headless-friendly alternative pipeline |
| `scripts/run_capture_session.py` | session recorder producing video plus CSV |
| `scripts/mock_streamer.py` | drives Unity with generated angles to test the rig without a camera |

---

## Joint angle estimation

| File | Description |
|---|---|
| `src/processing/coordinate_frame.py` | torso reference frame, Gram–Schmidt orthonormalised |
| `src/processing/angle_solver.py` | ZXY decomposition, bilateral four-DOF solver, rotation-observability flag |
| `src/processing/angle_filter.py` | `KalmanFilter2State`, `MovingAverageFilter`, `SavitzkyGolayFilter`, and per-side filter banks |
| `src/processing/calibration.py` | reference-pose calibration with span and gain validation |
| `src/processing/gesture_recognizer.py` | rule-based gesture classification |

---

## Pose estimation frameworks

| File | Description |
|---|---|
| `src/pose/base.py` | common `PoseEstimator` interface and 33-keypoint landmark schema |
| `src/pose/mediapipe_runner.py` | MediaPipe Pose (Solutions API, falling back to the Tasks API) |
| `src/pose/movenet_runner.py` | MoveNet Lightning / Thunder |
| `src/pose/posenet_runner.py` | PoseNet |
| `scripts/compare_frameworks.py`, `benchmarks/` | side-by-side comparison on identical input |

---

## Unity integration

| File | Description |
|---|---|
| `UdpAngleReceiver.cs` | threaded UDP receiver, parses `S,` and `B,` packets |
| `ArmBoneController.cs` | direct bone control via `Transform.localRotation` and `Quaternion.Euler`, frame-rate-independent interpolation |
| `ArmAngleController.cs` (`AvatarMuscleController`) | humanoid muscle-space controller via `HumanPoseHandler` |
| `MultiAvatarManager.cs` (`MonoArmManager`) | wires the receiver to whichever controllers are in the scene |
| `AngleSmoother.cs` | per-channel `SmoothDampAngle` helper |
| `PoseDebugUI.cs` | world-space live angle HUD |
| `Editor/SceneBuilder.cs` | **MonoArm → Build Scene / Diagnose Scene / Undo Last Build** |

---

## Logging, visualisation and evaluation

| File | Description |
|---|---|
| `src/processing/angle_logger.py` | CSV loggers (single and bilateral) plus a rolling live plot |
| `scripts/plot_angles.py` | four-panel time-series figure from a session CSV |
| `scripts/compare_filters.py` | offline filter comparison |
| `scripts/benchmark_latency.py` | per-component latency profile |
| `src/evaluation/metrics.py` | MAE, RMSE, MPJAE, PCK, jitter |
| `src/evaluation/statistics.py` | paired significance tests and effect sizes |
| `src/evaluation/eval_plots.py` | publication-format figures |
| `src/evaluation/occlusion_test.py` | occlusion robustness benchmark |
| `src/evaluation/ablation.py` | ablation harness |
| `webapp/server.py` figure endpoints | serves the figures the scripts above produce into the dashboard gallery |
| `scripts/evaluate_h36m.py`, `scripts/evaluate_panoptic.py` | ground-truth validation against public datasets (datasets not bundled) |

---

## Tests

```bash
python -m pytest tests/ -q
```

| File | Covers |
|---|---|
| `tests/test_webapp.py` | metrics, calibration guards, pipeline on a real photograph, HTTP endpoints, WebSocket data plane |
| `tests/test_pose_estimators.py` | estimator interface and landmark schema |
| `tests/test_evaluation.py` | metrics and statistics |
| `tests/test_refactored.py` | fusion model shapes — requires PyTorch |

---

## How to run

```bash
# Web dashboard + live Unity stream (needs a camera)
python scripts/run_web.py

# OpenCV desktop pipeline
python scripts/run_demo.py --filter kalman

# Drive Unity without a camera, to check the avatar rig
python scripts/mock_streamer.py --mode sinusoidal

# Latency profile for this machine
python scripts/benchmark_latency.py --frames 300

# Tests
python -m pytest tests/ -q
```

## Unity scene setup

1. Open `Unity/UnityMedia/` in Unity 2022 or newer.
2. Open `Assets/HumanoidScene1.unity`.
3. Menu → **MonoArm → Build Scene**, then save.
4. Start the Python side (`scripts/run_web.py`).
5. Press Play.

Outputs are written under `outputs/` (git-ignored): `outputs/web/logs/` for
dashboard sessions, `outputs/web/calibration_<side>.json` for calibration
parameters.
