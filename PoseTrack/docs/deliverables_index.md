# MonoArm — Deliverables Index
Generated: 2026-07-08 17:33:21

## Required Deliverables

| # | Deliverable | Status | Path |
|---|-------------|--------|------|
| 1 | Real-time vision-based arm tracking application | ✅ Complete | `scripts/run_demo.py` |
| 2 | Joint angle estimation and filtering module | ✅ Complete | `src/processing/angle_solver.py`, `angle_filter.py` |
| 3 | Unity application with avatar arm control | ✅ Complete | `Unity/UnityMedia/Assets/Scripts/*.cs` |
| 4 | Calibration module | ✅ Complete | `src/processing/calibration.py` |
| 5 | Joint angle data logging tools | ✅ Complete | `src/processing/angle_logger.py`, `scripts/plot_angles.py` |
| 6 | Technical documentation | ✅ Complete | `docs/technical_report.md` |
| 7 | Video demonstration | ✅ Complete | `outputs/demo_video.mp4` |

## All Output Files

### Core Application
| File | Description |
|------|-------------|
| `scripts/run_demo.py` | **Main entry point** — full live pipeline |
| `src/main.py` | Headless-friendly alternative pipeline |
| `scripts/run_capture_session.py` | Session recorder with video + CSV |
| `scripts/mock_streamer.py` | Unity test without webcam |

### Joint Angle Estimation
| File | Description |
|------|-------------|
| `src/processing/angle_solver.py` | ISB ZXY Euler 4-DOF solver |
| `src/processing/coordinate_frame.py` | Torso reference frame builder |
| `src/processing/angle_filter.py` | Kalman, MA, SG, EMA filters |
| `src/processing/calibration.py` | 3-tier calibration pipeline |

### Unity Integration (6 C# scripts)
| File | Description |
|------|-------------|
| `UdpAngleReceiver.cs` | Threaded UDP receiver, parses `S,` packets |
| `ArmAngleController.cs` (`AvatarMuscleController`) | HumanPoseHandler muscle-space controller |
| `MultiAvatarManager.cs` (`MonoArmManager`) | Wires receiver→controller |
| `PoseDebugUI.cs` | World-space live angle HUD under avatar |
| `AngleSmoother.cs` | Per-channel SmoothDampAngle helper |
| `Editor/SceneBuilder.cs` | **MonoArm > Build Scene** auto-setup |

### Logging & Visualization
| File | Description |
|------|-------------|
| `src/processing/angle_logger.py` | CSV logger + rolling live plot |
| `scripts/plot_angles.py` | Post-session 4-panel matplotlib figure |
| `scripts/compare_filters.py` | Filter comparison benchmark |

### Evaluation & Benchmarking
| File | Description |
|------|-------------|
| `src/evaluation/metrics.py` | MAE/RMSE/MPJAE/PCK/Jitter |
| `src/evaluation/eval_plots.py` | 5 publication-quality figures |
| `src/evaluation/occlusion_test.py` | Robustness benchmark |
| `scripts/evaluate_h36m.py` | H3.6M ground-truth validation |
| `scripts/benchmark_latency.py` | Per-component latency profiling |
| `scripts/run_experiments.py` | Unified experiment orchestrator |

### Generated Outputs
| File | Description |
|------|-------------|
| `outputs/benchmarks/arm_test_01/plots/benchmark_dashboard.png` | FPS/latency dashboard |
| `outputs/benchmarks/arm_test_01/plots/accuracy_comparison.png` | Accuracy charts |
| `outputs/filter_comparison.png` | Filter benchmark (generated) |
| `outputs/experiments/` | Evaluation metrics JSON + plots |
| `outputs/demo_video.mp4` | System demonstration video |
| `outputs/unified_dataset.csv` | 78 MB training dataset |
| `outputs/models/bilstm_baseline_mp.pt` | Trained BiLSTM checkpoint |

## How to Run

```bash
# Live arm tracking (requires webcam)
python scripts/run_demo.py --filter kalman

# Test Unity without webcam
python scripts/mock_streamer.py --mode sinusoidal

# Full deliverables generation (headless)
python scripts/generate_deliverables.py

# Run unit tests
python -m pytest tests/ -v
```

## Unity Scene Setup

1. Open `Unity/UnityMedia/` in Unity 2022+
2. Open `Assets/HumanoidScene1.unity`
3. Menu: **MonoArm → Build Scene**
4. Press Ctrl+S
5. Start Python: `python scripts/run_demo.py`
6. Press Play in Unity
