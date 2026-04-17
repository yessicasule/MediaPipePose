# Multi-Framework Pose Estimation Benchmarking

## Overview

This module provides a complete workflow for synchronized video + motion capture data collection and consistent multi-framework pose estimation benchmarking using **MediaPipe**, **PoseNet**, and **MoveNet**.

### Key Features

- **Synchronized Data Recording**: Collect video and motion capture data with frame-level synchronization
- **Consistent Benchmarking**: Extract frames once, process through all frameworks for fair comparison
- **Comprehensive Analysis**: Detailed latency, FPS, accuracy, and detection rate comparisons
- **Visual Reports**: Generate publication-ready comparison plots and dashboards

## Quick Start

### Option 1: Interactive Workflow
```bash
python run_benchmark_workflow.py
```

### Option 2: Full Automated Workflow
```bash
# Record new session
python run_benchmark_workflow.py --all --session_name test_001 --duration 60

# Use existing video
python run_benchmark_workflow.py --all --video_path path/to/video.mp4
```

### Option 3: Benchmark Only (with existing frames)
```bash
python run_benchmark_workflow.py --benchmark --frames_dir outputs/synchronized/my_session/frames
```

## Workflow Steps

### 1. Data Recording (Optional)

Record synchronized video + motion capture data:
```bash
python src/synchronized_recorder.py --session_name my_session --duration 30
```

This creates:
- `outputs/synchronized/my_session/video/recording.mp4` - Recorded video
- `outputs/synchronized/my_session/frames/` - Extracted frames
- `outputs/synchronized/my_session/motion_data/mocap_data.json` - Motion capture data
- `outputs/synchronized/my_session/metadata.json` - Session metadata

### 2. Extract Frames (if using external video)

```bash
python -m benchmarks.extract_frames \
    --video "path/to/video.mp4" \
    --out_dir "outputs/benchmarks/session/frames" \
    --stride 1
```

### 3. Run Benchmarks

#### All Frameworks
```bash
python -m benchmarks.run_all_benchmarks \
    --session_name my_benchmark \
    --frames_dir "outputs/benchmarks/session/frames"
```

#### Individual Frameworks

**MediaPipe:**
```bash
python -m benchmarks.run_mediapipe_on_frames \
    --frames_dir "frames" \
    --out_json "results/mediapipe.json"
```

**MoveNet:**
```bash
python -m benchmarks.run_movenet_on_frames \
    --frames_dir "frames" \
    --out_json "results/movenet.json" \
    --model lightning
```

**PoseNet (Node.js):**
```bash
cd benchmarks/posenet_tfjs && npm install
node run_posenet_on_frames.mjs \
    --frames_dir "../../frames" \
    --out_json "../results/posenet.json"
```

### 4. Generate Visualizations

```bash
python -m benchmarks.visualize_benchmarks \
    --results_dir "outputs/benchmarks/session/results" \
    --frames_dir "outputs/benchmarks/session/frames" \
    --output_dir "outputs/benchmarks/session/results/visualizations"
```

## Output Files

### Benchmark Results
- `mediapipe_results.json` - MediaPipe benchmark data
- `posenet_results.json` - PoseNet benchmark data  
- `movenet_results.json` - MoveNet benchmark data
- `benchmark_summary.json` - Combined comparison summary

### Visualizations
- `latency_comparison.png` - Per-frame latency comparison
- `fps_latency_comparison.png` - FPS and latency bar charts
- `accuracy_comparison.png` - Keypoint score comparison
- `benchmark_dashboard.png` - Comprehensive summary dashboard
- `frame_comparison.png` - Side-by-side frame analysis
- `benchmark_report.txt` - Text summary report

## Configuration Options

### Recording Options
| Option | Description | Default |
|--------|-------------|---------|
| `--session_name` | Session identifier | Timestamp-based |
| `--duration` | Recording duration (0=unlimited) | 30 seconds |
| `--output_dir` | Output directory | outputs/synchronized |

### Benchmark Options
| Option | Description | Default |
|--------|-------------|---------|
| `--frames_dir` | Pre-extracted frames directory | Required |
| `--max_frames` | Maximum frames to process | All |
| `--no-mediapipe` | Skip MediaPipe | False |
| `--no-posenet` | Skip PoseNet | False |
| `--no-movenet` | Skip MoveNet | False |
| `--no-viz` | Skip visualization | False |

### Model Variants
- **MediaPipe**: `full`, `lite`, `heavy`
- **MoveNet**: `lightning`, `thunder`
- **PoseNet**: `MobileNetV1`, `ResNet50`

## Dependencies

### Python
```
mediapipe>=0.10.9
opencv-python>=4.8.1
numpy>=1.24
tensorflow>=2.14
tensorflow-hub>=0.15
matplotlib>=3.7
psutil>=5.9
Pillow>=10.0
```

### Node.js (for PoseNet)
```bash
cd benchmarks/posenet_tfjs && npm install
```

## Architecture

```
run_benchmark_workflow.py
├── src/synchronized_recorder.py     # Data collection
├── benchmarks/
│   ├── extract_frames.py             # Frame extraction
│   ├── run_mediapipe_on_frames.py    # MediaPipe benchmark
│   ├── run_movenet_on_frames.py      # MoveNet benchmark
│   ├── run_all_benchmarks.py          # Unified runner
│   ├── visualize_benchmarks.py        # Visualization generator
│   └── posenet_tfjs/
│       └── run_posenet_on_frames.mjs  # PoseNet (Node.js)
└── outputs/
    └── benchmarks/
        └── session/
            ├── frames/                 # Extracted frames
            ├── results/                # Benchmark JSON results
            │   └── visualizations/     # Comparison plots
            └── benchmark_summary.json  # Combined summary
```

## Metrics Compared

### Performance
- **FPS**: Overall throughput (frames processed per second)
- **Latency**: Per-frame inference time (mean, p50, p90, p95, p99)
- **Stability**: Latency variance across frames

### Accuracy
- **Keypoint Score**: Average confidence per keypoint (0-1)
- **Detection Rate**: Percentage of frames with detected pose

## Tips

1. **Consistent Data**: Use the same extracted frames for all framework benchmarks
2. **Warm-up Runs**: First few frames may be slower due to model loading
3. **Frame Count**: More frames = more reliable benchmark statistics
4. **Resolution**: Match video resolution to intended use case

## Troubleshooting

### "No .jpg frames found"
Run frame extraction first: `python -m benchmarks.extract_frames --video video.mp4 --out_dir frames`

### "Node.js not found" 
Install Node.js or skip PoseNet: `python run_benchmark_workflow.py --no-posenet`

### "Model not found"
Download MediaPipe model:
```bash
mkdir models
curl -o models/pose_landmarker_full.task \
  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task
```
