## PoseNet vs MoveNet benchmark (video frames)

This benchmark extracts frames from a video once, then runs **MoveNet (TF Hub, Python)** and **PoseNet (TFJS, Node)** on the same frame set to compare:

- **Speed**: per-frame latency, FPS
- **Output quality proxy**: average keypoint confidence score (per frame)

### 1) Create frames from your video (Python)

From `PoseTrack/`:

```bash
python -m benchmarks.extract_frames --video "PATH\TO\video.mp4" --out_dir "benchmarks\frames\my_video" --stride 1
```

Useful options:
- `--stride N`: keep every Nth frame
- `--max_frames K`: stop after K extracted frames
- `--resize_width W --resize_height H`: resize before saving (optional)

### 2) Run MoveNet on the extracted frames (Python)

```bash
python -m benchmarks.run_movenet_on_frames --frames_dir "benchmarks\frames\my_video" --out_json "benchmarks\results\movenet.json" --model lightning
```

### 3) Run PoseNet on the extracted frames (Node)

Install Node deps once (from `PoseTrack/benchmarks/posenet_tfjs/`):

```bash
npm install
```

Run:

```bash
node run_posenet_on_frames.mjs --frames_dir "..\frames\my_video" --out_json "..\results\posenet.json" --architecture MobileNetV1
```

### 4) Plot graphs (Python)

```bash
python -m benchmarks.plot_benchmark --movenet_json "benchmarks\results\movenet.json" --posenet_json "benchmarks\results\posenet.json" --out_dir "benchmarks\plots"
```

This creates:
- `latency_ms.png` (per-frame inference latency)
- `fps.png` (overall FPS)
- `avg_keypoint_score.png` (per-frame average keypoint score)

