"""
MediaPipe Pose benchmark on extracted frames.

Uses `mp.solutions.pose.Pose` (no external .task model file) so it works
out-of-the-box with the existing Python dependencies.
"""

import argparse
import json
import os
import platform
import time
from pathlib import Path

import numpy as np


def _load_image_rgb_uint8(path: Path) -> np.ndarray:
    from PIL import Image

    img = Image.open(path).convert("RGB")
    return np.asarray(img, dtype=np.uint8)


def _system_info() -> dict:
    info = {
        "os": platform.platform(),
        "python": platform.python_version(),
        "processor": platform.processor(),
        "cwd": os.getcwd(),
    }
    try:
        import psutil  # type: ignore

        info["cpu_cores_logical"] = psutil.cpu_count(logical=True)
        info["ram_gb"] = round(psutil.virtual_memory().total / (1024**3), 2)
    except Exception:
        pass
    return info


def run(frames_dir: Path, out_json: Path, model_complexity: int, max_frames: int | None) -> dict:
    import mediapipe as mp

    frames = sorted(frames_dir.glob("*.jpg"))
    if not frames:
        raise RuntimeError(f"No .jpg frames found in: {frames_dir}")
    if max_frames is not None:
        frames = frames[:max_frames]

    pose = mp.solutions.pose.Pose(
        static_image_mode=True,
        model_complexity=model_complexity,
        smooth_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    per_frame: list[dict] = []
    latencies: list[float] = []
    mean_scores: list[float] = []

    wall_t0 = time.perf_counter()
    for i, fp in enumerate(frames):
        rgb = _load_image_rgb_uint8(fp)
        t0 = time.perf_counter()
        res = pose.process(rgb)
        dt_ms = (time.perf_counter() - t0) * 1000.0

        if res.pose_landmarks:
            kps = np.array([[lm.x, lm.y, lm.visibility] for lm in res.pose_landmarks.landmark], dtype=float)
            mean_score = float(np.mean(kps[:, 2]))
            kps_list = kps.tolist()
        else:
            mean_score = 0.0
            kps_list = None

        latencies.append(float(dt_ms))
        mean_scores.append(float(mean_score))
        per_frame.append(
            {
                "frame_index": i,
                "frame_file": fp.name,
                "inference_ms": float(dt_ms),
                "avg_keypoint_score": float(mean_score),
                "keypoints_xy_vis": kps_list,  # 33 landmarks: [x_norm, y_norm, visibility]
            }
        )

    wall_elapsed = time.perf_counter() - wall_t0
    fps = len(frames) / wall_elapsed if wall_elapsed > 0 else 0.0

    result = {
        "library": "MediaPipePose",
        "framework": "mediapipe",
        "model_complexity": model_complexity,
        "frames_dir": str(frames_dir),
        "n_frames": len(frames),
        "wall_elapsed_s": float(wall_elapsed),
        "fps": float(fps),
        "latency_ms": {
            "mean": float(np.mean(latencies)),
            "p50": float(np.percentile(latencies, 50)),
            "p90": float(np.percentile(latencies, 90)),
            "p95": float(np.percentile(latencies, 95)),
        },
        "avg_keypoint_score": {
            "mean": float(np.mean(mean_scores)),
            "p50": float(np.percentile(mean_scores, 50)),
        },
        "per_frame": per_frame,
        "system": _system_info(),
        "created_at_unix": time.time(),
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames_dir", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--model_complexity", type=int, default=1, help="0|1|2 (default: 1)")
    ap.add_argument("--max_frames", type=int, default=None, help="Optional cap on number of frames")
    args = ap.parse_args()

    res = run(Path(args.frames_dir), Path(args.out_json), args.model_complexity, args.max_frames)
    print(json.dumps({k: res[k] for k in res if k != "per_frame"}, indent=2))


if __name__ == "__main__":
    main()

