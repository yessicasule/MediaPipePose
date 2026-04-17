import argparse
import json
import os
import platform
import time
from pathlib import Path

import numpy as np


def _load_image_rgb_uint8(path: Path) -> np.ndarray:
    # Lazy import to keep import errors clearer if deps missing.
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


def _load_movenet(model_name: str):
    import tensorflow as tf
    import tensorflow_hub as hub

    if model_name not in {"lightning", "thunder"}:
        raise ValueError("--model must be one of: lightning, thunder")

    # TF Hub signatures: "serving_default"
    # lightning: 192x192, thunder: 256x256
    if model_name == "lightning":
        url = "https://tfhub.dev/google/movenet/singlepose/lightning/4"
        input_size = 192
    else:
        url = "https://tfhub.dev/google/movenet/singlepose/thunder/4"
        input_size = 256

    model = hub.load(url)
    fn = model.signatures["serving_default"]

    def infer(rgb_uint8: np.ndarray) -> tuple[np.ndarray, float]:
        # rgb_uint8: HxWx3, uint8
        img = tf.convert_to_tensor(rgb_uint8, dtype=tf.uint8)
        # resize_with_pad returns float32; MoveNet TF Hub expects int32 input.
        img = tf.image.resize_with_pad(img, input_size, input_size)
        img = tf.cast(img, tf.int32)
        img = tf.expand_dims(img, axis=0)
        t0 = time.perf_counter()
        out = fn(input=img)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        # output: [1,1,17,3] = (y,x,score)
        kps = out["output_0"].numpy()[0, 0]
        return kps, dt_ms

    return infer, input_size


def run(frames_dir: Path, out_json: Path, model: str, max_frames: int | None) -> dict:
    frames = sorted([p for p in frames_dir.glob("*.jpg")])
    if not frames:
        raise RuntimeError(f"No .jpg frames found in: {frames_dir}")
    if max_frames is not None:
        frames = frames[:max_frames]

    infer, input_size = _load_movenet(model)

    per_frame = []
    latencies = []
    mean_scores = []

    wall_t0 = time.perf_counter()
    for i, fp in enumerate(frames):
        rgb = _load_image_rgb_uint8(fp)
        kps, dt_ms = infer(rgb)
        scores = kps[:, 2].astype(float)
        mean_score = float(np.mean(scores))

        latencies.append(dt_ms)
        mean_scores.append(mean_score)
        per_frame.append(
            {
                "frame_index": i,
                "frame_file": fp.name,
                "inference_ms": float(dt_ms),
                "avg_keypoint_score": mean_score,
                "keypoints_yx_score": kps.astype(float).tolist(),
            }
        )

    wall_elapsed = time.perf_counter() - wall_t0
    fps = len(frames) / wall_elapsed if wall_elapsed > 0 else 0.0

    result = {
        "library": "MoveNet",
        "framework": "tensorflow_hub",
        "model": model,
        "input_size": input_size,
        "frames_dir": str(frames_dir),
        "n_frames": len(frames),
        "wall_elapsed_s": wall_elapsed,
        "fps": fps,
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
    ap.add_argument("--frames_dir", required=True, help="Directory of extracted .jpg frames")
    ap.add_argument("--out_json", required=True, help="Output JSON path")
    ap.add_argument("--model", default="lightning", help="lightning|thunder (default: lightning)")
    ap.add_argument("--max_frames", type=int, default=None, help="Optional cap on number of frames")
    args = ap.parse_args()

    res = run(Path(args.frames_dir), Path(args.out_json), args.model, args.max_frames)
    print(json.dumps({k: res[k] for k in res if k != "per_frame"}, indent=2))


if __name__ == "__main__":
    main()

