import argparse
import json
import os
import time
from pathlib import Path

import cv2


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def extract_frames(
    video_path: Path,
    out_dir: Path,
    stride: int,
    max_frames: int | None,
    resize_width: int | None,
    resize_height: int | None,
) -> dict:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0 else None

    _ensure_dir(out_dir)
    meta_path = out_dir / "frames_metadata.json"

    extracted = 0
    read_idx = 0
    saved = []
    t0 = time.perf_counter()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if read_idx % stride != 0:
            read_idx += 1
            continue

        if resize_width and resize_height:
            frame = cv2.resize(frame, (resize_width, resize_height), interpolation=cv2.INTER_AREA)

        fname = f"frame_{extracted:06d}.jpg"
        fpath = out_dir / fname

        ok = cv2.imwrite(str(fpath), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        if not ok:
            raise RuntimeError(f"Failed to write frame: {fpath}")

        saved.append(fname)
        extracted += 1
        read_idx += 1

        if max_frames is not None and extracted >= max_frames:
            break

    cap.release()
    elapsed_s = time.perf_counter() - t0

    meta = {
        "video_path": str(video_path),
        "out_dir": str(out_dir),
        "stride": stride,
        "max_frames": max_frames,
        "resize": {"width": resize_width, "height": resize_height},
        "video_fps": fps,
        "video_frame_count": total,
        "extracted_frames": extracted,
        "elapsed_s": elapsed_s,
        "frames": saved,
        "created_at_unix": time.time(),
        "cwd": os.getcwd(),
    }

    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="Path to input video (mp4, mov, etc.)")
    ap.add_argument("--out_dir", required=True, help="Output directory for extracted JPG frames")
    ap.add_argument("--stride", type=int, default=1, help="Keep every Nth frame (default: 1)")
    ap.add_argument("--max_frames", type=int, default=None, help="Stop after extracting K frames")
    ap.add_argument("--resize_width", type=int, default=None, help="Optional resize width")
    ap.add_argument("--resize_height", type=int, default=None, help="Optional resize height")
    args = ap.parse_args()

    if args.stride < 1:
        raise SystemExit("--stride must be >= 1")

    if (args.resize_width is None) ^ (args.resize_height is None):
        raise SystemExit("Provide both --resize_width and --resize_height, or neither.")

    meta = extract_frames(
        video_path=Path(args.video),
        out_dir=Path(args.out_dir),
        stride=args.stride,
        max_frames=args.max_frames,
        resize_width=args.resize_width,
        resize_height=args.resize_height,
    )
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()

