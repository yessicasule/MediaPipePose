"""
benchmark_latency.py — Measures end-to-end pipeline latency.

Method:
  Sends a known UDP packet and measures time until it would
  arrive in Unity. Also benchmarks per-component times:
    - Frame capture
    - MediaPipe inference
    - Angle computation
    - Filter update
    - UDP send

Usage:
    python scripts/benchmark_latency.py
    python scripts/benchmark_latency.py --frames 300 --camera 0
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.pose.mediapipe_runner import MediaPipeRunner
from src.processing.joint_angle_estimator import compute_all
from src.processing.angle_filter import AngleFilterSystem
from src.streaming.udp_streamer import UdpAngleStreamer


def benchmark(n_frames: int = 200, camera_id: int = 0):
    print(f"\nLatency Benchmark — {n_frames} frames, camera {camera_id}")
    print("=" * 55)

    runner  = MediaPipeRunner()
    filt    = AngleFilterSystem("kalman")
    streamer = UdpAngleStreamer(hz=60)

    cap = cv2.VideoCapture(camera_id, cv2.CAP_MSMF)
    if not cap.isOpened():
        cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # warm up
    for _ in range(15):
        cap.read()
    time.sleep(0.3)
    streamer.start()

    times = {"capture": [], "mediapipe": [], "angles": [], "filter": [], "udp": [], "total": []}
    detected = 0

    for i in range(n_frames):
        t0 = time.perf_counter()

        ret, frame = cap.read()
        t1 = time.perf_counter()
        if not ret:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        lm  = runner.process(rgb)
        t2  = time.perf_counter()

        if lm is not None:
            detected += 1
            raw = compute_all(lm)
            t3  = time.perf_counter()

            fil = filt.update(raw)
            t4  = time.perf_counter()

            streamer.update_angles(
                shoulder_pitch = fil["shoulder_elevation"],
                shoulder_yaw   = fil["shoulder_yaw"],
                shoulder_roll  = fil["shoulder_roll"],
                elbow_flex     = fil["elbow_flexion"],
            )
            t5 = time.perf_counter()

            times["capture"].append(  (t1 - t0) * 1000)
            times["mediapipe"].append((t2 - t1) * 1000)
            times["angles"].append(   (t3 - t2) * 1000)
            times["filter"].append(   (t4 - t3) * 1000)
            times["udp"].append(      (t5 - t4) * 1000)
            times["total"].append(    (t5 - t0) * 1000)

    cap.release()
    streamer.stop()
    runner.close()

    if not times["total"]:
        print("No poses detected — point the camera at a person.")
        return

    detection_rate = detected / n_frames * 100

    print(f"\nDetection rate : {detection_rate:.1f}%  ({detected}/{n_frames} frames)")
    print(f"\n{'Component':<15}  {'Mean ms':>8}  {'P50 ms':>8}  {'P95 ms':>8}  {'Max ms':>8}")
    print("-" * 55)

    for name, vals in times.items():
        arr = np.array(vals)
        label = "END-TO-END" if name == "total" else name
        print(f"{label:<15}  {arr.mean():>8.2f}  {np.percentile(arr,50):>8.2f}  "
              f"{np.percentile(arr,95):>8.2f}  {arr.max():>8.2f}")

    total = np.array(times["total"])
    target_ms = 100.0
    passing   = (total < target_ms).mean() * 100
    print(f"\n< {target_ms:.0f} ms (spec requirement) : {passing:.1f}% of frames pass")

    if total.mean() < target_ms:
        print(f"✅  Mean latency {total.mean():.1f} ms — WITHIN 100 ms spec")
    else:
        print(f"⚠️   Mean latency {total.mean():.1f} ms — EXCEEDS 100 ms spec")

    # Save results
    out = Path("outputs/latency_benchmark.txt")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        f.write(f"Frames: {n_frames}  Detection: {detection_rate:.1f}%\n")
        for name, vals in times.items():
            arr = np.array(vals)
            f.write(f"{name}: mean={arr.mean():.2f} p95={np.percentile(arr,95):.2f} max={arr.max():.2f}\n")
    print(f"\nResults saved → {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--camera", type=int, default=0)
    args = ap.parse_args()
    benchmark(args.frames, args.camera)
