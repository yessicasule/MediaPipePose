#!/usr/bin/env python3
"""
Benchmark script to compare pose estimation frameworks
"""

import cv2
import time
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config.config import Config
from src.pose.mediapipe_runner import MediaPipeRunner
from src.pose.movenet_runner import MoveNetRunner
from src.pose.posenet_runner import PoseNetRunner
from src.processing.joint_angle_estimator import compute_all
from src.processing.angle_filter import AngleFilterSystem


def get_fps(frame_times):
    if len(frame_times) < 2:
        return 0
    return 1.0 / (sum(frame_times) / len(frame_times))


def measure_jitter(angles_list):
    import numpy as np
    if len(angles_list) < 2:
        return 0, 0
    arr = list(angles_list)
    std = np.std(arr)
    var = np.var(arr)
    return float(std), float(var)


def benchmark_framework(name, runner, cap, duration=10):
    """Run benchmark for a pose framework"""
    print(f"\n{'='*50}")
    print(f"Benchmarking: {name}")
    print(f"{'='*50}")
    
    frame_times = []
    angles_list = []
    valid_frames = 0
    start = time.perf_counter()
    
    while time.perf_counter() - start < duration:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        
        t0 = time.perf_counter()
        landmarks = runner.process(frame)
        t1 = time.perf_counter()
        frame_times.append(t1 - t0)
        
        if landmarks is not None:
            valid_frames += 1
            try:
                angles = compute_all(landmarks)
                angles_list.append(angles.get("elbow_flexion", 0))
            except:
                pass
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    fps = get_fps(frame_times)
    avg_time = sum(frame_times) / len(frame_times) * 1000
    std, var = measure_jitter(angles_list)
    
    print(f"Valid frames: {valid_frames}")
    print(f"Average processing time: {avg_time:.2f} ms")
    print(f"FPS: {fps:.1f}")
    print(f"Elbow angle std: {std:.2f}, variance: {var:.2f}")
    
    runner.close()
    return {
        "name": name,
        "fps": fps,
        "avg_time_ms": avg_time,
        "valid_frames": valid_frames,
        "angle_std": std,
        "angle_var": var
    }


def main():
    print("Pose Estimation Framework Comparison")
    print(f"Duration per framework: 10 seconds")
    
    cap = cv2.VideoCapture(Config.CAMERA_ID)
    if not cap.isOpened():
        print("Error: Cannot open camera")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.FRAME_HEIGHT)
    
    results = []
    
    print("\n1. MediaPipe Pose")
    runner = MediaPipeRunner(
        min_detection_confidence=Config.MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=Config.MIN_TRACKING_CONFIDENCE
    )
    results.append(benchmark_framework("MediaPipe Pose", runner, cap, duration=10))
    
    print("\n2. MoveNet Lightning")
    runner = MoveNetRunner("movenet_lightning")
    results.append(benchmark_framework("MoveNet Lightning", runner, cap, duration=10))
    
    print("\n3. PoseNet")
    runner = PoseNetRunner()
    results.append(benchmark_framework("PoseNet", runner, cap, duration=10))
    
    cap.release()
    
    print(f"\n{'='*50}")
    print("SUMMARY")
    print(f"{'='*50}")
    print(f"{'Framework':<20} {'FPS':<8} {'ms/frame':<10} {'angle_std':<12}")
    for r in results:
        print(f"{r['name']:<20} {r['fps']:<8.1f} {r['avg_time_ms']:<10.2f} {r['angle_std']:<12.2f}")
    
    best = max(results, key=lambda x: x["fps"])
    print(f"\nBest performer: {best['name']} ({best['fps']:.1f} FPS)")


if __name__ == "__main__":
    main()