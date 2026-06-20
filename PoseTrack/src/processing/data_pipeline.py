"""
data_pipeline.py — Batch processing utilities for benchmark sessions.
Extracts frames from video, runs all three pose frameworks, saves JSON results.
Used by run_benchmark_all.py; not called during live demo.
"""
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.config import DATA_DIR, FRAMES_DIR, LANDMARKS_DIR, ANGLES_DIR
from src.utils.io_utils import extract_frames_from_video
from src.processing.joint_angle_estimator import Vec3, compute_all

try:
    from src.pose.mediapipe_runner import MediaPipeRunner
    mediapipe_available = True
except ImportError:
    mediapipe_available = False

try:
    from src.pose.movenet_runner import MoveNetRunner
    movenet_available = True
except ImportError:
    movenet_available = False

try:
    from src.pose.posenet_runner import PoseNetRunner
    posenet_available = True
except ImportError:
    posenet_available = False


def process_video_session(video_path, session_dir, frameworks=None, max_frames=None):
    """Run pose estimation on a video file and save per-framework JSON results."""
    session_dir = Path(session_dir)
    results_dir = session_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    if frameworks is None:
        frameworks = []
        if mediapipe_available: frameworks.append("mediapipe")
        if movenet_available:   frameworks.append("movenet")
        if posenet_available:   frameworks.append("posenet")

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    results = {fw: {"frames": [], "fps": fps} for fw in frameworks}

    runners = {}
    if "mediapipe" in frameworks and mediapipe_available:
        runners["mediapipe"] = MediaPipeRunner()
    if "movenet"   in frameworks and movenet_available:
        runners["movenet"]   = MoveNetRunner()
    if "posenet"   in frameworks and posenet_available:
        runners["posenet"]   = PoseNetRunner()

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret or (max_frames and frame_idx >= max_frames):
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        for fw, runner in runners.items():
            lm = runner.process(rgb)
            entry = {"frame": frame_idx, "detected": lm is not None, "angles": None}
            if lm is not None:
                entry["angles"] = compute_all(lm)
            results[fw]["frames"].append(entry)
        frame_idx += 1

    cap.release()
    for fw, runner in runners.items():
        if hasattr(runner, "close"):
            runner.close()
        out = results_dir / f"{fw}.json"
        with open(out, "w") as f:
            json.dump(results[fw], f)

    return results


def compute_framework_metrics(results_json_path):
    """Load a framework JSON and return mean/std per joint angle."""
    with open(results_json_path) as f:
        data = json.load(f)

    angle_keys = ["shoulder_elevation", "shoulder_yaw", "shoulder_roll", "elbow_flexion"]
    collected  = {k: [] for k in angle_keys}

    for frame in data.get("frames", []):
        if frame.get("detected") and frame.get("angles"):
            for k in angle_keys:
                v = frame["angles"].get(k)
                if v is not None:
                    collected[k].append(v)

    metrics = {}
    for k, vals in collected.items():
        arr = np.array(vals)
        metrics[k] = {
            "mean": float(arr.mean()) if len(arr) else 0.0,
            "std":  float(arr.std())  if len(arr) else 0.0,
            "n":    len(arr),
        }
    metrics["detection_rate"] = (
        sum(1 for f in data["frames"] if f.get("detected")) / len(data["frames"])
        if data["frames"] else 0.0
    )
    return metrics


def build_comparison_dataframe(session_dir):
    """Read all framework JSONs in a session and return a comparison DataFrame."""
    session_dir  = Path(session_dir)
    results_dir  = session_dir / "results"
    rows = []
    for json_file in sorted(results_dir.glob("*.json")):
        fw = json_file.stem
        m  = compute_framework_metrics(json_file)
        rows.append({"framework": fw, **{
            f"{k}_{stat}": m[k][stat]
            for k in ["shoulder_elevation","shoulder_yaw","shoulder_roll","elbow_flexion"]
            for stat in ["mean","std"]
        }, "detection_rate": m["detection_rate"]})
    return pd.DataFrame(rows)
