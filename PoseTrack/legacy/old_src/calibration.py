import json
import time
from pathlib import Path

import cv2
import numpy as np

from src.joint_angle_estimator import compute_elbow_flexion, compute_shoulder_elevation

CALIBRATION_FILE = Path("outputs/data/calibration.json")

REFERENCE_POSES = [
    ("arm_down",    "Lower your arm straight down beside your body"),
    ("arm_forward", "Raise your arm straight forward to shoulder height"),
    ("elbow_flexed","Bend your elbow to ~90 degrees"),
]


def _capture_angles(landmarks, n_samples: int = 30, delay_s: float = 2.0) -> dict:
    samples = {"elbow_flexion": [], "shoulder_elevation": []}
    t_end = time.perf_counter() + delay_s
    collected = 0
    while collected < n_samples and time.perf_counter() < t_end + 1.0:
        if landmarks is not None:
            samples["elbow_flexion"].append(compute_elbow_flexion(landmarks))
            samples["shoulder_elevation"].append(compute_shoulder_elevation(landmarks))
            collected += 1
        time.sleep(delay_s / n_samples)

    return {k: float(np.mean(v)) if v else 0.0 for k, v in samples.items()}


def run_calibration(cap, pose) -> dict:
    import mediapipe as mp
    mp_pose = mp.solutions.pose

    print("\n=== CALIBRATION ===")
    print("You will be asked to hold three reference poses.")
    print("Hold each pose still when prompted.\n")

    raw = {}
    for pose_id, instruction in REFERENCE_POSES:
        print(f"[{pose_id}]  {instruction}")
        input("  Press ENTER when ready...")
        print("  Collecting... hold still for 2 seconds.")

        angles_list = {"elbow_flexion": [], "shoulder_elevation": []}
        t_end = time.perf_counter() + 2.5
        while time.perf_counter() < t_end:
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res   = pose.process(rgb)
            if res.pose_landmarks:
                lm = res.pose_landmarks.landmark
                angles_list["elbow_flexion"].append(compute_elbow_flexion(lm))
                angles_list["shoulder_elevation"].append(compute_shoulder_elevation(lm))
        raw[pose_id] = {k: float(np.mean(v)) if v else 0.0 for k, v in angles_list.items()}
        print(f"  Captured: elbow={raw[pose_id]['elbow_flexion']:.1f}°  "
              f"shoulder_elev={raw[pose_id]['shoulder_elevation']:.1f}°")

    params = {
        "elbow_flex_min":      raw["arm_down"]["elbow_flexion"],
        "elbow_flex_max":      raw["elbow_flexed"]["elbow_flexion"],
        "shoulder_elev_min":   raw["arm_down"]["shoulder_elevation"],
        "shoulder_elev_max":   raw["arm_forward"]["shoulder_elevation"],
        "raw_poses":           raw,
        "calibrated_at":       time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    CALIBRATION_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CALIBRATION_FILE, "w") as f:
        json.dump(params, f, indent=2)

    print(f"\nCalibration saved to {CALIBRATION_FILE}")
    return params


def load_calibration() -> dict:
    if CALIBRATION_FILE.exists():
        with open(CALIBRATION_FILE) as f:
            return json.load(f)
    return {}


def apply_calibration(angle: float, raw_min: float, raw_max: float,
                       avatar_min: float = 0.0, avatar_max: float = 150.0) -> float:
    if raw_max == raw_min:
        return avatar_min
    ratio = (angle - raw_min) / (raw_max - raw_min)
    return float(np.clip(avatar_min + ratio * (avatar_max - avatar_min), avatar_min, avatar_max))
