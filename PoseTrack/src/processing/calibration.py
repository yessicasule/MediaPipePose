import numpy as np
import json
from pathlib import Path
from typing import Dict, Optional


class CalibrationData:
    def __init__(self):
        self.reference_poses = {}
        self.offset = {
            "shoulder_elevation": 0.0,
            "shoulder_yaw": 0.0,
            "shoulder_roll": 0.0,
            "elbow_flexion": 0.0
        }
        self.scale = {
            "shoulder_elevation": 1.0,
            "shoulder_yaw": 1.0,
            "shoulder_roll": 1.0,
            "elbow_flexion": 1.0
        }

    def add_reference_pose(self, name: str, angles: Dict[str, float]):
        self.reference_poses[name] = angles.copy()

    def compute_offsets(self):
        if "arm_down" in self.reference_poses and "arm_forward" in self.reference_poses:
            down = self.reference_poses["arm_down"]
            fwd = self.reference_poses["arm_forward"]
            self.offset["shoulder_elevation"] = -down.get("shoulder_elevation", 0)
            self.offset["shoulder_yaw"] = -down.get("shoulder_yaw", 0)
            self.offset["shoulder_roll"] = -down.get("shoulder_roll", 0)
            self.offset["elbow_flexion"] = -down.get("elbow_flexion", 0)

    def apply_calibration(self, angles: Dict[str, float]) -> Dict[str, float]:
        calibrated = {}
        for key in angles:
            raw = angles.get(key, 0)
            calibrated[key] = self.scale.get(key, 1.0) * raw + self.offset.get(key, 0)
        return calibrated

    def save(self, path: Path):
        data = {
            "reference_poses": self.reference_poses,
            "offset": self.offset,
            "scale": self.scale
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def load(path: Path) -> 'CalibrationData':
        with open(path, 'r') as f:
            data = json.load(f)
        cal = CalibrationData()
        cal.reference_poses = data.get("reference_poses", {})
        cal.offset = data.get("offset", cal.offset)
        cal.scale = data.get("scale", cal.scale)
        return cal


class CalibrationManager:
    def __init__(self):
        self.calibration = CalibrationData()
        self.calibrating = False
        self.current_pose_name = None

    def start_calibration_pose(self, pose_name: str):
        self.calibrating = True
        self.current_pose_name = pose_name
        print(f"Calibration: Hold '{pose_name}' pose. Press SPACE to capture...")

    def capture_pose(self, angles: Dict[str, float]):
        if self.calibrating and self.current_pose_name:
            self.calibration.add_reference_pose(self.current_pose_name, angles)
            print(f"Captured {self.current_pose_name}: {angles}")
            return True
        return False

    def finalize_calibration(self):
        self.calibration.compute_offsets()
        self.calibrating = False
        self.current_pose_name = None

    def apply(self, angles: Dict[str, float]) -> Dict[str, float]:
        return self.calibration.apply_calibration(angles)

    def save(self, path: Path):
        self.calibration.save(path)

    def load(self, path: Path):
        self.calibration = CalibrationData.load(path)