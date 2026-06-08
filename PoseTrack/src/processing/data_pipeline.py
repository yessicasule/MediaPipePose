import os
import json
import numpy as np
import pandas as pd
import subprocess
from pathlib import Path
import sys
import cv2

# Ensure we can import from PoseTrack
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from config.config import DATA_DIR, FRAMES_DIR, LANDMARKS_DIR, ANGLES_DIR
from src.utils.io_utils import extract_frames_from_video
from src.processing.joint_angle_estimator import Vec3, compute_all

try:
    from src.pose.mediapipe_runner import MediaPipeRunner
    mediapipe_available = True
except ImportError as e:
    print(f"Warning: MediaPipe is not available: {e}")
    mediapipe_available = False

try:
    from src.pose.movenet_runner import MoveNetRunner
    movenet_available = True
except ImportError as e:
    print(f"Warning: MoveNet is not available: {e}")
    movenet_available = False

class DataPipeline:
    def __init__(self, output_dir=None):
        self.output_dir = Path(output_dir) if output_dir else DATA_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def run_mediapipe_on_frames(self, frames_dir: Path) -> list:
        """Runs MediaPipe on a directory of frames and returns raw landmarks."""
        if not mediapipe_available:
            print("MediaPipe is not available. Skipping.")
            return []
        print("Running MediaPipe pose estimation...")
        runner = MediaPipeRunner(model_complexity=1)
        frames = sorted(list(frames_dir.glob("*.jpg")))
        results = []
        
        for idx, fp in enumerate(frames):
            img = cv2.imread(str(fp))
            if img is None:
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            landmarks = runner.process(img_rgb)
            
            frame_kps = []
            if landmarks:
                # landmarks is a list/collection of landmarks
                for lm in landmarks:
                    # MediaPipe landmarks have x, y, z, visibility
                    frame_kps.append([lm.x, lm.y, lm.z, getattr(lm, 'visibility', 1.0)])
            else:
                frame_kps = None
                
            results.append({
                "frame_index": idx,
                "frame_file": fp.name,
                "keypoints": frame_kps
            })
            
        runner.close()
        return results

    def run_movenet_on_frames(self, frames_dir: Path, model_name="lightning") -> list:
        """Runs MoveNet on a directory of frames and returns raw landmarks."""
        if not movenet_available:
            print("MoveNet is not available. Skipping.")
            return []
        print("Running MoveNet pose estimation...")
        runner = MoveNetRunner(model_name=f"movenet_{model_name}")
        frames = sorted(list(frames_dir.glob("*.jpg")))
        results = []
        
        for idx, fp in enumerate(frames):
            img = cv2.imread(str(fp))
            if img is None:
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            landmarks = runner.process(img_rgb)
            
            frame_kps = []
            if landmarks is not None:
                # MoveNet outputs [17, 3] corresponding to y, x, score
                for kp in landmarks:
                    # Normalize/store as [x, y, score]
                    frame_kps.append([float(kp[1]), float(kp[0]), float(kp[2])])
            else:
                frame_kps = None
                
            results.append({
                "frame_index": idx,
                "frame_file": fp.name,
                "keypoints": frame_kps
            })
            
        runner.close()
        return results

    def run_posenet_on_frames(self, frames_dir: Path) -> list:
        """Runs PoseNet Node.js subprocess on a directory of frames and parses the output JSON."""
        print("Running PoseNet pose estimation via Node.js...")
        posenet_script = Path(__file__).resolve().parent.parent.parent / "benchmarks" / "posenet_tfjs" / "run_posenet_on_frames.mjs"
        temp_out_json = self.output_dir / "temp_posenet_output.json"
        
        cmd = [
            "node", str(posenet_script),
            "--frames_dir", str(frames_dir),
            "--out_json", str(temp_out_json),
            "--architecture", "MobileNetV1"
        ]
        
        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            with open(temp_out_json, 'r') as f:
                data = json.load(f)
            
            # Clean up temp file
            if temp_out_json.exists():
                temp_out_json.unlink()
                
            results = []
            for item in data.get("per_frame", []):
                results.append({
                    "frame_index": item["frame_index"],
                    "frame_file": item["frame_file"],
                    "keypoints": item["keypoints_xy_score"] # list of [x_px, y_px, score]
                })
            return results
        except Exception as e:
            print(f"Failed to run PoseNet: {e}")
            return []

    def compute_angles_for_framework(self, tracking_results: list, framework: str) -> list:
        """Converts keypoints from different frameworks to left-arm angles."""
        angles_list = []
        
        for item in tracking_results:
            frame_idx = item["frame_index"]
            kps = item["keypoints"]
            
            if not kps:
                angles_list.append({
                    "frame": frame_idx,
                    "shoulder_pitch": 0.0,
                    "shoulder_roll": 0.0,
                    "shoulder_yaw": 0.0,
                    "elbow_flexion": 0.0,
                    "valid": False
                })
                continue
                
            # Build list of Vec3 objects according to the framework's joint indices
            landmarks = [None] * 33
            
            try:
                if framework == "mediapipe":
                    # MediaPipe has 33 joints.
                    # Mapping directly: LShoulder = 11, RShoulder = 12, LElbow = 13, LWrist = 15, LHip = 23, RHip = 24
                    for i, kp in enumerate(kps):
                        landmarks[i] = Vec3(kp[0], kp[1], kp[2])
                elif framework in ["movenet", "posenet"]:
                    # COCO mapping:
                    # LShoulder = 5, RShoulder = 6, LElbow = 7, LWrist = 9, LHip = 11, RHip = 12
                    # We map them to the indices expected by joint_angle_estimator (11, 12, 13, 15, 23, 24)
                    coco_to_estimator = {
                        5: 11,  # LShoulder
                        6: 12,  # RShoulder
                        7: 13,  # LElbow
                        9: 15,  # LWrist
                        11: 23, # LHip
                        12: 24  # RHip
                    }
                    for coco_idx, est_idx in coco_to_estimator.items():
                        if coco_idx < len(kps):
                            kp = kps[coco_idx]
                            landmarks[est_idx] = Vec3(kp[0], kp[1], 0.0) # Z set to 0.0 for 2D models
                            
                # Compute joint angles using standard joint_angle_estimator
                angles = compute_all(landmarks)
                
                angles_list.append({
                    "frame": frame_idx,
                    "shoulder_pitch": angles.get("shoulder_elevation", 0.0),
                    "shoulder_roll": angles.get("shoulder_roll", 0.0),
                    "shoulder_yaw": angles.get("shoulder_yaw", 0.0),
                    "elbow_flexion": angles.get("elbow_flexion", 0.0),
                    "valid": True
                })
            except Exception as e:
                angles_list.append({
                    "frame": frame_idx,
                    "shoulder_pitch": 0.0,
                    "shoulder_roll": 0.0,
                    "shoulder_yaw": 0.0,
                    "elbow_flexion": 0.0,
                    "valid": False
                })
                
        return angles_list

    def parse_h36m_gt(self, txt_path: Path) -> list:
        """Parses Human3.6M 3D coordinates text file and returns Left Arm angles."""
        print(f"Parsing H3.6M ground truth from {txt_path}...")
        raw_lines = txt_path.read_text().strip().split("\n")
        
        gt_angles = []
        for idx, line in enumerate(raw_lines):
            parts = [float(p) for p in line.split(",") if p.strip()]
            if not parts:
                continue
                
            # Reshape 99 values to (33 joints, 3 coordinates)
            joints = np.array(parts).reshape(-1, 3)
            
            # Map H3.6M joints to expected estimator indices:
            # joint 17: LShoulder -> map to landmarks[11]
            # joint 25: RShoulder -> map to landmarks[12]
            # joint 18: LElbow    -> map to landmarks[13]
            # joint 19: LWrist    -> map to landmarks[15]
            # joint 6:  LHip      -> map to landmarks[23]
            # joint 1:  RHip      -> map to landmarks[24]
            landmarks = [None] * 25
            landmarks[11] = Vec3(joints[17, 0], joints[17, 1], joints[17, 2])
            landmarks[12] = Vec3(joints[25, 0], joints[25, 1], joints[25, 2])
            landmarks[13] = Vec3(joints[18, 0], joints[18, 1], joints[18, 2])
            landmarks[15] = Vec3(joints[19, 0], joints[19, 1], joints[19, 2])
            landmarks[23] = Vec3(joints[6, 0], joints[6, 1], joints[6, 2])
            landmarks[24] = Vec3(joints[1, 0], joints[1, 1], joints[1, 2])
            
            angles = compute_all(landmarks)
            
            gt_angles.append({
                "frame": idx,
                "shoulder_pitch": angles.get("shoulder_elevation", 0.0),
                "shoulder_roll": angles.get("shoulder_roll", 0.0),
                "shoulder_yaw": angles.get("shoulder_yaw", 0.0),
                "elbow_flexion": angles.get("elbow_flexion", 0.0)
            })
            
        return gt_angles

    def run_full_pipeline(self, video_path: Path, h36m_gt_path: Path, output_csv_path: Path):
        """Processes video and ground truth end-to-end and creates unified dataset."""
        # 1. Extract frames
        frames_dir = self.output_dir / "temp_frames"
        print(f"Extracting frames from {video_path}...")
        extract_frames_from_video(str(video_path), str(frames_dir))
        
        # 2. Run frameworks
        mp_raw = self.run_mediapipe_on_frames(frames_dir)
        mv_raw = self.run_movenet_on_frames(frames_dir)
        pn_raw = self.run_posenet_on_frames(frames_dir)
        
        # 3. Clean up frames
        for f in frames_dir.glob("*.jpg"):
            f.unlink()
        if frames_dir.exists():
            frames_dir.rmdir()
            
        # 4. Compute angles
        mp_angles = self.compute_angles_for_framework(mp_raw, "mediapipe")
        mv_angles = self.compute_angles_for_framework(mv_raw, "movenet")
        pn_angles = self.compute_angles_for_framework(pn_raw, "posenet")
        
        # 5. Parse ground truth
        gt_angles = self.parse_h36m_gt(h36m_gt_path)
        
        # 6. Merge all
        min_len = min(len(mp_angles), len(mv_angles), len(pn_angles), len(gt_angles))
        
        data_rows = []
        for i in range(min_len):
            row = {
                "frame": i,
                
                # MediaPipe
                "mp_shoulder_pitch": mp_angles[i]["shoulder_pitch"],
                "mp_shoulder_roll": mp_angles[i]["shoulder_roll"],
                "mp_shoulder_yaw": mp_angles[i]["shoulder_yaw"],
                "mp_elbow_flexion": mp_angles[i]["elbow_flexion"],
                
                # MoveNet
                "mv_shoulder_pitch": mv_angles[i]["shoulder_pitch"],
                "mv_shoulder_roll": mv_angles[i]["shoulder_roll"],
                "mv_shoulder_yaw": mv_angles[i]["shoulder_yaw"],
                "mv_elbow_flexion": mv_angles[i]["elbow_flexion"],
                
                # PoseNet
                "pn_shoulder_pitch": pn_angles[i]["shoulder_pitch"],
                "pn_shoulder_roll": pn_angles[i]["shoulder_roll"],
                "pn_shoulder_yaw": pn_angles[i]["shoulder_yaw"],
                "pn_elbow_flexion": pn_angles[i]["elbow_flexion"],
                
                # Ground Truth
                "gt_shoulder_pitch": gt_angles[i]["shoulder_pitch"],
                "gt_shoulder_roll": gt_angles[i]["shoulder_roll"],
                "gt_shoulder_yaw": gt_angles[i]["shoulder_yaw"],
                "gt_elbow_flexion": gt_angles[i]["elbow_flexion"]
            }
            data_rows.append(row)
            
        df = pd.DataFrame(data_rows)
        output_csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv_path, index=False)
        print(f"Pipeline complete. Unified training data saved to {output_csv_path}")
        return df

if __name__ == "__main__":
    # Test script if executed directly
    pipeline = DataPipeline()
    print("DataPipeline initialized.")
