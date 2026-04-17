"""
Synchronized Video and Motion Capture Data Recorder

Collects synchronized video frames with corresponding motion data for consistent
multi-framework benchmarking. This ensures the exact same video data is provided
to all frameworks (MediaPipe, PoseNet, MoveNet) for fair comparison.
"""
import csv
import json
import time
import uuid
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
import cv2
import numpy as np


@dataclass
class FrameData:
    """Container for synchronized frame data"""
    frame_index: int
    timestamp: float
    video_frame_path: str
    motion_data: Dict[str, Any]
    camera_extrinsics: Optional[Dict[str, float]] = None
    camera_intrinsics: Optional[Dict[str, float]] = None


@dataclass
class SessionMetadata:
    """Metadata for recording session"""
    session_id: str
    session_name: str
    start_time: str
    end_time: Optional[str] = None
    duration_seconds: float = 0.0
    total_frames: int = 0
    video_fps: float = 0.0
    video_width: int = 0
    video_height: int = 0
    motion_capture_system: str = "none"
    motion_capture_rate: float = 0.0
    frameworks_to_benchmark: List[str] = field(default_factory=lambda: ["mediapipe", "posenet", "movenet"])


class SynchronizedRecorder:
    """
    Records synchronized video and motion capture data.
    
    Features:
    - High-quality video recording with timestamp sync
    - Mock motion capture data (ready for real mocap integration)
    - Frame-level synchronization markers
    - Metadata preservation for consistent benchmarking
    """
    
    LANDMARK_NAMES = {
        0: "nose", 1: "left_eye_inner", 2: "left_eye", 3: "left_eye_outer",
        4: "right_eye_inner", 5: "right_eye", 6: "right_eye_outer",
        7: "left_ear", 8: "right_ear", 9: "mouth_left", 10: "mouth_right",
        11: "left_shoulder", 12: "right_shoulder", 13: "left_elbow",
        14: "right_elbow", 15: "left_wrist", 16: "right_wrist",
        23: "left_hip", 24: "right_hip"
    }
    
    def __init__(
        self,
        session_name: Optional[str] = None,
        output_dir: str = "outputs/synchronized",
        camera_id: int = 0,
        target_fps: float = 30.0,
        video_width: int = 1280,
        video_height: int = 720,
        motion_capture_enabled: bool = False,
        motion_capture_system: str = "none"
    ):
        self.session_id = str(uuid.uuid4())[:8]
        self.session_name = session_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(output_dir)
        self.session_dir = self.output_dir / f"{self.session_name}_{self.session_id}"
        
        self.camera_id = camera_id
        self.target_fps = target_fps
        self.video_width = video_width
        self.video_height = video_height
        
        self.motion_capture_enabled = motion_capture_enabled
        self.motion_capture_system = motion_capture_system
        self.motion_capture_rate = 100.0 if motion_capture_enabled else 0.0
        
        self.metadata = SessionMetadata(
            session_id=self.session_id,
            session_name=self.session_name,
            start_time=datetime.now().isoformat(),
            motion_capture_system=motion_capture_system,
            motion_capture_rate=self.motion_capture_rate,
            video_fps=target_fps,
            video_width=video_width,
            video_height=video_height
        )
        
        self.frames: List[FrameData] = []
        self.frame_count = 0
        self.start_time = None
        self.cap = None
        self.writer = None
        self.is_recording = False
        
        self._create_directories()
    
    def _create_directories(self):
        """Create output directories"""
        self.session_dir.mkdir(parents=True, exist_ok=True)
        (self.session_dir / "frames").mkdir(exist_ok=True)
        (self.session_dir / "motion_data").mkdir(exist_ok=True)
        print(f"Session directory: {self.session_dir}")
    
    def _get_camera_calibration(self) -> Dict[str, Any]:
        """Get camera calibration parameters"""
        return {
            "fx": self.video_width * 0.9,
            "fy": self.video_height * 0.9,
            "cx": self.video_width / 2,
            "cy": self.video_height / 2,
            "width": self.video_width,
            "height": self.video_height
        }
    
    def _simulate_motion_capture(self, frame_idx: int, timestamp: float) -> Dict[str, Any]:
        """
        Simulate motion capture data.
        Replace this with actual mocap SDK integration (OptiTrack, Vicon, etc.)
        """
        t = timestamp
        mocap_data = {
            "markers": [],
            "body_segments": {},
            "timestamp": timestamp,
            "mocap_frame": frame_idx
        }
        
        for idx, name in self.LANDMARK_NAMES.items():
            if idx in [11, 12, 13, 14, 15, 16, 0]:
                phase = t * 2.0 + idx * 0.1
                base_y = {"left_shoulder": 0.3, "right_shoulder": 0.3, 
                          "left_elbow": 0.4, "right_elbow": 0.4,
                          "left_wrist": 0.5, "right_wrist": 0.5,
                          "nose": 0.15}[name]
                
                mocap_data["markers"].append({
                    "id": idx,
                    "name": name,
                    "x": 0.5 + 0.1 * np.sin(phase) + 0.02 * np.random.randn(),
                    "y": base_y + 0.05 * np.sin(phase * 1.5) + 0.02 * np.random.randn(),
                    "z": 0.1 * np.cos(phase),
                    "visibility": 0.95 + 0.05 * np.random.rand()
                })
        
        return mocap_data
    
    def _connect_mocap_to_camera(
        self,
        mocap_timestamp: float,
        frame_timestamp: float
    ) -> float:
        """Calculate synchronization offset between mocap and camera"""
        return abs(mocap_timestamp - frame_timestamp)
    
    def start_recording(self, duration_seconds: Optional[float] = None):
        """
        Start synchronized recording
        
        Args:
            duration_seconds: Optional recording duration (None = manual stop)
        """
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera {self.camera_id}")
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.video_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.video_height)
        self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)
        
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        self.video_width = actual_width
        self.video_height = actual_height
        self.target_fps = actual_fps or self.target_fps
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_path = self.session_dir / "video" / f"recording_{self.session_id}.mp4"
        self.video_path.parent.mkdir(exist_ok=True)
        self.writer = cv2.VideoWriter(
            str(self.video_path), fourcc, self.target_fps,
            (actual_width, actual_height)
        )
        
        self.metadata.video_width = actual_width
        self.metadata.video_height = actual_height
        self.metadata.video_fps = actual_fps or self.target_fps
        
        self.start_time = time.perf_counter()
        self.is_recording = True
        self.frame_count = 0
        
        self.metadata.start_time = datetime.now().isoformat()
        
        print(f"\nRecording started:")
        print(f"  Resolution: {actual_width}x{actual_height}")
        print(f"  FPS: {actual_fps or self.target_fps:.1f}")
        print(f"  Duration: {'Unlimited (press q to stop)' if duration_seconds is None else f'{duration_seconds}s'}")
        print(f"  Output: {self.session_dir}")
    
    def record_frame(self, frame: Optional[np.ndarray] = None) -> bool:
        """
        Record a single frame with synchronized motion data
        
        Args:
            frame: Optional pre-captured frame (None = capture from camera)
        
        Returns:
            True if frame recorded, False if recording stopped
        """
        if not self.is_recording:
            return False
        
        timestamp = time.perf_counter() - self.start_time
        
        if frame is None:
            ret, frame = self.cap.read()
            if not ret:
                return False
        
        frame_filename = f"frame_{self.frame_count:06d}.jpg"
        frame_path = self.session_dir / "frames" / frame_filename
        cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        mocap_data = self._simulate_motion_capture(self.frame_count, timestamp)
        
        sync_offset = self._connect_mocap_to_camera(
            mocap_data["timestamp"], timestamp
        )
        
        frame_data = FrameData(
            frame_index=self.frame_count,
            timestamp=timestamp,
            video_frame_path=str(frame_path),
            motion_data=mocap_data,
            camera_extrinsics={"sync_offset_ms": sync_offset * 1000},
            camera_intrinsics=self._get_camera_calibration()
        )
        
        self.frames.append(frame_data)
        
        self.writer.write(frame)
        self.frame_count += 1
        
        return True
    
    def record_loop(self, max_duration: Optional[float] = None):
        """Recording loop with display"""
        display_text = f"Recording: {self.session_name[:20]}"
        
        try:
            while self.is_recording:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                timestamp = time.perf_counter() - self.start_time
                
                annotated_frame = frame.copy()
                cv2.putText(annotated_frame, display_text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(annotated_frame, f"Frame: {self.frame_count}", (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(annotated_frame, f"Time: {timestamp:.1f}s", (10, 110),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                cv2.imshow("Synchronized Recording", annotated_frame)
                
                if not self.record_frame(frame):
                    break
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.stop_recording()
                    break
                
                if max_duration and timestamp >= max_duration:
                    print(f"\nMax duration reached: {max_duration}s")
                    self.stop_recording()
                    break
                    
        except KeyboardInterrupt:
            self.stop_recording()
    
    def stop_recording(self):
        """Stop recording and save all data"""
        if not self.is_recording:
            return
        
        self.is_recording = False
        
        if self.cap:
            self.cap.release()
        if self.writer:
            self.writer.release()
        
        cv2.destroyAllWindows()
        
        self.metadata.end_time = datetime.now().isoformat()
        self.metadata.duration_seconds = time.perf_counter() - self.start_time
        self.metadata.total_frames = self.frame_count
        
        self._save_session_data()
        
        print(f"\nRecording stopped:")
        print(f"  Frames: {self.frame_count}")
        print(f"  Duration: {self.metadata.duration_seconds:.2f}s")
        print(f"  Avg FPS: {self.frame_count / self.metadata.duration_seconds:.2f}")
    
    def _save_session_data(self):
        """Save all session data"""
        self._save_metadata()
        self._save_motion_data()
        self._save_frame_index()
        
        print(f"\nData saved to: {self.session_dir}")
    
    def _save_metadata(self):
        """Save session metadata"""
        meta_path = self.session_dir / "metadata.json"
        with open(meta_path, 'w') as f:
            json.dump(asdict(self.metadata), f, indent=2)
    
    def _save_motion_data(self):
        """Save motion capture data"""
        mocap_combined = {
            "session_id": self.session_id,
            "session_name": self.session_name,
            "total_frames": len(self.frames),
            "motion_capture_system": self.motion_capture_system,
            "motion_capture_rate": self.motion_capture_rate,
            "frames": [f.motion_data for f in self.frames]
        }
        
        mocap_path = self.session_dir / "motion_data" / "mocap_data.json"
        with open(mocap_path, 'w') as f:
            json.dump(mocap_combined, f, indent=2)
    
    def _save_frame_index(self):
        """Save frame index for quick access"""
        frame_index = {
            "session_id": self.session_id,
            "total_frames": len(self.frames),
            "frames": [
                {
                    "frame_index": f.frame_index,
                    "timestamp": f.timestamp,
                    "video_frame_path": f.video_frame_path,
                    "frame_filename": Path(f.video_frame_path).name
                }
                for f in self.frames
            ],
            "video_path": str(self.video_path)
        }
        
        index_path = self.session_dir / "frame_index.json"
        with open(index_path, 'w') as f:
            json.dump(frame_index, f, indent=2)
        
        csv_path = self.session_dir / "frame_index.csv"
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["frame_index", "timestamp", "frame_filename"])
            for f in self.frames:
                writer.writerow([f.frame_index, f.timestamp, Path(f.video_frame_path).name])
    
    def get_session_path(self) -> Path:
        """Get the session directory path"""
        return self.session_dir
    
    def get_video_path(self) -> Path:
        """Get the recorded video path"""
        return self.video_path
    
    def get_frames_dir(self) -> Path:
        """Get the extracted frames directory"""
        return self.session_dir / "frames"


def record_synchronized_session(
    session_name: Optional[str] = None,
    duration: float = 30.0,
    output_dir: str = "outputs/synchronized"
) -> SynchronizedRecorder:
    """
    Convenience function to record a synchronized session
    
    Args:
        session_name: Optional session name (default: timestamp-based)
        duration: Recording duration in seconds (0 = unlimited)
        output_dir: Output directory
    
    Returns:
        SynchronizedRecorder instance with recorded data
    """
    recorder = SynchronizedRecorder(
        session_name=session_name,
        output_dir=output_dir,
        target_fps=30.0,
        video_width=1280,
        video_height=720
    )
    
    recorder.start_recording()
    
    if duration > 0:
        recorder.record_loop(max_duration=duration)
    else:
        recorder.record_loop()
    
    return recorder


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Synchronized Video + Motion Capture Recorder")
    parser.add_argument("--session_name", type=str, default=None, help="Session name")
    parser.add_argument("--duration", type=float, default=30.0, help="Recording duration (0=unlimited)")
    parser.add_argument("--output_dir", type=str, default="outputs/synchronized", help="Output directory")
    parser.add_argument("--camera_id", type=int, default=0, help="Camera ID")
    
    args = parser.parse_args()
    
    recorder = SynchronizedRecorder(
        session_name=args.session_name,
        output_dir=args.output_dir,
        camera_id=args.camera_id
    )
    
    recorder.start_recording()
    recorder.record_loop(max_duration=args.duration if args.duration > 0 else None)
