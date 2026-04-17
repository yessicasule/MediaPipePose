"""
Upper-Body Keypoint Tracking Using MediaPipe Pose Landmarker
2-Week Assignment Implementation

This script:
- Tracks upper-body landmarks (head, shoulders, elbows, wrists) from live webcam feed
- Records video with keypoints and skeleton overlay
- Saves landmark data to CSV and JSON
- Reports FPS and system specifications
"""

import cv2
import numpy as np
import json
import csv
import time
import platform
import psutil
from datetime import datetime
from pathlib import Path
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Upper-body landmark indices (based on MediaPipe Pose model)
# 0: nose, 1-10: face, 11-12: shoulders, 13-14: elbows, 15-16: wrists
UPPER_BODY_LANDMARKS = {
    'nose': 0,
    'left_eye_inner': 1,
    'left_eye': 2,
    'left_eye_outer': 3,
    'right_eye_inner': 4,
    'right_eye': 5,
    'right_eye_outer': 6,
    'left_ear': 7,
    'right_ear': 8,
    'mouth_left': 9,
    'mouth_right': 10,
    'left_shoulder': 11,
    'right_shoulder': 12,
    'left_elbow': 13,
    'right_elbow': 14,
    'left_wrist': 15,
    'right_wrist': 16
}

# Connections for upper-body skeleton visualization
UPPER_BODY_CONNECTIONS = [
    # Face
    (0, 1), (1, 2), (2, 3),  # Left eye
    (0, 4), (4, 5), (5, 6),  # Right eye
    (0, 7), (0, 8),  # Ears
    (9, 10),  # Mouth
    # Upper body
    (11, 12),  # Shoulders
    (11, 13), (13, 15),  # Left arm
    (12, 14), (14, 16),  # Right arm
    (0, 11), (0, 12),  # Head to shoulders
]


class PoseTracker:
    def __init__(self, model_path='models/pose_landmarker_full.task', record_duration=60):
        """
        Initialize the Pose Tracker
        
        Args:
            model_path: Path to the MediaPipe Pose Landmarker model file
            record_duration: Duration to record in seconds (default 60)
        """
        self.model_path = model_path
        self.record_duration = record_duration
        self.landmark_data = []
        self.frame_count = 0
        self.start_time = None
        self.fps_history = []
        
        # Initialize MediaPipe Pose Landmarker
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            output_segmentation_masks=False,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.detector = vision.PoseLandmarker.create_from_options(options)
        
        # Create output directories
        self.output_dir = Path('output')
        self.output_dir.mkdir(exist_ok=True)
        
        # Generate output filenames with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.video_path = self.output_dir / f'pose_tracking_{timestamp}.mp4'
        self.csv_path = self.output_dir / f'landmarks_{timestamp}.csv'
        self.json_path = self.output_dir / f'landmarks_{timestamp}.json'
        self.system_info_path = self.output_dir / f'system_info_{timestamp}.txt'
        
    def get_system_info(self):
        """Collect system specifications"""
        info = {
            'platform': platform.system(),
            'platform_version': platform.version(),
            'processor': platform.processor(),
            'python_version': platform.python_version(),
            'cpu_count': psutil.cpu_count(),
            'cpu_freq': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
            'memory_total_gb': round(psutil.virtual_memory().total / (1024**3), 2),
            'memory_available_gb': round(psutil.virtual_memory().available / (1024**3), 2),
            'timestamp': datetime.now().isoformat()
        }
        return info
    
    def save_system_info(self):
        """Save system information to file"""
        info = self.get_system_info()
        with open(self.system_info_path, 'w') as f:
            f.write("System Specifications\n")
            f.write("=" * 50 + "\n\n")
            for key, value in info.items():
                f.write(f"{key.replace('_', ' ').title()}: {value}\n")
        return info
    
    def draw_upper_body_skeleton(self, image, landmarks):
        """
        Draw upper-body skeleton on the image
        
        Args:
            image: OpenCV image (BGR format)
            landmarks: MediaPipe pose landmarks
        """
        h, w = image.shape[:2]
        
        # Draw connections
        for connection in UPPER_BODY_CONNECTIONS:
            start_idx, end_idx = connection
            if start_idx < len(landmarks) and end_idx < len(landmarks):
                start = landmarks[start_idx]
                end = landmarks[end_idx]
                
                # Only draw if both landmarks are visible
                if start.visibility > 0.5 and end.visibility > 0.5:
                    start_point = (int(start.x * w), int(start.y * h))
                    end_point = (int(end.x * w), int(end.y * h))
                    cv2.line(image, start_point, end_point, (0, 255, 0), 2)
        
        # Draw keypoints
        for idx, landmark in enumerate(landmarks):
            if idx in UPPER_BODY_LANDMARKS.values() and landmark.visibility > 0.5:
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                
                # Color code by body part
                if idx in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:  # Head
                    color = (255, 0, 0)  # Blue
                elif idx in [11, 12]:  # Shoulders
                    color = (0, 255, 255)  # Yellow
                elif idx in [13, 14]:  # Elbows
                    color = (255, 165, 0)  # Orange
                elif idx in [15, 16]:  # Wrists
                    color = (0, 0, 255)  # Red
                else:
                    color = (255, 255, 255)  # White
                
                cv2.circle(image, (x, y), 5, color, -1)
                cv2.circle(image, (x, y), 8, color, 2)
        
        return image
    
    def extract_landmark_data(self, landmarks, world_landmarks, frame_idx, timestamp):
        """
        Extract landmark data for upper-body keypoints
        
        Args:
            landmarks: Normalized image coordinates
            world_landmarks: 3D world coordinates
            frame_idx: Current frame index
            timestamp: Frame timestamp
        """
        frame_data = {
            'frame_index': frame_idx,
            'timestamp': timestamp,
            'landmarks': {}
        }
        
        for name, idx in UPPER_BODY_LANDMARKS.items():
            if idx < len(landmarks):
                landmark = landmarks[idx]
                world_landmark = world_landmarks[idx] if world_landmarks and idx < len(world_landmarks) else None
                
                frame_data['landmarks'][name] = {
                    'x': landmark.x,
                    'y': landmark.y,
                    'z': landmark.z,
                    'visibility': landmark.visibility,
                    'world_x': world_landmark.x if world_landmark else None,
                    'world_y': world_landmark.y if world_landmark else None,
                    'world_z': world_landmark.z if world_landmark else None
                }
        
        return frame_data
    
    def save_to_csv(self):
        """Save landmark data to CSV file"""
        if not self.landmark_data:
            print("No landmark data to save")
            return
        
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Write header
            header = ['frame_index', 'timestamp']
            for name in UPPER_BODY_LANDMARKS.keys():
                header.extend([
                    f'{name}_x', f'{name}_y', f'{name}_z',
                    f'{name}_visibility',
                    f'{name}_world_x', f'{name}_world_y', f'{name}_world_z'
                ])
            writer.writerow(header)
            
            # Write data
            for frame_data in self.landmark_data:
                row = [frame_data['frame_index'], frame_data['timestamp']]
                for name in UPPER_BODY_LANDMARKS.keys():
                    if name in frame_data['landmarks']:
                        lm = frame_data['landmarks'][name]
                        row.extend([
                            lm['x'], lm['y'], lm['z'],
                            lm['visibility'],
                            lm['world_x'], lm['world_y'], lm['world_z']
                        ])
                    else:
                        row.extend([None] * 7)
                writer.writerow(row)
        
        print(f"✅ Landmark data saved to {self.csv_path}")
    
    def save_to_json(self):
        """Save landmark data to JSON file"""
        output_data = {
            'metadata': {
                'total_frames': len(self.landmark_data),
                'duration_seconds': self.record_duration,
                'average_fps': np.mean(self.fps_history) if self.fps_history else 0,
                'upper_body_landmarks': list(UPPER_BODY_LANDMARKS.keys()),
                'timestamp': datetime.now().isoformat()
            },
            'frames': self.landmark_data
        }
        
        with open(self.json_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"✅ Landmark data saved to {self.json_path}")
    
    def run(self):
        """Main tracking loop"""
        # Initialize camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Error: Cannot access camera")
            return
        
        # Get camera properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(self.video_path), fourcc, fps, (width, height))
        
        # Save system info
        system_info = self.save_system_info()
        print("\n" + "=" * 50)
        print("System Specifications")
        print("=" * 50)
        for key, value in system_info.items():
            print(f"{key.replace('_', ' ').title()}: {value}")
        print("=" * 50 + "\n")
        
        print(f"🎥 Starting pose tracking...")
        print(f"📹 Recording for {self.record_duration} seconds")
        print(f"💾 Output directory: {self.output_dir}")
        print(f"Press 'q' to quit early\n")
        
        self.start_time = time.time()
        last_fps_time = time.time()
        fps_frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("❌ Error: Cannot read frame")
                    break
                
                # Calculate elapsed time
                elapsed = time.time() - self.start_time
                if elapsed >= self.record_duration:
                    print(f"\n⏱️ Recording duration ({self.record_duration}s) reached")
                    break
                
                # Convert frame to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                
                # Detect pose
                detection_result = self.detector.detect(mp_image)
                
                # Process results
                if detection_result.pose_landmarks:
                    landmarks = detection_result.pose_landmarks[0]
                    world_landmarks = detection_result.pose_world_landmarks[0] if detection_result.pose_world_landmarks else None
                    
                    # Draw skeleton
                    frame = self.draw_upper_body_skeleton(frame, landmarks)
                    
                    # Extract and store landmark data
                    frame_data = self.extract_landmark_data(
                        landmarks, world_landmarks, self.frame_count, elapsed
                    )
                    self.landmark_data.append(frame_data)
                else:
                    # No pose detected - still record frame data
                    frame_data = {
                        'frame_index': self.frame_count,
                        'timestamp': elapsed,
                        'landmarks': {}
                    }
                    self.landmark_data.append(frame_data)
                
                # Calculate FPS
                fps_frame_count += 1
                if time.time() - last_fps_time >= 1.0:
                    current_fps = fps_frame_count / (time.time() - last_fps_time)
                    self.fps_history.append(current_fps)
                    fps_frame_count = 0
                    last_fps_time = time.time()
                
                # Display info on frame
                avg_fps = np.mean(self.fps_history) if self.fps_history else 0
                status_text = f"Frame: {self.frame_count} | Time: {elapsed:.1f}s | FPS: {avg_fps:.1f}"
                cv2.putText(frame, status_text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                if detection_result.pose_landmarks:
                    cv2.putText(frame, "Pose Detected", (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, "No Pose", (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Write frame to video
                out.write(frame)
                
                # Display frame
                cv2.imshow('Pose Tracking', frame)
                
                self.frame_count += 1
                
                # Check for quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\n⚠️ Recording stopped by user")
                    break
        
        finally:
            # Cleanup
            cap.release()
            out.release()
            cv2.destroyAllWindows()
            self.detector.close()
            
            # Save data
            print("\n💾 Saving data...")
            self.save_to_csv()
            self.save_to_json()
            
            # Print summary
            print("\n" + "=" * 50)
            print("Recording Summary")
            print("=" * 50)
            print(f"Total frames: {self.frame_count}")
            print(f"Duration: {elapsed:.2f} seconds")
            print(f"Average FPS: {np.mean(self.fps_history):.2f}" if self.fps_history else "N/A")
            print(f"Video saved: {self.video_path}")
            print(f"CSV saved: {self.csv_path}")
            print(f"JSON saved: {self.json_path}")
            print(f"System info saved: {self.system_info_path}")
            print("=" * 50)


if __name__ == "__main__":
    # Check if model file exists
    model_path = Path('models/pose_landmarker_full.task')
    if not model_path.exists():
        print(f"❌ Error: Model file not found at {model_path}")
        print("Please download the model from:")
        print("https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task")
        exit(1)
    
    # Create and run tracker
    tracker = PoseTracker(
        model_path=str(model_path),
        record_duration=60  # Record for 60 seconds
    )
    tracker.run()

