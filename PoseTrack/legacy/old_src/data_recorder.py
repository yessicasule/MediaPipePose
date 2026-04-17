"""
Data recording utilities for pose landmarks
"""
import csv
import json
import time
from datetime import datetime
from config.config import Config

class DataRecorder:
    def __init__(self, session_name=None):
        """
        Initialize data recorder
        
        Args:
            session_name: Optional custom session name
        """
        if session_name is None:
            session_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.session_name = session_name
        self.landmark_data = []
        self.start_time = None
        self.frame_count = 0
        
        self.csv_path = f"{Config.DATA_OUTPUT_DIR}/landmarks_{session_name}.csv"
        self.json_path = f"{Config.DATA_OUTPUT_DIR}/landmarks_{session_name}.json"
        self.metadata_path = f"{Config.DATA_OUTPUT_DIR}/metadata_{session_name}.json"
        
        self._init_csv()
        
        print(f"DataRecorder initialized: {session_name}")
    
    def _init_csv(self):
        """Initialize CSV file with headers"""
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            headers = ['frame_index', 'timestamp']
            
            for idx in Config.UPPER_BODY_LANDMARKS:
                landmark_name = Config.LANDMARK_NAMES[idx]
                headers.extend([
                    f'{landmark_name}_x',
                    f'{landmark_name}_y',
                    f'{landmark_name}_z',
                    f'{landmark_name}_visibility'
                ])
            
            writer.writerow(headers)
    
    def start_recording(self):
        """Start recording session"""
        self.start_time = time.time()
        self.frame_count = 0
        print("Recording started")
    
    def record_frame(self, landmarks):
        """
        Record landmarks for a single frame
        
        Args:
            landmarks: MediaPipe pose landmarks object
        """
        if self.start_time is None:
            self.start_recording()
        
        self.frame_count += 1
        timestamp = time.time() - self.start_time
        
        # Prepare row data
        row_data = [self.frame_count, f"{timestamp:.4f}"]
        frame_landmarks = {}
        
        # Extract upper-body landmarks
        for idx in Config.UPPER_BODY_LANDMARKS:
            landmark = landmarks.landmark[idx]
            landmark_name = Config.LANDMARK_NAMES[idx]
            
            # Add to CSV row
            row_data.extend([
                f"{landmark.x:.6f}",
                f"{landmark.y:.6f}",
                f"{landmark.z:.6f}",
                f"{landmark.visibility:.6f}"
            ])
            
            # Add to JSON structure
            frame_landmarks[landmark_name] = {
                'x': landmark.x,
                'y': landmark.y,
                'z': landmark.z,
                'visibility': landmark.visibility
            }
        
        # Write to CSV
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row_data)
        
        # Store for JSON
        self.landmark_data.append({
            'frame_index': self.frame_count,
            'timestamp': timestamp,
            'landmarks': frame_landmarks
        })
    
    def save_session(self, fps=None, system_info=None):
        """
        Save complete session data
        
        Args:
            fps: Average FPS during recording
            system_info: System specification dictionary
        """
        with open(self.json_path, 'w') as f:
            json.dump(self.landmark_data, f, indent=2)
        
        duration = time.time() - self.start_time if self.start_time else 0
        
        metadata = {
            'session_name': self.session_name,
            'start_time': datetime.now().isoformat(),
            'duration_seconds': duration,
            'total_frames': self.frame_count,
            'average_fps': fps,
            'tracked_landmarks': [Config.LANDMARK_NAMES[idx] 
                                 for idx in Config.UPPER_BODY_LANDMARKS],
            'mediapipe_settings': {
                'min_detection_confidence': Config.MIN_DETECTION_CONFIDENCE,
                'min_tracking_confidence': Config.MIN_TRACKING_CONFIDENCE,
                'model_complexity': Config.MODEL_COMPLEXITY
            }
        }
        
        if system_info:
            metadata['system_info'] = system_info
        
        with open(self.metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\nSession data saved:")
        print(f"   CSV:      {self.csv_path}")
        print(f"   JSON:     {self.json_path}")
        print(f"   Metadata: {self.metadata_path}")
        print(f"   Frames:   {self.frame_count}")
        print(f"   Duration: {duration:.2f}s")
        if fps:
            print(f"   Avg FPS:  {fps:.2f}")
