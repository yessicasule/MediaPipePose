import cv2
import mediapipe as mp
import time
import platform
import psutil
from config.config import Config
from src.data_recorder import DataRecorder
from src.video_recorder import VideoRecorder
from src.joint_angle_estimator import compute_all
from src.angle_filter import KalmanFilter1D
from src.udp_angle_streamer import UdpAngleStreamer

class PoseTracker:
    def __init__(self, session_name=None):
        """
        Initialize pose tracker with recording
        
        Args:
            session_name: Optional session name for outputs
        """
        Config.create_output_directories()
        
        self.session_name = session_name
        
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=Config.MODEL_COMPLEXITY,
            smooth_landmarks=Config.SMOOTH_LANDMARKS,
            min_detection_confidence=Config.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=Config.MIN_TRACKING_CONFIDENCE
        )
        
        self.data_recorder = DataRecorder(session_name)
        self.video_recorder = VideoRecorder(session_name)
        
        self.cap = None
        self.frame_count = 0
        self.detection_count = 0
        self.start_time = None
        self.is_recording = False

        self._angle_filters = {
            k: KalmanFilter1D(process_noise=0.01, measurement_noise=1.0)
            for k in ("elbow_flexion", "shoulder_elevation", "shoulder_horizontal")
        }
        self._angles = {"elbow_flexion": 0.0, "shoulder_elevation": 0.0, "shoulder_horizontal": 0.0}
        self._streamer: UdpAngleStreamer | None = None

        print("PoseTracker initialized")
    
    def get_system_info(self):
        """Get system specifications"""
        return {
            'os': f"{platform.system()} {platform.release()}",
            'processor': platform.processor(),
            'cpu_cores': psutil.cpu_count(logical=True),
            'ram_gb': round(psutil.virtual_memory().total / (1024**3), 2)
        }
    
    def start_camera(self):
        """Initialize camera"""
        self.cap = cv2.VideoCapture(Config.CAMERA_ID)
        
        if not self.cap.isOpened():
            print("ERROR: Cannot access camera!")
            return False
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.FRAME_HEIGHT)
        
        print("Camera started")
        return True
    
    def start_recording(self):
        """Start recording session"""
        if not self.is_recording:
            ret, frame = self.cap.read()
            if ret:
                h, w, _ = frame.shape
                self.video_recorder.start_recording(w, h, Config.OUTPUT_VIDEO_FPS)
                self.data_recorder.start_recording()
                self.start_time = time.time()
                self.is_recording = True
                self.frame_count = 0
                self.detection_count = 0
    
    def start_streaming(self, host: str = "127.0.0.1", port: int = 9000, hz: float = 30.0):
        self._streamer = UdpAngleStreamer(host, port, hz)
        self._streamer.start()
        print(f"UDP streaming to {host}:{port} at {hz} Hz")

    def stop_streaming(self):
        if self._streamer:
            self._streamer.stop()
            self._streamer = None

    def process_frame(self, frame):
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)

        pose_detected = False
        if results.pose_landmarks:
            pose_detected = True
            self.detection_count += 1

            self.mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )

            h, w, _ = frame.shape
            for idx in Config.UPPER_BODY_LANDMARKS:
                landmark = results.pose_landmarks.landmark[idx]
                if landmark.visibility > 0.5:
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    cv2.circle(frame, (x, y), 6, (0, 255, 255), -1)

            raw = compute_all(results.pose_landmarks.landmark)
            for k, f in self._angle_filters.items():
                self._angles[k] = f.update(raw[k])

            if self._streamer:
                ef = self._angles["elbow_flexion"]
                se = self._angles["shoulder_elevation"]
                sh = self._angles["shoulder_horizontal"]
                self._streamer.update_angles(se, sh, 0.0, ef)

            if self.is_recording:
                self.data_recorder.record_frame(results.pose_landmarks)

        return frame, pose_detected, results
    
    def add_overlay_info(self, frame, fps, pose_detected):
        status = "TRACKING" if pose_detected else "NO POSE"
        color  = (0, 255, 0) if pose_detected else (0, 0, 255)

        if self.is_recording:
            cv2.circle(frame, (20, 20), 10, (0, 0, 255), -1)
            cv2.putText(frame, "REC", (35, 27),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.putText(frame, f"Status: {status}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if pose_detected:
            ef = self._angles["elbow_flexion"]
            se = self._angles["shoulder_elevation"]
            sh = self._angles["shoulder_horizontal"]
            cv2.putText(frame, f"Elbow:    {ef:5.1f} deg", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
            cv2.putText(frame, f"Sh.Elev:  {se:5.1f} deg", (10, 145),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
            cv2.putText(frame, f"Sh.Horiz: {sh:5.1f} deg", (10, 170),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)

        if self.is_recording:
            duration = time.time() - self.start_time
            cv2.putText(frame, f"Duration: {duration:.1f}s", (10, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"Frames: {self.frame_count}", (10, 225),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return frame
    
    def run(self, duration=None):
        """
        Run pose tracking session
        
        Args:
            duration: Maximum recording duration in seconds (None = manual stop)
        """
        if not self.start_camera():
            return
        
        print("\n" + "="*60)
        print("POSE TRACKING SESSION")
        print("="*60)
        print("\nControls:")
        print("   Press SPACE to start/stop recording")
        print("   Press 'q' to quit")
        print("="*60 + "\n")
        
        recording_start_time = None
        
        while True:
            ret, frame = self.cap.read()
            
            if not ret:
                print("ERROR: Cannot read frame!")
                break
            
            self.frame_count += 1
            
            # Process frame
            processed_frame, pose_detected, results = self.process_frame(frame)
            
            # Calculate FPS
            if self.start_time:
                elapsed = time.time() - self.start_time
                fps = self.frame_count / elapsed if elapsed > 0 else 0
            else:
                fps = 0
            
            # Add overlay
            display_frame = self.add_overlay_info(processed_frame, fps, pose_detected)
            
            # Write to video if recording
            if self.is_recording:
                self.video_recorder.write_frame(display_frame)
                
                # Check duration limit
                if duration and recording_start_time:
                    if time.time() - recording_start_time >= duration:
                        print(f"\nMaximum duration ({duration}s) reached")
                        break
            
            # Display
            cv2.imshow('Pose Tracking - Upper Body', display_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord(' '):  # Spacebar
                if not self.is_recording:
                    self.start_recording()
                    recording_start_time = time.time()
                    print("Recording started - Press SPACE to stop")
                else:
                    break
        
        # Cleanup
        self.stop()
    
    def stop(self):
        """Stop tracking and save data"""
        if self.is_recording:
            self.video_recorder.stop_recording()
            
            if self.start_time:
                elapsed = time.time() - self.start_time
                fps = self.frame_count / elapsed if elapsed > 0 else 0
            else:
                fps = 0
            
            system_info = self.get_system_info()
            self.data_recorder.save_session(fps, system_info)
            
            detection_rate = (self.detection_count / self.frame_count * 100) if self.frame_count > 0 else 0
            
            print("\n" + "="*60)
            print("SESSION SUMMARY")
            print("="*60)
            print(f"Total frames: {self.frame_count}")
            print(f"Poses detected: {self.detection_count}")
            print(f"Detection rate: {detection_rate:.1f}%")
            print(f"Average FPS: {fps:.2f}")
            print(f"Duration: {elapsed:.2f}s")
            print("="*60 + "\n")
        
        self.stop_streaming()
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        self.pose.close()
        print("Session ended\n")
