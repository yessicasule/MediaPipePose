"""
Video recording with pose overlay
"""
import cv2
from datetime import datetime
from config.config import Config

class VideoRecorder:
    def __init__(self, session_name=None):
        """
        Initialize video recorder
        
        Args:
            session_name: Optional custom session name
        """
        if session_name is None:
            session_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.session_name = session_name
        self.video_path = f"{Config.VIDEO_OUTPUT_DIR}/pose_tracking_{session_name}.mp4"
        self.writer = None
        self.is_recording = False
        
        print(f"VideoRecorder initialized: {session_name}")
    
    def start_recording(self, frame_width, frame_height, fps):
        """
        Start video recording
        
        Args:
            frame_width: Frame width in pixels
            frame_height: Frame height in pixels
            fps: Frames per second
        """
        fourcc = cv2.VideoWriter_fourcc(*Config.VIDEO_CODEC)
        self.writer = cv2.VideoWriter(
            self.video_path,
            fourcc,
            fps,
            (frame_width, frame_height)
        )
        
        if not self.writer.isOpened():
            print("ERROR: Could not open video writer!")
            return False
        
        self.is_recording = True
        print(f"Video recording started: {self.video_path}")
        return True
    
    def write_frame(self, frame):
        """
        Write a frame to video
        
        Args:
            frame: Frame to write (numpy array)
        """
        if self.is_recording and self.writer:
            self.writer.write(frame)
    
    def stop_recording(self):
        """Stop video recording and save file"""
        if self.writer:
            self.writer.release()
            self.is_recording = False
            print(f"Video saved: {self.video_path}")
    
    def get_video_path(self):
        """Get the path to the saved video"""
        return self.video_path
