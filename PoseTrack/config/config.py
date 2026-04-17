import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_VIDEOS_DIR = DATA_DIR / "raw_videos"
FRAMES_DIR = DATA_DIR / "frames"
LANDMARKS_DIR = DATA_DIR / "landmarks"
ANGLES_DIR = DATA_DIR / "angles"
OUTPUTS_DIR = BASE_DIR / "outputs"

# UDP Streaming Settings
UDP_IP = "127.0.0.1"
UDP_PORT = 9000
STREAM_HZ = 30.0

# Filter Settings
DEFAULT_FILTER_TYPE = "kalman" # options: 'ema', 'kalman'

# Normalizations
# Unity expects angles in degrees.
# 0 degrees = arm straight down
# 90 degrees = arm straight out/forward
ANGLE_UNITS = "degrees"

class Config:
    CAMERA_ID = 0
    FRAME_WIDTH = 640
    FRAME_HEIGHT = 480
    
    MIN_DETECTION_CONFIDENCE = 0.5
    MIN_TRACKING_CONFIDENCE = 0.5
    MODEL_COMPLEXITY = 1
    SMOOTH_LANDMARKS = True
    
    RECORD_DURATION = 60
    OUTPUT_VIDEO_FPS = 30
    VIDEO_CODEC = 'mp4v'
    
    @classmethod
    def ensure_directories(cls):
        for d in [RAW_VIDEOS_DIR, FRAMES_DIR, LANDMARKS_DIR, ANGLES_DIR, OUTPUTS_DIR]:
            d.mkdir(parents=True, exist_ok=True)
        print("Output directories created/verified")

Config.ensure_directories()
