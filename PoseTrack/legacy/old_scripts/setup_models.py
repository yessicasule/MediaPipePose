"""
Setup and verify MediaPipe Pose models
"""
import os
import sys

# Set matplotlib backend before any imports that might use it
os.environ['MPLBACKEND'] = 'Agg'

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import urllib.request

def download_pose_model():
    """Download MediaPipe Pose Landmarker model"""
    
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    
    model_path = os.path.join(model_dir, "pose_landmarker_lite.task")
    
    if os.path.exists(model_path):
        print(f"Model already exists at {model_path}")
        return model_path
    
    print("Downloading Pose Landmarker model...")
    
    model_url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
    
    try:
        urllib.request.urlretrieve(model_url, model_path)
        print(f"Model downloaded successfully to {model_path}")
        print(f"   Size: {os.path.getsize(model_path) / (1024*1024):.2f} MB")
        return model_path
    except Exception as e:
        print(f"Error downloading model: {e}")
        return None

def verify_mediapipe_installation():
    """Verify MediaPipe installation"""
    
    print("\n" + "="*60)
    print("MEDIAPIPE INSTALLATION VERIFICATION")
    print("="*60 + "\n")
    
    try:
        import mediapipe as mp
        print(f"MediaPipe version: {mp.__version__}")
    except ImportError as e:
        print(f"MediaPipe not installed: {e}")
        return False
    
    try:
        import cv2
        print(f"OpenCV version: {cv2.__version__}")
    except ImportError as e:
        print(f"OpenCV not installed: {e}")
        return False
    
    try:
        import numpy as np
        print(f"NumPy version: {np.__version__}")
    except ImportError as e:
        print(f"NumPy not installed: {e}")
        return False
    
    print("\n" + "="*60)
    print("AVAILABLE MEDIAPIPE SOLUTIONS")
    print("="*60 + "\n")
    
    solutions = ['pose', 'hands', 'face_mesh', 'holistic', 'drawing_utils']
    for solution in solutions:
        if hasattr(mp.solutions, solution):
            print(f"mp.solutions.{solution}")
        else:
            print(f"mp.solutions.{solution} not available")
    
    return True

def test_pose_detection_models():
    """Test both legacy and new task-based Pose models"""
    
    print("\n" + "="*60)
    print("TESTING POSE MODELS")
    print("="*60 + "\n")
    
    print("Test 1: Legacy MediaPipe Pose Solution")
    try:
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        print("Legacy Pose model initialized successfully")
        pose.close()
    except Exception as e:
        print(f"Legacy Pose model failed: {e}")
    
    print("\nTest 2: Task-based Pose Landmarker")
    model_path = download_pose_model()
    
    if model_path and os.path.exists(model_path):
        try:
            base_options = python.BaseOptions(model_asset_path=model_path)
            options = vision.PoseLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.VIDEO
            )
            landmarker = vision.PoseLandmarker.create_from_options(options)
            print("Task-based Pose Landmarker initialized successfully")
            landmarker.close()
        except Exception as e:
            print(f"Task-based Pose Landmarker failed: {e}")
    else:
        print("Model file not available, skipping task-based test")
    
    print("\n" + "="*60)

def main():
    """Main setup verification"""
    
    print("\nMediaPipe Pose Tracking - Setup Verification\n")
    
    if not verify_mediapipe_installation():
        print("\nInstallation verification failed")
        print("Please run: pip install -r requirements.txt")
        return
    
    test_pose_detection_models()
    
    print("\n" + "="*60)
    print("SETUP SUMMARY")
    print("="*60 + "\n")
    
    print("All checks completed")
    print("\nRecommendations:")
    print("   For this project, we'll use the LEGACY Pose solution (simpler)")
    print("   It works directly with video streams without extra model files")
    print("   The task-based API is newer but requires model file management")
    print("\nNext: Run 'python test_mediapipe.py' to test with your camera\n")

if __name__ == "__main__":
    main()
