import cv2
import mediapipe as mp

class MediaPipeRunner:
    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=1):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            model_complexity=model_complexity
        )

    def process(self, image_rgb):
        """Processes RGB image and returns landmarks if found, else None"""
        results = self.pose.process(image_rgb)
        if results.pose_landmarks:
            return results.pose_landmarks.landmark
        return None

    def close(self):
        self.pose.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
