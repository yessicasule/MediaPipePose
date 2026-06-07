import cv2
import mediapipe as mp
import urllib.request
from pathlib import Path

_MODEL_DIR  = Path(__file__).resolve().parent / "models"
_MODEL_PATH = _MODEL_DIR / "pose_landmarker_lite.task"
_MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)

def _ensure_model():
    if not _MODEL_PATH.exists():
        _MODEL_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Downloading MediaPipe pose model to {_MODEL_PATH} ...")
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
        print("Download complete.")


class MediaPipeRunner:
    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=1):
        self._use_tasks = False
        try:
            self._init_solutions(min_detection_confidence, min_tracking_confidence, model_complexity)
        except Exception:
            self._use_tasks = True
            self._init_tasks(min_detection_confidence, min_tracking_confidence)

    def _init_solutions(self, det_conf, track_conf, complexity):
        try:
            import mediapipe.python.solutions.pose as _pose_mod
        except ModuleNotFoundError:
            import importlib
            _pose_mod = importlib.import_module("mediapipe.solutions.pose")
        self._pose_mod = _pose_mod
        self.pose = _pose_mod.Pose(
            min_detection_confidence=det_conf,
            min_tracking_confidence=track_conf,
            model_complexity=complexity,
        )

    def _init_tasks(self, det_conf, track_conf):
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python.core.base_options import BaseOptions
        _ensure_model()
        options = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(_MODEL_PATH)),
            running_mode=vision.RunningMode.VIDEO,
            min_pose_detection_confidence=det_conf,
            min_pose_presence_confidence=det_conf,
            min_tracking_confidence=track_conf,
        )
        self.pose = vision.PoseLandmarker.create_from_options(options)
        self._ts_ms = 0

    def process(self, image_rgb):
        if self._use_tasks:
            return self._process_tasks(image_rgb)
        results = self.pose.process(image_rgb)
        if results.pose_landmarks:
            return results.pose_landmarks.landmark
        return None

    def _process_tasks(self, image_rgb):
        self._ts_ms += 33
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        result = self.pose.detect_for_video(mp_img, self._ts_ms)
        if result.pose_landmarks:
            return result.pose_landmarks[0]
        return None

    def close(self):
        self.pose.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
