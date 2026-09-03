"""
mediapipe_runner.py — MediaPipe Pose Estimator
===============================================

Implements the PoseEstimator interface using Google MediaPipe Pose.

API Compatibility
-----------------
MediaPipe ships two pose inference APIs:

1. Legacy Solutions API (mediapipe < 0.10.x)
   mp.solutions.pose.Pose()
   Returns: results.pose_landmarks.landmark (direct list access)

2. Tasks API (mediapipe >= 0.10.x, recommended)
   vision.PoseLandmarker
   Returns: result.pose_landmarks[0] (NormalizedLandmark list)

This runner tries the Solutions API first and falls back to the Tasks API
if the solutions module is unavailable. Both paths return the same
Landmark interface to the rest of the pipeline.

Timestamp Handling (Tasks API)
-------------------------------
The Tasks API requires a monotonically increasing timestamp in milliseconds.
We track elapsed wall-clock time from process() calls rather than adding a
fixed 33 ms per frame, which was incorrect in the original code and caused
drift at non-30 fps framerates.

Coordinate System
-----------------
MediaPipe returns landmark coordinates normalised to image dimensions:
    x ∈ [0, 1]  — horizontal, left-to-right
    y ∈ [0, 1]  — vertical, top-to-bottom
    z           — pseudo-depth relative to hip (same scale as x)

References:
    https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker
    Bazarevsky et al. (2020). BlazePose: On-device Real-time Body Pose
    Tracking. arXiv:2006.10204.
"""

from __future__ import annotations

import time
import urllib.request
from pathlib import Path

import numpy as np

from .base import PoseEstimator, Landmark, N_LANDMARKS

# --------------------------------------------------------------------------
# Model download (Tasks API only)
# --------------------------------------------------------------------------
_MODEL_DIR   = Path(__file__).resolve().parent / "models"
_MODEL_PATH  = _MODEL_DIR / "pose_landmarker_lite.task"
_MODEL_URL   = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_lite/float16/latest/"
    "pose_landmarker_lite.task"
)


def _ensure_model() -> None:
    if not _MODEL_PATH.exists():
        _MODEL_DIR.mkdir(parents=True, exist_ok=True)
        print(f"[MediaPipe] Downloading model → {_MODEL_PATH}")
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
        print("[MediaPipe] Download complete.")


class MediaPipeRunner(PoseEstimator):
    """
    MediaPipe Pose runner implementing the PoseEstimator interface.

    Returns 33 Landmark objects in MediaPipe's standard indexing.
    z coordinates are MediaPipe's pseudo-depth estimates (relative to hip).

    Parameters
    ----------
    detection_confidence : float
        Minimum confidence for person detection [0, 1].
    tracking_confidence : float
        Minimum confidence for landmark tracking [0, 1].
        When tracking confidence drops below this, detection re-runs.
    model_complexity : int
        0 = Lite (fastest, least accurate)
        1 = Full (balanced)  ← default
        2 = Heavy (most accurate, slowest)
    """

    @property
    def name(self) -> str:
        return "MediaPipe"

    def __init__(
        self,
        detection_confidence: float = 0.5,
        tracking_confidence:  float = 0.5,
        model_complexity:     int   = 1,
    ) -> None:
        self._use_tasks   = False
        self._start_time  = time.perf_counter()
        self._last_ts_ms  = -1   # last timestamp handed to detect_for_video()

        try:
            self._init_solutions(detection_confidence, tracking_confidence, model_complexity)
        except Exception as e:
            print(f"[MediaPipe] Solutions API unavailable ({e}), trying Tasks API...")
            self._use_tasks = True
            self._init_tasks(detection_confidence, tracking_confidence)

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_solutions(
        self,
        det_conf:   float,
        track_conf: float,
        complexity: int,
    ) -> None:
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

    def _init_tasks(self, det_conf: float, track_conf: float) -> None:
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

    # ------------------------------------------------------------------
    # PoseEstimator interface
    # ------------------------------------------------------------------

    def process(self, image_rgb: np.ndarray) -> list[Landmark] | None:
        if self._use_tasks:
            return self._process_tasks(image_rgb)
        return self._process_solutions(image_rgb)

    def _process_solutions(self, image_rgb: np.ndarray) -> list[Landmark] | None:
        results = self.pose.process(image_rgb)
        if not results.pose_landmarks:
            return None
        return [
            Landmark(
                x=float(lm.x),
                y=float(lm.y),
                z=float(lm.z),
                visibility=float(lm.visibility),
            )
            for lm in results.pose_landmarks.landmark
        ]

    def _process_tasks(self, image_rgb: np.ndarray) -> list[Landmark] | None:
        import mediapipe as mp

        # Use real elapsed time for the timestamp (rather than a hardcoded
        # per-frame increment, which drifts at non-30fps rates). detect_for_video()
        # requires STRICTLY increasing timestamps, but an int millisecond count
        # can repeat when two frames are processed inside the same millisecond
        # (fast offline loops over pre-extracted frames, or coarse timer
        # resolution), so step forward by 1ms whenever that happens.
        ts_ms = int((time.perf_counter() - self._start_time) * 1000)
        if ts_ms <= self._last_ts_ms:
            ts_ms = self._last_ts_ms + 1
        self._last_ts_ms = ts_ms

        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        result = self.pose.detect_for_video(mp_img, ts_ms)

        if not result.pose_landmarks:
            return None
        return [
            Landmark(
                x=float(lm.x),
                y=float(lm.y),
                z=float(lm.z),
                visibility=float(lm.visibility),
            )
            for lm in result.pose_landmarks[0]
        ]

    def close(self) -> None:
        try:
            self.pose.close()
        except Exception:
            pass
