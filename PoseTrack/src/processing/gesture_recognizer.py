from collections import deque
import time
import numpy as np


GESTURES = ["REST", "RAISE_ARM", "REACH_FORWARD", "ELBOW_FLEX", "WAVE"]


class GestureRecognizer:
    """Rule-based gesture recognizer operating on filtered joint angles — no ML model required."""

    WINDOW_SEC   = 1.5    # history window in seconds
    MIN_FPS      = 20     # assumed minimum update rate
    WINDOW_MAXN  = int(WINDOW_SEC * MIN_FPS * 2)  # sample cap

    def __init__(self):
        self._history: deque = deque(maxlen=self.WINDOW_MAXN)
        self._last_gesture   = "REST"
        self._gesture_start  = time.perf_counter()
        self._last_time      = time.perf_counter()

    def update(self, angles: dict) -> tuple:
        """Push one frame of angles and return (gesture_name, confidence 0-1)."""
        now = time.perf_counter()
        dt  = now - self._last_time
        self._last_time = now
        self._history.append((now, angles.copy()))

        gesture, confidence = self._classify()
        if gesture != self._last_gesture:
            self._last_gesture  = gesture
            self._gesture_start = now

        return gesture, round(confidence, 3)

    def reset(self):
        self._history.clear()
        self._last_gesture = "REST"

    def _classify(self) -> tuple:
        if len(self._history) < 5:
            return "REST", 0.5

        now    = time.perf_counter()
        window = [(t, a) for t, a in self._history if now - t <= self.WINDOW_SEC]
        if not window:
            return "REST", 1.0

        pitches = np.array([a["shoulder_pitch"]  for _, a in window])
        yaws    = np.array([a["shoulder_yaw"]    for _, a in window])
        elbows  = np.array([a["elbow_flexion"]   for _, a in window])
        times   = np.array([t                    for t, _ in window])
        duration = times[-1] - times[0] if len(times) > 1 else 0.0

        # WAVE: detect oscillation via zero-crossings of mean-centered elbow signal
        if duration >= 1.0 and len(elbows) >= 10:
            elbow_mean  = elbows.mean()
            centered    = elbows - elbow_mean
            sign_changes = np.diff(np.sign(centered))
            crossings   = int(np.abs(sign_changes[sign_changes != 0]).sum() / 2)
            if crossings >= 3 and elbows.std() > 8:
                conf = min(1.0, crossings / 6)
                return "WAVE", conf

        # RAISE_ARM: shoulder_pitch > 70° sustained over 0.8 s
        recent_05 = [(t, a) for t, a in window if now - t <= 0.8]
        if recent_05:
            recent_pitches = np.array([a["shoulder_pitch"] for _, a in recent_05])
            frac_raised = (recent_pitches > 70).mean()
            if frac_raised >= 0.8:
                return "RAISE_ARM", float(frac_raised)

        # REACH_FORWARD: shoulder_yaw > 45° sustained over 0.6 s
        recent_06 = [(t, a) for t, a in window if now - t <= 0.6]
        if recent_06:
            recent_yaws = np.array([a["shoulder_yaw"] for _, a in recent_06])
            frac_reach  = (recent_yaws > 45).mean()
            if frac_reach >= 0.75:
                return "REACH_FORWARD", float(frac_reach)

        # ELBOW_FLEX: elbow_flexion < 50° (bent) sustained over 0.5 s
        recent_05b = [(t, a) for t, a in window if now - t <= 0.5]
        if recent_05b:
            recent_el = np.array([a["elbow_flexion"] for _, a in recent_05b])
            frac_bent = (recent_el < 50).mean()
            if frac_bent >= 0.8:
                return "ELBOW_FLEX", float(frac_bent)

        is_neutral = (
            pitches.mean() < 30 and
            np.abs(yaws).mean() < 20 and
            elbows.mean() > 140
        )
        if is_neutral:
            return "REST", 0.95

        return "REST", 0.5


def draw_gesture_overlay(frame, gesture: str, confidence: float):
    """Draw a gesture badge on a cv2 frame."""
    import cv2

    COLORS = {
        "REST":          (120, 120, 120),
        "RAISE_ARM":     (0,   220, 80),
        "REACH_FORWARD": (0,   180, 255),
        "ELBOW_FLEX":    (255, 160, 0),
        "WAVE":          (200, 0,   255),
    }
    color = COLORS.get(gesture, (255, 255, 255))
    h, w  = frame.shape[:2]

    cv2.rectangle(frame, (w - 280, 10), (w - 10, 65), (20, 20, 20), -1)
    cv2.rectangle(frame, (w - 280, 10), (w - 10, 65), color, 2)

    cv2.putText(frame, gesture.replace("_", " "),
                (w - 270, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    bar_w = int(confidence * 240)
    cv2.rectangle(frame, (w - 270, 48), (w - 270 + bar_w, 58), color, -1)
    cv2.rectangle(frame, (w - 270, 48), (w - 30, 58), (80, 80, 80), 1)

    return frame
