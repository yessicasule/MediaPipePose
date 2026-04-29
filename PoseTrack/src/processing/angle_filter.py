import numpy as np

try:
    from scipy.signal import savgol_filter as _savgol
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False


class MovingAverageFilter:
    def __init__(self, window_size: int = 5):
        self.window_size = window_size
        self._buf = []

    def update(self, value: float) -> float:
        self._buf.append(value)
        if len(self._buf) > self.window_size:
            self._buf.pop(0)
        return float(np.mean(self._buf))

    def reset(self):
        self._buf = []


class ExponentialMovingAverageFilter:
    def __init__(self, alpha: float = 0.2):
        self.alpha = alpha
        self._value = None

    def update(self, value: float) -> float:
        if self._value is None:
            self._value = value
        else:
            self._value = self.alpha * value + (1 - self.alpha) * self._value
        return float(self._value)

    def reset(self):
        self._value = None


class KalmanFilter1D:
    """Scalar Kalman filter — models a slowly-drifting angle measurement."""

    def __init__(self, process_noise: float = 0.01, measurement_noise: float = 1.0):
        self._Q    = process_noise
        self._R    = measurement_noise
        self._x    = 0.0
        self._P    = 1.0
        self._init = False

    def update(self, value: float) -> float:
        if not self._init:
            self._x    = value
            self._init = True
            return value
        self._P += self._Q
        K       = self._P / (self._P + self._R)
        self._x = self._x + K * (value - self._x)
        self._P = (1 - K) * self._P
        return float(self._x)

    def reset(self):
        self._x    = 0.0
        self._P    = 1.0
        self._init = False


class SavitzkyGolayFilter:
    """
    Real-time (causal) Savitzky-Golay filter.

    Buffers the last `window_length` samples and fits a polynomial of degree
    `polyorder` to them each frame, returning the filtered value at the most
    recent point.  Until the buffer fills, raw values are returned so the
    pipeline starts immediately without a forced delay.

    Requires scipy.  Falls back silently to raw values if scipy is absent.
    """

    def __init__(self, window_length: int = 11, polyorder: int = 3):
        if window_length % 2 == 0:
            window_length += 1                   # SG requires odd window
        if polyorder >= window_length:
            polyorder = window_length - 1
        self.window_length = window_length
        self.polyorder     = polyorder
        self._buf: list[float] = []

    def update(self, value: float) -> float:
        self._buf.append(value)
        if len(self._buf) > self.window_length:
            self._buf.pop(0)
        if not _SCIPY_AVAILABLE or len(self._buf) < self.window_length:
            return value
        filtered = _savgol(self._buf, self.window_length, self.polyorder)
        return float(filtered[-1])

    def reset(self):
        self._buf = []


class AngleFilterSystem:
    """
    Per-joint filter bank.  Supported filter_type values:
        "kalman"  — 1-D Kalman filter (default, best for real-time)
        "ema"     — Exponential moving average
        "ma"      — Simple moving average
        "sg"      — Savitzky-Golay (requires scipy)
    """

    def __init__(self, filter_type: str = "kalman"):
        self.filter_type = filter_type.lower()
        self.filters = {
            "elbow":          self._create_filter(),
            "shoulder_pitch": self._create_filter(),
            "shoulder_yaw":   self._create_filter(),
            "shoulder_roll":  self._create_filter(),
        }

    def _create_filter(self):
        if self.filter_type == "ema":
            return ExponentialMovingAverageFilter(alpha=0.2)
        if self.filter_type == "ma":
            return MovingAverageFilter(window_size=7)
        if self.filter_type == "sg":
            return SavitzkyGolayFilter(window_length=11, polyorder=3)
        return KalmanFilter1D(process_noise=0.01, measurement_noise=1.5)

    def update(self, joint_angles: dict) -> dict:
        return {
            "elbow_flexion":      self.filters["elbow"].update(
                                      joint_angles.get("elbow_flexion", 0.0)),
            "shoulder_elevation": self.filters["shoulder_pitch"].update(
                                      joint_angles.get("shoulder_elevation", 0.0)),
            "shoulder_yaw":       self.filters["shoulder_yaw"].update(
                                      joint_angles.get("shoulder_yaw", 0.0)),
            "shoulder_roll":      self.filters["shoulder_roll"].update(
                                      joint_angles.get("shoulder_roll", 0.0)),
        }

    def reset(self):
        for f in self.filters.values():
            f.reset()
