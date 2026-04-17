import numpy as np

class MovingAverageFilter:
    def __init__(self, window_size: int = 5):
        self.window_size = window_size
        self.values = []
        
    def update(self, value: float) -> float:
        self.values.append(value)
        if len(self.values) > self.window_size:
            self.values.pop(0)
        return float(np.mean(self.values))

    def reset(self):
        self.values = []

class ExponentialMovingAverageFilter:
    def __init__(self, alpha: float = 0.2):
        self.alpha = alpha
        self.value = None
        
    def update(self, value: float) -> float:
        if self.value is None:
            self.value = value
        else:
            self.value = self.alpha * value + (1 - self.alpha) * self.value
        return float(self.value)

    def reset(self):
        self.value = None

class KalmanFilter1D:
    def __init__(self, process_noise: float = 0.01, measurement_noise: float = 1.0):
        self._Q  = process_noise
        self._R  = measurement_noise
        self._x  = 0.0
        self._P  = 1.0
        self._init = False

    def update(self, value: float) -> float:
        if not self._init:
            self._x    = value
            self._init = True
            return value

        self._P += self._Q
        K        = self._P / (self._P + self._R)
        self._x  = self._x + K * (value - self._x)
        self._P  = (1 - K) * self._P
        return float(self._x)

    def reset(self):
        self._x    = 0.0
        self._P    = 1.0
        self._init = False

class AngleFilterSystem:
    def __init__(self, filter_type: str = "kalman"):
        """
        Manages per-joint filters. filter_type can be 'ema' or 'kalman'.
        """
        self.filter_type = filter_type.lower()
        self.filters = {
            "elbow": self._create_filter(),
            "shoulder_pitch": self._create_filter(),
            "shoulder_yaw": self._create_filter(),
            "shoulder_roll": self._create_filter()
        }
        
    def _create_filter(self):
        if self.filter_type == "ema":
            return ExponentialMovingAverageFilter(alpha=0.2)
        else:
            # Default to Kalman filter for optimal stability
            return KalmanFilter1D(process_noise=0.01, measurement_noise=1.5)
            
    def update(self, joint_angles: dict) -> dict:
        """
        Applies filter to raw angles. joint_angles must contain keys:
        'elbow_flexion', 'shoulder_elevation', 'shoulder_yaw', 'shoulder_roll'
        """
        filtered = {
            "elbow_flexion": self.filters["elbow"].update(joint_angles.get("elbow_flexion", 0.0)),
            "shoulder_elevation": self.filters["shoulder_pitch"].update(joint_angles.get("shoulder_elevation", 0.0)),
            "shoulder_yaw": self.filters["shoulder_yaw"].update(joint_angles.get("shoulder_yaw", 0.0)),
            "shoulder_roll": self.filters["shoulder_roll"].update(joint_angles.get("shoulder_roll", 0.0))
        }
        return filtered

    def reset(self):
        for f in self.filters.values():
            f.reset()
