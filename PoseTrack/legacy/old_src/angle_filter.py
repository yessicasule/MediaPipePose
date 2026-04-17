from collections import deque
import numpy as np


class MovingAverageFilter:
    def __init__(self, window: int = 7):
        self._buf = deque(maxlen=window)

    def update(self, value: float) -> float:
        self._buf.append(value)
        return float(np.mean(self._buf))

    def reset(self):
        self._buf.clear()


class SavitzkyGolayFilter:
    def __init__(self, window: int = 11, poly: int = 3):
        if window % 2 == 0:
            window += 1
        self._window = window
        self._poly   = poly
        self._buf    = deque(maxlen=window)
        self._coeffs = self._compute_coeffs(window, poly)

    @staticmethod
    def _compute_coeffs(window, poly):
        half   = window // 2
        x      = np.arange(-half, half + 1)
        A      = np.column_stack([x**i for i in range(poly + 1)])
        coeffs = np.linalg.pinv(A)[0]
        return coeffs

    def update(self, value: float) -> float:
        self._buf.append(value)
        if len(self._buf) < self._window:
            return float(np.mean(self._buf))
        arr = np.array(self._buf, dtype=float)
        return float(np.dot(self._coeffs, arr))

    def reset(self):
        self._buf.clear()


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
