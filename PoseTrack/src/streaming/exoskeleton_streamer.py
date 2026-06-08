"""
exoskeleton_streamer.py — Phase 9: Exoskeleton Calibration Layer.

Outputs structured calibration packets per frame over UDP port 9001.
Also writes a rolling CSV log.

Packet format (JSON per line):
{
  "frame": 142,
  "timestamp": "2026-06-08T12:00:00Z",
  "joint": "left_elbow_flexion",
  "angle": 42.1,
  "confidence": 0.95,
  "uncertainty": 1.3,
  "source": "DeepFusionPose-GAN",
  "action": "APPLY"   // APPLY if confidence > threshold, else HOLD
}
"""

import sys, json, socket, threading, time, csv
from datetime import datetime, timezone
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))


JOINT_NAMES = ["shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow_flexion"]


class ExoskeletonStreamer:
    """
    Thread-safe calibration signal broadcaster.

    Usage:
        streamer = ExoskeletonStreamer()
        streamer.start()
        ...
        streamer.update(angles_mean, angles_std, frame_idx)
        ...
        streamer.stop()
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 9001,
        hz: float = 30.0,
        confidence_threshold: float = 0.85,
        log_csv: str = "outputs/exoskeleton_log.csv",
        source: str = "DeepFusionPose-GAN",
    ):
        self._host      = host
        self._port      = port
        self._interval  = 1.0 / hz
        self._threshold = confidence_threshold
        self._source    = source
        self._sock      = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._lock      = threading.Lock()
        self._running   = False
        self._thread    = None

        # angles_mean and angles_std are lists of 4 values
        self._angles_mean = [0.0] * 4
        self._angles_std  = [1.0] * 4
        self._frame       = 0
        self._prev_safe   = [0.0] * 4   # last APPLY value per joint

        # CSV log
        self._log_path = Path(log_csv)
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        self._csv_file = open(self._log_path, "w", newline="", encoding="utf-8")
        fieldnames = ["timestamp", "frame", "joint", "angle", "confidence",
                      "uncertainty", "action", "source"]
        self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=fieldnames)
        self._csv_writer.writeheader()

    def update(self, angles_mean: list, angles_std: list, frame_idx: int):
        """Call this every inference frame with predicted mean and std."""
        with self._lock:
            self._angles_mean = list(angles_mean)
            self._angles_std  = list(angles_std)
            self._frame       = frame_idx

    def start(self):
        self._running = True
        self._thread  = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        print(f"[ExoskeletonStreamer] Calibration signal → {self._host}:{self._port} "
              f"| log → {self._log_path}")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        self._sock.close()
        self._csv_file.close()
        print(f"[ExoskeletonStreamer] Stopped. Log saved → {self._log_path}")

    def __enter__(self): self.start(); return self
    def __exit__(self, *_): self.stop()

    # ── Private ───────────────────────────────────────────────────────────

    def _loop(self):
        target = time.perf_counter()
        while self._running:
            with self._lock:
                means = self._angles_mean[:]
                stds  = self._angles_std[:]
                frame = self._frame

            ts = datetime.now(timezone.utc).isoformat(timespec="seconds")

            for i, joint in enumerate(JOINT_NAMES):
                angle       = means[i]
                uncertainty = stds[i]
                confidence  = round(1.0 / (1.0 + uncertainty), 4)

                if confidence >= self._threshold:
                    action = "APPLY"
                    self._prev_safe[i] = angle
                else:
                    action = "HOLD"
                    angle  = self._prev_safe[i]  # use last safe value

                packet = {
                    "frame":       frame,
                    "timestamp":   ts,
                    "joint":       joint,
                    "angle":       round(angle, 3),
                    "confidence":  confidence,
                    "uncertainty": round(uncertainty, 4),
                    "source":      self._source,
                    "action":      action,
                }

                # UDP send
                try:
                    data = (json.dumps(packet) + "\n").encode()
                    self._sock.sendto(data, (self._host, self._port))
                except OSError:
                    self._running = False
                    return

                # CSV log
                self._csv_writer.writerow({
                    "timestamp":   ts,
                    "frame":       frame,
                    "joint":       joint,
                    "angle":       packet["angle"],
                    "confidence":  packet["confidence"],
                    "uncertainty": packet["uncertainty"],
                    "action":      action,
                    "source":      self._source,
                })

            self._csv_file.flush()

            target += self._interval
            sleep = target - time.perf_counter()
            if sleep > 0:
                time.sleep(sleep)
            else:
                target = time.perf_counter()
