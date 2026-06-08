"""
udp_streamer.py — Phase 8: Multi-avatar 4-stream UDP broadcaster.

Sends 4 angle packets per tick on port 9000 (Unity) using distinct prefixes:
    MP,<pitch>,<roll>,<yaw>,<elbow>              ← Raw MediaPipe
    MV,<pitch>,<roll>,<yaw>,<elbow>              ← Raw MoveNet
    FU,<pitch>,<roll>,<yaw>,<elbow>,<unc>        ← Fusion + MC uncertainty
    GR,<pitch>,<roll>,<yaw>,<elbow>              ← GAN Refined

Runs in a background thread; call update_*() from the main inference thread.
"""

import socket
import threading
import time


class MultiAvatarStreamer:
    """
    Thread-safe UDP streamer for 4 simultaneous pose sources.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 9000, hz: float = 30.0):
        self._host     = host
        self._port     = port
        self._interval = 1.0 / hz
        self._sock     = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._lock     = threading.Lock()
        self._running  = False
        self._thread   = None

        # (pitch, roll, yaw, elbow)
        self._mp = (0.0, 0.0, 0.0, 0.0)
        self._mv = (0.0, 0.0, 0.0, 0.0)
        # (pitch, roll, yaw, elbow, uncertainty)
        self._fu = (0.0, 0.0, 0.0, 0.0, 0.0)
        self._gr = (0.0, 0.0, 0.0, 0.0)

    # ── Angle setters (called from inference thread) ───────────────────────

    def update_mediapipe(self, pitch: float, roll: float, yaw: float, elbow: float):
        with self._lock:
            self._mp = (pitch, roll, yaw, elbow)

    def update_movenet(self, pitch: float, roll: float, yaw: float, elbow: float):
        with self._lock:
            self._mv = (pitch, roll, yaw, elbow)

    def update_fusion(self, pitch: float, roll: float, yaw: float,
                      elbow: float, uncertainty: float = 0.0):
        with self._lock:
            self._fu = (pitch, roll, yaw, elbow, uncertainty)

    def update_gan(self, pitch: float, roll: float, yaw: float, elbow: float):
        with self._lock:
            self._gr = (pitch, roll, yaw, elbow)

    # ── Lifecycle ───────────────────────────────────────────────────────────

    def start(self):
        self._running = True
        self._thread  = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        print(f"[MultiAvatarStreamer] Sending to {self._host}:{self._port} @ {1/self._interval:.0f} Hz")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        self._sock.close()

    def __enter__(self): self.start(); return self
    def __exit__(self, *_): self.stop()

    # ── Broadcast loop ──────────────────────────────────────────────────────

    def _loop(self):
        target = time.perf_counter()
        while self._running:
            with self._lock:
                mp = self._mp
                mv = self._mv
                fu = self._fu
                gr = self._gr

            packets = [
                f"MP,{mp[0]:.3f},{mp[1]:.3f},{mp[2]:.3f},{mp[3]:.3f}\n",
                f"MV,{mv[0]:.3f},{mv[1]:.3f},{mv[2]:.3f},{mv[3]:.3f}\n",
                f"FU,{fu[0]:.3f},{fu[1]:.3f},{fu[2]:.3f},{fu[3]:.3f},{fu[4]:.4f}\n",
                f"GR,{gr[0]:.3f},{gr[1]:.3f},{gr[2]:.3f},{gr[3]:.3f}\n",
            ]
            for pkt in packets:
                try:
                    self._sock.sendto(pkt.encode(), (self._host, self._port))
                except OSError:
                    self._running = False
                    return

            target += self._interval
            sleep = target - time.perf_counter()
            if sleep > 0:
                time.sleep(sleep)
            else:
                target = time.perf_counter()


# ── Legacy single-stream class (backward compatible) ───────────────────────

class UdpAngleStreamer:
    """
    Original single-stream streamer kept for backward compatibility.
    Use MultiAvatarStreamer for Phase 8 onwards.
    """
    def __init__(self, host="127.0.0.1", port=9000, hz=30.0):
        self._host     = host
        self._port     = port
        self._interval = 1.0 / hz
        self._sock     = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._angles   = (0.0, 0.0, 0.0, 0.0)
        self._lock     = threading.Lock()
        self._running  = False
        self._thread   = None

    def start(self):
        self._running = True
        self._thread  = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread: self._thread.join(timeout=2.0)
        self._sock.close()

    def update_angles(self, shoulder_pitch, shoulder_yaw, shoulder_roll, elbow_flex):
        with self._lock:
            self._angles = (shoulder_pitch, shoulder_yaw, shoulder_roll, elbow_flex)

    def _loop(self):
        next_send = time.perf_counter()
        while self._running:
            with self._lock:
                sp, sy, sr, ef = self._angles
            packet = f"S,{sp:.3f},{sy:.3f},{sr:.3f},{ef:.3f}\n"
            try:
                self._sock.sendto(packet.encode(), (self._host, self._port))
            except OSError:
                break
            next_send += self._interval
            sleep_for = next_send - time.perf_counter()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                next_send = time.perf_counter()
