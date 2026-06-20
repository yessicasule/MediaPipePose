import socket
import threading
import time


class UdpAngleStreamer:
    """Sends 'S,pitch,yaw,roll,elbow' packets over UDP at a fixed rate."""

    def __init__(self, host="127.0.0.1", port=9000, hz=30.0):
        self._addr = (host, port)
        self._interval = 1.0 / hz
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._lock = threading.Lock()
        self._running = False
        self._thread = None
        self._pitch = self._yaw = self._roll = self._elbow = 0.0

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        self._sock.close()

    def update_angles(self, shoulder_pitch, shoulder_yaw, shoulder_roll, elbow_flex):
        with self._lock:
            self._pitch = shoulder_pitch
            self._yaw   = shoulder_yaw
            self._roll  = shoulder_roll
            self._elbow = elbow_flex

    def _loop(self):
        target = time.perf_counter()
        while self._running:
            with self._lock:
                msg = f"S,{self._pitch:.2f},{self._yaw:.2f},{self._roll:.2f},{self._elbow:.2f}\n"
            try:
                self._sock.sendto(msg.encode(), self._addr)
            except OSError:
                self._running = False
                return
            target += self._interval
            sleep = target - time.perf_counter()
            if sleep > 0:
                time.sleep(sleep)
            else:
                target = time.perf_counter()
