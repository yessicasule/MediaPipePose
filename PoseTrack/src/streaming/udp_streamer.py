import socket
import threading
import time

class UdpAngleStreamer:
    def __init__(self, host: str = "127.0.0.1", port: int = 9000, hz: float = 30.0):
        self._host     = host
        self._port     = port
        self._interval = 1.0 / hz
        self._sock     = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # S, shoulder_pitch, shoulder_yaw, shoulder_roll, elbow_flex
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
        if self._thread:
            self._thread.join(timeout=2.0)
        self._sock.close()

    def update_angles(self, shoulder_pitch: float, shoulder_yaw: float,
                      shoulder_roll: float, elbow_flex: float):
        """
        Mainly called by Real-time pose estimation thread to feed new data.
        """
        with self._lock:
            self._angles = (shoulder_pitch, shoulder_yaw, shoulder_roll, elbow_flex)

    def _loop(self):
        next_send = time.perf_counter()
        while self._running:
            with self._lock:
                sp, sy, sr, ef = self._angles
            # S = Start marker, Unity parses CSV indices
            packet = f"S,{sp:.3f},{sy:.3f},{sr:.3f},{ef:.3f}\n"
            try:
                self._sock.sendto(packet.encode(), (self._host, self._port))
            except OSError:
                break
                
            next_send += self._interval
            sleep_for  = next_send - time.perf_counter()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                 # Catch up logic if lagging
                 next_send = time.perf_counter()
