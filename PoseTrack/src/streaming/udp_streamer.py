"""
udp_streamer.py — Thread-safe UDP Angle Transmission
=====================================================

Transmits arm joint angles over UDP at a fixed rate to a Unity receiver.

Packet Format
-------------
    S,<shoulder_flex>,<shoulder_abd>,<shoulder_rot>,<elbow_flex>\\n

All values are in degrees, formatted to two decimal places.
Each packet is terminated with a newline character as a frame delimiter.

Example packet:
    S,45.23,-12.10,8.75,90.00\\n

The 'S' prefix identifies this as a single-arm pose packet.
The fixed-rate loop uses perf_counter timing to avoid clock drift,
sending at exactly the configured Hz without accumulating latency.

Threading Model
---------------
- A background daemon thread runs the fixed-rate send loop.
- The main thread updates angles via update_angles(), protected by a lock.
- On each send iteration, angles are snapshot-copied under the lock
  so the send call itself (which may block briefly on the OS) is lock-free.
"""

from __future__ import annotations

import socket
import threading
import time
from dataclasses import dataclass

from ..processing.angle_solver import ArmAngles


class UdpAngleSender:
    """
    Thread-safe, fixed-rate UDP sender for arm joint angles.

    Parameters
    ----------
    host : str
        Destination IP address (default: localhost).
    port : int
        Destination UDP port (default: 9000, must match Unity receiver).
    hz : float
        Target transmission rate in Hz (default: 30).
    """

    def __init__(
        self,
        host: str  = "127.0.0.1",
        port: int  = 9000,
        hz:   float = 30.0,
    ) -> None:
        self._addr     = (host, port)
        self._interval = 1.0 / max(hz, 1.0)
        self._sock     = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._lock     = threading.Lock()
        self._running  = False
        self._thread: threading.Thread | None = None

        # Current angle values (degrees)
        self._flex: float = 0.0
        self._abd:  float = 0.0
        self._rot:  float = 0.0
        self._elb:  float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the background sending thread."""
        if self._running:
            return
        self._running = True
        self._thread  = threading.Thread(target=self._loop, daemon=True, name="UdpSender")
        self._thread.start()

    def stop(self) -> None:
        """Stop the background thread and close the socket."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        try:
            self._sock.close()
        except OSError:
            pass

    def update(self, angles: ArmAngles) -> None:
        """
        Update the angles to be sent on the next packet.

        This is the primary way to push new angle data into the sender.
        Thread-safe.

        Parameters
        ----------
        angles : ArmAngles
            Latest computed (and filtered) arm joint angles.
        """
        with self._lock:
            self._flex = angles.shoulder_flexion
            self._abd  = angles.shoulder_abduction
            self._rot  = angles.shoulder_rotation
            self._elb  = angles.elbow_flexion

    def send_now(self, angles: ArmAngles) -> None:
        """
        Send a single packet immediately (bypasses the background thread).
        Useful for one-off sends in the data generator.
        """
        msg = self._format(
            angles.shoulder_flexion,
            angles.shoulder_abduction,
            angles.shoulder_rotation,
            angles.elbow_flexion,
        )
        try:
            self._sock.sendto(msg, self._addr)
        except OSError:
            pass

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _loop(self) -> None:
        """Fixed-rate send loop — runs on the background thread."""
        target = time.perf_counter()
        while self._running:
            with self._lock:
                msg = self._format(self._flex, self._abd, self._rot, self._elb)
            try:
                self._sock.sendto(msg, self._addr)
            except OSError:
                self._running = False
                return

            target += self._interval
            sleep_s = target - time.perf_counter()
            if sleep_s > 0:
                time.sleep(sleep_s)
            else:
                # Running behind schedule — reset target to avoid spiral
                target = time.perf_counter()

    @staticmethod
    def _format(flex: float, abd: float, rot: float, elb: float) -> bytes:
        """
        Format angles into the packet string and encode as UTF-8 bytes.

        Packet format: S,<flex>,<abd>,<rot>,<elb>\\n
        """
        return f"S,{flex:.2f},{abd:.2f},{rot:.2f},{elb:.2f}\n".encode("utf-8")

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
