#!/usr/bin/env python3
"""
Data Generator Application
Transmits simulated arm joint angles over UDP for Unity testing.
"""

import socket
import time
import math
import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config.config import Config, UDP_IP, UDP_PORT


class AngleSimulator:
    def __init__(self, mode: str = "wave"):
        self.mode = mode
        self.t = 0.0
    
    def get_angles(self, dt: float) -> tuple:
        self.t += dt
        if self.mode == "wave":
            pitch = 30 + 40 * math.sin(self.t * 1.5)
            yaw = 20 * math.sin(self.t * 0.8)
            roll = 10 * math.cos(self.t * 1.2)
            elbow = 10 + 60 * (0.5 + 0.5 * math.sin(self.t * 2.0))
        elif self.mode == "circle":
            pitch = 30 + 30 * math.cos(self.t * 0.7)
            yaw = 45 * math.sin(self.t * 0.5)
            roll = 15 * math.sin(self.t * 0.9)
            elbow = 20 + 40 * (0.5 + 0.5 * math.cos(self.t * 1.2))
        elif self.mode == "static":
            pitch, yaw, roll, elbow = 45, 15, 5, 90
        elif self.mode == "random":
            import random
            pitch = random.uniform(10, 90)
            yaw = random.uniform(-30, 30)
            roll = random.uniform(-15, 15)
            elbow = random.uniform(20, 120)
        else:
            pitch, yaw, roll, elbow = 0, 0, 0, 0
        return pitch, yaw, roll, elbow


def main():
    parser = argparse.ArgumentParser(description="Simulated arm angle UDP sender")
    parser.add_argument("--host", default=UDP_IP, help="Target host")
    parser.add_argument("--port", type=int, default=UDP_PORT, help="Target port")
    parser.add_argument("--hz", type=float, default=30.0, help="Update rate")
    parser.add_argument("--mode", default="wave", choices=["wave", "circle", "static", "random"],
                      help="Animation mode")
    parser.add_argument("--duration", type=int, default=0, help="Duration in seconds (0=infinite)")
    args = parser.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sim = AngleSimulator(args.mode)
    
    interval = 1.0 / args.hz
    print(f"Sending angles to {args.host}:{args.port} at {args.hz} Hz (mode: {args.mode})")
    print("Press Ctrl+C to stop" if args.duration == 0 else f"Running for {args.duration}s")

    start_time = time.perf_counter()
    try:
        while True:
            loop_start = time.perf_counter()
            pitch, yaw, roll, elbow = sim.get_angles(interval)
            packet = f"S,{pitch:.2f},{yaw:.2f},{roll:.2f},{elbow:.2f}\n"
            sock.sendto(packet.encode(), (args.host, args.port))
            
            elapsed = time.perf_counter() - loop_start
            sleep_time = interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
            
            if args.duration > 0 and time.perf_counter() - start_time >= args.duration:
                break
    except KeyboardInterrupt:
        print("\nStopped")
    finally:
        sock.close()


if __name__ == "__main__":
    main()