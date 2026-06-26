"""
mock_streamer.py — Simulated Angle Data Generator for Unity Testing
====================================================================

Sends simulated arm joint angles over UDP in the MonoArm packet format:
    S,<shoulder_flex>,<shoulder_abd>,<shoulder_rot>,<elbow_flex>\\n

This allows the entire Unity side to be developed and tested without a
camera or the vision pipeline running.

Modes
-----
    sinusoidal  : All DOFs follow independent sinusoidal sweeps.
    step        : Angles jump between defined positions at a fixed interval.
    static      : All angles held at zero (useful for testing jitter).
    random      : Angles undergo a bounded random walk.

Usage
-----
    python scripts/mock_streamer.py --mode sinusoidal --hz 30
    python scripts/mock_streamer.py --mode step --hz 20
"""

from __future__ import annotations

import argparse
import math
import random
import socket
import sys
import time


def _fmt(flex: float, abd: float, rot: float, elb: float) -> bytes:
    """Format one UDP packet."""
    return f"S,{flex:.2f},{abd:.2f},{rot:.2f},{elb:.2f}\n".encode("utf-8")


def run_sinusoidal(sock: socket.socket, addr: tuple, hz: float) -> None:
    """
    Sinusoidal sweep across all DOFs with different periods.
    Shoulder flexion: 5 s period, ±70°
    Shoulder abduction: 7 s period, 0–80°
    Shoulder rotation: 9 s period, ±40°
    Elbow flexion: 4 s period, 0–130°
    """
    interval = 1.0 / hz
    t0 = time.perf_counter()
    target = t0
    print("[mock_streamer] Mode: sinusoidal — press Ctrl+C to stop")
    while True:
        t = time.perf_counter() - t0
        flex = 70.0  * math.sin(2 * math.pi * t / 5.0)
        abd  = 40.0  * (1 - math.cos(2 * math.pi * t / 7.0))
        rot  = 40.0  * math.sin(2 * math.pi * t / 9.0)
        elb  = 65.0  * (1 - math.cos(2 * math.pi * t / 4.0))
        sock.sendto(_fmt(flex, abd, rot, elb), addr)
        print(f"\r  flex={flex:+6.1f}°  abd={abd:5.1f}°  rot={rot:+5.1f}°  elb={elb:5.1f}°", end="")
        target += interval
        sleep_s = target - time.perf_counter()
        if sleep_s > 0:
            time.sleep(sleep_s)
        else:
            target = time.perf_counter()


def run_step(sock: socket.socket, addr: tuple, hz: float) -> None:
    """
    Step through a set of reference poses, holding each for 3 seconds.
    Useful for testing calibration and angle mapping.
    """
    poses = [
        ("Arm down (rest)",     0.0,  0.0,  0.0,   0.0),
        ("Arm forward 90°",    90.0,  0.0,  0.0,   0.0),
        ("Arm side 90°",        0.0, 90.0,  0.0,   0.0),
        ("Elbow bent 90°",      0.0,  0.0,  0.0,  90.0),
        ("Arm up (overhead)",  90.0, 90.0,  0.0,   0.0),
        ("Int. rotation",       0.0,  0.0, 45.0,  90.0),
        ("Ext. rotation",       0.0,  0.0,-45.0,  90.0),
    ]
    interval = 1.0 / hz
    hold_s   = 3.0
    print("[mock_streamer] Mode: step — press Ctrl+C to stop")
    while True:
        for label, flex, abd, rot, elb in poses:
            print(f"\n  Pose: {label}  →  flex={flex:+5.1f}  abd={abd:5.1f}  rot={rot:+5.1f}  elb={elb:5.1f}")
            t_end  = time.perf_counter() + hold_s
            target = time.perf_counter()
            while time.perf_counter() < t_end:
                sock.sendto(_fmt(flex, abd, rot, elb), addr)
                target += interval
                sleep_s = target - time.perf_counter()
                if sleep_s > 0:
                    time.sleep(sleep_s)
                else:
                    target = time.perf_counter()


def run_static(sock: socket.socket, addr: tuple, hz: float) -> None:
    """Send zero angles continuously — baseline jitter test."""
    interval = 1.0 / hz
    target = time.perf_counter()
    print("[mock_streamer] Mode: static (all zeros) — press Ctrl+C to stop")
    while True:
        sock.sendto(_fmt(0.0, 0.0, 0.0, 0.0), addr)
        target += interval
        sleep_s = target - time.perf_counter()
        if sleep_s > 0:
            time.sleep(sleep_s)
        else:
            target = time.perf_counter()


def run_random_walk(sock: socket.socket, addr: tuple, hz: float) -> None:
    """Bounded random walk — stress-test filter responsiveness."""
    interval = 1.0 / hz
    flex, abd, rot, elb = 0.0, 0.0, 0.0, 0.0
    limits = [(-90, 90), (0, 90), (-60, 60), (0, 150)]
    vals   = [flex, abd, rot, elb]
    target = time.perf_counter()
    print("[mock_streamer] Mode: random walk — press Ctrl+C to stop")
    while True:
        for i, (lo, hi) in enumerate(limits):
            vals[i] = float(max(lo, min(hi, vals[i] + random.gauss(0, 2.0))))
        sock.sendto(_fmt(*vals), addr)
        print(f"\r  flex={vals[0]:+6.1f}°  abd={vals[1]:5.1f}°  rot={vals[2]:+5.1f}°  elb={vals[3]:5.1f}°", end="")
        target += interval
        sleep_s = target - time.perf_counter()
        if sleep_s > 0:
            time.sleep(sleep_s)
        else:
            target = time.perf_counter()


def main() -> None:
    parser = argparse.ArgumentParser(description="MonoArm UDP angle data generator.")
    parser.add_argument("--host",   default="127.0.0.1", help="Destination IP (default: 127.0.0.1)")
    parser.add_argument("--port",   type=int, default=9000, help="UDP port (default: 9000)")
    parser.add_argument("--hz",     type=float, default=30.0, help="Send rate in Hz (default: 30)")
    parser.add_argument("--mode",   choices=["sinusoidal", "step", "static", "random"],
                        default="sinusoidal", help="Animation mode")
    args = parser.parse_args()

    addr = (args.host, args.port)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    print(f"[mock_streamer] Sending to {args.host}:{args.port} at {args.hz:.0f} Hz")
    print(f"[mock_streamer] Packet format: S,flex,abd,rot,elbow\\n")

    try:
        if args.mode == "sinusoidal":
            run_sinusoidal(sock, addr, args.hz)
        elif args.mode == "step":
            run_step(sock, addr, args.hz)
        elif args.mode == "static":
            run_static(sock, addr, args.hz)
        elif args.mode == "random":
            run_random_walk(sock, addr, args.hz)
    except KeyboardInterrupt:
        print("\n[mock_streamer] Stopped.")
    finally:
        sock.close()


if __name__ == "__main__":
    main()
