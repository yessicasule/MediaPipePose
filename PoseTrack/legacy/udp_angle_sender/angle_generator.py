import argparse
import math
import socket
import time


def _sine_wave(t, period, amplitude, offset=0.0):
    return offset + amplitude * math.sin(2 * math.pi * t / period)


def _generate_angles(t):
    shoulder_pitch = _sine_wave(t, period=4.0, amplitude=45.0, offset=15.0)
    shoulder_yaw   = _sine_wave(t, period=6.0, amplitude=30.0, offset=0.0)
    shoulder_roll  = _sine_wave(t, period=8.0, amplitude=20.0, offset=0.0)
    elbow_flex     = _sine_wave(t, period=3.0, amplitude=60.0, offset=60.0)
    return shoulder_pitch, shoulder_yaw, shoulder_roll, elbow_flex


def run(host: str, port: int, hz: float):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    interval = 1.0 / hz
    print(f"Sending to {host}:{port} at {hz} Hz  (Ctrl+C to stop)")
    print("Format: S,shoulder_pitch,shoulder_yaw,shoulder_roll,elbow_flex")

    t_start = time.perf_counter()
    next_send = t_start

    try:
        while True:
            now = time.perf_counter()
            t = now - t_start

            sp, sy, sr, ef = _generate_angles(t)
            packet = f"S,{sp:.3f},{sy:.3f},{sr:.3f},{ef:.3f}\n"
            sock.sendto(packet.encode(), (host, port))

            next_send += interval
            sleep_for = next_send - time.perf_counter()
            if sleep_for > 0:
                time.sleep(sleep_for)

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        sock.close()


def main():
    parser = argparse.ArgumentParser(description="UDP Arm Angle Data Generator")
    parser.add_argument("--host", default="127.0.0.1", help="Destination host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=9000, help="Destination UDP port (default: 9000)")
    parser.add_argument("--hz",   type=float, default=30.0, help="Transmit rate in Hz (default: 30)")
    args = parser.parse_args()
    run(args.host, args.port, args.hz)


if __name__ == "__main__":
    main()
