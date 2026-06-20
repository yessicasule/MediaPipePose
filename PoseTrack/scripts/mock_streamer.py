"""
mock_streamer.py — Test Unity WITHOUT trained models.

Sends realistic animated arm angle data over UDP to Unity
so you can verify the scene wiring before Kaggle models are ready.

Simulates 4 streams (MP, MV, FU, GR) with sinusoidal motion
that makes the arm visibly move up/down and bend.

Usage:
    python mock_streamer.py
    python mock_streamer.py --host 127.0.0.1 --port 9000 --fps 30
"""
import argparse, math, socket, time

def send(sock, addr, prefix, values):
    body = ",".join(f"{v:.3f}" for v in values)
    sock.sendto(f"{prefix},{body}".encode(), addr)

def run(host, port, fps):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    addr = (host, port)
    dt   = 1.0 / fps
    t    = 0.0

    print(f"Mock streamer → {host}:{port}  at {fps} fps")
    print("Press Ctrl+C to stop.\n")
    print("Animating: shoulder_pitch 20-90°, elbow_flexion 30-140°")

    try:
        while True:
            # ── Base motion (realistic arm raise + elbow bend) ─────────────
            pitch = 55 + 35 * math.sin(t * 0.8)          # 20°–90°
            roll  = 10 +  8 * math.sin(t * 0.5 + 1.0)    # 2°–18°
            yaw   = 15 +  5 * math.sin(t * 0.3 + 0.5)    # 10°–20°
            elbow = 85 + 55 * math.sin(t * 0.6 + 0.8)    # 30°–140°

            # ── Each framework adds slightly different noise ────────────────
            mp = [pitch,                roll,                yaw,                elbow              ]
            mv = [pitch + 2.1*math.sin(t*3.1), roll + 1.5*math.cos(t*2.7),
                  yaw   + 1.8*math.sin(t*2.3), elbow+ 2.0*math.cos(t*3.5)]
            fu = [pitch + 0.5*math.sin(t*5.0), roll + 0.4*math.cos(t*4.5),
                  yaw   + 0.3*math.sin(t*4.2), elbow+ 0.6*math.cos(t*5.3)]
            gr = [pitch + 0.1*math.sin(t*6.0), roll + 0.1*math.cos(t*5.5),
                  yaw   + 0.1*math.sin(t*5.2), elbow+ 0.1*math.cos(t*6.3)]
            unc = 0.5 + 0.3 * math.sin(t * 0.4)           # uncertainty 0.2–0.8

            send(sock, addr, "MP", mp)
            send(sock, addr, "MV", mv)
            send(sock, addr, "FU", fu + [unc])
            send(sock, addr, "GR", gr)

            print(f"\r  t={t:6.1f}s  pitch={pitch:5.1f}°  elbow={elbow:5.1f}°  unc={unc:.2f}", end="")
            t   += dt
            time.sleep(dt)

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        sock.close()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9000)
    ap.add_argument("--fps",  type=int, default=30)
    args = ap.parse_args()
    run(args.host, args.port, args.fps)
