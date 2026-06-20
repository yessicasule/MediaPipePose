"""
run_demo.py — Complete integrated demonstration script.

THE single command to run for the final presentation:
    python scripts/run_demo.py

What it does:
  1. Opens webcam, overlays MediaPipe pose landmarks
  2. Computes + filters joint angles (Kalman by default)
  3. Recognizes gestures (WAVE, RAISE_ARM, REACH_FORWARD, ELBOW_FLEX, REST)
  4. Streams all 4 UDP prefixes to Unity (MP / MV / FU / GR)
     - MediaPipe-only mode when no trained models present
     - Full Fusion + GAN pipeline when models are available
  5. Logs joint angles to CSV (press L to toggle)
  6. Shows live FPS + latency in overlay
  7. Saves angle plot automatically on exit

Controls:
    Q  — quit
    C  — run 3-pose calibration (arm_down → arm_forward → elbow_flex)
    L  — toggle logging on/off (auto-saves CSV + plot on stop)
    S  — save snapshot PNG
    F  — cycle filter: Kalman → EMA → Moving Average → Savitzky-Golay
"""

import argparse, sys, time, socket
from pathlib import Path
from collections import deque

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.pose.mediapipe_runner import MediaPipeRunner
from src.processing.joint_angle_estimator import compute_all
from src.processing.angle_filter import AngleFilterSystem
from src.processing.calibration import CalibrationManager
from src.processing.angle_logger import AngleLogger, AngleVisualizer
from src.processing.gesture_recognizer import GestureRecognizer, draw_gesture_overlay
from src.streaming.udp_streamer import UdpAngleStreamer

FILTER_NAMES = ["kalman", "ema", "ma", "sg"]

# ── Optional model loader ─────────────────────────────────────────────────────
def _try_load_models(fusion_ckpt, gan_ckpt, scaler_path):
    try:
        import torch, json
        if not all(Path(p).exists() for p in [fusion_ckpt, gan_ckpt, scaler_path]):
            return None, None, None, None
        from src.models.fusion_network import DeepFusionPoseModel
        device = "cuda" if torch.cuda.is_available() else "cpu"
        with open(scaler_path) as f:
            sc = json.load(f)
        col_min = np.array(sc["col_min"], dtype=np.float32)
        col_max = np.array(sc["col_max"], dtype=np.float32)
        fusion = DeepFusionPoseModel()
        fusion.load_state_dict(torch.load(fusion_ckpt, map_location=device))
        fusion.to(device)
        gan = DeepFusionPoseModel()
        gan.load_state_dict(torch.load(gan_ckpt, map_location=device))
        gan.eval().to(device)
        print(f"✅  Fusion + GAN models loaded ({device})")
        return fusion, gan, (col_min, col_max), device
    except Exception as e:
        print(f"⚠️  Models not loaded ({e}) — running MediaPipe-only mode")
        return None, None, None, None


# ── UDP helper ────────────────────────────────────────────────────────────────
def _udp_send(sock, addr, prefix, vals):
    body = ",".join(f"{v:.3f}" for v in vals)
    sock.sendto(f"{prefix},{body}".encode(), addr)


# ── Overlay ───────────────────────────────────────────────────────────────────
def _draw_panel(frame, angles, fps, latency, logging_on, filter_name, calibrated, mode):
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (315, 235), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.60, frame, 0.40, 0, frame)

    def put(txt, y, col=(215, 215, 215), sc=0.50):
        cv2.putText(frame, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, sc, col, 1, cv2.LINE_AA)

    put(f"DeepFusionPose  [{mode}]",             22,  (80, 200, 255), 0.55)
    put(f"FPS {fps:4.1f}   Latency {latency:5.1f} ms", 42,  (140, 255, 140))
    put(f"Filter {filter_name.upper()}   {'[LOG]' if logging_on else '[log off]'}", 60,
        (0, 220, 100) if logging_on else (130, 130, 130))
    put(f"Calib {'OK' if calibrated else 'NONE'}",   76,
        (0, 220, 100) if calibrated else (80, 80, 220))

    cv2.line(frame, (8, 86), (308, 86), (55, 55, 55), 1)
    put(f"Shoulder Pitch  {angles.get('shoulder_elevation',0):6.1f}°", 102)
    put(f"Shoulder Yaw    {angles.get('shoulder_yaw',0):6.1f}°",       118)
    put(f"Shoulder Roll   {angles.get('shoulder_roll',0):6.1f}°",      134)
    put(f"Elbow Flexion   {angles.get('elbow_flexion',0):6.1f}°",      150)
    cv2.line(frame, (8, 160), (308, 160), (55, 55, 55), 1)
    put("Q=Quit  C=Calib  L=Log  S=Snap  F=Filter", 176, (100, 100, 100), 0.42)
    return frame


# ── Calibration ───────────────────────────────────────────────────────────────
def _run_calibration(cap, runner, filt, cal_mgr):
    poses = [
        ("arm_down",    "Hang arm DOWN at your side"),
        ("arm_forward", "Extend arm FORWARD horizontally"),
        ("elbow_flex",  "BEND elbow to ~90 degrees"),
    ]
    for name, instr in poses:
        print(f"\n  Calibration: {instr}  — press SPACE to capture, ESC to skip")
        cal_mgr.start_calibration_pose(name)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            lm  = runner.process(rgb)
            cv2.putText(frame, instr, (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
            cv2.putText(frame, "SPACE = capture  |  ESC = skip",
                        (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
            if lm is not None:
                fil = filt.update(compute_all(lm))
                cv2.putText(frame, f"elbow={fil['elbow_flexion']:.1f}  pitch={fil['shoulder_elevation']:.1f}",
                            (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 100), 1)
            cv2.imshow("DeepFusionPose — Demo", frame)
            k = cv2.waitKey(1) & 0xFF
            if k == 32 and lm is not None:
                cal_mgr.capture_pose(filt.update(compute_all(lm)))
                break
            if k == 27:
                break
    cal_mgr.finalize_calibration()
    out = ROOT / "outputs" / "calibration.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    cal_mgr.save(out)
    print(f"  Calibration saved → {out}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="DeepFusionPose — Final Demo")
    ap.add_argument("--camera",      type=int, default=0)
    ap.add_argument("--host",        default="127.0.0.1")
    ap.add_argument("--port",        type=int, default=9000)
    ap.add_argument("--filter",      default="kalman", choices=FILTER_NAMES)
    ap.add_argument("--fusion_ckpt", default=str(ROOT / "outputs/models/fusion_best.pt"))
    ap.add_argument("--gan_ckpt",    default=str(ROOT / "outputs/models/gan_generator_best.pt"))
    ap.add_argument("--scaler",      default=str(ROOT / "outputs/models/fusion_scaler.json"))
    ap.add_argument("--no_models",   action="store_true")
    args = ap.parse_args()

    # Models
    fusion_model = gan_model = scaler_data = dev = None
    if not args.no_models:
        fusion_model, gan_model, scaler_data, dev = _try_load_models(
            args.fusion_ckpt, args.gan_ckpt, args.scaler)
    mode = "Fusion+GAN" if fusion_model else "MediaPipe-only"

    # Components
    runner     = MediaPipeRunner()
    fi         = FILTER_NAMES.index(args.filter)
    filt       = AngleFilterSystem(FILTER_NAMES[fi])
    cal_mgr    = CalibrationManager()
    gesture    = GestureRecognizer()
    logger     = AngleLogger(output_dir=ROOT / "outputs" / "sessions")
    visualizer = AngleVisualizer()
    streamer   = UdpAngleStreamer(host=args.host, port=args.port, hz=30)

    # Camera
    cap = cv2.VideoCapture(args.camera, cv2.CAP_MSMF)
    if not cap.isOpened():
        cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    for _ in range(10): cap.read()
    time.sleep(0.3)
    streamer.start()

    # State
    SEQ_LEN     = 60
    ang_buf     = deque(maxlen=SEQ_LEN)
    logging_on  = False
    calibrated  = False
    snap_n      = 0
    frame_n     = 0
    fps         = 0.0
    latency_ms  = 0.0
    t_fps       = time.perf_counter()
    addr        = (args.host, args.port)

    print(f"\n  Mode: [{mode}]")
    print("  Controls: Q=Quit  C=Calibrate  L=Log  S=Snapshot  F=Cycle-filter\n")

    try:
        while True:
            t0 = time.perf_counter()
            ret, frame = cap.read()
            if not ret:
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            lm  = runner.process(rgb)

            raw = {"shoulder_elevation": 0.0, "shoulder_yaw": 0.0,
                   "shoulder_roll": 0.0, "elbow_flexion": 90.0}
            if lm is not None:
                raw = compute_all(lm)

            fil = filt.update(raw)
            if calibrated:
                fil = cal_mgr.apply(fil)

            # Gesture
            g_name, g_conf = gesture.update({
                "shoulder_pitch": fil.get("shoulder_elevation", 0),
                "shoulder_yaw":   fil.get("shoulder_yaw", 0),
                "shoulder_roll":  fil.get("shoulder_roll", 0),
                "elbow_flexion":  fil.get("elbow_flexion", 90),
            })

            four = [fil.get("shoulder_elevation", 0),
                    fil.get("shoulder_roll", 0),
                    fil.get("shoulder_yaw", 0),
                    fil.get("elbow_flexion", 90)]
            ang_buf.append(four * 3)

            fu_a = four
            gr_a = four
            unc  = 1.0

            if fusion_model is not None and len(ang_buf) >= SEQ_LEN:
                import torch
                from src.models.fusion_network import mc_predict
                col_min, col_max = scaler_data
                seq  = np.array(list(ang_buf), dtype=np.float32)
                rng  = col_max - col_min; rng[rng == 0] = 1.0
                seq  = 2.0 * (seq - col_min) / rng - 1.0
                x    = torch.tensor(seq).unsqueeze(0).to(dev)
                m, s = mc_predict(fusion_model, x, n_samples=10)
                fu_a = m[0, -1, :].tolist()
                unc  = float(s[0, -1, :].mean())
                with torch.no_grad():
                    gr_a = gan_model(x)[0, -1, :].tolist()

            # UDP → Unity
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            _udp_send(sock, addr, "MP", four)
            _udp_send(sock, addr, "MV", [v + np.random.normal(0, 2) for v in four])
            _udp_send(sock, addr, "FU", fu_a + [unc])
            _udp_send(sock, addr, "GR", gr_a)
            sock.close()

            if logging_on:
                logger.log(fil)

            frame_n += 1
            latency_ms = (time.perf_counter() - t0) * 1000
            if frame_n % 15 == 0:
                fps = 15 / (time.perf_counter() - t_fps)
                t_fps = time.perf_counter()

            _draw_panel(frame, fil, fps, latency_ms, logging_on,
                        FILTER_NAMES[fi], calibrated, mode)
            draw_gesture_overlay(frame, g_name, g_conf)
            cv2.imshow("DeepFusionPose — Demo", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                _run_calibration(cap, runner, filt, cal_mgr)
                calibrated = True
            elif key == ord('l'):
                if not logging_on:
                    logger.start(); logging_on = True
                    print("  Logging started.")
                else:
                    logger.stop(); logging_on = False
                    visualizer.plot_history(logger)
                    print("  Logging stopped + plot saved.")
            elif key == ord('s'):
                p = ROOT / "outputs" / f"snapshot_{snap_n:03d}.png"
                p.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(p), frame)
                snap_n += 1
                print(f"  Snapshot → {p}")
            elif key == ord('f'):
                fi = (fi + 1) % len(FILTER_NAMES)
                filt = AngleFilterSystem(FILTER_NAMES[fi])
                print(f"  Filter → {FILTER_NAMES[fi].upper()}")

    except KeyboardInterrupt:
        pass
    finally:
        if logging_on:
            logger.stop()
            visualizer.plot_history(logger)
        cap.release()
        streamer.stop()
        runner.close()
        cv2.destroyAllWindows()
        print("\n  Demo ended.")


if __name__ == "__main__":
    main()
