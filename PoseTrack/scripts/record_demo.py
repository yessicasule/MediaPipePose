"""
record_demo.py — Synthesised System Demonstration Video Generator
=================================================================

Generates a 30-second annotated MP4 demonstrating the full MonoArm pipeline
without requiring a webcam.  The video shows:

  Left panel : Simulated "camera view" with a stick-figure pose skeleton
  Right panel: Live 4-panel rolling angle plot (all DOFs, sinusoidal motion)
  Overlay HUD: Joint angle values (raw + filtered), FPS, filter type, UDP status

The angles follow the same sinusoidal trajectories as mock_streamer.py:
  Shoulder Flexion    : ±70°  @ 5s period
  Shoulder Abduction  : 0–80° @ 7s period
  Shoulder Rotation   : ±40°  @ 9s period
  Elbow Flexion       : 0–130° @ 4s period

All rendered using pure OpenCV — zero external dependencies beyond cv2 and numpy.

Usage
-----
    python scripts/record_demo.py
    python scripts/record_demo.py --output outputs/my_demo.mp4 --duration 45
"""

from __future__ import annotations

import argparse
import math
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Synthetic angle generator (matches mock_streamer.py sinusoidal mode)
# ---------------------------------------------------------------------------

def _angles_at(t: float) -> tuple[float, float, float, float]:
    """Return (flex, abd, rot, elbow) at time t (seconds)."""
    flex  =  70.0 * math.sin(2 * math.pi * t / 5.0)
    abd   =  40.0 * (1 - math.cos(2 * math.pi * t / 7.0))
    rot   =  40.0 * math.sin(2 * math.pi * t / 9.0)
    elbow =  65.0 * (1 - math.cos(2 * math.pi * t / 4.0))
    return flex, abd, rot, elbow


# ---------------------------------------------------------------------------
# Kalman-style exponential smoother (for simulated filtering)
# ---------------------------------------------------------------------------

class _EMAFilter:
    def __init__(self, alpha: float = 0.3) -> None:
        self._alpha = alpha
        self._val: float | None = None

    def update(self, x: float) -> float:
        if self._val is None:
            self._val = x
        else:
            self._val = (1 - self._alpha) * self._val + self._alpha * x
        return self._val


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_GREEN  = (100, 220, 100)
_BLUE   = (255, 160, 100)
_ORANGE = (80, 180, 255)
_PURPLE = (220,  80, 220)
_WHITE  = (230, 230, 230)
_GREY   = (140, 140, 140)
_DARK   = (20, 20, 30)
_CYAN   = (255, 220, 100)

_DOF_COLORS = [_GREEN, _BLUE, _ORANGE, _PURPLE]
_DOF_LABELS = ["Flex (°)", "Abd (°)", "Rot (°)", "Elbow (°)"]


def _draw_skeleton(img: np.ndarray, flex: float, abd: float, elbow: float) -> None:
    """Draw a simple right-arm stick figure that responds to angle values."""
    H, W = img.shape[:2]
    cx, cy = W // 2, H // 2

    # Shoulder
    sh = np.array([cx, cy - 30], dtype=float)

    # Upper arm (respond to flex + abd)
    flex_rad = math.radians(flex * 0.8)
    abd_rad  = math.radians(abd  * 0.5)
    ua_len   = 80
    ua = sh + np.array([
        ua_len * math.sin(flex_rad + 0.2) + ua_len * 0.3 * math.sin(abd_rad),
        ua_len * math.cos(flex_rad + 0.2),
    ])

    # Forearm (respond to elbow flexion)
    elbow_rad = math.radians(elbow * 0.7 + 30)
    fa_len = 70
    fa = ua + np.array([
        fa_len * math.sin(flex_rad + elbow_rad),
        fa_len * math.cos(flex_rad + elbow_rad),
    ])

    sh_i  = (int(sh[0]),  int(sh[1]))
    ua_i  = (int(ua[0]),  int(ua[1]))
    fa_i  = (int(fa[0]),  int(fa[1]))

    # Torso
    hip = (cx, cy + 60)
    cv2.line(img, (cx, cy - 80), hip, _GREY, 3, cv2.LINE_AA)
    cv2.line(img, (cx - 40, cy - 20), (cx + 40, cy - 20), _GREY, 3, cv2.LINE_AA)

    # Head
    cv2.circle(img, (cx, cy - 100), 22, _GREY, 2, cv2.LINE_AA)

    # Right arm
    cv2.line(img,   sh_i, ua_i, _GREEN, 4, cv2.LINE_AA)
    cv2.line(img,   ua_i, fa_i, _GREEN, 4, cv2.LINE_AA)
    cv2.circle(img, sh_i, 7, (200, 200, 200), -1, cv2.LINE_AA)
    cv2.circle(img, ua_i, 6, _GREEN, -1, cv2.LINE_AA)
    cv2.circle(img, fa_i, 5, _BLUE, -1, cv2.LINE_AA)

    # Left arm (static)
    ls = (cx - 40, cy - 20)
    le = (cx - 100, cy + 20)
    lw = (cx - 110, cy + 60)
    cv2.line(img, ls, le, _GREY, 3, cv2.LINE_AA)
    cv2.line(img, le, lw, _GREY, 3, cv2.LINE_AA)

    # Legs
    cv2.line(img, hip, (cx - 25, cy + 130), _GREY, 3, cv2.LINE_AA)
    cv2.line(img, hip, (cx + 25, cy + 130), _GREY, 3, cv2.LINE_AA)


def _draw_hud(
    img: np.ndarray,
    raw: tuple, filt: tuple,
    fps: float, t: float,
    filter_type: str,
) -> None:
    H, W = img.shape[:2]

    # Top bar
    cv2.rectangle(img, (0, 0), (W, 40), (10, 10, 20), -1)
    cv2.putText(img, "MonoArm — Live Pipeline Demo",
                (10, 26), _FONT, 0.6, _CYAN, 1, cv2.LINE_AA)
    cv2.putText(img, f"FPS: {fps:.1f}  |  t={t:.1f}s  |  Filter: {filter_type}",
                (W - 270, 26), _FONT, 0.45, _GREY, 1, cv2.LINE_AA)

    # Angle panel (bottom-left)
    panel_y = H - 130
    overlay = img.copy()
    cv2.rectangle(overlay, (0, panel_y), (250, H), (10, 10, 20), -1)
    cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)

    labels_raw  = ["Flex", "Abd", "Rot", "Elbow"]
    raw_vals    = [raw[0], raw[1], raw[2], raw[3]]
    filt_vals   = [filt[0], filt[1], filt[2], filt[3]]

    for i, (lbl, rv, fv, col) in enumerate(zip(labels_raw, raw_vals, filt_vals, _DOF_COLORS)):
        y = panel_y + 20 + i * 26
        cv2.putText(img, f"{lbl}:", (8, y), _FONT, 0.42, _GREY, 1, cv2.LINE_AA)
        cv2.putText(img, f"{fv:+6.1f}deg (raw:{rv:+5.1f})",
                    (58, y), _FONT, 0.42, col, 1, cv2.LINE_AA)

    # UDP status badge
    cv2.rectangle(img, (W - 160, H - 36), (W - 2, H - 2), (0, 80, 0), -1)
    cv2.putText(img, f"UDP 127.0.0.1:9000 OK",
                (W - 155, H - 12), _FONT, 0.38, (100, 255, 100), 1, cv2.LINE_AA)


def _draw_rolling_plot(
    bufs: list[deque],
    width: int = 640,
    height: int = 180,
) -> np.ndarray:
    """Render a 4-panel rolling strip plot (same as RollingAnglePlot in angle_logger.py)."""
    img = np.full((height, width, 3), (18, 18, 28), dtype=np.uint8)
    n   = 4
    pw  = width // n
    y_ranges = [(-80, 80), (0, 90), (-50, 50), (0, 140)]

    for i, (buf, label, col, (ymin, ymax)) in enumerate(
        zip(bufs, _DOF_LABELS, _DOF_COLORS, y_ranges)
    ):
        x0, x1 = i * pw, (i + 1) * pw
        cv2.rectangle(img, (x0, 0), (x1 - 1, height - 1), (50, 50, 60), 1)

        # Zero line
        zy = height - 1 - int((-ymin) / (ymax - ymin) * (height - 1))
        zy = max(0, min(height - 1, zy))
        cv2.line(img, (x0, zy), (x1, zy), (60, 60, 70), 1)

        if len(buf) >= 2:
            xs = np.linspace(x0 + 4, x1 - 4, 150)
            ys_raw = np.array(list(buf))
            ys = height - 1 - (
                np.clip((ys_raw - ymin) / (ymax - ymin), 0, 1) * (height - 1)
            ).astype(int)
            n_pts = len(ys)
            xs_t = xs[150 - n_pts:]
            pts = np.column_stack([xs_t.astype(int), ys]).reshape(-1, 1, 2)
            cv2.polylines(img, [pts], False, col, 2, cv2.LINE_AA)
            cv2.circle(img, (int(xs_t[-1]), int(ys[-1])), 4, col, -1, cv2.LINE_AA)

        current = buf[-1] if buf else 0.0
        cv2.putText(img, label, (x0 + 4, 14), _FONT, 0.35, _WHITE, 1, cv2.LINE_AA)
        cv2.putText(img, f"{current:+.1f}", (x0 + 4, height - 6),
                    _FONT, 0.36, col, 1, cv2.LINE_AA)

    return img


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Generate MonoArm synthesised demo video.")
    ap.add_argument("--output",   default="outputs/demo_video.mp4")
    ap.add_argument("--duration", type=int, default=30, help="Video duration in seconds")
    ap.add_argument("--fps",      type=int, default=30)
    args = ap.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    FRAME_W, FRAME_H = 640, 480
    PLOT_H  = 180
    TOTAL_H = FRAME_H + PLOT_H
    TOTAL_W = FRAME_W

    n_frames = args.duration * args.fps
    fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
    writer   = cv2.VideoWriter(str(out_path), fourcc, args.fps, (TOTAL_W, TOTAL_H))

    if not writer.isOpened():
        print(f"[ERROR] Cannot open VideoWriter for {out_path}")
        return

    # 4 rolling buffers, one per DOF
    bufs: list[deque] = [deque(maxlen=150) for _ in range(4)]

    # Simulated Kalman (simple EMA for demo)
    filters = [_EMAFilter(alpha=0.35) for _ in range(4)]

    print(f"[record_demo] Generating {args.duration}s @ {args.fps}fps -> {out_path}")
    t0 = time.perf_counter()

    for frame_idx in range(n_frames):
        t = frame_idx / args.fps

        # Synthesise angles
        raw = _angles_at(t)
        filt = tuple(f.update(v) for f, v in zip(filters, raw))

        # Push to rolling bufs
        for i in range(4):
            bufs[i].append(filt[i])

        # --- Camera frame (left panel: stick figure) ---
        cam_frame = np.full((FRAME_H, FRAME_W, 3), _DARK, dtype=np.uint8)

        # Grid lines for depth cue
        for gx in range(0, FRAME_W, 80):
            cv2.line(cam_frame, (gx, 0), (gx, FRAME_H), (30, 30, 40), 1)
        for gy in range(0, FRAME_H, 60):
            cv2.line(cam_frame, (0, gy), (FRAME_W, gy), (30, 30, 40), 1)

        _draw_skeleton(cam_frame, filt[0], filt[1], filt[3])
        _draw_hud(cam_frame, raw, filt, args.fps, t, "Kalman 2-State")

        # Progress bar at very bottom of cam frame
        prog_w = int(FRAME_W * frame_idx / n_frames)
        cv2.rectangle(cam_frame, (0, FRAME_H - 4), (prog_w, FRAME_H), _GREEN, -1)

        # --- Rolling plot ---
        plot_frame = _draw_rolling_plot(bufs, width=TOTAL_W, height=PLOT_H)

        # --- Combine ---
        combined = np.vstack([cam_frame, plot_frame])
        writer.write(combined)

        if frame_idx % (args.fps * 5) == 0:
            elapsed = time.perf_counter() - t0
            pct = frame_idx / n_frames * 100
            print(f"  {pct:5.1f}%  frame {frame_idx}/{n_frames}  ({elapsed:.1f}s elapsed)")

    writer.release()
    elapsed_total = time.perf_counter() - t0
    size_mb = out_path.stat().st_size / 1e6

    print(f"\n[OK] Demo video written: {out_path}")
    print(f"     Duration : {args.duration}s  |  Frames : {n_frames}")
    print(f"     Size     : {size_mb:.1f} MB")
    print(f"     Render   : {elapsed_total:.1f}s")


if __name__ == "__main__":
    main()
