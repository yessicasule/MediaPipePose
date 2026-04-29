"""
Render a side-by-side skeleton comparison video from benchmark JSON outputs.

Left panel  → MediaPipe skeleton drawn on the original frame
Right panel → MoveNet skeleton drawn on the same frame

Both frameworks' keypoints come from the per-frame arrays saved by
run_mediapipe_on_frames.py and run_movenet_on_frames.py, so no re-inference
is needed — just load the JSONs and draw.

Usage:
    python benchmarks/render_comparison_video.py \
        --frames_dir  data/sessions/arm_test_01/frames \
        --mediapipe   outputs/benchmarks/.../results/mediapipe.json \
        --movenet     outputs/benchmarks/.../results/movenet.json \
        --out         outputs/comparison_arm_test_01.mp4

Optional:
    --fps 30          output video frame rate
    --conf 0.3        minimum keypoint confidence to draw (both frameworks)
    --max_frames 300  stop after N frames
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Skeleton definitions
# ---------------------------------------------------------------------------

# MediaPipe Pose — 33 landmarks
_MP_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),
    (9,10),
    (11,12),
    (11,13),(13,15),(15,17),(15,19),(15,21),(17,19),
    (12,14),(14,16),(16,18),(16,20),(16,22),(18,20),
    (11,23),(12,24),(23,24),
    (23,25),(24,26),(25,27),(26,28),
    (27,29),(28,30),(29,31),(30,32),(27,31),(28,32),
]
# Arm keypoints highlighted in brighter colour
_MP_ARM_KPS = {11,12,13,14,15,16}

# MoveNet / COCO — 17 landmarks  (stored as [y_norm, x_norm, score])
# 0:nose 1:l_eye 2:r_eye 3:l_ear 4:r_ear
# 5:l_shoulder 6:r_shoulder 7:l_elbow 8:r_elbow
# 9:l_wrist 10:r_wrist 11:l_hip 12:r_hip
# 13:l_knee 14:r_knee 15:l_ankle 16:r_ankle
_MN_CONNECTIONS = [
    (0,1),(0,2),(1,3),(2,4),           # face
    (5,6),                             # shoulder bar
    (5,7),(7,9),                       # left arm
    (6,8),(8,10),                      # right arm
    (5,11),(6,12),(11,12),             # torso
    (11,13),(13,15),(12,14),(14,16),   # legs
]
_MN_ARM_KPS = {5,6,7,8,9,10}

# Colour palette (BGR)
_COL_MP_BONE   = (100, 220,  80)   # green-ish
_COL_MP_ARM    = ( 50, 255, 255)   # yellow
_COL_MP_KP     = (255, 255, 255)
_COL_MN_BONE   = (220,  80, 100)   # red-ish
_COL_MN_ARM    = ( 50, 200, 255)   # orange
_COL_MN_KP     = (255, 255, 255)

_HEADER_H   = 44   # pixels for top info strip per panel
_FOOTER_H   = 32   # pixels for bottom stats strip

# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def _draw_mp_skeleton(canvas, kps_xy_vis, conf_thresh, h, w):
    """kps_xy_vis: list of [x_norm, y_norm, visibility] — 33 points."""
    if kps_xy_vis is None:
        return
    pts = []
    for x_n, y_n, vis in kps_xy_vis:
        pts.append((int(x_n * w), int(y_n * h), vis))

    for a, b in _MP_CONNECTIONS:
        if a >= len(pts) or b >= len(pts):
            continue
        xa, ya, va = pts[a]
        xb, yb, vb = pts[b]
        if va < conf_thresh or vb < conf_thresh:
            continue
        is_arm = (a in _MP_ARM_KPS) or (b in _MP_ARM_KPS)
        colour = _COL_MP_ARM if is_arm else _COL_MP_BONE
        thickness = 3 if is_arm else 2
        cv2.line(canvas, (xa, ya), (xb, yb), colour, thickness, cv2.LINE_AA)

    for i, (x, y, vis) in enumerate(pts):
        if vis < conf_thresh:
            continue
        r = 5 if i in _MP_ARM_KPS else 3
        cv2.circle(canvas, (x, y), r, _COL_MP_KP, -1, cv2.LINE_AA)


def _draw_mn_skeleton(canvas, kps_yx_score, conf_thresh, h, w):
    """kps_yx_score: list of [y_norm, x_norm, score] — 17 points (MoveNet order)."""
    if kps_yx_score is None:
        return
    pts = []
    for y_n, x_n, sc in kps_yx_score:
        pts.append((int(x_n * w), int(y_n * h), sc))

    for a, b in _MN_CONNECTIONS:
        if a >= len(pts) or b >= len(pts):
            continue
        xa, ya, sa = pts[a]
        xb, yb, sb = pts[b]
        if sa < conf_thresh or sb < conf_thresh:
            continue
        is_arm = (a in _MN_ARM_KPS) or (b in _MN_ARM_KPS)
        colour = _COL_MN_ARM if is_arm else _COL_MN_BONE
        thickness = 3 if is_arm else 2
        cv2.line(canvas, (xa, ya), (xb, yb), colour, thickness, cv2.LINE_AA)

    for i, (x, y, sc) in enumerate(pts):
        if sc < conf_thresh:
            continue
        r = 5 if i in _MN_ARM_KPS else 3
        cv2.circle(canvas, (x, y), r, _COL_MN_KP, -1, cv2.LINE_AA)


def _make_header(w, label, colour_bgr, latency_ms, avg_score):
    strip = np.zeros((_HEADER_H, w, 3), dtype=np.uint8)
    strip[:] = (30, 30, 30)
    dot_col = tuple(int(c) for c in colour_bgr)
    cv2.rectangle(strip, (8, 10), (20, _HEADER_H - 10), dot_col, -1)
    cv2.putText(strip, label, (28, _HEADER_H - 12),
                cv2.FONT_HERSHEY_DUPLEX, 0.65, (240, 240, 240), 1, cv2.LINE_AA)
    info = f"{latency_ms:.1f} ms   score {avg_score:.2f}"
    cv2.putText(strip, info, (w - 210, _HEADER_H - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)
    return strip


def _make_footer(total_w, frame_idx, mp_lat, mn_lat):
    strip = np.zeros((_FOOTER_H, total_w, 3), dtype=np.uint8)
    strip[:] = (20, 20, 20)
    text = (f"Frame {frame_idx:04d}   "
            f"MediaPipe {mp_lat:.1f} ms   MoveNet {mn_lat:.1f} ms")
    cv2.putText(strip, text, (12, _FOOTER_H - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1, cv2.LINE_AA)
    return strip


def _divider(h, w=3):
    div = np.zeros((h, w, 3), dtype=np.uint8)
    div[:] = (60, 60, 60)
    return div


# ---------------------------------------------------------------------------
# Main render loop
# ---------------------------------------------------------------------------

def render(
    frames_dir: Path,
    mp_data: list,
    mn_data: list,
    out_path: Path,
    fps: float,
    conf_thresh: float,
    max_frames: int | None,
):
    frames = sorted(frames_dir.glob("*.jpg"))
    if not frames:
        raise RuntimeError(f"No .jpg frames in {frames_dir}")

    n = min(len(frames), len(mp_data), len(mn_data))
    if max_frames is not None:
        n = min(n, max_frames)

    # Probe frame size
    probe = cv2.imread(str(frames[0]))
    fh, fw = probe.shape[:2]

    panel_w = fw
    total_w = panel_w * 2 + 3   # 3px divider
    total_h = _HEADER_H + fh + _FOOTER_H

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (total_w, total_h))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for {out_path}")

    print(f"Rendering {n} frames → {out_path}")

    for i in range(n):
        frame = cv2.imread(str(frames[i]))
        if frame is None:
            continue

        mp_frame = mp_data[i]
        mn_frame = mn_data[i]

        mp_lat   = mp_frame.get("inference_ms", 0.0)
        mp_score = mp_frame.get("avg_keypoint_score", 0.0)
        mn_lat   = mn_frame.get("inference_ms", 0.0)
        mn_score = mn_frame.get("avg_keypoint_score", 0.0)

        # --- Left panel: MediaPipe ---
        left = frame.copy()
        _draw_mp_skeleton(left, mp_frame.get("keypoints_xy_vis"), conf_thresh, fh, fw)
        left_header = _make_header(panel_w, "MediaPipe Pose (33 kps)",
                                   _COL_MP_ARM, mp_lat, mp_score)

        # --- Right panel: MoveNet ---
        right = frame.copy()
        _draw_mn_skeleton(right, mn_frame.get("keypoints_yx_score"), conf_thresh, fh, fw)
        right_header = _make_header(panel_w, "MoveNet Lightning (17 kps)",
                                    _COL_MN_ARM, mn_lat, mn_score)

        # --- Footer ---
        footer = _make_footer(total_w, i, mp_lat, mn_lat)

        # --- Assemble ---
        top_row   = np.hstack([left_header,  _divider(_HEADER_H), right_header])
        body_row  = np.hstack([left,          _divider(fh),        right])
        composed  = np.vstack([top_row, body_row, footer])

        writer.write(composed)

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{n} frames written")

    writer.release()
    print(f"Done. Video saved to: {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Render side-by-side skeleton comparison video from benchmark JSONs"
    )
    ap.add_argument("--frames_dir",  required=True, help="Directory of .jpg frames")
    ap.add_argument("--mediapipe",   required=True, help="mediapipe.json from run_mediapipe_on_frames.py")
    ap.add_argument("--movenet",     required=True, help="movenet.json from run_movenet_on_frames.py")
    ap.add_argument("--out",         required=True, help="Output .mp4 path")
    ap.add_argument("--fps",         type=float, default=30.0)
    ap.add_argument("--conf",        type=float, default=0.3,
                    help="Min keypoint confidence to draw (default 0.3)")
    ap.add_argument("--max_frames",  type=int,   default=None)
    args = ap.parse_args()

    mp_result = json.loads(Path(args.mediapipe).read_text(encoding="utf-8"))
    mn_result = json.loads(Path(args.movenet).read_text(encoding="utf-8"))

    mp_data = sorted(mp_result["per_frame"], key=lambda x: x["frame_index"])
    mn_data = sorted(mn_result["per_frame"], key=lambda x: x["frame_index"])

    render(
        frames_dir  = Path(args.frames_dir),
        mp_data     = mp_data,
        mn_data     = mn_data,
        out_path    = Path(args.out),
        fps         = args.fps,
        conf_thresh = args.conf,
        max_frames  = args.max_frames,
    )


if __name__ == "__main__":
    main()
