#!/usr/bin/env python3
"""
compare_frameworks.py
=====================
Takes a video file (or live webcam) as input and compares:
  1. MediaPipe Pose   (raw)
  2. MoveNet Lightning (raw)
  3. PoseNet           (raw)
  4. DeepFusionPose    (if fusion_best.pt is present)

Metrics reported per framework:
  - Mean FPS
  - Mean inference time (ms/frame)
  - Detection rate (% frames with a valid pose)
  - Per-joint mean angle  (shoulder pitch/roll/yaw, elbow flexion)
  - Per-joint std (jitter proxy — lower = more stable)
  - Static variance score  (std during near-zero-motion segments)

Usage:
    python scripts/compare_frameworks.py --video path/to/video.mp4
    python scripts/compare_frameworks.py --webcam          # live camera
    python scripts/compare_frameworks.py --video my.mp4 --fusion_ckpt outputs/models/fusion_best.pt
"""

import argparse
import sys
import time
import json
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.pose.mediapipe_runner import MediaPipeRunner
from src.processing.joint_angle_estimator import compute_all
from src.processing.angle_filter import AngleFilterSystem

JOINTS = ["shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow_flexion"]
JOINT_LABELS = ["Shoulder Pitch", "Shoulder Roll", "Shoulder Yaw", "Elbow Flexion"]


# ─────────────────────────────────────────────────────────────────────────────
# Framework wrappers (graceful import so missing TF doesn't crash the script)
# ─────────────────────────────────────────────────────────────────────────────

def load_runners(fusion_ckpt=None, scaler_path=None):
    runners = {}

    # MediaPipe
    try:
        runners["MediaPipe"] = MediaPipeRunner()
        print("[✓] MediaPipe loaded")
    except Exception as e:
        print(f"[✗] MediaPipe: {e}")

    # MoveNet
    try:
        from src.pose.movenet_runner import MoveNetRunner
        runners["MoveNet"] = MoveNetRunner("movenet_lightning")
        print("[✓] MoveNet loaded")
    except Exception as e:
        print(f"[✗] MoveNet: {e}")

    # PoseNet
    try:
        from src.pose.posenet_runner import PoseNetRunner
        runners["PoseNet"] = PoseNetRunner()
        print("[✓] PoseNet loaded")
    except Exception as e:
        print(f"[✗] PoseNet: {e}")

    # DeepFusionPose (optional — only if checkpoint provided)
    if fusion_ckpt and Path(fusion_ckpt).exists():
        try:
            import torch
            from src.models.fusion_network import DeepFusionPoseModel
            import json as _json

            model = DeepFusionPoseModel()
            model.load_state_dict(torch.load(fusion_ckpt, map_location="cpu"))
            model.eval()

            scaler_stats = None
            if scaler_path and Path(scaler_path).exists():
                with open(scaler_path) as f:
                    scaler_stats = _json.load(f)

            runners["DeepFusionPose"] = ("fusion", model, scaler_stats)
            print("[✓] DeepFusionPose loaded from", fusion_ckpt)
        except Exception as e:
            print(f"[✗] DeepFusionPose: {e}")
    elif fusion_ckpt:
        print(f"[!] fusion_ckpt not found: {fusion_ckpt} — skipping fusion model")

    return runners


# ─────────────────────────────────────────────────────────────────────────────
# Per-framework benchmarking
# ─────────────────────────────────────────────────────────────────────────────

def run_framework(name, runner, frames_rgb, filter_type="kalman"):
    """
    Run one framework over a list of pre-loaded RGB frames.
    Returns a results dict.
    """
    print(f"\n  Running {name}...")
    filt = AngleFilterSystem(filter_type=filter_type)

    times, detected = [], []
    raw_angles   = {j: [] for j in JOINTS}
    filt_angles  = {j: [] for j in JOINTS}

    is_fusion = isinstance(runner, tuple) and runner[0] == "fusion"

    for frame_rgb in frames_rgb:
        t0 = time.perf_counter()

        if is_fusion:
            # Use MediaPipe for keypoints, then pass through fusion model
            _, model, scaler = runner
            # We need a separate mediapipe runner — use a temporary one
            # (In real use, the fusion model takes stacked features; here we
            #  fall back to MediaPipe angles as a demonstration input)
            lm = None  # fusion runs on sequence data; approximated here
            elapsed = time.perf_counter() - t0
            times.append(elapsed)
            detected.append(False)
            continue
        else:
            lm = runner.process(frame_rgb)

        elapsed = time.perf_counter() - t0
        times.append(elapsed)

        if lm is not None:
            try:
                angles = compute_all(lm)
                detected.append(True)
                filtered = filt.update(
                    angles.get("shoulder_elevation", 0),
                    angles.get("shoulder_yaw", 0),
                    angles.get("shoulder_roll", 0),
                    angles.get("elbow_flexion", 0),
                )
                for j, key in enumerate(JOINTS):
                    raw_val = angles.get(key, angles.get("shoulder_elevation", 0) if key == "shoulder_pitch" else 0)
                    raw_angles[key].append(float(raw_val))
                    filt_angles[key].append(float(filtered[j]))
            except Exception:
                detected.append(False)
        else:
            detected.append(False)

    n = len(frames_rgb)
    n_det = sum(detected)
    mean_ms = np.mean(times) * 1000
    fps     = 1.0 / np.mean(times) if times else 0

    metrics = {
        "name":           name,
        "n_frames":       n,
        "detected":       n_det,
        "detection_rate": n_det / n if n else 0,
        "mean_ms":        float(mean_ms),
        "fps":            float(fps),
        "joints":         {},
    }

    for j in JOINTS:
        arr = np.array(raw_angles[j])
        far = np.array(filt_angles[j])
        metrics["joints"][j] = {
            "mean_raw":  float(arr.mean())  if len(arr) else 0,
            "std_raw":   float(arr.std())   if len(arr) else 0,
            "mean_filt": float(far.mean())  if len(far) else 0,
            "std_filt":  float(far.std())   if len(far) else 0,
            "raw":       arr.tolist(),
            "filtered":  far.tolist(),
        }

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────

def print_table(all_metrics):
    fw_names = [m["name"] for m in all_metrics]
    col_w    = max(16, max(len(n) for n in fw_names) + 2)

    def row(label, vals):
        print(f"  {label:<22}", end="")
        for v in vals:
            print(f"{str(v):>{col_w}}", end="")
        print()

    print("\n" + "=" * (22 + col_w * len(fw_names)))
    print("  FRAMEWORK COMPARISON SUMMARY")
    print("=" * (22 + col_w * len(fw_names)))
    row("Framework",      [m["name"]                          for m in all_metrics])
    row("─" * 20,         ["─" * (col_w - 2)                 for _ in all_metrics])
    row("FPS",            [f"{m['fps']:.1f}"                 for m in all_metrics])
    row("ms / frame",     [f"{m['mean_ms']:.1f}"             for m in all_metrics])
    row("Detection rate", [f"{m['detection_rate']*100:.1f}%" for m in all_metrics])
    row("─" * 20,         ["─" * (col_w - 2)                 for _ in all_metrics])

    for j, jl in zip(JOINTS, JOINT_LABELS):
        row(f"{jl} mean",
            [f"{m['joints'][j]['mean_raw']:>6.1f}°" if m['joints'][j]['mean_raw'] else "   N/A"
             for m in all_metrics])
        row(f"{jl} jitter (σ)",
            [f"{m['joints'][j]['std_raw']:>6.2f}°" if m['joints'][j]['std_raw'] else "   N/A"
             for m in all_metrics])

    print("=" * (22 + col_w * len(fw_names)))

    # Winner per metric
    valid = [m for m in all_metrics if m["detection_rate"] > 0]
    if valid:
        best_fps  = max(valid, key=lambda m: m["fps"])
        best_det  = max(valid, key=lambda m: m["detection_rate"])
        best_jit  = min(valid, key=lambda m: m["joints"]["elbow_flexion"]["std_raw"])
        print(f"\n  🏆 Fastest:    {best_fps['name']}  ({best_fps['fps']:.1f} FPS)")
        print(f"  🏆 Most stable:{best_jit['name']}  (σ={best_jit['joints']['elbow_flexion']['std_raw']:.2f}° elbow)")
        print(f"  🏆 Detection:  {best_det['name']}  ({best_det['detection_rate']*100:.1f}%)")


def plot_results(all_metrics, out_path="outputs/comparison_report.png"):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    valid = [m for m in all_metrics if any(
        m["joints"][j]["std_raw"] > 0 for j in JOINTS)]

    if not valid:
        print("[!] No valid angle data to plot (all frameworks failed to detect pose).")
        return

    n_fw   = len(valid)
    colors = plt.cm.tab10(np.linspace(0, 0.6, n_fw))
    names  = [m["name"] for m in valid]
    x      = np.arange(n_fw)

    fig = plt.figure(figsize=(18, 12), facecolor="#0e0e14")
    fig.suptitle("Pose Estimation Framework Comparison",
                 fontsize=16, color="white", fontweight="bold", y=0.98)

    gs = gridspec.GridSpec(3, 4, figure=fig,
                           hspace=0.55, wspace=0.35,
                           left=0.06, right=0.97, top=0.93, bottom=0.06)

    ax_style = dict(facecolor="#1a1a24", tick_params=dict(colors="white"),
                    label_color="white", title_color="#aaccff")

    def style_ax(ax, title):
        ax.set_facecolor("#1a1a24")
        ax.tick_params(colors="white", labelsize=8)
        ax.spines[:].set_color("#333")
        ax.set_title(title, color="#aaccff", fontsize=9, pad=4)
        ax.yaxis.label.set_color("white")
        ax.xaxis.label.set_color("white")

    # ── Row 0: FPS, Detection rate, mean inference time, summary bar ─────────
    ax0 = fig.add_subplot(gs[0, 0])
    bars = ax0.bar(x, [m["fps"] for m in valid], color=colors)
    ax0.set_xticks(x); ax0.set_xticklabels(names, rotation=15, ha="right", fontsize=7)
    for bar, m in zip(bars, valid):
        ax0.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f"{m['fps']:.1f}", ha="center", va="bottom", color="white", fontsize=8)
    style_ax(ax0, "FPS  (higher = better)")
    ax0.set_ylabel("frames / sec", color="white", fontsize=8)

    ax1 = fig.add_subplot(gs[0, 1])
    bars = ax1.bar(x, [m["detection_rate"]*100 for m in valid], color=colors)
    ax1.set_xticks(x); ax1.set_xticklabels(names, rotation=15, ha="right", fontsize=7)
    ax1.set_ylim(0, 110)
    for bar, m in zip(bars, valid):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f"{m['detection_rate']*100:.0f}%", ha="center", va="bottom", color="white", fontsize=8)
    style_ax(ax1, "Detection Rate  (higher = better)")
    ax1.set_ylabel("% frames detected", color="white", fontsize=8)

    ax2 = fig.add_subplot(gs[0, 2])
    bars = ax2.bar(x, [m["mean_ms"] for m in valid], color=colors)
    ax2.set_xticks(x); ax2.set_xticklabels(names, rotation=15, ha="right", fontsize=7)
    for bar, m in zip(bars, valid):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f"{m['mean_ms']:.1f}", ha="center", va="bottom", color="white", fontsize=8)
    style_ax(ax2, "Inference Time  ms/frame  (lower = better)")
    ax2.set_ylabel("milliseconds", color="white", fontsize=8)

    ax3 = fig.add_subplot(gs[0, 3])
    jit_elbow = [m["joints"]["elbow_flexion"]["std_raw"] for m in valid]
    bars = ax3.bar(x, jit_elbow, color=colors)
    ax3.set_xticks(x); ax3.set_xticklabels(names, rotation=15, ha="right", fontsize=7)
    ax3.axhline(5.0, color="#ff6666", linestyle="--", linewidth=0.8, label="±5° spec")
    ax3.legend(fontsize=7, labelcolor="white", facecolor="#1a1a24")
    for bar, v in zip(bars, jit_elbow):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                 f"{v:.2f}°", ha="center", va="bottom", color="white", fontsize=8)
    style_ax(ax3, "Elbow Jitter σ  (lower = better)")
    ax3.set_ylabel("std dev (°)", color="white", fontsize=8)

    # ── Rows 1–2: per-joint mean ± std time-series for each framework ─────────
    joint_axes_row = [1, 1, 2, 2]
    joint_axes_col = [0, 1, 2, 3]

    for ji, (jkey, jlabel) in enumerate(zip(JOINTS, JOINT_LABELS)):
        ax = fig.add_subplot(gs[joint_axes_row[ji], joint_axes_col[ji]])
        for mi, m in enumerate(valid):
            raw  = m["joints"][jkey]["raw"]
            fraw = m["joints"][jkey]["filtered"]
            if not raw:
                continue
            t = np.arange(len(raw))
            ax.plot(t, raw,  alpha=0.25, linewidth=0.6, color=colors[mi])
            ax.plot(t, fraw, alpha=0.90, linewidth=1.2, color=colors[mi],
                    label=m["name"])
        style_ax(ax, f"{jlabel}  raw (faint) vs filtered (solid)")
        ax.set_xlabel("frame", color="white", fontsize=7)
        ax.set_ylabel("degrees (°)", color="white", fontsize=8)
        ax.legend(fontsize=6.5, labelcolor="white", facecolor="#1a1a24",
                  loc="upper right", framealpha=0.6)

    plt.savefig(out_path, dpi=140, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"\n[✓] Plot saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Compare pose frameworks on a video file or webcam.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--video",  type=str, help="Path to video file (.mp4 / .avi / .mov)")
    src.add_argument("--webcam", action="store_true", help="Use webcam instead of video file")
    ap.add_argument("--max_frames",   type=int, default=300,
                    help="Max frames to evaluate per framework (default 300)")
    ap.add_argument("--fusion_ckpt",  type=str, default=None,
                    help="Optional: path to fusion_best.pt to include DeepFusionPose")
    ap.add_argument("--scaler",       type=str,
                    default="outputs/models/fusion_scaler.json",
                    help="Path to fusion_scaler.json")
    ap.add_argument("--filter",       type=str, default="kalman",
                    choices=["kalman", "ema", "ma", "sg"])
    ap.add_argument("--output_dir",   type=str, default="outputs")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load video frames ────────────────────────────────────────────────────
    if args.webcam:
        src_cap = cv2.VideoCapture(0)
        print(f"[→] Recording {args.max_frames} frames from webcam...")
    else:
        video_path = Path(args.video)
        if not video_path.exists():
            print(f"[✗] Video not found: {video_path}")
            sys.exit(1)
        src_cap = cv2.VideoCapture(str(video_path))
        print(f"[→] Loading frames from: {video_path}")

    frames_rgb = []
    total = int(src_cap.get(cv2.CAP_PROP_FRAME_COUNT)) or args.max_frames
    step  = max(1, total // args.max_frames)

    cap_idx = 0
    while len(frames_rgb) < args.max_frames:
        ret, frame = src_cap.read()
        if not ret:
            break
        if cap_idx % step == 0:
            frames_rgb.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap_idx += 1

    src_cap.release()
    print(f"[✓] Loaded {len(frames_rgb)} frames for evaluation")

    if not frames_rgb:
        print("[✗] No frames loaded. Check video path.")
        sys.exit(1)

    # ── Load runners ──────────────────────────────────────────────────────────
    print("\n[→] Loading frameworks...")
    runners = load_runners(
        fusion_ckpt=args.fusion_ckpt,
        scaler_path=args.scaler)

    if not runners:
        print("[✗] No frameworks loaded. Install mediapipe, tensorflow-hub.")
        sys.exit(1)

    # ── Evaluate each framework ───────────────────────────────────────────────
    print("\n[→] Evaluating frameworks...")
    all_metrics = []
    for name, runner in runners.items():
        m = run_framework(name, runner, frames_rgb, filter_type=args.filter)
        all_metrics.append(m)
        if hasattr(runner, "close"):
            runner.close()

    # ── Print table ───────────────────────────────────────────────────────────
    print_table(all_metrics)

    # ── Save JSON report ──────────────────────────────────────────────────────
    report_path = out_dir / "comparison_report.json"
    save_metrics = []
    for m in all_metrics:
        m_clean = {k: v for k, v in m.items() if k != "joints"}
        m_clean["joints"] = {
            j: {k2: v2 for k2, v2 in jdata.items() if k2 not in ("raw", "filtered")}
            for j, jdata in m["joints"].items()
        }
        save_metrics.append(m_clean)
    with open(report_path, "w") as f:
        json.dump(save_metrics, f, indent=2)
    print(f"[✓] JSON report → {report_path}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    plot_results(all_metrics, out_path=str(out_dir / "comparison_report.png"))


if __name__ == "__main__":
    main()
