"""
evaluate_panoptic.py — MonoArm Framework Validation on CMU Panoptic Studio
=============================================================================

Real (non-simulated) ground-truth validation of MediaPipe / MoveNet /
PoseNet, used as a substitute for Human3.6M when H3.6M account approval
is unavailable (registration at vision.imar.ro has been effectively
closed for an extended period).

Unlike evaluate_h36m.py's default "synthetic" mode (which injects
literature-calibrated noise into ground-truth angles to *simulate*
predictions), this script always runs the real pose estimators frame by
frame against real extracted video images and compares the result to
real Panoptic Studio 3D ground truth. There is no synthetic mode here.

Data layout expected
---------------------
    <sequence_dir>/
        hdPose3d_stage1_coco19/body3DScene_00000000.json, ...
        hdImgs/<camera>/<camera>_00000000.jpg, ...   (e.g. camera "00_00")

Usage
-----
    python -m scripts.evaluate_panoptic \\
        --sequence_dir ../outputs/panoptic-toolbox-master/171204_pose1_sample \\
        --camera 00_00 \\
        --frameworks mediapipe movenet_lightning posenet
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import cv2
import numpy as np

from src.evaluation.panoptic_loader import iter_panoptic_sequence
from src.evaluation.metrics import (
    evaluate_framework, print_metrics_table, metrics_to_dict, JOINTS,
)
from src.evaluation.eval_plots import (
    plot_validation_summary, plot_bland_altman,
    plot_scatter_gt, plot_error_cdf, plot_timeseries_vs_gt,
)
from src.pose import load_estimator
from src.processing.angle_solver import compute_arm_angles
from src.processing.angle_filter import AngleFilterBank


def load_gt(sequence_dir: Path, max_frames: int | None) -> dict[str, np.ndarray]:
    pose_dir = sequence_dir / "hdPose3d_stage1_coco19"
    gts = list(iter_panoptic_sequence(pose_dir, max_frames=max_frames))
    if not gts:
        print(f"[FAIL] No GT frames parsed from {pose_dir}")
        sys.exit(1)

    frame_idx = [g.frame_idx for g in gts]
    arrays = {j: np.array([getattr(g, j) for g in gts], dtype=np.float64) for j in JOINTS}
    return frame_idx, arrays


def run_live_predictions(
    sequence_dir: Path,
    camera: str,
    frame_idx: list[int],
    frameworks: list[str],
    stream_hz: float,
) -> dict[str, dict[str, np.ndarray]]:
    """Run each real pose framework on the exact frames that have GT, in order."""
    img_dir = sequence_dir / "hdImgs" / camera
    preds: dict[str, dict[str, np.ndarray]] = {}

    for fw_name in frameworks:
        print(f"\n  Running {fw_name} on {len(frame_idx)} frames...")
        try:
            runner = load_estimator(fw_name)
        except Exception as e:
            print(f"  [FAIL] Could not load {fw_name}: {e}")
            continue

        filt = AngleFilterBank("kalman", stream_hz=stream_hz)
        angles_per_joint: dict[str, list[float]] = {j: [] for j in JOINTS}
        n_missing = 0

        for idx in frame_idx:
            fp = img_dir / f"{camera}_{idx:08d}.jpg"
            bgr = cv2.imread(str(fp))
            if bgr is None:
                n_missing += 1
                for j in JOINTS:
                    angles_per_joint[j].append(float("nan"))
                continue

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            lms = runner.process(rgb)
            a = compute_arm_angles(lms) if lms is not None else None
            if a is not None:
                a = filt.update(a)
                for j in JOINTS:
                    angles_per_joint[j].append(getattr(a, j))
            else:
                for j in JOINTS:
                    angles_per_joint[j].append(float("nan"))

        runner.close()

        for j in JOINTS:
            arr = np.array(angles_per_joint[j])
            nan_mask = np.isnan(arr)
            if nan_mask.any() and not nan_mask.all():
                ix = np.arange(len(arr))
                valid = ix[~nan_mask]
                arr = np.interp(ix, valid, arr[valid])
            elif nan_mask.all():
                arr = np.zeros(len(arr))
            angles_per_joint[j] = arr

        preds[fw_name] = angles_per_joint
        det_rate = 100.0 * (len(frame_idx) - n_missing) / len(frame_idx)
        print(f"  [{fw_name}] done. Frames read: {det_rate:.0f}%")

    return preds


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Real (non-simulated) validation against CMU Panoptic Studio ground truth."
    )
    ap.add_argument("--sequence_dir", required=True,
                     help="Path to a downloaded+extracted Panoptic sequence folder")
    ap.add_argument("--camera", default="00_00",
                     help="HD camera id whose frames were extracted (default: 00_00)")
    ap.add_argument("--max_frames", type=int, default=None)
    ap.add_argument("--frameworks", nargs="+",
                     default=["mediapipe", "movenet_lightning", "posenet"])
    ap.add_argument("--stream_hz", type=float, default=29.97,
                     help="Panoptic HD capture rate (used for the Kalman filter dt)")
    ap.add_argument("--output_dir", default="outputs/validation_panoptic")
    args = ap.parse_args()

    sequence_dir = Path(args.sequence_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[->] Loading Panoptic ground truth...")
    t0 = time.perf_counter()
    frame_idx, gt_arrays = load_gt(sequence_dir, args.max_frames)
    print(f"[OK] {len(frame_idx)} GT frames loaded in {time.perf_counter()-t0:.1f}s "
          f"(real CMU Panoptic 3D mocap-derived ground truth, camera {args.camera})")

    pred_data = run_live_predictions(
        sequence_dir, args.camera, frame_idx, args.frameworks, args.stream_hz,
    )
    if not pred_data:
        print("[FAIL] No framework produced predictions.")
        sys.exit(1)

    all_results = []
    for fw in args.frameworks:
        if fw not in pred_data:
            continue
        result = evaluate_framework(framework=fw, pred_arrays=pred_data[fw], gt_arrays=gt_arrays)
        all_results.append(result)
        print(f"  [{fw}]  MPJAE={result.mpjae:.2f} deg  r={result.mean_r:.3f}  "
              f"PCK@5={result.mean_pck_5:.1f}%")

    print_metrics_table(all_results)

    json_path = out_dir / "metrics_report.json"
    with open(json_path, "w") as f:
        json.dump(metrics_to_dict(all_results), f, indent=2)
    print(f"[OK] JSON metrics -> {json_path}")
    print("[NOTE] These numbers are measured from real inference on real "
          "CMU Panoptic Studio video + real 3D mocap ground truth -- not simulated.")

    print("\n[->] Generating figures...")
    plot_validation_summary(all_results, str(out_dir / "validation_summary.png"))
    plot_bland_altman(all_results, pred_data, gt_arrays, str(out_dir / "bland_altman.png"))
    plot_scatter_gt(all_results, pred_data, gt_arrays, str(out_dir / "scatter_gt.png"))
    plot_error_cdf(all_results, pred_data, gt_arrays, str(out_dir / "error_cdf.png"))
    plot_timeseries_vs_gt(all_results, pred_data, gt_arrays,
                           str(out_dir / "timeseries_vs_gt.png"),
                           n_frames=min(500, len(frame_idx)))

    print(f"\n[OK] All outputs saved to {out_dir}/")


if __name__ == "__main__":
    main()
