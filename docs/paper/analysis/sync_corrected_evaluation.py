"""
sync_corrected_evaluation.py -- Corrected real-data validation for the MonoArm paper.

Background
----------
The first pass of this validation (see conversation history) assumed that
video frame i and ground-truth file body3DScene_<i>.json referred to the
same instant. A post-hoc check (shifting predicted vs. ground-truth arrays
and re-measuring error) showed this assumption was wrong: error drops
sharply and consistently across both frameworks and both non-gated,
non-wrapping DOFs (shoulder abduction, elbow flexion) when the predicted
signal is shifted forward by roughly 9-12 frames relative to ground truth.

This script fixes that properly, using a pre-registered, non-cherry-picked
procedure:

  1. The sync offset is estimated ONCE, from a single reference channel
     (shoulder abduction -- the only DOF free of both the rotation-
     reliability gate and the +-180 deg wraparound) and a single reference
     framework (MediaPipe -- already designated PRIMARY_FRAMEWORK elsewhere
     in this codebase, e.g. scripts/run_full_pipeline.py). The offset that
     maximises Pearson correlation between predicted and ground-truth
     abduction is used.
  2. MoveNet-Lightning's own independently-implied offset (from its own
     abduction channel) is reported as a consistency check, but the
     offset actually applied to ALL subsequent metrics for BOTH frameworks
     is the one fixed in step 1. No offset is tuned per framework or per
     joint to make any result look better.
  3. All metrics (full-sequence, wraparound-corrected flexion, rotation-
     reliability-conditioned subset) are recomputed at the fixed offset,
     and all five result figures are regenerated with the same plotting
     code used originally, so the paper's numbers and figures are
     internally consistent and reproducible from this one script.
"""

from __future__ import annotations

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "PoseTrack"))

import numpy as np

from scripts.evaluate_panoptic import load_gt, run_live_predictions
from src.evaluation.metrics import evaluate_framework, print_metrics_table, metrics_to_dict, JOINTS
from src.evaluation.eval_plots import (
    plot_validation_summary, plot_bland_altman, plot_scatter_gt,
    plot_error_cdf, plot_timeseries_vs_gt,
)
from src.evaluation.statistics import compare_systems, comparison_report_to_dicts

SEQ_DIR = Path(__file__).resolve().parents[3] / "outputs" / "panoptic-toolbox-master" / "171204_pose1_sample"
OUT_DIR = Path(__file__).resolve().parents[3] / "outputs" / "validation_panoptic_synccorrected"
CAMERA = "00_00"
STREAM_HZ = 29.97
REFERENCE_JOINT = "shoulder_abduction"
REFERENCE_FRAMEWORK = "mediapipe"
SHIFT_SEARCH_RANGE = range(-30, 31)
MIN_OVERLAP = 40


def shifted_pair(pred: np.ndarray, gt: np.ndarray, shift: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (pred, gt) arrays aligned so that pred[i] <-> gt[i+shift]."""
    if shift >= 0:
        p = pred[: len(pred) - shift] if shift > 0 else pred
        g = gt[shift:]
    else:
        p = pred[-shift:]
        g = gt[: len(gt) + shift]
    n = min(len(p), len(g))
    return p[:n], g[:n]


def best_correlation_shift(pred: np.ndarray, gt: np.ndarray, search_range) -> tuple[int, float]:
    best_shift, best_r = 0, -np.inf
    for shift in search_range:
        p, g = shifted_pair(pred, gt, shift)
        if len(p) < MIN_OVERLAP:
            continue
        if np.std(p) < 1e-9 or np.std(g) < 1e-9:
            continue
        r = float(np.corrcoef(p, g)[0, 1])
        if r > best_r:
            best_shift, best_r = shift, r
    return best_shift, best_r


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/5] Loading real ground truth and running real inference (unshifted)...")
    frame_idx, gt = load_gt(SEQ_DIR, None)
    preds = run_live_predictions(SEQ_DIR, CAMERA, frame_idx, ["mediapipe", "movenet_lightning"], STREAM_HZ)

    print("\n[2/5] Estimating sync offset from the pre-registered reference "
          f"channel ({REFERENCE_FRAMEWORK} / {REFERENCE_JOINT})...")
    ref_shift, ref_r = best_correlation_shift(
        preds[REFERENCE_FRAMEWORK][REFERENCE_JOINT], gt[REFERENCE_JOINT], SHIFT_SEARCH_RANGE
    )
    print(f"  Reference offset (applied to everything below): shift = {ref_shift:+d} frames, "
          f"Pearson r = {ref_r:.3f} at that shift")

    consistency = {}
    for fw in ["mediapipe", "movenet_lightning"]:
        s, r = best_correlation_shift(preds[fw][REFERENCE_JOINT], gt[REFERENCE_JOINT], SHIFT_SEARCH_RANGE)
        consistency[fw] = {"independent_best_shift": s, "r_at_best_shift": r}
        print(f"  [consistency check] {fw} independently implies shift={s:+d} (r={r:.3f})")

    SHIFT = ref_shift  # the ONE offset used everywhere below

    print(f"\n[3/5] Recomputing all metrics at the fixed offset shift={SHIFT:+d}...")
    gt_aligned = {}
    preds_aligned = {fw: {} for fw in preds}
    n_aligned = None
    for j in JOINTS:
        for fw in preds:
            p, g = shifted_pair(preds[fw][j], gt[j], SHIFT)
            preds_aligned[fw][j] = p
            if n_aligned is None:
                n_aligned = len(p)
        gt_aligned[j] = shifted_pair(preds[REFERENCE_FRAMEWORK][j], gt[j], SHIFT)[1]

    n_aligned = len(gt_aligned[REFERENCE_JOINT])
    print(f"  Aligned overlap: n = {n_aligned} frames (was 101 before trimming for the shift)")

    full_results = [
        evaluate_framework(fw, preds_aligned[fw], gt_aligned, joints=JOINTS)
        for fw in preds_aligned
    ]
    print_metrics_table(full_results)

    print("\n[4/5] Wraparound-corrected shoulder flexion (aligned data)...")
    def circ_diff(a, b):
        d = a - b
        return (d + 180) % 360 - 180

    wraparound = {}
    for fw in preds_aligned:
        naive = np.abs(preds_aligned[fw]["shoulder_flexion"] - gt_aligned["shoulder_flexion"])
        circ = np.abs(circ_diff(preds_aligned[fw]["shoulder_flexion"], gt_aligned["shoulder_flexion"]))
        wraparound[fw] = {
            "naive_mae": float(naive.mean()),
            "circular_mae": float(circ.mean()),
            "n_wrap_affected": int((naive - circ > 1).sum()),
        }
        print(f"  {fw}: naive={naive.mean():.2f} circular={circ.mean():.2f} "
              f"(n_wrap_affected={wraparound[fw]['n_wrap_affected']})")

    print("\n[5/6] Statistical significance: MediaPipe vs. MoveNet-Lightning "
          "(paired t-test, Wilcoxon, Cohen's d_z, bootstrap CI, Holm-Bonferroni)...")
    print("  CAVEAT: frames within this single continuous motion are NOT independent "
          "observations, so these tests should be read as descriptive, not as formal "
          "inference in the textbook sense -- reported for completeness since the "
          "framework implements this protocol, not as a claim of a properly powered "
          "significance test.")
    stats_report = compare_systems(preds_aligned, gt_aligned, joints=JOINTS, seed=42)
    stats_rows = comparison_report_to_dicts(stats_report)
    for row in stats_rows:
        print(f"  {row['joint']:20s} MAE(a)={row['mae_a']:6.2f} "
              f"MAE(b)={row['mae_b']:6.2f}  wilcoxon_p={row['wilcoxon_p']:.4f}  "
              f"cohens_d={row['cohens_d']:+.2f}  significant(Holm)={row['significant']}")

    print("\n[6/6] Rotation-reliability-conditioned subset (aligned data)...")
    mask = gt_aligned["elbow_flexion"] >= 25.0
    gt_masked = {j: a[mask] for j, a in gt_aligned.items()}
    preds_masked = {fw: {j: a[mask] for j, a in preds_aligned[fw].items()} for fw in preds_aligned}
    cond_results = [
        evaluate_framework(fw, preds_masked[fw], gt_masked, joints=JOINTS)
        for fw in preds_masked
    ]
    print(f"  Reliable-rotation frames: {mask.sum()} / {n_aligned}")
    print_metrics_table(cond_results)

    # MoveNet's own elbow-flexion gating rate (unaffected by shift choice --
    # this is a property of MoveNet's own predictions vs its own threshold)
    movenet_pred_elbow = preds_aligned["movenet_lightning"]["elbow_flexion"]
    movenet_gate_rate = float(np.mean(movenet_pred_elbow >= 25.0))

    out = {
        "shift_frames_applied": SHIFT,
        "reference_framework": REFERENCE_FRAMEWORK,
        "reference_joint": REFERENCE_JOINT,
        "reference_pearson_r_at_shift": ref_r,
        "consistency_check": consistency,
        "n_frames_aligned": n_aligned,
        "full": metrics_to_dict(full_results),
        "wraparound": wraparound,
        "rotation_reliable_subset": metrics_to_dict(cond_results),
        "n_frames_reliable": int(mask.sum()),
        "significance_tests": stats_rows,
        "movenet_pred_elbow_gate_rate": movenet_gate_rate,
        "movenet_pred_elbow_mean": float(movenet_pred_elbow.mean()),
        "gt_elbow_mean_aligned": float(gt_aligned["elbow_flexion"].mean()),
        "gt_elbow_gate_rate_aligned": float(np.mean(gt_aligned["elbow_flexion"] >= 25.0)),
    }
    with open(OUT_DIR / "sync_corrected_analysis.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {OUT_DIR / 'sync_corrected_analysis.json'}")

    print("\nGenerating figures at corrected alignment...")
    plot_validation_summary(full_results, str(OUT_DIR / "validation_summary.png"))
    plot_bland_altman(full_results, preds_aligned, gt_aligned, str(OUT_DIR / "bland_altman.png"))
    plot_scatter_gt(full_results, preds_aligned, gt_aligned, str(OUT_DIR / "scatter_gt.png"))
    plot_error_cdf(full_results, preds_aligned, gt_aligned, str(OUT_DIR / "error_cdf.png"))
    plot_timeseries_vs_gt(full_results, preds_aligned, gt_aligned,
                           str(OUT_DIR / "timeseries_vs_gt.png"), n_frames=n_aligned)
    print(f"Done. All outputs in {OUT_DIR}")


if __name__ == "__main__":
    main()
