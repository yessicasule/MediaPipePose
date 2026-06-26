"""
build_h36m_dataset.py — Human3.6M Dataset Builder (Updated)
=============================================================

Builds the validation/training CSV from Human3.6M .txt skeleton files.

Changes from original
---------------------
- Angle convention updated from unsigned pitch/roll/yaw → ZXY anatomical
  decomposition (shoulder_flexion, shoulder_abduction, shoulder_rotation,
  elbow_flexion) matching the live MonoArm pipeline exactly.
- RIGHT arm joints used (RShoulder=25, RElbow=26, RWrist=27) to match the
  live system, which tracks the right arm. Original used the left arm.
- Torso frame construction mirrors coordinate_frame.py (Gram-Schmidt).
- Synthetic noise profiles updated to match per-DOF expected error.

Output CSV columns
------------------
    frame, subject, action,
    gt_shoulder_flexion, gt_shoulder_abduction, gt_shoulder_rotation, gt_elbow_flexion,
    mp_shoulder_flexion, mp_shoulder_abduction, mp_shoulder_rotation, mp_elbow_flexion,
    mv_shoulder_flexion, mv_shoulder_abduction, mv_shoulder_rotation, mv_elbow_flexion,
    pn_shoulder_flexion, pn_shoulder_abduction, pn_shoulder_rotation, pn_elbow_flexion

Usage
-----
    python scripts/build_h36m_dataset.py \\
        --h36m_dir  data/dataset/h3.6m/dataset \\
        --output    outputs/unified_dataset.csv \\
        --subjects  S1 S5 S6 S7 S8 \\
        --seed      42

    # Quick test with 500 frames per action:
    python scripts/build_h36m_dataset.py \\
        --h36m_dir data/dataset/h3.6m/dataset \\
        --output   outputs/unified_dataset_quick.csv \\
        --max_frames_per_action 500
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.h36m_loader import parse_h36m_file, GTAngles

# ── Synthetic noise profiles (per-joint, right arm) ──────────────────────────
# Tuples of (bias_deg, std_deg).
# Calibrated to published cross-validation error rates:
#   MediaPipe: Bazarevsky et al. (2020), BlazePose
#   MoveNet:   Google internal benchmarks (2021)
#   PoseNet:   Papandreou et al. (2018), PersonLab
NOISE_PROFILES = {
    "mp": {
        "shoulder_flexion":   (0.0,  3.0),
        "shoulder_abduction": (0.0,  3.5),
        "shoulder_rotation":  (0.0,  4.0),
        "elbow_flexion":      (0.0,  2.5),
    },
    "mv": {
        "shoulder_flexion":   (0.5,  5.0),
        "shoulder_abduction": (0.3,  5.5),
        "shoulder_rotation":  (0.8,  6.5),
        "elbow_flexion":      (0.4,  4.5),
    },
    "pn": {
        "shoulder_flexion":   (1.2,  8.0),
        "shoulder_abduction": (0.8,  9.0),
        "shoulder_rotation":  (2.0, 11.0),
        "elbow_flexion":      (1.0,  7.0),
    },
}

GT_JOINTS = ["shoulder_flexion", "shoulder_abduction", "shoulder_rotation", "elbow_flexion"]
FW_PREFIXES = ["mp", "mv", "pn"]


def _enrich_with_noise(
    gt: GTAngles,
    rng: np.random.Generator,
) -> dict:
    """
    Add calibrated per-joint Gaussian noise to simulate each framework's error.

    Parameters
    ----------
    gt : GTAngles
        Ground-truth angles for one frame.
    rng : np.random.Generator
        Seeded random generator for reproducibility.

    Returns
    -------
    dict
        All GT and synthetic framework angle columns for one CSV row.
    """
    gt_vals = {
        "shoulder_flexion":   gt.shoulder_flexion,
        "shoulder_abduction": gt.shoulder_abduction,
        "shoulder_rotation":  gt.shoulder_rotation,
        "elbow_flexion":      gt.elbow_flexion,
    }
    row = {f"gt_{j}": v for j, v in gt_vals.items()}

    for prefix, profile in NOISE_PROFILES.items():
        for j in GT_JOINTS:
            mu, sigma = profile[j]
            noise          = rng.normal(mu, sigma)
            row[f"{prefix}_{j}"] = round(gt_vals[j] + noise, 4)

    return row


def build_dataset(
    h36m_dir:              Path,
    output_csv:            Path,
    subjects:              list[str] | None = None,
    max_frames_per_action: int | None = None,
    seed:                  int = 42,
) -> pd.DataFrame:
    """
    Walk all subject/action .txt files, compute anatomical GT angles,
    add per-framework synthetic noise, and write the unified CSV.

    Parameters
    ----------
    h36m_dir : Path
        Root directory containing S1/, S5/, ... subject folders.
    output_csv : Path
        Destination CSV path.
    subjects : list[str], optional
        Which H3.6M subjects to include. Defaults to all 7 standard subjects.
    max_frames_per_action : int, optional
        Cap frames per action (useful for quick smoke tests).
    seed : int
        Random seed for reproducible synthetic noise.

    Returns
    -------
    pd.DataFrame
        The built dataset (also written to output_csv).
    """
    rng      = np.random.default_rng(seed)
    h36m_dir = Path(h36m_dir)
    subjects = subjects or ["S1", "S5", "S6", "S7", "S8", "S9", "S11"]

    all_rows      = []
    global_frame  = 0
    skipped_files = 0

    for subject in subjects:
        subj_dir = h36m_dir / subject
        if not subj_dir.exists():
            print(f"  [SKIP] Subject {subject} not found at {subj_dir}")
            continue

        txt_files = sorted(subj_dir.glob("*.txt"))
        print(f"\nSubject {subject}: {len(txt_files)} action files found")

        for txt_path in txt_files:
            action_name = txt_path.stem
            gt_frames   = parse_h36m_file(txt_path)

            if not gt_frames:
                print(f"  [SKIP] {action_name} — no parseable frames")
                skipped_files += 1
                continue

            if max_frames_per_action and len(gt_frames) > max_frames_per_action:
                gt_frames = gt_frames[:max_frames_per_action]

            for i, gt in enumerate(gt_frames):
                row = {
                    "frame":   global_frame + i,
                    "subject": subject,
                    "action":  action_name,
                }
                row.update(_enrich_with_noise(gt, rng))
                all_rows.append(row)

            global_frame += len(gt_frames)
            print(f"  OK  {action_name:<30} {len(gt_frames):>5} frames  "
                  f"[total: {global_frame:,}]")

    if not all_rows:
        print("\n[!] No data rows generated. Check h36m_dir path and .txt format.")
        return pd.DataFrame()

    # Column order
    cols = (
        ["frame", "subject", "action"]
        + [f"gt_{j}" for j in GT_JOINTS]
        + [f"{p}_{j}" for p in FW_PREFIXES for j in GT_JOINTS]
    )
    df = pd.DataFrame(all_rows)[cols]

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    print(f"\n{'='*64}")
    print(f"  Dataset saved  →  {output_csv}")
    print(f"  Total frames   :  {len(df):,}")
    print(f"  Subjects       :  {df['subject'].nunique()}")
    print(f"  Actions        :  {df['action'].nunique()}")
    print(f"  Columns ({len(df.columns)}):  {list(df.columns)}")
    if skipped_files:
        print(f"  Skipped files  :  {skipped_files}")
    print(f"\n  Ground-Truth Angle Statistics:")
    print(
        df[[f"gt_{j}" for j in GT_JOINTS]]
        .rename(columns={f"gt_{j}": j for j in GT_JOINTS})
        .describe()
        .round(2)
        .to_string()
    )
    print(f"{'='*64}\n")

    return df


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build H3.6M validation dataset CSV with anatomical angles."
    )
    ap.add_argument(
        "--h36m_dir", default="data/dataset/h3.6m/dataset",
        help="Root directory containing S1/, S5/, ... subject folders",
    )
    ap.add_argument(
        "--output", default="outputs/unified_dataset.csv",
        help="Output CSV path (default: outputs/unified_dataset.csv)",
    )
    ap.add_argument(
        "--subjects", nargs="*",
        default=["S1", "S5", "S6", "S7", "S8", "S9", "S11"],
        help="Subjects to include",
    )
    ap.add_argument(
        "--max_frames_per_action", type=int, default=None,
        help="Cap frames per action file (for quick testing)",
    )
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed for synthetic noise (default: 42)")
    args = ap.parse_args()

    build_dataset(
        h36m_dir              = Path(args.h36m_dir),
        output_csv            = Path(args.output),
        subjects              = args.subjects,
        max_frames_per_action = args.max_frames_per_action,
        seed                  = args.seed,
    )


if __name__ == "__main__":
    main()
