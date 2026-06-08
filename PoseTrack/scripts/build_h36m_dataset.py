"""
build_h36m_dataset.py
─────────────────────────────────────────────────────────────────────────────
Builds the unified training CSV from Human3.6M .txt skeleton files ONLY.
No video, no webcam, no MediaPipe/MoveNet/PoseNet required at this step.

What it does
────────────
For each H3.6M subject/action .txt file:
  - Parses the 99-value Euler-angle skeleton (33 joints × 3 axes) per frame
  - Extracts Left-Arm 3D joint positions: LShoulder, LElbow, LWrist, LHip
  - Computes 4 ground-truth angles: shoulder_pitch, shoulder_roll,
                                     shoulder_yaw, elbow_flexion
  - Creates THREE synthetic "framework" columns (mp_, mv_, pn_) by adding
    calibrated Gaussian noise to the ground truth. This simulates realistic
    measurement error from each framework:
        MediaPipe  σ = 3.0°  (best, low noise)
        MoveNet    σ = 5.0°  (medium noise)
        PoseNet    σ = 8.0°  (highest noise, 2D projection loss)

Output CSV (17 columns):
  frame | mp_shoulder_pitch | mp_shoulder_roll | mp_shoulder_yaw | mp_elbow_flexion
        | mv_shoulder_pitch | mv_shoulder_roll | mv_shoulder_yaw | mv_elbow_flexion
        | pn_shoulder_pitch | pn_shoulder_roll | pn_shoulder_yaw | pn_elbow_flexion
        | gt_shoulder_pitch | gt_shoulder_roll | gt_shoulder_yaw | gt_elbow_flexion

Why synthetic noise?
────────────────────
Running MediaPipe/MoveNet/PoseNet requires camera frames extracted from
proprietary H3.6M videos (not included in the public skeleton release).
Synthetic Gaussian noise is the standard academic proxy used in fusion papers
when the exact video frames are unavailable — it preserves the temporal
structure of real motion while modelling each tracker's known error profile.
(If you later run real trackers on frames, simply replace the mp_/mv_/pn_
columns in the CSV with real output — the fusion model trains identically.)

Usage
─────
python scripts/build_h36m_dataset.py \\
    --h36m_dir  data/dataset/h3.6m/dataset \\
    --output    outputs/unified_dataset.csv \\
    --seed      42
"""

import argparse
import sys
import math
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))


# ─────────────────────────────────────────────────────────────────
# H3.6M joint index reference
# The .txt files store 33 joints × 3 Euler angles = 99 values/row.
# We only need Left-Arm joints:
#   Joint 17 → LShoulder
#   Joint 18 → LElbow
#   Joint 19 → LWrist
#   Joint  6 → LHip       (for torso reference vector)
#   Joint  1 → RHip       (for lateral axis reference)
#   Joint 14 → Spine/Chest (for torso up vector)
# ─────────────────────────────────────────────────────────────────

H36M_JOINTS = {
    "root":       0,
    "rhip":       1,
    "rkne":       2,
    "rank":       3,
    "lhip":       6,
    "lkne":       7,
    "lank":       8,
    "spine":     12,
    "spine1":    13,
    "neck":      14,
    "lshoulder": 17,
    "rshoulder": 25,
    "lelbow":    18,
    "lwrist":    19,
}


def _vec(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Unit vector from a → b."""
    v = b - a
    n = np.linalg.norm(v)
    return v / n if n > 1e-8 else v


def _angle_deg(v1: np.ndarray, v2: np.ndarray) -> float:
    """Angle between two unit vectors in degrees."""
    c = np.clip(np.dot(v1 / (np.linalg.norm(v1) + 1e-9),
                        v2 / (np.linalg.norm(v2) + 1e-9)), -1.0, 1.0)
    return math.degrees(math.acos(c))


def compute_left_arm_angles(joints_xyz: np.ndarray) -> dict:
    """
    Given joints_xyz shape (33, 3), compute the 4 left-arm angles.

    Returns dict: shoulder_pitch, shoulder_roll, shoulder_yaw, elbow_flexion
    """
    ls = joints_xyz[H36M_JOINTS["lshoulder"]]
    le = joints_xyz[H36M_JOINTS["lelbow"]]
    lw = joints_xyz[H36M_JOINTS["lwrist"]]
    lh = joints_xyz[H36M_JOINTS["lhip"]]
    rh = joints_xyz[H36M_JOINTS["rhip"]]
    sp = joints_xyz[H36M_JOINTS["spine1"]]  # torso chest reference

    upper_arm = _vec(ls, le)          # shoulder → elbow
    forearm   = _vec(le, lw)          # elbow → wrist
    torso_up  = _vec((lh + rh) / 2, sp)  # hip midpoint → spine (up vector)
    lat_axis  = _vec(rh, lh)              # right hip → left hip (lateral axis)

    # shoulder_pitch: angle between torso_up and upper_arm
    shoulder_pitch = _angle_deg(torso_up, upper_arm)

    # shoulder_roll: angle between lat_axis and projected upper_arm on frontal plane
    frontal_normal = np.cross(torso_up, lat_axis)
    frontal_normal /= (np.linalg.norm(frontal_normal) + 1e-9)
    ua_proj = upper_arm - np.dot(upper_arm, frontal_normal) * frontal_normal
    shoulder_roll = _angle_deg(lat_axis, ua_proj)

    # shoulder_yaw: angle between frontal plane projection and lateral
    sagittal_normal = np.cross(torso_up, frontal_normal)
    sagittal_normal /= (np.linalg.norm(sagittal_normal) + 1e-9)
    ua_sag = upper_arm - np.dot(upper_arm, sagittal_normal) * sagittal_normal
    shoulder_yaw = _angle_deg(torso_up, ua_sag)

    # elbow_flexion: 180° – angle between upper_arm and forearm vectors
    elbow_flexion = 180.0 - _angle_deg(upper_arm, forearm)

    return {
        "shoulder_pitch": round(shoulder_pitch, 4),
        "shoulder_roll":  round(shoulder_roll,  4),
        "shoulder_yaw":   round(shoulder_yaw,   4),
        "elbow_flexion":  round(elbow_flexion,  4),
    }


def parse_h36m_txt(txt_path: Path) -> list[dict]:
    """
    Parse one H3.6M .txt file (one row = one frame, 99 comma-separated floats).
    Returns a list of angle dicts, one per frame.
    """
    rows = []
    lines = txt_path.read_text(encoding="utf-8").strip().split("\n")
    for line in lines:
        vals = [v for v in line.strip().split(",") if v.strip()]
        if len(vals) < 99:
            continue
        joints_flat = np.array([float(v) for v in vals[:99]])
        joints_xyz  = joints_flat.reshape(33, 3)
        try:
            angles = compute_left_arm_angles(joints_xyz)
            rows.append(angles)
        except Exception:
            # Skip malformed frames
            continue
    return rows


def add_framework_noise(
    gt_rows: list[dict],
    rng: np.random.Generator,
) -> list[dict]:
    """
    For each frame, add calibrated Gaussian noise to simulate each framework:
        MediaPipe  → σ = 3.0°
        MoveNet    → σ = 5.0°
        PoseNet    → σ = 8.0°  (also adds slight systematic bias per joint)
    """
    JOINTS = ["shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow_flexion"]

    # Framework noise profiles (mean_bias°, std°)
    PROFILES = {
        "mp": {"bias": 0.0,  "std": 3.0},   # MediaPipe — best
        "mv": {"bias": 0.5,  "std": 5.0},   # MoveNet   — medium
        "pn": {"bias": 1.2,  "std": 8.0},   # PoseNet   — 2D, most noise
    }

    results = []
    for row in gt_rows:
        enriched = dict(row)  # copy of gt
        for fw, profile in PROFILES.items():
            for j in JOINTS:
                noise = rng.normal(profile["bias"], profile["std"])
                enriched[f"{fw}_{j}"] = round(row[j] + noise, 4)
        results.append(enriched)
    return results


def build_dataset(
    h36m_dir: Path,
    output_csv: Path,
    subjects: list[str] = None,
    max_frames_per_action: int = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Walk all subject/action .txt files, parse angles, add noise, concat.
    """
    rng = np.random.default_rng(seed)
    h36m_dir = Path(h36m_dir)
    subjects = subjects or ["S1", "S5", "S6", "S7", "S8", "S9", "S11"]

    all_rows = []
    global_frame = 0

    for subject in subjects:
        subj_dir = h36m_dir / subject
        if not subj_dir.exists():
            print(f"  [SKIP] {subject} not found at {subj_dir}")
            continue

        txt_files = sorted(subj_dir.glob("*.txt"))
        print(f"\nSubject {subject}: {len(txt_files)} action files")

        for txt_path in txt_files:
            action_name = txt_path.stem  # e.g. "walking_1"
            gt_rows = parse_h36m_txt(txt_path)

            if not gt_rows:
                print(f"  [SKIP] {action_name} — no parseable frames")
                continue

            if max_frames_per_action and len(gt_rows) > max_frames_per_action:
                gt_rows = gt_rows[:max_frames_per_action]

            enriched = add_framework_noise(gt_rows, rng)

            for i, row in enumerate(enriched):
                all_rows.append({
                    "frame":            global_frame + i,
                    "subject":          subject,
                    "action":           action_name,

                    # MediaPipe (simulated)
                    "mp_shoulder_pitch": row["mp_shoulder_pitch"],
                    "mp_shoulder_roll":  row["mp_shoulder_roll"],
                    "mp_shoulder_yaw":   row["mp_shoulder_yaw"],
                    "mp_elbow_flexion":  row["mp_elbow_flexion"],

                    # MoveNet (simulated)
                    "mv_shoulder_pitch": row["mv_shoulder_pitch"],
                    "mv_shoulder_roll":  row["mv_shoulder_roll"],
                    "mv_shoulder_yaw":   row["mv_shoulder_yaw"],
                    "mv_elbow_flexion":  row["mv_elbow_flexion"],

                    # PoseNet (simulated)
                    "pn_shoulder_pitch": row["pn_shoulder_pitch"],
                    "pn_shoulder_roll":  row["pn_shoulder_roll"],
                    "pn_shoulder_yaw":   row["pn_shoulder_yaw"],
                    "pn_elbow_flexion":  row["pn_elbow_flexion"],

                    # Ground truth
                    "gt_shoulder_pitch": row["shoulder_pitch"],
                    "gt_shoulder_roll":  row["shoulder_roll"],
                    "gt_shoulder_yaw":   row["shoulder_yaw"],
                    "gt_elbow_flexion":  row["elbow_flexion"],
                })

            global_frame += len(enriched)
            print(f"  OK {action_name:<28}  {len(enriched):>5} frames  "
                  f"[total so far: {global_frame}]")

    df = pd.DataFrame(all_rows)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    print(f"\n{'='*60}")
    print(f"Unified dataset saved → {output_csv}")
    print(f"Total frames:   {len(df):,}")
    print(f"Subjects:       {df['subject'].nunique()}")
    print(f"Actions:        {df['action'].nunique()}")
    print(f"Columns ({len(df.columns)}):  {list(df.columns)}")
    print(f"\nSample statistics (Ground Truth angles):")
    print(df[["gt_shoulder_pitch", "gt_shoulder_roll",
              "gt_shoulder_yaw", "gt_elbow_flexion"]].describe().round(2).to_string())
    return df


def main():
    ap = argparse.ArgumentParser(
        description="Build unified training CSV from H3.6M skeleton .txt files."
    )
    ap.add_argument(
        "--h36m_dir",
        default="data/dataset/h3.6m/dataset",
        help="Root dir containing S1/, S5/, ... subject folders",
    )
    ap.add_argument(
        "--output",
        default="outputs/unified_dataset.csv",
        help="Output CSV path",
    )
    ap.add_argument(
        "--subjects",
        nargs="*",
        default=["S1", "S5", "S6", "S7", "S8", "S9", "S11"],
        help="Which subjects to include",
    )
    ap.add_argument(
        "--max_frames_per_action",
        type=int,
        default=None,
        help="Cap frames per action file (useful for quick testing)",
    )
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    build_dataset(
        h36m_dir=Path(args.h36m_dir),
        output_csv=Path(args.output),
        subjects=args.subjects,
        max_frames_per_action=args.max_frames_per_action,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
