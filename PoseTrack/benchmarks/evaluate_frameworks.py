import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.processing.joint_angle_estimator import Vec3, elbow_flexion_deg
from src.evaluation.metrics import compute_jitter, compute_static_pose_stability, compute_fps


# Common joints (COCO-style names). We map each framework to these.
COCO = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]


def _load(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def _norm_xy_from_result(res: dict, frame: dict) -> dict[str, tuple[float, float, float] | None]:
    """
    Returns dict name -> (x_norm, y_norm, score) or None.
    """
    lib = res.get("library")

    if lib == "MoveNet":
        # keypoints_yx_score: [ [y,x,score] * 17 ] in COCO order.
        kps = frame.get("keypoints_yx_score")
        if not kps:
            return {k: None for k in COCO}
        out = {}
        for i, name in enumerate(COCO):
            y, x, s = kps[i]
            out[name] = (float(x), float(y), float(s))
        return out

    if lib == "PoseNet":
        # keypoints_xy_score in PoseNet order (COCO 17), pixel coords.
        kps = frame.get("keypoints_xy_score")
        sz = frame.get("image_size") or {}
        w = float(sz.get("width") or 0)
        h = float(sz.get("height") or 0)
        if not kps or w <= 0 or h <= 0:
            return {k: None for k in COCO}
        out = {}
        for i, name in enumerate(COCO):
            x, y, s = kps[i]
            out[name] = (float(x) / w, float(y) / h, float(s))
        return out

    if lib == "MediaPipePose":
        # keypoints_xy_vis: [ [x_norm, y_norm, vis] * 33 ].
        # Map a subset to COCO indices using MediaPipe landmark indices.
        kps = frame.get("keypoints_xy_vis")
        if not kps:
            return {k: None for k in COCO}

        mp = kps
        mp_idx = {
            "nose": 0,
            "left_eye": 2,
            "right_eye": 5,
            "left_ear": 7,
            "right_ear": 8,
            "left_shoulder": 11,
            "right_shoulder": 12,
            "left_elbow": 13,
            "right_elbow": 14,
            "left_wrist": 15,
            "right_wrist": 16,
            "left_hip": 23,
            "right_hip": 24,
            "left_knee": 25,
            "right_knee": 26,
            "left_ankle": 27,
            "right_ankle": 28,
        }
        out = {k: None for k in COCO}
        for name, idx in mp_idx.items():
            x, y, v = mp[idx]
            out[name] = (float(x), float(y), float(v))
        return out

    raise ValueError(f"Unknown library: {lib}")


def _jitter_xy(frames_xy: list[dict[str, tuple[float, float, float] | None]], key: str, min_score: float) -> float:
    pts = []
    for fr in frames_xy:
        v = fr.get(key)
        if not v:
            continue
        x, y, s = v
        if s < min_score:
            continue
        pts.append((x, y))
    if len(pts) < 3:
        return float("nan")
    pts = np.asarray(pts, dtype=float)
    d = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    return float(np.mean(d))


def _elbow_angle_series(
    frames_xy: list[dict[str, tuple[float, float, float] | None]],
    side: str,
    min_score: float,
) -> np.ndarray:
    sh = f"{side}_shoulder"
    el = f"{side}_elbow"
    wr = f"{side}_wrist"
    angles = []
    for fr in frames_xy:
        a = fr.get(sh)
        b = fr.get(el)
        c = fr.get(wr)
        if not a or not b or not c:
            continue
        if a[2] < min_score or b[2] < min_score or c[2] < min_score:
            continue
        angles.append(
            elbow_flexion_deg(
                Vec3(a[0], a[1], 0.0),
                Vec3(b[0], b[1], 0.0),
                Vec3(c[0], c[1], 0.0),
            )
        )
    return np.asarray(angles, dtype=float)


def evaluate(res: dict, min_score: float, static_first_n: int) -> dict:
    pf = sorted(res["per_frame"], key=lambda x: x["frame_index"])
    frames_xy = [_norm_xy_from_result(res, fr) for fr in pf]

    # Practical metrics
    left_elbow_jitter = _jitter_xy(frames_xy, "left_elbow", min_score)
    right_elbow_jitter = _jitter_xy(frames_xy, "right_elbow", min_score)

    # Static elbow-angle stability proxy: std dev on first N valid angle samples.
    left_angles = _elbow_angle_series(frames_xy, "left", min_score)
    right_angles = _elbow_angle_series(frames_xy, "right", min_score)

    left_static = left_angles[:static_first_n] if left_angles.size else left_angles
    right_static = right_angles[:static_first_n] if right_angles.size else right_angles

    # Robustness proxy during motion: fraction of frames with confident wrist+elbow+shoulder.
    def robust_fraction(side: str) -> float:
        ok = 0
        total = 0
        for fr in frames_xy:
            total += 1
            a = fr.get(f"{side}_shoulder")
            b = fr.get(f"{side}_elbow")
            c = fr.get(f"{side}_wrist")
            if a and b and c and a[2] >= min_score and b[2] >= min_score and c[2] >= min_score:
                ok += 1
        return ok / total if total else float("nan")

    n_frames = int(res.get("n_frames", len(pf)))
    wall_s   = float(res.get("wall_elapsed_s", 0.0))

    # Use metrics.py for angle-domain jitter and static stability
    left_angle_jitter  = compute_jitter(left_angles.tolist())
    right_angle_jitter = compute_jitter(right_angles.tolist())
    left_static_std    = compute_static_pose_stability(left_static.tolist())
    right_static_std   = compute_static_pose_stability(right_static.tolist())
    fps_verified       = compute_fps(n_frames, wall_s) if wall_s > 0 else float(res.get("fps", 0.0))

    out = {
        "library": res.get("library"),
        "framework": res.get("framework"),
        "fps": fps_verified,
        "latency_ms_mean": float(res.get("latency_ms", {}).get("mean", float("nan"))),
        "avg_keypoint_score_mean": float(res.get("avg_keypoint_score", {}).get("mean", float("nan"))),
        "jitter": {
            "left_elbow_xy_mean_step":    left_elbow_jitter,
            "right_elbow_xy_mean_step":   right_elbow_jitter,
            "left_elbow_angle_mean_step":  left_angle_jitter,
            "right_elbow_angle_mean_step": right_angle_jitter,
        },
        "elbow_angle": {
            "left_deg_mean":       float(np.nanmean(left_angles))  if left_angles.size  else float("nan"),
            "left_deg_std_static": left_static_std,
            "right_deg_mean":      float(np.nanmean(right_angles)) if right_angles.size else float("nan"),
            "right_deg_std_static": right_static_std,
        },
        "robustness": {
            "left_arm_confident_fraction":  robust_fraction("left"),
            "right_arm_confident_fraction": robust_fraction("right"),
        },
        "n_frames": n_frames,
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="+", required=True, help="One or more results JSON files")
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--min_score", type=float, default=0.3)
    ap.add_argument("--static_first_n", type=int, default=60)
    args = ap.parse_args()

    evals = []
    for p in args.results:
        res = _load(Path(p))
        evals.append(evaluate(res, min_score=args.min_score, static_first_n=args.static_first_n))

    out = {
        "min_score": args.min_score,
        "static_first_n": args.static_first_n,
        "evaluations": evals,
    }
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

