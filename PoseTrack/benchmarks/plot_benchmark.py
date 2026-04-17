import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _series(res: dict) -> tuple[np.ndarray, np.ndarray]:
    # returns (latency_ms, avg_kp_score) aligned by frame_index
    pf = sorted(res["per_frame"], key=lambda x: x["frame_index"])
    lat = np.array([x["inference_ms"] for x in pf], dtype=float)
    sc = np.array([x["avg_keypoint_score"] for x in pf], dtype=float)
    return lat, sc


def _save(fig, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--movenet_json", required=True)
    ap.add_argument("--posenet_json", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    mov = _load(Path(args.movenet_json))
    pos = _load(Path(args.posenet_json))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mov_lat, mov_sc = _series(mov)
    pos_lat, pos_sc = _series(pos)

    n = min(len(mov_lat), len(pos_lat))
    mov_lat, pos_lat = mov_lat[:n], pos_lat[:n]
    mov_sc, pos_sc = mov_sc[:n], pos_sc[:n]
    x = np.arange(n)

    # latency per frame
    fig = plt.figure(figsize=(11, 4.2))
    plt.plot(x, mov_lat, label=f"MoveNet ({mov.get('model', '')})", linewidth=1.2)
    plt.plot(x, pos_lat, label=f"PoseNet ({pos.get('architecture', '')})", linewidth=1.2)
    plt.title("Per-frame inference latency (ms)")
    plt.xlabel("Frame index")
    plt.ylabel("Latency (ms)")
    plt.legend()
    plt.grid(True, alpha=0.25)
    _save(fig, out_dir / "latency_ms.png")

    # avg keypoint score per frame
    fig = plt.figure(figsize=(11, 4.2))
    plt.plot(x, mov_sc, label="MoveNet avg keypoint score", linewidth=1.2)
    plt.plot(x, pos_sc, label="PoseNet avg keypoint score", linewidth=1.2)
    plt.title("Per-frame average keypoint confidence score")
    plt.xlabel("Frame index")
    plt.ylabel("Avg score")
    plt.legend()
    plt.grid(True, alpha=0.25)
    _save(fig, out_dir / "avg_keypoint_score.png")

    # overall FPS bar chart
    fig = plt.figure(figsize=(7.5, 4.2))
    names = ["MoveNet", "PoseNet"]
    fps = [float(mov.get("fps", 0.0)), float(pos.get("fps", 0.0))]
    plt.bar(names, fps, color=["#1f77b4", "#ff7f0e"])
    plt.title("Overall throughput (FPS)")
    plt.ylabel("FPS")
    for i, v in enumerate(fps):
        plt.text(i, v, f"{v:.2f}", ha="center", va="bottom")
    plt.grid(True, axis="y", alpha=0.25)
    _save(fig, out_dir / "fps.png")

    summary = {
        "aligned_frames_plotted": int(n),
        "movenet": {
            "fps": float(mov.get("fps", 0.0)),
            "latency_ms_mean": float(mov.get("latency_ms", {}).get("mean", float(np.mean(mov_lat)))),
            "avg_keypoint_score_mean": float(mov.get("avg_keypoint_score", {}).get("mean", float(np.mean(mov_sc)))),
        },
        "posenet": {
            "fps": float(pos.get("fps", 0.0)),
            "latency_ms_mean": float(pos.get("latency_ms", {}).get("mean", float(np.mean(pos_lat)))),
            "avg_keypoint_score_mean": float(pos.get("avg_keypoint_score", {}).get("mean", float(np.mean(pos_sc)))),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

