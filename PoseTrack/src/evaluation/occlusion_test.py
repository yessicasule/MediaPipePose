"""
occlusion_test.py — Phase 7: Artificial occlusion benchmarking.

Simulates tracker dropout by zeroing out one framework's angles for k% of frames.
Tests all 5 models (3 baselines + fusion + GAN) under 0/25/50/75% occlusion.
"""

import sys, json, argparse
import numpy as np
import pandas as pd
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import torch
from torch.utils.data import Dataset, DataLoader

from src.models.fusion_network import DeepFusionPoseModel, FusionDataset
from src.models.baseline_models import BiLSTMPoseModel, JointAngleDataset
from src.evaluation.metrics import compute_jitter

FRAMEWORK_SLICES = {
    "mediapipe": (0, 4),
    "movenet":   (4, 8),
    "posenet":   (8, 12),
}


class OccludedFusionDataset(Dataset):
    """
    Wraps FusionDataset and randomly zeros one framework's angles
    in k% of frames within each sequence.
    occlude_framework: 'mediapipe' | 'movenet' | 'posenet'
    occlusion_rate: 0.0 – 1.0
    """
    def __init__(self, base_dataset: FusionDataset, occlude_framework: str,
                 occlusion_rate: float, seed: int = 42):
        self.base  = base_dataset
        self.start, self.end = FRAMEWORK_SLICES[occlude_framework]
        self.rate  = occlusion_rate
        self.rng   = np.random.default_rng(seed)

    def __len__(self): return len(self.base)

    def __getitem__(self, idx):
        x, y = self.base[idx]
        x = x.clone()
        L = x.shape[0]
        n_occlude = int(L * self.rate)
        if n_occlude > 0:
            frames_to_zero = self.rng.choice(L, n_occlude, replace=False)
            x[frames_to_zero, self.start:self.end] = 0.0
        return x, y


def _eval_model(model, loader, device):
    """Returns (MAE, RMSE, Jitter) averaged over all joints."""
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            out = model(xb)
            all_preds.append(out.cpu().numpy())
            all_targets.append(yb.numpy())
    preds   = np.concatenate(all_preds).reshape(-1, 4)
    targets = np.concatenate(all_targets).reshape(-1, 4)
    mae  = float(np.mean(np.abs(preds - targets)))
    rmse = float(np.sqrt(np.mean((preds - targets) ** 2)))
    jitter = float(np.mean([compute_jitter(preds[:, j].tolist()) for j in range(4)]))
    return mae, rmse, jitter


def run_occlusion_benchmark(
    csv_path: str,
    scaler_path: str,
    fusion_ckpt: str,
    gan_ckpt: str,
    baseline_ckpts: dict,    # {"mp": "path.pt", "mv": ..., "pn": ...}
    output_json: str,
    seq_len: int = 60,
    batch: int  = 32,
    device: str = "cpu",
) -> dict:
    with open(scaler_path) as f:
        scaler = json.load(f)

    base_ds = FusionDataset(csv_path, sequence_length=seq_len, scaler_stats=scaler)
    occlusion_levels = [0.0, 0.25, 0.50, 0.75]
    frameworks = ["mediapipe", "movenet", "posenet"]

    # Load all models
    fusion_model = DeepFusionPoseModel().to(device)
    fusion_model.load_state_dict(torch.load(fusion_ckpt, map_location=device))

    gan_model = DeepFusionPoseModel().to(device)
    gan_model.load_state_dict(torch.load(gan_ckpt, map_location=device))

    baseline_models = {}
    for fw, ckpt in baseline_ckpts.items():
        m = BiLSTMPoseModel().to(device)
        m.load_state_dict(torch.load(ckpt, map_location=device))
        baseline_models[fw] = m

    results = []
    for fw in frameworks:
        for rate in occlusion_levels:
            occ_ds = OccludedFusionDataset(base_ds, fw, rate)
            loader = DataLoader(occ_ds, batch_size=batch, shuffle=False, num_workers=0)

            # Fusion
            f_mae, f_rmse, f_jit = _eval_model(fusion_model, loader, device)
            # GAN
            g_mae, g_rmse, g_jit = _eval_model(gan_model, loader, device)

            row = {
                "occluded_framework": fw,
                "occlusion_pct": int(rate * 100),
                "fusion_MAE":   f_mae,  "fusion_RMSE":   f_rmse,  "fusion_Jitter":   f_jit,
                "gan_MAE":      g_mae,  "gan_RMSE":      g_rmse,  "gan_Jitter":      g_jit,
            }

            # Baselines — only their own framework's loader matters
            for fw_b, bm in baseline_models.items():
                # Baseline takes only 4 features (its own framework slice)
                class _SliceDS(Dataset):
                    def __init__(self, parent, sl):
                        self.parent = parent
                        self.sl = sl
                    def __len__(self): return len(self.parent)
                    def __getitem__(self, i):
                        x, y = self.parent[i]
                        return x[:, sl[0]:sl[1]], y
                sl = FRAMEWORK_SLICES[fw_b]
                bl_loader = DataLoader(_SliceDS(occ_ds, sl), batch_size=batch, shuffle=False, num_workers=0)
                b_mae, b_rmse, b_jit = _eval_model(bm, bl_loader, device)
                row[f"{fw_b}_MAE"]    = b_mae
                row[f"{fw_b}_RMSE"]   = b_rmse
                row[f"{fw_b}_Jitter"] = b_jit

            results.append(row)
            print(f"  [{fw} @ {int(rate*100)}%] Fusion MAE={f_mae:.3f}, GAN MAE={g_mae:.3f}")

    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Occlusion results saved → {output_json}")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv",         required=True)
    ap.add_argument("--scaler",      required=True)
    ap.add_argument("--fusion_ckpt", required=True)
    ap.add_argument("--gan_ckpt",    required=True)
    ap.add_argument("--mp_ckpt",     required=True)
    ap.add_argument("--mv_ckpt",     required=True)
    ap.add_argument("--pn_ckpt",     required=True)
    ap.add_argument("--output_json", default="outputs/occlusion_results.json")
    ap.add_argument("--seq_len",     type=int, default=60)
    ap.add_argument("--batch",       type=int, default=32)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_occlusion_benchmark(
        csv_path=args.csv, scaler_path=args.scaler,
        fusion_ckpt=args.fusion_ckpt, gan_ckpt=args.gan_ckpt,
        baseline_ckpts={"mp": args.mp_ckpt, "mv": args.mv_ckpt, "pn": args.pn_ckpt},
        output_json=args.output_json,
        seq_len=args.seq_len, batch=args.batch, device=device,
    )


if __name__ == "__main__":
    main()
