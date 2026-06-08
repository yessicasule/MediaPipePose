"""
run_experiments.py — Phase 10: Full Ablation Study.

Runs 5 model configurations, evaluates each, and produces a Markdown comparison table.

Ablation configs:
    full         – Transformer + CrossAttn + BiLSTM + Bayesian + GAN
    no_transform – BiLSTM only (no Transformer or CrossAttn)
    no_fusion    – Single-framework BiLSTM (MediaPipe only as proxy)
    no_bayesian  – Fusion without MC Dropout uncertainty
    no_gan       – Fusion trained normally, no GAN fine-tuning
"""

import sys, json, argparse, time
import numpy as np
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.fusion_network import (
    DeepFusionPoseModel, FusionDataset, train_fusion_model,
    evaluate_fusion_model, mc_predict
)
from src.models.baseline_models import (
    BiLSTMPoseModel, JointAngleDataset, train_baseline_model, evaluate_baseline_model
)
from src.models.gan_refinery import TemporalDiscriminator, train_gan, evaluate_gan
from src.evaluation.metrics import compute_jitter


# ────────────────────────────────────────────
# Ablation variant: No-Transformer (BiLSTM-only fusion)
# ────────────────────────────────────────────

class NoTransformerFusion(nn.Module):
    """Fusion model without Transformer/CrossAttention — just embedding + BiLSTM."""
    def __init__(self, input_dim=12, hidden=128, layers=2, output_dim=4, dropout=0.2):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU())
        self.bilstm = nn.LSTM(64, hidden, layers, batch_first=True,
                              bidirectional=True, dropout=dropout if layers > 1 else 0)
        self.head = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(hidden * 2, 128), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(128, output_dim)
        )
    def forward(self, x):
        return self.head(self.bilstm(self.embed(x))[0])


def _timed_train(model, train_loader, val_loader, epochs, lr, device, ckpt):
    t0 = time.time()
    train_fusion_model(model, train_loader, val_loader, epochs=epochs, lr=lr,
                       device=device, checkpoint_path=ckpt)
    return time.time() - t0


def run_ablation(
    csv_path: str,
    scaler_path: str,
    output_dir: str,
    seq_len: int = 60,
    batch: int = 32,
    epochs: int = 50,
    lr: float = 1e-4,
    device: str = "cpu",
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(scaler_path) as f:
        scaler = json.load(f)

    full_ds = FusionDataset(csv_path, sequence_length=seq_len, scaler_stats=scaler)
    n_train = int(0.8 * len(full_ds))
    train_ds, val_ds = torch.utils.data.random_split(
        full_ds, [n_train, len(full_ds) - n_train],
        generator=torch.Generator().manual_seed(42)
    )
    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch, shuffle=False, num_workers=0)
    eval_loader  = DataLoader(full_ds,  batch_size=batch, shuffle=False, num_workers=0)

    ablation_results = {}

    # ── 1. FULL model ──────────────────────────────────────────
    print("\n=== CONFIG: full ===")
    model_full = DeepFusionPoseModel()
    ckpt_full = str(output_dir / "ablation_full.pt")
    _timed_train(model_full, train_loader, val_loader, epochs, lr, device, ckpt_full)
    model_full.load_state_dict(torch.load(ckpt_full, map_location=device))
    # GAN refine
    disc = TemporalDiscriminator()
    train_gan(model_full, disc, train_loader, val_loader,
              pretrain_epochs=5, gan_epochs=20, device=device,
              checkpoint_path=str(output_dir / "ablation_full_gan.pt"))
    model_full.load_state_dict(torch.load(str(output_dir / "ablation_full_gan.pt"), map_location=device))
    ablation_results["full"] = evaluate_fusion_model(model_full, eval_loader, device, n_mc=30)

    # ── 2. No-Transformer ──────────────────────────────────────
    print("\n=== CONFIG: no_transformer ===")
    model_notrans = NoTransformerFusion()
    ckpt_notrans = str(output_dir / "ablation_no_transformer.pt")
    _timed_train(model_notrans, train_loader, val_loader, epochs, lr, device, ckpt_notrans)
    model_notrans.load_state_dict(torch.load(ckpt_notrans, map_location=device))
    # Evaluate with MC Dropout style (model already has dropout)
    ablation_results["no_transformer"] = evaluate_fusion_model(model_notrans, eval_loader, device, n_mc=30)

    # ── 3. No-Fusion (MediaPipe BiLSTM only, 4 features) ───────
    print("\n=== CONFIG: no_fusion ===")
    mp_ds = JointAngleDataset(csv_path, "mp", sequence_length=seq_len)
    mp_n  = int(0.8 * len(mp_ds))
    mp_train, mp_val = torch.utils.data.random_split(
        mp_ds, [mp_n, len(mp_ds) - mp_n], generator=torch.Generator().manual_seed(42))
    mp_train_l = DataLoader(mp_train, batch_size=batch, shuffle=True,  num_workers=0)
    mp_val_l   = DataLoader(mp_val,   batch_size=batch, shuffle=False, num_workers=0)
    model_nofuse = BiLSTMPoseModel()
    train_baseline_model(model_nofuse, mp_train_l, mp_val_l, epochs=epochs, lr=lr, device=device)
    mp_eval_l = DataLoader(mp_ds, batch_size=batch, shuffle=False, num_workers=0)
    ablation_results["no_fusion"] = evaluate_baseline_model(model_nofuse, mp_eval_l, device)

    # ── 4. No-Bayesian (full fusion, no MC, deterministic inference) ─
    print("\n=== CONFIG: no_bayesian ===")
    model_nobay = DeepFusionPoseModel()
    ckpt_nobay = str(output_dir / "ablation_no_bayesian.pt")
    _timed_train(model_nobay, train_loader, val_loader, epochs, lr, device, ckpt_nobay)
    model_nobay.load_state_dict(torch.load(ckpt_nobay, map_location=device))
    # Deterministic eval (model.eval() — no MC)
    model_nobay.eval()
    all_preds, all_tgts = [], []
    with torch.no_grad():
        for xb, yb in eval_loader:
            all_preds.append(model_nobay(xb.to(device)).cpu().numpy())
            all_tgts.append(yb.numpy())
    preds_f   = np.concatenate(all_preds).reshape(-1, 4)
    targets_f = np.concatenate(all_tgts).reshape(-1, 4)
    mae  = float(np.mean(np.abs(preds_f - targets_f)))
    rmse = float(np.sqrt(np.mean((preds_f - targets_f) ** 2)))
    jitter = float(np.mean([compute_jitter(preds_f[:, j].tolist()) for j in range(4)]))
    ablation_results["no_bayesian"] = {"average": {"MAE": mae, "RMSE": rmse, "Jitter": jitter, "Uncertainty": 0.0}}

    # ── 5. No-GAN (full fusion, NO adversarial fine-tuning) ────
    print("\n=== CONFIG: no_gan ===")
    model_nogan = DeepFusionPoseModel()
    ckpt_nogan = str(output_dir / "ablation_no_gan.pt")
    _timed_train(model_nogan, train_loader, val_loader, epochs, lr, device, ckpt_nogan)
    model_nogan.load_state_dict(torch.load(ckpt_nogan, map_location=device))
    ablation_results["no_gan"] = evaluate_fusion_model(model_nogan, eval_loader, device, n_mc=30)

    # ── Save & print table ─────────────────────────────────────
    results_path = output_dir / "ablation_evaluation.json"
    with open(results_path, "w") as f:
        json.dump(ablation_results, f, indent=2)

    _print_markdown_table(ablation_results, output_dir)
    return ablation_results


def _print_markdown_table(results: dict, output_dir: Path):
    header = "| Config         | MAE ↓  | RMSE ↓ | Jitter ↓ | Uncertainty |"
    sep    = "|----------------|--------|--------|----------|-------------|"
    rows   = [header, sep]
    for cfg, data in results.items():
        avg = data.get("average", {})
        mae  = avg.get("MAE",         0)
        rmse = avg.get("RMSE",        0)
        jit  = avg.get("Jitter",      0)
        unc  = avg.get("Uncertainty", 0)
        rows.append(f"| {cfg:<14} | {mae:6.3f} | {rmse:6.3f} | {jit:8.3f} | {unc:11.3f} |")

    table = "\n".join(rows)
    md_path = output_dir / "ablation_results.md"
    md_path.write_text("# DeepFusionPose — Ablation Study Results\n\n" + table + "\n", encoding="utf-8")
    print("\n" + table)
    print(f"\nMarkdown table saved → {md_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv",        required=True)
    ap.add_argument("--scaler",     required=True)
    ap.add_argument("--output_dir", default="outputs/ablation")
    ap.add_argument("--epochs",     type=int,   default=50)
    ap.add_argument("--seq_len",    type=int,   default=60)
    ap.add_argument("--batch",      type=int,   default=32)
    ap.add_argument("--lr",         type=float, default=1e-4)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    run_ablation(args.csv, args.scaler, args.output_dir,
                 args.seq_len, args.batch, args.epochs, args.lr, device)


if __name__ == "__main__":
    main()
