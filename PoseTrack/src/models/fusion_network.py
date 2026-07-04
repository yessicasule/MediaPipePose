"""
fusion_network.py — Phase 4 (DeepFusionPose) + Phase 5 (Bayesian MC Dropout)

Architecture:
    Input (12) → Feature Embedding (64) + Positional Encoding
               → Transformer Encoder (2 blocks, 8 heads)
               → Cross-Attention (query=transformer out, key/value=framework sub-vectors)
               → BiLSTM (hidden=128, 2 layers, bidirectional)
               → Dense Head (256 → 128 → 4)
               → Predicted joint angles [B, L, 4]

Bayesian: MC Dropout at inference — run N=50 stochastic forward passes,
          return (mean_angles, std_uncertainty).
"""

import sys
import json
import math
import copy
import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from config.config import Config
from src.models.datasets import BaseSequenceDataset
from src.models.trainer import BaseTrainer
from src.evaluation.metrics import evaluate_predictions, metrics_to_model_dict

# ─────────────────────────────────────────────
# 1.  Dataset
# ─────────────────────────────────────────────

INPUT_COLS = [
    "mp_shoulder_pitch", "mp_shoulder_roll", "mp_shoulder_yaw", "mp_elbow_flexion",
    "mv_shoulder_pitch", "mv_shoulder_roll", "mv_shoulder_yaw", "mv_elbow_flexion",
    "pn_shoulder_pitch", "pn_shoulder_roll", "pn_shoulder_yaw", "pn_elbow_flexion",
]
TARGET_COLS = [
    "gt_shoulder_pitch", "gt_shoulder_roll", "gt_shoulder_yaw", "gt_elbow_flexion"
]


class FusionDataset(BaseSequenceDataset):
    """
    Loads the unified 17-column CSV and creates sliding-window sequences.
    input_dim=12, output_dim=4.
    Normalises inputs to [-1, 1] using training-split statistics.
    """

    def __init__(
        self,
        csv_path: str,
        sequence_length: int = 60,
        step_size: int = 5,
        scaler_stats: dict = None,   # pass None to fit; pass dict to apply
    ):
        super().__init__(
            csv_path=csv_path,
            feature_cols=INPUT_COLS,
            target_cols=TARGET_COLS,
            sequence_length=sequence_length,
            step_size=step_size,
            scaler_stats=scaler_stats,
            normalize=True
        )


# ─────────────────────────────────────────────
# 2.  Positional Encoding
# ─────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        pe = pe.unsqueeze(0)          # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: [B, L, d_model]
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ─────────────────────────────────────────────
# 3.  Transformer Encoder Block
# ─────────────────────────────────────────────

class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model: int = 64, n_heads: int = 8, d_ff: int = 256, dropout: float = 0.1):
        super().__init__()
        self.self_attn  = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ff         = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x):
        # Self-attention with residual
        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm1(x + self.drop(attn_out))
        # Feed-forward with residual
        x = self.norm2(x + self.drop(self.ff(x)))
        return x


# ─────────────────────────────────────────────
# 4.  Cross Attention (framework-selective)
# ─────────────────────────────────────────────

class CrossAttentionFusion(nn.Module):
    """
    Query = Transformer output (fused representation).
    Keys/Values = per-framework sub-vectors (3 × [B, L, 4] → projected to [B, L, d_model]).
    """
    def __init__(self, d_model: int = 64, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.proj_mp = nn.Linear(4, d_model)
        self.proj_mv = nn.Linear(4, d_model)
        self.proj_pn = nn.Linear(4, d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, query, raw_12):
        # raw_12: [B, L, 12]  — split into 3 × [B, L, 4]
        mp = self.proj_mp(raw_12[:, :, 0:4])
        mv = self.proj_mv(raw_12[:, :, 4:8])
        pn = self.proj_pn(raw_12[:, :, 8:12])
        # Concat along sequence as key/value: [B, 3*L, d_model]
        kv = torch.cat([mp, mv, pn], dim=1)
        attn_out, _ = self.cross_attn(query, kv, kv)
        return self.norm(query + self.drop(attn_out))


# ─────────────────────────────────────────────
# 5.  Full DeepFusionPose Model
# ─────────────────────────────────────────────

class DeepFusionPoseModel(nn.Module):
    """
    Phase 4 + 5 model. Dropout layers are active at inference for MC Dropout.
    """

    def __init__(
        self,
        input_dim: int = 12,
        d_model: int = 64,
        n_heads: int = 8,
        n_transformer_blocks: int = 2,
        d_ff: int = 256,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        output_dim: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()

        # Feature embedding
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.ReLU(),
        )
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)

        # Transformer Encoder
        self.transformer = nn.Sequential(
            *[TransformerEncoderBlock(d_model, n_heads, d_ff, dropout)
              for _ in range(n_transformer_blocks)]
        )

        # Cross Attention
        self.cross_attn = CrossAttentionFusion(d_model, n_heads, dropout)

        # BiLSTM
        self.bilstm = nn.LSTM(
            input_size=d_model,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        # Dense regression head
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, output_dim),
        )

    def forward(self, x):
        # x: [B, L, 12]
        raw_12 = x.clone()
        emb = self.pos_enc(self.embedding(x))       # [B, L, 64]
        enc = self.transformer(emb)                  # [B, L, 64]
        fused = self.cross_attn(enc, raw_12)         # [B, L, 64]
        lstm_out, _ = self.bilstm(fused)             # [B, L, 256]
        out = self.head(lstm_out)                    # [B, L, 4]
        return out


# ─────────────────────────────────────────────
# 6.  MC Dropout inference (Phase 5)
# ─────────────────────────────────────────────

def mc_predict(model: nn.Module, x: torch.Tensor, n_samples: int = 50) -> tuple:
    """
    Runs N stochastic forward passes with Dropout active.
    Returns (mean, std) both shape [B, L, 4].
    """
    model.train()   # keep Dropout active
    preds = []
    with torch.no_grad():
        for _ in range(n_samples):
            preds.append(model(x).unsqueeze(0))     # [1, B, L, 4]
    stack = torch.cat(preds, dim=0)                 # [N, B, L, 4]
    mean = stack.mean(dim=0)                        # [B, L, 4]
    std  = stack.std(dim=0)                         # [B, L, 4]
    return mean, std


def confidence_from_std(std: torch.Tensor) -> torch.Tensor:
    """Maps std → confidence in [0,1]:  c = 1 / (1 + std)."""
    return 1.0 / (1.0 + std)


# ─────────────────────────────────────────────
# 7.  Evaluation helper
# ─────────────────────────────────────────────

def evaluate_fusion_model(model: nn.Module, loader: DataLoader, device: str = "cpu",
                           n_mc: int = 50) -> dict:
    """Returns per-joint MAE, RMSE, Jitter + mean uncertainty."""
    all_means, all_stds, all_targets = [], [], []
    for xb, yb in loader:
        xb = xb.to(device)
        mean, std = mc_predict(model, xb, n_mc)
        all_means.append(mean.cpu().numpy())
        all_stds.append(std.cpu().numpy())
        all_targets.append(yb.numpy())

    means   = np.concatenate(all_means, axis=0) # [N, L, 4]
    stds    = np.concatenate(all_stds, axis=0)  # [N, L, 4]
    targets = np.concatenate(all_targets, axis=0) # [N, L, 4]

    fm = evaluate_predictions("FusionModel", means, targets, joint_names=['shoulder_pitch', 'shoulder_roll', 'shoulder_yaw', 'elbow_flexion'])
    metrics = metrics_to_model_dict(fm)

    # Add Uncertainty
    stds_flat = stds.reshape(-1, 4)
    mean_uncertainty = np.mean(stds_flat, axis=0)
    
    joint_names = ["shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow_flexion"]
    for i, name in enumerate(joint_names):
        metrics[name]["Uncertainty"] = float(mean_uncertainty[i])
    metrics["average"]["Uncertainty"] = float(np.mean(mean_uncertainty))
    
    return metrics


# ─────────────────────────────────────────────
# 8.  Entry point
# ─────────────────────────────────────────────

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Train DeepFusionPose (Phase 4+5)")
    ap.add_argument("--csv",        required=True, help="Path to unified training CSV")
    ap.add_argument("--output_dir", default="outputs/models")
    ap.add_argument("--seq_len",    type=int, default=60)
    ap.add_argument("--epochs",     type=int, default=80)
    ap.add_argument("--batch",      type=int, default=32)
    ap.add_argument("--lr",         type=float, default=1e-4)
    ap.add_argument("--n_mc",       type=int, default=50, help="MC Dropout samples at inference")
    args = ap.parse_args()

    # Load custom config if specified
    Config.ensure_directories()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ── Dataset ──────────────────────────────────────
    full_dataset = FusionDataset(args.csv, sequence_length=args.seq_len)

    # Save scaler for inference
    scaler_path = output_dir / "fusion_scaler.json"
    with open(scaler_path, "w") as f:
        json.dump(full_dataset.scaler_dict(), f, indent=2)
    print(f"Scaler saved → {scaler_path}")

    n_total = len(full_dataset)
    n_train = int(0.8 * n_total)
    n_val   = n_total - n_train
    train_ds, val_ds = torch.utils.data.random_split(full_dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, num_workers=0)

    # ── Model ────────────────────────────────────────
    model = DeepFusionPoseModel(dropout=0.2)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,}")

    # ── Train ────────────────────────────────────────
    ckpt = str(output_dir / "fusion_best.pt")
    
    trainer = BaseTrainer(
        model=model,
        device=device,
        checkpoint_path=ckpt,
        patience=15,
        clip_grad=1.0
    )
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.MSELoss()

    print(f"Beginning DeepFusionPose training using BaseTrainer...")
    _ = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler
    )

    # ── Evaluate (load best) ─────────────────────────
    model.load_state_dict(torch.load(ckpt, map_location=device))
    eval_loader = DataLoader(full_dataset, batch_size=args.batch, shuffle=False, num_workers=0)
    metrics = evaluate_fusion_model(model, eval_loader, device=device, n_mc=args.n_mc)

    eval_path = output_dir / "fusion_evaluation.json"
    with open(eval_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n=== FUSION MODEL EVALUATION ===")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
