"""
gan_refinery.py — Phase 6: GAN Refinement of Fusion Network output.

Generator:    The trained DeepFusionPoseModel (loaded from checkpoint), fine-tuned adversarially.
Discriminator: 1D-Conv network classifying angle sequences as Real vs Fake.

Loss:
    G_total = λ1·MSE(G(x), gt)  +  λ2·-log(D(G(x)))  +  λ3·temporal_smoothness(G(x))
    D_total = BCE(D(gt), 1)     +  BCE(D(G(x).detach()), 0)
"""

import sys
import json
import argparse
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from config.config import Config
from src.models.fusion_network import DeepFusionPoseModel, FusionDataset, mc_predict
from src.models.trainer import GANTrainer
from src.evaluation.metrics import evaluate_predictions, metrics_to_model_dict

# ────────────────────────────────────────────
# Discriminator
# ────────────────────────────────────────────

class TemporalDiscriminator(nn.Module):
    """
    1D-Conv discriminator on angle sequences.
    Input:  [B, L, 4]  (joint angles over time)
    Output: [B, 1]     (real/fake probability)
    """
    def __init__(self, input_dim: int = 4, seq_len: int = 60):
        super().__init__()
        self.conv = nn.Sequential(
            # [B, 4, L] → [B, 64, L]
            nn.Conv1d(input_dim, 64,  kernel_size=3, padding=1), nn.LeakyReLU(0.2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1), nn.LeakyReLU(0.2),
            nn.Conv1d(128, 256, kernel_size=3, padding=1), nn.LeakyReLU(0.2),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)   # [B, 256, 1]
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 64), nn.LeakyReLU(0.2),
            nn.Linear(64, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        # x: [B, L, 4] → permute to [B, 4, L] for Conv1d
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = self.pool(x)
        return self.classifier(x)


# ────────────────────────────────────────────
# Evaluation helper
# ────────────────────────────────────────────

def evaluate_gan(generator: nn.Module, loader: DataLoader, device: str = "cpu") -> dict:
    generator.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            all_preds.append(generator(xb).cpu().numpy())
            all_targets.append(yb.numpy())

    preds   = np.concatenate(all_preds,   axis=0) # [N, L, 4]
    targets = np.concatenate(all_targets, axis=0) # [N, L, 4]

    fm = evaluate_predictions("GANRefinery", preds, targets, joint_names=['shoulder_pitch', 'shoulder_roll', 'shoulder_yaw', 'elbow_flexion'])
    return metrics_to_model_dict(fm)


# ────────────────────────────────────────────
# Entry point
# ────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv",         required=True)
    ap.add_argument("--fusion_ckpt", required=True, help="Path to fusion_best.pt")
    ap.add_argument("--scaler",      required=True, help="Path to fusion_scaler.json")
    ap.add_argument("--output_dir",  default="outputs/models")
    ap.add_argument("--seq_len",     type=int,   default=60)
    ap.add_argument("--batch",       type=int,   default=32)
    ap.add_argument("--pretrain",    type=int,   default=20)
    ap.add_argument("--gan_epochs",  type=int,   default=60)
    args = ap.parse_args()

    Config.ensure_directories()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    with open(args.scaler) as f:
        scaler_stats = json.load(f)

    full_ds = FusionDataset(args.csv, sequence_length=args.seq_len, scaler_stats=scaler_stats)
    n_train = int(0.8 * len(full_ds))
    train_ds, val_ds = torch.utils.data.random_split(full_ds, [n_train, len(full_ds) - n_train])
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, num_workers=0)
    eval_loader  = DataLoader(full_ds,  batch_size=args.batch, shuffle=False, num_workers=0)

    # Load pre-trained fusion generator
    generator = DeepFusionPoseModel()
    generator.load_state_dict(torch.load(args.fusion_ckpt, map_location=device))

    discriminator = TemporalDiscriminator(input_dim=4, seq_len=args.seq_len)

    output_dir = Path(args.output_dir)
    ckpt = str(output_dir / "gan_generator_best.pt")

    trainer = GANTrainer(
        generator=generator,
        discriminator=discriminator,
        device=device,
        checkpoint_path=ckpt,
        patience=15,
        clip_grad=1.0,
        lambda_recon=10.0,
        lambda_adv=1.0,
        lambda_smooth=5.0
    )

    opt_g = optim.AdamW(generator.parameters(),     lr=1e-4, betas=(0.5, 0.999))
    opt_d = optim.AdamW(discriminator.parameters(), lr=4e-5, betas=(0.5, 0.999))
    
    mse_criterion = nn.MSELoss()
    bce_criterion = nn.BCELoss()

    trainer.fit_gan(
        train_loader=train_loader,
        val_loader=val_loader,
        pretrain_epochs=args.pretrain,
        gan_epochs=args.gan_epochs,
        opt_g=opt_g,
        opt_d=opt_d,
        mse_criterion=mse_criterion,
        bce_criterion=bce_criterion
    )

    generator.load_state_dict(torch.load(ckpt, map_location=device))
    metrics = evaluate_gan(generator, eval_loader, device=device)

    eval_path = output_dir / "gan_evaluation.json"
    with open(eval_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print("\n=== GAN EVALUATION ===")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
