"""
gan_refinery.py — Phase 6: GAN Refinement of Fusion Network output.

Generator:    The trained DeepFusionPoseModel (loaded from checkpoint), fine-tuned adversarially.
Discriminator: 1D-Conv network classifying angle sequences as Real vs Fake.

Loss:
    G_total = λ1·MSE(G(x), gt)  +  λ2·-log(D(G(x)))  +  λ3·temporal_smoothness(G(x))
    D_total = BCE(D(gt), 1)     +  BCE(D(G(x).detach()), 0)
"""

import sys, json, argparse
import numpy as np
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.models.fusion_network import DeepFusionPoseModel, FusionDataset, mc_predict
from src.evaluation.metrics import compute_jitter

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
# Loss components
# ────────────────────────────────────────────

def temporal_smoothness_loss(pred: torch.Tensor) -> torch.Tensor:
    """Penalises sudden angle accelerations (second derivative)."""
    first_diff  = pred[:, 1:, :]  - pred[:, :-1, :]
    second_diff = first_diff[:, 1:, :] - first_diff[:, :-1, :]
    return second_diff.pow(2).mean()


# ────────────────────────────────────────────
# GAN Training Loop
# ────────────────────────────────────────────

def train_gan(
    generator: nn.Module,
    discriminator: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    pretrain_epochs: int = 20,
    gan_epochs: int = 60,
    lr_g: float = 1e-4,
    lr_d: float = 4e-5,
    lambda_recon: float = 10.0,
    lambda_adv:   float = 1.0,
    lambda_smooth: float = 5.0,
    device: str = "cpu",
    checkpoint_path: str = "outputs/models/gan_generator_best.pt",
) -> dict:
    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)

    generator.to(device)
    discriminator.to(device)

    opt_g = optim.AdamW(generator.parameters(),     lr=lr_g, betas=(0.5, 0.999))
    opt_d = optim.AdamW(discriminator.parameters(), lr=lr_d, betas=(0.5, 0.999))

    mse  = nn.MSELoss()
    bce  = nn.BCELoss()

    history = {"pretrain_loss": [], "g_loss": [], "d_loss": [], "val_mse": []}

    # ── Phase A: Pretrain generator alone (stabilise before adversarial) ──
    print(f"Pre-training generator for {pretrain_epochs} epochs...")
    for epoch in range(1, pretrain_epochs + 1):
        generator.train()
        ep_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt_g.zero_grad()
            pred = generator(xb)
            loss = mse(pred, yb) + lambda_smooth * temporal_smoothness_loss(pred)
            loss.backward()
            nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
            opt_g.step()
            ep_loss += loss.item() * xb.size(0)
        ep_loss /= len(train_loader.dataset)
        history["pretrain_loss"].append(ep_loss)
        if epoch % 5 == 0 or epoch == 1:
            print(f"  Pretrain epoch {epoch:3d}/{pretrain_epochs} | loss={ep_loss:.4f}")

    # ── Phase B: Adversarial training ────────────────────────────────────
    print(f"\nAdversarial training for {gan_epochs} epochs...")
    best_val = float("inf")
    for epoch in range(1, gan_epochs + 1):
        generator.train()
        discriminator.train()
        g_ep, d_ep = 0.0, 0.0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            real_label = torch.ones(xb.size(0), 1, device=device)
            fake_label = torch.zeros(xb.size(0), 1, device=device)

            # ── Update Discriminator ──
            opt_d.zero_grad()
            fake_seq    = generator(xb).detach()
            d_real_loss = bce(discriminator(yb),        real_label)
            d_fake_loss = bce(discriminator(fake_seq),  fake_label)
            d_loss = (d_real_loss + d_fake_loss) * 0.5
            d_loss.backward()
            opt_d.step()

            # ── Update Generator ──
            opt_g.zero_grad()
            fake_seq    = generator(xb)
            adv_loss    = bce(discriminator(fake_seq),  real_label)
            recon_loss  = mse(fake_seq, yb)
            smooth_loss = temporal_smoothness_loss(fake_seq)
            g_loss = lambda_recon * recon_loss + lambda_adv * adv_loss + lambda_smooth * smooth_loss
            g_loss.backward()
            nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
            opt_g.step()

            g_ep += g_loss.item() * xb.size(0)
            d_ep += d_loss.item() * xb.size(0)

        g_ep /= len(train_loader.dataset)
        d_ep /= len(train_loader.dataset)

        # Validation MSE only
        generator.eval()
        val_mse = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = generator(xb)
                val_mse += mse(pred, yb).item() * xb.size(0)
        val_mse /= len(val_loader.dataset)

        history["g_loss"].append(g_ep)
        history["d_loss"].append(d_ep)
        history["val_mse"].append(val_mse)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  GAN epoch {epoch:3d}/{gan_epochs} | G={g_ep:.4f} | D={d_ep:.4f} | val_mse={val_mse:.4f}")

        if val_mse < best_val:
            best_val = val_mse
            torch.save(generator.state_dict(), checkpoint_path)

    print(f"Best GAN val MSE: {best_val:.4f} → {checkpoint_path}")
    return history


def evaluate_gan(generator: nn.Module, loader: DataLoader, device: str = "cpu") -> dict:
    generator.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            all_preds.append(generator(xb).cpu().numpy())
            all_targets.append(yb.numpy())

    preds   = np.concatenate(all_preds,   axis=0).reshape(-1, 4)
    targets = np.concatenate(all_targets, axis=0).reshape(-1, 4)

    mae  = np.mean(np.abs(preds - targets), axis=0)
    rmse = np.sqrt(np.mean((preds - targets) ** 2, axis=0))
    jitter = np.array([compute_jitter(preds[:, j].tolist()) for j in range(4)])

    joint_names = ["shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow_flexion"]
    metrics = {}
    for i, name in enumerate(joint_names):
        metrics[name] = {"MAE": float(mae[i]), "RMSE": float(rmse[i]), "Jitter": float(jitter[i])}
    metrics["average"] = {
        "MAE": float(mae.mean()), "RMSE": float(rmse.mean()), "Jitter": float(jitter.mean())
    }
    return metrics


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

    train_gan(
        generator, discriminator,
        train_loader, val_loader,
        pretrain_epochs=args.pretrain,
        gan_epochs=args.gan_epochs,
        device=device,
        checkpoint_path=ckpt,
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
