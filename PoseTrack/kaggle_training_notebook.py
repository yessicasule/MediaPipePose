# DeepFusionPose — Full Training Pipeline (Kaggle GPU Notebook)
# ================================================================
# Upload the following files from your local PoseTrack project to Kaggle:
#   /kaggle/working/src/models/baseline_models.py
#   /kaggle/working/src/models/fusion_network.py
#   /kaggle/working/src/models/gan_refinery.py
#   /kaggle/working/src/evaluation/metrics.py
#   /kaggle/working/src/evaluation/occlusion_test.py
#   /kaggle/working/scripts/run_experiments.py
#   /kaggle/input/deepfusionpose-data/unified_dataset.csv  ← your merged CSV
#
# Then run this notebook cell by cell.

# ─── Cell 1: Environment setup ───────────────────────────────────────────────
import subprocess, sys

def pip(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

pip("torch torchvision")
pip("pandas numpy scikit-learn matplotlib")
print("Packages ready.")

# ─── Cell 2: Paths ───────────────────────────────────────────────────────────
import os
from pathlib import Path

# Upload your H3.6M dataset folder as a Kaggle Dataset named "h36m-skeleton-data"
# It should contain: S1/ S5/ S6/ S7/ S8/ S9/ S11/  with .txt files inside each
H36M_DIR   = "/kaggle/input/h36m-skeleton-data"   # ← your Kaggle dataset mount
OUTPUT_DIR = Path("/kaggle/working/outputs/models")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH   = "/kaggle/working/outputs/unified_dataset.csv"

# Add src to Python path (upload your full PoseTrack/src + scripts as a dataset too)
sys.path.insert(0, "/kaggle/working")

# ─── Cell 3: Build the unified CSV from H3.6M skeletons ──────────────────────
# This uses build_h36m_dataset.py — NO VIDEO needed.
# Parses 3D joint positions → computes GT angles → adds simulated tracker noise.
from scripts.build_h36m_dataset import build_dataset

df = build_dataset(
    h36m_dir=H36M_DIR,
    output_csv=CSV_PATH,
    subjects=["S1", "S5", "S6", "S7", "S8", "S9", "S11"],
    seed=42,
)
print(f"\nCSV shape: {df.shape}")
print(df.head(3))

# ─── Cell 4: Device check ────────────────────────────────────────────────────
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")
if DEVICE == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ─── Cell 4: Phase 3 — Train 3 Baseline BiLSTM models ───────────────────────
import json
from torch.utils.data import DataLoader
from src.models.baseline_models import (
    BiLSTMPoseModel, JointAngleDataset,
    train_baseline_model, evaluate_baseline_model
)

FRAMEWORKS = {"mp": "MediaPipe", "mv": "MoveNet", "pn": "PoseNet"}
BASELINE_CKPTS = {}
baseline_results = {}

for fw, name in FRAMEWORKS.items():
    print(f"\n{'='*50}")
    print(f"Training baseline BiLSTM — {name}")
    print('='*50)
    ds = JointAngleDataset(CSV_PATH, fw, sequence_length=60, step_size=5)
    if len(ds) == 0:
        print(f"  SKIP: no data for {name}")
        continue
    n_train = int(0.8 * len(ds))
    train_ds, val_ds = torch.utils.data.random_split(
        ds, [n_train, len(ds) - n_train],
        generator=torch.Generator().manual_seed(42)
    )
    train_l = DataLoader(train_ds, batch_size=32, shuffle=True,  num_workers=2, pin_memory=True)
    val_l   = DataLoader(val_ds,   batch_size=32, shuffle=False, num_workers=2, pin_memory=True)
    eval_l  = DataLoader(ds,       batch_size=32, shuffle=False, num_workers=2, pin_memory=True)

    model = BiLSTMPoseModel()
    train_baseline_model(model, train_l, val_l, epochs=30, lr=1e-3, device=DEVICE)
    metrics = evaluate_baseline_model(model, eval_l, device=DEVICE)
    baseline_results[name] = metrics

    ckpt = str(OUTPUT_DIR / f"bilstm_baseline_{fw}.pt")
    torch.save(model.state_dict(), ckpt)
    BASELINE_CKPTS[fw] = ckpt
    print(f"  Saved → {ckpt}")
    print(f"  MAE={metrics['average']['MAE']:.3f}  RMSE={metrics['average']['RMSE']:.3f}")

with open(OUTPUT_DIR / "baseline_results.json", "w") as f:
    json.dump(baseline_results, f, indent=2)

# ─── Cell 5: Phase 4+5 — Train DeepFusionPose (Transformer + BiLSTM + MC Dropout) ──
from src.models.fusion_network import (
    DeepFusionPoseModel, FusionDataset,
    train_fusion_model, evaluate_fusion_model
)

print("\n" + "="*60)
print("Phase 4+5: Training DeepFusionPose Fusion Network")
print("="*60)

full_ds = FusionDataset(CSV_PATH, sequence_length=60)

# Save scaler for later use
SCALER_PATH = str(OUTPUT_DIR / "fusion_scaler.json")
with open(SCALER_PATH, "w") as f:
    json.dump(full_ds.scaler_dict(), f, indent=2)
print(f"Scaler saved → {SCALER_PATH}")

n_train = int(0.8 * len(full_ds))
train_ds, val_ds = torch.utils.data.random_split(
    full_ds, [n_train, len(full_ds) - n_train],
    generator=torch.Generator().manual_seed(42)
)
train_l = DataLoader(train_ds, batch_size=32, shuffle=True,  num_workers=2, pin_memory=True)
val_l   = DataLoader(val_ds,   batch_size=32, shuffle=False, num_workers=2, pin_memory=True)
eval_l  = DataLoader(full_ds,  batch_size=32, shuffle=False, num_workers=2, pin_memory=True)

FUSION_CKPT = str(OUTPUT_DIR / "fusion_best.pt")
fusion_model = DeepFusionPoseModel()
total_params = sum(p.numel() for p in fusion_model.parameters() if p.requires_grad)
print(f"Model parameters: {total_params:,}")

train_fusion_model(
    fusion_model, train_l, val_l,
    epochs=80, lr=1e-4, device=DEVICE,
    checkpoint_path=FUSION_CKPT,
)

fusion_model.load_state_dict(torch.load(FUSION_CKPT, map_location=DEVICE))
fusion_metrics = evaluate_fusion_model(fusion_model, eval_l, device=DEVICE, n_mc=50)

with open(OUTPUT_DIR / "fusion_evaluation.json", "w") as f:
    json.dump(fusion_metrics, f, indent=2)
print("\n=== FUSION METRICS ===")
print(json.dumps(fusion_metrics["average"], indent=2))

# ─── Cell 6: Phase 6 — GAN Refinement ───────────────────────────────────────
from src.models.gan_refinery import (
    TemporalDiscriminator, train_gan, evaluate_gan
)

print("\n" + "="*60)
print("Phase 6: GAN Refinement")
print("="*60)

GAN_CKPT = str(OUTPUT_DIR / "gan_generator_best.pt")

# Reload fusion model (fine-tune from pre-trained weights)
generator = DeepFusionPoseModel()
generator.load_state_dict(torch.load(FUSION_CKPT, map_location=DEVICE))
discriminator = TemporalDiscriminator(input_dim=4, seq_len=60)

train_gan(
    generator, discriminator,
    train_l, val_l,
    pretrain_epochs=20,
    gan_epochs=60,
    device=DEVICE,
    checkpoint_path=GAN_CKPT,
)

generator.load_state_dict(torch.load(GAN_CKPT, map_location=DEVICE))
gan_metrics = evaluate_gan(generator, eval_l, device=DEVICE)

with open(OUTPUT_DIR / "gan_evaluation.json", "w") as f:
    json.dump(gan_metrics, f, indent=2)
print("\n=== GAN METRICS ===")
print(json.dumps(gan_metrics["average"], indent=2))

# ─── Cell 7: Phase 7 — Occlusion Benchmark ───────────────────────────────────
from src.evaluation.occlusion_test import run_occlusion_benchmark

print("\n" + "="*60)
print("Phase 7: Occlusion Benchmark")
print("="*60)

occlusion_results = run_occlusion_benchmark(
    csv_path=CSV_PATH,
    scaler_path=SCALER_PATH,
    fusion_ckpt=FUSION_CKPT,
    gan_ckpt=GAN_CKPT,
    baseline_ckpts=BASELINE_CKPTS,
    output_json=str(OUTPUT_DIR.parent / "occlusion_results.json"),
    seq_len=60,
    batch=32,
    device=DEVICE,
)

# ─── Cell 8: Phase 10 — Ablation Study ───────────────────────────────────────
from scripts.run_experiments import run_ablation

print("\n" + "="*60)
print("Phase 10: Ablation Study")
print("="*60)

ablation_results = run_ablation(
    csv_path=CSV_PATH,
    scaler_path=SCALER_PATH,
    output_dir=str(OUTPUT_DIR.parent / "ablation"),
    seq_len=60,
    batch=32,
    epochs=50,
    lr=1e-4,
    device=DEVICE,
)

# ─── Cell 9: Final Summary Table ────────────────────────────────────────────
import pandas as pd

rows = []
for name, metrics in {**baseline_results, "Fusion": fusion_metrics, "GAN": gan_metrics}.items():
    avg = metrics.get("average", {})
    rows.append({
        "Model":  name,
        "MAE":    round(avg.get("MAE", 0), 3),
        "RMSE":   round(avg.get("RMSE", 0), 3),
        "Jitter": round(avg.get("Jitter", 0), 3),
    })

df = pd.DataFrame(rows)
print("\n=== FINAL COMPARISON ===")
print(df.to_markdown(index=False))
df.to_csv(str(OUTPUT_DIR.parent / "final_comparison.csv"), index=False)
print(f"\nAll model checkpoints saved to: {OUTPUT_DIR}")

# ─── Cell 10: Package outputs for download ───────────────────────────────────
import shutil
shutil.make_archive("/kaggle/working/DeepFusionPose_models", "zip", str(OUTPUT_DIR))
print("✅ Download: /kaggle/working/DeepFusionPose_models.zip")
print("   Contains all .pt checkpoints + evaluation JSONs + ablation report")
