import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from pathlib import Path

# Ensure we can import from PoseTrack
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.evaluation.metrics import compute_jitter

class JointAngleDataset(Dataset):
    def __init__(self, csv_path: str, input_prefix: str, sequence_length: int = 30, step_size: int = 5):
        """
        PyTorch Dataset for sequence-based joint angle mapping.
        csv_path: Path to unified dataset CSV
        input_prefix: 'mp', 'mv', or 'pn' (MediaPipe, MoveNet, PoseNet)
        sequence_length: Number of frames in a sequence window
        step_size: Stride for sliding window segmentation
        """
        df = pd.read_csv(csv_path)
        
        # Target angle columns (Ground Truth)
        target_cols = [
            'gt_shoulder_pitch',
            'gt_shoulder_roll',
            'gt_shoulder_yaw',
            'gt_elbow_flexion'
        ]
        
        # Input angle columns (Framework)
        input_cols = [
            f'{input_prefix}_shoulder_pitch',
            f'{input_prefix}_shoulder_roll',
            f'{input_prefix}_shoulder_yaw',
            f'{input_prefix}_elbow_flexion'
        ]
        
        # Extract features and targets
        self.inputs = df[input_cols].values.astype(np.float32)
        self.targets = df[target_cols].values.astype(np.float32)
        
        # Segment into sliding window sequences
        self.sequence_length = sequence_length
        self.samples = []
        
        n_frames = len(df)
        for start_idx in range(0, n_frames - sequence_length + 1, step_size):
            end_idx = start_idx + sequence_length
            x_seq = self.inputs[start_idx:end_idx]
            y_seq = self.targets[start_idx:end_idx]
            self.samples.append((x_seq, y_seq))
            
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x), torch.tensor(y)


class BiLSTMPoseModel(nn.Module):
    def __init__(self, input_dim: int = 4, hidden_dim: int = 64, num_layers: int = 2, output_dim: int = 4):
        super(BiLSTMPoseModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        
    def forward(self, x):
        # x shape: [Batch, SeqLen, InputDim]
        lstm_out, _ = self.lstm(x) # lstm_out shape: [Batch, SeqLen, HiddenDim * 2]
        out = self.fc(lstm_out)    # out shape: [Batch, SeqLen, OutputDim]
        return out


def train_baseline_model(
    model: nn.Module, 
    train_loader: DataLoader, 
    val_loader: DataLoader, 
    epochs: int = 30, 
    lr: float = 0.001,
    device: str = "cpu"
) -> dict:
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    history = {"train_loss": [], "val_loss": []}
    
    print("Beginning baseline BiLSTM training...")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * x_batch.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                outputs = model(x_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item() * x_batch.size(0)
        val_loss /= len(val_loader.dataset)
        
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            
    return history


def evaluate_baseline_model(model: nn.Module, data_loader: DataLoader, device: str = "cpu") -> dict:
    model.eval()
    model.to(device)
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for x_batch, y_batch in data_loader:
            x_batch = x_batch.to(device)
            outputs = model(x_batch)
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(y_batch.numpy())
            
    # Concatenate along batch axis
    preds = np.concatenate(all_preds, axis=0) # Shape: [N_Samples, SeqLen, 4]
    targets = np.concatenate(all_targets, axis=0)
    
    # Flatten across sequences to compute overall point metrics (MAE, RMSE)
    preds_flat = preds.reshape(-1, 4)
    targets_flat = targets.reshape(-1, 4)
    
    mae = np.mean(np.abs(preds_flat - targets_flat), axis=0)
    rmse = np.sqrt(np.mean((preds_flat - targets_flat)**2, axis=0))
    
    # Jitter is measured frame-to-frame across sequences.
    # Compute jitter for each sequence, then average.
    jitter_values = []
    for seq_idx in range(preds.shape[0]):
        seq_jitter = []
        for joint_idx in range(4):
            pred_joint_seq = preds[seq_idx, :, joint_idx].tolist()
            seq_jitter.append(compute_jitter(pred_joint_seq))
        jitter_values.append(seq_jitter)
    mean_jitter = np.mean(jitter_values, axis=0)
    
    joint_names = ['shoulder_pitch', 'shoulder_roll', 'shoulder_yaw', 'elbow_flexion']
    metrics = {}
    for i, name in enumerate(joint_names):
        metrics[name] = {
            "MAE": float(mae[i]),
            "RMSE": float(rmse[i]),
            "Jitter": float(mean_jitter[i])
        }
    
    # Add overall averages
    metrics["average"] = {
        "MAE": float(np.mean(mae)),
        "RMSE": float(np.mean(rmse)),
        "Jitter": float(np.mean(mean_jitter))
    }
    
    return metrics


def main():
    if len(sys.argv) < 3:
        print("Usage: python baseline_models.py <csv_path> <output_dir>")
        sys.exit(1)
        
    csv_path = sys.argv[1]
    output_dir = Path(sys.argv[2])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    frameworks = ["mp", "mv", "pn"]
    framework_names = {
        "mp": "MediaPipe",
        "mv": "MoveNet",
        "pn": "PoseNet"
    }
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    results_summary = {}
    
    for fw in frameworks:
        name = framework_names[fw]
        print(f"\n--- Training baseline model for {name} ---")
        
        # Create dataset and loaders
        dataset = JointAngleDataset(csv_path, fw, sequence_length=30, step_size=5)
        if len(dataset) == 0:
            print(f"Skipping {name}: Not enough data samples.")
            continue
            
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
        
        model = BiLSTMPoseModel()
        _ = train_baseline_model(model, train_loader, val_loader, epochs=25, lr=0.001, device=device)
        
        # Evaluate
        eval_loader = DataLoader(dataset, batch_size=16, shuffle=False)
        metrics = evaluate_baseline_model(model, eval_loader, device=device)
        results_summary[name] = metrics
        
        # Save model
        model_path = output_dir / f"bilstm_baseline_{fw}.pt"
        torch.save(model.state_dict(), model_path)
        print(f"Saved {name} model to {model_path}")
        
    # Save evaluation summary
    summary_path = output_dir / "baseline_evaluation_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(results_summary, f, indent=4)
        
    print(f"\n=== BASELINE EVALUATION SUMMARY ===")
    print(json.dumps(results_summary, indent=4))

if __name__ == "__main__":
    main()
