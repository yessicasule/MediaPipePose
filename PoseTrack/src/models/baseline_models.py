import os
import sys
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path

from config.config import Config
from src.models.datasets import BaseSequenceDataset
from src.models.trainer import BaseTrainer
from src.evaluation.metrics import evaluate_predictions, metrics_to_model_dict

class JointAngleDataset(BaseSequenceDataset):
    def __init__(self, csv_path: str, input_prefix: str, sequence_length: int = 30, step_size: int = 5):
        """
        PyTorch Dataset for sequence-based joint angle mapping.
        """
        input_cols = [
            f'{input_prefix}_shoulder_pitch',
            f'{input_prefix}_shoulder_roll',
            f'{input_prefix}_shoulder_yaw',
            f'{input_prefix}_elbow_flexion'
        ]
        target_cols = [
            'gt_shoulder_pitch',
            'gt_shoulder_roll',
            'gt_shoulder_yaw',
            'gt_elbow_flexion'
        ]
        super().__init__(
            csv_path=csv_path,
            feature_cols=input_cols,
            target_cols=target_cols,
            sequence_length=sequence_length,
            step_size=step_size,
            normalize=False
        )


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
    
    fm = evaluate_predictions(
        framework="Baseline",
        preds=preds,
        targets=targets,
        joint_names=['shoulder_pitch', 'shoulder_roll', 'shoulder_yaw', 'elbow_flexion']
    )
    return metrics_to_model_dict(fm)


def main():
    if len(sys.argv) < 3:
        print("Usage: python baseline_models.py <csv_path> <output_dir>")
        sys.exit(1)
        
    csv_path = sys.argv[1]
    output_dir = Path(sys.argv[2])
    
    # Ensure directories are created at application start
    Config.ensure_directories()
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
        model_path = output_dir / f"bilstm_baseline_{fw}.pt"
        
        trainer = BaseTrainer(
            model=model,
            device=device,
            checkpoint_path=str(model_path),
            patience=30,  # match baseline patience or epochs
            clip_grad=1.0
        )
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        _ = trainer.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=25,
            optimizer=optimizer,
            criterion=criterion
        )
        
        # Load best model checkpoint before evaluation
        model.load_state_dict(torch.load(model_path, map_location=device))
        
        # Evaluate
        eval_loader = DataLoader(dataset, batch_size=16, shuffle=False)
        metrics = evaluate_baseline_model(model, eval_loader, device=device)
        results_summary[name] = metrics
        
        print(f"Saved {name} model to {model_path}")
        
    # Save evaluation summary
    summary_path = output_dir / "baseline_evaluation_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(results_summary, f, indent=4)
        
    print(f"\n=== BASELINE EVALUATION SUMMARY ===")
    print(json.dumps(results_summary, indent=4))

if __name__ == "__main__":
    main()
