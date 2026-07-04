import os
import unittest
import tempfile
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path

from src.models.datasets import BaseSequenceDataset
from src.models.trainer import BaseTrainer, GANTrainer

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 4)
    def forward(self, x):
        return self.fc(x)

class TestRefactoredComponents(unittest.TestCase):
    def setUp(self):
        # Create a temporary CSV file representing the dataset
        self.temp_dir = tempfile.TemporaryDirectory()
        self.csv_path = Path(self.temp_dir.name) / "test_data.csv"
        
        # Build dummy dataset
        data = {
            "frame": list(range(100)),
            "subject": ["S1"] * 100,
            "action": ["Directions"] * 100,
            "mp_shoulder_pitch": np.linspace(0, 10, 100),
            "mp_shoulder_roll": np.linspace(0, -10, 100),
            "mp_shoulder_yaw": np.linspace(-5, 5, 100),
            "mp_elbow_flexion": np.linspace(20, 80, 100),
            "gt_shoulder_pitch": np.linspace(0, 10, 100),
            "gt_shoulder_roll": np.linspace(0, -10, 100),
            "gt_shoulder_yaw": np.linspace(-5, 5, 100),
            "gt_elbow_flexion": np.linspace(20, 80, 100),
        }
        df = pd.DataFrame(data)
        df.to_csv(self.csv_path, index=False)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_base_sequence_dataset_sliding_window(self):
        # Feature and target columns (old pitch/roll/yaw convention)
        feature_cols = [
            "mp_shoulder_pitch", "mp_shoulder_roll", "mp_shoulder_yaw", "mp_elbow_flexion"
        ]
        target_cols = [
            "gt_shoulder_pitch", "gt_shoulder_roll", "gt_shoulder_yaw", "gt_elbow_flexion"
        ]
        
        # Test sequence length 30, step size 5
        dataset = BaseSequenceDataset(
            csv_path=str(self.csv_path),
            feature_cols=feature_cols,
            target_cols=target_cols,
            sequence_length=30,
            step_size=5,
            normalize=False
        )
        
        # Expected number of windows: (100 - 30) // 5 + 1 = 15
        self.assertEqual(len(dataset), 15)
        
        # Verify first item shape
        x, y = dataset[0]
        self.assertEqual(x.shape, (30, 4))
        self.assertEqual(y.shape, (30, 4))

    def test_base_sequence_dataset_normalization(self):
        feature_cols = [
            "mp_shoulder_pitch", "mp_shoulder_roll", "mp_shoulder_yaw", "mp_elbow_flexion"
        ]
        target_cols = [
            "gt_shoulder_pitch", "gt_shoulder_roll", "gt_shoulder_yaw", "gt_elbow_flexion"
        ]
        
        # Test sequence length 20, step size 10, normalize=True
        dataset = BaseSequenceDataset(
            csv_path=str(self.csv_path),
            feature_cols=feature_cols,
            target_cols=target_cols,
            sequence_length=20,
            step_size=10,
            normalize=True
        )
        
        # Ensure scaling stats exist
        scaler_dict = dataset.scaler_dict()
        self.assertIn("col_min", scaler_dict)
        self.assertIn("col_max", scaler_dict)
        
        # Normalized input values should be in range [-1.0, 1.0]
        x, _ = dataset[0]
        self.assertTrue(torch.all(x >= -1.0001))
        self.assertTrue(torch.all(x <= 1.0001))

    def test_base_trainer_fit(self):
        feature_cols = [
            "mp_shoulder_pitch", "mp_shoulder_roll", "mp_shoulder_yaw", "mp_elbow_flexion"
        ]
        target_cols = [
            "gt_shoulder_pitch", "gt_shoulder_roll", "gt_shoulder_yaw", "gt_elbow_flexion"
        ]
        
        dataset = BaseSequenceDataset(
            csv_path=str(self.csv_path),
            feature_cols=feature_cols,
            target_cols=target_cols,
            sequence_length=10,
            step_size=5,
            normalize=False
        )
        
        loader = DataLoader(dataset, batch_size=4, shuffle=True)
        model = SimpleModel()
        checkpoint_file = Path(self.temp_dir.name) / "best_model.pt"
        
        trainer = BaseTrainer(
            model=model,
            device="cpu",
            checkpoint_path=str(checkpoint_file),
            patience=2,
            clip_grad=1.0
        )
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.MSELoss()
        
        # Run fit for 2 epochs
        history = trainer.fit(
            train_loader=loader,
            val_loader=loader,
            epochs=2,
            optimizer=optimizer,
            criterion=criterion
        )
        
        self.assertEqual(len(history["train_loss"]), 2)
        self.assertEqual(len(history["val_loss"]), 2)
        
        # Checkpoint file should have been created
        self.assertTrue(checkpoint_file.exists())

if __name__ == "__main__":
    unittest.main()
