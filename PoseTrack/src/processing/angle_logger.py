import csv
import time
from pathlib import Path
from typing import Dict, Optional
import numpy as np


class AngleLogger:
    def __init__(self, output_dir: Optional[Path] = None):
        if output_dir is None:
            from config.config import ANGLES_DIR
            output_dir = ANGLES_DIR
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.csv_path = None
        self.csv_file = None
        self.csv_writer = None
        self.start_time = None
        self.running = False
        
        self.angle_history = []
        self.max_history = 1000

    def start(self, session_name: Optional[str] = None):
        if session_name is None:
            session_name = f"session_{int(time.time())}"
        
        self.csv_path = self.output_dir / f"{session_name}.csv"
        self.csv_file = open(self.csv_path, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            "timestamp", "elapsed_s", 
            "shoulder_elevation", "shoulder_yaw", "shoulder_roll", "elbow_flexion"
        ])
        
        self.start_time = time.perf_counter()
        self.running = True
        print(f"Logging to {self.csv_path}")

    def log(self, angles: Dict[str, float]):
        if not self.running:
            return
            
        elapsed = time.perf_counter() - self.start_time
        self.csv_writer.writerow([
            f"{time.time():.3f}",
            f"{elapsed:.4f}",
            f"{angles.get('shoulder_elevation', 0):.2f}",
            f"{angles.get('shoulder_yaw', 0):.2f}",
            f"{angles.get('shoulder_roll', 0):.2f}",
            f"{angles.get('elbow_flexion', 0):.2f}"
        ])
        self.csv_file.flush()
        
        self.angle_history.append({
            'elapsed': elapsed,
            **angles
        })
        if len(self.angle_history) > self.max_history:
            self.angle_history.pop(0)

    def stop(self):
        self.running = False
        if self.csv_file:
            self.csv_file.close()
            print(f"Logging stopped. File: {self.csv_path}")

    def get_statistics(self) -> Dict[str, float]:
        if not self.angle_history:
            return {}
        
        keys = ["shoulder_elevation", "shoulder_yaw", "shoulder_roll", "elbow_flexion"]
        stats = {}
        for key in keys:
            values = [a[key] for a in self.angle_history if key in a]
            if values:
                stats[f"{key}_mean"] = np.mean(values)
                stats[f"{key}_std"] = np.std(values)
                stats[f"{key}_min"] = np.min(values)
                stats[f"{key}_max"] = np.max(values)
        return stats


class AngleVisualizer:
    def __init__(self):
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            self.plt = plt
            self.available = True
        except ImportError:
            self.available = False

    def plot_history(self, logger: AngleLogger, output_path: Optional[Path] = None):
        if not self.available or not logger.angle_history:
            return
            
        data = logger.angle_history
        elapsed = [d['elapsed'] for d in data]
        
        fig, axes = self.plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.flatten()
        
        keys = ["shoulder_elevation", "shoulder_yaw", "shoulder_roll", "elbow_flexion"]
        titles = ["Shoulder Elevation", "Shoulder Yaw", "Shoulder Roll", "Elbow Flexion"]
        
        for i, (key, title) in enumerate(zip(keys, titles)):
            values = [d.get(key, 0) for d in data]
            axes[i].plot(elapsed, values)
            axes[i].set_title(title)
            axes[i].set_xlabel("Time (s)")
            axes[i].set_ylabel("Angle (deg)")
            axes[i].grid(True)
        
        self.plt.tight_layout()
        
        if output_path is None:
            from config.config import OUTPUTS_DIR
            output_path = OUTPUTS_DIR / f"angle_plot_{int(time.time())}.png"
        
        self.plt.savefig(output_path)
        self.plt.close()
        print(f"Plot saved to {output_path}")