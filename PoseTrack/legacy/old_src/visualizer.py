"""
Visualization utilities for pose data
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from config.config import Config
import os

class PoseVisualizer:
    def __init__(self, csv_path):
        """
        Initialize visualizer with landmark data
        
        Args:
            csv_path: Path to CSV file with landmark data
        """
        self.csv_path = csv_path
        self.data = pd.read_csv(csv_path)
        self.session_name = os.path.basename(csv_path).replace('landmarks_', '').replace('.csv', '')
    
    def plot_elbow_trajectory(self, elbow='left', save=True):
        """
        Plot trajectory of elbow joint
        
        Args:
            elbow: 'left' or 'right'
            save: Whether to save the plot
        """
        landmark_name = f"{elbow}_elbow"
        
        x_col = f"{landmark_name}_x"
        y_col = f"{landmark_name}_y"
        
        if x_col not in self.data.columns or y_col not in self.data.columns:
            print(f"ERROR: {landmark_name} data not found!")
            return
        
        x_data = self.data[x_col].values
        y_data = self.data[y_col].values
        timestamps = self.data['timestamp'].values
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'{elbow.capitalize()} Elbow Trajectory Analysis', 
                     fontsize=16, fontweight='bold')
        
        ax1 = axes[0, 0]
        scatter = ax1.scatter(x_data, y_data, c=timestamps, 
                             cmap='viridis', s=20, alpha=0.6)
        ax1.plot(x_data, y_data, 'b-', alpha=0.3, linewidth=0.5)
        ax1.set_xlabel('X Coordinate (normalized)')
        ax1.set_ylabel('Y Coordinate (normalized)')
        ax1.set_title('2D Trajectory (X-Y Plane)')
        ax1.grid(True, alpha=0.3)
        ax1.invert_yaxis()
        plt.colorbar(scatter, ax=ax1, label='Time (s)')
        
        ax2 = axes[0, 1]
        ax2.plot(timestamps, x_data, 'r-', linewidth=1.5)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('X Coordinate')
        ax2.set_title('Horizontal Movement Over Time')
        ax2.grid(True, alpha=0.3)
        
        ax3 = axes[1, 0]
        ax3.plot(timestamps, y_data, 'g-', linewidth=1.5)
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Y Coordinate')
        ax3.set_title('Vertical Movement Over Time')
        ax3.grid(True, alpha=0.3)
        ax3.invert_yaxis()
        
        ax4 = axes[1, 1]
        
        dx = np.diff(x_data)
        dy = np.diff(y_data)
        dt = np.diff(timestamps)
        
        dt[dt == 0] = 1e-6
        
        speed = np.sqrt(dx**2 + dy**2) / dt
        speed_timestamps = timestamps[1:]
        
        ax4.plot(speed_timestamps, speed, 'purple', linewidth=1.5)
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Speed (units/s)')
        ax4.set_title('Movement Speed')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            output_path = f"{Config.PLOT_OUTPUT_DIR}/elbow_trajectory_{elbow}_{self.session_name}.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved: {output_path}")
        
        plt.show()
    
    def plot_upper_body_summary(self, save=True):
        """Plot summary of all upper-body landmarks"""
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Upper-Body Landmarks Summary', fontsize=16, fontweight='bold')
        
        landmarks_to_plot = [
            ('left_shoulder', 'Left Shoulder'),
            ('right_shoulder', 'Right Shoulder'),
            ('left_elbow', 'Left Elbow'),
            ('right_elbow', 'Right Elbow')
        ]
        
        for idx, (landmark_name, title) in enumerate(landmarks_to_plot):
            ax = axes[idx // 2, idx % 2]
            
            x_col = f"{landmark_name}_x"
            y_col = f"{landmark_name}_y"
            
            if x_col in self.data.columns and y_col in self.data.columns:
                x_data = self.data[x_col].values
                y_data = self.data[y_col].values
                
                ax.scatter(x_data, y_data, c=self.data['timestamp'], 
                          cmap='viridis', s=10, alpha=0.6)
                ax.plot(x_data, y_data, 'b-', alpha=0.2, linewidth=0.5)
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_title(title)
                ax.grid(True, alpha=0.3)
                ax.invert_yaxis()
        
        plt.tight_layout()
        
        if save:
            output_path = f"{Config.PLOT_OUTPUT_DIR}/upper_body_summary_{self.session_name}.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved: {output_path}")
        
        plt.show()
    
    def print_statistics(self):
        """Print basic statistics about the tracking session"""
        print("\n" + "="*60)
        print("TRACKING STATISTICS")
        print("="*60 + "\n")
        
        print(f"Total frames: {len(self.data)}")
        print(f"Duration: {self.data['timestamp'].max():.2f} seconds")
        print(f"Average frame interval: {self.data['timestamp'].diff().mean():.4f}s")
        
        print("\nLandmark Movement Ranges:")
        for landmark_idx in Config.UPPER_BODY_LANDMARKS:
            landmark_name = Config.LANDMARK_NAMES[landmark_idx]
            x_col = f"{landmark_name}_x"
            y_col = f"{landmark_name}_y"
            
            if x_col in self.data.columns:
                x_range = self.data[x_col].max() - self.data[x_col].min()
                y_range = self.data[y_col].max() - self.data[y_col].min()
                avg_visibility = self.data[f"{landmark_name}_visibility"].mean()
                
                print(f"\n{landmark_name}:")
                print(f"  X range: {x_range:.4f}")
                print(f"  Y range: {y_range:.4f}")
                print(f"  Avg visibility: {avg_visibility:.4f}")
        
        print("\n" + "="*60 + "\n")
