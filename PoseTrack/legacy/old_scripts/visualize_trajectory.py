"""
Trajectory Visualization for Upper-Body Keypoints
Plots the trajectory of one arm keypoint (elbow joint) over time
"""

import json
import csv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse


def load_data_from_json(json_path):
    """Load landmark data from JSON file"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def load_data_from_csv(csv_path):
    """Load landmark data from CSV file"""
    frames = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            frames.append(row)
    return frames


def extract_elbow_trajectory(data, elbow_side='left', source_format='json'):
    """
    Extract elbow trajectory from landmark data
    
    Args:
        data: Landmark data (dict for JSON, list for CSV)
        elbow_side: 'left' or 'right'
        source_format: 'json' or 'csv'
    
    Returns:
        Dictionary with timestamps, x, y, z coordinates, and visibility
    """
    timestamps = []
    x_coords = []
    y_coords = []
    z_coords = []
    visibility = []
    
    elbow_key = f'{elbow_side}_elbow'
    
    if source_format == 'json':
        for frame in data['frames']:
            if elbow_key in frame['landmarks']:
                lm = frame['landmarks'][elbow_key]
                timestamps.append(frame['timestamp'])
                x_coords.append(lm['x'])
                y_coords.append(lm['y'])
                z_coords.append(lm['z'])
                visibility.append(lm['visibility'])
    else:  # CSV
        for frame in data:
            x_key = f'{elbow_key}_x'
            y_key = f'{elbow_key}_y'
            z_key = f'{elbow_key}_z'
            vis_key = f'{elbow_key}_visibility'
            
            if x_key in frame and frame[x_key]:
                timestamps.append(float(frame['timestamp']))
                x_coords.append(float(frame[x_key]))
                y_coords.append(float(frame[y_key]))
                z_coords.append(float(frame[z_key]))
                visibility.append(float(frame[vis_key]))
    
    return {
        'timestamps': np.array(timestamps),
        'x': np.array(x_coords),
        'y': np.array(y_coords),
        'z': np.array(z_coords),
        'visibility': np.array(visibility)
    }


def plot_trajectory(trajectory, elbow_side='left', output_path=None):
    """
    Plot elbow trajectory in 2D and 3D
    
    Args:
        trajectory: Dictionary with trajectory data
        elbow_side: 'left' or 'right'
        output_path: Path to save the plot
    """
    fig = plt.figure(figsize=(16, 6))
    
    # Filter by visibility (only plot points with visibility > 0.5)
    valid_mask = trajectory['visibility'] > 0.5
    
    # 2D Trajectory (X-Y plane)
    ax1 = fig.add_subplot(131)
    if np.any(valid_mask):
        ax1.plot(trajectory['x'][valid_mask], trajectory['y'][valid_mask], 
                'b-', alpha=0.6, linewidth=1.5, label='Trajectory')
        ax1.scatter(trajectory['x'][valid_mask], trajectory['y'][valid_mask],
                   c=trajectory['timestamps'][valid_mask], cmap='viridis',
                   s=20, alpha=0.7, edgecolors='black', linewidth=0.5)
        ax1.scatter(trajectory['x'][valid_mask][0], trajectory['y'][valid_mask][0],
                   c='green', s=100, marker='o', label='Start', zorder=5)
        ax1.scatter(trajectory['x'][valid_mask][-1], trajectory['y'][valid_mask][-1],
                   c='red', s=100, marker='s', label='End', zorder=5)
    ax1.set_xlabel('X (normalized)', fontsize=12)
    ax1.set_ylabel('Y (normalized)', fontsize=12)
    ax1.set_title(f'{elbow_side.capitalize()} Elbow Trajectory (X-Y Plane)', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.invert_yaxis()  # Invert Y to match image coordinates
    
    # 2D Trajectory (X-Z plane)
    ax2 = fig.add_subplot(132)
    if np.any(valid_mask):
        ax2.plot(trajectory['x'][valid_mask], trajectory['z'][valid_mask],
                'r-', alpha=0.6, linewidth=1.5, label='Trajectory')
        ax2.scatter(trajectory['x'][valid_mask], trajectory['z'][valid_mask],
                   c=trajectory['timestamps'][valid_mask], cmap='plasma',
                   s=20, alpha=0.7, edgecolors='black', linewidth=0.5)
        ax2.scatter(trajectory['x'][valid_mask][0], trajectory['z'][valid_mask][0],
                   c='green', s=100, marker='o', label='Start', zorder=5)
        ax2.scatter(trajectory['x'][valid_mask][-1], trajectory['z'][valid_mask][-1],
                   c='red', s=100, marker='s', label='End', zorder=5)
    ax2.set_xlabel('X (normalized)', fontsize=12)
    ax2.set_ylabel('Z (depth)', fontsize=12)
    ax2.set_title(f'{elbow_side.capitalize()} Elbow Trajectory (X-Z Plane)', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3D Trajectory
    ax3 = fig.add_subplot(133, projection='3d')
    if np.any(valid_mask):
        ax3.plot(trajectory['x'][valid_mask], trajectory['y'][valid_mask], trajectory['z'][valid_mask],
                'b-', alpha=0.6, linewidth=1.5, label='Trajectory')
        scatter = ax3.scatter(trajectory['x'][valid_mask], trajectory['y'][valid_mask], trajectory['z'][valid_mask],
                            c=trajectory['timestamps'][valid_mask], cmap='viridis',
                            s=20, alpha=0.7, edgecolors='black', linewidth=0.5)
        ax3.scatter(trajectory['x'][valid_mask][0], trajectory['y'][valid_mask][0], trajectory['z'][valid_mask][0],
                   c='green', s=100, marker='o', label='Start', zorder=5)
        ax3.scatter(trajectory['x'][valid_mask][-1], trajectory['y'][valid_mask][-1], trajectory['z'][valid_mask][-1],
                   c='red', s=100, marker='s', label='End', zorder=5)
        plt.colorbar(scatter, ax=ax3, label='Time (seconds)')
    ax3.set_xlabel('X (normalized)', fontsize=12)
    ax3.set_ylabel('Y (normalized)', fontsize=12)
    ax3.set_zlabel('Z (depth)', fontsize=12)
    ax3.set_title(f'{elbow_side.capitalize()} Elbow 3D Trajectory', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.invert_yaxis()  # Invert Y to match image coordinates
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Trajectory plot saved to {output_path}")
    else:
        plt.show()


def plot_trajectory_over_time(trajectory, elbow_side='left', output_path=None):
    """
    Plot elbow coordinates over time
    
    Args:
        trajectory: Dictionary with trajectory data
        elbow_side: 'left' or 'right'
        output_path: Path to save the plot
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    valid_mask = trajectory['visibility'] > 0.5
    
    # X coordinate over time
    axes[0].plot(trajectory['timestamps'][valid_mask], trajectory['x'][valid_mask],
                'b-', linewidth=2, label='X coordinate')
    axes[0].set_xlabel('Time (seconds)', fontsize=12)
    axes[0].set_ylabel('X (normalized)', fontsize=12)
    axes[0].set_title(f'{elbow_side.capitalize()} Elbow X Coordinate Over Time', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Y coordinate over time
    axes[1].plot(trajectory['timestamps'][valid_mask], trajectory['y'][valid_mask],
                'g-', linewidth=2, label='Y coordinate')
    axes[1].set_xlabel('Time (seconds)', fontsize=12)
    axes[1].set_ylabel('Y (normalized)', fontsize=12)
    axes[1].set_title(f'{elbow_side.capitalize()} Elbow Y Coordinate Over Time', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    axes[1].invert_yaxis()  # Invert Y to match image coordinates
    
    # Z coordinate (depth) over time
    axes[2].plot(trajectory['timestamps'][valid_mask], trajectory['z'][valid_mask],
                'r-', linewidth=2, label='Z coordinate (depth)')
    axes[2].set_xlabel('Time (seconds)', fontsize=12)
    axes[2].set_ylabel('Z (depth)', fontsize=12)
    axes[2].set_title(f'{elbow_side.capitalize()} Elbow Z Coordinate (Depth) Over Time', fontsize=14, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Time series plot saved to {output_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Visualize elbow trajectory from pose tracking data')
    parser.add_argument('--input', type=str, required=True,
                       help='Path to JSON or CSV file with landmark data')
    parser.add_argument('--elbow', type=str, choices=['left', 'right'], default='left',
                       help='Which elbow to visualize (default: left)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output directory for plots (default: same as input)')
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ Error: Input file not found: {input_path}")
        return
    
    # Determine format
    if input_path.suffix == '.json':
        data = load_data_from_json(input_path)
        source_format = 'json'
    elif input_path.suffix == '.csv':
        data = load_data_from_csv(input_path)
        source_format = 'csv'
    else:
        print(f"❌ Error: Unsupported file format. Please use .json or .csv")
        return
    
    # Extract trajectory
    print(f"📊 Extracting {args.elbow} elbow trajectory...")
    trajectory = extract_elbow_trajectory(data, args.elbow, source_format)
    
    if len(trajectory['timestamps']) == 0:
        print(f"❌ Error: No valid trajectory data found for {args.elbow} elbow")
        return
    
    print(f"✅ Found {len(trajectory['timestamps'])} data points")
    print(f"   Time range: {trajectory['timestamps'][0]:.2f}s - {trajectory['timestamps'][-1]:.2f}s")
    
    # Determine output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = input_path.parent
    
    output_dir.mkdir(exist_ok=True)
    
    # Generate plots
    trajectory_path = output_dir / f'{args.elbow}_elbow_trajectory.png'
    timeseries_path = output_dir / f'{args.elbow}_elbow_timeseries.png'
    
    print(f"\n📈 Generating trajectory plots...")
    plot_trajectory(trajectory, args.elbow, trajectory_path)
    plot_trajectory_over_time(trajectory, args.elbow, timeseries_path)
    
    print(f"\n✅ Visualization complete!")
    print(f"   Trajectory plot: {trajectory_path}")
    print(f"   Time series plot: {timeseries_path}")


if __name__ == "__main__":
    main()

