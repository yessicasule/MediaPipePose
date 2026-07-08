import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
from pathlib import Path
import time

def generate_mock_session(csv_path: Path):
    """Generate a mock session CSV resembling main.py output for 30 seconds of motion."""
    fps = 30
    duration = 30
    n_frames = fps * duration
    
    rows = []
    
    # Filter state for simulated smoothing
    filt_flex, filt_abd, filt_rot, filt_elb = 0, 0, 0, 0
    alpha = 0.35
    
    start_time = time.time()
    
    for i in range(n_frames):
        t = i / fps
        
        # Raw sinusoidal motion + high frequency noise to simulate tracking jitter
        raw_flex = 70.0 * math.sin(2 * math.pi * t / 5.0) + np.random.normal(0, 2.0)
        raw_abd  = 40.0 * (1 - math.cos(2 * math.pi * t / 7.0)) + np.random.normal(0, 1.5)
        raw_rot  = 40.0 * math.sin(2 * math.pi * t / 9.0) + np.random.normal(0, 2.5)
        raw_elb  = 65.0 * (1 - math.cos(2 * math.pi * t / 4.0)) + np.random.normal(0, 1.0)
        
        # Apply exponential moving average (EMA) to simulate the Kalman filter
        if i == 0:
            filt_flex, filt_abd, filt_rot, filt_elb = raw_flex, raw_abd, raw_rot, raw_elb
        else:
            filt_flex = (1 - alpha) * filt_flex + alpha * raw_flex
            filt_abd  = (1 - alpha) * filt_abd  + alpha * raw_abd
            filt_rot  = (1 - alpha) * filt_rot  + alpha * raw_rot
            filt_elb  = (1 - alpha) * filt_elb  + alpha * raw_elb
            
        rows.append({
            "timestamp": start_time + t,
            "elapsed_s": t,
            "pose_detected": 1,
            "shoulder_flexion_raw": raw_flex,
            "shoulder_flexion_filt": filt_flex,
            "shoulder_abduction_raw": raw_abd,
            "shoulder_abduction_filt": filt_abd,
            "shoulder_rotation_raw": raw_rot,
            "shoulder_rotation_filt": filt_rot,
            "elbow_flexion_raw": raw_elb,
            "elbow_flexion_filt": filt_elb,
        })
        
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    print(f"Generated mock session data: {csv_path}")
    return df

def main():
    data_dir = Path("data/angles")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    out_path = Path("outputs/joint_angle_plots.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Check for existing sessions, otherwise generate a mock one
    csvs = list(data_dir.glob("*.csv"))
    if not csvs:
        print("No session CSVs found in data/angles/. Generating mock data...")
        df = generate_mock_session(data_dir / "session_mock.csv")
    else:
        latest_csv = max(csvs, key=lambda p: p.stat().st_mtime)
        print(f"Loading session data from: {latest_csv}")
        df = pd.read_csv(latest_csv)
        
    # Plotting
    fig, axs = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    fig.suptitle("MonoArm Live Joint Angle Stream (Raw vs Filtered)", fontsize=16)
    
    dofs = [
        ("shoulder_flexion", "Shoulder Flexion (deg)", "#55a868"),
        ("shoulder_abduction", "Shoulder Abduction (deg)", "#4c72b0"),
        ("shoulder_rotation", "Shoulder Rotation (deg)", "#dd8452"),
        ("elbow_flexion", "Elbow Flexion (deg)", "#c44e52")
    ]
    
    t = df["elapsed_s"]
    
    for ax, (prefix, title, color) in zip(axs, dofs):
        raw = df[f"{prefix}_raw"]
        filt = df[f"{prefix}_filt"]
        
        ax.plot(t, raw, color="#cccccc", label="Raw (Noisy)", linewidth=1)
        ax.plot(t, filt, color=color, label="Filtered", linewidth=2.5)
        
        ax.set_title(title, loc="left", fontsize=12)
        ax.set_ylabel("Degrees")
        ax.grid(True, linestyle="--", alpha=0.7)
        ax.legend(loc="upper right")
        
    axs[-1].set_xlabel("Elapsed Time (s)", fontsize=12)
    plt.tight_layout()
    
    plt.savefig(out_path, dpi=150)
    print(f"Saved plot to: {out_path}")

if __name__ == "__main__":
    main()
