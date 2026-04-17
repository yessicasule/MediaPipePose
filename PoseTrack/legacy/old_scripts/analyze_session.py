"""
Analyze and visualize a recorded session
"""
import sys
import os
import glob
from src.visualizer import PoseVisualizer
from config.config import Config

def main():
    csv_files = glob.glob(f"{Config.DATA_OUTPUT_DIR}/landmarks_*.csv")
    
    if not csv_files:
        print("No recorded sessions found")
        print(f"Please record a session first using: python main.py")
        return
    
    latest_csv = max(csv_files, key=os.path.getctime)
    
    print(f"\nAnalyzing session: {os.path.basename(latest_csv)}\n")
    
    viz = PoseVisualizer(latest_csv)
    viz.print_statistics()
    
    print("\nGenerating visualizations...\n")
    viz.plot_elbow_trajectory('left', save=True)
    viz.plot_elbow_trajectory('right', save=True)
    viz.plot_upper_body_summary(save=True)
    
    print("\nAnalysis complete\n")

if __name__ == "__main__":
    main()
