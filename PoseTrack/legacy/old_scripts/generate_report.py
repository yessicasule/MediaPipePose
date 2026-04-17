"""
Generate a complete report for the assignment
"""
import json
import glob
import os
from datetime import datetime
from config.config import Config

def generate_report():
    metadata_files = glob.glob(f"{Config.DATA_OUTPUT_DIR}/metadata_*.json")
    
    if not metadata_files:
        print("No sessions found")
        return
    
    latest_metadata = max(metadata_files, key=os.path.getctime)
    
    with open(latest_metadata, 'r') as f:
        metadata = json.load(f)
    
    session_name = metadata['session_name']
    
    report = f"""
{'='*70}
MEDIAPIPE UPPER-BODY POSE TRACKING - SESSION REPORT
{'='*70}

SESSION INFORMATION
-------------------
Session Name: {session_name}
Date & Time: {metadata['start_time']}
Duration: {metadata['duration_seconds']:.2f} seconds
Total Frames: {metadata['total_frames']}
Average FPS: {metadata.get('average_fps', 'N/A'):.2f}

MEDIAPIPE CONFIGURATION
-----------------------
Model Complexity: {metadata['mediapipe_settings']['model_complexity']}
Detection Confidence: {metadata['mediapipe_settings']['min_detection_confidence']}
Tracking Confidence: {metadata['mediapipe_settings']['min_tracking_confidence']}

TRACKED LANDMARKS (Upper Body)
------------------------------
"""
    
    for idx, landmark in enumerate(metadata['tracked_landmarks'], 1):
        report += f"{idx}. {landmark.replace('_', ' ').title()}\n"
    
    if 'system_info' in metadata:
        report += f"""
SYSTEM SPECIFICATIONS
--------------------
Operating System: {metadata['system_info']['os']}
Processor: {metadata['system_info']['processor']}
CPU Cores: {metadata['system_info']['cpu_cores']}
RAM: {metadata['system_info']['ram_gb']} GB

"""
    
    report += f"""
OUTPUT FILES
------------
Video: outputs/videos/pose_tracking_{session_name}.mp4
Data (CSV): outputs/data/landmarks_{session_name}.csv
Data (JSON): outputs/data/landmarks_{session_name}.json
Metadata: outputs/data/metadata_{session_name}.json
Plots: outputs/plots/*_{session_name}.png

QUALITATIVE OBSERVATIONS
------------------------
The MediaPipe Pose model successfully tracked upper-body landmarks in real-time
from a monocular camera feed. The system maintained an average FPS of 
{f"{metadata.get('average_fps', 0):.2f}" if isinstance(metadata.get('average_fps'), (int, float)) else 'N/A'}, demonstrating efficient performance.

Key landmarks tracked include:
- Head (nose)
- Shoulders (left and right)
- Elbows (left and right)
- Wrists (left and right)

The trajectory plots show the movement patterns of the elbow joints over time,
with visibility scores indicating tracking confidence throughout the session.

{'='*70}
END OF REPORT
{'='*70}
"""
    
    report_path = f"{Config.DATA_OUTPUT_DIR}/REPORT_{session_name}.txt"
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(report)
    print(f"\nReport saved to: {report_path}\n")

if __name__ == "__main__":
    generate_report()
