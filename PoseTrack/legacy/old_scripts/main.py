"""
Main entry point for pose tracking
"""
from src.pose_tracker import PoseTracker
from datetime import datetime

def main():
    session_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("\n" + "="*60)
    print("MEDIAPIPE UPPER-BODY POSE TRACKING")
    print("="*60)
    print(f"\nSession: {session_name}\n")
    
    tracker = PoseTracker(session_name=session_name)
    tracker.run(duration=60)
    
    print("Session completed\n")

if __name__ == "__main__":
    main()
