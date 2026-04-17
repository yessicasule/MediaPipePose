import argparse
import sys
from pathlib import Path
import cv2

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.pose.mediapipe_runner import MediaPipeRunner
from src.processing.joint_angle_estimator import compute_all

def run_test(mode: str, camera: int):
    print(f"Running test in '{mode}' mode...")
    runner = MediaPipeRunner()
    cap = cv2.VideoCapture(camera)
    
    if not cap.isOpened():
        print(f"Cannot open camera {camera}")
        return
        
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            landmarks = runner.process(rgb)
            
            if landmarks:
                if mode == "basic":
                    print("Pose detected!")
                    cv2.putText(frame, "Pose Detected", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                elif mode == "detailed":
                    angles = compute_all(landmarks)
                    text = f"P:{angles['shoulder_elevation']:.1f} Y:{angles['shoulder_yaw']:.1f} R:{angles['shoulder_roll']:.1f} E:{angles['elbow_flexion']:.1f}"
                    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                elif mode == "upper_body":
                    # Assume upper-body constraint logic
                    cv2.putText(frame, "Upper Body Tracking", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
            else:
                cv2.putText(frame, "No Pose", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                
            cv2.imshow('Test Pose', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        runner.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Test Pose Capabilities")
    parser.add_argument("--mode", choices=["basic", "detailed", "upper_body"], default="basic")
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()
    run_test(args.mode, args.camera)
