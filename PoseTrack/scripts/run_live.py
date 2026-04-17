import cv2
import time
import argparse
import sys
from pathlib import Path

# Fix python path for isolated execution
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.pose.mediapipe_runner import MediaPipeRunner
from src.processing.joint_angle_estimator import compute_all
from src.processing.angle_filter import AngleFilterSystem
from src.streaming.udp_streamer import UdpAngleStreamer
from config.config import Config

def main():
    parser = argparse.ArgumentParser(description="Run Real-Time Pose Estimation to Unity")
    parser.add_argument("--host", default=Config.UDP_IP, help="UDP Host IP")
    parser.add_argument("--port", type=int, default=Config.UDP_PORT, help="UDP Port")
    parser.add_argument("--filter", default="kalman", choices=["ema", "kalman"], help="Filter type")
    parser.add_argument("--camera", type=int, default=Config.CAMERA_ID, help="Camera ID")
    parser.add_argument("--video", type=str, default=None, help="Optional video file instead of camera")
    args = parser.parse_args()

    print(f"Starting Real-Time Pipeline on {args.host}:{args.port} using {args.filter} filter")

    # Components
    pose_runner = MediaPipeRunner(
        min_detection_confidence=Config.MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=Config.MIN_TRACKING_CONFIDENCE
    )
    filter_sys = AngleFilterSystem(filter_type=args.filter)
    streamer = UdpAngleStreamer(host=args.host, port=args.port, hz=Config.OUTPUT_VIDEO_FPS)
    
    # Input
    cap = cv2.VideoCapture(args.video if args.video else args.camera)
    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.FRAME_HEIGHT)

    # Start UDP stream
    streamer.start()

    print("Pipeline running. Press 'q' to stop.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                if args.video:
                    # loop video
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    break

            # Process
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            landmarks = pose_runner.process(image_rgb)

            if landmarks is not None:
                # Visibility check could be added here before calculation
                raw_angles = compute_all(landmarks)
                filtered_angles = filter_sys.update(raw_angles)
                
                # Stream out
                streamer.update_angles(
                    shoulder_pitch = filtered_angles["shoulder_elevation"],
                    shoulder_yaw   = filtered_angles["shoulder_yaw"],
                    shoulder_roll  = filtered_angles["shoulder_roll"],
                    elbow_flex     = filtered_angles["elbow_flexion"]
                )
            else:
                # Retains last known in streamer thread by ignoring update
                pass

            # Visual feedback (optional)
            cv2.imshow("Real-Time Feed", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        pass
    finally:
        print("Shutting down...")
        cap.release()
        cv2.destroyAllWindows()
        pose_runner.close()
        streamer.stop()

if __name__ == "__main__":
    main()
