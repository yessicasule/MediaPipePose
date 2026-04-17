"""
Detailed MediaPipe Pose test with landmark information
"""
import cv2
import mediapipe as mp
import time

def test_mediapipe_with_details():
    """Test MediaPipe with detailed landmark information"""
    
    # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,  # 0=lite, 1=full, 2=heavy
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Cannot access camera")
        return False
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("\n" + "="*60)
    print("MEDIAPIPE POSE DETECTION - DETAILED TEST")
    print("="*60)
    print("\nCamera and MediaPipe initialized")
    print("\nUpper Body Landmarks to track:")
    print("   0: Nose")
    print("   11: Left Shoulder")
    print("   12: Right Shoulder")
    print("   13: Left Elbow")
    print("   14: Right Elbow")
    print("   15: Left Wrist")
    print("   16: Right Wrist")
    print("\nControls:")
    print("   Press 'q' to quit")
    print("   Press 's' to show/hide landmark info")
    print("="*60 + "\n")
    
    frame_count = 0
    start_time = time.time()
    show_info = True
    
    # Upper body landmark indices
    upper_body_indices = [0, 11, 12, 13, 14, 15, 16]
    landmark_names = {
        0: "Nose",
        11: "L_Shoulder",
        12: "R_Shoulder",
        13: "L_Elbow",
        14: "R_Elbow",
        15: "L_Wrist",
        16: "R_Wrist"
    }
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Cannot read frame")
            break
        
        frame = cv2.flip(frame, 1)
        
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        
        results = pose.process(image_rgb)
        
        image_rgb.flags.writeable = True
        
        frame_count += 1
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time if elapsed_time > 0 else 0
        
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
            )
            
            status = "Pose Detected"
            color = (0, 255, 0)
            
            if show_info:
                y_offset = 90
                for idx in upper_body_indices:
                    landmark = results.pose_landmarks.landmark[idx]
                    name = landmark_names[idx]
                    
                    h, w, _ = frame.shape
                    x_px = int(landmark.x * w)
                    y_px = int(landmark.y * h)
                    visibility = landmark.visibility
                    
                    text = f"{name}: ({x_px:4d}, {y_px:4d}) V:{visibility:.2f}"
                    cv2.putText(frame, text, (10, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                    y_offset += 20
                    
                    if visibility > 0.5:
                        cv2.circle(frame, (x_px, y_px), 5, (0, 255, 255), -1)
        else:
            status = "No Pose Detected - Step into frame"
            color = (0, 0, 255)
        
        cv2.putText(frame, status, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Frame: {frame_count}", (500, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow('MediaPipe Pose - Detailed Test', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            show_info = not show_info
            print(f"Landmark info: {'ON' if show_info else 'OFF'}")
    
    cap.release()
    cv2.destroyAllWindows()
    pose.close()
    
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Total frames processed: {frame_count}")
    print(f"Average FPS: {fps:.2f}")
    print(f"Duration: {elapsed_time:.2f} seconds")
    print("="*60 + "\n")
    
    return True

if __name__ == "__main__":
    test_mediapipe_with_details()
