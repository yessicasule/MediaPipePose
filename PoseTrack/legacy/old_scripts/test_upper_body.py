"""
Track upper-body landmarks specifically
"""
import cv2
import mediapipe as mp
import time

def test_upper_body_tracking():
    print("\n" + "="*60)
    print("UPPER-BODY LANDMARK TRACKING TEST")
    print("="*60 + "\n")
    
    # Define upper-body landmarks
    UPPER_BODY_LANDMARKS = {
        0: "Nose",
        1: "Left Eye (Inner)",
        2: "Left Eye",
        3: "Left Eye (Outer)",
        4: "Right Eye (Inner)",
        5: "Right Eye",
        6: "Right Eye (Outer)",
        7: "Left Ear",
        8: "Right Ear",
        9: "Mouth (Left)",
        10: "Mouth (Right)",
        11: "Left Shoulder",
        12: "Right Shoulder",
        13: "Left Elbow",
        14: "Right Elbow",
        15: "Left Wrist",
        16: "Right Wrist"
    }
    
    KEY_LANDMARKS = [0, 11, 12, 13, 14, 15, 16]
    
    print("Tracking Upper-Body Landmarks:")
    for idx in KEY_LANDMARKS:
        print(f"   {idx}: {UPPER_BODY_LANDMARKS[idx]}")
    
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    
    pose = mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("ERROR: Cannot access camera")
        return False
    
    print("\nControls:")
    print("   Press 'i' to toggle info display")
    print("   Press 'q' to quit\n")
    
    frame_count = 0
    start_time = time.time()
    show_info = True
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        frame_count += 1
        frame = cv2.flip(frame, 1)
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)
        
        fps = frame_count / (time.time() - start_time)
        
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )
            
            h, w, _ = frame.shape
            
            for idx in KEY_LANDMARKS:
                landmark = results.pose_landmarks.landmark[idx]
                
                x_px = int(landmark.x * w)
                y_px = int(landmark.y * h)
                visibility = landmark.visibility
                
                if visibility > 0.5:
                    cv2.circle(frame, (x_px, y_px), 8, (0, 255, 255), -1)
                    cv2.circle(frame, (x_px, y_px), 10, (255, 0, 0), 2)
                
                if show_info and visibility > 0.5:
                    label = UPPER_BODY_LANDMARKS[idx]
                    cv2.putText(frame, label, (x_px + 15, y_px),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            if show_info:
                y_offset = 120
                cv2.rectangle(frame, (5, 115), (350, 115 + len(KEY_LANDMARKS) * 25 + 10), 
                            (0, 0, 0), -1)
                cv2.rectangle(frame, (5, 115), (350, 115 + len(KEY_LANDMARKS) * 25 + 10), 
                            (255, 255, 255), 2)
                
                for idx in KEY_LANDMARKS:
                    landmark = results.pose_landmarks.landmark[idx]
                    x_px = int(landmark.x * w)
                    y_px = int(landmark.y * h)
                    visibility = landmark.visibility
                    
                    name = UPPER_BODY_LANDMARKS[idx]
                    info_text = f"{name}: ({x_px:3d},{y_px:3d}) V:{visibility:.2f}"
                    
                    color = (0, 255, 0) if visibility > 0.5 else (0, 0, 255)
                    cv2.putText(frame, info_text, (10, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                    y_offset += 25
            
            status = "TRACKING UPPER BODY"
            status_color = (0, 255, 0)
        else:
            status = "NO POSE DETECTED"
            status_color = (0, 0, 255)
        
        cv2.putText(frame, status, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow('Upper-Body Tracking Test', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('i'):
            show_info = not show_info
            print(f"Info display: {'ON' if show_info else 'OFF'}")
    
    cap.release()
    cv2.destroyAllWindows()
    pose.close()
    
    print(f"\nTest completed - {frame_count} frames processed at {fps:.2f} FPS\n")
    
    return True

if __name__ == "__main__":
    test_upper_body_tracking()
