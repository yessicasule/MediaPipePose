import argparse
import json
import socket
import time

import cv2

from src.angles import Vec3, elbow_flexion_deg, shoulder_elevation_deg


def _send(sock: socket.socket, host: str, port: int, payload: dict) -> None:
    msg = json.dumps(payload).encode("utf-8")
    sock.sendto(msg, (host, port))


def _mediapipe_keypoints(frame_bgr):
    import mediapipe as mp

    if not hasattr(_mediapipe_keypoints, "_pose"):
        _mediapipe_keypoints._pose = mp.solutions.pose.Pose(  # type: ignore[attr-defined]
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        _mediapipe_keypoints._mp = mp  # type: ignore[attr-defined]

    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    res = _mediapipe_keypoints._pose.process(rgb)  # type: ignore[attr-defined]
    if not res.pose_landmarks:
        return None
    lms = res.pose_landmarks.landmark
    # Indices: shoulders 11/12, elbows 13/14, wrists 15/16, hips 23/24
    def v(i):
        return Vec3(float(lms[i].x), float(lms[i].y), float(lms[i].z))

    return {
        "left_shoulder": v(11),
        "right_shoulder": v(12),
        "left_elbow": v(13),
        "right_elbow": v(14),
        "left_wrist": v(15),
        "right_wrist": v(16),
        "left_hip": v(23),
        "right_hip": v(24),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--framework", default="mediapipe", help="mediapipe (default)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5005)
    ap.add_argument("--camera", type=int, default=0)
    args = ap.parse_args()

    if args.framework != "mediapipe":
        raise SystemExit("Only --framework mediapipe is implemented for real-time streaming right now.")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit("Could not open camera.")

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    t0 = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        kp = _mediapipe_keypoints(frame)
        if kp:
            left_elbow = elbow_flexion_deg(kp["left_shoulder"], kp["left_elbow"], kp["left_wrist"])
            right_elbow = elbow_flexion_deg(kp["right_shoulder"], kp["right_elbow"], kp["right_wrist"])
            left_sh_elev = shoulder_elevation_deg(kp["left_hip"], kp["left_shoulder"], kp["left_elbow"])
            right_sh_elev = shoulder_elevation_deg(kp["right_hip"], kp["right_shoulder"], kp["right_elbow"])

            pkt = {
                "t": time.time() - t0,
                "left_elbow_deg": float(left_elbow),
                "right_elbow_deg": float(right_elbow),
                "left_shoulder_elev_deg": float(left_sh_elev),
                "right_shoulder_elev_deg": float(right_sh_elev),
            }
            _send(sock, args.host, args.port, pkt)

            cv2.putText(frame, f"L elbow: {left_elbow:.1f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"R elbow: {right_elbow:.1f}", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Pose -> Unity (press q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

