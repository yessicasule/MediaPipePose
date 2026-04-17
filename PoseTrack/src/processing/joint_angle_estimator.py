import numpy as np
import math

SHOULDER = 11
ELBOW    = 13
WRIST    = 15
HIP_L    = 23
HIP_R    = 24

def _vec(a, b):
    # Depending on the landmark format (MediaPipe objects have .x, .y, .z)
    try:
        return np.array([b.x - a.x, b.y - a.y, b.z - a.z], dtype=float)
    except AttributeError:
        return np.array([b[0] - a[0], b[1] - a[1], b[2] - a[2]], dtype=float)

def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-6 else v

def _angle_between(u, v):
    cos_a = np.clip(np.dot(_unit(u), _unit(v)), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_a)))

def compute_elbow_flexion(landmarks) -> float:
    upper_arm = _vec(landmarks[SHOULDER], landmarks[ELBOW])
    forearm   = _vec(landmarks[ELBOW],    landmarks[WRIST])
    straight_angle = _angle_between(upper_arm, forearm)
    return 180.0 - straight_angle

def compute_shoulder_elevation(landmarks) -> float:
    # Also known as shoulder pitch
    try:
        hip_mid = np.array([
            (landmarks[HIP_L].x + landmarks[HIP_R].x) / 2,
            (landmarks[HIP_L].y + landmarks[HIP_R].y) / 2,
            (landmarks[HIP_L].z + landmarks[HIP_R].z) / 2,
        ])
        torso_up = _unit(np.array([
            landmarks[SHOULDER].x - hip_mid[0],
            landmarks[SHOULDER].y - hip_mid[1],
            landmarks[SHOULDER].z - hip_mid[2],
        ]))
    except AttributeError:
        hip_mid = np.array([
            (landmarks[HIP_L][0] + landmarks[HIP_R][0]) / 2,
            (landmarks[HIP_L][1] + landmarks[HIP_R][1]) / 2,
            (landmarks[HIP_L][2] + landmarks[HIP_R][2]) / 2,
        ])
        torso_up = _unit(np.array([
            landmarks[SHOULDER][0] - hip_mid[0],
            landmarks[SHOULDER][1] - hip_mid[1],
            landmarks[SHOULDER][2] - hip_mid[2],
        ]))
        
    upper_arm = _unit(_vec(landmarks[SHOULDER], landmarks[ELBOW]))
    return _angle_between(torso_up, upper_arm)

def compute_shoulder_horizontal(landmarks) -> float:
    # Also known as shoulder yaw
    try:
        shoulder_l = np.array([landmarks[11].x, landmarks[11].y, landmarks[11].z])
        shoulder_r = np.array([landmarks[12].x, landmarks[12].y, landmarks[12].z])
    except AttributeError:
        shoulder_l = np.array([landmarks[11][0], landmarks[11][1], landmarks[11][2]])
        shoulder_r = np.array([landmarks[12][0], landmarks[12][1], landmarks[12][2]])
        
    torso_lateral = _unit(shoulder_r - shoulder_l)
    upper_arm = _unit(_vec(landmarks[SHOULDER], landmarks[ELBOW]))
    proj = upper_arm - np.dot(upper_arm, np.array([0, 1, 0])) * np.array([0, 1, 0])
    return _angle_between(torso_lateral, proj)

def compute_shoulder_roll(landmarks) -> float:
    upper_arm = _unit(_vec(landmarks[SHOULDER], landmarks[ELBOW]))
    forearm = _unit(_vec(landmarks[ELBOW], landmarks[WRIST]))
    
    try:
        hip_mid = np.array([
            (landmarks[HIP_L].x + landmarks[HIP_R].x) / 2,
            (landmarks[HIP_L].y + landmarks[HIP_R].y) / 2,
            (landmarks[HIP_L].z + landmarks[HIP_R].z) / 2,
        ])
        torso = _unit(np.array([
            landmarks[SHOULDER].x - hip_mid[0],
            landmarks[SHOULDER].y - hip_mid[1],
            landmarks[SHOULDER].z - hip_mid[2],
        ]))
    except AttributeError:
        hip_mid = np.array([
            (landmarks[HIP_L][0] + landmarks[HIP_R][0]) / 2,
            (landmarks[HIP_L][1] + landmarks[HIP_R][1]) / 2,
            (landmarks[HIP_L][2] + landmarks[HIP_R][2]) / 2,
        ])
        torso = _unit(np.array([
            landmarks[SHOULDER][0] - hip_mid[0],
            landmarks[SHOULDER][1] - hip_mid[1],
            landmarks[SHOULDER][2] - hip_mid[2],
        ]))
    
    cross_ua_fa = np.cross(upper_arm, forearm)
    y = np.dot(cross_ua_fa, upper_arm)
    x = np.dot(forearm, torso)
    
    roll = math.atan2(y, x)
    return float(np.degrees(roll))

def compute_all(landmarks) -> dict:
    try:
        return {
            "elbow_flexion":         compute_elbow_flexion(landmarks),
            "shoulder_elevation":    compute_shoulder_elevation(landmarks),
            "shoulder_yaw":          compute_shoulder_horizontal(landmarks),
            "shoulder_roll":         compute_shoulder_roll(landmarks)
        }
    except Exception:
        return {
            "elbow_flexion":       0.0,
            "shoulder_elevation":  0.0,
            "shoulder_yaw":        0.0,
            "shoulder_roll":       0.0
        }
