import numpy as np


SHOULDER = 11
ELBOW    = 13
WRIST    = 15
HIP_L    = 23
HIP_R    = 24


def _vec(a, b):
    return np.array([b.x - a.x, b.y - a.y, b.z - a.z], dtype=float)


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
    upper_arm = _unit(_vec(landmarks[SHOULDER], landmarks[ELBOW]))
    return _angle_between(torso_up, upper_arm)


def compute_shoulder_horizontal(landmarks) -> float:
    shoulder_l = np.array([landmarks[11].x, landmarks[11].y, landmarks[11].z])
    shoulder_r = np.array([landmarks[12].x, landmarks[12].y, landmarks[12].z])
    torso_lateral = _unit(shoulder_r - shoulder_l)

    upper_arm = _unit(_vec(landmarks[SHOULDER], landmarks[ELBOW]))
    proj = upper_arm - np.dot(upper_arm, np.array([0, 1, 0])) * np.array([0, 1, 0])
    return _angle_between(torso_lateral, proj)


def compute_all(landmarks) -> dict:
    try:
        return {
            "elbow_flexion":         compute_elbow_flexion(landmarks),
            "shoulder_elevation":    compute_shoulder_elevation(landmarks),
            "shoulder_horizontal":   compute_shoulder_horizontal(landmarks),
        }
    except Exception:
        return {
            "elbow_flexion":       0.0,
            "shoulder_elevation":  0.0,
            "shoulder_horizontal": 0.0,
        }
