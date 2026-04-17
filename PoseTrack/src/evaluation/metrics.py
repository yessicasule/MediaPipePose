import numpy as np

def compute_fps(total_frames: int, duration_seconds: float) -> float:
    if duration_seconds <= 0:
        return 0.0
    return total_frames / duration_seconds

def compute_jitter(angles_list: list) -> float:
    """
    Computes frame-to-frame variance (jitter).
    angles_list: list of angle values for a specific joint across frames.
    Returns the mean absolute difference between consecutive frames.
    """
    if len(angles_list) < 2:
        return 0.0
    diffs = np.abs(np.diff(angles_list))
    return float(np.mean(diffs))

def compute_failure_rate(total_frames: int, valid_frames: int) -> float:
    """
    % of frames with missing keypoints or lost tracking.
    """
    if total_frames == 0:
        return 0.0
    failed_frames = total_frames - valid_frames
    return (failed_frames / total_frames) * 100.0

def compute_static_pose_stability(angles_list: list) -> float:
    """
    Validates static holding. Expected standard deviation < 5°
    """
    if len(angles_list) < 2:
        return 0.0
    return float(np.std(angles_list))

def validate_static_pose(angles_list: list, threshold: float = 5.0) -> bool:
    """
    Returns True if the pose is considered stable (std dev < threshold).
    """
    return compute_static_pose_stability(angles_list) < threshold

def evaluate_session(tracking_data: dict) -> dict:
    """
    Consolidates session evaluation.
    tracking_data format:
    {
        "total_time": 10.5,
        "total_frames": 315,
        "valid_frames": 300,
        "joints": {
            "elbow_flexion": [90.1, 90.2, 90.1, ...],
            ...
        }
    }
    """
    report = {
        "fps": compute_fps(tracking_data.get("total_frames", 0), tracking_data.get("total_time", 0.0)),
        "failure_rate_percent": compute_failure_rate(
            tracking_data.get("total_frames", 0), 
            tracking_data.get("valid_frames", 0)
        ),
        "joint_metrics": {}
    }
    
    joints = tracking_data.get("joints", {})
    for joint_name, angles in joints.items():
        report["joint_metrics"][joint_name] = {
            "jitter": compute_jitter(angles),
            "stability_std": compute_static_pose_stability(angles),
            "is_stable": validate_static_pose(angles)
        }
        
    return report
