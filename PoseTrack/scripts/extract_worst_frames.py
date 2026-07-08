import pandas as pd
import numpy as np
import cv2
import math
from pathlib import Path

# Match the names used in unified_dataset.csv
DOFS = [
    ("shoulder_pitch", "Flexion"),
    ("shoulder_roll", "Abduction"),
    ("shoulder_yaw", "Rotation"),
    ("elbow_flexion", "Elbow Flex")
]
FRAMEWORKS = [
    ("gt", "Ground Truth", (230, 230, 230)),
    ("mp", "MediaPipe", (100, 220, 100)),
    ("mv", "MoveNet", (255, 160, 100)),
    ("pn", "PoseNet", (220, 80, 220))
]

def _draw_skeleton(img: np.ndarray, flex: float, abd: float, elbow: float, color: tuple) -> None:
    """Draw a simple right-arm stick figure that responds to angle values."""
    H, W = img.shape[:2]
    cx, cy = W // 2, H // 2

    # Shoulder
    sh = np.array([cx, cy - 30], dtype=float)

    # Upper arm (respond to flex + abd)
    flex_rad = math.radians(flex * 0.8)
    abd_rad  = math.radians(abd  * 0.5)
    ua_len   = 80
    ua = sh + np.array([
        ua_len * math.sin(flex_rad + 0.2) + ua_len * 0.3 * math.sin(abd_rad),
        ua_len * math.cos(flex_rad + 0.2),
    ])

    # Forearm (respond to elbow flexion)
    elbow_rad = math.radians(elbow * 0.7 + 30)
    fa_len = 70
    fa = ua + np.array([
        fa_len * math.sin(flex_rad + elbow_rad),
        fa_len * math.cos(flex_rad + elbow_rad),
    ])

    sh_i  = (int(sh[0]),  int(sh[1]))
    ua_i  = (int(ua[0]),  int(ua[1]))
    fa_i  = (int(fa[0]),  int(fa[1]))

    _GREY = (140, 140, 140)

    # Torso
    hip = (cx, cy + 60)
    cv2.line(img, (cx, cy - 80), hip, _GREY, 3, cv2.LINE_AA)
    cv2.line(img, (cx - 40, cy - 20), (cx + 40, cy - 20), _GREY, 3, cv2.LINE_AA)
    cv2.circle(img, (cx, cy - 100), 22, _GREY, 2, cv2.LINE_AA)

    # Right arm
    cv2.line(img,   sh_i, ua_i, color, 4, cv2.LINE_AA)
    cv2.line(img,   ua_i, fa_i, color, 4, cv2.LINE_AA)
    cv2.circle(img, sh_i, 7, (200, 200, 200), -1, cv2.LINE_AA)
    cv2.circle(img, ua_i, 6, color, -1, cv2.LINE_AA)
    cv2.circle(img, fa_i, 5, color, -1, cv2.LINE_AA)

    # Left arm (static)
    ls = (cx - 40, cy - 20)
    le = (cx - 100, cy + 20)
    lw = (cx - 110, cy + 60)
    cv2.line(img, ls, le, _GREY, 3, cv2.LINE_AA)
    cv2.line(img, le, lw, _GREY, 3, cv2.LINE_AA)

    # Legs
    cv2.line(img, hip, (cx - 25, cy + 130), _GREY, 3, cv2.LINE_AA)
    cv2.line(img, hip, (cx + 25, cy + 130), _GREY, 3, cv2.LINE_AA)

def generate_panel(row, prefix, title, color):
    # Retrieve angles
    flex = row[f"{prefix}_{DOFS[0][0]}"]
    abd = row[f"{prefix}_{DOFS[1][0]}"]
    rot = row[f"{prefix}_{DOFS[2][0]}"]
    elb = row[f"{prefix}_{DOFS[3][0]}"]
    
    img = np.full((350, 300, 3), (20, 20, 30), dtype=np.uint8)
    
    # Draw skeleton
    _draw_skeleton(img, flex, abd, elb, color)
    
    # Text
    _FONT = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, title, (10, 30), _FONT, 0.7, color, 2, cv2.LINE_AA)
    
    cv2.putText(img, f"Flex: {flex:+.1f}", (10, 310), _FONT, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(img, f"Abd:  {abd:+.1f}", (10, 330), _FONT, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    
    cv2.putText(img, f"Rot:  {rot:+.1f}", (150, 310), _FONT, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(img, f"Elbw: {elb:+.1f}", (150, 330), _FONT, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    
    return img

def main():
    csv_path = Path("outputs/unified_dataset.csv")
    out_dir = Path("outputs/worst_frames")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Calculate Mean Per Joint Angle Error (MPJAE) across frameworks
    errors = []
    for fw, _, _ in FRAMEWORKS[1:]: # Skip GT
        fw_err = 0
        for dof_col, _ in DOFS:
            err = (df[f"{fw}_{dof_col}"] - df[f"gt_{dof_col}"]).abs()
            fw_err += err
        errors.append(fw_err / 4.0) # Mean error across 4 DOFs
        
    # Take the max error across frameworks for each frame to find the truly worst frames
    df["max_framework_error"] = pd.concat(errors, axis=1).max(axis=1)
    
    # Sort and take top 10
    worst = df.sort_values(by="max_framework_error", ascending=False).head(10)
    
    print("Generating top 10 worst frame comparisons...")
    for i, (_, row) in enumerate(worst.iterrows(), 1):
        panels = []
        for prefix, title, color in FRAMEWORKS:
            panel = generate_panel(row, prefix, title, color)
            panels.append(panel)
            
        combined = np.hstack(panels)
        
        # Add a title bar
        title_bar = np.full((50, combined.shape[1], 3), (10, 10, 15), dtype=np.uint8)
        cv2.putText(title_bar, f"Worst Frame #{i} (Dataset Row {row.name}) - Max Error: {row['max_framework_error']:.1f} deg", 
                    (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        
        final_img = np.vstack([title_bar, combined])
        
        out_path = out_dir / f"worst_frame_{i:02d}.png"
        cv2.imwrite(str(out_path), final_img)
        print(f"Saved {out_path}")

if __name__ == "__main__":
    main()
