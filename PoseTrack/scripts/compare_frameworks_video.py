import pandas as pd
import numpy as np
import cv2
import math
import time
from pathlib import Path
from extract_worst_frames import generate_panel, FRAMEWORKS, DOFS

def main():
    csv_path = Path("outputs/unified_dataset.csv")
    out_path = Path("outputs/comparison_test_22.mp4") # The user mentioned this name implicitly or I can just name it comparison_video.mp4
    out_path = Path("outputs/frameworks_comparison.mp4")
    
    duration_sec = 30
    fps = 30
    n_frames_to_render = duration_sec * fps
    
    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # We will just take the first N frames
    df_subset = df.head(n_frames_to_render)
    
    TOTAL_W = 300 * 4 # 4 panels of 300 width
    TOTAL_H = 350 + 50 # 350 height + 50 title bar
    
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (TOTAL_W, TOTAL_H))
    
    if not writer.isOpened():
        print(f"[ERROR] Cannot open VideoWriter for {out_path}")
        return
        
    print(f"[compare_frameworks_video] Generating {duration_sec}s @ {fps}fps -> {out_path}")
    t0 = time.perf_counter()
    
    for i, (_, row) in enumerate(df_subset.iterrows()):
        panels = []
        for prefix, title, color in FRAMEWORKS:
            panel = generate_panel(row, prefix, title, color)
            panels.append(panel)
            
        combined = np.hstack(panels)
        
        # Add a title bar
        title_bar = np.full((50, combined.shape[1], 3), (10, 10, 15), dtype=np.uint8)
        cv2.putText(title_bar, f"Dataset Frame: {row['frame']}  |  Subject: {row['subject']}  |  Action: {row['action']}", 
                    (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        
        final_img = np.vstack([title_bar, combined])
        writer.write(final_img)
        
        if i % (fps * 5) == 0:
            elapsed = time.perf_counter() - t0
            pct = i / n_frames_to_render * 100
            print(f"  {pct:5.1f}%  frame {i}/{n_frames_to_render}  ({elapsed:.1f}s elapsed)")
            
    writer.release()
    elapsed_total = time.perf_counter() - t0
    size_mb = out_path.stat().st_size / 1e6
    
    print(f"\n[OK] Comparison video written: {out_path}")
    print(f"     Duration : {duration_sec}s  |  Frames : {n_frames_to_render}")
    print(f"     Size     : {size_mb:.1f} MB")
    print(f"     Render   : {elapsed_total:.1f}s")

if __name__ == "__main__":
    main()
