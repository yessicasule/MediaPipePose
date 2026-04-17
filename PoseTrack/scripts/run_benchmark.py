import os
import sys
import argparse
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.capture.video_recorder import VideoRecorder
from src.utils.io_utils import extract_frames_from_video, save_json
from config.config import DATA_DIR

def run_benchmark(video_path: str, models: list):
    print("--- 1. Extraction ---")
    frames_dir = os.path.join(DATA_DIR, "frames", Path(video_path).stem)
    extract_frames_from_video(video_path, frames_dir)
    
    print("--- 2. Pose Estimation ---")
    results = {}
    
    for model in models:
        print(f"Running {model}...")
        if model == "mediapipe":
            from src.pose.mediapipe_runner import MediaPipeRunner
            # Implement processing frames loop logic inside or here
            print("MediaPipe processing omitted for brevity in template")
            pass
        elif model == "movenet":
            # the TF logic
            print("MoveNet processing omitted for brevity in template")
            pass
        elif model == "posenet":
            from src.pose.posenet_runner import PoseNetRunner
            runner = PoseNetRunner()
            runner.process_frames(frames_dir, str(DATA_DIR / "landmarks" / "posenet"))
    
    # After models finish dumping to data/landmarks/, computing angles & metrics follows
    print("--- 3. Angle Computation & Metrics ---")
    # This invokes src.evaluation.metrics on parsed landmarks JSON datasets...
    print("Benchmark complete.")

def main():
    parser = argparse.ArgumentParser(description="Run Offline Benchmarking")
    parser.add_argument("--video", required=True, help="Input video to benchmark")
    parser.add_argument("--models", nargs="+", default=["mediapipe", "movenet", "posenet"])
    args = parser.parse_args()
    
    run_benchmark(args.video, args.models)

if __name__ == "__main__":
    main()
