"""
Complete Benchmark Workflow: Synchronized Data Collection and Multi-Framework Processing

This script provides a complete workflow for:
1. Recording synchronized video + motion capture data
2. Extracting frames from recorded video
3. Processing frames through MediaPipe, PoseNet, and MoveNet
4. Generating comprehensive comparison visualizations

Usage:
    python run_benchmark_workflow.py                    # Interactive mode
    python run_benchmark_workflow.py --record          # Record session
    python run_benchmark_workflow.py --benchmark       # Run benchmarks only
    python run_benchmark_workflow.py --all             # Full workflow
"""
import argparse
import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, List


@dataclass
class WorkflowConfig:
    """Configuration for the benchmark workflow"""
    mode: str = "interactive"
    session_name: Optional[str] = None
    recording_duration: float = 30.0
    video_path: Optional[str] = None
    frames_dir: Optional[str] = None
    output_dir: str = "outputs/benchmarks"
    max_frames: Optional[int] = None
    run_mediapipe: bool = True
    run_posenet: bool = True
    run_movenet: bool = True
    generate_visualizations: bool = True
    delete_frames_after: bool = False


class BenchmarkWorkflow:
    """
    Complete benchmark workflow orchestrator.
    
    Manages the entire pipeline from data collection to benchmark comparison.
    """
    
    def __init__(self, config: WorkflowConfig):
        self.config = config
        self.session_name = config.session_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.base_dir = Path(config.output_dir)
        self.session_dir = self.base_dir / self.session_name
        
    def print_header(self, text: str):
        """Print formatted header"""
        print(f"\n{'#'*70}")
        print(f"  {text}")
        print(f"{'#'*70}")
    
    def print_step(self, step: int, total: int, text: str):
        """Print step indicator"""
        print(f"\n[Step {step}/{total}] {text}")
        print("-" * 50)
    
    def check_dependencies(self) -> bool:
        """Check if required dependencies are installed"""
        self.print_header("DEPENDENCY CHECK")
        
        checks = {
            "MediaPipe": self._check_mediapipe,
            "OpenCV": self._check_opencv,
            "TensorFlow": self._check_tensorflow,
            "Node.js (PoseNet)": self._check_nodejs
        }
        
        all_ok = True
        for name, check_fn in checks.items():
            try:
                ok, msg = check_fn()
                status = "OK" if ok else "MISSING"
                print(f"  {name}: {status} - {msg}")
                if not ok:
                    all_ok = False
            except Exception as e:
                print(f"  {name}: ERROR - {e}")
                all_ok = False
        
        return all_ok
    
    def _check_mediapipe(self) -> tuple[bool, str]:
        try:
            import mediapipe
            return True, f"v{mediapipe.__version__}"
        except ImportError:
            return False, "Run: pip install mediapipe"
    
    def _check_opencv(self) -> tuple[bool, str]:
        try:
            import cv2
            return True, f"v{cv2.__version__}"
        except ImportError:
            return False, "Run: pip install opencv-python"
    
    def _check_tensorflow(self) -> tuple[bool, str]:
        try:
            import tensorflow as tf
            return True, f"v{tf.__version__}"
        except ImportError:
            return False, "Run: pip install tensorflow"
    
    def _check_nodejs(self) -> tuple[bool, str]:
        try:
            result = subprocess.run(["node", "--version"], capture_output=True, text=True)
            if result.returncode == 0:
                return True, f"v{result.stdout.strip()}"
            return False, "Node.js not found"
        except FileNotFoundError:
            return False, "Run: npm install (in benchmarks/posenet_tfjs/)"
    
    def record_data(self) -> Path:
        """Record synchronized video and motion capture data"""
        self.print_step(1, 4, "Recording Synchronized Data")
        
        from src.synchronized_recorder import SynchronizedRecorder
        
        recorder = SynchronizedRecorder(
            session_name=self.session_name,
            output_dir=str(self.config.output_dir),
            target_fps=30.0,
            video_width=1280,
            video_height=720
        )
        
        recorder.start_recording()
        
        duration = self.config.recording_duration if self.config.recording_duration > 0 else None
        recorder.record_loop(max_duration=duration)
        
        video_path = recorder.get_video_path()
        frames_dir = recorder.get_frames_dir()
        
        print(f"\nRecording complete!")
        print(f"  Video: {video_path}")
        print(f"  Frames: {frames_dir}")
        print(f"  Total frames: {len(list(frames_dir.glob('*.jpg')))}")
        
        return frames_dir
    
    def extract_frames_from_video(self, video_path: Path) -> Path:
        """Extract frames from video file"""
        self.print_step(1, 4, "Extracting Frames from Video")
        
        frames_dir = self.session_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        
        from benchmarks.extract_frames import extract_frames
        
        meta = extract_frames(
            video_path=video_path,
            out_dir=frames_dir,
            stride=1,
            max_frames=self.config.max_frames,
            resize_width=None,
            resize_height=None
        )
        
        print(f"\nFrames extracted:")
        print(f"  Count: {meta['extracted_frames']}")
        print(f"  Location: {frames_dir}")
        
        return frames_dir
    
    def run_benchmarks(self, frames_dir: Path) -> dict:
        """Run all framework benchmarks"""
        self.print_step(2, 4, "Running Framework Benchmarks")
        
        sys.path.insert(0, str(Path(__file__).parent))
        from benchmarks.run_all_benchmarks import UnifiedBenchmarkRunner, BenchmarkConfig
        
        config = BenchmarkConfig(
            session_name=self.session_name,
            frames_dir=str(frames_dir),
            output_dir=str(self.config.output_dir),
            max_frames=self.config.max_frames,
            run_mediapipe=self.config.run_mediapipe,
            run_posenet=self.config.run_posenet,
            run_movenet=self.config.run_movenet
        )
        
        runner = UnifiedBenchmarkRunner(config)
        results = runner.run_all_benchmarks()
        
        print(f"\nBenchmarks complete!")
        print(f"  MediaPipe: {results.mediapipe.get('fps', 0):.1f} fps" if results.mediapipe else "  MediaPipe: skipped")
        print(f"  PoseNet: {results.posenet.get('fps', 0):.1f} fps" if results.posenet else "  PoseNet: skipped")
        print(f"  MoveNet: {results.movenet.get('fps', 0):.1f} fps" if results.movenet else "  MoveNet: skipped")
        
        return asdict(results)
    
    def generate_visualizations(self, results_dir: Path, frames_dir: Path) -> dict:
        """Generate comparison visualizations"""
        self.print_step(3, 4, "Generating Visualizations")
        
        from benchmarks.visualize_benchmarks import run as visualize_run
        
        viz_dir = results_dir / "visualizations"
        
        summary = visualize_run(
            results_dir=results_dir,
            output_dir=viz_dir,
            frames_dir=frames_dir,
            generate_frame_viz=True
        )
        
        print(f"\nVisualizations saved to: {viz_dir}")
        
        return summary
    
    def print_final_summary(self, results: dict, frames_dir: Path):
        """Print final summary"""
        self.print_step(4, 4, "Workflow Complete")
        
        print(f"\nSession: {self.session_name}")
        print(f"Frames processed: {len(list(frames_dir.glob('*.jpg')))}")
        print(f"\nResults location: {self.session_dir}")
        
        if 'comparison_summary' in results:
            print(f"\nQuick Summary:")
            for key, value in results['comparison_summary'].items():
                if isinstance(value, dict):
                    print(f"  {key}: {value}")
    
    def run_interactive(self):
        """Run interactive workflow"""
        print(f"\n{'#'*70}")
        print("  SYNCHRONIZED POSE ESTIMATION BENCHMARK WORKFLOW")
        print(f"{'#'*70}")
        print(f"\nSession: {self.session_name}")
        print(f"Output: {self.session_dir}")
        
        if not self.check_dependencies():
            print("\nWarning: Some dependencies are missing. Install them before running benchmarks.")
        
        self.print_header("STEP 1: Data Recording")
        print("""
Options:
  1. Record new session (use camera)
  2. Use existing video file
  3. Skip recording (use existing frames)
        """)
        
        choice = input("Select option (1/2/3): ").strip()
        
        frames_dir = None
        
        if choice == "1":
            duration = float(input("Recording duration in seconds (default 30): ") or "30")
            self.config.recording_duration = duration
            frames_dir = self.record_data()
            self.config.frames_dir = str(frames_dir)
            
        elif choice == "2":
            video_path = input("Enter video path: ").strip()
            if video_path and Path(video_path).exists():
                frames_dir = self.extract_frames_from_video(Path(video_path))
                self.config.frames_dir = str(frames_dir)
            else:
                print("Invalid video path")
                return
        
        elif choice == "3":
            frames_dir = Path(input("Enter frames directory: ").strip())
            if not frames_dir.exists():
                print("Invalid directory")
                return
            self.config.frames_dir = str(frames_dir)
        
        if frames_dir and not self.config.max_frames:
            print(f"\nFound {len(list(frames_dir.glob('*.jpg')))} frames")
        
        self.print_header("STEP 2: Run Benchmarks")
        print("""
Select frameworks to benchmark:
  1. All (MediaPipe, PoseNet, MoveNet)
  2. MediaPipe only
  3. MediaPipe + MoveNet
  4. Custom selection
        """)
        
        bench_choice = input("Select option (1/2/3/4): ").strip()
        
        if bench_choice == "2":
            self.config.run_posenet = False
            self.config.run_movenet = False
        elif bench_choice == "3":
            self.config.run_posenet = False
        elif bench_choice == "4":
            self.config.run_mediapipe = input("Include MediaPipe? (y/n): ").lower() == 'y'
            self.config.run_posenet = input("Include PoseNet? (y/n): ").lower() == 'y'
            self.config.run_movenet = input("Include MoveNet? (y/n): ").lower() == 'y'
        
        if not self.config.frames_dir:
            self.config.frames_dir = input("Enter frames directory: ").strip()
        
        frames_dir = Path(self.config.frames_dir)
        results = self.run_benchmarks(frames_dir)
        
        if self.config.generate_visualizations:
            self.generate_visualizations(self.session_dir / "results", frames_dir)
        
        self.print_final_summary(results, frames_dir)
    
    def run_full(self):
        """Run complete automated workflow"""
        self.print_header("COMPLETE BENCHMARK WORKFLOW")
        
        self.check_dependencies()
        
        frames_dir = None
        
        if self.config.frames_dir:
            frames_dir = Path(self.config.frames_dir)
            if not frames_dir.exists():
                raise FileNotFoundError(f"Frames directory not found: {frames_dir}")
            print(f"\nUsing existing frames: {frames_dir}")
            
        elif self.config.video_path:
            frames_dir = self.extract_frames_from_video(Path(self.config.video_path))
            
        else:
            print("\nNo video or frames provided. Recording new session...")
            frames_dir = self.record_data()
        
        results = self.run_benchmarks(frames_dir)
        
        if self.config.generate_visualizations:
            self.generate_visualizations(self.session_dir / "results", frames_dir)
        
        self.print_final_summary(results, frames_dir)
        
        if self.config.delete_frames_after:
            print(f"\nDeleting extracted frames...")
            shutil.rmtree(frames_dir, ignore_errors=True)
            print("Frames deleted.")
        
        return results
    
    def run_benchmarks_only(self):
        """Run benchmarks only (skip recording)"""
        self.print_header("BENCHMARK RUNNER")
        
        if not self.config.frames_dir:
            raise ValueError("--frames_dir required for benchmark-only mode")
        
        frames_dir = Path(self.config.frames_dir)
        if not frames_dir.exists():
            raise FileNotFoundError(f"Frames directory not found: {frames_dir}")
        
        results = self.run_benchmarks(frames_dir)
        
        if self.config.generate_visualizations:
            self.generate_visualizations(self.session_dir / "results", frames_dir)
        
        self.print_final_summary(results, frames_dir)
        
        return results


def main():
    parser = argparse.ArgumentParser(
        description="Complete Benchmark Workflow: Synchronized Data Collection & Multi-Framework Processing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python run_benchmark_workflow.py

  # Record and benchmark
  python run_benchmark_workflow.py --record --duration 60

  # Benchmark existing frames
  python run_benchmark_workflow.py --benchmark --frames_dir outputs/synchronized/session_001/frames

  # Full automated workflow
  python run_benchmark_workflow.py --all --video_path path/to/video.mp4 --session_name test_001

  # Custom benchmark selection
  python run_benchmark_workflow.py --benchmark --frames_dir ./frames --no-posenet
        """
    )
    
    parser.add_argument("--mode", type=str, choices=["interactive", "record", "benchmark", "all"],
                       default="interactive", help="Workflow mode")
    parser.add_argument("--session_name", type=str, default=None, help="Session name")
    parser.add_argument("--duration", type=float, default=30.0,
                       help="Recording duration in seconds (0=unlimited)")
    parser.add_argument("--video_path", type=str, default=None,
                       help="Input video path for frame extraction")
    parser.add_argument("--frames_dir", type=str, default=None,
                       help="Pre-extracted frames directory")
    parser.add_argument("--output_dir", type=str, default="outputs/benchmarks",
                       help="Output directory")
    parser.add_argument("--max_frames", type=int, default=None,
                       help="Maximum frames to process")
    parser.add_argument("--no-mediapipe", action="store_true", help="Skip MediaPipe")
    parser.add_argument("--no-posenet", action="store_true", help="Skip PoseNet")
    parser.add_argument("--no-movenet", action="store_true", help="Skip MoveNet")
    parser.add_argument("--no-viz", action="store_true", help="Skip visualization generation")
    parser.add_argument("--delete_frames", action="store_true",
                       help="Delete extracted frames after benchmarking")
    
    args = parser.parse_args()
    
    config = WorkflowConfig(
        mode=args.mode,
        session_name=args.session_name,
        recording_duration=args.duration,
        video_path=args.video_path,
        frames_dir=args.frames_dir,
        output_dir=args.output_dir,
        max_frames=args.max_frames,
        run_mediapipe=not args.no_mediapipe,
        run_posenet=not args.no_posenet,
        run_movenet=not args.no_movenet,
        generate_visualizations=not args.no_viz,
        delete_frames_after=args.delete_frames
    )
    
    workflow = BenchmarkWorkflow(config)
    
    try:
        if config.mode == "interactive":
            workflow.run_interactive()
        elif config.mode == "record":
            frames_dir = workflow.record_data()
            print(f"\nRecording complete. Frames at: {frames_dir}")
        elif config.mode == "benchmark":
            workflow.run_benchmarks_only()
        elif config.mode == "all":
            workflow.run_full()
            
    except KeyboardInterrupt:
        print("\n\nWorkflow interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
