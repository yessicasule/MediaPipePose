"""
Unified Multi-Framework Pose Estimation Benchmark Runner

Processes synchronized video data through MediaPipe, PoseNet, and MoveNet
for comprehensive performance and accuracy comparison.
"""
import argparse
import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark run"""
    session_name: str
    video_path: Optional[str] = None
    frames_dir: Optional[str] = None
    output_dir: str = "outputs/benchmarks"
    max_frames: Optional[int] = None
    stride: int = 1
    run_mediapipe: bool = True
    run_posenet: bool = True
    run_movenet: bool = True
    posenet_architecture: str = "MobileNetV1"
    movenet_model: str = "lightning"
    mediapipe_model: str = "full"
    resize_width: Optional[int] = None
    resize_height: Optional[int] = None


@dataclass 
class BenchmarkResults:
    """Container for all benchmark results"""
    session_name: str
    timestamp: str
    total_duration_s: float = 0.0
    frames_processed: int = 0
    mediapipe: Optional[Dict] = None
    posenet: Optional[Dict] = None
    movenet: Optional[Dict] = None
    comparison_summary: Dict = field(default_factory=dict)


class UnifiedBenchmarkRunner:
    """
    Unified runner for multi-framework pose estimation benchmarking.
    
    Workflow:
    1. Extract frames from video (if not already done)
    2. Run benchmarks on all specified frameworks
    3. Collect and compare results
    4. Generate comprehensive comparison report
    """
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.results = BenchmarkResults(
            session_name=config.session_name,
            timestamp=datetime.now().isoformat()
        )
        
        self.base_dir = Path(config.output_dir)
        self.session_dir = self.base_dir / config.session_name
        self.results_dir = self.session_dir / "results"
        
        if config.frames_dir:
            self.frames_dir = Path(config.frames_dir)
        else:
            self.frames_dir = self.session_dir / "frames"
        
        self._setup_directories()
    
    def _setup_directories(self):
        """Create output directories"""
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        print(f"Session directory: {self.session_dir}")
    
    def extract_frames(self, force: bool = False) -> Path:
        """
        Extract frames from video for consistent benchmarking
        
        Args:
            force: Force re-extraction even if frames exist
        
        Returns:
            Path to extracted frames directory
        """
        if self.config.frames_dir:
            frames_path = Path(self.config.frames_dir)
            if frames_path.exists() and list(frames_path.glob("*.jpg")):
                print(f"Using existing frames: {frames_path}")
                return frames_path
        
        if self.frames_dir.exists() and not force:
            existing_frames = list(self.frames_dir.glob("*.jpg"))
            if existing_frames:
                print(f"Using existing frames: {self.frames_dir}")
                return self.frames_dir
        
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        
        if not self.config.video_path:
            raise ValueError("video_path required for frame extraction")
        
        video_path = Path(self.config.video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        print(f"\nExtracting frames from: {video_path}")
        
        from benchmarks.extract_frames import extract_frames
        
        meta = extract_frames(
            video_path=video_path,
            out_dir=self.frames_dir,
            stride=self.config.stride,
            max_frames=self.config.max_frames,
            resize_width=self.config.resize_width,
            resize_height=self.config.resize_height
        )
        
        print(f"Extracted {meta['extracted_frames']} frames")
        return self.frames_dir
    
    def run_mediapipe_benchmark(self) -> Dict[str, Any]:
        """Run MediaPipe benchmark"""
        if not self.config.run_mediapipe:
            return None
        
        output_json = self.results_dir / "mediapipe.json"
        
        if output_json.exists():
            print(f"\nLoading existing MediaPipe results: {output_json}")
            with open(output_json) as f:
                return json.load(f)
        
        print(f"\n{'='*60}")
        print("Running MediaPipe Benchmark")
        print(f"{'='*60}")
        
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from benchmarks.run_mediapipe_on_frames import run
        
        result = run(
            frames_dir=self.frames_dir,
            output_json=output_json,
            model=self.config.mediapipe_model,
            max_frames=self.config.max_frames
        )
        
        return result
    
    def run_movenet_benchmark(self) -> Dict[str, Any]:
        """Run MoveNet benchmark"""
        if not self.config.run_movenet:
            return None
        
        output_json = self.results_dir / "movenet.json"
        
        if output_json.exists():
            print(f"\nLoading existing MoveNet results: {output_json}")
            with open(output_json) as f:
                return json.load(f)
        
        print(f"\n{'='*60}")
        print("Running MoveNet Benchmark")
        print(f"{'='*60}")
        
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from benchmarks.run_movenet_on_frames import run
        
        result = run(
            frames_dir=self.frames_dir,
            out_json=output_json,
            model=self.config.movenet_model
        )
        
        return result
    
    def run_posenet_benchmark(self) -> Dict[str, Any]:
        """Run PoseNet benchmark (Node.js)"""
        if not self.config.run_posenet:
            return None
        
        output_json = self.results_dir / "posenet.json"
        
        if output_json.exists():
            print(f"\nLoading existing PoseNet results: {output_json}")
            with open(output_json) as f:
                return json.load(f)
        
        print(f"\n{'='*60}")
        print("Running PoseNet Benchmark (Node.js)")
        print(f"{'='*60}")
        
        posenet_script = Path(__file__).parent / "posenet_tfjs" / "run_posenet_on_frames.mjs"
        if not posenet_script.exists():
            raise FileNotFoundError(f"PoseNet script not found: {posenet_script}")
        
        cmd = [
            "node", str(posenet_script),
            "--frames_dir", str(self.frames_dir),
            "--out_json", str(output_json),
            "--architecture", self.config.posenet_architecture
        ]
        
        print(f"Running: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            raise RuntimeError(f"PoseNet benchmark failed: {result.stderr}")
        
        with open(output_json) as f:
            return json.load(f)
    
    def _compare_results(self) -> Dict[str, Any]:
        """Compare results across frameworks"""
        summary = {}
        
        for name in ["mediapipe", "posenet", "movenet"]:
            result = getattr(self.results, name)
            if result:
                score = 0
                if "avg_keypoint_score" in result:
                    score = result["avg_keypoint_score"].get("mean", 0)
                elif "keypoint_score" in result:
                    score = result["keypoint_score"].get("mean", 0)
                
                summary[name] = {
                    "fps": result.get("fps", 0),
                    "mean_latency_ms": result.get("latency_ms", {}).get("mean", 0),
                    "p90_latency_ms": result.get("latency_ms", {}).get("p90", 0),
                    "avg_keypoint_score": score
                }
        
        if len(summary) >= 2:
            fps_values = [(n, s["fps"]) for n, s in summary.items()]
            fastest = max(fps_values, key=lambda x: x[1])
            summary["fastest_fps"] = {"framework": fastest[0], "fps": fastest[1]}
            
            score_values = [(n, s["avg_keypoint_score"]) for n, s in summary.items()]
            most_accurate = max(score_values, key=lambda x: x[1])
            summary["highest_accuracy"] = {"framework": most_accurate[0], "score": most_accurate[1]}
        
        return summary
    
    def run_all_benchmarks(self) -> BenchmarkResults:
        """Run all configured benchmarks"""
        start_time = time.time()
        
        print(f"\n{'#'*70}")
        print(f"UNIFIED MULTI-FRAMEWORK POSE ESTIMATION BENCHMARK")
        print(f"{'#'*70}")
        print(f"Session: {self.config.session_name}")
        print(f"Frames directory: {self.frames_dir}")
        print(f"Frame count: {len(list(self.frames_dir.glob('*.jpg')))}")
        print(f"Frameworks: mediapipe={self.config.run_mediapipe}, "
              f"posenet={self.config.run_posenet}, movenet={self.config.run_movenet}")
        print(f"{'#'*70}")
        
        self.results.mediapipe = self.run_mediapipe_benchmark()
        self.results.movenet = self.run_movenet_benchmark()
        self.results.posenet = self.run_posenet_benchmark()
        
        self.results.comparison_summary = self._compare_results()
        self.results.total_duration_s = time.time() - start_time
        self.results.frames_processed = len(list(self.frames_dir.glob("*.jpg")))
        
        self._save_results()
        
        return self.results
    
    def _save_results(self):
        """Save final benchmark results"""
        output_path = self.session_dir / "benchmark_summary.json"
        
        with open(output_path, 'w') as f:
            json.dump(asdict(self.results), f, indent=2)
        
        print(f"\n{'='*60}")
        print("BENCHMARK COMPLETE")
        print(f"{'='*60}")
        print(f"Results saved to: {output_path}")
        
        if self.results.comparison_summary:
            print(f"\nComparison Summary:")
            for key, value in self.results.comparison_summary.items():
                if isinstance(value, dict):
                    print(f"  {key}: {value}")
                else:
                    print(f"  {key}: {value}")
    
    def print_summary(self):
        """Print formatted results summary"""
        print(f"\n{'='*70}")
        print(f"BENCHMARK RESULTS: {self.config.session_name}")
        print(f"{'='*70}")
        
        for framework in ["mediapipe", "posenet", "movenet"]:
            result = getattr(self.results, framework)
            if not result:
                continue
            
            print(f"\n{framework.upper()}:")
            print(f"  FPS: {result.get('fps', 0):.2f}")
            print(f"  Mean Latency: {result.get('latency_ms', {}).get('mean', 0):.2f}ms")
            print(f"  P90 Latency: {result.get('latency_ms', {}).get('p90', 0):.2f}ms")
            
            avg_score = result.get('avg_keypoint_score', {}).get('mean', 0) or \
                       result.get('keypoint_score', {}).get('mean', 0)
            print(f"  Avg Keypoint Score: {avg_score:.3f}")


def run_from_config(config: BenchmarkConfig) -> BenchmarkResults:
    """Convenience function to run benchmark from config"""
    runner = UnifiedBenchmarkRunner(config)
    
    runner.extract_frames()
    results = runner.run_all_benchmarks()
    runner.print_summary()
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Unified Multi-Framework Pose Estimation Benchmark"
    )
    
    parser.add_argument("--session_name", required=True, help="Session name for output")
    parser.add_argument("--video_path", type=str, default=None,
                       help="Path to input video (if frames not pre-extracted)")
    parser.add_argument("--frames_dir", type=str, default=None,
                       help="Path to pre-extracted frames directory")
    parser.add_argument("--output_dir", type=str, default="outputs/benchmarks",
                       help="Output directory for results")
    parser.add_argument("--max_frames", type=int, default=None,
                       help="Maximum frames to process")
    parser.add_argument("--stride", type=int, default=1,
                       help="Frame extraction stride")
    parser.add_argument("--no-mediapipe", action="store_true",
                       help="Skip MediaPipe benchmark")
    parser.add_argument("--no-posenet", action="store_true",
                       help="Skip PoseNet benchmark")
    parser.add_argument("--no-movenet", action="store_true",
                       help="Skip MoveNet benchmark")
    parser.add_argument("--posenet_arch", type=str, default="MobileNetV1",
                       choices=["MobileNetV1", "ResNet50"],
                       help="PoseNet architecture")
    parser.add_argument("--movenet_model", type=str, default="lightning",
                       choices=["lightning", "thunder"],
                       help="MoveNet model")
    parser.add_argument("--mediapipe_model", type=str, default="full",
                       choices=["full", "lite", "heavy"],
                       help="MediaPipe model")
    parser.add_argument("--resize_width", type=int, default=None,
                       help="Resize frames to width")
    parser.add_argument("--resize_height", type=int, default=None,
                       help="Resize frames to height")
    
    args = parser.parse_args()
    
    if not args.frames_dir and not args.video_path:
        parser.error("Either --frames_dir or --video_path required")
    
    config = BenchmarkConfig(
        session_name=args.session_name,
        video_path=args.video_path,
        frames_dir=args.frames_dir,
        output_dir=args.output_dir,
        max_frames=args.max_frames,
        stride=args.stride,
        run_mediapipe=not args.no_mediapipe,
        run_posenet=not args.no_posenet,
        run_movenet=not args.no_movenet,
        posenet_architecture=args.posenet_arch,
        movenet_model=args.movenet_model,
        mediapipe_model=args.mediapipe_model,
        resize_width=args.resize_width,
        resize_height=args.resize_height
    )
    
    results = run_from_config(config)
    
    return results


if __name__ == "__main__":
    main()
