import subprocess
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).parent.resolve()
PYTHON = sys.executable


def _run(command, description, cwd=HERE):
    print(f"\n{'='*60}")
    print(f"STEP: {description}")
    print(f"{'='*60}")
    try:
        subprocess.run(command, shell=True, check=True, cwd=cwd)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        return False
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return False


def _ask_continue(step_name):
    resp = input(f"\n{step_name} failed. Continue anyway? (y/n): ")
    return resp.strip().lower() == "y"


def main():
    print("\n" + "="*70)
    print("COMPLETE POSE TRACKING WORKFLOW")
    print("="*70)
    print("\n1. System check")
    print("2. Camera test and recording session  (preview then record)")
    print("3. Analysis and visualization")
    print("4. Report generation")
    print("5. Frame extraction")
    print("6. Multi-framework benchmark (MediaPipe / MoveNet / PoseNet)")
    print("7. Framework comparison visualizations")
    print("\n" + "="*70)
    input("\nPress ENTER to start...")

    session_name = datetime.now().strftime("%Y%m%d_%H%M%S")

    basic_steps = [
        (f'"{PYTHON}" check_system.py',   "System Specifications Check"),
        (f'"{PYTHON}" test_camera.py',     "Camera Access Test (preview only, press q to quit)"),
        (f'"{PYTHON}" analyze_session.py', "Analyze Session and Generate Plots"),
        (f'"{PYTHON}" generate_report.py', "Generate Session Report"),
    ]

    for i, (cmd, desc) in enumerate(basic_steps, 1):
        print(f"\n\n{'='*70}")
        print(f"STEP {i}/7: {desc}")
        print(f"{'='*70}")

        if i == 2:
            print("\nNote: this is a preview-only camera test. Press q to close when done.")
            input("Press ENTER to start the camera test...")
            ok = _run(cmd, desc)
            if not ok and not _ask_continue(desc):
                print("\nWorkflow cancelled"); return

            print("\nNote: press SPACE to start recording, press SPACE again to stop, then q to quit")
            input("Press ENTER to start recording...")
            ok = _run(f'"{PYTHON}" main.py', "Record Pose Tracking Session")
            if not ok:
                resp = input("\nRecording ended. Continue with analysis? (y/n): ")
                if resp.strip().lower() != "y":
                    print("\nWorkflow cancelled"); return
            continue

        ok = _run(cmd, desc)

        if not ok:
            if not _ask_continue(desc):
                print("\nWorkflow cancelled"); return

    print(f"\n\n{'='*70}")
    print("STEP 5/7: Frame Extraction")
    print(f"{'='*70}")
    video_path = input("Enter path to recorded video (leave blank to skip benchmark): ").strip()

    frames_dir = None
    if video_path and Path(video_path).exists():
        frames_dir = HERE / "outputs" / "benchmarks" / session_name / "frames"
        ok = _run(
            f'"{PYTHON}" -m benchmarks.extract_frames '
            f'--video "{video_path}" --out_dir "{frames_dir}"',
            "Extract Frames from Video"
        )
        if not ok:
            frames_dir = None
    elif video_path == "":
        existing = input("Or enter an existing frames directory path (blank to skip): ").strip()
        if existing and Path(existing).exists():
            frames_dir = Path(existing)

    if frames_dir and Path(frames_dir).exists():
        print(f"\n\n{'='*70}")
        print("STEP 6/7: Multi-Framework Benchmark")
        print(f"{'='*70}")
        ok = _run(
            f'"{PYTHON}" run_benchmark_workflow.py --mode benchmark '
            f'--frames_dir "{frames_dir}" --session_name "{session_name}"',
            "Run MediaPipe / MoveNet / PoseNet Benchmarks"
        )
        if not ok and not _ask_continue("Benchmark"):
            print("\nWorkflow cancelled"); return

        print(f"\n\n{'='*70}")
        print("STEP 7/7: Comparison Visualizations")
        print(f"{'='*70}")
        results_dir = HERE / "outputs" / "benchmarks" / session_name / "results"
        _run(
            f'"{PYTHON}" -m benchmarks.visualize_benchmarks '
            f'--results_dir "{results_dir}" --frames_dir "{frames_dir}"',
            "Generate Framework Comparison Visualizations"
        )
    else:
        print("\nSkipping benchmark steps (no frames available).")

    print("\n\n" + "="*70)
    print("WORKFLOW COMPLETE")
    print("="*70)
    print(f"\n  Videos:     {HERE / 'outputs' / 'videos'}")
    print(f"  Data/CSV:   {HERE / 'outputs' / 'data'}")
    print(f"  Plots:      {HERE / 'outputs' / 'plots'}")
    print(f"  Benchmarks: {HERE / 'outputs' / 'benchmarks' / session_name}")
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nWorkflow interrupted by user")
        sys.exit(0)
