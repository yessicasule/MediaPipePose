"""
Comprehensive Visualization and Comparison Plots for Multi-Framework Benchmarking

Generates detailed comparison visualizations for MediaPipe, PoseNet, and MoveNet
benchmark results including latency, accuracy, FPS, and keypoint analysis.
"""
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Any

import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


plt.style.use('seaborn-v0_8-whitegrid')
COLORS = {
    'mediapipe': '#4285F4',
    'posenet': '#FBBC05', 
    'movenet': '#34A853'
}


def load_json(path: Path) -> dict:
    """Load JSON file"""
    with open(path, 'r') as f:
        return json.load(f)


def get_latencies(result: dict) -> np.ndarray:
    """Extract latency array from result"""
    per_frame = result.get('per_frame', [])
    return np.array([f['inference_ms'] for f in per_frame], dtype=float)


def get_scores(result: dict) -> np.ndarray:
    """Extract keypoint scores array from result"""
    per_frame = result.get('per_frame', [])
    return np.array([f.get('avg_keypoint_score', 0) for f in per_frame], dtype=float)


def plot_latency_comparison(results: Dict[str, dict], output_dir: Path):
    """Create per-frame latency comparison plot"""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    
    for name, result in results.items():
        latencies = get_latencies(result)
        frames = np.arange(len(latencies))
        
        axes[0].plot(frames, latencies, label=name.capitalize(), 
                    color=COLORS[name], alpha=0.7, linewidth=1)
        axes[0].fill_between(frames, latencies, alpha=0.2, color=COLORS[name])
        
        mean_lat = np.mean(latencies)
        axes[0].axhline(mean_lat, color=COLORS[name], linestyle='--', 
                        alpha=0.5, linewidth=1)
    
    axes[0].set_ylabel('Latency (ms)')
    axes[0].set_title('Per-Frame Inference Latency Comparison')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)
    
    for name, result in results.items():
        latencies = get_latencies(result)
        axes[1].hist(latencies, bins=50, alpha=0.5, label=name.capitalize(),
                    color=COLORS[name], edgecolor='white', linewidth=0.5)
    
    axes[1].set_xlabel('Frame Index / Latency (ms)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Latency Distribution Histogram')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / 'latency_comparison.png'
    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_fps_bar_chart(results: Dict[str, dict], output_dir: Path):
    """Create FPS comparison bar chart"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    names = [n.capitalize() for n in results.keys()]
    fps_values = [results[n].get('fps', 0) for n in results.keys()]
    colors = [COLORS[n] for n in results.keys()]
    
    bars = axes[0].bar(names, fps_values, color=colors, edgecolor='white', linewidth=1.5)
    axes[0].set_ylabel('FPS')
    axes[0].set_title('Overall Throughput (FPS)')
    axes[0].grid(True, axis='y', alpha=0.3)
    
    for bar, val in zip(bars, fps_values):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{val:.1f}', ha='center', va='bottom', fontweight='bold')
    
    mean_latencies = [
        results[n].get('latency_ms', {}).get('mean', 0) for n in results.keys()
    ]
    p90_latencies = [
        results[n].get('latency_ms', {}).get('p90', 0) for n in results.keys()
    ]
    
    x = np.arange(len(names))
    width = 0.35
    
    bars1 = axes[1].bar(x - width/2, mean_latencies, width, label='Mean', 
                        color=[c + '80' for c in colors])
    bars2 = axes[1].bar(x + width/2, p90_latencies, width, label='P90',
                        color=colors)
    
    axes[1].set_ylabel('Latency (ms)')
    axes[1].set_title('Latency Comparison (Mean vs P90)')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names)
    axes[1].legend()
    axes[1].grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / 'fps_latency_comparison.png'
    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_accuracy_comparison(results: Dict[str, dict], output_dir: Path):
    """Create accuracy/keypoint score comparison"""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    
    for name, result in results.items():
        scores = get_scores(result)
        frames = np.arange(len(scores))
        
        axes[0].plot(frames, scores, label=name.capitalize(),
                    color=COLORS[name], alpha=0.7, linewidth=1)
        axes[0].fill_between(frames, scores, alpha=0.2, color=COLORS[name])
    
    axes[0].set_ylabel('Avg Keypoint Score')
    axes[0].set_title('Per-Frame Average Keypoint Confidence Score')
    axes[0].legend(loc='lower right')
    axes[0].set_ylim([0, 1.1])
    axes[0].grid(True, alpha=0.3)
    
    for name, result in results.items():
        scores = get_scores(result)
        axes[1].boxplot(scores, positions=[list(results.keys()).index(name)],
                       widths=0.6, patch_artist=True,
                       boxprops=dict(facecolor=COLORS[name], alpha=0.5),
                       medianprops=dict(color='black', linewidth=2))
    
    axes[1].set_xticklabels([n.capitalize() for n in results.keys()])
    axes[1].set_ylabel('Score Distribution')
    axes[1].set_title('Keypoint Score Distribution (Box Plot)')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / 'accuracy_comparison.png'
    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_summary_dashboard(results: Dict[str, dict], output_dir: Path):
    """Create comprehensive summary dashboard"""
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)
    
    ax_fps = fig.add_subplot(gs[0, 0])
    ax_lat = fig.add_subplot(gs[0, 1])
    ax_score = fig.add_subplot(gs[0, 2])
    ax_lat_hist = fig.add_subplot(gs[1, 0])
    ax_score_hist = fig.add_subplot(gs[1, 1])
    ax_table = fig.add_subplot(gs[1, 2])
    ax_radar = fig.add_subplot(gs[2, :2], projection='polar')
    ax_detailed = fig.add_subplot(gs[2, 2])
    
    names = [n.capitalize() for n in results.keys()]
    fps_values = [results[n].get('fps', 0) for n in results.keys()]
    colors = [COLORS[n] for n in results.keys()]
    
    bars = ax_fps.bar(names, fps_values, color=colors, edgecolor='white')
    ax_fps.set_ylabel('FPS')
    ax_fps.set_title('Throughput')
    for bar, val in zip(bars, fps_values):
        ax_fps.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   f'{val:.1f}', ha='center', va='bottom', fontsize=9)
    
    mean_lat = [results[n].get('latency_ms', {}).get('mean', 0) for n in results.keys()]
    p90_lat = [results[n].get('latency_ms', {}).get('p90', 0) for n in results.keys()]
    x = np.arange(len(names))
    width = 0.35
    ax_lat.bar(x - width/2, mean_lat, width, label='Mean', color='steelblue', alpha=0.8)
    ax_lat.bar(x + width/2, p90_lat, width, label='P90', color='coral', alpha=0.8)
    ax_lat.set_xticks(x)
    ax_lat.set_xticklabels(names)
    ax_lat.set_ylabel('Latency (ms)')
    ax_lat.set_title('Latency Comparison')
    ax_lat.legend(fontsize=8)
    
    avg_scores = [np.mean(get_scores(results[n])) for n in results.keys()]
    bars = ax_score.bar(names, avg_scores, color=colors, edgecolor='white')
    ax_score.set_ylabel('Score')
    ax_score.set_title('Avg Keypoint Score')
    ax_score.set_ylim([0, 1.1])
    for bar, val in zip(bars, avg_scores):
        ax_score.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                     f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    for name, result in results.items():
        latencies = get_latencies(result)
        ax_lat_hist.hist(latencies, bins=30, alpha=0.5, label=name.capitalize(),
                        color=COLORS[name])
    ax_lat_hist.set_xlabel('Latency (ms)')
    ax_lat_hist.set_ylabel('Frequency')
    ax_lat_hist.set_title('Latency Distribution')
    ax_lat_hist.legend(fontsize=8)
    
    for name, result in results.items():
        scores = get_scores(result)
        ax_score_hist.hist(scores, bins=30, alpha=0.5, label=name.capitalize(),
                          color=COLORS[name])
    ax_score_hist.set_xlabel('Keypoint Score')
    ax_score_hist.set_ylabel('Frequency')
    ax_score_hist.set_title('Score Distribution')
    ax_score_hist.legend(fontsize=8)
    
    table_data = []
    headers = ['Framework', 'FPS', 'Mean(ms)', 'P90(ms)', 'Score']
    for name in results.keys():
        r = results[name]
        table_data.append([
            name.capitalize(),
            f"{r.get('fps', 0):.1f}",
            f"{r.get('latency_ms', {}).get('mean', 0):.2f}",
            f"{r.get('latency_ms', {}).get('p90', 0):.2f}",
            f"{np.mean(get_scores(r)):.3f}"
        ])
    
    ax_table.axis('off')
    table = ax_table.table(cellText=table_data, colLabels=headers,
                           loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    ax_table.set_title('Summary Table', pad=20, fontweight='bold')
    
    categories = ['FPS', 'Low Latency', 'Accuracy', 'Stability']
    n_categories = len(categories)
    angles = [n / float(n_categories) * 2 * np.pi for n in range(n_categories)]
    angles += angles[:1]
    
    ax_radar.set_title('Normalized Performance Radar', pad=20, fontweight='bold')
    ax_radar.set_theta_offset(np.pi / 2)
    ax_radar.set_theta_direction(-1)
    ax_radar.set_rlabel_position(0)
    
    max_fps = max(fps_values) if fps_values else 1
    max_lat = max([max(get_latencies(results[n])) for n in results.keys()]) if results else 1
    max_score = 1.0
    
    for name, result in results.items():
        values = [
            results[name].get('fps', 0) / max_fps,
            1 - (results[name].get('latency_ms', {}).get('mean', 0) / max_lat),
            np.mean(get_scores(results[name])) / max_score,
            1 - (np.std(get_scores(results[name])) / max_score)
        ]
        values += values[:1]
        ax_radar.plot(angles, values, 'o-', linewidth=2, label=name.capitalize(),
                     color=COLORS[name])
        ax_radar.fill(angles, values, alpha=0.15, color=COLORS[name])
    
    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels(categories, fontsize=10)
    ax_radar.set_ylim([0, 1])
    ax_radar.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=9)
    ax_radar.grid(True)
    
    detection_rates = []
    for name, result in results.items():
        per_frame = result.get('per_frame', [])
        detected = sum(1 for f in per_frame if f.get('pose_detected', True))
        rate = detected / len(per_frame) * 100 if per_frame else 0
        detection_rates.append(rate)
    
    ax_detailed.bar(names, detection_rates, color=colors, edgecolor='white')
    ax_detailed.set_ylabel('Detection Rate (%)')
    ax_detailed.set_title('Pose Detection Rate')
    ax_detailed.set_ylim([0, 105])
    for i, rate in enumerate(detection_rates):
        ax_detailed.text(i, rate + 2, f'{rate:.1f}%', ha='center', fontsize=9)
    
    fig.suptitle('Multi-Framework Pose Estimation Benchmark Dashboard',
                fontsize=16, fontweight='bold', y=0.98)
    
    output_path = output_dir / 'benchmark_dashboard.png'
    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_frame_comparison(results: Dict[str, dict], frames_dir: Path, 
                          output_dir: Path, max_frames: int = 9):
    """Create side-by-side pose visualization for sample frames"""
    frames = sorted(list(frames_dir.glob("*.jpg")))[:max_frames]
    
    if not frames:
        print("  No frames to visualize")
        return
    
    cols = min(3, len(frames))
    rows = (len(frames) + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols * 2, figsize=(cols * 6, rows * 4))
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx, frame_path in enumerate(frames):
        row = idx // cols
        col_base = (idx % cols) * 2
        
        frame = cv2.imread(str(frame_path))
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        axes[row, col_base].imshow(frame_rgb)
        axes[row, col_base].set_title(f'Frame {idx}', fontsize=10)
        axes[row, col_base].axis('off')
        
        for i, (name, result) in enumerate(results.items()):
            per_frame = result.get('per_frame', [])
            if idx < len(per_frame):
                info = per_frame[idx]
                lat = info.get('inference_ms', 0)
                score = info.get('avg_keypoint_score', 0)
                axes[row, col_base + 1].text(0.1, 0.9 - i * 0.15,
                    f'{name.capitalize()}: {lat:.1f}ms, {score:.2f}',
                    transform=axes[row, col_base + 1].transAxes,
                    fontsize=9, color=COLORS[name], fontweight='bold')
        
        axes[row, col_base + 1].text(0.1, 0.9,
            f'Frame {Path(frame_path).stem}', transform=axes[row, col_base + 1].transAxes,
            fontsize=10, fontweight='bold')
        axes[row, col_base + 1].axis('off')
        
        for i in range(col_base + 2, cols * 2):
            if row < len(axes) and i < len(axes[row]):
                axes[row, i].axis('off')
    
    plt.tight_layout()
    output_path = output_dir / 'frame_comparison.png'
    fig.savefig(output_path, dpi=120, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {output_path}")


def generate_comparison_report(results: Dict[str, dict], output_dir: Path):
    """Generate text comparison report"""
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("MULTI-FRAMEWORK POSE ESTIMATION BENCHMARK REPORT")
    report_lines.append("=" * 70)
    report_lines.append("")
    
    for name, result in results.items():
        report_lines.append(f"\n{name.upper()} RESULTS:")
        report_lines.append("-" * 40)
        report_lines.append(f"  FPS: {result.get('fps', 0):.2f}")
        report_lines.append(f"  Total Frames: {result.get('n_frames', len(result.get('per_frame', [])))}")
        report_lines.append(f"  Wall Time: {result.get('wall_elapsed_s', 0):.2f}s")
        
        lat_stats = result.get('latency_ms', {})
        report_lines.append(f"  Latency:")
        report_lines.append(f"    Mean: {lat_stats.get('mean', 0):.2f}ms")
        report_lines.append(f"    Std: {lat_stats.get('std', 0):.2f}ms")
        report_lines.append(f"    Min: {lat_stats.get('min', 0):.2f}ms")
        report_lines.append(f"    Max: {lat_stats.get('max', 0):.2f}ms")
        report_lines.append(f"    P50: {lat_stats.get('p50', 0):.2f}ms")
        report_lines.append(f"    P90: {lat_stats.get('p90', 0):.2f}ms")
        report_lines.append(f"    P95: {lat_stats.get('p95', 0):.2f}ms")
        
        scores = get_scores(result)
        report_lines.append(f"  Keypoint Score: {np.mean(scores):.3f} ± {np.std(scores):.3f}")
        
        detection = result.get('detection_rate', {})
        if detection:
            report_lines.append(f"  Detection Rate: {detection.get('rate', 0)*100:.1f}%")
    
    report_lines.append("\n" + "=" * 70)
    report_lines.append("COMPARISON SUMMARY")
    report_lines.append("=" * 70)
    
    fps_vals = [(n, results[n].get('fps', 0)) for n in results.keys()]
    fastest = max(fps_vals, key=lambda x: x[1])
    report_lines.append(f"\nFastest (FPS): {fastest[0].capitalize()} ({fastest[1]:.2f} fps)")
    
    lat_vals = [(n, results[n].get('latency_ms', {}).get('mean', float('inf'))) 
               for n in results.keys()]
    lowest_lat = min(lat_vals, key=lambda x: x[1])
    report_lines.append(f"Lowest Latency: {lowest_lat[0].capitalize()} ({lowest_lat[1]:.2f}ms)")
    
    score_vals = [(n, np.mean(get_scores(results[n]))) for n in results.keys()]
    highest_acc = max(score_vals, key=lambda x: x[1])
    report_lines.append(f"Highest Accuracy: {highest_acc[0].capitalize()} ({highest_acc[1]:.3f})")
    
    report_lines.append("\n" + "=" * 70)
    
    report_path = output_dir / 'benchmark_report.txt'
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    print(f"  Saved: {report_path}")
    
    return '\n'.join(report_lines)


def run(
    results_dir: Path,
    output_dir: Path,
    frames_dir: Optional[Path] = None,
    generate_frame_viz: bool = True
):
    """Run all visualization functions"""
    print(f"\n{'='*60}")
    print("GENERATING BENCHMARK VISUALIZATIONS")
    print(f"{'='*60}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    result_files = {
        'mediapipe': 'mediapipe.json',
        'posenet': 'posenet.json', 
        'movenet': 'movenet.json'
    }
    
    loaded_count = 0
    for name, filename in result_files.items():
        path = results_dir / filename
        if path.exists():
            results[name] = load_json(path)
            print(f"  Loaded: {path}")
            loaded_count += 1
    
    if not results:
        raise RuntimeError("No benchmark results found. Run benchmarks first.")
    
    print(f"\n  Total frameworks loaded: {loaded_count}/{len(result_files)}")
    
    print(f"\nGenerating plots...")
    
    plot_latency_comparison(results, output_dir)
    plot_fps_bar_chart(results, output_dir)
    plot_accuracy_comparison(results, output_dir)
    plot_summary_dashboard(results, output_dir)
    
    if generate_frame_viz and frames_dir and frames_dir.exists():
        plot_frame_comparison(results, frames_dir, output_dir)
    
    report = generate_comparison_report(results, output_dir)
    
    summary_path = output_dir / 'visualization_summary.json'
    summary = {
        'results_loaded': list(results.keys()),
        'frames_analyzed': len(results.get('mediapipe', {}).get('per_frame', [])),
        'outputs': {
            'latency_comparison': str(output_dir / 'latency_comparison.png'),
            'fps_latency_comparison': str(output_dir / 'fps_latency_comparison.png'),
            'accuracy_comparison': str(output_dir / 'accuracy_comparison.png'),
            'benchmark_dashboard': str(output_dir / 'benchmark_dashboard.png'),
            'frame_comparison': str(output_dir / 'frame_comparison.png'),
            'benchmark_report': str(output_dir / 'benchmark_report.txt')
        }
    }
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*60}")
    print("VISUALIZATION COMPLETE")
    print(f"{'='*60}")
    print(f"Output directory: {output_dir}")
    
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Generate benchmark comparison visualizations"
    )
    parser.add_argument("--results_dir", required=True,
                       help="Directory containing framework result JSON files")
    parser.add_argument("--frames_dir", type=str, default=None,
                       help="Directory with extracted frames for visualization")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory for visualizations")
    parser.add_argument("--no_frame_viz", action="store_true",
                       help="Skip frame-by-frame visualization")
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = results_dir / "visualizations"
    
    frames_dir = Path(args.frames_dir) if args.frames_dir else None
    
    summary = run(
        results_dir=results_dir,
        output_dir=output_dir,
        frames_dir=frames_dir,
        generate_frame_viz=not args.no_frame_viz
    )
    
    print("\n" + summary['outputs']['benchmark_report'])


if __name__ == "__main__":
    main()
