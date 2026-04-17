#!/bin/bash
# Environment activation scripts for Bash
# Usage: source activate_env.sh [realtime|benchmark|posenet]

if [ "$1" == "realtime" ]; then
    echo "Activating Real-Time System Environment (MediaPipe, UDP, Unity)"
    # conda activate pose_realtime
elif [ "$1" == "benchmark" ]; then
    echo "Activating Benchmark Environment (TensorFlow, MoveNet)"
    # conda activate pose_benchmark
elif [ "$1" == "posenet" ]; then
    echo "PoseNet uses Node.js, checking Node version..."
    node -v
else
    echo "Usage: source activate_env.sh [realtime|benchmark|posenet]"
fi
