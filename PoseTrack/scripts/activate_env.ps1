# Environment activation scripts for PowerShell
# Usage: .\activate_env.ps1 [realtime|benchmark|posenet]

param (
    [string]$env_type
)

if ($env_type -eq "realtime") {
    Write-Host "Activating Real-Time System Environment (MediaPipe, UDP, Unity)"
    # Insert your specific activation command here, for example:
    # conda activate pose_realtime
    # or
    # .\.venv_realtime\Scripts\Activate.ps1
} elseif ($env_type -eq "benchmark") {
    Write-Host "Activating Benchmark Environment (TensorFlow, MoveNet)"
    # Insert your specific activation command here
    # conda activate pose_benchmark
} elseif ($env_type -eq "posenet") {
    Write-Host "PoseNet uses Node.js, checking Node version..."
    node -v
} else {
    Write-Host "Usage: .\activate_env.ps1 [realtime|benchmark|posenet]"
}
