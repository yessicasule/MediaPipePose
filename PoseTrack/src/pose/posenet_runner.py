import os
import json
import subprocess
from pathlib import Path

class PoseNetRunner:
    def __init__(self, node_script_path: str = None):
        if node_script_path is None:
            self.node_script_path = str(
                Path(__file__).resolve().parent.parent.parent
                / "benchmarks" / "posenet_tfjs" / "run_posenet.mjs"
            )
        else:
            self.node_script_path = node_script_path

    def process_frames(self, input_dir: str, output_dir: str):
        print(f"Calling PoseNet Node script: {self.node_script_path}")
        print(f"Input dir: {input_dir}")
        print(f"Output dir: {output_dir}")
        try:
            result = subprocess.run(
                ["node", self.node_script_path, "--input", input_dir, "--output", output_dir],
                capture_output=True, text=True, check=True
            )
            print("PoseNet execution complete.")
            return result.stdout
        except subprocess.CalledProcessError as e:
            print(f"PoseNet execution failed:\n{e.stderr}")
            raise
