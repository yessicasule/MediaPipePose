import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import process from "node:process";
import { performance } from "node:perf_hooks";

import * as tf from "@tensorflow/tfjs";
import * as posenet from "@tensorflow-models/posenet";
import jpeg from "jpeg-js";
import yargs from "yargs";
import { hideBin } from "yargs/helpers";

function loadJpegRgbUint8(filePath) {
  const buf = fs.readFileSync(filePath);
  const decoded = jpeg.decode(buf, { useTArray: true }); // {width,height,data:RGBA}
  const { width, height, data } = decoded;
  const rgb = new Uint8Array(width * height * 3);
  for (let i = 0, j = 0; i < data.length; i += 4, j += 3) {
    rgb[j + 0] = data[i + 0];
    rgb[j + 1] = data[i + 1];
    rgb[j + 2] = data[i + 2];
  }
  return { width, height, rgb };
}

function systemInfo() {
  return {
    os: `${os.platform()} ${os.release()}`,
    node: process.version,
    cpu: os.cpus()?.[0]?.model ?? "",
    cpu_cores_logical: os.cpus()?.length ?? null,
    ram_gb: Math.round((os.totalmem() / (1024 ** 3)) * 100) / 100,
    cwd: process.cwd(),
  };
}

async function main() {
  const args = await yargs(hideBin(process.argv))
    .option("frames_dir", { type: "string", demandOption: true })
    .option("out_json", { type: "string", demandOption: true })
    .option("architecture", { type: "string", default: "MobileNetV1" }) // MobileNetV1 | ResNet50
    .option("outputStride", { type: "number", default: 16 })
    .option("inputResolution", { type: "number", default: 257 })
    .option("multiplier", { type: "number", default: 0.75 })
    .option("quantBytes", { type: "number", default: 2 })
    .option("max_frames", { type: "number", default: 0, describe: "Optional cap (0 = all)" })
    .strict()
    .parse();

  const framesDir = path.resolve(args.frames_dir);
  const outJson = path.resolve(args.out_json);

  const files = fs
    .readdirSync(framesDir)
    .filter((f) => f.toLowerCase().endsWith(".jpg"))
    .sort()
    .map((f) => path.join(framesDir, f));
  if (files.length === 0) {
    throw new Error(`No .jpg frames found in: ${framesDir}`);
  }

  const cappedFiles = args.max_frames && args.max_frames > 0 ? files.slice(0, args.max_frames) : files;

  // Ensure we use an available backend (pure TFJS typically uses "cpu").
  await tf.setBackend("cpu");
  await tf.ready();

  const net = await posenet.load({
    architecture: args.architecture,
    outputStride: args.outputStride,
    inputResolution: { width: args.inputResolution, height: args.inputResolution },
    multiplier: args.multiplier,
    quantBytes: args.quantBytes,
  });

  const perFrame = [];
  const latencies = [];
  const meanScores = [];

  const wallT0 = performance.now();
  for (let i = 0; i < cappedFiles.length; i++) {
    const fp = cappedFiles[i];
    const img = loadJpegRgbUint8(fp);

    const input = tf.tensor3d(img.rgb, [img.height, img.width, 3], "int32");
    const t0 = performance.now();
    const pose = await net.estimateSinglePose(input, { flipHorizontal: false });
    const dtMs = performance.now() - t0;
    input.dispose();

    const scores = pose.keypoints.map((k) => Number(k.score ?? 0));
    const meanScore =
      scores.length > 0 ? scores.reduce((a, b) => a + b, 0) / scores.length : 0;

    latencies.push(dtMs);
    meanScores.push(meanScore);
    perFrame.push({
      frame_index: i,
      frame_file: path.basename(fp),
      inference_ms: Number(dtMs),
      avg_keypoint_score: Number(meanScore),
      // PoseNet keypoints are pixel coordinates in the input image.
      // Store as [x_px, y_px, score] for each keypoint, plus image size for normalization.
      image_size: { width: img.width, height: img.height },
      keypoints_xy_score: pose.keypoints.map((k) => [
        Number(k.position?.x ?? 0),
        Number(k.position?.y ?? 0),
        Number(k.score ?? 0),
      ]),
    });
  }
  const wallElapsedS = (performance.now() - wallT0) / 1000.0;
  const fps = cappedFiles.length / (wallElapsedS > 0 ? wallElapsedS : 1);

  function percentile(arr, p) {
    if (arr.length === 0) return 0;
    const a = Array.from(arr).sort((x, y) => x - y);
    const idx = Math.floor((p / 100) * (a.length - 1));
    return a[idx];
  }

  const result = {
    library: "PoseNet",
    framework: "tfjs",
    architecture: args.architecture,
    outputStride: args.outputStride,
    inputResolution: args.inputResolution,
    multiplier: args.multiplier,
    quantBytes: args.quantBytes,
    frames_dir: framesDir,
    n_frames: cappedFiles.length,
    wall_elapsed_s: wallElapsedS,
    fps,
    latency_ms: {
      mean: latencies.reduce((a, b) => a + b, 0) / Math.max(1, latencies.length),
      p50: percentile(latencies, 50),
      p90: percentile(latencies, 90),
      p95: percentile(latencies, 95),
    },
    avg_keypoint_score: {
      mean: meanScores.reduce((a, b) => a + b, 0) / Math.max(1, meanScores.length),
      p50: percentile(meanScores, 50),
    },
    per_frame: perFrame,
    system: systemInfo(),
    created_at_unix: Date.now() / 1000,
  };

  fs.mkdirSync(path.dirname(outJson), { recursive: true });
  fs.writeFileSync(outJson, JSON.stringify(result, null, 2), "utf-8");

  const { per_frame, ...summary } = result;
  console.log(JSON.stringify(summary, null, 2));
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});

