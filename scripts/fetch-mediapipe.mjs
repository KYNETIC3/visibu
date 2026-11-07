// Copies the pinned MediaPipe WASM files into public/ and downloads the pinned
// Holistic model once, so the app never loads code or models from a CDN.
import { createHash } from 'node:crypto';
import { copyFile, mkdir, readFile, readdir, writeFile } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const wasmSrc = join(root, 'node_modules/@mediapipe/tasks-vision/wasm');
const outDir = join(root, 'public/mediapipe');
const wasmOut = join(outDir, 'wasm');

const MODEL = {
  file: 'holistic_landmarker.task',
  url: 'https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/1/holistic_landmarker.task',
  bytes: 13683609,
};

await mkdir(wasmOut, { recursive: true });
for (const name of await readdir(wasmSrc)) {
  await copyFile(join(wasmSrc, name), join(wasmOut, name));
}

const modelPath = join(outDir, MODEL.file);
if (!existsSync(modelPath)) {
  console.log(`Downloading ${MODEL.file} ...`);
  const res = await fetch(MODEL.url);
  if (!res.ok) throw new Error(`Model download failed: HTTP ${res.status}`);
  const buf = Buffer.from(await res.arrayBuffer());
  if (buf.length !== MODEL.bytes) {
    throw new Error(`Model size ${buf.length} does not match pinned size ${MODEL.bytes}`);
  }
  await writeFile(modelPath, buf);
}

const sha = createHash('sha256').update(await readFile(modelPath)).digest('hex');
console.log(`MediaPipe assets ready. ${MODEL.file} sha256 ${sha}`);
