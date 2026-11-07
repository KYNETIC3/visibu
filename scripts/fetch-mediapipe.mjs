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
  sha256: 'e2dab61191e2dcd0a15f943d8e3ed1dce13c82dfa597b9dd39f562975a50c3f8',
};

const sha256 = (buf) => createHash('sha256').update(buf).digest('hex');

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
  if (sha256(buf) !== MODEL.sha256) throw new Error('Downloaded model does not match the pinned sha256');
  await writeFile(modelPath, buf);
}

// Check every run, so a stale or replaced file in public/ is caught too.
const sha = sha256(await readFile(modelPath));
if (sha !== MODEL.sha256) {
  throw new Error(`${modelPath} does not match the pinned sha256. Delete it and run npm run assets again.`);
}
console.log(`MediaPipe assets ready. ${MODEL.file} sha256 ${sha}`);
