/// <reference lib="webworker" />
// Runs MediaPipe Holistic off the main thread. Video frames arrive as
// ImageBitmaps and never leave the device; only landmark numbers go back.
import { FilesetResolver, HolisticLandmarker } from '@mediapipe/tasks-vision';
import { packFrame } from './frame';
import type { Delegate, FromWorker, ToWorker } from './messages';

declare const self: DedicatedWorkerGlobalScope;

let landmarker: HolisticLandmarker | null = null;
let lastTimestamp = -1;

function post(message: FromWorker, transfer: Transferable[] = []) {
  self.postMessage(message, transfer);
}

async function create(wasmPath: string, modelPath: string, delegate: Delegate) {
  const fileset = await FilesetResolver.forVisionTasks(wasmPath, true);
  return HolisticLandmarker.createFromOptions(fileset, {
    baseOptions: { modelAssetPath: modelPath, delegate },
    runningMode: 'VIDEO',
  });
}

async function init(msg: Extract<ToWorker, { type: 'init' }>) {
  const started = performance.now();
  const order: Delegate[] = msg.preferGpu ? ['GPU', 'CPU'] : ['CPU'];
  let lastError: unknown;
  for (const delegate of order) {
    try {
      landmarker = await create(msg.wasmPath, msg.modelPath, delegate);
      post({ type: 'ready', delegate, loadMs: performance.now() - started });
      return;
    } catch (err) {
      lastError = err;
    }
  }
  post({ type: 'error', message: `Tracker failed to start: ${String(lastError)}` });
}

function track(msg: Extract<ToWorker, { type: 'frame' }>) {
  const { bitmap } = msg;
  try {
    if (!landmarker) return;
    // MediaPipe requires strictly increasing timestamps in VIDEO mode.
    const timestamp = Math.max(msg.timestamp, lastTimestamp + 1);
    lastTimestamp = timestamp;
    const started = performance.now();
    const result = landmarker.detectForVideo(bitmap, timestamp);
    const inferenceMs = performance.now() - started;
    const frame = packFrame(result);
    post({ type: 'result', frame, timestamp, inferenceMs }, [frame.buffer]);
  } catch (err) {
    post({ type: 'error', message: String(err) });
  } finally {
    bitmap.close();
  }
}

self.onmessage = (event: MessageEvent<ToWorker>) => {
  const msg = event.data;
  if (msg.type === 'init') void init(msg);
  else track(msg);
};
