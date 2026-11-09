/// <reference lib="webworker" />
// Runs MediaPipe Holistic off the main thread. Video frames arrive as
// ImageBitmaps and never leave the device; only landmark numbers go back.
import { FilesetResolver, HolisticLandmarker } from '@mediapipe/tasks-vision';
import { packFrame } from './frame';
import type { Delegate, FromWorker, ToWorker } from './messages';

declare const self: DedicatedWorkerGlobalScope & { import?: (url: string) => Promise<unknown> };

// Dev only: Vite rewrites MediaPipe's runtime import() of its loader script to
// `<url>?import`, which it refuses to serve from /public. MediaPipe uses
// self.import when defined, so load the loader through a blob URL instead
// (Vite leaves blob: URLs alone). Production builds don't rewrite imports.
if (import.meta.env.DEV) {
  self.import = async (url: string) => {
    const code = await (await fetch(url)).text();
    const blobUrl = URL.createObjectURL(new Blob([code], { type: 'text/javascript' }));
    try {
      return await import(/* @vite-ignore */ blobUrl);
    } finally {
      URL.revokeObjectURL(blobUrl);
    }
  };
}

let landmarker: HolisticLandmarker | null = null;
let lastTimestamp = -1;

function post(message: FromWorker, transfer: Transferable[] = []) {
  self.postMessage(message, transfer);
}

async function create(wasmPath: string, modelPath: string, delegate: Delegate) {
  // This is a module worker, where importScripts() is unavailable, so MediaPipe
  // falls back to import() of the loader. Only the ES-module loader
  // (vision_wasm_module_internal.js) sets globalThis.ModuleFactory; the classic
  // one declares a module-scoped `var` and fails with "ModuleFactory not set."
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
  post({ type: 'error', message: `Tracker failed to start: ${String(lastError)}`, fatal: true });
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
    post({ type: 'error', message: String(err), fatal: false });
  } finally {
    bitmap.close();
  }
}

self.onmessage = (event: MessageEvent<ToWorker>) => {
  const msg = event.data;
  if (msg.type === 'init') void init(msg);
  else track(msg);
};
