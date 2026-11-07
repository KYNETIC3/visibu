// Messages between the main thread and the tracker worker.

export type Delegate = 'GPU' | 'CPU';

export type ToWorker =
  | { type: 'init'; wasmPath: string; modelPath: string; preferGpu: boolean }
  | { type: 'frame'; bitmap: ImageBitmap; timestamp: number };

export type FromWorker =
  | { type: 'ready'; delegate: Delegate; loadMs: number }
  | { type: 'result'; frame: Float32Array; timestamp: number; inferenceMs: number }
  | { type: 'error'; message: string; fatal: boolean };
