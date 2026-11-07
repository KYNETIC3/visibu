// Main-thread side of the tracker: owns the worker and sends it the newest
// camera frame only when it is free, so slow devices drop frames instead of
// building a backlog.
import type { Delegate, FromWorker, ToWorker } from './messages';

/** A single bad frame is skipped; this many in a row means tracking has failed. */
export const MAX_CONSECUTIVE_FRAME_ERRORS = 10;

export interface TrackerCallbacks {
  onReady(delegate: Delegate, loadMs: number): void;
  onFrame(frame: Float32Array, timestamp: number, inferenceMs: number): void;
  /** Called once tracking cannot continue. The tracker has already stopped. */
  onError(message: string): void;
}

export class Tracker {
  private worker: Worker;
  private busy = false;
  private ready = false;
  private running = false;
  private failed = false;
  private rafId = 0;
  private frameErrors = 0;

  constructor(private callbacks: TrackerCallbacks, preferGpu = true) {
    this.worker = new Worker(new URL('./tracker.worker.ts', import.meta.url), { type: 'module' });
    this.worker.onmessage = (event: MessageEvent<FromWorker>) => this.handle(event.data);
    this.worker.onerror = (event) => this.fail(event.message || 'Tracker worker crashed');
    const base = import.meta.env.BASE_URL;
    this.send({
      type: 'init',
      wasmPath: `${base}mediapipe/wasm`,
      modelPath: `${base}mediapipe/holistic_landmarker.task`,
      preferGpu,
    });
  }

  start(video: HTMLVideoElement) {
    if (this.running || this.failed) return;
    this.running = true;
    const loop = async () => {
      if (!this.running) return;
      if (this.ready && !this.busy && video.readyState >= 2) {
        this.busy = true;
        try {
          // The raw, unmirrored frame. Mirroring is done in CSS only, because
          // mirrored input swaps MediaPipe's left and right hand labels.
          const bitmap = await createImageBitmap(video);
          if (!this.running) {
            bitmap.close();
            this.busy = false;
            return;
          }
          this.send({ type: 'frame', bitmap, timestamp: performance.now() }, [bitmap]);
        } catch {
          this.busy = false;
        }
      }
      if (this.running) this.rafId = requestAnimationFrame(loop);
    };
    this.rafId = requestAnimationFrame(loop);
  }

  stop() {
    this.running = false;
    cancelAnimationFrame(this.rafId);
  }

  dispose() {
    this.stop();
    this.worker.terminate();
  }

  private send(message: ToWorker, transfer: Transferable[] = []) {
    this.worker.postMessage(message, transfer);
  }

  private fail(message: string) {
    if (this.failed) return;
    this.failed = true;
    this.dispose();
    this.callbacks.onError(message);
  }

  private handle(msg: FromWorker) {
    if (this.failed) return;
    switch (msg.type) {
      case 'ready':
        this.ready = true;
        this.callbacks.onReady(msg.delegate, msg.loadMs);
        break;
      case 'result':
        this.busy = false;
        this.frameErrors = 0;
        this.callbacks.onFrame(msg.frame, msg.timestamp, msg.inferenceMs);
        break;
      case 'error':
        this.busy = false;
        if (msg.fatal || ++this.frameErrors >= MAX_CONSECUTIVE_FRAME_ERRORS) this.fail(msg.message);
        break;
    }
  }
}
