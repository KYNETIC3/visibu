// Stage 0 go/no-go screen: starts the camera, runs Holistic tracking in the
// worker, draws the landmarks, and reports whether this device is fast enough.
import { useEffect, useRef, useState } from 'react';
import { Tracker } from '../tracking/tracker';
import { hasPart, LAYOUT, type Part } from '../tracking/frame';
import { RollingStats, type TrackingStats } from '../tracking/stats';
import type { Delegate } from '../tracking/messages';

const TARGET_FPS = 15;

type Status =
  | { kind: 'idle' }
  | { kind: 'starting' }
  | { kind: 'running'; delegate: Delegate; loadMs: number }
  | { kind: 'error'; message: string };

const PART_COLOURS: Partial<Record<Part, string>> = {
  pose: '#8a8f98',
  leftHand: '#e8590c',
  rightHand: '#1c7ed6',
};

export function TrackingCheck() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const trackerRef = useRef<Tracker | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const ctxRef = useRef<CanvasRenderingContext2D | null>(null);
  const statsRef = useRef<RollingStats>(null!);
  const [status, setStatus] = useState<Status>({ kind: 'idle' });
  const [stats, setStats] = useState<TrackingStats | null>(null);

  useEffect(() => () => stop(), []);

  async function start() {
    stop();
    setStats(null);
    setStatus({ kind: 'starting' });
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' },
        audio: false,
      });
      streamRef.current = stream;
      const video = videoRef.current!;
      video.srcObject = stream;
      await video.play();

      statsRef.current = new RollingStats();
      const tracker = new Tracker({
        onReady: (delegate, loadMs) => {
          setStatus({ kind: 'running', delegate, loadMs });
          tracker.start(video);
        },
        onFrame: (frame, _timestamp, inferenceMs) => {
          statsRef.current.add({
            at: performance.now(),
            inferenceMs,
            leftHand: hasPart(frame, 'leftHand'),
            rightHand: hasPart(frame, 'rightHand'),
          });
          draw(frame);
        },
        // The tracker has already stopped itself; release the camera too.
        onError: (message) => {
          stop();
          setStatus({ kind: 'error', message });
        },
      });
      trackerRef.current = tracker;
    } catch (err) {
      stop();
      setStatus({ kind: 'error', message: cameraErrorMessage(err) });
    }
  }

  function stop() {
    trackerRef.current?.dispose();
    trackerRef.current = null;
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    const video = videoRef.current;
    if (video) video.srcObject = null;
  }

  // Refresh the numbers twice a second rather than on every frame.
  useEffect(() => {
    if (status.kind !== 'running') return;
    const id = setInterval(() => setStats(statsRef.current.summary()), 500);
    return () => clearInterval(id);
  }, [status.kind]);

  function draw(frame: Float32Array) {
    const canvas = canvasRef.current;
    const video = videoRef.current;
    if (!canvas || !video) return;
    if (canvas.width !== video.videoWidth) canvas.width = video.videoWidth;
    if (canvas.height !== video.videoHeight) canvas.height = video.videoHeight;
    const ctx = (ctxRef.current ??= canvas.getContext('2d'));
    if (!ctx) return;
    const { width, height } = canvas;
    ctx.clearRect(0, 0, width, height);
    // Read x and y straight from the packed frame to avoid allocating points.
    for (const [part, colour] of Object.entries(PART_COLOURS) as [Part, string][]) {
      if (!hasPart(frame, part)) continue;
      const { offset, points } = LAYOUT[part];
      const radius = part === 'pose' ? 3 : 4;
      ctx.fillStyle = colour;
      ctx.beginPath();
      for (let i = 0; i < points; i++) {
        const x = frame[offset + i * 3] * width;
        const y = frame[offset + i * 3 + 1] * height;
        ctx.moveTo(x + radius, y);
        ctx.arc(x, y, radius, 0, Math.PI * 2);
      }
      ctx.fill();
    }
  }

  const fast = stats && stats.frames > 10 ? stats.fps >= TARGET_FPS : null;

  return (
    <section className="tracking-check" aria-labelledby="tracking-heading">
      <h2 id="tracking-heading">Tracking check</h2>
      <p className="lede">
        Checks whether this device can track hands, face and body fast enough. Video stays on this
        device; only landmark numbers are used.
      </p>

      <div className="stage">
        <video ref={videoRef} className="mirror" playsInline muted aria-label="Your camera" />
        <canvas ref={canvasRef} className="mirror overlay" aria-hidden="true" />
        {status.kind === 'idle' && (
          <button className="primary center" onClick={start}>
            Start camera
          </button>
        )}
        {status.kind === 'starting' && <p className="center notice">Loading tracker…</p>}
      </div>

      {status.kind === 'error' && (
        <div className="error-box">
          <p className="error" role="alert">
            {status.message}
          </p>
          <button className="primary" onClick={start}>
            Try again
          </button>
        </div>
      )}

      {status.kind === 'running' && (
        <dl className="stats" aria-live="polite">
          <div>
            <dt>Speed</dt>
            <dd>
              {stats ? `${stats.fps.toFixed(1)} FPS` : '…'}
              {fast !== null && (
                <span className={fast ? 'pass' : 'fail'}>
                  {fast ? ' Fast enough' : ` Below ${TARGET_FPS} FPS`}
                </span>
              )}
            </dd>
          </div>
          <div>
            <dt>Tracking time per frame</dt>
            <dd>{stats ? `${stats.inferenceMs.toFixed(0)} ms` : '…'}</dd>
          </div>
          <div>
            <dt>Both hands visible</dt>
            <dd>{stats ? `${Math.round(stats.bothHandsRate * 100)}% of frames` : '…'}</dd>
          </div>
          <div>
            <dt>At least one hand visible</dt>
            <dd>{stats ? `${Math.round(stats.anyHandRate * 100)}% of frames` : '…'}</dd>
          </div>
          <div>
            <dt>Processor</dt>
            <dd>{status.delegate === 'GPU' ? 'Graphics card (GPU)' : 'Main processor (CPU)'}</dd>
          </div>
          <div>
            <dt>Start-up time</dt>
            <dd>{(status.loadMs / 1000).toFixed(1)} s</dd>
          </div>
        </dl>
      )}
    </section>
  );
}

function cameraErrorMessage(err: unknown): string {
  const name = err instanceof DOMException ? err.name : '';
  if (name === 'NotAllowedError') return 'Camera access was blocked. Allow the camera in your browser settings, then try again.';
  if (name === 'NotFoundError') return 'No camera was found on this device.';
  if (!window.isSecureContext) return 'The camera needs a secure (HTTPS) page.';
  return `The camera could not start: ${String(err)}`;
}
