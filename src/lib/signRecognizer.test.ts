import { describe, expect, it } from 'vitest';
import { packFrame, type Point } from '../tracking/frame';
import {
  DEFAULT_CONFIG,
  HeuristicSignClassifier,
  MediaPipeSignRecognizer,
  type SignEvent,
} from './signRecognizer';

type Shape = 'open' | 'fist' | 'point';

/** Builds a 21-point hand with its wrist at (x, y). */
function hand(shape: Shape, x: number, y: number): Point[] {
  const pts: Point[] = Array.from({ length: 21 }, () => ({ x, y, z: 0 }));
  // finger joints: [mcp, pip, dip, tip] per finger
  const fingers = [
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16],
    [17, 18, 19, 20],
  ];
  fingers.forEach((f, i) => {
    const dx = (i - 2) * 0.02;
    const extended = shape === 'open' || (shape === 'point' && (i === 1 || i === 2));
    f.forEach((idx, j) => {
      const reach = extended ? 0.04 + j * 0.03 : j < 2 ? 0.04 + j * 0.03 : 0.04;
      pts[idx] = { x: x + dx, y: y - reach, z: 0 };
    });
  });
  return pts;
}

function run(frames: Point[][]): string[] {
  const c = new HeuristicSignClassifier();
  return frames.map((f) => c.classify(f, null)?.word).filter((w): w is string => !!w);
}

const steps = (n: number, fn: (i: number) => Point[]) => Array.from({ length: n }, (_, i) => fn(i));

describe('HeuristicSignClassifier', () => {
  it('ignores a still hand', () => {
    expect(run(steps(20, () => hand('open', 0.5, 0.6)))).toEqual([]);
  });

  it('recognises HELLO from an open hand rising', () => {
    expect(run(steps(15, (i) => hand('open', 0.5, 0.7 - i * 0.012)))).toContain('HELLO');
  });

  it('recognises YES from a nodding fist', () => {
    const words = run(steps(15, (i) => hand('fist', 0.5, 0.6 + (i % 4 < 2 ? 0 : 0.04))));
    expect(words).toContain('YES');
  });

  it('recognises NO from a wagging pointed finger', () => {
    const words = run(steps(15, (i) => hand('point', 0.5 + (i % 4 < 2 ? 0 : 0.04), 0.6)));
    expect(words).toContain('NO');
  });

  it('does not call every raised open hand THANK YOU', () => {
    expect(run(steps(20, () => hand('open', 0.5, 0.3)))).toEqual([]);
  });
});

describe('MediaPipeSignRecognizer', () => {
  it('emits a held sign once, not repeatedly', () => {
    const r = new MediaPipeSignRecognizer(new HeuristicSignClassifier(), DEFAULT_CONFIG);
    const events: SignEvent[] = [];
    r.start({} as HTMLVideoElement, { onSign: (e) => events.push(e), onHands: () => {} });
    // keep nodding a fist for many frames
    for (let i = 0; i < 60; i++) {
      const h = hand('fist', 0.5, 0.6 + (i % 4 < 2 ? 0 : 0.04));
      r.processFrame(
        packFrame({ poseLandmarks: [], leftHandLandmarks: [], rightHandLandmarks: [h], faceLandmarks: [] }),
      );
    }
    r.stop();
    expect(events.map((e) => e.word)).toEqual(['YES']);
  });
});
