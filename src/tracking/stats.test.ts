import { describe, expect, it } from 'vitest';
import { RollingStats } from './stats';

describe('RollingStats', () => {
  it('is empty before any frames', () => {
    expect(new RollingStats().summary().fps).toBe(0);
  });

  it('measures frame rate, inference time and hand visibility', () => {
    const stats = new RollingStats(3000, 0);
    // 21 frames, 50 ms apart = 20 FPS. Every other frame shows both hands.
    for (let i = 0; i <= 20; i++) {
      stats.add({ at: i * 50, inferenceMs: 30, leftHand: true, rightHand: i % 2 === 0 });
    }
    const s = stats.summary();
    expect(s.fps).toBeCloseTo(20);
    expect(s.inferenceMs).toBe(30);
    expect(s.bothHandsRate).toBeCloseTo(11 / 21);
    expect(s.anyHandRate).toBe(1);
  });

  it('drops frames older than the window', () => {
    const stats = new RollingStats(1000, 0);
    stats.add({ at: 0, inferenceMs: 500, leftHand: false, rightHand: false });
    stats.add({ at: 2000, inferenceMs: 10, leftHand: true, rightHand: true });
    stats.add({ at: 2100, inferenceMs: 10, leftHand: true, rightHand: true });
    expect(stats.summary()).toMatchObject({ frames: 2, inferenceMs: 10, bothHandsRate: 1 });
  });

  it('ignores warm-up frames', () => {
    const stats = new RollingStats(3000, 1);
    stats.add({ at: 0, inferenceMs: 5700, leftHand: false, rightHand: false });
    stats.add({ at: 5700, inferenceMs: 10, leftHand: true, rightHand: true });
    stats.add({ at: 5750, inferenceMs: 10, leftHand: true, rightHand: true });
    expect(stats.summary()).toMatchObject({ frames: 2, inferenceMs: 10, fps: 20 });
  });
});
