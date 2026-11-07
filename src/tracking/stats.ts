// Rolling tracking statistics for the stage 0 go/no-go check:
// frames per second, inference time, and how often each hand is visible.

export interface FrameSample {
  at: number; // ms, from performance.now()
  inferenceMs: number;
  leftHand: boolean;
  rightHand: boolean;
}

export interface TrackingStats {
  fps: number;
  inferenceMs: number;
  bothHandsRate: number;
  anyHandRate: number;
  frames: number;
}

export class RollingStats {
  private samples: FrameSample[] = [];
  private skipped = 0;

  /**
   * @param windowMs how much recent history to summarise
   * @param warmupFrames frames to ignore at the start; MediaPipe's first call
   *   takes seconds while it compiles, which would skew the averages
   */
  constructor(
    private windowMs = 3000,
    private warmupFrames = 1,
  ) {}

  add(sample: FrameSample) {
    if (this.skipped < this.warmupFrames) {
      this.skipped++;
      return;
    }
    this.samples.push(sample);
    const cutoff = sample.at - this.windowMs;
    while (this.samples.length > 0 && this.samples[0].at < cutoff) this.samples.shift();
  }

  summary(): TrackingStats {
    const n = this.samples.length;
    if (n === 0) return { fps: 0, inferenceMs: 0, bothHandsRate: 0, anyHandRate: 0, frames: 0 };
    const span = this.samples[n - 1].at - this.samples[0].at;
    const fps = n > 1 && span > 0 ? ((n - 1) * 1000) / span : 0;
    let inference = 0;
    let both = 0;
    let any = 0;
    for (const s of this.samples) {
      inference += s.inferenceMs;
      if (s.leftHand && s.rightHand) both++;
      if (s.leftHand || s.rightHand) any++;
    }
    return { fps, inferenceMs: inference / n, bothHandsRate: both / n, anyHandRate: any / n, frames: n };
  }
}
