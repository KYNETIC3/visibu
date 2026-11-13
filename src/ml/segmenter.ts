// Turns the tracker's frame stream into one landmark sequence per signed message.
//
// The model was trained on whole How2Sign sentences, so it translates a whole
// message at once rather than a sliding window (which would cut sentences and
// drop context). A message starts when a hand appears and ends when the hands
// have been down for `endPauseMs`, when the signer presses "Done", or at
// `maxFrames`. Frames are resampled to the training rate (15 fps) by holding
// the latest frame, matching the nearest-frame sampling used in training.
import { LAYOUT } from '../tracking/frame'
import { TARGET_FPS } from './features'

export interface SegmenterConfig {
  fps: number
  /** Hands down this long ends the message. */
  endPauseMs: number
  /** Frames kept from before the hands appeared. */
  prerollFrames: number
  /** Frames kept after the hands went down. */
  tailFrames: number
  minFrames: number
  maxFrames: number
}

export const DEFAULT_SEGMENTER: SegmenterConfig = {
  fps: TARGET_FPS,
  endPauseMs: 1200,
  prerollFrames: 4,
  tailFrames: 4,
  minFrames: 8,
  maxFrames: 450,
}

export type SegmentEvent =
  | { type: 'started' }
  | { type: 'segment'; frames: Float32Array[]; durationMs: number }
  | { type: 'discarded'; reason: 'too-short' }

export const handsIn = (frame: Float32Array) =>
  !Number.isNaN(frame[LAYOUT.leftHand.offset]) || !Number.isNaN(frame[LAYOUT.rightHand.offset])

export class MessageSegmenter {
  private period: number
  private nextTick: number | null = null
  private preroll: Float32Array[] = []
  private frames: Float32Array[] = []
  private lastHandsAt = 0
  private lastHandsIndex = -1
  private startedAt = 0
  private signing = false

  constructor(private cfg: SegmenterConfig = DEFAULT_SEGMENTER) {
    this.period = 1000 / cfg.fps
  }

  get isSigning(): boolean {
    return this.signing
  }

  /** Milliseconds since the current message started (0 when not signing). */
  elapsed(now: number): number {
    return this.signing ? now - this.startedAt : 0
  }

  reset(): void {
    this.nextTick = null
    this.preroll = []
    this.frames = []
    this.signing = false
    this.lastHandsIndex = -1
  }

  /** Feed one tracked frame. Returns events caused by it. */
  push(frame: Float32Array, timestamp: number): SegmentEvent[] {
    const events: SegmentEvent[] = []
    const hands = handsIn(frame)
    if (this.nextTick === null || timestamp - this.nextTick > 1000) this.nextTick = timestamp
    // Hold this frame for every 15 fps tick it covers (at most ~1 s, after a stall).
    while (this.nextTick <= timestamp) {
      this.tick(frame, hands, this.nextTick, events)
      this.nextTick += this.period
    }
    return events
  }

  /** End the current message now (the signer pressed "Done"). */
  finish(): SegmentEvent[] {
    if (!this.signing) return []
    return [this.end()]
  }

  private tick(frame: Float32Array, hands: boolean, t: number, events: SegmentEvent[]) {
    if (!this.signing) {
      this.preroll.push(frame)
      if (this.preroll.length > this.cfg.prerollFrames + 1) this.preroll.shift()
      if (hands) {
        this.signing = true
        this.startedAt = t
        this.lastHandsAt = t
        this.frames = this.preroll
        this.preroll = []
        this.lastHandsIndex = this.frames.length - 1
        events.push({ type: 'started' })
      }
      return
    }
    this.frames.push(frame)
    if (hands) {
      this.lastHandsAt = t
      this.lastHandsIndex = this.frames.length - 1
    }
    if (t - this.lastHandsAt >= this.cfg.endPauseMs || this.frames.length >= this.cfg.maxFrames) {
      events.push(this.end())
    }
  }

  private end(): SegmentEvent {
    const keep = Math.min(this.frames.length, this.lastHandsIndex + 1 + this.cfg.tailFrames)
    const frames = this.frames.slice(0, keep)
    const durationMs = (frames.length * 1000) / this.cfg.fps
    this.signing = false
    this.frames = []
    this.preroll = []
    this.lastHandsIndex = -1
    if (frames.length < this.cfg.minFrames) return { type: 'discarded', reason: 'too-short' }
    return { type: 'segment', frames, durationMs }
  }
}
