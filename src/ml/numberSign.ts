// Choose a suggestion by holding up the ASL number 1, 2 or 3.
//
// This is a fixed control gesture, like pressing a button, not sign
// recognition: it only reads which fingers are straight. It never translates
// anything and never sends anything. Picking a suggestion only puts its text
// in the review box; the signer still approves the message with the button.
//
// ASL numbers, from the hand's landmarks (MediaPipe hand: 0 wrist, 1-4 thumb,
// 5-8 index, 9-12 middle, 13-16 ring, 17-20 pinky):
//   1  index up; middle, ring, pinky folded; thumb not out
//   2  index + middle up and apart; ring, pinky folded; thumb not out
//   3  thumb, index, middle out; ring, pinky folded
// Angles and distances are scaled by palm size, so they work at any distance
// and with either hand, mirrored or not.
import { LAYOUT } from '../tracking/frame'

export type PickNumber = 1 | 2 | 3
type State = 'out' | 'in' | 'unsure'

type V3 = [number, number, number]

function point(frame: Float32Array, offset: number, i: number, aspect: number): V3 {
  const o = offset + i * 3
  return [frame[o] * aspect, frame[o + 1], frame[o + 2] * aspect]
}

const dist = (a: V3, b: V3) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2])

/** Straight finger: tip-to-knuckle distance close to the summed bone lengths, and the tip past the middle joint. */
function finger(p: V3[], mcp: number): State {
  const [m, pip, dip, tip] = [p[mcp], p[mcp + 1], p[mcp + 2], p[mcp + 3]]
  const bones = dist(m, pip) + dist(pip, dip) + dist(dip, tip)
  const straightness = dist(m, tip) / bones
  const pastPip = dist(p[0], tip) > dist(p[0], pip)
  if (straightness > 0.9 && pastPip) return 'out'
  if (straightness < 0.65 || !pastPip) return 'in'
  return 'unsure'
}

function thumb(p: V3[], palm: number): State {
  const tip = p[4]
  // Tucked: the tip rests on the folded fingers or the palm.
  const nearest = Math.min(...[6, 10, 14, 9, 13].map((j) => dist(tip, p[j])))
  if (nearest < 0.35 * palm) return 'in'
  if (dist(tip, p[5]) > 0.6 * palm && dist(tip, p[9]) > 0.75 * palm) return 'out'
  return 'unsure'
}

/** The number one hand shows, or null. `offset` is the hand's offset in the packed frame. */
export function handNumber(frame: Float32Array, offset: number, aspect: number): PickNumber | null {
  if (Number.isNaN(frame[offset])) return null
  const p = Array.from({ length: 21 }, (_, i) => point(frame, offset, i, aspect))
  const palm = dist(p[0], p[9])
  if (!(palm > 1e-4)) return null
  const [index, middle, ring, pinky] = [5, 9, 13, 17].map((m) => finger(p, m))
  const th = thumb(p, palm)
  if (ring !== 'in' || pinky !== 'in' || index !== 'out') return null
  if (middle === 'in' && th !== 'out') return 1
  if (middle !== 'out') return null
  // Fingers held together (tips no wider apart than the knuckles) is not 2 or 3.
  if (dist(p[8], p[12]) < 1.4 * dist(p[5], p[9])) return null
  if (th === 'out') return 3
  if (th === 'in') return 2
  return null
}

/** The number shown in a frame. Both hands must agree when both are visible. */
export function frameNumber(frame: Float32Array, aspect: number): PickNumber | null {
  const l = handNumber(frame, LAYOUT.leftHand.offset, aspect)
  const r = handNumber(frame, LAYOUT.rightHand.offset, aspect)
  if (l && r) return l === r ? l : null
  return l ?? r
}

/**
 * Whether the visible hand is being held still. Signing moves the hands; a
 * number held up to choose a suggestion does not. Without this, handshapes
 * that look like 1, 2 or 3 in the middle of ordinary signing could choose.
 */
export class HandStillness {
  private history: { t: number; x: number; y: number }[] = []

  constructor(
    /** Look back this far. */
    private windowMs = 300,
    /** Allowed wrist movement within the window, as a share of the palm length. */
    private maxMove = 0.25,
  ) {}

  update(frame: Float32Array, t: number, aspect: number): boolean {
    const off = Number.isNaN(frame[LAYOUT.rightHand.offset]) ? LAYOUT.leftHand.offset : LAYOUT.rightHand.offset
    if (Number.isNaN(frame[off])) {
      this.history = []
      return false
    }
    const wrist = point(frame, off, 0, aspect)
    const palm = dist(wrist, point(frame, off, 9, aspect))
    this.history.push({ t, x: wrist[0], y: wrist[1] })
    while (this.history.length && this.history[0].t < t - this.windowMs) this.history.shift()
    if (this.history.length < 2 || t - this.history[0].t < this.windowMs * 0.6) return false
    return this.history.every((h) => Math.hypot(h.x - wrist[0], h.y - wrist[1]) <= this.maxMove * palm)
  }

  reset(): void {
    this.history = []
  }
}

export interface PickerConfig {
  /** How long a number must be held. */
  holdMs: number
  /** Tracking glitches shorter than this (a missed or misread frame) do not reset the hold. */
  graceMs: number
}

export const DEFAULT_PICKER: PickerConfig = { holdMs: 800, graceMs: 250 }

export interface PickerUpdate {
  /** The number being held, if any. */
  candidate: PickNumber | null
  /** 0..1 towards selecting it. */
  progress: number
  /** Set on the frame where a number is selected. */
  picked: PickNumber | null
}

/**
 * Turns per-frame numbers into one selection per deliberate hold. After a
 * selection, the hand must drop or change before anything else is selected,
 * so holding "2" selects it once.
 */
export class NumberSignPicker {
  private cand: PickNumber | null = null
  private start = 0
  private last = 0
  private lastPicked: PickNumber | null = null

  constructor(
    private max: number,
    private cfg: PickerConfig = DEFAULT_PICKER,
  ) {}

  push(n: PickNumber | null, t: number): PickerUpdate {
    if (n !== null && n > this.max) n = null
    if (n !== null && n === this.cand) {
      this.last = t
    } else if (this.cand === null || t - this.last > this.cfg.graceMs) {
      this.cand = n
      this.start = this.last = t
    }
    if (this.cand !== this.lastPicked) this.lastPicked = null // re-arm once the hand changes
    if (this.cand === null) return { candidate: null, progress: 0, picked: null }
    if (this.lastPicked === this.cand) return { candidate: this.cand, progress: 1, picked: null }
    const progress = Math.min(1, (this.last - this.start) / this.cfg.holdMs)
    if (progress >= 1) {
      this.lastPicked = this.cand
      return { candidate: this.cand, progress: 1, picked: this.cand }
    }
    return { candidate: this.cand, progress, picked: null }
  }

  reset(): void {
    this.cand = null
    this.lastPicked = null
  }
}
