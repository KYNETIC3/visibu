// Landmark features for the sign translation model. A line-for-line port of
// ml/common/features.py; src/ml/features.test.ts checks both give the same
// numbers (fixture written by `python -m tools.make_parity_fixture`).
// Change both files together and bump FEATURE_VERSION.
import { FRAME_LENGTH, HAND_POINTS, LAYOUT } from '../tracking/frame'

export const FEATURE_VERSION = 1
export const TARGET_FPS = 15

const POSE_SEL = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16]
const LIPS = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
  78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]
const LEFT_BROW = [276, 283, 282, 295, 285, 300, 293, 334, 296, 336]
const RIGHT_BROW = [46, 53, 52, 65, 55, 70, 63, 105, 66, 107]
const EYES = [33, 133, 159, 145, 362, 263, 386, 374]
const NOSE = [1, 4]
const OVAL = [10, 152, 234, 454]
const FACE_SEL = [...LIPS, ...LEFT_BROW, ...RIGHT_BROW, ...EYES, ...NOSE, ...OVAL]

const N_POSE = POSE_SEL.length
const N_FACE = FACE_SEL.length
const N_GLOBAL = N_POSE + N_FACE + 2 * HAND_POINTS
export const FEATURE_DIM = N_GLOBAL * 2 + 2 * HAND_POINTS * 3 + N_FACE * 2 + N_GLOBAL * 2 + 4

const FALLBACK_CENTER_Y = 0.6
const FALLBACK_SCALE = 0.35
const MIN_SCALE = 1e-3
const CLIP = 10
const FACE_NOSE_TIP = 1
const FACE_EYE_OUTER_R = 33
const FACE_EYE_OUTER_L = 263

const POSE = LAYOUT.pose.offset / 3
const LEFT = LAYOUT.leftHand.offset / 3
const RIGHT = LAYOUT.rightHand.offset / 3
const FACE = LAYOUT.face.offset / 3

/** Causal state carried from frame to frame within one message. */
export class FeatureState {
  center: [number, number] | null = null
  scale: number | null = null
  prevGlobal: Float64Array | null = null
  prevPresence: Uint8Array | null = null
  constructor(readonly aspect: number) {}
}

/** Features for one packed frame (FRAME_LENGTH floats). Updates `state`. */
export function frameFeatures(frame: Float32Array, state: FeatureState, out = new Float32Array(FEATURE_DIM)): Float32Array {
  if (frame.length !== FRAME_LENGTH) throw new Error(`frame has ${frame.length} values, expected ${FRAME_LENGTH}`)
  const a = state.aspect
  const X = (p: number) => frame[p * 3] * a
  const Y = (p: number) => frame[p * 3 + 1]
  const Z = (p: number) => frame[p * 3 + 2] * a
  const present = (p: number) => !Number.isNaN(frame[p * 3])

  const poseOk = present(POSE)
  const faceOk = present(FACE)
  const leftOk = present(LEFT)
  const rightOk = present(RIGHT)

  if (poseOk) {
    const ls = POSE + 11
    const rs = POSE + 12
    const s = Math.hypot(X(ls) - X(rs), Y(ls) - Y(rs))
    if (s >= MIN_SCALE) {
      state.center = [(X(ls) + X(rs)) / 2, (Y(ls) + Y(rs)) / 2]
      state.scale = s
    }
  }
  let cx: number, cy: number, sc: number
  if (state.center === null || state.scale === null) {
    cx = 0.5 * a
    cy = FALLBACK_CENTER_Y
    sc = FALLBACK_SCALE
  } else {
    ;[cx, cy] = state.center
    sc = state.scale
  }

  const vals = new Float64Array(FEATURE_DIM)
  const g = new Float64Array(N_GLOBAL * 2)
  const pres = new Uint8Array(N_GLOBAL)
  const putGlobal = (row: number, idx: number[], start: number, ok: boolean) => {
    if (!ok) return
    idx.forEach((i, k) => {
      g[(row + k) * 2] = (X(start + i) - cx) / sc
      g[(row + k) * 2 + 1] = (Y(start + i) - cy) / sc
      pres[row + k] = 1
    })
  }
  const handIdx = Array.from({ length: HAND_POINTS }, (_, i) => i)
  putGlobal(0, POSE_SEL, POSE, poseOk)
  putGlobal(N_POSE, FACE_SEL, FACE, faceOk)
  putGlobal(N_POSE + N_FACE, handIdx, LEFT, leftOk)
  putGlobal(N_POSE + N_FACE + HAND_POINTS, handIdx, RIGHT, rightOk)

  let o = 0
  vals.set(g, o)
  o += N_GLOBAL * 2

  for (const [start, ok] of [[LEFT, leftOk], [RIGHT, rightOk]] as const) {
    if (ok) {
      const hs = Math.max(Math.hypot(X(start + 9) - X(start), Y(start + 9) - Y(start)), MIN_SCALE)
      for (let i = 0; i < HAND_POINTS; i++) {
        vals[o + i * 3] = (X(start + i) - X(start)) / hs
        vals[o + i * 3 + 1] = (Y(start + i) - Y(start)) / hs
        vals[o + i * 3 + 2] = (Z(start + i) - Z(start)) / hs
      }
    }
    o += HAND_POINTS * 3
  }

  if (faceOk) {
    const nose = FACE + FACE_NOSE_TIP
    const er = FACE + FACE_EYE_OUTER_R
    const el = FACE + FACE_EYE_OUTER_L
    const fs = Math.max(Math.hypot(X(er) - X(el), Y(er) - Y(el)), MIN_SCALE)
    FACE_SEL.forEach((i, k) => {
      vals[o + k * 2] = (X(FACE + i) - X(nose)) / fs
      vals[o + k * 2 + 1] = (Y(FACE + i) - Y(nose)) / fs
    })
  }
  o += N_FACE * 2

  if (state.prevGlobal && state.prevPresence) {
    for (let p = 0; p < N_GLOBAL; p++) {
      if (pres[p] && state.prevPresence[p]) {
        vals[o + p * 2] = g[p * 2] - state.prevGlobal[p * 2]
        vals[o + p * 2 + 1] = g[p * 2 + 1] - state.prevGlobal[p * 2 + 1]
      }
    }
  }
  o += N_GLOBAL * 2

  vals[o] = poseOk ? 1 : 0
  vals[o + 1] = faceOk ? 1 : 0
  vals[o + 2] = leftOk ? 1 : 0
  vals[o + 3] = rightOk ? 1 : 0

  state.prevGlobal = g
  state.prevPresence = pres
  // A part detected with some points missing must not poison the model input (NaN becomes 0).
  for (let i = 0; i < FEATURE_DIM; i++) out[i] = Number.isNaN(vals[i]) ? 0 : Math.min(CLIP, Math.max(-CLIP, vals[i]))
  return out
}

/** Features for a whole message: `frames` packed frames -> Float32Array [T * FEATURE_DIM]. */
export function sequenceFeatures(frames: Float32Array[], aspect: number): Float32Array {
  const state = new FeatureState(aspect)
  const out = new Float32Array(frames.length * FEATURE_DIM)
  frames.forEach((f, t) => frameFeatures(f, state, out.subarray(t * FEATURE_DIM, (t + 1) * FEATURE_DIM)))
  return out
}
