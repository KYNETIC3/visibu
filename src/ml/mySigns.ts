// The user's own signs: a meaning they typed, plus a few recorded takes.
//
// Each take is stored as a "fingerprint": the word model's averaged encoder
// output for the clip (256 numbers), centred and scaled to length 1. Never
// video, never the landmark sequence. A sign is recognised by comparing a new
// clip's fingerprint with the average of a sign's takes (cosine similarity);
// the threshold comes from the model's manifest (export_signs_onnx.py calibrate()).

export interface MySign {
  id: string
  /** What the sign means, as the user typed it. Shown as a suggestion. */
  meaning: string
  /** Fingerprints of the recorded takes (centred, unit length). */
  takes: number[][]
  /** Word-model version the takes were made with; other versions cannot be compared. */
  modelVersion: string
  createdAt: number
  updatedAt: number
}

export interface Matching {
  mean: number[]
  threshold: number
}

export interface SignMatch {
  sign: MySign
  /** Cosine similarity, -1..1. At or above the threshold counts as the same sign. */
  score: number
}

function norm(v: ArrayLike<number>): number {
  let s = 0
  for (let i = 0; i < v.length; i++) s += v[i] * v[i]
  return Math.sqrt(s)
}

/** A raw fingerprint, centred with the model's mean and scaled to length 1. */
export function unitFingerprint(raw: ArrayLike<number>, mean: ArrayLike<number>): number[] {
  const out = Array.from(raw, (x, i) => x - (mean[i] ?? 0))
  const n = Math.max(norm(out), 1e-8)
  return out.map((x) => x / n)
}

export function similarity(a: ArrayLike<number>, b: ArrayLike<number>): number {
  let s = 0
  for (let i = 0; i < Math.min(a.length, b.length); i++) s += a[i] * b[i]
  return s / Math.max(norm(a) * norm(b), 1e-8)
}

/** The average of a sign's takes, scaled to length 1. */
export function prototype(takes: number[][]): number[] {
  if (takes.length === 0) return []
  const sum = new Array<number>(takes[0].length).fill(0)
  for (const t of takes) t.forEach((x, i) => (sum[i] += x))
  const n = Math.max(norm(sum), 1e-8)
  return sum.map((x) => x / n)
}

/** Signs that can be compared with fingerprints from this model version. */
export const usable = (signs: MySign[], version: string) => signs.filter((s) => s.modelVersion === version && s.takes.length > 0)

/** The user's signs that look like `fingerprint`, most similar first, at or above the threshold. */
export function similarSigns(fingerprint: number[], signs: MySign[], threshold: number, exceptId?: string): SignMatch[] {
  return signs
    .filter((s) => s.id !== exceptId)
    .map((sign) => ({ sign, score: similarity(fingerprint, prototype(sign.takes)) }))
    .filter((m) => m.score >= threshold)
    .sort((a, b) => b.score - a.score)
}

/** The closest of the user's signs, if it is close enough to count as that sign. */
export function bestMatch(fingerprint: number[], signs: MySign[], threshold: number): SignMatch | null {
  return similarSigns(fingerprint, signs, threshold)[0] ?? null
}

/** For each take, its similarity to the average of the other takes (to spot a take that went wrong). */
export function takeAgreement(takes: number[][]): number[] {
  if (takes.length < 2) return takes.map(() => 1)
  return takes.map((t, i) => similarity(t, prototype(takes.filter((_, j) => j !== i))))
}

/** Rounded copy for storage (4 decimals is far below the matching precision needed). */
export const compact = (v: number[]) => v.map((x) => Math.round(x * 1e4) / 1e4)
