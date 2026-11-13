// Beam search with n-gram repetition blocking, returning the n best sentences.
// Line-for-line port of ml/models/beam.py; src/ml/features.test.ts checks both
// give identical results on the cases in __fixtures__/parity.json.
//
// A hypothesis's score is its mean token log-prob (<eos> included when it
// finished). exp(score) is the model score shown in the app, not an accuracy.

export interface BeamConfig {
  beam: number
  nBest: number
  maxLen: number
  noRepeatNgram: number
  /** Token ids never generated (pad, bos, unk). */
  banned: number[]
}

export interface Hypothesis {
  /** Without <bos> / <eos>. */
  tokens: number[]
  score: number
  finished: boolean
}

/** Next-token log-probs for each prefix, row-major [prefixes.length, V]. */
export type StepFn = (prefixes: number[][]) => Promise<Float32Array | Float64Array> | Float32Array | Float64Array

const NEG = -1e9

export function blockedTokens(tokens: number[], n: number): Set<number> {
  const out = new Set<number>()
  if (n <= 0 || tokens.length < n - 1) return out
  if (n === 1) return new Set(tokens)
  const start = tokens.length - (n - 1)
  for (let i = 0; i + n <= tokens.length; i++) {
    let match = true
    for (let j = 0; j < n - 1; j++) {
      if (tokens[i + j] !== tokens[start + j]) {
        match = false
        break
      }
    }
    if (match) out.add(tokens[i + n - 1])
  }
  return out
}

/** Indices of the k largest values, highest first, lower index first on ties. */
function topK(row: Float64Array, k: number): number[] {
  const best: number[] = []
  for (let i = 0; i < row.length; i++) {
    if (best.length === k && row[i] <= row[best[k - 1]]) continue
    let pos = best.length
    while (pos > 0 && row[best[pos - 1]] < row[i]) pos--
    best.splice(pos, 0, i)
    if (best.length > k) best.pop()
  }
  return best
}

export async function beamSearch(step: StepFn, bos: number, eos: number, cfg: BeamConfig): Promise<Hypothesis[]> {
  let beams: { tokens: number[]; total: number }[] = [{ tokens: [], total: 0 }]
  const finished: Hypothesis[] = []
  for (let t = 0; t < cfg.maxLen; t++) {
    const lp = await step(beams.map((b) => [bos, ...b.tokens]))
    const V = lp.length / beams.length
    const cands: { total: number; b: number; k: number }[] = []
    beams.forEach((beam, b) => {
      const row = Float64Array.from(lp.subarray(b * V, (b + 1) * V))
      for (const k of cfg.banned) row[k] = NEG
      for (const k of blockedTokens(beam.tokens, cfg.noRepeatNgram)) if (k !== eos) row[k] = NEG
      if (beam.tokens.length === 0) row[eos] = NEG // an empty sentence is never a useful suggestion
      for (const k of topK(row, cfg.beam * 2)) if (row[k] > NEG / 2) cands.push({ total: beam.total + row[k], b, k })
    })
    cands.sort((x, y) => y.total - x.total || x.b - y.b || x.k - y.k)
    const next: typeof beams = []
    for (const c of cands) {
      const toks = beams[c.b].tokens
      if (c.k === eos) finished.push({ tokens: [...toks], score: c.total / (toks.length + 1), finished: true })
      else next.push({ tokens: [...toks, c.k], total: c.total })
      if (next.length === cfg.beam) break
    }
    beams = next
    if (beams.length === 0 || finished.length >= cfg.beam) break
  }
  // Finished sentences first; sentences cut off at maxLen only fill any remaining slots.
  const byScore = (a: Hypothesis, b: Hypothesis) => b.score - a.score
  const live = beams.map((b) => ({ tokens: b.tokens, score: b.total / Math.max(1, b.tokens.length), finished: false })).sort(byScore)
  const pool = [...finished.sort(byScore), ...live]
  const out: Hypothesis[] = []
  const seen = new Set<string>()
  for (const h of pool) {
    const key = h.tokens.join(',')
    if (seen.has(key) || h.tokens.length === 0) continue
    seen.add(key)
    out.push(h)
    if (out.length === cfg.nBest) break
  }
  return out
}
