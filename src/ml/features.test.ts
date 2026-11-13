import { describe, expect, it } from 'vitest'
import { beamSearch, blockedTokens, type BeamConfig } from './beam'
import { ctcGreedyDecode } from './ctc'
import { FEATURE_DIM, FEATURE_VERSION, sequenceFeatures } from './features'
import { detokenize, tokenize } from './tokenizer'
import parity from './__fixtures__/parity.json'

// Written by `python -m tools.make_parity_fixture` (in ml/) from the Python training code.
interface Fixture {
  feature_version: number
  feature_dim: number
  sequences: { aspect: number; frames: (number | null)[][][]; features: number[][] }[]
  detokenize: { tokens: string[]; text: string }[]
  tokenize: { text: string; tokens: string[] }[]
  ctc: { log_probs: number[][]; decoded: number[] }
  beam: { table: number[][]; bos: number; eos: number; config: BeamConfig; expected: { tokens: number[]; score: number; finished: boolean }[] }[]
}
const fixture = parity as unknown as Fixture

describe('Python/TypeScript parity', () => {
  it('uses the same feature version and size', () => {
    expect(FEATURE_VERSION).toBe(fixture.feature_version)
    expect(FEATURE_DIM).toBe(fixture.feature_dim)
  })

  it('computes the same features as the training code', () => {
    for (const seq of fixture.sequences) {
      const frames = seq.frames.map((f) => Float32Array.from(f.flat(), (v) => (v === null ? NaN : v)))
      const got = sequenceFeatures(frames, seq.aspect)
      let worst = 0
      seq.features.forEach((row, t) =>
        row.forEach((v, i) => {
          worst = Math.max(worst, Math.abs(got[t * FEATURE_DIM + i] - v))
        }),
      )
      expect(worst).toBeLessThan(1e-5)
    }
  })

  it('tokenises and detokenises the same way', () => {
    for (const c of fixture.tokenize) expect(tokenize(c.text)).toEqual(c.tokens)
    for (const c of fixture.detokenize) expect(detokenize(c.tokens)).toBe(c.text)
  })

  it('decodes CTC output the same way', () => {
    const lp = fixture.ctc.log_probs
    expect(ctcGreedyDecode(lp.flat(), lp.length, lp[0].length)).toEqual(fixture.ctc.decoded)
  })

  it('beam search returns the same suggestions as the training code', async () => {
    for (const c of fixture.beam) {
      const step = (prefixes: number[][]) => Float64Array.from(prefixes.flatMap((p) => c.table[p[p.length - 1]]))
      const got = await beamSearch(step, c.bos, c.eos, c.config)
      expect(got.map((h) => [h.tokens, h.finished])).toEqual(c.expected.map((h) => [h.tokens, h.finished]))
      got.forEach((h, i) => expect(h.score).toBeCloseTo(c.expected[i].score, 9))
    }
  })
})

describe('repetition blocking', () => {
  it('blocks a token that would repeat a trigram', () => {
    expect([...blockedTokens([5, 6, 7, 5, 6], 3)]).toEqual([7])
    expect(blockedTokens([5, 6, 7], 3).size).toBe(0)
    expect(blockedTokens([5, 6], 0).size).toBe(0)
  })
})
