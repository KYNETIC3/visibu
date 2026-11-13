import { describe, expect, it } from 'vitest'
import { withMySign, type SignProposal } from '../lib/signRecognizer'
import { bestMatch, prototype, similarSigns, similarity, takeAgreement, unitFingerprint, usable, type MySign } from './mySigns'

const D = 8
const basis = (i: number, noise = 0, seed = 1): number[] =>
  Array.from({ length: D }, (_, k) => (k === i ? 1 : 0) + noise * Math.sin(seed * (k + 1) * 12.9898))
const mean = new Array(D).fill(0)
const sign = (id: string, meaning: string, axis: number, version = 'v1'): MySign => ({
  id, meaning, modelVersion: version, createdAt: 0, updatedAt: 0,
  takes: [1, 2, 3].map((s) => unitFingerprint(basis(axis, 0.1, s), mean)),
})

describe('fingerprints', () => {
  it('centres and scales to length 1', () => {
    const u = unitFingerprint([3, 4, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0])
    expect(Math.hypot(...u)).toBeCloseTo(1)
    expect(similarity(u, [2, 4, 0, 0, 0, 0, 0, 0])).toBeCloseTo(1)
  })

  it('averages takes into one prototype', () => {
    expect(prototype([[1, 0], [0, 1]]).map((x) => +x.toFixed(4))).toEqual([0.7071, 0.7071])
  })

  it('flags a take that does not agree with the others', () => {
    const takes = [basis(0, 0.05, 1), basis(0, 0.05, 2), basis(3, 0.05, 3)].map((t) => unitFingerprint(t, mean))
    const agree = takeAgreement(takes)
    expect(agree[2]).toBeLessThan(0.3)
    expect(agree[0]).toBeGreaterThan(0.4)
  })
})

describe('matching my signs', () => {
  const mine = [sign('a', 'Can I get a latte?', 0), sign('b', 'See you soon', 1)]

  it('finds the sign a new take belongs to, and nothing for an unknown sign', () => {
    expect(bestMatch(unitFingerprint(basis(1, 0.1, 7), mean), mine, 0.42)?.sign.meaning).toBe('See you soon')
    expect(bestMatch(unitFingerprint(basis(5, 0.1, 7), mean), mine, 0.42)).toBeNull()
  })

  it('lists look-alikes for the "already have this sign" question, skipping the sign being re-recorded', () => {
    const fp = prototype(sign('new', 'x', 0).takes)
    expect(similarSigns(fp, mine, 0.42).map((m) => m.sign.id)).toEqual(['a'])
    expect(similarSigns(fp, mine, 0.42, 'a')).toEqual([])
  })

  it('ignores signs recorded with another model version', () => {
    expect(usable([sign('a', 'x', 0, 'old'), sign('b', 'y', 1)], 'v1').map((s) => s.id)).toEqual(['b'])
  })
})

describe('using my signs in translation', () => {
  const words = { signs: 250, testTop1: 0.6, version: 'v1', matching: { mean, threshold: 0.42 } }
  const base: SignProposal = {
    id: 1, text: 'Dog.', confidence: 0.9, lowConfidence: false,
    suggestions: [{ text: 'Dog.', confidence: 0.9, lowConfidence: false, source: 'sign', sign: 'dog' }],
    embedding: Float32Array.from(basis(0, 0.1, 9)), glosses: [], durationMs: 900, latencyMs: 30,
    model: { name: 'm', version: '1', stage: 'research', ctcTargets: 'english-words', licence: '', dataset: '', backend: 'wasm', loadMs: 1, words },
  }

  it('puts my meaning first when the sign matches', () => {
    const p = withMySign(base, [sign('a', 'Can I get a latte?', 0)])
    expect(p.text).toBe('Can I get a latte?')
    expect(p.suggestions[0]).toMatchObject({ source: 'mine', text: 'Can I get a latte?' })
    expect(p.suggestions[1].text).toBe('Dog.')
  })

  it('leaves the translation alone when nothing matches', () => {
    expect(withMySign(base, [sign('b', 'See you soon', 1)])).toBe(base)
  })
})
