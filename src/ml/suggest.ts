// Combines the two models' outputs into the (at most 3) suggestions the signer sees.
//
//  * word model   one of 250 everyday signs, offered as short sentences (signPhrases.ts)
//  * sentence model   How2Sign translation, whole sentences
//
// A short stretch of signing is most likely a single sign, so the word model's
// suggestions come first; a long one is most likely a sentence. The two models'
// scores are not on the same scale, so they are never compared directly.
import { signPhrases, signWord } from './signPhrases'

export interface RankedSuggestion {
  text: string
  /** The producing model's score, 0..1. Not a measured accuracy. */
  confidence: number
  source: 'sign' | 'sentence'
  /** For word suggestions: the sign that was recognised, in plain English. */
  sign?: string
}

export interface SignGuess {
  label: string
  prob: number
}

/** Signing up to this long is treated as a single sign first. */
export const SINGLE_SIGN_MS = 3500

/** Runner-up signs below this probability are not worth showing. */
export const MIN_RUNNER_UP = 0.05

/** Word-model suggestions: the top sign's phrases, then the next likely signs' first phrase. */
export function wordSuggestions(guesses: SignGuess[], max = 3): RankedSuggestion[] {
  const out: RankedSuggestion[] = []
  guesses.forEach((g, rank) => {
    if (rank > 0 && g.prob < MIN_RUNNER_UP) return
    const phrases = signPhrases(g.label)
    for (const text of rank === 0 ? phrases.slice(0, 2) : phrases.slice(0, 1)) {
      out.push({ text, confidence: g.prob, source: 'sign', sign: signWord(g.label) })
    }
  })
  return out.slice(0, max)
}

export function mergeSuggestions(
  words: RankedSuggestion[],
  sentences: RankedSuggestion[],
  durationMs: number,
  max = 3,
): RankedSuggestion[] {
  const short = durationMs <= SINGLE_SIGN_MS
  // Keep at least one of each kind when both exist, so neither model is hidden.
  const first = short ? words : sentences
  const second = short ? sentences : words
  const ordered = [...first.slice(0, max - (second.length ? 1 : 0)), ...second, ...first.slice(max - 1)]
  const seen = new Set<string>()
  const out: RankedSuggestion[] = []
  for (const s of ordered) {
    const key = s.text.trim().toLowerCase()
    if (!key || seen.has(key)) continue
    seen.add(key)
    out.push(s)
    if (out.length === max) break
  }
  return out
}

/** Joins accepted pieces into one message, adding a full stop between unpunctuated pieces. */
export function mergeTexts(texts: string[]): string {
  return texts
    .map((t) => t.trim())
    .filter(Boolean)
    .map((t, i, all) => (i < all.length - 1 && !/[.!?]$/.test(t) ? `${t}.` : t))
    .join(' ')
}
