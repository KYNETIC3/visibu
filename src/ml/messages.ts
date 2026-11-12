// Messages between the main thread and the translator worker.

export type ModelStage = 'smoke' | 'research'

export interface ModelInfo {
  name: string
  version: string
  stage: ModelStage
  ctcTargets: 'gloss' | 'english-words'
  licence: string
  dataset: string
  backend: string
  loadMs: number
  /** The 250-sign word model, if installed. */
  words: {
    signs: number
    testTop1: number | null
    version: string
    /** For comparing sign fingerprints (the user's own signs): centring vector and same-sign threshold. */
    matching: { mean: number[]; threshold: number } | null
  } | null
}

export type ToTranslator =
  | { type: 'init'; baseUrl: string }
  | { type: 'translate'; id: number; frames: Float32Array; count: number; aspect: number }
  /** Only the word model: the clip's fingerprint and top signs (teaching the user's own signs). */
  | { type: 'fingerprint'; id: number; frames: Float32Array; count: number; aspect: number }

export interface Suggestion {
  text: string
  /** The producing model's score, 0..1 (sentence: geometric-mean token probability; sign: softmax). Not a calibrated accuracy. */
  confidence: number
  source: 'sign' | 'sentence' | 'mine'
  /** For word suggestions: the recognised sign in plain English. */
  sign?: string
}

export interface TranslationResult {
  id: number
  /** The best suggestion (same as suggestions[0]). */
  text: string
  confidence: number
  /** Up to 3 distinct suggestions from the word and sentence models, best first (see suggest.ts). */
  suggestions: Suggestion[]
  /** Top signs from the word model, most likely first (empty without it). */
  signs: { label: string; prob: number }[]
  /** The word model's fingerprint of the clip (null without it), for matching the user's own signs. */
  embedding: Float32Array | null
  /** Greedy CTC output. Glosses only when the model was trained on glosses. */
  ctcTokens: string[]
  frames: number
  latencyMs: number
}

export type FromTranslator =
  | { type: 'ready'; model: ModelInfo }
  | { type: 'unavailable'; reason: string }
  | { type: 'error'; message: string; id?: number }
  | { type: 'result'; result: TranslationResult }
  | { type: 'fingerprint'; id: number; embedding: Float32Array | null; signs: { label: string; prob: number }[] }
