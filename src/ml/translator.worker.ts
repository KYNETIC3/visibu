/// <reference lib="webworker" />
// Runs the sign translation models with ONNX Runtime Web, off the main thread.
// Receives landmark frames (numbers only, never video), returns proposed text.
// Models are self-hosted under /models/visibu-slt/ and checked against the
// sha256 values in their manifest. Nothing is sent anywhere.
import * as ort from 'onnxruntime-web/wasm'
import wasmUrl from 'onnxruntime-web/ort-wasm-simd-threaded.wasm?url'
import { beamSearch, type BeamConfig } from './beam'
import { ctcGreedyDecode } from './ctc'
import { FEATURE_DIM, FEATURE_VERSION, sequenceFeatures } from './features'
import type { FromTranslator, ModelInfo, ToTranslator } from './messages'
import { mergeSuggestions, wordSuggestions, type RankedSuggestion } from './suggest'
import { detokenize } from './tokenizer'

declare const self: DedicatedWorkerGlobalScope

interface Manifest {
  name: string
  version: string
  stage: 'smoke' | 'research'
  feature_version: number
  feature_dim: number
  fps: number
  max_tokens: number
  /** Present on models exported with a batched (beam search) decoder. */
  decoding?: { beam: number; n_best: number; no_repeat_ngram: number }
  special_ids: { pad: number; bos: number; eos: number; unk: number; ctc_blank: number }
  ctc_targets: 'gloss' | 'english-words'
  dataset: string
  licence: string
  files: Record<string, { sha256: string; bytes: number }>
}

let encoder: ort.InferenceSession | null = null
let decoder: ort.InferenceSession | null = null
let manifest: Manifest | null = null
let itos: string[] = []
let signs: ort.InferenceSession | null = null
let signLabels: string[] = []

interface SignsManifest {
  version: string
  feature_version: number
  feature_dim: number
  signs: number
  test: { top1: number } | null
  matching?: { mean: number[]; threshold: number }
  files: Record<string, { sha256: string; bytes: number }>
}

/** The optional 250-sign word model. Missing or broken -> sentence model only. */
async function loadSigns(baseUrl: string, opts: ort.InferenceSession.SessionOptions): Promise<ModelInfo['words']> {
  const dir = `${baseUrl}models/visibu-signs/`
  try {
    const res = await fetch(`${dir}manifest.json`, { cache: 'no-cache' })
    if (!res.ok || !(res.headers.get('content-type') ?? '').includes('json')) return null
    const m = (await res.json()) as SignsManifest
    if (m.feature_version !== FEATURE_VERSION || m.feature_dim !== FEATURE_DIM) return null
    const [model, labels] = await Promise.all(['signs.onnx', 'labels.json'].map((f) => fetchChecked(dir + f, m.files[f])))
    signLabels = JSON.parse(new TextDecoder().decode(labels)) as string[]
    signs = await ort.InferenceSession.create(new Uint8Array(model), opts)
    const matching = m.matching ? { mean: m.matching.mean, threshold: m.matching.threshold } : null
    return { signs: signLabels.length, testTop1: m.test?.top1 ?? null, version: m.version, matching }
  } catch (err) {
    console.warn('[translator] word model not loaded:', err)
    signs = null
    return null
  }
}

const post = (m: FromTranslator) => self.postMessage(m)

async function sha256(buf: ArrayBuffer): Promise<string> {
  const d = await crypto.subtle.digest('SHA-256', buf)
  return [...new Uint8Array(d)].map((b) => b.toString(16).padStart(2, '0')).join('')
}

async function fetchChecked(url: string, expected: { sha256: string; bytes: number }): Promise<ArrayBuffer> {
  const res = await fetch(url, { cache: 'no-cache' })
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`)
  const buf = await res.arrayBuffer()
  if (buf.byteLength !== expected.bytes || (await sha256(buf)) !== expected.sha256) {
    throw new Error(`${url} does not match its manifest checksum`)
  }
  return buf
}

async function init(baseUrl: string) {
  const started = performance.now()
  const dir = `${baseUrl}models/visibu-slt/`
  let res: Response
  try {
    res = await fetch(`${dir}manifest.json`, { cache: 'no-cache' })
  } catch {
    post({ type: 'unavailable', reason: 'The sign translation model could not be loaded.' })
    return
  }
  // A missing file may come back as the app's HTML page (dev server fallback).
  if (!res.ok || !(res.headers.get('content-type') ?? '').includes('json')) {
    post({ type: 'unavailable', reason: 'No sign translation model is installed in this build.' })
    return
  }
  try {
    const m = (await res.json()) as Manifest
    if (m.feature_version !== FEATURE_VERSION || m.feature_dim !== FEATURE_DIM) {
      throw new Error(`model expects feature version ${m.feature_version}, app has ${FEATURE_VERSION}`)
    }
    const [enc, dec, vocab] = await Promise.all(
      ['encoder.onnx', 'decoder.onnx', 'vocab.json'].map((f) => fetchChecked(dir + f, m.files[f])),
    )
    itos = (JSON.parse(new TextDecoder().decode(vocab)) as { itos: string[] }).itos

    // Single-threaded WASM: works without cross-origin isolation and everywhere
    // WebAssembly runs. We are already inside a worker, so no proxy worker.
    ort.env.wasm.wasmPaths = { wasm: wasmUrl }
    ort.env.wasm.numThreads = 1
    ort.env.wasm.proxy = false
    const opts: ort.InferenceSession.SessionOptions = { executionProviders: ['wasm'], graphOptimizationLevel: 'all' }
    encoder = await ort.InferenceSession.create(new Uint8Array(enc), opts)
    decoder = await ort.InferenceSession.create(new Uint8Array(dec), opts)
    const words = await loadSigns(baseUrl, opts)
    manifest = m
    const info: ModelInfo = {
      name: m.name,
      version: m.version,
      stage: m.stage,
      ctcTargets: m.ctc_targets,
      licence: m.licence,
      dataset: m.dataset,
      backend: 'wasm',
      loadMs: Math.round(performance.now() - started),
      words,
    }
    post({ type: 'ready', model: info })
  } catch (err) {
    post({ type: 'error', message: `Sign translation model failed to load: ${String(err)}` })
  }
}

async function translate(msg: Extract<ToTranslator, { type: 'translate' }>) {
  if (!encoder || !decoder || !manifest) {
    post({ type: 'error', id: msg.id, message: 'Model not loaded' })
    return
  }
  const started = performance.now()
  const { count, aspect } = msg
  const frameLen = msg.frames.length / count
  const frames = Array.from({ length: count }, (_, t) => msg.frames.subarray(t * frameLen, (t + 1) * frameLen))
  const feats = sequenceFeatures(frames, aspect)

  const encOut = await encoder.run({ features: new ort.Tensor('float32', feats, [1, count, FEATURE_DIM]) })
  const memory = encOut.memory
  const ctc = encOut.ctc_log_probs
  const [, steps, vocab] = ctc.dims
  const ctcIds = ctcGreedyDecode(ctc.data as Float32Array, steps, vocab, manifest.special_ids.ctc_blank)

  const { bos, eos, pad, unk } = manifest.special_ids
  const memData = memory.data as Float32Array
  const [, S, D] = memory.dims
  // Older exports take one sequence per decoder call; newer ones take a batch of beams.
  const batched = manifest.decoding !== undefined
  const dec = decoder
  const runDecoder = async (prefixes: number[][]): Promise<Float32Array> => {
    const B = prefixes.length
    const L = prefixes[0].length
    const mem = new Float32Array(B * S * D)
    for (let b = 0; b < B; b++) mem.set(memData, b * S * D)
    const tokens = new ort.Tensor('int64', BigInt64Array.from(prefixes.flat().map(BigInt)), [B, L])
    const out = await dec.run({ memory: new ort.Tensor('float32', mem, [B, S, D]), tokens })
    return out.next_log_probs.data as Float32Array
  }
  const step = async (prefixes: number[][]): Promise<Float32Array> => {
    if (batched) return runDecoder(prefixes)
    const rows: Float32Array[] = []
    for (const p of prefixes) rows.push(await runDecoder([p])) // one run at a time per session
    const all = new Float32Array(rows.reduce((n, r) => n + r.length, 0))
    let off = 0
    for (const r of rows) {
      all.set(r, off)
      off += r.length
    }
    return all
  }
  const d = manifest.decoding ?? { beam: 4, n_best: 3, no_repeat_ngram: 3 }
  const cfg: BeamConfig = {
    beam: d.beam,
    nBest: d.n_best,
    maxLen: manifest.max_tokens,
    noRepeatNgram: d.no_repeat_ngram,
    banned: [pad, bos, unk],
  }
  const hyps = await beamSearch(step, bos, eos, cfg)
  const sentences: RankedSuggestion[] = hyps.map((h) => ({
    text: detokenize(h.tokens.map((i) => itos[i] ?? '<unk>')),
    confidence: Math.exp(h.score),
    source: 'sentence',
  }))

  const { guesses, embedding } = await runSigns(feats, count)
  const durationMs = (count / manifest.fps) * 1000
  const suggestions = mergeSuggestions(wordSuggestions(guesses.slice(0, 3)), sentences, durationMs)
  post({
    type: 'result',
    result: {
      id: msg.id,
      text: suggestions[0]?.text ?? '',
      confidence: suggestions[0]?.confidence ?? 0,
      suggestions,
      signs: guesses,
      embedding,
      ctcTokens: ctcIds.map((i) => itos[i] ?? '<unk>'),
      frames: count,
      latencyMs: Math.round(performance.now() - started),
    },
  })
}

/** The word model on one clip: top signs and the clip's fingerprint (null without the model). */
async function runSigns(feats: Float32Array, count: number) {
  if (!signs) return { guesses: [] as { label: string; prob: number }[], embedding: null }
  const out = await signs.run({ features: new ort.Tensor('float32', feats, [1, count, FEATURE_DIM]) })
  const probs = out.probs.data as Float32Array
  const guesses = [...probs.keys()]
    .sort((a, b) => probs[b] - probs[a] || a - b)
    .slice(0, 5)
    .map((i) => ({ label: signLabels[i], prob: probs[i] }))
  const embedding = out.embedding ? Float32Array.from(out.embedding.data as Float32Array) : null
  return { guesses, embedding }
}

async function fingerprint(msg: Extract<ToTranslator, { type: 'fingerprint' }>) {
  const frameLen = msg.frames.length / msg.count
  const frames = Array.from({ length: msg.count }, (_, t) => msg.frames.subarray(t * frameLen, (t + 1) * frameLen))
  const { guesses, embedding } = await runSigns(sequenceFeatures(frames, msg.aspect), msg.count)
  post({ type: 'fingerprint', id: msg.id, embedding, signs: guesses })
}

self.onmessage = (event: MessageEvent<ToTranslator>) => {
  const msg = event.data
  if (msg.type === 'init') void init(msg.baseUrl)
  else if (msg.type === 'fingerprint')
    fingerprint(msg).catch((err) => post({ type: 'error', id: msg.id, message: `Fingerprint failed: ${String(err)}` }))
  else
    translate(msg).catch((err) => post({ type: 'error', id: msg.id, message: `Translation failed: ${String(err)}` }))
}
