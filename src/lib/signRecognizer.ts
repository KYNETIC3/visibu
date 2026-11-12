import { useCallback, useEffect, useRef, useState } from 'react'
import { TARGET_FPS } from '../ml/features'
import { DEFAULT_SEGMENTER, handsIn, MessageSegmenter, type SegmenterConfig } from '../ml/segmenter'
import type { FromTranslator, ModelInfo, Suggestion, ToTranslator, TranslationResult } from '../ml/messages'
import { bestMatch, unitFingerprint, usable, type MySign } from '../ml/mySigns'

/**
 * Simple event bus for MediaPipe frames, so the tracker and the recogniser are
 * not coupled. useTracker publishes every tracked frame here.
 */
class FrameBus {
  private subscribers = new Set<(frame: Float32Array, timestamp: number) => void>()

  subscribe(fn: (frame: Float32Array, timestamp: number) => void): () => void {
    this.subscribers.add(fn)
    return () => this.subscribers.delete(fn)
  }

  publish(frame: Float32Array, timestamp: number): void {
    for (const fn of this.subscribers) fn(frame, timestamp)
  }

  clear(): void {
    this.subscribers.clear()
  }
}

export const frameBus = new FrameBus()

/** A machine translation of one signed message, waiting for the signer to approve, edit or reject it. */
export interface SignProposal {
  id: number
  /** Proposed English. Never shown to the partner until the signer approves it. */
  text: string
  /** Mean token probability from the model, 0..1. A model score, not a measured accuracy. */
  confidence: number
  lowConfidence: boolean
  /** Up to 3 distinct suggestions, best first (the first is `text`). The signer may pick one. */
  suggestions: (Suggestion & { lowConfidence: boolean })[]
  /** The word model's fingerprint of the clip (raw, uncentred), or null without the word model. */
  embedding: Float32Array | null
  /** Glosses from the CTC head. Empty unless the model was trained on gloss annotations. */
  glosses: string[]
  durationMs: number
  latencyMs: number
  model: ModelInfo
}

export type RecognizerStatus =
  | { state: 'loading' }
  | { state: 'ready'; model: ModelInfo }
  | { state: 'unavailable'; reason: string }
  | { state: 'error'; reason: string }

/** What the recogniser is doing right now. */
export type Activity = 'waiting' | 'signing' | 'translating'

export interface RecognizerHandlers {
  onHands: (visible: boolean) => void
  onActivity: (activity: Activity) => void
  onProposal: (proposal: SignProposal) => void
  /** Signing was too short to translate, or the model produced no words. */
  onNothing?: (reason: 'too-short' | 'empty') => void
  onError?: (message: string) => void
  /** True if a finished stretch of signing was a control gesture (e.g. holding up 1, 2 or 3 to choose), not a message. */
  isControlGesture?: (frames: Float32Array[]) => boolean
  /** Capture mode (teaching a sign): each finished stretch of signing goes here instead of being translated. */
  onCapture?: (frames: Float32Array[]) => void
}

export interface Fingerprint {
  embedding: Float32Array | null
  signs: { label: string; prob: number }[]
}

export interface SignRecognizer {
  /** Load the model (once). Status changes are reported to `onStatus`. */
  load(onStatus: (s: RecognizerStatus) => void): void
  /** Start listening to tracked frames. The video element supplies the aspect ratio. */
  start(video: HTMLVideoElement, handlers: RecognizerHandlers): void
  /** End the current message now and translate it. */
  finish(): void
  /** Fingerprint one captured stretch of signing with the word model. */
  fingerprint(frames: Float32Array[]): Promise<Fingerprint>
  /** Stop listening and drop any message in progress. Late results are ignored. */
  stop(): void
  /** Stop and release the model. */
  dispose(): void
}

/** Below this model score a proposal is flagged as low confidence. */
export const LOW_CONFIDENCE = 0.5

/**
 * Continuous sign translation: tracked frames -> one landmark sequence per
 * message -> ONNX encoder + decoder in a Web Worker -> proposed English text.
 */
export class OnnxSignRecognizer implements SignRecognizer {
  private worker: Worker | null = null
  private status: RecognizerStatus = { state: 'loading' }
  private onStatus: ((s: RecognizerStatus) => void) | null = null
  private handlers: RecognizerHandlers | null = null
  private video: HTMLVideoElement | null = null
  private unsubscribe: (() => void) | null = null
  private segmenter: MessageSegmenter
  private handsVisible = false
  private nextId = 1
  private pending = new Map<number, { durationMs: number }>()
  private generation = 0
  private pendingGeneration = new Map<number, number>()
  private fingerprints = new Map<number, { resolve: (f: Fingerprint) => void; reject: (e: Error) => void }>()

  constructor(
    private createWorker: () => Worker = () =>
      new Worker(new URL('../ml/translator.worker.ts', import.meta.url), { type: 'module' }),
    config: SegmenterConfig = DEFAULT_SEGMENTER,
  ) {
    this.segmenter = new MessageSegmenter(config)
  }

  load(onStatus: (s: RecognizerStatus) => void): void {
    this.onStatus = onStatus
    onStatus(this.status)
    if (this.worker) return
    try {
      this.worker = this.createWorker()
    } catch (err) {
      this.setStatus({ state: 'error', reason: `Could not start the translator: ${String(err)}` })
      return
    }
    this.worker.onmessage = (e: MessageEvent<FromTranslator>) => this.handle(e.data)
    this.worker.onerror = (e) => this.setStatus({ state: 'error', reason: e.message || 'Translator crashed' })
    this.send({ type: 'init', baseUrl: import.meta.env.BASE_URL })
  }

  start(video: HTMLVideoElement, handlers: RecognizerHandlers): void {
    this.stop()
    this.video = video
    this.handlers = handlers
    this.handsVisible = false
    this.unsubscribe = frameBus.subscribe((frame, ts) => this.processFrame(frame, ts))
    handlers.onActivity('waiting')
  }

  /** Feed one frame directly (used by the frame bus and by tests). */
  processFrame(frame: Float32Array, timestamp: number): void {
    if (!this.handlers) return
    const hands = handsIn(frame)
    if (hands !== this.handsVisible) {
      this.handsVisible = hands
      this.handlers.onHands(hands)
    }
    // Without a model there is nothing to translate; keep reporting hands only.
    if (this.status.state !== 'ready') return
    for (const ev of this.segmenter.push(frame, timestamp)) this.onSegmentEvent(ev)
  }

  finish(): void {
    for (const ev of this.segmenter.finish()) this.onSegmentEvent(ev)
  }

  fingerprint(frames: Float32Array[]): Promise<Fingerprint> {
    if (this.status.state !== 'ready' || frames.length === 0) return Promise.reject(new Error('The sign model is not ready.'))
    const { packed, count } = pack(frames)
    const id = this.nextId++
    return new Promise((resolve, reject) => {
      this.fingerprints.set(id, { resolve, reject })
      this.send({ type: 'fingerprint', id, frames: packed, count, aspect: this.aspect() }, [packed.buffer])
    })
  }

  private aspect(): number {
    const v = this.video
    return v && v.videoWidth > 0 && v.videoHeight > 0 ? v.videoWidth / v.videoHeight : 16 / 9
  }

  stop(): void {
    this.unsubscribe?.()
    this.unsubscribe = null
    this.handlers = null
    this.segmenter.reset()
    this.generation++
  }

  dispose(): void {
    this.stop()
    this.worker?.terminate()
    this.worker = null
    this.status = { state: 'loading' }
  }

  private onSegmentEvent(ev: ReturnType<MessageSegmenter['finish']>[number]) {
    if (!this.handlers) return
    if (ev.type === 'started') {
      this.handlers.onActivity('signing')
    } else if (ev.type === 'discarded') {
      this.handlers.onActivity('waiting')
      this.handlers.onNothing?.('too-short')
    } else if (this.handlers.onCapture) {
      this.handlers.onActivity('waiting')
      this.handlers.onCapture(ev.frames)
    } else if (this.handlers.isControlGesture?.(ev.frames)) {
      this.handlers.onActivity('waiting')
    } else {
      const { packed, count } = pack(ev.frames)
      const id = this.nextId++
      this.pending.set(id, { durationMs: ev.durationMs })
      this.pendingGeneration.set(id, this.generation)
      this.handlers.onActivity('translating')
      this.send({ type: 'translate', id, frames: packed, count, aspect: this.aspect() }, [packed.buffer])
    }
  }

  private handle(msg: FromTranslator) {
    switch (msg.type) {
      case 'ready':
        this.setStatus({ state: 'ready', model: msg.model })
        break
      case 'unavailable':
        this.setStatus({ state: 'unavailable', reason: msg.reason })
        break
      case 'fingerprint': {
        const w = this.fingerprints.get(msg.id)
        this.fingerprints.delete(msg.id)
        w?.resolve({ embedding: msg.embedding, signs: msg.signs })
        break
      }
      case 'error':
        if (msg.id === undefined) this.setStatus({ state: 'error', reason: msg.message })
        else if (this.fingerprints.has(msg.id)) {
          this.fingerprints.get(msg.id)!.reject(new Error(msg.message))
          this.fingerprints.delete(msg.id)
        } else {
          const current = this.takePending(msg.id)
          if (current) {
            this.handlers?.onActivity('waiting')
            this.handlers?.onError?.(msg.message)
          }
        }
        break
      case 'result':
        this.onResult(msg.result)
        break
    }
  }

  /** Returns the pending entry if it belongs to the current start() session. */
  private takePending(id: number) {
    const entry = this.pending.get(id)
    const gen = this.pendingGeneration.get(id)
    this.pending.delete(id)
    this.pendingGeneration.delete(id)
    return entry && gen === this.generation ? entry : null
  }

  private onResult(r: TranslationResult) {
    const entry = this.takePending(r.id)
    if (!entry || !this.handlers || this.status.state !== 'ready') return
    this.handlers.onActivity('waiting')
    if (!r.text.trim()) {
      this.handlers.onNothing?.('empty')
      return
    }
    const model = this.status.model
    this.handlers.onProposal({
      id: r.id,
      text: r.text,
      confidence: r.confidence,
      lowConfidence: r.confidence < LOW_CONFIDENCE,
      suggestions: r.suggestions.map((s) => ({ ...s, lowConfidence: s.confidence < LOW_CONFIDENCE })),
      embedding: r.embedding,
      glosses: model.ctcTargets === 'gloss' ? r.ctcTokens : [],
      durationMs: entry.durationMs,
      latencyMs: r.latencyMs,
      model,
    })
  }

  private setStatus(s: RecognizerStatus) {
    this.status = s
    this.onStatus?.(s)
  }

  private send(msg: ToTranslator, transfer: Transferable[] = []) {
    this.worker?.postMessage(msg, transfer)
  }
}

function pack(frames: Float32Array[]) {
  const count = frames.length
  const frameLen = frames[0].length
  const packed = new Float32Array(count * frameLen)
  frames.forEach((f, i) => packed.set(f, i * frameLen))
  return { packed, count }
}

/**
 * Puts the user's own sign first when a translation matches one of them
 * (same fingerprint, at or above the model's threshold). Otherwise unchanged.
 */
export function withMySign(p: SignProposal, signs: MySign[]): SignProposal {
  const words = p.model.words
  if (!p.embedding || !words?.matching) return p
  const match = bestMatch(unitFingerprint(p.embedding, words.matching.mean), usable(signs, words.version), words.matching.threshold)
  if (!match) return p
  const mine = { text: match.sign.meaning, confidence: Math.max(0, match.score), lowConfidence: false, source: 'mine' as const, sign: match.sign.meaning }
  const rest = p.suggestions.filter((s) => s.text.trim().toLowerCase() !== mine.text.trim().toLowerCase())
  return { ...p, text: mine.text, confidence: mine.confidence, lowConfidence: false, suggestions: [mine, ...rest].slice(0, 3) }
}

/** The app's sign recogniser: the trained ONNX translation model. */
export function createSignRecognizer(): SignRecognizer {
  return new OnnxSignRecognizer()
}

/** Shown in place of live sign captions when no model is installed. Never saved. */
export const EXAMPLE_SIGN_CAPTION = 'Hello, I would like to book an appointment for Thursday.'

interface Options {
  video: HTMLVideoElement | null
  active: boolean
  /** The user's own signs; a translation that matches one shows its meaning first. */
  mySigns?: MySign[]
  /** Teaching mode: finished stretches of signing go here instead of being translated. */
  onCapture?: (frames: Float32Array[]) => void
}

/** A short stretch of signing around a pick-by-sign is that pick, not a message. */
const PICK_SLACK_MS = 2500
/** Longer stretches are kept even if a pick happened during them: a message is never dropped. */
const PICK_MAX_MS = 3500

/** At most this many translations wait for review; the oldest is dropped beyond it. */
export const MAX_PENDING = 6

export function useSignRecognition({ video, active, mySigns, onCapture }: Options) {
  const [status, setStatus] = useState<RecognizerStatus>({ state: 'loading' })
  const [handsVisible, setHandsVisible] = useState<boolean | null>(null)
  const [activity, setActivity] = useState<Activity>('waiting')
  // Translations waiting for the signer, oldest first. Signing continues while they wait.
  const [proposals, setProposals] = useState<SignProposal[]>([])
  const [notice, setNotice] = useState<string | null>(null)
  const recognizerRef = useRef<SignRecognizer | null>(null)
  // When the signer last chose a suggestion by holding up 1, 2 or 3.
  const lastPickRef = useRef(-Infinity)
  const mySignsRef = useRef(mySigns ?? [])
  const captureRef = useRef(onCapture)
  useEffect(() => {
    mySignsRef.current = mySigns ?? []
    captureRef.current = onCapture
  }, [mySigns, onCapture])
  const capturing = onCapture !== undefined

  // One recogniser (and one model load) for the life of the screen.
  useEffect(() => {
    const r = createSignRecognizer()
    recognizerRef.current = r
    r.load(setStatus)
    return () => {
      r.dispose()
      recognizerRef.current = null
    }
  }, [])

  useEffect(() => {
    const r = recognizerRef.current
    if (!r || !video || !active) {
      setHandsVisible(null)
      setActivity('waiting')
      return
    }
    r.start(video, {
      onHands: setHandsVisible,
      onActivity: setActivity,
      onProposal: (p) => {
        setNotice(null)
        setProposals((list) => [...list, withMySign(p, mySignsRef.current)].slice(-MAX_PENDING))
      },
      onCapture: capturing ? (frames) => captureRef.current?.(frames) : undefined,
      onNothing: (reason) =>
        setNotice(reason === 'too-short' ? 'That was too short to translate. Sign, then lower your hands.' : 'No words were recognised. Please try again or type your message.'),
      onError: (message) => setNotice(message),
      // Holding up a number to choose also looks like signing; drop that stretch.
      isControlGesture: (frames) => {
        const ms = (frames.length / TARGET_FPS) * 1000
        return ms <= PICK_MAX_MS && performance.now() - lastPickRef.current < ms + PICK_SLACK_MS
      },
    })
    return () => r.stop()
  }, [active, video, capturing])

  const finish = useCallback(() => recognizerRef.current?.finish(), [])
  /** Remove reviewed (accepted or discarded) translations. */
  const resolve = useCallback((ids: number[]) => setProposals((list) => list.filter((p) => !ids.includes(p.id))), [])
  const clearNotice = useCallback(() => setNotice(null), [])
  /** Call when a suggestion was chosen by holding up a number, so that stretch is not translated. */
  const notePick = useCallback(() => {
    lastPickRef.current = performance.now()
  }, [])

  const fingerprint = useCallback((frames: Float32Array[]) => {
    const r = recognizerRef.current
    return r ? r.fingerprint(frames) : Promise.reject(new Error('The sign model is not loaded.'))
  }, [])

  return { status, handsVisible, activity, proposals, notice, finish, resolve, clearNotice, notePick, fingerprint }
}
