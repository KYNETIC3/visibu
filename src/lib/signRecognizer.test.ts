import { describe, expect, it } from 'vitest'
import { FRAME_LENGTH, LAYOUT, packFrame, type Point } from '../tracking/frame'
import type { FromTranslator, ModelInfo, ToTranslator } from '../ml/messages'
import { MessageSegmenter, type SegmentEvent } from '../ml/segmenter'
import { OnnxSignRecognizer, type Activity, type RecognizerStatus, type SignProposal } from './signRecognizer'

const hand = (): Point[] => Array.from({ length: 21 }, (_, i) => ({ x: 0.5 + i * 0.001, y: 0.5, z: 0 }))
const withHands = () => packFrame({ poseLandmarks: [], leftHandLandmarks: [], rightHandLandmarks: [hand()], faceLandmarks: [] })
const noHands = () => new Float32Array(FRAME_LENGTH).fill(NaN)

/** Feeds frames at 30 fps; returns all segmenter events. */
function feed(seg: MessageSegmenter, frames: Float32Array[], t0 = 0): SegmentEvent[] {
  return frames.flatMap((f, i) => seg.push(f, t0 + (i * 1000) / 30))
}
const repeat = (n: number, make: () => Float32Array) => Array.from({ length: n }, make)

describe('MessageSegmenter', () => {
  it('resamples to 15 fps and ends a message after the hands go down', () => {
    const seg = new MessageSegmenter()
    // 5 s idle, 3 s signing, 2 s hands down — at 30 fps.
    const events = feed(seg, [...repeat(150, noHands), ...repeat(90, withHands), ...repeat(60, noHands)])
    expect(events.map((e) => e.type)).toEqual(['started', 'segment'])
    const segEv = events[1] as Extract<SegmentEvent, { type: 'segment' }>
    // 3 s at 15 fps = 45 frames, plus up to 4 preroll and 4 tail frames.
    expect(segEv.frames.length).toBeGreaterThanOrEqual(45)
    expect(segEv.frames.length).toBeLessThanOrEqual(45 + 4 + 4 + 1)
  })

  it('keeps one message across short pauses between signs', () => {
    const seg = new MessageSegmenter()
    const events = feed(seg, [...repeat(30, withHands), ...repeat(15, noHands), ...repeat(30, withHands), ...repeat(60, noHands)])
    expect(events.filter((e) => e.type === 'segment')).toHaveLength(1)
  })

  it('discards very short movements', () => {
    const seg = new MessageSegmenter()
    const events = feed(seg, [...repeat(4, withHands), ...repeat(60, noHands)])
    expect(events.map((e) => e.type)).toEqual(['started', 'discarded'])
  })

  it('ends a message on demand', () => {
    const seg = new MessageSegmenter()
    feed(seg, repeat(40, withHands))
    expect(seg.finish().map((e) => e.type)).toEqual(['segment'])
    expect(seg.finish()).toEqual([])
  })

  it('caps a message at maxFrames', () => {
    const seg = new MessageSegmenter({ fps: 15, endPauseMs: 1200, prerollFrames: 0, tailFrames: 0, minFrames: 8, maxFrames: 30 })
    const events = feed(seg, repeat(120, withHands))
    const segs = events.filter((e) => e.type === 'segment') as Extract<SegmentEvent, { type: 'segment' }>[]
    expect(segs.length).toBeGreaterThanOrEqual(1)
    expect(segs[0].frames.length).toBe(30)
  })
})

class FakeWorker {
  sent: ToTranslator[] = []
  onmessage: ((e: MessageEvent<FromTranslator>) => void) | null = null
  onerror: ((e: ErrorEvent) => void) | null = null
  postMessage(m: ToTranslator) {
    this.sent.push(m)
  }
  terminate() {}
  reply(m: FromTranslator) {
    this.onmessage?.({ data: m } as MessageEvent<FromTranslator>)
  }
}

const MODEL: ModelInfo = {
  name: 'visibu-slt', version: 'test', stage: 'research', ctcTargets: 'english-words',
  licence: 'CC BY-NC 4.0', dataset: 'How2Sign', backend: 'wasm', loadMs: 1, words: null,
}
const video = { videoWidth: 1280, videoHeight: 720 } as HTMLVideoElement

function setup() {
  const worker = new FakeWorker()
  const r = new OnnxSignRecognizer(() => worker as unknown as Worker)
  const statuses: RecognizerStatus[] = []
  const proposals: SignProposal[] = []
  const activity: Activity[] = []
  const hands: boolean[] = []
  r.load((s) => statuses.push(s))
  const handlers = {
    onHands: (v: boolean) => hands.push(v),
    onActivity: (a: Activity) => activity.push(a),
    onProposal: (p: SignProposal) => proposals.push(p),
  }
  return { r, worker, statuses, proposals, activity, hands, handlers }
}

const signOnce = (r: OnnxSignRecognizer, t0 = 0) =>
  [...repeat(45, withHands), ...repeat(60, noHands)].forEach((f, i) => r.processFrame(f, t0 + (i * 1000) / 30))

describe('OnnxSignRecognizer', () => {
  it('reports an unavailable model and still reports hands', () => {
    const { r, worker, statuses, hands, handlers, activity } = setup()
    expect(worker.sent[0]).toMatchObject({ type: 'init' })
    worker.reply({ type: 'unavailable', reason: 'No model' })
    expect(statuses.at(-1)).toEqual({ state: 'unavailable', reason: 'No model' })
    r.start(video, handlers)
    signOnce(r)
    expect(hands).toEqual([true, false])
    expect(worker.sent.filter((m) => m.type === 'translate')).toHaveLength(0)
    expect(activity).toEqual(['waiting'])
  })

  it('does not translate a stretch spent holding up a number to choose a suggestion', () => {
    const { r, worker, activity, handlers } = setup()
    worker.reply({ type: 'ready', model: MODEL })
    r.start(video, { ...handlers, isControlGesture: () => true })
    signOnce(r)
    expect(worker.sent.filter((m) => m.type === 'translate')).toHaveLength(0)
    expect(activity).toEqual(['waiting', 'signing', 'waiting'])
  })

  it('keeps translating new signing while earlier translations wait', () => {
    const { r, worker, handlers } = setup()
    worker.reply({ type: 'ready', model: MODEL })
    r.start(video, handlers)
    signOnce(r)
    signOnce(r, 10_000)
    expect(worker.sent.filter((m) => m.type === 'translate')).toHaveLength(2)
  })

  it('sends one message for translation and proposes the result, never auto-sending it', () => {
    const { r, worker, proposals, activity, handlers } = setup()
    worker.reply({ type: 'ready', model: MODEL })
    r.start(video, handlers)
    signOnce(r)
    const req = worker.sent.find((m) => m.type === 'translate')
    expect(req).toBeDefined()
    if (req?.type !== 'translate') return
    expect(req.aspect).toBeCloseTo(16 / 9)
    expect(req.frames.length).toBe(req.count * FRAME_LENGTH)
    expect(activity).toEqual(['waiting', 'signing', 'translating'])

    worker.reply({ type: 'result', result: { id: req.id, text: 'I would like a coffee.', confidence: 0.3, suggestions: [{ text: 'I would like a coffee.', confidence: 0.3, source: 'sentence' }, { text: 'I like coffee.', confidence: 0.6, source: 'sentence' }], signs: [], embedding: null, ctcTokens: ['coffee'], frames: req.count, latencyMs: 40 } })
    expect(proposals).toHaveLength(1)
    expect(proposals[0]).toMatchObject({ text: 'I would like a coffee.', lowConfidence: true, glosses: [] })
    expect(proposals[0].suggestions).toEqual([
      { text: 'I would like a coffee.', confidence: 0.3, lowConfidence: true, source: 'sentence' },
      { text: 'I like coffee.', confidence: 0.6, lowConfidence: false, source: 'sentence' },
    ])
    expect(activity.at(-1)).toBe('waiting')
  })

  it('drops results that arrive after the camera was stopped', () => {
    const { r, worker, proposals, handlers } = setup()
    worker.reply({ type: 'ready', model: MODEL })
    r.start(video, handlers)
    signOnce(r)
    const req = worker.sent.find((m) => m.type === 'translate')
    r.stop()
    r.start(video, handlers)
    if (req?.type === 'translate') {
      worker.reply({ type: 'result', result: { id: req.id, text: 'Late', confidence: 0.9, suggestions: [{ text: 'Late', confidence: 0.9, source: 'sentence' }], signs: [], embedding: null, ctcTokens: [], frames: 1, latencyMs: 1 } })
    }
    expect(proposals).toHaveLength(0)
  })

  it('does not track after stop', () => {
    const { r, worker, handlers } = setup()
    worker.reply({ type: 'ready', model: MODEL })
    r.start(video, handlers)
    r.stop()
    signOnce(r)
    expect(worker.sent.filter((m) => m.type === 'translate')).toHaveLength(0)
  })

  it('only shows CTC tokens as glosses for gloss-trained models', () => {
    const { r, worker, proposals, handlers } = setup()
    worker.reply({ type: 'ready', model: { ...MODEL, ctcTargets: 'gloss' } })
    r.start(video, handlers)
    signOnce(r)
    const req = worker.sent.find((m) => m.type === 'translate')
    if (req?.type === 'translate') {
      worker.reply({ type: 'result', result: { id: req.id, text: 'Coffee please.', confidence: 0.8, suggestions: [{ text: 'Coffee please.', confidence: 0.8, source: 'sentence' }], signs: [], embedding: null, ctcTokens: ['COFFEE', 'PLEASE'], frames: 1, latencyMs: 1 } })
    }
    expect(proposals[0].glosses).toEqual(['COFFEE', 'PLEASE'])
    expect(proposals[0].lowConfidence).toBe(false)
  })
})

it('frame layout offsets are what the segmenter expects', () => {
  expect(LAYOUT.rightHand.offset).toBe((33 + 21) * 3)
})
