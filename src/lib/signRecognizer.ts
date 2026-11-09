import { useEffect, useRef, useState, useCallback } from 'react';
import { readPart, type Point } from '../tracking/frame';

/**
 * A single recognised sign with confidence and sentence-boundary flag.
 */
export interface SignEvent {
  /** English word or phrase for the recognised sign. */
  word: string;
  /** 0–1 confidence. */
  confidence: number;
  /** True when this sign ends a sentence (pause, or a sentence-final sign). */
  endsSentence: boolean;
}

/**
 * Interface for a sign language classifier.
 * Implement this to plug in a real ML model (e.g., TensorFlow.js, ONNX, MediaPipe custom model).
 */
export interface SignClassifier {
  /**
   * Classify a single frame's hand landmarks.
   * Called once per frame; implementations may keep motion history.
   * @param primary   21 landmarks of the dominant hand, or null
   * @param secondary 21 landmarks of the other hand, or null
   * @param body      face / chest positions, when tracked
   * @returns Recognised sign or null if no sign detected in this frame
   */
  classify(primary: Point[] | null, secondary: Point[] | null, body?: BodyContext): SignEvent | null;

  /** Reset any internal temporal state (e.g., between sentences). */
  reset(): void;

  /** Optional: warm-up / model loading. Returns true when ready. */
  init?(): Promise<boolean>;
}

/**
 * Configuration for the MediaPipe-backed sign recognizer.
 */
export interface SignRecognizerConfig {
  /** Minimum confidence to emit a sign event. */
  minConfidence: number;
  /** Frames a sign must persist before being emitted (temporal smoothing). */
  minHoldFrames: number;
  /** Frames of no-hand / low-confidence before a sentence boundary is emitted. */
  sentenceGapFrames: number;
  /** Whether to prefer left or right hand for one-handed signs. 'auto' = dominant hand by visibility. */
  preferredHand: 'left' | 'right' | 'auto';
}

/** Default recognizer configuration. */
export const DEFAULT_CONFIG: SignRecognizerConfig = {
  minConfidence: 0.65,
  minHoldFrames: 3,
  sentenceGapFrames: 15,
  preferredHand: 'auto',
};

/**
 * Simple event bus for MediaPipe frames.
 * Allows the tracker (wherever it lives) to push frames without direct coupling.
 */
class FrameBus {
  private subscribers = new Set<(frame: Float32Array, timestamp: number) => void>();

  /** Subscribe to frames. Returns an unsubscribe function. */
  subscribe(fn: (frame: Float32Array, timestamp: number) => void): () => void {
    this.subscribers.add(fn);
    return () => this.subscribers.delete(fn);
  }

  /** Publish a frame to all subscribers. */
  publish(frame: Float32Array, timestamp: number): void {
    for (const fn of this.subscribers) fn(frame, timestamp);
  }

  /** Clear all subscribers (for testing / disposal). */
  clear(): void {
    this.subscribers.clear();
  }
}

/** Singleton frame bus. The tracker should call `frameBus.publish(frame, timestamp)` on each result. */
export const frameBus = new FrameBus();

/** Frames of hand history the heuristic classifier looks back over. */
const HISTORY = 15;
/** Minimum frames of history before any motion-based sign can fire. */
const MIN_MOTION_FRAMES = 8;
/** Per-frame movement below this is treated as tracking jitter. */
const JITTER = 0.004;

interface Sample {
  wrist: Point;
  tip: Point;
}

const dist2 = (a: Point, b: Point) => Math.hypot(a.x - b.x, a.y - b.y);

/** Counts direction reversals along one axis, ignoring jitter. */
function reversals(values: number[]): number {
  let flips = 0;
  let lastSign = 0;
  for (let i = 1; i < values.length; i++) {
    const d = values[i] - values[i - 1];
    if (Math.abs(d) < JITTER) continue;
    const s = Math.sign(d);
    if (lastSign !== 0 && s !== lastSign) flips++;
    lastSign = s;
  }
  return flips;
}

const range = (values: number[]) => Math.max(...values) - Math.min(...values);

/**
 * Heuristic-based sign classifier (reference implementation).
 * Recognises a tiny vocabulary from hand shape + motion over a short window.
 * DEMONSTRATION ONLY — replace with a trained model by implementing SignClassifier.
 */
export class HeuristicSignClassifier implements SignClassifier {
  private history: Sample[] = [];

  classify(primary: Point[] | null, _secondary: Point[] | null, body?: BodyContext): SignEvent | null {
    if (!primary || !primary[0] || !primary[8]) return null;

    this.history.push({ wrist: primary[0], tip: primary[8] });
    if (this.history.length > HISTORY) this.history.shift();
    if (this.history.length < MIN_MOTION_FRAMES) return null;

    const wristX = this.history.map((s) => s.wrist.x);
    const wristY = this.history.map((s) => s.wrist.y);
    const tipX = this.history.map((s) => s.tip.x);
    const first = this.history[0];
    const last = this.history[this.history.length - 1];

    const open = this.isOpenHand(primary);
    const fist = this.isFist(primary);
    const pointing = this.isPointing(primary);

    // NO: index (and middle) finger snapping side to side.
    if (pointing && range(tipX) > 0.03 && reversals(tipX) >= 2) {
      return { word: 'NO', confidence: 0.75, endsSentence: true };
    }

    // YES: fist nodding up and down.
    if (fist && range(wristY) > 0.03 && reversals(wristY) >= 2) {
      return { word: 'YES', confidence: 0.75, endsSentence: true };
    }

    if (!open) return null;

    // THANK YOU: flat hand starts at the chin and moves forward/down, away from the face.
    const startedAtFace = body?.face
      ? dist2(first.tip, body.face) < 0.12
      : first.tip.y < 0.35;
    const movedAway = body?.face
      ? dist2(last.tip, body.face) - dist2(first.tip, body.face) > 0.08
      : last.tip.y - first.tip.y > 0.06;
    if (startedAtFace && movedAway) {
      return { word: 'THANK YOU', confidence: 0.78, endsSentence: true };
    }

    // PLEASE: flat hand circling on the chest.
    const nearChest = body?.chest ? dist2(last.wrist, body.chest) < 0.2 : last.wrist.y > 0.45;
    const circling =
      range(wristX) > 0.04 &&
      range(wristY) > 0.04 &&
      reversals(wristX) >= 1 &&
      reversals(wristY) >= 1 &&
      dist2(first.wrist, last.wrist) < Math.max(range(wristX), range(wristY)) * 0.6;
    if (nearChest && circling) {
      return { word: 'PLEASE', confidence: 0.7, endsSentence: false };
    }

    // HELLO: open hand moving up, or out sideways, in one sweep.
    const rise = first.wrist.y - last.wrist.y;
    const sweep = Math.abs(last.wrist.x - first.wrist.x);
    if ((rise > 0.06 || sweep > 0.08) && reversals(wristY) === 0) {
      return { word: 'HELLO', confidence: 0.8, endsSentence: true };
    }

    return null;
  }

  reset(): void {
    this.history = [];
  }

  private isOpenHand(hand: Point[]): boolean {
    const wrist = hand[0];
    const tips = [8, 12, 16, 20];
    const pips = [6, 10, 14, 18];
    let extended = 0;
    for (let i = 0; i < tips.length; i++) {
      if (dist3(hand[tips[i]], wrist) > dist3(hand[pips[i]], wrist) * 1.1) extended++;
    }
    return extended >= 4;
  }

  private isFist(hand: Point[]): boolean {
    const wrist = hand[0];
    const tips = [8, 12, 16, 20];
    const pips = [6, 10, 14, 18];
    let curled = 0;
    for (let i = 0; i < tips.length; i++) {
      if (dist3(hand[tips[i]], wrist) < dist3(hand[pips[i]], wrist)) curled++;
    }
    return curled >= 4;
  }

  private isPointing(hand: Point[]): boolean {
    const wrist = hand[0];
    const indexOut = dist3(hand[8], wrist) > dist3(hand[6], wrist) * 1.1;
    const ringCurled = dist3(hand[16], wrist) < dist3(hand[14], wrist);
    const pinkyCurled = dist3(hand[20], wrist) < dist3(hand[18], wrist);
    return indexOut && ringCurled && pinkyCurled;
  }
}

function dist3(a: Point, b: Point): number {
  return Math.hypot(a.x - b.x, a.y - b.y, a.z - b.z);
}

/** Pulls the face (mouth) and chest positions from a frame, if tracked. */
function readBody(frame: Float32Array): BodyContext {
  const face = readPart(frame, 'face');
  const pose = readPart(frame, 'pose');
  let faceP: Point | null = face?.[13] ?? null; // upper inner lip
  if (!faceP && pose?.[0]) faceP = pose[0]; // nose
  let chest: Point | null = null;
  if (pose?.[11] && pose?.[12]) {
    const l = pose[11];
    const r = pose[12];
    chest = { x: (l.x + r.x) / 2, y: (l.y + r.y) / 2 + 0.08, z: (l.z + r.z) / 2 };
  }
  const ok = (p: Point | null) => (p && Number.isFinite(p.x) && Number.isFinite(p.y) ? p : null);
  return { face: ok(faceP), chest: ok(chest) };
}

/** Frames without a fresh detection before the same word may be emitted again. */
const REPEAT_COOLDOWN_FRAMES = 20;

/**
 * MediaPipe-backed sign recognizer.
 * Subscribes to the frame bus, runs the classifier, emits SignEvents via callbacks.
 */
export class MediaPipeSignRecognizer implements SignRecognizer {
  private classifier: SignClassifier;
  private config: SignRecognizerConfig;
  private unsubscribe: (() => void) | null = null;
  private handlers: RecognizerHandlers | null = null;

  private candidate: SignEvent | null = null;
  private candidateFrames = 0;
  private lastWord: string | null = null;
  private quietFrames = 0;
  private gapFrames = 0;
  private pendingWords = false;
  private handsVisible = false;

  constructor(classifier: SignClassifier, config: SignRecognizerConfig = DEFAULT_CONFIG) {
    this.classifier = classifier;
    this.config = config;
  }

  start(_video: HTMLVideoElement, handlers: RecognizerHandlers): void {
    this.handlers = handlers;
    this.classifier.reset();
    this.candidate = null;
    this.candidateFrames = 0;
    this.lastWord = null;
    this.quietFrames = 0;
    this.gapFrames = 0;
    this.pendingWords = false;
    this.handsVisible = false;
    this.unsubscribe?.();
    this.unsubscribe = frameBus.subscribe((frame) => this.processFrame(frame));
  }

  stop(): void {
    this.unsubscribe?.();
    this.unsubscribe = null;
    this.handlers = null;
    this.classifier.reset();
  }

  /** Feed one frame directly (used by the frame bus and by tests). */
  processFrame(frame: Float32Array): void {
    if (!this.handlers) return;

    const leftHand = readPart(frame, 'leftHand');
    const rightHand = readPart(frame, 'rightHand');
    const anyHands = !!leftHand || !!rightHand;

    if (anyHands !== this.handsVisible) {
      this.handsVisible = anyHands;
      this.handlers.onHands(anyHands);
    }

    if (!anyHands) {
      // Hands down: forget motion and allow the same sign again.
      this.classifier.reset();
      this.candidate = null;
      this.candidateFrames = 0;
      this.lastWord = null;
      this.noDetection();
      return;
    }

    const [primary, secondary] = this.selectHands(leftHand, rightHand);
    const result = this.classifier.classify(primary, secondary, readBody(frame));

    if (!result || result.confidence < this.config.minConfidence) {
      this.candidate = null;
      this.candidateFrames = 0;
      this.noDetection();
      return;
    }

    // Still the sign we just emitted: hold, don't repeat.
    if (result.word === this.lastWord) {
      this.quietFrames = 0;
      this.gapFrames = 0;
      return;
    }

    this.gapFrames = 0;
    if (this.candidate && this.candidate.word === result.word) {
      this.candidateFrames++;
      this.candidate = { ...this.candidate, confidence: Math.max(this.candidate.confidence, result.confidence) };
    } else {
      this.candidate = result;
      this.candidateFrames = 1;
    }

    if (this.candidateFrames >= this.config.minHoldFrames) this.emit();
  }

  private noDetection(): void {
    this.quietFrames++;
    this.gapFrames++;
    if (this.quietFrames >= REPEAT_COOLDOWN_FRAMES) this.lastWord = null;
    if (this.pendingWords && this.gapFrames >= this.config.sentenceGapFrames) {
      this.pendingWords = false;
      this.handlers?.onBoundary?.();
    }
  }

  private emit(): void {
    if (!this.candidate || !this.handlers) return;
    const event = this.candidate;
    this.lastWord = event.word;
    this.quietFrames = 0;
    this.candidate = null;
    this.candidateFrames = 0;
    this.pendingWords = !event.endsSentence;
    this.handlers.onSign(event);
  }

  private selectHands(left: Point[] | null, right: Point[] | null): [Point[] | null, Point[] | null] {
    if (this.config.preferredHand === 'left') return left ? [left, right] : [right, left];
    if (this.config.preferredHand === 'right') return right ? [right, left] : [left, right];
    return right ? [right, left] : [left, right];
  }
}

export interface BodyContext {
  /** Mouth (or nose) position, if the face/pose is tracked. */
  face: Point | null;
  /** Upper-chest position, if the pose is tracked. */
  chest: Point | null;
}

export interface RecognizerHandlers {
  onSign: (e: SignEvent) => void;
  onHands: (visible: boolean) => void;
  /** Called when a pause ends a sentence whose last sign didn't end it. */
  onBoundary?: () => void;
}

/**
 * Public interface for a sign recognizer.
 * Consumes pre-computed MediaPipe frames via the frame bus.
 */
export interface SignRecognizer {
  /** Start recognition. The video element is kept for API compatibility but not used for tracking. */
  start(video: HTMLVideoElement, handlers: RecognizerHandlers): void;
  /** Stop recognition and release resources. */
  stop(): void;
}

/**
 * Factory for the sign recognizer.
 * Returns a MediaPipe-backed recognizer wired to the frame bus.
 * The tracker (wherever it runs) MUST call `frameBus.publish(frame, timestamp)` for this to work.
 */
export function createSignRecognizer(): SignRecognizer {
  const classifier = new HeuristicSignClassifier();
  return new MediaPipeSignRecognizer(classifier, DEFAULT_CONFIG);
}

/** Shown in place of live sign captions until a model is connected. Never saved. */
export const EXAMPLE_SIGN_CAPTION = 'Hello, I would like to book an appointment for Thursday.';

interface Options {
  video: HTMLVideoElement | null;
  active: boolean;
  onSentence: (text: string) => void;
}

export function useSignRecognition({ video, active, onSentence }: Options) {
  const [words, setWords] = useState<string[]>([]);
  const [lastSign, setLastSign] = useState<SignEvent | null>(null);
  const [handsVisible, setHandsVisible] = useState<boolean | null>(null);
  const [modelConnected, setModelConnected] = useState(false);
  const sentenceCb = useRef(onSentence);
  const recognizerRef = useRef<SignRecognizer | null>(null);

  useEffect(() => {
    sentenceCb.current = onSentence;
  }, [onSentence]);

  useEffect(() => {
    if (!video) return;

    // Create recognizer on first mount
    if (!recognizerRef.current) {
      recognizerRef.current = createSignRecognizer();
      setModelConnected(true);
    }

    const r = recognizerRef.current;
    if (!active) {
      r.stop();
      setHandsVisible(null);
      setWords([]);
      setLastSign(null);
      return;
    }

    let buffer: string[] = [];
    let clearTimer: ReturnType<typeof setTimeout> | null = null;

    r.start(video, {
      onHands: setHandsVisible,
      onBoundary: () => {
        if (buffer.length === 0) return;
        sentenceCb.current(buffer.join(' '));
        buffer = [];
        clearTimer = setTimeout(() => setWords([]), 2600);
      },
      onSign: (e) => {
        if (clearTimer !== null) {
          clearTimeout(clearTimer);
          clearTimer = null;
        }
        setLastSign(e);
        buffer = [...buffer, e.word];
        setWords(buffer);
        if (e.endsSentence) {
          sentenceCb.current(buffer.join(' '));
          buffer = [];
          clearTimer = setTimeout(() => setWords([]), 2600);
        }
      },
    });

    return () => {
      r.stop();
      if (clearTimer !== null) clearTimeout(clearTimer);
    };
  }, [active, video]);

  const clear = useCallback(() => setWords([]), []);

  return { words, lastSign, handsVisible, modelConnected, clear };
}