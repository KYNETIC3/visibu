import { useEffect, useRef, useState, useCallback } from 'react';
import { readPart, hasPart, type Point } from '../tracking/frame';

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
   * @param leftHand  21 landmarks (x,y,z normalized 0–1) or null if not visible
   * @param rightHand 21 landmarks (x,y,z normalized 0–1) or null if not visible
   * @returns Recognised sign or null if no sign detected in this frame
   */
  classify(leftHand: Point[] | null, rightHand: Point[] | null): SignEvent | null;

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

/**
 * Heuristic-based sign classifier (reference implementation).
 * Recognises a tiny vocabulary from hand shape + motion heuristics.
 * Replace with a real ML model by implementing SignClassifier.
 */
class HeuristicSignClassifier implements SignClassifier {
  private lastLeft: Point[] | null = null;
  private lastRight: Point[] | null = null;
  private frameCount = 0;

  classify(leftHand: Point[] | null, rightHand: Point[] | null): SignEvent | null {
    this.frameCount++;

    // Simple heuristic: detect "HELLO" (open hand wave) and "THANK YOU" (fingers to chin)
    // This is a DEMONSTRATION ONLY — swap for a trained model.

    const activeHand = this.selectHand(leftHand, rightHand);
    if (!activeHand) return null;

    const isOpen = this.isOpenHand(activeHand);
    const isFist = this.isFist(activeHand);
    const isPointing = this.isPointing(activeHand);
    const movedUp = this.movedUp(activeHand);

    // HELLO: open hand, moving up/out
    if (isOpen && movedUp && this.frameCount > 10) {
      return { word: 'HELLO', confidence: 0.82, endsSentence: true };
    }

    // THANK YOU: flat hand moving from chin forward
    if (isOpen && !movedUp && this.isNearFace(activeHand)) {
      return { word: 'THANK YOU', confidence: 0.78, endsSentence: true };
    }

    // YES: fist nodding
    if (isFist && movedUp) {
      return { word: 'YES', confidence: 0.75, endsSentence: true };
    }

    // NO: index finger wagging
    if (isPointing && this.wagging(activeHand)) {
      return { word: 'NO', confidence: 0.72, endsSentence: true };
    }

    // PLEASE: flat hand circling on chest
    if (isOpen && this.isCircling(activeHand)) {
      return { word: 'PLEASE', confidence: 0.7, endsSentence: false };
    }

    return null;
  }

  reset(): void {
    this.lastLeft = null;
    this.lastRight = null;
    this.frameCount = 0;
  }

  private selectHand(left: Point[] | null, right: Point[] | null): Point[] | null {
    if (left && !right) return left;
    if (right && !left) return right;
    if (left && right) {
      // Prefer hand with higher average confidence (visible landmarks)
      const leftScore = this.visibilityScore(left);
      const rightScore = this.visibilityScore(right);
      return leftScore >= rightScore ? left : right;
    }
    return null;
  }

  private visibilityScore(hand: Point[]): number {
    let sum = 0;
    for (const p of hand) sum += Number.isFinite(p.x) && Number.isFinite(p.y) && Number.isFinite(p.z) ? 1 : 0;
    return sum / hand.length;
  }

  private isOpenHand(hand: Point[]): boolean {
    // Fingers extended: tips farther from wrist than MCPs
    const wrist = hand[0];
    const tips = [4, 8, 12, 16, 20]; // thumb, index, middle, ring, pinky tips
    const mcps = [2, 5, 9, 13, 17];
    let extended = 0;
    for (let i = 0; i < tips.length; i++) {
      const tip = hand[tips[i]];
      const mcp = hand[mcps[i]];
      if (tip && mcp && this.dist(tip, wrist) > this.dist(mcp, wrist) * 1.2) extended++;
    }
    return extended >= 4;
  }

  private isFist(hand: Point[]): boolean {
    const wrist = hand[0];
    const tips = [4, 8, 12, 16, 20];
    let curled = 0;
    for (const tipIdx of tips) {
      const tip = hand[tipIdx];
      if (tip && this.dist(tip, wrist) < this.dist(hand[5], wrist) * 0.8) curled++;
    }
    return curled >= 4;
  }

  private isPointing(hand: Point[]): boolean {
    const wrist = hand[0];
    const indexTip = hand[8];
    const indexMcp = hand[5];
    const middleTip = hand[12];
    if (!indexTip || !indexMcp || !middleTip) return false;
    return (
      this.dist(indexTip, wrist) > this.dist(indexMcp, wrist) * 1.5 &&
      this.dist(middleTip, wrist) < this.dist(hand[9], wrist) * 1.1
    );
  }

  private movedUp(hand: Point[]): boolean {
    if (!this.lastLeft && !this.lastRight) return false;
    const last = this.lastLeft ?? this.lastRight;
    if (!last) return false;
    const currWrist = hand[0];
    const lastWrist = last[0];
    return currWrist && lastWrist && currWrist.y < lastWrist.y - 0.02;
  }

  private isNearFace(hand: Point[]): boolean {
    const wrist = hand[0];
    // Face landmarks are in upper portion of frame; wrist y < 0.4 is rough heuristic
    return wrist && wrist.y < 0.4;
  }

  private wagging(hand: Point[]): boolean {
    // Simplified: horizontal oscillation of index tip
    if (!this.lastLeft && !this.lastRight) return false;
    const last = this.lastLeft ?? this.lastRight;
    if (!last) return false;
    const currTip = hand[8];
    const lastTip = last[8];
    return currTip && lastTip && Math.abs(currTip.x - lastTip.x) > 0.03;
  }

  private isCircling(_hand: Point[]): boolean {
    // Placeholder for circular motion detection
    return false;
  }

  private dist(a: Point, b: Point): number {
    const dx = a.x - b.x;
    const dy = a.y - b.y;
    const dz = a.z - b.z;
    return Math.sqrt(dx * dx + dy * dy + dz * dz);
  }
}

/**
 * MediaPipe-backed sign recognizer.
 * Subscribes to the frame bus, runs the classifier, emits SignEvents via callbacks.
 */
class MediaPipeSignRecognizer implements SignRecognizer {
  private classifier: SignClassifier;
  private config: SignRecognizerConfig;
  private unsubscribe: (() => void) | null = null;
  private handlers: { onSign: (e: SignEvent) => void; onHands: (visible: boolean) => void } | null = null;

  // Temporal smoothing state
  private candidate: SignEvent | null = null;
  private candidateFrames = 0;
  private lastEmitted: SignEvent | null = null;
  private gapFrames = 0;
  private handsVisible = false;

  constructor(classifier: SignClassifier, config: SignRecognizerConfig = DEFAULT_CONFIG) {
    this.classifier = classifier;
    this.config = config;
  }

  start(
    _video: HTMLVideoElement,
    handlers: { onSign: (e: SignEvent) => void; onHands: (visible: boolean) => void }
  ): void {
    this.handlers = handlers;
    this.classifier.reset();
    this.candidate = null;
    this.candidateFrames = 0;
    this.lastEmitted = null;
    this.gapFrames = 0;
    this.handsVisible = false;

    this.unsubscribe = frameBus.subscribe((frame, _timestamp) => this.processFrame(frame));
  }

  stop(): void {
    if (this.unsubscribe) {
      this.unsubscribe();
      this.unsubscribe = null;
    }
    this.handlers = null;
    this.classifier.reset();
  }

  private processFrame(frame: Float32Array): void {
    if (!this.handlers) return;

    const leftHand = hasPart(frame, 'leftHand') ? readPart(frame, 'leftHand') : null;
    const rightHand = hasPart(frame, 'rightHand') ? readPart(frame, 'rightHand') : null;
    const anyHands = !!leftHand || !!rightHand;

    // Hand visibility callback
    if (anyHands !== this.handsVisible) {
      this.handsVisible = anyHands;
      this.handlers.onHands(anyHands);
    }

    if (!anyHands) {
      this.gapFrames++;
      if (this.gapFrames >= this.config.sentenceGapFrames && this.candidate) {
        this.flushCandidate(true);
      }
      return;
    }

    this.gapFrames = 0;

    // Select hand based on config
    const [primary, secondary] = this.selectHands(leftHand, rightHand);
    if (!primary) return;

    // Classify
    const result = this.classifier.classify(primary, secondary);
    if (!result || result.confidence < this.config.minConfidence) {
      // Low confidence or no detection — increment gap, maybe flush
      this.gapFrames++;
      if (this.gapFrames >= this.config.sentenceGapFrames && this.candidate) {
        this.flushCandidate(true);
      }
      return;
    }

    // Temporal smoothing: require N consistent frames
    if (this.candidate && this.sameSign(this.candidate, result)) {
      this.candidateFrames++;
      // Update confidence to running max
      this.candidate = { ...this.candidate, confidence: Math.max(this.candidate.confidence, result.confidence) };
    } else {
      // New sign or first detection
      this.candidate = result;
      this.candidateFrames = 1;
    }

    // Emit when held long enough
    if (this.candidateFrames >= this.config.minHoldFrames && this.candidate !== this.lastEmitted) {
      this.flushCandidate(false);
    }
  }

  private selectHands(left: Point[] | null, right: Point[] | null): [Point[] | null, Point[] | null] {
    switch (this.config.preferredHand) {
      case 'left':
        return [left, right];
      case 'right':
        return [right, left];
      case 'auto':
      default: {
        // Prefer hand with more visible landmarks
        const leftScore = left ? this.visibilityScore(left) : 0;
        const rightScore = right ? this.visibilityScore(right) : 0;
        return leftScore >= rightScore ? [left, right] : [right, left];
      }
    }
  }

  private visibilityScore(hand: Point[]): number {
    let sum = 0;
    for (const p of hand) sum += Number.isFinite(p.x) && Number.isFinite(p.y) && Number.isFinite(p.z) ? 1 : 0;
    return sum / hand.length;
  }

  private sameSign(a: SignEvent, b: SignEvent): boolean {
    return a.word === b.word;
  }

  private flushCandidate(endsSentence: boolean): void {
    if (!this.candidate || !this.handlers) return;
    const event: SignEvent = { ...this.candidate, endsSentence };
    this.handlers.onSign(event);
    this.lastEmitted = this.candidate;
    this.candidate = null;
    this.candidateFrames = 0;
  }
}

/**
 * Public interface for a sign recognizer.
 * Consumes pre-computed MediaPipe frames via the frame bus.
 */
export interface SignRecognizer {
  /** Start recognition. The video element is kept for API compatibility but not used for tracking. */
  start(
    video: HTMLVideoElement,
    handlers: { onSign: (e: SignEvent) => void; onHands: (visible: boolean) => void }
  ): void;
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