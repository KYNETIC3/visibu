import { useEffect, useRef, useState } from 'react'

/**
 * Plug point for a real sign-language model (e.g. MediaPipe hand landmarks + a
 * classifier). A recognizer reads frames from the <video> element and reports
 * recognised signs; the UI does not care how.
 */
export interface SignEvent {
  /** English word or phrase for the recognised sign. */
  word: string
  /** 0–1 confidence. */
  confidence: number
  /** True when this sign ends a sentence (pause, or a sentence-final sign). */
  endsSentence: boolean
}

export interface SignRecognizer {
  readonly isDemo: boolean
  start(video: HTMLVideoElement, handlers: { onSign: (e: SignEvent) => void; onHands: (visible: boolean) => void }): void
  stop(): void
}

/**
 * Stand-in recogniser so the experience can be designed and tested end-to-end
 * before a model is connected. It "recognises" scripted phrases on a timer.
 */
export class DemoSignRecognizer implements SignRecognizer {
  readonly isDemo = true
  private timer: number | null = null
  private phrases = [
    'Hello, I would like to book an appointment for Thursday.',
    'Could you please write that down for me?',
    'Thank you, that is very helpful.',
  ]

  start(_video: HTMLVideoElement, { onSign, onHands }: { onSign: (e: SignEvent) => void; onHands: (v: boolean) => void }) {
    this.stop()
    onHands(true)
    let p = 0
    let w = 0
    let rest = 0
    this.timer = window.setInterval(() => {
      if (rest > 0) {
        rest--
        return
      }
      const words = this.phrases[p % this.phrases.length].split(' ')
      const last = w === words.length - 1
      onSign({ word: words[w], confidence: 0.86 + Math.random() * 0.12, endsSentence: last })
      w++
      if (last) {
        w = 0
        p++
        rest = 5 // pause between sentences
      }
    }, 850)
  }

  stop() {
    if (this.timer !== null) window.clearInterval(this.timer)
    this.timer = null
  }
}

/** Swap this factory for a real model when one is available. */
export function createSignRecognizer(): SignRecognizer {
  return new DemoSignRecognizer()
}

interface Options {
  video: HTMLVideoElement | null
  active: boolean
  onSentence: (text: string) => void
}

export function useSignRecognition({ video, active, onSentence }: Options) {
  const [words, setWords] = useState<string[]>([])
  const [lastSign, setLastSign] = useState<SignEvent | null>(null)
  const [handsVisible, setHandsVisible] = useState<boolean | null>(null)
  const [r] = useState<SignRecognizer>(createSignRecognizer)
  const sentenceCb = useRef(onSentence)
  useEffect(() => {
    sentenceCb.current = onSentence
  })

  useEffect(() => {
    if (!active || !video) {
      r.stop()
      return
    }
    let buffer: string[] = []
    let clearTimer: number | null = null
    r.start(video, {
      onHands: setHandsVisible,
      onSign: (e) => {
        if (clearTimer !== null) {
          window.clearTimeout(clearTimer)
          clearTimer = null
        }
        setLastSign(e)
        buffer = [...buffer, e.word]
        setWords(buffer)
        if (e.endsSentence) {
          sentenceCb.current(buffer.join(' '))
          buffer = []
          // Keep the finished sentence readable for a moment, then clear.
          clearTimer = window.setTimeout(() => setWords([]), 2600)
        }
      },
    })
    return () => {
      r.stop()
      if (clearTimer !== null) window.clearTimeout(clearTimer)
    }
  }, [active, video, r])

  return {
    words,
    lastSign,
    handsVisible,
    isDemo: r.isDemo,
    clear: () => setWords([]),
  }
}
