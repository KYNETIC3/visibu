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
  start(video: HTMLVideoElement, handlers: { onSign: (e: SignEvent) => void; onHands: (visible: boolean) => void }): void
  stop(): void
}

/**
 * Return a real recognizer here once a model is available. While this returns
 * null, the Translate screen shows a static example caption that is never
 * added to the conversation or saved to history.
 */
export function createSignRecognizer(): SignRecognizer | null {
  return null
}

/** Shown in place of live sign captions until a model is connected. Never saved. */
export const EXAMPLE_SIGN_CAPTION = 'Hello, I would like to book an appointment for Thursday.'

interface Options {
  video: HTMLVideoElement | null
  active: boolean
  onSentence: (text: string) => void
}

export function useSignRecognition({ video, active, onSentence }: Options) {
  const [words, setWords] = useState<string[]>([])
  const [lastSign, setLastSign] = useState<SignEvent | null>(null)
  const [handsVisible, setHandsVisible] = useState<boolean | null>(null)
  const [r] = useState<SignRecognizer | null>(createSignRecognizer)
  const sentenceCb = useRef(onSentence)
  useEffect(() => {
    sentenceCb.current = onSentence
  })

  useEffect(() => {
    if (!r) return
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
    modelConnected: r !== null,
    clear: () => setWords([]),
  }
}
