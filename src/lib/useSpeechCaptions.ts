import { useCallback, useEffect, useRef, useState } from 'react'

/* Minimal typings for the Web Speech API (not in lib.dom yet). */
interface SpeechAlternative { transcript: string }
interface SpeechResult { isFinal: boolean; 0: SpeechAlternative; length: number }
interface SpeechResultEvent { resultIndex: number; results: ArrayLike<SpeechResult> }
interface SpeechErrorEvent { error: string }
interface Recognition {
  lang: string
  continuous: boolean
  interimResults: boolean
  start(): void
  stop(): void
  abort(): void
  onresult: ((e: SpeechResultEvent) => void) | null
  onerror: ((e: SpeechErrorEvent) => void) | null
  onend: (() => void) | null
  onspeechstart: (() => void) | null
}
type RecognitionCtor = new () => Recognition

function getCtor(): RecognitionCtor | null {
  const w = window as unknown as { SpeechRecognition?: RecognitionCtor; webkitSpeechRecognition?: RecognitionCtor }
  return w.SpeechRecognition ?? w.webkitSpeechRecognition ?? null
}

export type SpeechStatus = 'off' | 'listening' | 'unsupported' | 'denied' | 'error'

interface Options {
  lang: string
  onFinal: (text: string) => void
  onSpeechStart?: () => void
}

/**
 * Live captions for the hearing partner, using the browser's speech recognition.
 * `interim` is the sentence still being spoken; `onFinal` fires per finished phrase.
 */
export function useSpeechCaptions({ lang, onFinal, onSpeechStart }: Options) {
  const supported = getCtor() !== null
  const [status, setStatus] = useState<SpeechStatus>(supported ? 'off' : 'unsupported')
  const [interim, setInterim] = useState('')
  const [speaking, setSpeaking] = useState(false)
  const recRef = useRef<Recognition | null>(null)
  const wantRef = useRef(false)
  const cbRef = useRef({ onFinal, onSpeechStart })
  useEffect(() => {
    cbRef.current = { onFinal, onSpeechStart }
  })

  const stop = useCallback(() => {
    wantRef.current = false
    recRef.current?.abort()
    recRef.current = null
    setInterim('')
    setSpeaking(false)
    setStatus((s) => (s === 'unsupported' ? s : 'off'))
  }, [])

  const start = useCallback(() => {
    const Ctor = getCtor()
    if (!Ctor) {
      setStatus('unsupported')
      return
    }
    wantRef.current = true
    const rec = new Ctor()
    rec.lang = lang
    rec.continuous = true
    rec.interimResults = true
    rec.onspeechstart = () => {
      setSpeaking(true)
      cbRef.current.onSpeechStart?.()
    }
    rec.onresult = (e) => {
      let live = ''
      for (let i = e.resultIndex; i < e.results.length; i++) {
        const r = e.results[i]
        const text = r[0].transcript
        if (r.isFinal) {
          cbRef.current.onFinal(text)
          setSpeaking(false)
        } else {
          live += text
        }
      }
      setInterim(live.trim())
    }
    rec.onerror = (e) => {
      if (e.error === 'not-allowed' || e.error === 'service-not-allowed') {
        wantRef.current = false
        setStatus('denied')
      } else if (e.error !== 'no-speech' && e.error !== 'aborted') {
        setStatus('error')
      }
    }
    // Browsers end recognition after silence; restart while the user wants captions.
    rec.onend = () => {
      if (wantRef.current) {
        try {
          rec.start()
        } catch {
          /* already started */
        }
      }
    }
    recRef.current = rec
    try {
      rec.start()
      setStatus('listening')
    } catch {
      setStatus('error')
    }
  }, [lang])

  useEffect(() => () => {
    wantRef.current = false
    recRef.current?.abort()
  }, [])

  return { supported, status, interim, speaking, start, stop }
}
