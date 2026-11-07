import { useCallback, useEffect, useRef, useState, type FormEvent } from 'react'
import { CameraPanel, QualityCard } from '../components/CameraPanel'
import { Icon } from '../components/Icon'
import { PartnerView } from '../components/PartnerView'
import { formatTime, useConversation } from '../lib/history'
import { useSettings } from '../lib/settings'
import { EXAMPLE_SIGN_CAPTION, useSignRecognition } from '../lib/signRecognizer'
import type { useCamera } from '../lib/useCamera'
import { useSpeechCaptions } from '../lib/useSpeechCaptions'

type Mode = 'sign' | 'speech'

const SIZES = [
  { px: 30, label: 'Small captions', fs: 14 },
  { px: 42, label: 'Medium captions', fs: 18 },
  { px: 58, label: 'Large captions', fs: 23 },
]

const QUICK_REPLIES = ['Please wait a moment', 'Could you repeat that?', 'Please write it down', 'Thank you']

interface Props {
  camera: ReturnType<typeof useCamera>
  onAlert: () => void
  setStatus: (s: 'live' | 'paused' | 'off') => void
}

export function Translate({ camera, onAlert, setStatus }: Props) {
  const { settings, update } = useSettings()
  const [mode, setMode] = useState<Mode>('sign')
  const [paused, setPaused] = useState(false)
  const [partnerText, setPartnerText] = useState<string | null>(null)
  const [copied, setCopied] = useState(false)
  const [flashKey, setFlashKey] = useState(0)
  const [recentPartner, setRecentPartner] = useState('')
  const [draft, setDraft] = useState('')
  const [videoEl, setVideoEl] = useState<HTMLVideoElement | null>(null)
  const { turns, addTurn } = useConversation(settings.saveHistory, settings.signLanguage)
  const listEnd = useRef<HTMLLIElement>(null)

  const live = camera.status === 'live'
  const attach = camera.attach
  const attachVideo = useCallback(
    (el: HTMLVideoElement | null) => {
      attach(el)
      setVideoEl(el)
    },
    [attach],
  )

  // Start the camera when the screen opens.
  const startCamera = camera.start
  useEffect(() => {
    void startCamera()
  }, [startCamera])

  useEffect(() => {
    setStatus(!live ? 'off' : paused ? 'paused' : 'live')
  }, [live, paused, setStatus])

  const sign = useSignRecognition({
    video: videoEl,
    active: live && !paused && mode === 'sign' && partnerText === null,
    onSentence: (text) => addTurn('you', 'signed', text),
  })

  const speech = useSpeechCaptions({
    lang: settings.spokenLanguage,
    onFinal: (text) => {
      addTurn('partner', 'spoke', text)
      setRecentPartner(text.trim())
      if (settings.vibrate) navigator.vibrate?.(120)
    },
    onSpeechStart: () => {
      if (settings.flashOnSpeech) {
        setFlashKey((k) => k + 1)
        onAlert()
      }
    },
  })

  const { start: startSpeech, stop: stopSpeech } = speech
  useEffect(() => {
    if (mode === 'speech' && !paused) startSpeech()
    else stopSpeech()
  }, [mode, paused, startSpeech, stopSpeech])

  // Glow when hands leave the frame, so a signer knows they are not being read.
  const prevHands = useRef(sign.handsVisible)
  useEffect(() => {
    if (prevHands.current === true && sign.handsVisible === false && settings.glowWhenHandsLeave) onAlert()
    prevHands.current = sign.handsVisible
  }, [sign.handsVisible, settings.glowWhenHandsLeave, onAlert])

  useEffect(() => {
    listEnd.current?.scrollIntoView({ block: 'nearest', behavior: settings.reduceMotion ? 'auto' : 'smooth' })
  }, [turns.length, settings.reduceMotion])

  const isSpeech = mode === 'speech'
  // Real caption text only. The example below is display-only: never copied, shown or saved.
  const captionText = isSpeech ? speech.interim || recentPartner : sign.words.join(' ')
  const showExample = !isSpeech && !sign.modelConnected && live && !paused
  const displayText = showExample ? EXAMPLE_SIGN_CAPTION : captionText
  const captionWords = displayText ? displayText.split(/\s+/) : []
  const inProgress = isSpeech ? speech.interim.length > 0 : sign.words.length > 0 && !sign.lastSign?.endsSentence
  const lastTurn = turns[turns.length - 1]

  const copy = async () => {
    const text = captionText || lastTurn?.text || ''
    try {
      await navigator.clipboard.writeText(text)
      setCopied(true)
      window.setTimeout(() => setCopied(false), 1600)
    } catch {
      /* clipboard blocked */
    }
  }

  const sendTyped = (text: string) => {
    const clean = text.trim()
    if (!clean) return
    addTurn('you', 'typed', clean)
    setPartnerText(clean)
  }

  const onSubmit = (e: FormEvent) => {
    e.preventDefault()
    sendTyped(draft)
    setDraft('')
  }

  return (
    <main id="main" className="page split" tabIndex={-1}>
      <h1 className="sr-only">Live translate</h1>

      <section aria-label="Camera" className="split__side">
        <CameraPanel
          attach={attachVideo}
          status={camera.status}
          errorName={camera.errorName}
          lighting={camera.lighting}
          mirror={settings.mirrorCamera}
          paused={paused}
          alert={isSpeech && speech.speaking}
          handsVisible={mode === 'sign' ? sign.handsVisible : null}
          lastSign={sign.lastSign}
          showConfidence={settings.showConfidence}
          showSign={mode === 'sign' && sign.modelConnected}
          onStart={() => void camera.start()}
        />

        <div className="row">
          <button type="button" className="btn" onClick={() => setPaused((p) => !p)} aria-pressed={paused}>
            <Icon name={paused ? 'play' : 'pause'} />
            {paused ? 'Resume' : 'Pause'}
          </button>
          <button type="button" className="btn" onClick={() => update({ mirrorCamera: !settings.mirrorCamera })} aria-pressed={settings.mirrorCamera}>
            <Icon name="mirror" />
            Mirror view
          </button>
          <a className="btn" href="#/setup">
            <Icon name="refresh" />
            Re-check setup
          </a>
        </div>

        <QualityCard lighting={camera.lighting} handsVisible={sign.handsVisible} live={live} />

        {!sign.modelConnected && (
          <p className="card__hint" style={{ padding: '0 4px' }}>
            <strong>Sign recognition is not connected yet.</strong> The caption shows an example sentence that is never
            saved. Speech captions, quick replies and typed messages are real.
          </p>
        )}
      </section>

      <section aria-label="Translation" className="split__main">
        <div className="row row--between" style={{ gap: 14 }}>
          <div role="group" aria-label="Who is communicating" className="segmented">
            <button type="button" className="segmented__btn" aria-pressed={!isSpeech} onClick={() => setMode('sign')}>
              <Icon name="hand" /> I'm signing
            </button>
            <button type="button" className="segmented__btn" aria-pressed={isSpeech} onClick={() => setMode('speech')}>
              <Icon name="mic" /> Partner is speaking
            </button>
          </div>
          <div role="group" aria-label="Caption size" className="row" style={{ gap: 8 }}>
            <span style={{ fontSize: 15, color: 'var(--muted)', fontWeight: 600 }}>Caption size</span>
            {SIZES.map((s) => (
              <button
                key={s.px}
                type="button"
                className="size-btn"
                aria-label={s.label}
                aria-pressed={settings.captionSize === s.px}
                onClick={() => update({ captionSize: s.px })}
                style={{ fontSize: s.fs }}
              >
                A
              </button>
            ))}
          </div>
        </div>

        <ModeBanner mode={mode} speech={speech} />

        <div key={flashKey} className={`card card--xl caption-card${flashKey ? ' caption-card--flash' : ''}`}>
          <div className="row row--between">
            {settings.showSpeaker ? (
              <span className={`pill ${isSpeech ? 'pill--partner' : 'pill--you'}`}>
                <Icon name={isSpeech ? 'mic' : 'hand'} size={16} />
                {isSpeech ? 'Partner · speaking' : 'You · signing'}
              </span>
            ) : (
              <span />
            )}
            {showExample ? (
              <span className="pill pill--muted" style={{ height: 32, fontSize: 14 }}>
                Example · not saved
              </span>
            ) : (
              <span className="card__hint">Live caption</span>
            )}
          </div>

          <p className={`caption${showExample ? ' caption--example' : ''}`} aria-label={showExample ? 'Example caption' : 'Live caption'}>
            {captionWords.length === 0 ? (
              <span className="caption__placeholder">
                {paused
                  ? 'Paused. Press Resume to continue.'
                  : isSpeech
                    ? 'Listening for your partner…'
                    : live
                      ? 'Start signing. Your words will appear here.'
                      : 'Turn on the camera to start.'}
              </span>
            ) : (
              captionWords.map((w, i) => (
                <span key={`${i}-${w}`} className={`caption__word${isSpeech && speech.interim ? ' caption__interim' : ''}`}>
                  {w}
                </span>
              ))
            )}
            {inProgress && <span className="caret" aria-hidden="true" />}
          </p>

          <div className="caption-card__actions">
            <button type="button" className="btn btn--primary" onClick={() => setPartnerText(captionText || lastTurn?.text || '')}>
              <Icon name="screen" /> Show to partner
            </button>
            <button type="button" className="btn" onClick={copy}>
              <Icon name="copy" /> {copied ? 'Copied' : 'Copy text'}
            </button>
            <button
              type="button"
              className="btn"
              onClick={() => {
                sign.clear()
                setRecentPartner('')
              }}
            >
              <Icon name="close" /> Clear
            </button>
            <span className="sr-only" role="status">
              {copied ? 'Caption copied to clipboard' : ''}
            </span>
          </div>
        </div>

        <div className="split" style={{ gap: 20 }}>
          <section className="card card--lift stack" style={{ flex: '3 1 360px', minWidth: 0 }} aria-labelledby="conv-h">
            <div className="row row--between">
              <h2 id="conv-h" className="card__title">
                Conversation
              </h2>
              <a href="#/history" style={{ fontWeight: 600, fontSize: 15 }}>
                Open history
              </a>
            </div>
            {turns.length === 0 ? (
              <p className="card__hint">Finished sentences from both of you collect here.</p>
            ) : (
              <ol className="turns" aria-live="polite" aria-relevant="additions" style={{ maxHeight: 420, overflowY: 'auto' }}>
                {turns.map((t, i) => (
                  <li key={t.id} ref={i === turns.length - 1 ? listEnd : undefined} className={`turn turn--${t.who}`}>
                    <div className="turn__meta">
                      <span>
                        {t.who === 'you' ? 'You' : 'Partner'} · {t.via}
                      </span>
                      <span className="turn__time">{formatTime(t.at)}</span>
                    </div>
                    <div className="turn__text">{t.text}</div>
                  </li>
                ))}
              </ol>
            )}
          </section>

          <section className="card card--lift stack" style={{ flex: '2 1 280px', minWidth: 0 }} aria-labelledby="quick-h">
            <div>
              <h2 id="quick-h" className="card__title">
                Quick replies
              </h2>
              <p className="card__hint">Tap to add to the conversation and show your partner.</p>
            </div>
            <div className="row" style={{ gap: 8 }}>
              {QUICK_REPLIES.map((q) => (
                <button key={q} type="button" className="chip" onClick={() => sendTyped(q)}>
                  {q}
                </button>
              ))}
            </div>
            <form className="field" onSubmit={onSubmit}>
              <label htmlFor="msg" className="field__label">
                Type instead
              </label>
              <div className="row" style={{ flexWrap: 'nowrap', gap: 8 }}>
                <input id="msg" className="input" style={{ flex: '1 1 auto' }} value={draft} onChange={(e) => setDraft(e.target.value)} placeholder="Type a message" autoComplete="off" />
                <button type="submit" className="btn btn--primary" disabled={!draft.trim()}>
                  Send
                </button>
              </div>
            </form>
          </section>
        </div>
      </section>

      {partnerText !== null && <PartnerView text={partnerText} onClose={() => setPartnerText(null)} />}
    </main>
  )
}

function ModeBanner({ mode, speech }: { mode: Mode; speech: ReturnType<typeof useSpeechCaptions> }) {
  if (mode === 'sign') {
    return (
      <div className="banner banner--you" role="status">
        <Icon name="hand" size={24} />
        <div>
          <div className="banner__title">Watching your hands</div>
          <div className="banner__text">Sign at a comfortable pace. Words appear as each sign is recognised.</div>
        </div>
      </div>
    )
  }
  if (speech.status === 'unsupported') {
    return (
      <div className="banner banner--warn" role="alert">
        <Icon name="alert" size={24} strokeWidth={2} />
        <div>
          <div className="banner__title">Speech captions aren't available in this browser</div>
          <div className="banner__text">Open Visibu in Chrome, Edge or Safari to caption your partner's speech. You can still type.</div>
        </div>
      </div>
    )
  }
  if (speech.status === 'denied') {
    return (
      <div className="banner banner--warn" role="alert">
        <Icon name="alert" size={24} strokeWidth={2} />
        <div>
          <div className="banner__title">Microphone access is blocked</div>
          <div className="banner__text">Allow the microphone for this site in your browser's address bar, then switch modes again.</div>
        </div>
      </div>
    )
  }
  return (
    <div className="banner banner--partner" role="status">
      <span className={`bars${speech.speaking ? '' : ' bars--idle'}`} aria-hidden="true">
        <span />
        <span />
        <span />
        <span />
        <span />
      </span>
      <div>
        <div className="banner__title">{speech.speaking ? 'Your partner is speaking' : 'Listening for your partner'}</div>
        <div className="banner__text">Their words appear below as they talk. The screen edge glows when speech starts.</div>
      </div>
    </div>
  )
}
