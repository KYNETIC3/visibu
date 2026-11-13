import { useCallback, useEffect, useRef, useState, type FormEvent } from 'react'
import { CameraPanel, QualityCard } from '../components/CameraPanel'
import { Icon } from '../components/Icon'
import { PartnerView } from '../components/PartnerView'
import { FingerKey } from '../components/HandOverlay'
import { SignPanel } from '../components/ProposalReview'
import { formatTime, useConversation } from '../lib/history'
import { useSettings } from '../lib/settings'
import { useMySigns } from '../lib/useMySigns'
import { EXAMPLE_SIGN_CAPTION, useSignRecognition } from '../lib/signRecognizer'
import { useTracker } from '../lib/useTracker'
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
  const signing = live && !paused && mode === 'sign' && partnerText === null

  // Choosing a suggestion by signing 1, 2 or 3 (for the oldest waiting translation).
  const pickSetting = settings.signSuggestions && settings.pickBySign
  // Sign translation proposes messages while the signer keeps signing; nothing is
  // added or shown until the signer accepts it. Holding up 1, 2 or 3 is a choice, not a message.
  const { signs: mySigns } = useMySigns()
  const sign = useSignRecognition({ video: videoEl, active: signing, mySigns })
  const pickBySign = pickSetting && (sign.proposals[0]?.suggestions.length ?? 0) > 1
  const aspect = videoEl && videoEl.videoWidth > 0 && videoEl.videoHeight > 0 ? videoEl.videoWidth / videoEl.videoHeight : 16 / 9

  // Tracker for sign recognition (runs MediaPipe Holistic in a worker). It
  // publishes frames to the bus while the signer is signing.
  useTracker({
    video: videoEl,
    active: signing,
    onError: (msg) => {
      console.error('[Tracker]', msg);
      // Could surface to UI via a toast/banner if desired
    },
  })
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
  const captionText = isSpeech ? speech.interim || recentPartner : ''
  const modelState = sign.status.state
  const modelMissing = modelState === 'unavailable' || modelState === 'error'
  const showExample = !isSpeech && modelMissing && live && !paused
  const displayText = showExample ? EXAMPLE_SIGN_CAPTION : captionText
  const captionWords = displayText ? displayText.split(/\s+/) : []
  const inProgress = isSpeech ? speech.interim.length > 0 : sign.activity === 'signing'
  const signPlaceholder =
    modelState === 'loading'
      ? 'Loading the sign models…'
      : sign.activity === 'signing'
        ? 'Signing… lower your hands when done.'
        : sign.activity === 'translating'
          ? 'Translating…'
          : 'Ready. Sign, then lower your hands.'
  // Sign status lives in the sign panel, not over the camera picture.

  const acceptProposals = (text: string, ids: number[]) => {
    addTurn('you', 'signed', text)
    sign.resolve(ids)
    setPartnerText(text)
  }
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
          signStatus={null}
          showHandPoints={settings.showHandPoints && mode === 'sign'}
          onStart={() => void camera.start()}
        />
        {settings.showHandPoints && mode === 'sign' && live && <FingerKey />}

        <div className="row" style={{ gap: 8 }}>
          <button type="button" className="btn btn--sm" onClick={() => setPaused((p) => !p)} aria-pressed={paused}>
            <Icon name={paused ? 'play' : 'pause'} />
            {paused ? 'Resume' : 'Pause'}
          </button>
          <button type="button" className="btn btn--sm" onClick={() => update({ mirrorCamera: !settings.mirrorCamera })} aria-pressed={settings.mirrorCamera}>
            <Icon name="mirror" />
            Mirror view
          </button>
          <a className="btn btn--sm" href="#/setup">
            <Icon name="refresh" />
            Re-check setup
          </a>
        </div>

        <QualityCard lighting={camera.lighting} handsVisible={sign.handsVisible} live={live} />

        <SignModelNote status={sign.status} />
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
          {isSpeech && (
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
          )}
        </div>

        <ModeBanner mode={mode} speech={speech} />

        {!isSpeech && !showExample && (
          <SignPanel
            proposals={sign.proposals}
            status={paused ? 'Paused. Press Resume to continue.' : live ? signPlaceholder : 'Turn on the camera to start.'}
            activity={paused || !live ? 'off' : sign.activity}
            notice={sign.notice}
            onDone={sign.finish}
            showConfidence={settings.showConfidence}
            showSuggestions={settings.signSuggestions}
            pickBySign={pickBySign && signing}
            aspect={aspect}
            onAccept={acceptProposals}
            onPickedBySign={sign.notePick}
            onDiscard={sign.resolve}
          />
        )}

        {(isSpeech || showExample) && (
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
                        ? signPlaceholder
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

            {!isSpeech && sign.notice && (
              <p className="card__hint" role="status" style={{ fontSize: 17 }}>
                {sign.notice}
              </p>
            )}

            <div className="caption-card__actions">
              {!isSpeech && sign.activity === 'signing' && (
                <button type="button" className="btn btn--primary" onClick={sign.finish}>
                  <Icon name="check" /> Done signing
                </button>
              )}
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
                  sign.clearNotice()
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
        )}

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
  // Signing mode shows its status in the bar under the queue instead.
  if (mode === 'sign') return null
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

function SignModelNote({ status }: { status: ReturnType<typeof useSignRecognition>['status'] }) {
  if (status.state === 'ready') {
    const m = status.model
    return (
      <p className="card__hint" style={{ padding: '0 4px' }}>
        <strong>{m.stage === 'smoke' ? 'Test model loaded.' : 'Research model, not validated with Deaf signers.'}</strong>{' '}
        Sentences: trained on How2Sign (non-commercial research licence).
        {m.words && <> Single signs: {m.words.signs} everyday signs{m.words.testTop1 !== null ? `, right first time for ${Math.round(m.words.testTop1 * 100)}% of signs by people it had not seen` : ''}.</>} Translations can be wrong, so you accept each one. Version {m.version}.
      </p>
    )
  }
  if (status.state === 'loading') {
    return (
      <p className="card__hint" style={{ padding: '0 4px' }} role="status">
        Loading the sign translation model on this device…
      </p>
    )
  }
  return (
    <p className="card__hint" style={{ padding: '0 4px' }}>
      <strong>Sign translation is not available.</strong> {status.reason} The caption shows an example sentence that is
      never saved. Speech captions, quick replies and typed messages are real.
    </p>
  )
}
