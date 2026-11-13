import { useCallback, useEffect, useRef, useState, type FormEvent } from 'react'
import { CameraPanel } from '../components/CameraPanel'
import { Icon } from '../components/Icon'
import { PartnerView } from '../components/PartnerView'
import { SignPanel } from '../components/ProposalReview'
import { Sheet } from '../components/Sheet'
import { formatTime, useConversation } from '../lib/history'
import { navigate } from '../lib/router'
import { useSettings } from '../lib/settings'
import { useMySigns } from '../lib/useMySigns'
import { useSignRecognition } from '../lib/signRecognizer'
import { useTracker } from '../lib/useTracker'
import type { useCamera } from '../lib/useCamera'
import { useSpeechCaptions } from '../lib/useSpeechCaptions'

type Mode = 'sign' | 'speech'

const QUICK_REPLIES = ['One moment', 'Could you repeat that?', 'Thank you']

interface Props {
  camera: ReturnType<typeof useCamera>
  onAlert: () => void
}

/**
 * One job per screen: the camera, one translation to check, and a reply bar.
 * Everything else (the conversation so far, settings, setup) sits one tap away.
 */
export function Translate({ camera, onAlert }: Props) {
  const { settings, update } = useSettings()
  const [mode, setMode] = useState<Mode>('sign')
  const [paused, setPaused] = useState(false)
  const [partnerText, setPartnerText] = useState<string | null>(null)
  const [showConversation, setShowConversation] = useState(false)
  const [flashKey, setFlashKey] = useState(0)
  const [recentPartner, setRecentPartner] = useState('')
  const [draft, setDraft] = useState('')
  const [videoEl, setVideoEl] = useState<HTMLVideoElement | null>(null)
  const { turns, addTurn } = useConversation(settings.saveHistory, settings.signLanguage)

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
      console.error('[Tracker]', msg)
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
    if (mode === 'speech' && !paused && partnerText === null) startSpeech()
    else stopSpeech()
  }, [mode, paused, partnerText, startSpeech, stopSpeech])

  // Glow when hands leave the frame, so a signer knows they are not being read.
  const prevHands = useRef(sign.handsVisible)
  useEffect(() => {
    if (prevHands.current === true && sign.handsVisible === false && settings.glowWhenHandsLeave) onAlert()
    prevHands.current = sign.handsVisible
  }, [sign.handsVisible, settings.glowWhenHandsLeave, onAlert])

  const isSpeech = mode === 'speech'
  const modelState = sign.status.state
  const modelMissing = modelState === 'unavailable' || modelState === 'error'

  let status = 'Ready'
  let hint = 'Sign your message, then lower your hands.'
  if (!live) {
    status = 'Camera off'
    hint = 'Turn on the camera to start.'
  } else if (paused) {
    status = 'Paused'
    hint = 'Press play on the camera to continue.'
  } else if (modelState === 'loading') {
    status = 'Getting ready'
    hint = 'Loading the sign models on this device…'
  } else if (sign.activity === 'signing') {
    status = 'Reading your signs'
    hint = 'Signing… lower your hands when you’re done.'
  } else if (sign.activity === 'translating') {
    status = 'Translating'
    hint = 'Translating…'
  }

  const captionText = speech.interim || recentPartner
  const captionWords = captionText ? captionText.split(/\s+/) : []

  const acceptProposals = (text: string, ids: number[]) => {
    addTurn('you', 'signed', text)
    sign.resolve(ids)
    setPartnerText(text)
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
    <main id="main" className="translate" tabIndex={-1}>
      <h1 className="sr-only">Translate</h1>

      <div className="translate__top">
        <div role="group" aria-label="Who is communicating" className="segmented">
          <button type="button" className="segmented__btn" aria-pressed={!isSpeech} onClick={() => setMode('sign')}>
            <Icon name="hand" /> I'm signing
          </button>
          <button type="button" className="segmented__btn" aria-pressed={isSpeech} onClick={() => setMode('speech')}>
            <Icon name="mic" /> Partner speaking
          </button>
        </div>
        <button
          type="button"
          className="icon-btn glass"
          aria-label={turns.length ? `Conversation so far, ${turns.length} message${turns.length === 1 ? '' : 's'}` : 'Conversation so far'}
          onClick={() => setShowConversation(true)}
        >
          <Icon name="bubble" size={20} strokeWidth={1.7} />
          {turns.length > 0 && (
            <span className="badge" aria-hidden="true">
              {turns.length}
            </span>
          )}
        </button>
      </div>

      <div className="translate__body">
        {!isSpeech && (
          <>
            <section aria-label="Camera" className="translate__camera">
              <CameraPanel
                attach={attachVideo}
                status={camera.status}
                errorName={camera.errorName}
                lighting={camera.lighting}
                mirror={settings.mirrorCamera}
                paused={paused}
                alert={false}
                handsVisible={sign.handsVisible}
                signStatus={null}
                showHandPoints={settings.showHandPoints}
                onStart={() => void camera.start()}
                controls={
                  <>
                    <button type="button" onClick={() => setPaused((p) => !p)} aria-label={paused ? 'Resume' : 'Pause'}>
                      <Icon name={paused ? 'play' : 'pause'} size={20} />
                    </button>
                    <button type="button" onClick={() => update({ mirrorCamera: !settings.mirrorCamera })} aria-label="Mirror camera" aria-pressed={settings.mirrorCamera}>
                      <Icon name="mirror" size={20} />
                    </button>
                  </>
                }
              />
            </section>

            <section aria-label="Translation" className="translate__card">
              <div className="tcard">
                {modelMissing ? (
                  <>
                    <span className="tcard__status">
                      <span className="tcard__dot" aria-hidden="true" /> Sign translation unavailable
                    </span>
                    <p className="tcard__hint">You can still type, use a quick reply, or switch to Partner speaking.</p>
                    {sign.status.state !== 'ready' && 'reason' in sign.status && <p className="tcard__note">{String(sign.status.reason)}</p>}
                  </>
                ) : (
                  <SignPanel
                    proposals={sign.proposals}
                    status={status}
                    hint={hint}
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
              </div>
            </section>
          </>
        )}

        {isSpeech && (
          <section aria-label="Live caption" key={flashKey} className={`listen-card${flashKey ? ' caption-card--flash' : ''}`}>
            <div className="row row--between">
              <span className={`listen-card__label${speech.speaking ? ' listen-card__label--on' : ''}`}>
                <span className={`bars${speech.speaking && !paused ? '' : ' bars--idle'}`} aria-hidden="true">
                  <span />
                  <span />
                  <span />
                  <span />
                  <span />
                </span>
                {paused ? 'Paused' : speech.speaking ? 'Your partner is speaking' : 'Listening'}
              </span>
              <button type="button" className="btn btn--sm" onClick={() => setPaused((p) => !p)}>
                <Icon name={paused ? 'play' : 'pause'} /> {paused ? 'Resume' : 'Pause'}
              </button>
            </div>

            <SpeechProblem status={speech.status} />

            <p className="caption" aria-live="polite">
              {captionWords.length === 0 ? (
                <span className="caption__placeholder">{paused ? 'Paused.' : 'Listening for your partner…'}</span>
              ) : (
                captionWords.map((w, i) => (
                  <span key={`${i}-${w}`} className={`caption__word${speech.interim ? ' caption__interim' : ''}`}>
                    {w}
                  </span>
                ))
              )}
              {speech.interim.length > 0 && <span className="caret" aria-hidden="true" />}
            </p>
          </section>
        )}
      </div>

      <form aria-label="Reply without signing" className="reply-bar glass" onSubmit={onSubmit}>
        <div className="reply-bar__chips">
          {QUICK_REPLIES.map((q) => (
            <button key={q} type="button" className="chip" onClick={() => sendTyped(q)}>
              {q}
            </button>
          ))}
        </div>
        <div className="reply-bar__input">
          <label htmlFor="msg" className="sr-only">
            Type a message
          </label>
          <input id="msg" className="input" value={draft} onChange={(e) => setDraft(e.target.value)} placeholder="Type a message" autoComplete="off" />
          <button type="submit" className="round-send" aria-label="Send" disabled={!draft.trim()}>
            <Icon name="arrowUp" size={20} strokeWidth={2.2} />
          </button>
        </div>
      </form>

      {showConversation && (
        <Sheet
          title="Conversation"
          onClose={() => setShowConversation(false)}
          footer={
            <button
              type="button"
              className="btn btn--text"
              onClick={() => {
                setShowConversation(false)
                navigate('history')
              }}
            >
              See all conversations in History
            </button>
          }
        >
          {turns.length === 0 ? (
            <p className="card__hint">Messages you and your partner share collect here.</p>
          ) : (
            <ol className="bubbles" aria-label="Messages">
              {turns.map((t) => (
                <li key={t.id} className={`bubble bubble--${t.who}`}>
                  <span className="bubble__meta">
                    {t.who === 'you' ? 'You' : 'Partner'} · {formatTime(t.at)}
                  </span>
                  <span className="bubble__text">{t.text}</span>
                </li>
              ))}
            </ol>
          )}
        </Sheet>
      )}

      {partnerText !== null && (
        <PartnerView
          text={partnerText}
          onClose={() => setPartnerText(null)}
          onReply={() => {
            setPartnerText(null)
            setMode('speech')
          }}
        />
      )}
    </main>
  )
}

function SpeechProblem({ status }: { status: ReturnType<typeof useSpeechCaptions>['status'] }) {
  if (status === 'unsupported') {
    return (
      <div className="banner banner--warn" role="alert">
        <Icon name="alert" size={22} strokeWidth={2} />
        <div>
          <div className="banner__title">Speech captions aren’t available in this browser</div>
          <div className="banner__text">Open Visibu in Chrome, Edge or Safari. You can still type.</div>
        </div>
      </div>
    )
  }
  if (status === 'denied') {
    return (
      <div className="banner banner--warn" role="alert">
        <Icon name="alert" size={22} strokeWidth={2} />
        <div>
          <div className="banner__title">Microphone access is blocked</div>
          <div className="banner__text">Allow the microphone for this site in your browser’s address bar.</div>
        </div>
      </div>
    )
  }
  return null
}
