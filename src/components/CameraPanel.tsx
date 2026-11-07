import type { CameraStatus, Lighting } from '../lib/useCamera'
import type { SignEvent } from '../lib/signRecognizer'
import { Icon } from './Icon'

interface Props {
  attach: (el: HTMLVideoElement | null) => void
  status: CameraStatus
  lighting: Lighting
  mirror: boolean
  paused: boolean
  alert: boolean
  handsVisible: boolean | null
  lastSign: SignEvent | null
  showConfidence: boolean
  showSign: boolean
  showFrame?: boolean
  onStart: () => void
}

const EMPTY_COPY: Record<Exclude<CameraStatus, 'live'>, { title: string; body: string; action?: string }> = {
  idle: { title: 'Camera is off', body: 'Turn on the camera so Visibu can read your signs. Video stays on this device.', action: 'Turn on camera' },
  requesting: { title: 'Waiting for permission…', body: 'Choose “Allow” in your browser’s prompt to use the camera.' },
  denied: { title: 'Camera access is blocked', body: 'Allow the camera for this site in your browser’s address bar, then try again.', action: 'Try again' },
  unavailable: { title: 'No camera found', body: 'Connect a camera, or open Visibu on a device that has one.', action: 'Try again' },
  error: { title: 'Camera could not start', body: 'Another app may be using it. Close it and try again.', action: 'Try again' },
}

export function CameraPanel(p: Props) {
  const live = p.status === 'live'
  const empty = p.status === 'live' ? null : EMPTY_COPY[p.status]
  const conf = p.lastSign ? Math.round(p.lastSign.confidence * 100) : 0
  const classes = ['camera', live && !p.paused && 'camera--live', p.alert && 'camera--alert'].filter(Boolean).join(' ')

  return (
    <div className={classes}>
      <video
        ref={p.attach}
        className={`camera__video${p.mirror ? ' camera__video--mirror' : ''}`}
        muted
        playsInline
        autoPlay
        aria-label="Your camera preview"
        hidden={!live}
      />

      {empty && (
        <div className="camera__empty" role="status">
          <Icon name="camera" size={40} strokeWidth={1.4} />
          <h2 style={{ fontSize: 22 }}>{empty.title}</h2>
          <p>{empty.body}</p>
          {empty.action && (
            <button type="button" className="btn btn--lg" onClick={p.onStart}>
              {empty.action}
            </button>
          )}
        </div>
      )}

      {live && (
        <>
          {p.showFrame && <div className="camera__frame" aria-hidden="true" />}
          <div className="camera__top">
            <span className="pill pill--dark">
              <span className={`dot${p.paused ? '' : ' pulse'}`} style={{ background: p.paused ? '#C9CCD1' : '#F26B5B' }} />
              {p.paused ? 'Paused' : 'Camera on'}
            </span>
            {p.handsVisible !== null && (
              <span className="pill pill--dark">
                <Icon name={p.handsVisible ? 'check' : 'alert'} size={14} strokeWidth={2} />
                {p.handsVisible ? 'Hands in frame' : 'Hands out of frame'}
              </span>
            )}
          </div>

          {p.showSign && (
            <div className="camera__sign" aria-live="off">
              <div style={{ flex: '1 1 auto', minWidth: 0 }}>
                <div className="eyebrow">Sign recognised</div>
                <div className="sign-word">{p.lastSign ? p.lastSign.word.replace(/[^\p{L}']/gu, '').toUpperCase() : 'Ready'}</div>
              </div>
              {p.showConfidence && (
                <div style={{ flex: '0 0 132px' }}>
                  <div className="row row--between" style={{ fontSize: 14, gap: 4 }}>
                    <span>Sure</span>
                    <strong>{conf}%</strong>
                  </div>
                  <div className="meter" role="meter" aria-label="Recognition confidence" aria-valuemin={0} aria-valuemax={100} aria-valuenow={conf}>
                    <div className="meter__fill" style={{ width: `${conf}%` }} />
                  </div>
                </div>
              )}
            </div>
          )}
        </>
      )}
    </div>
  )
}

const LIGHTING: Record<Lighting, { label: string; tone: 'ok' | 'warn' | 'unknown' }> = {
  good: { label: 'Good', tone: 'ok' },
  dim: { label: 'Too dark', tone: 'warn' },
  bright: { label: 'Too bright', tone: 'warn' },
  unknown: { label: 'Checking…', tone: 'unknown' },
}

export function QualityCard({ lighting, handsVisible, live }: { lighting: Lighting; handsVisible: boolean | null; live: boolean }) {
  const light = live ? LIGHTING[lighting] : { label: 'Camera off', tone: 'unknown' as const }
  const hands = !live || handsVisible === null ? { label: '—', tone: 'unknown' as const } : handsVisible ? { label: 'Good', tone: 'ok' as const } : { label: 'Move into frame', tone: 'warn' as const }
  const items = [
    { k: 'Lighting', ...light },
    { k: 'Framing', ...hands },
    { k: 'Privacy', label: 'On device', tone: 'ok' as const },
  ]
  return (
    <section className="card card--lift stack" aria-labelledby="quality-h">
      <h2 id="quality-h" className="card__title" style={{ fontSize: 16 }}>
        Camera quality
      </h2>
      <div className="quality">
        {items.map((i) => (
          <div key={i.k} className="stack" style={{ gap: 4 }}>
            <span className="quality__label">{i.k}</span>
            <span className={`quality__value quality__value--${i.tone}`}>
              {i.tone !== 'unknown' && <Icon name={i.tone === 'ok' ? 'check' : 'alert'} size={14} strokeWidth={2} />}
              {i.label}
            </span>
          </div>
        ))}
      </div>
    </section>
  )
}
