import type { CameraStatus, Lighting } from '../lib/useCamera'
import { HandOverlay } from './HandOverlay'
import { Icon } from './Icon'

interface Props {
  attach: (el: HTMLVideoElement | null) => void
  status: CameraStatus
  /** DOMException name from the last failed request, e.g. NotReadableError. */
  errorName?: string
  lighting: Lighting
  mirror: boolean
  paused: boolean
  alert: boolean
  handsVisible: boolean | null
  /** What sign translation is doing, e.g. "Signing · 3 s". Null hides the overlay. */
  signStatus: { title: string; detail: string } | null
  showFrame?: boolean
  /** Draw the tracked hand points, coloured by finger. */
  showHandPoints?: boolean
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
          <p>
            {empty.body}
            {p.status === 'error' && p.errorName ? ` (${p.errorName})` : ''}
          </p>
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
          {p.showHandPoints && !p.paused && <HandOverlay mirror={p.mirror} />}
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

          {p.signStatus && (
            <div className="camera__sign" aria-live="off">
              <div style={{ flex: '1 1 auto', minWidth: 0 }}>
                <div className="eyebrow">{p.signStatus.title}</div>
                <div className="sign-word">{p.signStatus.detail}</div>
              </div>
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
