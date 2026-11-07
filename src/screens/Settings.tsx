import { useEffect, useRef, useState } from 'react'
import { Icon } from '../components/Icon'
import { SwitchRow } from '../components/Switch'
import { DEFAULT_SETTINGS, SIGN_LANGUAGES, usePrefersLessMotion, useSettings, type Settings as S, type SignLanguage } from '../lib/settings'

export function Settings() {
  const { settings, update } = useSettings()
  const lessMotion = usePrefersLessMotion()
  const [saved, setSaved] = useState(false)
  const timer = useRef<number | undefined>(undefined)

  useEffect(() => () => window.clearTimeout(timer.current), [])

  const set = (patch: Partial<S>) => {
    update(patch)
    setSaved(true)
    window.clearTimeout(timer.current)
    timer.current = window.setTimeout(() => setSaved(false), 1800)
  }

  return (
    <main id="main" className="page split" tabIndex={-1}>
      <div className="split__main">
        <div className="row row--between" style={{ alignItems: 'flex-end' }}>
          <div>
            <h1 className="page-title">Accessibility</h1>
            <p className="page-sub">Make Visibu comfortable to read and impossible to miss. Changes save automatically.</p>
          </div>
          <span role="status" aria-live="polite">
            {saved && (
              <span className="pill pill--ok toast">
                <Icon name="check" size={14} strokeWidth={2} /> Saved
              </span>
            )}
          </span>
        </div>

        <section className="card card--lift" aria-labelledby="h-cap">
          <h2 id="h-cap" className="card__title" style={{ fontSize: 20, marginBottom: 8 }}>
            Captions
          </h2>
          <div className="field" style={{ padding: '4px 0 14px' }}>
            <div className="row row--between">
              <label htmlFor="cap-size" style={{ fontWeight: 700 }}>
                Caption text size
              </label>
              <span style={{ fontWeight: 700, color: 'var(--you-ink)' }}>{settings.captionSize} px</span>
            </div>
            <input id="cap-size" className="range" type="range" min={24} max={72} step={2} value={settings.captionSize} onChange={(e) => set({ captionSize: Number(e.target.value) })} />
            <div className="row row--between card__hint" style={{ fontSize: 14 }}>
              <span>Smaller</span>
              <span>Larger</span>
            </div>
          </div>
          <SwitchRow title="Show who is talking" description="A label and icon on every caption, not only colour." checked={settings.showSpeaker} onChange={(v) => set({ showSpeaker: v })} />
          <SwitchRow title="Show recognition confidence" description="See how sure Visibu is about each sign." checked={settings.showConfidence} onChange={(v) => set({ showConfidence: v })} />
          <SwitchRow title="Save conversations" description="Keep transcripts on this device to read later." checked={settings.saveHistory} onChange={(v) => set({ saveHistory: v })} />
        </section>

        <section className="card card--lift" aria-labelledby="h-alert">
          <h2 id="h-alert" className="card__title" style={{ fontSize: 20 }}>
            Visual alerts
          </h2>
          <p className="card__hint" style={{ marginBottom: 4 }}>
            Sound is replaced by things you can see and feel.
          </p>
          <SwitchRow title="Glow when someone starts speaking" description="A soft glow around the screen and the caption area." checked={settings.flashOnSpeech} onChange={(v) => set({ flashOnSpeech: v })} />
          <SwitchRow title="Glow when your hands leave the frame" description="So you know when signs are not being read." checked={settings.glowWhenHandsLeave} onChange={(v) => set({ glowWhenHandsLeave: v })} />
          <SwitchRow title="Vibrate on new messages" description="On phones and tablets that support it." checked={settings.vibrate} onChange={(v) => set({ vibrate: v })} />
        </section>

        <section className="card card--lift" aria-labelledby="h-disp">
          <h2 id="h-disp" className="card__title" style={{ fontSize: 20 }}>
            Display and motion
          </h2>
          <SwitchRow title="High contrast captions" description="White text on black, stronger borders everywhere." checked={settings.highContrast} onChange={(v) => set({ highContrast: v })} />
          <SwitchRow title="Reduce motion" description="Turn off animations. Visibu also follows your device setting." checked={settings.reduceMotion} onChange={(v) => set({ reduceMotion: v })} />
          <SwitchRow title="Mirror camera view" description="Show your camera like a mirror." checked={settings.mirrorCamera} onChange={(v) => set({ mirrorCamera: v })} />
          <div className="opt">
            <label htmlFor="sign-lang" className="opt__title" style={{ flex: '1 1 auto' }}>
              Sign language
            </label>
            <select id="sign-lang" className="select" style={{ background: 'var(--surface)' }} value={settings.signLanguage} onChange={(e) => set({ signLanguage: e.target.value as SignLanguage })}>
              {SIGN_LANGUAGES.map((l) => (
                <option key={l.id} value={l.id}>
                  {l.name}
                </option>
              ))}
            </select>
          </div>
        </section>

        <div>
          <button
            type="button"
            className="btn"
            onClick={() => {
              if (window.confirm('Reset all accessibility settings to their defaults?')) set({ ...DEFAULT_SETTINGS, setupDone: settings.setupDone })
            }}
          >
            Reset to defaults
          </button>
        </div>
      </div>

      <aside aria-label="Caption preview" className="split__side" style={{ position: 'sticky', top: 24 }}>
        <h2 className="eyebrow" style={{ fontSize: 15, color: 'var(--ink-2)' }}>
          Preview
        </h2>
        <div className={`preview${settings.flashOnSpeech && !lessMotion ? ' preview--flash' : ''}`}>
          {settings.showSpeaker && (
            <span className="pill pill--partner" style={{ alignSelf: 'flex-start', height: 32, fontSize: 14 }}>
              <Icon name="mic" size={14} strokeWidth={1.8} /> Partner · speaking
            </span>
          )}
          <p style={{ fontSize: settings.captionSize, lineHeight: 1.2, fontWeight: 600, letterSpacing: '-0.01em', overflowWrap: 'anywhere' }}>Would morning or afternoon work better for you?</p>
          {settings.showConfidence && <p style={{ fontSize: 15, opacity: 0.85 }}>Recognition confidence: 94%</p>}
        </div>
        <p className="card__hint">This is how captions look on the Translate screen.</p>
      </aside>
    </main>
  )
}
