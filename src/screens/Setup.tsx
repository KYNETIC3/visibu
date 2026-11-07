import { useEffect, useRef, useState } from 'react'
import { CameraPanel } from '../components/CameraPanel'
import { Icon } from '../components/Icon'
import { navigate } from '../lib/router'
import { SIGN_LANGUAGES, useSettings } from '../lib/settings'
import type { useCamera } from '../lib/useCamera'

const STEPS = [
  { title: 'Allow the camera', sub: 'Stays on your device' },
  { title: 'Light and framing', sub: 'Face and hands visible' },
  { title: 'Your languages', sub: 'Sign and spoken' },
]

const SPOKEN = [
  { id: 'en-US', name: 'English (US)' },
  { id: 'en-GB', name: 'English (UK)' },
  { id: 'en-NG', name: 'English (Nigeria)' },
  { id: 'en-AU', name: 'English (Australia)' },
]

export function Setup({ camera }: { camera: ReturnType<typeof useCamera> }) {
  const { settings, update } = useSettings()
  const [step, setStep] = useState(1)
  const panelHeading = useRef<HTMLHeadingElement>(null)
  const live = camera.status === 'live'
  const first = useRef(true)

  useEffect(() => {
    if (first.current) {
      first.current = false
      return
    }
    panelHeading.current?.focus()
  }, [step])

  const lightingTip =
    camera.lighting === 'dim'
      ? 'It looks a little dark. Face a window or turn on a lamp.'
      : camera.lighting === 'bright'
        ? 'It looks very bright. Move away from light behind you.'
        : camera.lighting === 'good'
          ? 'Lighting looks good.'
          : 'Checking your lighting…'

  const finish = () => {
    update({ setupDone: true })
    navigate('translate')
  }

  return (
    <main id="main" className="page" tabIndex={-1} style={{ maxWidth: 1240 }}>
      <div className="stack" style={{ maxWidth: 720, marginBottom: 28 }}>
        <h1 className="page-title">Let's get your camera ready</h1>
        <p className="page-sub" style={{ fontSize: 19 }}>
          Three quick visual checks so Visibu can see your hands clearly. No sound is needed at any step.
        </p>
        <div className="row" style={{ gap: 12, flexWrap: 'nowrap' }}>
          <div className="meter" style={{ flex: '1 1 auto' }} role="progressbar" aria-label="Setup progress" aria-valuemin={1} aria-valuemax={3} aria-valuenow={step} aria-valuetext={`Step ${step} of 3`}>
            <div className="meter__fill" style={{ width: `${(step / 3) * 100}%` }} />
          </div>
          <span style={{ fontWeight: 700, fontSize: 15, whiteSpace: 'nowrap' }}>Step {step} of 3</span>
        </div>
      </div>

      <div className="split">
        <ol className="steps">
          {STEPS.map((s, i) => {
            const n = i + 1
            return (
              <li key={s.title}>
                <button type="button" className="step-btn" aria-current={n === step ? 'step' : undefined} onClick={() => setStep(n)}>
                  <span className={`step-num${n === step ? ' step-num--on' : n < step ? ' step-num--done' : ''}`}>
                    {n < step ? <Icon name="check" size={18} strokeWidth={2.4} /> : n}
                    {n < step && <span className="sr-only">done</span>}
                  </span>
                  <span>
                    <span className="step-title">{s.title}</span>
                    <span className="step-sub">{s.sub}</span>
                  </span>
                </button>
              </li>
            )
          })}
        </ol>

        <section className="card card--xl split__main" aria-labelledby="step-h" style={{ gap: 24 }}>
          <div key={step} className="panel-in split" style={{ gap: 32, alignItems: 'center' }}>
            <div style={{ flex: '1 1 300px', minWidth: 0 }}>
              {step === 3 ? (
                <div className="stack" style={{ aspectRatio: '4 / 3', borderRadius: 24, background: 'var(--you-tint)', alignItems: 'center', justifyContent: 'center', textAlign: 'center', padding: 24, gap: 8 }}>
                  <span className="eyebrow" style={{ color: 'var(--you-ink)' }}>
                    Signing in
                  </span>
                  <span style={{ fontSize: 56, fontWeight: 700, lineHeight: 1 }}>{settings.signLanguage}</span>
                  <span style={{ color: 'var(--you-ink)' }}>Captions in {SPOKEN.find((l) => l.id === settings.spokenLanguage)?.name ?? settings.spokenLanguage}</span>
                </div>
              ) : (
                <CameraPanel
                  attach={camera.attach}
                  status={camera.status}
                  lighting={camera.lighting}
                  mirror={settings.mirrorCamera}
                  paused={false}
                  alert={false}
                  handsVisible={null}
                  lastSign={null}
                  showConfidence={false}
                  showSign={false}
                  showFrame={step === 2}
                  onStart={() => void camera.start()}
                />
              )}
            </div>

            <div className="stack" style={{ flex: '1 1 280px', minWidth: 0 }}>
              <h2 id="step-h" ref={panelHeading} tabIndex={-1} style={{ fontSize: 28, lineHeight: 1.2, outline: 'none' }}>
                {STEPS[step - 1].title}
              </h2>

              {step === 1 && (
                <>
                  <p style={{ color: 'var(--ink-2)' }}>
                    Your browser will ask to use your camera. Choose <strong>Allow</strong>. Video stays on this device and is never recorded.
                  </p>
                  <ul style={{ margin: 0, paddingLeft: 20, color: 'var(--ink-2)' }}>
                    <li>Only your hands and upper body are read</li>
                    <li>You can turn the camera off any time</li>
                  </ul>
                  {live ? (
                    <span className="pill pill--ok" role="status" style={{ alignSelf: 'flex-start' }}>
                      <Icon name="check" size={16} strokeWidth={2} /> Camera is on
                    </span>
                  ) : (
                    <button type="button" className="btn btn--primary btn--lg" style={{ alignSelf: 'flex-start' }} onClick={() => void camera.start()} disabled={camera.status === 'requesting'}>
                      <Icon name="camera" /> {camera.status === 'requesting' ? 'Waiting for permission…' : 'Allow camera'}
                    </button>
                  )}
                </>
              )}

              {step === 2 && (
                <>
                  <p style={{ color: 'var(--ink-2)' }}>Sit an arm's length away. Keep your face and both hands inside the green box.</p>
                  <p role="status" style={{ fontWeight: 700, color: camera.lighting === 'good' ? 'var(--ok-ink)' : camera.lighting === 'unknown' ? 'var(--muted)' : '#8f4209' }}>
                    {live ? lightingTip : 'Turn on the camera in step 1 to check lighting.'}
                  </p>
                  <div className="tips">
                    <div className="tip tip--do"><strong>Do</strong>Face a window or lamp</div>
                    <div className="tip tip--avoid"><strong>Avoid</strong>Bright light behind you</div>
                    <div className="tip tip--do"><strong>Do</strong>Plain background</div>
                    <div className="tip tip--avoid"><strong>Avoid</strong>Hands too close to the lens</div>
                  </div>
                </>
              )}

              {step === 3 && (
                <>
                  <fieldset style={{ border: 0, margin: 0, padding: 0 }}>
                    <legend className="field__label" style={{ marginBottom: 8 }}>
                      Sign language
                    </legend>
                    <div className="row" style={{ gap: 8 }}>
                      {SIGN_LANGUAGES.map((l) => (
                        <button key={l.id} type="button" className="chip" aria-pressed={settings.signLanguage === l.id} aria-label={l.name} onClick={() => update({ signLanguage: l.id })}>
                          {l.id}
                        </button>
                      ))}
                    </div>
                  </fieldset>
                  <div className="field">
                    <label htmlFor="spoken" className="field__label">
                      Language your partner speaks
                    </label>
                    <select id="spoken" className="select" value={settings.spokenLanguage} onChange={(e) => update({ spokenLanguage: e.target.value })}>
                      {SPOKEN.map((l) => (
                        <option key={l.id} value={l.id}>
                          {l.name}
                        </option>
                      ))}
                    </select>
                  </div>
                </>
              )}
            </div>
          </div>

          <div className="row row--between" style={{ paddingTop: 20, borderTop: '1px solid var(--surface-muted)' }}>
            <button type="button" className="btn btn--lg" onClick={() => setStep((s) => Math.max(1, s - 1))} disabled={step === 1}>
              Back
            </button>
            {step < 3 ? (
              <button type="button" className="btn btn--primary btn--lg" onClick={() => setStep((s) => s + 1)}>
                Continue
              </button>
            ) : (
              <button type="button" className="btn btn--primary btn--lg" onClick={finish}>
                Start translating
              </button>
            )}
          </div>
        </section>
      </div>
    </main>
  )
}
