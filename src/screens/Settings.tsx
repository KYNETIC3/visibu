import { useId, useState, type ReactNode } from 'react'
import { Icon } from '../components/Icon'
import { SwitchRow } from '../components/Switch'
import { ConfirmDialog } from '../components/Sheet'
import { navigate } from '../lib/router'
import { DEFAULT_SETTINGS, SIGN_LANGUAGES, TEXT_SCALE, THEMES, useSettings, type Settings as S, type SignLanguage } from '../lib/settings'

function Group({ label, foot, children }: { label: string; foot?: string; children: ReactNode }) {
  const id = useId()
  return (
    <section className="group" aria-labelledby={id}>
      <h2 id={id} className="group__label">
        {label}
      </h2>
      <div className="inset">{children}</div>
      {foot && <p className="group__foot">{foot}</p>}
    </section>
  )
}

function LinkCell({ href, children }: { href: string; children: ReactNode }) {
  return (
    <a className="cell" href={href}>
      <span className="cell__main">{children}</span>
      <span className="cell__chev">
        <Icon name="chevronRight" size={16} strokeWidth={2.2} />
      </span>
    </a>
  )
}

export function Settings() {
  const { settings, update } = useSettings()
  const [confirmReset, setConfirmReset] = useState(false)
  const set = (patch: Partial<S>) => update(patch)
  const scale = settings.textScale || 1
  const scaleLabel = Math.abs(scale - 1) < 0.01 ? 'Default' : `${Math.round(scale * 100)}%`

  return (
    <main id="main" className="narrow" tabIndex={-1}>
      <h1 className="large-title">Settings</h1>

      <Group label="Appearance">
        <div className="cell cell--block">
          <div role="group" aria-label="Appearance" className="segmented segmented--full">
            {THEMES.map((t) => (
              <button key={t.id} type="button" className="segmented__btn" aria-pressed={settings.theme === t.id} onClick={() => set({ theme: t.id })}>
                {t.label}
              </button>
            ))}
          </div>
        </div>
        <div className="cell cell--block">
          <div className="row row--between">
            <label htmlFor="text-size">Text size</label>
            <span className="cell__sub">{scaleLabel}</span>
          </div>
          <div className="text-size">
            <span aria-hidden="true" style={{ fontSize: 14, fontWeight: 600 }}>
              A
            </span>
            <input
              id="text-size"
              className="range"
              type="range"
              min={TEXT_SCALE.min}
              max={TEXT_SCALE.max}
              step={TEXT_SCALE.step}
              value={scale}
              aria-valuetext={scaleLabel}
              onChange={(e) => set({ textScale: Math.round(Number(e.target.value) * 10) / 10 })}
            />
            <span aria-hidden="true" style={{ fontSize: 24, fontWeight: 600 }}>
              A
            </span>
          </div>
        </div>
        <SwitchRow title="Reduce motion" description="Turns off animations. Visibu also follows your device setting." checked={settings.reduceMotion} onChange={(v) => set({ reduceMotion: v })} />
        <SwitchRow title="High contrast" description="Stronger text and borders." checked={settings.highContrast} onChange={(v) => set({ highContrast: v })} />
      </Group>

      <Group label="Captions">
        <div className="cell cell--block">
          <div className="row row--between">
            <label htmlFor="cap-size">Caption size</label>
            <span className="cell__sub">{settings.captionSize} px</span>
          </div>
          <input id="cap-size" className="range" type="range" min={24} max={72} step={2} value={settings.captionSize} onChange={(e) => set({ captionSize: Number(e.target.value) })} />
        </div>
        <SwitchRow title="Save conversations" description="Kept on this device only, in History." checked={settings.saveHistory} onChange={(v) => set({ saveHistory: v })} />
      </Group>

      <Group label="Sign translation" foot="You always check a translation before your partner sees it.">
        <SwitchRow title="Show other ways to say it" description="Up to three translations to choose from." checked={settings.signSuggestions} onChange={(v) => set({ signSuggestions: v })} />
        {settings.signSuggestions && (
          <SwitchRow title="Pick by signing 1, 2 or 3" description="Hold the number up for about a second." checked={settings.pickBySign} onChange={(v) => set({ pickBySign: v })} />
        )}
        <SwitchRow title="Show hand points" description="Coloured dots on each finger show what Visibu tracks." checked={settings.showHandPoints} onChange={(v) => set({ showHandPoints: v })} />
        <SwitchRow title="Show technical details" description="Confidence, recognised sign and gloss." checked={settings.showConfidence} onChange={(v) => set({ showConfidence: v })} />
        <div className="cell">
          <label htmlFor="sign-lang" className="cell__main">
            Sign language
          </label>
          <select id="sign-lang" className="select" value={settings.signLanguage} onChange={(e) => set({ signLanguage: e.target.value as SignLanguage })}>
            {SIGN_LANGUAGES.map((l) => (
              <option key={l.id} value={l.id}>
                {l.name}
              </option>
            ))}
          </select>
        </div>
      </Group>

      <Group label="Visual alerts" foot="Sound is replaced by things you can see and feel.">
        <SwitchRow title="Glow when someone speaks" checked={settings.flashOnSpeech} onChange={(v) => set({ flashOnSpeech: v })} />
        <SwitchRow title="Glow when your hands leave the frame" checked={settings.glowWhenHandsLeave} onChange={(v) => set({ glowWhenHandsLeave: v })} />
        <SwitchRow title="Vibrate on new messages" description="On phones and tablets that support it." checked={settings.vibrate} onChange={(v) => set({ vibrate: v })} />
      </Group>

      <Group label="Camera">
        <SwitchRow title="Mirror camera" checked={settings.mirrorCamera} onChange={(v) => set({ mirrorCamera: v })} />
        <LinkCell href="#/setup">Camera setup</LinkCell>
        <LinkCell href="#/tracking">Tracking check</LinkCell>
      </Group>

      <Group
        label="About"
        foot="Visibu is a student research prototype. Sign translation is experimental and has not been tested with Deaf signers. It does not replace a qualified ASL interpreter."
      >
        <button
          type="button"
          className="cell cell--link"
          onClick={() => {
            set({ welcomeSeen: false })
            navigate('translate')
          }}
        >
          Show the welcome again
        </button>
        <button type="button" className="cell cell--danger" onClick={() => setConfirmReset(true)}>
          Reset all settings
        </button>
      </Group>

      {confirmReset && (
        <ConfirmDialog
          title="Reset all settings?"
          message="Appearance, text size, captions and alerts go back to their defaults. Your conversations are kept."
          confirmLabel="Reset"
          onConfirm={() => {
            set({ ...DEFAULT_SETTINGS, setupDone: settings.setupDone, welcomeSeen: settings.welcomeSeen })
            setConfirmReset(false)
          }}
          onCancel={() => setConfirmReset(false)}
        />
      )}
    </main>
  )
}
