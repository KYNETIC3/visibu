import { useCallback, useEffect, useRef, useState } from 'react'
import { Header } from './components/Header'
import { Icon } from './components/Icon'
import { ROUTES, useRoute } from './lib/router'
import { SettingsProvider, SIGN_LANGUAGES, useSettings } from './lib/settings'
import { useCamera } from './lib/useCamera'
import { History } from './screens/History'
import { Settings } from './screens/Settings'
import { Setup } from './screens/Setup'
import { Translate } from './screens/Translate'

type LiveStatus = 'live' | 'paused' | 'off'

function Shell() {
  const route = useRoute()
  const { settings } = useSettings()
  const camera = useCamera()
  const [status, setStatus] = useState<LiveStatus>('off')
  const [glowKey, setGlowKey] = useState(0)
  const firstRender = useRef(true)

  // Screen title + move focus to the new page for keyboard and screen-reader users.
  useEffect(() => {
    const r = ROUTES.find((x) => x.id === route)!
    document.title = `${r.title} · Visibu`
    if (firstRender.current) {
      firstRender.current = false
      return
    }
    document.getElementById('main')?.focus({ preventScroll: true })
    window.scrollTo(0, 0)
  }, [route])

  // The camera only runs on screens that use it.
  const stopCamera = camera.stop
  useEffect(() => {
    if (route !== 'translate' && route !== 'setup') stopCamera()
  }, [route, stopCamera])

  const onAlert = useCallback(() => setGlowKey((k) => k + 1), [])
  const signName = SIGN_LANGUAGES.find((l) => l.id === settings.signLanguage)?.id ?? settings.signLanguage

  const headerEnd =
    route === 'translate' ? (
      <>
        {status === 'live' && (
          <span className="pill pill--ok">
            <span className="dot pulse" /> Live · translating
          </span>
        )}
        {status === 'paused' && (
          <span className="pill pill--muted">
            <Icon name="pause" size={14} /> Paused
          </span>
        )}
        {status === 'off' && (
          <span className="pill pill--muted">
            <Icon name="camera" size={14} /> Camera off
          </span>
        )}
        <a className="btn" href="#/settings" aria-label={`Languages: ${signName} to English. Change in settings.`} style={{ minHeight: 44 }}>
          {signName} → English
          <Icon name="chevron" size={16} strokeWidth={1.8} />
        </a>
      </>
    ) : route === 'setup' ? (
      <a href="#/" style={{ fontWeight: 600 }}>
        Skip for now
      </a>
    ) : null

  return (
    <div className="app">
      <a className="skip-link" href="#main" onClick={(e) => { e.preventDefault(); document.getElementById('main')?.focus() }}>
        Skip to content
      </a>
      <Header route={route} end={headerEnd} />

      {route === 'translate' && !settings.setupDone && (
        <div className="page" style={{ paddingBottom: 0 }}>
          <div className="banner banner--you" role="note">
            <Icon name="camera" size={24} />
            <div style={{ flex: '1 1 auto' }}>
              <div className="banner__title">New here? Check your camera first</div>
              <div className="banner__text">A one-minute visual guide to lighting and framing helps Visibu read your signs.</div>
            </div>
            <a className="btn" href="#/setup">
              Start setup
            </a>
          </div>
        </div>
      )}

      {route === 'translate' && <Translate camera={camera} onAlert={onAlert} setStatus={setStatus} />}
      {route === 'setup' && <Setup camera={camera} />}
      {route === 'history' && <History />}
      {route === 'settings' && <Settings />}

      {glowKey > 0 && <div key={glowKey} className="edge-glow" aria-hidden="true" />}
    </div>
  )
}

export default function App() {
  return (
    <SettingsProvider>
      <Shell />
    </SettingsProvider>
  )
}
