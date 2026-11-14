import { useCallback, useEffect, useRef, useState } from 'react'
import { Header } from './components/Header'
import { Welcome } from './components/Welcome'
import { navigate, ROUTES, useRoute } from './lib/router'
import { SettingsProvider, useSettings } from './lib/settings'
import { useCamera } from './lib/useCamera'
import { History } from './screens/History'
import { MySigns } from './screens/MySigns'
import { Settings } from './screens/Settings'
import { Setup } from './screens/Setup'
import { TrackingCheck } from './screens/TrackingCheck'
import { Translate } from './screens/Translate'

function Shell() {
  const route = useRoute()
  const { settings, update } = useSettings()
  const camera = useCamera()
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

  // The shared camera only runs on screens that use it. The tracking check
  // owns its own camera and tracker.
  const stopCamera = camera.stop
  useEffect(() => {
    if (route !== 'translate' && route !== 'setup' && route !== 'signs') stopCamera()
  }, [route, stopCamera])

  const onAlert = useCallback(() => setGlowKey((k) => k + 1), [])

  return (
    <div className="app">
      <a
        className="skip-link"
        href="#main"
        onClick={(e) => {
          e.preventDefault()
          document.getElementById('main')?.focus()
        }}
      >
        Skip to content
      </a>
      <Header route={route} />

      {route === 'translate' && <Translate camera={camera} onAlert={onAlert} />}
      {route === 'signs' && <MySigns camera={camera} />}
      {route === 'setup' && <Setup camera={camera} />}
      {route === 'history' && <History />}
      {route === 'settings' && <Settings />}
      {route === 'tracking' && (
        <main id="main" className="page page--narrow" tabIndex={-1}>
          <h1 className="sr-only">Device tracking check</h1>
          <TrackingCheck />
        </main>
      )}

      {route === 'translate' && !settings.welcomeSeen && (
        <Welcome
          onDone={() => update({ welcomeSeen: true })}
          onSetup={() => {
            update({ welcomeSeen: true })
            navigate('setup')
          }}
        />
      )}

      {glowKey > 0 && <div key={glowKey} className="edge-glow" aria-hidden="true" />}
    </div>
  )
}

export function App() {
  return (
    <SettingsProvider>
      <Shell />
    </SettingsProvider>
  )
}
