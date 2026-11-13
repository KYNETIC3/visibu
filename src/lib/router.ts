import { useEffect, useState } from 'react'

export type Route = 'translate' | 'signs' | 'setup' | 'history' | 'settings' | 'tracking'

/** `tab` routes appear in the main tab bar; the rest are reached from Settings. */
export const ROUTES: { id: Route; href: string; label: string; title: string; tab: boolean }[] = [
  { id: 'translate', href: '#/', label: 'Translate', title: 'Translate', tab: true },
  { id: 'signs', href: '#/signs', label: 'My signs', title: 'My signs', tab: true },
  { id: 'history', href: '#/history', label: 'History', title: 'History', tab: true },
  { id: 'settings', href: '#/settings', label: 'Settings', title: 'Settings', tab: false },
  { id: 'setup', href: '#/setup', label: 'Camera setup', title: 'Camera setup', tab: false },
  { id: 'tracking', href: '#/tracking', label: 'Tracking check', title: 'Tracking check', tab: false },
]

function parse(hash: string): Route {
  const path = hash.replace(/^#\/?/, '').split(/[?/]/)[0]
  return (ROUTES.find((r) => r.id === path)?.id ?? 'translate') as Route
}

/** Tiny hash router: no server config needed to deploy as static files. */
export function useRoute(): Route {
  const [route, setRoute] = useState<Route>(() => parse(window.location.hash))
  useEffect(() => {
    const on = () => setRoute(parse(window.location.hash))
    window.addEventListener('hashchange', on)
    return () => window.removeEventListener('hashchange', on)
  }, [])
  return route
}

export function navigate(route: Route) {
  window.location.hash = ROUTES.find((r) => r.id === route)!.href
}
