import { useEffect, useState } from 'react'

export type Route = 'translate' | 'setup' | 'history' | 'settings'

export const ROUTES: { id: Route; href: string; label: string; title: string }[] = [
  { id: 'translate', href: '#/', label: 'Translate', title: 'Live translate' },
  { id: 'history', href: '#/history', label: 'History', title: 'Conversation history' },
  { id: 'settings', href: '#/settings', label: 'Accessibility', title: 'Accessibility settings' },
  { id: 'setup', href: '#/setup', label: 'Camera setup', title: 'Camera setup' },
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
