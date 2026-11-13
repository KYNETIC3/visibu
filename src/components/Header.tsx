import type { ReactNode } from 'react'
import { ROUTES, type Route } from '../lib/router'
import { Icon, type IconName } from './Icon'

const TAB_ICONS: Partial<Record<Route, IconName>> = {
  translate: 'hand',
  signs: 'bookmark',
  history: 'clock',
}

/** Brand, a glass tab capsule (Translate, My signs, History) and the Settings button. */
export function Header({ route, end }: { route: Route; end?: ReactNode }) {
  return (
    <header className="header">
      <a href="#/" className="brand" aria-label="Visibu, translate">
        <span>Visibu</span>
      </a>
      <nav aria-label="Main" className="nav glass">
        {ROUTES.filter((r) => r.tab).map((r) => (
          <a key={r.id} href={r.href} className="nav__link" aria-current={r.id === route ? 'page' : undefined}>
            <Icon name={TAB_ICONS[r.id] ?? 'hand'} size={18} strokeWidth={1.7} />
            {r.label}
          </a>
        ))}
      </nav>
      <div className="header__end">
        {end}
        <a href="#/settings" className="icon-btn glass" aria-label="Settings" aria-current={route === 'settings' ? 'page' : undefined}>
          <Icon name="gear" size={20} strokeWidth={1.6} />
        </a>
      </div>
    </header>
  )
}
