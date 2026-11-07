import type { ReactNode } from 'react'
import { ROUTES, type Route } from '../lib/router'
import { Logo } from './Icon'

export function Header({ route, end }: { route: Route; end?: ReactNode }) {
  return (
    <header className="header">
      <a href="#/" className="brand" aria-label="Visibu, live translate">
        <Logo />
        <span>Visibu</span>
      </a>
      <nav aria-label="Main" className="nav">
        {ROUTES.map((r) => (
          <a key={r.id} href={r.href} className="nav__link" aria-current={r.id === route ? 'page' : undefined}>
            {r.label}
          </a>
        ))}
      </nav>
      {end && <div className="header__end">{end}</div>}
    </header>
  )
}
