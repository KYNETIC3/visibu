import { Icon, type IconName } from './Icon'
import { Sheet } from './Sheet'

const POINTS: { icon: IconName; title: string; text: string }[] = [
  { icon: 'check', title: 'You approve every message', text: 'Translations can be wrong. Nothing is shown until you approve it.' },
  { icon: 'lock', title: 'Stays on this device', text: 'Your camera video is never uploaded.' },
  { icon: 'person', title: 'Not an interpreter', text: 'A research prototype. For anything important, use a qualified ASL interpreter.' },
]

/** One-time notice in place of the always-on disclaimer. It can be shown again from Settings. */
export function Welcome({ onDone, onSetup }: { onDone: () => void; onSetup: () => void }) {
  return (
    <Sheet title="Before you start" onClose={onDone} narrow hideClose>
      <ul className="welcome-list">
        {POINTS.map((p) => (
          <li key={p.title}>
            <span className="welcome-list__icon">
              <Icon name={p.icon} size={22} strokeWidth={1.8} />
            </span>
            <span>
              <span className="welcome-list__title">{p.title}</span>
              <br />
              <span className="welcome-list__text">{p.text}</span>
            </span>
          </li>
        ))}
      </ul>
      <div className="stack" style={{ gap: 8 }}>
        <button type="button" className="btn btn--primary btn--lg" data-autofocus onClick={onSetup}>
          Check my camera
        </button>
        <button type="button" className="btn btn--text btn--lg" onClick={onDone}>
          Not now
        </button>
      </div>
    </Sheet>
  )
}
