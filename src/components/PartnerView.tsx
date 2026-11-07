import { useEffect, useRef } from 'react'
import { Icon } from './Icon'

/** Full-screen, very large caption to turn toward a hearing partner. */
export function PartnerView({ text, onClose }: { text: string; onClose: () => void }) {
  const closeRef = useRef<HTMLButtonElement>(null)

  useEffect(() => {
    const previous = document.activeElement as HTMLElement | null
    closeRef.current?.focus()
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose()
      if (e.key === 'Tab') {
        // Only one control in the dialog: keep focus on it.
        e.preventDefault()
        closeRef.current?.focus()
      }
    }
    document.addEventListener('keydown', onKey)
    document.body.style.overflow = 'hidden'
    return () => {
      document.removeEventListener('keydown', onKey)
      document.body.style.overflow = ''
      previous?.focus()
    }
  }, [onClose])

  return (
    <div className="partner-view" role="dialog" aria-modal="true" aria-labelledby="pv-label" aria-describedby="pv-text">
      <div className="row row--between">
        <span id="pv-label" style={{ fontSize: 18, fontWeight: 700, color: 'var(--you-ink)' }}>
          Translated by Visibu · turn the screen toward your partner
        </span>
        <button ref={closeRef} type="button" className="btn" onClick={onClose}>
          <Icon name="close" /> Close
        </button>
      </div>
      <p id="pv-text" className="partner-view__text">
        {text || 'Nothing to show yet.'}
      </p>
    </div>
  )
}
