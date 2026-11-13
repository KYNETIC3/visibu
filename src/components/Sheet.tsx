import { useId, type ReactNode } from 'react'
import { useModal } from '../lib/useModal'
import { Icon } from './Icon'

interface SheetProps {
  title: string
  onClose: () => void
  children: ReactNode
  footer?: ReactNode
  narrow?: boolean
  /** Hide the close button (the content has its own way out). */
  hideClose?: boolean
}

/** A modal sheet: centred on wide screens, a bottom sheet on phones. Tap outside to close. */
export function Sheet({ title, onClose, children, footer, narrow, hideClose }: SheetProps) {
  const id = useId()
  const ref = useModal(onClose)
  return (
    <div
      className="sheet-backdrop"
      onMouseDown={(e) => {
        if (e.target === e.currentTarget && !hideClose) onClose()
      }}
    >
      <div ref={ref} className={`sheet${narrow ? ' sheet--narrow' : ''}`} role="dialog" aria-modal="true" aria-labelledby={`${id}-t`}>
        <div className="sheet__head">
          <h2 id={`${id}-t`} className="sheet__title">
            {title}
          </h2>
          {!hideClose && (
            <button type="button" className="icon-btn icon-btn--plain" aria-label="Close" onClick={onClose}>
              <Icon name="close" size={18} strokeWidth={2} />
            </button>
          )}
        </div>
        <div className="sheet__body">{children}</div>
        {footer && <div className="sheet__foot">{footer}</div>}
      </div>
    </div>
  )
}

interface ConfirmProps {
  title: string
  message: string
  confirmLabel: string
  onConfirm: () => void
  onCancel: () => void
}

/** Apple-style alert for destructive actions. Cancel has focus first. */
export function ConfirmDialog({ title, message, confirmLabel, onConfirm, onCancel }: ConfirmProps) {
  const id = useId()
  const ref = useModal(onCancel)
  return (
    <div className="alert-backdrop">
      <div ref={ref} className="alert" role="alertdialog" aria-modal="true" aria-labelledby={`${id}-t`} aria-describedby={`${id}-m`}>
        <div className="alert__body">
          <h2 id={`${id}-t`} className="alert__title">
            {title}
          </h2>
          <p id={`${id}-m`} className="alert__msg">
            {message}
          </p>
        </div>
        <div className="alert__actions">
          <button type="button" data-autofocus onClick={onCancel}>
            Cancel
          </button>
          <button type="button" className="alert__danger" onClick={onConfirm}>
            {confirmLabel}
          </button>
        </div>
      </div>
    </div>
  )
}

