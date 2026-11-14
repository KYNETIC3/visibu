import { useId } from 'react'

interface Props {
  title: string
  description?: string
  checked: boolean
  onChange: (next: boolean) => void
}

/** A labelled on/off settings cell. Uses role="switch" so screen readers announce on/off. */
export function SwitchRow({ title, description, checked, onChange }: Props) {
  const id = useId()
  return (
    <div className="cell">
      <span className="cell__main">
        <span id={`${id}-t`}>{title}</span>
        {description && (
          <span id={`${id}-d`} className="cell__sub">
            {description}
          </span>
        )}
      </span>
      <button
        type="button"
        role="switch"
        className="switch"
        aria-checked={checked}
        aria-labelledby={`${id}-t`}
        aria-describedby={description ? `${id}-d` : undefined}
        onClick={() => onChange(!checked)}
      >
        <span className="switch__knob" />
      </button>
    </div>
  )
}
