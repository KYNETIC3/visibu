import { useId } from 'react'

interface Props {
  title: string
  description?: string
  checked: boolean
  onChange: (next: boolean) => void
}

/** A labelled on/off row. Uses role="switch" so screen readers announce on/off. */
export function SwitchRow({ title, description, checked, onChange }: Props) {
  const id = useId()
  return (
    <div className="opt">
      <div className="opt__body">
        <div id={`${id}-t`} className="opt__title">
          {title}
        </div>
        {description && (
          <div id={`${id}-d`} className="opt__desc">
            {description}
          </div>
        )}
      </div>
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
