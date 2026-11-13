import { useEffect, useLayoutEffect, useRef, useState } from 'react'
import { useModal } from '../lib/useModal'
import { Icon } from './Icon'

/** Largest size first; the text steps down until it fits the space, so nothing ever scrolls. */
const SIZES = [128, 112, 96, 84, 72, 62, 54, 46, 40, 34, 30, 26, 22]

interface Props {
  text: string
  onClose: () => void
  /** Switch to "Partner speaking" so the partner can answer. */
  onReply?: () => void
}

/** Full-screen, very large text to turn toward a hearing partner. Fits one screen, buttons always visible. */
export function PartnerView({ text, onClose, onReply }: Props) {
  const ref = useModal(onClose)
  const stageRef = useRef<HTMLDivElement>(null)
  const textRef = useRef<HTMLParagraphElement>(null)
  const [box, setBox] = useState({ w: 0, h: 0 })

  // Re-fit when the window or the text changes.
  useEffect(() => {
    const stage = stageRef.current
    if (!stage) return
    const ro = new ResizeObserver(() => setBox({ w: stage.clientWidth, h: stage.clientHeight }))
    ro.observe(stage)
    return () => ro.disconnect()
  }, [])

  useLayoutEffect(() => {
    const stage = stageRef.current
    const p = textRef.current
    if (!stage || !p) return
    // Sized directly on the element: measuring needs the DOM, and no re-render is needed.
    // SIZES is largest first and fitting is monotonic, so binary-search the first size that fits.
    const fits = (i: number) => {
      p.style.fontSize = `${SIZES[i]}px`
      return p.scrollHeight <= stage.clientHeight && p.scrollWidth <= stage.clientWidth
    }
    let lo = 0
    let hi = SIZES.length - 1
    while (lo < hi) {
      const mid = (lo + hi) >> 1
      if (fits(mid)) hi = mid
      else lo = mid + 1
    }
    p.style.fontSize = `${SIZES[lo]}px`
  }, [text, box])

  return (
    <div ref={ref} className="partner-view" role="dialog" aria-modal="true" aria-labelledby="pv-label" aria-describedby="pv-text">
      <div className="row row--between" style={{ flexWrap: 'nowrap' }}>
        <span id="pv-label" className="partner-view__label">
          <Icon name="hand" size={22} strokeWidth={1.7} />
          From the person signing
        </span>
        <button type="button" className="icon-btn glass" aria-label="Close" onClick={onClose}>
          <Icon name="close" size={20} strokeWidth={2} />
        </button>
      </div>
      <div ref={stageRef} className="partner-view__stage">
        <p id="pv-text" ref={textRef} className="partner-view__text">
          {text || 'Nothing to show yet.'}
        </p>
      </div>
      <div className="partner-view__actions">
        <button type="button" className="btn" data-autofocus onClick={onClose}>
          Done
        </button>
        {onReply && (
          <button type="button" className="btn btn--primary" onClick={onReply}>
            <Icon name="mic" size={20} /> Reply by speaking
          </button>
        )}
      </div>
    </div>
  )
}
