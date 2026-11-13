import { useEffect, useRef, useState } from 'react'
import { frameNumber, HandStillness, NumberSignPicker, type PickNumber } from '../ml/numberSign'
import { frameBus } from './signRecognizer'

interface Options {
  /** Listen to tracked frames only while this is true. */
  active: boolean
  /** How many suggestions there are; higher numbers are ignored. */
  count: number
  /** Camera width / height, so finger shapes are measured undistorted. */
  aspect: number
  onPick: (n: PickNumber) => void
}

/** Watches tracked frames for a 1, 2 or 3 held up still. Never sends anything. */
export function useNumberPick({ active, count, aspect, onPick }: Options) {
  const [hold, setHold] = useState<{ candidate: PickNumber | null; progress: number }>({ candidate: null, progress: 0 })
  const onPickRef = useRef(onPick)
  useEffect(() => {
    onPickRef.current = onPick
  }, [onPick])

  useEffect(() => {
    if (!active) return
    const picker = new NumberSignPicker(count)
    const still = new HandStillness()
    let shown = { candidate: null as PickNumber | null, progress: 0 }
    const unsubscribe = frameBus.subscribe((frame, ts) => {
      // Only a number held still counts; a passing handshape during signing does not.
      const steady = still.update(frame, ts, aspect)
      const u = picker.push(steady ? frameNumber(frame, aspect) : null, ts)
      // Re-render only when what is shown changes (progress in 10% steps).
      const next = { candidate: u.candidate, progress: Math.round(u.progress * 10) / 10 }
      if (next.candidate !== shown.candidate || next.progress !== shown.progress) {
        shown = next
        setHold(next)
      }
      if (u.picked) onPickRef.current(u.picked)
    })
    return () => {
      unsubscribe()
      setHold({ candidate: null, progress: 0 })
    }
  }, [active, count, aspect])

  return active ? hold : { candidate: null, progress: 0 }
}
