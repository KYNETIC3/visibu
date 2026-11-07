import { useCallback, useEffect, useRef, useState } from 'react'

export type CameraStatus = 'idle' | 'requesting' | 'live' | 'denied' | 'unavailable' | 'error'
export type Lighting = 'good' | 'dim' | 'bright' | 'unknown'

/**
 * Owns the webcam stream. Video never leaves the device.
 * Also samples frame brightness once a second so we can coach on lighting.
 */
export function useCamera() {
  const videoRef = useRef<HTMLVideoElement | null>(null)
  const streamRef = useRef<MediaStream | null>(null)
  /** The getUserMedia request in flight, so repeat calls share it instead of racing. */
  const pendingRef = useRef<Promise<void> | null>(null)
  /** Bumped by stop(); a request from an older generation discards its stream. */
  const genRef = useRef(0)
  const [status, setStatus] = useState<CameraStatus>('idle')
  const [lighting, setLighting] = useState<Lighting>('unknown')
  const [errorName, setErrorName] = useState('')

  const attach = useCallback((el: HTMLVideoElement | null) => {
    videoRef.current = el
    if (el && streamRef.current && el.srcObject !== streamRef.current) {
      el.srcObject = streamRef.current
      void el.play().catch(() => {})
    }
  }, [])

  const start = useCallback((): Promise<void> => {
    if (streamRef.current) {
      setStatus('live')
      return Promise.resolve()
    }
    // StrictMode runs effects twice in dev: share the request already in flight.
    if (pendingRef.current) return pendingRef.current
    if (!navigator.mediaDevices?.getUserMedia) {
      setStatus('unavailable')
      return Promise.resolve()
    }
    const gen = genRef.current
    setStatus('requesting')
    setErrorName('')
    const request = (async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { width: { ideal: 1280 }, height: { ideal: 960 }, facingMode: 'user' },
          audio: false,
        })
        if (gen !== genRef.current) {
          // stop() ran while permission was pending: release the camera at once.
          stream.getTracks().forEach((t) => t.stop())
          return
        }
        streamRef.current = stream
        if (videoRef.current) {
          videoRef.current.srcObject = stream
          await videoRef.current.play().catch(() => {})
        }
        if (gen === genRef.current) setStatus('live')
      } catch (err) {
        if (gen !== genRef.current) return
        const name = err instanceof DOMException ? err.name : ''
        setErrorName(name)
        setStatus(name === 'NotAllowedError' || name === 'SecurityError' ? 'denied' : name === 'NotFoundError' ? 'unavailable' : 'error')
      }
    })()
    pendingRef.current = request
    // Attached after the assignment above, so it clears the slot even if the
    // request settled synchronously. Only clears if it is still this request.
    void request.finally(() => {
      if (pendingRef.current === request) pendingRef.current = null
    })
    return request
  }, [])

  const stop = useCallback(() => {
    genRef.current++
    pendingRef.current = null
    streamRef.current?.getTracks().forEach((t) => t.stop())
    streamRef.current = null
    if (videoRef.current) videoRef.current.srcObject = null
    setStatus('idle')
    setLighting('unknown')
  }, [])

  // Brightness sampling on a tiny offscreen canvas.
  useEffect(() => {
    if (status !== 'live') return
    const canvas = document.createElement('canvas')
    canvas.width = 32
    canvas.height = 24
    const ctx = canvas.getContext('2d', { willReadFrequently: true })
    const id = window.setInterval(() => {
      const v = videoRef.current
      if (!ctx || !v || v.readyState < 2) return
      ctx.drawImage(v, 0, 0, canvas.width, canvas.height)
      const { data } = ctx.getImageData(0, 0, canvas.width, canvas.height)
      let sum = 0
      for (let i = 0; i < data.length; i += 4) sum += 0.2126 * data[i] + 0.7152 * data[i + 1] + 0.0722 * data[i + 2]
      const avg = sum / (data.length / 4)
      setLighting(avg < 60 ? 'dim' : avg > 215 ? 'bright' : 'good')
    }, 1000)
    return () => window.clearInterval(id)
  }, [status])

  // Release the camera on unmount. This deliberately does not cancel a pending
  // request: StrictMode's simulated unmount would discard it, and the remount
  // would then race a second getUserMedia against the device it still holds.
  useEffect(() => () => streamRef.current?.getTracks().forEach((t) => t.stop()), [])

  return { videoRef, attach, status, errorName, lighting, start, stop }
}
