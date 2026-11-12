import { useEffect, useRef } from 'react'
import { frameBus } from '../lib/signRecognizer'
import { readPart, type Point } from '../tracking/frame'

/**
 * Draws the tracked hand points over the camera picture, each finger in its
 * own colour, so the signer can see exactly
 * which points the models read. The models get all 21 points of each hand
 * whether or not this is shown; it is for the person, not the model.
 */

const FINGERS = [
  { name: 'Thumb', color: '#F2B33D', points: [1, 2, 3, 4] },
  { name: 'Index', color: '#4FA3FF', points: [5, 6, 7, 8] },
  { name: 'Middle', color: '#4CC38A', points: [9, 10, 11, 12] },
  { name: 'Ring', color: '#F06BBE', points: [13, 14, 15, 16] },
  { name: 'Pinky', color: '#B49BFF', points: [17, 18, 19, 20] },
] as const

const PALM = [0, 5, 9, 13, 17, 0]
/** Clear the drawing when no frame has arrived for this long (hands gone or tracking paused). */
const STALE_MS = 400

interface Props {
  mirror: boolean
}

export function HandOverlay({ mirror }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const mirrorRef = useRef(mirror)
  useEffect(() => {
    mirrorRef.current = mirror
  }, [mirror])

  useEffect(() => {
    const canvas = canvasRef.current
    const ctx = canvas?.getContext('2d')
    if (!canvas || !ctx) return
    const video = canvas.parentElement?.querySelector('video') ?? null
    let latest: Float32Array | null = null
    let latestAt = 0
    let raf = 0
    let drawn = false

    const unsubscribe = frameBus.subscribe((frame) => {
      latest = frame
      latestAt = performance.now()
    })

    const draw = () => {
      raf = requestAnimationFrame(draw)
      const dpr = window.devicePixelRatio || 1
      const W = canvas.clientWidth
      const H = canvas.clientHeight
      if (canvas.width !== Math.round(W * dpr) || canvas.height !== Math.round(H * dpr)) {
        canvas.width = Math.round(W * dpr)
        canvas.height = Math.round(H * dpr)
      }
      const stale = !latest || performance.now() - latestAt > STALE_MS
      if (stale) {
        if (drawn) ctx.clearRect(0, 0, canvas.width, canvas.height)
        drawn = false
        return
      }
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
      ctx.clearRect(0, 0, W, H)
      // The video uses object-fit: cover, so map normalised points through the same crop.
      const vw = video?.videoWidth || W
      const vh = video?.videoHeight || H
      const scale = Math.max(W / vw, H / vh)
      const dw = vw * scale
      const dh = vh * scale
      const ox = (W - dw) / 2
      const oy = (H - dh) / 2
      const at = (p: Point): [number, number] => {
        const x = ox + p.x * dw
        return [mirrorRef.current ? W - x : x, oy + p.y * dh]
      }
      for (const part of ['leftHand', 'rightHand'] as const) {
        const hand = readPart(latest!, part)
        if (hand) drawHand(ctx, hand.map(at))
      }
      drawn = true
    }
    raf = requestAnimationFrame(draw)
    return () => {
      cancelAnimationFrame(raf)
      unsubscribe()
    }
  }, [])

  return <canvas ref={canvasRef} className="camera__hands" aria-hidden="true" />
}

function drawHand(ctx: CanvasRenderingContext2D, pts: [number, number][]) {
  const size = Math.hypot(pts[9][0] - pts[0][0], pts[9][1] - pts[0][1])
  // Thin lines and small joints, so the hand itself stays visible.
  const r = Math.max(1.5, Math.min(3.5, size * 0.03))
  ctx.lineCap = 'round'
  ctx.lineJoin = 'round'

  // Palm outline.
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.6)'
  ctx.lineWidth = r * 0.6
  ctx.beginPath()
  PALM.forEach((i, k) => (k ? ctx.lineTo(...pts[i]) : ctx.moveTo(...pts[i])))
  ctx.stroke()

  for (const f of FINGERS) {
    const chain: number[] = f.name === 'Thumb' ? [0, ...f.points] : [...f.points]
    // Faint dark halo, then the finger colour, so lines read on any background.
    for (const [color, width] of [['rgba(0, 0, 0, 0.4)', r * 1.3], [f.color, r * 0.7]] as const) {
      ctx.strokeStyle = color
      ctx.lineWidth = width
      ctx.beginPath()
      chain.forEach((i, k) => (k ? ctx.lineTo(...pts[i]) : ctx.moveTo(...pts[i])))
      ctx.stroke()
    }
    for (const i of f.points.slice(0, 3)) dot(ctx, pts[i], r, f.color)
    dot(ctx, pts[f.points[3]], r * 1.6, f.color) // fingertip, slightly larger
  }
  dot(ctx, pts[0], r * 1.3, '#ffffff')
}

function dot(ctx: CanvasRenderingContext2D, [x, y]: [number, number], r: number, color: string) {
  ctx.beginPath()
  ctx.arc(x, y, r + 1, 0, Math.PI * 2)
  ctx.fillStyle = 'rgba(0, 0, 0, 0.4)'
  ctx.fill()
  ctx.beginPath()
  ctx.arc(x, y, r, 0, Math.PI * 2)
  ctx.fillStyle = color
  ctx.fill()
}

/** The colour key, shown under the camera (readable text, not drawn on the video). */
export function FingerKey() {
  return (
    <ul className="finger-key" aria-label="Finger colours on the camera">
      {FINGERS.map((f) => (
        <li key={f.name}>
          <span className="finger-key__dot" style={{ background: f.color }} aria-hidden="true" />
          {f.name}
        </li>
      ))}
    </ul>
  )
}
