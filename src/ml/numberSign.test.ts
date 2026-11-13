import { describe, expect, it } from 'vitest'
import { FRAME_LENGTH, LAYOUT } from '../tracking/frame'
import { frameNumber, HandStillness, NumberSignPicker, type PickNumber } from './numberSign'

type P = [number, number, number]
type Finger = 'up' | 'down'

const WRIST: P = [0.5, 0.8, 0]
const MCP: Record<'index' | 'middle' | 'ring' | 'pinky', P> = { index: [0.46, 0.7, 0], middle: [0.5, 0.69, 0], ring: [0.54, 0.7, 0], pinky: [0.58, 0.72, 0] }

function straight(mcp: P, dx: number): P[] {
  const n = Math.hypot(dx, 1)
  const [ux, uy] = [dx / n, -1 / n]
  const at = (d: number): P => [mcp[0] + ux * d, mcp[1] + uy * d, 0]
  return [at(0.04), at(0.07), at(0.095)]
}

function folded(mcp: P): P[] {
  return [
    [mcp[0], mcp[1] - 0.04, -0.01],
    [mcp[0], mcp[1] - 0.02, -0.03],
    [mcp[0], mcp[1], -0.02],
  ]
}

/** A synthetic MediaPipe hand (21 points), palm facing the camera. */
function hand(fingers: { index: Finger; middle: Finger; ring: Finger; pinky: Finger }, thumb: 'out' | 'in', spread = 0.3): P[] {
  const thumbPts: P[] =
    thumb === 'out'
      ? [[0.45, 0.77, 0], [0.42, 0.74, 0], [0.39, 0.71, 0], [0.36, 0.68, 0]]
      : [[0.45, 0.77, 0], [0.44, 0.73, -0.01], [0.48, 0.69, -0.02], [0.52, 0.67, -0.03]]
  const f = (name: keyof typeof MCP, dx: number) => [MCP[name], ...(fingers[name] === 'up' ? straight(MCP[name], dx) : folded(MCP[name]))]
  return [WRIST, ...thumbPts, ...f('index', -spread), ...f('middle', spread), ...f('ring', 0.4), ...f('pinky', 0.6)]
}

function frame(right: P[] | null, left: P[] | null = null): Float32Array {
  const out = new Float32Array(FRAME_LENGTH).fill(NaN)
  if (right) right.forEach((p, i) => out.set(p, LAYOUT.rightHand.offset + i * 3))
  if (left) left.forEach((p, i) => out.set(p, LAYOUT.leftHand.offset + i * 3))
  return out
}

const ONE = hand({ index: 'up', middle: 'down', ring: 'down', pinky: 'down' }, 'in')
const TWO = hand({ index: 'up', middle: 'up', ring: 'down', pinky: 'down' }, 'in')
const THREE = hand({ index: 'up', middle: 'up', ring: 'down', pinky: 'down' }, 'out')

describe('reading the number a hand shows', () => {
  it('reads 1, 2 and 3', () => {
    expect(frameNumber(frame(ONE), 1)).toBe(1)
    expect(frameNumber(frame(TWO), 1)).toBe(2)
    expect(frameNumber(frame(THREE), 1)).toBe(3)
  })

  it('works with either hand, and mirrored', () => {
    const mirror = (h: P[]) => h.map(([x, y, z]): P => [1 - x, y, z])
    expect(frameNumber(frame(null, TWO), 1)).toBe(2)
    expect(frameNumber(frame(mirror(THREE)), 1)).toBe(3)
  })

  it('ignores hands that are not 1, 2 or 3', () => {
    const fist = hand({ index: 'down', middle: 'down', ring: 'down', pinky: 'down' }, 'in')
    const open = hand({ index: 'up', middle: 'up', ring: 'up', pinky: 'up' }, 'out')
    const together = hand({ index: 'up', middle: 'up', ring: 'down', pinky: 'down' }, 'in', 0)
    const six = hand({ index: 'up', middle: 'up', ring: 'up', pinky: 'down' }, 'in')
    for (const h of [fist, open, together, six]) expect(frameNumber(frame(h), 1)).toBeNull()
    expect(frameNumber(frame(null), 1)).toBeNull()
  })

  it('measures the hand undistorted in a widescreen camera', () => {
    // A real hand in a 16:9 image is squeezed sideways in normalised x and z.
    const a = 16 / 9
    const squeeze = (h: P[]) => h.map(([x, y, z]): P => [0.5 + (x - 0.5) / a, y, z / a])
    expect(frameNumber(frame(squeeze(ONE)), a)).toBe(1)
    expect(frameNumber(frame(squeeze(TWO)), a)).toBe(2)
    expect(frameNumber(frame(squeeze(THREE)), a)).toBe(3)
  })

  it('needs both hands to agree when both are up', () => {
    expect(frameNumber(frame(TWO, TWO), 1)).toBe(2)
    expect(frameNumber(frame(ONE, TWO), 1)).toBeNull()
  })
})

describe('choosing by holding a number', () => {
  const run = (p: NumberSignPicker, seq: (PickNumber | null)[], from = 0, step = 66) =>
    seq.map((n, i) => p.push(n, from + i * step))

  it('selects after the number is held, once', () => {
    const p = new NumberSignPicker(3)
    const out = run(p, Array(20).fill(2))
    const picks = out.filter((u) => u.picked)
    expect(picks).toHaveLength(1)
    expect(picks[0].picked).toBe(2)
    expect(out.findIndex((u) => u.picked)).toBe(13) // 13 frames * 66 ms >= 800 ms
    expect(out[5].progress).toBeGreaterThan(0.3)
  })

  it('does not select a number flashed briefly', () => {
    const p = new NumberSignPicker(3)
    expect(run(p, [2, 2, 2, null, null, null, null, null, 3, 3, 3]).some((u) => u.picked)).toBe(false)
  })

  it('tolerates a one-frame tracking glitch', () => {
    const p = new NumberSignPicker(3)
    const seq: (PickNumber | null)[] = [1, 1, 1, 1, 1, 1, null, 1, 1, 1, 1, 3, 1, 1, 1, 1]
    expect(run(p, seq).find((u) => u.picked)?.picked).toBe(1)
  })

  it('selects again only after the hand changes', () => {
    const p = new NumberSignPicker(3)
    run(p, Array(15).fill(1))
    expect(run(p, Array(15).fill(1), 1000).some((u) => u.picked)).toBe(false)
    run(p, Array(6).fill(null), 2000)
    expect(run(p, Array(15).fill(1), 3000).some((u) => u.picked)).toBe(true)
  })

  it('ignores numbers above the number of suggestions', () => {
    const p = new NumberSignPicker(2)
    expect(run(p, Array(20).fill(3)).some((u) => u.picked)).toBe(false)
  })
})

describe('telling a held number from signing', () => {
  const shift = (h: P[], dx: number): P[] => h.map(([x, y, z]): P => [x + dx, y, z])

  it('counts a hand held still', () => {
    const s = new HandStillness()
    const steady = Array.from({ length: 10 }, (_, i) => s.update(frame(TWO), i * 66, 1))
    expect(steady.at(-1)).toBe(true)
    expect(steady[0]).toBe(false) // needs a moment of history first
  })

  it('does not count a moving hand, even in a number shape', () => {
    const s = new HandStillness()
    // Wrist moves about half a palm length every frame, as in signing.
    const moving = Array.from({ length: 10 }, (_, i) => s.update(frame(shift(TWO, i * 0.05)), i * 66, 1))
    expect(moving.some(Boolean)).toBe(false)
  })

  it('forgets when the hand leaves', () => {
    const s = new HandStillness()
    for (let i = 0; i < 10; i++) s.update(frame(TWO), i * 66, 1)
    expect(s.update(frame(null), 700, 1)).toBe(false)
    expect(s.update(frame(TWO), 766, 1)).toBe(false)
  })
})
