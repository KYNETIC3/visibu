import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { clearSessions, dayGroup, formatShort, formatTime, loadSessions, upsertSession, type Session } from './history'

const at = (y: number, m: number, d: number, h = 12, min = 0) => new Date(y, m, d, h, min).getTime()

describe('dayGroup', () => {
  const now = new Date(2026, 10, 13, 9, 30) // Fri 13 Nov 2026, 09:30

  it('puts anything since midnight in Today', () => {
    expect(dayGroup(at(2026, 10, 13, 0, 0), now)).toBe('Today')
    expect(dayGroup(at(2026, 10, 13, 9, 29), now)).toBe('Today')
  })

  it('splits Yesterday and Earlier at midnight', () => {
    expect(dayGroup(at(2026, 10, 12, 23, 59), now)).toBe('Yesterday')
    expect(dayGroup(at(2026, 10, 12, 0, 0), now)).toBe('Yesterday')
    expect(dayGroup(at(2026, 10, 11, 23, 59), now)).toBe('Earlier')
  })

  it('uses calendar days, not 24-hour blocks', () => {
    // Whatever the local time zone does with daylight saving, the first minute
    // of the previous calendar day is Yesterday and the last minute before it is Earlier.
    for (const [y, m, d] of [
      [2026, 2, 9], // day after a March clock change in many zones
      [2026, 9, 26], // day after an October clock change in many zones
      [2026, 10, 2],
    ]) {
      const today = new Date(y, m, d, 0, 30)
      expect(dayGroup(new Date(y, m, d - 1, 0, 0).getTime(), today)).toBe('Yesterday')
      expect(dayGroup(new Date(y, m, d - 1, 0, 0).getTime() - 60_000, today)).toBe('Earlier')
    }
  })
})

describe('formatShort', () => {
  const now = new Date(2026, 10, 13, 9, 30)

  it('shows the time for today and yesterday', () => {
    const t = at(2026, 10, 13, 8, 5)
    expect(formatShort(t, now)).toBe(formatTime(t))
    const y = at(2026, 10, 12, 16, 12)
    expect(formatShort(y, now)).toBe(formatTime(y))
  })

  it('shows a date for older conversations', () => {
    const old = at(2026, 10, 9, 8, 30)
    expect(formatShort(old, now)).not.toBe(formatTime(old))
    expect(formatShort(old, now)).toMatch(/9/)
  })
})

describe('clearSessions', () => {
  beforeEach(() => {
    const store = new Map<string, string>()
    vi.stubGlobal('window', {
      localStorage: {
        getItem: (k: string) => store.get(k) ?? null,
        setItem: (k: string, v: string) => void store.set(k, v),
        removeItem: (k: string) => void store.delete(k),
      },
      dispatchEvent: () => true,
    })
  })

  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('removes every saved conversation', () => {
    const session = (id: string): Session => ({
      id,
      title: id,
      startedAt: 1,
      updatedAt: 2,
      signLanguage: 'ASL',
      turns: [{ id: `${id}-t`, who: 'you', via: 'typed', text: 'Hi', at: 1 }],
    })
    upsertSession(session('a'))
    upsertSession(session('b'))
    expect(loadSessions()).toHaveLength(2)
    clearSessions()
    expect(loadSessions()).toHaveLength(0)
  })
})
