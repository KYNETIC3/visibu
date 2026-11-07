import { useCallback, useEffect, useState } from 'react'
import { readArray, writeJSON } from './storage'

export type Speaker = 'you' | 'partner'
export type Via = 'signed' | 'spoke' | 'typed'

export interface Turn {
  id: string
  who: Speaker
  via: Via
  text: string
  at: number
}

export interface Session {
  id: string
  title: string
  startedAt: number
  updatedAt: number
  signLanguage: string
  turns: Turn[]
}

const KEY = 'visibu.sessions.v1'
const EVENT = 'visibu:sessions'

export const uid = () =>
  typeof crypto !== 'undefined' && 'randomUUID' in crypto
    ? crypto.randomUUID()
    : `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`

export function loadSessions(): Session[] {
  return readArray<Session>(KEY).sort((a, b) => b.updatedAt - a.updatedAt)
}

function save(sessions: Session[]) {
  writeJSON(KEY, sessions)
  window.dispatchEvent(new Event(EVENT))
}

export function upsertSession(session: Session) {
  const rest = loadSessions().filter((s) => s.id !== session.id)
  save([session, ...rest])
}

/** Sentences an earlier demo recogniser saved as if they were signed. */
const LEGACY_DEMO_PHRASES = new Set([
  'Hello, I would like to book an appointment for Thursday.',
  'Could you please write that down for me?',
  'Thank you, that is very helpful.',
])

/** One-time cleanup: drop saved demo sentences, and sessions left empty. */
export function purgeDemoTurns() {
  const sessions = readArray<Session>(KEY)
  let changed = false
  const cleaned = sessions
    .map((s) => {
      const turns = s.turns.filter((t) => !(t.via === 'signed' && LEGACY_DEMO_PHRASES.has(t.text)))
      if (turns.length === s.turns.length) return s
      changed = true
      return { ...s, turns, title: titleFor(turns) }
    })
    .filter((s) => s.turns.length > 0)
  if (changed || cleaned.length !== sessions.length) writeJSON(KEY, cleaned)
}

export function deleteSession(id: string) {
  save(loadSessions().filter((s) => s.id !== id))
}

/** Live list of saved sessions, updated whenever any screen saves. */
export function useSessions(): Session[] {
  const [sessions, setSessions] = useState<Session[]>(loadSessions)
  useEffect(() => {
    const sync = () => setSessions(loadSessions())
    window.addEventListener(EVENT, sync)
    window.addEventListener('storage', sync)
    return () => {
      window.removeEventListener(EVENT, sync)
      window.removeEventListener('storage', sync)
    }
  }, [])
  return sessions
}

/** Title from the first message, so sessions are easy to find again. */
export function titleFor(turns: Turn[]): string {
  const first = turns[0]?.text.trim()
  if (!first) return 'New conversation'
  return first.length > 42 ? `${first.slice(0, 40).trimEnd()}…` : first
}

/** The conversation in progress on the Translate screen. */
export function useConversation(saveHistory: boolean, signLanguage: string) {
  const [sessionId] = useState(uid)
  const [startedAt] = useState(Date.now)
  const [turns, setTurns] = useState<Turn[]>([])

  const addTurn = useCallback((who: Speaker, via: Via, text: string) => {
    const clean = text.trim()
    if (!clean) return
    setTurns((t) => [...t, { id: uid(), who, via, text: clean, at: Date.now() }])
  }, [])

  useEffect(() => {
    if (!saveHistory || turns.length === 0) return
    upsertSession({
      id: sessionId,
      title: titleFor(turns),
      startedAt,
      updatedAt: Date.now(),
      signLanguage,
      turns,
    })
  }, [turns, saveHistory, sessionId, startedAt, signLanguage])

  return { turns, addTurn }
}

export function formatTime(at: number): string {
  return new Date(at).toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' })
}

export function formatWhen(at: number): string {
  const d = new Date(at)
  const today = new Date()
  const yesterday = new Date()
  yesterday.setDate(today.getDate() - 1)
  const same = (a: Date, b: Date) => a.toDateString() === b.toDateString()
  if (same(d, today)) return `Today, ${formatTime(at)}`
  if (same(d, yesterday)) return `Yesterday, ${formatTime(at)}`
  return d.toLocaleDateString([], { day: 'numeric', month: 'short' }) + `, ${formatTime(at)}`
}

export function transcriptText(turns: Turn[]): string {
  return turns
    .map((t) => `[${formatTime(t.at)}] ${t.who === 'you' ? 'You' : 'Partner'} (${t.via}): ${t.text}`)
    .join('\n')
}
