import { useCallback, useEffect, useState } from 'react'
import { compact, type MySign } from '../ml/mySigns'
import { uid } from './history'
import { readArray, writeJSON } from './storage'

// Stored only in this browser (localStorage): meanings and fingerprints, never video.
const KEY = 'visibu.mySigns.v1'
const EVENT = 'visibu:mySigns'

export function loadMySigns(): MySign[] {
  return readArray<MySign>(KEY).filter((s) => typeof s?.meaning === 'string' && Array.isArray(s.takes))
}

function save(signs: MySign[]) {
  writeJSON(KEY, signs)
  window.dispatchEvent(new Event(EVENT))
}

export function useMySigns() {
  const [signs, setSigns] = useState<MySign[]>(loadMySigns)

  useEffect(() => {
    const on = () => setSigns(loadMySigns())
    window.addEventListener(EVENT, on)
    window.addEventListener('storage', on)
    return () => {
      window.removeEventListener(EVENT, on)
      window.removeEventListener('storage', on)
    }
  }, [])

  const add = useCallback((meaning: string, takes: number[][], modelVersion: string) => {
    const now = Date.now()
    save([...loadMySigns(), { id: uid(), meaning: meaning.trim(), takes: takes.map(compact), modelVersion, createdAt: now, updatedAt: now }])
  }, [])

  const update = useCallback((id: string, change: Partial<Pick<MySign, 'meaning' | 'takes' | 'modelVersion'>>) => {
    save(
      loadMySigns().map((s) =>
        s.id === id
          ? { ...s, ...change, meaning: (change.meaning ?? s.meaning).trim(), takes: (change.takes ?? s.takes).map(compact), updatedAt: Date.now() }
          : s,
      ),
    )
  }, [])

  const remove = useCallback((id: string) => save(loadMySigns().filter((s) => s.id !== id)), [])

  return { signs, add, update, remove }
}
