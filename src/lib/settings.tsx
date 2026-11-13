import { createContext, useContext, useEffect, useLayoutEffect, useMemo, useState, type ReactNode } from 'react'
import { readJSON, writeJSON } from './storage'

export type SignLanguage = 'ASL' | 'BSL' | 'Auslan' | 'NSL'
export type Theme = 'system' | 'light' | 'dark'

export const THEMES: { id: Theme; label: string }[] = [
  { id: 'system', label: 'Match device' },
  { id: 'light', label: 'Light' },
  { id: 'dark', label: 'Dark' },
]

export interface Settings {
  theme: Theme
  captionSize: number
  showSpeaker: boolean
  showConfidence: boolean
  saveHistory: boolean
  flashOnSpeech: boolean
  glowWhenHandsLeave: boolean
  vibrate: boolean
  highContrast: boolean
  reduceMotion: boolean
  mirrorCamera: boolean
  /** Show up to three translations to choose from when reviewing a signed message. */
  signSuggestions: boolean
  /** Choose a suggestion by holding up 1, 2 or 3 fingers. */
  pickBySign: boolean
  /** Draw the tracked hand points on the camera, coloured by finger. */
  showHandPoints: boolean
  signLanguage: SignLanguage
  /** BCP-47 language for speech captions, e.g. en-US */
  spokenLanguage: string
  setupDone: boolean
}

export const DEFAULT_SETTINGS: Settings = {
  theme: 'system',
  captionSize: 42,
  showSpeaker: true,
  showConfidence: true,
  saveHistory: true,
  flashOnSpeech: true,
  glowWhenHandsLeave: true,
  vibrate: true,
  highContrast: false,
  reduceMotion: false,
  mirrorCamera: true,
  signSuggestions: true,
  pickBySign: true,
  showHandPoints: true,
  signLanguage: 'ASL',
  spokenLanguage: 'en-US',
  setupDone: false,
}

export const SIGN_LANGUAGES: { id: SignLanguage; name: string }[] = [
  { id: 'ASL', name: 'American Sign Language (ASL)' },
  { id: 'BSL', name: 'British Sign Language (BSL)' },
  { id: 'Auslan', name: 'Auslan' },
  { id: 'NSL', name: 'Nigerian Sign Language (NSL)' },
]

const KEY = 'visibu.settings.v1'

interface SettingsContextValue {
  settings: Settings
  update: (patch: Partial<Settings>) => void
}

const SettingsContext = createContext<SettingsContextValue | null>(null)

export function SettingsProvider({ children }: { children: ReactNode }) {
  const [settings, setSettings] = useState<Settings>(() => readJSON(KEY, DEFAULT_SETTINGS))

  // Layout effect so the theme and contrast apply before the first paint (no flash).
  useLayoutEffect(() => {
    writeJSON(KEY, settings)
    const root = document.documentElement
    root.style.setProperty('--caption-size', `${settings.captionSize}px`)
    root.dataset.theme = settings.theme
    root.dataset.contrast = settings.highContrast ? 'high' : 'normal'
    root.dataset.motion = settings.reduceMotion ? 'reduce' : 'normal'
  }, [settings])

  const value = useMemo<SettingsContextValue>(
    () => ({ settings, update: (patch) => setSettings((s) => ({ ...s, ...patch })) }),
    [settings],
  )

  return <SettingsContext.Provider value={value}>{children}</SettingsContext.Provider>
}

export function useSettings(): SettingsContextValue {
  const ctx = useContext(SettingsContext)
  if (!ctx) throw new Error('useSettings must be used inside <SettingsProvider>')
  return ctx
}

/** True when the user asked for less motion, in Visibu or in their OS. */
export function usePrefersLessMotion(): boolean {
  const { settings } = useSettings()
  const [os, setOs] = useState(() => window.matchMedia?.('(prefers-reduced-motion: reduce)').matches ?? false)
  useEffect(() => {
    const mq = window.matchMedia?.('(prefers-reduced-motion: reduce)')
    if (!mq) return
    const on = () => setOs(mq.matches)
    mq.addEventListener('change', on)
    return () => mq.removeEventListener('change', on)
  }, [])
  return settings.reduceMotion || os
}
