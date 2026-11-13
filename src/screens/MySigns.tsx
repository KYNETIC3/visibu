import { useCallback, useEffect, useRef, useState } from 'react'
import { CameraPanel } from '../components/CameraPanel'
import { FingerKey } from '../components/HandOverlay'
import { Icon } from '../components/Icon'
import { bestMatch, prototype, similarSigns, takeAgreement, unitFingerprint, usable, type MySign, type SignMatch } from '../ml/mySigns'
import { signPhrases, signWord } from '../ml/signPhrases'
import { useMySigns } from '../lib/useMySigns'
import { useSettings } from '../lib/settings'
import { useSignRecognition } from '../lib/signRecognizer'
import { useTracker } from '../lib/useTracker'
import type { useCamera } from '../lib/useCamera'

/** Takes recorded per sign. More takes match more reliably; three keeps teaching quick. */
const TAKES = 3
/** A built-in sign counts as "already known" when the takes agree on it this confidently. */
const BUILT_IN_PROB = 0.5

interface Take {
  fp: number[]
  top: { label: string; prob: number } | null
}

type Mode =
  | { kind: 'idle' }
  | { kind: 'record'; meaning: string; editId: string | null; takes: Take[] }
  | { kind: 'review'; meaning: string; editId: string | null; takes: Take[] }
  | { kind: 'try' }

interface Props {
  camera: ReturnType<typeof useCamera>
}

/**
 * Teach Visibu your own signs: type what a sign means, sign it a few times.
 * Only fingerprints (numbers describing the movement) are kept, on this device.
 */
export function MySigns({ camera }: Props) {
  const { settings, update: updateSettings } = useSettings()
  const { signs, add, update, remove } = useMySigns()
  const [mode, setMode] = useState<Mode>({ kind: 'idle' })
  const [meaning, setMeaning] = useState('')
  const [message, setMessage] = useState('')
  const [tryResult, setTryResult] = useState<{ match: SignMatch | null; closest: SignMatch | null } | null>(null)
  const [videoEl, setVideoEl] = useState<HTMLVideoElement | null>(null)
  const [busy, setBusy] = useState(false)
  const live = camera.status === 'live'
  const listening = live && (mode.kind === 'record' || mode.kind === 'try')

  const modeRef = useRef(mode)
  useEffect(() => {
    modeRef.current = mode
  }, [mode])

  const startCamera = camera.start
  useEffect(() => {
    void startCamera()
  }, [startCamera])
  const attach = camera.attach
  const attachVideo = useCallback(
    (el: HTMLVideoElement | null) => {
      attach(el)
      setVideoEl(el)
    },
    [attach],
  )

  // Captured stretches of signing come here (not translated) while recording or trying.
  const captureRef = useRef<((frames: Float32Array[]) => Promise<void>) | null>(null)
  const onCapture = useCallback((frames: Float32Array[]) => {
    void captureRef.current?.(frames)
  }, [])
  const sign = useSignRecognition({ video: videoEl, active: listening, onCapture })
  useTracker({ video: videoEl, active: listening })

  const words = sign.status.state === 'ready' ? sign.status.model.words : null
  const matching = words?.matching ?? null
  const version = words?.version ?? ''
  const ready = matching !== null

  const fingerprint = sign.fingerprint
  const handleTake = async (frames: Float32Array[]) => {
    if (!matching) return
    setBusy(true)
    try {
      const f = await fingerprint(frames)
      if (!f.embedding) throw new Error('The sign model gave no fingerprint.')
      const fp = unitFingerprint(f.embedding, matching.mean)
      const m = modeRef.current
      if (m.kind === 'record') {
        const takes = [...m.takes, { fp, top: f.signs[0] ?? null }]
        setMessage(`Take ${takes.length} of ${TAKES} recorded.`)
        setMode(takes.length >= TAKES ? { ...m, kind: 'review', takes } : { ...m, takes })
      } else if (m.kind === 'try') {
        const mine = usable(signs, version)
        const closest = similarSigns(fp, mine, -1)[0] ?? null
        const match = bestMatch(fp, mine, matching.threshold)
        setTryResult({ match, closest })
        setMessage(match ? `Matched your sign: ${match.sign.meaning}.` : 'No match with your signs.')
      }
    } catch (err) {
      setMessage(`Could not read that take: ${String(err instanceof Error ? err.message : err)} Please try again.`)
    } finally {
      setBusy(false)
    }
  }
  useEffect(() => {
    captureRef.current = handleTake
  })

  const startTeaching = (text: string, editId: string | null = null) => {
    const clean = text.trim()
    if (!clean || !ready) return
    setTryResult(null)
    setMode({ kind: 'record', meaning: clean, editId, takes: [] })
    setMessage(`Recording “${clean}”. Sign it, then lower your hands. Take 1 of ${TAKES}.`)
  }

  const cancel = () => {
    setMode({ kind: 'idle' })
    setMessage('Cancelled. Nothing was saved.')
  }

  const finishSave = (text: string) => {
    setMode({ kind: 'idle' })
    setMeaning('')
    setMessage(text)
  }

  return (
    <main id="main" className="page split" tabIndex={-1}>
      <h1 className="sr-only">My signs</h1>

      <section aria-label="Camera" className="split__side">
        <CameraPanel
          attach={attachVideo}
          status={camera.status}
          errorName={camera.errorName}
          lighting={camera.lighting}
          mirror={settings.mirrorCamera}
          paused={false}
          alert={false}
          handsVisible={listening ? sign.handsVisible : null}
          signStatus={null}
          showHandPoints={settings.showHandPoints}
          onStart={() => void camera.start()}
        />
        {settings.showHandPoints && live && <FingerKey />}
        <div className="row" style={{ gap: 8 }}>
          <button type="button" className="btn btn--sm" onClick={() => updateSettings({ mirrorCamera: !settings.mirrorCamera })} aria-pressed={settings.mirrorCamera}>
            <Icon name="mirror" size={18} /> Mirror view
          </button>
        </div>
        <p className="card__hint" style={{ padding: '0 4px' }}>
          Only numbers describing your hand and body movement are saved, in this browser. No video is recorded or sent anywhere.
        </p>
      </section>

      <section aria-label="My signs" className="split__main">
        <section className="card stack teach" aria-labelledby="teach-h">
          <div>
            <h2 id="teach-h" className="card__title">
              {mode.kind !== 'idle' && mode.kind !== 'try' && mode.editId ? 'Change a sign' : 'Teach a new sign'}
            </h2>
            <p className="card__hint">Type what it means, then sign it {TAKES} times. Lower your hands after each take.</p>
          </div>

          {!ready && <p className="teach__note">{sign.status.state === 'loading' ? 'Loading the sign model…' : 'The sign model is not available, so signs cannot be taught.'}</p>}

          {mode.kind === 'idle' || mode.kind === 'try' ? (
            <form
              className="teach__form"
              onSubmit={(e) => {
                e.preventDefault()
                startTeaching(meaning)
              }}
            >
              <label htmlFor="teach-meaning" className="field__label">
                What does the sign mean?
              </label>
              <div className="row" style={{ flexWrap: 'nowrap', gap: 8 }}>
                <input
                  id="teach-meaning"
                  className="input"
                  style={{ flex: '1 1 auto', minWidth: 0 }}
                  value={meaning}
                  onChange={(e) => setMeaning(e.target.value)}
                  placeholder="e.g. Can I get a latte, please?"
                  autoComplete="off"
                  maxLength={160}
                />
                <button type="submit" className="btn btn--primary btn--sm" disabled={!meaning.trim() || !ready || !live}>
                  <Icon name="hand" size={18} /> Record
                </button>
              </div>
              {!live && <p className="teach__note">Turn on the camera to record.</p>}
            </form>
          ) : (
            <Recording mode={mode} busy={busy} activity={sign.activity} onDone={sign.finish} onCancel={cancel} onRedo={(i) => setMode({ ...mode, kind: 'record', takes: mode.takes.filter((_, j) => j !== i) })} />
          )}

          {mode.kind === 'review' && matching && (
            <Review
              mode={mode}
              signs={usable(signs, version)}
              threshold={matching.threshold}
              onRedo={(i) => {
                setMode({ ...mode, kind: 'record', takes: mode.takes.filter((_, j) => j !== i) })
                setMessage(`Redo take ${i + 1}: sign it, then lower your hands.`)
              }}
              onSaveNew={() => {
                add(mode.meaning, mode.takes.map((t) => t.fp), version)
                finishSave(`Saved “${mode.meaning}”.`)
              }}
              onReplaceTakes={(id) => {
                update(id, { takes: mode.takes.map((t) => t.fp), modelVersion: version })
                finishSave(`Changed the sign for “${mode.meaning}”.`)
              }}
              onChangeMeaning={(target) => {
                update(target.id, { meaning: mode.meaning, takes: mode.takes.map((t) => t.fp), modelVersion: version })
                if (mode.editId && mode.editId !== target.id) remove(mode.editId)
                finishSave(`“${target.meaning}” now means “${mode.meaning}”.`)
              }}
              onCancel={cancel}
            />
          )}
        </section>

        <section className="card stack" aria-labelledby="try-h">
          <div className="row row--between" style={{ gap: 8 }}>
            <div>
              <h2 id="try-h" className="card__title">
                Try your signs
              </h2>
              <p className="card__hint">Sign one of your signs to see if Visibu recognises it.</p>
            </div>
            {mode.kind === 'try' ? (
              <button type="button" className="btn btn--sm" onClick={() => setMode({ kind: 'idle' })}>
                Stop
              </button>
            ) : (
              <button type="button" className="btn btn--sm" disabled={!ready || !live || mode.kind !== 'idle' || usable(signs, version).length === 0} onClick={() => { setTryResult(null); setMode({ kind: 'try' }) }}>
                <Icon name="play" size={18} /> Start
              </button>
            )}
          </div>
          {mode.kind === 'try' && (
            <p className="teach__live" aria-hidden="true">
              <span className={`signbar__dot signbar__dot--${busy ? 'translating' : sign.activity}`} /> {busy ? 'Reading…' : sign.activity === 'signing' ? 'Signing… lower your hands when done.' : 'Ready. Sign one of your signs.'}
            </p>
          )}
          {tryResult && (
            <p className={`teach__result${tryResult.match ? ' teach__result--ok' : ''}`}>
              {tryResult.match ? (
                <>
                  <Icon name="check" size={18} strokeWidth={2} /> “{tryResult.match.sign.meaning}” · {Math.round(tryResult.match.score * 100)}% match
                </>
              ) : (
                <>
                  <Icon name="alert" size={18} strokeWidth={2} /> No match.
                  {tryResult.closest && ` Closest: “${tryResult.closest.sign.meaning}” (${Math.round(tryResult.closest.score * 100)}%, needs ${Math.round((matching?.threshold ?? 0) * 100)}%).`}
                </>
              )}
            </p>
          )}
        </section>

        <SignList signs={signs} version={version} disabled={mode.kind === 'record' || mode.kind === 'review'} onRename={(id, m) => update(id, { meaning: m })} onRerecord={(s) => startTeaching(s.meaning, s.id)} onRemove={remove} />

        <span className="sr-only" role="status">
          {message}
        </span>
      </section>
    </main>
  )
}

function Recording({
  mode,
  busy,
  activity,
  onDone,
  onCancel,
  onRedo,
}: {
  mode: Extract<Mode, { kind: 'record' | 'review' }>
  busy: boolean
  activity: string
  onDone: () => void
  onCancel: () => void
  onRedo: (i: number) => void
}) {
  const recording = mode.kind === 'record'
  return (
    <div className="stack" style={{ gap: 10 }}>
      <p className="teach__meaning">“{mode.meaning}”</p>
      <ol className="teach__takes" aria-label="Takes">
        {Array.from({ length: TAKES }, (_, i) => {
          const done = i < mode.takes.length
          const current = recording && i === mode.takes.length
          return (
            <li key={i} className={`teach__take${done ? ' teach__take--done' : ''}${current ? ' teach__take--now' : ''}`}>
              {done ? <Icon name="check" size={16} strokeWidth={2.2} /> : <span>{i + 1}</span>}
              <span className="sr-only">{done ? `Take ${i + 1} recorded` : current ? `Take ${i + 1}, recording` : `Take ${i + 1}`}</span>
              {done && recording && (
                <button type="button" className="teach__redo" onClick={() => onRedo(i)} aria-label={`Redo take ${i + 1}`}>
                  Redo
                </button>
              )}
            </li>
          )
        })}
      </ol>
      {recording && (
        <p className="teach__live">
          <span className={`signbar__dot signbar__dot--${busy ? 'translating' : activity}`} aria-hidden="true" />
          {busy ? 'Reading the take…' : activity === 'signing' ? 'Signing… lower your hands when done.' : `Sign it now (take ${mode.takes.length + 1} of ${TAKES}).`}
        </p>
      )}
      {recording && (
        <div className="row" style={{ gap: 8 }}>
          {activity === 'signing' && (
            <button type="button" className="btn btn--primary btn--sm" onClick={onDone}>
              <Icon name="check" size={18} /> Done
            </button>
          )}
          <button type="button" className="btn btn--sm" onClick={onCancel}>
            Cancel
          </button>
        </div>
      )}
    </div>
  )
}

function Review({
  mode,
  signs,
  threshold,
  onRedo,
  onSaveNew,
  onReplaceTakes,
  onChangeMeaning,
  onCancel,
}: {
  mode: Extract<Mode, { kind: 'review' }>
  signs: MySign[]
  threshold: number
  onRedo: (i: number) => void
  onSaveNew: () => void
  onReplaceTakes: (id: string) => void
  onChangeMeaning: (target: MySign) => void
  onCancel: () => void
}) {
  const fps = mode.takes.map((t) => t.fp)
  const proto = prototype(fps)
  const agreement = takeAgreement(fps)
  const odd = agreement.findIndex((a) => a < threshold)
  const lookalike = similarSigns(proto, signs, threshold, mode.editId ?? undefined)[0] ?? null
  // A built-in sign the takes agree on, confidently.
  const tops = mode.takes.map((t) => t.top).filter((t): t is { label: string; prob: number } => t !== null)
  const label = tops[0]?.label
  const builtIn = label && tops.length === mode.takes.length && tops.every((t) => t.label === label) && tops.reduce((s, t) => s + t.prob, 0) / tops.length >= BUILT_IN_PROB ? label : null

  if (odd >= 0) {
    return (
      <div className="teach__ask" role="alert">
        <p>
          <strong>Take {odd + 1} looks different from the others.</strong> Redo it so Visibu learns one clear version of the sign.
        </p>
        <div className="row" style={{ gap: 8 }}>
          <button type="button" className="btn btn--primary btn--sm" onClick={() => onRedo(odd)}>
            Redo take {odd + 1}
          </button>
          <button type="button" className="btn btn--sm" onClick={onCancel}>
            Cancel
          </button>
        </div>
      </div>
    )
  }

  if (lookalike) {
    const pct = Math.round(lookalike.score * 100)
    const same = lookalike.sign.meaning.trim().toLowerCase() === mode.meaning.trim().toLowerCase()
    return (
      <div className="teach__ask" role="alert">
        <p>
          <strong>This sign already has a meaning.</strong> It looks like your sign “{lookalike.sign.meaning}” ({pct}% match).
          {same ? ' Update that sign with these new takes?' : ` Do you want to change it to “${mode.meaning}”?`}
        </p>
        <div className="row" style={{ gap: 8 }}>
          {same ? (
            <button type="button" className="btn btn--primary btn--sm" onClick={() => onReplaceTakes(lookalike.sign.id)}>
              Update “{lookalike.sign.meaning}”
            </button>
          ) : (
            <button type="button" className="btn btn--primary btn--sm" onClick={() => onChangeMeaning(lookalike.sign)}>
              Change to “{mode.meaning}”
            </button>
          )}
          {!same && (
            <button type="button" className="btn btn--sm" onClick={mode.editId ? () => onReplaceTakes(mode.editId!) : onSaveNew}>
              Keep both
            </button>
          )}
          <button type="button" className="btn btn--sm" onClick={onCancel}>
            Cancel
          </button>
        </div>
      </div>
    )
  }

  return (
    <div className="teach__ask" role="status">
      {builtIn ? (
        <p>
          <strong>Visibu already knows this sign as “{signWord(builtIn)}”</strong> (it suggests “{signPhrases(builtIn)[0]}”). Use “{mode.meaning}” instead when you sign it?
        </p>
      ) : (
        <p>
          <strong>All {TAKES} takes recorded.</strong> Save “{mode.meaning}”?
        </p>
      )}
      <div className="row" style={{ gap: 8 }}>
        <button type="button" className="btn btn--primary btn--sm" onClick={mode.editId ? () => onReplaceTakes(mode.editId!) : onSaveNew}>
          <Icon name="check" size={18} /> {builtIn ? 'Yes, use mine' : 'Save'}
        </button>
        <button type="button" className="btn btn--sm" onClick={onCancel}>
          Cancel
        </button>
      </div>
    </div>
  )
}

function SignList({
  signs,
  version,
  disabled,
  onRename,
  onRerecord,
  onRemove,
}: {
  signs: MySign[]
  version: string
  disabled: boolean
  onRename: (id: string, meaning: string) => void
  onRerecord: (s: MySign) => void
  onRemove: (id: string) => void
}) {
  const [editing, setEditing] = useState<{ id: string; text: string } | null>(null)
  const [confirming, setConfirming] = useState<string | null>(null)
  return (
    <section className="card stack" aria-labelledby="list-h">
      <h2 id="list-h" className="card__title">
        Your signs {signs.length > 0 && <span className="signpanel__count">{signs.length}</span>}
      </h2>
      {signs.length === 0 ? (
        <p className="signpanel__empty">No signs yet. Teach one above; it then shows up first in Translate when you sign it.</p>
      ) : (
        <ul className="mysigns">
          {signs.map((s) => {
            const outdated = version !== '' && s.modelVersion !== version
            return (
              <li key={s.id} className="mysigns__item">
                {editing?.id === s.id ? (
                  <form
                    className="mysigns__edit"
                    onSubmit={(e) => {
                      e.preventDefault()
                      if (editing.text.trim()) onRename(s.id, editing.text)
                      setEditing(null)
                    }}
                  >
                    <label htmlFor={`rename-${s.id}`} className="sr-only">
                      New meaning for {s.meaning}
                    </label>
                    <input id={`rename-${s.id}`} className="input" value={editing.text} onChange={(e) => setEditing({ id: s.id, text: e.target.value })} autoFocus maxLength={160} />
                    <button type="submit" className="btn btn--primary btn--sm" disabled={!editing.text.trim()}>
                      Save
                    </button>
                    <button type="button" className="btn btn--sm" onClick={() => setEditing(null)}>
                      Cancel
                    </button>
                  </form>
                ) : (
                  <>
                    <div className="mysigns__main">
                      <span className="mysigns__meaning">{s.meaning}</span>
                      <span className="mysigns__meta">
                        {outdated ? 'Recorded with an older model: change the sign to use it again' : `${s.takes.length} takes · ${new Date(s.updatedAt).toLocaleDateString()}`}
                      </span>
                    </div>
                    {confirming === s.id ? (
                      <span className="mysigns__actions" role="group" aria-label={`Delete ${s.meaning}?`}>
                        <span className="mysigns__meta">Delete?</span>
                        <button type="button" className="btn btn--sm" onClick={() => { onRemove(s.id); setConfirming(null) }}>
                          Delete
                        </button>
                        <button type="button" className="btn btn--sm" onClick={() => setConfirming(null)}>
                          Keep
                        </button>
                      </span>
                    ) : (
                      <span className="mysigns__actions">
                        <button type="button" className="btn btn--sm" disabled={disabled} onClick={() => setEditing({ id: s.id, text: s.meaning })} aria-label={`Rename ${s.meaning}`}>
                          Rename
                        </button>
                        <button type="button" className="btn btn--sm" disabled={disabled} onClick={() => onRerecord(s)} aria-label={`Change the sign for ${s.meaning}`}>
                          Change sign
                        </button>
                        <button type="button" className="btn btn--sm btn--icon" disabled={disabled} onClick={() => setConfirming(s.id)} aria-label={`Delete ${s.meaning}`}>
                          <Icon name="trash" size={18} />
                        </button>
                      </span>
                    )}
                  </>
                )}
              </li>
            )
          })}
        </ul>
      )}
    </section>
  )
}
