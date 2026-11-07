import { useMemo, useState } from 'react'
import { Icon } from '../components/Icon'
import { deleteSession, formatTime, formatWhen, transcriptText, useSessions, type Session } from '../lib/history'

type Filter = 'all' | 'today' | 'week' | 'older'

const FILTERS: { id: Filter; label: string }[] = [
  { id: 'all', label: 'All' },
  { id: 'today', label: 'Today' },
  { id: 'week', label: 'This week' },
  { id: 'older', label: 'Older' },
]

function bucket(at: number): Filter {
  const now = new Date()
  const startOfDay = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime()
  if (at >= startOfDay) return 'today'
  if (at >= startOfDay - 6 * 86400000) return 'week'
  return 'older'
}

function duration(s: Session): string {
  const mins = Math.max(1, Math.round((s.updatedAt - s.startedAt) / 60000))
  return `${mins} min`
}

export function History() {
  const sessions = useSessions()
  const [filter, setFilter] = useState<Filter>('all')
  const [query, setQuery] = useState('')
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [notice, setNotice] = useState('')

  const visible = useMemo(() => {
    const q = query.trim().toLowerCase()
    return sessions.filter(
      (s) =>
        (filter === 'all' || bucket(s.updatedAt) === filter) &&
        (!q || s.title.toLowerCase().includes(q) || s.turns.some((t) => t.text.toLowerCase().includes(q))),
    )
  }, [sessions, filter, query])

  const current = visible.find((s) => s.id === selectedId) ?? visible[0] ?? null

  const say = (msg: string) => {
    setNotice(msg)
    window.setTimeout(() => setNotice(''), 2000)
  }

  const copy = async (s: Session) => {
    try {
      await navigator.clipboard.writeText(transcriptText(s.turns))
      say('Transcript copied')
    } catch {
      say('Copy is blocked in this browser')
    }
  }

  const share = async (s: Session) => {
    const text = transcriptText(s.turns)
    if (navigator.share) {
      try {
        await navigator.share({ title: s.title, text })
      } catch {
        /* user cancelled */
      }
    } else {
      await copy(s)
    }
  }

  const remove = (s: Session) => {
    if (window.confirm(`Delete “${s.title}”? This can't be undone.`)) {
      deleteSession(s.id)
      setSelectedId(null)
      say('Conversation deleted')
    }
  }

  return (
    <main id="main" className="page" tabIndex={-1}>
      <div className="page-head">
        <div>
          <h1 className="page-title">Conversation history</h1>
          <p className="page-sub">Saved on this device only. Read back, copy or share anything you discussed.</p>
        </div>
        <div className="field" style={{ flex: '0 1 380px' }}>
          <label htmlFor="search" className="field__label">
            Search conversations
          </label>
          <div style={{ position: 'relative' }}>
            <span style={{ position: 'absolute', left: 14, top: 15, color: 'var(--muted)' }}>
              <Icon name="search" />
            </span>
            <input id="search" type="search" className="input" style={{ width: '100%', paddingLeft: 42, background: 'var(--surface)' }} placeholder="Words, places or people" value={query} onChange={(e) => setQuery(e.target.value)} />
          </div>
        </div>
      </div>

      <div role="group" aria-label="Filter conversations" className="row" style={{ gap: 8, marginBottom: 24 }}>
        {FILTERS.map((f) => (
          <button key={f.id} type="button" className="chip" aria-pressed={filter === f.id} onClick={() => setFilter(f.id)}>
            {f.label}
          </button>
        ))}
      </div>

      <p className="sr-only" role="status">
        {notice || `${visible.length} conversation${visible.length === 1 ? '' : 's'} shown`}
      </p>

      {sessions.length === 0 ? (
        <div className="card card--xl empty-state">
          <Icon name="hand" size={40} strokeWidth={1.4} />
          <h2 style={{ fontSize: 22, color: 'var(--ink)' }}>No conversations yet</h2>
          <p>Conversations you have on the Translate screen are saved here automatically.</p>
          <a className="btn btn--primary btn--lg" href="#/">
            Start translating
          </a>
        </div>
      ) : (
        <div className="split" style={{ gap: 24 }}>
          <ul className="session-list" aria-label="Conversations">
            {visible.length === 0 && <li className="card__hint" style={{ padding: 16 }}>No conversations match.</li>}
            {visible.map((s) => {
              const signed = s.turns.filter((t) => t.who === 'you').length
              const spoken = s.turns.filter((t) => t.who === 'partner').length
              return (
                <li key={s.id}>
                  <button type="button" className="session-btn" aria-current={current?.id === s.id ? 'true' : undefined} onClick={() => setSelectedId(s.id)}>
                    <span className="row row--between" style={{ flexWrap: 'nowrap', gap: 8, width: '100%' }}>
                      <span style={{ fontWeight: 700, fontSize: 18 }}>{s.title}</span>
                      <span style={{ fontSize: 14, color: 'var(--muted)', whiteSpace: 'nowrap' }}>{formatWhen(s.updatedAt)}</span>
                    </span>
                    <span style={{ fontSize: 15, color: 'var(--ink-2)' }}>{s.turns[s.turns.length - 1]?.text}</span>
                    <span className="row" style={{ gap: 6, marginTop: 4 }}>
                      <span className="count-tag count-tag--you">{signed} from you</span>
                      <span className="count-tag count-tag--partner">{spoken} from partner</span>
                    </span>
                  </button>
                </li>
              )
            })}
          </ul>

          {current && (
            <section className="card card--xl split__main" aria-labelledby="conv-title">
              <div className="row row--between" style={{ gap: 12 }}>
                <div>
                  <h2 id="conv-title" style={{ fontSize: 26, lineHeight: 1.2 }}>
                    {current.title}
                  </h2>
                  <p className="card__hint">
                    {formatWhen(current.startedAt)} · {duration(current)} · {current.signLanguage} → English
                  </p>
                </div>
                <div className="row" style={{ gap: 8 }}>
                  <button type="button" className="btn" onClick={() => copy(current)}>
                    <Icon name="copy" /> Copy transcript
                  </button>
                  <button type="button" className="btn" onClick={() => share(current)}>
                    <Icon name="share" /> Share
                  </button>
                  <button type="button" className="btn btn--icon" aria-label={`Delete ${current.title}`} onClick={() => remove(current)}>
                    <Icon name="trash" />
                  </button>
                </div>
              </div>
              <ol className="turns" style={{ gap: 14 }}>
                {current.turns.map((t) => (
                  <li key={t.id} className={`turn turn--${t.who}`}>
                    <div className="turn__meta">
                      <span>
                        {t.who === 'you' ? 'You' : 'Partner'} · {t.via}
                      </span>
                      <span className="turn__time">{formatTime(t.at)}</span>
                    </div>
                    <div className="turn__text turn__text--lg">{t.text}</div>
                  </li>
                ))}
              </ol>
              {notice && (
                <span className="pill pill--ok toast" aria-hidden="true" style={{ alignSelf: 'flex-start' }}>
                  <Icon name="check" size={14} strokeWidth={2} /> {notice}
                </span>
              )}
            </section>
          )}
        </div>
      )}
    </main>
  )
}
