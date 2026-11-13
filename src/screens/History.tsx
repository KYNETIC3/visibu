import { useMemo, useState } from 'react'
import { Icon } from '../components/Icon'
import { ConfirmDialog } from '../components/Sheet'
import {
  clearSessions,
  dayGroup,
  deleteSession,
  formatShort,
  formatTime,
  formatWhen,
  transcriptText,
  useSessions,
  type DayGroup,
  type Session,
} from '../lib/history'

const GROUPS: DayGroup[] = ['Today', 'Yesterday', 'Earlier']

type Confirm = { kind: 'one'; session: Session } | { kind: 'all' } | null

/** Conversations listed like Messages: one row each, grouped by day, open one to read it. */
export function History() {
  const sessions = useSessions()
  const [openId, setOpenId] = useState<string | null>(null)
  const [editing, setEditing] = useState(false)
  const [query, setQuery] = useState('')
  const [confirm, setConfirm] = useState<Confirm>(null)
  const [notice, setNotice] = useState('')

  const say = (msg: string) => {
    setNotice(msg)
    window.setTimeout(() => setNotice(''), 2500)
  }

  // Search only shows with many conversations; once hidden it must not keep filtering.
  const searchable = sessions.length > 5
  const visible = useMemo(() => {
    const q = searchable ? query.trim().toLowerCase() : ''
    if (!q) return sessions
    return sessions.filter((s) => s.title.toLowerCase().includes(q) || s.turns.some((t) => t.text.toLowerCase().includes(q)))
  }, [sessions, query, searchable])

  const open = sessions.find((s) => s.id === openId) ?? null

  const doConfirm = () => {
    if (!confirm) return
    if (confirm.kind === 'one') {
      deleteSession(confirm.session.id)
      if (openId === confirm.session.id) setOpenId(null)
      if (sessions.length <= 1) setEditing(false)
      say('Conversation deleted')
    } else {
      clearSessions()
      setOpenId(null)
      setEditing(false)
      say('All conversations deleted')
    }
    setConfirm(null)
  }

  const copy = async (s: Session) => {
    try {
      await navigator.clipboard.writeText(transcriptText(s.turns))
      say('Copied')
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

  return (
    <main id="main" className="narrow" tabIndex={-1}>
      {open ? (
        <>
          <div className="row row--between" style={{ flexWrap: 'nowrap' }}>
            <button type="button" className="back-btn" onClick={() => setOpenId(null)}>
              <Icon name="chevronLeft" size={22} strokeWidth={2} /> History
            </button>
            <div className="row" style={{ gap: 8, flexWrap: 'nowrap' }}>
              <button type="button" className="icon-btn icon-btn--plain" aria-label="Copy transcript" onClick={() => copy(open)}>
                <Icon name="copy" size={20} />
              </button>
              <button type="button" className="icon-btn icon-btn--plain" aria-label="Share transcript" onClick={() => share(open)}>
                <Icon name="share" size={20} />
              </button>
              <button type="button" className="icon-btn icon-btn--plain btn--danger" aria-label="Delete this conversation" onClick={() => setConfirm({ kind: 'one', session: open })}>
                <Icon name="trash" size={20} />
              </button>
            </div>
          </div>
          <div>
            <h1 className="large-title" style={{ fontSize: 28 }}>
              {formatWhen(open.startedAt)}
            </h1>
            <p className="card__hint">
              {open.turns.length} message{open.turns.length === 1 ? '' : 's'} · {open.signLanguage} and English
            </p>
          </div>
          <section className="card card--xl">
            <ol className="bubbles" aria-label="Messages">
              {open.turns.map((t) => (
                <li key={t.id} className={`bubble bubble--${t.who}`}>
                  <span className="bubble__meta">
                    {t.who === 'you' ? 'You' : 'Partner'} · {formatTime(t.at)}
                  </span>
                  <span className="bubble__text">{t.text}</span>
                </li>
              ))}
            </ol>
          </section>
        </>
      ) : (
        <>
          <div className="narrow__head">
            <h1 className="large-title">History</h1>
            {sessions.length > 0 &&
              (editing ? (
                <div className="row" style={{ gap: 4 }}>
                  <button type="button" className="btn btn--text btn--danger" onClick={() => setConfirm({ kind: 'all' })}>
                    Clear All
                  </button>
                  <button type="button" className="btn btn--text" style={{ fontWeight: 700 }} onClick={() => setEditing(false)}>
                    Done
                  </button>
                </div>
              ) : (
                <button type="button" className="btn btn--text" onClick={() => setEditing(true)}>
                  Edit
                </button>
              ))}
          </div>

          {sessions.length === 0 ? (
            <section className="card card--xl empty-state">
              <span className="welcome-list__icon" style={{ width: 64, height: 64, borderRadius: 20 }}>
                <Icon name="clock" size={30} strokeWidth={1.6} />
              </span>
              <h2 style={{ fontSize: 22, color: 'var(--ink)' }}>No conversations</h2>
              <p>Conversations are saved here, on this device only.</p>
            </section>
          ) : (
            <>
              {searchable && (
                <div className="search">
                  <label htmlFor="hist-search" className="sr-only">
                    Search conversations
                  </label>
                  <span className="search__icon">
                    <Icon name="search" />
                  </span>
                  <input id="hist-search" type="search" className="input" placeholder="Search" value={query} onChange={(e) => setQuery(e.target.value)} />
                </div>
              )}

              {visible.length === 0 && <p className="card__hint">No conversations match “{query}”.</p>}

              {GROUPS.map((g) => {
                const rows = visible.filter((s) => dayGroup(s.updatedAt) === g)
                if (rows.length === 0) return null
                return (
                  <section key={g} className="group" aria-labelledby={`g-${g}`}>
                    <h2 id={`g-${g}`} className="group__label">
                      {g}
                    </h2>
                    <ul className="inset">
                      {rows.map((s) => (
                        <li key={s.id} className={editing ? 'row-with-delete' : undefined}>
                          {editing && (
                            <button type="button" className="minus-btn" aria-label={`Delete conversation: ${s.title}`} onClick={() => setConfirm({ kind: 'one', session: s })}>
                              <span className="minus-btn__dot" aria-hidden="true" />
                            </button>
                          )}
                          <button type="button" className="cell" style={{ minHeight: 68 }} onClick={() => setOpenId(s.id)}>
                            <span className="cell__main">
                              <span className="cell__title">{s.title}</span>
                              <span className="cell__sub">
                                {s.turns.length} message{s.turns.length === 1 ? '' : 's'}
                              </span>
                            </span>
                            <span className="cell__aside">{formatShort(s.updatedAt)}</span>
                            <span className="cell__chev">
                              <Icon name="chevronRight" size={16} strokeWidth={2.2} />
                            </span>
                          </button>
                        </li>
                      ))}
                    </ul>
                  </section>
                )
              })}
            </>
          )}
        </>
      )}

      <p className="sr-only" role="status">
        {notice}
      </p>

      {confirm && (
        <ConfirmDialog
          title={confirm.kind === 'all' ? 'Clear all history?' : 'Delete this conversation?'}
          message={
            confirm.kind === 'all'
              ? `All ${sessions.length} conversation${sessions.length === 1 ? '' : 's'} will be removed from this device. You can’t undo this.`
              : `“${confirm.session.title}” will be removed from this device. You can’t undo this.`
          }
          confirmLabel={confirm.kind === 'all' ? 'Clear All' : 'Delete'}
          onConfirm={doConfirm}
          onCancel={() => setConfirm(null)}
        />
      )}
    </main>
  )
}
