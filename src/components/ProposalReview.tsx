import { useEffect, useId, useRef, useState, type CSSProperties } from 'react'
import type { Activity, SignProposal } from '../lib/signRecognizer'
import { useNumberPick } from '../lib/useNumberPick'
import { mergeTexts } from '../ml/suggest'
import { Icon } from './Icon'

interface Props {
  /** Translations waiting for review, oldest first. Signing continues while they wait. */
  proposals: SignProposal[]
  /** Short status above the text, e.g. "Ready" or "Translating". */
  status: string
  /** Large hint shown while nothing is waiting, e.g. "Sign your message, then lower your hands." */
  hint: string
  activity: Activity | 'off'
  notice: string | null
  onDone: () => void
  showConfidence: boolean
  /** Show the other translations to choose from (a setting). */
  showSuggestions: boolean
  /** Choose a suggestion for the open translation by holding up 1, 2 or 3 fingers (a setting). */
  pickBySign: boolean
  /** Camera width / height. */
  aspect: number
  /** Accept one or more translations as one message. */
  onAccept: (text: string, ids: number[]) => void
  onDiscard: (ids: number[]) => void
  /** A suggestion was chosen by holding up a number (that stretch of signing is not a message). */
  onPickedBySign?: () => void
}

/**
 * Sign translation in one panel: status at the top, translations in a
 * scrolling list (the open one shows its suggestions, older ones fold to one
 * line), and "send all" pinned at the bottom. Nothing is sent automatically.
 */
export function SignPanel(p: Props) {
  const { proposals } = p
  const id = useId()
  // The signer's edits, by proposal id. Untouched items use the model's best suggestion.
  const [texts, setTexts] = useState<Record<number, string>>({})
  const [openId, setOpenId] = useState<number | null>(null)
  const [announce, setAnnounce] = useState('')
  const listRef = useRef<HTMLOListElement>(null)
  const textOf = (sp: SignProposal) => texts[sp.id] ?? sp.text
  const setText = (pid: number, t: string) => setTexts((m) => ({ ...m, [pid]: t }))

  const count = proposals.length
  const newest = proposals[count - 1]?.id ?? null
  // A new translation opens and scrolls into view; older ones fold up. Focus stays where it is.
  const prevNewest = useRef(newest)
  useEffect(() => {
    if (newest !== null && newest !== prevNewest.current) {
      setOpenId(newest)
      setAnnounce(`New translation to check. ${count} waiting.`)
      // After the newly opened item has laid out, so it is fully in view.
      requestAnimationFrame(() =>
        requestAnimationFrame(() => {
          const list = listRef.current
          if (list) list.scrollTo({ top: list.scrollHeight, behavior: 'smooth' })
        }),
      )
    }
    prevNewest.current = newest
  }, [newest, count])

  const open = proposals.find((sp) => sp.id === openId) ?? proposals[count - 1]
  const smoke = proposals.some((sp) => sp.model.stage === 'smoke')
  const merged = mergeTexts(proposals.map(textOf))

  const accept = (ps: SignProposal[]) => {
    const text = mergeTexts(ps.map(textOf))
    if (!text) return
    p.onAccept(text, ps.map((sp) => sp.id))
    setAnnounce(ps.length > 1 ? `${ps.length} translations sent as one message.` : 'Message accepted.')
  }

  return (
    <section className="signpanel" aria-labelledby={`${id}-h`}>
      <header className="signpanel__head">
        <span className={`tcard__dot tcard__dot--${count > 0 ? 'ready' : p.activity}`} aria-hidden="true" />
        <div className="signpanel__status">
          <h2 id={`${id}-h`} className="signpanel__title">
            {count > 1 ? `Check your translations (${count})` : count === 1 ? 'Check your translation' : p.status}
          </h2>
          {p.notice && <p className="signpanel__notice">{p.notice}</p>}
        </div>
        {p.activity === 'signing' && (
          <button type="button" className="btn btn--sm" onClick={p.onDone}>
            <Icon name="check" size={18} /> Done signing
          </button>
        )}
      </header>

      {smoke && (
        <p className="signpanel__warn" role="note">
          <Icon name="alert" size={16} strokeWidth={2} /> Test model: its text is not a real translation.
        </p>
      )}

      {count === 0 ? (
        <p className="signpanel__empty">{p.hint}</p>
      ) : (
        <ol ref={listRef} className="pending" aria-label="Translations to check">
          {proposals.map((sp, i) =>
            sp.id === open?.id ? (
              <OpenItem
                key={sp.id}
                index={i}
                proposal={sp}
                text={textOf(sp)}
                onText={(t) => setText(sp.id, t)}
                showConfidence={p.showConfidence}
                showSuggestions={p.showSuggestions}
                pickBySign={p.pickBySign}
                aspect={p.aspect}
                onChosen={(t, how) => {
                  if (how === 'sign') p.onPickedBySign?.()
                  setAnnounce(`Chose “${t}”${how === 'sign' ? ' by sign' : ''}. Accept it to send.`)
                }}
                onAccept={() => accept([sp])}
                onDiscard={() => p.onDiscard([sp.id])}
              />
            ) : (
              <li key={sp.id} className="pending__item pending__item--folded">
                <button type="button" className="pending__peek" aria-expanded="false" onClick={() => setOpenId(sp.id)}>
                  <span className="pending__num" aria-hidden="true">
                    {i + 1}
                  </span>
                  <span className="pending__peek-text">{textOf(sp)}</span>
                  <span className="sr-only">, open to edit</span>
                </button>
                <ItemActions index={i} disabled={!textOf(sp).trim()} onAccept={() => accept([sp])} onDiscard={() => p.onDiscard([sp.id])} />
              </li>
            ),
          )}
        </ol>
      )}

      {count > 1 && (
        <footer className="signpanel__foot">
          <p className="signpanel__merged" title={merged}>
            <span className="sr-only">All together: </span>
            {merged}
          </p>
          <div className="signpanel__foot-actions">
            <button type="button" className="btn btn--primary btn--sm" onClick={() => accept(proposals)} disabled={!merged}>
              <Icon name="check" size={18} /> Show all {count} as one
            </button>
            <button type="button" className="btn btn--sm" onClick={() => p.onDiscard(proposals.map((sp) => sp.id))}>
              Discard all
            </button>
          </div>
        </footer>
      )}

      <span className="sr-only" role="status">
        {announce}
      </span>
    </section>
  )
}

function ItemActions({ index, disabled, onAccept, onDiscard }: { index: number; disabled: boolean; onAccept: () => void; onDiscard: () => void }) {
  return (
    <span className="pending__actions">
      <button type="button" className="btn btn--primary btn--sm" onClick={onAccept} disabled={disabled} aria-label={`Show translation ${index + 1} to your partner`}>
        <Icon name="check" size={18} /> Show to partner
      </button>
      <button type="button" className="btn btn--sm btn--icon" onClick={onDiscard} aria-label={`Discard translation ${index + 1}`}>
        <Icon name="trash" size={18} />
      </button>
    </span>
  )
}

interface ItemProps {
  index: number
  proposal: SignProposal
  text: string
  onText: (t: string) => void
  showConfidence: boolean
  showSuggestions: boolean
  pickBySign: boolean
  aspect: number
  onChosen: (text: string, how: 'tap' | 'sign') => void
  onAccept: () => void
  onDiscard: () => void
}

function OpenItem({ index, proposal, text, onText, showConfidence, showSuggestions, pickBySign, aspect, onChosen, onAccept, onDiscard }: ItemProps) {
  const id = useId()
  const suggestions = showSuggestions && proposal.suggestions.length > 1 ? proposal.suggestions : []
  const selected = suggestions.findIndex((s) => s.text === text)
  const current = suggestions[selected] ?? proposal.suggestions[0] ?? proposal
  const signPicking = pickBySign && suggestions.length > 1
  const choose = (i: number, how: 'tap' | 'sign') => {
    const s = suggestions[i]
    if (!s) return
    onText(s.text)
    onChosen(s.text, how)
  }
  const hold = useNumberPick({ active: signPicking, count: suggestions.length, aspect, onPick: (n) => choose(n - 1, 'sign') })
  const sign = proposal.suggestions.find((s) => s.source === 'sign')?.sign
  const mine = current.source === 'mine'

  return (
    <li className="pending__item pending__item--open">
      <div className="pending__row">
        <span className="pending__num" aria-hidden="true">
          {index + 1}
        </span>
        <label htmlFor={`${id}-t`} className="sr-only">
          Translation {index + 1}
        </label>
        <textarea
          id={`${id}-t`}
          className="input pending__text"
          value={text}
          rows={1}
          onChange={(e) => onText(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault()
              if (text.trim()) onAccept()
            }
          }}
          aria-describedby={`${id}-m`}
        />
        <ItemActions index={index} disabled={!text.trim()} onAccept={onAccept} onDiscard={onDiscard} />
      </div>

      {suggestions.length > 1 && (
        <ol className="chips" aria-label={`Other translations for ${index + 1}`}>
          {suggestions.map((s, i) => {
            const holding = hold.candidate === i + 1 && selected !== i
            return (
              <li key={i}>
                <button
                  type="button"
                  className="chips__btn"
                  aria-pressed={selected === i}
                  aria-label={`${signPicking ? `Option ${i + 1}: ` : ''}${s.text}${s.lowConfidence ? ', low confidence' : ''}`}
                  onClick={() => choose(i, 'tap')}
                >
                  {signPicking && (
                    <span className="chips__num" aria-hidden="true">
                      {i + 1}
                    </span>
                  )}
                  <span aria-hidden="true">{s.text}</span>
                  {showConfidence && (
                    <span className="chips__pct" aria-hidden="true">
                      {Math.round(s.confidence * 100)}%
                    </span>
                  )}
                  {holding && <span className="chips__hold" aria-hidden="true" style={{ '--hold': hold.progress } as CSSProperties} />}
                </button>
              </li>
            )
          })}
        </ol>
      )}

      <p id={`${id}-m`} className="pending__meta">
        {current.lowConfidence ? <Icon name="alert" size={14} strokeWidth={2} /> : <Icon name="pencil" size={14} strokeWidth={1.8} />}
        <span>{mine ? 'Your sign · tap the text to edit' : current.lowConfidence ? 'Unsure. Check it, or tap the text to edit' : 'Tap the text to edit'}</span>
        {showConfidence && !mine && sign && <span>· sign “{sign}”</span>}
        {signPicking && <span className="pending__pick">· hold up 1–3 to pick{hold.candidate && selected !== hold.candidate - 1 ? ` (${hold.candidate}…)` : ''}</span>}
        {showConfidence && proposal.glosses.length > 0 && <span>· gloss {proposal.glosses.join(' ')}</span>}
      </p>
    </li>
  )
}
