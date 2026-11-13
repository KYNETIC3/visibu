import { describe, expect, it } from 'vitest'
import { signPhrases, signWord } from './signPhrases'
import { mergeSuggestions, mergeTexts, SINGLE_SIGN_MS, wordSuggestions, type RankedSuggestion } from './suggest'

const sentence = (text: string, confidence = 0.3): RankedSuggestion => ({ text, confidence, source: 'sentence' })

describe('sign phrases', () => {
  it('turns a single sign into everyday sentences', () => {
    expect(signPhrases('hello')).toContain('Hi, how are you doing?')
    expect(signPhrases('thankyou')[0]).toBe('Thank you.')
  })

  it('falls back to the word itself for other signs', () => {
    expect(signPhrases('giraffe')).toEqual(['Giraffe.'])
    expect(signWord('callonphone')).toBe('call on the phone')
  })
})

describe('word suggestions', () => {
  it('offers the top sign two ways, then the runners-up', () => {
    const s = wordSuggestions([
      { label: 'hello', prob: 0.7 },
      { label: 'bye', prob: 0.1 },
      { label: 'thankyou', prob: 0.05 },
    ])
    expect(s.map((x) => x.text)).toEqual(['Hello!', 'Hi, how are you doing?', 'Goodbye!'])
    expect(s[0]).toMatchObject({ source: 'sign', sign: 'hello', confidence: 0.7 })
  })
})

describe('unlikely signs', () => {
  it('drops runner-up signs that are very unlikely', () => {
    const s = wordSuggestions([
      { label: 'dog', prob: 0.94 },
      { label: 'car', prob: 0.004 },
    ])
    expect(s.map((x) => x.text)).toEqual(['Dog.'])
  })
})

describe('merging the two models', () => {
  const words = wordSuggestions([
    { label: 'hello', prob: 0.7 },
    { label: 'bye', prob: 0.1 },
  ])
  const sentences = [sentence('And then we are going to talk.'), sentence('So that is what.')]

  it('puts single signs first for a short stretch of signing, keeping one sentence', () => {
    const s = mergeSuggestions(words, sentences, 1500)
    expect(s.map((x) => x.source)).toEqual(['sign', 'sign', 'sentence'])
  })

  it('puts sentences first for long signing, keeping one sign', () => {
    const s = mergeSuggestions(words, sentences, SINGLE_SIGN_MS + 1)
    expect(s.map((x) => x.source)).toEqual(['sentence', 'sentence', 'sign'])
  })

  it('works with only one model and drops duplicates', () => {
    expect(mergeSuggestions([], [sentence('Hi.'), sentence('hi.'), sentence('Hello.')], 900).map((x) => x.text)).toEqual(['Hi.', 'Hello.'])
    expect(mergeSuggestions(words, [], 9000)).toHaveLength(3)
  })
})

describe('combining accepted pieces', () => {
  it('joins pieces into one message with sentence breaks', () => {
    expect(mergeTexts(['Hello!', 'how are you', 'Thank you.'])).toBe('Hello! how are you. Thank you.')
    expect(mergeTexts([' ', 'Yes.'])).toBe('Yes.')
  })
})
