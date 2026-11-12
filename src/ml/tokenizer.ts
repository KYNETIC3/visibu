// Word-level tokeniser, mirrored from ml/common/tokenizer.py.

const SPECIALS = new Set(['<pad>', '<bos>', '<eos>', '<unk>'])
const PUNCT = new Set(['.', ',', '!', '?', ';', ':'])
const TOKEN_RE = /[a-z0-9]+(?:'[a-z]+)*|[.,!?;:]/g

export function normalize(text: string): string {
  return text.toLowerCase().replace(/[‘’]/g, "'").replace(/[“”]/g, '"')
}

export function tokenize(text: string): string[] {
  return normalize(text).match(TOKEN_RE) ?? []
}

export function detokenize(tokens: string[]): string {
  let out = ''
  let capNext = true
  for (const t of tokens) {
    if (SPECIALS.has(t)) continue
    let word = t
    if (word === 'i' || word.startsWith("i'")) word = 'I' + word.slice(1)
    if (capNext && !PUNCT.has(word)) {
      word = word.slice(0, 1).toUpperCase() + word.slice(1)
      capNext = false
    }
    if (PUNCT.has(word)) {
      out += word
      if (word === '.' || word === '!' || word === '?') capNext = true
    } else {
      out += (out ? ' ' : '') + word
    }
  }
  return out
}
