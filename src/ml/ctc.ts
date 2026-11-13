// Greedy CTC decoding (best path): argmax per frame, collapse repeats, drop blanks.
// Mirrors ctc_greedy_decode in ml/models/ctc_head.py.

/** logProbs is row-major [frames x vocab]. */
export function ctcGreedyDecode(logProbs: Float32Array | number[], frames: number, vocab: number, blank = 0): number[] {
  const ids: number[] = []
  let prev = -1
  for (let t = 0; t < frames; t++) {
    let best = 0
    let bestVal = -Infinity
    for (let k = 0; k < vocab; k++) {
      const v = logProbs[t * vocab + k]
      if (v > bestVal) {
        bestVal = v
        best = k
      }
    }
    if (best !== prev && best !== blank) ids.push(best)
    prev = best
  }
  return ids
}
