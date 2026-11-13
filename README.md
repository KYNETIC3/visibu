# Visibu

A browser prototype for coffee counter conversations between a Deaf ASL signer and a hearing barista. Built for [LaunchPad Hacks 2026](https://launchpad-hacks.devpost.com/).

Visibu is a research prototype, not a finished ASL translator. The signer signs a whole message and an on-device model proposes an English sentence. The signer approves, edits or discards it, and only approved text is shown to the barista. The barista's speech comes back as live captions. It does not replace a qualified ASL interpreter.

**New here? Start with [docs/GETTING-STARTED.md](docs/GETTING-STARTED.md):** setup, every feature step by step, and troubleshooting.

## Status

- **Stage 0 (foundations):** done. The tracking check screen tests whether a device can track hands, face and body fast enough.
- **Redesigned interface:** a calm, accessible UI for the conversation itself.
- **Sign translation (research):** the rule-based five-sign demo has been replaced. In its place: a continuous sign-to-English pipeline (Transformer encoder with a CTC head, plus a Transformer decoder), trained in Python on How2Sign and run in the browser with ONNX Runtime Web. Two research models are trained:
  - a 250-sign word model (test top-1 61%, top-5 80% on unseen signers);
  - a sentence model on the full How2Sign set (val BLEU 3.18, still early).

  Signers can also teach their own signs in **My signs**. See [ml/README.md](ml/README.md).

## Screens

- **Translate:** the camera sits on the side and captions are large. "I'm signing" and "Partner is speaking" modes, a full-screen **Show to partner** view, quick replies and typed messages.
- **My signs:** teach Visibu your own signs. Type what a sign means, then sign it 3 times. If it looks like a sign you already taught (or one Visibu knows), it asks whether to change that sign's meaning, keep both, or cancel. Saved signs can be renamed, re-recorded ("Change sign"), deleted and tried out. In Translate, a matching sign is offered first as "Your sign".
- **Camera setup:** a three-step visual guide (allow camera, light and framing with a live brightness check, languages). No sound is needed at any step.
- **History:** conversations saved on this device only, with search, date filters, copy, share and delete.
- **Accessibility:** caption size, light or dark theme (follows the device by default), high contrast, reduced motion, and visual alerts (screen glow, vibration) in place of sound.
- **Tracking check:** the Stage 0 device check (speed, hand visibility, GPU or CPU).

## Run it locally

You need Node.js 20 or newer and a webcam.

```bash
npm install
npm run dev
```

Open http://localhost:5173. The first run downloads the pinned MediaPipe model (13.7 MB) into `public/mediapipe/`.

To test the production build with the production Content Security Policy:

```bash
npm run build && npm run preview
```

Then open http://localhost:4173.

## Tests

```bash
npm test
npm run lint
```

`npm test` includes parity tests that check the browser feature code against the Python training code. The ML tests run from `ml/`:

```bash
cd ml && source .venv/bin/activate && python -m pytest -q tests
```

## Sign translation

The **UI, hand tracking (MediaPipe) and speech-to-text (Web Speech API) pipelines work end to end**. Sign translation works like this:

1. MediaPipe Holistic tracks hands, face and upper body in a worker (unchanged).
2. `src/ml/segmenter.ts` collects one landmark sequence per message, at 15 fps. A message ends when the hands are down for 1.2 s or the signer presses **Done signing**.
3. `src/ml/translator.worker.ts` computes features (`src/ml/features.ts`, the same code as training), then runs `encoder.onnx` and `decoder.onnx` with ONNX Runtime Web (WASM) in a Web Worker.
4. `createSignRecognizer()` in `src/lib/signRecognizer.ts` returns that recogniser. It reports hands, activity (waiting, signing, translating) and a **proposal**.
5. The **sign panel** (`src/components/ProposalReview.tsx`) lists translations as they arrive, while the signer keeps signing. Each has up to 3 **suggestions**, chosen in this order:
   - a matching sign the user taught (`src/ml/mySigns.ts`);
   - the 250-sign word model (`signs.onnx`), whose single signs become short sentences (`src/ml/signPhrases.ts`);
   - the sentence model (beam search).

   The signer can pick a suggestion by tapping it or by holding up 1, 2 or 3 fingers still, and can edit the text. Each translation is then accepted on its own or **sent all as one message** (saved as "You · signed" and opened in **Show to partner**), or discarded. Picking only fills in the text: nothing is spoken or shown to the partner until the signer accepts. Suggestions, sign picking and the finger points on the camera can each be turned off in Accessibility → Sign translation.
6. **My signs:**
   - Each taught sign is stored as fingerprints: the word model's averaged encoder output for each take, 256 numbers, in localStorage. No video and no landmark sequences are stored.
   - A new clip matches a taught sign when the cosine similarity to the average of its takes is at least the threshold in the model's manifest (0.42). That threshold was set so 1% of different-sign pairs pass, on simulated repeats (augmented How2Sign clips); real repeats vary more.
   - Signs taught with a different word-model version are flagged for re-recording.

The trained models are committed in `public/models/` (licences in [public/models/README.md](public/models/README.md); the How2Sign sentence model is CC BY-NC 4.0, non-commercial), so a clone works without extra downloads. Weights-only training checkpoints are in [ml/trained/](ml/trained/README.md); to retrain or fine-tune, see [docs/GETTING-STARTED.md](docs/GETTING-STARTED.md#12-change-or-retrain-the-models). If the model files are missing, the Translate screen says sign translation is unavailable, and typed messages, quick replies and speech captions keep working.

## Accessibility

- Atkinson Hyperlegible Next type (self-hosted), body text 17px or larger, touch targets 44px or larger, text contrast 4.5:1 or better.
- Speakers are told apart by label and icon as well as colour.
- Every control is keyboard reachable with a visible focus ring. There is a skip link, focus moves to each new page, and finished messages are announced to screen readers.
- All animation stops when the OS or the in-app setting asks for reduced motion.

## Privacy

- Camera video never leaves the device. Only landmark numbers are processed.
- Conversation history and taught signs (meanings plus fingerprints, never video) are stored in this browser only.
- MediaPipe code, models and fonts are self-hosted and version-pinned, and nothing loads from a CDN.
- Sign translation runs in the browser. The ONNX Runtime WASM file is bundled with the app, and the models are served from this site and checked against sha256 values in their manifest. No new network destinations were added.
- The Content Security Policy in `vercel.json` only allows connections to this site, Supabase and Deepgram. This blocks MediaPipe's usage-metrics call to `odml.pa.googleapis.com`.

## Limitations

- Hearing learners built and tested this prototype, and no Deaf users have tested it.
- No usable sign translation model has been trained yet. The only model so far is a smoke test trained on 156 clips; its output is meaningless and the app labels it "Test model".
- The training data (How2Sign) is instructional videos, not coffee orders, recorded with few signers in a studio. A useful coffee-counter model needs fine-tuning on consented coffee-counter signing.
- How2Sign has no gloss annotations, so the model translates directly to English and shows no glosses.
- Face landmarks are model inputs, but nobody has measured whether facial grammar is learned.
- The model score is not a calibrated accuracy. Accuracy figures will be added once a trained model is evaluated (BLEU, chrF and ROUGE-L on How2Sign val and test, then testing with Deaf signers).
- How2Sign is licensed for non-commercial use only. Check with the organisers whether a prize-eligible entry may use it (see [ml/README.md](ml/README.md#blockers)).

## Credits

See [CREDITS.md](CREDITS.md).
