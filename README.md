# Visibu

A browser prototype for coffee counter conversations between a Deaf ASL signer and a hearing barista. Built for [LaunchPad Hacks 2026](https://launchpad-hacks.devpost.com/).

Visibu is not an ASL translator. It recognises a small set of signs taught by the signer, the signer confirms every message before it is spoken, and the barista's speech comes back as live captions. It does not replace a qualified ASL interpreter.

## Status

- **Stage 0 (foundations):** done. The tracking check screen tests whether a device can track hands, face and body fast enough.
- **Redesigned interface:** a calm, accessible UI for the conversation itself. Sign recognition is not connected to it yet (see below).

## Screens

- **Translate:** the camera sits on the side and captions are large. "I'm signing" and "Partner is speaking" modes, a full-screen **Show to partner** view, quick replies and typed messages.
- **Camera setup:** a three-step visual guide (allow camera, light and framing with a live brightness check, languages). No sound is needed at any step.
- **History:** conversations saved on this device only, with search, date filters, copy, share and delete.
- **Accessibility:** caption size, high contrast, reduced motion, and visual alerts (screen glow, vibration) in place of sound.
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

## Connecting sign recognition

The **UI, hand tracking (MediaPipe), and speech-to-text (Web Speech API) pipelines are fully functional**. The Translate screen, camera setup, history, and accessibility features all work end-to-end.

`createSignRecognizer()` in `src/lib/signRecognizer.ts` currently returns `null`, so the Translate screen shows one static example caption labelled "Example · not saved" (never added to the conversation or saved). The **gesture classification model interface is ready to be linked** — implement the `SignRecognizer` interface and return it from `createSignRecognizer()` to connect your classifier. The UI needs nothing else.

## Accessibility

- Atkinson Hyperlegible Next type (self-hosted), body text 17px or larger, touch targets 44px or larger, text contrast 4.5:1 or better.
- Speakers are told apart by label and icon as well as colour.
- Every control is keyboard reachable with a visible focus ring. There is a skip link, focus moves to each new page, and finished messages are announced to screen readers.
- All animation stops when the OS or the in-app setting asks for reduced motion.

## Privacy

- Camera video never leaves the device. Only landmark numbers are processed.
- Conversation history is stored in this browser only.
- MediaPipe code, models and fonts are self-hosted and version-pinned, and nothing loads from a CDN.
- The Content Security Policy in `vercel.json` only allows connections to this site, Supabase and Deepgram. This blocks MediaPipe's usage-metrics call to `odml.pa.googleapis.com`.

## Limitations

- Hearing learners built and tested this prototype, and no Deaf users have tested it.
- The tracker reads hands, face and body, but nothing in the app reads facial grammar yet.
- Sign recognition is not yet connected to the Translate screen.
- The redesigned interface is light-only for now.
- Accuracy figures will be added once they are measured.

## Credits

See [CREDITS.md](CREDITS.md).
