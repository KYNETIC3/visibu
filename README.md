# Visibu

A browser prototype for coffee counter conversations between a Deaf ASL signer and a hearing barista. Built for [LaunchPad Hacks 2026](https://launchpad-hacks.devpost.com/).

Visibu is not an ASL translator. It recognises a small set of signs taught by the signer, the signer confirms every message before it is spoken, and the barista's speech comes back as live captions. It does not replace a qualified ASL interpreter.

## Status

Stage 0 (foundations) is done. It has the tracking check screen only.

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
```

## Privacy

- Camera video never leaves the device. Only landmark numbers are processed.
- MediaPipe code and models are self-hosted and version-pinned, and nothing loads from a CDN.
- The Content Security Policy in `vercel.json` only allows connections to this site, Supabase and Deepgram. This blocks MediaPipe's usage-metrics call to `odml.pa.googleapis.com`.

## Limitations

- Hearing learners built and tested this prototype, and no Deaf users have tested it.
- The tracker reads hands, face and body, but nothing in the app reads facial grammar yet.
- Accuracy figures will be added once they are measured.

## Credits

See [CREDITS.md](CREDITS.md).
