# Visibu

Calm, accessible sign-language and speech translation for Deaf, hard-of-hearing and speech-impaired people.

- **Sign → text**: the camera sits on the side and recognised signs become large live captions.
- **Speech → text**: your partner's speech is captioned live (Web Speech API: Chrome, Edge, Safari).
- **Show to partner**: one tap shows a caption full-screen in very large text.
- **History**: conversations are saved on the device only, searchable, copyable and shareable.
- **Accessibility settings**: caption size, high contrast, reduced motion, visual alerts (screen glow, vibration) in place of sound.

## Run it

```bash
npm install
npm run dev
```

Open http://localhost:5173 and allow the camera and microphone.

## Connecting a sign-language model

No sign-language model is connected yet: `createSignRecognizer()` in `src/lib/signRecognizer.ts` returns `null`, and the Translate screen shows a static, clearly labelled example caption that is never added to the conversation or saved. To plug in a real model (for example MediaPipe hand landmarks plus a classifier), implement the `SignRecognizer` interface and return it from `createSignRecognizer()`. The UI needs nothing else.

## Accessibility notes

- Atkinson Hyperlegible Next type, body text 17px or larger, touch targets 44px or larger, text contrast 4.5:1 or better.
- Speakers are told apart by label and icon as well as colour.
- Every control is keyboard reachable with a visible focus ring. There is a skip link, focus moves to each new page, and finished messages are announced to screen readers.
- All animation stops when the OS or the in-app setting asks for reduced motion.
