# Visibu: getting started

A step-by-step guide to running Visibu on your own computer and using every feature. No machine-learning knowledge is needed.

> **This is a demo.** Visibu will sometimes misread a sign or write the wrong sentence. You check every message before it is shown, and you can teach it your own signs in **My signs**. It has not been tested with Deaf signers and does not replace a qualified ASL interpreter.

---

## Contents

1. [What you need](#1-what-you-need)
2. [One-time setup](#2-one-time-setup)
3. [Start the app](#3-start-the-app)
4. [Check your camera](#4-check-your-camera)
5. [Translate your signing](#5-translate-your-signing)
6. [When your partner speaks](#6-when-your-partner-speaks)
7. [Teach Visibu your own signs (My signs)](#7-teach-visibu-your-own-signs-my-signs)
8. [Settings worth knowing](#8-settings-worth-knowing)
9. [Tips for better results](#9-tips-for-better-results)
10. [Troubleshooting](#10-troubleshooting)
11. [For developers](#11-for-developers)
12. [Change or retrain the models](#12-change-or-retrain-the-models)

---

## 1. What you need

- A computer with a **webcam**: Mac, Windows or Linux.
- **Node.js 20 or newer**. Check with `node -v`, or install it from https://nodejs.org.
- **Git**.
- A modern browser: **Chrome, Edge or Safari**.

The trained models are **included in the repository** (`public/models/`), so there is nothing else to download.

---

## 2. One-time setup

**Step 1. Get the code**

```bash
git clone https://github.com/KYNETIC3/visibu.git
cd visibu
```

**Step 2. Switch to the branch with the new features**

Skip this step once [KYNETIC3/visibu#8](https://github.com/KYNETIC3/visibu/pull/8) is merged.

```bash
git checkout feature/trained-sign-translation
```

**Step 3. Install the app's packages**

```bash
npm install
```

That's it: the models come with the code. To check they are there:

```bash
ls public/models/visibu-signs public/models/visibu-slt
```

You should see `signs.onnx` in the first folder and `encoder.onnx` / `decoder.onnx` in the second.

> **Already cloned before?** Just run `git pull` (on the branch above), then `npm install`.

---

## 3. Start the app

```bash
npm run dev
```

Then open **http://localhost:5173** in your browser.

- The first start downloads the hand-tracking model (about 14 MB), so give it a moment.
- Keep that terminal window open while you use the app. Press `Ctrl + C` in it to stop.
- When the browser asks for the **camera**, choose **Allow**. Video never leaves your computer.

The navigation bar at the top has: **Translate · My signs · History · Accessibility · Camera setup · Tracking check**.

---

## 4. Check your camera

Open **Camera setup** (or click **Start setup** on the banner). It takes about a minute:

1. **Allow the camera.**
2. **Light and framing:** face a window or lamp, not one behind you. Fit your head, shoulders and both hands in the picture. The live check tells you if it's too dark or too bright.
3. **Languages:** ASL → English.

---

## 5. Translate your signing

Open **Translate** and make sure **I'm signing** is selected.

**Step 1. Sign**
- Sign one word (like *hello*) or a whole sentence at a comfortable pace.
- The coloured dots on your fingers show what Visibu is tracking: **thumb** yellow, **index** blue, **middle** green, **ring** pink, **pinky** purple.

**Step 2. Lower your hands**
- Lowering your hands for about a second tells Visibu you've finished. You can also press **Done**.
- The status line at the top of the panel shows **Ready → Signing… → Translating…**.

**Step 3. Check the translation**
- It appears under **To check**. The newest one is open, with up to **3 suggestions** under the text box. For example, *hello* gives "Hello!" and "Hi, how are you doing?".
- "Unsure, check it" means the model isn't confident.
- "Your sign" means it matched a sign you taught (section 7).

**Step 4. Choose or edit**
- **Click** a suggestion to use it, **or**
- **hold up 1, 2 or 3 fingers, still, for about a second** to pick that suggestion hands-free. A bar fills while you hold.
- You can also type in the box to fix any word.

**Step 5. Send it**
- **Accept** (or press `Enter`) adds it to the conversation and shows it full-screen to your partner.
- **×** discards it.

**Keep signing.** You don't have to stop and accept each one. New translations line up in the panel and older ones fold into one line; click one to open it. When you're ready:
- **Send all N as one** joins everything waiting into one message, with a preview above the button.
- **Clear** discards everything waiting.

> Nothing is ever shown to your partner until you accept it.

---

## 6. When your partner speaks

Click **Partner is speaking**. Their speech appears as live captions, and the screen edge glows when they start talking.

- **Quick replies** ("Please wait a moment", "Thank you"…) send a ready-made message in one tap.
- **Type instead** lets you type a reply.
- **History** keeps past conversations on this computer only, with search, copy and delete.

---

## 7. Teach Visibu your own signs (My signs)

Visibu is a demo and won't know every sign, or your way of signing. **My signs** lets you show it a sign and tell it what it means.

**To teach a sign**
1. Open **My signs**.
2. In **What does the sign mean?**, type the meaning, e.g. *Can I get a latte, please?*
3. Click **Record**.
4. **Sign it, then lower your hands.** Do this **3 times**. A tick appears for each take.
   - If one take looks different from the others, Visibu asks you to **redo** it.
   - Click **Redo** under any take to record it again.
5. Click **Save**.

**If the sign already has a meaning**

If your new sign looks like one you've already taught, Visibu asks:

> *"This sign already has a meaning. It looks like your sign 'X'. Do you want to change it?"*

- **Change to "…"**: the old sign now means your new text.
- **Keep both**: saves it as a separate sign.
- **Cancel**: nothing changes.

If it looks like one of Visibu's 250 built-in signs (e.g. *hello*), it asks whether to use **your** meaning instead.

**Try it**
- Under **Try your signs**, click **Start** and sign one of your signs. You'll see the match, e.g. *"See you soon" · 92% match*, or the closest sign if nothing matched.
- In **Translate**, your sign now comes up **first**, labelled **Your sign**.

**Manage your signs**

Under **Your signs** you can:
- **Rename**: change the meaning.
- **Change sign**: record new takes for the same meaning.
- **🗑 Delete**: you're asked to confirm.

> Your signs are saved **only in this browser on this computer**, as numbers describing the movement, never video. A different browser or computer starts empty.

---

## 8. Settings worth knowing

Open **Accessibility**.

| Setting | What it does |
|---|---|
| **Show suggestions** | Shows up to 3 options per translation. |
| **Choose a suggestion by signing 1, 2 or 3** | The hands-free pick. Turn it off if it gets in the way. |
| **Show hand points on the camera** | The coloured finger dots. |
| **Show recognition confidence** | Shows a % next to each suggestion. It's a model score, not a guarantee. |
| **Glow when your hands leave the frame** | Tells you when you're out of view. |
| **Mirror camera view** | Shows the camera like a mirror. |
| **High contrast / Reduce motion / caption size** | For readability. |

---

## 9. Tips for better results

- **Light from the front**, plain background, no window behind you.
- **Both hands and your face in the picture**, about an arm's length from the camera.
- **Pause with your hands down** between messages. That's how Visibu knows a message has ended.
- **Short everyday signs** (hello, thank you, water, dog…) work best. Long sentences are often wrong; this is a demo.
- **Teach it** the phrases you use most in **My signs**. That's the most reliable way to get them right.

---

## 10. Troubleshooting

| Problem | Fix |
|---|---|
| **"Safari/Chrome can't connect to the server"** | The app isn't running. Run `npm run dev` again and keep the terminal open. |
| **"Sign translation is not available"** | The models are missing. Run `git pull`, check that `public/models/` has both folders, then refresh the page. |
| **"Camera access is blocked"** | Click the camera icon in the address bar → Allow, then refresh. Close other apps using the camera (Zoom, FaceTime). |
| **"Hands out of frame"** | Step back so both hands are visible, and improve the lighting. |
| **Nothing happens after signing** | Lower your hands fully for a second, or press **Done**. |
| **A suggestion got picked by accident** | Turn off **Choose a suggestion by signing 1, 2 or 3** in Accessibility. |
| **The computer feels slow** | Close other heavy tabs. Translation runs on your own computer. |
| **My signs disappeared** | They're per browser. Use the same browser on the same computer, and don't clear site data. |
| **The translation is wrong** | Expected sometimes. Pick another suggestion, edit the text, or teach the sign in **My signs**. |

---

## 11. For developers

```bash
npm test          # app tests (54)
npm run lint
npm run build     # production build (also type-checks)
npm run preview   # serve the build at http://localhost:4173
```

- **ML training, evaluation and export** (Python, `ml/`): see [ml/README.md](../ml/README.md). It covers the Kaggle cloud training and the commands that produce and `--install` the models.
- **How sign translation works inside the app:** the "Sign translation" section of the main [README](../README.md).
- **Licences for every dataset:** [CREDITS.md](../CREDITS.md). The How2Sign-trained models are **non-commercial**; check before using them in anything paid or judged for a prize.

---

## 12. Change or retrain the models

Everything used to train the models is in `ml/`, and the trained weights are in `ml/trained/`, so you can retrain, fine-tune or swap models yourself. The full reference is [ml/README.md](../ml/README.md). These are the common paths.

**Step 1. Set up Python (once).** You need Python 3.11 or newer.

```bash
cd ml
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m pytest -q tests          # should say "11 passed"
```

**Step 2. Pick what you want to do.**

| I want to… | Do this |
|---|---|
| **Teach new signs without any training** | Use **My signs** in the app (section 7). No Python needed. |
| **Re-export a model** (e.g. after changing export settings) | `python -m export.export_signs_onnx --checkpoint trained/pretrain-signs/model.pt --install` |
| **Fine-tune the sentence model** on more or different sentences | Prepare your data (ml/README "Pipeline"), copy `trained/slt-how2sign/vocab.json` into your prepared folder, then:<br>`python -m training.train_slt --prepared <your prepared folder> --out checkpoints/my-run --init-model trained/slt-how2sign/model.pt` |
| **Train the sentence model from scratch** on How2Sign | Easiest on Kaggle's free GPU, with no big download: see ml/README "Full How2Sign from pre-extracted landmarks" (`python -m tools.make_kaggle_kernel --user <your kaggle name> --job slt`). |
| **Retrain the 250-sign word model** | On Kaggle: ml/README "Pre-training on isolated signs" (`--job pretrain`). You need to accept the `asl-signs` competition rules on Kaggle first. |

**Step 3. Test, install and check.**

```bash
python -m training.evaluate --checkpoint <your best.pt> --prepared <prepared folder> --split val   # scores
python -m export.export_onnx --checkpoint <your best.pt> --prepared <prepared folder> --stage research --install
cd .. && npm test && npm run dev    # try it in the app
```

`--install` replaces the files in `public/models/`. Commit them on a **new branch** and open a pull request with your scores, so the team can compare against the current model (val BLEU 3.18).

**Ground rules**
- Never evaluate on data you trained on. The scripts keep train, val and test apart.
- Don't commit datasets (`ml/datasets/` is ignored) or your Kaggle token (`~/.kaggle/`).
- Kaggle needs your own account and API token: create it under Kaggle → Settings → API, and keep it private.
- How2Sign-trained models are **non-commercial** (see `public/models/README.md`).

