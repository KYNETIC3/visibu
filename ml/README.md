# Visibu sign translation (ML)

This folder trains the model behind Visibu's sign translation and exports it for the browser.

**Status, November 2026.** The whole pipeline is built and tested end to end on real How2Sign data: landmark extraction, training, evaluation, ONNX export and browser inference. **No useful model has been trained yet.** The only model so far is a smoke-test model trained on 156 clips. Its output is meaningless (it repeats "and"), and the app labels it as a test model. The full training set could not be downloaded automatically (see [Blockers](#blockers)).

## What it does

```
webcam -> MediaPipe Holistic (existing tracker, in a worker)
       -> landmark features, 15 fps                   src/ml/features.ts  = common/features.py
       -> one sequence per signed message             src/ml/segmenter.ts
       -> recognition model: Transformer encoder + CTC head      encoder.onnx
       -> translation model: Transformer decoder, beam search    decoder.onnx
       -> up to 3 suggested English sentences + model scores     src/ml/beam.ts = models/beam.py
       -> signer picks one (tap, or sign 1/2/3), edits it, approves or discards it
                                                      src/components/ProposalReview.tsx
       -> only approved text reaches the partner
```

Everything runs on the device. There is no Python server and no cloud inference, and camera video never leaves the browser.

## Why it differs from the original plan

The plan was two stages: landmarks → **gloss** sequence (CTC), then gloss → English.

**How2Sign does not publish gloss annotations.** The official download script lists them under `## TODO`, and the "Gloss Annotation" links on how2sign.github.io point to an unrelated Open Images file. Without glosses, neither the gloss recogniser nor the gloss-to-English model can be trained on How2Sign.

The design keeps two separately exported models, trained jointly on what How2Sign does publish (sentence-aligned English):

| | Model A: recognition | Model B: translation |
|---|---|---|
| Input | landmark features [T, 786] | encoder memory + English so far |
| Network | Linear → Conv1d (stride 2) → 4-layer Transformer encoder, sinusoidal positions | 3-layer Transformer decoder, tied embeddings |
| Output | memory [T/2, 256] and CTC log-probs | next English word |
| ONNX file | `encoder.onnx` | `decoder.onnx` |

Loss = cross-entropy of the decoder + 0.3 × CTC.

- **With a gloss file** (`prepare_sequences.py --gloss file.tsv`, columns `SENTENCE_NAME` and `GLOSS`), the CTC head is a real continuous gloss recogniser. The app then shows the glosses beside the proposal.
- **Without one** (the How2Sign default), the CTC head trains on the sentence's English words as an auxiliary alignment loss. Those outputs are not glosses and the app never shows them.

The decoder learns English word order from the encoder's view of the whole signed sequence. It does not join tokens with spaces.

## Setup

```bash
cd ml
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -c "import torch; print(torch.backends.mps.is_available())"   # True on Apple Silicon
python -m pytest -q tests
```

The landmark extractor uses the app's MediaPipe model, so run `npm run assets` in the repo root once first.

All commands below run from `ml/` with the virtual environment active.

## Dataset: How2Sign

- **Source:** https://how2sign.github.io and https://github.com/how2sign/how2sign.github.io (`download_how2sign.sh`).
- **Licence:** [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). Attribution required and **no commercial use**. Videos and data must not be redistributed for commercial purposes; this repo never commits them (`ml/datasets/` is git-ignored). Trained weights are derived from the data, so treat them as non-commercial too.
- **Files used:** re-aligned English translations (`how2sign_realigned_{train,val,test}.csv`) and sentence-level front-view clips (`{split}_rgb_front_clips.zip`).

### Verified format

From the downloaded files, `python -m preprocessing.inspect_dataset`:

- **Annotation layout:** tab-separated, with columns `VIDEO_ID, VIDEO_NAME, SENTENCE_ID, SENTENCE_NAME, START_REALIGNED, END_REALIGNED, SENTENCE`.
- **Clip names:** each clip is `<SENTENCE_NAME>.mp4`, at 1280×720 and 24 fps.
- **Signer id:** `SENTENCE_NAME` = `<SENTENCE_ID>-<n>-rgb_front`. `n` runs 1–11, matching the signer numbers on the How2Sign site, so we use it as the signer id. That mapping is our inference; the dataset does not document it.

| Split | Sentences | Source videos | Hours | Median clip | Clips in archive |
|---|---|---|---|---|---|
| train | 31,165 | 2,192 | 59.3 | 5.3 s | not downloaded (blocker) |
| val | 1,741 | 115 | 3.3 | 5.7 s | 1,739 |
| test | 2,357 | 149 | 4.3 | 5.2 s | not downloaded |

- **Leakage:** no source video appears in two splits.
- **Signers:** most signers appear in every split, so How2Sign is not signer-independent. Signer 10 appears only in test; evaluation reports that subset separately as the only unseen-signer result.
- **Vocabulary:** 15,393 English word types in train; 7,604 occur at least 3 times. Out-of-vocabulary rate at that cut-off: 3.3% on val, 3.4% on test.

### Limitations that matter for Visibu

- **Topic:** the content is instructional "how-to" videos, not coffee-counter dialogue. Train-set counts: "latte" 7, "espresso" 28, "decaf" 0, "cash" 2. The model will not know a coffee order unless it is fine-tuned on coffee-counter signing.
- **Recording:** few signers, a green screen, studio lighting and one fixed camera, unlike a phone at a counter.
- **Tracking gaps:** on fast signing, MediaPipe often loses one or both hands. In the first 8 val clips, hand detection ranged from 0% to 100% of frames.
- **Facial grammar:** face landmarks are model inputs, but nothing measures whether non-manual markers are learned.

## Pre-training on isolated signs (Kaggle asl-signs)

Optional. It gives the sign encoder a head start on reading handshapes before it learns sentences from How2Sign.

- **Data:** Google's "Isolated Sign Language Recognition" Kaggle competition (`asl-signs`, PopSign ASL). It has 94,477 clips of 250 single signs from 21 people, distributed as **MediaPipe Holistic landmarks**, so no video and no extraction step. About 37 GB zipped.
- **Licence:** CC BY 4.0. The competition rules (7.A) allow commercial and non-commercial use; 7.B forbids redistributing the data, so it is never committed. Unlike How2Sign, this data does not conflict with the hackathon's commercial-use rule.
- **Download** needs a Kaggle account that has accepted the competition rules, plus an API token in `~/.kaggle/access_token`:
  ```
  python -m pip install -r requirements.txt
  kaggle competitions download asl-signs -p datasets/asl-signs/raw   # resumable
  ```
- **Import:** `python -m preprocessing.import_asl_signs` reads the parquet files straight from the zip (no 57 GB extraction) and writes float16 `.npy` frames in the app's 553-point layout.
  - Face: Kaggle has 468 points, not 478 (no iris). The features only use points up to 454, so nothing is lost.
  - Frame rate: not published. Assumed 30 fps (`--source-fps`) and resampled to 15 fps.
  - Aspect ratio: not published. **Estimated at 3:4 portrait** (`--aspect 0.75`). On How2Sign, where the 16:9 aspect is known, a face's cheek-to-cheek / forehead-to-chin ratio measured 0.78 (p10–p90 0.73–0.85, 300 clips). Three Kaggle sample clips then implied aspects of 0.68, 0.75 and 0.80. Re-check on more clips after import.
  - Splits are **by participant**: validation 2044 and 4718, test 16069 and 18796, the other 17 train. No person appears in two splits.
- **Pre-train:** `python -m training.pretrain_signs --out checkpoints/pretrain-signs` trains the same SignEncoder (by default the same size as `train_slt.py`) with a 250-way classifier on the time-averaged encoder output. It reports top-1 and top-5 accuracy on the validation participants. `--evaluate-test` reports the test participants once, at the end.
- **Use it:** `python -m training.train_slt ... --init-encoder checkpoints/pretrain-signs/best.pt`. The encoder weights and their feature statistics are loaded, the classifier is dropped, and the app never sees the 250 labels.
- **Training on Kaggle instead (optional).** The competition data already sits on Kaggle's servers, so pre-training can run there with no 37 GB download, and the Mac can sleep meanwhile. This uses Kaggle's NVIDIA (CUDA) GPU for this training step only, a deliberate exception to "train on the Mac" agreed with the project owner on 11 Nov 2026. The app still runs everything on the device.
  ```
  python -m tools.make_kaggle_kernel --user <kaggle username>      # packs this repo's code into one script
  kaggle kernels push -p exports/kaggle-kernel                      # private notebook, GPU, no internet
  kaggle kernels status <user>/visibu-sign-pretrain
  kaggle kernels output <user>/visibu-sign-pretrain -p checkpoints/pretrain-signs
  ```
  The notebook imports the landmarks into its temporary disk (they are not saved or shared), pre-trains with an 11-hour budget (Kaggle stops runs at 12 h), evaluates once on the test participants, and keeps `best.pt` plus the reports as output. A local simulation of the notebook (paths redirected, CPU) ran end to end.
- **Tested** end to end on three real sample clips (import, pre-training on MPS, test evaluation, and starting `train_slt` from the result). Not yet run on the full data, so there are no accuracy numbers yet.

## Full How2Sign from pre-extracted landmarks (How2Sign Holistic)

Downloading the 31 GB train videos failed here (Drive quota, slow network), and only 4,488 train clips were recovered. A model trained on those overfitted after 8 epochs: val BLEU 1.85, outputs mostly generic how-to English.

- **Data:** "How2Sign Holistic" ([Hugging Face](https://huggingface.co/datasets/PSewmuthu/How2Sign_Holistic), and [on Kaggle](https://www.kaggle.com/datasets/psewmuthu/how2sign-holistic) as `psewmuthu/how2sign-holistic`). It holds MediaPipe Holistic landmarks for every How2Sign sentence clip, with the realigned CSVs. The page says MIT, but the data derives from How2Sign, so **How2Sign's CC BY-NC 4.0 terms still apply**.
- **Format, verified on 76 val clips against this repo's own extraction of the same clips:**
  - Arrays are float32 `[T, 543, 3]`, ordered pose (33), face (468), left hand, right hand, at every source frame (24 fps). A part that was not detected is all zeros.
  - Mean xy difference from ours: face 0.0007, hands about 0.002–0.005, pose about 0.01 (the older Holistic pose model). After conversion the features differ by a median of 0.011 (features are about unit scale).
  - Hand detection rates are close: left hand missing 18% vs 16%, right hand 11% vs 9%. Frame counts after resampling to 15 fps match within 1.
- **Import:** `preprocessing/import_how2sign_holistic.py` converts zeros to NaN, maps the points into the 553-point layout (iris slots NaN) and resamples to 15 fps. It writes the same `landmarks/` and `annotations/` files as `extract_landmarks.py`, so the rest of the pipeline is unchanged.
- **Training on Kaggle:** the dataset is already on Kaggle, so a private notebook attaches it with no download and no internet access. The notebook also attaches the pre-training notebook's output for `--init-encoder`. It runs import, manifests, vocabulary, feature statistics, `train_slt.py --device cuda` within an 11-hour budget, then a val report.

  ```bash
  python -m tools.make_kaggle_kernel --user <kaggle username> --job slt -- --batch-size 32
  kaggle kernels push -p exports/kaggle-slt
  kaggle kernels output <user>/visibu-slt-how2sign -p checkpoints/slt-how2sign
  ```

  The same steps ran end to end on this Mac with a 76-clip sample (CPU, 2 epochs).
- **Result (12 Nov 2026):**
  - The Kaggle run took 2.4 h on GPU. It trained on 29,547 sentences (1,075 longer than 20 s were skipped), with a 7,608-word vocabulary and 12.3 M parameters.
  - Validation loss was best at epoch 10 (CE 4.04) and early stopping ended the run at epoch 22.
  - Scored on the same 1,713 val clips as the 4,418-sentence model, using this repo's own MediaPipe extraction (the app's landmarks), with beam 4:

  | | BLEU | BLEU-1 | chrF | ROUGE-L | best-of-3 BLEU | CPU latency (median) |
  |---|---|---|---|---|---|---|
  | 4,418 recovered sentences | 1.85 | 24.8 | 16.2 | 13.0 | 2.13 | 89 ms |
  | full How2Sign | **3.18** | 25.9 | **20.4** | **15.7** | 3.49 | 142 ms |

  - The ONNX export gave identical suggestions to PyTorch on 30/30 val clips. Installed as the app's sentence model.
  - Still far from usable translation; published How2Sign baselines are about 8–15 BLEU.
- **Expect:** better sentences on How2Sign's topics. It will still not handle everyday phrases such as "hello", which How2Sign barely contains (23 of 4,418 recovered train sentences start with hello/hi).

## Pipeline

```bash
# 1. Annotations (small) and clips (val 1.7 GB, test 2.2 GB, train 31 GB)
python -m preprocessing.download_how2sign --accept-licence annotations
python -m preprocessing.download_how2sign --accept-licence clips --split val
python -m preprocessing.download_how2sign --accept-licence clips --split test
python -m preprocessing.download_how2sign --accept-licence clips --split train   # see Blockers

# 2. Landmarks: MediaPipe Holistic 1.2.2, the same .task model as the app, 15 fps, float16
python -m preprocessing.extract_landmarks --split val
python -m preprocessing.extract_landmarks --split test
python -m preprocessing.extract_landmarks --split train          # resumable

# 3. Manifests, vocabulary (train only), feature statistics (train only)
python -m preprocessing.inspect_dataset
python -m preprocessing.build_vocabulary
python -m preprocessing.prepare_sequences
python -m preprocessing.normalize_landmarks

# 4. Train (MPS if available, else CPU). Logs: checkpoints/<run>/train_log.jsonl
python -m training.train_slt --out checkpoints/slt-base
python -m training.train_slt --out checkpoints/slt-base --resume   # continue; keep the same --epochs

# 5. Evaluate on val for model selection; run test once, at the end
python -m training.evaluate --checkpoint checkpoints/slt-base/best.pt --split val
python -m training.evaluate --checkpoint checkpoints/slt-base/best.pt --split test

# 6. Export, check against PyTorch, install into the app (public/models/visibu-slt/)
python -m export.export_onnx --checkpoint checkpoints/slt-base/best.pt --stage research \
  --eval-report reports/eval_slt-base_val.json --install
python -m export.validate_onnx --checkpoint checkpoints/slt-base/best.pt --split val --limit 50
```

### Smoke test, before the train split is available

This trains on part of val and evaluates on held-out val videos. It never touches test, and its numbers mean nothing:

```bash
python -m tools.make_smoke_split --source val
python -m preprocessing.normalize_landmarks --manifest smoke_train
python -m training.train_slt --train-manifest smoke_train --val-manifest smoke_eval --out checkpoints/smoke \
  --epochs 6 --batch-size 8 --d-model 128 --enc-layers 2 --dec-layers 2 --ff 256 --warmup-steps 20 --workers 0
python -m training.evaluate --checkpoint checkpoints/smoke/best.pt --split smoke_eval
python -m export.export_onnx --checkpoint checkpoints/smoke/best.pt --stage smoke --install
```

## Features (`common/features.py`, mirrored in `src/ml/features.ts`)

786 values per frame at 15 fps. x and z are multiplied by the image aspect ratio first, so distances are not stretched.

| Block | Size | What it encodes |
|---|---|---|
| Global positions | 127 × 2 | 11 upper-body pose points, 74 face points, both hands; relative to the shoulder midpoint, scaled by shoulder width |
| Hand shape | 2 × 21 × 3 | each hand relative to its wrist, scaled by palm length, including depth (orientation) |
| Face shape | 74 × 2 | lips, eyebrows, eyes, nose and face outline relative to the nose tip, scaled by eye distance (non-manual markers) |
| Motion | 127 × 2 | frame-to-frame change of the global positions |
| Presence | 4 | pose, face, left hand, right hand detected |

- **Missing parts:** a part that isn't detected gives zeros and a 0 presence flag.
- **Anchor:** the shoulder anchor is carried forward causally.
- **Standardisation:** per-feature mean and standard deviation from train are baked into `encoder.onnx`.
- **Parity:** `python -m tools.make_parity_fixture` writes `src/ml/__fixtures__/parity.json` from synthetic frames, and `npm test` checks that the TypeScript port matches to within 1e-5.

## Measured on this machine (Apple M2, 8 GB)

**Landmark extraction:**

- 37 ms per frame at 640 px wide; 52 ms at 1280 px.
- Half resolution gave the same detections (mean difference 0.0008 of image width) and is the default.
- Throughput tops out at about 20 frames/s, however many processes run: MediaPipe already uses every core.
- The GPU delegate fails in MediaPipe's Python build ("ImageToTensorConverterMetal" needs GPU input).
- **Full How2Sign: about 3.6 M frames, so about 50 hours. Train alone: about 44 hours.**
- Under sustained load on this 8 GB machine, with swap in use, throughput dropped to about 0.06 clips/s, roughly 5× slower. Close other apps for long runs.

**Training:** the default model has 12.3 M parameters (6 encoder layers, 3 decoder layers, d=256). Up to 80 epochs, stopping early after 12 without improvement.

- Measured with the earlier 10.75 M-parameter (4-encoder-layer) model: one pass over 156 clips took 12.5 s on MPS and 18.5 s on CPU, about 40 minutes per epoch on the full train set.
- The deeper encoder adds roughly a third to that, so expect **about 50 minutes per epoch, about 20–40 hours in total**. This is an estimate, not a measurement. `--enc-layers 4` restores the smaller model.
- The Mac can do this overnight; no cloud is needed. If memory is tight, use `--batch-size 8 --max-frames 225`.

**CTC on MPS:** PyTorch has no MPS kernel for CTC loss, so that one loss runs on the CPU. Gradients still flow back to the GPU.

**ONNX parity:**

- Encoder memory, CTC and decoder outputs match PyTorch to within 7e-6, at 8 to 301 frames.
- Beam search gave identical suggestions (all three, same order) in ONNX Runtime and PyTorch on 21 of 21 held-out clips. The decoder accepts several beams at once (`[B, L]` tokens).
- The browser's beam search (`src/ml/beam.ts`) gives identical results to `models/beam.py` on the fixture cases in `npm test`.
- One real held-out clip gave identical text in the browser, in both dev and the production build under its Content Security Policy. The model score also matched, checked in dev (0.115 vs 0.115).

**Browser:** ONNX Runtime Web 1.31.1, WASM.

- Model load: 0.5–0.8 s.
- Translation: 0.7–0.8 s with the smoke model under greedy decoding, which ran to the 60-word cap. With beam search and repetition blocking it stops early; ONNX Runtime CPU (Python) decoding took 42 ms median on held-out clips. Expect a trained model's beam search to cost several decoder calls per word, so measure again after training.

## Decoding and suggestions

- **Beam search**, width 4, returns the 3 best distinct sentences (`models/beam.py`, ported line for line to `src/ml/beam.ts`).
- **Repetition blocking:** no 3-word sequence may repeat (`no_repeat_ngram=3`), and a sentence can't be empty. `<unk>` is never generated.
- **Score:** mean token log-probability; `exp(score)` is the "model score" in the app. It is not a calibrated accuracy.
- The settings travel in `manifest.json` (`decoding`). Models exported before this change have a batch-1 decoder; the app still runs them, one beam at a time.
- `training/evaluate.py` decodes with the same beam search (`--beam 1` for greedy) and reports **oracle BLEU over the 3 suggestions**: the score if the signer always picked the closest one. That measures how much the suggestion picker can help.

### Choosing a suggestion by signing 1, 2 or 3

`src/ml/numberSign.ts` reads which fingers are straight in the tracked hand: index only is 1; index and middle apart is 2; thumb, index and middle is 3. The number must be held for 0.8 s (tracking glitches under 250 ms are ignored), and the hand has to drop or change before it selects again.

This is a fixed control gesture, like a button, not sign recognition, and no model is involved. It only puts that suggestion's text in the review box; the signer still approves with the button. It is on by default and can be turned off in Accessibility → Sign translation, as can the suggestions themselves. It has been tested on synthetic hand shapes only, not yet on real signers' hands.

## Browser deployment decisions

- **WASM only** (`onnxruntime-web/wasm`, 14.8 MB of wasm, 3.9 MB gzipped).
  - WebGPU is not enabled. The WebGPU build is 29.5 MB, WebGPU isn't available in every browser, and this model is small and decodes one word at a time, where GPU dispatch overhead outweighs the gain.
  - Revisit if a larger model is trained, and measure first.
- **Single thread, inside a Web Worker** (`src/ml/translator.worker.ts`). Multithreaded WASM needs cross-origin isolation headers, which the app does not set. Inference never blocks the UI.
- **Self-hosted files.** Vite bundles the wasm into `dist/assets`, and models are served from `/models/visibu-slt/`.
  - The worker checks every file against the sha256 values in `manifest.json` before using it.
  - No CDN, and nothing is fetched from another origin.
  - The existing Content Security Policy already allows all of this (`'wasm-unsafe-eval'`, `connect-src 'self'`), so `vercel.json` is unchanged.
- **Missing model.** If `/models/visibu-slt/manifest.json` is absent, the app says sign translation is unavailable. Typed messages, quick replies and speech captions keep working.
- **Model files are not committed** (`public/models/` is git-ignored) because they derive from non-commercial data. Install them with `export_onnx.py --install` on the machine that builds the app.

## Blockers

1. **The train clips can't be downloaded automatically.** Google Drive answered "Quota exceeded" for the 31 GB `train_rgb_front_clips.zip`. This is common for popular Drive files. To get it:
   - open the file in a signed-in browser (https://drive.google.com/file/d/1VX7n0jjW0pW3GEdgOks3z8nqE6iI6EnW), use "Add shortcut to Drive" or "Make a copy", and download from your own Drive; or
   - retry after 24 hours.

   Save it as `ml/datasets/how2sign/raw/train_rgb_front_clips.zip`, then rerun the `clips --split train` command; it unzips an existing archive.
2. **Licence vs. prize.** How2Sign is CC BY-NC 4.0. `CREDITS.md` says every dataset must allow commercial use because first place wins a cash prize. Whether a prize-eligible hackathon entry counts as commercial use is a question for the organisers or the How2Sign authors (amanda.duarte@upc.edu), not something this code can settle.
3. **No glosses** (see above). Real gloss supervision needs annotations from somewhere else.
4. **Time.** About 50 hours of extraction plus about 20–40 hours of training on this Mac.

## What has to be true before anyone relies on it

- Report val and test BLEU, chrF and ROUGE-L from `training/evaluate.py`, plus the signer-10 unseen-signer subset, without tuning on test.
- Fine-tune and evaluate on coffee-counter signing recorded with consent from Deaf signers. How2Sign alone won't produce coffee orders.
- Test with Deaf ASL signers across signing styles and speeds, camera positions, occlusion, facial grammar and spatial references.
- Treat the model score as a score, not an accuracy, until it is calibrated against real error rates.
- It is a research prototype. It does not replace a qualified interpreter.
