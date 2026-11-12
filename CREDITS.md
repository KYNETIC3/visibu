# Credits and disclosures

LaunchPad Hacks requires every entry to credit libraries, datasets, assets and AI assistance. Update this file whenever something new is added.

## Work done during the event

All code in this repository was written during LaunchPad Hacks, which ran from 16 October to 20 November 2026. The repository was created on 6 November 2026.

## AI assistance

- Most of the code was written by Claude (Anthropic), working in Claude Code.
- The team reviewed and tested the code, learned the signs, and made the demo.

## Libraries

| Library | Use | Licence |
|---|---|---|
| [MediaPipe Tasks Vision](https://www.npmjs.com/package/@mediapipe/tasks-vision) 1.2.2 | Hand, face and body tracking | Apache-2.0 |
| MediaPipe Holistic Landmarker model (float16, version 1) | Tracking model | See the [MediaPipe model card](https://developers.google.com/edge/mediapipe/solutions/vision/holistic_landmarker) |
| [React](https://react.dev) | User interface | MIT |
| [Vite](https://vite.dev) | Build tool | MIT |
| [Vitest](https://vitest.dev) | Tests | MIT |
| [TypeScript](https://www.typescriptlang.org) | Language | Apache-2.0 |
| [ONNX Runtime Web](https://onnxruntime.ai) 1.31.1 | Runs the sign translation models in the browser | MIT |
| [PyTorch](https://pytorch.org) 2.15.0 | Trains the sign translation models (not shipped) | BSD-3-Clause |
| [MediaPipe (Python)](https://pypi.org/project/mediapipe/) 1.2.2 | Extracts training landmarks (not shipped) | Apache-2.0 |
| [OpenCV](https://opencv.org), [ONNX](https://onnx.ai), [sacreBLEU](https://github.com/mjpost/sacrebleu), [rouge-score](https://pypi.org/project/rouge-score/) | Video decoding, export, evaluation (not shipped) | Apache-2.0 / Apache-2.0 / Apache-2.0 / Apache-2.0 |

## Datasets

Any dataset added later must allow commercial use, because first place wins a cash prize.

| Dataset | Use | Licence |
|---|---|---|
| [How2Sign](https://how2sign.github.io) (Duarte et al., CVPR 2021) | Training and evaluating the sign translation model: re-aligned English translations and front-view sentence clips. Not redistributed; nothing from it is committed. | CC BY-NC 4.0 (**non-commercial**) |
| [How2Sign Holistic](https://huggingface.co/datasets/PSewmuthu/How2Sign_Holistic) (PSewmuthu, 2025; also on Kaggle as `psewmuthu/how2sign-holistic`) | MediaPipe Holistic landmarks for every How2Sign sentence clip. Used, on Kaggle only, to train the sentence model on the full How2Sign corpus without downloading its videos. Not redistributed; nothing from it is committed. | Labelled MIT, but derived from How2Sign, so treated as **CC BY-NC 4.0** (non-commercial), like How2Sign itself. |
| [Google – Isolated Sign Language Recognition](https://www.kaggle.com/competitions/asl-signs) (Kaggle, 2023; PopSign ASL data) | Pre-training the sign encoder on 250 isolated signs, from the MediaPipe landmarks Kaggle distributes (no video). Not redistributed; nothing from it is committed. | **CC BY 4.0**, and the competition rules (section 7.A) allow use "for any purpose, whether commercial or non-commercial". Attribution required. Section 7.B forbids redistributing the data to anyone who hasn't accepted the rules, so it stays git-ignored in `ml/datasets/`. |

**Open issue:** How2Sign's non-commercial licence conflicts with the rule above. Ask the LaunchPad Hacks organisers, and if needed the How2Sign authors, before a How2Sign-trained model is used in the judged submission. Until then, the submitted build should ship without a model (the app says sign translation is unavailable) or describe the model as a non-commercial research demo.
