# Visibu models

These are the trained models the app loads in the browser. They are committed so a fresh clone works straight away.

| Folder | Model | Trained on | Licence of the training data |
|---|---|---|---|
| `visibu-signs/` | Word model: 250 everyday signs, plus the "fingerprint" used by **My signs** | Google – Isolated Sign Language Recognition (Kaggle `asl-signs`, PopSign ASL) | CC BY 4.0. Commercial use allowed with attribution to Google and PopSign. |
| `visibu-slt/` | Sentence model: How2Sign sentences → English | How2Sign (Duarte et al., CVPR 2021), via the pre-extracted How2Sign Holistic landmarks | **CC BY-NC 4.0: non-commercial use only**, with attribution to the How2Sign authors |

- Every file is checked against the sha256 values in its folder's `manifest.json` before the app uses it.
- Scores and training details are in each `manifest.json` and in [ml/README.md](../../ml/README.md).
- To retrain or replace a model, see [docs/GETTING-STARTED.md](../../docs/GETTING-STARTED.md#12-change-or-retrain-the-models). The export scripts' `--install` option overwrites these folders.
- Before using the How2Sign sentence model in anything commercial, or judged for a cash prize, check with the event organisers or the How2Sign authors.
