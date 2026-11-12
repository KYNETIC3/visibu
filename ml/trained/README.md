# Trained checkpoints (PyTorch)

Weights-only copies of the two trained models: the optimiser state is removed, so they fit on GitHub. Use them to export again, evaluate, or keep training. The browser versions of the same models are in `public/models/`.

| Folder | What | Use it with |
|---|---|---|
| `pretrain-signs/model.pt` | Sign encoder + 250-sign classifier. Test top-1 60.7%, top-5 80.5% on unseen participants. | `export.export_signs_onnx --checkpoint trained/pretrain-signs/model.pt --install`<br>`training.train_slt --init-encoder trained/pretrain-signs/model.pt` |
| `slt-how2sign/model.pt` | Sentence model trained on 29,547 How2Sign sentences. Val BLEU 3.18. | `export.export_onnx --checkpoint trained/slt-how2sign/model.pt --prepared <folder with its vocab.json> --stage research --install`<br>`training.train_slt --init-model trained/slt-how2sign/model.pt` (fine-tune) |

Each folder also holds the run's training log, configuration and, for the sentence model, its `vocab.json` and `feature_stats.json`.

The sentence model derives from How2Sign (**CC BY-NC 4.0, non-commercial**). The word model derives from Kaggle `asl-signs` (CC BY 4.0). See [CREDITS.md](../../CREDITS.md).
