"""Unit tests for the shared ML code. Run from ml/:  python -m pytest -q tests"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.features import FEATURE_DIM, sequence_features  # noqa: E402
from common.layout import LEFT_START, RIGHT_START, TOTAL_POINTS  # noqa: E402
from common.tokenizer import BOS_ID, EOS_ID, Vocab, detokenize, tokenize  # noqa: E402
from models.ctc_head import ctc_greedy_decode  # noqa: E402
from models.slt import ModelConfig, SLTModel  # noqa: E402


def frames(T=12, seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(0.2, 0.8, size=(T, TOTAL_POINTS, 3)).astype(np.float32)


def test_feature_shape_and_finite():
    f = frames()
    f[3, LEFT_START:LEFT_START + 21] = np.nan
    x = sequence_features(f, 16 / 9)
    assert x.shape == (12, FEATURE_DIM)
    assert np.isfinite(x).all()
    assert x[3, -2] == 0.0 and x[3, -1] == 1.0  # left hand absent, right present


def test_features_are_causal():
    f = frames(20)
    full = sequence_features(f, 4 / 3)
    prefix = sequence_features(f[:9], 4 / 3)
    assert np.allclose(full[:9], prefix)


def test_all_missing_is_zero_except_flags():
    f = np.full((3, TOTAL_POINTS, 3), np.nan, dtype=np.float32)
    x = sequence_features(f, 1.0)
    assert np.all(x == 0)


def test_hand_shape_invariant_to_translation():
    f = frames(2)
    g = f.copy()
    g[:, RIGHT_START:RIGHT_START + 21, :2] += 0.05
    a, b = sequence_features(f, 1.0), sequence_features(g, 1.0)
    hand = slice(127 * 2 + 63, 127 * 2 + 126)  # right-hand local block
    assert np.allclose(a[:, hand], b[:, hand], atol=1e-5)


def test_tokenizer_round_trip():
    assert tokenize("Hi, I'd like a LATTE!") == ["hi", ",", "i'd", "like", "a", "latte", "!"]
    assert detokenize(["hi", ",", "i'd", "like", "a", "latte", "!", "thanks"]) == "Hi, I'd like a latte! Thanks"
    v = Vocab.build(["a latte please", "a tea please"], min_freq=1)
    assert v.decode(v.encode("a latte please")[1:]) == "A latte please"
    assert v.ctc_targets("a latte, please.") == [v.stoi["a"], v.stoi["latte"], v.stoi["please"]]


def test_ctc_greedy_collapses_repeats_and_blanks():
    lp = torch.full((1, 6, 4), -10.0)
    for t, k in enumerate([0, 2, 2, 0, 2, 3]):
        lp[0, t, k] = 0
    assert ctc_greedy_decode(lp) == [[2, 2, 3]]


def test_model_trains_one_step_and_decodes():
    torch.manual_seed(0)
    m = SLTModel(ModelConfig(in_dim=FEATURE_DIM, vocab_size=30, d_model=32, enc_layers=1, dec_layers=1, heads=2, ff=64))
    x = torch.randn(2, 17, FEATURE_DIM)
    lens = torch.tensor([17, 11])
    tin = torch.tensor([[BOS_ID, 5, 6, 7], [BOS_ID, 8, 0, 0]])
    ctc_lp, ctc_lens, logits = m(x, lens, tin)
    assert ctc_lp.shape[:2] == (2, 9) and ctc_lens.tolist() == [9, 6]
    assert logits.shape == (2, 4, 30)
    logits.sum().backward()
    m.eval()
    ids, conf, _, _ = m.greedy(x, lens, BOS_ID, EOS_ID, max_len=5)
    assert len(ids) == 2 and all(0 < c <= 1 for c in conf)


def test_padding_does_not_change_outputs():
    torch.manual_seed(0)
    m = SLTModel(ModelConfig(in_dim=FEATURE_DIM, vocab_size=30, d_model=32, enc_layers=2, dec_layers=1, heads=2, ff=64)).eval()
    x = torch.randn(1, 13, FEATURE_DIM)
    padded = torch.cat([x, torch.zeros(1, 7, FEATURE_DIM)], 1)
    a, _ = m.encoder(x, torch.tensor([13]))
    b, _ = m.encoder(padded, torch.tensor([13]))
    assert torch.allclose(a, b[:, : a.shape[1]], atol=1e-5)


def test_beam_search_matches_greedy_at_width_one_and_blocks_repeats():
    import numpy as np
    from models.beam import BeamConfig, beam_search, blocked_tokens

    rng = np.random.default_rng(3)
    logits = rng.normal(size=(10, 10))
    table = logits - np.log(np.exp(logits).sum(1, keepdims=True))
    step = lambda prefixes: table[[p[-1] for p in prefixes]]
    # Width 1, no blocking = greedy (except an empty sentence is never allowed).
    greedy, last = [], 1
    for _ in range(8):
        row = table[last].copy()
        row[[0, 1]] = -1e9
        if not greedy:
            row[2] = -1e9
        nxt = int(row.argmax())
        if nxt == 2:
            break
        greedy.append(nxt)
        last = nxt
    hyps = beam_search(step, 1, 2, BeamConfig(beam=1, n_best=1, max_len=8, no_repeat_ngram=0, banned=(0, 1)))
    assert hyps[0].tokens == greedy
    # Wider beams return distinct suggestions, none repeating a trigram.
    hyps = beam_search(step, 1, 2, BeamConfig(beam=4, n_best=3, max_len=12, no_repeat_ngram=3, banned=(0, 1)))
    assert len({tuple(h.tokens) for h in hyps}) == len(hyps) > 1
    for h in hyps:
        t = h.tokens
        assert len({tuple(t[i:i + 3]) for i in range(len(t) - 2)}) == max(0, len(t) - 2)
    assert blocked_tokens([5, 6, 7, 5, 6], 3) == {7}


def test_asl_signs_import_resampling_and_split():
    from preprocessing.import_asl_signs import TEST_PARTICIPANTS, VAL_PARTICIPANTS, resample, split_of

    frames = np.arange(30, dtype=np.float32)[:, None, None] * np.ones((1, 553, 3), np.float32)
    out = resample(frames, 30.0)
    assert len(out) == 15 and out[1, 0, 0] == 2.0  # every second frame at 30 -> 15 fps
    assert not (VAL_PARTICIPANTS & TEST_PARTICIPANTS)
    assert {split_of(p) for p in VAL_PARTICIPANTS} == {"val"}
    assert split_of("26734") == "train"


def test_how2sign_holistic_import_layout_missing_parts_and_fps():
    from common.layout import FACE_START, POSE_START
    from preprocessing.import_how2sign_holistic import holistic_to_frames

    T = 48  # 2 s at 24 fps
    src = np.zeros((T, 543, 3), dtype=np.float32)
    src[:, 0:33] = 0.1      # pose
    src[:, 33:501] = 0.2    # face
    src[:, 522:543] = 0.4   # right hand only; left hand stays all zeros (not detected)
    src[5, 522:543] = 0.0   # right hand lost in one frame
    out = holistic_to_frames(src)
    assert out.shape == (30, TOTAL_POINTS, 3)  # resampled to 15 fps
    assert np.allclose(out[:, POSE_START], 0.1) and np.allclose(out[:, FACE_START], 0.2)
    assert np.isnan(out[:, FACE_START + 468:]).all()  # no iris in Holistic
    assert np.isnan(out[:, LEFT_START]).all()
    assert np.isnan(out[3, RIGHT_START, 0])  # source frame 5 -> output frame 3 (nearest)
    assert np.allclose(out[0, RIGHT_START], 0.4)
