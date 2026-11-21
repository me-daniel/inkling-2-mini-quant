"""Round-trips the llama-perplexity logits format with a Python copy of the C++ writer (perplexity.cpp, log_softmax)."""

import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import kldfile  # noqa: E402


def encode(logits: np.ndarray) -> np.ndarray:
    """One position, as perplexity.cpp writes it: [scale, min_log_prob] as float32, then uint16 codes."""
    n_vocab = logits.size
    max_l, min_l = float(logits.max()), float(logits.min())
    min_l = max(min_l, max_l - 16)
    lse = float(np.log(np.exp(logits.astype(np.float64) - max_l).sum()))
    min_log_prob = np.float32(min_l - max_l - lse)
    scale = np.float32((max_l - min_l) / 65535.0)
    nv = 2 * ((n_vocab + 1) // 2) + 4
    out = np.zeros(nv, dtype="<u2")
    out[:4] = np.array([scale, min_log_prob], dtype="<f4").view("<u2")
    if scale:
        q = np.where(logits > min_l, np.rint((logits - min_l) / scale), 0)
        out[4 : 4 + n_vocab] = q.astype("<u2")
    return out


def write_file(path: Path, logits: np.ndarray, tokens: np.ndarray) -> None:
    n_chunk, n_scored, n_vocab = logits.shape
    n_ctx = tokens.shape[1]
    assert n_scored == n_ctx - 1 - n_ctx // 2
    with open(path, "wb") as f:
        f.write(b"_logits_" + struct.pack("<iii", n_ctx, n_vocab, n_chunk))
        f.write(tokens.astype("<i4").tobytes())
        for c in range(n_chunk):
            for i in range(n_scored):
                f.write(encode(logits[c, i]).tobytes())


def test_roundtrip_and_kl(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    n_chunk, n_ctx, n_vocab = 3, 16, 51  # odd vocab exercises the padding of nv
    n_scored = n_ctx - 1 - n_ctx // 2
    tokens = rng.integers(0, n_vocab, size=(n_chunk, n_ctx))
    a = rng.normal(0, 3, size=(n_chunk, n_scored, n_vocab)).astype(np.float32)
    b = (a + rng.normal(0, 0.3, size=a.shape)).astype(np.float32)
    pa, pb = tmp_path / "a.kld", tmp_path / "b.kld"
    write_file(pa, a, tokens)
    write_file(pb, b, tokens)

    h = kldfile.read_header(pa)
    assert (h.n_ctx, h.n_vocab, h.n_chunk, h.n_scored) == (n_ctx, n_vocab, n_chunk, n_scored)

    for c, nxt, lp in kldfile.iter_chunks(pa):
        assert np.array_equal(nxt, tokens[c, n_ctx // 2 + 1 :])
        exact = a[c] - np.log(np.exp(a[c].astype(np.float64)).sum(axis=1, keepdims=True))
        # values more than 16 below the top logit are clipped to the floor by the writer
        kept = a[c] > a[c].max(axis=1, keepdims=True) - 16
        assert np.max(np.abs(lp - exact)[kept]) < 16 / 65535 + 1e-5  # one code step
        assert np.all(lp[~kept] >= exact[~kept] - 1e-5)  # clipped values sit at the floor, above the truth

    def kl(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        lx = x - np.log(np.exp(x.astype(np.float64)).sum(axis=-1, keepdims=True))
        ly = y - np.log(np.exp(y.astype(np.float64)).sum(axis=-1, keepdims=True))
        return np.sum(np.exp(lx) * (lx - ly), axis=-1).mean(axis=-1)

    # KL check on logits narrow enough that nothing is clipped
    a, b = a / 2, b / 2
    write_file(pa, a, tokens)
    write_file(pb, b, tokens)
    got, want = kldfile.kl_per_chunk(pa, pb), kl(a, b)
    assert np.allclose(got, want, rtol=1e-3, atol=1e-6)
    assert np.allclose(kldfile.kl_per_chunk(pa, pa), 0.0, atol=1e-9)


def test_rejects_other_tokens(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    a = rng.normal(size=(1, 7, 9)).astype(np.float32)
    t1 = rng.integers(0, 9, size=(1, 16))
    t2 = t1.copy()
    t2[0, 3] = (t2[0, 3] + 1) % 9
    write_file(tmp_path / "a.kld", a, t1)
    write_file(tmp_path / "b.kld", a, t2)
    try:
        kldfile.kl_per_chunk(tmp_path / "a.kld", tmp_path / "b.kld")
    except ValueError:
        return
    raise AssertionError("different tokens were accepted")
