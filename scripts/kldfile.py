"""Reads llama-perplexity's --kl-divergence-base files (the "_logits_" format of tools/perplexity/perplexity.cpp).

Layout: b"_logits_", int32 n_ctx, int32 n_vocab, int32 n_chunk, int32 tokens[n_chunk * n_ctx], then per chunk
n_ctx - 1 - n_ctx // 2 scored positions, each nv = 2 * ((n_vocab + 1) // 2) + 4 uint16 values: two float32
(scale, min_log_prob) followed by n_vocab codes q, with log_prob = min_log_prob + scale * q. Codes of 0 are values
clipped at max_logit - 16 (the writer's floor), so the tail below that is not recoverable.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

MAGIC = b"_logits_"


@dataclass(frozen=True)
class Header:
    n_ctx: int
    n_vocab: int
    n_chunk: int
    tokens: np.ndarray  # int32 [n_chunk, n_ctx]
    data_offset: int

    @property
    def nv(self) -> int:
        return 2 * ((self.n_vocab + 1) // 2) + 4

    @property
    def n_scored(self) -> int:
        return self.n_ctx - 1 - self.n_ctx // 2


def read_header(path: str | Path) -> Header:
    with open(path, "rb") as f:
        if f.read(8) != MAGIC:
            raise ValueError(f"{path}: not a llama-perplexity logits file")
        n_ctx, n_vocab, n_chunk = np.frombuffer(f.read(12), dtype="<i4")
        tokens = np.frombuffer(f.read(4 * n_chunk * n_ctx), dtype="<i4").reshape(n_chunk, n_ctx)
        h = Header(int(n_ctx), int(n_vocab), int(n_chunk), tokens, f.tell())
    expected = h.data_offset + h.n_chunk * h.n_scored * h.nv * 2
    size = Path(path).stat().st_size
    if size != expected:
        raise ValueError(f"{path}: {size} bytes, expected {expected}")
    return h


def iter_chunks(path: str | Path) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
    """Yields (chunk index, int32 next-token ids [n_scored], float32 log-probs [n_scored, n_vocab])."""
    h = read_header(path)
    raw = np.memmap(path, dtype="<u2", mode="r", offset=h.data_offset, shape=(h.n_chunk, h.n_scored, h.nv))
    first = h.n_ctx // 2
    for c in range(h.n_chunk):
        block = np.asarray(raw[c])
        params = np.ascontiguousarray(block[:, :4]).view("<f4")  # [n_scored, 2]: scale, min_log_prob
        codes = block[:, 4 : 4 + h.n_vocab].astype(np.float32)
        logp = params[:, 1:2] + params[:, 0:1] * codes
        yield c, h.tokens[c, first + 1 : first + 1 + h.n_scored], logp


def same_tokens(*paths: str | Path) -> bool:
    hs = [read_header(p) for p in paths]
    return all(h.n_ctx == hs[0].n_ctx and h.n_vocab == hs[0].n_vocab and np.array_equal(h.tokens, hs[0].tokens) for h in hs)


def kl_per_chunk(base: str | Path, other: str | Path) -> np.ndarray:
    """Mean KL(base || other) per chunk from two logits files over the same tokens, float64 [n_chunk]."""
    if not same_tokens(base, other):
        raise ValueError("files were scored on different tokens")
    out = []
    for (_, _, lp), (_, _, lq) in zip(iter_chunks(base), iter_chunks(other)):
        p = np.exp(lp.astype(np.float64))
        p /= p.sum(axis=1, keepdims=True)
        lq64 = lq.astype(np.float64)
        lq64 -= np.log(np.exp(lq64).sum(axis=1, keepdims=True))
        lp64 = np.log(p)
        out.append(float(np.mean(np.sum(p * (lp64 - lq64), axis=1))))
    return np.array(out)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: kldfile.py BASE.kld OTHER.kld   (prints mean KL per chunk and overall)")
    per = kl_per_chunk(sys.argv[1], sys.argv[2])
    print(" ".join(f"{v:.6f}" for v in per))
    print(f"mean KL {per.mean():.6f}")
