"""Behaviour the per-pass residual must show on synthetic logits files. Skips until analysis/per_pass.py is written."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "tests"))
import per_pass  # noqa: E402
from test_kldfile import write_file  # noqa: E402

N_CHUNK, N_CTX, N_VOCAB = 4, 32, 64
N_SCORED = N_CTX - 1 - N_CTX // 2


def make(tmp_path: Path, deltas: dict[str, np.ndarray], combined: np.ndarray) -> tuple[Path, list[Path], Path]:
    rng = np.random.default_rng(0)
    tokens = rng.integers(0, N_VOCAB, size=(N_CHUNK, N_CTX))
    base = rng.normal(0, 1.5, size=(N_CHUNK, N_SCORED, N_VOCAB)).astype(np.float32)
    write_file(tmp_path / "base.kld", base, tokens)
    singles = []
    for k in ("p1", "p2", "p3"):
        write_file(tmp_path / f"{k}.kld", (base + deltas[k]).astype(np.float32), tokens)
        singles.append(tmp_path / f"{k}.kld")
    write_file(tmp_path / "all.kld", (base + combined).astype(np.float32), tokens)
    return tmp_path / "base.kld", singles, tmp_path / "all.kld"


def deltas(scale: tuple[float, float, float]) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(1)
    return {k: s * rng.normal(0, 0.05, size=(N_CHUNK, N_SCORED, N_VOCAB)) for k, s in zip(("p1", "p2", "p3"), scale)}


def call(*args):
    try:
        return per_pass.residual_ratio(*args)
    except NotImplementedError:
        pytest.skip("analysis/per_pass.py not written yet")


def test_additive_gives_zero_residual(tmp_path: Path) -> None:
    d = deltas((1, 1, 1))
    out = call(*make(tmp_path, d, d["p1"] + d["p2"] + d["p3"]))
    assert out["r"] < 0.02
    assert out["r_chunks"].shape == (N_CHUNK,)


def test_last_pass_only_counts_once(tmp_path: Path) -> None:
    d = deltas((1, 1, 1))
    out = call(*make(tmp_path, d, d["p3"]))  # all passes perturbed, but only pass 3 reaches the output
    assert 0.9 < out["share_last"] < 1.1
    assert out["r"] > 0.5  # the single-pass effects of passes 1 and 2 are missing from "all"


def test_compounding_exceeds_the_sum(tmp_path: Path) -> None:
    d = deltas((1, 1, 1))
    total = d["p1"] + d["p2"] + d["p3"]
    out = call(*make(tmp_path, d, 2.0 * total))
    assert out["r"] > 0.4


def test_shift_invariance(tmp_path: Path) -> None:
    # adding a per-token constant to the logits changes nothing in the distribution, so it must not change r
    d = deltas((1, 1, 1))
    shifted = {k: v + 3.0 for k, v in d.items()}
    out = call(*make(tmp_path, shifted, d["p1"] + d["p2"] + d["p3"] - 7.0))
    assert out["r"] < 0.02


def test_fisher_norm_matches_kl(tmp_path: Path) -> None:
    # for small changes KL(base || S) ~ 1/2 ||Delta_S||_p^2; check against the independent KL code in kldfile
    import kldfile

    d = {k: v * 0.2 for k, v in deltas((1, 1, 1)).items()}
    base, singles, combined = make(tmp_path, d, d["p1"] + d["p2"] + d["p3"])
    acc = per_pass.accumulate(base, singles, combined)
    n_pos = N_CHUNK * N_SCORED
    half_fisher = 0.5 * acc[:, per_pass.TERMS.index("d3"), 0].sum() / n_pos
    kl = kldfile.kl_per_chunk(base, singles[2]).mean()
    assert abs(half_fisher - kl) / kl < 0.05
