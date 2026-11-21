"""analysis/noise.py: the paired chunk bootstrap of KL ratios."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "analysis"))
import noise  # noqa: E402


def test_ratio_is_paired_ratio_of_one_condition() -> None:
    rng = np.random.default_rng(0)
    a, b = rng.uniform(0.01, 0.02, 16), rng.uniform(0.01, 0.02, 16)
    r, p = noise.ratio(a, b), noise.paired_ratio([a], [b])
    assert r["ratio"] == p["ratio"] and r["ci95"] == p["ci95"]


def test_sum_of_conditions_and_scale() -> None:
    rng = np.random.default_rng(1)
    s1, s2, s3 = (rng.uniform(0.01, 0.02, 16) for _ in range(3))
    total = s1 + s2 + s3
    out = noise.paired_ratio([total], [s1, s2, s3])
    assert abs(out["ratio"] - 1) < 1e-12 and np.allclose(out["ci95"], 1) and np.allclose(out["chunk_range"], 1)
    mean = noise.paired_ratio([total], [s1, s2, s3], scale=3.0)
    assert abs(mean["ratio"] - 3) < 1e-12
    # the bootstrap is paired: a ratio that is constant per chunk has a zero-width CI
    paired = noise.paired_ratio([2 * s1], [s1])
    assert np.allclose(paired["ci95"], 2)
    assert paired["chunk_range"] == (2.0, 2.0)
