import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "analysis"))
from battery_A import ratio_ci, verdict  # noqa: E402


def test_constant_ratio_gives_that_ratio_with_a_tight_ci():
    rng = np.random.default_rng(1)
    base = rng.uniform(0.01, 0.05, 32)
    r, lo, hi = ratio_ci({"a": (0.8 * base, base), "b": (0.8 * base, base)}, n_boot=500)
    assert abs(r - 0.8) < 1e-12 and abs(lo - 0.8) < 1e-12 and abs(hi - 0.8) < 1e-12
    assert verdict(lo, hi).startswith("PASS")


def test_mean_over_domains_and_regression_verdict():
    base = np.full(32, 0.02)
    r, lo, hi = ratio_ci({"a": (1.2 * base, base), "b": (1.4 * base, base)}, n_boot=200)
    assert abs(r - 1.3) < 1e-12 and verdict(lo, hi) == "REGRESSION"
    assert verdict(0.85, 0.95) == "PARTIAL transfer" and verdict(0.9, 1.05) == "no evidence of transfer"
