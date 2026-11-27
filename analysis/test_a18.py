import numpy as np

from a18 import g_stat, outcome


def toy(seed=1, docs=12, scale=1.0):
    r = np.random.default_rng(seed)
    a = r.uniform(0.01, 0.05, docs * 4)
    return a, a * scale * r.uniform(0.9, 1.1, a.size)


def test_identical_files_give_one():
    a, _ = toy()
    g, lo, hi = g_stat({"x": (a, a), "y": (a, a)})
    assert g == lo == hi == 1.0


def test_symmetric_under_swap():
    p = {d: toy(seed=i, scale=1.2) for i, d in enumerate("abc")}
    g, lo, hi = g_stat(p)
    g2, lo2, hi2 = g_stat({d: (b, a) for d, (a, b) in p.items()})
    assert np.isclose(g * g2, 1) and np.isclose(lo * hi2, 1) and np.isclose(hi * lo2, 1)


def test_documents_are_resampled_whole():
    # one document 100x worse in every chunk: the CI must cover dropping or doubling that document
    a = np.full(48, 0.02)
    b = a.copy()
    b[:4] *= 100
    g, lo, hi = g_stat({"x": (b, a)})
    assert lo < g < hi and np.isclose(np.log(g), np.log(100) / 12)


def test_outcome_bands():
    assert outcome(0.95, 0.93, 0.98, "A", "B") == "A closer"
    assert outcome(0.99, 0.985, 0.995, "A", "B") == "equivalent at +-5%"  # bound < 1 but G > 0.97
    assert outcome(1.05, 1.02, 1.08, "A", "B") == "B closer"
    assert outcome(1.0, 0.90, 1.10, "A", "B") == "inconclusive"
