"""gram-error.py on synthetic data: with strongly correlated inputs the full-Gram error can rank two encodings the
opposite way from the diagonal one, and an upper-triangle-only Gram is symmetrised to the full one."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location("gram_error", Path(__file__).resolve().parent.parent / "scripts" / "gram-error.py")
ge = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ge)

RHO = 0.99
HALVES = {"gate": slice(0, 2), "up": slice(2, 4), "both": slice(0, 4)}


def gram() -> np.ndarray:
    # two blocks of two strongly correlated input columns
    b = np.array([[1.0, RHO], [RHO, 1.0]])
    g = np.zeros((4, 4))
    g[:2, :2] = b
    g[2:, 2:] = 2 * b
    return g.astype(np.float32)


def weights() -> np.ndarray:
    return np.array([[1.0, 0.5, 0.2, -0.3],
                     [-0.4, 0.8, 0.6, 0.1],
                     [0.3, -0.2, 0.9, 0.7],
                     [0.5, 0.5, -0.6, 0.4]], dtype=np.float32)


def test_full_and_diag_rank_differently():
    w, g = weights(), gram()
    # A: errors along (1, -1) inside each correlated pair, the direction the Gram barely sees (eigenvalue 1 - rho)
    da = 0.1 * np.array([[1, -1, 1, -1], [-1, 1, 1, -1], [1, -1, -1, 1], [1, -1, 1, -1]], dtype=np.float32)
    # B: smaller errors on one column only, which the Gram sees at full weight
    db = 0.04 * np.array([[1, 0, 1, 0], [1, 0, 0, 1], [0, 1, 1, 0], [1, 0, 0, 1]], dtype=np.float32)
    ra, rb = ge.rel_errors(w, w + da, g, HALVES), ge.rel_errors(w, w + db, g, HALVES)
    for h in HALVES:
        assert ra["full"][h] < rb["full"][h]
        assert ra["diag"][h] > rb["diag"][h]
        assert ra["plain"][h] > rb["plain"][h]
    # closed form for 'both' against a float64 reference
    g64, w64 = g.astype(np.float64), w.astype(np.float64)
    full = np.trace(da @ g64 @ da.T) / np.trace(w64 @ g64 @ w64.T)
    diag = np.trace(da @ np.diag(np.diag(g64)) @ da.T) / np.trace(w64 @ np.diag(np.diag(g64)) @ w64.T)
    assert ra["full"]["both"] == pytest.approx(full, rel=1e-5)
    assert ra["diag"]["both"] == pytest.approx(diag, rel=1e-5)
    assert ra["plain"]["both"] == pytest.approx(np.linalg.norm(da) / np.linalg.norm(w64), rel=1e-6)


def test_halves_are_ratios_of_their_own_sums():
    w, g = weights(), gram()
    d = np.zeros_like(w)
    d[0] = 0.1  # error only in a gate row
    r = ge.rel_errors(w, w + d, g, HALVES)
    assert r["full"]["up"] == 0 and r["full"]["gate"] > r["full"]["both"] > 0


def test_symmetrise_upper_triangle():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((50, 6)).astype(np.float32)
    x[:, 1] = x[:, 0] + 0.01 * x[:, 1]
    g = x.T @ x
    u = np.triu(g)
    s = ge.symmetrise(u)
    np.testing.assert_array_equal(s, s.T)
    np.testing.assert_allclose(s, g, rtol=1e-6, atol=1e-5)
    np.testing.assert_array_equal(np.diag(s), np.diag(u))  # diagonal counted once
    # garbage below the diagonal is ignored, as the capture leaves it at 0 anyway
    np.testing.assert_array_equal(ge.symmetrise(u + np.tril(np.ones_like(u), -1)), s)
    w = rng.standard_normal((4, 6)).astype(np.float32)
    dq = w + 0.05 * rng.standard_normal((4, 6)).astype(np.float32)
    h = {"both": slice(0, 4)}
    a, b = ge.rel_errors(w, dq, s, h), ge.rel_errors(w, dq, g, h)
    for m in ge.METRICS:
        assert a[m]["both"] == pytest.approx(b[m]["both"], rel=1e-5)
    # using U alone (not symmetrised) gives a different, wrong full error
    assert ge.rel_errors(w, dq, u, h)["full"]["both"] != pytest.approx(b["full"]["both"], rel=1e-3)


def test_aggregate_and_bootstrap():
    o = np.array([1.0, 2.0, 3.0, np.nan])
    u = np.array([0.5, 2.5, 2.0, np.nan])
    c = np.array([1.0, 1.0, 2.0, 0.0])
    a = ge.aggregate(o, c, c)
    assert a["unweighted"] == pytest.approx(2.0)
    assert a["count-weighted"] == pytest.approx((1 + 2 + 6) / 4)
    p, lo, hi = ge.boot_diff(u, o, c, n=2000, seed=0)
    assert p == pytest.approx((-0.5 + 0.5 - 2.0) / 4)
    assert lo <= p <= hi
    assert ge.boot_diff(u, o, c, n=2000, seed=0) == (p, lo, hi)


def test_bias_rest_split():
    rng = np.random.default_rng(1)
    n = 200
    x = rng.standard_normal((n, 6)) + np.array([3.0, -2.0, 0.5, 0.0, 1.0, -1.0])  # inputs with a large mean
    g = (x.T @ x).astype(np.float32)
    m = x.mean(axis=0)
    w = rng.standard_normal((4, 6)).astype(np.float32)
    dq = w + 0.05 * rng.standard_normal((4, 6)).astype(np.float32)
    r = ge.rel_errors(w, dq, g, HALVES, mean=m, count=n)
    d, w64 = (dq - w).astype(np.float64), w.astype(np.float64)
    c = (x - m).T @ (x - m)  # centred scatter: G = C + n m m^T
    for h, s in HALVES.items():
        den = np.trace(w64[s] @ (x.T @ x) @ w64[s].T)
        assert r["bias"][h] == pytest.approx(n * np.sum((d[s] @ m) ** 2) / den, rel=1e-6)
        assert r["rest"][h] == pytest.approx(np.trace(d[s] @ c @ d[s].T) / den, rel=1e-4)
        assert r["bias"][h] + r["rest"][h] == pytest.approx(r["full"][h], rel=1e-12)
    assert "bias" not in ge.rel_errors(w, dq, g, HALVES)
