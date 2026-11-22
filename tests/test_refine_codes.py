"""refine-codes.py: IQ2_S / IQ3_XXS decode and group encode against gguf-py, and the Gram-objective search."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

import gguf
from gguf import GGMLQuantizationType as Q
from gguf.quants import dequantize

ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location("refine_codes", ROOT / "scripts" / "refine-codes.py")
rc = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(rc)

V2A = ROOT / "models" / "v2a.gguf"
REFINED = ROOT / "models" / "refined" / "gu-refined-6-20.gguf"  # its directory holds the other refined files
LAYERS = {6: Q.IQ2_S, 20: Q.IQ3_XXS}
TYPES = (Q.IQ2_S, Q.IQ3_XXS)
# Accelerate's BLAS raises spurious FP flags under numpy on macOS (results are checked for finiteness)
pytestmark = pytest.mark.filterwarnings("ignore:.*encountered in matmul:RuntimeWarning")


def real_tensor(path, layer):
    if not path.exists():
        pytest.skip(f"{path} missing")
    return {t.name: t for t in gguf.GGUFReader(path).tensors}[f"blk.{layer}.ffn_gate_up_exps.weight"]


def random_raw(f, rows, nb, rng):
    """Random valid blocks: random code bytes, d = a fixed small fp16 per block (no NaN/inf), random scales."""
    raw = rng.integers(0, 256, size=(rows, nb, f.bsize), dtype=np.uint8)
    d = (rng.uniform(0.5, 2.0, size=(rows, nb)) * 1e-3).astype(np.float16)
    raw[..., :2] = d.reshape(rows, nb, 1).view(np.uint8)
    return raw.reshape(rows, nb * f.bsize)


# (1) decode equals gguf-py's dequantize bit for bit on real v2a blocks
@pytest.mark.parametrize("layer", list(LAYERS))
def test_decode_matches_gguf_py_on_v2a(layer):
    t = real_tensor(V2A, layer)
    assert t.tensor_type == LAYERS[layer]
    f = rc.FMTS[t.tensor_type]
    for e in (0, 41, 127):
        raw = np.asarray(t.data[e])
        mine = rc.decode(f, raw)
        theirs = dequantize(raw, t.tensor_type).astype(np.float32).reshape(mine.shape)
        assert mine.shape == (2304, 2304)
        np.testing.assert_array_equal(mine.view(np.uint32), theirs.view(np.uint32))


@pytest.mark.parametrize("qt", TYPES)
def test_decode_matches_gguf_py_on_random_blocks(qt):
    f = rc.FMTS[qt]
    raw = random_raw(f, 16, 3, np.random.default_rng(0))
    mine = rc.decode(f, raw)
    np.testing.assert_array_equal(mine.view(np.uint32), dequantize(raw, qt).reshape(mine.shape).view(np.uint32))


# (2) re-encoding the unchanged codes reproduces the original bytes exactly
@pytest.mark.parametrize("layer", list(LAYERS))
def test_reencode_unchanged_is_identity_on_v2a(layer):
    t = real_tensor(V2A, layer)
    f = rc.FMTS[t.tensor_type]
    raw = np.array(t.data[5])
    _, idx, neg = rc.fields(f, raw)
    # scramble every code bit first, so the test cannot pass by leaving bytes alone
    scrambled = raw & rc.fixed_bits_mask(f, raw.shape[1])
    if f.qtype == Q.IQ3_XXS:
        # IQ3_XXS: set_group keeps the partner half of a sign word, so the partner must keep its parity: complement
        # all 8 sign bits of every word (each 4-bit half keeps its parity, every bit changes)
        b = raw.reshape(raw.shape[0], -1, f.bsize)
        aux = rc._aux(b)
        new = aux & np.uint32(0xF0000000)
        for l in range(4):
            byte = ~rc.KSIGNS[(aux >> np.uint32(7 * l)) & 127]
            new |= (byte.astype(np.uint32) & 127) << np.uint32(7 * l)
        s = scrambled.reshape(b.shape)
        for i in range(4):
            s[..., 66 + i:98:4] = ((new >> np.uint32(8 * i)) & 0xFF).astype(np.uint8)
        assert not np.array_equal(scrambled, raw)
    rows = np.arange(raw.shape[0])
    for gi in range(idx.shape[1]):
        rc.set_group(f, scrambled, rows, gi, idx[:, gi], neg[:, gi])
    np.testing.assert_array_equal(scrambled, raw)


@pytest.mark.parametrize("qt", TYPES)
def test_set_group_writes_only_its_group(qt):
    f = rc.FMTS[qt]
    rng = np.random.default_rng(1)
    raw = random_raw(f, 8, 2, rng)
    db0, idx0, neg0 = rc.fields(f, raw)
    rows = np.array([1, 4, 6])
    gi = 37
    new_idx = rng.integers(0, f.grid.shape[0], size=len(rows))
    new_neg = rng.integers(0, 2, size=(len(rows), f.k)).astype(bool)
    if qt == Q.IQ3_XXS:  # keep the sign word even: match the partner group's parity
        par = neg0[rows, gi ^ 1].sum(axis=1) % 2
        bad = new_neg.sum(axis=1) % 2 != par
        new_neg[bad, 0] = ~new_neg[bad, 0]
    out = raw.copy()
    rc.set_group(f, out, rows, gi, new_idx, new_neg)
    db1, idx1, neg1 = rc.fields(f, out)
    np.testing.assert_array_equal(db1, db0)
    np.testing.assert_array_equal(idx1[rows, gi], new_idx)
    np.testing.assert_array_equal(neg1[rows, gi], new_neg)
    keep = np.ones(idx0.shape, dtype=bool)
    keep[rows, gi] = False
    np.testing.assert_array_equal(idx1[keep], idx0[keep])
    np.testing.assert_array_equal(neg1[keep], neg0[keep])
    if qt == Q.IQ3_XXS:
        with pytest.raises(ValueError):
            odd = neg0[rows[:1], gi].copy()
            odd[0, 0] = ~odd[0, 0]
            rc.set_group(f, out.copy(), rows[:1], gi, idx0[rows[:1], gi], odd)


# (3) synthetic Gram with strongly correlated inputs: the objective decreases monotonically, output is valid
def correlated_gram(cols, n, rng, rho=0.97):
    z = rng.standard_normal((n, 4))
    x = np.sqrt(rho) * z[:, rng.integers(0, 4, size=cols)] + np.sqrt(1 - rho) * rng.standard_normal((n, cols))
    x += 0.5  # a mean, as in the captured Grams
    return x.T @ x


@pytest.mark.parametrize("guard", [False, True])
@pytest.mark.parametrize("shrink", [0.0, 0.1])
@pytest.mark.parametrize("qt", TYPES)
def test_refine_decreases_objective_and_stays_valid(qt, shrink, guard):
    f = rc.FMTS[qt]
    rng = np.random.default_rng(2)
    rows, nb = 12, 2
    raw = random_raw(f, rows, nb, rng)
    dq = rc.decode(f, raw)
    w = (dq + 0.004 * rng.standard_normal(dq.shape)).astype(np.float32)
    g = correlated_gram(dq.shape[1], 2000, rng)
    e0 = dq.astype(np.float64) - w
    j_start = float(np.sum(e0 * (e0 @ g)))
    h = g + shrink * np.diag(np.diag(g))  # amendment 9b's objective matrix (shrink 0: amendment 9)
    out, dq1, sw = rc.refine(f, raw, w, h, max_sweeps=4, tol=0.0, g_report=g, gain_guard=guard)
    u = w.astype(np.float64) @ h  # amendment 9c: no row's output gain under h falls below its start
    gain0 = np.einsum("ri,ri->r", dq.astype(np.float64), u) / np.einsum("ri,ri->r", w.astype(np.float64), u)
    gain1 = np.einsum("ri,ri->r", dq1.astype(np.float64), u) / np.einsum("ri,ri->r", w.astype(np.float64), u)
    if guard:
        assert np.all(gain1 >= gain0 - 1e-12)
        assert sw[-1]["min_row_ogain_change"] >= -1e-12
    js = [sw[0]["J_before"]] + [s["J_after"] for s in sw]
    assert sw[0]["JG_before"] == pytest.approx(j_start, rel=1e-9)
    assert js[0] == pytest.approx(float(np.sum(e0 * (e0 @ h))), rel=1e-9)
    assert all(b <= a for a, b in zip(js, js[1:]))
    assert js[-1] < 0.9 * js[0]
    for s in sw:  # the summed exact dJ of accepted moves equals the recomputed objective
        assert s["J_tracked"] == pytest.approx(s["J_after"], rel=1e-8)
    rc.validate(f, out, dq1)
    m = rc.fixed_bits_mask(f, raw.shape[1])
    np.testing.assert_array_equal(out & m, raw & m)


@pytest.mark.parametrize("guarded", [False, True])
@pytest.mark.parametrize("qt", TYPES)
def test_ranked_winner_is_the_exact_best_candidate(qt, guarded):
    """The BLAS ranking picks the candidate with the smallest exact dJ among the documented candidate set, and with
    the output-gain guard among the candidates whose numerator change d . uw stays >= -slack."""
    f = rc.FMTS[qt]
    rng = np.random.default_rng(3)
    raw = random_raw(f, 40, 1, rng)
    db, idx, neg = rc.fields(f, raw)
    dq = rc.values(f, db, idx, neg).astype(np.float64)
    w = dq + 0.004 * rng.standard_normal(dq.shape)
    g = correlated_gram(dq.shape[1], 500, rng)
    E = dq - w
    R = E @ g
    gi = 3
    c = slice(gi * f.k, gi * f.k + f.k)
    Gb, Rg, cur = g[c, c], R[:, c], dq[:, c]
    uw = (w @ g)[:, c]
    slack = np.zeros(len(cur))
    thr = np.einsum("ri,ri->r", cur, uw) - slack if guarded else None
    if qt == Q.IQ2_S:
        ci, cn = rc._pick_iq2s(f, Gb, Rg, cur, db[:, gi], neg[:, gi], 1e-6 * np.mean(np.diag(g)), uw, thr)
    else:
        ci, cn = rc._pick_iq3xxs(f, Gb, Rg, cur, db[:, gi], neg[:, gi ^ 1], uw, thr)
    checked = 0
    for r in range(dq.shape[0]):
        if qt == Q.IQ2_S:
            pats = cn[r][None]
        else:
            par = neg[r, gi ^ 1].sum() % 2
            pats = rc._PATS_BY_PAR[par]
        best = np.inf
        for p in pats:
            v = (db[r, gi] * f.grid * np.where(p, np.float32(-1), np.float32(1))).astype(np.float64)
            d = v - cur[r]
            dj = 2 * d @ Rg[r] + np.einsum("ci,ij,cj->c", d, Gb, d)
            if guarded:
                dj = np.where(d @ uw[r] >= -slack[r] - 1e-15, dj, np.inf)
            best = min(best, dj.min())
        if not best < 0:
            continue  # no feasible improving candidate (the unchanged code, dJ = 0, wins): the exact check rejects
        v = (db[r, gi] * f.grid[ci[r]] * np.where(cn[r], np.float32(-1), np.float32(1))).astype(np.float64)
        d = v - cur[r]
        assert 2 * d @ Rg[r] + d @ Gb @ d == pytest.approx(best, rel=1e-6, abs=1e-12 * abs(best) + 1e-18)
        checked += 1
    assert checked >= 3


# (4) every output block decodes to values in the representable set
@pytest.mark.parametrize("name,layers,grams", [
    ("gu-refined-6-20.gguf", {6: Q.IQ2_S, 20: Q.IQ3_XXS}, "gu-6-20"),
    ("gu-refined-6-20-9b.gguf", {6: Q.IQ2_S, 20: Q.IQ3_XXS}, "gu-6-20"),
    ("gu-refined-9-10-11.gguf", {9: Q.IQ3_XXS, 10: Q.IQ3_XXS, 11: Q.IQ3_XXS}, "gu-9-10-11"),
])
def test_refined_output_is_representable(name, layers, grams):
    """Every expert of a refined file decodes (gguf-py == ours) to representable codes; d and scale bits equal v2a's;
    the file holds exactly the refined gate_up tensors; experts under amendment 9b's 2304-token floor keep v2a's
    bytes exactly."""
    path = REFINED.parent / name
    if not path.exists():
        pytest.skip(f"{path} missing")
    assert sorted(t.name for t in gguf.GGUFReader(path).tensors) == sorted(
        f"blk.{l}.ffn_gate_up_exps.weight" for l in layers)
    for layer, qt in layers.items():
        t = real_tensor(path, layer)
        assert t.tensor_type == qt and [int(x) for x in t.shape] == [2304, 2304, 128]
        f = rc.FMTS[qt]
        o = real_tensor(V2A, layer)
        m = rc.fixed_bits_mask(f, t.data.shape[2])
        counts = np.fromfile(ROOT / "models" / "grams" / grams / f"blk.{layer}.ffn_gate_up_exps.weight.gram.counts.i64",
                             dtype=np.int64)
        for e in range(128):
            raw = np.asarray(t.data[e])
            rc.validate(f, raw)
            np.testing.assert_array_equal(raw & m, np.asarray(o.data[e]) & m)
            if counts[e] < 2304:
                np.testing.assert_array_equal(raw, np.asarray(o.data[e]))


def test_validate_rejects_bad_values():
    f = rc.FMTS[Q.IQ2_S]
    raw = random_raw(f, 2, 1, np.random.default_rng(4))
    rc.validate(f, raw)
    with pytest.raises(AssertionError):
        rc.validate(f, raw, rc.decode(f, raw) * np.float32(1.5))


# (5) Astra review fixes (reviews/2026-11-22-refiner-astra.md), each test written to fail before its fix
def iq2s_row(db_d, idx, neg):
    """One IQ2_S row of len(idx) // 32 blocks: fp16 d = db_d (so db = d / 8 with all sub-block scales 0), group gi
    gets grid index idx[gi] and signs neg[gi] (8 bools)."""
    f = rc.FMTS[Q.IQ2_S]
    nb = len(idx) // f.gpb
    raw = np.zeros((1, nb, f.bsize), dtype=np.uint8)
    raw[..., :2] = np.array([db_d], dtype=np.float16).view(np.uint8)
    raw = raw.reshape(1, -1)
    for gi in range(len(idx)):
        rc.set_group(f, raw, np.array([0]), gi, np.array([idx[gi]]), np.asarray(neg[gi], dtype=bool)[None])
    return raw


def test_iq2s_searches_signs_astra_point1():
    """Astra point 1: db = 1, grid 0 (all 8s), q[0:2] = (8, 8), w[0:2] = (1, 1), q[8] = w[8] = -8, rest exact; G =
    identity with the {0, 1, 8} block all ones. Flipping only q[0]'s sign lowers the regularised objective from
    205.8 to 17 and raises the gain numerator by 94.4; signs fixed from the continuous target cannot find it."""
    f = rc.FMTS[Q.IQ2_S]
    neg = np.zeros((32, 8), dtype=bool)
    neg[1, 0] = True  # q[8] = -8
    raw = iq2s_row(8.0, np.zeros(32, dtype=np.int64), neg)
    q = rc.decode(f, raw)
    assert q[0, 0] == 8 and q[0, 1] == 8 and q[0, 8] == -8 and np.all(np.abs(q) == 8)
    w = q.copy()
    w[0, :2] = 1
    g = np.eye(256)
    g[np.ix_([0, 1, 8], [0, 1, 8])] = 1.0
    h = g + 0.1 * np.diag(np.diag(g))
    e = q.astype(np.float64) - w
    assert float(np.sum(e * (e @ h))) == pytest.approx(205.8)
    out, dq1, sw = rc.refine(f, raw, w, h, max_sweeps=4, tol=0.0, g_report=g, gain_guard=True)
    assert sw[-1]["J_after"] <= 17 + 1e-9
    assert sw[-1]["min_row_ogain_change"] >= 0
    rc.validate(f, out, dq1)


def test_zero_unregularised_denominator_is_null_astra_point2(monkeypatch):
    """Astra point 2: 2304 identical tokens (1, 1, 0, ...) and BF16 rows (1, -1, 0, ...): the count passes, tr(G) > 0,
    tr(W H W^T) > 0, but tr(W G W^T) = 0. The undefined ratios are reported as null instead of raising."""
    f = rc.FMTS[Q.IQ2_S]
    raw = random_raw(f, 2, 1, np.random.default_rng(5))
    w = np.zeros((2, 256), dtype=np.float32)
    w[:, 0], w[:, 1] = 1, -1
    x = np.zeros(256)
    x[:2] = 1
    u = np.triu(2304 * np.outer(x, x)).astype(np.float32)[None]
    n = "blk.7.ffn_gate_up_exps.weight"

    class T:
        tensor_type = Q.IQ2_S
        data = raw[None]

    monkeypatch.setattr(rc, "_W", {"q": {n: T}, "b": {n: T}})
    monkeypatch.setattr(rc.ge, "load_gram", lambda l: (u, np.array([2304])))
    monkeypatch.setattr(rc.ge, "deq", lambda t, e: w)
    layer, e, raw1, rec = rc._one((7, 0, 2, 1e-4, 1e-6, 0.1, 2304, True))
    assert rec["tr_WGWt"] == 0 and rec["tr_WHWt"] > 0
    assert rec["sweeps"] and all(s["rel_before"] is None and s["rel_after"] is None for s in rec["sweeps"])
    assert all(np.isfinite(s["relreg_before"]) for s in rec["sweeps"])
    assert "rel n/a" in rc._msg(rec)
    rc.validate(f, raw1)


def test_gain_floor_is_a_change_from_baseline_astra_point3():
    """Astra point 3: w all 8s, q equal except q[0] = 43, diagonal G with G[0,0] = 2^-64 and 10 elsewhere. Moving 43
    to 8 removes the error but lowers the gain numerator by 308 * 2^-64 (times 1.1 under H), which vanishes when added
    to the full numerator. Tracked as a change from 0, the move is rejected."""
    f = rc.FMTS[Q.IQ2_S]
    idx = np.zeros(32, dtype=np.int64)
    idx[0] = 1  # grid 1 = (43, 8, 8, 8, 8, 8, 8, 8)
    raw = iq2s_row(8.0, idx, np.zeros((32, 8), dtype=bool))
    q = rc.decode(f, raw)
    assert q[0, 0] == 43 and np.all(q[0, 1:] == 8)
    w = np.full_like(q, 8)
    g = np.diag(np.r_[2.0 ** -64, np.full(255, 10.0)])
    h = g + 0.1 * np.diag(np.diag(g))
    out, dq1, sw = rc.refine(f, raw, w, h, max_sweeps=4, tol=0.0, g_report=g, gain_guard=True)
    np.testing.assert_array_equal(out, raw)
    out2, _, _ = rc.refine(f, raw, w, h, max_sweeps=4, tol=0.0, g_report=g, gain_guard=False)
    assert not np.array_equal(out2, raw)  # without the floor the move is taken (the test exercises the floor)


@pytest.mark.parametrize("guarded", [False, True])
def test_iq2s_ranked_winner_over_all_documented_sign_patterns(guarded):
    """The IQ2_S ranking returns the exact best over every documented sign pattern x all 1024 magnitudes, and the sign
    polish ends in a point no single feasible sign flip improves, never worse than the ranked winner."""
    f = rc.FMTS[Q.IQ2_S]
    rng = np.random.default_rng(6)
    raw = random_raw(f, 40, 1, rng)
    db, idx, neg = rc.fields(f, raw)
    dq = rc.values(f, db, idx, neg).astype(np.float64)
    w = dq + 0.004 * rng.standard_normal(dq.shape)
    g = correlated_gram(dq.shape[1], 500, rng)
    R = (dq - w) @ g
    gi = 5
    c = slice(gi * f.k, gi * f.k + f.k)
    Gb, Rg, cur = g[c, c], R[:, c], dq[:, c]
    uw = (w @ g)[:, c]
    thr = np.einsum("ri,ri->r", cur, uw) if guarded else None
    ridge = 1e-6 * np.mean(np.diag(g))
    pats = rc._iq2s_sign_patterns(Gb, Rg, cur, db[:, gi], neg[:, gi], ridge)
    assert len(pats) == 3
    ci, cn = rc._pick_iq2s(f, Gb, Rg, cur, db[:, gi], neg[:, gi], ridge, uw, thr)

    def dj(r, gidx, ng):
        v = (db[r, gi] * f.grid[gidx] * np.where(ng, np.float32(-1), np.float32(1))).astype(np.float64)
        d = v - cur[r]
        return 2 * d @ Rg[r] + d @ Gb @ d, d @ uw[r]

    checked = 0
    for r in range(dq.shape[0]):
        best = np.inf
        for p in pats:
            v = (db[r, gi] * f.grid * np.where(p[r], np.float32(-1), np.float32(1))).astype(np.float64)
            d = v - cur[r]
            s = 2 * d @ Rg[r] + np.einsum("ci,ij,cj->c", d, Gb, d)
            if guarded:
                s = np.where(d @ uw[r] >= -1e-15 * abs(thr[r]) - 1e-18, s, np.inf)
            best = min(best, s.min())
        if best < 0:
            assert dj(r, ci[r], cn[r])[0] == pytest.approx(best, rel=1e-6, abs=1e-12 * abs(best) + 1e-18)
            checked += 1
    assert checked >= 3
    pn = rc._polish_iq2s(f, Gb, Rg, cur, db[:, gi], ci, cn, uw, thr)
    improved = 0
    for r in range(dq.shape[0]):
        d0, d1 = dj(r, ci[r], cn[r])[0], dj(r, ci[r], pn[r])
        assert d1[0] <= d0 + 1e-12 * abs(d0) + 1e-18
        improved += d1[0] < d0 - 1e-9 * abs(d0)
        for i in range(f.k):
            fl = pn[r].copy()
            fl[i] = ~fl[i]
            df = dj(r, ci[r], fl)
            if not guarded or df[1] >= 0:
                assert df[0] >= d1[0] - 1e-9 * abs(d1[0]) - 1e-18
    assert improved >= 1
