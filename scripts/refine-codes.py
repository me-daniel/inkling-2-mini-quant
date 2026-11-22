"""Code-level refinement of IQ2_S / IQ3_XXS gate_up experts under the full input Gram (Addendum H amendment 9).

    python scripts/refine-codes.py [--quant models/v2a.gguf] [--bf16 models/Inkling-2-Mini-BF16.gguf]
        [--grams models/grams/gu-6-20] [--layers 6 20] [--experts 0:128:1] [--workers 4]
        [--out models/refined/gu-refined-6-20.gguf]

Keeps every block's fp16 d and 4-bit sub-block scales; changes only grid indices and signs. Per expert, with
E = dequantized minus BF16 (all 2304 rows at once, float64) and the full symmetric Gram G (gram-error.py's
symmetrise of the upper triangle), the objective is J = tr(E H E^T) with H = G + shrink * Diag(G), separable over
rows; R = E H. Amendment 9b guards (defaults): shrink = 0.1, and only experts with at least --min-tokens = 2304
Gram tokens are refined (fewer tokens than inputs means a rank-deficient G); the others keep --quant's bytes.
Amendment 9c (default; --no-gain-guard drops it): a move is accepted only if dJ < 0 and the row's output gain
(q_r H w_r^T) / (w_r H w_r^T) does not fall below its llama-quantize value; candidates that would break it are left
out of the ranking, so each row takes its best feasible candidate (H w_r^T is precomputed per expert; a move
changes the numerator by d . (H w_r^T)[idx]). --shrink 0 --min-tokens 0 --no-gain-guard gives amendment 9 as first
registered. Below, "G" in the search means H.

One group = the values that share one grid index: 8 inputs for IQ2_S, 4 for IQ3_XXS. For group columns idx, all
rows at once:
  delta* = -solve(G[idx,idx] + ridge I, R[:,idx]^T)^T, ridge = 1e-6 * mean(diag G) (used for delta* only)
  IQ2_S candidates (Astra review point 1): all 1024 grid magnitudes, each with three sign patterns per row:
    (i) target: sign(current + delta*) (current sign where that is exactly 0);
    (ii) separable: weight by weight, the sign that minimises the row's exact dJ with the group's other weights held
      at their current values: dJ_i = H_ii v_i^2 + 2 v_i (R_i - H_ii cur_i) + const, v_i = db * m_i * s_i; the
      square does not depend on the sign, so s_i = -sign(db * (R_i - H_ii cur_i)) (current sign where that is 0).
      Every grid magnitude m_i is positive, so this pattern is the same for all 1024 magnitudes; IQ2_S stores the
      8 signs freely (no parity), so every pattern is representable;
    (iii) current: the code's own signs.
    The ranked winner over these 3 x 1024 candidates then gets a sign polish: single sign flips of the winner, weights
    0..7 in order, each taken if it lowers the exact dJ (and keeps the output-gain floor when guarded), repeated
    until no flip helps (at most 16 passes). The separable rule ignores the coupling H_ij between new values of the
    same group, so on its own it can flip two correlated weights together and lose; the polish then finds the
    single flip (Astra's example: 205.8 -> 17).
  IQ3_XXS candidates: all 256 grids x the 8 sign patterns whose parity keeps the 8-value sign word even (the
    partner group of the word is held fixed; ksigns stores 7 bits and sets the 8th to make the count of minus
    signs even). Exhaustive, so delta* is not needed there.
  Candidates are ranked by the expansion dJ(v) = v^T Gb v + 2 v.(R_idx - Gb cur) + const (one BLAS product);
  the winner per row is then scored exactly, dJ = 2 d.R[:,idx] + d G[idx,idx] d with d = v - cur in weight units
  (v = db * grid * sign in float32, exactly as gguf-py decodes it; float64 arithmetic, unregularised G), and
  accepted if dJ < 0. Accepted rows update E[:, idx], R += d G[idx, :] and the raw block bytes.
Output-gain floor (amendment 9c; Astra review point 3): per row the change of the numerator q_r H w_r^T from
llama-quantize's value is tracked from 0 as the sum of the accepted moves' d . (H w_r^T)[idx], and a move is taken
only if that change stays >= 0 (comparing the full numerator with its start lets float rounding hide a tiny loss).
Processing order: groups by ascending input column (block 0 group 0, 1, ..., then block 1, ...), all rows in
parallel; experts in ascending index. A sweep visits every group once; R and J are recomputed from E at each
sweep start; sweeps repeat until (J_before - J_after) / J_before < 1e-4 or 4 sweeps (J under H).

Outputs: a GGUF holding only blk.L.ffn_gate_up_exps.weight for the given layers (type, shape and byte layout as in
--quant; written to .part, then renamed) and a JSON log next to it (per layer and expert: J under H and under G,
J / tr(W H W^T) as relreg_*, J_G / tr(W G W^T) as rel_*, null where the denominator is not > 0 (Astra review point 2),
before/after each sweep, accepted moves, the tracked J from the summed accepted dJ, seconds). Every expert is
checked: the decoder equals gguf-py's dequantize bit for bit before and after, d and scale bits are unchanged,
and every group decodes to a representable code.
"""

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

import gguf
from gguf import GGMLQuantizationType as Q
from gguf import quants as gq
from gguf.quants import dequantize

_spec = importlib.util.spec_from_file_location("gram_error", Path(__file__).resolve().parent / "gram-error.py")
ge = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ge)

QK_K = 256
KSIGNS = np.frombuffer(gq.IQ2_XXS.ksigns, dtype=np.uint8)
POPC = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


class Fmt:
    def __init__(self, qtype, k, bsize, cls):
        cls.init_grid()
        self.qtype, self.k, self.bsize = qtype, k, bsize
        self.gpb = QK_K // k  # groups per block
        self.grid = np.ascontiguousarray(cls.grid.reshape(-1, k), dtype=np.float32)  # [n_grid, k], unsigned
        self.levels = np.array(cls.grid_map, dtype=np.float32)
        assert gq.GGML_QUANT_SIZES[qtype] == (QK_K, bsize)


FMTS = {Q.IQ2_S: Fmt(Q.IQ2_S, 8, 82, gq.IQ2_S), Q.IQ3_XXS: Fmt(Q.IQ3_XXS, 4, 98, gq.IQ3_XXS)}


def _aux(b: np.ndarray) -> np.ndarray:
    """IQ3_XXS scales_and_signs as uint32 [rows, nb, 8] from blocks [rows, nb, 98] (little endian, unaligned)."""
    s = b[..., 66:98].reshape(*b.shape[:-1], 8, 4).astype(np.uint32)
    return s[..., 0] | (s[..., 1] << 8) | (s[..., 2] << 16) | (s[..., 3] << 24)


def fields(f: Fmt, raw: np.ndarray):
    """raw uint8 [rows, nb * bsize] -> (db float32 [rows, ngroups], idx int64 [rows, ngroups], neg bool [rows, ngroups, k]).
    db is computed exactly as gguf-py computes it; neg marks a minus sign."""
    rows = raw.shape[0]
    b = raw.reshape(rows, -1, f.bsize)
    nb = b.shape[1]
    d = b[..., :2].copy().view(np.float16).reshape(rows, nb, 1).astype(np.float32)
    g = np.arange(f.gpb)
    if f.qtype == Q.IQ2_S:
        qs, sg, qh, sc = b[..., 2:34], b[..., 34:66], b[..., 66:74], b[..., 74:82]
        idx = qs.astype(np.int64) | (((qh[..., g // 4] >> (2 * (g % 4)).astype(np.uint8)) & 3).astype(np.int64) << 8)
        s = (sc[..., g // 4] >> (4 * ((g % 4) // 2)).astype(np.uint8)) & 0x0F  # uint8
        db = d * (np.float32(0.5) + s) * np.float32(0.25)
        neg = ((sg[..., None] >> np.arange(8, dtype=np.uint8)) & 1).astype(bool)
    else:
        idx = b[..., 2:66].astype(np.int64)
        aux = _aux(b)  # [rows, nb, 8]
        a = aux[..., g // 8]  # per group
        db = d * (np.float32(0.5) + (a >> 28).astype(np.float32)) * np.float32(0.5)
        byte = KSIGNS[(a >> (7 * ((g % 8) // 2)).astype(np.uint32)) & 127]  # 8 sign bits of the word
        nib = (byte >> (4 * (g % 2)).astype(np.uint8)) & 0x0F
        neg = ((nib[..., None] >> np.arange(4, dtype=np.uint8)) & 1).astype(bool)
    return db.reshape(rows, -1), idx.reshape(rows, -1), neg.reshape(rows, -1, f.k)


def values(f: Fmt, db: np.ndarray, idx: np.ndarray, neg: np.ndarray) -> np.ndarray:
    """Dequantized float32 values [rows, ngroups * k], same operation order as gguf-py: db * grid * sign."""
    sign = np.where(neg, np.float32(-1), np.float32(1))
    return (db[..., None] * f.grid[idx] * sign).reshape(db.shape[0], -1)


def decode(f: Fmt, raw: np.ndarray) -> np.ndarray:
    return values(f, *fields(f, raw))


def set_group(f: Fmt, raw: np.ndarray, rows: np.ndarray, gi: int, idx: np.ndarray, neg: np.ndarray) -> None:
    """In place: give group gi (global group index along the row) of the given rows the grid index idx [n] and the
    signs neg [n, k] (True = minus). Leaves d, scales and every other group's bits unchanged."""
    blk, g = divmod(gi, f.gpb)
    o = blk * f.bsize
    idx = np.asarray(idx, dtype=np.int64)
    bits = (np.asarray(neg, dtype=np.uint8) << np.arange(f.k, dtype=np.uint8)).sum(axis=1).astype(np.uint8)
    if f.qtype == Q.IQ2_S:
        if np.any((idx < 0) | (idx >= 1024)):
            raise ValueError("IQ2_S grid index out of range")
        raw[rows, o + 2 + g] = (idx & 0xFF).astype(np.uint8)
        raw[rows, o + 34 + g] = bits
        q, sh = o + 66 + g // 4, 2 * (g % 4)
        raw[rows, q] = (raw[rows, q] & np.uint8(~(3 << sh) & 0xFF)) | (((idx >> 8) & 3) << sh).astype(np.uint8)
    else:
        if np.any((idx < 0) | (idx >= 256)):
            raise ValueError("IQ3_XXS grid index out of range")
        raw[rows, o + 2 + g] = idx.astype(np.uint8)
        w = o + 66 + 4 * (g // 8)
        s = raw[rows, w:w + 4].astype(np.uint32)
        aux = s[:, 0] | (s[:, 1] << 8) | (s[:, 2] << 16) | (s[:, 3] << 24)
        l, h = (g % 8) // 2, g % 2
        byte = KSIGNS[(aux >> np.uint32(7 * l)) & 127]
        byte = (byte & np.uint8(~(0x0F << (4 * h)) & 0xFF)) | (bits << np.uint8(4 * h))
        if np.any(POPC[byte] % 2):
            raise ValueError("IQ3_XXS sign word with an odd number of minus signs is not representable")
        aux = (aux & np.uint32(~(127 << (7 * l)) & 0xFFFFFFFF)) | ((byte.astype(np.uint32) & 127) << np.uint32(7 * l))
        for i in range(4):
            raw[rows, w + i] = ((aux >> np.uint32(8 * i)) & 0xFF).astype(np.uint8)


def fixed_bits_mask(f: Fmt, row_bytes: int) -> np.ndarray:
    """uint8 mask [row_bytes] of the bits refinement must not touch (d, scales)."""
    m = np.zeros((row_bytes // f.bsize, f.bsize), dtype=np.uint8)
    m[:, :2] = 0xFF
    if f.qtype == Q.IQ2_S:
        m[:, 74:82] = 0xFF
    else:
        m[:, 66 + 3:98:4] = 0xF0  # top 4 bits of each little-endian uint32 = the sub-block scale
    return m.reshape(-1)


def validate(f: Fmt, raw: np.ndarray, ref: np.ndarray | None = None) -> None:
    """Every group decodes to a representable code: gguf-py's dequantize equals our decode bit for bit, every value
    is finite and equals db * level * (+-1) for a level of the grid map, and IQ3_XXS sign words have even parity."""
    db, idx, neg = fields(f, raw)
    mine = values(f, db, idx, neg)
    theirs = dequantize(raw, f.qtype).reshape(mine.shape)
    if not np.array_equal(mine.view(np.uint32), theirs.astype(np.float32).view(np.uint32)):
        raise AssertionError("decode differs from gguf-py dequantize")
    if ref is not None and not np.array_equal(mine.view(np.uint32), ref.view(np.uint32)):
        raise AssertionError("decode differs from the tracked values")
    if not np.all(np.isfinite(mine)):
        raise AssertionError("non-finite values")
    lv = np.abs(mine).reshape(*db.shape, f.k)
    cand = db[..., None, None] * f.levels  # [rows, ngroups, 1, n_levels]
    if not np.all(np.any(lv[..., None] == cand, axis=-1)):
        raise AssertionError("a value is not db * level")
    if f.qtype == Q.IQ3_XXS:
        word = neg.reshape(neg.shape[0], -1, 2 * f.k).sum(axis=2)
        if np.any(word % 2):
            raise AssertionError("odd sign parity")


def _feasible(score, lin_u, thr):
    """Mask candidates whose output-gain numerator would fall below the row's floor (amendment 9c)."""
    return score if thr is None else np.where(lin_u >= thr[:, None], score, np.inf)


def _pick_iq2s(f, Gb, Rg, cur, db, neg_cur, ridge, uw=None, thr=None):
    """uw [rows, k] = (H w_r^T) on the group's columns, thr [rows]: with both, only candidates with v . uw >= thr are
    ranked (the output-gain guard); rows with no feasible candidate return an arbitrary one, rejected later."""
    with np.errstate(all="ignore"):  # Accelerate's BLAS raises spurious FP flags under numpy on macOS
        return _pick_iq2s_(f, Gb, Rg, cur, db, neg_cur, ridge, uw, thr)


def _iq2s_sign_patterns(Gb, Rg, cur, db, neg_cur, ridge):
    """The documented IQ2_S sign patterns [target, separable, current], each bool [rows, k] (True = minus)."""
    k = Gb.shape[0]
    with np.errstate(all="ignore"):
        delta = -np.linalg.solve(Gb + ridge * np.eye(k), Rg.T).T
    tgt = cur + delta
    negt = np.where(tgt == 0, neg_cur, tgt < 0)
    lin = db.astype(np.float64)[:, None] * (Rg - np.diag(Gb) * cur)  # sign s_i minimises 2 s_i * lin_i
    negs = np.where(lin == 0, neg_cur, lin > 0)
    return [negt, negs, np.asarray(neg_cur, dtype=bool)]


def _pick_iq2s_(f, Gb, Rg, cur, db, neg_cur, ridge, uw, thr):
    k = f.k
    b = Rg - cur @ Gb
    iu, ju = np.triu_indices(k)
    coef = np.where(iu == ju, 1.0, 2.0)
    P = f.grid[:, iu].astype(np.float64) * f.grid[:, ju]  # [n_grid, k(k+1)/2]
    B = np.hstack([P, f.grid.astype(np.float64)])
    d64 = db.astype(np.float64)[:, None]
    rows = np.arange(len(cur))
    best_s = np.full(len(cur), np.inf)
    best_i = np.zeros(len(cur), dtype=np.int64)
    best_n = np.zeros((len(cur), k), dtype=bool)
    for negp in _iq2s_sign_patterns(Gb, Rg, cur, db, neg_cur, ridge):
        sig = np.where(negp, -1.0, 1.0)
        feat = Gb[iu, ju] * coef * sig[:, iu] * sig[:, ju]  # [rows, k(k+1)/2]
        A = np.hstack([d64 ** 2 * feat, 2 * d64 * sig * b])
        score = A @ B.T
        if thr is not None:
            score = _feasible(score, (d64 * sig * uw) @ f.grid.T.astype(np.float64), thr)
        i = np.argmin(score, axis=1)
        s = score[rows, i]
        better = s < best_s  # ties keep the earlier pattern
        best_s[better], best_i[better], best_n[better] = s[better], i[better], negp[better]
    return best_i, best_n


def _polish_iq2s(f, Gb, Rg, cur, db, ci, cn, uw=None, thr=None, max_passes=16):
    """Single sign flips of the IQ2_S winner (grid index ci, signs cn): weight i flips if that lowers the exact
    dJ = 2 d.R + d G d (d = v - cur) and, with thr, keeps v . uw >= thr. Returns the new signs [rows, k]."""
    with np.errstate(all="ignore"):
        cn = np.array(cn, dtype=bool)
        v = (db[:, None] * f.grid[ci] * np.where(cn, np.float32(-1), np.float32(1))).astype(np.float64)
        d = v - cur
        Gd = d @ Gb
        lin = np.einsum("ri,ri->r", v, uw) if thr is not None else None
        for _ in range(max_passes):
            moved = False
            for i in range(f.k):
                dl = -2 * v[:, i]
                ch = 2 * dl * (Rg[:, i] + Gd[:, i]) + dl * dl * Gb[i, i]
                ok = ch < 0
                if thr is not None:
                    ok &= lin + dl * uw[:, i] >= thr
                if not ok.any():
                    continue
                a = np.nonzero(ok)[0]
                v[a, i] = -v[a, i]
                d[a, i] += dl[a]
                Gd[a] += dl[a, None] * Gb[i][None]
                if lin is not None:
                    lin[a] += dl[a] * uw[a, i]
                cn[a, i] = ~cn[a, i]
                moved = True
            if not moved:
                break
    return cn


_PATS = np.array([[(p >> i) & 1 for i in range(4)] for p in range(16)], dtype=bool)
_PATS_BY_PAR = [_PATS[np.array([bin(p).count("1") % 2 == q for p in range(16)])] for q in (0, 1)]  # 8 each


def _pick_iq3xxs(f, Gb, Rg, cur, db, neg_partner, uw=None, thr=None):
    with np.errstate(all="ignore"):
        return _pick_iq3xxs_(f, Gb, Rg, cur, db, neg_partner, uw, thr)


def _pick_iq3xxs_(f, Gb, Rg, cur, db, neg_partner, uw, thr):
    n_grid = f.grid.shape[0]
    b = Rg - cur @ Gb
    par = neg_partner.sum(axis=1) % 2  # this group's parity must equal the partner's for an even word
    cidx = np.zeros(len(cur), dtype=np.int64)
    cneg = np.zeros((len(cur), f.k), dtype=bool)
    d64 = db.astype(np.float64)
    for q in (0, 1):
        r = np.nonzero(par == q)[0]
        if len(r) == 0:
            continue
        pats = _PATS_BY_PAR[q]  # [8, 4]
        cv = (np.where(pats, -1.0, 1.0)[:, None, :] * f.grid[None].astype(np.float64)).reshape(-1, f.k)  # [8*n_grid, k]
        qc = np.einsum("ci,ci->c", cv @ Gb, cv)
        A = np.hstack([(d64[r] ** 2)[:, None], 2 * d64[r][:, None] * b[r]])
        B = np.vstack([qc[None], cv.T])
        score = A @ B
        if thr is not None:
            score = _feasible(score, d64[r][:, None] * (uw[r] @ cv.T), thr[r])
        best = np.argmin(score, axis=1)
        cidx[r] = best % n_grid
        cneg[r] = pats[best // n_grid]
    return cidx, cneg


def refine(f: Fmt, raw: np.ndarray, w: np.ndarray, g: np.ndarray, max_sweeps: int = 4, tol: float = 1e-4,
           ridge_rel: float = 1e-6, g_report: np.ndarray | None = None, gain_guard: bool = True):
    """raw uint8 [rows, row_bytes] (not modified), w float32 [rows, cols] (the target), g float64 [cols, cols]
    symmetric, the matrix of the objective. With g_report, also logs J under that matrix (JG_before/after).
    gain_guard (amendment 9c): a move is accepted only if dJ < 0 and the row's output gain
    (q_r g w_r^T) / (w_r g w_r^T) stays >= its starting value; candidates breaking that are not ranked.
    Returns (new raw, new values float32, list of per-sweep dicts)."""
    raw = raw.copy()
    db, idx, neg = fields(f, raw)
    dq = values(f, db, idx, neg)
    w64 = w.astype(np.float64)
    E = dq.astype(np.float64) - w64
    with np.errstate(all="ignore"):
        U = w64 @ g  # rows of g w_r^T (g symmetric)
    den_r = np.einsum("ri,ri->r", w64, U)
    num0 = np.einsum("ri,ri->r", dq.astype(np.float64), U)  # the starting (llama-quantize) numerators
    gch = np.zeros(len(num0))  # tracked change of each row's numerator from num0 (Astra point 3)
    ridge = ridge_rel * float(np.mean(np.diag(g)))
    k, ngroups = f.k, db.shape[1]
    sweeps = []
    for s in range(max_sweeps):
        t0 = time.time()
        with np.errstate(all="ignore"):  # Accelerate's BLAS raises spurious FP flags under numpy on macOS
            R = E @ g
        j0 = float(np.sum(E * R))
        if g_report is not None:
            with np.errstate(all="ignore"):
                jg0 = float(np.sum(E * (E @ g_report)))
        tracked, acc_n, blocked = j0, 0, 0
        for gi in range(ngroups):
            c = slice(gi * k, gi * k + k)
            Gb, Rg, cur = g[c, c], R[:, c], dq[:, c].astype(np.float64)
            uw = U[:, c]
            thr = np.einsum("ri,ri->r", cur, uw) - gch if gain_guard else None
            with np.errstate(all="ignore"):
                if f.qtype == Q.IQ2_S:
                    ci, cn = _pick_iq2s(f, Gb, Rg, cur, db[:, gi], neg[:, gi], ridge, uw, thr)
                    cn = _polish_iq2s(f, Gb, Rg, cur, db[:, gi], ci, cn, uw, thr)
                else:
                    ci, cn = _pick_iq3xxs(f, Gb, Rg, cur, db[:, gi], neg[:, gi ^ 1], uw, thr)
                v = db[:, gi, None] * f.grid[ci] * np.where(cn, np.float32(-1), np.float32(1))  # float32, as decode
                dd = v.astype(np.float64) - cur
                dj = 2 * np.sum(dd * Rg, axis=1) + np.einsum("ri,ri->r", dd @ Gb, dd)
                dnum = np.einsum("ri,ri->r", dd, uw)
            ok = dj < 0
            if gain_guard:
                keep = gch + dnum >= 0
                blocked += int(np.sum(ok & ~keep))
                ok &= keep
            a = np.nonzero(ok)[0]
            if len(a) == 0:
                continue
            dq[a, c] = v[a]
            E[a, c] = v[a].astype(np.float64) - w64[a, c]
            with np.errstate(all="ignore"):
                R[a] += dd[a] @ g[c, :]
            set_group(f, raw, a, gi, ci[a], cn[a])
            idx[a, gi], neg[a, gi] = ci[a], cn[a]
            gch[a] += dnum[a]
            tracked += float(dj[a].sum())
            acc_n += len(a)
        with np.errstate(all="ignore"):
            j1 = float(np.sum(E * (E @ g)))
        num_now = np.einsum("ri,ri->r", dq.astype(np.float64), U)  # recomputed, not tracked
        pos = den_r > 0
        sweeps.append({"sweep": s + 1, "J_before": j0, "J_after": j1, "J_tracked": tracked, "accepted": acc_n,
                       "gain_blocked": blocked, "ogain_before": _ratio(num0.sum(), den_r.sum()),
                       "ogain_after": _ratio(num_now.sum(), den_r.sum()),
                       "min_row_ogain_change": float(np.min((num_now[pos] - num0[pos]) / den_r[pos])) if pos.any() else 0.0,
                       "seconds": round(time.time() - t0, 2)})
        if g_report is not None:
            with np.errstate(all="ignore"):
                sweeps[-1].update(JG_before=jg0, JG_after=float(np.sum(E * (E @ g_report))))
        if j0 <= 0 or (j0 - j1) / j0 < tol:
            break
    return raw, dq, sweeps


# ---- driver -------------------------------------------------------------------------------------------------------

_W: dict = {}


def _ratio(a, b):
    """a / b as a float, or None (JSON null) where b is not > 0 (Astra review point 2)."""
    return float(a) / float(b) if b > 0 else None


def _f5(x):
    return "n/a" if x is None else f"{x:.5f}"


def _msg(rec) -> str:
    sw = rec["sweeps"]
    if not sw:
        return rec.get("skipped", "")
    return (f"rel {_f5(sw[0]['rel_before'])} -> {_f5(sw[-1]['rel_after'])} (reg {_f5(sw[0]['relreg_before'])} -> "
            f"{_f5(sw[-1]['relreg_after'])}; output gain {_f5(sw[0]['ogain_before'])} -> {_f5(sw[-1]['ogain_after'])}) "
            f"in {len(sw)} sweeps")


def _worker_init(quant, bf16, grams):
    ge.GRAMS = grams
    _W["q"] = {t.name: t for t in gguf.GGUFReader(quant).tensors}
    _W["b"] = {t.name: t for t in gguf.GGUFReader(bf16).tensors}


def _one(job):
    layer, e, sweeps, tol, ridge_rel, shrink, min_tokens, gain_guard = job
    n = f"blk.{layer}.ffn_gate_up_exps.weight"
    tq, tb = _W["q"][n], _W["b"][n]
    f = FMTS[tq.tensor_type]
    t0 = time.time()
    raw0 = np.array(tq.data[e], dtype=np.uint8)
    validate(f, raw0)  # decode == gguf-py on the original blocks
    u, counts = ge.load_gram(layer)
    with np.errstate(all="ignore"):
        g = ge.symmetrise(np.asarray(u[e], dtype=np.float32)).astype(np.float64)
    w = ge.deq(tb, e)
    w64 = w.astype(np.float64)
    with np.errstate(all="ignore"):
        den = float(np.sum(w64 * (w64 @ g)))
    h = g + shrink * np.diag(np.diag(g))
    with np.errstate(all="ignore"):
        den_h = float(np.sum(w64 * (w64 @ h)))
    rec = {"expert": e, "count": int(counts[e]), "tr_WGWt": den, "tr_WHWt": den_h}
    if not np.trace(g) > 0 or counts[e] < min_tokens:
        why = "empty Gram" if not np.trace(g) > 0 else f"{int(counts[e])} tokens < {min_tokens}"
        rec.update(skipped=why, sweeps=[], seconds=round(time.time() - t0, 2))
        return layer, e, raw0, rec
    raw1, dq1, sw = refine(f, raw0, w, h, sweeps, tol, ridge_rel, g_report=g, gain_guard=gain_guard)
    if gain_guard and sw and sw[-1]["min_row_ogain_change"] < -1e-9:
        raise AssertionError(f"layer {layer} expert {e}: a row's output gain fell by {sw[-1]['min_row_ogain_change']:.3g}")
    validate(f, raw1, dq1)
    mask = fixed_bits_mask(f, raw0.shape[1])
    if not np.array_equal(raw0 & mask, raw1 & mask):
        raise AssertionError(f"layer {layer} expert {e}: d or scale bits changed")
    for x in sw:
        x["relreg_before"], x["relreg_after"] = _ratio(x["J_before"], den_h), _ratio(x["J_after"], den_h)
        x["rel_before"], x["rel_after"] = _ratio(x["JG_before"], den), _ratio(x["JG_after"], den)
    rec.update(sweeps=sw, blocks_changed=int(np.sum(np.any((raw0 != raw1).reshape(raw0.shape[0], -1, f.bsize), axis=2))),
               seconds=round(time.time() - t0, 2))
    return layer, e, raw1, rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quant", default="models/v2a.gguf")
    ap.add_argument("--bf16", default="models/Inkling-2-Mini-BF16.gguf")
    ap.add_argument("--grams", default="models/grams/gu-6-20")
    ap.add_argument("--layers", type=int, nargs="+", default=[6, 20])
    ap.add_argument("--experts", default="0:128:1", help="start:stop:step; experts outside stay unchanged")
    ap.add_argument("--sweeps", type=int, default=4)
    ap.add_argument("--tol", type=float, default=1e-4)
    ap.add_argument("--ridge", type=float, default=1e-6, help="ridge for delta*, times mean(diag H)")
    ap.add_argument("--shrink", type=float, default=0.1, help="objective matrix H = G + shrink * Diag(G) (amendment 9b)")
    ap.add_argument("--min-tokens", type=int, default=2304, help="refine only experts with at least this many Gram tokens")
    ap.add_argument("--no-gain-guard", dest="gain_guard", action="store_false",
                    help="drop amendment 9c's per-row output-gain floor (offline record only)")
    ap.add_argument("--workers", type=int, default=1, help="experts refined in parallel processes")
    ap.add_argument("--out", default="models/refined/gu-refined-6-20.gguf")
    ap.add_argument("--min-free", type=int, default=30)
    args = ap.parse_args()
    if not os.path.exists(f"{args.grams}/done"):
        sys.exit(f"STOP: {args.grams}/done missing")
    fp = ge.free_pct()
    if fp < args.min_free:
        sys.exit(f"STOP: memory_pressure free {fp}% < {args.min_free}%")
    experts = list(range(*(int(x) for x in args.experts.split(":"))))
    qt = {t.name: t for t in gguf.GGUFReader(args.quant).tensors}
    for l in args.layers:
        n = f"blk.{l}.ffn_gate_up_exps.weight"
        t = qt[n]
        if t.tensor_type not in FMTS:
            sys.exit(f"STOP: {n} is {t.tensor_type.name}; only IQ2_S and IQ3_XXS are supported")
        if [int(x) for x in t.shape] != [ge.N_IN, ge.N_OUT, ge.N_EXP]:
            sys.exit(f"STOP: {n} shape {list(t.shape)}")
        if f"blk.{l}.ffn_gate_up_exps.scale" in qt:
            sys.exit(f"STOP: {args.quant} has a .scale for layer {l} (not supported)")
    out = {l: np.array(qt[f"blk.{l}.ffn_gate_up_exps.weight"].data, dtype=np.uint8) for l in args.layers}
    log = {"script": "scripts/refine-codes.py", "args": vars(args), "numpy": np.__version__,
           "order": "experts ascending; groups by ascending input column; all rows in parallel",
           "layers": {str(l): {"type": qt[f"blk.{l}.ffn_gate_up_exps.weight"].tensor_type.name, "experts": []}
                      for l in args.layers}}
    del qt
    jobs = [(l, e, args.sweeps, args.tol, args.ridge, args.shrink, args.min_tokens, args.gain_guard)
            for l in args.layers for e in experts]
    t0 = time.time()

    def take(res):
        l, e, raw, rec = res
        out[l][e] = raw
        log["layers"][str(l)]["experts"].append(rec)
        print(f"[{time.time() - t0:7.0f} s] layer {l} expert {e}: {_msg(rec)} ({rec['seconds']} s)", flush=True)

    if args.workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(args.workers, initializer=_worker_init,
                                 initargs=(args.quant, args.bf16, args.grams)) as ex:
            for res in ex.map(_one, jobs):
                take(res)
    else:
        _worker_init(args.quant, args.bf16, args.grams)
        for j in jobs:
            take(_one(j))
    for l in args.layers:
        log["layers"][str(l)]["experts"].sort(key=lambda r: r["expert"])
        log["layers"][str(l)]["n_refined"] = sum(1 for r in log["layers"][str(l)]["experts"] if r["sweeps"])
    log["seconds_total"] = round(time.time() - t0, 1)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    part = args.out + ".part"
    wr = gguf.GGUFWriter(part, arch="inkling2_loop")
    wr.add_string("refine.source", args.quant)
    wr.add_string("refine.grams", args.grams)
    for l in args.layers:
        n = f"blk.{l}.ffn_gate_up_exps.weight"
        wr.add_tensor(n, out[l], raw_dtype=Q[log["layers"][str(l)]["type"]])
    wr.write_header_to_file()
    wr.write_kv_data_to_file()
    wr.write_tensors_to_file()
    wr.close()
    chk = {t.name: t for t in gguf.GGUFReader(part).tensors}  # re-read: names, types, shapes, bytes
    for l in args.layers:
        n = f"blk.{l}.ffn_gate_up_exps.weight"
        t = chk[n]
        if (t.tensor_type.name != log["layers"][str(l)]["type"] or [int(x) for x in t.shape] != [ge.N_IN, ge.N_OUT, ge.N_EXP]
                or not np.array_equal(np.asarray(t.data), out[l])):
            sys.exit(f"STOP: re-read of {part} differs for {n}")
    del chk
    os.replace(part, args.out)
    Path(args.out + ".json.part").write_text(json.dumps(log, indent=1) + "\n", encoding="utf-8")
    os.replace(args.out + ".json.part", args.out + ".json")
    print(f"wrote {args.out} and {args.out}.json in {log['seconds_total']} s")


if __name__ == "__main__":
    main()
