"""Full-Gram weight error of the gate_up experts, UD-IQ3_XXS against ours (v2a) at the same type (Addendum H amendment 7).

    python scripts/gram-error.py [--layers 6 20] [--experts 0:128:1] [--extra NAME=GGUF ...]

Inputs (layout of the Gram files from the sibling repo's patch 0010: cblas_ssyrk(RowMajor, Upper, Trans) into
g[ex * ne0 * ne0 + row * ne0 + col], so index [expert][row][col] with row <= col filled and the rest left at 0):
  models/grams/gu-6-20/blk.L.ffn_gate_up_exps.weight.gram.f32         float32 [128][2304][2304], upper triangle
  models/grams/gu-6-20/blk.L.ffn_gate_up_exps.weight.gram.counts.i64  int64 [128], rows (tokens) per expert
  BF16 reference, ours (models/v2a.gguf) and UD (models/UD-IQ3_XXS.gguf), tensor blk.L.ffn_gate_up_exps.weight,
  numpy [128][2304 out][2304 in]; out rows 0-1151 = gate, 1152-2303 = up. UD's dequantized values are multiplied by
  blk.L.ffn_gate_up_exps.scale[e] where that tensor exists.

Per expert e, half h (gate, up, both), with G = U + U^T - diag(U) and dW = W_q - W:
  full  = sum_{r in h} dW_r G dW_r^T / sum_{r in h} W_r G W_r^T
  diag  = the same with diag(G)
  plain = ||dW_h||_F / ||W_h||_F
Aggregated over experts unweighted, weighted by counts[e], and weighted by trace(G_e) (amendment 7's "route-weighted").
Paired bootstrap over experts (10,000 resamples, seed 0) of the count-weighted mean of UD minus ours on full.

Writes results/pilot/gram-error.tsv (one row per layer, expert, half, file) and results/pilot/gram-error.md.
--extra NAME=GGUF adds further encodings of the same tensors (same type as ours, no .scale; e.g. the refined
gate_up of scripts/refine-codes.py, amendment 9) as extra columns, with paired bootstraps against ours and UD; if
GGUF.json is a refine-codes log, its runtime and its own before/after objectives are summarised and checked
against the errors computed here. Without --extra the output is unchanged.
"""

import argparse
import os
import sys
import time

import numpy as np

import gguf
from gguf import GGMLQuantizationType as Q
from gguf.quants import dequantize

BF16, OURS, UD = "models/Inkling-2-Mini-BF16.gguf", "models/v2a.gguf", "models/UD-IQ3_XXS.gguf"
GRAMS = "models/grams/gu-6-20"
N_EXP, N_IN, N_OUT = 128, 2304, 2304
HALVES = {"gate": slice(0, 1152), "up": slice(1152, 2304), "both": slice(0, 2304)}
METRICS = ("full", "diag", "plain")
EXTRA: dict[str, str] = {}  # name -> path, from --extra
REG: float | None = None  # --reg s adds metric "reg": the full error under G + s * Diag(G) (amendment 9b)
SPLIT = ("bias", "rest")  # full = bias + rest, needs the per-expert input mean (in_sum / count)
AGGS = ("unweighted", "count-weighted", "trace-weighted")


def symmetrise(u: np.ndarray) -> np.ndarray:
    """Full symmetric Gram from an upper-triangle-only one: U + U^T - diag(U). Ignores whatever is below the diagonal."""
    u = np.triu(u)
    return u + u.T - np.diag(np.diag(u))


def row_forms(x: np.ndarray, g: np.ndarray, mean=None, count: float = 0) -> dict[str, np.ndarray]:
    """Per-row quadratic forms of x [rows, n_in] under the symmetric Gram g: x_r G x_r^T, x_r diag(G) x_r^T, x_r x_r^T,
    and with a mean m and token count n also n (x_r m)^2, the part of x_r G x_r^T carried by the input mean
    (G = C + n m m^T with C the centred scatter). float32 matmul, float64 accumulation."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    with np.errstate(all="ignore"):  # Accelerate's BLAS raises spurious FP flags under numpy on macOS; checked below
        y = x @ g.astype(np.float32, copy=False)
        full = np.einsum("ij,ij->i", y.astype(np.float64), x.astype(np.float64))
        x2 = x.astype(np.float64) ** 2
        out = {"full": full, "diag": x2 @ np.diag(g).astype(np.float64), "plain": x2.sum(axis=1)}
        if REG is not None:
            out["reg"] = full + REG * out["diag"]
        if mean is not None:
            out["bias"] = float(count) * (x.astype(np.float64) @ np.asarray(mean, dtype=np.float64)) ** 2
    for k, v in out.items():
        if not np.all(np.isfinite(v)):
            raise FloatingPointError(f"non-finite {k} forms")
    return out


def rel_errors(w: np.ndarray, wq: np.ndarray, g: np.ndarray, halves=HALVES, mean=None,
               count: float = 0) -> dict[str, dict[str, float]]:
    """Relative errors of wq against w [rows, n_in] under g (symmetric): {metric: {half: value}} (see module doc).
    With the input mean and token count also bias = n ||dW m||^2 / tr(W G W^T) and rest = full - bias."""
    fw, fd = row_forms(w, g), row_forms(wq - w, g, mean, count)
    res: dict[str, dict[str, float]] = {}
    for m in METRICS + (("reg",) if REG is not None else ()) + (("bias",) if mean is not None else ()):
        res[m] = {}
        for h, s in halves.items():
            den = fw["full" if m == "bias" else m][s].sum()
            v = fd[m][s].sum() / den if den > 0 else np.nan
            res[m][h] = float(np.sqrt(v)) if m == "plain" else float(v)
    if mean is not None:
        res["rest"] = {h: res["full"][h] - res["bias"][h] for h in halves}
    return res


def wmean(v: np.ndarray, wt: np.ndarray) -> float:
    ok = np.isfinite(v) & (wt > 0)
    return float(np.sum(v[ok] * wt[ok]) / np.sum(wt[ok]))


def aggregate(v: np.ndarray, counts: np.ndarray, traces: np.ndarray) -> dict[str, float]:
    ok = np.isfinite(v)
    return {"unweighted": float(v[ok].mean()), "count-weighted": wmean(v, counts), "trace-weighted": wmean(v, traces)}


def boot_diff(ud: np.ndarray, ours: np.ndarray, wt: np.ndarray, n: int = 10_000, seed: int = 0):
    """Paired bootstrap over experts of the weighted mean of ud - ours. Returns (point, lo, hi) with a 95% percentile CI."""
    ok = np.isfinite(ud) & np.isfinite(ours) & (wt > 0)
    d, w = (ud - ours)[ok], wt[ok].astype(np.float64)
    idx = np.random.default_rng(seed).integers(0, len(d), size=(n, len(d)))
    bs = (d[idx] * w[idx]).sum(axis=1) / w[idx].sum(axis=1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return float(np.sum(d * w) / np.sum(w)), float(lo), float(hi)


def deq(t: gguf.ReaderTensor, e: int) -> np.ndarray:
    raw = np.asarray(t.data[e])
    if t.tensor_type == Q.BF16:
        return (raw.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
    return dequantize(raw, t.tensor_type).astype(np.float32).reshape(N_OUT, N_IN)


def free_pct() -> int:
    out = os.popen("memory_pressure").read()
    for line in out.splitlines():
        if "free percentage" in line:
            return int(line.rsplit(":", 1)[1].strip().rstrip("%"))
    return -1


def check_inputs(layers: list[int]):
    tens = {k: {t.name: t for t in gguf.GGUFReader(p).tensors}
            for k, p in (("bf16", BF16), ("ours", OURS), ("ud", UD), *EXTRA.items())}
    facts = []
    for l in layers:
        n = f"blk.{l}.ffn_gate_up_exps.weight"
        tb, to, tu = tens["bf16"][n], tens["ours"][n], tens["ud"][n]
        for t in (tb, to, tu):
            if [int(x) for x in t.shape] != [N_IN, N_OUT, N_EXP]:
                sys.exit(f"STOP: {n} shape {list(t.shape)}")
        if tb.tensor_type != Q.BF16:
            sys.exit(f"STOP: BF16 file stores {n} as {tb.tensor_type.name}")
        if to.tensor_type != tu.tensor_type:
            sys.exit(f"STOP: layer {l} types differ: ours {to.tensor_type.name}, UD {tu.tensor_type.name}")
        sc = tens["ud"].get(f"blk.{l}.ffn_gate_up_exps.scale")
        if sc is not None and [int(x) for x in sc.shape] not in ([N_EXP], [1, N_EXP], [N_EXP, 1]):
            sys.exit(f"STOP: UD scale of layer {l} has shape {list(sc.shape)}")
        for k in ("ours", *EXTRA):
            if k in EXTRA:
                tx = tens[k].get(n)
                if tx is None or [int(x) for x in tx.shape] != [N_IN, N_OUT, N_EXP] or tx.tensor_type != to.tensor_type:
                    sys.exit(f"STOP: {EXTRA[k]} lacks {n} with ours' shape and type")
            if tens[k].get(f"blk.{l}.ffn_gate_up_exps.scale") is not None:
                sys.exit(f"STOP: {k} has a gate_up .scale in layer {l}")
        facts.append((l, to.tensor_type.name, None if sc is None else np.asarray(sc.data, dtype=np.float64).reshape(-1)))
    return tens, facts


def load_gram(l: int):
    base = f"{GRAMS}/blk.{l}.ffn_gate_up_exps.weight.gram"
    if os.path.getsize(base + ".f32") != N_EXP * N_IN * N_IN * 4:
        sys.exit(f"STOP: {base}.f32 has {os.path.getsize(base + '.f32')} bytes")
    u = np.memmap(base + ".f32", dtype=np.float32, mode="r", shape=(N_EXP, N_IN, N_IN))
    counts = np.fromfile(base + ".counts.i64", dtype=np.int64)
    if counts.shape != (N_EXP,):
        sys.exit(f"STOP: counts shape {counts.shape}")
    return u, counts


def layout_check(l: int, u: np.memmap, counts: np.ndarray) -> list[str]:
    """Strict lower triangle is zero, upper is not; diag(G_e) and counts against the capture's own imatrix (in_sum2)."""
    notes = []
    for e in [int(x) for x in np.argsort(-counts)[:3]]:
        a = np.asarray(u[e])
        lo, up = np.abs(np.tril(a, -1)).max(), np.abs(np.triu(a, 1)).max()
        if lo != 0 or up == 0:
            sys.exit(f"STOP: layer {l} expert {e}: max |strict lower| {lo}, max |strict upper| {up}")
        notes.append(f"expert {e}: max |strict lower| = {lo:g}, max |strict upper| = {up:.4g}")
    imf = f"{GRAMS}/imatrix.gguf"
    if os.path.exists(imf):
        im = {t.name: t for t in gguf.GGUFReader(imf).tensors}
        n = f"blk.{l}.ffn_gate_up_exps.weight"
        s2 = np.asarray(im[n + ".in_sum2"].data, dtype=np.float64).reshape(N_EXP, N_IN)
        ic = np.asarray(im[n + ".counts"].data, dtype=np.float64).reshape(-1)
        d = np.stack([np.diag(np.asarray(u[e])) for e in range(N_EXP)]).astype(np.float64)
        rel = np.abs(d - s2).max() / np.abs(s2).max()
        notes.append(f"diag(G) vs imatrix in_sum2, all experts: max |diff| / max |in_sum2| = {rel:.3g}; "
                     f"counts equal imatrix counts in {int(np.sum(ic == counts))} of {N_EXP} experts")
    return notes


def load_means(l: int, counts: np.ndarray):
    """Per-expert input means m_e = in_sum_e / count_e from the capture's imatrix.gguf, or (None, reason)."""
    imf = f"{GRAMS}/imatrix.gguf"
    if not os.path.exists(imf):
        return None, f"{imf} missing: bias/rest split skipped"
    im = {t.name: t for t in gguf.GGUFReader(imf).tensors}
    n = f"blk.{l}.ffn_gate_up_exps.weight"
    if n + ".in_sum" not in im:
        return None, f"{n}.in_sum missing from {imf}: bias/rest split skipped"
    t = im[n + ".in_sum"]
    if [int(x) for x in t.shape] != [N_IN, N_EXP]:
        return None, f"{n}.in_sum has gguf shape {[int(x) for x in t.shape]}, not [{N_IN}, {N_EXP}]: split skipped"
    s1 = np.asarray(t.data, dtype=np.float64).reshape(N_EXP, N_IN)
    ic = np.asarray(im[n + ".counts"].data, dtype=np.float64).reshape(-1)
    if not np.array_equal(ic, counts.astype(np.float64)):
        bad = int(np.sum(ic != counts))
        return None, f"imatrix counts differ from Gram counts in {bad} of {N_EXP} experts: split skipped"
    with np.errstate(all="ignore"):
        m = np.where(ic[:, None] > 0, s1 / np.maximum(ic, 1)[:, None], 0.0)
    return m, f"input means from {n}.in_sum (gguf shape [{N_IN}, {N_EXP}]) / counts; counts equal the Gram's"


def run(layers, experts, tsv):
    tens, facts = check_inputs(layers)
    rec, meta = [], {}
    for l, ty, scale in facts:
        n = f"blk.{l}.ffn_gate_up_exps.weight"
        u, counts = load_gram(l)
        notes = layout_check(l, u, counts)
        means, mnote = load_means(l, counts)
        notes.append(mnote)
        mets = METRICS + (("reg",) if REG is not None else ()) + (SPLIT if means is not None else ())
        traces = np.full(N_EXP, np.nan)
        res = {}
        t0 = time.time()
        for e in experts:
            g = symmetrise(np.asarray(u[e], dtype=np.float32))
            traces[e] = float(np.trace(g.astype(np.float64)))
            w = deq(tens["bf16"][n], e)
            qo = deq(tens["ours"][n], e)
            qu = deq(tens["ud"][n], e)
            if scale is not None:
                qu = qu * np.float32(scale[e])
            kw = {} if means is None else {"mean": means[e], "count": float(counts[e])}
            res[e] = {"ours": rel_errors(w, qo, g, **kw), "ud": rel_errors(w, qu, g, **kw)}
            for k in EXTRA:
                res[e][k] = rel_errors(w, deq(tens[k][n], e), g, **kw)
            del g, w, qo, qu
        print(f"layer {l}: {len(experts)} experts in {time.time() - t0:.0f} s", flush=True)
        if means is not None:  # the centred scatter is PSD, so rest >= 0 if in_sum and the Gram saw the same rows
            q = min(res[e][f]["rest"][h] / res[e][f]["full"][h] for e in experts for f in ("ours", "ud", *EXTRA) for h in HALVES
                    if res[e][f]["full"][h] > 0)
            notes.append(f"min over experts, files and halves of rest / full = {q:.6f} (must be >= 0 up to rounding)")
        meta[l] = {"type": ty, "scale": scale, "counts": counts, "traces": traces, "notes": notes, "res": res,
                   "mets": mets}
        for e in experts:
            for h in HALVES:
                for f in ("ours", "ud", *EXTRA):
                    r = res[e][f]
                    rec.append(f"{l}\t{ty}\t{e}\t{counts[e]}\t{traces[e]:.6e}\t{h}\t{f}\t"
                               + "\t".join(f"{r[m][h]:.6e}" if m in r else "nan" for m in tsv_cols()))
    with open(tsv, "w", encoding="utf-8") as fh:
        fh.write("layer\ttype\texpert\tcount\tgram_trace\thalf\tfile\t" + "\t".join(tsv_cols()) + "\n" + "\n".join(rec) + "\n")
    return meta


def tsv_cols() -> tuple[str, ...]:
    return METRICS + SPLIT + (("reg",) if REG is not None else ())


def fmt(x: float) -> str:
    return f"{x:.5f}" if abs(x) >= 1e-3 else f"{x:.3e}"


def report(meta, experts, md, tsv):
    title = "# gate_up error under the full input Gram, UD-IQ3_XXS against ours (v2a)"
    origin = "Written by `scripts/gram-error.py` (Addendum H amendment 7)"
    if EXTRA:
        title += " and " + ", ".join(EXTRA)
        origin += "; extra encodings (amendment 9): " + ", ".join(f"{k} = `{v}`" for k, v in EXTRA.items())
    if REG is not None:
        origin += f"; reg = full with G + {REG:g} * Diag(G) in numerator and denominator (amendment 9b's objective)"
    out = [title, "", origin + ". Per-expert rows: `" + tsv + "`.",
           f"Grams: `{GRAMS}` (patched llama-imatrix on cand.gguf at P = 3, first 100 chunks of `data/calib-wt.txt`, "
           "pooled over all visits), symmetrised from the upper triangle. dW = dequantized minus BF16; UD's dequantized "
           "values times its per-expert `.scale` where present. full = tr(dW G dW^T) / tr(W G W^T), diag = the same "
           "with diag(G), plain = ||dW|| / ||W|| (Frobenius, not squared). bias = n_e ||dW m_e||^2 / tr(W G W^T) with "
           "m_e = in_sum_e / n_e the expert's mean input (G is uncentred, so the mean contributes n_e (dW_r m_e)^2 per "
           "row), rest = full - bias (the centred part). Per expert and half, then averaged over "
           f"{len(experts)} experts: unweighted, weighted by the Gram token count, weighted by tr(G_e). "
           "`UD < ours` counts experts where UD's per-expert error is lower.", ""]
    verdict = {}
    for l, m in meta.items():
        res, counts, traces = m["res"], m["counts"].astype(np.float64), m["traces"]
        sc = m["scale"]
        out += [f"## Layer {l} ({m['type']}; UD `.scale` "
                + ("present, " + f"min {sc.min():.4g}, max {sc.max():.4g}" if sc is not None else "absent") + ")", ""]
        out += [f"- Gram layout: {x}" for x in m["notes"]]
        nz = int(np.sum(m["counts"][experts] > 0))
        out += [f"- Experts with tokens: {nz} of {len(experts)}; tokens per expert min {int(m['counts'][experts].min())}, "
                f"median {int(np.median(m['counts'][experts]))}, max {int(m['counts'][experts].max())}", ""]
        out += ["| metric | half | mean | ours | UD | UD / ours | UD < ours |"
                + "".join(f" {k} | {k} / ours | {k} < ours |" for k in EXTRA),
                "|---|---|---|---|---|---|---|" + "---|---|---|" * len(EXTRA)]
        arr = {}
        for met in ("full",) + tuple(x for x in m["mets"] if x in ("reg",) + SPLIT) + ("diag", "plain"):
            for h in HALVES:
                o = np.full(N_EXP, np.nan)
                u = np.full(N_EXP, np.nan)
                xs = {k: np.full(N_EXP, np.nan) for k in EXTRA}
                for e in experts:
                    o[e], u[e] = res[e]["ours"][met][h], res[e]["ud"][met][h]
                    for k in EXTRA:
                        xs[k][e] = res[e][k][met][h]
                arr[(met, h)] = (o, u)
                ok = np.isfinite(o) & np.isfinite(u)
                win = f"{int(np.sum(u[ok] < o[ok]))} of {int(ok.sum())}"
                verdict[(l, "win", met, h)] = win
                ao, au = aggregate(o, counts, traces), aggregate(u, counts, traces)
                ax = {}
                for k, x in xs.items():
                    arr[(met, h, k)] = x
                    okx = np.isfinite(o) & np.isfinite(x)
                    verdict[(l, "win", met, h, k)] = f"{int(np.sum(x[okx] < o[okx]))} of {int(okx.sum())}"
                    ax[k] = aggregate(x, counts, traces)
                for a in AGGS:
                    out.append(f"| {met} | {h} | {a} | {fmt(ao[a])} | {fmt(au[a])} | {au[a] / ao[a]:.4f} | {win} |"
                               + "".join(f" {fmt(ax[k][a])} | {ax[k][a] / ao[a]:.4f} | {verdict[(l, 'win', met, h, k)]} |"
                                         for k in EXTRA))
                    verdict[(l, met, h, a)] = (ao[a], au[a])
                    for k in EXTRA:
                        verdict[(l, met, h, a, k)] = ax[k][a]
        out += ["", "Paired bootstrap over experts (10,000 resamples, seed 0), count-weighted mean of UD minus ours, full:", "",
                "| half | UD - ours | 95% CI |", "|---|---|---|"]
        for h in HALVES:
            o, u = arr[("full", h)]
            p, lo, hi = boot_diff(u, o, counts)
            verdict[(l, "boot", h)] = (p, lo, hi)
            out.append(f"| {h} | {p:+.4e} | [{lo:+.4e}, {hi:+.4e}] |")
        for k in EXTRA:
            out += ["", f"Paired bootstrap over experts (10,000 resamples, seed 0), count-weighted mean of {k} minus ours "
                    f"and {k} minus UD:", "", f"| metric | half | {k} - ours | 95% CI | {k} - UD | 95% CI |",
                    "|---|---|---|---|---|---|"]
            for met in ("full", "reg", "bias", "rest", "diag"):
                if met not in m["mets"]:
                    continue
                for h in HALVES:
                    o, u = arr[(met, h)]
                    x = arr[(met, h, k)]
                    p1, lo1, hi1 = boot_diff(x, o, counts)
                    p2, lo2, hi2 = boot_diff(x, u, counts)
                    verdict[(l, "boot", met, h, k)] = ((p1, lo1, hi1), (p2, lo2, hi2))
                    out.append(f"| {met} | {h} | {p1:+.4e} | [{lo1:+.4e}, {hi1:+.4e}] | {p2:+.4e} | [{lo2:+.4e}, {hi2:+.4e}] |")
        out.append("")
    out += reading(meta, verdict)
    for k in EXTRA:
        out += ["", *extra_summary(k, meta, verdict, experts)]
    with open(md, "w", encoding="utf-8") as fh:
        fh.write("\n".join(out) + "\n")


def reading(meta, v) -> list[str]:
    """Amendment 7's rule, applied literally: supported if in both layers UD < ours on full and UD >= ours on diag;
    'downstream' if UD is also worse (>) on full in both layers; otherwise neither branch fires."""
    layers = list(meta)
    out = ["## Reading (amendment 7 rule, applied literally)", "",
           "Rule: supported if UD is lower than ours on the full error and higher or equal on the diagonal one in both "
           "layers; if UD is also worse on the full error, layer-local output error does not explain UD's edge.", "",
           "| mean | half | " + " | ".join(f"layer {l} full UD < ours | layer {l} diag UD >= ours" for l in layers)
           + " | branch |", "|---|---|" + "---|---|" * len(layers) + "---|"]
    branches = {}
    for a in AGGS:
        for h in HALVES:
            fl = [v[(l, "full", h, a)][1] < v[(l, "full", h, a)][0] for l in layers]
            dg = [v[(l, "diag", h, a)][1] >= v[(l, "diag", h, a)][0] for l in layers]
            if all(fl) and all(dg):
                b = "supported"
            elif not any(fl):
                b = "UD worse on full: downstream"
            else:
                b = "neither"
            branches[(a, h)] = b
            out.append(f"| {a} | {h} | " + " | ".join(f"{'yes' if f else 'no'} | {'yes' if d else 'no'}"
                                                      for f, d in zip(fl, dg)) + f" | {b} |")
    out.append("")
    # templated sentences; every number comes from the tables above
    s = []
    for l in layers:
        fo, fu = v[(l, "full", "both", "count-weighted")]
        do, du = v[(l, "diag", "both", "count-weighted")]
        p, lo, hi = v[(l, "boot", "both")]
        s.append(f"In layer {l} (both halves, count-weighted), UD's full error is {fmt(fu)} against our {fmt(fo)} "
                 f"(ratio {fu / fo:.3f}; UD - ours {p:+.3e}, 95% CI [{lo:+.3e}, {hi:+.3e}]) and its diagonal error "
                 f"{fmt(du)} against our {fmt(do)} (ratio {du / do:.3f}); UD is lower in {v[(l, 'win', 'full', 'both')]} "
                 f"experts on full and {v[(l, 'win', 'diag', 'both')]} on diag.")
        if (l, "bias", "both", "count-weighted") in v:
            bo, bu = v[(l, "bias", "both", "count-weighted")]
            ro, ru = v[(l, "rest", "both", "count-weighted")]
            s.append(f"Split of the full error in layer {l} (added after registration, not part of the rule): the input-mean "
                     f"part is {fmt(bo)} ours against {fmt(bu)} UD (ratio {bu / bo:.3f}), the centred rest {fmt(ro)} "
                     f"against {fmt(ru)} (ratio {ru / ro:.3f}).")
    b = branches[("count-weighted", "both")]
    allb = sorted(set(branches.values()))
    s.append(f"Under the registered rule this is \"{b}\" for both halves together, count-weighted; across the three "
             f"means and three halves the branches are: {', '.join(allb)}.")
    out.append(" ".join(s))
    return out


def extra_summary(k: str, meta, v, experts) -> list[str]:
    """Templated sentences on an extra encoding against ours and UD, plus its refine-codes log if one sits next to it."""
    out = [f"## {k} against ours and UD", ""]
    s = []
    for l in meta:
        fo, fu = v[(l, "full", "both", "count-weighted")]
        fx = v[(l, "full", "both", "count-weighted", k)]
        (p1, lo1, hi1), _ = v[(l, "boot", "full", "both", k)]
        share = (fo - fx) / (fo - fu) if fo != fu else float("nan")
        s.append(f"In layer {l} (both halves, count-weighted), {k}'s full error is {fmt(fx)} against our {fmt(fo)} and "
                 f"UD's {fmt(fu)} ({k} - ours {p1:+.3e}, 95% CI [{lo1:+.3e}, {hi1:+.3e}]; (ours - {k}) / (ours - UD) = {share:.3f}; lower than ours in {v[(l, 'win', 'full', 'both', k)]} experts).")
        if (l, "bias", "both", "count-weighted", k) in v:
            bo, bu = v[(l, "bias", "both", "count-weighted")]
            ro, ru = v[(l, "rest", "both", "count-weighted")]
            bx, rx = v[(l, "bias", "both", "count-weighted", k)], v[(l, "rest", "both", "count-weighted", k)]
            s.append(f"Bias part {fmt(bx)} (ours {fmt(bo)}, UD {fmt(bu)}), rest {fmt(rx)} (ours {fmt(ro)}, UD {fmt(ru)}).")
        if (l, "reg", "both", "count-weighted", k) in v:
            go, gu = v[(l, "reg", "both", "count-weighted")]
            gx = v[(l, "reg", "both", "count-weighted", k)]
            s.append(f"Regularised (the search objective) {fmt(gx)} (ours {fmt(go)}, UD {fmt(gu)}).")
        do, du = v[(l, "diag", "both", "count-weighted")]
        dx = v[(l, "diag", "both", "count-weighted", k)]
        po, pu = v[(l, "plain", "both", "count-weighted")]
        px = v[(l, "plain", "both", "count-weighted", k)]
        s.append(f"Diagonal {fmt(dx)} (ours {fmt(do)}, UD {fmt(du)}), plain {fmt(px)} (ours {fmt(po)}, UD {fmt(pu)}).")
    out.append(" ".join(s))
    # in-sample caveat: an expert with fewer tokens than inputs has a rank-deficient Gram, whose null space a search
    # under that same Gram can fill at zero measured cost
    out += ["", f"Full error (both halves) by Gram token count; an expert with fewer than {N_IN} tokens has a "
            "rank-deficient Gram (the same Gram the search minimised, so these are in-sample errors):", "",
            f"| layer | tokens | experts | share of tokens | ours (count-w.) | UD (count-w.) | {k} (count-w.) | "
            f"{k} / ours | {k} / UD |", "|---|---|---|---|---|---|---|---|---|"]
    for l, m in meta.items():
        res, c = m["res"], m["counts"].astype(np.float64)
        ex = np.array(experts)
        for lab, sel in ((f"< {N_IN}", c[ex] < N_IN), (f">= {N_IN}", c[ex] >= N_IN)):
            e = ex[sel]
            if len(e) == 0:
                continue
            wt = c[e]
            f = {x: wmean(np.array([res[i][x]["full"]["both"] for i in e]), wt) for x in ("ours", "ud", k)}
            out.append(f"| {l} | {lab} | {len(e)} | {wt.sum() / c[ex].sum():.3f} | {fmt(f['ours'])} | {fmt(f['ud'])} | "
                       f"{fmt(f[k])} | {f[k] / f['ours']:.3f} | {f[k] / f['ud']:.3f} |")
    logf = EXTRA[k] + ".json"
    if not os.path.exists(logf):
        return out
    import json
    log = json.load(open(logf, encoding="utf-8"))
    if log.get("script") != "scripts/refine-codes.py":
        return out
    out += ["", f"Refinement log `{logf}` (scripts/refine-codes.py; wall time {log['seconds_total']:.0f} s for "
            f"{sum(len(x['experts']) for x in log['layers'].values())} expert tensors, "
            f"workers {log['args'].get('workers')}, sweeps <= {log['args'].get('sweeps')}, tol {log['args'].get('tol')}):", "",
            "| layer | experts refined | skipped | sweeps used (count) | accepted moves | blocks changed | "
            "expert seconds (sum) | max abs(log - here), before | max abs(log - here), after | J never rose |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    regchk: list[str] = []
    for l in meta:
        recs = [r for r in log["layers"].get(str(l), {}).get("experts", []) if r["expert"] in experts]
        done = [r for r in recs if r["sweeps"]]
        nsw: dict[int, int] = {}
        for r in done:
            nsw[len(r["sweeps"])] = nsw.get(len(r["sweeps"]), 0) + 1
        res = meta[l]["res"]
        db = max((abs(r["sweeps"][0]["rel_before"] - res[r["expert"]]["ours"]["full"]["both"]) for r in done), default=0.0)
        da = max((abs(r["sweeps"][-1]["rel_after"] - res[r["expert"]][k]["full"]["both"]) for r in done), default=0.0)
        mono = all(sw["J_after"] <= sw["J_before"] for r in done for sw in r["sweeps"])
        if REG is not None and done and "relreg_before" in done[0]["sweeps"][0]:
            gb = max(abs(r["sweeps"][0]["relreg_before"] - res[r["expert"]]["ours"]["reg"]["both"]) for r in done)
            ga = max(abs(r["sweeps"][-1]["relreg_after"] - res[r["expert"]][k]["reg"]["both"]) for r in done)
            regchk.append(f"layer {l}: max abs(log - here) on reg {gb:.2e} before, {ga:.2e} after")
        nblk = N_OUT * N_IN // 256
        out.append(f"| {l} | {len(done)} | {len(recs) - len(done)} | "
                   + ", ".join(f"{a}: {b}" for a, b in sorted(nsw.items())) + " | "
                   f"{sum(sw['accepted'] for r in done for sw in r['sweeps'])} | "
                   f"{sum(r.get('blocks_changed', 0) for r in done)} of {nblk * len(done)} | "
                   f"{sum(r['seconds'] for r in done):.0f} | {db:.2e} | {da:.2e} | {'yes' if mono else 'no'} |")
    sk = sorted({r.get("skipped", "") for x in log["layers"].values() for r in x["experts"] if not r["sweeps"]})
    out += ["", f"Search settings from the log: shrink {log['args'].get('shrink', 0)}, min tokens "
            f"{log['args'].get('min_tokens', 0)}; 'J never rose' is under the search objective; the max abs columns "
            "compare the log's own full-G relative errors with the ones computed here."
            + (" " + "; ".join(regchk) + "." if regchk else "")
            + (f" Skipped experts keep the source bytes (reasons: {len(sk)} distinct, e.g. '{sk[0]}')." if sk else "")]
    return out


def main() -> None:
    global GRAMS, REG
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, nargs="+", default=[6, 20])
    ap.add_argument("--experts", default="0:128:1", help="start:stop:step")
    ap.add_argument("--tsv", default="results/pilot/gram-error.tsv")
    ap.add_argument("--md", default="results/pilot/gram-error.md")
    ap.add_argument("--grams", default=GRAMS, help="Gram directory (default the amendment 7 capture)")
    ap.add_argument("--min-free", type=int, default=30, help="memory_pressure free percentage required to start")
    ap.add_argument("--reg", type=float, default=None, metavar="S",
                    help="also report the full error under G + S * Diag(G) (amendment 9b: S = 0.1)")
    ap.add_argument("--extra", action="append", default=[], metavar="NAME=GGUF",
                    help="another encoding of the same tensors to compare (e.g. refined=models/refined/gu-refined-6-20.gguf)")
    args = ap.parse_args()
    GRAMS = args.grams
    REG = args.reg
    for x in args.extra:
        k, _, path = x.partition("=")
        if not path or k in ("ours", "ud", "bf16") or k in EXTRA:
            sys.exit(f"STOP: bad --extra {x}")
        EXTRA[k] = path
    if not os.path.exists(f"{GRAMS}/done"):
        sys.exit(f"STOP: {GRAMS}/done missing (capture not finished)")
    fp = free_pct()
    if fp < args.min_free:
        sys.exit(f"STOP: memory_pressure free {fp}% < {args.min_free}%")
    experts = list(range(*(int(x) for x in args.experts.split(":"))))
    meta = run(args.layers, experts, args.tsv)
    report(meta, experts, args.md, args.tsv)
    print(f"wrote {args.tsv}, {args.md}")


if __name__ == "__main__":
    main()
