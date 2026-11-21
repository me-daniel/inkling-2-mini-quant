"""Audit of the per-pass result on chunks 0 and 1 (light, CPU, 16 rows at a time).

Recomputes from the raw logits files: KL per chunk (kldfile method and a llama-perplexity-style p side),
Fisher r per chunk (to compare with perpass-residual.json), the clipping floor's mass, floor crossings,
1/2 Fisher vs KL, full-vocabulary centred Euclidean (the registered form), head Euclidean at 5/10/14 nats,
Fisher restricted to entries clean in all five files, per-token r, and the top-1% residual energy share.
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(sys.argv[1])
CHUNKS = [int(x) for x in sys.argv[2].split(",")]
OUT = Path(sys.argv[3])
sys.path.insert(0, str(ROOT / "scripts"))
import kldfile  # noqa: E402

K = ROOT / "models/logits"
NAMES = ["base", "p1", "p2", "p3", "all"]
paths = {n: K / f"pp-{n}.kld" for n in NAMES}
assert kldfile.same_tokens(*paths.values())
h = kldfile.read_header(paths["base"])
raws = {n: np.memmap(p, dtype="<u2", mode="r", offset=h.data_offset, shape=(h.n_chunk, h.n_scored, h.nv)) for n, p in paths.items()}
ROWS = 16
TERMS = ("d1", "d2", "d3", "all", "sum", "res", "d12", "d23", "d13")
HEADS = (5.0, 10.0, 14.0)


def decode(block):
    params = np.ascontiguousarray(block[:, :4]).view("<f4")
    codes = block[:, 4 : 4 + h.n_vocab]
    lp = params[:, 1:2].astype(np.float64) + params[:, 0:1].astype(np.float64) * codes.astype(np.float64)
    return lp, codes == 0, params


def fisher(p, x):
    m = np.sum(p * x, axis=1, keepdims=True)
    return np.sum(p * (x - m) ** 2, axis=1)  # per row


def euclid(mask, x):
    n = mask.sum(axis=1, keepdims=True)
    m = np.sum(np.where(mask, x, 0.0), axis=1, keepdims=True) / n
    return np.sum(np.where(mask, (x - m) ** 2, 0.0), axis=1)


res = {}
for c in CHUNKS:
    acc = {"fisher": np.zeros(len(TERMS)), "fisher_clean": np.zeros(len(TERMS)), "full_euclid": np.zeros(len(TERMS)),
           **{f"head{hn:g}": np.zeros(len(TERMS)) for hn in HEADS}}
    per_row_f = []  # per-token Fisher sq norms (res, all)
    kl = {k: 0.0 for k in ("p1", "p2", "p3", "all")}
    kl_llama_p = {k: 0.0 for k in kl}
    half_f = {k: 0.0 for k in kl}
    tail_mass, n_clip_rows, n_rows = 0.0, 0, 0
    cross_mass = {k: 0.0 for k in kl}  # base mass on entries floored in S but not in base
    cross_fisher = {k: 0.0 for k in kl}  # their share of the Fisher sq norm
    head_counts = []
    for s in range(0, h.n_scored, ROWS):
        dec = {n: decode(np.asarray(raws[n][c, s : s + ROWS])) for n in NAMES}
        lb, fb, _ = dec["base"]
        n_rows += lb.shape[0]
        p = np.exp(lb)
        p /= p.sum(axis=1, keepdims=True)
        logp = np.log(p)
        tail_mass += float(np.sum(np.where(fb, p, 0.0)))
        n_clip_rows += int(np.sum(fb.any(axis=1)))
        clean = ~fb
        for k in kl:
            lq, fq, _ = dec[k]
            lqn = lq - np.log(np.exp(lq).sum(axis=1, keepdims=True))
            kl[k] += float(np.sum(p * (logp - lqn)))
            # llama-perplexity's p side: unnormalised file probabilities, entries with log p > -16 only
            pl = np.where(lb > -16.0, np.exp(lb), 0.0)
            kl_llama_p[k] += float(np.sum(pl * (lb - lqn)))
            x = lq - lb
            half_f[k] += 0.5 * float(np.sum(fisher(p, x)))
            cr = fq & ~fb
            cross_mass[k] += float(np.sum(np.where(cr, p, 0.0)))
            m = np.sum(p * x, axis=1, keepdims=True)
            cross_fisher[k] += float(np.sum(np.where(cr, p * (x - m) ** 2, 0.0)))
            clean &= ~fq
        d = {"d1": dec["p1"][0] - lb, "d2": dec["p2"][0] - lb, "d3": dec["p3"][0] - lb, "all": dec["all"][0] - lb}
        d["sum"] = d["d1"] + d["d2"] + d["d3"]
        d["res"] = d["all"] - d["sum"]
        d["d12"], d["d23"], d["d13"] = d["d1"] + d["d2"], d["d2"] + d["d3"], d["d1"] + d["d3"]
        pc = np.where(clean, p, 0.0)
        pc /= pc.sum(axis=1, keepdims=True)
        heads = {hn: lb >= lb.max(axis=1, keepdims=True) - hn for hn in HEADS}
        head_counts.append(heads[10.0].sum(axis=1))
        full = np.ones_like(lb, dtype=bool)
        rowf = {}
        for i, t in enumerate(TERMS):
            f = fisher(p, d[t])
            rowf[t] = f
            acc["fisher"][i] += f.sum()
            acc["fisher_clean"][i] += fisher(pc, d[t]).sum()
            acc["full_euclid"][i] += euclid(full, d[t]).sum()
            for hn in HEADS:
                acc[f"head{hn:g}"][i] += euclid(heads[hn], d[t]).sum()
        per_row_f.append(np.stack([rowf["res"], rowf["all"]], axis=1))
    prf = np.concatenate(per_row_f)
    hc = np.concatenate(head_counts)

    def stats(a):
        g = dict(zip(TERMS, a))
        return {"r": np.sqrt(g["res"] / g["all"]), "norm_ratio": np.sqrt(g["all"] / g["sum"]),
                "p1_over_p3": np.sqrt(g["d1"] / g["d3"]), "p2_over_p3": np.sqrt(g["d2"] / g["d3"]),
                "visit_mult": g["all"] / ((g["d1"] + g["d2"] + g["d3"]) / 3),
                **{f"cos{a}{b}": (g[f"d{a}{b}"] - g[f"d{a}"] - g[f"d{b}"]) / (2 * np.sqrt(g[f"d{a}"] * g[f"d{b}"]))
                   for a, b in (("1", "2"), ("2", "3"), ("1", "3"))}}

    res_sorted = np.sort(prf[:, 0])[::-1]
    top = max(1, int(round(0.01 * len(res_sorted))))
    ok = prf[:, 1] > 0
    res[c] = {
        "n_rows": n_rows,
        "kl_kldfile": {k: v / n_rows for k, v in kl.items()},
        "kl_llama_p_side": {k: v / n_rows for k, v in kl_llama_p.items()},
        "half_fisher_over_kl": {k: half_f[k] / kl[k] for k in kl},
        "tail_mass_mean": tail_mass / n_rows,
        "rows_with_clipping": n_clip_rows / n_rows,
        "cross_mass_mean": {k: v / n_rows for k, v in cross_mass.items()},
        "cross_fisher_share": {k: cross_fisher[k] / (2 * half_f[k]) for k in kl},
        "head10_entries": {"min": int(hc.min()), "median": float(np.median(hc)), "max": int(hc.max())},
        "per_token_r_fisher": {"median": float(np.median(np.sqrt(prf[ok, 0] / prf[ok, 1]))),
                               "p90": float(np.percentile(np.sqrt(prf[ok, 0] / prf[ok, 1]), 90))},
        "top1pct_residual_energy_share": float(res_sorted[:top].sum() / res_sorted.sum()),
        **{name: {k: float(v) for k, v in stats(a).items()} for name, a in acc.items()},
    }
    print(json.dumps({c: res[c]}, indent=1), flush=True)

OUT.write_text(json.dumps(res, indent=1) + "\n")
