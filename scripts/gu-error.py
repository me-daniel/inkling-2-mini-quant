"""Same-type weight error of the gate_up experts: Unsloth UD-IQ3_XXS against our builds, both measured against BF16.

    python scripts/gu-error.py OUT.tsv [--experts 0:128:8]

For each layer, takes UD's ffn_gate_up_exps type and every one of our files (pool-2bit, pool-3bit, cand) that stores
that layer's gate_up at the same type. Per sampled expert e (same indices for every file) it dequantizes with
gguf.quants.dequantize, multiplies UD's result by blk.N.ffn_gate_up_exps.scale[e] when that tensor exists, and computes
  rel   = ||W_q - W|| / ||W||                                   (Frobenius)
  rel_w = sqrt(sum_ij h_j (W_q - W)_ij^2 / sum_ij h_j W_ij^2)    h_j = in_sum2[e, j] / counts[e] (our imatrix, as q2opt.py)
then averages over the sampled experts. Also writes one 'bytes' row per layer for UD and cand (whole tensor, from the
file). cand rows whose bytes equal the pool tensor's are flagged 'same_as' so they are not counted twice.

    python scripts/gu-error.py --report results/pilot/gu-error.tsv results/pilot/gu-error.md

renders the tables of the TSV as Markdown (per type: UD against our same-type tensor; per layer: types and bytes).
"""

import argparse
import hashlib
import sys

import numpy as np

import gguf
from gguf import GGMLQuantizationType as Q
from gguf.quants import dequantize

FILES = {
    "bf16": "models/Inkling-2-Mini-BF16.gguf",
    "ud": "models/UD-IQ3_XXS.gguf",
    "cand": "models/cand.gguf",
    "pool-3bit": "models/pool-3bit.gguf",
    "pool-2bit": "models/pool-2bit.gguf",
}
IMATRIX = "models/ggml-org/imatrix.gguf"
OURS = ("pool-2bit", "pool-3bit", "cand")


def rows(t: gguf.ReaderTensor, e: int) -> np.ndarray:
    raw = np.asarray(t.data[e])
    if t.tensor_type == Q.BF16:
        return (raw.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
    return dequantize(raw, t.tensor_type).astype(np.float32)


def digest(t: gguf.ReaderTensor) -> str:
    return hashlib.sha1(memoryview(np.ascontiguousarray(t.data)).cast("B")).hexdigest()


def report(tsv: str, md: str) -> None:
    lines = open(tsv, encoding="utf-8").read().splitlines()
    head, rec = lines[1].split("\t"), [dict(zip(lines[1].split("\t"), x.split("\t"))) for x in lines[2:]]
    del head
    out = ["# gate_up weight error, UD-IQ3_XXS against ours at the same type", "",
           f"Tables written by `scripts/gu-error.py --report` from `results/pilot/gu-error.tsv`. Sample: {lines[0][2:]}.",
           "rel = ||W_q - W|| / ||W|| against the BF16 file, rel_w the same weighted per input column by our imatrix "
           "(in_sum2 / counts), both averaged over the sampled experts. UD's dequantized values are multiplied by its "
           "per-expert `.scale` where present. Our row: cand where cand's tensor is its own encoding, else the pool file "
           "at that type (cand rows identical to a pool tensor are not repeated).", ""]
    layers = sorted({int(r["layer"]) for r in rec})
    by = {(int(r["layer"]), r["file"]): r for r in rec}
    for ty in sorted({r["ud_type"] for r in rec}):
        rows_ = []
        tot = [0, 0, 0]
        for l in layers:
            ud = by[(l, "ud")]
            if ud["ud_type"] != ty:
                continue
            ours = [r for r in rec if int(r["layer"]) == l and r["file"] != "ud" and r["rel"] and not r["same_as"]]
            for o in ours:
                d, dw = float(ud["rel"]) - float(o["rel"]), float(ud["rel_w"]) - float(o["rel_w"])
                tot = [tot[0] + 1, tot[1] + (d > 0), tot[2] + (dw > 0)]
                rows_.append(f"| {l} | {o['file']} | {ud['rel']} | {o['rel']} | {d:+.5f} | {ud['rel_w']} | {o['rel_w']} | {dw:+.5f} |")
            if not ours:
                rows_.append(f"| {l} | (none of ours at {ty}) | {ud['rel']} | | | {ud['rel_w']} | | |")
        out += [f"## {ty}", "", "| layer | our file | UD rel | our rel | UD - ours | UD rel_w | our rel_w | UD - ours |",
                "|---|---|---|---|---|---|---|---|"] + rows_
        out += ["", f"UD higher than ours in {tot[1]} of {tot[0]} same-type layers (rel), {tot[2]} of {tot[0]} (rel_w).", ""]
    out += ["## Types and bytes per layer (whole tensor, from the files)", "",
            "| layer | UD type | cand type | UD bytes | cand bytes | UD - cand |", "|---|---|---|---|---|---|"]
    groups: dict[tuple[str, str], list[int]] = {}
    sud = sc = 0
    for l in layers:
        u, c = by[(l, "ud")], by[(l, "cand")]
        bu, bc = int(u["bytes"]), int(c["bytes"])
        sud, sc = sud + bu, sc + bc
        out.append(f"| {l} | {u['type']} | {c['type']} | {bu:,} | {bc:,} | {bu - bc:+,} |")
        if bu != bc or u["type"] != c["type"]:
            groups.setdefault((c["type"], u["type"]), []).append(l)
    out += [f"| all | | | {sud:,} | {sc:,} | {sud - sc:+,} |", "", "Byte difference by type change (cand -> UD):", "",
            "| cand -> UD | layers | UD - cand bytes |", "|---|---|---|"]
    for (ct, ut), ls in sorted(groups.items(), key=lambda kv: kv[1][0]):
        d = sum(int(by[(l, 'ud')]['bytes']) - int(by[(l, 'cand')]['bytes']) for l in ls)
        out.append(f"| {ct} -> {ut} | {', '.join(map(str, ls))} | {d:+,} |")
    with open(md, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")


def main() -> None:
    if len(sys.argv) == 4 and sys.argv[1] == "--report":
        report(sys.argv[2], sys.argv[3])
        return
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--experts", default="0:128:8", help="start:stop:step over the 128 experts")
    args = ap.parse_args()
    experts = list(range(*(int(x) for x in args.experts.split(":"))))

    tens = {k: {t.name: t for t in gguf.GGUFReader(p).tensors} for k, p in FILES.items()}
    imat = {t.name: t for t in gguf.GGUFReader(IMATRIX).tensors}
    out = [f"# experts {args.experts} ({len(experts)} of 128): {experts}",
           "layer\tud_type\tfile\ttype\tbytes\tsame_as\tud_scale\trel\trel_w\trel_sd\trel_w_sd"]
    for layer in range(24):
        name = f"blk.{layer}.ffn_gate_up_exps.weight"
        ud = tens["ud"][name]
        ud_scale = tens["ud"].get(f"blk.{layer}.ffn_gate_up_exps.scale")
        ud_type = ud.tensor_type.name
        w_bf = tens["bf16"][name]
        sums = np.asarray(imat[name + ".in_sum2"].data, dtype=np.float64)
        counts = np.asarray(imat[name + ".counts"].data, dtype=np.float64).reshape(-1)
        same = [k for k in OURS if tens[k][name].tensor_type == ud.tensor_type]
        same_as = {}
        if "cand" in same:
            dc = digest(tens["cand"][name])
            for k in ("pool-2bit", "pool-3bit"):
                if k in same and digest(tens[k][name]) == dc:
                    same_as["cand"] = k
        cols = {k: [] for k in ["ud"] + same}
        for e in experts:
            w = rows(w_bf, e).astype(np.float64)
            h = sums[e] / counts[e] if counts[e] > 0 else np.ones(w.shape[1])
            h = np.maximum(h, 1e-12)
            n, nw = np.sum(w * w), np.sum(h * (w * w).sum(axis=0))
            for k in cols:
                q = rows(tens[k][name], e).astype(np.float64)
                if k == "ud" and ud_scale is not None:
                    q *= float(ud_scale.data[e])
                d2 = (q - w) ** 2
                cols[k].append((np.sqrt(d2.sum() / n), np.sqrt(np.sum(h * d2.sum(axis=0)) / nw)))
        for k in ["ud", "cand"] + [x for x in same if x != "cand"]:
            t = tens[k][name]
            if k in cols:
                a = np.array(cols[k])
                m, s = a.mean(axis=0), a.std(axis=0, ddof=1)
                err = f"{m[0]:.5f}\t{m[1]:.5f}\t{s[0]:.5f}\t{s[1]:.5f}"
            else:
                err = "\t\t\t"
            sc = ("yes" if ud_scale is not None else "no") if k == "ud" else ""
            row = f"{layer}\t{ud_type}\t{k}\t{t.tensor_type.name}\t{int(t.n_bytes)}\t{same_as.get(k, '')}\t{sc}\t{err}"
            out.append(row)
            print(row, file=sys.stderr, flush=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
