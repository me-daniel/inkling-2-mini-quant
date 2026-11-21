"""Writes donor GGUFs with Gaussian-perturbed routed experts for the Addendum A noise conditions (CHANGELOG.md).

    python scripts/noise.py BASE.gguf TWO_BIT.gguf OUT_PREFIX --layers 6-11 --seeds 1 2 3

For each layer, projection and expert, the noise has the same relative l2 error as the 2-bit level has against
BASE for that expert: t = ||W_2bit - W_base||^2 / ||W_base||^2, noise ~ N(0, s^2) with s^2 = t * ||W_base||^2 / n.
Each seed gives one donor file OUT_PREFIX-<seed>.gguf holding only the perturbed expert tensors, stored as Q4_0.
The achieved relative error after Q4_0 encoding is measured and written to OUT_PREFIX.tsv.
"""

import argparse
import sys

import numpy as np

import gguf
from gguf import GGMLQuantizationType as Q
from gguf.quants import dequantize, quantize

KINDS = ("ffn_gate_up_exps", "ffn_down_exps")


def rel_err(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sum((a.astype(np.float64) - b) ** 2) / np.sum(b.astype(np.float64) ** 2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("two_bit")
    ap.add_argument("out_prefix")
    ap.add_argument("--layers", required=True, help="e.g. 6-11")
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--kinds", nargs="+", default=list(KINDS), choices=KINDS)
    args = ap.parse_args()

    lo, hi = (int(x) for x in args.layers.split("-"))
    base = {t.name: t for t in gguf.GGUFReader(args.base).tensors}
    two = {t.name: t for t in gguf.GGUFReader(args.two_bit).tensors}

    writers = {s: gguf.GGUFWriter(f"{args.out_prefix}-{s}.gguf", arch="inkling2_loop") for s in args.seeds}
    rows = ["layer\tkind\texpert\tt_2bit\t" + "\t".join(f"t_seed{s}" for s in args.seeds)]
    for layer in range(lo, hi + 1):
        for k, kind in enumerate(KINDS):
            if kind not in args.kinds:
                continue
            name = f"blk.{layer}.{kind}.weight"
            tb, t2 = base[name], two[name]
            n_expert = tb.data.shape[0]
            out = {s: [] for s in args.seeds}
            for e in range(n_expert):
                wb = dequantize(np.asarray(tb.data[e]), tb.tensor_type)
                w2 = dequantize(np.asarray(t2.data[e]), t2.tensor_type)
                t = rel_err(w2, wb)
                # Q4_0 storage adds its own, roughly independent error; take it out of the noise budget
                t_store = rel_err(dequantize(quantize(wb, Q.Q4_0), Q.Q4_0), wb)
                sigma = np.sqrt(max(t - t_store, 0.0) * np.sum(wb.astype(np.float64) ** 2) / wb.size)
                achieved = []
                for s in args.seeds:
                    rng = np.random.default_rng([s, layer, k, e])
                    wn = (wb + sigma * rng.standard_normal(wb.shape)).astype(np.float32)
                    qn = quantize(wn, Q.Q4_0)
                    achieved.append(rel_err(dequantize(qn, Q.Q4_0), wb))
                    out[s].append(qn)
                rows.append(f"{layer}\t{kind}\t{e}\t{t:.6f}\t" + "\t".join(f"{a:.6f}" for a in achieved))
            for s in args.seeds:
                data = np.stack(out[s])
                # data has the byte shape [expert, row, bytes]; the writer derives the logical shape from it
                writers[s].add_tensor(name, data, raw_dtype=Q.Q4_0)
            print(f"{name}: mean t_2bit {np.mean([float(r.split()[3]) for r in rows[-n_expert:]]):.4f}", file=sys.stderr)

    for w in writers.values():
        w.write_header_to_file()
        w.write_kv_data_to_file()
        w.write_tensors_to_file()
        w.close()
    with open(f"{args.out_prefix}.tsv", "w", encoding="utf-8") as f:
        f.write("\n".join(rows) + "\n")


if __name__ == "__main__":
    main()
