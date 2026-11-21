"""Addendum A (CHANGELOG.md): relative l2 error t of the routed experts of each pool against the Q8_0 reference.

    python scripts/t_blocks.py REF.gguf IMATRIX.gguf OUT.tsv POOL_NAME=POOL.gguf [...]

Per layer, projection and pool: t = sum_e ||W_q,e - W_e||^2 / sum_e ||W_e||^2, and t_w with each expert weighted
by its routing count from the imatrix (counts of looped layers are pooled over the three visits). Experts are
dequantized one at a time; layers run in parallel processes.
"""

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

import gguf
from gguf.quants import dequantize

KINDS = ("ffn_gate_up_exps", "ffn_down_exps")


def one(job: tuple[str, str, str, str, int, str]) -> str:
    ref_path, imat_path, pool_name, pool_path, layer, kind = job
    name = f"blk.{layer}.{kind}.weight"
    ref = {t.name: t for t in gguf.GGUFReader(ref_path).tensors}[name]
    pool = {t.name: t for t in gguf.GGUFReader(pool_path).tensors}[name]
    imat = {t.name: t for t in gguf.GGUFReader(imat_path).tensors}
    counts = np.asarray(imat[name + ".counts"].data, dtype=np.float64).reshape(-1)
    err = np.zeros(counts.size)
    norm = np.zeros(counts.size)
    for e in range(counts.size):
        w = dequantize(np.asarray(ref.data[e]), ref.tensor_type).astype(np.float64)
        q = dequantize(np.asarray(pool.data[e]), pool.tensor_type).astype(np.float64)
        err[e] = np.sum((q - w) ** 2)
        norm[e] = np.sum(w**2)
    t = err.sum() / norm.sum()
    tw = (counts * err).sum() / (counts * norm).sum()
    return f"{layer}\t{kind}\t{pool_name}\t{pool.tensor_type.name}\t{t:.6f}\t{tw:.6f}"


def main() -> None:
    ref_path, imat_path, out, *pools = sys.argv[1:]
    n_layer = 1 + max(int(t.name.split(".")[1]) for t in gguf.GGUFReader(ref_path).tensors if t.name.startswith("blk."))
    jobs = [(ref_path, imat_path, *p.split("=", 1), layer, kind) for p in pools for layer in range(n_layer) for kind in KINDS]
    with ProcessPoolExecutor(max_workers=6) as ex:
        rows = list(ex.map(one, jobs))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write("layer\tkind\tpool\ttype\tt\tt_weighted\n" + "\n".join(rows) + "\n")
    print(f"wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
