"""Real-file checks, light: (a) the writer's rounding noise floor on the residual, from the first 128 scored rows of
chunk 0 of the five 2-bit pass-split logits files (about 51 MB read per file); (b) float32 against float64 decoding;
(c) whether the 3-bit rung changes B2's down tensors (sha256 of 12 tensors in two GGUFs).

    .venv/bin/python -I reviews/2026-11-21-code-review-analysis-opus/check_real_logits.py
"""

import hashlib
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import gguf  # noqa: E402
import kldfile  # noqa: E402

K = ROOT / "models/logits"
ROWS = 128
names = ("base", "p1", "p2", "p3", "all")
h = kldfile.read_header(K / "pp-base.kld")
lp, var, f32 = {}, {}, []
for n in names:
    raw = np.memmap(K / f"pp-{n}.kld", dtype="<u2", mode="r", offset=h.data_offset, shape=(h.n_chunk, h.n_scored, h.nv))
    block = np.asarray(raw[0, :ROWS])
    par = np.ascontiguousarray(block[:, :4]).view("<f4").astype(np.float64)
    codes = block[:, 4 : 4 + h.n_vocab].astype(np.float64)
    lp[n] = par[:, 1:2] + par[:, 0:1] * codes
    f32.append(float(np.max(np.abs(lp[n] - next(kldfile.iter_blocks(K / f"pp-{n}.kld", ROWS))[2].astype(np.float64)))))
    var[n] = np.where(codes > 0, par[:, 0:1] ** 2 / 12, 0.0)  # code 0 is the floor, exact
    del raw, block, codes
lb = lp["base"]
p = np.exp(lb)
p /= p.sum(1, keepdims=True)
head = lb >= lb.max(1, keepdims=True) - 10.0
res = lp["all"] - lp["p1"] - lp["p2"] - lp["p3"] + 2 * lb
alld = lp["all"] - lb
nres = 4 * var["base"] + var["p1"] + var["p2"] + var["p3"] + var["all"]
nall = var["base"] + var["all"]


def norms(x):
    m = (p * x).sum(1, keepdims=True)
    mh = np.where(head, x, 0).sum(1, keepdims=True) / head.sum(1, keepdims=True)
    return np.array([(p * (x - m) ** 2).sum(), np.where(head, (x - mh) ** 2, 0).sum(), ((x - x.mean(1, keepdims=True)) ** 2).sum()])


def noise(v):  # expected squared norm of the rounding noise alone (centring ignored: an upper bound)
    return np.array([(p * v).sum(), np.where(head, v, 0).sum(), v.sum()])


sr, sa, nr, na = norms(res), norms(alld), noise(nres), noise(nall)
print(f"== chunk 0, rows 0-{ROWS - 1}, 2-bit pass split ==")
print(f"max |float32 decode - float64 decode|: {max(f32):.2e} nats; median writer step "
      f"{np.median([np.median(np.sqrt(12 * var[n][var[n] > 0])) for n in names]):.2e} nats")
for i, nm in enumerate(("fisher", "head_euclid", "full_euclid")):
    print(f"{nm}: rounding-noise share of ||R||^2 {nr[i] / sr[i]:.1e}, of ||all||^2 {na[i] / sa[i]:.1e}; "
          f"r {np.sqrt(sr[i] / sa[i]):.5f}, noise-corrected {np.sqrt(max(sr[i] - nr[i], 0) / (sa[i] - na[i])):.5f}")

print("\n== B2 expert tensors: IQ4_NL base vs pool-3bit (the 3-bit rung's donor) ==")
want = {f"blk.{l}.ffn_{k}_exps.weight" for l in range(6, 12) for k in ("down", "gate_up")}


def digests(path):
    return {t.name: (t.tensor_type.name, hashlib.sha256(memoryview(t.data).cast("B")).hexdigest()[:16])
            for t in gguf.GGUFReader(path).tensors if t.name in want}


a, b = digests(ROOT / "models/Inkling-2-Mini-IQ4_NL.gguf"), digests(ROOT / "models/pool-3bit.gguf")
for kind in ("down", "gate_up"):
    ks = sorted(k for k in a if f"_{kind}_" in k)
    print(f"{kind}: identical in {sum(a[k] == b[k] for k in ks)} of {len(ks)} layers; types base {sorted({a[k][0] for k in ks})}, "
          f"pool-3bit {sorted({b[k][0] for k in ks})}")
