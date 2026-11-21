"""analysis/per_pass.py on synthetic logits files (CPU, about 1 MB):
1. every statistic against an independent per-chunk 4x4 Fisher Gram matrix (no polarisation);
2. the bootstrap CI against an independent loop with the same index draw;
3. mutants of _sq_norms and HEAD_NATS run against tests/test_per_pass.py: which tests notice?

    .venv/bin/python -I reviews/2026-11-21-code-review-analysis-opus/check_per_pass.py
"""

import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "analysis"), str(ROOT / "tests"), str(ROOT / "scripts")]
import kldfile  # noqa: E402
import per_pass  # noqa: E402
import test_per_pass  # noqa: E402
from test_kldfile import write_file  # noqa: E402

rng = np.random.default_rng(7)
NC, NCTX, NV = 5, 40, 97
NS = NCTX - 1 - NCTX // 2
tok = rng.integers(0, NV, size=(NC, NCTX))
base = rng.normal(0, 2.5, size=(NC, NS, NV))
base[..., :20] -= 25  # a fifth of the vocabulary below the 16-nat floor
common = rng.normal(0, 0.1, size=base.shape)
d = {k: 0.5 * common + rng.normal(0, 0.1, size=base.shape) for k in ("p1", "p2", "p3")}  # correlated parts
allc = 1.3 * (d["p1"] + d["p2"] + d["p3"]) + rng.normal(0, 0.05, size=base.shape)
tmp = Path(tempfile.mkdtemp())
files = {}
for k, x in (("base", base), *((k, base + d[k]) for k in d), ("all", base + allc)):
    files[k] = tmp / f"{k}.kld"
    write_file(files[k], x.astype(np.float32), tok)

L = {k: np.stack([lp for _, _, lp in kldfile.iter_chunks(p)]).astype(np.float64) for k, p in files.items()}
p = np.exp(L["base"])
p /= p.sum(-1, keepdims=True)
D = np.stack([L[k] - L["base"] for k in ("p1", "p2", "p3", "all")])
Dc = D - (p * D).sum(-1, keepdims=True)
G = np.einsum("acsv,bcsv,csv->abc", Dc, Dc, p)  # [4, 4, chunk]
g = G.sum(-1)
q = lambda v, M: float(v @ M @ v)  # noqa: E731
res_v, ones = np.array([-1.0, -1, -1, 1]), np.array([1.0, 1, 1, 0])
want = {"r": np.sqrt(q(res_v, g) / g[3, 3]), "norm_ratio": np.sqrt(g[3, 3] / q(ones, g)),
        "share_last": np.sqrt(g[2, 2] / g[3, 3]), "visit_multiplier": g[3, 3] / (np.trace(g[:3, :3]) / 3),
        "pass1_over_pass3": np.sqrt(g[0, 0] / g[2, 2]), "pass2_over_pass3": np.sqrt(g[1, 1] / g[2, 2]),
        "cos_12": g[0, 1] / np.sqrt(g[0, 0] * g[1, 1]), "cos_23": g[1, 2] / np.sqrt(g[1, 1] * g[2, 2]),
        "cos_13": g[0, 2] / np.sqrt(g[0, 0] * g[2, 2])}
acc = per_pass.accumulate(files["base"], [files[k] for k in ("p1", "p2", "p3")], files["all"], rows=7)
s = per_pass.summarize(acc, n_boot=2000, seed=3)
print("== 1. Fisher statistics: largest relative difference, per_pass vs independent Gram ==")
print(f"over {len(want)} statistics: {max(abs(s['fisher'][k] - v) / abs(v) for k, v in want.items()):.1e}")
rc = np.sqrt(np.array([q(res_v, G[..., c]) / G[3, 3, c] for c in range(NC)]))
print(f"r per chunk: {float(np.max(np.abs(s['fisher']['r_chunks'] - rc) / rc)):.1e}")
idx = np.random.default_rng(3).integers(0, NC, size=(2000, NC))
br = [np.sqrt(q(res_v, G[..., i].sum(-1)) / G[3, 3, i].sum()) for i in idx]
lo, hi = s["fisher"]["ci95"]["r"]
print(f"== 2. r CI: per_pass [{lo:.12f}, {hi:.12f}], independent [{np.percentile(br, 2.5):.12f}, {np.percentile(br, 97.5):.12f}]")

print("== 3. mutants against tests/test_per_pass.py ==")
orig_sq, orig_head = per_pass._sq_norms, per_pass.HEAD_NATS


def mutant(fisher_centre=True, head_centre=True):
    def f(pp, head, deltas):
        out = {}
        for k, x in deltas.items():
            m = np.sum(pp * x, axis=1, keepdims=True) if fisher_centre else 0.0
            nh = head.sum(axis=1, keepdims=True)
            mh = np.sum(np.where(head, x, 0.0), axis=1, keepdims=True) / nh if head_centre else 0.0
            out[k] = (float(np.sum(pp * (x - m) ** 2)), float(np.sum(np.where(head, (x - mh) ** 2, 0.0))),
                      float(np.sum((x - x.mean(axis=1, keepdims=True)) ** 2)))
        return out
    return f


tests = sorted(n for n in dir(test_per_pass) if n.startswith("test_"))


def caught() -> list[str]:
    hit = []
    for n in tests:
        try:
            getattr(test_per_pass, n)(Path(tempfile.mkdtemp()))
        except AssertionError:
            hit.append(n)
    return hit


print(f"{len(tests)} tests; unmutated, failing: {caught() or 'none'}")
for label, f in (("Fisher norm without centring", mutant(fisher_centre=False)),
                 ("head norm without centring", mutant(head_centre=False))):
    per_pass._sq_norms = f
    print(f"{label}: failing tests {caught() or 'none'}")
per_pass._sq_norms = orig_sq
per_pass.HEAD_NATS = 3.0
print(f"HEAD_NATS 10 -> 3: failing tests {caught() or 'none'}")
per_pass.HEAD_NATS = orig_head

# the shift test: does the shift reach per_pass at all? The writer stores log-softmax, so a per-token constant vanishes.
t = Path(tempfile.mkdtemp())
x = rng.normal(0, 1.5, size=(1, NS, NV)).astype(np.float32)
write_file(t / "a.kld", x, tok[:1])
write_file(t / "b.kld", (x + 3.0).astype(np.float32), tok[:1])
la, lb2 = (next(kldfile.iter_chunks(t / n))[2] for n in ("a.kld", "b.kld"))
print(f"max |logp(x + 3) - logp(x)| after the writer: {float(np.max(np.abs(la - lb2))):.1e} nats "
      "(the +3.0 and -7.0 shifts in test_shift_invariance never reach per_pass)")
