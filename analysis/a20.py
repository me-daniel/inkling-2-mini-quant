"""Addendum H amendment 20: the scale-refinement pilot (arms A codes only, B codes + scales; layers 6 and 20).

    python -I analysis/a20.py   (models/refined/a20-{A,B}-6-20.gguf.json, logs/ppl-a20-{A,B}-S-P3.log)

Offline: per layer, J under G summed over the refined experts (first sweep's JG_before, last sweep's JG_after), from
the refiner's JSON logs. Online: KL(B) - KL(A) on S, paired over chunks (analysis/a19.py's paired_delta).
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "analysis"))
from a19 import paired_delta  # noqa: E402
from maptable import parse  # noqa: E402


def j_sums(path: Path) -> dict:
    log = json.loads(path.read_text())
    out = {}
    for layer, v in log["layers"].items():
        ex = [r for r in v["experts"] if r["sweeps"]]
        out[layer] = {"before": sum(r["sweeps"][0]["JG_before"] for r in ex),
                      "after": sum(r["sweeps"][-1]["JG_after"] for r in ex), "n": len(ex),
                      "scale_moves": sum(s.get("scale_moves", 0) for r in ex for s in r["sweeps"]),
                      "d_moves": sum(s.get("d_moves", 0) for r in ex for s in r["sweeps"])}
    return out


def main() -> None:
    a = j_sums(ROOT / "models/refined/a20-A-6-20.gguf.json")
    b = j_sums(ROOT / "models/refined/a20-B-6-20.gguf.json")
    lines = ["# Amendment 20: scale-refinement pilot (layers 6 and 20)", "",
             "Built by `analysis/a20.py`. J under G summed over refined experts; both arms start from v2c's codes.", "",
             "| layer | experts | J start | J after A | J after B | B / A | scale moves (B) | d moves (B) |",
             "|---|---|---|---|---|---|---|---|"]
    ok = True
    for layer in a:
        r = b[layer]["after"] / a[layer]["after"]
        ok &= r <= 0.98
        lines.append(f"| {layer} | {a[layer]['n']} | {a[layer]['before']:.6g} | {a[layer]['after']:.6g} | "
                     f"{b[layer]['after']:.6g} | {r:.4f} | {b[layer]['scale_moves']:,} | {b[layer]['d_moves']:,} |")
    ka = parse(ROOT / "logs/ppl-a20-A-S-P3.log")
    kb = parse(ROOT / "logs/ppl-a20-B-S-P3.log")
    d, lo, hi = paired_delta(np.array(kb["per_chunk"]), np.array(ka["per_chunk"]))
    lines += ["", f"KL on S: A {ka['mean_kl']:.5f}, B {kb['mean_kl']:.5f}; B - A {d:+.5f} [{lo:+.5f}, {hi:+.5f}]", ""]
    if ok and d < 0:
        verdict = "J at least 2% lower in both layers and KL lower: register the full-file run."
    elif ok:
        verdict = "J at least 2% lower in both layers but KL not lower: the local objective does not carry; T2 stops."
    else:
        verdict = "J not 2% lower in both layers: T2 stops."
    lines.append(f"Registered rule: {verdict}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
