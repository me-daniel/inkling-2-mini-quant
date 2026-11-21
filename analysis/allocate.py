"""Allocation for the release candidate (Addenda A, C, D). Written by Claude at Daniel's request; Daniel reviews.

Model (the linearity theorem, arXiv 2411.17525): the KL of a build is the sum over blocks of the KL each block's
choice adds on its own (measured in the block map against the Q8_0 reference), plus the non-expert choice.
With 5 blocks and a handful of levels every combination is enumerated, so the optimum is exact under the model.
Bytes come from scripts/budget.py on the same rules the build will use. Prints the best allocations under the budget.
"""

from __future__ import annotations

import csv
import itertools
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BLOCKS = {"B1": "0|1|2|3|4|5", "B2": "6|7|8|9|10|11", "B3": "12|13|14|15|16|17", "B4": "18|19", "B5": "20|21|22|23"}
# level -> (gate_up type, down type); "3bit" etc. as measured in the map
LEVELS = {"3bit": ("IQ3_XXS", "IQ4_NL"), "gu2s": ("IQ2_S", "IQ4_NL"), "gu2xxs": ("IQ2_XXS", "IQ4_NL"), "2bit": ("IQ2_XXS", "Q2_0")}
BUDGET = 11_663_310_784 - 11_977_760  # Unsloth UD-IQ3_XXS file bytes minus the constant GGUF overhead


def measured() -> dict[str, float]:
    rows = csv.DictReader(open(ROOT / "results/pilot/kl.tsv"), delimiter="\t")
    return {r["run"]: float(r["mean_kl"]) for r in rows}


def rules(choice: dict[str, str], ne: str) -> dict:
    out = []
    for b, lvl in choice.items():
        gu, dn = LEVELS[lvl]
        out += [[rf"^blk\.({BLOCKS[b]})\.ffn_gate_up_exps\.weight$", gu], [rf"^blk\.({BLOCKS[b]})\.ffn_down_exps\.weight$", dn]]
    return {"default": ne, "rules": out}


sys.path.insert(0, str(ROOT / "scripts"))
import budget  # noqa: E402

_TENSORS: list | None = None


def price(alloc: dict) -> int:
    """Same arithmetic as scripts/budget.py, on a tensor list read once."""
    global _TENSORS
    if _TENSORS is None:
        _TENSORS = [(t.name, int(t.n_elements), t.tensor_type, int(t.n_bytes)) for t in budget.gguf.GGUFReader(budget.REF).tensors]
    rules = [(budget.re.compile(rx), budget.Q[t.upper()]) for rx, t in alloc["rules"]]
    default = budget.Q[alloc["default"].upper()]
    total = 0
    for name, n, qtype, nbytes in _TENSORS:
        if qtype in (budget.Q.F32, budget.Q.F16, budget.Q.BF16):
            total += nbytes
            continue
        total += budget.tensor_bytes(n, next((q for rx, q in rules if rx.search(name)), default))
    return total


def main(top: int = 8) -> None:
    kl = measured()
    ne_cost = {"Q8_0": 0.0, "Q6_K": kl.get("map-NE-Q6K-S-P3", float("nan"))}
    avail = {b: [l for l in LEVELS if f"map-{b}-{l}-S-P3" in kl] for b in BLOCKS}
    results = []
    for combo in itertools.product(*avail.values()):
        choice = dict(zip(BLOCKS, combo))
        for ne, ne_kl in ne_cost.items():
            if ne_kl != ne_kl:  # not measured
                continue
            pred = ne_kl + sum(kl[f"map-{b}-{l}-S-P3"] for b, l in choice.items())
            results.append((pred, choice, ne))
    results.sort(key=lambda r: r[0])
    shown = 0
    for pred, choice, ne in results:
        alloc = rules(choice, ne)
        b = price(alloc)
        if b > BUDGET:
            continue
        print(f"predicted KL {pred:.5f}  bytes {b:,} ({(BUDGET - b) / 1e6:.0f} MB spare)  NE {ne}  {choice}")
        if shown == 0:
            (ROOT / "results/pilot/allocation-best.json").write_text(json.dumps({"predicted_kl": pred, "choice": choice, "ne": ne, "bytes": b, "rules": alloc}, indent=1))
        shown += 1
        if shown >= top:
            break


if __name__ == "__main__":
    main()
