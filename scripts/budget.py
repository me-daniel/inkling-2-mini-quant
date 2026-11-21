"""Prices an allocation in exact bytes before building it.

    python scripts/budget.py ALLOC.json            # bytes of the GGUF that allocation would produce
    python scripts/budget.py --check FILE.gguf ... # compare the model with real files' tensor bytes

ALLOC.json: {"default": "Q8_0", "rules": [["regex", "TYPE"], ...]} applied to the tensors of the BF16 conversion,
first matching rule wins, mirroring llama-quantize --tensor-type. 1-D tensors and the F32 loop tensors stay as
they are in the reference files (taken from the Q8_0 reference), since llama-quantize does not quantize them.
"""

import json
import re
import sys
from pathlib import Path

import gguf
from gguf.constants import GGML_QUANT_SIZES, GGMLQuantizationType as Q

ROOT = Path(__file__).resolve().parent.parent
REF = ROOT / "models/Inkling-2-Mini-Q8_0.gguf"


def tensor_bytes(n_elements: int, qtype: Q) -> int:
    block, size = GGML_QUANT_SIZES[qtype]
    assert n_elements % block == 0, (n_elements, qtype)
    return n_elements // block * size


def price(alloc: dict) -> int:
    rules = [(re.compile(rx), Q[t.upper()]) for rx, t in alloc["rules"]]
    default = Q[alloc["default"].upper()]
    total = 0
    for t in gguf.GGUFReader(REF).tensors:
        if t.tensor_type in (Q.F32, Q.F16, Q.BF16):  # kept as is by llama-quantize
            total += int(t.n_bytes)
            continue
        qtype = next((q for rx, q in rules if rx.search(t.name)), default)
        total += tensor_bytes(int(t.n_elements), qtype)
    return total


def check(paths: list[str]) -> None:
    for p in paths:
        r = gguf.GGUFReader(p)
        real = sum(int(t.n_bytes) for t in r.tensors)
        model = sum(tensor_bytes(int(t.n_elements), t.tensor_type) for t in r.tensors)
        size = Path(p).stat().st_size
        print(f"{Path(p).name}: tensor bytes {real:,}, formula {model:,}, file {size:,} (overhead {size - real:,})")


if __name__ == "__main__":
    if sys.argv[1] == "--check":
        check(sys.argv[2:])
    else:
        print(price(json.loads(Path(sys.argv[1]).read_text())))
