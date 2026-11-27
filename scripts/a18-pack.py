"""Amendment 18 step 0: every file must carry the same tokenizer, chat template and vocabulary as our Q8_0 reference,
so that KL measures the weights, not the packaging. Also prints loop count, file bytes and tensor count.

    python -I scripts/a18-pack.py REF.gguf NAME=FILE.gguf ...   (exit 1 on any mismatch; writes nothing)
"""

import hashlib
import sys
from pathlib import Path

from gguf import GGUFReader

KEYS = ("tokenizer.ggml.tokens", "tokenizer.ggml.merges", "tokenizer.ggml.token_type", "tokenizer.chat_template",
        "tokenizer.ggml.bos_token_id", "tokenizer.ggml.eos_token_id", "inkling2_loop.vocab_size")


def field_hash(r: GGUFReader, key: str) -> str:
    f = r.fields.get(key)
    if f is None:
        return "missing"
    h = hashlib.sha256()
    for i in f.data:
        h.update(bytes(f.parts[i]))
    return h.hexdigest()[:16]


def info(path: str) -> dict:
    r = GGUFReader(path)
    loop = r.fields.get("inkling2_loop.loop.count")
    return {"hash": {k: field_hash(r, k) for k in KEYS}, "loop": int(loop.parts[loop.data[0]][0]) if loop else None,
            "bytes": Path(path).stat().st_size, "tensors": len(r.tensors)}


def main() -> None:
    ref = info(sys.argv[1])
    bad = 0
    print(f"reference {sys.argv[1]}: loop {ref['loop']}, {ref['tensors']} tensors")
    for spec in sys.argv[2:]:
        name, path = spec.split("=", 1)
        x = info(path)
        diff = [k for k in KEYS if x["hash"][k] != ref["hash"][k]]
        bad += bool(diff)
        print(f"{name}: {x['bytes']:,} B, {x['tensors']} tensors, loop {x['loop']}, "
              + ("tokenizer/template/vocab identical" if not diff else "DIFFERS: " + ", ".join(diff)))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
