"""Builds data/calib-mix.txt for the amendment 13 Grams: WikiText-2 train (data/calib-wt.txt), the v6 calibration
conversations (rendered with the model's own chat template, as the file's format notes ask) and the v6 prose,
interleaved in blocks of about 6,000 characters, round robin, so the first 300 chunks of 512 tokens hold roughly
equal parts of each. The conversations carry special-token strings: llama-imatrix needs --parse-special.

    python -I scripts/calib-mix.py          (prints block counts, characters per source and the sha256)
    python -I scripts/calib-mix.py --code   (amendment 16: data/calib-mixc.txt, the same three sources plus
                                             data/calib-code.txt (scripts/a16-data.py) as a fourth, same round robin)
"""

import hashlib
import json
import re
import sys
from pathlib import Path

import gguf
from jinja2.sandbox import ImmutableSandboxedEnvironment

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data/raw"
BLOCK = 6000


def blocks(text: str) -> list[str]:
    out, cur = [], []
    n = 0
    for line in text.splitlines(keepends=True):
        cur.append(line)
        n += len(line)
        if n >= BLOCK:
            out.append("".join(cur))
            cur, n = [], 0
    if cur:
        out.append("".join(cur))
    return out


def template() -> str:
    f = gguf.GGUFReader(ROOT / "models/cand.gguf").fields["tokenizer.chat_template"]
    # {% generation %} marks assistant tokens for training masks (a transformers extension jinja2 lacks); it emits nothing
    return re.sub(r"\{%-?\s*(end)?generation\s*-?%\}", "", bytes(f.parts[f.data[0]]).decode())


def raise_exception(msg: str):
    raise ValueError(msg)


def conversations() -> list[str]:
    d = json.loads((RAW / "bartowski-v6_conversations.json").read_text(encoding="utf-8"))
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    env.globals["raise_exception"] = raise_exception
    # the template calls tojson(sort_keys=..., separators=...) as transformers allows; jinja2's own filter takes only indent
    env.filters["tojson"] = lambda x, indent=None, sort_keys=False, separators=None: json.dumps(
        x, indent=indent, sort_keys=sort_keys, separators=separators, ensure_ascii=False)
    tpl = env.from_string(template())
    return [tpl.render(messages=c["messages"], tools=c.get("tools") or None, add_generation_prompt=False)
            for c in d["conversations"] + d["extension_conversations"]]


def main() -> None:
    src = {
        "wikitext-train": blocks((ROOT / "data/calib-wt.txt").read_text(encoding="utf-8")),
        "v6-conversations": [b for c in conversations() for b in blocks(c + "\n")],
        "v6-prose": blocks((RAW / "bartowski-v6_prose.txt").read_text(encoding="utf-8")),
    }
    code = "--code" in sys.argv[1:]
    if code:
        src["code"] = blocks((ROOT / "data/calib-code.txt").read_text(encoding="utf-8"))
    order, i = [], 0
    while any(i < len(v) for v in src.values()):
        order += [v[i] for v in src.values() if i < len(v)]
        i += 1
    text = "".join(order)
    out = ROOT / ("data/calib-mixc.txt" if code else "data/calib-mix.txt")
    out.write_text(text, encoding="utf-8")
    for k, v in src.items():
        print(f"{k}: {len(v)} blocks, {sum(map(len, v)):,} characters")
    print(f"{out.relative_to(ROOT)}: {len(text):,} characters, sha256 {hashlib.sha256(text.encode()).hexdigest()}")


if __name__ == "__main__":
    main()
