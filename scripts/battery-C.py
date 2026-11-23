"""Amendment 15, component C: task-trace KL texts from the Q8_0 reference's own B replies (before v1 or v2m score them).

    python -I scripts/battery-C.py   (needs results/battery/B-q8.jsonl complete; writes data/battery/C-B<k>.txt)

Same format as T2 (scripts/t2-traces.py): per reply its reasoning, a newline, its answer (B2: the tool call as JSON),
replies joined by blank lines in item order; plain text, because llama-perplexity tokenizes without special tokens
(so not chat-template rendered as amendment 15 first said). Each file is cut to 8 x 1,024 tokens, or fewer whole
chunks if the replies are shorter.
"""

import hashlib
import json
from pathlib import Path

from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent.parent
TOK = Tokenizer.from_file(str(ROOT / "models/Inkling-2-Mini/tokenizer.json"))


def main() -> None:
    recs = [json.loads(x) for x in (ROOT / "results/battery/B-q8.jsonl").read_text().splitlines()]
    out = []
    for t in ("B1", "B2", "B3", "B4"):
        texts = []
        for r in (x for x in recs if x["task"] == t):
            ans = r.get("content") or ""
            if r.get("tool_calls"):
                ans = (ans + "\n" + json.dumps([c.get("function") for c in r["tool_calls"]], ensure_ascii=False)).strip()
            texts.append(((r.get("reasoning_content") or "") + "\n" + ans).strip())
        ids = TOK.encode("\n\n".join(texts) + "\n", add_special_tokens=False).ids
        n = min(8, len(ids) // 1024)
        text = TOK.decode(ids[: n * 1024])
        p = ROOT / f"data/battery/C-{t}.txt"
        p.write_text(text, encoding="utf-8")
        out.append(f"- `{p.relative_to(ROOT)}`: {n} chunks of 1,024 tokens from {len(texts)} replies, sha256 {hashlib.sha256(text.encode()).hexdigest()}")
    (ROOT / "results/battery/C-data.md").write_text("# Amendment 15 component C texts\n\n" + "\n".join(out) + "\n", encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()
