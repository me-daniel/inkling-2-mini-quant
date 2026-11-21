"""Builds the pilot data fixed in CHANGELOG.md (2026-11-21) and writes results/data.md.

S  = WikiText-2-raw test, chunks 1-16 of 4,096 tokens (model tokenizer)
T1 = WikiText-2-raw test, chunks 17-32
T2 = prompts: the first 20 GSM8K train questions (traces are generated later by the reference)
Also reports 13-gram overlap of S and T1 with bartowski's v6 calibration text (the imatrix source).
"""

import hashlib
import json
import re
from pathlib import Path

import pyarrow.parquet as pq
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data"
CHUNK, N_CHUNKS = 4096, 16
# llama-perplexity re-tokenizes each file, so each split carries a small margin past its 16 chunks
MARGIN = 256

WIKI = RAW / "wikitext/wikitext-2-raw-v1/test-00000-of-00001.parquet"
GSM8K = RAW / "gsm8k/main/train-00000-of-00001.parquet"
CALIB = [RAW / "bartowski-v6_prose.txt", RAW / "bartowski-v6_conversations.json"]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ngrams(text: str, n: int = 13) -> set[tuple[str, ...]]:
    words = re.findall(r"\w+", text.lower())
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


def main() -> None:
    tok = Tokenizer.from_file(str(ROOT / "models/Inkling-2-Mini/tokenizer.json"))
    wiki = "".join(pq.read_table(WIKI).column("text").to_pylist())
    ids = tok.encode(wiki, add_special_tokens=False).ids

    span = CHUNK * N_CHUNKS + MARGIN
    if len(ids) < 2 * span:
        raise SystemExit(f"WikiText-2 test has {len(ids)} tokens, need {2 * span}")
    splits = {"S": ids[:span], "T1": ids[span : 2 * span]}

    rows = []
    for name, part in splits.items():
        path = OUT / f"{name}.txt"
        path.write_text(tok.decode(part), encoding="utf-8")
        retok = len(tok.encode(path.read_text(encoding="utf-8"), add_special_tokens=False).ids)
        if retok < CHUNK * N_CHUNKS:
            raise SystemExit(f"{name}: only {retok} tokens after re-tokenizing")
        rows.append((name, path, retok))

    questions = pq.read_table(GSM8K).column("question").to_pylist()[:20]
    t2 = OUT / "T2-prompts.jsonl"
    t2.write_text("".join(json.dumps({"id": i, "question": q}) + "\n" for i, q in enumerate(questions)), encoding="utf-8")

    calib = set().union(*(ngrams(p.read_text(encoding="utf-8")) for p in CALIB))
    overlap = {}
    for name, path, _ in rows:
        grams = ngrams(path.read_text(encoding="utf-8"))
        overlap[name] = (len(grams & calib), len(grams))

    lines = [
        "# Pilot data",
        "",
        "Written by `scripts/prepare-data.py`. Do not edit by hand. Split rules: `CHANGELOG.md`, entry 2026-11-21.",
        "",
        "| File | Tokens | sha256 (first 16) | 13-gram overlap with bartowski v6 |",
        "|---|---|---|---|",
    ]
    for name, path, n in rows:
        hit, total = overlap[name]
        lines.append(f"| `data/{path.name}` | {n:,} | `{sha(path)[:16]}` | {hit:,} of {total:,} ({hit / total:.3%}) |")
    lines.append(f"| `data/{t2.name}` | 20 prompts | `{sha(t2)[:16]}` | n/a |")
    lines += [
        "",
        "Sources (sha256, first 16):",
        f"- WikiText-2-raw test parquet `{sha(WIKI)[:16]}` (Salesforce/wikitext rev b08601e)",
        f"- GSM8K main train parquet `{sha(GSM8K)[:16]}` (openai/gsm8k rev 740312a)",
        *[f"- `{p.name}` `{sha(p)[:16]}` (gist e26453c, version 14d426c)" for p in CALIB],
        "",
    ]
    (ROOT / "results/data.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
