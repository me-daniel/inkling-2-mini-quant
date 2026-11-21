"""Builds the pilot data fixed in CHANGELOG.md (2026-11-21) and writes results/data.md.

S  = WikiText-2-raw test, chunks 1-16 of 4,096 tokens (model tokenizer)
T1 = WikiText-2-raw test, chunks 17-32
T2 = prompts: GSM8K train questions 1-20 and 51-70 (traces are generated later by the reference; 21-50 are the stability battery)
Also reports 13-gram overlap of S and T1 with bartowski's v6 calibration text (the imatrix source).

--t3 builds only T3 (CHANGELOG.md, Addendum H amendment: T3) and leaves S, T1 and T2 untouched:
T3 = WikiText-2-raw test, chunks 33-48, the next span after T1 under the same slicing rule.
It writes data/T3.txt and replaces the T3 section of results/data.md (13-gram overlap against S, T1,
T2 and the calibration text). A plain run keeps an existing T3 section.
"""

import argparse
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
T3_HEAD = "## T3"

WIKI = RAW / "wikitext/wikitext-2-raw-v1/test-00000-of-00001.parquet"
GSM8K = RAW / "gsm8k/main/train-00000-of-00001.parquet"
CALIB = [RAW / "bartowski-v6_prose.txt", RAW / "bartowski-v6_conversations.json"]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ngrams(text: str, n: int = 13) -> set[tuple[str, ...]]:
    words = re.findall(r"\w+", text.lower())
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


def build_t3(tok: Tokenizer, ids: list[int]) -> None:
    span = CHUNK * N_CHUNKS + MARGIN
    if len(ids) < 3 * span:
        raise SystemExit(f"WikiText-2 test has {len(ids)} tokens, need {3 * span}")
    start, end = 2 * span, 3 * span
    path = OUT / "T3.txt"
    path.write_text(tok.decode(ids[start:end]), encoding="utf-8")
    n = len(tok.encode(path.read_text(encoding="utf-8"), add_special_tokens=False).ids)
    if n < CHUNK * N_CHUNKS:
        raise SystemExit(f"T3: only {n} tokens after re-tokenizing")

    t2_prompts = OUT / "T2-prompts.jsonl"
    prompts = "\n".join(json.loads(line)["question"] for line in t2_prompts.read_text(encoding="utf-8").splitlines())
    others = [
        ("`data/S.txt`", (OUT / "S.txt").read_text(encoding="utf-8"), sha(OUT / "S.txt")),
        ("`data/T1.txt`", (OUT / "T1.txt").read_text(encoding="utf-8"), sha(OUT / "T1.txt")),
        ("`data/T2.txt` (reference traces)", (OUT / "T2.txt").read_text(encoding="utf-8"), sha(OUT / "T2.txt")),
        ("`data/T2-prompts.jsonl` (questions)", prompts, sha(t2_prompts)),
        *[(f"`{p.name}` (calibration)", p.read_text(encoding="utf-8"), sha(p)) for p in CALIB],
    ]
    grams = ngrams(path.read_text(encoding="utf-8"))
    calib = set().union(*(ngrams(text) for label, text, _ in others if "calibration" in label))
    rows = [(label, len(grams & ngrams(text)), digest) for label, text, digest in others]
    rows.append(("calibration text, both files", len(grams & calib), ""))

    lines = [
        T3_HEAD,
        "",
        "Written by `scripts/prepare-data.py --t3`. Rule: `CHANGELOG.md`, Addendum H amendment: T3.",
        f"T3 = WikiText-2-raw test tokens {start:,} to {end:,} (chunks 33-48 of 4,096 plus the {MARGIN}-token margin), "
        "the span after T1 under the S/T1 slicing rule.",
        "",
        "| File | Tokens | sha256 |",
        "|---|---|---|",
        f"| `data/{path.name}` | {n:,} | `{sha(path)}` |",
        "",
        f"13-gram overlap of T3 ({len(grams):,} distinct 13-grams, lowercased `\\w+` words) with:",
        "",
        "| Text | sha256 (first 16) | Shared 13-grams | Share of T3 |",
        "|---|---|---|---|",
        *[f"| {label} | {f'`{d[:16]}`' if d else 'n/a'} | {hit:,} | {hit / len(grams):.3%} |" for label, hit, d in rows],
        "",
        f"Source: WikiText-2-raw test parquet `{sha(WIKI)[:16]}` (Salesforce/wikitext rev b08601e), already on disk; no download.",
        "",
    ]
    report = ROOT / "results/data.md"
    old = report.read_text(encoding="utf-8")
    head = old.split("\n" + T3_HEAD, 1)[0].rstrip("\n") + "\n"
    report.write_text(head + "\n" + "\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--t3", action="store_true", help="build only T3 and its section of results/data.md")
    args = ap.parse_args()

    tok = Tokenizer.from_file(str(ROOT / "models/Inkling-2-Mini/tokenizer.json"))
    wiki = "".join(pq.read_table(WIKI).column("text").to_pylist())
    ids = tok.encode(wiki, add_special_tokens=False).ids
    if args.t3:
        build_t3(tok, ids)
        return

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

    qs = pq.read_table(GSM8K).column("question").to_pylist()
    questions = qs[:20] + qs[50:70]
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
    lines.append(f"| `data/{t2.name}` | 40 prompts | `{sha(t2)[:16]}` | n/a |")
    lines += [
        "",
        "Sources (sha256, first 16):",
        f"- WikiText-2-raw test parquet `{sha(WIKI)[:16]}` (Salesforce/wikitext rev b08601e)",
        f"- GSM8K main train parquet `{sha(GSM8K)[:16]}` (openai/gsm8k rev 740312a)",
        *[f"- `{p.name}` `{sha(p)[:16]}` (gist e26453c, version 14d426c)" for p in CALIB],
        "",
    ]
    report = ROOT / "results/data.md"
    old = report.read_text(encoding="utf-8") if report.exists() else ""
    t3 = old.split("\n" + T3_HEAD, 1)[1] if "\n" + T3_HEAD in old else None
    report.write_text("\n".join(lines) + ("\n" + T3_HEAD + t3 if t3 is not None else ""), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
