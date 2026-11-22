"""Builds data/calib-wt.txt for Addendum H amendment 5 (cal-wt): calibration text from WikiText-2-raw TRAIN, disjoint
from the test split (S, T1, T3) and the validation split (stability prompt).

Fixed rule, no choosing: join the train rows exactly as scripts/prepare-data.py joins the test rows, tokenize with the
model tokenizer, keep the first 300 x 512 tokens. Writes results/calib-wt.md with the token count, sha256 and the
13-gram overlap (lowercased \\w+ words) against S, T1, T2 and T3.

    python scripts/calib-wt.py
"""

import hashlib
import importlib.util
from pathlib import Path

import pyarrow.parquet as pq
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent.parent
TRAIN = ROOT / "data/raw/wikitext/wikitext-2-raw-v1/train-00000-of-00001.parquet"
N_TOK = 300 * 512

spec = importlib.util.spec_from_file_location("prep", ROOT / "scripts/prepare-data.py")
prep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prep)


def main() -> None:
    text = "".join(pq.read_table(TRAIN).column("text").to_pylist())
    tok = Tokenizer.from_file(str(ROOT / "models/Inkling-2-Mini/tokenizer.json"))
    ids = tok.encode(text, add_special_tokens=False).ids
    out = tok.decode(ids[:N_TOK])
    path = ROOT / "data/calib-wt.txt"
    path.write_text(out, encoding="utf-8")
    n = len(tok.encode(out, add_special_tokens=False).ids)
    grams = prep.ngrams(out)
    rows = []
    for name in ("S.txt", "T1.txt", "T2.txt", "T3.txt"):
        p = ROOT / "data" / name
        if p.exists():
            rows.append(f"| {name} | {len(grams & prep.ngrams(p.read_text(encoding='utf-8'))):,} |")
    md = ["# calib-wt (Addendum H amendment 5)", "",
          f"- Source: WikiText-2-raw train parquet `{prep.sha(TRAIN)[:16]}` (Salesforce/wikitext rev b08601e).",
          f"- Rule: first {N_TOK:,} tokens of the joined train rows (model tokenizer); re-tokenized length {n:,}.",
          f"- `data/calib-wt.txt` sha256 `{hashlib.sha256(path.read_bytes()).hexdigest()}`, {len(grams):,} distinct 13-grams.", "",
          "| Text | Shared 13-grams with calib-wt |", "|---|---|"] + rows
    (ROOT / "results/calib-wt.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
