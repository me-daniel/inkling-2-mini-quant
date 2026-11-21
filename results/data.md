# Pilot data

Written by `scripts/prepare-data.py`. Do not edit by hand. Split rules: `CHANGELOG.md`, entry 2026-11-21.

| File | Tokens | sha256 (first 16) | 13-gram overlap with bartowski v6 |
|---|---|---|---|
| `data/S.txt` | 65,792 | `7ab968dccbaa49aa` | 0 of 43,499 (0.000%) |
| `data/T1.txt` | 65,792 | `b92d716fcae5582f` | 0 of 42,630 (0.000%) |
| `data/T2-prompts.jsonl` | 40 prompts | `db381a9a4b500762` | n/a |

Sources (sha256, first 16):
- WikiText-2-raw test parquet `5f1bea067869d048` (Salesforce/wikitext rev b08601e)
- GSM8K main train parquet `ea82612ea9582142` (openai/gsm8k rev 740312a)
- `bartowski-v6_prose.txt` `2a2c5562163fc800` (gist e26453c, version 14d426c)
- `bartowski-v6_conversations.json` `55e576b29563c2c6` (gist e26453c, version 14d426c)

## T3

Written by `scripts/prepare-data.py --t3`. Rule: `CHANGELOG.md`, Addendum H amendment: T3.
T3 = WikiText-2-raw test tokens 131,584 to 197,376 (chunks 33-48 of 4,096 plus the 256-token margin), the span after T1 under the S/T1 slicing rule.

| File | Tokens | sha256 |
|---|---|---|
| `data/T3.txt` | 65,792 | `5a26eea1323fbade49482767b9ea4de14f539262b82135ab0e0e0a917be689a7` |

13-gram overlap of T3 (44,283 distinct 13-grams, lowercased `\w+` words) with:

| Text | sha256 (first 16) | Shared 13-grams | Share of T3 |
|---|---|---|---|
| `data/S.txt` | `7ab968dccbaa49aa` | 0 | 0.000% |
| `data/T1.txt` | `b92d716fcae5582f` | 0 | 0.000% |
| `data/T2.txt` (reference traces) | `5aebb8fea2fcc9ed` | 0 | 0.000% |
| `data/T2-prompts.jsonl` (questions) | `db381a9a4b500762` | 0 | 0.000% |
| `bartowski-v6_prose.txt` (calibration) | `2a2c5562163fc800` | 0 | 0.000% |
| `bartowski-v6_conversations.json` (calibration) | `55e576b29563c2c6` | 0 | 0.000% |
| calibration text, both files | n/a | 0 | 0.000% |

Source: WikiText-2-raw test parquet `5f1bea067869d048` (Salesforce/wikitext rev b08601e), already on disk; no download.
