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
