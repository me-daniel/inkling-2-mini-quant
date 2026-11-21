# Unslop: Hugging Face card

- Date: 2026-11-21
- Model: GPT-6.2 Astra, effort medium (Codex CLI, isolated, read-only)
- Input: drafts/release/README.md rendered by scripts/card.py at 833d57d
- Applied: all wording, except the heading (renamed "How we made and tested it") and the reasoning-budget sentence (restored to what the flag does).

## Prompt

You are editing a Hugging Face model card so it reads like a careful human wrote it, not an AI. Apply these rules ("unslop"):
- Remove AI patterns: significance inflation, promotional words, superficial -ing phrases, AI vocabulary (additionally, crucial, delve, enhance, showcase, underscore, pivotal, landscape...), "serves as / features / boasts", negative parallelisms ("not just X, it's Y"), forced groups of three, synonym cycling, false ranges, filler, hedging, generic conclusions, chatbot phrases.
- Style: no em or en dashes; no colon used to join two halves of a sentence (a colon before a list or code block is fine); no inline bold headers at the start of bullets ("**Runs fully on the Mac GPU.** ..."), turn those into plain sentences or short prose; sentence-case headings; straight quotes.
- Add voice: vary sentence rhythm, be specific, first person plural "we" is fine, keep it plain and direct. Short is better.
Hard constraints:
- Keep every number, file name, model name, link, image, table and code block exactly as they are. Do not add new facts or claims.
- Keep the YAML front matter unchanged.
- Never use the words best, fastest, first, beats, same size, loop-aware. Keep claims as scoped as the original (tested set, date, Mac only).
- Keep the meaning that Unsloth's file is closer to the original but slower on a Mac at the moment, and that our registered test against it failed.
Return only the full rewritten Markdown file, nothing else.

--- FILE: README.md ---

## Reply (verbatim)

---
base_model: thinkingmachines/Inkling-2-Mini
base_model_relation: quantized
license: apache-2.0
tags:
- gguf
- llama.cpp
- inkling2_loop
- apple-silicon
---

# Inkling 2 Mini, Metal GGUF (11.5 GB)

This GGUF keeps the whole model on Metal. Every expert tensor uses a type the Mac GPU can run. Unsloth's UD-IQ3_XXS has 20 NVFP4 expert tensors that currently fall back to the CPU on a Mac.

## How we chose the layout

We measured the cost of shrinking each of 5 layer groups, including the core that runs three times. We chose the layout from 4 candidates we tested.

Before release, we registered a quality test against Unsloth's file. Our file failed that test. We then compared it with other all-Metal GGUFs of similar size on held-out text.

## Results

Lower KL on held-out text means the file is closer to the original model. We measured prompt and generation speeds on an Apple M5 Pro.

| File | Size | KL | Fully on Metal | Prompt t/s | Generation t/s |
|---|---|---|---|---|---|
| **This file** | 11.51 GB | 0.02686 | yes | 224 | 19.0 |
| AtomicChat AD-IQ2_S | 11.56 GB | 0.03497 | yes | 236 | 18.3 |
| Unsloth UD-IQ3_XXS | 11.66 GB | 0.01939 | no | 68 | 10.9 |

![Held-out KL of every file we compared](figures/mac-comparison.png)

![Speed against KL on the Mac](figures/speed-vs-kl.png)

Of the 4 all-Metal GGUFs under 11.7 GB we tested on 21 Nov 2026, this file had the lowest KL. Unsloth's file was closer to the original, but on our Mac it was 3.3× slower at reading prompts and 1.75× slower at generating text. For all files, both test sets, confidence intervals and stability checks, see [DETAILS.md](DETAILS.md).

## Use

Use a llama.cpp build with Inkling 2 support (upstream after 20 Nov 2026).

```sh
llama-cli -m Inkling-2-Mini-Metal.gguf -ngl 99 --jinja -c 32768 --temp 1.0 --reasoning-budget 16384
```

Temperature 1.0 is the release default. The reasoning budget caps how long the model can spend in the reasoning loops described in the base model card. Stock upstream llama.cpp (`c661033`) gives identical scores.

## Research background

We made this file as part of a master's thesis pilot on looped models. Inkling 2 Mini runs layers 6 to 17 three times with the same weights. We enabled quantization error on one pass at a time to measure how the errors add up across passes. The report covers that experiment. We chose the file's layout independently of it.

## Limits

KL measures how closely a file matches the reference model's output distribution. Task quality needs a separate test.

We used one machine, one llama.cpp build and one importance matrix for the comparison. It covers the files we tested that day. On CUDA, or once Metal supports NVFP4, Unsloth's UD-IQ3_XXS is the better choice.

## Provenance

We used `thinkingmachines/Inkling-2-Mini` at revision `ed40e7d` and the importance matrix from `ggml-org/Inkling-2-Mini-GGUF`.

We used llama.cpp `29b52d3` (b12600) with 7 patches for measurement. The file runs without those patches.

Scripts computed every number from the run logs. For full tables, settings and file revisions, see [DETAILS.md](DETAILS.md).
