# Link check, 21 Nov 2026

Every external URL in `drafts/note.md` and `reviews/2026-11-21-scoop-check.md` was refetched with curl (follow redirects, 30 s timeout) on 21 Nov 2026. 40 unique URLs; all 40 returned HTTP 200. Titles, dates and states come from the fetched pages or API JSON. Claim checks are regex matches of the quoted phrase or number against the fetched text. Scripts and raw downloads: scratchpad `lc/` (status.tsv, chk.py, q.py). The note was not edited.

Result: 37 live, 3 mismatch (the pages are live, but they changed after the scoop check: #32669, the GitHub repo search count, the HF search count). None dead. Every link in the note's Prior work section is live and matches.

## Links in drafts/note.md (Prior work)

| Item | URL | HTTP | Page title / date | Claim checked | Mark |
|---|---|---|---|---|---|
| LoopQ | arxiv.org/abs/2605.16343 | 200 | "LoopQ: Quantization for Looped Language Models", submitted 8 May 2026 | "recursive error accumulation" found; "W4A4" found | live |
| Feedback Exposure | arxiv.org/abs/2609.30820 | 200 | "Quantizing Looped Transformers: Feedback Exposure and Calibration Blindness", 25 Sep 2026 | "Huginn" found | live |
| Tilted Bowl | arxiv.org/abs/2609.39277 | 200 | "A Tilted Bowl Is Not a Slippery Slope: Compressing Looped Models", 30 Sep 2026 | "33 models from five families" in full text; "fresh" found | live |
| Loopy | arxiv.org/abs/2610.05265 | 200 | "Loopy: Low-Bit Quantization Framework for Looped Language Models", 4 Oct 2026 | "recurrent depth" found | live |
| LoopQuant | arxiv.org/html/2610.16368v1 | 200 | "LoopQuant: Accurate and Efficient Post-Training Quantization for Looped Transformers", page shows 12 Oct 2026 | "Shapley", "stronger-than-additive", "A.3.9" found | live |
| LookAtMyQuant card | huggingface.co/shinmegamiboson/Inkling-2-Mini-LookAtMyQuant-8.9GB | 200 | repo title matches | "48 logical visits", "Bernoulli-KL", "12:48", 0.0208, 0.0189 found | live |

## Further links in reviews/2026-11-21-scoop-check.md

| Item | URL | HTTP | What the fetch shows | Mark |
|---|---|---|---|---|
| HF search API | huggingface.co/api/models?search=Inkling-2-Mini... | 200 | Returns 100 (the limit); scoop check said 88. Newest now Harsha001/Inkling-2-Mini-ROCmFPX, 21 Nov 21:38 UTC | mismatch (count grew) |
| HF quantized listing | huggingface.co/models?other=base_model:quantized:... | 200 | "Quantized Models for thinkingmachines/Inkling-2-Mini"; count of 74 not rechecked | live |
| unsloth GGUF | huggingface.co/unsloth/Inkling-2-Mini-GGUF | 200 | 0.0269 and 94.7 found | live |
| Unsloth guide | unsloth.ai/docs/models/inkling-2-mini | 200 | "Inkling 2 Mini - Run Locally Guide"; "BF16" found | live |
| bartowski | huggingface.co/bartowski/thinkingmachines_Inkling-2-Mini-GGUF | 200 | 0.69 found | live |
| ggml-org | huggingface.co/ggml-org/Inkling-2-Mini-GGUF | 200 | 0.0030 and 0.0095 found | live |
| AtomicChat | huggingface.co/AtomicChat/Inkling-2-Mini-GGUF | 200 | "we have not tested it" and #32695 found | live |
| InterpreticAI IQ3_XXS | huggingface.co/InterpreticAI/Inkling-2-Mini-IQ3_XXS-GGUF | 200 | title matches | live |
| InterpreticAI Q4_K_M | huggingface.co/InterpreticAI/Inkling-2-Mini-GGUF | 200 | title matches | live |
| mudler APEX | huggingface.co/mudler/Inkling-2-Mini-APEX-GGUF | 200 | `29b52d398` found | live |
| Xylolabs | huggingface.co/Xylolabs/Inkling-2-Mini-GGUF | 200 | #32704 found | live |
| hermitdave | huggingface.co/hermitdave/Inkling-2-Mini-MLX-RCO-3.5bpw | 200 | 0.0669 found | live |
| netprojects | huggingface.co/netprojects/Inkling-2-Mini-MLX-2.5bit | 200 | title matches | live |
| araysuter | huggingface.co/araysuter/Inkling-2-Mini-MLX-mixed-2bit | 200 | title matches | live |
| baytizzel | huggingface.co/baytizzel/Inkling-2-Mini-MLX-M5Pro-48gb | 200 | title matches | live |
| LookAtMyQuant GitHub | github.com/ShinMegamiBoson/lookatmyquant | 200 | "A quantization compiler that delivers 12x lower KL than Q4_K_M ..." | live |
| ISTA-DASLab API | huggingface.co/api/models?author=ISTA-DASLab... | 200 | newest Kimi-K3-P48NVFP4-MoESQ, 3 Nov; 0 Inkling repos | live |
| GitHub repo search | api.github.com/search/repositories?q=inkling-2-mini... | 200 | total_count 63 now; 40 items is the page cap. Scoop check said "40 repos" | mismatch (the 40 was the page cap or an old count) |
| arXiv API query | export.arxiv.org/api/query?... looped transformer | 200 | 8 entries; no "Inkling 2" | live |
| Feedback Exposure HTML | arxiv.org/html/2609.30820v1 | 200 | title matches; "Huginn", "nine" found | live |
| Tilted Bowl HTML | arxiv.org/html/2609.39277v1 | 200 | title matches; "33 models" found | live |
| Dory1.0-4B | arxiv.org/html/2611.03998v1 | 200 | "Dory1.0-4B: A Looped Normalized-GPT", 02 Nov 2026; "NVFP4" found | live |
| llama.cpp master Metal source | raw.githubusercontent.com/ggml-org/llama.cpp/master/.../ggml-metal-device.m | 200 | mul_mat check still returns false for F8_E4M3, NVFP4, TQ1_0; GET_ROWS still excludes NVFP4 | live |
| Unsloth fork Metal source | raw.githubusercontent.com/unslothai/llama.cpp/b12588-mix-c0c3de8/... | 200 | GGML_TYPE_NVFP4 still listed in both exclusions (2 matches) | live |
| Unsloth releases | api.github.com/repos/unslothai/llama.cpp/releases | 200 | newest b12588-mix-c0c3de8 | live |
| PR #20456 (web, API) | github.com/.../pull/20456, api .../pulls/20456 | 200, 200 | "metal : add NVFP4 quantization support", open, created 12 Mar, updated 23 Jun, 0 comments, mergeable_state dirty | live |
| Open PRs list | api.github.com/.../pulls?state=open... | 200 | 100 PRs; none titled Metal plus NVFP4 | live |
| PR #32677 | api .../pulls/32677 | 200 | "llama, metal : support BF16/F32 src1 for MUL_MAT and MUL_MAT_ID", open, 20 Nov | live |
| #32610 | api .../issues/32610 | 200 | "CUDA: few-row MMQ for NVFP4", open | live |
| #32704 | api .../issues/32704 | 200 | "model : inkling2_loop re-share the loop slots after the scale pass", merged 21 Nov 11:10:39 UTC | live |
| #32669 | api .../issues/32669 | 200 | "model: apply .scale tensors automatically + fix correctness", **merged 21 Nov 15:30:02 UTC**. Scoop check says "still open" | mismatch (stale) |
| #32710 | api .../issues/32710 | 200 | "Eval bug: inkling2_loop: perplexity nan on CUDA", open | live |
| #32707 | api .../issues/32707 | 200 | "Eval bug: ggml-org/Inkling-2-Mini-GGUF:IQ4_NL prefill is abominably slow and crashes on 5090", open | live |

## Notes

- #32669 merged after the scoop check. Mainline now applies `.scale` tensors on its own. The note does not cite it, so the note needs no change for this.
- The two count mismatches are snapshot drift. If the note ever quotes them, give the date.
- Not checked: the quantized-listing count (74, client-rendered page) and the full LookAtMyQuant card beyond the quoted strings.
