# Council: 48-hour plan

- Date: 2026-11-21
- Model: gpt-6.2-astra
- Prompt: `reviews/2026-11-21-council-prompt.md`

## Reply (verbatim)

## 1. Verdict

**Est. — Choose a hybrid. Spend the next 48 hours on one pass-specific intervention from Plan A, using Plan B’s idea of reusing quantized tensor bytes. Aim to publish the pilot and its reproduction instructions. Defer the model upload.** One reason decides it: holding the physical weights fixed and changing *which visit uses them* answers a question that another 11.7 GB GGUF cannot answer. The intervention also advances the thesis if the measured effect is small.

**V — The competitive picture changed on 21 November.** Unsloth now directs llama.cpp users to a `Mainline_Support` directory containing an 11.7 GB UD-IQ3_XXS and an 8.49 GB UD-IQ2_XXS. **Verify** that the exact `.gguf_file` runs correctly on the pinned stock binary. The Hub also showed new mlx-community oQ3e-MTP and oQ4e-MTP entries updated minutes apart. Their cards describe precision choices by tensor role or source format. [Unsloth mainline files](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF/tree/main/Mainline_Support) · [oQ3e card](https://huggingface.co/mlx-community/Inkling-2-Mini-oQ3e-mtp) · [oQ4e card](https://huggingface.co/mlx-community/Inkling-2-Mini-oQ4e-mtp)

**V — Upstream status.** Release b12600 identifies `29b52d3` and explicitly credits #32694, despite an older PR-page snapshot displaying “Open.” #32704 shows open and concerns missing expert scales on later visits. #32695 shows draft. Record the stock and patched SHAs separately. [Release b12600](https://github.com/ggml-org/llama.cpp/releases/tag/b12600) · [Scale fix](https://github.com/ggml-org/llama.cpp/pull/32704) · [Alternate implementation](https://github.com/ggml-org/llama.cpp/pull/32695)

## 2. What Plan B would establish

**Est. — A five-block P=3 map is useful allocation evidence. Its novelty is narrow.** It tells a builder where a precision change costs the most KL per byte saved on this checkpoint. Its blocks differ in layer count, depth, source precision and visit count. A large L6–11 effect cannot identify which pass mattered or whether errors compounded. Publish raw KL changes alongside bytes saved. **Verify** any priority claim. I found no measured pass-specific intervention in the cards reviewed, and the Hub inventory is changing within hours.

**Est. — Beating InterpreticAI is plausible, with insufficient evidence to call it likely.** Its imatrix build already has a credible starting point. A win matters to someone choosing between those two downloads. Unsloth’s same-size mainline candidate makes InterpreticAI alone an inadequate release bar. **V** — AtomicChat reports 0.0271 KL for its larger 12.58 GB AD-IQ3_XXS against its BF16 conversion. Unsloth reports 0.0269 against UD-Q8_K_XL on different data. Ranking those decimals would be meaningless. A cross-fork comparison needs identical scored tokens, one reference distribution and a measured engine discrepancy. [AtomicChat measurements](https://huggingface.co/AtomicChat/Inkling-2-Mini-GGUF) · [Unsloth measurements](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF)

## 3. The 48-hour deliverable

**Est. — Pick one six-layer core block before seeing logits.** Start with the published 17.6 GB IQ4_NL base. Prepare one lower-precision copy of the selected expert projections from the same source revision. At fixed P=3, point only the selected visit at those tensors. Run baseline, pass 1 only, pass 2 only, pass 3 only and all three. Each pass restarts from the common input and communicates through a gated message, so an early perturbation and a final-pass perturbation have different paths to the logits. [Base tensor recipe](https://huggingface.co/ggml-org/Inkling-2-Mini-GGUF) · [Model architecture](https://huggingface.co/thinkingmachines/Inkling-2-Mini)

**Est. — Cost:** one bounded loader/graph change, one donor block and five serial conditions. Compare the all-pass centered-logit difference with the **vector sum** of the three isolated differences on identical tokens. Report the residual and the repeatability floor. Adding three KL values would answer a different question. This is an exploratory result conditional on an already-quantized base. Leave message ablation, route locking and confirmatory H1 for Plan A.

## 4. Measurement traps

- **V / Verify — Reference and runtime.** Published Q8_0 is 27.8 GB before runtime buffers. Its 0.0030 KL against expanded BF16 was measured on 40×512 WikiText tokens. It supplies neither a universal correction nor a BF16 master. Download it if needed for product comparisons. Measure peak memory, repeated-run drift and stock-versus-patched drift on identical inputs. [ggml-org reference results](https://huggingface.co/ggml-org/Inkling-2-Mini-GGUF)
- **Est. — Sampling.** Sixteen 4,096-token chunks cost about 13.2 GB of reference logits using the supplied scaling. A sweep that selects the allocation consumes development data. Freeze separate evaluation documents, hash them, and check exact and near overlap with bartowski v6 calibration text. Report paired document-level uncertainty and top-1 agreement; correlated tokens exaggerate apparent precision.
- **Verify — Actual tensor bytes.** Inspect types for every 1,152-wide down projection. Q2_0 fits, but stock quantization ignores its imatrix. Check names, dimensions, scales, offsets, alignment, metadata and payload hashes. Require a no-op splice to reproduce baseline logits and all-pass switching to reproduce an ordinary tied-weight build containing the same donor bytes. [AtomicChat quantizer notes](https://huggingface.co/AtomicChat/Inkling-2-Mini-GGUF)
- **Verify — User-facing utility.** Compare exact text-model bytes and itemize mmproj and MTP separately. A 16 GB claim requires measured peak VRAM, context, KV type, offload and throughput on the 3080 Ti. Thirty capped GSM8K prompts supply a smoke test. Record correctness, cap hits, repetition and malformed output against baseline under identical decoding.

## 5. Kill criteria and claim language

**Verify — Withhold the numerical pilot** if visit tracing, no-op parity, all-pass parity or the memory cap fails. If controls pass but the residual sits at the repeatability floor, publish that bounded result. Permitted wording is “On these documents, at P=3, changing these projections on visit *k* produced this measured logit change.” Exclude claims of general compounding, first-ever priority, pass-aware allocation or preserved reasoning ability.

**Est. — For a later model upload**, require student approval, stock-runtime validation and a frozen candidate no larger than the matched baselines. Require paired held-out KL improvement beyond measured replay drift against the same-pipeline uniform build, InterpreticAI and the working same-size Unsloth mainline file. An unresolved comparison earns a map-only release.

## 6. Schedule from now

| Elapsed hours, Est. | Work and decision |
|---|---|
| 0–2 | Student dates the exploratory changelog before data, selects block and projections, freezes documents, metric and controls, and reserves untouched thesis data. |
| 2–6 | Pin revisions, fetch base and donor source, inspect tensor manifests, trace 48 visits and time one chunk. Enforce one heavy run and roughly 30 GB peak RAM. |
| 6–12 | Implement selective tensor binding. Student checks visit identity and scale attachment. |
| 12–20 | Queue no-op, tied-build and repeatability controls serially while the student rests. |
| 20–24 | Student decides whether controls passed. If they failed, spend day two repairing the harness and document the failure. |
| 24–36 | Run the five conditions serially. Stream logit differences by chunk and retain paired summaries. |
| 36–42 | Replay a prespecified subset, inspect residuals and write reproduction commands. |
| 42–48 | Recheck Hub files and PRs. Student approves the scoped pilot note or publishes the validated harness alone. The student retains the later allocation decision. No GCP spend is needed for this plan. |
