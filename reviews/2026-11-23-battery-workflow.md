# Test battery research and design (workflow, 23 Nov 2026)

- Three Opus 5.5 research agents (tasks, KL domains, statistics; web), a Fable 5.2 (medium) design, an Opus critique, a Fable (medium) revision. Prompts held public facts only.
- Registered form: CHANGELOG, Addendum H amendment 15. Links in the reports are the agents' own; only those cited in the amendment were opened again.

## Final design

## Critique points: accept or reject

1. Accept: Q8 generation is budgeted at 1.8x, and Q8 now generates on half the items only.
2. Accept: B1 length and cap come from the existing T2 traces; I plan on 450 tokens per item until then.
3. Accept, conditionally: `-np 4` is used only if the pilot measures at least 1.6x aggregate throughput; the 2-3x claim is not relied on.
4. Accept: D runs only under `-np 4`; at `-np 1` it is dropped.
5. Accept: language and code/LaTeX share of the calibration file is measured before freezing, with a 2% demotion threshold.
6. Accept: cluster bootstrap by source document, 8 documents per domain.
7. Accept: the primary statistic is the mean over domains of per-domain geometric-mean ratios.
8. Accept: the pass bar is a CI upper bound of 0.90 or less; 0.90 to 1.0 is partial transfer.
9. Accept: H1 and H3 are co-primary (both must pass); H2 and per-domain checks are secondary.
10. Accept: B4 is scored by schema validity.
11. Accept as secondary, provided the pilot confirms per-item scores can be logged.
12. Accept: raw completions are logged and parsed by one frozen parser; reasoning-off is checked in the pilot.
13. Accept: H3 uses correctness against ground truth; agreement with Q8 is secondary.
14. Accept: H3 excludes B2. This costs power, stated below.
15. Accept as descriptive only: 8 chunks is too few for a test.
16. Accept: logit size is computed from the real vocabulary and free disk is checked.
17. Accept in part: half the code documents come from pre-2023 human repos. Pretraining exposure does not bias KL against Q8, so "after cutoff" is no longer required.
18. Accept: EvalPlus runs in Docker (CPU, not counted as GPU time).

# v1 vs v2m transfer battery (final design for pre-registration)

All times are estimates from stated speeds (220 tok/s scoring, 18.5 tok/s generation, Q8 generation at 1.8x and scoring at 1.5x the time); nothing has been run. Both branches total about 3.8 GPU-hours.

Run order: calibration-content check -> pilot -> freeze file (item IDs, hashes, seed 1234, caps, parser, `-np`, margins) -> Q8_0 -> v1 -> v2m. T3 stays sealed and separate.

## 1. Components

Sizes are given as `-np 4` branch / `-np 1` branch. The pilot picks the branch before any v1 or v2m test data exists.

| # | Component | Source and size | Settings | GPU min, 3 models | Metric |
|---|---|---|---|---|---|
| A | Fresh-domain KL | 4 domains x 8 documents x 4 chunks x 1024 tokens | `llama-perplexity --kl-divergence -c 1024`, raw text | 35 | per-chunk KL vs Q8_0 |
| L | Long-context KL | German, 8 chunks x 4096, other documents than A | as A, `-c 4096` | 9 | per-chunk KL |
| M | Multiple-choice log-prob | about 1,000 items: MMLU and ARC-Challenge (`--multiple-choice`), `--winogrande` | prompt-only scoring | 25 | per-item correct, margin if loggable |
| B1 | Math | GSM8K test, 80 / 60 seeded items | llama-server `--jinja`, greedy, reasoning on, cap = T2 p99 | B total 118 / 126 | answer correct vs gold |
| B2 | Tool calling | BFCL live_simple + live_multiple, 100 / 60 | greedy, reasoning off, cap 256, raw text, frozen own parser | (in B) | AST match vs ground truth |
| B3 | Code | MBPP+ 60 / 40 | greedy, reasoning off, cap 512, EvalPlus in Docker | (in B) | tests pass |
| B4 | Structured output | JSONSchemaBench 60 / 40, no grammar | greedy, reasoning off, cap 384 | (in B) | valid under `jsonschema` |
| C | Task-trace KL | Q8 transcripts from B1-B4, chat-template rendered, up to 8 x 1024 per task | as A | 9 | per-chunk KL |
| D | Noise floor | v1 repeat on 30 B1 + 30 B3 items, `-np 4` branch only | as B | 11 / 0 | v1-vs-v1 flip rate |
| P | Pilot | Q8 on about 20 discarded items; `llama-batched-bench` on v1 | as B | 10 | throughput, cap, reasoning-off check |

Fresh domains for A:
1. Python: whole files from MIT/Apache repos, half with human commit history before 2023, half recent.
2. Rust: same rule.
3. German prose: Bundestag plenary protocols, 2026.
4. Math: LaTeX bodies of recent CC BY arXiv math papers.

Notes:
- Freshness check: run language ID and a code/LaTeX detector over the v2m calibration file and v1's imatrix source. A domain above 2% share is relabelled "near-calibration" and leaves H1. Code in v1's imatrix is recorded as a bias toward v1.
- Q8 generation: Q8 runs the first half of each B set (seed order). That is enough for C and for secondary agreement.
- B1 size: n = 36,000 / 27,000 tokens divided by the mean T2 trace length, at most 80 / 60, at least 40.
- Batching: `-np` is identical for all models. Per-slot context is set so no item truncates.
- Logit files: size = 32 x 512 x vocabulary x 2 bytes per domain. Check free disk, delete after each domain.
- Failures: a cap hit or unparseable output counts as incorrect.
- Status: B2 is a near-calibration control. B1 shares its domain with T2, so it is a task guard.

## 2. Primary endpoints and decision rule

H1 and H3 are co-primary, each at alpha 0.05. "Default" needs both, so alpha is not split.

1. H1, transfer: R = mean over fresh domains of exp(mean log(KL_v2m / KL_v1)), with a 95% cluster bootstrap CI (resample documents within domain, 10,000 resamples).
   - Pass: upper bound at or below 0.90 (at least half the T2 gain retained).
   - Partial transfer: upper bound between 0.90 and 1.0.
   - Regress: lower bound above 1.0.
   - Otherwise: "no evidence of transfer".
2. H3, no task regression: correctness against ground truth on B1+B3+B4 (200 / 140 items), v2m minus v1, one-sided 95% bound from mid-p McNemar.
   - Pass: lower bound above -5 points.
   - Regress: upper bound below 0.
   - Power: about 0.8 only if discordance is at or below 8% (n = 200) or 5.6% (n = 140). An inconclusive result is likely and is not read as a pass.

Ship rule for v1.1:
- H1 passes and H3 passes: v2m is the default.
- H1 passes or is partial, and H3 is inconclusive: v2m ships as an alternative file, v1 stays default.
- H1 regress or H3 regress: v2m does not ship.

Power check before freezing: take the per-chunk paired SD of log ratio from T1. If 32 chunks per domain cannot bound 0.90 at a true ratio of 0.80, raise A to 12 documents per domain (about 9 extra minutes).

## 3. Secondary endpoints, descriptive only

- Per-domain ratios with CIs for A, C and L, and the pre-2023 vs recent code split.
- M: paired accuracy (McNemar) and, if per-item scores can be logged, the paired log-prob margin.
- B2: AST score per model with an exact interval; failures split into parse-fail, wrong function, wrong argument.
- Agreement with Q8 per task on the Q8 half.
- 99th-percentile KLD and "Same top p" per domain.
- Per-task accuracy with exact intervals; B4 required-field and type errors.
- D flip rate, tokens per item, cap-hit and loop rate, and a sensitivity analysis without capped items.

## 4. Cut, and why

- Accuracy as a superiority endpoint: power is 0.3 or less at this n.
- Q8 generation on all items: not needed once H3 uses ground truth.
- MATH-500, AIME, LiveCodeBench, IFEval, full GSM8K: over budget.
- Chinese Wikipedia, WildChat, xLAM-60k, The Stack v2: near-calibration, gated, or redundant with German.
- Sampling above temperature 0: no common random numbers across models.
- HumanEval+: redundant with MBPP+.

## 5. Total GPU time

- A 35, L 9, M 25, C 9, pilot 10, loads and overhead 10: 98 min in both branches.
- `-np 4` branch: B 118 + D 11 = 129 min, total about 227 min.
- `-np 1` branch: B 126, total about 224 min.

If the pilot shows more than 5% overrun, cut in this registered order: M to 500 items, then B4, then B3 to 30. H3 pools what remains.

Open items to verify in the pilot, since I have not checked them: that `llama-perplexity` multiple-choice modes expose per-item scores and that their dataset files are available, that `chat_template_kwargs` turns reasoning off, and that chunk boundaries line up with document boundaries in A.
## Critique

Critique of the v1 vs v2m transfer battery (numbered; each point gives a fix)

1. The Q8 runtime is underestimated. Q8_0 is about 1.8x the file size, and generation is bandwidth-bound, so Q8 decodes at roughly 10-11 tok/s, not 18.5. Fix: budget Q8 generation at 1.8x the per-model time (B: about 130 min, not 73), or the total goes past 4 h.

2. The B1 tokens per item are probably too low. 23 min per model means about 300 tokens per item, which is short for a looped reasoning model. Fix: before freezing, take the length distribution from the existing T2 GSM8K traces. Only raise the cap above 1024 if p95 is over it. Otherwise cap hits inflate discordance for every model.

3. The biggest cheap time saving is batching. Run llama-server with `-np 4` (continuous batching) and measure aggregate throughput in the pilot with `llama-batched-bench`. MoE decode on Apple silicon often scales 2-3x. Keep `-np` identical for all three models.

4. D's noise floor is empty. Greedy decoding with one slot on the same binary is deterministic, so the v1 vs v1 flip rate is 0 by construction. Batching makes it nonzero, because batch composition changes the numerics. Fix: if `-np` > 1, keep D and run it under the same `-np`. If `-np` = 1, drop D and spend its 12 min on point 11.

5. The fresh domains are not checked as fresh. The llama.cpp chat/tool-use calibration conversations almost certainly contain Python, JSON and some non-English text. Fix: before freezing, run language ID plus a code/LaTeX detector over the actual v2m calibration file, report the share per domain, and demote any domain above about 2% to "near-calibration". The same check applies to v1's imatrix. If that imatrix had code in it, the test is biased toward v1, so record that bias.

6. The bootstrap ignores clustering. Chunks from the same file or protocol are correlated. Fix: run a cluster bootstrap by source document, stratified by domain, with at least 8 documents per domain.

7. Pooling lets one domain dominate. Code and LaTeX KL will be several times larger than prose KL. Fix: make the primary statistic the mean over domains of the per-domain geometric-mean ratio, exp(mean log(KL_v2m/KL_v1)).

8. The H1 pass bar is too weak. "Upper bound < 1.0" passes on a 1% gain. T1 and T2 show ratios of 0.74 and 0.79. Fix: pre-register a meaningful threshold, for example "retains at least half the T2 gain", which means upper bound ≤ 0.90. Report anything between 0.90 and 1.0 as partial transfer.

9. The gatekeeping order contradicts the ship rule. A failed H2 blocks H3, yet shipping depends on H3. The per-domain "regress" CIs also add four untested comparisons. Fix: make H1 and H3 co-primary as an intersection-union test (both must pass, no alpha split). Move H2 and the per-domain checks to secondary. Ship only if H3 non-inferiority passes. If H3 is inconclusive, ship v2m as an alternative file, not the default.

10. B4's metric measures chaos, not quality. Exact canonical equality with Q8 fails as soon as any free-text field drifts by one token. Fix: use schema validity via `jsonschema` as a binary per-item outcome, with required-field and type errors as secondary.

11. A cheap, high-n, paired check is missing. `llama-perplexity --multiple-choice` (MMLU, ARC-Challenge), `--winogrande` and `--hellaswag` only process prompts, at about 220 tok/s. About 1,000 items take under 10 min per model. Fix: add them as secondary. The per-item log-prob margin is a continuous paired outcome with far more power than generated accuracy.

12. The BFCL harness has a confound. With `--jinja`, llama.cpp's template-specific tool parser may not recognise this template and fall back to the generic format, so "parse-fail" then measures llama.cpp, not the quant. Fix: log the raw completion text (`reasoning_format none`) and parse it with one frozen parser of your own. Check in the pilot that `chat_template_kwargs` really turns reasoning off. If it does not, the 256-token cap cuts every call.

13. B3 and B1 agreement is a weak signal. "Same verdict as Q8" counts two different failures as agreement, and MBPP+ is near saturation. Fix: keep it as a regression guard, but use correctness against ground truth for the H3 pairs (v2m vs v1 McNemar on correct/incorrect). Keep agreement with Q8 secondary.

14. H3 pooling mixes near-calibration B2 into the no-regression test. Fix: run H3 on B1+B3+B4. B2 gets its own descriptive interval.

15. Length is untested for a looped model. Errors that compound through three passes may show only at long context. Fix: add one domain at `-c 4096` with 8 chunks (about 3 min per model).

16. Check the logit file size against the real vocabulary: 32 x 512 scored tokens x vocab x 2 bytes. With a 200k vocabulary that is 6.5 GB, not 5 GB. Check free disk space before running.

17. "After cutoff" code may itself be LLM-written and close to the model's own distribution. Fix: require a human-authored repo history from before 2023 for at least half of the code documents, or record the bias.

18. EvalPlus executes generated code. Run it in its Docker sandbox, not on the host.

Revised time with batching (point 3): A 30 min, B (B1-B4 plus Q8 at 1.8x) about 100-120 min, MC log-prob 25 min, C 15 min, long-context check 9 min. Total about 3-3.5 h. Without batching, cut B3 to 40 items and B1 to 60 up front instead of waiting for the pilot.
## Draft

# v1 vs v2m transfer battery (design for pre-registration)

All times are estimates from the stated speeds (220 tok/s scoring, 18.5 tok/s generation); nothing has been run. Total is about 3.7 GPU-hours.

Run order: freeze file (item IDs, file hashes, seed 1234, caps, parsers, margins) -> Q8_0 pilot -> Q8_0 full -> v1 -> v2m. T3 stays sealed and separate.

## 1. Components

| # | Component | Source and size | Settings | GPU min: v1+v2m (+Q8 once) | Metric |
|---|---|---|---|---|---|
| A | Fresh-domain KL | 4 domains x 32 chunks x 1024 tokens (512 scored per chunk) | `llama-perplexity --kl-divergence -c 1024`, raw text | 20 (+10) | per-chunk KL vs Q8_0 |
| B1 | Math task | GSM8K test, 80 seeded items | llama-server `--jinja`, greedy, reasoning on, cap = Q8 pilot p99 (max 1024) | 46 (+23) | final answer equals Q8 answer |
| B2 | Tool calling | BFCL live_simple 50 + live_multiple 50 | greedy, reasoning off, cap 256 | 20 (+10) | same function and normalised arguments as Q8 |
| B3 | Code | MBPP+ 60 seeded tasks (EvalPlus, `--greedy`) | reasoning off, cap 512 | 18 (+9) | same pass/fail verdict as Q8 |
| B4 | Structured output | JSONSchemaBench, 60 seeded schemas, no grammar | greedy, reasoning off, cap 384 | 14 (+7) | canonical JSON equals Q8 output |
| C | Task-trace KL | Q8's own B1-B4 transcripts, chat-template rendered, 16 x 1024 per task | as A (same method as T2) | 10 (+5) | per-chunk KL vs Q8_0 |
| D | Noise floor and pilot | v1 repeat on 30 B1 + 30 B2 items; Q8 pilot on about 50 discarded items | as B | 12 (+10) | flip rate v1 vs v1; cap and timing calibration |

Fresh domains for A, all absent from v2m's calibration and dated after the model's cutoff where possible:
1. Python: whole files from MIT/Apache GitHub repos created after the cutoff, no vendored code.
2. Rust: same rule.
3. German prose: Bundestag plenary protocols, 2026 sessions.
4. Math: LaTeX bodies of recent CC BY arXiv math papers, preamble stripped.

Notes on the components:
- Exclusions: no source that appears in common imatrix calibration sets (Magicoder, OpenCoder, McEval, OpenMathInstruct-2, FineWeb), and check v1's imatrix provenance.
- Logit files: about 5 GB per domain, so delete after each domain.
- B2 status: tool calling overlaps the calibration set, so it is a near-calibration control, not a fresh domain.
- B1 status: GSM8K test shares its domain with T2, so it is a task guard, not a transfer test.
- Failures: a cap hit or unparseable output counts as disagreement.

## 2. Primary endpoints and decision rule

Fixed-sequence gatekeeping, each at alpha 0.05, tested only if the previous one passes, so alpha is not split.

1. H1, transfer: ratio KL(v2m)/KL(v1) pooled over the 4 fresh domains in A (128 chunks), with a 95% paired bootstrap CI over chunks, stratified by domain, 10,000 resamples.
   - Pass: CI upper bound below 1.0, and no single domain with a point ratio above 1.05.
   - Regress: CI lower bound above 1.0, or any domain's own CI lower bound above 1.0.
   - Otherwise: "no evidence of transfer; the gain is specific to calibration-like text".
2. H2, task-trace transfer: the same ratio and bootstrap on the code and JSON traces from C (32 chunks). Pass: CI upper bound below 1.0.
3. H3, no task regression: agreement with the Q8_0 answer pooled over B1-B4 (300 items), v2m minus v1, one-sided 95% bound from mid-p McNemar on discordant pairs.
   - Pass (non-inferior): lower bound above -5 points.
   - Power: about 0.8 if discordance is 12% or less. D measures the discordance and the v1-vs-v1 floor; both are recorded before v2m runs.

Verdict wording:
- "v2m transfers": H1 passes.
- "v2m does not regress on tasks": H3 passes.
- v1.1 ships v2m as default only if H1 passes and H3 is not failed in the regress direction (upper bound below 0).

Power check before freezing: compute the per-chunk correlation between v1 and v2m from the existing T1 output. If the paired SD implies more than 32 chunks per domain, raise A to 48 (about 4 extra minutes in total).

## 3. Secondary endpoints, descriptive only

- Per-domain KL ratios with CIs for A and C, including the tool and GSM8K traces.
- 99th-percentile KLD and "Same top p" per domain, since overfit can show in the tails while the mean holds.
- Per-chunk sign test on log(KL_v2m/KL_v1).
- Per-task agreement with Q8 and raw accuracy, with exact intervals: GSM8K accuracy, MBPP+ pass rate, BFCL AST score, JSON schema validity.
- Tool-call failures split into parse-fail, wrong function and wrong argument.
- Tokens per item, cap-hit rate and loop rate per model, plus a sensitivity analysis that drops items where any model hit the cap.

## 4. Cut, and why

- Accuracy as a superiority endpoint: at n of 80-300, power for a 1-3 point gap is 0.3 or less.
- MATH-500, AIME, LiveCodeBench: 1-20k reasoning tokens per item is over budget, and LiveCodeBench documents no local-endpoint option.
- IFEval (541 items): about 1 h for three models, generative only, low paired power. JSON without a grammar covers format-following more cheaply.
- Full GSM8K test (1319 items): about 6 h per model.
- Chinese Wikipedia and WildChat as KL domains: language effects on imatrix quants are near null in the literature, German already serves as the null check, and dialogue overlaps the calibration set.
- xLAM-60k and The Stack v2: near-calibration or likely pretraining text, and both are gated.
- Multiple seeds or temperature above 0: fixed-seed sampling gives no common random numbers once logits differ. Greedy makes every discordance a model effect, and loop rate is logged instead.
- HumanEval+: redundant with MBPP+.

## 5. Total GPU time

- A: 30 min
- B1-B4: 147 min
- C: 15 min
- D: 22 min
- Model loads and overhead: 10 min
- Total: about 224 min (3.7 h)

If the Q8 pilot shows tokens per item running more than 15% over these estimates, cut in this registered order: B4 first, then B3 down to 40 items, then B1 down to 60. H3 is then pooled over what remains, and H2 uses the code traces alone if B4 is dropped.
## Research reports

## Report: tasks
Task lens: small task evaluations for a paired v1 vs v2m comparison against a llama-server endpoint.

1. Budget comes first. At about 18.5 tok/s, 4 GPU-hours buys about 266k generated tokens, or about 133k per model if both are run. Prompt processing at about 220 tok/s is roughly 12x cheaper. Two consequences follow:
   - Anything with long reasoning per item (AIME at 5-20k+ tokens, MATH-500 at 1-4k) is out, or must be capped at a handful of items.
   - The most efficient design is teacher-forced: generate reference continuations once with Q8_0 on task prompts (code, tool calls, JSON, math, German), then score v1 and v2m on those same tokens with `llama-perplexity --kl-divergence` (KL, top-1 agreement, worst-token tails). This is the domain-transfer test, run at prompt-processing speed.
   - Free-running generation is then only a small regression guard.
   - These are my calculations from the stated speeds.

2. Flips beat accuracy, and KL predicts flips. Dutta et al. found accuracy within about 2% of baseline while 5-13.6% of answers flipped. KL vs flips had Spearman 0.981 on MMLU, and easier tasks flip less. Primary task metric to pre-register: per-item agreement with the Q8_0 answer (flip rate), with accuracy secondary. https://arxiv.org/abs/2407.09141 (search summary of the full text; the abstract page itself not opened)

3. Power: use paired per-item tests. Kübler et al. (ICLR 2026) recommend per-sample McNemar over task-level averages, and note that output changes even at temperature 0 because of numerical noise. https://arxiv.org/abs/2602.10144 (opened)
   - The v1 vs v2m KL gap is small (0.027 vs 0.020). At n=200 there may be only about 10-20 discordant items.
   - So pre-register the task checks as non-inferiority (v2m no worse than v1 by more than X), not as a superiority claim.
   - Fix the seed, temperature 0, and the same build, and run a v1-vs-v1 repeat to measure the noise floor of flips.

4. Math difficulty scaling. Harder sets lose up to 4x more than GSM8K: AIME-120 > MATH-500 > GSM8K. At 4-bit, the 1.5B model drops 1.8 points on GSM8K and 3.3 on MATH-500; the 32B drops about 0.4 on average. These used 32k max tokens and 3 seeds, which is infeasible here. Aggressive quantization lengthens outputs, so log tokens per item as a cheap signal. https://arxiv.org/html/2504.04823v2 (opened)
   - Pick 150-250 GSM8K test items plus about 50-100 MATH-500 level 3-5 items with a token cap.
   - GSM8K is the same domain as T2 (train prompts), so it is not a fresh-domain test. GSM-Symbolic P1/P2 gives template variants built from 100 GSM8K test originals. https://arxiv.org/html/2410.05229 (search only)

5. llama.cpp GGUF evidence (Llama-3.1-8B, 13 K-quants):
   - GSM8K was the most quant-sensitive task and IFEval second. MMLU, HellaSwag and TruthfulQA stayed within about 1 point.
   - Perplexity was "not a complete predictor": Q5_0, Q5_1 and Q5_K_S share PPL 7.43, yet GSM8K ranged 75.7-79.1.
   - Single-run noise is about ±1-2 points, so differences that size are within noise. https://arxiv.org/html/2601.14277v1 (opened)

6. IFEval: 541 prompts, short outputs (my estimate: a few hundred tokens). It runs through lm-eval `local-chat-completions` with `--apply_chat_template`. That backend has no loglikelihood tasks, so it is generative only. https://huggingface.co/datasets/google/IFEval (opened), https://www.philschmid.de/evaluate-llms-with-lm-eval-and-tgi-vllm (search only)
   - IFBench has 58 out-of-distribution constraints on held-out WildChat prompts, so it is less contaminated. https://arxiv.org/html/2507.02833v2 (search only, prompt count not verified)

7. Tool calling (BFCL):
   - Cheapest per item (short outputs) and closest to the chat template's tool path.
   - Runs against an OpenAI-compatible server: add a `MODEL_CONFIG_MAPPING` entry with `OpenAICompletionsHandler` and `is_fc_model=True`, set `OPENAI_BASE_URL`, then `bfcl generate --test-category live_simple`. https://gist.github.com/bbrowning/b0b3b36e36eb78e19417d63ca1f5d022 (opened)
   - A 2026 study of BFCL v4 multi-turn at F16/Q8_0/Q5_K_M/Q4_K_M: xLAM-2-3B fell 58.0 to 51.75 at Q4_K_M and Qwen3-1.7B 14.5 to 8.75, both significant under paired McNemar. Aggregate accuracy hid which failure grew (wrong arguments, silent wrong trajectories), and 4 of 6 models were floor- or format-limited. https://neurips.cc/virtual/2026/171271 (opened)
   - Recommendation: live_simple plus live_multiple/parallel (a few hundred items). Report parse-fail, wrong-function and wrong-argument rates separately, and check BFCL's handling of malformed calls.

8. Code:
   - EvalPlus supports `--backend openai --base-url http://localhost:PORT/v1 --greedy`. MBPP+ has 378 tasks (v0.2+); HumanEval+ has 164. https://github.com/evalplus/evalplus (opened)
   - LiveCodeBench release_v6 has 1,055 problems (May 2023 to Apr 2025) with `--start_date` filtering for problems after the model's cutoff. Its README documents no OpenAI base-URL option. https://github.com/LiveCodeBench/LiveCodeBench (opened)
   - GGUF is among the formats studied for code; correctness drops differ by method. https://arxiv.org/abs/2607.14181 (abstract only)

9. Structured output: llama.cpp grammar guarantees valid syntax, so the signal is semantic correctness, or schema adherence without the grammar. JSONSchemaBench has 10K real schemas, and constrained decoding improved downstream accuracy by up to 4%. Sample about 100 schemas. https://arxiv.org/html/2501.10868v3 (search only)

10. "Hollow" correctness: under NF4, accuracy moved at most 3.1 points, yet correct-but-unsupported reasoning rose in 3B models, undetectable from surface features (F1 0.53). GSM8K was unaffected; LogiQA and ARC-Challenge shifted most. Teacher-forced KL on reasoning traces (item 1) is the cheap proxy. https://arxiv.org/abs/2607.09999 (opened)

11. Divergence beats accuracy on llama.cpp schemes (BF16 down to Q2_K): accuracy was non-monotonic while JSD and TVD rose steadily. https://arxiv.org/abs/2609.07664 (opened)

12. Contamination:
    - It does not bias a v1 vs v2m comparison (same weights).
    - It does matter for the fresh-domain claim. Pick domains absent from the v2m calibration (WikiText-2, chat/tool conversations, prose). Code, math competition text, JSON and German qualify; tool-call transcripts do not, because they overlap the calibration set.
    - AIME 2026 and HMMT Feb 2026 exist on Hugging Face (MathArena), but their position relative to the model's training cutoff is unknown. https://huggingface.co/datasets/MathArena/aime_2026/blob/main/README.md (search only)

Suggested split, my estimate, to measure first on 10 items:
- Teacher-forced KL: 4 domains x 5 x 2048 tokens per model, about 10 min each.
- GSM8K: 200 items.
- BFCL live: about 250 items.
- IFEval: 541 items, or a 200-item subset.
- MBPP+: 100 items.
- All greedy, capped tokens, both models.
- Total about 3-4 h. Drop MBPP+ or trim IFEval if the measured tokens per item run high.

## Report: kl-domains
Lens kl-domains: findings

1. Mechanics that set the budget. `llama-perplexity --kl-divergence` scores only the second half of each chunk (`const int first = n_ctx/2;` in the KL function, line 1772). It prints a running cumulative mean per chunk, so you get per-chunk means by differencing consecutive lines, and the paired bootstrap needs nothing extra. The base logit file stores about n_vocab x 2 bytes per scored token, plus 4 more bytes per token. The README warns it reaches 11 to 37 GiB on a full WikiText test. Its ± is per-token Gaussian, which understates the error under autocorrelation, so bootstrap over chunks instead. Opened: https://github.com/ggml-org/llama.cpp/blob/master/tools/perplexity/README.md and https://raw.githubusercontent.com/ggml-org/llama.cpp/master/tools/perplexity/perplexity.cpp

2. Power, my estimate and not a published figure. Assume chunk-level KL has a CV of about 0.4 and the per-chunk correlation between v1 and v2m is r = 0.5 to 0.8. The SD of the paired difference is then 0.25 to 0.4 x the mean. Resolving a 20% gap at 2.8 SE (alpha 0.05 two-sided, 80% power) needs n = (SD_diff / 0.071)^2, about 13 to 32 chunks. Pre-register 32 chunks x `-c 1024` per domain: 16k scored tokens and 32k processed, about 2.5 min per model at 220 tok/s. Three models (Q8 base, v1, v2m) over 6 domains come to about 45 min. The logit file is about 16k x n_vocab x 2 bytes, roughly 5 GB per domain at a 150k vocab, so delete it after each domain. Calibrate r from the existing T1 per-chunk numbers before freezing n.

3. Community evidence that calibration overfits is thin and mixed (llama.cpp discussion #5006). Wiki-calibrated imatrix beat random tokens on wiki PPL (7.73 vs 8.25) and on HellaSwag (+2.25 ± 0.53), so it transferred. But kalomaze's out-of-domain lyrics test (about 2.5k tokens) ranked pretrain-style calibration below no-imatrix (15.68 vs 15.55). No thread there tests code. The useful point: a test domain that matches the calibration domain proves nothing, which is the exact worry about T1. Opened: https://github.com/ggml-org/llama.cpp/discussions/5006

4. For an MoE like this, code is the domain most likely to expose expert-level overfit. A 118B-MoE imatrix study found code calibration moves expert importance the most. down_exps correlation with a prose baseline was only 0.48 to 0.60, against 0.959 overall for literature. That fits v2m, which re-chose expert codes. The study did not measure KLD. Opened: https://huggingface.co/Myric/Laguna-S-2.1-imatrix-calibration-study

5. Language mismatch has measurable effects for GPTQ: Chinese PPL 11.23 vs 12.24 for matched vs English calibration. For AWQ the effect is about 1% or less. Math and code in the calibration mix widen activation tails. Opened: https://arxiv.org/html/2601.18306v1. For imatrix K-quants, Norwegian or Malayalam imatrices showed no significant gain over an English one (n = 2000 per bar, Bonferroni). So expect small language effects, which makes German or Chinese a good null-check domain. Opened: https://arxiv.org/html/2503.03592v3

6. Sources to avoid because common calibration sets already contain them. eaddario/imatrix-calibration (MIT) draws on Magicoder, OpenCoder and McEval for code, OpenMathInstruct-2 for math, BitAgent and Efficient_ToolCalling for tools, and FineWeb or FineWeb-2 in 18 languages. Exclude these and their parents, and grep v1's imatrix provenance. Opened: https://huggingface.co/datasets/eaddario/imatrix-calibration

7. Recommended fresh domains, each 32 x 1024 tokens, all dated after the model's training cutoff where possible.
   - a) Code: GitHub repos created after the cutoff under MIT or Apache-2.0 licenses (GitHub search `created:>DATE license:mit`). Use 2 languages, Python plus Rust or TypeScript, in whole files, no vendored code. Not The Stack v2: it is from 2023, gated, needs a Software Heritage agreement, and is likely in pretraining. Opened the dataset page via search summary: https://huggingface.co/datasets/bigcode/the-stack-v2-train-smol-ids (search only)
   - b) German prose: Bundestag plenary protocols of the 21st Wahlperiode, 2026 sessions. They come as XML/JSON, appear the next working day, and are official works with no copyright under § 5 UrhG. https://www.bundestag.de/services/opendata (search only); https://zenodo.org/records/18177196 (search only)
   - c) Chinese or another non-Latin script: Wikipedia pages created after the cutoff, CC BY-SA, from the dumps (search only).
   - d) Math: LaTeX source of recent arXiv math.* papers whose license field is CC BY, preamble stripped (search only).
   - e) Structured or tool calls: xLAM-60k (CC-BY-4.0, about 60k JSON query/tools/answers, login-gated), rendered with the chat template. Treat it as a near-calibration control, not a fresh domain. The fresher variant is the Q8's own tool-call traces on held-out prompts, scored like T2. Opened: https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k
   - f) Dialogue: WildChat (ODC-BY, 66 to 68 languages), filtered to English and German turns (search only): https://huggingface.co/datasets/allenai/WildChat

8. Pre-registration details.
   - Fix the tokenization: raw text, one doc per chunk boundary, no BOS tricks.
   - Freeze the chunk seed and the file hashes before running anything.
   - Primary statistic: per-domain ratio KL(v2m)/KL(v1) with a 95% paired bootstrap CI over chunks.
   - Pass rule: CI upper bound below 1.0 on code, and no domain above 1.05.
   - Also report the 99th-percentile KLD and "same top p", because overfit can show in the tails while the mean holds.

## Report: stats
Stats lens: paired v1-vs-v2m battery design. The numbers in points 2 and 3 come from my own simulation (exact two-sided binomial on discordant pairs, alpha 0.05, 20k sims) and normal-approximation formulas. They are not quoted from a source.

1. The analysis unit is the per-item paired difference, not the two marginal scores. Miller's paired SE is sqrt(SE_A² + SE_B² − 2·SE_A·SE_B·corr), and two quants of one model are highly correlated, so pairing is close to free variance reduction. His sample-size formula is n = (z_{α/2}+z_β)²·(ω² + σ²/K)/δ². His example of a 3-point gap needs about 969 items. He also advises against lowering temperature just to cut variance. https://arxiv.org/html/2411.00640v1

2. For binary tasks, power depends only on the discordance rate d (items where v1 and v2m differ) and the true gap δ. Exact McNemar power, one-sided win direction:
   - d=4%: n=300 gives power 0.08 / 0.30 / 0.71 for δ=1 / 2 / 3 points; n=1319 (full GSM8K test) gives 0.40 / 0.95 / 1.0.
   - d=8%: n=300 gives 0.06 / 0.17 / 0.37; n=1319 gives 0.23 / 0.70 / 0.97.
   - d=12%: n=1319 gives 0.16 / 0.52 / 0.87.
   - MDE ≈ 2.8·sqrt(d/n). At n=150 and d=8% that is 6.5 points.
   Resolution Diagnostics gives the same closed form, N* = (z+z)²·σ_D²/δ², with σ_D² = p_A·q_A + p_B·q_B − 2ρ·sqrt(p_A·q_A·p_B·q_B). It warns that G*Power and pwr shortcuts underestimate N by about 2x. Its calibration found exact McNemar about 3 points conservative and mid-p close to nominal. https://arxiv.org/html/2605.30315

3. The budget makes this binding. A GSM8K trace of about 400 tokens at 18.5 tok/s is about 22 s, so roughly 150 items per model-hour. With about 2 GPU-hours for tasks, n is about 150 per model, and a superiority claim of 1-3 points has power ≤0.3. Pre-register task accuracy as a non-inferiority guardrail, not a win. One-sided 95%, 80% power, true δ=0, margin m: n = (1.645+0.842)²·d/m².
   - m=5 points: n=99 / 198 / 297 for d=4 / 8 / 12%.
   - m=3 points: n=275 / 550 / 825.
   The realistic choice is m=5 with n of about 150-200. Pre-register m and the assumed d. Measure d on a 30-item pilot that is thrown away.

4. Primary endpoint: KL on fresh domains (code, German prose), per chunk. Token-level KL has thousands of correlated observations, so use chunks as the unit: for example 20-30 chunks × 512-1024 tokens per domain. Compare per-chunk paired log(KL_v2m/KL_v1) with a sign test or Wilcoxon, plus a chunk-level paired bootstrap CI. llama-perplexity's printed ± assumes per-token KLD is Gaussian and independent, so it is too narrow for the claim. Read it, but don't test on it. It also reports "Same top p", which is a useful secondary. https://github.com/ggml-org/llama.cpp/blob/master/tools/perplexity/README.md

5. The best cheap task-relevant metric is flips against Q8_0, not accuracy. Dutta et al. found accuracy held within 1% while flips reached 13.6%. Flips correlated with KL (Spearman 0.981 on MMLU), and high-flip models scored worse on MT-Bench (Llama2-70B about −5%, turn 2 up to −10%). Pre-register the per-item "final answer agrees with Q8_0" indicator, compared v1 vs v2m by McNemar. Cost-saving idea, my own reasoning: generate Q8_0 answers on new task items once, then teacher-force them through v1 and v2m with --kl-divergence. Prompt processing runs at 220 tok/s, about 12x faster than generation, so this gives task-domain KL per item at low cost. https://arxiv.org/html/2407.09141v1

6. Sampling settings:
   - Seed noise on reasoning tasks is large: pass@1 SD of 5-15 points, a temperature sweep moves results up to 15%, and the paper recommends ≥30 seeds on small sets. That is unaffordable here, so use greedy (T=0) with an identical prompt, template and build for both models. Greedy makes every discordance a model effect rather than sampling noise.
   - Responses near the token cap were almost always wrong in that paper. https://arxiv.org/html/2504.07086v2
   - The DeepSeek-R1 card recommends T=0.6 "to prevent endless repetitions". So greedy can loop, and loop rate becomes a pre-registered secondary endpoint for each model. https://huggingface.co/deepseek-ai/DeepSeek-R1
   - Fixed-seed sampling does not give common random numbers once the logits differ. That is my reasoning, not a sourced claim.

7. Truncation rule, fixed in advance: set the cap from the Q8_0 pilot at its p99 trace length (cap-hit at most 1-2%). A cap hit or unparseable answer counts as wrong. Report the cap-hit rate per model, and run a sensitivity analysis that excludes items where either model hit the cap.

8. Small-n intervals: CLT intervals underestimate uncertainty below a few hundred items. Use exact or Bayesian intervals instead, for example the bayes_evals library. https://arxiv.org/abs/2503.01747

9. Multiplicity:
   - One primary: fresh-domain KL ratio v2m < v1, both domains pooled or both required.
   - One key secondary: GSM8K non-inferiority at m=5.
   - Use fixed-sequence gatekeeping (test the secondary only if the primary passes), so alpha is not split.
   - Everything else is descriptive or Holm-adjusted: flips vs Q8_0, Same-top-p, loop rate, cap-hit rate, tool-call/JSON validity.
   - Freeze items, seeds, caps, parser and margins in a dated file before any v2m run, and keep T3 sealed.

10. Context only, from the abstract: well-tuned INT8 lost 1-3% accuracy across more than 500k evaluations. That is the same size as the effect being tested here, which supports a non-inferiority framing. https://arxiv.org/abs/2411.02355

Simulation script: a throwaway local script, not kept.