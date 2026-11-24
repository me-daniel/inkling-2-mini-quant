# Review of the remaining decisions before A2 (v2c, default switch)

- Date: 2026-11-24, ~05:50
- Model: GPT-6.2 Astra, effort high (Codex CLI, isolated)
- Repo HEAD: f9b4dd6 (tree dirty: untracked v2c offline reports only)
- Why: Daniel asked for Astra (high) and Fable (medium) to answer the open decisions; he decides the default switch himself at the end.

## Prompt

You are a skeptical reviewer of the next decisions in a model-quantization project. You did not do the work. Answer in under 600 words. For each numbered decision give your recommendation (one of the options, or a change), the main risk, and one thing the author must check first. Then add at most 3 points the list is missing. No praise.

## Setup (public facts)

- Model: Inkling 2 Mini, a looped MoE (layers 6-17 run three times per token on the same experts; 128 experts, top-8), quantized to llama.cpp GGUF for Macs. All files run fully on the Mac GPU (Metal) with stock llama.cpp, unlike the main comparator Unsloth UD-IQ3_XXS (11.66 GB), which keeps some NVFP4 expert tensors on the CPU and generates 1.75x slower.
- Metric: KL divergence of next-token distributions against a Q8_0 reference; ratios are geometric means of per-chunk KL with document-level bootstrap CIs.
- Public now (release v1.1, tagged on Hugging Face and GitHub): v1 (11.51 GB) is the default file. v2m (11.57 GB) is an alternative file: the same tensor types as a llama-quantize build at UD's per-layer types, with the gate_up expert codes of 22 layers re-chosen by a codes-only search minimising tr(E H E^T) (E = quantized minus BF16, H = per-expert input Gram + 0.1 diag), Grams from 153,600 tokens of mixed calibration text (WikiText train, chat/tool conversations, prose; almost no code). down projections are not refined.
- v2m vs v1: sealed one-shot test T3 (WikiText) KL 0.0208 vs 0.0285, diff -0.0077 [-0.0090, -0.0063]; UD 0.0199 (v2m worse than UD by +0.0009 [+0.0003, +0.0016]). Fresh battery A (8 docs x 4096 tokens per domain), v2m/v1: python 0.917 [0.869, 0.973], rust 0.955 [0.910, 1.009], german 0.844 [0.822, 0.866]. Registered bar "at least 10% lower with 95% confidence" over those three: R = 0.905 [0.881, 0.932], so partial, which is why v2m is an alternative and not the default. Tasks (200 items: GSM8K, MBPP+, JSON-schema output): v2m - v1 = +3.0 points, discordant 6 / 0, passed the non-regression rule. Speed and memory equal to v1 within noise.
- Share of v1's KL gap to UD that v2m closes: 91% WikiText, 69% reasoning traces, 62% math, 49% German, 30% Python, 15% Rust.
- Running now (pre-registered as amendment 16, public before any of its data): v2c, the same search with 65,545 tokens of Python and Rust code added to the calibration text (400 chunks instead of 300; more tokens and code change together). Decided on a new fresh set A2 (python, rust, go, typescript; 8 docs x 4096 tokens each, from repositories used nowhere else; go and typescript are in no calibration text). Primary: v2c/v2m over A2 python+rust, pass if the 95% upper bound < 1. Guards: v2c/v2m on German (battery A's text, already seen), WikiText T1 and reasoning traces T2, each upper bound <= 1.05. Tasks: v2c vs v1 under the same non-regression rule (about 70 min of GPU). If all pass, v2c is proposed to replace v2m as the alternative file.
- Also pre-registered (amendment 17, public before A2 is scored): the alternative file becomes the default if, on A2's four code domains, the mean of per-domain ratios candidate/v1 has 95% upper bound < 1 and no domain's upper bound > 1.05, and it passed the task rule. The owner has now said he decides the default switch himself at the very end, with this rule's verdict as input.
- Timeline: v2c is assembled in about 4 hours, then A2 scoring (about 1 hour GPU). One Mac, 48 GB, one heavy job at a time; a sibling project (a smaller 8.5-8.9 GB file family) shares the GPU.

## The decisions

1. Task run for v2c. (a) Always run it, as registered. (b) Run it only if A2's primary test and the guards pass, since otherwise v2c can be neither the alternative nor a default candidate.
2. If amendment 16 passes, how v2c ships (upload only after the owner's go at the end). (a) Replace the bytes behind the existing file name "...-refined.gguf" and tag v1.2; tag v1.1 keeps v2m reachable. (b) A third file name, v2m stays. (c) Hold v2c unpublished as a research result.
3. T3 for v2c. T3 was opened once, for v2m. v2c never saw it, but T3 is no longer sealed. (a) Score v2c on T3 and report it as descriptive. (b) Leave T3 alone; A2 is the fresh test.
4. What to give the owner for the default decision beyond amendment 17's verdict. Proposed: one table per candidate (v1, v2m, v2c if it passes, UD) over every set, the task results, speed and memory on the same Mac (v2c not benched yet, about 15 min GPU), and the rule's verdict, with no recommendation of our own. Or include a recommendation?
5. Next research while waiting, ranked by expected release value: (a) extend the search to the down projections (unrefined now; a diagnostic that swaps UD's NVFP4 down projections for our Metal-runnable IQ4_NL loses 0.0008-0.0012 KL to UD on every set, about the size of v2m's remaining T3 gap); (b) add IQ2_XXS support to the search for the sibling's smaller file, whose layers are mostly IQ2_XXS; (c) stop adding methods and write up. Any new candidate would need its own registration and a fresh test set.
6. Public record. (a) Push text-only result records (changelog, result tables, scripts; no weights) to GitHub as each one lands. (b) Hold every push until the owner's decision at the end.

## Reply

1. **(a) Run the v2c tasks as registered.** A failed A2 result settles release eligibility, but the task result still tells readers whether the KL and task outcomes agree. **Main risk:** making task reporting conditional on a favorable KL result after committing to the run. **Check first:** verify that the harness uses the same 200 items, scoring settings, and paired non-regression rule as the v1 comparison.

2. **(b) Give v2c a distinct filename after the owner approves.** Keep v2m available and point the release page to the chosen alternative. **Main risk:** replacing bytes under `...-refined.gguf` lets two users cite the same filename for different models, even though the old tag remains reachable. **Check first:** verify that the release manifest maps each filename to its revision and checksum.

3. **(b) Leave T3 alone.** Its v2m result already informed the project’s view of the remaining gap. A2 supplies fresh evidence for the code question, and T1 supplies the registered WikiText guard. **Main risk:** a descriptive v2c T3 score later gets cited as a second independent confirmation. **Check first:** identify any release claim about v2c closing the T3 gap and remove it unless directly measured.

4. **Include a recommendation alongside the amendment 17 verdict.** Give the owner one side-by-side table with absolute KL, paired ratios and intervals, task outcomes, size, speed, and peak memory. Mark unmeasured cells. Spend the 15 minutes to benchmark v2c if it qualifies. Recommend retaining v1 on a failed verdict. On a pass, explain whether the code gain justifies a general default given the non-code and task evidence. **Main risk:** a four-code-domain win gets read as a general-purpose win. **Check first:** establish which workloads the default is intended to serve before interpreting A2.

5. **Rank (a), then (b), then (c) for release value.** Start with a small down-projection pilot. Draft the existing results while the Mac is occupied. **Main risk:** the NVFP4-to-IQ4_NL swap measures a format change. Its KL cost gives no estimate of how much a codes-only IQ4_NL search can recover. **Check first:** on development data, test whether refining a representative subset of down projections improves end-to-end KL at fixed tensor types before committing the GPU to a full run.

6. **(a) Push text-only records as each registered analysis finishes.** Publish failures on the same schedule as passes and mark the owner’s release decision as pending. **Main risk:** a partial result gets quoted as the release verdict. **Check first:** verify that each record identifies the candidate hash, amendment, and status of every outstanding gate.

**Missing from the list**

- Specify which candidate enters amendment 17 before opening A2. If A2 selects v2c over v2m and also supports promotion, disclose the reuse when interpreting the promotion interval.
- Attribute any v2c gain to the combined calibration change. Isolating the effect of code requires a token-matched control.
- Audit the bootstrap implementation. The eight documents per domain are the resampling units. Keep candidate scores paired and preserve the registered domain weighting.

## Response (both reviews; Astra's points as A1-A6 and A+1-3, Fable's as F1-F6 and F+1-3)

1. Task run (A1 a, F1 a unless gated in advance). Accepted: run it as registered. Amendment 16 words it unconditionally, so no gate is added.
2. Shipping v2c (A2 and F2 both: a new file name, never new bytes under an old name). Accepted. v2m stays on the main branch too (A2), because amendment 17 promised that file names stay so pinned downloads keep working, and dropping v2m from main (F2) would break them. Upload only with Daniel's go at the end.
3. T3 for v2c (A3 no, F3 descriptive only). Rejected for now: the T3 reference logits were deleted after the sibling's T3 run (13 GB) and would have to be rebuilt, and a second look at an opened set adds a number that could be misread as confirmation (A3). A2 is the test.
4. Decision pack (A4, F4 agree). Accepted: one table, every row marked fresh, seen or descriptive; absolute KL, ratios with CIs, tasks, size, speed and memory; v2c benched if it qualifies, with the GPU free of other jobs; a short recommendation in its own section with the strongest counter-argument. v1 and v2m are both scored on A2 (`scripts/a16-score.sh` runs cand, v2m, v2c and UD-IQ3_XXS).
5. Next research (A5: a, b, c; F5: c, then a bounded pilot after a diagnostic; b is the sibling's). Accepted the common part: the decision pack first, then a diagnostic on battery A's code text (seen data, diagnostic only) that swaps one tensor class at a time between v2m and UD to see where the code gap sits. A down-projection search starts only if down projections carry a real share of it, under its own registration and a fresh set. IQ2_XXS support stays with the sibling.
6. Public record (A6, F6 both a). Accepted: text-only records go to GitHub as they land, marked "decision pending" where a decision is open.
7. A+1 (amendment 17 reuses A2 after A2 selects v2c). Accepted: `analysis/a16.py` now also prints v2m's amendment 17 line, marked descriptive, whenever v2c is the candidate. The candidate rule itself was fixed in amendment 17 before A2.
8. A+2 and F+2 (tokens and code change together). Accepted as a reading rule: a pass is credited to "the larger calibration with code", not to code. A token-matched control is not run now. F+2's power point is also accepted: H-C1 rests on 16 documents, so a fail reads as "no evidence of a gain", not as "code does not help".
9. A+3 (bootstrap). Checked, no change: `analysis/battery_A.py` `ratio_ci` resamples whole documents (4 chunks each) within each domain, on paired log ratios, with equal domain weights.
10. F+1 (the bar moved; owner discretion). Accepted as disclosure: amendment 17's rule and Daniel's own decision are both stated before A2 is scored (CHANGELOG record of 24 Nov). The rule's verdict is published either way; if the decision departs from it, the card says so.
11. F+3 (task rule power). Accepted: the task check is reported as a screen for large drops (200 items), not a guarantee of no regression.
