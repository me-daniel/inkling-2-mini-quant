# Review of the remaining decisions before A2 (v2c, default switch)

- Date: 2026-11-24, ~05:50
- Model: Claude Fable 5.2, effort medium (subagent, prompt file only, no other reads)
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

Review of the six decisions (I read only the prompt file; nothing here is checked against the repo or the amendments' actual text).

1. Task run for v2c
- Recommendation: (b), but only if the gate is published before A2 is scored. Otherwise (a).
- Risk: skipping a registered step after seeing A2 looks like selective reporting, even when the result could change nothing.
- Check first: whether amendment 16 words the task run as unconditional. If it does, file the gate as a dated note within the 4-hour assembly window.

2. How v2c ships
- Recommendation: a change to (b). Give v2c a new, versioned file name and drop v2m from the main branch, leaving it reachable at tag v1.1. That keeps two files, and no name ever changes bytes.
- Risk of (a): every published v2m number (T3, battery A, tasks) is tied to "...-refined.gguf". After a silent swap, the same name means two models.
- Check first: who links or pins the current name, and that the card lists a SHA256 per file.

3. T3 for v2c
- Recommendation: (a), scored only after the A2 verdict is fixed, labelled "opened set, descriptive", and used in no rule.
- Risk: "v2c vs UD on T3" becomes the headline or tips the default decision, although T3 is no longer a test.
- Check first: that no v2c choice (400 chunks, the 0.1 diagonal, the code mix) was made with T3 numbers in view. If one was, say so next to the number.

4. Material for the default decision
- Recommendation: table plus rule verdict, plus a short recommendation in a separate section that states the strongest counter-argument. A bare table of every set hands the owner an unmarked multiple-comparison problem.
- Risk: the table mixes registered fresh sets with already-seen ones. Mark each row as fresh, seen or descriptive.
- Check first: that v1 and v2m are both scored on A2 (amendment 17 needs candidate/v1, the primary only gives v2c/v2m), and that the v2c speed bench runs with the sibling job off the GPU.

5. Next research
- Recommendation: (c) now, since the GPU is occupied for the next 5 hours anyway and writing costs none. Then (a) as a bounded pilot. (b) belongs to the sibling's plan, not this release.
- Risk of (a): its ceiling is about 0.001 KL, roughly 5% of v2m's T3 KL, and the search will recover only part of a format cost. It also needs a different Gram (expert-internal activations) and another fresh set.
- Check first: where the remaining Python and Rust gap sits. v2m closes 15-30% there; a per-tensor-class swap diagnostic on code text says whether down projections are the lever at all.

6. Public record
- Recommendation: (a). The timestamps are what make the registrations credible; holding results until the decision removes that.
- Risk: interim tables get read as a release. Label each "not released, decision pending".
- Check first: that A2's document list and hashes are public before scoring.

Missing from the list

- The bar moved after the data. v2m missed the registered "at least 10% lower" bar (R = 0.905, upper bound 0.932). Amendment 17 replaces it with "upper bound < 1", on code domains only, for a default file most people use for chat. On top of that the owner now decides himself. Publish that change before A2 is scored, and state whether discretion can only decline a switch or can also override a failed rule.
- A2's primary rests on 16 documents, and the expected v2c/v2m effect is smaller than v2m/v1 was (rust 0.955 [0.910, 1.009] on battery A). A fail is likely to be a power failure; write down beforehand that a null is inconclusive. A pass cannot be credited to code, since tokens and code change together; a 400-chunk no-code control is needed for that claim.
- The task rule cannot see a regression. With 200 items and 6 discordant pairs, a real drop of 2-3 points would pass. Either enlarge the task set or stop presenting the task result as a safety guarantee.

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
