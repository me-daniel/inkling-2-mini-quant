ROLE_LINE_HERE

You are reviewing the RESULTS of a preregistered comparison of quantized GGUF files of a looped MoE language model (layers 6-17 run three times, P = 3). The registration was written and published before any data. Below: (1) the registration, (2) the registered primary results on the fresh set A3, (3) descriptive KL on the older sets S, T1, T2 that selected our files, (4) per-role byte totals from the file headers. Parts B (speed) and C (same-type tensor swaps between v2c and ISTA IQ3_S, fixed bytes and type, which compare encoders) are still running and are not shown.

Answer as a numbered list of at most 10 points. Each point: a label [measured] / [inference] / [needs source], one claim, and a concrete action if any. Keep it under 900 words. Be blunt; disagreement is useful.

## 1. Registration (verbatim, names and paths removed)

### Addendum H amendment 18: head-to-head against ISTA-DASLab's GSQ-RCO files and Unsloth (2026-11-27, before any of its data)

Why: on 26 Nov ISTA-DASLab published `Inkling-2-Mini-GSQ-RCO-GGUF`: four all-Metal files with learned-grid encoding (GSQ, joint codes and scales) and a per-tensor type budget (RCO). Our cards' "closest" claims cover only the files tested on 21 Nov, and Unsloth's UD-IQ3_XXS was already closer than v2c on fresh code (A2). The design was reviewed before data by Astra (high), Sol (xhigh) and Fable (medium); the reviews are in `reviews/2026-11-27-a18-*.md`.

Scope: this compares released files, not methods. Each of our files differs from ISTA's in map, encoder, calibration and pipeline together. Method attribution is the descriptive part D below. Every KL claim is "closer to our Q8_0": D_KL(Q8_0 ‖ file) at P = 3, our Q8_0 and llama.cpp build (29b52d3 plus `patches/llama.cpp`), identical token ids, -c 1024 with the second half of each chunk scored.

Files:
- Ours: v1 (cand), v2m (refined), v2c (refined2), c1(the 8.9G file, sha256 a01a9156…bca9, read-only).
- ISTA at f0da109: IQ3_S, IQ3_XXS, Q2_K_XL, IQ2_S (`scripts/fetch-ista.sh`, sha256 against the Hub in `results/a18/ista-sha256.txt`).
- Unsloth UD-IQ3_XXS at ea0c79b (`Mainline_Support`, as in amendment 4).
- Excluded, named: AtomicChat AD-IQ2_S, bartowski IQ2_S and mudler APEX-I-Nano. They lost to v1 on 21 Nov.
- Exact bytes are reported from the files.

Step 0, before scoring: a packaging check. Every file must carry the same tokenizer tokens (hash), chat template, loop count and vocabulary as our Q8_0. Tensor types, fallbacks and bytes are read from the headers (`scripts/gguf-types.py`).

A3, the fresh set (`scripts/a18-data.py`, hashed in `results/a18/data.md` before any model is scored on it). 5 domains, 12 documents × 4,096 tokens each, 48 chunks per domain:
- python and javascript: 12 new repositories each, battery A's half-old, half-recent rule. JavaScript is in no calibration text of ours.
- german and english: Wikipedia articles created in the 30 days before the build.
- math: arXiv LaTeX, 13–26 Nov 2026.

Sampling rules, fallbacks and overlap with calib-mix, calib-mixc and calib-code are fixed in the script. Not included: a chat-template domain, because our llama-perplexity does not parse special tokens (Ling's patch 0013 is not ported here). The task battery stands in for chat use. The two code domains together get 40% weight, and so do the two Wikipedia domains. Conclusions are scoped to this mixture and context.

Primary statistic per pair: G = exp(mean over the 5 domains of the per-domain mean of log(KL_ours / KL_other) per chunk). It is symmetric: swapping the files gives 1/G.
- 95% CI from a paired, domain-stratified cluster bootstrap. Resample the 12 documents of each domain with replacement, carrying all 4 chunks and both files, and recompute G. 10,000 resamples, seed 0; `analysis/a18.py`, with a toy-fixture test.
- No joint claim across pairs, so no multiplicity adjustment. Each pair has its own actions.

Outcomes per pair, written now:
- Ours closer: upper bound < 1 and G ≤ 0.97.
- Other closer: lower bound > 1 and G ≥ 1/0.97.
- Equivalent at ±5%: the whole CI inside [0.95, 1/0.95].
- Otherwise: inconclusive.

Sensitivity results, reported beside every primary result:
- leave-English-out G;
- leave-one-document-out range;
- per-domain G with CIs and the count of domains favouring each file;
- absolute and token-weighted KL per file and domain;
- p99 KL and top-1 agreement per domain.

Primary pairs:
- P1: v2c (11.567 GB) against ISTA IQ3_S (11.43 GB). ISTA is 137 MB smaller, so an ISTA win means dominance and a v2c win is a trade-off of +1.2% bytes.
- P2: c1 (8.932 GB) against ISTA IQ2_S (8.98 GB). c1 is 48 MB smaller, so a c1 win means dominance.
- P3: v2c against UD-IQ3_XXS (11.66 GB).
- Descriptive: the KL-against-bytes frontier of all 9 files on A3. No equal-byte estimate is drawn from interpolation.

Descriptive, in this order of GPU priority (fixed now):
- A. KL on A3 for all 9 files. KL on S, T1 and T2 for ISTA's 4 files and UD; these sets are seen for us, and S selected our files.
- B. Speed for P1 and P2 with `scripts/bench-v2m.sh` (pp512, tg128, depth 16K): back to back in both orders, quiet machine, all layers on Metal checked in the log.
- C. Reciprocal same-type tensor swaps, scored on S and T1 only (A3 is kept for the registered files). Where v2c and ISTA IQ3_S hold the same type for the same tensor, swap that tensor class both ways (gate_up, then downs). That compares the two encodings at fixed bytes and fixed type. Splices use `scripts/splice.py --verify`.
- D. If ISTA wins P1: "ISTA map, stock encoder", a llama-quantize build at ISTA IQ3_S's exact per-tensor types with our imatrix, from a BF16 rebuilt from the release. Scored on S, T1 and A3. Path-specific log contrasts are reported, never shares:
  - ISTA against ISTA-map-stock = encoder + calibration + pipeline;
  - ISTA-map-stock against v2a (UD map, stock) = map difference at stock encoding.
- E. Tasks (amendment 15's 300 items, same prompts, decoding and scorer) only for a pair whose KL outcome is equivalent or inconclusive. Report:
  - discordant wins and losses per suite;
  - a paired item bootstrap within suites;
  - a one-sided 95% lower bound on ours minus the other file, against a −5-point margin.

  With 300 items this screens only for large differences; the power from the observed discordance is reported.

Actions, written now:
- P1, ISTA closer:
  - the card gets a dated correction naming the closer files;
  - no further release from the current fixed-scale refiner;
  - backlog T2 (alternating scale updates) first;
  - C and D decide between encoder work and map work.
- P1, ours closer: the card adds the comparison with revisions and date. The "closest" wording stays dropped while UD is closer (P3). Backlog T8 (GSQ head-to-head) is dropped.
- P1, equivalent or inconclusive: no more head-to-heads at 11.5 GB until T2 has a result. Tasks (E) decide the card line.
- P2, ISTA closer: the sibling project corrects c1's claim and stops tuning KL-per-byte promotions.
- P2, ours closer while P1 is lost or tied: c1's gain-restored Q2_0 downs and per-byte ranking are tried at 11.5 GB.
- Either way: the shared guidelines' edge claim is rewritten to the measured result. Card changes are the owner's call.


## 2. Registered results on A3 (built from logs by analysis/a18.py)

## Absolute KL on A3 (mean per chunk; top-1 agreement %; p99 per-token KL)

| file | GB | python | javascript | german | english | math | token-weighted mean |
|---|---|---|---|---|---|---|---|
| c1 (8.9G) | 8.932 | 0.04313 (94.2; 0.469) | 0.05007 (94.5; 0.676) | 0.10443 (85.4; 0.796) | 0.07863 (88.6; 0.637) | 0.06524 (91.5; 0.609) | 0.06830 |
| ISTA IQ2_S | 8.976 | 0.05381 (93.7; 0.670) | 0.06673 (93.7; 0.940) | 0.14865 (83.2; 1.214) | 0.11395 (86.6; 1.126) | 0.08178 (90.3; 0.838) | 0.09298 |
| ISTA Q2_K_XL | 9.793 | 0.03766 (94.7; 0.459) | 0.04534 (94.8; 0.635) | 0.09813 (86.1; 0.833) | 0.07732 (88.8; 0.758) | 0.05592 (92.0; 0.589) | 0.06287 |
| ISTA IQ3_XXS | 10.608 | 0.02814 (95.2; 0.334) | 0.03383 (95.5; 0.481) | 0.06438 (88.4; 0.559) | 0.05338 (90.5; 0.542) | 0.03899 (93.3; 0.378) | 0.04374 |
| ISTA IQ3_S | 11.425 | 0.02133 (95.9; 0.247) | 0.02400 (96.0; 0.326) | 0.04549 (90.4; 0.409) | 0.03713 (92.2; 0.378) | 0.02823 (94.3; 0.276) | 0.03124 |
| v1 | 11.514 | 0.01995 (96.1; 0.226) | 0.02080 (96.4; 0.281) | 0.03838 (90.9; 0.314) | 0.03233 (92.3; 0.284) | 0.02598 (94.3; 0.246) | 0.02749 |
| v2m (refined) | 11.567 | 0.01724 (96.5; 0.208) | 0.01908 (96.5; 0.270) | 0.03066 (92.0; 0.266) | 0.02540 (93.1; 0.238) | 0.02253 (95.1; 0.221) | 0.02298 |
| v2c (refined2) | 11.567 | 0.01474 (96.5; 0.176) | 0.01788 (96.5; 0.242) | 0.03028 (92.2; 0.256) | 0.02545 (93.3; 0.246) | 0.02147 (95.0; 0.208) | 0.02196 |
| Unsloth UD-IQ3_XXS | 11.663 | 0.01432 (96.8; 0.170) | 0.01547 (96.9; 0.212) | 0.02646 (92.2; 0.212) | 0.02289 (93.8; 0.207) | 0.01826 (95.5; 0.172) | 0.01948 |

### P1: v2c (refined2) / ISTA IQ3_S (bytes v2c (refined2) minus ISTA IQ3_S: +141 MB)

- G = 0.710 [0.692, 0.727] -> **v2c (refined2) closer**
- leave-English-out G = 0.716 [0.697, 0.736]; leave-one-document-out range 0.707 to 0.713; domains favouring v2c (refined2): 5 of 5
- per domain: python 0.692 [0.649, 0.740], javascript 0.756 [0.704, 0.808], german 0.664 [0.647, 0.683], english 0.684 [0.651, 0.724], math 0.757 [0.723, 0.790]

### P2: c1 (8.9G) / ISTA IQ2_S (bytes c1 (8.9G) minus ISTA IQ2_S: -44 MB)

- G = 0.757 [0.734, 0.781] -> **c1 (8.9G) closer**
- leave-English-out G = 0.773 [0.747, 0.800]; leave-one-document-out range 0.752 to 0.761; domains favouring c1 (8.9G): 5 of 5
- per domain: python 0.816 [0.748, 0.886], javascript 0.777 [0.725, 0.837], german 0.702 [0.667, 0.739], english 0.694 [0.648, 0.750], math 0.804 [0.751, 0.855]

### P3: v2c (refined2) / Unsloth UD-IQ3_XXS (bytes v2c (refined2) minus Unsloth UD-IQ3_XXS: -97 MB)

- G = 1.116 [1.095, 1.136] -> **Unsloth UD-IQ3_XXS closer**
- leave-English-out G = 1.119 [1.097, 1.142]; leave-one-document-out range 1.112 to 1.119; domains favouring v2c (refined2): 0 of 5
- per domain: python 1.026 [0.981, 1.074], javascript 1.153 [1.086, 1.225], german 1.136 [1.109, 1.164], english 1.101 [1.054, 1.145], math 1.169 [1.144, 1.194]


## 3. Descriptive, seen sets

## KL on the seen sets (P = 3; mean KL, top-1 agreement %)

S selected our files, so these sets favour them. Built by `analysis/a18-seen.py`.

| file | S | T1 | T2 |
|---|---|---|---|
| c1 (8.9G) | 0.07209 (89.0) | 0.06683 (89.7) | 0.04889 (92.1) |
| ISTA IQ2_S | 0.09835 (87.5) | 0.09754 (87.8) | 0.04966 (92.4) |
| ISTA Q2_K_XL | 0.06550 (89.9) | 0.06474 (89.9) | 0.03139 (94.0) |
| ISTA IQ3_XXS | 0.04455 (91.5) | 0.04278 (91.7) | 0.02493 (94.7) |
| ISTA IQ3_S | 0.03076 (92.7) | 0.02916 (93.2) | 0.01782 (95.4) |
| v2c (refined2) | 0.02099 (94.2) | 0.01990 (94.2) | 0.01481 (95.8) |
| Unsloth UD-IQ3_XXS | 0.02050 (94.3) | 0.01939 (94.4) | 0.01361 (96.0) |

## 4. Bytes by role from headers (GB)

| file | gate_up | down | non-expert | notes |
|---|---|---|---|---|
| v2c | 5.712 | 4.586 | 1.256 | UD-style per-layer gate_up layout, refined codes (fixed scales, Gram objective with per-row gain floor) |
| ISTA IQ3_S | 5.223 | 4.300 | 1.890 | GSQ learned codes+scales, RCO per-tensor budget; layers 0-2 gate_up IQ1_S, downs Q2_0; BF16 token_embd (0.93 GB, kept on CPU by llama.cpp, so not on Metal) |
| c1 | 4.544 | 3.153 | 1.223 | gain-restored Q2_0 downs, KL-ranked promotions |
| ISTA IQ2_S | 4.082 | 3.153 | 1.729 | |

A separate preregistered follow-up (amendment 19) now tests eight single cherry-picks of ISTA's layout choices on v2c (starved layers 0-2, 8-bit attn_v in the loop, Q8_0 embedding, BF16 small attention tensors, IQ3_S gate_up in three layers), selected on S, confirmed on T1, then once on a new fresh set A4. Amendment 20 pilots joint scale+code refinement (GSQ's core idea) under the gain floor in two layers.
