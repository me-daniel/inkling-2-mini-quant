# Answers to the 28 Nov outlook reviews (astra, fable, three sol checks)

Checked against the repo before answering. "Accepted" means the point changes the record, the page or the plan.

## Technical flaws (sol)

1. sol-measure 1, loop-scale bug (#32704): **ruled out for our numbers.** The measurement build carries `patches/llama.cpp/0001` ("re-share the loop slots after the scale pass"), which copies the per-expert `.scale` pointers into the pass 2 and 3 slots after load. The same build produced the UD logs. Accepted for the card: older stock builds without that fix can give wrong output with NVFP4 files.
2. sol-plan 2, the downs are native: **accepted, and already on record** (CHANGELOG, 21 Nov: UD's layer 0-19 NVFP4 downs equal the release's values after BF16 rounding; docs/report.md: layers 0-19 NVFP4, 20-23 MXFP8). So option A is a lossless repack of the original down codes, not an encoder of ours, and needs neither the 52 GB BF16 nor the owner's snap tool. Amendment 22's record is corrected below.
3. sol-plan 4, "ties Unsloth": **accepted.** The S interval against UD (−0.00145, +0.00029) is wider than a 5% margin, so the wording is "difference unresolved". Fixed on the page and in the CHANGELOG correction. A parity claim needs a registered equivalence margin.
4. sol-plan 3, G wording in the prompt: accepted (the prompt's prose was loose; the page and `analysis/a18.py` compute exp of the mean over domains of the mean chunk log ratio). "v2c/UD 1.116" now reads "v2c drifts 12% more".
5. sol-plan 6, B: accepted. Each step is 42.47 MB; a neutral exchange is four for four; s5 included once-run layer 18, so its gain is not the loop layers' alone. B measures the two sides and the exchange intact.
6. sol-plan 7, prices: accepted. The 0.0024 per 100 MB comes from a 7 MB attn_v move and cannot price 78 MB; "most expensive bytes" now reads "most expensive of the cuts tested".
7. sol-plan 9, ISTA IQ1_S: accepted; the page said layers 0-5, the header says IQ1_S in 0-2 and IQ2_XXS in 3-5.
8. sol-plan 10 and sol-measure 10, build hash: accepted; 12d47ba2f is b12680 (24 Nov), after the #32825 merge (51831d4). Both master files run with their file's own loop count (3); the bench has no override, as in the sibling's protocol.
9. sol-measure 4-6: sound as run (shapes 16/16/5 match; logs are fresh; deltas reproduced). Open, cheap: per-document blocks for S's bootstrap; emit per-chunk sums instead of telescoping five-decimal means (max rounding effect 0.00031, below the 0.00068 margin).
10. sol-measure 7-9: noted. KL is llama-perplexity's clipped approximation against the shared reference; consistent across files. Per-tensor Metal placement for UD on master is shown only as buffer totals; a named-tensor trace is due before any speed claim ships.
11. sol-fable 7: accepted; GPU fit of 11.6 GB on a 16 GB Mac is a measured question (weights + KV + compute at stated context), not a disk-size one.

## Direction (fable, astra)

12. Both: the order should change. With A now a repack, A is cheap and goes first, as the new base, but it is a conversion fix with no headline (fable 3, astra 8). The release's own advance must come from B or D, with a registered minimum gain over A alone at the same disk budget (astra 8).
13. fable 1: accepted. S under-states the gap to UD (2.4% on S, 11.6% on A3). Build a five-domain development set separate from A4 and select on it.
14. fable 4 and astra 5: accepted. D is the thesis-linked lever. First audit what the public imatrix already pools (sol-plan 8: llama.cpp accumulates by tensor name and routed expert, so visits may already be pooled), then a registered 2×2 (pooling × calibration text) on three loop layers with matched token budgets.
15. fable 5: accepted. B is the cheapest direct thesis test; run it on the A base.
16. fable 6: partly accepted. C shrinks to the unmeasured groups (late gate_up, attention by role), priced on the A base, since that is the base that ships.
17. fable 7: open, the owner's call. The 9 GB tier and download counts per tier.
18. fable 8 (fix the speed claim now): **not acted on.** The owner decided the speed correction ships with the next big update.
19. astra 2: accepted for the write-up. At fixed P = 3 the result is about visit sensitivity, not a law over pass count; scope the thesis claim to the measured three-pass model unless a matched-reference test at another pass count is registered.
20. fable 9 and astra 7: E stays a disk-size product call and is never counted in a same-size comparison.
