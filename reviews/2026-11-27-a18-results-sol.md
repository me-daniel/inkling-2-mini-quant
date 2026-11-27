# Review of amendment 18 part A results: sol, 27 Nov 2026

Prompt: `2026-11-27-a18-results-prompt.md` with the ROLE line for sol. Sol (technique), gpt-6.2-sol via codex-ask.sh with web search, effort xhigh.

ROLE line: ROLE: you review TECHNIQUE. Check the statistic, the bootstrap, the sensitivity results and the internal consistency of the numbers (e.g. absolute KL against G, seen vs fresh sets, T2 where c1 and ISTA IQ2_S nearly tie). Flag anything that looks like a measurement artifact and how to rule it out.

## Reply (verbatim)

1. [measured] **The primary statistic reconciles with the absolute KL.** Geometric means of the five reported domain ratios reproduce P1 0.710, P2 0.757, and P3 1.116. Ratios of token-weighted mean KL are 0.703, 0.735, and 1.128. Those ratios give more weight to chunks with higher comparator KL. Every reported token-weighted mean also matches the simple average of the five domain means at displayed precision. Verify that the scored-token counts are equal as intended.

2. [measured] **All three A3 calls survive the reported sensitivity checks.** Dropping English leaves each CI on the same side of 1, and every leave-one-document-out point estimate still clears its pair’s effect-size threshold. For P3, all five domain point estimates favor Unsloth, but Python’s CI spans 1. The displayed P3 domain values give a leave-English-out estimate of 1.11955, which is compatible with the reported 1.119 after rounding the underlying inputs. Publish the unrounded estimates.

3. [needs source] **The bootstrap design fits the statistic, but the implementation needs an audit.** Replay `analysis/a18.py` from the keyed chunk export. Confirm 12 document draws within each domain, with all four chunks and both files carried together, and equal domain weights on every draw. With the same draws, swapping the files should turn \(G\) into \(1/G\) and \([L,U]\) into \([1/U,1/L]\). Also verify that leave-one-document-out recomputes the affected domain from its remaining 11 documents.

4. [inference] **The intervals could be too narrow if nominally distinct documents share sources.** Check for shared repositories and near-duplicates across A3 domains, calibration text, and earlier evaluation sets. If dependencies turn up, rerun a sensitivity that keeps shared sources together. If the code sampling fixed six old and six recent repositories per language, also check a bootstrap that preserves those quotas.

5. [needs source] **A reference or scoring mismatch could shift all the ratios without showing up in the bootstrap.** On a small matched sample, verify the actual three-pass execution, token IDs, scored positions, and vocabulary order. Recompute full-vocabulary \(D_{\mathrm{KL}}(Q8_0\parallel\text{file})\) directly from logits and run Q8_0 against itself through the production scorer to measure its numerical floor. Audit any zero, negative, or nonfinite chunk KL handling before taking logs. Confirm that p99 pools the same scored tokens per domain.

6. [measured] **T2 is a near tie on mean KL and reverses the top-1 ordering for P2.** The c1/ISTA IQ2_S ratios of mean KL are 0.733 on S, 0.685 on T1, 0.984 on T2, and 0.735 on A3. On T2, c1 leads by 0.00077 KL while ISTA leads by 0.3 percentage points in top-1 agreement. Re-score both files against one freshly generated reference on identical T2 positions, then report descriptive paired document intervals for both metrics. Keep S’s role in selecting the files explicit.

7. [measured] **The byte table has an unexplained 12–13 MB per file.** The role totals fall short of the reported sizes by more than rounding can explain. Publish exact file sizes and reconcile them with tensor payloads, metadata, and padding. The reported size ordering supports c1’s A3 dominance on bytes and KL, and a size–KL trade-off for v2c against ISTA IQ3_S.

8. [measured] **The registered win and equivalence rules can overlap.** For example, \(G=0.965\) with CI \([0.952,0.990]\) qualifies as both “ours closer” and “equivalent.” All three A3 calls are unambiguous. Specify precedence before applying the rules to future data.

9. [inference] **The reciprocal swaps estimate the effect of replacing stored tensor payloads in each recipient.** Calibration and upstream pipeline differences travel with those payloads, and the effect can depend on the recipient. Verify that untouched tensors are byte-identical and report both swap directions separately.
