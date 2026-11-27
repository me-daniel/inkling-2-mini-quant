# Review of amendment 18 part A results: astra, 27 Nov 2026

Prompt: `2026-11-27-a18-results-prompt.md` with the ROLE line for astra. Astra (concept), gpt-6.2-astra via codex-ask.sh with web search, effort high.

ROLE line: ROLE: you check the CONCEPT. Do the conclusions follow from the registered outcomes? What may and may not be claimed (scope: released files, not methods)? Which follow-up (amendments 19/20, or something else) is best supported by these results? You may web-search ISTA-DASLab's Inkling-2-Mini-GSQ-RCO-GGUF card and the GSQ paper.

## Reply (verbatim)

1. [measured] **P1 is a KL–size trade-off.** On A3, v2c’s geometric mean chunk-KL ratio to ISTA IQ3_S was **0.710** (95% CI 0.692–0.727). That clears the registered win threshold in aggregate, and all five domain intervals favor v2c. Use the measured cost of **141 MB more** (about 1.2%) in the dated card comparison.

2. [measured] **P2 is the one registered win on both KL and file size.** c1’s ratio to ISTA IQ2_S was **0.757** (0.734–0.781), all five domain intervals favor c1, and c1 is **44 MB smaller**. Claim dominance over that specific release on those two measures. P1 also went to v2c, so the registered condition for transferring c1’s promotions to 11.5 GB was unmet.

3. [measured] **P3 rules out calling v2c “closest” on A3.** The v2c/Unsloth ratio was **1.116** (1.095–1.136), so Unsloth was closer for **97 MB more**. All five point estimates favor Unsloth, but Python’s interval includes 1 (0.981–1.074). Keep “closest” off the card and put this trade-off beside P1 in the shared guidelines.

4. [inference] **The defensible card claim is pairwise fidelity to your Q8_0.** Name the release revisions, patched build, P = 3, five-domain A3 mixture, and 1,024-token context with second-half scoring. Link the Step 0 packaging checks, whose pass results are absent from the excerpt. Task quality, chat, longer contexts, and fidelity to the original checkpoint require separate evidence. ISTA’s published task scores use the released checkpoint as reference. [ISTA model card](https://huggingface.co/ISTA-DASLab/Inkling-2-Mini-GSQ-RCO-GGUF)

5. [measured] **A3 carries the confirmation.** Removing English or any one document leaves all three point estimates on the same side of their registered effect thresholds. S, T1, and T2 add descriptive context only. For example, c1’s mean-KL edge over IQ2_S narrows to 0.04889 versus 0.04966 on T2, where top-1 agreement favors IQ2_S. Keep those results out of the primary estimate.

6. [inference] **The headers support an allocation hypothesis.** Relative to ISTA IQ3_S, v2c spends 489 MB more on `gate_up`, 286 MB more on downs, and 634 MB less on non-expert tensors. P2 also favors the file with more `gate_up` bytes, while down totals match to reported precision. Neither contrast identifies the cause of the KL gap. ISTA held its BF16 embedding fixed outside RCO’s search and reports CPU placement, so calling it an RCO mistake or counting its bytes as Metal residency would misstate the comparison. [ISTA model card](https://huggingface.co/ISTA-DASLab/Inkling-2-Mini-GSQ-RCO-GGUF)

7. [inference] **Finish B and C first.** B supplies the missing speed comparison. C is the strongest pending test of whether the *released encodings* differ at fixed tensor type and bytes. Report both swap directions by tensor class and host file on S/T1. Calibration and pipeline travel with the encoded weights, so even a consistent swap effect cannot establish encoder-algorithm superiority. The registered results retire T8 and trigger neither D nor E.

8. [inference] **Of amendments 19 and 20, prioritize 19 for the next file decision.** The headers give a concrete reason to test individual allocation choices, while P1 provides no reason to copy ISTA’s whole map. Report each edit’s byte delta alongside KL, including savings from a downgrade. Preserve S selection, T1 confirmation, and the single A4 evaluation. Use Unsloth as the practical benchmark for any resulting candidate.

9. [inference] **Keep amendment 20 as a controlled pilot.** The GSQ paper motivates jointly optimizing assignments and scales. A3 supplies no evidence that fixed scales caused v2c’s residual gap to Unsloth. Compare joint refinement with fixed-scale refinement on the same two layers, holding map, bytes, calibration, and gain floor fixed. A gain there would justify a broader test. [GSQ paper](https://arxiv.org/abs/2604.18556)
