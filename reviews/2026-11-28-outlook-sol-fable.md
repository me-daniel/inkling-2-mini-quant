# Technique review of fable's reply: sol, 28 Nov 2026

Model: gpt-6.2-sol via codex-ask.sh, effort xhigh. Inputs at b1990a6 (tree dirty only in untracked card drafts). Prompt: `reviews/2026-11-28-outlook-prompt.md` sections 1-4 and fable's reply.

## Reply (verbatim)

1. [inference] The arithmetic checks out. UD’s S KL is 0.02050, v2c trails by 2.4%, and the donor lowers v2c’s KL by 5.2%. Applying that reduction to A3’s domain-weighted ratio gives a 5.8% gap, conditional on transfer across datasets and aggregation methods. Donor–UD equivalence remains unestablished. A separate five-domain development set supports selection while preserving A4 for confirmation.

2. [inference] The 29% KL lead applies to the larger ISTA comparison. Both task intervals include zero and allow gains of several points. They cannot establish practical equivalence or predict the task difference against UD. That requires a direct paired comparison with a prespecified equivalence margin and adequate precision.

3. [inference] The donor gain is resolved on S and similar in magnitude on T1 and T2. The swap changes format and encoded values together. The 2% bound observed in the ISTA swaps cannot predict what an independent NVFP4 encoder will achieve. Shared gate-up placement also leaves the full layout comparison open. The original checkpoint is an alternative to the 52 GB rebuild. IQ4_NL and NVFP4 both store 4.5 bits per weight, and NVFP4’s 64-weight blocks divide 1152. [ggml format definitions](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-common.h)

4. [inference] The pass ratios and corpus gaps motivate D. They cannot identify a calibration or pooling effect. The registered 2×2 can estimate both effects and their interaction on the three tested layers if it matches calibration sample budgets and holds refinement fixed. Match corpus domains and lengths to isolate text provenance. Twelve layer refinements take 5–6.2 CPU hours. Full-model transfer needs separate validation.

5. [inference] The 42.5 MB transfer per pair and roughly four GPU minutes per step check out. The ladder tests the net fixed-size exchange on v2c. Additive pass errors can coexist with a larger penalty from downgrading the chosen late layers. Separate upgrade and downgrade measurements expose that tradeoff. Controlled pass splits are needed to attribute it to reuse. [ggml block sizes](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-common.h#L2776-L2805)

6. [measured] The reported losses and one-third arithmetic are sound. They show that summing single-edit prices missed these combinations. v2c prices describe v2c, and the donor may better approximate A’s down layout. Check price transfer with matched edits and measure the chosen combination on the intended shipping base. Four or five runs cover only the sampled edits.

7. [needs source] Neither fit claim follows from disk size. Metal’s recommended working set is a performance guideline, and reported values differ across 16 GB M4s. Test offloaded weights, KV cache and compute buffers at a stated context and batch size. The 0.44 GB difference is correct. A paired dev-set comparison can estimate the KL gap. Download counts measure downloads and only approximate audience. [Apple Metal documentation](https://developer.apple.com/documentation/metal/mtldevice/recommendedmaxworkingsetsize.md) · [16 GB M4 logs](https://github.com/ggml-org/llama.cpp/issues/27667)

8. [measured] The supplied rates imply a 1.6% prompt and 5.7% generation advantage on the tested build. A dated correction should identify the build and benchmark settings. Upstream Metal NVFP4 support merged on November 23. [llama.cpp merge](https://github.com/ggml-org/llama.cpp/pull/32825)

9. [inference] E raises v2c to about 11.80 GB, above UD’s 11.66 GB, and lowers S KL by 3.0%. Its quality result remains reportable with disk size stated. Reusing the same ISTA task items adds no independent items. The ISTA layout and joint-refinement findings apply to the variants tested.

10. [inference] A fresh comparison at each size can establish a gain against the corresponding baseline. Attributing it to reuse requires pass-controlled evidence. Null B and D results constrain the tested swaps and refinements to the precision of their intervals. They cannot establish that allocation generally fails to exploit reuse.
