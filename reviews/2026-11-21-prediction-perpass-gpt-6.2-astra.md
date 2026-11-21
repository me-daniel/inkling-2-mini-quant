# Pre-registered prediction: per-pass test (written before any per-pass data)

- Date: 2026-11-21, committed before the first per-pass run
- Model: GPT-6.2 Astra, effort medium (Codex CLI, no web)
- Prompt: architecture facts and the run-4 design with addenda A and B, see `reviews/2026-11-21-prediction-prompt.md`

## Reply (verbatim)

1. **Pass ordering.** My prediction is \(\|\Delta_3\|>\|\Delta_2\|>\|\Delta_1\|\) in corpus RMS, with a **guessed** ratio \(\|\Delta_1\|:\|\Delta_2\|:\|\Delta_3\|\approx 0.15:0.4:1\). The architectural **derivation** is about paths. A pass-3 B2 perturbation can reach the coda through both the final pass’s B2 block state and its subsequent computation. A pass-2 perturbation must first cross one message interface, and a pass-1 perturbation must cross two. The restart removes the direct current-state path between passes. It does not erase information that pass 2 has incorporated from pass 1. The numerical guess assumes each interface transmits roughly 40% of the downstream perturbation norm. Rank 64 constrains the gate controller, while the term \(\mathrm{gain}\odot\delta\mathrm{message}\) can transmit changes in many more directions. Neither rank nor restart proves contraction.

2. **Additivity and the residual.** Let \(y\) be the vector of centred log-probabilities at one token, and let \(h_k\) be the B2 weight change at use \(k\), treating the three uses as independent arguments before setting their changes equal. On a smooth branch with fixed routes, Taylor expansion **gives**
   \[
   \Delta_{\rm all}=\sum_k J_kh_k+\tfrac12\sum_k H_{kk}[h_k,h_k]
       +\sum_{j<k}H_{jk}[h_j,h_k]+O(\|h\|^3).
   \]
   Subtracting the three measured single-pass effects cancels the linear and pure quadratic terms. Thus \(R=\sum_{j<k}H_{jk}[h_j,h_k]+O(\|h\|^3)\), with only mixed terms surviving at higher orders. A product of propagation Jacobians can already contribute to a *single* \(J_j\). It contributes to \(R\) when the later perturbation changes how the earlier one propagates. For example, changing a later down-projection acts on an activation already changed by an earlier pass. Input-dependent gain and forget gates, depth mixing, attention, SwiGLU and the final softcap also supply mixed curvature. Vocabulary centring cancels the log-softmax normalizer exactly. A top-8 boundary crossing can invalidate the fixed-route expansion altogether.

   My **guess** is a median token \(r\) of 0.05–0.15 and a corpus RMS ratio of 0.08–0.20, with a longer upper tail for individual tokens. I expect approximate vector additivity and a dominant pass-3 contribution. Those properties can coexist. To assess “counts once,” measure \(\|\Delta_{\rm all}-\Delta_3\|/\|\Delta_{\rm all}\|\) separately. To assess compounding, compare the norms directly: even a substantial \(R\) can reduce the all-pass norm if it opposes the sum.

3. **Tied versus independent noise.** For base probabilities \(p\), set \(F=\operatorname{diag}(p)-pp^\top\). The local KL **approximation** is \(\tfrac12\Delta^\top F\Delta\). Write \(A_k=\mathbb E\|J_kh\|_F^2\), where \(\|v\|_F^2=v^\top Fv\), and \(C_{jk}=\mathbb E[(J_jh)^\top F(J_kh)]\), averaging over matched noise draws and positions. Independent zero-mean draws cancel the cross terms in expectation. Tying yields
   \[
   \frac{\mathbb E\,{\rm KL}_{\rm tied}}{\mathbb E\,{\rm KL}_{\rm ind}}
   \simeq 1+\frac{2\sum_{j<k}C_{jk}}{\sum_k A_k}.
   \]
   The excess *KL* is \(\sum C_{jk}\) after the Taylor factor of one half. This derivation uses local output linearity and noise independence. The stated layerwise perplexity rule alone does not establish it for repeated weight uses.

   A ratio of 1 means zero **net** alignment, 2 means cross terms add another independent-noise baseline, and 3 is the linear upper bound, requiring equal Fisher magnitudes and perfect positive alignment. Negative net alignment gives a ratio below 1. If the Fisher RMS magnitudes follow the ratio in point 1, even perfect alignment gives only \(1.55^2/(0.15^2+0.4^2+1)\approx2.03\). My **guess** is 1.05–1.35. Reused experts and the common prelude favor some alignment, but different message paths can rotate or reverse the output changes.

4. **Route flips.** A near tie between experts ranked eighth and ninth can turn a small input change into a discrete expert substitution. An all-pass run may cross a boundary that every single-pass run misses, or reverse a single-pass flip. Tying can make boundary crossings more correlated across uses, with either sign of the KL effect. With route logs, compare each condition’s selected expert *sets* against base and stratify residual energy by first divergence and flip count. Include earlier context tokens whose changed K/V or convolution state reaches the measured token. A same-token route match alone cannot certify a smooth path. With outputs only, report the fraction of total residual energy in the largest 1% of token residuals. This measures concentration; attributing it to routing would require route evidence. If reruns are feasible, replaying base route indices while recomputing their continuous weights gives a direct control.

5. **Aggregation and uncertainty.** My primary statistic is \(r_{\rm RMS}=\sqrt{\sum_i\|R_i\|_2^2/\sum_i\|\Delta_{{\rm all},i}\|_2^2}\), using the same vocabulary centring everywhere. It measures the fraction in RMS amplitude and avoids division by tiny token-level effects. Report the median token ratio alongside it because a few high-energy positions can dominate the primary statistic. Report zero-denominator counts separately. Bootstrap whole documents, keeping all conditions paired and recomputing the ratio in each replicate. Use whole 4,096-token chunks if those are the independent sampling units; group chunks from the same document when possible. Sixteen independent clusters may distinguish 0.05 from 0.20 if the effect is consistent, but token count alone gives no power guarantee. Report the cluster interval and leave-one-cluster-out range. Both quantify corpus variability conditional on the noise draws.

6. **Falsifiable registrations.** These are **guesses**. Use pooled Euclidean RMS for the quantization predictions and pooled mean KL for noise.

   1. The pass-1/pass-3 RMS ratio will be 0.05–0.30 and the pass-2/pass-3 ratio 0.25–0.65, with pass 2 exceeding pass 1. Any failed bound or reversed ordering contradicts this prediction.
   2. Quantized all-pass \(r_{\rm RMS}\) will be 0.08–0.20. A value outside that interval contradicts it.
   3. The corpus norm ratio \(\|\Delta_{\rm all}\|/\|\sum_k\Delta_k\|\), stacking tokens before taking norms, will be 0.90–1.10. A value outside contradicts it, with a value above 1.10 supporting net amplification.
   4. Tied/independent mean KL will be 1.05–1.35. A measured ratio outside contradicts the run-level prediction. Generalizing its sign to other draws requires seed replication.

7. **Interpretation pitfalls.** Compute exact KL from base probabilities separately from Euclidean centred-log-probability norms. Rare vocabulary items can dominate the latter while carrying little Fisher weight. Match Gaussian perturbation norms tensor by tensor against the actual **4-bit-to-2-bit** difference, including separate W13 and W2 scales. Quantization error can be structured and weight-correlated. Independent Gaussian KL therefore predicts the sum of *Gaussian* single-pass KLs locally; quantized single-pass KLs cannot test that equality. At relative weight error 0.1–0.3, higher-order effects may matter, and a single draw per condition confounds tying with draw variability. Finally, teacher-force identical tokens, reset caches consistently at chunk boundaries, preserve separate pass caches, and measure a repeated-base numerical floor. A stale cache, a different token history, or subtraction near that floor can all produce an apparent interaction.
