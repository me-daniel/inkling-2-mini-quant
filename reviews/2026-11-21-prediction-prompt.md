You are asked for a pre-registered theoretical prediction, written before any data exist. Reason from the architecture below; do not search the web. Number every point. Be quantitative where you can, and say clearly which statements are derivations, which are approximations, and which are guesses.

## Architecture (Inkling 2 Mini as implemented in llama.cpp)
- Decoder-only, 24 physical layers, hidden size 2304, vocab 201,024, final logit softcap 30 (cap * tanh(x / cap)).
- Every layer: attention (sliding window or full) then a routed MoE FFN: 128 SwiGLU experts, top-8, no shared expert, router bias. Weights of expert e in layer l: gate_up W13 (2304 -> 2*1152) and down W2 (1152 -> 2304).
- "Block attention residuals": instead of a plain residual stream, each layer's attention input and FFN input are mix(blocks, cur, query) = softmax-weighted average (over depth) of rms-normed stored block states (outputs of earlier 6-layer blocks) and the current state; blocks have 6 layers.
- Loop: layers 0-5 (prelude) run once; the core 6-17 runs P = 3 times; layers 18-23 (coda) run once. Each pass starts again from the prelude output (the same tensor for all passes). From pass 2 on, each core layer's Q, K, V and FFN inputs are modulated by a message from the PREVIOUS pass only: message = mix(previous pass's block states + previous pass's output, msg_query) and input' = (1 - forget) * input + gain * message, with gain = tanh(G g), forget = tanh(F g), g = D [rmsnorm(input); rmsnorm(message)] of rank 64. Pass 3 sees pass 2 only, not pass 1. The coda sees the prelude blocks and the LAST pass's blocks.
- Each (pass, layer) has its own KV cache and convolution state; within a pass, later tokens attend to earlier tokens' keys/values of the same pass.

## Intervention
Block B2 = layers 6-11 (the first half of the core). Its routed-expert weights are replaced by a 2-bit quantized version (IQ2_XXS for W13, Q2_0 for W2; relative l2 weight error t roughly 0.1-0.3, Est.) on a chosen subset of passes, everything else fixed at a 4-bit base. Conditions: none (base), pass 1 only, pass 2 only, pass 3 only, all three passes. Also two noise conditions: Gaussian noise with the same relative l2 error as the 2-bit level, either the SAME draw on all three passes (tied, like real shared weights) or a DIFFERENT draw per pass (independent).
Measured per token position over 65,536 WikiText tokens (16 chunks of 4,096): the change of centred log-probabilities relative to the base, Delta_S for subset S, and KL(base || condition).

## Questions
1. Expected ordering and rough ratio of ||Delta_1||, ||Delta_2||, ||Delta_3|| (pass-k-only effects), and why, given the restart and the previous-pass-only messages.
2. Expand the output to second order in the per-pass perturbations. Write the residual R = Delta_all - (Delta_1 + Delta_2 + Delta_3) in terms of the expansion. Which terms make R nonzero (cross-pass Jacobian products, gating nonlinearity, softcap, router top-8 flips)? Predict the typical size of r = ||R|| / ||Delta_all|| (per token and aggregated) and which regime we will most likely see: r ~ 0 ("counts per pass", additive), Delta_all ~ Delta_3 ("counts once, last pass dominates"), or ||Delta_all|| > ||sum Delta_k|| ("compounds").
3. For the noise conditions: under the linearity theorem (Malinovskii et al., arXiv 2411.17525, perplexity increase ~ sum over layers of alpha_l * relative l2 error), KL from independent per-pass noise should be ~ the sum of single-pass KLs. For tied noise, derive the expected excess KL from the cross terms 2 * sum_{j<k} E[Delta_j^T F Delta_k] (F = Fisher of the output distribution). What sign and size do you expect, and what does a tied/independent ratio of 1, 2 or 3 mean?
4. Routing: top-8 of 128 with perturbed inputs is discontinuous. How do route flips affect R and the noise comparison, and how should the analysis separate smooth from flip-driven effects with the data we have (per-token values, optionally router choices)?
5. Statistics: what aggregate of r is robust (mean of per-token r, ratio of sums of squares, median)? Is 65,536 correlated tokens (16 documents-chunks) enough to distinguish r = 0.05 from r = 0.2, and how should the bootstrap be clustered?
6. Write 3 to 5 numbered, falsifiable predictions with numeric thresholds that we can register now, each with what result would contradict it.
7. Pitfalls in this design that could make the measured r or the tied/independent ratio mean something else.

Keep it to 900-1,500 words.
