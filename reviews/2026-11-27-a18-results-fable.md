# Review of amendment 18 part A results: fable (direction), 27 Nov 2026

Prompt: `2026-11-27-a18-results-prompt.md` with the ROLE line for direction ("is the project spending its next GPU days on the right thing? Is chasing UD-IQ3_XXS at 11.6 GB worth it, or should effort go to the small file, to tasks, to the write-up, or to something else? Argue against the current plan where you can."). Fable, effort medium, given only the prompt file.

## Reply (verbatim)

1. [measured] The 11.6 GB chase is over a small gap that looks like generalisation, not layout. v2c/UD is 1.024 on S, 1.026 on T1, 1.088 on T2, and G = 1.116 on A3. The gap is smallest on python (1.026, CI spans 1) and largest on math (1.169) and javascript (1.153), which our calibration does not cover. Action: before any layout work, treat calibration breadth and selection on S as the prime suspects.

2. [measured] Refined2 bought almost nothing outside code. From v2m to v2c on A3: python −14.5%, javascript −6.3%, math −4.7%, german −1.2%, english +0.2%. Action: stop refiner rounds selected on S; a further round will most likely repeat this pattern.

3. [inference] Amendment 19 is the weakest use of GPU days. It cherry-picks eight layout choices from a file that lost P1 by 29%, and selects them on S, the set whose selection already fails to transfer (point 1). ISTA spends 1.89 GB on non-expert tensors against our 1.26 GB and loses; that is evidence against its layout, not for it. Action: pause A19 until part C is in, then run at most the two picks with a mechanism you can state in advance.

4. [inference] Gate amendment 20 on part C. C compares the two encoders at fixed bytes and type, which is the cheap test of whether joint scale+code learning pays here. If ISTA's tensors do not win the same-type swaps, A20 has no motivation. Action: no A20 GPU time before C reports.

5. [inference] Even a full win over UD would be invisible to users. The gap is 0.0025 nats token-weighted (0.02196 against 0.01948), with top-1 agreement within 0.5 points in every domain. On the frontier's own slope (about −0.41 log-KL per GB between ISTA's IQ3_XXS and IQ3_S), UD's extra 97 MB explains roughly 4 of the 11.6 points, leaving about 7% at equal bytes. The registration forbids this as an estimate, so it is orientation only. Action: drop "closest at 11.6 GB" as a goal; the 300-item battery cannot detect a 7% KL difference.

6. [measured] The small file is the stronger result, and it is under-tested. c1 dominates ISTA IQ2_S (G = 0.757, 44 MB smaller) and sits within 8.6% KL of ISTA Q2_K_XL while 0.86 GB smaller. Two holes:
   - T2: c1 0.04889 against ISTA IQ2_S 0.04966, a 1.6% lead, and ISTA has the higher top-1 (92.4 against 92.1). The lead nearly vanishes on one seen set. Action: say what T2 contains and explain this before claiming anything general.
   - Exclusions: AtomicChat, bartowski IQ2_S and APEX-I-Nano were dropped because they lost to v1, an 11.5 GB file. That says nothing about c1 at 8.9 GB. Action: score them against c1, on A4 or a new fresh set, before any 9 GB "closest" wording.

7. [needs source] Whether Unsloth has a UD file near 9 GB is not in this document. If it has one, P2's dominance claim needs that pair. If it has none, 9 GB is the one size where our file leads with no stronger rival, and that is where GPU days should go. Action: check the Hub listing, then decide.

8. [inference] No task evidence will come out of A18 as registered. Part E runs only for equivalent or inconclusive pairs, and all three pairs resolved, so every claim stays "closer to Q8_0". The pair with the best chance of a detectable task difference is c1 against ISTA IQ2_S (0.068 against 0.093; german top-1 85.4 against 83.2). Action: run the 300 items on that pair, labelled exploratory, with the power statement. It is cheap, and it is the only route to a claim users care about.

9. [needs source] There is a hole in our frontier between 8.9 and 11.5 GB. ISTA IQ3_XXS (10.6 GB, KL 0.0437) and Q2_K_XL (9.8 GB) have no counterpart from us. If the default Metal budget on a 16 GB Mac is about two thirds of RAM, an 11.6 GB file does not fit and a 10.6 GB one does; that figure is from memory, not this document. Action: verify the budget; if it holds, a c1-method file at about 10.5 GB is worth more than any gain at 11.6 GB.

10. [inference] The write-up is the most time-sensitive item. ISTA published on 26 Nov; a public preregistration with tight CIs, 5 of 5 domains and a self-reported loss to Unsloth is credible now and stale in a month. Parts B and C complete it, and B may matter more than KL, since ISTA's 0.93 GB BF16 embedding stays on CPU. Note that part D never runs (ISTA did not win P1), so attribution rests on C alone and the text must say so. Action: fix the byte deltas first (registration says 137 and 48 MB, results say 141 and 44), then draft the write-up before starting new experiments.
