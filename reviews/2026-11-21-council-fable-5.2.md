# Council: 48-hour plan

- Date: 2026-11-21
- Model: fable-5.2
- Prompt: `reviews/2026-11-21-council-prompt.md`

## Reply (verbatim)

# Council reply (Fable 5.2), 21 Nov 2026

Live checks ran around 01:10 UTC on 21 Nov. Labels: V = vendor/authors, I = independent, Est. = my estimate, Verify = not checked.

## What changed since the brief

- **Unsloth's ladder now targets mainline.** `unsloth/Inkling-2-Mini-GGUF` has a `Mainline_Support/` folder with the full UD ladder in the `inkling2_loop` layout, e.g. UD-IQ3_XXS at 11,663,310,784 bytes. Files end in `.gguf_file` and need PR #32704; without it they report perplexity around 130 instead of 7.65 (V).
- **PR #32704** is open, ready for review, authored by danielhanchen, with no reviews yet; two approvals are required (I, PR page). **#32695** is still a draft with no maintainer comment.
- **IndrasMirror/Inkling-2-Mini-XQ-Uncensored** announces per-layer expert formats "chosen from each expert's measured output error on held-out text". The pipeline is private, the file is not uploaded and every quality field is "TBD" (V).
- No bartowski or mradermacher repo matches "Inkling-2-Mini" yet (I, HF API search, 32 hits).
- hermitdave's RCO card does not mention the loop; all 48 expert tensors sit at the 3-bit floor (V).

## 1. Verdict

**Hybrid: run Plan B's pipeline tonight, but lead with a measurement note (block map plus a pass-count sweep). The model upload is optional and gated.**

The deciding reason: steps 1-3 of B (reference, floors, splicer, KLD harness) are Plan A's G0 work, so doing them now costs the thesis nothing. The "special model" on its own is weak, because the mainline niche it relied on closed a few hours ago.

## 2. Is B's "special" claim real?

Mostly not, as written.

- **Novelty of the map.** A 5-block one-at-a-time sweep is a standard sensitivity sweep. AtomicChat already spares "the 8 busiest layers" (V), and IndrasMirror claims measured per-layer allocation (V, unpublished). The numbers for this model would be new; the method is not.
- **Confound.** At P=3, blocks L6-11 and L12-17 are visited three times. The map cannot tell "sensitive because reused" from "sensitive because of depth". Calling the build "loop-aware" (B step 5) would overclaim.
- **InterpreticAI as the bar.** Their card gives no tensor types and no quality numbers (V). If it is stock `llama-quantize` IQ3_XXS, the down projections fell back to IQ4_NL, which makes it B's own "3-bit pool" (Est.). Beating it with 5 blocks and 2-3 levels is likely, by perhaps 5-15% KLD (Est.), but it would say little.
- **The real bar** is Unsloth UD-IQ3_XXS in the mainline layout at 11.66 GB, measured on your harness. Their 0.0269 is against their own Q8_K_XL on undisclosed domains (V) and cannot be compared with your numbers. AtomicChat AD-IQ3_XXS (12.58 GB, fork only) is larger and needs a second engine, so leave it out and say so.
- **Remaining niche.** A BF16-derived file has no `.scale` tensors, so it loads on release builds from b12600 with no patch. That holds only until #32704 merges, probably within days (Est.).

## 3. What makes it more special in 48 hours

**Add a pass-count sweep.** The pass count is a load-time key, so no patch is needed.

- Build Q8_0 reference logits at P=1, 2 and 3. Score each pool file against the same-P reference.
- If KLD grows faster than linearly with P, that is the first direct test of AtomicChat's untested guess that "rounding adds up over the 48 layer passes".
- Cost: two extra references of 13.2 GB each (16 chunks, scaled from 52.7 GB per 64) and six extra evaluations, about 1.5-2 h unattended (Est.).
- Limits: Mini is trained at P=3, so P=1 and P=2 are off-distribution, and whether the varloop card's warning also applies to Mini is Verify. This is a pilot, not the H1 test. H1 still needs the untie patch.
- Thesis safety: log it in the changelog before data and use a corpus disjoint from the confirmatory set.

**Second, cheap:** post your failing-without-fix test and reproduction as a review comment on #32704. It has no reviewers. About 30 minutes, and you post it yourself.

I would skip the 16 GB target and the IQ2 build. UD-Q2_K_XL (9.14 GB) and AD-IQ2_XXS (10.43 GB) already sit there, and stock Q2_0 ignores the imatrix, so you would start behind.

## 4. Measurement traps

1. **Reference.** Q8_0 sits 0.0030-0.0033 from BF16 (V, ggml-org and AtomicChat) and only 97.7% top-1 (V). At a KLD near 0.027 that floor is about 11%. Report paired differences only, never absolute KLD against others' cards. If GCP is approved (2-3 GPU-h, Est.), a BF16 reference on the same chunks removes the issue.
2. **16 chunks.** 65,536 tokens is enough for paired comparison but not for choosing and judging on the same text. Use set S for the map and a fresh set T for the head-to-head, with a bootstrap CI over chunks.
3. **Calibration overlap.** I did not inspect bartowski v6's contents (Verify). Grep S and T against it. Unsloth and InterpreticAI used unknown calibration data, so overlap can cut either way; state that.
4. **Splicing.** Splicing every block from pool A must reproduce pool A's sha256. One spliced mix must match a direct `--tensor-type` requantize tensor by tensor. Check alignment and `general.file_type`.
5. **Additivity.** One-at-a-time deltas need not add. Report predicted against measured KLD for the final build.
6. **Size.** Compare language-model GGUF bytes only, within 0.5%. Exclude mmproj (0.64 GB) and the DSpark drafter.
7. **Engine.** Run the UD file on your pinned #32704 build and your own file on the same binary.
8. **Stability.** 30 GSM8K items have a standard error near 7 points, so they are a smoke test only. Unsloth recommends temperature 1.0 with no truncation (V), so tail mass matters. Add ten 2,000-token generations at those settings, a prompt of at least 16K tokens, JSON tool-call validity, and two non-English prompts.

## 5. Kill criteria and card wording

**Stop everything if:**
- Q8_0 run-to-run KLD is not near zero, or
- the pass-count override does not change runtime in line with 24/48/72 visits, or
- your IQ4_NL against Q8_0 is far from the 0.0095 scale.

**No model upload unless**, on set T:
- paired KLD beats both the same-size uniform build and UD-IQ3_XXS at equal or fewer bytes,
- the 95% CI excludes zero,
- top-1 is not worse, and
- every stability check is clean.

If it fails, publish the note only.

**May claim:**
- "block-wise allocation chosen from a measured 5-block table at P=3";
- exact bytes, tokens, reference and commit;
- "loads on release builds from b12600 without patches";
- "exploratory".

**May not claim:**
- "loop-aware" or "pass-aware";
- "first" or "best";
- any task-quality claim;
- any comparison with AD or UD card numbers;
- anything about the fork files;
- causal language about the loop from the P sweep.

## 6. Plan (T = start)

| Time | Work |
|---|---|
| T+0 | **Decision 1:** approve the hybrid. Write the changelog entry (exploratory; S/T corpora; gates above). |
| T+0 to 1.5 | Download, convert, build Q8_0. Fetch UD-IQ3_XXS (mainline) and InterpreticAI's file. |
| T+1.5 to 3 | Reference logits at P=1/2/3 on S and at P=3 on T. Run-to-run floor. Build pools (3-bit, IQ4_XS). Splicer hash tests. |
| T+3 to 5 | Five-block map at P=3 on S. Pass sweep for each pool. |
| Sleep | |
| T+12 | **Decision 2:** you pick the allocation from the table. |
| T+12 to 14 | Build. Head-to-head on T. Stability battery. |
| T+14 | **Decision 3:** upload gate. Either way, draft the note: map, P sweep, limits. |
| T+16 to 20 | Rerun the P sweep on a second corpus (code or multilingual). Post the #32704 review comment. |
| T+24 | **Decision 4:** publish the note, with the model if the gate passed. |
| T+24 to 48 | Back to Plan A: start the untie patch, and write into the changelog what the pilot showed and what stays confirmatory. Recheck HF for bartowski and the #32704 merge before any further claim. |

## Sources

- https://huggingface.co/api/models?search=Inkling-2-Mini&sort=lastModified&direction=-1&limit=100
- https://huggingface.co/unsloth/Inkling-2-Mini-GGUF
- https://huggingface.co/unsloth/Inkling-2-Mini-GGUF/raw/main/Mainline_Support/README.md
- https://huggingface.co/AtomicChat/Inkling-2-Mini-GGUF
- https://huggingface.co/InterpreticAI/Inkling-2-Mini-IQ3_XXS-GGUF
- https://huggingface.co/IndrasMirror/Inkling-2-Mini-XQ-Uncensored
- https://huggingface.co/hermitdave/Inkling-2-Mini-MLX-RCO-3.5bpw
- https://github.com/ggml-org/llama.cpp/pull/32704
- https://github.com/ggml-org/llama.cpp/pull/32695
