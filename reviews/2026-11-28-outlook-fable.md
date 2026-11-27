# Outlook review: fable (direction), 28 Nov 2026

Model: Fable, effort medium, prompt text only, no tools used. Inputs at b1990a6 (tree dirty only in untracked card drafts). Prompt: `reviews/2026-11-28-outlook-prompt.md` with the direction ROLE line.

## Reply (verbatim)

Direction review: the plan's order (A, then C, then D) is backwards for both the thesis and the release. Put D and B first, shrink C, and treat A as catch-up.

1. [inference] The donor "tie" is on the selection set, not where it counts. Donor minus UD on S is −0.00060, so UD is about 0.02050 on S and v2c trails by only 2.4% there, against 11.6% on A3. NVFP4 downs buy about 5% on S; if that transfers proportionally, v3 still trails UD by about 6% on fresh domains. Action: build a five-domain dev set, separate from A4, and select on it rather than on S.

2. [measured] Users cannot see the gap being chased. A 29% KL lead over ISTA gave no resolved task difference at 300 items (+2.0 [−0.8, +4.5], +1.7 [−0.4, +3.2]), so 5-12% against UD is invisible in use. Action: make "tie or beat UD at 11.6 GB" a reported number, not the release gate.

3. [inference] A is the best-evidenced lever, but it is catch-up. The gain is resolved on S and consistent on T1 and T2. With gate_up placement already copied, though, v3 = A is Unsloth's layout re-encoded. Same-type encoders differ by at most 2%, so expect the donor's number and no more. It also needs the 52 GB rebuild and splits the release into two files. Action: do A third in owner-hours and ship it without a headline.

4. [inference] D is the only option that serves the thesis and the product together, so it goes first. The 0.87-1.23 ratio with the third pass dominating says per-visit importance should matter. The S-versus-A3 gap and ISTA's narrowing on T2 both point at calibration data. As proposed, D confounds the two changes. Action: the owner writes the visit-pooled accumulation first; then run a registered 2×2 pilot (pooling × calibration text) on three loop layers, about 6 CPU hours, before the 10-12 hour full re-refine.

5. [inference] B is the cheapest direct test of the thesis claim and should run this week. Each step is 42.5 MB and one 4-minute KL run. If loop errors add across passes, moving bits from once-run layers 19-23 into loop layers 6, 7, 8, 12 must help; if it does not, that is also a thesis figure. Action: register, then run the four-step ladder on v2c.

6. [measured] Cut C to a stub. Every size-neutral ISTA-style subset lost by 0.0012-0.0013, and non-additivity (about 0.0004) is a third of that effect, so an allocator fed single-step prices will mispredict. Prices measured on a donor base describe a file that never ships. Action: price only the unmeasured groups (late gate_up, attention by role) on v2c, 4-5 runs. Keep the allocator as a thesis function only if B shows a loop-versus-late asymmetry.

7. [needs source] The 9 GB tier probably has the larger audience and is our weaker position. An 11.6 GB file does not fit the GPU budget of a 16 GB Mac; c1 does. Yet UD-IQ2_XXS matched c1 at 0.44 GB less, unresolved, on a borrowed set. Action: check download counts per size tier for Unsloth and ISTA, run c1 against UD-IQ2_XXS on the new dev set, and if D works, apply it at 9 GB first.

8. [measured] Fix the speed claim now, not in v3. The card says 3.3× / 1.75×; on master since 23 Nov it is 443/31.4 against 436/29.7. A stale claim left up for weeks undermines every other number on the card. Action: ask the owner for a go on a dated two-line correction this week.

9. [inference] Drop entirely:
   - E as a quality argument: it pushes the file above UD's size, so ship it or not as a disk-size product call and never count it in a comparison.
   - Further ISTA-layout subsets.
   - More task-battery runs against ISTA.
   - Upstream work until the error-versus-pass analysis is drafted.
   - Joint scale-and-code refinement stays stopped.

10. [inference] The one thing worth announcing is a loop-aware method of our own (visit-pooled importance or pass-aware allocation) with a resolved gain on a fresh set, shown at both sizes and tied to the thesis ratio figure. If B and D both come out null, announce nothing big: ship NVFP4 downs quietly as a point release and put the weeks into the write-up, where "passes add, allocation cannot exploit it" is still a chapter.
