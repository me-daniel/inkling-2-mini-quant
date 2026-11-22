# Unslop: paper (paper/main.tex)

- Date: 2026-11-23
- Pass 1: Fable 5.2, effort medium, edited `paper/main.tex` in place. Commit `3b7103f`.
- Pass 2: GPT-6.2 Sol, effort medium (`.claude/skills/crosscheck/codex-ask.sh gpt-6.2-sol PROMPT REPLY medium`, Codex CLI, isolated, read-only). The reply was the full LaTeX file, applied unchanged. Commit `3388d3a`.
- Input: `paper/main.tex` only, no other repository content and nothing from `private/`.
- Checks after each pass: `paper/check-numbers.py --diff` (the multiset of numbers is identical before and after both passes), labels, refs, citations, `\code`, `\nolinkurl`, environments, sections and the TikZ figure unchanged (script), no dashes, "we", "our" or banned words; `paper/check-numbers.py` finds every number in `results/`, `CHANGELOG.md` or the logs (348 of 348).

## Corrections after pass 1 (commit `dce3b24`)

Fable flagged four sentences it doubted but left alone. All four were changed:
1. "All of them sit near 1": now names the B3 interval, which excludes 1.
2. "The pairwise Fisher cosines ... which is where the surcharge comes from" quoted the 2-bit cosines for a noise result: now gives the tied and independent noise cosines (`results/pilot/perpass-h1-h5.md`).
3. "the candidate beat it clearly": now "had lower KL than it on every chunk of both sets" (0 of 16, 0 of 5).
4. "UD's gate_up tensors carry the whole gap (0.00810 of the 0.00782)": now says they recover slightly more than the whole layout.

## Corrections after pass 2

1. Abstract, "The full input second moment accounts for the advantage of UD-IQ3_XXS's encoding": overclaim, restored to the measured ranking (worse under the per-column metric, better under the full Gram).
2. Introduction: restored "Looped models break that assumption." (the paragraph jumped without it).
3. Stability: restored that thirty GSM8K items give no accuracy estimate.
4. Pilot: "On these three layers the interval includes zero" had lost its conclusion; restored that refined and UD tensors are level within that interval.
5. Limits: restored that the nine tied-to-independent ratios are not nine independent replications.

## Prompts

Pass 1 (Fable): the unslop rules in `.claude/skills/unslop/SKILL.md`, first person singular, no dashes, no colon joining clauses, and the hard constraints below.

Pass 2 (Sol), sent before the file:

````text
You are editing a LaTeX technical report (a master's thesis pilot on quantizing a looped mixture-of-experts language model) so it reads like its sole author, Daniel, a master's student, wrote it himself in a plain, careful technical voice, first person singular. Not an AI voice. Another editor has already done one pass. This is a second, light pass: change only what still needs changing, and leave good sentences alone.

Unslop rules:
- Remove AI patterns: significance inflation, promotional words, superficial -ing phrases, vague attributions, AI vocabulary (additionally, crucial, delve, enhance, showcase, underscore, pivotal, landscape, testament, intricate, interplay, notably, robust, leverage), "serves as / stands as / features / boasts", negative parallelisms ("not just X, it's Y", punchy "X, not Y" endings), forced groups of three, synonym cycling (pick one term and repeat it), false ranges, filler ("in order to", "it is important to note"), excessive hedging, generic conclusions, tidy aphorisms, chatbot phrases.
- Voice: vary sentence rhythm, be specific, plain and direct. First person singular ("I", "my") for the author's own work, never "we" or "our".

Prose rules:
- No em dashes or en dashes anywhere, including LaTeX "---" and "--".
- No colon joining two clauses of a sentence. A colon before a list, an equation or a short enumeration is fine.
- Sentence-case headings, straight quotes in the LaTeX sense (``...'' is fine), no emojis, no new bold.

Hard constraints (a script checks them; any violation gets your edit reverted):
- Keep every number exactly as written, the same number of times (byte counts, bracketed intervals, ratios, hashes, dates, counts, signs). Do not round, convert, spell out, merge, drop or add any number.
- Do not change the preamble (everything before \begin{document}), the tikzpicture, any \label, \ref, \cite, \citep, \citet, \code{...}, \nolinkurl{...}, math ($...$ and the equation environment) or the numeric table cells. Captions and prose cells may be lightly edited, keeping their numbers.
- Do not add facts, claims, explanations or caveats. Do not remove any caveat, limit or scope qualifier (exploratory, post hoc, one machine, one engine commit, "on these documents at P = 3", "as registered", "point only", "inference", "untested", "descriptive").
- Keep the FAIL of the release gate against UD-IQ3_XXS stated plainly, and keep the later comparison labelled post hoc and as not changing the FAIL.
- Never use the words best, fastest, beats, loop-aware, novel.
- Keep technical terms consistent: v1, v2a, v2r, v2m, candidate, KL, top-1, pass, visit, block, B1 to B5, gate_up, down, S, T1, T2, T3, UD-IQ3_XXS, Gram, residual share, norm ratio.
- The result must be valid LaTeX with the same environments, sections, tables and figures in the same order.

Return only the full edited LaTeX file, from \documentclass to \end{document}, nothing else: no preamble text, no code fence, no notes.

--- FILE ---
````
