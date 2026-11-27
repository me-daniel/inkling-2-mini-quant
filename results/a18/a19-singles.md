# Amendment 19: single changes on v2c, KL on S at P = 3

Built by `analysis/a19.py` from `logs/ppl-a19-*-S-P3.log`; delta = KL(change) - KL(v2c), paired over S's 16 chunks, chunk-bootstrap 95% CI. MB are decimal, from the headers.

v2c: KL 0.02099

| change | what | KL | delta [95% CI] | file MB | Metal MB |
|---|---|---|---|---|---|
| s1 | gate_up 0-2 IQ1_S (c1) | 0.02348 | +0.00249 [+0.00181, +0.00318] | -212.3 | -212.3 |
| s2 | down 0,1,3 Q2_0 (c1) | 0.02270 | +0.00171 [+0.00125, +0.00219] | -286.7 | -286.7 |
| s3 | attn_v 6-17 Q8_0 | 0.02082 | -0.00017 [-0.00029, -0.00005] | +6.9 | +6.9 |
| s4 | token_embd Q8_0 | 0.02036 | -0.00063 [-0.00106, -0.00028] | +231.6 | +0.0 |
| s5 | gate_up 6,7,8,12,18 IQ3_XXS stock | 0.02043 | -0.00056 [-0.00100, -0.00013] | +212.3 | +212.3 |
| s6 | sink + conv projections BF16 | 0.02106 | +0.00007 [-0.00010, +0.00024] | +2.3 | +2.3 |
| s7 | s6 + attn_rel BF16 | 0.02101 | +0.00002 [-0.00010, +0.00015] | +27.3 | +27.3 |
| s8 | gate_up 9,10,15 IQ3_S stock | 0.02089 | -0.00010 [-0.00050, +0.00030] | +95.6 | +95.6 |

## Subsets the rule builds next (additive prediction)

- file: s2+s3+s4: predicted KL 0.02190 (delta +0.00091), file -48.2 MB, Metal -279.8 MB
- file: s2+s3+s4+s7: predicted KL 0.02192 (delta +0.00093), file -20.9 MB, Metal -252.5 MB
- metal: s2+s3+s4+s8: predicted KL 0.02180 (delta +0.00081), file +47.3 MB, Metal -184.2 MB
- metal: s2+s3+s4+s7+s8: predicted KL 0.02182 (delta +0.00083), file +74.6 MB, Metal -156.9 MB

## Subsets, measured on S

| subset | KL | delta [95% CI] | beats v2c (upper < 0) |
|---|---|---|---|
| s2+s3+s4 | 0.02228 | +0.00129 [+0.00086, +0.00177] | no |
| s2+s3+s4+s7 | 0.02219 | +0.00120 [+0.00080, +0.00162] | no |
| s2+s3+s4+s8 | 0.02221 | +0.00122 [+0.00061, +0.00187] | no |
| s2+s3+s4+s7+s8 | 0.02226 | +0.00127 [+0.00072, +0.00183] | no |
