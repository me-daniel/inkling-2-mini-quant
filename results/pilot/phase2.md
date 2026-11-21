# Pilot phase 2: block map and pass sweep

Written by `analysis/phase2_report.py` from `results/pilot/kl.tsv`. KL against the Q8_0 reference on S (16 x 4,096 tokens).

## Block map at P = 3 (one block changed, everything else Q8_0)

| Block | Layers | 2-bit KL | per layer | 3-bit KL | per layer |
|---|---|---|---|---|---|
| B1 | 0-5, before the loop | 0.0279 | 0.0046 | 0.0053 | 0.00088 |
| B2 | 6-11, looped | 0.0623 | 0.0104 | 0.0082 | 0.00137 |
| B3 | 12-17, looped | 0.0852 | 0.0142 | 0.0077 | 0.00128 |
| B4 | 18-19, after the loop | 0.0359 | 0.0180 | 0.0023 | 0.00116 |
| B5 | 20-23, after the loop, MXFP8 source | 0.0667 | 0.0167 | 0.0047 | 0.00117 |

Additivity: sum of the five blocks against the uniform build. 2-bit 0.2781 vs 0.3046 (ratio 1.10); 3-bit 0.0282 vs 0.0221 (ratio 0.78).

## Pass sweep (uniform builds, each against the reference at the same pass count)

| Level | P = 2 | P = 3 | P = 4 |
|---|---|---|---|
| 2bit | 0.4285 | 0.3046 | 0.3365 |
| 3bit | 0.0456 | 0.0221 | 0.0327 |

Run time per sweep evaluation (mean of both levels), against the layer visits 12 + 12 P:

- P = 2: 157 s, ratio to P = 3 0.75 (visits ratio 0.75)
- P = 3: 210 s, ratio to P = 3 1.00 (visits ratio 1.00)
- P = 4: 308 s, ratio to P = 3 1.47 (visits ratio 1.25)
