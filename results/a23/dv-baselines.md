# Amendment 23: baselines on DV

Built by `analysis/a23-dv.py`. Mean KL against Q8_0, P = 3.

| file | python | javascript | german | english | math |
|---|---|---|---|---|---|
| v2c | 0.01252 | 0.01836 | 0.03132 | 0.02492 | 0.02031 |
| v2c-A | 0.01181 | 0.01737 | 0.02975 | 0.02401 | 0.01977 |
| UD-IQ3_XXS | 0.01210 | 0.01553 | 0.02720 | 0.02192 | 0.01755 |
| c1 | 0.03769 | 0.04708 | 0.10281 | 0.07814 | 0.06033 |
| c1-A | 0.03720 | 0.04638 | 0.10185 | 0.07699 | 0.05976 |
| UD-IQ2_XXS | 0.03681 | 0.04753 | 0.09922 | 0.07628 | 0.05595 |

| a / b | G [95% CI] | per domain |
|---|---|---|
| v2c-A / UD-IQ3_XXS | 1.068 [1.051, 1.086] | python 0.973, javascript 1.090, german 1.088, english 1.088, math 1.106 |
| v2c-A / v2c | 0.952 [0.940, 0.964] | python 0.940, javascript 0.945, german 0.951, english 0.961, math 0.961 |
| c1 / UD-IQ2_XXS | 1.035 [1.015, 1.053] | python 1.020, javascript 1.010, german 1.041, english 1.036, math 1.067 |
| c1-A / c1 | 0.989 [0.982, 0.997] | python 0.988, javascript 0.989, german 0.990, english 0.985, math 0.993 |
| c1-A / UD-IQ2_XXS | 1.023 [1.004, 1.043] | python 1.008, javascript 0.998, german 1.031, english 1.021, math 1.059 |
