# Amendment 18 parts B and C (built by `analysis/a18-bc.py`)

## B. Speed (llama-bench, t/s; mean of the two orders, spread in brackets)

Runs 6 (v2c first) and 7 (ista-IQ3_S first).

| test | v2c | ista-IQ3_S | v2c / ista-IQ3_S |
|---|---|---|---|
| pp512 | 271.7 (2.7) | 269.6 (7.1) | 1.008 |
| tg128 | 21.1 (0.1) | 21.2 (0.0) | 0.999 |
| pp512 @ d4096 | 189.3 (3.9) | 190.4 (3.2) | 0.994 |
| tg128 @ d4096 | 20.0 (0.1) | 19.8 (0.6) | 1.013 |
| pp512 @ d16384 | 133.9 (0.2) | 132.9 (1.2) | 1.008 |
| tg128 @ d16384 | 18.5 (0.0) | 18.6 (0.1) | 0.996 |

Process peak memory footprint of the depth runs (GB, per order; mapped Metal buffers are not counted, see `results/a18/load.md` for GPU bytes): v2c 2.81 / 2.81; ista-IQ3_S 2.57 / 2.81

Runs 8 (c1 first) and 9 (ista-IQ2_S first).

| test | c1 | ista-IQ2_S | c1 / ista-IQ2_S |
|---|---|---|---|
| pp512 | 275.7 (2.6) | 283.1 (0.8) | 0.974 |
| tg128 | 22.4 (0.1) | 23.3 (0.1) | 0.960 |
| pp512 @ d4096 | 195.7 (4.1) | 197.9 (1.3) | 0.989 |
| tg128 @ d4096 | 21.2 (0.2) | 22.1 (0.1) | 0.962 |
| pp512 @ d16384 | 137.7 (1.2) | 137.3 (0.6) | 1.002 |
| tg128 @ d16384 | 19.6 (0.1) | 20.3 (0.0) | 0.962 |

Process peak memory footprint of the depth runs (GB, per order; mapped Metal buffers are not counted, see `results/a18/load.md` for GPU bytes): c1 2.81 / 2.80; ista-IQ2_S 2.81 / 2.82

## C. Reciprocal swaps (KL at P = 3; swap minus recipient, paired chunk bootstrap 95% CI)

| swap | what | S | S Δ [CI] | T1 | T1 Δ [CI] |
|---|---|---|---|---|---|
| v2c-gu6i | ISTA's gate_up in 6 layers (6, 11, 13, 17, 20, 22) into v2c | 0.02071 | -0.00028 [-0.00075, +0.00021] | 0.01965 | -0.00025 [-0.00057, +0.00009] |
| ista-gu6v | v2c's gate_up in those 6 layers into ISTA | 0.03040 | -0.00036 [-0.00088, +0.00017] | 0.02924 | +0.00008 [-0.00041, +0.00062] |
| v2c-dni | ISTA's IQ4_NL downs in 21 layers into v2c | 0.02096 | -0.00003 [-0.00029, +0.00025] | 0.01966 | -0.00024 [-0.00061, +0.00010] |
| ista-dnv | v2c's IQ4_NL downs in those 21 layers into ISTA | 0.03040 | -0.00036 [-0.00091, +0.00017] | 0.02872 | -0.00044 [-0.00074, -0.00014] |
| v2c-embi | ISTA's BF16 token_embd into v2c (added before data, labelled) | 0.02037 | -0.00062 [-0.00095, -0.00035] | 0.01962 | -0.00028 [-0.00047, -0.00009] |

Recipients: v2c S 0.02099, T1 0.01990; ista-IQ3_S S 0.03076, T1 0.02916
