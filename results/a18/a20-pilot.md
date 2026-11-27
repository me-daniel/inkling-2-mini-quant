# Amendment 20: scale-refinement pilot (layers 6 and 20)

Built by `analysis/a20.py`. J under G summed over refined experts; both arms start from v2c's codes.

| layer | experts | J start | J after A | J after B | B / A | scale moves (B) | d moves (B) |
|---|---|---|---|---|---|---|---|
| 6 | 126 | 4.58042e+07 | 4.46784e+07 | 4.41977e+07 | 0.9892 | 5,760,864 | 429,148 |
| 20 | 124 | 1.79605e+07 | 1.73292e+07 | 1.73199e+07 | 0.9995 | 578,676 | 1,209,264 |

KL on S: A 0.02109, B 0.02087; B - A -0.00022 [-0.00062, +0.00007]

Registered rule: J not 2% lower in both layers: T2 stops.
