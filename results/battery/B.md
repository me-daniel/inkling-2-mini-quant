| task | n | cand | q8 | v2m |
|---|---|---|---|---|
| B1 | 80 | 78 | 78 | 78 |
| B2 | 100 | 75 | 78 | 78 |
| B3 | 60 | 47 | 47 | 49 |
| B4 | 60 | 53 | 57 | 57 |

| task | v2m right, v1 wrong | v1 right, v2m wrong | capped replies v1 / v2m |
|---|---|---|---|
| B1 | 0 | 0 | 0 / 0 |
| B2 | 8 | 5 | 0 / 0 |
| B3 | 2 | 0 | 1 / 3 |
| B4 | 4 | 0 | 2 / 0 |

H3 (B1+B3+B4, n=200): v2m - v1 = +3.0 points, one-sided 95% bounds [+1.1, +3.0], discordant 6 for v2m / 0 for v1 -> PASS (no regression)
