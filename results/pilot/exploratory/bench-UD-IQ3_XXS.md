| model                          |       size |     params | backend    | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | --------------: | -------------------: |
| inkling2_loop ?B IQ3_XXS - 3.0625 bpw |  10.85 GiB |    26.07 B | MTL,BLAS   |       5 |           pp512 |         72.86 ± 0.52 |
| inkling2_loop ?B IQ3_XXS - 3.0625 bpw |  10.85 GiB |    26.07 B | MTL,BLAS   |       5 |           tg128 |         11.80 ± 1.33 |

build: 9f5b9d6 (8)
