| model                          |       size |     params | backend    | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | --------------: | -------------------: |
| inkling2_loop ?B Q6_K          |  10.71 GiB |    26.07 B | MTL,BLAS   |       5 |           pp512 |       244.94 ± 34.84 |
| inkling2_loop ?B Q6_K          |  10.71 GiB |    26.07 B | MTL,BLAS   |       5 |           tg128 |         19.93 ± 1.00 |

build: 9f5b9d6 (8)
