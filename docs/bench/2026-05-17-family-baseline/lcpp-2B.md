| model                          |       size |     params | backend    | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | --------------: | -------------------: |
| qwen35 2B Q4_K - Medium        |   1.18 GiB |     1.88 B | MTL,BLAS   |      12 |           pp128 |      2928.57 ± 65.68 |
| qwen35 2B Q4_K - Medium        |   1.18 GiB |     1.88 B | MTL,BLAS   |      12 |           pp512 |      3770.28 ± 32.69 |
| qwen35 2B Q4_K - Medium        |   1.18 GiB |     1.88 B | MTL,BLAS   |      12 |          pp1024 |       3755.14 ± 2.86 |
| qwen35 2B Q4_K - Medium        |   1.18 GiB |     1.88 B | MTL,BLAS   |      12 |            tg32 |        185.67 ± 0.87 |
| qwen35 2B Q4_K - Medium        |   1.18 GiB |     1.88 B | MTL,BLAS   |      12 |           tg128 |        184.07 ± 0.06 |

build: 0253fb21f (9187)
