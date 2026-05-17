| model                          |       size |     params | backend    | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | --------------: | -------------------: |
| qwen35 9B Q4_K - Medium        |   5.28 GiB |     8.95 B | MTL,BLAS   |      12 |           pp128 |        746.22 ± 5.44 |
| qwen35 9B Q4_K - Medium        |   5.28 GiB |     8.95 B | MTL,BLAS   |      12 |           pp512 |        838.48 ± 0.80 |
| qwen35 9B Q4_K - Medium        |   5.28 GiB |     8.95 B | MTL,BLAS   |      12 |          pp1024 |       822.90 ± 14.26 |
| qwen35 9B Q4_K - Medium        |   5.28 GiB |     8.95 B | MTL,BLAS   |      12 |            tg32 |         63.73 ± 0.16 |
| qwen35 9B Q4_K - Medium        |   5.28 GiB |     8.95 B | MTL,BLAS   |      12 |           tg128 |         61.46 ± 1.17 |

build: 0253fb21f (9187)
