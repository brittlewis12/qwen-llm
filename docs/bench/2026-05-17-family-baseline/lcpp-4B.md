| model                          |       size |     params | backend    | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | --------------: | -------------------: |
| qwen35 4B Q4_K - Medium        |   2.54 GiB |     4.21 B | MTL,BLAS   |      12 |           pp128 |      1291.49 ± 11.31 |
| qwen35 4B Q4_K - Medium        |   2.54 GiB |     4.21 B | MTL,BLAS   |      12 |           pp512 |       1482.91 ± 1.16 |
| qwen35 4B Q4_K - Medium        |   2.54 GiB |     4.21 B | MTL,BLAS   |      12 |          pp1024 |       1486.40 ± 0.42 |
| qwen35 4B Q4_K - Medium        |   2.54 GiB |     4.21 B | MTL,BLAS   |      12 |            tg32 |         92.35 ± 0.26 |
| qwen35 4B Q4_K - Medium        |   2.54 GiB |     4.21 B | MTL,BLAS   |      12 |           tg128 |         92.00 ± 0.06 |

build: 0253fb21f (9187)
