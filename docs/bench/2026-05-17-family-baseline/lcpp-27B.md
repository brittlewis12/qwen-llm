| model                          |       size |     params | backend    | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | --------------: | -------------------: |
| qwen35 27B Q4_K - Medium       |  15.65 GiB |    26.90 B | MTL,BLAS   |      12 |           pp128 |        225.96 ± 0.33 |
| qwen35 27B Q4_K - Medium       |  15.65 GiB |    26.90 B | MTL,BLAS   |      12 |           pp512 |        237.30 ± 3.33 |
| qwen35 27B Q4_K - Medium       |  15.65 GiB |    26.90 B | MTL,BLAS   |      12 |          pp1024 |        214.88 ± 2.29 |
| qwen35 27B Q4_K - Medium       |  15.65 GiB |    26.90 B | MTL,BLAS   |      12 |            tg32 |         21.75 ± 0.04 |
| qwen35 27B Q4_K - Medium       |  15.65 GiB |    26.90 B | MTL,BLAS   |      12 |           tg128 |         21.03 ± 0.16 |

build: 0253fb21f (9187)
