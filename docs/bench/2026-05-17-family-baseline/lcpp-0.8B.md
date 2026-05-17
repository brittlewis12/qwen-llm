| model                          |       size |     params | backend    | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | --------------: | -------------------: |
| qwen35 0.8B Q4_K - Medium      | 497.39 MiB |   752.39 M | MTL,BLAS   |      12 |           pp128 |     5459.18 ± 288.20 |
| qwen35 0.8B Q4_K - Medium      | 497.39 MiB |   752.39 M | MTL,BLAS   |      12 |           pp512 |      7965.23 ± 33.26 |
| qwen35 0.8B Q4_K - Medium      | 497.39 MiB |   752.39 M | MTL,BLAS   |      12 |          pp1024 |      7803.46 ± 47.36 |
| qwen35 0.8B Q4_K - Medium      | 497.39 MiB |   752.39 M | MTL,BLAS   |      12 |            tg32 |        249.63 ± 1.45 |
| qwen35 0.8B Q4_K - Medium      | 497.39 MiB |   752.39 M | MTL,BLAS   |      12 |           tg128 |        251.49 ± 1.08 |

build: 0253fb21f (9187)
