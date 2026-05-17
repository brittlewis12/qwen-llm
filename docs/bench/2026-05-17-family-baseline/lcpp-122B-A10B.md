| model                          |       size |     params | backend    | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | --------------: | -------------------: |
| qwen35moe 122B.A10B Q4_K - Medium |  71.73 GiB |   122.11 B | MTL,BLAS   |      12 |           pp128 |        264.13 ± 1.61 |
| qwen35moe 122B.A10B Q4_K - Medium |  71.73 GiB |   122.11 B | MTL,BLAS   |      12 |           pp512 |        449.81 ± 3.72 |
| qwen35moe 122B.A10B Q4_K - Medium |  71.73 GiB |   122.11 B | MTL,BLAS   |      12 |          pp1024 |        436.94 ± 3.61 |
| qwen35moe 122B.A10B Q4_K - Medium |  71.73 GiB |   122.11 B | MTL,BLAS   |      12 |            tg32 |         34.21 ± 0.01 |
| qwen35moe 122B.A10B Q4_K - Medium |  71.73 GiB |   122.11 B | MTL,BLAS   |      12 |           tg128 |         33.80 ± 0.13 |

build: 0253fb21f (9187)
