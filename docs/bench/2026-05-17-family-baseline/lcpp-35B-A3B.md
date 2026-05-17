| model                          |       size |     params | backend    | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | --------------: | -------------------: |
| qwen35moe 35B.A3B Q4_K - Medium |  20.60 GiB |    34.66 B | MTL,BLAS   |      12 |           pp128 |       805.65 ± 19.52 |
| qwen35moe 35B.A3B Q4_K - Medium |  20.60 GiB |    34.66 B | MTL,BLAS   |      12 |           pp512 |       1425.09 ± 9.25 |
| qwen35moe 35B.A3B Q4_K - Medium |  20.60 GiB |    34.66 B | MTL,BLAS   |      12 |          pp1024 |       1411.16 ± 4.03 |
| qwen35moe 35B.A3B Q4_K - Medium |  20.60 GiB |    34.66 B | MTL,BLAS   |      12 |            tg32 |         72.01 ± 0.24 |
| qwen35moe 35B.A3B Q4_K - Medium |  20.60 GiB |    34.66 B | MTL,BLAS   |      12 |           tg128 |         72.64 ± 0.18 |

build: 0253fb21f (9187)
