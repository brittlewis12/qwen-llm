# A3B All-Quant Guard — v0.365

Clean A3B quant-family guard after the v0.362 fast IQ3 SwiGLU and v0.364 fast
IQ4_XS down changes. Same physical box, sequential runs, no concurrent GPU
benches. Rows are `runs=1` spot checks intended to catch broad regressions and
rerank, not replace repeated release sweeps.

## Identity

- `qwen-llm`: `6c6ce6941`
- `llama.cpp`: `c818263f2a` (build 9833, backends `MTL,BLAS`)
- GPU: `Apple M4 Max`
- QWEN env: none

## Results

| A3B quant | pp512 lcpp | pp512 qwen | pp512 read | tg128 lcpp | tg128 qwen | tg128 read |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Q3_K_M | `1443.2` | `1552.9` | `1.08x` | `80.82` | `103.19` | `1.28x` |
| Q4_K_M | `1413.8` | `1520.6` | `1.08x` | `77.80` | `108.09` | `1.39x` |
| Q6_K | `1388.7` | `1452.0` | `1.05x` | `79.02` | `100.09` | `1.27x` |
| Q8_0 | `1454.5` | `1564.7` | `1.08x` | `73.19` | `91.19` | `1.25x` |
| UD-IQ4_XS | `1441.1` | `1512.6` | `1.05x` | `79.99` | `102.92` | `1.29x` |

## Read

All measured local A3B quant rows are now green for both `pp512` and `tg128`.
The low-bit decode branch should stop being scored as a coverage/parity gap; the
next A3B work should use hardware-headroom criteria, especially long-context
attention/GDN and broader-family guards.

## Method

- qwen rows: `qwen-bench suite --pp 512 --tg 128 --runs 1`
- llama.cpp rows: `llama-bench -p 512 -n 128 -r 1 -o json`
- Per model: qwen then llama.cpp, sequentially, never in parallel
