# v0.450 Narrow Current-HEAD Guardrail

Goal: get a fast, trustworthy current-HEAD sanity read after v0.444-v0.449
without running a full paired family sweep.

## Commands

```sh
target/release/qwen-bench ctx-sweep \
  -m /Users/tito/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  --checkpoints 8192,16384,32768 --window 5 --fresh-per-checkpoint \
  > target/profiles/v0450-a3b-q4-ctx-sweep-8192-32768.out

target/release/qwen-bench suite \
  -m /Users/tito/models/unsloth-Qwen3.5-122B-A10B-GGUF/UD-Q4_K_XL/Qwen3.5-122B-A10B-UD-Q4_K_XL.gguf \
  --tg 128 --runs 1 \
  > target/profiles/v0450-a10b-q4xl-suite-tg128.json

target/release/qwen-bench suite \
  -m /Users/tito/models/Qwen3.6-27B-Q4_K_M.gguf \
  --pp 512,4096 --tg 128 --runs 1 \
  > target/profiles/v0450-27b-q4-suite-pp512-4096-tg128.json
```

Power state in JSON rows: AC power, high-power mode, no thermal/performance
warnings.

## Results

| Model | Shape | Result |
| --- | --- | ---: |
| A3B Q4_K_M | ctx8192 decode window | `101.7 t/s` |
| A3B Q4_K_M | ctx16384 decode window | `99.3 t/s` |
| A3B Q4_K_M | ctx32768 decode window | `93.5 t/s` |
| A10B Q4_K_XL | tg128 | `45.62 t/s` |
| 27B Q4_K_M | pp512 | `244.31 t/s` |
| 27B Q4_K_M | pp4096 | `224.93 t/s` |
| 27B Q4_K_M | tg128 | `23.51 t/s` |

## Read

No fresh guardrail regression appears. A3B long decode still degrades gradually
rather than cliffing, A10B decode remains healthy, and dense 27B prompt/decode
sanity rows remain in the expected band. This packet does not replace a paired
llama.cpp family sweep; it is a cheap current-HEAD anchor for deciding the next
optimization branch.
