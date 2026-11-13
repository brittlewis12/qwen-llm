# Prefill standing on M5 Ultra — 2026-11-12

Status: measurement packet for the three tier-1 families on current main
`5ee5c8c4f`. M5 Ultra, 256 GiB, macOS 27.0.1, Metal 4 tensor peak
**131.7 TFLOP/s**. This packet is a fresh prefill standing, not a merge or a
default change.

## Artifacts and build

| Family | Artifact | Path on metagross |
| --- | --- | --- |
| V4.1 | GSQ-RCO IQ2_XS, four shards + IQ4_NL disk Engram | `~/models/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF/DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XS-00001-of-00004.gguf` |
| GLM-5.3 Flash | stock UD-IQ3_XXS, four shards | `~/models/glm-stock-iq3-download/UD-IQ3_XXS/GLM-5.3-Flash-UD-IQ3_XXS-00001-of-00004.gguf` |
| Flash-Next | GSQ-RCO IQ3_S, two shards | `~/models/ISTA-DASLab-Qwen3.8-Flash-Next-GSQ-RCO-GGUF/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf` |

`qwen` `a702fb0c…` and `qwen-bench` `0738d77f…` were built from clean
`5ee5c8c4f` (`build-info` status `match`, `build_dirty=false`). The tokenizer
binary used by the serve probe is `qwen-tok` from the sibling
`perf/v41-prefill` worktree (`qwen-tok.sha256` in the raw directory); it is
tokenizer-only and does not participate in timing.

## Protocol

- `qwen-bench suite` with the synthetic uniform-vocabulary stream, seed 1,
  fresh state per retained repetition. Cold cells are `--pp N --depth 0`;
  append cells are `--pp W --depth D` after the suite's untimed fill.
- V4.1 cold cells use the production `QWEN_V41_PREFILL_EXECUTION=shrinking_ced`
  (also the default). Nonzero-depth appends remain Full, as production does.
- Cold 4K/8K/32K: two separate processes with reversed row order (A: 4K then
  8K; B: 8K then 4K; 32K A/B), 180 s between cold arms. Cold 128K: two
  separate single-retained-sample processes with a 300 s cooldown (GLM used
  60 s under the corrected shared-host rule).
- Append: one width per process, `--runs 2`; V4.1 used 120 s between widths.
- The session began under the strict repo rule (pressure 0). The lead's
  corrected shared-host rule was adopted partway through: settle to pressure
  <=1, accept 0–1 during/after a run, abort at >=2, record the level per cell.
  Cells measured only under the corrected rule are identified by their
  thermal record. A cell that failed twice for the same reason was skipped.
- Every build, model load and server load held the canonical Metal lease
  through `scripts/bench/with-metal-lease --task prefill-standing --for 15m`
  (builds: 8–15 m; measurement holds <=15 m). Measurement processes ran at
  normal priority; builds used `nice -n 10`. Host load and thermal pressure
  are recorded per invocation.

## Roofline accounting

Achieved TFLOP/s is `2 × active_parameters_per_token × tok/s / 10^12`
against the measured **131.7 TFLOP/s** Metal 4 tensor peak. Active parameters
are counted from the GGUF headers (all shards), not from a runtime census:

- dense tensors counted once;
- routed expert tensors counted at `top_k / expert_count`;
- one token-embedding row counted per token;
- the vocabulary head is excluded from the per-token count (it runs once per
  prefill call at the endpoint, as llama-bench `pp` does);
- Flash-Next's 51,200,245,760-parameter `per_layer_token_embd` PLE table is
  excluded from the dense count because each token gathers rows rather than
  multiplying the table; one 2,560-wide row is counted;
- V4.1's 110.6 GB Engram companion is a disk lookup table, not a matmul and
  is excluded; the two 157,286,400-parameter `engram_wkv` backbone tensors are
  counted because they are always-on dense projections.

| Family | dense | routed total | top-k / experts | embedding row | active / token | peak tok/s at 131.7 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| V4.1 | 6,975,323,608 | 543,581,798,400 | 6 / 384 | 5,120 | 15,468,794,328 | 4,257 |
| GLM-5.3 | 7,837,063,006 | 311,653,564,416 | 8 / 288 | 4,096 | 16,494,110,558 | 3,992 |
| Flash-Next | 3,676,300,160 | 120,795,955,200 | 10 / 512 | 2,560 | 6,035,598,720 | 10,910 |

The head is 661,913,600 (V4.1), 634,388,480 (GLM) and 635,699,200
(Flash-Next) parameters; it would add about 2 × head / call to a cell's
useful FLOPs but is not in the per-token number. Attention/index products,
dequantization, padding and launch costs are outside this useful-matmul
numerator; the percentage is an end-to-end bound, not hardware occupancy.

## Cold prefill

| Family | Cell | tok/s | samples | n | TFLOP/s | % of 131.7 | thermal |
| --- | --- | ---: | --- | ---: | ---: | ---: | --- |
| V4.1 | pp4096 | **1,912.68** | 1907.7 / 1913.4 / 1907.5 / 1922.1 | 4 | 59.2 | 44.9% | 0 |
| V4.1 | pp8192 | **2,425.40** | 2416.1 / 2429.6 / 2427.0 / 2429.0 | 4 | 75.0 | 57.0% | 0 |
| V4.1 | pp32768 | **2,891.25** | 2881.7 / 2891.9 / 2893.2 / 2898.2 | 4 | 89.5 | 67.9% | 0 |
| V4.1 | pp131072 | **2,699.61** | 2697.0 / 2702.2 (cooled singles) | 2 | 83.5 | 63.4% | 0 |
| GLM-5.3 | pp4096 | **1,142.59** | 1142.8 / 1142.4 | 2 | 37.7 | 28.6% | 0 |
| GLM-5.3 | pp8192 | **1,127.05** | 1125.4 / 1128.6 | 2 | 37.2 | 28.2% | 0 |
| GLM-5.3 | pp32768 | **1,096.74** | 1097.4 / 1096.1 (clean retry; first attempt hit pressure 2 and is excluded) | 2 | 36.2 | 27.5% | 0 |
| GLM-5.3 | pp131072 | **1,003.28** | 1002.0 / 1004.6 | 2 | 33.1 | 25.1% | 1 / 0 |
| Flash-Next | pp4096 | **2,218.63** | 2218.9 / 2219.4 / 2219.1 / 2217.2 | 4 | 26.8 | 20.3% | 0 |
| Flash-Next | pp8192 | **1,930.50** | 1930.2 / 1931.4 / 1929.1 / 1931.3 | 4 | 23.3 | 17.7% | 0 |
| Flash-Next | pp32768 | **1,878.88** | 1879.9 / 1878.4 / 1878.9 / 1878.4 | 4 | 22.7 | 17.2% | 0 |

The 8K CED row is now **2,425.4**, up from the scoreboard's 2,315. The cold
32K and 128K CED rows are **2,891.3** and **2,699.6**, up from 2,302 and
1,981. GLM 4K is **1,142.6**, up from 1,031. Flash-Next 4K is **2,218.6**,
level with the 2,219 scoreboard row.

## Appends

| Family | Depth / new rows | tok/s | samples | n | TFLOP/s | % of 131.7 | thermal |
| --- | --- | ---: | --- | ---: | ---: | ---: | --- |
| V4.1 | 48K / 256 | **583.87** | 585.4 / 582.3 | 2 | 18.1 | 13.7% | 0 |
| V4.1 | 48K / 1024 | **1,093.57** | 1093.4 / 1093.7 | 2 | 33.8 | 25.7% | 0 |
| V4.1 | 48K / 2048 | **1,227.66** | 1180.9 / 1274.4 | 2 | 38.0 | 28.8% | 0 |
| V4.1 | 80K / 256 | **470.54** | 478.6 / 462.4 | 2 | 14.6 | 11.1% | 0 |
| V4.1 | 80K / 1024 | skipped | 894.2 / 802.5 (pressure 2 twice; not accepted) | — | — | — | 2 |
| V4.1 | 80K / 2048 | **1,048.51** | 1058.4 / 1038.6 | 2 | 32.4 | 24.6% | 0 |
| GLM-5.3 | 48K / 256–2048 | not completed | shared-host lease contention | — | — | — | — |
| GLM-5.3 | 80K / 256–2048 | not completed | shared-host lease contention | — | — | — | — |
| Flash-Next | 48K / 256–2048 | not completed | shared-host lease contention | — | — | — | — |
| Flash-Next | 80K / 256–2048 | not completed | shared-host lease contention | — | — | — | — |

## Serve: 48K cold-root TTFT

| Family | Reps | TTFT | prompt tokens | prefill mode | prefill_ms |
| --- | --- | ---: | ---: | --- | ---: |
| V4.1 | 21.585 s / 21.356 s (mean **21.470 s**) | 21.47 s | 49,182 | `shrinking_ced+full` | 21,487.8 / 21,262.6 ms |

Thermal 0 before and after both reps; fresh `qwen serve` process per rep,
60 s apart. The TTFT is 49,182 / 21.47 = **2,291 tok/s** including the first
token. GLM-5.3 and Flash-Next serve cells were queued but not reached before
the deadline (shared-host lease contention).

## MLX-Serve 26.11.2 baseline

The matched V4.1 artifact was re-run through mlx-serve **26.11.2** under
`~/models/mlx/mlx-serve-26.11.2/`, first as the four split shards and then as
a merged single GGUF. Two independent blockers were found, so no local MLX
prefill number is claimed:

1. **ds4 bridge rejects split GGUF.** `--engine ds4` reports
   `SplitGgufUnsupported: ... is one shard of a 4-file split GGUF`.
2. **After merging, ds4 rejects the Engram.** The merged 155,225,823,904-byte
   GGUF loads far enough to stage the Metal kernels, then fails with
   `ds4: unsupported V4.1 Engram row encoding` for the companion
   `DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf`.

The MLX-native GGUF path (`--mlx-gguf`, which is where the project's earlier
sorted-run adaptation came from) was then queued, but the shared M5 lease was
continuously held by other lanes through the deadline, so it did not produce a
cell. This is a reproducibility conflict with the earlier
mlx-serve V4.1 comparison (~1,800 tok/s non-replay / ~2,790 with bounded
replay): that result needs either a supported Engram encoding or an MLX pack,
not the production IQ4_NL companion as served here.

Public competitor figures quoted on the scoreboard were not re-measured and
must not be treated as same-machine cells: MLX-Serve ~3,880 short / ~1,800
long; tacos8me 2,498 at 8K; oMLX oQ2e 2,453 / 2,279. For a local MLX re-run
the minimal packs would be: `ddalcu/DeepSeek-V4.1-Flash-1017-iQ-MLX-3.0bpw`
(about 200 GB at 3.0 bpw, not downloaded), `TensorFold/GLM-5.3-Flash-MLX-oQ4-MTP`
(about 180 GB, not on disk), and `ddalcu/Qwen3.8-Flash-Next-iQ-MLX-3.3bpw`
(about 64 GB download / 52 GB resident per the mlx-serve reference, not on
disk). No large download was started.

## Thermal and host load

All accepted V4.1 cold and 48K append cells were at pressure 0 before and
after. The V4.1 80K/1024 cell was pressure 2 twice and is not accepted. The
first GLM 32K attempt was pressure 2 and is excluded; the clean retry and both
128K singles were 0/0. All accepted Flash-Next cold cells were 0/0. The
1-minute host load ranged from about 4 to 73 during the session; the high
readings coincide with other lanes' `qos-nice0`/`v41-prefill` campaigns and
`~/ghost` shadPS4 work. Per-invocation load/thermal records are in the raw
JSONL.

## Raw evidence

`metagross:~/qwen-runs/2026-11-12-prefill-standing/`:
`v41-cold-small.jsonl`, `v41-cold-128k.jsonl`, `v41-append-48000-A-w.jsonl`,
`v41-append-80000-A-w.jsonl`, `v41-80k-2048.jsonl`, `glm-cold-small.jsonl`,
`glm-cold-128k.jsonl`, `glm-32k-128k.jsonl`, `fn-cold-short.jsonl`,
`v41-serve-cold-48k.jsonl`, `mlx-v41-*.jsonl`, `stage-results.log`, plus the
per-invocation stdout/stderr logs, `gguf_active.py` and the model-load lease
history. The merge log and derived merged GGUF are deleted after the record.

## Limits

- The shared M5 was heavily loaded by other lanes and `~/ghost` work for much
  of the session. Five campaigns aborted under the strict early protocol; the
  corrected 0–1 rule was used for the later cells. Some cells were still
  skipped after two failures.
- Cold 128K points are single retained samples per process, as in the
  established scoreboard protocol. Append cells are two retained samples from
  one process; the reversed second append arm was not completed for any family.
- GLM and Flash-Next append cells and GLM/Flash-Next serve TTFT were not
  reached before the deadline. They remain open cells, not measured zeros.
- The MLX comparison is incomplete for the reason above. No quality claim is
  made for bounded replay; it remains below the project's exactness gate.
