# V4.1 three-agent standing: the −4.9% is a page-cache/time artifact

M5 Ultra (`metagross`), 2026-11-12/13. Branch `fix/v41-serve-regression`,
based on main `540c0a2ab`. No product change was made and the branch is not
merged.

## Question

The 2026-11-12 standing recorded DeepSeek V4.1 Flash 1017 three-agent serve
**−4.94%** for `6526a6919` → `f691af6e1` (AB 0.9487, BA 0.9526; clean
thermal-0 BA pair −4.74%). GLM gained 4.6% and Flash-Next was flat, so the
prime suspects were the V4.1 layer-interleave/CED refactor (`95b883202`,
`fe12a168a`, `5c89cb6ee`, `4acd6d87a`, `6fff1290c`) and the shared serve
changes in the same range.

## Native decode is flat

`v41_regression_probe` (identical source in both arms, untracked measurement
harness) loads the four-shard IQ2_XS artifact plus its Engram companion once,
then measures 128 production rounds of B=3 `step_batch` and B=1 `Forward::step`
at 32K and 48K. Solo causal waves are left at their production default, and the
measurement block enters `GpuThreadQos` exactly as the serve engine does.
Two runs per pair, release profile, both arms built from the same source.

Release + QoS, baseline `6526a6919` vs main `540c0a2ab` (current/baseline
`ms_round` ratio; lower is faster):

| pair | 32K B3 | 32K B1 | 48K B3 | 48K B1 |
| --- | ---: | ---: | ---: | ---: |
| AB | 0.966 | 0.990 | 0.984 | 0.986 |
| BA | 1.003 | 1.016 | 0.990 | 0.990 |
| AB2 | 1.035 | 1.010 | 0.989 | 1.004 |
| BA2 | 1.003 | 1.001 | — | — |
| **geomean (n=14)** | **1.002** | **1.004** | **0.988** | **0.993** |

All 14 paired ratios have geomean **0.9975** and range 0.966-1.035: no decode
regression. An `iter`/perf-ablations probe without the QoS guard gave 32K B3
1.009 and 32K B1 0.995 in its one clean pair; its 48K baseline run was
load-affected and is not used.

The midpoint of the range, `a3736629a` (57 of 113 commits), measured with the
same release+QoS probe against baseline: **1.001 / 0.998 / 1.003 / 1.003** for
32K B3 / 32K B1 / 48K B3 / 48K B1. There is no step at the midpoint, at main,
or at the measured current.

### Bisect table

| point | probe | 32K B3 | 32K B1 | 48K B3 | 48K B1 | verdict |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `6526a6919` baseline | release+QoS | 1.000 | 1.000 | 1.000 | 1.000 | reference |
| `a3736629a` midpoint | release+QoS | 1.001 | 0.998 | 1.003 | 1.003 | flat |
| `540c0a2ab` main | release+QoS | 1.002 | 1.004 | 0.988 | 0.993 | flat |
| `f691af6e1` measured current | serve greedy | — | — | — | — | flat (below) |

The layer-slice `MS=0` path leaves no trace in the phases record: every V4.1
arm reports `layer_rounds=0` and `prefill_mode=layer_major_batched+final_logits_copy`.
With `between=None`, `prefill_layer_slice` calls the incumbent
`engine.prefill`/`engine.cold_prefill`; `run_chunk` receives `interleave=None`;
and no decode file in the range changes outside `perf-ablations`. The flat
native decode agrees with the source.

## Why the serve pair moved

The standing arms are not cache- or time-symmetric.

| packet | arm | tok/s | ms/transition | thermal | Engram decode reads |
| --- | --- | ---: | ---: | ---: | ---: |
| AB | baseline | 94.500 | 27.744 | 0 | 2676.8 MB |
| AB | current | 89.650 | 28.955 | 1 | 2558.8 MB |
| BA | baseline | 98.025 | 26.832 | 0 | 496.3 MB |
| BA | current | 93.375 | 27.914 | 0 | 2989.0 MB |

- In the clean BA pair the **current server ran first** and read 2.99 GB from
  the IQ4_NL Engram companion during decode; baseline ran second and read only
  0.50 GB. The second arm inherited the first arm's page cache.
- In AB, baseline was at thermal 0 for the whole arm and current at thermal 1
  for the whole arm (the harness flagged it), so the same-binary comparison is
  also unbalanced.
- The record's own same-binary drift is 3.7% for baseline (AB→BA) and 4.2% for
  current (AB→BA), i.e. as large as the reported gap.

### Agent sampling is not the same work

Two self-warmed agent-sampling arms (both thermal 0, each measured after an
identical warm arm of its own binary) gave baseline **91.175** tok/s and
current **88.725** tok/s. But the arms generated 3647 vs 3549 window tokens and
their per-turn response hashes differ: real agent sampling with concurrent
timing does not execute the same token stream, so the same 40 s window does not
price the same work. Current also read 6.8% more Engram bytes because the
streams diverged.

### Greedy control removes the stream difference

Same protocol, `--sampling greedy`, so both binaries execute identical token
streams (2447 transitions per arm, same prompts and token counts), all arms
thermal 0:

| arm | order | tok/s | ms/transition | Engram decode reads |
| --- | --- | ---: | ---: | ---: |
| baseline | AB first | 90.900 | 28.724 | 2417.6 MB |
| current | AB second | 91.175 | 28.269 | 317.8 MB |
| current | BA first | 96.900 | 27.181 | 0.0 MB |
| baseline | BA second | 95.150 | 27.594 | 2405.1 MB |

Within-packet current/baseline is **1.003** (AB) and **1.018** (BA): V4.1 is
not slower. The same baseline binary moved **+4.7%** between packets with
essentially identical disk reads (2417.6 → 2405.1 MB), and the same current
binary moved +6.3% while its Engram reads fell from 318 MB to zero. Both are
host/page-cache/time effects, not decode changes.

## Decision

- **No product fix or revert.** The reported V4.1 gap is not reproducible in
  native decode, in a same-token serve control, or at the commit-range
  midpoint. The layer-slice default (`QWEN_SERVE_V41_LAYER_SLICE_MS=0`) is
  inert and costs nothing measurable.
- The standing protocol needs a cache/time control before it is used to
  attribute a few-percent V4.1 change: log `process_disk_read_bytes` and
  window-token/response hashes per arm, and add a deterministic greedy control
  or self-warmed arms. A 40 s agent-sampling window on its own has ~5% drift
  and several-fold Engram-read variance.
- No roadmap edit. Raw data:
  `metagross:~/qwen-runs/2026-11-12-v41-regression/`. Aven note on `SRV-2K0Y`:
  `VW68R006QF6PNMRC`.
