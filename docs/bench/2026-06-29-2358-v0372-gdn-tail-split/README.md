# v0.372 GDN Tail Split Attribution

Purpose: add a visible phase diagnostic for the GDN tail so GDN work is ranked by
subphase evidence instead of the aggregate `gdn tail` bucket.

New knob:

```text
QWEN_PHASE_GDN_TAIL_SPLIT=1
```

This affects only `qwen-bench phase`. It splits the GDN tail into:

- `gdn tail conv`
- `gdn tail l2`
- `gdn tail step`
- `gdn tail norm`

## Measurements

```text
QWEN_PHASE_MOE_FFN_SPLIT=deep \
QWEN_PHASE_GDN_PROJ_SPLIT=1 \
QWEN_PHASE_GDN_TAIL_SPLIT=1 \
  target/release/qwen-bench phase \
  -m /Users/tito/models/unsloth-Qwen3.5-122B-A10B-GGUF/UD-Q4_K_XL/Qwen3.5-122B-A10B-UD-Q4_K_XL.gguf \
  --ctx 8192 \
  > target/profiles/v0372-a10b-q4xl-phase-ctx8192-gdn-tail-split.out

QWEN_PHASE_MOE_FFN_SPLIT=deep \
QWEN_PHASE_GDN_PROJ_SPLIT=1 \
QWEN_PHASE_GDN_TAIL_SPLIT=1 \
  target/release/qwen-bench phase \
  -m /Users/tito/models/Qwen3.5-35B-A3B-Q4_K_M.gguf \
  --ctx 32768 \
  > target/profiles/v0372-a3b-q4-phase-ctx32768-gdn-tail-split.out
```

| Model row | conv | l2 | step | norm | Read |
| --- | ---: | ---: | ---: | ---: | --- |
| A10B `ctx8192` | `0.14 ms` | `0.17 ms` | `0.69 ms` | `0.17 ms` | step largest, still small |
| A3B `ctx32768` | `0.11 ms` | `0.15 ms` | `0.34 ms` | `0.14 ms` | no large tail villain |

## Decision

Keep the diagnostic; it is phase-only and clarifies the roadmap. Do not make GDN
tail the next implementation branch from this evidence. The recurrence step is the
largest tail subphase, but it is only `0.69 ms` on A10B `ctx8192` and `0.34 ms` on
A3B `ctx32768`, smaller than attention, routed gate/up, routed down, GDN
projection, and lm-head buckets.

Read: naive GDN-tail fusion or local row-count retunes are low EV. GDN work needs
a broader byte/dataflow change, and the next implementation branch should stay on
larger measured buckets unless a new counter signal appears.
