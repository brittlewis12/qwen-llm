# v0.132 Dense Fused-SwiGLU Same-Process A/B

Status: dirty-code harness after `v0.131`, using a runtime override for
`QWEN_PREFILL_DENSE_FFN_FUSED_SWIGLU_Q4` inside one already-loaded process.
All runs used `QWEN_PREFILL_ATTN_MATRIX_G6=1,QWEN_PREFILL_ATTN_MATRIX_G8=1`,
AC power, and clean `pmset` / memory-pressure probes.

Artifacts:

- `pp4096.tsv`, `pp4096.log`
- `pp8192.tsv`, `pp8192.log`
- `pp16384-a.tsv`, `pp16384-a.log`
- `pp16384-b.tsv`, `pp16384-b.log`
- `phase-pp4096-base.tsv`, `phase-pp4096-fused.tsv`, `phase-pp4096-base-b.tsv`

## Same-process A/B

`qwen-bench pp-ffn-ab` warms base and fused once, then alternates order by pair:
pair 0 runs `base -> fused`; pair 1 runs `fused -> base`.

| Shape | Base rows | Fused rows | Read |
| --- | ---: | ---: | --- |
| `pp4096` | `179.14`, `187.50` | `177.88`, `185.19` | fused loses both orderings |
| `pp8192` | `185.70`, `182.90` | `180.65`, `177.04` | fused loses both orderings |
| `pp16384-a` | `178.25`, `183.84` | `195.67`, `194.18` | apparent large win, suspicious |
| `pp16384-b` | `174.82`, `170.10` | `174.84`, `170.39` | repeat is flat |

## Phase Trace Read

The preceding `pp4096` phase traces ran `base -> fused -> base` with layer-phase
tracing enabled. The first base trace was faster across nearly every phase, while
the fused and second-base traces moved together. That pattern does not prove a
local fused-FFN win; it proves the trace lane can expose whole-run drift.

Selected sums:

| Phase | Base A ms | Fused ms | Base B ms | Read |
| --- | ---: | ---: | ---: | --- |
| `gdn.ffn` | `9374.80` | `10005.68` | `10205.28` | fused resembles late baseline |
| `attn.ffn` | `3136.42` | `3366.42` | `3433.05` | fused resembles late baseline |
| `gdn_front` | `3625.07` | `3901.84` | `3955.86` | unrelated phase drift is comparable |
| `attn` | `1849.75` | `1976.90` | `2011.37` | whole-run drift, not FFN-local |

## Read

- The same-process harness is useful and should stay: it removes model-load drift
  and gives a better candidate-local gate than separate-process sweeps.
- The dense fused-Q4 SwiGLU branch still should not be defaulted. Medium/long
  same-process rows lose at `pp4096` and `pp8192`, and the first `pp16384` win did
  not reproduce.
- The surprising `pp16384-a` row is worth remembering as a possible long-context
  interaction, but it is not evidence by itself. A future branch needs a repeatable
  phase-local mechanism, not one lucky total row.
