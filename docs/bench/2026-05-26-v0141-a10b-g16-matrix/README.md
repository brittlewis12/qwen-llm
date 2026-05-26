# v0.141 A10B G16 Matrix Attention

Status: default promotion after `v0.145`. The branch enables the matrix-attention
sidecar for the A10B group-16 shape by default when the existing packed G16
prompt path is active. `QWEN_PREFILL_ATTN_MATRIX_G16=0` rolls back to packed
attention; `QWEN_PREFILL_ATTN_MATRIX_G16=1` force-enables strict matrix behavior.

## Clean Gates

Clean rows used `build_dirty=0`, AC power, no recorded thermal/performance/CPU
power warnings, and `96%` free memory before/after.

| Shape | Base | G16 matrix | Read |
| --- | ---: | ---: | --- |
| `pp1024` | `411.86` | `430.67` | `+4.6%` |
| `pp16384` | `289.97` | `338.07` | `+16.6%` |

Clean current-build repeat packet after `v0.144`, with paired llama.cpp anchors:

| Shape | Qwen base rows | Qwen G16 rows | G16/base | llama.cpp default | G16/lcpp |
| --- | ---: | ---: | ---: | ---: | ---: |
| `pp512` | `367.61`, `368.77` | `382.72`, `383.36` | `+4.0%` | `427.24` | `0.90x` |
| `pp1024` | `413.23`, `414.49` | `435.08`, `428.02` | `+4.3%` | `422.34` | `1.02x` |
| `pp4096` | `380.16`, `373.43` | `408.67`, `402.29` | `+7.6%` | `374.29` | `1.08x` |
| `pp16384` | `289.36`, `293.28` | `340.41`, `340.14` | `+16.8%` | `331.58` | `1.03x` |

The paired llama.cpp default packet reports `flash_attn=false`. An explicit
`-fa 0` packet also reports `flash_attn=false` and lands at `427.54`, `413.45`,
`352.21`, and `320.18 t/s` for `pp512/1024/4096/16384`; use the default packet as
the same-session target because it is at least as fast on the longer rows.

## Warmed Dirty Spike Rows

These rows were pre-commit spikes and should be treated as EV evidence, not as a
promotion gate. They were useful because A10B cold no-warmup rows are heavily
confounded by first-touch/model-residency effects.

| Shape | Base | G16 matrix | Read |
| --- | ---: | ---: | --- |
| `pp512` | `361.09` | `379.78` | `+5.2%` |
| `pp1024` | `410.32` | `428.08` | `+4.3%` |
| `pp4096` | `376.47` | `404.63` | `+7.5%` |
| `pp16384` | `281.09` | `329.29` | `+17.1%` |

Chunk-4096 interaction spikes:

| Shape | Base chunk1024 | Base chunk4096 | G16 chunk1024 | G16 chunk4096 | Read |
| --- | ---: | ---: | ---: | ---: | --- |
| `pp4096` | `376.47` | `403.52` | `404.63` | `412.26` | chunk helps base; G16 still wins |
| `pp16384` | `281.09` | `293.72` | `329.29` | `329.68` | G16 dominates; chunk mostly flat |

## Phase / Coverage

The dirty `pp512` phase trace with G16 matrix active showed all 12 A10B attention
layers emitting `prefill-attn-matrix-g16-shape`. Matrix attention body phases were
about `14.3 ms` total (`KQ 5.46 ms`, softmax `3.11 ms`, `KQV 5.76 ms`), versus
the prior packed-attention body bucket around `73.21 ms` at the same prompt size.
The top A10B buckets remained routed MoE: `routed_swiglu 585.07 ms` and
`routed_down 317.90 ms` in that trace.

Clean current-commit trace after rebuilding `qwen-bench` from `b78a5d167` keeps
the same coverage and attribution shape:

- `12/12` A10B attention layers emit `prefill-attn-matrix-g16-shape` at layers
  `3,7,11,15,19,23,27,31,35,39,43,47`.
- Expected fast-path counts are present in the phase summary: `47` routed MoE
  layers, `12` attention layers, and `36` GDN layers.
- Matrix body phases total `14.53 ms` (`KQ 5.22 ms`, softmax `3.39 ms`,
  `KQV 5.92 ms`).
- Dominant residual buckets are routed MoE: `routed_swiglu 636.17 ms` and
  `routed_down 340.65 ms`; `routed_reduce` is tiny at `7.33 ms`.

Do not read the traced `tokens/s` as a throughput gate; phase tracing adds heavy
serialization/IO overhead. Use this row for coverage and attribution only.

## Correctness / Scratch

- Initial bare `QWEN_PREFILL_ATTN_MATRIX_G16=1` A10B smoke exposed a test-path
  scratch-allocation bug: `matrix scratch max_pos=4 < required last_pos=6`.
- A follow-up test-scratch fix sizes prefill correctness scratch by total/prefix
  position. Bare `QWEN_PREFILL_ATTN_MATRIX_G16=1` A10B smoke now passes without
  `QWEN_PREFILL_ATTN_MATRIX_MAX_POS`: final logits `0.999934`, GDN state
  `0.999809`, conv `0.999657`, KV K `0.999567`, and KV V `0.999414`.
- Read: numeric smoke correctness and test scratch are no longer the blocker;
  clean coverage, repeats, and paired llama.cpp anchors still are.

Default-promotion validation:

- `cargo fmt --check` and release `qwen-bench` rebuild passed after the policy
  change.
- Active matrix smoke passed with no `QWEN_PREFILL_ATTN_MATRIX_G16` env by lowering
  only the packed activation threshold for the smoke:
  `QWEN_PREFILL_ATTN_PACKED_G16_MIN_POS=1 QWEN_PREFILL_TRACE_ATTN_PHASES=1 cargo test --release -p qwen-llm prefill_tokens_matches_single_token_loop_122b_a10b_moe_smoke -- --nocapture`.
- The trace showed `prefill-attn-matrix-g16-shape` for all 12 A10B attention layers
  in both smoke chunks. Final logits were `0.999934`, GDN state `0.999810`, conv
  `0.999657`, KV K `0.999567`, and KV V `0.999413`.

## Current Read

- This is the first A10B-specific attention branch with strong medium and long
  evidence, and it changes the A10B long-context slope materially.
- The mechanism is not a chunk-size artifact: larger chunks help the packed base,
  but the G16 matrix path still carries the larger long-context win.
- The branch clears the qwen default performance gate: every repeated qwen shape is
  positive, same-session llama.cpp is beaten from `pp1024` through `pp16384`, and
  active matrix correctness is green for the G16 shape.
- `pp512` remains a lcpp gap despite the G16 win over qwen base, so the next
  scoreboard work after defaulting should return to routed MoE.
- After G16 matrix, the next A10B gap should be re-attributed; routed MoE is likely
  back on top, not packed attention.

## Artifacts

- `v0141-clean-a10b-pp1024-g16-matrix.json`
- `v0141-clean-a10b-pp16384-g16-matrix.json`
- `v0141-a10b-pp512-g16-matrix-warmup-spike.json`
- `v0141-a10b-pp1024-g16-matrix-warmup-spike.json`
- `v0141-a10b-pp4096-g16-matrix-warmup-spike.json`
- `v0141-a10b-pp16384-g16-matrix-warmup-spike.json`
- `v0141-a10b-pp4096-chunk4096-g16-matrix-warmup-spike.json`
- `v0141-a10b-pp16384-chunk4096-g16-matrix-warmup-spike.json`
- `v0141-a10b-pp512-g16-matrix-phase-summary.tsv`
- `v0143-clean-a10b-pp512-g16-matrix-phase.out`
- `v0143-clean-a10b-pp512-g16-matrix-phase-summary.tsv`
- `v0143-clean-a10b-pp512-g16-matrix-coverage.txt`
- `v0144-clean-a10b-pp512-g16-matrix-repeat.json`
- `v0144-clean-a10b-pp1024-g16-matrix-repeat.json`
- `v0144-clean-a10b-pp4096-g16-matrix-repeat.json`
- `v0144-clean-a10b-pp16384-g16-matrix-repeat.json`
- `v0144-lcpp-a10b-pp512-1024-4096-16384-default-fa.json`
- `v0144-lcpp-a10b-pp512-1024-4096-16384-fa0.json`
