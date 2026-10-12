# Lower-state grouped gate/up

Base: `950a2db3`. Experimental M32/N32/K32 Half-operand gate/up retains
the incumbent per-output arithmetic and N16 path for small buckets, while
halving accumulator state and dynamic threadgroup memory (16 to 8 KiB).
It doubles output threadgroups and activation loads. The candidate is not
promoted: production remains unchanged.

Seven focused tests pass under Metal validation, including an independent
CPU Half-dot oracle, incumbent bit equality, tails, offsets and preflight.
The real-input packet captures block 4 normalized input and routes, measures
gate/up and unchanged down separately, and runs whole-model warm ABBA with
three common teacher-forced continuation tokens at actual N2048/N512 widths.

## Results

Maya, Fast Half, M4 Max 128 GiB, production lease, no Metal validation during
timing. `maya-abba.jsonl` retains warm A/B followed by A1/B1/B2/A2, one resident
model and fresh whole-model sessions. No environment intervention.

| Actual rows | Gate/up A/B ms | Unchanged down ms | Whole A/B ms | Whole A/B ratio |
| --- | ---: | ---: | ---: | ---: |
| 2048 | 67.946 / 69.998 | 42.833 | 10553.17 / 10684.90 | 0.9877 |
| 512 | 32.671 / 30.893 | 23.911 | 2948.45 / 2960.03 | 0.9961 |

The N2048 target loses at both the sampled leaf and whole-model level. The
small N512 leaf improvement does not convert into a whole-model gain. No UD
sweep or threshold tuning follows this result.

All recorded gate/up hashes match the captured production output; unchanged
down and whole join/three-step continuation hashes match across every arm.
All recorded values are finite and numerical differences/KL are zero. This
does not assert that every intermediate whole-model tensor was captured.

The normalized-input capture is 8 MiB at N512 / 32 MiB at N2048, additional to
the existing down diagnostic capture. It is not production memory. The
representative leaf times are not aggregate attribution for all routed blocks.

## Reproduction

The validated experimental source is retained at Git commit `d7a46ba7`.
It is removed from the active build after measurement. Use that revision in a
separate worktree to reproduce:

```sh
env -u MTL_DEBUG_LAYER QWEN_METAL_LEASE_WAIT=1 \
  GLM53_GGUF=/path/to/Maya.gguf GLM53_EXPERT_GATE_UP_OUT=/path/to/new.jsonl \
  /path/to/release/test-binary \
  glm5_next_metal::packed::expert_down::gate_up::expert_gate_up_packet \
  --ignored --exact --nocapture --test-threads=1
```

Candidate checkpoint validation: 62 grouped-MoE, 6 KDA and 86 GLM tests passed
with Metal validation; all-target release workspace check passed. Preserve
the result in history, not a permanently compiled unsuccessful shader.
