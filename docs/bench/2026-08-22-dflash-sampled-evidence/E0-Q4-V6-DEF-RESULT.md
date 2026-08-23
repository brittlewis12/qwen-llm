# E0 Q4 v6 Alternate-Prompt Development Result

Status: one alternate one-token generated-history sentinel passed. This adds
minimal token/trajectory diversity and no meaningful code-task, broad-prompt,
held-out, verifier, serving, or product authority.

## Decision

- **GO:** multi-hidden capture remained bitwise lockstep with ordinary target
  execution for three generated transitions on a second one-token prompt and a
  different sampled trajectory.
- **NO-GO:** do not describe raw `def` plus four outputs as meaningful code
  behavior, prompt breadth, replication, model-wide E0, or product readiness.
- **Next gate:** either buy a meaningful short multi-token E0 prompt or pivot to
  the independent authoritative K0 contract; do not accumulate more easy
  one-token rows.

## Frozen Run

- Source/build/runtime commit:
  `01803b4d26f4fa9ff55bbd2ee36d8ec03bd2ae92`.
- Build/runtime source state:
  `git-source-sha256-v2:aba0924ccec93f21d081893b515c653b32f16854574029deedbbc88add5aaba7`.
- Build identity: clean `match`, both dirty flags false, no problems or
  overrides.
- Run ID: `dflash-e0-1787459433029236000-29690`.
- Fixture: `e0-q4-def-generated-v6`, role `development-sentinel`.
- Prompt: UTF-8 `def`, token ID `[727]`, canonical token i32le SHA-256
  `110a3fffafc92b01e8c967241e5bdf4651c93128160ae079e6b0a34a2a78f0f4`.
- Config: four requested/emitted tokens, sampler-v1, seed 0, temperature 0.7,
  top-k 200, top-p 1.0, min-p 0.05, warmup enabled,
  `serial_then_capture`.
- Generated IDs: `[369, 36951, 1393, 1590]`, i32le SHA-256
  `7e770afada199f9c903ddca1c0dd04a0bee11f13c3539de6bd7552b047cfc552`.
- Target aggregate identity:
  `fb9bcd41434a7fc1c2ed3b2d19a4ceb6cd8adb9cc55b9363ff1042acc2b3e048`.
- Q4 drafter aggregate identity:
  `51f00de02d311528ff23560395600804bb1d79af63b37e9f0f99c431f8d4ed62`.
- Executable SHA-256:
  `c3367307bc3d0f0a7062ec9326003ff78f2b4d5c32ef31f5b055347b2151680d`.
- Binding SHA-256:
  `ee469af5bec9e1086bd4cc8c9997efeef97dca22638f504f912731640f32f4e0`.
- Reducer SHA-256:
  `43b9fa9216fabcff8f573152939be01f0d1b7980000a89f500fbf83de304c059`.

Schema-v3 binding fixed the complete prompt/config and exact arm order. The
one-invocation command and decision rule were committed in
`E0-Q4-V6-DEF-RUN-PLAN.md` before acquisition.

## Reduction And Objective

The strict reduction reports one input, one run, one pass, four emissions, five
validated transitions, zero failures/invalid-pre-observation runs, and no
observed failure. The trace contains exactly one prompt, four sample frontiers,
three generated target transitions, one boundary, and one continuation.

Producer terminal accounting records three target transitions, four draws in
each sampler, token-limit stop without EOS, equal continuation, and final
consumed prefix/DFlash context length five. The full-raw sidecar exactly matches
the preregistered `1,885,208,576` bytes with authenticated coverage of six arm
pairs at prefix lengths `1, 2, 3, 4, 4, 5`.

## Artifacts

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| Trace | 25,095,125 | `34701cdbe54839aef01f3d32a422ff5b1ff2f1f35d271414aad273572ed584e4` |
| State sidecar | 1,885,208,576 | `e7900ae44651e589eb803667f2d37e495b8b8beec820b1a70e8e87ef2627c67c` |
| Reduction | 825 | `61642463de33059e250d7f0c552a5933e73f49916c510058ff20818e8d7dc6b6` |

Originals remain at their frozen `/private/tmp` paths. Verified mode-`0444`
copies are in the local, non-immutable quarantine
`/Users/tito/Documents/qwen-evidence/dflash-e0/2026-08-22-v6-def-pass/`.

## Scope Limit

V6 is one host, Q4 target/drafter, seed/config, serial-first order, one-token
prompt, and four-token output. `def` changes the initial embedding and produces a
different generated trajectory, but does not constitute semantic code coverage
or substantial prompt diversity. Reverse order on this trajectory, longer or
multi-token contexts, additional seeds, Q8/BF16, independent hidden residual
semantics, sparse-q/K0, packed verification, rollback, acceptance, economics,
performance, serving, held-out behavior, and product integration remain open.
