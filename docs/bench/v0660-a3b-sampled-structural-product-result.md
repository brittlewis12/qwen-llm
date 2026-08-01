# v0.660 A3B Sampled Structural Product Result

Status: **GO** with
`authority=["exact-frozen-fixture-policy-review"]`. Exact bounded top-k over
resident transition logits clears every frozen gate. The policy review retains
the hidden, default-off force path for the authenticated A3B profile. It does
not authorize default admission, another timing packet, or a broader sampling
claim.

## Scope

The packet charges one exact fresh-process request:

- `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` under the authenticated copied profile;
- 419 prompt tokens and 128 generated tokens;
- temperature `0.7`, top-k `200`, top-p `1.0`, min-p `0.05`, seed `42`;
- 128 sampler calls, 127 target transitions, and token-limit termination; and
- six fresh counterbalanced pairs in order `AB BA BA AB AB BA`.

Arm B differs only by `--sampled-structural`. Prompt logits remain owned. The
127 transition selections use exact bounded top-k over a synchronized Shared
F32 row without a full logits copy or full-vocabulary candidate vector.

## Identity And Validity

```text
commit: e34906e2a8c72a379629d73db9ea77d56e29dccd
tree: 39bd3b81ccc992d02854e88b380d33ace9222877
source: git-source-sha256-v2:fbacdd0ae3f158c3f1ef04478511dba30a908c8f28cb5e7da8aef364350997b1
build/runtime status: match; clean; no overrides
```

- The packet contains exactly 79 inventoried nonterminal files plus the sole
  terminal `decision.json`. Every recorded size and SHA-256 matches; no failure,
  temporary, foreign, retry, timeout, leak, or cleanup record exists.
- All 15 child attempts have one launch, spawn, and completion. Candidate B is
  exposed once immediately before its first spawn.
- The sampler suite passes 17 tests, the CLI suite passes 60, and the release
  A3B test passes exact row, sampled-token, active KV/GDN/conv state, and
  continuation comparisons. Its output authenticates the frozen prompt-token
  digest.
- All 14 host gates pass on AC power with no thermal, performance, or competitor
  warning, 96% available memory, and median CPU idle of `88.25-96.50%`.
- Swap occupancy is unchanged. Compressor stored and occupied pages decline by
  517 and 136; advisory pageout, compression, and swapout counters are flat.

## Correctness And Structural Evidence

All 12 children emit byte-identical stdout, the same 128-token digest, 128 draws,
127 transitions, stop reason, sampling configuration, and pending-terminal
semantics. A rows have the exact schema-10 key set; B rows add only the frozen
schema-12 structural object.

Every B child records:

```text
borrowed_transition_calls:          127
resident_head_wait_calls:           127
validated_shared_row_calls:         127
max_heap_len/capacity:              200 / 200
fallback_calls:                     0
full_candidate_vector_allocations:  0
transition_logits_copy_bytes:       0
extra_command_buffers:              0
gpu_sampling_dispatches:            0
```

This is evidence for the named host-side work substitution, not GPU sampling,
prompt-logit borrowing, lm-head work removal, or a command-schedule change.

## Frozen Reduction

| Pair | Order | Generation saving | Request saving | TTFT delta | Spawn saving |
| ---: | :---: | ---: | ---: | ---: | ---: |
| 1 | AB | `4.333709%` | `50.390333 ms` | `+3.823833 ms` | `83.304084 ms` |
| 2 | BA | `3.851522%` | `32.106083 ms` | `+15.610708 ms` | `11.067792 ms` |
| 3 | BA | `6.713454%` | `87.605750 ms` | `-3.129583 ms` | `81.188125 ms` |
| 4 | AB | `4.937845%` | `60.782459 ms` | `+0.240667 ms` | `37.188917 ms` |
| 5 | AB | `5.067953%` | `61.407209 ms` | `+0.565666 ms` | `31.339041 ms` |
| 6 | BA | `5.232855%` | `61.844625 ms` | `+3.358542 ms` | `55.878584 ms` |
| **Median** | - | **`5.002899%`** | **`61.094834 ms`** | **`+1.962104 ms`** | **`46.533751 ms`** |

Generation and request wall improve in `6/6` pairs. AB/BA generation medians
are `4.937845%` and `5.232855%`; request medians are `60.782459` and
`61.844625 ms`. Every frozen performance gate passes.

The headline generation gate clears narrowly: the median exceeds `5%` by only
`0.002899` percentage points, three individual pairs are below `5%`, and the AB
median is below `5%`. Pair 2 regresses TTFT by more than `10 ms`, while the
preregistered gate applies to the median and passes. The result therefore
establishes a real positive exact-fixture mechanism, not a robust per-run
`>=5%` claim or uniform TTFT nonregression.

## Policy Decision

Retain `--sampled-structural` as a hidden, default-off force path under its
existing exact A3B eligibility checks. Do not automatically select it, widen its
profile, or infer transfer to another model, quant, top-k, request shape, serving
mode, or stochastic contract. v0.660 consumes the sole v0.659 performance
packet; no repeat or breadth packet is authorized.

The optimization leaves the active queue. Generic certified lm-head screening
is the next independent decode work-removal oracle; direct-to-session durable
restore remains the next bounded process-cold continuation candidate.

## Artifacts

The complete 80-file packet is under
`target/profiles/v0660-a3b-sampled-structural-product-p1/`.
`decision.json` SHA-256 is
`293edd588337cd14119315703344ed4a292ddf6e7debf291fe54112dd96a8c47`.
Post-acquisition independent review: `cx` session
`019fbd7c-16da-7103-93ab-4a034d14cd76`.
