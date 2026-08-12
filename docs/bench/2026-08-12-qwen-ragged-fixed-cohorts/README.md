# Qwen Ragged Fixed Cohorts

Date: 2026-08-12
Status: dense automatic slice GO; MoE automatic admission HOLD

## Question

Can the existing dense B=8 and Qwen MoE B=16 decode executors consume
requests at different prompt frontiers without changing each lane's greedy
result, and does that unlock useful aggregate throughput on heterogeneous
regular-file JSONL workloads?

## Implementation

`QWEN_FIXED_COHORT_RAGGED_PROMPTS=1` removes prompt-length equality from
fixed-cohort compatibility. Each lane retains its own logical position, KV/GDN
frontier, prompt length, and terminal accounting. Immutable-weight projection
rows remain batched at fixed width; attention and position-sensitive mixer work
receive the corresponding lane position.

The first product slice remains deliberately bounded:

- greedy decode only;
- fixed B=8 dense or B=16 Qwen MoE membership;
- no refill or continuous scheduler;
- one shared maximum capacity per cohort;
- requested-transition utilization must remain at least 3/4;
- a capacity-first ordering is selected only when it preserves cohort count,
  does not increase transition slots, and strictly reduces capacity slots;
- Metal-priced sessions, prefill scratch, executor scratch, reserve, and an
  optional CPU checkpoint are admitted before executor or sequence allocation;
- denied cohorts become input-ordered serial requests rather than aborting the
  file.

For `--execution-mode auto`, dense B=8 now admits ragged prompts only when the
file-wide short-prompt envelope and every proposed cohort pass a charged gate:

- every prompt is at most 256 tokens;
- every request asks for the same generation limit of at least 32 tokens;
- the fixed prefill chunk covers the longest prompt in one chunk;
- charged prompt tokens are at most twice productive decode transitions;
- ragged planning creates more full B=8 cohorts than incumbent planning;
- the entire file packs into admitted B=8 cohorts without serial remainders;
- the existing 3/4 transition-utilization and memory gates still pass.

`QWEN_FIXED_COHORT_RAGGED_PROMPTS=0` is the strict automatic rollback. Setting
it to one retains the broader explicit experiment for both dense B=8 and Qwen
MoE B=16. Prefix packing is reported as configured but ineffective for ragged
planning; within-cohort prefix fanout remains active.

## Exact Product Cells

All commands used the release binary at the reviewed worktree, warm model files,
`--temp 0`, fixed `--prefill-chunk 512`, and serialized GPU execution. Complete
candidate and serial JSONL files were byte-identical in every cell.

| Family / workload | Prompt tokens | Output | Serial wall | Ragged wall | Speedup |
|---|---:|---:|---:|---:|---:|
| Dense 0.8B Q8, heterogeneous short prompts | 10-137 | 8 x 64 | 2.82 s | 1.65 s | 1.709x |
| Dense 0.8B Q8, shared root + private suffixes | 1,984-2,222 | 8 x 32 | 2.64 s | 2.29 s | 1.153x |
| Qwen A3B Q4, heterogeneous short prompts | 10-139 | 16 x 64 | 14.21 s | 10.95 s | 1.298x |

The earlier implementation checkpoint, before final admission hardening, also
measured `2.23 -> 1.28 s` (`1.742x`) on dense short prompts and
`17.39 -> 11.08 s` (`1.569x`) on A3B. Those rows remain mechanism evidence;
the table above is authority for the final reviewed code.

The corrected dense shared-root cell selects a chunk-aligned 1,536-token
checkpoint from a 1,982-token exact LCP. It evaluates 4,240 private suffix
tokens, generates 256 tokens at 420.8 aggregate tok/s, and remains exact.

A3B long private suffixes remain the important negative constraint. An earlier
1,684-1,778-token, B=16, 24-token-output fixture was effectively flat
(`10.84 -> 10.78 s`, `1.006x`): 3,064 private suffix tokens erased the decode
benefit. This is why ragged admission remains explicit rather than default-on.


## Charged Automatic Decision

The product comparison charges the ragged fixed-width heuristic against the
automatic selector's current B=2 incumbent on the measured envelope; the token
charge is a conservative empirical policy, not an online B2 cost oracle. All
rows below use current-source executors, fixed `--prefill-chunk 512`, greedy
output, serialized process launches, and report zero process swaps.

| Family / workload | Output | B=2 execution | Ragged execution | Speedup |
|---|---:|---:|---:|---:|
| Dense 0.8B Q8, short | 8 x 64 | 1,748 ms | 1,246 ms | 1.403x |
| Dense 0.8B Q8, boundary | 8 x 32 | 943 ms | 690 ms | 1.367x |
| Dense 27B Q4, short | 8 x 64 | 20,570 ms | 10,333 ms | 1.991x |
| Dense 27B Q4, boundary | 8 x 32 | 11,451 ms | 6,417 ms | 1.784x |
| A3B Q4, short, counterbalanced | 16 x 64 | 9,060 ms | 8,343 ms | 1.086x |
| A3B Q4, boundary | 16 x 32 | 5,285 ms | 4,928 ms | 1.073x |
| A3B Q4, low-output boundary | 16 x 16 | 3,397 ms | 3,211 ms | 1.058x |

The initial A3B whole-process order (`14.88 -> 11.00 s`) overstated the lane:
the first load took 5.60 seconds and the second 2.41 seconds. Reversing order
left a real but sub-gate execution gain (`9.06 -> 8.34 s`, `1.086x`). That is
useful mechanism evidence, not authority to make MoE ragged automatic.

The dense automatic slice clears the 1.10x product bar at both measured model
scales and at the admitted 32-token boundary. The outer admitted fixture reaches
a 256-token longest prompt and a 1.976 prompt/transition charge: 0.8B execution
moves `1,369 -> 1,011 ms` (`1.354x`) and 27B moves
`17,475 -> 9,811 ms` (`1.781x`). A 16-token dense probe was only
`0.80 -> 0.68 s` whole wall and lies beyond the conservative 2x charge. The
policy therefore promotes dense B=8 only; MoE B=16 remains explicit.

The final exact automatic product packet on 0.8B measured `2.06 -> 1.53 s`
(`1.346x`) with byte-identical output SHA-256
`312f47f66e2242ee865648ff56fe9d2321213c3c60ebe3f0f60491d1aead8023`.
Selector schema 2 reports `automatic_dense_charged`; planner schema 8 reports
the plan decision (`admitted`, envelope rejection, charge rejection, or no
cohort gain). Realized planner telemetry separately reports runtime memory
rejection and serial fallback.


## Provenance

Final product artifacts were captured on macOS 15.7.9, host model `Mac16,5`
with 128 GiB unified memory. The source parent was
`a406ad3f55b688d696b6731024cfdc8fb588e110`; the measured dirty-worktree binary
SHA-256 was
`7eba4fbeb66e1684015a5b3441b7aa0876a9433d21a36ec8352f4d7a1012de5a`.
The `.err` files preserve complete argv-observable telemetry and `/usr/bin/time
-l` resource rows. Tested local model identities were:

| Asset | Bytes | mtime ns |
|---|---:|---:|
| `Qwen3.5-0.8B-Q8_0.gguf` | 811,843,840 | 1,774,269,806,260,313,011 |
| `Qwen3.6-27B-Q4_K_M.gguf` | 16,817,244,384 | 1,776,876,069,707,727,799 |
| `Qwen3.5-35B-A3B-Q4_K_M.gguf` | 22,016,023,168 | 1,774,274,392,389,261,747 |

These compatibility-oriented file identities are evidence provenance, not
portable content digests.

## Fixtures

The recovered JSONL workloads are now durable:

- `dense-short.jsonl`:
  `09d744dfa6e3fe38d80d9a2f9e898c42f821ca1db3b63a1b2ddf2a66216b4d10`
- `moe-short.jsonl`:
  `1ae2e560cbb305d5c3762283776492c67f0152825f78c05efe3dc189fcfd54f6`
- `dense-root.jsonl`:
  `23216cd6b75c8045c43a0a9227b0975afecb887663fb41364069906d01d9819f`
- `moe-long-root.jsonl`:
  `60319e76287f41bd45e80d4aaa010e5700e643447add2f8bf0cb13c5e7ef7732`
- `dense-short-16.jsonl`:
  `92e56b1535479728abe393ee59e4a5575d4dac5cd7d361039693a835182ada9f`
- `dense-short-32.jsonl`:
  `e39737926dad79b2e7cc59a4eb2e2ebbf08e0a58cdbb248c043674db76285172`
- `dense-policy-boundary.jsonl`:
  `26cb7fdb8a1d237d58948904f5f76cabf73eec90250068b2a25e9322754d815f`
- `moe-short-16.jsonl`:
  `e52485ee578792396f9b86468c86939110520f114755d6c62ecd83b908d5a961`
- `moe-short-32.jsonl`:
  `0116cb13da87227454d4c563f75add95ecad19d8724e771d3c973793b9ecfbf6`

## Validation

- Dense and MoE complete JSONL outputs and generated-token SHA-256 values match
  their serial controls exactly.
- Dense and MoE backend unit suites pass.
- 35 fixed-cohort planner/policy tests pass, including per-cohort charging,
  no-remainder fallback, memory rewriting, and longest-lane root alignment.
- `cargo clippy -p qwen-cli --bin qwen -- -D warnings` passes.
- `cargo clippy -p qwen-llm --lib -- -D warnings` passes.
- Initial adversarial review: HOLD; policy and documentation were tightened.
- Final adversarial re-review verdict: GO with no remaining actionable findings.

## Decision

Promote the charged short-prompt slice for automatic dense B=8 selection.
Retain incumbent equal-length/refill planning when the envelope, per-cohort
charge, cohort-gain, or utilization gate fails. Selector memory denial narrows
to B2; a later per-cohort denial falls back to input-ordered serial execution.
Keep automatic MoE ragged
selection on HOLD: B=16 is exact and useful, but the counterbalanced gain does
not clear the 1.10x product bar. The next architectural step remains refill or
continuous batching over the same per-request frontier contract, not another
model-specific executor.
