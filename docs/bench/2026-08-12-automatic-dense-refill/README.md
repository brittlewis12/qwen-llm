# Automatic Dense B8 Serial-Tail Rescue

Date: 2026-08-12

Status: product GO for bounded dense refill inside opt-in automatic execution.

## Question

Should `--execution-mode auto` use the already-qualified dense B8 serial-tail
rescue after selecting the dense fixed-cohort backend?

Before this change, automatic selection and explicit B8 execution planned the
same file differently. The selector saw only static cohorts, selected B8, then
execution forcibly disabled refill. On the 32-request short-prompt fixture, that
left 16 requests on serial fallback even though the default explicit B8 policy
had already qualified two bounded refill arenas.

## Contract

Automatic dense B8 now uses the same bounded policy for planning and execution:

- absent `QWEN_DENSE_BATCH8_REFILL` or explicit `=1` selects only
  `short_serial_fallback_rescue`;
- `=0` is strict static rollback;
- ragged automatic work remains refill-disabled;
- explicit `--batch-size 8` retains `=1` as the broader forced experiment;
- Qwen MoE never parses or enables the dense-only policy.

The selector prices the plan's maximum execution capacity, including refill
shared capacity, before choosing B8. Runtime arena admission remains a defensive
second check and restores the captured static plan if host conditions change.

## Command

Both candidate and rollback use one final-source binary. The rollback adds
`QWEN_DENSE_BATCH8_REFILL=0`; the candidate leaves the variable absent.

```bash
/usr/bin/time -l env \
  target/release/qwen \
  --model MODEL \
  --requests-jsonl ../2026-08-12-dense-refill-default/requests.jsonl \
  --execution-mode auto \
  --temp 0 \
  --prefill-chunk 256
```

## Result

| Model | Automatic rollback | Automatic rescue | Speedup |
|---|---:|---:|---:|
| Qwen3.5 0.8B Q8 | `2.16 s` | `1.76 s` | `1.227x` |
| Qwen3.6 27B Q4 | `19.76 s` | `15.89 s` | `1.244x` |

The candidate turns two static B8 cohorts plus 16 serial requests into two
bounded refill arenas covering all 32 requests. It keeps exactly eight live
sessions. Both models report zero process swaps and byte-identical JSONL between
candidate and rollback. Their output SHA-256 values remain the prior qualified
canaries:

- 0.8B: `22d3e4ab22276a4e0b7376686f0766b8b376eabb194d8e83bebb3416a8c50d59`
- 27B: `6fa252015e0ea7e8524db6756b6ff7115ed8f506b8de052d8d70260771afe648`

An explicit-auto `QWEN_DENSE_BATCH8_REFILL=1` falsifier uses the fully batched
16-request threshold fixture. It remains static at `0.90 s`, reports
`short_serial_fallback_rescue`, and forms no refill arena because there is no
serial fallback to rescue. This proves `=1` does not reopen forced refill under
automatic execution.

## Decision

Promote bounded rescue inside opt-in automatic dense execution. This is policy
composition over the existing exact executor, not a new scheduler or broader
continuous-batching claim. Keep ragged, longer-prompt, fully batched, and MoE
refill outside automatic scope.

The packet binds the exact measured source and records the transient runtime
binary hash. The binary is not retained: later evidence edits change the
full-worktree build stamp without changing candidate source. Same-binary
candidate/rollback outputs, compact raw telemetry, and machine-readable commands
remain retained. The fixture stays in its original directory to avoid duplication.
