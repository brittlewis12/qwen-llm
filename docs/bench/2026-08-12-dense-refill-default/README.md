# Dense B8 Short Serial-Tail Rescue

Date: 2026-08-12

Status: product GO for measured short-prompt serial-tail rescue; broader refill
remains explicit.

## Question

When should the bounded dense B8 refill mechanism become automatic inside an
explicit `--batch-size 8` request?

The mechanism can combine two static waves into one eight-lane arena and replace
finished lanes once. Its original 10% decode-step gate does not charge synchronous
replacement prefill or setup. The qualification therefore separates three policy
questions:

1. Does refill help when it absorbs requests the static planner would run serially?
2. Does that transfer from 0.8B to the flagship 27B dense model?
3. Which prompt and schedule boundaries remain profitable enough to default?

## Commands

All fixed-width arms use the same command, changing only the refill environment
and fixture. `MODEL` is one of the two paths in `validation.json`.

```bash
/usr/bin/time -l env QWEN_DENSE_BATCH8_REFILL=0 \
  target/release/qwen \
  --model MODEL \
  --requests-jsonl FIXTURE \
  --batch-size 8 \
  --temp 0 \
  --prefill-chunk 256
```

Set `QWEN_DENSE_BATCH8_REFILL=1` for the forced candidate. B2 controls replace
`--batch-size 8` with `--concurrency 2` and disable B2 prefix/file-root fanout.
Runs are serial and use warm model pages.

## Serial-Tail Rescue

The primary fixture has 32 requests at one 11-token frontier and generation
limits ranging from 1 to 40, repeated twice. Prefix reuse is below its minimum.
Static B8 admits only 16 requests; two depth cohorts fail its transition
utilization gate and fall back to serial. Refill absorbs all 32 into two bounded
arenas.

| Model | B2 | Static B8 | Refill B8 | Refill vs B2 | Refill vs static |
|---|---:|---:|---:|---:|---:|
| 0.8B Q8 | `2.070 s` | `2.140 s` | `1.685 s` | `1.228x` | `1.270x` |
| 27B Q4 | `23.79 s` | `19.73 s` | `16.04 s` | `1.483x` | `1.230x` |

The 0.8B rows are medians of `2.08/2.06 s` and `1.69/1.68 s`; the static arm
is one `2.14 s` sample. The 27B cell is one serial B2/static/refill bracket.
All outputs are byte-identical within each model. No process reports swaps, and
RSS is flat across each comparison.

Both refill runs execute 7 and 47 physical B8 steps. The second arena performs
346 productive and 30 padding transitions (`0.9202` utilization). Neither model
falls back for memory.

## Boundary Falsifiers

The exact 10% decode-step threshold is not strong default authority. A fully
batched 16-request fixture needs 20 static steps and 18 refill steps. On 0.8B,
static and forced refill take `0.90 -> 0.87 s` (`1.034x`). The automatic policy
therefore leaves work with no static serial fallback unchanged, even when an
explicit force would pass the structural gate.

Prompt cost also matters. With 1,029 equal-token, non-prefix prompts, forced
refill moves 0.8B static B8 `6.44 -> 5.99 s` (`1.075x`), below the standing
`1.10x` product gate. At exactly 64 prompt tokens, refill remains strong on both
anchors:

| Model | Static B8 | Refill B8 | Speedup |
|---|---:|---:|---:|
| 0.8B Q8 | `2.23 s` | `1.81 s` | `1.232x` |
| 27B Q4 | `23.69 s` | `20.17 s` | `1.174x` |

These cells freeze 64 tokens as the conservative automatic boundary. They do not
claim that token 65 is a measured crossover; they keep unmeasured longer prompts
on the prior static path.

## Decision

Use a tri-state refill policy for explicit dense B8:

- absent environment: `short_serial_fallback_rescue`; refill only a two-wave
  arena that absorbs at least one static serial fallback, has every prompt at or
  below 64 tokens, and clears the existing utilization and 10% decode-step gates;
- `QWEN_DENSE_BATCH8_REFILL=1`: force the broader experimental planner, including
  fully batched, longer-prompt, and explicitly ragged workloads;
- `QWEN_DENSE_BATCH8_REFILL=0`: strict rollback to static planning.

Automatic execution and Qwen MoE remain closed. Ragged prompts remain default-off
and do not implicitly enable refill. Prefix-selected cohorts are still excluded,
memory denial restores the captured static plan, publication remains input
ordered, and the executor keeps exactly eight live sessions.

Dense planner telemetry schema 7 reports `refill_policy` and the optional
`refill_default_max_prompt_tokens`, making the automatic boundary observable.
A retained final-source absent-environment run reports
`short_serial_fallback_rescue`, cap 64, two realized arenas, zero memory fallback,
and byte-identical output in `1.75 s`. Its stdout and stderr are retained beside
the fixtures. `validation.json` binds the candidate to the base commit plus exact
source hash and records the transient run-time binary hash. Later evidence edits
change the dirty-worktree build stamp, so the disposable current target binary is
not expected to retain that run-time hash.
Machine-readable process fields, fixture and raw-output hashes, exact policy,
and all four fixtures are retained in this directory.
