# KDA preparation history

History-only experiment: retain nine F32 convolution-history values per channel
across a packed chunk, then write final raw-input history once. Coefficient
loads, ordered FMAs, nonlinearities, reductions, barriers, and recurrence are
unchanged. No new runtime GPU allocation.

Six KDA tests pass with Metal validation, including bit equality for prepared
arrays and final history, one/two/three-row tails, and the independent CPU
chunk/chunk/decode oracle.

The model packet captures one actual KDA block after a 32-token prefix, replays
preparation alone from restored inputs at N2048/N512, and optionally measures
whole suffix prefill with three common teacher-forced continuation tokens.

## Results: Parked

Maya Fast Half on M4 Max 128 GiB. A retains device history; B uses registers.
One resident model, warm A/B followed by A1/B1/B2/A2. Prefix32 is built with A
in both whole-model arms; timed suffix rows are actual workspace rows.

| Rows | Leaf wall A/B ms | Whole retry A/B ms | Whole A/B ratio |
| --- | ---: | ---: | ---: |
| 2048 | 3.236 / 3.172 | 10242.99 / 10153.03 | 1.0089 |
| 512 | 0.821 / 0.737 | 2877.64 / 2872.37 | 1.0018 |

The leaf is too small to explain the whole N2048 mean difference. Multiplying
leaf savings by 34 KDA blocks gives only about 2-4 ms (wall/GPU differ), itself
a scale estimate rather than measured aggregate attribution. This does not
establish a useful whole-model gain or justify production promotion.

All five leaf arrays (prepared q/k/v, decay, final raw history) match bitwise
across all recorded arms. All recorded whole-model output hashes also match
at the join and three continuation steps, including across the two packets.

### Incomplete Attempt

`maya-abba.jsonl` completed both leaf screens and whole warm A/B. The N2048 A1
baseline prefill then took 84.146 seconds; the packet stopped before its output
record. The tool timed out at 1200 seconds and its automatic termination
reported EPERM. The identified process later disappeared without manual
termination, memory-setting changes or cache purging. No concrete cause was
established. Do not label this a candidate-kernel failure or a completed whole
comparison. The unresolved interval included file sync and continuation work.

The retry adds phase/continuation records, records warm durations, and uses
per-event flush with a final sync instead of syncing every event. It skips the
already completed capture/leaf work. `maya-whole-retry.jsonl` completes all 12
sessions in about 157 seconds with no recorded errors. First warmup retains
256 KiB; subsequent sessions have zero pre/post allocation growth. All post-drop
totals are 112402464768 bytes, so no accumulating session allocation is evident.

## Reproduction And Scope

The experiment is retained at Git commit `d7a46ba7`, not in the active build;
use that revision in a separate worktree. The whole-only retry is:

```sh
env -u MTL_DEBUG_LAYER QWEN_METAL_LEASE_WAIT=1 \
  GLM53_GGUF=/path/to/Maya.gguf GLM53_KDA_PREP_OUT=/path/to/new.jsonl \
  GLM53_KDA_PREP_WHOLE_ONLY=1 /path/to/release/test-binary \
  glm5_next_metal::packed::kda_prepare_ab::kda_prepare_ab_packet \
  --ignored --exact --nocapture --test-threads=1
```

Omit WHOLE_ONLY for leaf capture/replay; WHOLE=1 then additionally runs whole
ABBA. Capture GPU cost at width8192 is 256.28 MiB, individually priced; replay
reuses the session's five mutable buffers, restoring them outside timing.
No extra production allocation or modified state/snapshot contract.

The full experimental checkpoint passed 154 targeted tests and all-target
release checks. Archive the unsuccessful implementation instead of growing
the active shader/override surface. The established CPU KDA oracle remains.
