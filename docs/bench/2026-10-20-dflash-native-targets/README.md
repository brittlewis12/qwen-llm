# Native Flash-Next and GLM-Flash DFlash2

Scope: Qwen3.8-Flash-Next and GLM-5.3-Flash, not full GLM-5.3 or Muse.
The native target owns verification and rollback; only drafting and
proposal-aware prefix acceptance are shared. The opt-in paths and public-startup
checks are complete. Gameplay measurements below determine whether to use them.

## Execution contract

- Flash-Next captures the mean of four completed residual streams at zero-based
  layers 5, 15, 25, 35, 45. GLM-Flash uses 5, 14, 24, 33, 42.
- Draft normalization uses the artifact's epsilon: 1e-6 for the local Flash-Next
  head and 1e-5 for the local GLM-Flash head. Target normalization is unchanged.
- Verification uses each target's native packed arithmetic. Full retention keeps
  verified state; partial retention restores bounded recurrent/pending state and
  scalar-replays retained inputs. Correction and bonus tokens are not resampled.
- Emitted EOS and the final output-budget token remain unconsumed. Sampler state
  and output packets publish only after retained state and features are ready.
- Initial native drafter serving explicitly uses cold prefill, with zero matched
  tokens and no snapshot publication. Target snapshots alone do not contain the
  drafter's conditioning features. Ordinary serial/cache behavior is unchanged.
- Memory admission includes the resident head, request scratch, and the complete
  transport allowance once. Packed workspace narrowing preserves verification
  width and context. No memory bypass is used.

## Verification evidence

The October 20 checks below completed with Metal API validation and the
production GPU lease. Their logs and test binaries were stored under
`target/dflash-adapters-2026-10-20/`, which was cleared before the October 22
continuation; those ephemeral raw records are no longer available.

- The CLI library/binary and selected public CLI/info suites passed 1,489 tests,
  with 61 ignored. The first
  parallel run encountered a busy GPU lease and two trace-order failures; the
  trace tests passed independently and the complete serialized suite passed.
- Forty focused acceptance, capture, checkpoint, packet-boundary, and memory
  admission CPU checks passed before the final integration build.
- Flash-Next real-model checks passed: compact rollback and continuation;
  greedy/sampled packets; cancellation; late publication failures; and persistent
  GPU poison. Same-width all-row readouts match independently evaluated native
  learned heads bit-for-bit, preserving state and continuation.
- A fixed-width Flash-Next A/B/A probe changed only the last two of four inputs.
  Both earlier rows' final hyper states and logits remained bit-identical.
  This is a bounded fixture witness, not a general causality proof.
- Native width-one versus width-four Flash-Next execution differs at one tested
  boundary (cosine 0.997688, relative RMS 0.0684, same argmax). Its magnitude is
  not localized. Cross-width values are characterization, not a readout oracle;
  there is no claim of bit-identical serial generation or sampled traces.
- A small-buffer reproducer exposed a false allocation guard: 20,500 bytes of
  owned capture resources, a 65,536-byte price, and a 131,072-byte global device
  counter jump. Repeated identical banks could instead have zero global growth.
  The corrected guard validates each owned resource and rechecks live headroom;
  it does not inflate prices or ignore actual memory pressure.
- The first GLM attempt overlapped an unleased external llama.cpp evaluation and
  failed with GPU out-of-memory. That run establishes neither clean capacity nor
  correctness. The coordinated rerun passed capture/readout, prefill cancellation,
  rollback across ring phases and the sparse frontier, greedy/sampled seeded
  packets, fatal command failures, EOS boundaries, and postcommit publication
  failure. A failure witness initially tried to reset an intentionally poisoned
  session; it now drops that fixture before allocating the next independent
  case. Production failure semantics were not weakened.
- Both native resident smoke tests pass through actual drafter preparation and
  startup, repeated requests, cancellation, and subsequent reuse/recreation.
  They assert zero prefix reuse for drafter mode.

### October 22 revalidation

The resumed baseline includes the intervening split-QSA decode, HC injection
fusion, PLE page warming, and shared grouped-expert changes. Five focused
real-model tests passed again, serialized with Metal validation and the normal
lease: Flash-Next same-width readouts, fixed-width suffix-mutation causality,
compact rollback/continuation, greedy/sampled packets and cancellation; and GLM
greedy/sampled seeded packets and counters. The earlier cross-width Flash-Next
difference remains (relative RMS about 0.0684 at the same boundary); it is not
silently reclassified as numerical noise or a serial-equivalence guarantee.

Fresh raw records live in a private benchmark archive outside the disposable
build tree. The library tests used source `14377c46`; the release baseline
`7048f181` adds only documentation.

## Rollout measurement

The October 22 replay reconstructs the first 14 messages from
`~/code/llm/game/claude_flash_next_gsq.json`, the same saved source and message
count as the earlier native Qwen comparison; the old frozen wrapper was cleared
with `target/`, so a fresh input hash is recorded. It runs native
renderers, temperature 1, seed 42, a 256-token cap, and serial/spec/spec/serial
fresh-process ordering. Each family first receives a separate 16-token warmup
per arm, excluded from measurements: switching these large targets can otherwise
make the first measured arm pay cold disk reads. Per-process GPU wiring and
request preparation remain measured. Other filters use native release defaults.
It records the frozen binary hash, clean build commit, input hash, request stats,
full commands, output, diagnostics, and evidence of actual draft calls.

The release baseline is clean `7048f181` on an M4 Max with 128 GiB RAM. Metal
validation is enabled; these are exploratory timings, not production-performance
qualification. No competing GPU workload was intentionally launched. Top-k
is 20 for Flash-Next and 0 (disabled) for GLM; both use top-p 0.95 and
min-p 0. Request time excludes model loading; process wall includes startup,
loading and any lease wait. Both arms use native admission and prefill geometry.

[`results.json`](results.json) retains per-run timings, acceptance counters,
warmup exclusions, native geometry, and binary/input/output hashes without
gameplay text or machine-specific paths. The binary SHA-256 is
`314958d24ca29bba5146a4e6abaad1c6779ee6ba25b16c677fe01a3263f8326e`.

### Flash-Next result

The Q3_K_XL target with Q8_0 DFlash2 head is slower on this prefix:

| Mode | Prompt tokens | Output tokens | Decode tok/s | Request seconds | Process seconds |
|---|---:|---:|---:|---:|---:|
| Serial | 11,550 | 256 | 30.51 | 35.78 | 36.24 |
| DFlash2 | 11,550 | 256 | 8.81 | 56.62 | 57.19 |

Values are means of two observations. Speculative decode is **0.289x** serial;
request latency increases **58.3%**. Both speculative repetitions accept
153/714 proposed tokens (21.4%), with 102 draft calls, 98 restores and 226
replayed inputs; only four packets retain the full verified prefix. The dominant
cost is target work: about 20.48 s verification plus 7.40 s replay, versus
0.70 s drafting. This is real speculation, not a silent serial fallback.

All measured runs reach the cap. Each arm repeats its own output fingerprint,
but serial and speculative continuations differ. A capped continuation does not
establish completed-turn gameplay quality or EOS latency. Keep speculation
opt-in; these results do not support enabling it by default. Earlier
ordinary-Qwen results remain in `../2026-10-20-dflash-current-rollouts/README.md`.

### GLM-Flash result

The IQ3_XXS target with Q8_0 DFlash2 head is also slower:

| Mode | Prompt tokens | Output tokens | Decode tok/s | Request seconds | Process seconds |
|---|---:|---:|---:|---:|---:|
| Serial | 11,105 | 256 | 26.10 | 71.89 | 72.84 |
| DFlash2 | 11,105 | 256 | 5.36 | 111.98 | 113.66 |

Speculative decode is **0.205x** serial; request latency increases **55.8%**.
Each speculative repetition accepts 124/882 drafts (14.1%), makes 126 draft
calls and replays 250 retained inputs. Five tail packets are anchor-only;
there are no serial-verification fallbacks, cancellations or failures. Both
arms retain the same 512-row prefill workspace and 11,360-token capacity.
The initial serial warmup reads 117.5 GB during prefetch; every measured run
reports zero cold prefetch windows and zero bytes read. The slowdown is not
explained by one arm paying that initial cold read or receiving fewer prefill
rows.

All measured runs reach 256 tokens and repeat their arm's own output fingerprint.
As with Flash-Next, this is one sampled prefix, not a gameplay-quality result.
Keep native DFlash2 opt-in for both families. The initial GLM attempt failed
before inference because the runner resolved a split-shard symlink to a backing
filename without the required suffix; the corrected runner preserves supplied
paths, and the failed attempt contributes no timing sample.

### Reproduction

The durable runner is `scripts/bench/dflash_native_replay.py` (19 CPU tests).
It accepts explicit local model paths and refuses existing output directories:

```sh
uv run --script scripts/bench/dflash_native_replay.py \
  --binary "$FROZEN_QWEN" --expected-commit "$BUILD_COMMIT" \
  --source "$HOME/code/llm/game/claude_flash_next_gsq.json" \
  --output-dir "$NEW_PRIVATE_OUTPUT_DIR" \
  --family flash-next --model "$FLASH_TARGET" --drafter "$FLASH_DRAFTER"
```

For GLM use `--family glm-flash` and its target/head paths. Each invocation runs
both warmups and all four measured arms. The runner fixes Metal validation and
lease waiting, drops ambient tuning overrides, and checks successful v2 request
stats, clean build identity, native template, and nonzero speculative draft
calls. It preserves failed/interrupted manifests and waits for graceful shutdown
of its owned GPU process; it does not force-kill after a timeout. Input, binary,
stdout, stderr and stats have hashes. Target/head identity is path/size/mtime,
not a full-content hash. Gameplay content and generated output stay private.
