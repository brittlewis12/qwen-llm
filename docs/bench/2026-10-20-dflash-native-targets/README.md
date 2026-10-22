# Native Flash-Next and GLM-Flash DFlash2

Scope: Qwen3.8-Flash-Next and GLM-5.3-Flash, not full GLM-5.3 or Muse.
The native target owns verification and rollback; only drafting and
proposal-aware prefix acceptance are shared. The opt-in paths and public-startup
checks are complete; rollout measurements are pending.

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

## Rollout measurement

Pending. The October 22 continuation reconstructs the first 14 messages from
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

No native-family acceleration or gameplay-quality claim is made yet. Earlier
ordinary-Qwen results remain in `../2026-10-20-dflash-current-rollouts/README.md`.
