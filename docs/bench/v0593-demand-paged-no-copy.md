# v0.593 Demand-Paged One-Shot GGUF Views

Status: preregistered before timed work. This tracked file and its tracked runner
are part of the frozen source commit.

## Intent

Measure the no-copy product design v0.591 never tested: retain the exact 27B
read-only GGUF mapping, expose direct Metal views, and perform no CPU prefault.
Any GPU demand mapping, first-prefill work, or decode penalty remains inside the
external spawn-to-first-byte and spawn-to-exit endpoints.

This is a warm-filesystem-cache, fresh-process, one-shot policy experiment. It
does not claim storage-cold latency, loaded serving, or unlimited-generation
parity. Copied storage remains the persistent-workload baseline.

## Frozen Contract

- Model and prompt identities match v0.591 exactly.
- Effective prompt: 419 tokens, packed in one chunk, max context 1024.
- Both arms force native Q4_K token embedding residency.
- A: `QWEN_GGUF_NO_COPY=0`; no prefault override.
- B: `QWEN_GGUF_NO_COPY=1` and `QWEN_GGUF_NO_COPY_PREFAULT=0`.
- Output lengths: `1, 256, 32, 128`, in that order.
- Within every length: two pairs in order `BA, AB`.
- Cooldown after each complete-file cache read: at least 30 seconds.

Before every process, require AC power, no thermal/performance warning, and a
`memory_pressure -Q` availability reading of at least 50%. Repeat those checks
after process exit. The full file read is outside the timed endpoint and defines
the cache-warm condition; it is not claimed as free user-facing work.
Child-attributed hard faults and block input must be zero. This does not claim to
observe every device-initiated I/O counter, so the result remains explicitly
cache-conditioned.

The runner hashes source, binaries, model, prompt, this preregistration, and
itself before creating a fresh packet directory. It requires both tracked packet
files, a normalized environment, current clean HEAD, and matching clean build
identity. Any interruption is terminal for the packet. Do not delete individual
artifacts or resume it; archive the complete directory and preregister a new
packet attempt.

## Correctness And Accounting

Before timing:

- copied versus no-prefault retained storage is bit-exact for full packed-prefill
  logits, KV/GDN state, one forced transition, and continuation state;
- 195 active library tests pass, 124 are ignored, check and release build pass;
- A reports 851 copied descriptors and B reports 850 retained views plus the one
  20,480-byte EOF fallback;
- B reports prefault disabled, zero prefault pages, and zero prefault wall;
- product outputs match byte-for-byte across all four canonical processes at
  every requested output length.

Every timed row must generate the exact requested count and stop at the token
limit. An early stop invalidates that output-length fixture rather than silently
changing the slope. The exact timing-row and nested checkpoint schemas,
model/runtime identity, prompt, special-token policy, cache policy, decode
policy, terminal semantics, context, chunking, allocation checkpoints, and
transition count are fail-closed.

A row invalidated by post-run host state, child-attributed hard faults, or block
input is retained under an attempt-specific name. A within-pair output mismatch
is handled the same way. Its complete pair is rejected and rerun with a new
attempt number, up to three attempts. Pair decisions retain exact rejection
reasons. Command, parser, ledger, timing-schema, or cross-pair output failures
terminate the packet. No individual arm is dropped. Processes never overlap.

## Sequence And Gates

The tracked runner alone controls sequence. It runs output-1 `BA,AB`, evaluates
the gate, and refuses longer lengths on failure. On passage it runs output lengths
256, 32, and 128 in that fixed order.

Stop the cold-latency lane after output-1 unless both candidate rows win first-byte
and exit wall, with paired-median first-byte movement at least `1.20x` and 750 ms
absolute saving. A memory-only claim may remain.

For each subsequent length require:

- B wins first-byte and exit wall in both pairs;
- paired-median first-byte speedup is at least `1.20x` and saving at least 750 ms;
- paired-median exit speedup is at least `1.10x` and saving at least 500 ms;
- exact output, ledger, host, hard-fault, and block-input gates remain green.

Report model-load, prefill, transition count/wall/TPS, RSS, footprint, page
reclaims, and Metal allocation without converting any secondary field into the
primary result. Length-dependent first-byte savings differing from output-1 by
more than 200 ms produce a non-authorizing `needs_review` result requiring
explanation.

Fit `exit_wall = alpha + beta * (N-1)` separately by arm from the four
paired-median rows. Report measured rows before the fit. The fitted crossover is
sizing only and cannot extend policy beyond tested output lengths.

If all rows clear, they authorize only an explicit caller-owned one-shot request
for this exact content fingerprint, warm-cache precondition, and tested
prompt/context/prefill regime, through a cap no higher than 128 outputs. Product
integration must add an engine storage request and caller-owned lifecycle
selection. The current runtime gate is a structural layout allowlist, not
content identity, so it cannot satisfy that contract without content
verification. Persistent JSONL, server, unknown lifecycle, storage-cold use,
untested prompt regimes, and unbounded generation remain copied. Explicit force
outside the measured envelope carries no performance claim.
