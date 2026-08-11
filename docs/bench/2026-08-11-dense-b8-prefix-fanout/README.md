# Dense B=8 shared-prefix fanout

## Question

Can a fixed dense-Qwen cohort prefill one exact common token prefix, fork its
causal state into seven sibling sessions, and prefill only private suffixes
without changing output or weakening memory admission?

The shipped B=8 path previously prefetched every complete prompt serially.
That retained decode's measured weight-reuse win but repeated the dominant
prompt computation eight times for system-prompt, agent, and document cohorts.

## Candidate

`LoadedModel::restore_prepared_checkpoint` restores an immutable in-process
`PreparedCheckpoint` directly into a fresh sequence. It preserves model-owner
provenance, validates the complete canonical token prefix, checks destination
capacity and snapshot geometry before mutation, and retains pending-token and
exact-logit semantics. It does not insert into the RAM prefix index or touch the
durable codec.

For each complete dense B=8 cohort, the CLI now:

1. computes the exact eight-way token LCP;
2. requires at least 256 common tokens;
3. aligns a partial shared prefix down to the fixed prefill chunk boundary so
   cold and fanout paths retain the same packed-prefill segmentation;
4. preserves a whole identical prompt even when its length is not chunk-aligned;
5. prices the CPU snapshot plus seven remaining Metal sessions before work;
6. prefills the shared prefix once, captures one snapshot, and restores it into
   seven fresh sessions;
7. prefills each private suffix serially to the common final frontier; and
8. enters the unchanged fixed-B=8 decode executor.

If snapshot-inclusive admission fails but the original eight-session plan fits,
the cohort falls back to ordinary serial prefill. The snapshot is dropped before
decode. `QWEN_DENSE_BATCH8_PREFIX_FANOUT=0` is the explicit rollback.

This is transient state fanout, not cache publication, copy-on-write KV, or
batched prefill. Every mutating lane still owns complete private causal state.

## Results

### Qwen3.6 27B Q4_K_M

Eight 1,029-token requests shared an exact 1,024-token prefix and had distinct
five-token suffixes. Each requested eight generated tokens.

| execution | cohort prefill | prefill + decode | aggregate decode |
|---|---:|---:|---:|
| B=8, fanout disabled | 36.344 s | 37.250 s | 47.490 tok/s |
| B=8, shared-prefix fanout | **7.553 s** | **8.453 s** | 47.808 tok/s |
| relative | **4.812x** | **4.407x** | 1.007x |

Adding each process's cache-warm model load gives a phase-sum improvement from
39.270 s to 10.381 s, or `3.783x`. Decode remains flat, as intended.

The candidate spent:

- 4,111.818 ms on the one shared-prefix prefill;
- 40.260 ms capturing a 224,006,272-byte snapshot;
- 75.862 ms restoring all seven sibling sessions; and
- 3,324.455 ms on the eight private five-token suffix prefills.

The small-suffix term is now the obvious residual: packed prefill has a large
fixed cost when invoked separately for five tokens. That follow-up is
independent of the fanout mechanism and does not diminish this promotion.

The snapshot increased incremental admission from 1,775,255,552 to
1,999,261,824 bytes; the evaluator separately retained its 2 GiB reserve and
admitted a 4,146,745,472-byte total requirement.

All candidate rows were byte-identical to both the rollback arm and a
cache-disabled serial run. The three files share SHA-256
`d0c7036fc49242f77d51608197beb64b0c20845fd68dc5fca32a96894d5c0955`.
This covers 43 emitted tokens, 35 useful transitions, 21 finished-lane padding
transitions, and multiple EOS boundaries.

### Qwen3.5 0.8B Q4_K_M

The same 1,024-token shared-prefix fixture reduced prefill from 1,056.512 to
591.681 ms (`1.786x`) and total prefill plus decode by `1.661x`. A separate
504-token identical-prompt fixture exercised the no-suffix/final-logit branch:
prefill fell from 523.068 to 115.936 ms (`4.512x`).

Both fixtures were byte-identical across fanout, rollback, and cache-disabled
serial execution. Their output hashes are retained in `validation.json`.

## Decision

- Promote transient shared-prefix fanout as default-on inside the already
  explicit dense-Qwen `--batch-size 8` mode.
- Keep the 256-token floor, chunk-aligned partial boundary, exact token match,
  calculated admission, and environment rollback.
- Preserve the public prefix-cache settings as unsupported in B=8 mode; this
  optimization neither consumes cache budget nor publishes an entry.
- Treat direct prepared-checkpoint restore as a Qwen runtime primitive usable by
  dense and MoE sessions. A MoE B=8 executor remains separately qualified.
- Do not project this snapshot ABI onto DeepSeek V4; its causal snapshot and
  endpoint-observation semantics remain family-specific.
- Next, route very short private suffixes through a measured low-overhead path,
  then add compatibility-based cohort planning and serialized underfill.

## Commands

```bash
target/release/qwen \
  --model /Users/tito/models/Qwen3.6-27B-Q4_K_M.gguf \
  --requests-jsonl target/tmp/dense-b8-fanout-aligned.jsonl \
  --batch-size 8

QWEN_DENSE_BATCH8_PREFIX_FANOUT=0 target/release/qwen \
  --model /Users/tito/models/Qwen3.6-27B-Q4_K_M.gguf \
  --requests-jsonl target/tmp/dense-b8-fanout-aligned.jsonl \
  --batch-size 8

target/release/qwen \
  --model /Users/tito/models/Qwen3.6-27B-Q4_K_M.gguf \
  --requests-jsonl target/tmp/dense-b8-fanout-aligned.jsonl \
  --prefix-cache-max-mib 0 \
  --cache-prefix-auto-min-tokens 0
```
