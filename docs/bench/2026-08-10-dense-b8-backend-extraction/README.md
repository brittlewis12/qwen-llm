# Dense B=8 backend extraction

## Question

Can the complete-model B=8 proof become a reusable engine backend without
losing its throughput or weakening session, cancellation, and command-failure
contracts?

## Change

`DenseBatch8Executor` moves the measured graph from `qwen-bench` into
`qwen-llm`. It owns reusable shared scratch while callers retain eight private
`MetalSession`s. The library path now:

- validates token, frontier, capacity, inventory, dtype, shape, alignment, and
  buffer-range contracts before every command;
- rejects mutable aliases within or across sessions;
- follows the singleton engine's active residual/post-norm organization;
- restores host frontiers after an encode error or pre-commit cancellation;
- checks both command-buffer status and error after every committed step;
- poisons the executor after a committed command failure;
- returns eight row-wise argmax IDs; and
- copies all eight completed logit rows into the corresponding sessions so
  existing observation and checkpoint semantics remain available.

The benchmark now calls this engine executor. Its timing therefore includes
the added session-logit copies and all public-boundary validation; no result is
inherited from the benchmark-local implementation.

The dirty release binary and checkout matched at acquisition under source ID
`git-source-sha256-v2:80af9f674f1401b7a9d185150da67accf8d5fb47e5b84d20774eeb180b984f53`.

## Results

### Qwen3.6 27B Q4_K_M

| implementation | serialized tok/s | static B=8 tok/s | speedup |
|---|---:|---:|---:|
| benchmark-local proof | 25.717 | 66.117 | 2.570x |
| extracted engine backend | 25.689 | **65.786** | **2.564x** |

The production-oriented path retains 99.5% of the proof's static throughput.
Median B=8 wall is 121.607 ms versus 311.418 ms serialized. Copying about
7.6 MiB of logits and validating the public session boundary costs only about
0.5% on the bandwidth-bound 27B row.

All 64 argmax decisions agree through eight causal positions. The full-logit,
residual, recurrent-state, convolution, and written-KV evidence is identical to
the prior proof: worst logits cosine is `0.999999479`; every persistent-state
cosine remains above `0.99999963`.

### Qwen3.5 0.8B Q4_K_M smoke

The copy-inclusive backend reaches `590.16 tok/s` versus `378.70 tok/s`
serialized, a `1.560x` gain. This is about 2.3% below the earlier benchmark-only
candidate, as expected for a small dispatch-shaped graph where fixed validation
and logit-copy work are a larger share.

## Decision

- Retain the extracted executor as the implementation authority for dense B=8;
  the benchmark-local whole-model graph is no longer the product seam.
- Preserve the narrow contract: dense Qwen, exactly eight sessions, equal
  contiguous frontiers, F16 KV, and caller-managed underfill.
- Move directly to an explicit file-JSONL cohort surface. Do not build a daemon,
  arrival queue, ragged scheduler, or dynamic width first.
- Keep prefix caching, sampled decoding, stdin batching, and partial cohorts out
  of the first product slice. Each is independently valuable but not required
  to bank the measured full-cohort gain.
- Add calculated admission and one representative long-context continuation
  before enabling the mode automatically.

## Artifacts

- `build-info.json`
- `qwen35-0p8b-smoke.txt`
- `qwen36-27b.txt`
