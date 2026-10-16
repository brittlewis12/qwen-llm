# GLM Maya: Upstream One-Draft MTP Screen

Status: prepared, not measured. Preparation began October 15; the review and
fixture were completed October 16. No performance or correctness result is
claimed. Native qwen inference is unchanged.

## Question And Scope

Does upstream GLM NextN pay for its draft initialization, verification, and
recovery on one real-source 32K request? Compare ordinary generation (A) against
one-token draft-mtp (B), using fresh servers in A1/B1/B2/A2 order. Each request has
32,768 frozen prompt tokens and a 128-token greedy output budget. Short EOS is
retained as an invalid fixed-work comparison, not suppressed.

This is an upstream feasibility screen, not a native backend implementation,
sampled-quality evaluation, or a broad MTP promotion. A positive result would
motivate a native two-row verifier/recovery cost screen, not automatic integration.

## Identity

- qwen source revision: `28e45c3a`.
- Inspected llama.cpp source and server: `844b765d4`, b11678, 0.7.0-dev.
  This is the local `glm5-next-lora-wo` branch, not an unmodified release tag.
  Both upstream MTP and sparse gather removal are ancestors. No LoRA is supplied.
- Artifact: local three-shard Maya-M IQ2_S. `fixture-v2/artifact.json` records
  full-header hashes, sizes, mtimes, and vocabulary size. Weight payloads were
  not rehashed. The loader must use shard one and the identical sibling shards.
- `fixture-v2/prompt-manifest.json` pins four public source files and their
  revision/hashes. Prefix, source body, and suffix are tokenized separately;
  the body's token prefix is cut to preserve the question and chat suffix.
- Prompt SHA256: `08828a9b69ee06da2b4f18f2ca9577048d530f34d2efa8e06768ce6888d6b25c`.
- Vocabulary-only `llama-tokenize` prepares the fixture; no model weights or
  inference graph were loaded during preparation.

The source-review question is intended to elicit a nontrivial continuation.
The 128-token cap can truncate reasoning; this packet does not certify task quality.

## Memory Admission

Independent source/header review estimated these persistent allocations at
33,024 context capacity (GiB):

| Component | Ordinary | One draft |
| --- | ---: | ---: |
| Trunk payload | 104.6822 | 104.6822 |
| NextN payload | 0 | 3.3457 |
| Target MLA/indexer cache | 0.6063 | 0.6063 |
| Target recurrent state, including rollback rows | 0.1422 | 0.2843 |
| Draft MLA/indexer cache | 0 | 0.0551 |
| Known subtotal | 105.4306 | 108.9735 |

Embedded NextN shares the model object; it does not reload a second trunk.
These subtotals exclude graph/compute/output buffers, padding, host metadata,
and transient peaks. The draft context retains the full context capacity even
with one proposed token. The 118 GiB configured wired limit is not available
physical memory or an admission certificate.

Initial admission was deferred: about 35-37 GiB of anonymous pages plus wired
and compressed memory were in use, including an active simulator/UI-test
workflow and background media analysis. The user offered to free memory.
Do not run until live admission is rechecked. Do not kill other owners' work,
purge caches, change wired limits, or substitute a smaller artifact silently.

## Execution

From the worktree root, after live memory/ownership admission:

```sh
scripts/bench/with-metal-lease uv run --offline --no-project python -B \
  docs/bench/2026-10-15-glm-mtp-upstream/run.py \
  --server /Users/tito/code/llama.cpp/build/bin/llama-server \
  --model /Volumes/wdblack/weights-archive/glm-5.3-flash-maya-m/Maya-M/GLM-5.3-Flash-Maya-M-IQ2_S-00001-of-00003.gguf \
  --prompt docs/bench/2026-10-15-glm-mtp-upstream/fixture-v2/prompt.json \
  --artifact-manifest docs/bench/2026-10-15-glm-mtp-upstream/fixture-v2/artifact.json \
  --out docs/bench/2026-10-15-glm-mtp-upstream/abba
```

The output directory must not exist. The external lease covers all child
servers. Only owned child processes are terminated. No deployment is performed:
each temporary endpoint binds localhost, disables the web UI, and uses an
ownership nonce before accepting benchmark traffic.

Both arms use full GPU offload, fit off, one slot, 512 batch/ubatch, explicit
flash attention and F16 target/draft KV. Context shifting, prompt caching,
RAM cache, and optional prompt checkpoints are disabled. Mandatory recurrent
rollback storage remains. Native startup warmup is charged to startup.

`request_seconds` includes uncached target prefill, draft prompt catch-up, and
generation. Startup is reported separately; this nonstreaming client does not
measure TTFT. Upstream generation throughput uses the 127 tokens following the
first prompt-logit token when 128 outputs are returned.

The harness validates zero prompt-cache hits, 32,768 processed prompt tokens,
128 generated IDs, positive finite timings, and actual drafting in B. A must
not draft. Executable/shared-library hashes and shard identities must stay
stable across arms. The harness refuses inherited LLAMA_ARG_, GGML_, MTL_, and
DYLD_ overrides rather than silently measuring them.

After each measured request, a separately timed three-token continuation asks
the same live slot to reuse the original prompt plus generated tokens. Compare
continuations only when the original streams agree and cache counters prove
prefix reuse. This is a terminal-alignment probe, not a general state oracle.

## Validation And Limits

```sh
uv run --offline --no-project python -B \
  docs/bench/2026-10-15-glm-mtp-upstream/test_run.py
```

Twelve CPU-only harness tests pass. The frozen prompt has exactly 32,768 valid
IDs. Initial fixture preparation needed two repairs: later GGUF shards omit
architecture metadata, and this tokenizer parses specials by default without
accepting an explicit --parse-special flag. Neither attempt loaded weights.

RSS is sampled, not an exact allocation peak. Record memory pressure and swap
changes alongside timings. Output divergence or invalid continuation must stay
visible; high draft acceptance alone cannot establish a useful or correct win.
