# DeepSeek V4.1 bring-up assessment

Status: source-verified architecture assessment, 2026-09-10. No V4.1 execution
support is claimed in qwen-llm. Preserve V4 Flash-0731 and Qwen Flash-Next contracts.
Community refresh: 2026-09-11, approximately 01:18 UTC (see below).
Latest update: DwarfStar Metal support landed on main on 2026-09-12.

## Decision

Treat V4.1 as a separate execution and cache profile, not a dimension update to
V4. Build a correctness-first full-forward path before adding optimized CED
prefill with exact dependency pruning or explicitly approximate replay. Reuse
kernels only after checking their numerical and storage contracts; do not relax
V4's frozen schema to accommodate early converted files.

The first implementation milestone is model-free configuration, tensor-role,
hashing, and cache-ownership validation. Full-checkpoint local generation is a
separate capacity decision: the 552B backbone alone exceeds a 128 GiB host at
ordinary three- or four-bit storage, even with all Engram tables kept on SSD.
Do not download or allocate the full model as part of the architecture spike.
This is a full-residency limit, not a streaming impossibility: the subsequent
DwarfStar demonstration makes exact fetch-on-miss expert streaming a concrete
candidate for the local deployment lane. The September 12 release supplies
public code and artifacts; local qwen-llm qualification is still pending.

This document owns architecture contracts and bring-up gates. Unimplemented
performance work belongs in `docs/PERF-ROADMAP.md`; attempted optimizations in
`docs/PERF-LOG.md`; actual benchmark artifacts in `docs/bench/`. This assessment
is not a performance result.

## Sources and reproducibility

Official repository: https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash

Pinned revision: `dba1be0a40aa45a94ad051997016db3960a90277`.
Resolve source files using `/resolve/<revision>/<path>`, not mutable `main`.

| Source | SHA-256 |
|---|---|
| `DeepSeek_V41_Tech_Report.pdf` | `ba68e2e40408125ae6d2f63a9a241b61c73910691c74ec1a2a7023c851eac08d` |
| `config.json` | `8be45ce0476004a3f529fd896115a4a2e800a129ad2d3ec05b16050f52e21879` |
| `inference/model.py` | `4e9ae23620edc8028ccc5d5fef552ab7fdc7dcd6f79608754fe9f67644056f65` |
| `inference/engram.py` | `11f35ecbead8150c35aa002b3d180ef290b05a25afe883a11884f94d476d3897` |
| `inference/kernel.py` | `1236c3507019ed176f5dba5e04bcea58867cf654818c6cf138ed4845398c2455` |
| `encoding/encoding.py` | `502bdaec8a3fd88ebc24c4721a7038fbe42f2063c664638127056107920035c1` |
| `model.safetensors.index.json` | `74b0686a3d2891980d5e303251b075a3bccae2c2ff650747db2620a649b98fa8` |

The index declares 96,085 tensors across 48 shards and 510,286,023,000 tensor
payload bytes. This is checkpoint metadata, not a verified download census.
Bounded HTTP range reads of shard 47 and 48 headers independently confirm the
Engram shapes and dtypes below; weight payloads were not downloaded or hashed.

Initial upstream converter work inspected:
https://github.com/ggml-org/llama.cpp/pull/28696
at head `cd628010bc3fc0a787d156c969d52a0789451c96`.
That revision is conversion-only, temporarily labels the architecture
`deepseek4`, and explicitly cannot load in the existing runtime. Its metadata does not yet
constitute a complete V4.1 execution contract: cache/index source schedules,
candidate selection, and tokenizer-compressed Engram addressing need explicit
representation or a pinned, fully validated profile. Do not adopt its label as
evidence of runtime compatibility.
The refresh below supersedes that provisional naming: the PR now uses
`deepseek41`, still without a runtime.

Official Rust protocol toolkit: https://github.com/deepseek-ai/deepseek-recipe
(`deepseek-recipe-core`, `deepseek-recipe-encoding`, and protocol conversion).
Evaluate a pinned version against fixtures before adding a dependency.

## DwarfStar release: 2026-09-12

Supersedes the earlier "not yet public" DwarfStar status below. Local
`/Users/tito/code/ds4` fast-forwarded from `6289c51` to
`bd66c402070042bf0a79ad6ece8242de4c93680c`:
https://github.com/antirez/ds4/commit/bd66c402070042bf0a79ad6ece8242de4c93680c
(`DeepSeek v4.1 Flash support for Metal`). Untracked local logs were preserved;
no model was downloaded, binary rebuilt, or inference benchmark run.

The release includes Metal text and vision, CLI/agent/server integration,
two-Mac RDMA TP, session/state tests, conversion and calibrated artifacts.
V4.1 DSpark, pipeline execution, CUDA and ROCm remain unsupported. Its GGUF is
a project-specific contract, not automatically interchangeable with the
llama.cpp converter merely because both say `deepseek41`.

Published `antirez/deepseek-v4.1-flash-gguf` artifacts:

| Target | File size | Main weights | Engram |
|---|---:|---:|---:|
| `ds41f-q2` | about 341 GiB | about 152 GiB | about 189 GiB, disk-only |
| `ds41f-q4` | about 483 GiB | about 294 GiB | about 189 GiB, disk-only |

Q2 runs via SSD streaming on a 128 GB Mac, or resident main weights across
two 128 GB Macs. The table retains native FP8 values/scales rather than being
forced through a low-bit table quant to fit. `ds4_engram.c` implements explicit
`pread` of 264-byte rows with `F_NOCACHE` and disabled read-ahead, bounded
2,048-token batches, sorted row requests, duplicate reuse within worker slices,
and up to 16 readers. The graph overlaps Engram reads with encoder work. This is
concrete SSD row access, not a claim that ordinary MLX lazy tensor slicing is
out-of-core.

The user's September 11 screenshot reports an eight-second transition from
the decode expert cache to full encoder residency and about 800 prefill tok/s
afterward. It describes scalar cached, overlapped layer-major, and fully
resident encoder schedules. Keep the transition and post-prefill cache recovery
outside that quoted throughput: 800 tok/s is not end-to-end TTFT including them.

The published scheduler has already evolved beyond that three-way description:

- `ds41_prefill_count` uses cached scalar short appends, batched/layer-major
  sweeps and a wide-sweep path. Streaming batching normally starts at 256 new
  tokens, raised to 1,024 on continued prefills when at least half the experts
  fit in cache. Resident single-host prefills batch from eight; TP from 32.
- Wide sweeps use bounded intermediate state, up to 32,768 rows within a 3 GiB
  carry budget, and become selectable at 4,096 remaining tokens. This lets each
  layer's expert bank serve multiple compute tiles before being released.
- `ds41_encoder_acquire` considers full encoder residency from 16,384 remaining
  tokens, but explicitly skips it when wide sweeps are available/enabled. The
  code explains that a wide sweep reads each encoder layer once without
  discarding the decode cache. Encoder residency is a retained alternative,
  not an unconditional default for every sufficiently long prompt.
- Encoder acquisition borrows the admitted expert-cache budget and restores it
  on release/cancellation; it must not allocate a second weight set alongside
  the existing cache. A token-major tail helps warm subsequent decode.

Inspect `ds4.c` around `ds41_carry_cap`, `ds41_encoder_acquire`,
`ds41_decoder_prepare`, and `ds41_graph_prefill_sweep`. The deferred decoder
path reconstructs a depth-dependent 2,541-token dependency suffix, rather than
assuming the paper's approximate 128-token replay is exact. Batched/scalar
floating-point differences still need separate qualification.

`QA_BEFORE_RELEASES.md` section 17 retains source/quant audits, API
teacher-forced comparisons, cancellation and exact restore checks, long sessions
past 100K, and explicit failed experiments. In particular, an indirect Q4
expert-table path escaped the streaming budget and triggered a watchdog/reboot;
the release restricts that full-table path to resident execution. This is a
direct warning to keep qwen-llm's residency safety holds and independent host
memory accounting, not permission to copy pinning policies wholesale.

Bring-up consequence: DwarfStar is now a public native Metal reference for
configuration, source-faithful disk Engram, exact decoder dependency pruning,
and bounded expert streaming. Compare wide sweeps against encoder residency
using total request time and following decode, not only steady prefill speed.

## Community refresh: 2026-09-11

Evidence distinguishes code on main, runnable feature branches/images,
unmerged proposals, and author-reported demonstrations. No community benchmark
was rerun locally. The official HF revision remains
`dba1be0a40aa45a94ad051997016db3960a90277`.

### Local checkout updates

All paths are under `/Users/tito/code`. Nine existing checkouts fast-forwarded
without merges, rebases, stashes, or local-edit loss:

| Checkout | Updated HEAD | V4.1 status at inspection |
|---|---|---|
| `llama.cpp` | `df03399b8` | Converter PR only; no mainline runtime found |
| `mlx` | `a3af4d1` | Core primitives; no V4.1-specific PR/branch found |
| `mlx-lm` | `c62d795` | No V4.1 model; even the identified V4 model PR remains open |
| `ds4` | `6289c51` | Public streaming groundwork; V4.1 demo not yet pushed |
| `vllm` | `84030bbe3` | Model definitions/frontend merged; runtime registration/integration pending |
| `sglang` | `9df72e8f5a` | Cookbook on main; runtime on `dsv4.1` |
| `omlx` | `25bccf5f` | No V4.1 integration found; generic/V4 streaming PRs open |
| `Rapid-MLX` | `cc8681a2` | Native text research runtime and qualification, not product exposure |
| `vllm-metal` | `dfe1951` | No explicit V4.1 Metal integration found |

MTPLX upstream rewrote history: original local `main` at `4ce96908` was left
untouched after `pull --ff-only` refused. Latest `origin/main` is `21be78b3`;
it is available in the detached `MTPLX-upstream-latest` worktree. The sides have
663/1,231 unique commits, so silently resetting would discard local history.
No V4.1-specific work was found there. PR 254 is V4 target-only delegation to
external mlx-serve, not a V4.1 implementation.

No existing mlx-serve checkout was found in the searched home tree. Cloned
`ddalcu/mlx-serve` to `mlx-serve` at `fa76a4b5`, without initializing submodules.
Its DwarfStar gitlink is `efdadd41`; V4 SSD support does not imply V4.1 support.
Also cloned `PipeNetwork/deepseek-v41-mlx` to `deepseek-v41-mlx` at `8bdd543c`.
Installed environments and binaries were not rebuilt.

Additional fetched research refs, without changing existing checked-out branches:

- llama.cpp: `upstream/pr-28696` = `f8640de8`.
- vLLM: `origin/dsv41-feat` = `c9d909e8`.
- SGLang: `origin/dsv4.1` = `824bb45e`; other public branches arrived with fetch.
- mlx-lm: `origin/pr-1797` = `ffa5959f` (V4, not V4.1).
- MTPLX: `origin/pr-254` = `1d4cbe5a` (external V4 backend).

### DwarfStar: working private code, public streaming groundwork

Primary author report:
https://x.com/antirez/status/2098121665771110540
with replies `2098123316468847044` and `2098128125930492000`.

Antirez reports 15 tok/s with V4.1 SSD streaming on one 128 GB M5 Max and
25 tok/s with two Macs using tensor parallelism over RDMA. He explicitly says
he will push when QA is ready. The public branch list and main checkout do not
yet contain that implementation. These are demonstration numbers, not a
reproducible pinned quant/context/quality comparison with qwen-llm.

Public issue: https://github.com/antirez/ds4/issues/1023

Already inspectable groundwork in `ds4.c`, `ds4_metal.m`, and `ds4_ssd.c`:

- `91dda0b`: let expert caching adapt beyond the built-in preload ranking.
- `715ed7a`: age expert caches from the first routed layer.
- `85837d0`: increase cache use within the safe working-set budget.
- `660e1d4`: bound streaming model views while preserving auxiliary mappings.
- `b6af0ad`: bound GLM prefill priorities so decode can adapt.

The author also proposes keeping the encoder weights resident during large
prefills, exploiting CED's phase asymmetry. This is a next step in the report,
not a demonstrated completed optimization. Transfer the phase-aware residency
idea, not an assumption that decoder replay or phase-transition costs vanish.

### Apple model runtimes and serving

- llama.cpp PR 28696 now declares `deepseek41` and leaves V4's tensor list
  untouched. Still conversion-only. Reported output sizes are 473.1 GiB staging,
  246.3 GiB Q2_K, and 323.4 GiB Q3_K_M. The staging file has MXFP4 experts, so
  some nominally lower-bit recipes enlarge individual projections. These are
  conversion checks, not executable quality results.
- mlx-lm PR 1797 is V4-Flash-0731, not V4.1:
  https://github.com/ml-explore/mlx-lm/pull/1797 . No explicit V4.1 PR was found
  in mlx-lm or MLX core. Generic disk-offload PR 1588 is closed without merge;
  do not infer upstream availability from downstream references to it.
- MTPLX PR 254 remains open:
  https://github.com/youssofal/MTPLX/pull/254 . It launches an external V4
  target-only backend and does not add native V4.1 or DSpark.
- mlx-serve's V4/DwarfStar SSD backend is real, but no V4.1-specific code or
  branch was found. Track its backend pin after DwarfStar publishes support.
- oMLX PR 3468 implements SSD expert streaming, measured on Qwen Flash-Next,
  V4-0731 and GLM-5.3, not V4.1:
  https://github.com/jundot/omlx/pull/3468 . PR 2595 provides a separate
  checkpoint-backed SwitchGLU offload implementation, with native DeepSeek
  kernels explicitly deferred: https://github.com/jundot/omlx/pull/2595 .

PipeNetwork's standalone MLX port supplies initialized tiny-model parity,
chunked prefill, dequantization tests, explicit owner caches, strict loading and
real-checkpoint text/quantization results:
https://github.com/PipeNetwork/deepseek-v41-mlx . It is not stock mlx-lm support
and does not provide a qualified vision or DSpark runtime. Its layer-at-a-time
streaming machinery is not proof of DwarfStar-style fast cached-expert decode.
Read `docs/upstream-notes.md` together with the newer README: earlier sizing
estimates are superseded by actual load-transient results.

Rapid-MLX PR 3301 merged its native port under `scripts/deepseek_v41_native/`
and a qualification record, not server/catalog support:
https://github.com/raullenchai/Rapid-MLX/pull/3301 . On a 256 GiB M3 Ultra,
a 336/384-expert native affine 2-bit REAP build occupies about 199 GiB and
reports 7.92 tok/s on a very short correctly framed greedy smoke. It fails
their 12 tok/s product floor and is explicitly withheld from publication.

Open PR 3297 retains additional full-model experiments:
https://github.com/raullenchai/Rapid-MLX/pull/3297 . Expert-local gate/up fusion
was a small win; a larger mHC fusion speedup failed logit/top-1 agreement; giant
all-expert packing collapsed to 0.436 tok/s despite a 2.02x layer microbenchmark.
These are particularly useful negative results for native Metal design. They
are not directly comparable with the M5 Max streaming demonstration.

### NVIDIA/AMD serving: recipes point to development images

vLLM merged substantial definitions in PR 56228 and frontend support in 56208.
However PR 56214 (`dsv41-feat` -> main) still adds the model/draft registry
entries and integration changes. The recipe explicitly says no pip wheel
carries this architecture and points to `deepseekv41-flash-0909` images.
Source: https://github.com/vllm-project/recipes/blob/main/models/deepseek-ai/DeepSeek-V4.1-Flash.yaml
Its 522B headline conflicts with the official 552B-backbone report; retain the
official accounting and treat recipe defaults as deployment-specific.

Highest-value unmerged vLLM work:

- Runtime integration: https://github.com/vllm-project/vllm/pull/56214
- SWA bounded replay: https://github.com/vllm-project/vllm/pull/56227
- Engram asynchronous prefetch/shared host tables:
  https://github.com/vllm-project/vllm/pull/56357
- Candidate-only sparse MQA scoring:
  https://github.com/vllm-project/vllm/pull/56254
- Microbatch positions/Engram histories:
  https://github.com/vllm-project/vllm/pull/56224
- Mega attention: https://github.com/vllm-project/vllm/pull/56344
- Kernel tracking: https://github.com/vllm-project/vllm/issues/56217

PR 56357's GB200 measurements show nearly neutral decode and sub-1% TTFT gains
from overlapping small Engram transfers; it is not evidence that SSD expert
streaming is similarly cheap. PR 56254's sparse-indexer gains are context- and
batch-dependent, default-off, and have near- rather than exact top-k agreement.

SGLang's main cookbook uses `lmsysorg/sglang:dev-dsv41` (or the MI35x variant)
and explicitly says support has not shipped in a release. Its runtime lives
on `dsv4.1`, PR 38798:
https://github.com/sgl-project/sglang/pull/38798 . The branch includes Engram,
ratio-1/2 compression, vision, DSpark, and encoder/decoder replay paths.
Some closed/merged PRs target this feature branch, not main: static DSpark PD
support PR 38949 is one example.

Highest-value remaining SGLang work:

- HiCache plus encoder replay correctness/compatibility:
  https://github.com/sgl-project/sglang/pull/38957
- Candidate-only DeepGEMM indexing:
  https://github.com/sgl-project/sglang/pull/38944
- Bounded Blackwell prefill workspace:
  https://github.com/sgl-project/sglang/pull/38956
- Unified indexer execution decisions:
  https://github.com/sgl-project/sglang/pull/38962
- Preserve fused shared expert in VL routing:
  https://github.com/sgl-project/sglang/pull/38963
- Actual packed main-KV storage on Hopper:
  https://github.com/sgl-project/sglang/issues/38902

FP4 fake-quantized values do not imply packed FP4 storage: the Hopper issue
describes a path that requantizes into the legacy FP8/BF16 layout. Likewise,
masking dense indexer logits is not candidate-only computation. Audit the
selected hardware backend before assigning the paper's 890 B/token or bounded
indexer cost to a community runtime.

### Newly identified reference defect

PipeNetwork's `docs/upstream-notes.md` and `tests/test_parity.py` identify a
real ownership hazard visible in the pinned official source. `Indexer.forward`
reassigns `shared_attn.index_k` only when an owner publishes a new key. On a
ratio-2 step without pair completion, encoder owners can instead read the
previous forward's last owner (decoder layer 20). Main cache ownership and
index cache ownership are not being advanced under the same conditions.

Their fix publishes the owner's cache pointer even when there is no new row;
see `deepseek_v41_mlx/attention.py`. Their parity harness explicitly patches
the reference for this comparison and separately measures the unpatched error.
The reported tiny-model relative-max logit difference is 0.67; this survey
confirmed the source mechanism but did not rerun that numerical experiment.

Our oracle must therefore pin both the official source and any correction,
keep patched/unpatched negative controls, and test every pair phase. Do not
reproduce the stale-pointer behavior merely to match an incorrect golden.

## Confirmed architecture

Report sections 2.1-2.4, 3.2, and 4.2 agree with the released config on:

| Property | V4 Flash-0731 | V4.1 Flash |
|---|---:|---:|
| Backbone parameters, excluding Engram | 284B | 552B |
| Additional Engram table parameters | none | 196.614B |
| Active parameters per token | 13B | about 8B prefill / 16B decode |
| Backbone layers | 43 | 40: encoder 0-19, decoder 20-39 |
| Hidden width | 4096 | 5120 |
| Routed experts / selected / shared | 256 / 6 / 1 | 384 / 6 / 1 |
| Expert intermediate width | 2048 | 2304 |
| Main query heads / shared KV heads | 64 / 1 | 64 / 1 |
| KV width / rotary tail | 512 / 64 | 512 / 64 |
| Query low-rank width | 1024 | 1280 |
| Output groups / low-rank width | 8 / 1024 | 8 / 1024 |
| Indexer heads / width / selected rows | 64 / 128 / 512 | 32 / 128 / 512 |
| SWA window | 128 | 128 |
| Residual streams / Sinkhorn iterations | 4 / 20 | 4 / 20 |

V4.1 uses pure CSA2 plus local SWA: HCA is explicitly removed, not merely
unmentioned. Every layer retains its own local SWA KV, including decoder layers.
The active-count asymmetry is phase-dependent stack execution, not a new
per-token variable expert-count router. Decode traverses both halves.

### CSA2 ownership and selection

All indices here are zero-based:

| Layers | Ratio | Mode | Main KV and index-K owner | Selection producer |
|---|---:|---|---:|---:|
| 0-1 | 0 | SWA only | none | none |
| 2, 8, 14 | 2 | Full | self | self |
| 3-7, 9-13, 15-19 | 2 | Reuse | preceding Full | preceding Full |
| 20 | 1 | Full | 20 | 20 |
| 24, 28, 32, 36 | 1 | Reindex | 20 | self |
| Other decoder layers | 1 | Reuse | 20 | preceding Full/Reindex |

Thus 38 global-attention layers require only four global cache owners and
eight index-producing layers. All layers still compute their own main Q.

CSA2 removes V4 CSA's overlapping compression and compressor absolute-position
embedding. Ratio-2 compression pools non-overlapping pairs with a per-channel
softmax gate; ratio 1 is a projection plus RMSNorm. Indexer K is projected from
the normalized, pre-RoPE main latent, not through a separate hidden-state
compressor. The reference rotates and quantizes index keys before rotating and
quantizing the main latent. Preserve this ordering and partial-pair visibility.

Decoder layer 20 also selects at most 2,048 blocks of eight positions by each
block's maximum index score, pinning the newest reachable block. Its candidate
pool contains at most 16,384 positions. Reindex layers select their own top-512
within that pool. The pool is query-specific, not a permanent pruning of history.

Only the four later decoder indexers have context-bounded candidate scoring.
The first decoder indexer and the three encoder indexers still scan their
visible histories. Do not claim the whole model is asymptotically O(1) in context.

### CED and replay are separate contracts

Report section 2.2 credits YoCo as an inspiration. Decoder global KV comes from
the final encoder states; layer 20 produces it and all later decoder layers
reuse it. There remains a representation per token, not one vector summarizing
the whole prompt, and decoder SWA still comes from decoder-local hidden states.

In the reference, layer 20 consumes the four-stream encoder residual through
the carried input-mixing coefficients and its attention norm before projecting
global KV. A native encoder/decoder boundary must preserve this exact seam,
not substitute an unweighted stream mean or an arbitrary final hidden tensor.
The decoder's initial residual also needs those streams.

The released `Transformer.forward` runs every prompt position through all 40
layers. That supplies a full-forward correctness reference, not the optimized
8B-prefill serving implementation, subject to the index-owner correction noted
in the community refresh. A split prefill path can generate decoder
global KV without running every prompt position through decoder attention/MoE,
but it still has to construct decoder local state and first-token logits.

Report section 3.2.2 explicitly distinguishes:

- Decoder bounded replay: run the last 128 prompt positions through the decoder,
  truncating local attention to that segment. It is approximate relative to a
  full decoder forward and is simulated during post-training.
- Encoder bounded replay: on a global-prefix hit with missing encoder SWA, replay
  the last 128 cached positions plus the uncached suffix. Replayed positions read
  existing global KV without regenerating or overwriting it; only new positions
  publish new global KV.
- Exact reconstruction needs a larger depth-dependent local receptive field
  (the report uses layers times window), or retained exact local state.

Crucially, encoder replay makes suffix state depend on the cache-hit boundary.
The same text prefix can therefore induce different caches under different
replay histories. Keep exact snapshots separate from approximate prefix reuse;
record replay policy and relevant boundary/provenance in compatibility and
observability. Never silently promote approximate reuse into an exact cache hit.

Their production system still keeps short-lived encoder SWA in host memory.
It eliminates long-lived SWA persistence to SSD, not all live SWA storage.

### Single-Pass mHC

This is a mathematical wiring change, not just a faster implementation of V4.
Each attention/FFN sublayer uses input-mixing coefficients predicted at the
previous sublayer, while predicting coefficients for the following one.
Residual combination and output mixing still use the current coefficients.
Sinkhorn remains at 20 iterations.

The reference initializes the first input mix to one-hot stream zero, carries
the next mix through the stack, and uses the last FFN's mix for final collapse.
V4's separate final `output_hc_*` predictor is absent. Preserve the coefficient
handoff across Engram injection and the encoder/decoder split.

Mega-mHC fuses residual update, next coefficient prediction, input mixing,
normalization and activation conversion. The reported activation-traffic saving
does not establish a Metal wall-time gain. Start with unfused verified math.

### Engram: reusable access pattern, different operator

The two tables are in shards 47 and 48:

| Layer | FP8 table shape | E8M0 scale shape |
|---|---|---|
| 1 | `[384006168, 256]` | `[384006168, 8]` |
| 14 | `[384016682, 256]` | `[384016682, 8]` |

Each token fetches 24 rows per module: orders 2/3/4, eight heads per order,
256 channels per head. Concatenated input width is 6,144. The FP8 `wkv` matrix
is `[25600, 6144]`, producing four 5,120-wide keys and one shared value.
Each stream has its own normalized, signed-square-root sigmoid gate. Q/K norm
weights are `[4, 5120]`. There is no short causal convolution; the report says
its benefit did not justify inference complexity.

DeepSeek hashes tokenizer-compressed IDs (99,092 distinct IDs), not raw BPE IDs.
The normalization includes Unicode normalization, accent removal, lowercasing,
whitespace treatment, and special handling of partial UTF-8 tokens. Multipliers
come from NumPy's per-layer seeded generator; prime ranges are distinct across
orders, heads, and modules. Export verified maps/multipliers/primes as model
metadata or fixture constants rather than replacing the RNG with a Rust default.

Padding is the compressed form of token 2. Image spans terminate hash lookback
and receive no Engram contribution. Do not inherit Qwen PLE's EOS-boundary
policy. Hash history, modality boundaries, and partial compressor state all
belong to the sequence, not to shared model weights.

### Numerical formats

- Main global KV: post-RoPE E2M1, one E4M3 scale per 16 channels, no second-level
  global scale. A 512-channel entry is 256 + 32 = 288 bytes.
- Indexer Q/K: E2M1 with E8M0 scales per 32 channels; a key is 64 + 4 = 68 bytes.
  The released reference does not apply V4's Hadamard transform. Do not carry
  `LlamaCppB10222F16HadamardV1` into this family.
- Local SWA KV: FP8 over the entire post-RoPE vector, including the rotary tail;
  not V4's mixed FP8-NoPE/BF16-RoPE oracle or our V4 F16 session ABI.
- FP8 linear weights use 32x32 scale tiles. Engram table scales instead cover
  32 columns of one row. Treating the table as tiled linear weights corrupts it.
- Routed experts use packed E2M1 with E8M0 scales per 32 reduction elements.
  Validate nibble order and conversion against the source format, not its name.
- RMS epsilon is 1e-20; HC epsilon is 1e-6. BF16 casts and QAT round trips are
  part of the reference contract, even if initial CPU scratch uses F32.

The 890-byte claim follows directly from ownership:

```text
main KV + index K per entry = 288 + 68 = 356 bytes
three encoder owners at ratio 2 + one decoder owner at ratio 1
global bytes per input token = (3/2 + 1) * 356 = 890
```

At 1,048,576 tokens this is 890 MiB of packed global cache, excluding local
state, frontiers, alignment, temporary selections, and scratch. The reference
stores dequantized BF16 cache tensors after QAT, so it does not realize this
packed footprint. Its indexer also scores the full history before masking
candidates; a native gather-before-score implementation is needed to realize
the hierarchical compute saving.

## Memory feasibility

Do not repeat the earlier confusion between total parameters and accelerator
residency. Engram tables can remain CPU/storage-addressed, but backbone experts
remain a separate residency problem. Ideal backbone sizes, excluding scales,
format overhead, vision/draft accounting and runtime memory:

| Storage | 552B backbone |
|---|---:|
| Exact 3 bits/weight | 192.78 GiB |
| Exact 4 bits/weight | 257.05 GiB |

The two table payloads contain exactly 196,613,849,600 parameters. Native FP8
plus per-row E8M0 scales costs 202,758,032,400 bytes (about 188.83 GiB), excluding
projections. Hypothetical IQ4_NL tables cost 110,595,290,400 bytes (about 103 GiB);
this is storage arithmetic, not an established acceptable quantization recipe.

Across both modules, one text token needs 12,672 packed FP8 table bytes including
scales, before duplicate elimination, page-read amplification, and staging.
On unified memory, CPU row gathering avoids making the entire table a GPU
buffer; it does not make resident pages free. SSD-backed mmap can fault on the
CPU. Measure cold/hot row locality, page amplification, prefill throughput and
tail latency before promising low overhead. No whole-table GPU wiring or
unbounded dequantization. Existing whole-model residency-set safety holds apply.

The immediate bottleneck for full-model local deployment is backbone capacity,
not the sub-gigabyte global KV. A larger host, independently validated smaller
checkpoint, or explicit expert-streaming design is needed; none is assumed here.

## Code reuse map

| Existing seam | Reuse assessment |
|---|---|
| `crates/qwen-llm/src/gguf.rs`, retained Metal storage and admission | Reuse descriptors, split mappings, ownership and admission; freeze a new profile |
| `crates/qwen-llm/src/deepseek_v4.rs` | Keep strict V4 schema; separate V4.1 config/binder |
| `crates/qwen-llm/src/deepseek_v4_metal/attention.rs` | Shared-KV attention, sinks, RoPE and grouped output are groundwork; new cache formats and HC wiring |
| `crates/qwen-llm/src/deepseek_v4_metal/indexer.rs` | Selection primitives are useful; replace compressor, source ownership and index numerics |
| `crates/qwen-llm/src/deepseek_v4_metal/moe.rs` and `metal/moe.rs` | Reuse quantized expert primitives after 384-expert/5120x2304 geometry tests; no V4 early hash router |
| `crates/qwen-llm/src/qwen4exp_ple.rs` | Bounded selected-row gather is useful; current view accepts IQ4_NL only |
| `crates/qwen-llm/src/qwen4exp.rs` and `qwen4exp_ple_metal.rs` | Do not reuse Qwen hash semantics or convolutional injection as DeepSeek Engram |
| `crates/qwen-llm/src/qwen4exp_residency.rs` | Model for separating CPU table source from Metal weight views; not a ready-made FP8 table loader |
| `crates/qwen-llm/src/deepseek_v4_metal/session.rs` and `snapshot.rs` | Reuse lifecycle principles, not V4 state layout or identity domains |

Our current V4 session is fixed around 43 layers and 4096 hidden width. Its
authoritative attention/index caches are F16 with the pinned llama.cpp numerical
contract. Qwen Flash-Next support is currently a serial text lane, not a ready
serving, durable-prefix, or vision backend (`README.md`). Neither path can be
advertised as V4.1 support by changing family detection.

## Bring-up gates

1. **Metadata and reference contract.** Add separate V4.1 config/role definitions,
   validate the four cache owners and eight index sources, and retain deterministic
   fixtures for compressed tokenizer IDs, hash rows, boundary handling, packed
   FP4/FP8 formats and memory arithmetic. Do not register a provisional GGUF label
   as runnable. Pin converter metadata before admitting a converted artifact.
2. **Reduced-shape numerical oracle.** Exercise real math with initialized,
   deterministic small weights: Engram, lagged mHC, pair compression, source
   aliasing, full/reindex/reuse and candidate selection. The upstream self-test
   uses uninitialized weights and is only a plumbing check, not this oracle.
3. **Full-forward text correctness.** Build scalar execution and then packed
   prefill, initially traversing both halves for every token. Compare intermediate
   tensors, decisions, logits and state against a pinned numerical reference.
   A GGUF quant changes weights/possibly arithmetic; separate that target from
   the official FP8/FP4 activation path rather than demanding false bitwise parity.
4. **Exact session lifecycle.** Validate singleton/chunk equivalence, interleaved
   independent sessions, forks, cancellation, rollback and snapshot/restore.
   Keep global stores per owner and selection/candidate state per query. Do not
   copy the Python process-global `shared_attn` into the native engine.
5. **CED deployment path.** Split encoder prefill, decoder-global publication,
   decoder-tail replay and full-stack decode. Keep exact full-forward as the
   baseline and expose approximate replay explicitly. Encoder prefix replay must
   leave cached global rows untouched. Quality-gate replay independently of speed.
6. **Product extensions.** Add V4.1 prompt/DSML parsing, then native vision and
   DSpark behind separate gates. Integrate serving only after lifecycle tests;
   retain V4 prompt, cache and execution regression coverage throughout.

Boundary tests must include positions 0/1/2, odd/even pair completion, 127/128/129,
candidate block edges, the 16,384-candidate limit, every Full/Reindex transition,
and splits immediately around Engram layers 1/14 and the 19/20 boundary.
Tie handling, invalid indices, causal masking and absolute-vs-ring index domains
need explicit decision fixtures, not only final-logit tolerance tests.

V4.1 prompt encoding changes DSML tag names (leading spaces and `calls` instead
of V4's `tool_calls`), uses numeric reasoning effort 1-100, and supports explicit
mid-conversation system messages. Aliases are low=50, high=75, max=100. Pin exact
bytes and streaming parse fixtures; do not silently use the V4 encoder/parser.

Vision adds a 32-layer ViT, 2D RoPE, 3x3 pixel-unshuffle, two-layer projector,
image delimiters and modality-specific expert-selection bias. A text-only first
milestone must reject images rather than tokenize placeholders as ordinary text.

DSpark uses three draft blocks, five draft positions, Markov rank 256, 128 routed
experts with three active, and target layers 37/38/39. The reference releases its
forward path but no speculative scheduler loop. Existing speculative lifecycle
principles transfer; neither a Qwen DFlash adapter nor a V4 draft asset is a
drop-in implementation. Keep greedy and distribution-exact claims separate.

## Assessment limits

The report/config/reference and two table headers were inspected; no full model
was loaded, no Metal performance run was made, and no quality result was measured.
Remaining deployment questions are a complete accepted GGUF schema, a feasible
local backbone asset/residency plan, numerical fixture generation, and actual
batch-one performance. Benchmark tables and production throughput claims in the
report are vendor evidence, not qwen-llm acceptance results.

CED creates a useful model-specific representation boundary, but does not by
itself establish portable cross-model latent briefings or lossless semantic
compaction. Decoder-local state and learned coordinate compatibility still matter.
