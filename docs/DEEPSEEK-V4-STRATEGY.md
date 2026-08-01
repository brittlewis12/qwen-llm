# DeepSeek V4 support strategy

Status: active architecture spike on branch `deepseek-v4-spike` in the
`qwen-llm-dsv4` worktree. This document is scoped to Apple Silicon Metal and
the DeepSeek-V4-Flash-0731 target model. It uses dependency order and promotion
gates rather than delivery dates.

## Decision

Explore DeepSeek V4 as a second first-class model family, not as another Qwen
variant. Preserve the existing Qwen forward path and performance contracts.
Build separate DS4 model, session, cache, CPU-oracle, and Metal-forward types,
then share only components whose tensor and numerical contracts actually
match.

The initial shipping target is non-speculative generation from a standard
llama.cpp-schema GGUF. DSpark, custom quant recipes, SSD expert streaming, and
a project rename are independent follow-on decisions.

## Why this target

DeepSeek-V4-Flash-0731 is the official post-training release of V4 Flash. The
target backbone is 284B total / 13B active, MIT licensed, and supports a 1M
context. The complete official checkpoint is about 304B parameters because it
also carries three DSpark stages.

DeepSeek reports large 0731 gains over the April preview on agentic workloads,
including 82.7 on Terminal-Bench 2.1 and 54.4 on DeepSWE. Those are vendor
results from its unreleased harness, so they justify exploration but are not a
quality acceptance test for this engine.

The architecture is also a worthwhile systems target. At 1M context, the
paper estimates V4 Flash at 10% of DeepSeek-V3.2 single-token FLOPs and 7% of
its KV cache. The often-quoted 27% / 10% figures apply to V4 Pro, not Flash.

Primary sources:

- Model: https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731
- Paper: https://arxiv.org/abs/2606.19348
- Official encoding reference: the model repository's `encoding/` directory
- Standard GGUF execution reference: llama.cpp PR 24162 and descendants

## Frozen reference asset

The current local target is:

`/Users/tito/models/deepseek-v4-flash-0731/UD-IQ3_XXS/DeepSeek-V4-Flash-0731-UD-IQ3_XXS-00001-of-00004.gguf`

Observed facts from the complete four-shard mapping:

- Source repository: `unsloth/DeepSeek-V4-Flash-0731-GGUF`
- Quant: `UD-IQ3_XXS`, `general.file_type = 23`
- Size: 102,999,888,416 bytes (95.93 GiB)
- Shards: 4
- Tensors: 1,328
- Architecture: `deepseek4`
- No converted `mtp.*` / DSpark tensors

Pinned shard hashes:

| Shard | Bytes | SHA-256 |
|---|---:|---|
| 1 | 5,257,664 | `9758eb3d78e1afe8852543931703f4f1cd6fbb07f492d4ed853f5d2f6e43be5a` |
| 2 | 49,485,728,288 | `afcfd59721d4da86bc3301e16ca624af202d8af3fa3f9fbbfbb04b3b47666cfd` |
| 3 | 49,437,886,752 | `64eaf514a763597ba7bb50866583d8db5eabbbbce3cb2f616d749af3890155ca` |
| 4 | 4,071,015,712 | `5df52988c56348a22d15da809e9ac4f0cc59cc1c412347f1481dda4685ce89b2` |

The download records resolve shard 2 at a different repository commit from
the other shards. The GGUF split metadata and complete tensor schema validate,
but reproducible benchmark packets must pin the four content hashes rather
than claim one repository revision.

## Corrected architecture facts

The April paper, official 0731 config, current llama.cpp, vLLM, SGLang, and
DwarfStar agree on the following target geometry:

| Property | Flash-0731 |
|---|---:|
| Target layers | 43 |
| Hidden size | 4,096 |
| Vocabulary | 129,280 |
| Query heads | 64 |
| Shared K/V heads | 1 |
| K=V width | 512 (448 non-RoPE + 64 RoPE) |
| Q low-rank width | 1,024 |
| Output groups / low-rank width | 8 / 1,024 |
| Local window | 128 |
| Routed experts / selected | 256 / 6 |
| Expert width / shared experts | 2,048 / 1 |
| Hyper-connection streams | 4 |
| Sinkhorn iterations | 20 |
| CSA index heads / width / top-k | 64 / 128 / 512 |

The target layer schedule is:

- Layers 0-1: local sliding-window attention only (`ratio = 0`).
- Layers 2,4,...,42: CSA (`ratio = 4`), 21 layers.
- Layers 3,5,...,41: HCA (`ratio = 128`), 20 layers.
- The ratio array has three trailing zeros aligned with attached stages, but
  the current GGUF schema gives that tail no execution semantics. The frozen
  profile pins it as opaque metadata; standard llama.cpp conversion omits the
  corresponding `mtp.*` tensors.

Important terminology and semantic corrections:

- CSA and HCA are interleaved layer types. CSA does not run "on top of" HCA.
- HCA uses ratio 128, not ratio 2 or 4.
- V4 attention is shared-KV MQA with low-rank Q and grouped low-rank output.
  Calling it a V3 MLA baseline obscures a materially different cache contract.
- Every layer still has a 128-token local branch.
- CSA uses overlapping ratio-4 compression, a separate compressed index key,
  and a Lightning Indexer over the complete compressed history. It attends to
  the selected compressed rows plus the local window.
- HCA compresses non-overlapping groups of 128 and attends densely over all
  completed compressed rows plus the local window.
- mHC maintains four residual streams. Learned pre, post, and 4x4 combination
  mappings surround both attention and MoE; Sinkhorn normalization constrains
  the combination matrix. This is not a replacement for one residual add.
- Layers 0-2 obtain expert IDs from an I32 `[topk, vocab]` table but still use
  learned router scores for expert weights. Later layers use bias-assisted
  selection. The score transform is `sqrt(softplus(logit))`.

## Tokenizer and prompt encoding

The target GGUF contains a byte-level BPE vocabulary with:

- `tokenizer.ggml.model = gpt2`
- `tokenizer.ggml.pre = joyai-llm`
- 129,280 tokens and 127,741 merges
- BOS 0, EOS 1, padding 2
- automatic BOS/EOS insertion disabled

`joyai-llm` uses the same three-stage pretokenizer as DeepSeek V3/V4: numeric
groups of at most three Unicode numbers, fixed CJK/kana regions, then the
DeepSeek letter/mark/punctuation/whitespace expression. It is not Qwen's
pretokenizer even though the BPE machinery is reusable.

The project intentionally pins `unicode-general-category` 1.1.0 (Unicode
16.0) for native pretokenization. Newly assigned code points can therefore
tokenize differently from older Unicode tables even though byte fallback
still guarantees coverage. The Unicode version and representative post-15.0
letter/number behavior are explicit regression tests; future table upgrades
must be reviewed as tokenizer changes.

Raw tokenization is only half of frontend support. The official 0731 release
deliberately does not ship a Jinja template. Its Python encoder defines system,
user, assistant, tool, developer, latest-reminder, reasoning-effort, DSML, and
quick-task behavior. qwen-llm must port and fixture-test that encoder rather
than assume llama.cpp's injected template is byte-identical.

## What transfers from qwen-llm

Directly reusable:

- Validated split-GGUF mmap and tensor descriptors.
- Persistent Shared-buffer ownership, load policy, prefetch, and memory
  admission infrastructure.
- Ordinary Q/K/IQ projection matvec and matmul kernels where dimensions and
  dtypes match.
- Q8_0, Q6_K, IQ3_XXS, IQ3_S, IQ2_S, MXFP4 decoding primitives.
- Existing output-head and GPU argmax structure after final HC collapse.
- Parts of the routed-MoE scheduling and 256-expert top-k infrastructure.
- Kernel tracing, benchmark packet, checkpoint identity, and promotion-gate
  conventions.

Not directly reusable:

- `ArchKind`: it describes dense versus MoE Qwen FFNs, not model family.
- Qwen `Arch`, `Block`, `Model`, `MetalModel`, `MetalSession`, `KvCache`, and
  snapshot ABI.
- `attn_v4`: it hardcodes 256-wide GQA and full K/V caches. Its online-softmax
  structure is useful, but its interface and storage contract are not.
- Qwen's ordinary residual layer loop.
- Qwen's gated-attention and GDN scratch layouts.
- Qwen's tokenizer pre-split and chat renderer.

The frozen IQ3 asset also exposes a concrete MoE gap: 42 layers store routed
gate/up banks as IQ2_S, while the current production grouped-expert path does
not support IQ2_S gate/up. Routed down is mainly IQ3_XXS, with two MXFP4
outliers. Generic matmul dtype coverage is not equivalent to grouped-expert
coverage.

## Oracle hierarchy

Use independent sources for semantics and optimization:

| Source | Frozen revision | Role |
|---|---|---|
| Official DeepSeek checkpoint/code | 0731 release | Model, DSpark, and prompt source of truth |
| vLLM | `b40d859c7b07ae244bcd8c6eecdcdbd9a3afaa07` | Strongest PyTorch equations and optional accelerator cross-check |
| SGLang | `58974ca16ca2a4bb2f02f9ceb9622a0fd2ccf7f8` | Independent cache/indexer/attention and DSpark cross-check |
| llama.cpp | `876a4321163249c43ca4e986818fab5ab081f282` | Standard GGUF schema and generic CPU/Metal end-to-end oracle |
| DwarfStar | `54b36ed9ba42da31b24f2d1a5feb075c2475dbb1` | Complete native-Metal performance reference |
| omlx | `c59b4cb19639cb9dabeeef696354e999aa9963c5` | Secondary MLX graph and sparse-prefill reference |
| Gigatoken | `34a1599f0c0ae7d7cd0d1c530e6522320158b360` | JoyAI/DeepSeek pretokenizer behavioral reference |

Do not copy one implementation wholesale. Port equations into qwen-llm's
types, preserve source/license provenance, and require agreement between at
least two independent implementations for stateful operations. Independence is
about separately implemented executable semantics, not accelerator diversity:
CPU-only DwarfStar and llama.cpp evidence is sufficient. CUDA, ROCm, Colab, and
full-model redownloads are optional falsifiers, never S1 infrastructure gates.

## Dependency graph and promotion gates

### S0: architecture census and tokenizer - implemented in the spike

Deliverables:

- Separate `ModelFamily::DeepSeek4` discriminator.
- Generic DS4 metadata parsing plus a strict Flash-0731 profile and tensor
  binding.
- Closed native tokenizer dispatch for `deepseek4/gpt2/joyai-llm`.
- Family-aware CLI model inspection.

Gate:

- All 1,328 target tensors bind exactly once with no unexpected tensor.
- Geometry resolves to 2 local / 21 CSA / 20 HCA layers.
- Native tokenization matches the sequential regex reference and the pinned
  greeting vector: `Hello, world! -> [19923, 14, 2058, 3]`.
- Existing Qwen tokenizer dispatch and tests remain unchanged.

### S1: operation-level CPU semantic oracle

Implement family-specific F32/BF16 reference operations before a whole-model
loop:

- mHC pre, post, fused post/pre equation, HC head, and Sinkhorn state.
- Shared-KV Q projection, Q/K normalization, partial and inverse RoPE, sinks,
  and grouped low-rank output.
- Ratio-4 overlap compressor and ratio-128 compressor, including visibility
  boundaries and incomplete-window state.
- Lightning Indexer score, exact top-512 selection, Hadamard rotation, and
  activation quantization round trips.
- Hash and learned sqrt-softplus routing with clamped shared/routed SwiGLU.

Gate:

- Checked-in small vectors match directly executed DwarfStar scalar helpers and
  a pinned llama.cpp CPU harness operation by operation. vLLM and SGLang remain
  equation and source-structure cross-checks where their production paths are
  not locally runnable.
- Stateful compressor/cache snapshots from directly executed implementations
  match after every token across window boundaries, not only at final logits.
- mHC and routing agree with two independently implemented, locally executable
  paths. No particular device backend is required.

Spike status: in progress. `deepseek_v4_oracle` now provides allocation-explicit
F32 operations for mHC, Sinkhorn, shared-KV projection and attention, partial
forward/inverse YaRN RoPE, grouped low-rank output, both compressor state
machines, cache-format round trips, indexer scoring/top-k, routing, and clamped
SwiGLU. Normal tests use checked-in operation vectors and compare compressor
state at every ratio-4 token plus the ratio-128 127/128 and 255/256 boundaries.

The fixture generator has two deliberately distinguished sources of evidence:

- NumPy equation transcriptions record the pinned vLLM, SGLang, llama.cpp, and
  DwarfStar source locations. These are reproducible cross-language vectors,
  not claims that those runtimes were executed.
- `scripts/reference/dsv4_dwarfstar_oracle.c` is compiled against the pinned
  DwarfStar checkout and directly executes selected scalar mHC, RoPE,
  compressor-pooling, indexer-QAT, routing-helper, and SwiGLU paths. It does not
  yet execute complete compressor transitions or a standard-GGUF model.
  Generation fails on revision drift.

Run `uv run scripts/reference/generate_dsv4_oracle.py` from the repository root
with DwarfStar at `~/code/ds4`, or set `DSV4_DWARFSTAR_DIR`. Python 3.14 and
NumPy 2.5.1 are pinned by the script metadata. Use the same command with
`--check` for a byte-for-byte drift gate; the ignored
`fixture_regeneration_has_no_drift` test exposes that external-checkout gate to
the Rust harness. Both DwarfStar commit and `ds4.c` content hash are pinned.

S1 is not promoted yet because direct executable coverage is incomplete, not
because a remote accelerator is missing. The current DwarfStar harness directly
covers selected scalar helpers while shared-KV attention/output, complete mHC
composition, indexer scoring, and token-by-token compressor transitions still
rely on transparent NumPy transcriptions. The next evidence step is a pinned
llama.cpp CPU fixture harness plus broader DwarfStar scalar calls. Both use
small synthetic tensors and require neither a model download nor DwarfStar's
custom quant metadata.

The typed cache prerequisite is now implemented separately in
`deepseek_v4_cache`. It is still CPU-oracle infrastructure, not generation
support:

- The frozen geometry binds the exact layer schedule, compressor YaRN/RMS
  numerics, an explicit cache-storage version, and a caller-supplied strong
  checkpoint/numerics identity. Geometry-compatible weights cannot exchange
  snapshots accidentally.
- A detached whole-token transaction stages layers in order. Transaction views
  include the current raw row and a compressed row completed by that same token;
  only a complete commit publishes state. Late layer failure, including a raw
  ring overwrite or a CSA/HCA boundary, leaves logical state unchanged.
- Every layer owns a position-tagged local ring. CSA owns independent ratio-4
  attention and indexer frontiers plus aligned histories; HCA owns a ratio-128
  frontier and dense history. Completed histories allocate lazily in fixed-row
  typed slabs, avoiding both one allocation per row and whole-history
  relocation at geometric vector-growth boundaries.
- `F32QuantizationOracleV1` stores decoded F32 values after the intended mixed
  FP8/BF16 attention-cache round trip or Hadamard/MXFP4 indexer QAT. It records
  numerical semantics without pretending to be the packed S6 Metal ABI.
- Full typed snapshots validate ring tags, history shape/count, compressor
  phase, CSA alignment, profile, storage version, and checkpoint identity before
  replacement. They deep-copy F32 histories and are correctness artifacts, not
  a 1M-context persistence design.

The CPU transaction stages one bounded compressor candidate per compressed
layer so rollback is structurally simple. That copies roughly the complete set
of compressor frontiers per token and is intentionally not the Metal execution
plan. A production `DeepSeekV4Session` must use preallocated deltas, shadow
banks, or command-buffer ordering while preserving the same commit contract.

Tests cover same-token positions 3/127/255, post-boundary phases 4/128/256,
local-ring wrap, late failure and retry, exact storage round trips, empty and
terminal snapshots, continuation equivalence, and adversarial snapshot
corruption. Ratio-4 restore also validates untouched overlap rows between exact
boundaries, not only full-lane equality at positions divisible by four.

One known reference difference is frozen explicitly: the oracle follows vLLM
MXFP4 and DwarfStar for the indexer (UE8M0 power-of-two scale floor near
`2^-126`, E2M1 round-to-nearest-even). SGLang's current
`fp4_indexer.py` uses a `1e-4` pre-rounded scale floor and chooses the lower code
at exact E2M1 midpoints. This does not affect ordinary non-tiny activations. It
must be resolved against the official contract or an explicit bit-level spec
before packed-cache promotion; it does not create an accelerator requirement.

### S2: local-only Metal backbone

Implement a separate `DeepSeekV4MetalModel`, `DeepSeekV4Session`, and forward
loop for layers 0-1, including mHC, local shared-KV attention, MoE, and final HC
head. Add IQ2_S grouped expert gate/up support rather than dequantizing the
entire expert bank.

Gate:

- One local layer matches the CPU oracle at named intermediate boundaries.
- Layer 0 hash expert IDs and weights match exactly.
- No existing Qwen benchmark packet regresses by more than 2%.

This is a layer-local correctness gate, not a claim that skipping later layers
can produce meaningful model tokens.

### S3: HCA lane

Add ratio-128 compressor state, compressed-cache insertion, mixed local plus
dense-compressed attention, and prefill planning.

Gate:

- Boundary vectors at positions 127, 128, 255, and 256 match both CPU and an
  external implementation.
- One HCA layer matches named intermediate states for decode and batched
  prefill.
- Incomplete groups are never visible to attention.

### S4: CSA lane

Add ratio-4 overlap state, separate indexer compression, FP4 indexer path,
full-history score, exact top-512, and mixed sparse attention.

Gate:

- Overlap state matches across positions 3/4, 7/8, and snapshot restore.
- Indexer scores and selected IDs match before comparing attention output.
- When compressed rows are at most 512, dense-all and selected paths agree.
- Decode and batched prefill match the CPU oracle independently.

### S5: full 0731 target generation

Connect all 43 layers, the native tokenizer, official prompt encoder subset,
sampling, and stop handling. DSpark remains disabled and the compression-ratio
tail remains opaque metadata.

Gate:

- Exact greedy 128-token continuation matches the pinned llama.cpp oracle.
- Intermediate bisect can isolate any divergence to one layer and operation.
- Repeated runs are deterministic under the same host-validity contract used
  by Qwen benchmarks.
- IQ3 peak resident plus scratch memory passes M4 Max admission with explicit
  system headroom; no reliance on swap is allowed for the resident target.

### S6: Metal performance promotion

Only after S5 exactness:

- Pack intended mixed FP8/BF16 KV and FP4 indexer caches.
- Fuse mHC split/Sinkhorn/collapse, compressor projection/store, shared-KV
  sparse attention, and high-value MoE boundaries.
- Implement real batched prefill; repeated single-token decode is not an
  acceptable prompt path.

Gates:

- Decode throughput is at least current llama.cpp Metal on the same hashes,
  request, context, and memory policy.
- Fresh TTFT and warm decode have named phase attribution.
- Long-context tests at 32K, 128K, and beyond validate both memory slope and
  semantic cache visibility.
- Existing canonical Qwen packets regress by less than 2%.

### S7: DSpark - independent lane

The official 0731 package contains three DSpark stages, but standard llama.cpp
conversion drops all `mtp.*` tensors. DSpark requires a separate artifact or a
converter extension, target hidden capture, context projection, multiple draft
blocks, non-causal draft attention, a low-rank Markov head, and verification.

Open this lane only after target-only generation is correct and measured.

## Deferred lanes

### Custom quant recipes

Do not make DwarfStar recipe parity a prerequisite. Reopen only if a matched
quality study shows at least a 3% relevant-quality gain over standard Unsloth
quants, or reaches a resident size class unavailable through standard K/IQ
formats.

### SSD expert streaming

Keep streaming architecture-neutral and independent. Reopen for a workload
that cannot fit resident memory and only if measured decode loss is below 15%
at a useful cache budget. Do not use streaming to mask excessive DS4 scratch
or cache allocations.

### Project rename

The engine identity may eventually broaden beyond Qwen, but renaming packages,
environment variables, cache ABIs, and tools during architecture bring-up adds
noise without reducing technical risk. Revisit after S5.

## Risk register

| Risk | Response |
|---|---|
| IQ2_S routed gate/up has no production grouped path | Make it an explicit S2 deliverable and test against scalar dequantization |
| Existing Qwen session becomes branch-heavy | Keep DS4 model/session/forward types separate |
| CSA indexer dominates decode | Attribute full-history score and top-k before changing tile shapes |
| Generic Metal fallback hides memory blowups | Require explicit scratch accounting and resident-memory gates |
| Activation QAT differs across references | Freeze operation vectors and distinguish semantic BF16 from packed-cache promotion |
| No tiny official DS4 checkpoint exists | Build operation fixtures and a synthetic tiny family fixture; do not use passthrough layers as an exactness oracle |
| Prompt template differs across runtimes | Port the official 0731 encoder and compare rendered bytes, not rendered intent |
| Asset revisions drift | Pin all shard hashes and checkpoint metadata in every benchmark packet |
| DS4 variant churn | Freeze 0731; add another descriptor only for a measured quality gain and material schema change |

## Immediate next work

1. Add a pinned llama.cpp CPU fixture harness and extend the DwarfStar scalar
   harness until every S1 operation family has directly executed local evidence.
2. Export a deterministic schema/quant census from `DeepSeekV4Model` and pin it
   beside the target hashes.
3. Add official tokenizer and prompt-encoding fixtures beyond raw BPE.
4. Build the S2 local-layer scalar adapter over real dequantized layer-0 weights
   and the typed cache transaction/view contract
   without exposing generation dispatch.
5. Define the packed Metal cache row/slab ownership plan before any long-context
   allocation or durable DS4 snapshot ABI is introduced.
