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

Full-model differentials also pin execution shape. The qwen-owned session is a
singleton decode loop, so a continuing llama.cpp oracle must evaluate every
prompt token separately rather than batch-prefill the prefix. The HCA audit
found that `llm --snapshot` had ignored `--prompt-geometry`; `llm` commit
`e07acac20fcd2ee0faca90aa91078ff142724d63` fixes that path and records
`prompt_decode_mode` in snapshot metadata. Every position-1-and-later fixture
now pins `singleton`; position zero separately pins direct token injection.
At position 9, qwen versus the corrected b10222 singleton oracle gives cosine
0.999999975 and relative RMS 0.000261269. The former batched-prefix vector is
therefore not a valid decode oracle and is not retained as evidence.

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

The fixture generator has three deliberately distinguished sources of evidence:

- NumPy equation transcriptions record the pinned vLLM, SGLang, llama.cpp, and
  DwarfStar source locations. These are reproducible cross-language vectors,
  not claims that those runtimes were executed.
- `scripts/reference/dsv4_dwarfstar_oracle.c` is compiled against the pinned
  DwarfStar checkout and directly executes selected scalar mHC, RoPE,
  compressor-pooling, indexer-QAT, routing-helper, and SwiGLU paths. It also
  executes complete synthetic F16-projection compressor transitions at
  production output widths: ratio-4 attention and indexer through position 8,
  plus ratio-128 attention through position 256. Seventeen-wide projections
  exercise row stride, vector-plus-tail accumulation, and non-exact F16 weight
  and APE rounding without claiming the production 4,096-wide projection cost.
  Every post-token KV/score state is checked by a canonical F32 digest, and all
  six boundary outputs cover RMSNorm, compressed RoPE, attention E4M3, or
  indexer Hadamard/E2M1 QAT as applicable. It does not yet execute complete mHC
  projection/gating or a standard-GGUF model. Revision, tree, or source drift
  fails generation; harness drift fails the pinned fixture `--check` gate.
- `scripts/reference/dsv4_llama_cpp_cpu_oracle.cpp` builds against pinned,
  CPU-only static baseline GGML and directly executes `ggml_dsv4_hc_comb`,
  `ggml_dsv4_hc_pre`, and `ggml_dsv4_hc_post` as one three-token reference
  graph. Its self-contained vectors exercise nontrivial token strides and the
  exact `[destination, source, token]` combination layout. This is executable
  fused-primitive evidence, not yet a claim that llama.cpp's complete mHC
  projection/gating composition ran.

Run `uv run scripts/reference/generate_dsv4_oracle.py` from the repository root
with DwarfStar at `~/code/ds4` and llama.cpp at `~/code/llama.cpp`, or set
`DSV4_DWARFSTAR_DIR` and `DSV4_LLAMA_CPP_DIR`. Python 3.14 and NumPy 2.5.1 are
pinned by the script metadata; the llama.cpp lane additionally requires CMake
and a C++17 compiler. Use the same command with `--check` for a captured-
toolchain byte-for-byte drift gate; the ignored
`fixture_regeneration_has_no_drift` test exposes that external-checkout gate to
the Rust harness. Revisions, tracked-worktree state, semantic source hashes,
harness/build hashes, build toolchain, one-thread reference mode, and tensor
shapes are all recorded or enforced. CMake, C/C++ compilers, effective base and
release flags, linker flags, and system/SDK context are captured but not
hermetically pinned; the runtime math library is not independently pinned.
Cross-toolchain `expf` last-bit drift therefore requires explicit numerical
review rather than a silent fixture update.

S1 is not promoted yet because direct executable coverage is incomplete, not
because a remote accelerator is missing. The current DwarfStar and llama.cpp
harnesses directly cover selected scalar helpers, all three fused mHC CPU
primitives, and complete token-by-token compressor transitions. Shared-KV
attention/output, complete mHC projection/gating, and indexer scoring still
rely on transparent NumPy transcriptions. The next evidence step is complete
DwarfStar mHC composition, followed by locally executable shared-KV/indexer
seams. All use small synthetic tensors and require neither a model download nor
DwarfStar's custom quant metadata.

The DwarfStar transition differential compares `compressor_decode_one` at its
function boundary. Attention rows therefore include NoPE E4M3 simulation while
the RoPE tail remains F32; DwarfStar's caller-side F16 cache append is outside
that boundary. The separate typed-cache tests retain the intended mixed
FP8/BF16 storage contract. State digests canonicalize signed zero and map
DwarfStar's finite `-1e30` empty-score sentinel to negative infinity so they
compare semantics rather than representation-only sentinel choices.

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
plan. A resumable production session must use preallocated deltas or shadow
banks to preserve rollback. The current Metal correctness session instead sets
a poison bit before its first mutation and is deliberately fail-stop: any
incomplete token makes the session permanently non-retryable rather than
presenting partially committed state as reusable.

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

Status: promoted for sequential positions 0 through 2 on 2026-08-02. The native
`DeepSeekV4MetalResidency` and `DeepSeekV4Session` execute the local branch of
all 43 layers, not a truncated model: embedding, both mHC surrounds, shared-KV
attention, routed and shared MoE, final HC collapse, output norm, and the full
129,280-row vocabulary head remain qwen-owned Rust and Metal execution.

The continuing-session oracle also corrected two assumptions that position zero
could not distinguish. Effective llama.cpp b10222 uses adjacent-pair partial
RoPE for `deepseek4`, and `llama_core` leaves its cache at the llama.cpp default
F16 type. The native session therefore stores partially rotated raw KV as F16;
the mixed FP8-NoPE/BF16-RoPE operation oracle remains a separate packed-cache
contract rather than being silently substituted for the maintained F16 oracle.

Every CSA attention/indexer and HCA attention compressor projection now runs
from position zero and writes its F32 KV plus APE-adjusted score frontier. No
compressed row is published early. This made position 3 implementable from
retained state without replay and closed S2 without speculating about visibility
at the first compression boundary.

Gate:

- Token 35 at position 0 matches b10222 with argmax 201, cosine 0.999999999,
  relative RMS 0.000049306, mean absolute error 0.000145020, and max absolute
  error 0.001245499 across all logits.
- Exact token sequence `[35, 201]` matches the position-1 b10222 oracle with
  argmax 200, cosine 0.999999998, relative RMS 0.000065605, mean absolute error
  0.000233142, and max absolute error 0.001331806 across all logits.
- Extending the exact sequence to `[35, 201, 200]` matches position 2 with
  argmax 200, cosine 0.999999990, relative RMS 0.000155619, mean absolute error
  0.000689060, and max absolute error 0.003203392.
- Focused release differentials cover adjacent local and scaled YaRN RoPE,
  inverse RoPE, ordered F32-to-F16 same-token cache insertion, sink attention,
  and ratio-4 frontier lane plus APE semantics.
- No delegated llama.cpp runtime participates. Its full-vocabulary F32 vectors
  are checked-in numerical evidence only.

The DS4 model and forward loop remain family-isolated. The stricter shared
`get_rows` boundary did expose a pre-existing Qwen representation mismatch:
base decode, MTP, and DFlash stored I32 payloads in F32-tagged tensors, and a
fallible encode could release an unfinished Metal encoder. Commit `1487fbc`
repairs that shared seam before S4 promotion with typed I32 IDs/argmax outputs,
I32 subviews, dtype-pinned prefill scratch plans, and idempotent encoder cleanup
on early return or unwind. Focused release tests cover the original SIGTRAP and
the actual 0.8B DFlash packed verifier. Broad performance recertification is
still deferred because no dispatch or numerical kernel changed.

### S3: CSA lane

Status: dense-all baseline promoted through positions 3, 4, 7, and 8 on
2026-08-02. Before implementation, exact-token b10222 full-vocabulary fixtures
at positions 3 and 4 arbitrated same-token visibility and first continuation.
The second-boundary fixtures at positions 7 and 8 then made overlap roll a live
model differential rather than an operation-only claim. Every fixture has two
byte-identical fresh-session captures, full shard hashes, and pinned producer
commits.

The native lane now performs ratio-4 two-branch pooling, learned RMSNorm,
block-start adjacent-pair RoPE, F16 compressed-row publication, overlap roll,
and one denominator-only-sink softmax over local raw rows plus every completed
compressed row. The parallel indexer compressor publishes its normalized
Hadamard row, but index scoring and top-512 selection remain deferred while the
history is below 512 rows and dense-all is definitionally equivalent. The
session now continues through the independently promoted first HCA row and
its immediate position-128 continuation, then fails closed before position
129. The second ratio-128 publication remains structurally guarded at position
255 for its future differential.

Gate:

- Counterfactual sequence `[35, 201, 200, 34]` matches position 3 with argmax
  262, cosine 0.999999988, and relative RMS 0.000165250; extending with token
  262 matches position 4 with argmax 63,325, cosine 0.999999978, and relative
  RMS 0.000210703.
- Exact sequence `[35, 201, 200, 34, 35, 201, 200, 34]` matches the second
  boundary at position 7 with argmax 35, cosine 0.999999985, and relative RMS
  0.000183200; extending with token 35 matches position 8 with argmax 201,
  cosine 0.999999979, and relative RMS 0.000228075.
- Focused release operation gates match ratio-4 frontier publication and roll,
  normalized Hadamard-128, and same-token dense CSA against independent CPU
  oracles.
- Snapshot restore, dense-all versus selected-path equivalence at scale, and
  indexer score/ID differentials remain promotion requirements for sparse CSA;
  they are not blockers for the validated dense-all baseline.

### S4: HCA lane

Status: first-boundary decode slice promoted through positions 126, 127, and
128 on 2026-08-02. All 20 HCA layers now use the already-retained ratio-128
frontier to pool, RMS-normalize, apply block-start adjacent-pair RoPE, publish
an F16 compressed row, and include that row in the same-token softmax over the
128-row local window plus dense compressed history. Position 128 proves that
the row remains visible on continuation. The session fails closed at position
255 before a second HCA row can be published.

The long-prefix differential uses a different numerical gate from positions
0-8. Singleton-versus-singleton drift accumulates before any HCA row exists:
relative RMS is 0.000261269 at position 9 and 0.045916357 by position 126.
Applying the short-prefix 0.002 threshold at position 127 would therefore
misattribute inherited reduction drift to HCA. Promotion instead combines an
exact integrated operation gate, preserved full-model argmaxes, an explicit
boundary-discontinuity bound, and recovery on the immediate continuation.

Gate:

- Production 512-wide compressor-row coverage writes two complete 128-token
  frontiers, proves no early publication through positions 126 and 254, and
  matches the CPU oracle after pooling, RMSNorm, and F16 publication at both
  boundaries. Row 1 starts at position 128, so its position-255 differential
  also exercises non-identity block-start adjacent-pair YaRN RoPE.
- In one retained native session, positions 126, 127, and 128 preserve b10222
  argmaxes 34, 35, and 201. Their cosine / relative-RMS pairs are
  0.999086380 / 0.045916357, 0.998497359 / 0.063801241, and
  0.998833369 / 0.048343154 respectively.
- A controlled ablation that published but withheld the first HCA row worsened
  relative RMS to 0.075390582 at position 127 and 0.128496492 at position 128.
  Same-token consumption is therefore both directionally correct and necessary
  for the continuation rather than an inert implementation detail.

S4 remains open for the second boundary at positions 255/256, named
full-layer intermediate states, and batched prefill. Those are extension and
prefill gates; they do not block a bounded singleton-decode generation slice.

### S5: full 0731 target generation

Status: bounded raw CLI slice promoted on 2026-08-02. The release `qwen`
binary now opens the split GGUF once, dispatches `deepseek4` outside the Qwen
model binder, constructs the native JoyAI tokenizer and strict Metal residency,
forwards every raw prompt token, and reuses the common sampler and
producer-declared stop-token contract. Generated pieces are written as exact
token bytes without a synthetic stdout newline. Chat templates, batched
requests, prompt lookup, prefill controls, and prefix/checkpoint caches fail
before residency rather than being silently ignored, including when a
value-bearing option is explicitly supplied at its Qwen default.

The CLI reserves `prompt_tokens + max_generated_tokens - 1` forwards before
Metal residency or any token execution. This mirrors the generator's
pending-final-token semantics and guarantees an accepted request cannot
partially stream beyond the retained-session evidence through position 128.
The 103 GB split GGUF is already virtually mapped for family detection at that
point; residency remains untouched on rejection. The promoted capacity is
exported by the session implementation, so frontend and executor cannot drift
onto independent magic limits. The shared one-open handoff also passed a
release one-token Qwen 3.5 0.8B smoke, preserving the existing family path.

Expose the already-connected 43-layer session through first-class generation
dispatch, then connect the native tokenizer, official prompt encoder subset,
sampling, and stop handling. A bounded raw prompt path comes first so frontend
work does not postpone executable inference. DSpark remains disabled and the
compression-ratio tail remains opaque metadata.

Gate:

- Raw prompt `A` (token 35) generated exact greedy IDs `[201, 200, 200, 1778]`
  through the release `qwen` binary, matching the pinned b10222 singleton chain
  and crossing the first CSA publication at position 3. The first cold forward,
  including lazy pipeline construction, took 36.8 seconds; the three retained
  decode transitions ran at 14.34 tokens/second. This is executable evidence,
  not a delegated llama.cpp runner.
- An independent fresh-process repeat preserved all four IDs byte-for-byte;
  with the system Metal pipeline cache populated, prompt forward fell to 0.659
  seconds and retained decode rose to 15.68 tokens/second.
- Exact greedy continuation token IDs through the first HCA boundary match the
  pinned singleton llama.cpp oracle.
- Intermediate bisect can isolate any divergence to one layer and operation.
- Repeated runs are deterministic under the same host-validity contract used
  by Qwen benchmarks.
- IQ3 peak resident plus scratch memory passes M4 Max admission with explicit
  system headroom; no reliance on swap is allowed for the resident target.

S5 remains open for a CLI continuation through the first HCA boundary, the
official prompt encoder subset, long-prefix repeatability, and explicit peak
memory admission. The bounded raw slice is sufficient to establish first-class
native inference without weakening those promotion gates.

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

1. Extend the bounded CLI differential from the exact position-3 greedy chain
   through the retained position-127 HCA publication and position-128
   continuation; keep the position-129 evidence guard visible rather than
   hiding it with replay or delegation.
2. Port and fixture the minimum official 0731 prompt-encoder path required for
   ordinary system/user/assistant turns; keep raw mode available as the exact
   reproducibility interface.
3. Capture positions 129 and 254 as continuation checkpoints before requesting
   promotion to the second HCA boundary.
4. Capture positions 255 and 256 with singleton b10222 execution, promote the
   second HCA publication, and extend the dense HCA history gate.
5. Implement sparse CSA index scoring/top-512 selection before compressed
   history exceeds 512 rows, and extend slab ownership before the current CSA
   row-256 allocation guard at position 1027.
