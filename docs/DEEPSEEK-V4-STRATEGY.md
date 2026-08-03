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
quick-task behavior. The ordinary chat subset is now ported from two
independently maintained release-derived implementations: vLLM's V4 encoder and
SGLang's encoder, whose source explicitly records release provenance. Both
produce byte-identical user-only, system/user, and multi-turn Unicode prompts.

The promoted subset accepts an optional leading system turn, alternating plain
user/assistant history, and a final user turn. It emits BOS explicitly,
attaches `<｜Assistant｜></think>` to each user turn, and terminates historical
assistant content with EOS. Unknown per-message or wrapped top-level fields,
developer/tool roles, reasoning records, non-alternating history, and Qwen
thinking-policy flags fail closed. `--messages-max` applies after structural
JSON deserialization, then role/order/extra-field semantics validate only the
retained prefix; wrapper-level semantics are always rejected. Raw mode remains
the exact reproduction interface. Rich V4 tools, reasoning, latest-reminder,
response-format, and quick-task semantics remain separate fixture-gated
extensions rather than approximations of the official encoder.

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

### Evidence cadence

Correctness evidence is an explicit DAG, not an instruction to replay every
ancestor after every edit:

- Sub-second gates own layout, scalar semantics, cache bounds, state-machine
  rejection, manifest identity, and allocation arithmetic.
- Tens-of-seconds gates own retained packed transitions, sparse layer probes,
  short packed-versus-singleton schedule checks, and one exact endpoint from an
  already-certified prefix.
- Fresh long-prefix native runs, cold independent-oracle captures, broad model
  matrices, and performance packets are rare promotion audits measured in
  minutes. They do not belong in an ordinary kernel edit loop.

An immutable external oracle fixture is not invalidated by a native
implementation edit. Recapture it only when model bytes, prompt tokens,
semantics, or oracle revisions change, or when evidence shows the fixture is
defective. Fresh-from-zero native replay reopens for changes to tokenizer or
GGUF interpretation, position/RoPE semantics, cache mutation or addressing,
packed scheduling, quant decoding across the prefix, and future snapshot
restore. Otherwise operation evidence plus a retained strategic endpoint is the
shortest responsible gate.

The first typed Metal causal-state snapshot layer is now live. Each session owns
its exact committed token transcript, pre-reserved before inference, and may
capture either ready phase. The snapshot binds a constructor-supplied strong
model-content ID, an exhaustive domain-separated digest of the Flash config and
explicit causal/numerics/storage ABI, and the exact prefix. A poisoned session
cannot capture or restore.

The payload contains only persistent state: chronological visible F16 raw rows,
all F32 CSA-attention/CSA-indexer/HCA compressor state bits, and visible F16
attention/indexer publications. Scratch and observations are excluded. Restore
builds deterministic full physical images and validates identity, geometry,
prefix, numerical phase, overlap lanes, arena lengths, and whole-state digest
before poisoning the destination; success always becomes ready without an
observation. This preserves the distinction between “correct from restored
position” and “this build reproduced the prefix.”

The durable wire layer derives every section length before allocation, uses a
256-byte versioned header at a 16 KiB payload boundary, requires zero padding and
reserved fields, and closes the record with a BLAKE3 digest in addition to the
typed prefix and causal digests. Its hard limit is 64 MiB; the largest currently
admitted position-2049 record is 31,957,028 bytes. Decode temporarily retains
byte sections while constructing typed arenas, so peak host allocation is about
twice the payload. Request/sampler state remains deliberately outside this
model-session checkpoint.

The explicit `--deepseek-v4-snapshot PATH` surface derives a strong cached
content root over every complete retained GGUF shard. The snapshot parent must
be current-user-owned and not group/world-writable; the versioned identity cache
is a current-user `0700` directory. Snapshot files are staged as `0600`, synced,
decoded again, and published create-only by hard link. Existing paths are
accepted only for the same model/ABI/prefix/causal digest. Ancestor components
remain caller-trusted. A transient or crash-retained same-inode staging alias is
valid; link-only count/ctime changes are not integrity, while file type, private
mode, size, descriptor identity, and the complete digest are. A restore
record and exact token prefix are validated before Metal residency, then at
least one uncached endpoint token rebuilds the observation that causal snapshots
intentionally omit.

Gate:

- At the first CSA boundary, the 12,409,104-byte causal payload encodes as a
  12,425,520-byte record. Decode and backward restore preserve all 129,280
  continuation logits and the post-continuation state bit-for-bit, while the
  independent b10222 gate remains argmax 63,325 at cosine 0.999999958 / relative
  RMS 0.000290112. The complete live packet takes 1.534-1.539 seconds.
- A fresh CLI process hashes 102,999,888,416 ordered shard bytes in 4,840.5 ms,
  advances the first 1,024 tokens of the certified 1,025-token prompt, and
  publishes a 24,907,808-byte record. Prompt execution takes 21,746.0 ms and
  generates oracle ID 201.
- A second process gets a zero-byte identity-cache hit, validates the record in
  83.7 ms before residency, restores 1,024 tokens, and executes only the endpoint
  token. Prompt execution takes 922.2 ms, a 23.6x feedback-loop reduction, and
  again generates ID 201.
- The focused restored-endpoint gate completes in 1.035-1.117 seconds and
  reproduces the established full-vector SHA-256
  `73d357295a7821607869764af42aaafc845e1764afe8c23a0aab2e5f570a7956`,
  argmax 201, cosine 0.998133285, and relative RMS 0.061869968 against b10222.
  Its causal snapshot digest is
  `c4173d31f6ddb8bf3cb2a2c2eb6310de4c9ccdacc830d0c032489bb3db6fa77d`.
- Doubling physical compressed-history capacity preserves the 256-row slab ABI:
  the existing position-1024 record still validates before residency, restores
  into a zeroed 512-row image, and reproduces the same endpoint hash and oracle
  metrics. Capacity is not allowed to masquerade as a causal-format change.
- Eight retained chunks advance that old checkpoint through the second slab to
  position 2048 in 21.498 seconds. The resulting 31,940,608-byte causal payload
  publishes as a 31,957,024-byte record with digest
  `a91f82475c5ecc18ae91fd607224a227c275b73745badfbf9451a63acc32e246`.
  Restoring it and executing the uncached endpoint takes 1.141 seconds.

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

Status: dense-all baseline promoted through all 512 rows and position 2048 on
2026-08-02. Before initial implementation, exact-token b10222 full-vocabulary
fixtures at positions 3 and 4 arbitrated same-token visibility and first
continuation.
The second-boundary fixtures at positions 7 and 8 then made overlap roll a live
model differential rather than an operation-only claim. Every fixture has two
byte-identical fresh-session captures, full shard hashes, and pinned producer
commits.

The native lane now performs ratio-4 two-branch pooling, learned RMSNorm,
block-start adjacent-pair RoPE, F16 compressed-row publication, overlap roll,
and one denominator-only-sink softmax over local raw rows plus every completed
compressed row. The parallel indexer compressor publishes its normalized
Hadamard row, but index scoring and top-512 selection remain deferred while the
history has at most 512 rows and dense-all is definitionally equivalent. The
session now continues through both 256-row compressed-history slabs and the
position-2048 continuation, then fails closed at position 2049. Position 2051,
which would publish unallocated row 512, retains a more specific slab guard.

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

Status: HCA rows 0-15 and the continuation through position 2048 promoted on
2026-08-02. All 20 HCA layers use the retained ratio-128 frontier to pool,
RMS-normalize, apply block-start adjacent-pair RoPE, publish F16 compressed
rows, and include every published row in the same-token softmax over the
128-row local window plus dense compressed history. The first four rows retain
their named full-model boundary evidence. Later rows use generalized
production-width operation properties and strategic full-model endpoints
rather than a mechanical fixture campaign at every boundary. Position 2047
publishes HCA row 15 and CSA row 511; position 2048 proves immediate wrapped
continuation at the dense-all limit.

The long-prefix differential uses a different numerical gate from positions
0-8. Singleton-versus-singleton drift accumulates before any HCA row exists:
relative RMS is 0.000261269 at position 9 and 0.045916357 by position 126.
Applying the short-prefix 0.002 threshold at position 127 would therefore
misattribute inherited reduction drift to HCA. Promotion instead combines an
exact integrated operation gate, preserved full-model argmaxes, an explicit
boundary-discontinuity bound, and recovery on the immediate continuation.

Gate:

- Production 512-wide compressor-row coverage writes four complete 128-token
  frontiers, proves no early publication through positions 126, 254, 382, and
  510, and matches the CPU oracle after pooling, RMSNorm, and F16 publication
  at all four boundaries. Rows 1-3 start at positions 128, 256, and 384,
  exercising non-identity block-start adjacent-pair YaRN RoPE.
- In one retained native session, positions 126, 127, and 128 preserve b10222
  argmaxes 34, 35, and 201. Their cosine / relative-RMS pairs are
  0.999086380 / 0.045916357, 0.998497359 / 0.063801241, and
  0.998833369 / 0.048343154 respectively.
- A controlled ablation that published but withheld the first HCA row worsened
  relative RMS to 0.075390582 at position 127 and 0.128496492 at position 128.
  Same-token consumption is therefore both directionally correct and necessary
  for the continuation rather than an inert implementation detail.
- Fresh-session b10222 captures at positions 129 and 254 are byte-identical
  across repeats. Their vector hashes are
  `8cb27e714a0ed478b9d19d08d5def339debff9ed7f73c4bcfb3642dbdd4eeab6`
  and `0b959bcd97f045e4515190b035bed296002057fc587174d15b6aa6bd2c163159`.
- The same retained native session preserves argmaxes 200 and 34 at positions
  129 and 254. Their cosine / relative-RMS pairs are
  0.997522039 / 0.070976029 and 0.998094070 / 0.061915705. Position 129 stays
  within the explicit first-boundary allowance; position 254 recovers on both
  measures rather than accumulating further drift.
- Fresh-session b10222 captures at positions 255 and 256 are byte-identical
  across repeats. Their vector hashes are
  `7b93e37f0da402e0dd9fc1a40d9ed99921adebfc0bed7667cbd5f15a880f93a6`
  and `57d23cc00c74a8cb001e8edd07a13b4c07903be88d1fb2801632edf5cd5ccc4c`.
  Their manifests pin the expanded token geometry, all four model shards, and
  the exact `llm`, `llama_core`, `llama-cpp-rs`, and b10222 producer revisions.
- The retained native session preserves argmaxes 35 and 201 at positions 255
  and 256. Their cosine / relative-RMS pairs are
  0.998899999 / 0.050519499 and 0.998702828 / 0.053781075. Both improve on the
  position-254 pre-boundary control in both measures, so publishing the second
  row introduces no boundary discontinuity under the inherited-drift gate. All
  three positions additionally require cosine at least 0.997 and relative RMS
  at most 0.075, preventing correlated drift from satisfying only relational
  checks.
- A four-row Metal attention gate matches the CPU oracle and materially
  differs from both prior-three-row and local-only ablations, proving the newest
  published row is consumed. A second production-width falsifier uses tagged
  wrapped 128-row raw caches at positions 382-384 and 510-512 and independently
  matches CPU attention for HCA row counts 2/3/3 and 3/4/4 plus CSA counts
  95/96/96 and 127/128/128. This closes the ring-order, compressed-count, and
  same-token visibility alternatives.
- Fresh-session b10222 captures at positions 257, 382, 383, and 384 are
  byte-identical across repeats. Their vector hashes are
  `234d6168ae338832cb694b51e8042bcf40ec129579b49a4d7b90c97e4aa5aa6c`,
  `d26400242d98ed8cfe5a80f79f4e23343576aa657bee06140968cebf68a6b6a4`,
  `6a72ff9fd4ed33b5fc04cfcc41bf59cc7366c226412c1930a21511d35afb921d`,
  and `310e38754af708713a92a8b6aba2be7c77f2e678b8a6f91874ede867f9d4611e`.
- The retained session preserves b10222 argmaxes 200, 34, 35, and 201 at those
  positions. Their cosine / relative-RMS pairs are
  0.997692051 / 0.070415212, 0.990839296 / 0.141674404,
  0.993863232 / 0.135474860, and 0.998019910 / 0.065362616. Positions 382/383
  expose larger inherited interval drift, so they use explicit 0.990 / 0.15
  containment rather than the established 0.997 / 0.075 endpoint gate. The
  publication must improve or preserve both measures from position 382, and
  position 384 must strictly recover both and satisfy the established gate.
- Before increasing oracle context capacity, position 384 was recaptured with
  context 1,024 and remained byte-identical to the context-512 vector. A pinned
  sidecar records the exact command, producer and shard revisions, both context
  sizes, and shared vector hash. The fourth-interval fixtures therefore change
  capacity only after an executable invariance check.
- Fresh-session b10222 captures at positions 385, 510, 511, and 512 are
  byte-identical across repeats. Their vector hashes are
  `9ddcba113bb3a7f082cbf7b0fc6a78c23396f2dc86aebb00668052509439906c`,
  `682683248e39eae1051f2de392f3f3052032b89e6b719c43a0e98202ec1cb018`,
  `f90eea6537979dd51831216e0db3949c5869aabd75e505b627f05abf058f3340`,
  and `56f13995d0f9e0042a81e2015878164bd3d23a14515e093023f07edd87677376`.
- The retained session preserves b10222 argmaxes 200, 34, 35, and 201 at those
  positions. Their cosine / relative-RMS pairs are
  0.996155765 / 0.087700659, 0.992338381 / 0.124623164,
  0.995358983 / 0.106502983, and 0.997749833 / 0.071887024. Positions 385,
  510, and 511 use the existing 0.990 / 0.15 interval containment; publication
  must improve or preserve both measures, and position 512 must strictly
  recover both and satisfy the established 0.997 / 0.075 endpoint gate.
- A production-width ratio-128 property executes all 2,049 positions through
  row 15. It rejects every early publication, compares every emitted row with
  the independent CPU compressor after F16 storage, preserves all older rows,
  validates block-start YaRN positions 0 through 1,920, and leaves the history
  unchanged at position 2048.
- Serial publication-to-attention gates at positions 639, 1023, and 2047
  materially distinguish the newest row from a prior-count ablation. Tagged
  wrapped-ring gates at 638/639/640 and 1022/1023/1024 independently match HCA
  counts 4/5/5 and 7/8/8 plus CSA counts 159/160/160 and 255/256/256. The same gate now
  covers 1026/1027/1028 and 2046/2047/2048, including HCA counts 15/16/16 and
  CSA counts 511/512/512 at the second-slab endpoint.
- Production-width ratio-4 attention and Hadamard-indexer properties publish
  rows 0-511, match the overlap-roll CPU state through position 2048, preserve
  both complete slabs on continuation, and synchronously reject row 512. Dense
  attention accepts count 512 and rejects 513. Packed retained attention also
  matches the CPU oracle with 128 raw plus 512 compressed rows, exercising its
  full 640-thread score-production geometry.
- Before increasing oracle capacity, the pinned position-512 b10222 vector was
  recaptured at context 2,048 and remained byte-identical to context 1,024 at
  SHA-256 `56f13995d0f9e0042a81e2015878164bd3d23a14515e093023f07edd87677376`.
- Two fresh b10222 position-1024 captures were byte-identical at SHA-256
  `6d6360c975b654d18408f262541a363a695087db4fef7722b31a5ecdf70dd03e`.
  The native packed-prefix/singleton session preserved argmax 201 at cosine
  0.998247840 and relative RMS 0.060068528.
- Before increasing oracle capacity again, the position-1024 b10222 vector was
  recaptured at context 4,096 and remained byte-identical to its context-2,048
  vector. Two fresh position-2048 captures were then byte-identical at SHA-256
  `a3a134646f01f6f7568006079a14840ddc1aa8fa29218ffab9d439938e0947ff`.
- Native execution from the certified position-1024 checkpoint preserves the
  position-2048 oracle argmax 201 at cosine 0.998054535 and relative RMS
  0.063223233. Both the uninterrupted extension and a fresh durable restore
  produce full-vector SHA-256
  `13cb8a323341f3f23bcb98ddfb8c257b5125411065d14e1b40c5a75fa19ed53c`.
- A valid-token position-2049 call rejects before mutation, keeps
  `next_position == 2049`, and preserves every completed position-2048 logit
  bit. Position 2051 reports the unallocated CSA row-512 boundary before the
  general continuation limit. Full-logit agreement remains limited to named
  fixture positions.

The next semantic ownership boundary is CSA row 512 at position 2051: history
would exceed top-k 512, so sparse index scoring and stable selection must replace
dense-all before that token is admitted. The next HCA publication is later, at
position 2175. Named full-layer intermediate states remain useful for an actual
divergence, not as a mandatory campaign at every 128-token boundary.

### S5: full 0731 target generation

Status: bounded raw CLI and resident-memory admission slices promoted on
2026-08-02. The release `qwen` binary now opens the split GGUF once, dispatches
`deepseek4` outside the Qwen model binder, constructs the native JoyAI tokenizer
and strict Metal residency, forwards every raw prompt token, and reuses the
common sampler and producer-declared stop-token contract. Generated pieces are
written as exact token bytes without a synthetic stdout newline. Chat
templates, batched requests, prompt lookup, prefill controls, and
prefix/checkpoint caches fail before residency rather than being silently
ignored, including when a value-bearing option is explicitly supplied at its
Qwen default.

The CLI reserves `prompt_tokens + max_generated_tokens - 1` forwards before
Metal residency or any token execution. This mirrors the generator's
pending-final-token semantics and guarantees an accepted request cannot
partially stream beyond the retained-session evidence through position 2048.
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
- Raw prefix `"A\n\t@" * 32` tokenizes exactly to
  `[35, 201, 200, 34] * 32`. Two independent release CLI sessions reserved all
  then-promoted 129 forwards and generated `[35, 201]`, matching the pinned
  b10222 argmaxes after the position-127 HCA publication and its position-128
  continuation. Sequential 128-token prompt execution took 35.0 and 35.2
  seconds; the final transitions took 0.486 and 0.484 seconds.
- The earlier 257-forward gate used a repeated-pattern prompt to cross the
  second HCA publication, consume position 256, and generate oracle token 201 in
  100.9 seconds. Its then-next transition rejected before Metal residency.
  These are correctness observations for the unoptimized singleton prompt path.
- The subsequent 513-forward gate crossed the fourth HCA publication, consumed
  position 512, and generated oracle token 201 in 248.6 seconds. Its
  then-unsupported 514-forward request rejected before Metal residency.
- The first-slab gate accepted exactly 1,025 forwards. A native request
  publishes HCA row 7 and CSA row 255 at position 1023 and matches the
  position-1024 b10222 endpoint through retained layer-major chunks at
  0.998133285 cosine / 0.061869968 relative RMS. Its then-terminal rejection
  also established the fail-before-residency and fail-before-mutation contract.
- The exported budget now accepts exactly 2,049 forwards. The release CLI
  validates and restores the certified 2,048-token checkpoint before residency,
  executes the uncached position-2048 endpoint in 982.3 ms, and generates
  oracle ID 201 with `prefill_mode=causal_snapshot_restore`. End-to-end process
  time is 1.56 seconds; a 2,050-forward request rejects before residency.
- Ordinary 0731 messages match vLLM and SGLang byte-for-byte. On the Flash
  vocabulary, user-only, system/user, and multi-turn Unicode fixtures encode to
  exact token sequences of 5, 8, and 16 tokens. A live system/user request
  rendered to 11 tokens and qwen generated IDs `[46382, 15697, 11898, 1]`
  (`ghiaccioli` then EOS); the b10222 `llm` singleton oracle produced the same
  first-token argmax and complete greedy text.
- Intermediate bisect can isolate any divergence to one layer and operation.
- Repeated runs are deterministic under the same host-validity contract used
  by Qwen benchmarks.
- Allocation-free planning inventories 7 resident buffers (3 retained no-copy
  windows and 4 final-page copies) at 102,994,624,512 priced-upper bytes and
  521 unique session buffers at 166,877,972 logical / 171,212,800 priced-upper
  bytes. The session total includes the complete physical 128-token packed
  scratch and its 131,072-byte pre-chunk raw-ring snapshot rather than charging
  either to reserve. The complete priced upper bound is 103,165,837,312 bytes;
  a 536,870,912-byte dynamic reserve makes the admission requirement
  103,702,708,224 bytes.
- The load plan freezes configuration plus every descriptor's name, shape,
  dtype, shard, offset, and byte length. Realization revalidates those values,
  fallback policy, all view/alias/window geometry, and a deterministic planner
  rebuild before refreshing admission at the last allocation-free point.
- On the target M4 Max, Metal reported 126,701,535,232 recommended bytes and a
  475,136-byte baseline, giving 126,701,060,096 bytes of working-set headroom.
  The process signal was `Some(0)`, the established omitted-limit convention,
  so the explicit reason was `admitted_process_budget_omitted`.
- Live phase reconciliation observed 102,994,608,128 residency bytes and
  103,161,495,552 cumulative bytes after session construction and after the
  first forward. This remains below the 103,702,708,224-byte first-forward
  gate. Residency and session must fit their priced inventories without the
  reserve; only the first-forward endpoint gate may use the reserve. Residency
  is reconciled inside realization, before an unaccounted resident handle can
  be returned.
- A release `qwen -p A -n 1` run generated oracle ID 201 in 0.716 seconds of
  prompt execution. `vm_stat` pageouts, swapins, and swapouts were unchanged;
  `vm.swapusage` remained 1,825.94 MiB before and after; and `/usr/bin/time`
  reported zero swaps. This proves no incremental swap in that run, not that
  the host began swap-free.

The pricing contract is validated on this Apple M4 Max using shared-buffer
`heapBufferSizeAndAlign`, rounded again to the 16 KiB host-page allocation
granularity, plus the current no-copy mapping behavior. The packed scratch made
the extra page rounding necessary: size/alignment pricing alone undercounted a
live session by 8,940 bytes and was rejected before promotion. This remains a
platform-specific upper bound, not a portable Metal guarantee. Reconciliation
samples phase endpoints and therefore does not observe a transient allocation
that is created and released between samples. Admission is also not an atomic
reservation against another process allocating on the same device. The 512 MiB
reserve and fail-closed phase checks are the current operational protection for
those limits.

The bounded raw and ordinary-message slices now establish first-class native
inference and resident-memory admission through every full-session differential
promoted so far. Rich tools/reasoning are product extensions, not prerequisites
for the minimum ordinary prompt-encoder gate.

### S6: Metal performance promotion

Status: retained layer-major chunks of 1-128 tokens promoted through the full
2,049-forward session capacity on 2026-08-02. The physical scratch is sized and
admitted once for 128; short chunks use exact prefix views. One-token prompts
remain singleton only when they are the entire request. Longer prompts consume
successive explicit packed chunks, with no singleton replay or hidden fallback.

The packed path is DS4-owned and never calls `forward_token`. Its outer loop is
43 layers over a token matrix. Embedding, mHC function projections and controls,
Q/KV/compressor projections, attention output A/B, router, selected expert
buckets, shared expert, and residual updates execute across the batch. Only
position-dependent RoPE, raw-cache publication, and ratio-4/128 compressor
transitions remain chronologically ordered. CPU routing reads one `[N,256]`
matrix per layer, preserves top-k slot order, and groups selected rows by expert;
the two MXFP4 routed-down outliers retain an explicit row fallback.

Before each retained layer chunk, a bitwise `ushort` copy preserves that
layer's complete F16 raw ring. New rows publish by absolute modulo-128 position.
Packed attention derives each query's absolute raw window and compressed count,
then selects pre-chunk or current storage before modulo addressing. This keeps
future chunk rows from destroying historical keys needed by earlier queries;
append-only compressed rows remain hidden solely by per-query absolute counts.
The dynamic mass slab covers exactly 128 raw rows, 512 compressed rows, and one
denominator slot. The host dispatch uses 640 threads so every possible raw or
compressed score has one producer while the first 512 lanes also write output.

Intermediate teacher-forced chunks use the typed `advance_tokens` transition.
It executes every causal layer state but omits final HC collapse, output norm,
and the 129,280-row vocabulary projection. Successful advancement explicitly
invalidates prior logits and final hidden observations; only the final chunk
can make those observations valid again.

The first packed attention implementation honestly preserved semantics but
reused the singleton kernel, which recomputed every 512-wide score independently
for all 512 output lanes. N=128 took 30.7 seconds and its ordinary continuation
missed the established endpoint gate. Promotion did not relax either result.
Persistent Q8_0 projections now use one token-axis GEMV dispatch with the exact
singleton accumulation body, while a DS4 packed causal kernel computes each
raw/compressed score once in scalar dimension order and shares the resulting
mass across output lanes. Per-query counts preserve same-token publication and
prevent future compressed rows from leaking.

Gate:

- Packed N=1 preserves position-zero argmax 201 at cosine 0.999999548 and
  relative RMS 0.000951217 against the full b10222 vector.
- Packed `[35, 201, 200, 34]` publishes the first CSA row and preserves argmax
  262 at cosine 0.999999903 / relative RMS 0.000448886. Ordinary singleton
  decode from that retained state preserves position-4 argmax 63,325 at
  0.999999958 / 0.000290112.
- Packed `[35, 201, 200, 34] * 32` publishes the first HCA row and preserves
  position-127 argmax 35 at 0.998357518 / 0.057293913. Ordinary position-128
  decode wraps raw slot zero, preserves argmax 201, and recovers to
  0.998725888 / 0.050464456.
- The optimized N=128 path takes 2.909-3.060 seconds versus the prior 35.0-35.2
  second singleton prompt, an 11.4-12.0x improvement. The release CLI
  independently reports 2,934.8 ms, `prefill_mode=layer_major_128`, and
  generated IDs `[35, 201]` across the HCA boundary and continuation.
- The packed causal attention kernel matches ordered singleton rows within two
  F32 epsilons; token-axis Q8_0 GEMV is bitwise identical to successive
  singleton dispatches.
- Retained synthetic attention matches the independent CPU oracle across a
  sliding-window wrap at start 127, CSA and HCA boundary crossings at start
  125, a complete 128-token overwrite from start 128, and the maximum promoted
  CSA geometry at start 2,044 with 128 raw plus 512 compressed rows.
- Two retained 128-token chunks preserve position-255 argmax 35 under the
  established interval containment at 0.997055041 / 0.080157928, then ordinary
  position-256 decode recovers into the endpoint band at argmax 201 and
  0.998410769 / 0.057533156. The complete transition takes 6.179 seconds versus
  the prior 100.9-second singleton prompt path.
- Eight advance-only chunks plus one one-token endpoint reach position 1024 in
  21.144-21.821 seconds, down from 582.536 seconds for packed-128 plus singleton
  replay (26.7-27.6x). The full b10222 vector keeps argmax 201 at cosine
  0.998133285 and relative RMS 0.061869968 without changing the frozen endpoint
  gate. Both native runs produced the byte-identical full-vector SHA-256
  `73d357295a7821607869764af42aaafc845e1764afe8c23a0aab2e5f570a7956`.
- The release CLI independently reserves 1,025/1,025 forwards, reports
  `prefill_mode=layer_major_128_chunks`, and generates oracle ID 201 in 20,357.1
  ms of prompt execution with zero reported swaps.
- Starting from the durable position-1024 state, eight more advance-only chunks
  reach position 2048 in 21.498 seconds and the endpoint completes in 22.495
  seconds total. A fresh session restored at position 2048 reproduces the same
  endpoint bits in 1.141 seconds, and the release CLI reports 982.3 ms of prompt
  execution from that checkpoint.
- A successful advance revokes the session's current logits and final hidden
  observation; vectors already copied to the host remain ordinary owned values.
  A callback unwind from retained position 2 leaves that position unchanged,
  keeps observations invalid, sets poison, and rejects subsequent decode.

Broader S6 work remains:

- Pack intended mixed FP8/BF16 KV and FP4 indexer caches.
- Fuse mHC split/Sinkhorn/collapse, compressor projection/store, shared-KV
  sparse attention, and high-value MoE boundaries.
- Move CPU routing and grouped expert schedules onto the GPU only after named
  retained-chunk phase attribution identifies them as the next bottleneck.

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
| IQ2_S routed gate/up has no production grouped path | Bucket selected rows by expert and use generic IQ2_S matmul; add a grouped kernel only if profiling justifies it |
| Existing Qwen session becomes branch-heavy | Keep DS4 model/session/forward types separate |
| CSA indexer dominates decode | Attribute full-history score and top-k before changing tile shapes |
| Generic Metal fallback hides memory blowups | Require explicit scratch accounting and resident-memory gates |
| Activation QAT differs across references | Freeze operation vectors and distinguish semantic BF16 from packed-cache promotion |
| No tiny official DS4 checkpoint exists | Build operation fixtures and a synthetic tiny family fixture; do not use passthrough layers as an exactness oracle |
| Prompt template differs across runtimes | Port the official 0731 encoder and compare rendered bytes, not rendered intent |
| Asset revisions drift | Pin all shard hashes and checkpoint metadata in every benchmark packet |
| DS4 variant churn | Freeze 0731; add another descriptor only for a measured quality gain and material schema change |

## Immediate next work

1. Use the durable certified position-2048 checkpoint as the ordinary boundary
   development loop; reopen fresh 2,048-token replay only for a causal-prefix or
   restore-ABI change.
2. Keep position 2051 fail-closed until sparse CSA index scoring and stable
   top-512 selection replace dense-all. The second slab intentionally reaches,
   but does not cross, the semantic selection boundary.
3. Differential the selected row IDs, selected attention output, and immediate
   continuation at the first 513-row history before extending physical history
   again.
4. Extend the 0731 message encoder to reasoning and DSML tools only with exact
   release-derived byte fixtures and an end-to-end tool-call workload.
