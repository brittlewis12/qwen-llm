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

## Working records

This document owns architecture contracts, milestone gates, and the current
integrated interpretation. It is not the chronological experiment ledger.

- `docs/PERF-ROADMAP.md` owns the force-ranked optimization queue and
  optimization reopen conditions across model families. DS4 work enters that
  queue once a structural milestone becomes an optimization problem.
- `docs/PERF-LOG.md` owns append-only measured outcomes, including bounded
  KILLs and removed prototypes. Add one entry at each decision-changing
  checkpoint rather than rewriting history here.
- `docs/bench/` owns durable packet details when a result needs raw samples,
  artifact inventories, or an independently reusable protocol. Small
  model-free falsifiers may cite their exact command and retained test instead
  of manufacturing a packet directory.

The strategy may summarize promoted conclusions, but the performance log is
authoritative for what was tried, what gate fired, and why a branch closed.

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

## Pinned assets and local policy

The current product, quality, and optimization target is the 2026-08-04
converter refresh at:

`/Users/tito/models/deepseek-v4-flash-0731/UD-IQ3_XXS/DeepSeek-V4-Flash-0731-UD-IQ3_XXS-00001-of-00004.gguf`

Observed facts from its complete four-shard mapping:

- Source repository: `unsloth/DeepSeek-V4-Flash-0731-GGUF`
- Pinned asset ID: `deepseek-v4-flash-0731-ud-iq3_xxs-current-2026-08-04`
- Quant label: `UD-IQ3_XXS`, `general.file_type = 23`
- Size: 104,207,848,032 bytes (97.05 GiB)
- Shards / tensors: 4 / 1,328
- Architecture: `deepseek4`; no converted `mtp.*` / DSpark tensors
- Census manifest:
  `crates/qwen-llm/tests/fixtures/deepseek_v4_flash_0731_ud_iq3_xxs_current_2026_08_04_census_v1.json`

Pinned current shard hashes:

| Shard | Bytes | SHA-256 |
|---|---:|---|
| 1 | 5,257,696 | `dec1cee704800267d9d836d5a61aefc33705be939bbb3058fa9006d98191576d` |
| 2 | 49,910,532,416 | `3064d3c4c1d6363e9f9ad88e90a3e2c5fb2d6f7ae16ca72135c3ce6a5c984da5` |
| 3 | 49,257,859,456 | `2e9b2732eca7da8324f731653624a4f5c9846258926fd9f468cc703afb51a019` |
| 4 | 5,034,198,464 | `4ca79d8e5107dd1b9bb57b176a7c09948837425dee49f0f1dfd6547a3769fea7` |

The directory label does not describe this recipe completely. Routed gate/up
storage is IQ2_XS in 25 layers, IQ3_XXS in 17, and IQ3_S in one; routed down is
IQ3_XXS in 41 layers and MXFP4 in two. Admission therefore validates role-level
dtype coverage from the artifact rather than inferring it from `UD-IQ3_XXS`.
The refresh is also the first accepted qualitative target: the fixed boundary
probe changed from a low-margin Croatian token on the legacy asset to a sharp
`Hi` / `Hi!` response, while measured decode increased from 30.90 to 43.08
tokens/s. Those observations choose the product asset; they are not a general
quality benchmark.

The 2026-07-31 four-shard asset used for bring-up is historical lineage, not a
required local dependency. Its b10222 logits, decision transcripts, census,
and shard hashes remain immutable in the repository. Historical live tests use
`DSV4_LEGACY_MODEL` and may be reprovisioned under
`/Users/tito/models/deepseek-v4-flash-0731-old/`; ordinary development does not
keep its 95.93 GiB GGUF resident. The exact mixed-revision URLs, commits, Xet
objects, hashes, and engine ordered-content root are retained in
`crates/qwen-llm/tests/fixtures/deepseek_v4_flash_0731_ud_iq3_xxs_legacy_2026_07_31_provisioning_v1.json`.
Its pinned hashes are:

| Shard | Bytes | SHA-256 |
|---|---:|---|
| 1 | 5,257,664 | `9758eb3d78e1afe8852543931703f4f1cd6fbb07f492d4ed853f5d2f6e43be5a` |
| 2 | 49,485,728,288 | `afcfd59721d4da86bc3301e16ca624af202d8af3fa3f9fbbfbb04b3b47666cfd` |
| 3 | 49,437,886,752 | `64eaf514a763597ba7bb50866583d8db5eabbbbce3cb2f616d749af3890155ca` |
| 4 | 4,071,015,712 | `5df52988c56348a22d15da809e9ac4f0cc59cc1c412347f1481dda4685ce89b2` |

The legacy download records resolve shard 2 at a different repository commit
from the other shards. Historical packets therefore pin ordered content hashes
rather than claim one repository revision. They remain valid evidence for the
implementation state they measured, but current performance claims need a new
packet on the current asset rather than a silent repin.

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
quick-task behavior. The ordinary chat and reasoning-mode subset is ported from
two independently maintained release-derived implementations: vLLM's V4
encoder and SGLang's encoder, whose source explicitly records release
provenance. Fourteen generated cases require those pinned implementations to
produce byte-identical prompts before their bytes are accepted as fixtures.

The promoted subset accepts an optional leading system turn, alternating plain
user/assistant history, and a final user turn. Chat mode emits the established
`<｜Assistant｜></think>` transition. `--reasoning high` opens `<think>` on the
final turn, `max` also prepends the exact release effort instruction, and
`--preserve-reasoning` replays assistant `reasoning` / `reasoning_content`
fields across turns with conflict checks. Unknown per-message or wrapped
top-level fields, developer/tool roles, non-alternating history, and Qwen
thinking-policy flags fail closed. Reasoning controls are rejected before every
CLI early exit unless `--messages` owns the request. `--messages-max` applies
after structural JSON deserialization, then role/order/extra-field semantics
validate only the retained prefix; wrapper-level semantics are always rejected.
Raw mode remains the exact reproduction interface. Rich V4 tools,
latest-reminder, response-format, and quick-task semantics remain separate
fixture-gated extensions rather than approximations of the official encoder.

## What transfers from qwen-llm

Directly reusable:

- Validated split-GGUF mmap and tensor descriptors.
- Persistent Shared-buffer ownership, load policy, prefetch, and memory
  admission infrastructure.
- Ordinary Q/K/IQ projection matvec and matmul kernels where dimensions and
  dtypes match.
- Q8_0, Q6_K, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, and MXFP4 decoding primitives.
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

The legacy IQ3 asset stores routed gate/up banks as IQ2_S in 42 layers and
IQ3_S in one. The current product asset instead mixes IQ2_XS, IQ3_XXS, and IQ3_S.
Production singleton execution now has matching all-slot kernels for all four
gate/up dtypes, while packed IQ2_XS uses the generic block-256 matrix path.
Generic matrix coverage is a correctness contract, not a grouped-kernel
performance claim; optimize that packed path only from measured TTFT evidence.

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

Full-model differentials also pin execution shape. At shallow positions, the
qwen-owned singleton decode loop is compared with a llama.cpp oracle that also
evaluates every prompt token separately. The HCA audit found that
`llm --snapshot` had ignored `--prompt-geometry`; `llm` commit
`e07acac20fcd2ee0faca90aa91078ff142724d63` fixes that path and records
`prompt_decode_mode` in snapshot metadata. Position-zero fixtures separately
pin direct token injection. At position 9, qwen versus the corrected b10222
singleton oracle gives cosine 0.999999975 and relative RMS 0.000261269, so the
former batched-prefix vector is not retained as the shallow decode oracle.

That singleton preference is not extrapolated indefinitely. By positions
3070-3072, b10222's own legal singleton and batched schedules have materially
different vectors with the same decisions. Neither schedule is a privileged
bit oracle at depth. Future deep gates therefore use native schedule
self-consistency, bit-exact restore equivalence, operation properties at the
actual far indices, and external schedules as falsifiers. Two independent
external schedules may define a measured envelope; one schedule alone may not.

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

The former FP4 reference difference is resolved for this profile. Official
revision `7872f01b` and the independent vLLM packed implementation at
`b40d859c` establish BF16 input semantics, the `6 * 2^-126` amax floor, UE8M0
power-of-two scales, adjacent E2M1 pairs, and round-to-nearest-even including
signed zero. DwarfStar `54b36ed9` independently cross-checks the Hadamard plus
FP4-simulation graph. SGLang's `1e-4` floor and lower-code midpoint policy are
therefore not the Flash-0731 packed contract; no accelerator is required to
arbitrate them.

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

The position-3072 schedule bifurcation retires deep singleton fixture ladders.
After a mechanism has crossed its real boundary, larger indices are covered by
model-free production-width properties, native packed/singleton containment,
and durable-restore identity. A named full-model endpoint is required only at a
new algorithm, addressing, numerical-regime, memory, or ownership
discontinuity. An external batched endpoint remains a useful falsifier, but is
not called an envelope without a second legal external schedule.

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
validates identity, geometry, prefix, numerical phase, overlap lanes, arena
lengths, and whole-state digest before poisoning the destination. It then zeros
each destination publication buffer directly and copies only the visible wire
prefix; it does not construct a full-capacity host image, and no recoverable
operation remains after mutation begins. Success always becomes ready without
an observation. This preserves the distinction between “correct from restored
position” and “this build reproduced the prefix.”

The durable wire layer derives every section length before allocation, uses a
256-byte versioned header at a 16 KiB payload boundary, requires zero padding and
reserved fields, and closes the record with a BLAKE3 digest in addition to the
typed prefix and causal digests. The CLI policy permits records through 1 GiB.
A terminal 1,048,576-forward state has 4,194,304 prefix bytes, 5,636,096 raw
bytes, 12,206,080 compressor bytes, and 7,214,202,880 visible-publication bytes:
the payload is 7,236,239,360 bytes and the complete v1 record is 7,236,255,776
bytes. A missing snapshot path is sized immediately after tokenization, before
strong shard hashing or Metal residency, so this terminal record fails the
policy cheaply. Existing records are likewise budgeted before residency.
Permitted decode still temporarily retains wire sections while constructing
typed arenas, so its peak host allocation is about twice the visible payload.
Streaming long-context snapshots remain a separate product lane; inference does
not require them. Request/sampler state remains deliberately outside this
model-session checkpoint.

Physical history capacity is destination policy, not wire identity. Snapshot
v1 stores only visible rows and retains the same compatibility digest across
slab counts. A position-3072 record from a 768-row session restores canonically
into a full-context 262,144-row CSA / 8,192-row HCA session with every
unpublished destination bit zero, and re-encoding the restored state produces
the identical record. The reverse direction rejects before mutation when the
snapshot position exceeds the smaller session. No ABI bump or full physical
host image was needed for capacity generalization.

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
- A fresh CLI process hashes the legacy asset's 102,999,888,416 ordered shard
  bytes in 4,840.5 ms,
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
- Seven more retained chunks from position 2176 fill all 768 CSA rows and
  publish HCA row 23 at position 3071. The next-position-3072 checkpoint has a
  38,989,824-byte payload, 39,006,240-byte record, and causal digest
  `8ba373e16e2b9bde526d7b00326ae26331b2bedd68fee2fbfaf6055a2e4f8d25`.
  Fresh restore under the cooperative singleton schedule reproduces
  position-3072 logit SHA-256
  `7ec53d29a78a4d6ee932f292d67dc67c1d15bdd31c050ef2aa625f57fd257764`
  without changing the snapshot ABI or causal digest. The later GPU-route
  schedule preserves that causal record and produces
  `2ecde5de747a8637d38c2cd38538670b3e15932cee15b243b6e25f1b3dfe4c43`
  inside the same independent schedule envelope.

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

Status: the dynamic CSA lane is structurally promoted through the model's exact
1,048,576-forward context. Full-model evidence crosses the first
request-derived fourth-slab row at position 3075 and its continuation at 3076;
terminal-capacity far-index properties publish and consume rows
262,142/262,143 at positions 1,048,571/1,048,575 without replaying their prefix.
Exact-token b10222 full-vocabulary fixtures at positions 3/4 first arbitrated
same-token visibility; positions 7/8 made the overlap roll a live model
differential; positions 2051/2052 pin the first 513-row history; positions
3070/3071/3072 close the former fixed third slab. Every named external fixture
has two byte-identical fresh-session captures, full shard hashes, and pinned
producer commits.

The native lane performs ratio-4 two-branch pooling, learned RMSNorm,
block-start adjacent-pair RoPE, F16 compressed-row publication, overlap roll,
and denominator-only-sink attention over the local raw window plus compressed
history. Attention and indexer publications now own request-derived 256-row
slabs, with a 768-row compatibility floor and 262,144 rows at full context.
Rows 0-511 retain the exact dense-all path; row 512 and later use an explicit
`LlamaCppB10222F16HadamardV1` indexer contract: project 64x128 queries
from normalized Q-LoRA, apply adjacent-pair tail RoPE and normalized Hadamard,
project and scale 64 head weights from normalized attention input, sum
`relu(dot(q_h, k_row)) * weight_h`, then select descending score with lower-row
tie breaks. Selected IDs are compacted back into ascending cache order before
attention, matching b10222 mask-order accumulation. Production scratch stores
only that consumed order; score-ranked IDs remain available to operation probes
without paying their insertion sort or buffer cost on every query.

Singleton and retained packed paths share those score/selection kernels and the
cooperative selected-attention kernel. A packed chunk keeps its pre-boundary
prefix on the old dense kernel and switches only the suffix whose visibility
exceeds 512 rows. The sparse attention kernel receives both the original chunk
start and suffix offset, so a future row in the same chunk cannot overwrite an
older raw-ring key needed by an earlier sparse query. Its N=1 dispatch aliases
the current and preserved raw ring intentionally, computes each shared-KV score
once per query/head, and shares the mass across all 512 output lanes. The host
contract validates the physical compressed capacity and the shader bounds every
selected ID by both visibility and capacity. Invalid geometry or non-finite
scores produce a bounded safe selection, record an error status, and poison the
session after command completion rather than risking an out-of-bounds GPU read.

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
- A frontier restored directly at position 65,528 executes only the final eight
  inputs, publishes rows 16,382 and 16,383, and matches independent CPU
  attention/indexer compressors. Across the four emitted vectors, cosine is at
  least 0.999999792, relative RMS at most 0.000645559, and max absolute error at
  most 0.0029296875 after the production F16 round trip.
- At position 65,535, one Metal command scores all 16,384 physical index rows,
  selects final row 16,383 inside the deterministic far-history top 512, compacts
  those IDs into cache order, and feeds them directly to selected attention.
  Scores, mask, IDs, and output match the independent CPU contracts; a blind
  first-512 alternative is materially different. Earlier sparse gates separately
  pin future-row masking.
- A terminal ratio-4 frontier starts at position 1,048,568 and publishes rows
  262,142/262,143 for both attention and Hadamard-indexer caches. Against the
  iterative CPU compressor after the production F16 round trip, attention rows
  measure cosine 0.999999024/0.999992249, relative RMS
  0.001397355/0.003937333, and max error 0.010986328/0.037734985; indexer rows
  measure 0.999995600/0.999951636, 0.002969242/0.009835373, and
  0.005859375/0.016235352. The wider far-position envelope accounts for direct
  power versus iterative RoPE evaluation rather than changing the equation.
- At position 1,048,575, a one-head 128-wide integration gate scores all
  262,144 index rows, forces terminal row 262,143 into the deterministic top
  512, compacts the result into cache order, and feeds selected attention.
  Score, mask, IDs, and output match their CPU contracts, closing terminal
  capacity, stride, visibility, and top-k consumption without a million-token
  replay. Separate 64x128 production-head gates pin the same scoring equation
  and cache layout at smaller histories.
- Production 64x128 index scoring over 513 and 768 physical rows matches the
  CPU equation. Stable top-512 tests pin exact IDs, all-score ties retain rows
  0-511, a full 768-visible-row query retains exactly rows 256-767 in descending
  score / ascending cache order, and a non-finite visible score fails closed.
  Selected attention drops a tagged row, includes row 512, matches the CPU
  oracle, and materially differs from both dense-513 and blind-first-512
  alternatives.
- The selector no longer performs 512 synchronized full-history maximum scans.
  Histories above 1,024 rows use one deterministic 256-thread threadgroup per
  query and 32 MSB-first radix count passes to derive the exact cutoff score.
  The key transform preserves finite numerical order, canonicalizes exponent-
  zero values to match the deployed fast-math subnormal/`+0`/`-0` equivalence,
  rejects every visible NaN or infinity first, and retains lower row IDs at the
  cutoff. Contiguous per-lane prefix compaction emits ascending cache-order IDs;
  score-ranked IDs are insertion-sorted only when a diagnostic asks for them.
- The radix path reuses two 256-u32 threadgroup regions and adds no persistent
  or admission-priced storage. Exact gates cover the 1,024/1,025 dispatch seam,
  16 mixed packed queries, cutoff ties of 2/511/512/513/all rows, signed zeros,
  positive and negative subnormals, all three non-finite classes, invalid and
  per-query visibility, repeated dispatches, and terminal row 262,143. The
  16,384-row warm host packet now measures about 0.32 ms for one query and
  0.49 ms for 128 queries versus the former 4.76-4.80 and 13.27-13.45 ms.
- Two fresh b10222 position-2051 captures are byte-identical at SHA-256
  `064fd66567734085532b7307186b4087adea7c28cc6334105b0e26e05b50b8a6`;
  two independent position-2052 captures are byte-identical at
  `819a833db015eb57c553d3e77e9514141d5094fc0b07df9e9977553bfa51abaf`.
  Native singleton execution preserves argmaxes 35 and 201 at cosine / relative
  RMS 0.998268661 / 0.059772205 and 0.997481235 / 0.070970007.
- A packed four-token chunk crosses the 513-row boundary and an immediate packed
  continuation preserves argmax 201. Against the singleton continuation it has
  relative RMS 0.000265079; a model-free packed test also
  matches the CPU oracle while forcing the original-chunk raw-ring branch.
- The durable next-position-2052 checkpoint has a 31,967,504-byte payload,
  31,983,920-byte record, and causal digest
  `8b7906e362419dfe077eb5dd7d4909e154a5729cd100463dcfd9c2a86c172920`.
  Fresh restore remains ABI-compatible and reproduces cooperative-schedule
  position-2052 logit SHA-256
  `c4858badebd29be9ae861ff96161050f0f2165ab757245df4ae70e01c7e1663f`.
  Packed and singleton prefixes intentionally retain their own deterministic
  reduction histories; their snapshots need not be bit-identical.
- At positions 3070/3071/3072, b10222's legal singleton and batched prompt
  schedules differ materially even though all argmaxes agree. The old absolute
  singleton-relative floor is therefore reported, not treated as a
  schedule-invariant semantic oracle. A pinned three-schedule envelope uses
  angular chord and common-singleton-norm L2 diameter: the expanded native
  diameter must fit inside the independently measured b10222 schedule diameter
  plus the already-established historical allowance. Measured expanded versus
  allowed angular/L2 diameters are 0.202833/0.215389 versus
  0.289972/0.298147 at position 3070, 0.261382/0.271941 versus
  0.402803/0.421941 at position 3071, and 0.102987/0.105349 versus
  0.161559/0.168516 at position 3072.
- Complete position-3070 decision transcripts pin 21 CSA selectors and all 43
  MoE routes for native, batched b10222, and singleton b10222. All three agree
  through CSA layer 2 and routes through layer 4. Native first swaps two rows at
  layer 4 under near-identical scores and milliscale cutoff margins; the two
  b10222 schedules themselves first swap a cutoff row at layer 6. Route slot/set
  divergence begins at layers 5/7 for both native/reference and
  reference/reference comparisons. The first disagreement is therefore a
  cutoff-sensitive numerical bifurcation, not a publication, ordering, or
  large-margin semantic mismatch. Canonical vectors and transcripts are
  byte-identical across two fresh processes and validated model-free. The later
  dense-attention promotion changes low-order score and route-weight values but
  preserves every one of the 21 cache-order selected-ID lists and 43 ordered
  routed-expert lists from the exact-radix checkpoint. Fifteen selector layers
  have 152 internal ordinal changes, all by at most two ranks; rank-512/513 row
  IDs remain unchanged, as do all consumed sets and orders. The smallest cutoff
  margin is 0.000160 after the change, maximum score delta is 0.000263, and
  maximum route-weight delta is 0.00002378. The regenerated transcript is
  byte-identical across fresh runs at SHA-256
  `7ce0ab84f262d5e9ce8cd3098452872156f180a53d5282d71c68d4b97ce9e393`.
  GPU route publication later supersedes only the floating schedule: all
  consumed IDs remain exact and two fresh transcripts repeat at
  `e9d17dc01a3b05057020d552e1bd81359c5a27ed8b60495b50a40f0b605b1284`.
- The certified position-3072 v1 snapshot restores into a 1,024-row CSA session
  and reproduces its existing full-logit hash before crossing the old fixed
  allocation. Packed and singleton schedules then publish row 768 at position
  3075 and continue through 3076 at cosine / relative-RMS pairs
  0.999999986 / 0.000169229 and 0.999999998 / 0.000063240. Fresh restore of the
  row-768 state reproduces continuation bits exactly; the terminal causal digest
  is `73d2c01d1db99d5a84e3da691fd8bdb7a7ea720ee6fc49c9926ade6f55a838a3`,
  and position 3077 rejects before mutation in that deliberately bounded test
  session.

The official Hadamard-plus-MXFP4 scalar contract is frozen separately from the
production cache. Its generated fixture pins four 32-value blocks, 64 packed
value bytes, four UE8M0 scale bytes, BF16-before-amax semantics, exact midpoint
and scale boundaries, malformed-row rejection, and a decoded packed-score
transcription at SHA-256
`0e5e2b251a960d417e7977608a363b83e072e2d90bc286cc52820b0ea7dc2b1f`.
The fixture's contiguous 68-byte envelope is an oracle representation: upstream
Q uses separate value/scale tensors and paged K segregates values from scales at
the cache-block level. The scalar row type is intentionally narrower than the
physical encoding: any row whose decoded values overflow F32 is rejected before
scoring, including a BF16-maximum pack input. Packed Metal execution remains a
future numerics ABI and
must not silently replace the executable b10222 F16 cache contract used by
these vanilla-GGUF full-model fixtures. Causal-snapshot numerics ABI v1 includes
`LlamaCppB10222F16HadamardV1`; changing scoring, tie/order, or index-cache
numerics requires a version bump or explicit legacy acceptance.

### S4: HCA lane

Status: the dense HCA lane is structurally promoted through all 8,192 rows in
the model's exact context. Full-model semantic evidence currently reaches 24
published rows and position 3076; a bounded real-weight integration gate crosses
the first tiled query at position 65,663, and production-width properties
publish and consume terminal row 8,191 at position 1,048,575 without replaying
the prefix. All 20 HCA layers use the retained ratio-128 frontier to pool,
RMS-normalize, apply block-start adjacent-pair RoPE, publish F16 compressed rows,
and include every published row in the same-token softmax over the 128-row local
window plus dense compressed history. Counts through 512 use the cooperative
single-threadgroup reduction. Larger histories use a correctness-first tiled
two-pass kernel: each 512-row tile recomputes scalar 512-wide dot products,
finds the global maximum, and then accumulates masses and values in the same
raw-then-compressed order as the legacy path. It uses no global scratch and
makes no throughput claim. At production 64x512 geometry the cooperative and
tiled kernels are bit-identical at exactly 512 compressed rows, so the handoff
does not add a reduction discontinuity. The first four rows retain their named
full-model boundary evidence. Later rows use generalized production-width operation
properties and strategic full-model endpoints rather than a mechanical fixture
campaign at every boundary. Position 2047
publishes HCA row 15 and CSA row 511; position 2048 proves immediate wrapped
continuation at the former dense-all limit, positions 2051/2052 prove HCA
continues unchanged as CSA enters sparse selection, positions 2174/2175/2176
pin the first HCA publication reached while that sparse path is active, and
position 3071 publishes row 23 as CSA fills its third slab.

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
- A production-width ratio-128 property executes all 3,073 positions through
  row 23. It rejects every early publication, compares every emitted row with
  the independent CPU compressor after F16 storage, preserves all older rows,
  validates block-start YaRN positions 0 through 2,944, and leaves the history
  unchanged at position 3072.
- A second production-width property reconstructs the frontier at position
  65,408, executes exactly the final 128-token block, and publishes HCA row 511
  at position 65,535. Its frontier state and final F16 row match the independent
  CPU compressor; the same Metal command then consumes all 512 rows and matches
  CPU attention while a prior-511-row ablation is materially different. This
  proves far row addressing and terminal same-token visibility without a 65K
  replay.
- The one-head tiled differential is bit-identical to the legacy reduction at
  compressed counts 1, 511, and 512. At production 64-head geometry the
  cooperative and tiled kernels are bit-identical to each other at count 512;
  both differ from legacy in the same 66 of 32,768 output words. Above the
  switch tiled attention matches the independent CPU equation at counts 513,
  527, 528, 895, 896, 897, 8,191, and 8,192, including late rising maxima and a
  material final-row ablation. Counts 895/896/897 straddle a true 512-row tile
  boundary rather than only the cooperative-to-tiled dispatch boundary.
- A packed two-query gate at positions 65,662/65,663 executes a cooperative prefix
  and tiled suffix in one retained chunk. It preserves the original pre-chunk
  raw ring for the older query and consumes the newly published row 512 only for
  the later query. A separate terminal pair at positions
  1,048,574/1,048,575 pins visibility 8,191 to 8,192 and the same raw-cache
  causality at the maximum supported indices.
- A ratio-128 frontier starts directly at position 1,048,448, publishes row
  8,191 on position 1,048,575, and consumes all 8,192 rows in that same Metal
  command. The published F16 row measures cosine 0.999990738, relative RMS
  0.004303934, and max absolute error 0.032226563 against the iterative CPU
  compressor; attention recomputed from the actual stored row then matches the
  CPU contract at the tight operation tolerance.
- Scaled YaRN forward/inverse properties cover positions 65,535, 65,536,
  1,048,448, and 1,048,575. The production direct-power Metal schedule and the
  independent iterative-theta CPU schedule each remain inside relative-RMS
  0.01 / max-error 0.02 envelopes; observed worst cases are 0.002540418 /
  0.003356243 and 0.004722705 / 0.003458768 respectively. The unscaled
  theta-10,000 branch separately remains bounded at the terminal position with
  relative RMS 0.004380320 and max error 0.002259519.
- A deliberately bounded real-weight session begins with synthetic empty causal
  state at position 65,663 and executes one complete 43-layer token through
  sparse CSA and the first tiled HCA query. Two fresh sessions produce the
  byte-identical full-logit SHA-256
  `915be9ad610710c4bcb3cfb661ef1fcdd35e2ccebca82c20e15b05336528997a`;
  the next position rejects without changing logits. This proves mechanical
  integration on the legacy 95.93 GiB asset, not semantic correctness of the
  synthetic prefix.
- Serial publication-to-attention gates at positions 639, 1023, 2047, 2175,
  and 3071 materially distinguish the newest row from a prior-count ablation.
  Tagged
  wrapped-ring gates at 638/639/640 and 1022/1023/1024 independently match HCA
  counts 4/5/5 and 7/8/8 plus CSA counts 159/160/160 and 255/256/256. The same gate now
  covers 1026/1027/1028 and 2046/2047/2048, including HCA counts 15/16/16 and
  CSA counts 511/512/512 at the second-slab endpoint.
- Production-width ratio-4 attention and Hadamard-indexer properties publish
  rows 0-767 into three fixed slabs, match the overlap-roll CPU state through
  the sparse boundary, and preserve all older rows. Dense attention accepts
  count 512 and rejects 513; the selected path accepts the 513-row physical
  history while still attending to exactly 512 compressed rows. Packed retained
  attention exercises the full 640-thread raw-plus-selected geometry.
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
- Fresh-session b10222 captures at positions 2174, 2175, and 2176 are
  byte-identical across repeats at SHA-256
  `dd1662dd4310ce97796f666ccc41b7088c747409b8148487a9c26d6d69f95d4f`,
  `a3bd79f2b0becb7a97ad8688b698711b78d9df3c0c5b8a09c6f51d6011873e58`,
  and `2c4a75eaa53d206ee996ec01480dff2f5d41c3dae02185c9737de34b6dbeb8f7`.
- The singleton-schedule native control/boundary/continuation preserves
  argmaxes 34/35/201 at cosine / relative-RMS pairs
  0.996295665 / 0.092472428, 0.997194466 / 0.075045587, and
  0.997641601 / 0.069803888. Publication and continuation each improve both
  measures, so inherited sparse-interval drift is not misattributed to row 16.
- The product packed schedule reaches the boundary in one 124-token chunk at
  0.995150607 / 0.098514095, then recovers on the immediate singleton
  continuation to 0.997213076 / 0.076756856. The gate combines 0.990 / 0.15
  interval containment, a boundary allowance of 0.002 cosine / 0.01 relative
  RMS from the pre-boundary control, strict two-measure recovery, and an
  absolute continuation floor/ceiling of 0.997 / 0.08. Packed-versus-singleton
  schedule comparisons are separately bounded at 0.998 / 0.06; measured pairs
  are 0.998268217 / 0.058828178 and 0.999267484 / 0.038675588.
- The next-position-2176 checkpoint has a 32,821,760-byte payload,
  32,838,176-byte record, and causal digest
  `279a2f4ba1a7a6541144b5af6f2cbff745b498cca1fd24e414ec1cb7f86ffa68`.
  Fresh restore under the cooperative singleton schedule reproduces full-logit
  SHA-256
  `5218c60672d51e48f2dbd832584aac39c895e5b34b39724e96ca7705642293c8`;
  position 2177 rejects before mutation and preserves every completed
  position-2176 logit bit.

The former post-original-context discontinuity is now closed. Position 65,663
uses 513 dense compressed rows, the packed scheduler splits crossing chunks at
the first tiled query, and the same kernel scales to all 8,192 terminal rows.
The promoted strategic capacity is the model's exact 1,048,576 forwards rather
than decimal one million. There is no further CSA or HCA addressing or algorithm
discontinuity below that ceiling: sparse CSA always consumes at most 512 selected
rows, dense HCA scans derived 512-row tiles, and publication storage grows in
request-derived slabs. Named full-layer intermediate states remain useful for an
actual divergence, not at every compression boundary. Full-context feasibility
is a correctness and memory statement; practical long-prompt latency remains an
optimization problem.

### S5: full 0731 target generation

Status: bounded raw CLI and resident-memory admission slices promoted on
2026-08-02; request-derived execution and admission now extend through the exact
model context. The release `qwen` binary opens the split GGUF once, dispatches
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
partially stream beyond the engine-owned 1,048,576-forward evidence ceiling.
The 103 GB split GGUF is already virtually mapped for family detection at that
point; residency remains untouched on rejection. The promoted capacity is
exported by the session implementation, so frontend and executor cannot drift
onto independent magic limits. After tokenization, the CLI derives an immutable
request capacity; transcript, compressor slabs, sparse scratch, snapshot
constraints, admission, and the session guard all receive that same value. The
shared one-open handoff also passed a release one-token Qwen 3.5 0.8B smoke,
preserving the existing family path.

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
- The earlier 2,049-forward budget validated the release CLI against the
  certified 2,048-token checkpoint: it restored before residency, executed the
  uncached position-2048 endpoint in 982.3 ms, and generated
  oracle ID 201 with `prefill_mode=causal_snapshot_restore`. End-to-end process
  time is 1.56 seconds.
- The earlier 2,053-forward budget validated the 31,983,920-byte position-2052
  record before residency, restored 2,052 tokens, consumed the uncached sparse
  endpoint in 1,495.5 ms, and
  generated oracle ID 201 with `prefill_mode=causal_snapshot_restore`. The
  request reported `reserved_forwards=2053/2053`.
- The earlier exported budget accepted exactly 2,177 forwards. The release CLI
  validates the 32,838,176-byte position-2176 record before residency, restores
  2,176 tokens, consumes the uncached endpoint in 1,049.5 ms, and generates
  oracle ID 201. It reports `reserved_forwards=2177/2177`; position 2177 remains
  rejected in that historical build.
- The prior 3,073-forward gate used a release CLI process that
  validates the 39,006,240-byte position-3072 record before residency, restores
  3,072 tokens, executes the uncached endpoint in 1,389.9 ms, and generates
  oracle ID 201 with `reserved_forwards=3073/3073` and
  `prefill_mode=causal_snapshot_restore`. End-to-end process time is 1.79
  seconds with zero reported swaps.
- The current request contract accepts every exact budget through 1,048,576 and
  rejects 1,048,577 before residency. Capacity arithmetic is exhaustively
  checked over the entire interval: CSA requires `floor(F/4)` rows, HCA requires
  `floor(F/128)`, and each rounds to a 256-row slab with compatibility floors of
  768 and 512. Thus 3,075 forwards retain 768 CSA rows, 3,076 allocate 1,024,
  decimal 1,000,000 allocates 250,112 CSA / 7,936 HCA rows, and the exact model
  context allocates 262,144 CSA / 8,192 HCA rows.
- Ordinary 0731 messages match vLLM and SGLang byte-for-byte. On the Flash
  vocabulary, user-only, system/user, and multi-turn Unicode fixtures encode to
  exact token sequences of 5, 8, and 16 tokens. A live system/user request
  rendered to 11 tokens and qwen generated IDs `[46382, 15697, 11898, 1]`
  (`ghiaccioli` then EOS); the b10222 `llm` singleton oracle produced the same
  first-token argmax and complete greedy text.
- Fourteen executable-source fixtures cover chat, high/max reasoning,
  drop/preserve history, two rounds, missing reasoning, and both field aliases.
  Pinned vLLM and SGLang revisions plus encoder hashes regenerate every prompt
  byte-identically; invalid alias, role, wrapper, and CLI-scope combinations
  fail before model execution.
- Intermediate bisect can isolate any divergence to one layer and operation.
- Repeated runs are deterministic under the same host-validity contract used
  by Qwen benchmarks.
- Allocation-free full-context planning inventories 7 resident buffers (3
  retained no-copy windows and 4 final-page copies) at 102,994,608,640 logical /
  102,994,624,512 priced-upper bytes and 542 unique session buffers at
  7,631,942,884 logical / 7,636,467,712 priced-upper bytes. Of the logical
  session total, 7,214,202,880 bytes are published-history storage. The
  remainder includes the complete physical 128-token packed scratch,
  sparse-index score/selection matrices, and 131,072-byte pre-chunk raw-ring
  snapshot rather than charging them to reserve. The complete priced upper
  bound is 110,631,092,224 bytes; a 536,870,912-byte dynamic reserve makes the
  admission requirement 111,167,963,136 bytes. The final additions are the
  49,152-byte six-slot routed FFN row and 2,752 logical bytes of transient
  per-layer route and sparse-selection records.
- The load plan freezes configuration plus every descriptor's name, shape,
  dtype, shard, offset, and byte length. Realization revalidates those values,
  fallback policy, all view/alias/window geometry, and a deterministic planner
  rebuild before refreshing admission at the last allocation-free point.
- On the target M4 Max, Metal reported 126,701,535,232 recommended bytes and a
  475,136-byte baseline, giving 126,701,060,096 bytes of working-set headroom.
  The process signal was `Some(0)`, the established omitted-limit convention,
  so the explicit reason was `admitted_process_budget_omitted`.
- A post-collapse live full-context reconciliation still observed
  7,631,945,728 session bytes and 110,626,553,856 cumulative
  residency-plus-session bytes: the three small record buffers did not move
  Metal's endpoint allocation counter, while both observations remain below
  their newly increased priced inventories. Residency and session must fit
  those inventories without the reserve; only the first-forward endpoint gate
  may use the reserve.
  Residency is reconciled inside realization, before an unaccounted resident
  handle can be returned.
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

This admission and reconciliation contract is explicitly Metal-scoped. It does
not include the snapshot codec's temporary host vectors, whose separate 1 GiB
record limit and roughly twice-visible-payload decode peak are documented above.
The CLI preflights an absent snapshot's exact record size before shard hashing
or residency, so a full-context capture is rejected by policy rather than
silently consuming the 7.2 GiB wire payload. A future total-process admission
claim must account for streaming or those host allocations.

The bounded raw and ordinary-message slices establish first-class native
inference. Full-context operation properties, request guards, and live memory
reconciliation establish structural feasibility through every model position;
they do not claim practical million-token TTFT or replace a real-prefix semantic
endpoint. Rich tool/developer schemas are product extensions, not prerequisites
for the ordinary chat and reasoning prompt-encoder gate.

### S6: Metal performance promotion

Status: retained layer-major chunks of 1-128 tokens use request-derived storage
through the exact 1,048,576-forward model context. The physical scratch is sized
and admitted once for 128; short chunks use exact prefix views. One-token prompts
remain singleton only when they are the entire request. Longer prompts consume
successive explicit packed chunks, with no singleton replay or hidden fallback.
Full-model packed semantic evidence crosses the old fixed-capacity boundary
through position 3076. Production-width far-index properties, terminal
two-query causality gates, and a bounded real-weight post-64K execution replace
prohibitively slow deep-prefix fixture replay at the remaining structural
boundaries.

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
CSA chunks that straddle row 512 run a cooperative dense prefix followed by a
sparse suffix with per-query visible counts and cache-ordered selected IDs. The
dense helper allocates masses for the final query's exact raw-plus-compressed row
count and dispatches the greater of that count and 512 threads, up to 640. The
sparse suffix always has 128 raw plus 512 selected rows and uses the full
640-thread / 641-float shape so every score has one producer while the first 512
lanes also write output. HCA queries through 512 compressed rows retain the
cooperative reduction. Above 512, the packed scheduler keeps any cooperative
prefix and dispatches only the suffix from the first 513-row query to a
512-thread tiled two-pass kernel. Singleton HCA above 512 instead uses the
promoted 32-lane online kernel; no packed caller enters that schedule. The packed
suffix retains the original chunk start so future raw-ring writes cannot leak
into an earlier query.

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

The selector originally also materialized a score-ranked top-512 list for every
query, then sorted it by insertion even though attention consumes only ascending
cache-order IDs. Ranked output is now optional and enabled only by diagnostic
operation tests. Removing the singleton and packed ranked buffers reduces the
session by 264,192 logical / 278,528 priced-upper bytes. In the measured row-16
packet, the full two-schedule live gate fell from 14.651 to 12.574 seconds while
retaining every pinned logit and causal-state bit; the release restored endpoint
fell from 1,566.3 to 1,049.5 ms. These are observed paired runs, not a standalone
microbenchmark attribution.

The first production-shape far-context profile identified a separate singleton
attention cliff. The original kernel recomputed each 512-wide shared-KV dot
product independently for every output lane and took 41.49-41.63 ms per CSA
layer regardless of history length. Singleton decode now uses the packed
cooperative kernel: each score is computed once per query/head and shared through
threadgroup memory. It takes 0.303-0.304 ms, a 136.4-137.2x kernel speedup. The
legacy selected-attention host path remains test-only. Legacy dense attention
remains production only at singleton position zero and otherwise serves the
numerical differential.

The first profile measured deterministic top-512 selection at 3.490, 25.876,
and 114.212 ms over 16,384, 65,536, and 262,144 compressed rows. Replacing its
512 maximum scans with exact radix thresholding reduces all-tied medians to
0.190, 1.146, and 4.467 ms and production-like mixed-score medians to 0.190,
1.172, and 4.582 ms, an 18.4-24.9x improvement. Twenty-dispatch p95s are at
most 0.191, 1.173, and 4.589 ms. The same packet measures index scoring at
0.702, 1.961, and 8.232 ms and cooperative attention at 0.320, 0.323, and
0.323 ms. Conservatively projecting only those three isolated GPU phases across
all 21 CSA layers gives 25.451, 72.568, and 275.880 ms per token.

These are not full decode timings: they exclude projections, HCA, MoE, command
synchronization, and host work. They establish that full-history index scoring,
not exact selection or selected attention, is now the dominant CSA target.

The production 64-head x 128-dimension scorer now assigns one simdgroup to each
row. A 256-thread group stages eight disjoint F16 key rows once, each lane
computes two complete head dots in original dimension order, and lane zero
accumulates the 64 weighted terms in original head order. This preserves every
scalar score bit while reducing 16,384/65,536/262,144-row medians to
0.141/0.533/2.102 ms, a 3.95-4.99x scalar-bracket midpoint speedup; the raw
65,536-row endpoints span 3.75-4.16x. The resulting conservative
score-plus-select-plus-attention projection across 21 CSA layers is about
13.7/42.5/147.5 ms at 64K/262K/1M-token-equivalent histories. At this checkpoint
exact radix selection, not scoring, led the terminal isolated packet.

A real-weight synthetic state at position 65,663 proves the integrated effect.
Across two fresh-process scalar/cooperative/scalar brackets, cooperative scoring
removes 10.8-11.6 ms from both command-GPU and wall time. Full score vectors,
selected IDs/statuses, route decisions, logits, and causal state remain exact.
The canonical zero-state fixture pins logits at
`1c0f5e0475314e693bfe0664b5454a2ece26d9a5913f9a5218cdf39da59582d4`
and causal state at
`03f15887db83e4baf0ad5ba66f95b3e2a7fe461858d92aebe9f0b3009f83254e`.
Because Metal scratch allocation is intentionally uninitialized, constructed
deep states now initialize raw history, compressor state, and published rows
explicitly rather than relying on a misleading `zeros_*` constructor name.

The promoted selector resolves four key bits per pass instead of one. Each lane
keeps 16 private counts; eight simdgroups reduce an 8x16 table in existing
threadgroup storage; lane zero applies the same one-based descending rank.
Threshold equality, lower-row tie retention, cache-order compaction, optional
ranked output, and bounded statuses remain unchanged. Separate compile-time
instantiations keep the original bitwise kernel as an executable differential
without a production switch.

At 16,384/65,536/262,144 rows, mixed radix4 selection takes
0.123/0.499/1.875 ms versus bitwise 0.190/1.173/4.58 ms; all-tied radix4 takes
0.137/0.539/2.019 ms with terminal p95 2.035 ms. Packed 128-query geometry is
also exact and improves from a 1.593 ms bitwise-bracket midpoint to 1.251 ms.
Four complete real-weight position-65,663 brackets remove 1.06-1.29 ms
command-GPU and 1.06-1.70 ms wall while preserving the full decision
transcript, logits, and causal state. The conservative isolated 21-layer CSA
projection is now about 12.6/29.2/93.4 ms. Scoring and selection are near peers
at terminal history; complete-token attribution shows that neither is the
largest remaining far-context phase.

At production 64-head x 512-dimension geometry, the exact tiled-HCA kernel takes
about 0.68/1.44/5.26 ms per layer at 513/2,048/8,192 compressed rows. A retained
deterministic nonzero repeat measures 0.763/1.637/5.259 ms median and
0.777/1.904/5.263 ms p95. The stable terminal endpoint projects to about
105.2 ms over all 20 HCA layers, above the 93.4 ms terminal CSA subtotal.

A real-weight session with zero-initialized synthetic causal history constructed
for the final two context positions separates a first-touch observation from
warm inference. Across two packets, its first token takes 240.053-241.613 ms
command-GPU while wall time varies from 1.024 to 17.549 seconds. The excess wait
is consistent with residency effects but remains unattributed. The immediately
following terminal token takes 232.989-235.787 ms command-GPU and
235.998-237.261 ms wall, with only 1.474-3.009 ms outside the GPU. The repeated
terminal logit SHA-256 is
`4c54019668cb815036bd823ddee2c4156f481a48138b35896899d758184587be`.
This roughly 4.22-4.24 token/s endpoint passes a rough scale check against
37 ms short-context + 93.4 ms CSA + 105.2 ms HCA estimates. The terms are not
disjoint because the short-context command already includes shallow attention;
the check is not a phase decomposition. The first-touch wait is a separate
cold-start observation, not hidden Metal execution.

Three exact-reduction HCA candidates were bounded under one frozen gate: no more
than 0.05 ms/layer regression at 513 rows, at least 0.20 ms/layer saving at
2,048, and at least 1.50 ms/layer at 8,192 with terminal p95 below both baseline
arms. Reusing an F32 score slab, pairing two heads to share KV loads, and
combining both preserve every named output bit but save only 0.978, 1.191, and
0.954 ms/layer at terminal history. All were removed. The next bounded HCA
candidate changes the schedule materially through online softmax/value tiling
under an explicit numerical envelope; the gate is not weakened and pair-four is
not a mechanical follow-up.

The promoted singleton schedule assigns one 32-lane simdgroup to every head.
Each lane retains four F32 query/output vectors while the group stages one
512-value F16 shared-KV row in 1 KiB of threadgroup memory, computes its QK
reduction, and immediately updates F32 online-softmax/value state. The sink is
the initial zero-valued pseudo-row; chronological raw rows still precede dense
compressed rows. This reduces three logical cache traversals to one without a
split reducer, persistent scratch, cache-layout change, or snapshot-ABI change.
Packed multi-query HCA remains on the exact tiled schedule.

At 513/2,048/8,192 rows, old/online/old production-width medians are
0.732/0.418/0.777, 1.721/0.899/1.435, and 5.258/3.410/5.256 ms/layer. Terminal
online p95 is 3.413 ms; the 1.848 ms/layer midpoint saving clears the frozen
1.50 ms gate and reduces isolated 20-layer HCA from about 105.2 to 68.2 ms.
Ten structural boundaries cover sink dominance, near-equal and moving maxima,
wrapped/distinct raw buffers, and newest-row visibility. Worst production-width
legacy-relative cosine is 0.999999998, relative RMS 5.873e-5, and scaled maximum
error 1.041e-6; repeats are bit-identical.

The real-weight position-65,663 transcript preserves every consumed CSA/MoE ID
and status plus argmax 7,249 at logit cosine 1.0 and relative RMS 2.28e-7. Two
terminal campaigns comprising four legacy/online/legacy packets save
22.5-24.5 ms in command-GPU and wall time, moving warm inference from 233-237
to 210-213 ms, about 4.70-4.73 token/s, without an outside-GPU regression. Two
terminal online transcripts are exact
repeats and preserve every legacy consumed ID/status plus argmax 201; maximum
CSA score, route-weight, and cutoff-margin deltas are 2.861e-6, 6.11e-7, and
zero. The online terminal logit and causal pins are
`c6a6075667623b0127d3b18383c5f4e11b136502ab53163d0d8b9edde3cb244c`
and `adb621cb44f5ad957dc60568db9a346344d28995430518e0f9d641c7d7982317`.
CSA's unchanged 93.4 ms subtotal therefore leads online HCA's 68.2 ms subtotal;
the higher-complexity heads8/rows16 split-K HCA design remains deferred.

The retained legacy selected-attention differential still measures 43.36 ms at
the 128-raw-plus-512-compressed shape. Singleton dense attention had the same
output-lane score-recomputation structure in every SWA layer, HCA through 512
rows, and CSA through its first 512 rows. Positions at least one now share the
packed cooperative dense kernel; singleton position zero deliberately retains
the legacy kernel and its strongest exact fixture lineage. A paired five-sample
release packet reduces median singleton decode from 487.743 to 59.498 ms at context
about 128 and from 624.192 to 59.865 ms at context about 512: 2.050 to 16.807
tokens/s and 1.602 to 16.704 tokens/s. Context-dependent growth over that range
falls from 136.449 to 0.367 ms.

The baseline is clean commit `cfb4d3a` with only the profiler harness applied;
both sides use:

```bash
cargo test --release -p qwen-llm --test deepseek_v4_position_zero_live profile_native_deepseek_v4_singleton_decode_at_128_and_512 -- --ignored --exact --nocapture --test-threads=1
```

Baseline context-128/context-512 samples in milliseconds are
`[485.663, 485.194, 487.743, 487.997, 489.423]` and
`[621.798, 621.767, 627.592, 626.768, 624.192]`; candidate samples are
`[59.973708, 59.498083, 60.165667, 59.021916, 58.860167]` and
`[60.106792, 59.865125, 60.702625, 59.439334, 59.369167]`. This is an observed
near-flat five-sample curve, not a context-independent throughput claim.

At production 64x512 geometry, cooperative dense attention is bit-identical to
legacy for 128-row SWA and 129-row HCA. Its worst named CSA/HCA difference is 137
of 32,768 F32 words, maximum absolute error 2.33e-10, and relative RMS 1.1e-8.
At the 512-compressed-row HCA handoff, cooperative and tiled kernels are
bit-identical to each other; both differ from legacy in the same 66 words at
maximum absolute error 5.83e-11. This gives the <=512 and >512 paths one exact
production-shape reduction lineage rather than merely adjacent tolerances.

A clean current-upstream llama.cpp Metal packet at b10254 measures 36.137 and
36.255 ms/token at cached depths 128 and 512, or 27.673 and 27.582 tokens/s.
qwen-llm takes 1.646x/1.651x as long per token and delivers 60.7%/60.6% of
upstream throughput despite matching its near-flat context curve. A separately pinned b10235 exact
token-injection packet remains near 27 tokens/s and repeats its full vectors
byte-for-byte. A cold-first qwen prefix observation is materially behind warmed
llama_core medians, but no prefill ratio is promoted until warm treatment is
matched. Complete commands, revisions, decode samples, hashes, and comparison caveats are retained in
`docs/bench/2026-08-03-dsv4-metal-llama-baseline.md`.

A focused routing-seam packet replays identical context-128 and context-512
snapshots through five ordinary forwards, five instrumented forwards, and five
ordinary forwards. All three schedules produce the same final-logit hash at
each depth. Instrumentation is therefore observational: it adds no dispatches
to the ordinary path and does not change the profiled compute graph. Median
ordinary/instrumented/ordinary wall times are 58.849/58.748/58.957 ms at depth
128 and 59.344/59.636/59.873 ms at depth 512.

The same-queue Metal-clock interval from each router command's GPU end to its
expert command's GPU start totals a median 9.827 and 9.563 ms/token. CPU route
and copy work, measured inside rather than in addition to that interval, totals
1.124 and 1.116 ms; expert-command construction totals 1.423 and 1.406 ms.
Hash layers account for about 1.108/1.094 ms of the gap and the 40 learned
layers for 8.719/8.436 ms. Both depths decisively cross the preregistered 5 ms
gate. The next optimization is therefore a private GPU route record consumed
by dynamically indexed experts in one command per layer. Parallel expert-slot
execution, packed GPU bucketization, whole-token submission, and an externally
observable SSD-streaming ticket remain separate, measured follow-ons.

The attribution command at checkpoint `7040e03` is:

```bash
cargo test --release -p qwen-llm --test deepseek_v4_position_zero_live profile_native_deepseek_v4_routing_seam_at_128_and_512 -- --ignored --exact --nocapture
```

The promoted implementation publishes hash and learned routes on the GPU and
feeds their six IDs directly into indexed IQ2_S, IQ3_S, IQ3_XXS, and MXFP4
expert projections. Router, route, serial six-slot experts, shared expert, and
residual update now occupy one command per layer, reducing singleton
commit/completion boundaries from 86 to 43. Invalid route status or IDs zero all
routed outputs before the host observes and poisons the failed layer.

The first learned-route kernel was correctly rejected: a scalar six-pass scan
took 0.318 ms/layer and moved the removed idle interval into GPU work. A stable
eight-simdgroup reduction computes direct F32 GPU-schedule `sqrt(softplus)`
selection scores,
breaks ties by lower expert ID, and takes 0.0183 ms/layer in the same 100-dispatch
release probe. Hash routing takes 0.0943 ms/layer and occurs in only three
layers. Production-shape indexed-versus-static projection ratios are 1.017 for
IQ2_S gate/up, 1.029 for IQ3_S gate/up, 1.047 for IQ3_XXS down, and 1.034 for
MXFP4 down; every output bit is identical to the default fast static IQ2/IQ3
configuration or the sole MXFP4 static kernel.

Paired warmed packets place the new singleton path around 51.5-54.9 ms at depth
128 and 52.3-52.7 ms at depth 512, versus 59.498/59.865 ms before routing moved
to the GPU. A command-timestamp packet measures about 43.0/43.9 ms of aggregate
GPU work and 2.5/2.4 ms of CPU encoding at the two depths. The curve remains
flat and delivers roughly 19 tokens/s; current llama.cpp remains ahead at about
27.6 tokens/s, so this is a promoted seam removal rather than the S6 throughput
endpoint.

The latest five-sample singleton arrays are
`[56.411,54.937,56.051,54.550,51.702]` at depth 128 and
`[52.431,52.709,53.850,53.162,52.624]` at depth 512. The paired command-profile
arrays are `[64.775,51.773,50.918,52.924,51.902]` and
`[61.166,53.058,52.280,54.473,53.255]`; the first restored observation in each
series remains visibly cold and is retained rather than discarded.

The GPU numerical schedule preserves all 258 routed expert IDs, all 21 CSA
selected lists, and every rank-512/513 row ID at position 3070. Route weights
move by at most 2.217e-5 absolute / 8.870e-5 relative; downstream CSA scores and
cutoff margins move by at most 2.822e-4 and 4.840e-5 without changing a consumed
decision. Two fresh transcripts are byte-identical at SHA-256
`e9d17dc01a3b05057020d552e1bd81359c5a27ed8b60495b50a40f0b605b1284`.
The full deep endpoint remains inside the preregistered three-schedule envelope.

The current decomposition command is:

```bash
cargo test --release -p qwen-llm --test deepseek_v4_position_zero_live profile_native_deepseek_v4_single_command_at_128_and_512 -- --ignored --exact --nocapture
```

The per-layer map found no matched cohort with a 1.5x / 5 ms excess: after the
first visibly cold token, ordinary layers cluster near 0.86-0.94 ms, CSA
publication adds about 0.05 ms to each affected layer, and layer 42's final
head adds about 0.9 ms. Attribution therefore advances to encoder-boundary
timestamps rather than assigning attention or MoE cost from layer parity.
Dispatch-boundary counters are unavailable on this M4 Max, so the profiler
rotates four disjoint layer strata across 20 sequential product-warm tokens.
Only the sampled quarter of a token is split; all other layers retain the
production encoder shape. Timestamps are resolved and scaled independently
against each layer command, never across the 43 host-separated commands.

Ten named stages cover attention HC-pre, Q/KV preparation, attention core,
attention output, the attention-to-FFN HC bridge, router, routed experts, shared
expert, MoE combination, and layer tail. The split schedule and both ordinary
controls produce identical final vectors at SHA-256
`4a76e44320e1bfc6fb6a23ff2e5009d13d74ce9bdf5105d3a96ccabbb782e638`
near context 128 and
`e0c536148b2a1fb6b41623a2240d5ba5a8857c61f6813752a03d41d8a0207925`
near context 512. Median raw counter coverage is 1.0000. Four-token cadence
reconstructed GPU medians are 43.549/45.242 ms; sampled encoder gaps account
for only 0.401/0.394 ms.

| Four-token cadence reconstructed GPU stage, ms/token | ctx~128 | ctx~512 |
|---|---:|---:|
| attention HC-pre | 2.186 | 2.187 |
| Q/KV prepare + compressor frontier | 8.190 | 8.236 |
| dense attention core | 3.891 | 4.733 |
| inverse RoPE + output A/B | 8.059 | 8.226 |
| HC attention-post + FFN-pre | 2.384 | 2.396 |
| router projection + route | 0.992 | 1.019 |
| six routed experts | **14.030** | **13.929** |
| shared expert | 2.575 | 2.583 |
| routed/shared combine | 0.336 | 0.338 |
| layer tail + final head | 1.034 | 1.039 |

The census makes routed experts the highest-value next falsifier. Attention
output reads 3.066 GB of Q8_0 weights at roughly 369-386 GB/s, and the 0.894 GB
shared expert runs at about 346 GB/s. The six selected routed banks read 2.241
GB/token but realize only 159-164 GB/s, consistent with fragmentation across 24
serial gate/up/SwiGLU/down dispatches per layer while leaving quantized memory
access as a competing explanation. Router selection itself is no longer
material.

The next prototype therefore batches all six route slots while preserving each
row's existing accumulation order: one indexed gate+up+clamped-SwiGLU dispatch
and one indexed down dispatch per layer, with slot-private intermediate rows and
the established weighted reduction order. Every `(slot, FFN row)` computes both
gate/up reductions and SwiGLU inside one threadgroup synchronization domain;
the following down dispatch is the only cross-dispatch dependency. It earns a
production switch only if all four routed storage families remain bit-identical
to the serial indexed path, invalid status/IDs still zero every slot, the
routed-expert stage falls by at least 2 ms/token at both depths, and paired
product-warm decode does not regress. Otherwise the serial indexed path remains
the production baseline and the next measured candidates are Q/KV preparation
or whole-token submission.

The prototype clears every promotion gate. IQ2_S and IQ3_S gate/up kernels
retain the exact per-row quantized accumulation and materialize independent F32
gate/up totals before applying the established clamp/SwiGLU expression. One
two-dimensional dispatch covers six slot-private FFN rows; a second dispatch
applies either IQ3_XXS or MXFP4 down projection into the existing slot-major
expert outputs. This removes 22 dispatches/layer, or 946 dispatches/token,
without changing weighted reduction order.

All four production storage combinations are bit-identical to 24 serial indexed
dispatches for both the six intermediate FFN rows and final expert outputs.
Non-ready status zeros all six rows, and negative/oversized IDs zero exactly the
invalid slot before any bank pointer is formed. Production-shape warm probes
move IQ2_S+IQ3_XXS from 0.322 to 0.174 ms/layer, IQ2_S+MXFP4 from 0.326 to
0.190, IQ3_S+IQ3_XXS from 0.320 to 0.175, and IQ3_S+MXFP4 from 0.314 to 0.192.

The live product-warm routed stage falls from 14.030/13.929 to 7.729/7.751 ms
at contexts about 128/512, a 6.301/6.178 ms/token reduction. Reconstructed GPU
time falls from 43.549/45.242 to 37.548/39.301 ms. Ordinary candidate controls
run in about 44.6-47.0 ms at context 128 and 45.9-46.4 ms at context 512, while
the exact control/profile vectors remain `4a76e443...` and `e0c53614...`.
Position zero remains `dc2fd6f1...`; the position-3075/3076 continuation and
terminal causal digest remain `068c670b...`, `bfc09a03...`, and `73d2c01d...`.

The next measured seam was host synchronization rather than another weight
kernel. Ordinary CPU encoding was about 1.7-1.8 ms/token, while
`wall - aggregate_command_GPU - encode_CPU` remained 6.3-7.7 ms at context 128
and 6.5-6.9 ms at context 512, clearing the 5 ms whole-token submission gate.

The preregistered first collapse retained 43 ordered serial encoders inside one
command. It falsified command-buffer submission alone: product-warm medians
were 45.698/48.026 ms at contexts about 128/512 versus 45.370/47.771 ms from a
detached `43463fc` baseline. Keeping that command ownership but removing the 42
intra-command layer encoder boundaries exposes the actual seam. One serial
encoder now carries every layer and all 1,999 position-129 dispatches; product-
warm candidate repeats span 38.235-38.359/38.189-39.131 ms at contexts about
128/512 (25.56-26.19 token/s). Even the slower repeat removes 7.011/8.640 ms,
or 15.5%/18.1%, from the detached baseline. The remaining latency gap to the
pinned llama.cpp 36.137/36.255 ms baseline is about 6-8%, not the prior 26-32%.

The collapse does not trade away layer attribution or fail-stop ownership.
Forty-three immutable route records retain six IDs, six weights, and status;
sparse layers retain visible count, selected count, and status. Payload is
written before ready status, later layers use disjoint views, and the host
pre-poisons every record before encoding so a missing producer cannot inherit a
prior token's success. It bulk-reads and validates records in layer order only
after the command completes. A
failed Metal command reports no prefix; a semantic failure or callback unwind
leaves the session poisoned and publishes only the already verified callback
prefix. These three snapshot-excluded buffers add 2,752 logical bytes.

At position 129 the whole-token and retained 43-command comparator use one and
43 serial encoders respectively, execute the same 1,999 dispatches, and produce
bit-identical logits at SHA-256 `3be7c94d...`. Position zero remains
`dc2fd6f1...`; positions 3075/3076 and terminal causal state remain
`068c670b...`, `bfc09a03...`, and `73d2c01d...`. The exact-prefix pins
`3be7c94d...` and `809bc818...` were also reproduced in the detached `43463fc`
baseline, proving their correction from stale assertions is not attributed to
the command collapse.

The next attribution packet must preserve this one encoder rather than
reintroducing encoder boundaries as the observer. The remaining short-context
target is only about 2 ms/token to llama.cpp parity; far-context Lightning
Indexer scoring remains an independent measured lane.

The target M4 Max reports `stage-boundary=true` but
`dispatch-boundary=false`, so an in-encoder `MTLCounterSampleBuffer` profiler
would be unsupported rather than non-perturbative. The promoted fallback times
only host boundaries and the completed command's independent GPU duration; it
adds no command, encoder, dispatch, or GPU sample. It separates guards/phase,
record reset, command+encoder creation, CPU encoding, commit, the
commit-through-wait envelope, command status, bulk record reads, ordered
validation/callbacks, and causal publication. Host segments plus
`commit_wait_wall - command_gpu` reconstruct `forward_wall - command_gpu`
within 0.005 ms at the medians.

Ten-token product-warm packets settle the host/GPU split:

| Whole-token median, ms | ctx~128 | ctx~512 |
|---|---:|---:|
| profiled wall | 38.089 | 38.637 |
| command GPU | **37.125** | **37.627** |
| total outside GPU | **0.960** | **1.032** |
| CPU command encoding | 0.727 | 0.772 |
| commit/wait minus GPU | 0.215 | 0.247 |
| command+encoder creation | 0.006 | 0.013 |
| record read + validation/callback | 0.003 | 0.005 |
| all remaining named host phases | <0.001 each | <0.001 each |

The before/after ordinary-control medians are 38.126/36.843 ms at context 128
and 37.975/39.028 ms at context 512, bracketing the profiled arms under normal
run variance. Profiled and ordinary schedules retain exact final hashes; a
position-129 gate additionally preserves one encoder, 1,999 dispatches, every
logit bit, and the causal-state digest. No host component owns 0.75 ms at both
depths, and total outside-GPU time is only about 1 ms, so host lookup, record
scanning, and completion synchronization are closed as the next short-context
optimization target.

Two headless Metal captures independently bracket the command interval. The
standard timeline sees singleton compute spans around 34-38 ms. A pinned
`metal-counters` template joins five context-128 and six context-512 whole-token
windows to 234.7/232.2 GB/s GPU read bandwidth, 21.9%/22.6% occupancy,
40.8%/40.9% instruction-throughput limiting, 31.1%/30.8% integer/complex
limiting, and only 11.4%/11.4% compute-launch limiting. These percentages are
not additive time shares, and shader-list metadata is sampling-biased; they do
not authorize a kernel by themselves. They do falsify a global launch-only
story and support a targeted quantized-expert instruction-path falsifier.

The bounded all-slot follow-up rejects four short-context kernel hypotheses.
Removing the cross-simdgroup totals rendezvous remains bit-identical, but the
product command moves to 37.185/36.923 ms at contexts 128/512 versus the
37.125/37.627 ms attribution checkpoint: it does not improve both depths or
clear 0.75 ms/token. Doubling IQ2_S rows per simdgroup worsens isolated gate/up
from about 0.112 to 0.123 ms/layer. A Flash-0731 geometry specialization with
two-byte packed metadata loads saves only 0.008-0.013 ms/layer, below the
0.018-0.020 ms model-free kill gate. Four-lane grid conversion and exact sign
bit application regresses to about 0.157 ms/layer. All prototypes were removed
without a production switch.

The retained profiler now uses deterministic nonzero quant blocks and times
gate/up and down independently. At production geometry, IQ2_S gate/up owns
about 0.112 ms/layer, IQ3_XXS down about 0.062 ms/layer, and MXFP4 down about
0.088 ms/layer. The split is attribution evidence rather than an additive
product trace because each operation is warmed and timed independently. These
results close sub-threshold short-context kernel tuning under the current gate;
the promoted cooperative Lightning scorer reduces the measured 262,144-row
operation from 8.3-8.4 to 2.102 ms per CSA layer, while four-bit radix reduces
mixed selection from about 4.58 to 1.875 ms. Complete terminal attribution now
puts exact tiled HCA at 5.26 ms/layer and about 105.2 ms/token. Promoted online
singleton HCA reduces that to 3.410 ms/layer and about 68.2 ms/token, below the
complete 93.4 ms CSA subtotal. Exact CSA scoring and selection therefore become
the primary far-context optimization lane. The direct FP4 scoring premise that
first motivated that lane is tested and killed by the deeper packet below.

One later refreshed-asset workload found a bounded hole between those regimes,
not a reason to reopen local kernel tuning. Sparse CSA starts at compressed row
513 / token position 2,051, while production retained the scalar
remove-one-worst-row selector until row 1,024. Its work grows as
`(visible - 512) * visible` across that band. Moving the existing exact radix4
selector to the first pruned row takes an identical 2,385-token warmed product
bracket from scalar 6.69/6.68 to 22.62 decode token/s and cuts generation from
4,780.3/4,789.8 to 1,414.7 ms. Prefill remains bracketed, and every generated ID
and logged first-token logit bit remains exact. The scalar, one-bit parallel,
and radix4 schedules also agree from rows 513 through 1,024, including packed
publication cadence. This is a complexity-policy repair at positions
2,051-4,095; the contexts-128/512 bounded KILL and far-context CSA priority both
remain intact.

The first bounded packed-lane experiment separates schedule ceiling from cache
semantics. An eight-simdgroup matrix scorer over already-decoded F16 K and one
F16-rounded query reduces 262,144-row scoring from a 2.102 ms bracket midpoint
to 0.518-0.519 ms/layer across two campaigns; terminal saving is
1.584-1.586 ms/layer and the separate per-layer query conversion costs about
0.008 ms. It also clears the shallower frozen gates and repeats bit-for-bit.
This establishes about 1.585 ms/layer of available schedule budget and
justifies testing whether packed decode fits inside it; it does not prove that
it will. The probe is not the paper contract and has no production caller.

The official packed-semantic shadow now answers that question. Four simdgroups
pack one post-Hadamard row under exact BF16-before-amax semantics into separate
raw-byte value and scale planes. The 68-byte scalar row remains a fixture
envelope, not a cache layout. The first scorer decoded all 64 packed Q heads in
every eight-row K tile and KILLed at 1.634 ms terminal, only 0.492 ms/layer
faster than current. The admitted schedule decodes authoritative packed Q once
into a transient 16 KiB/query F16 unit-value slab while K remains packed in the
matrix scan. Q scales remain separate and block rescaling uses one combined
power-of-two exponent.

Two valid campaigns measure terminal current/FP4/current
`2.127/0.841/2.126` and `2.127/0.841/2.124` ms, saving 1.285/1.284 ms/layer;
candidate p95 is 0.842/0.842 ms. Q pack and unpack are inside the timed arm, and
one-row K publication packing is 0.0134-0.0137 ms. Every primitive byte,
malformed-row status, fixture score, selector decision, tail, offset, and repeat
gate passes. This promotes a test-only schedule, not a cache: synthetic K is
prepacked, status-0 planes are trusted, and F16 snapshot v1 is untouched.

The real-weight status-carrying sidecar closes that synthetic-K gap at the first
sparse boundary. Every CSA indexer row packs from the exact post-Hadamard,
pre-F16 publication seam into split value and scale planes. Publication writes
unavailable/writing/final status around a device barrier; one preflight checks
exact authoritative visibility, all 64 Q heads, and every visible K row before
the packed scorer can expose a selection. Invalid observation remains reportable
without a shadow decision, while invalid counterfactual consumption poisons the
causal token.

One refreshed-asset A/candidate/B campaign advances three fresh sessions through
2,061 forwards. All 21 packed and singleton reports have ready Q/K, count 512,
and selector status zero. Packed masks are exact in 8/21 layers and singleton
masks in 11/21; all remaining differences are one reciprocal rank-512/rank-513
exchange. This keeps the original exact-identity falsifier failed. A separate
FP4-ID/F16-cache counterfactual preserves packed argmax 35 at cosine
0.999999967 / relative RMS 0.000260574 and singleton argmax 201 at
0.999999996 / 0.000091580. Maximum absolute errors are 0.003527/0.001985.

That evidence is explicitly shallow. Selecting 512 rows from only 513 candidates
can differ by at most one reciprocal exchange, so it qualifies the sidecar and
counterfactual plumbing but cannot establish ranking stability as history grows.

Controls repeat bit-for-bit in logits, reports, and transcripts, and their
causal-state digests are exact. The candidate trace binds packed/singleton kind,
position, layer, visibility, and every consumed ID across exactly 21 packed and
42 cumulative singleton CSA layers. Snapshot v1 export/restore is forbidden for
the counterfactual; ordinary restore disables the sidecar and invalidates
capture lineage while preserving an ordinary F16 continuation. The diagnostics
plan includes every sidecar and scratch buffer. At full context it adds
398,481,944 logical bytes without changing the production plan.

This is a diagnostics decision/quality `GO`, not a production or speed
promotion. Every arm deliberately computes both scorers, and positions
2,053-2,060 span only 513-515 visible rows; candidate 47.721 ms versus a
46.593 ms GPU control midpoint is therefore not the terminal packed-shadow
comparison. The next gate was a no-double-score experimental arm with a paired
dual-score audit endpoint. Its result follows; a useful deep/current-product
packet still remains required before cache or snapshot migration.

The bounded no-double-score arm now passes its falsifier. Session authority is
an exhaustive F16-authoritative, paired-counterfactual, or FP4-only-experimental
mode; each forward derives F16-only, paired, or FP4-only execution. Common
query/head-weight preparation is shared, while singleton and packed FP4-only
positions encode no F16 score/selection pipeline. An explicit singleton audit
atomically restores paired scoring without changing FP4 consumption.

Paired A, FP4-only, and paired B remain bit-identical through packed position
2,051, the position-2,052 paired audit, and final position 2,060. Final logits
share SHA-256
`99726fc307b0068e05cd0dd7fec1910752e8709d95a5cfeac827580a90c1f7bd`;
final causal state shares digest
`68122e6f8c0ff2ab4b6265ed7993b28e25d29a92a79b8c6fa338f34977732d5c`;
all 210 consumed CSA-layer selections share source-bound trace
`28ddb115041232ba85e6a03fbd6a36646a6677a6a001f2d6d88a0f5d237b3ce3`.
The paired audit also reproduces the prior 11 exact / ten cutoff-exchange
selector topology.

Without reports in any timed arm, paired/FP4-only/paired GPU medians are
46.738/45.291/46.802 ms. Control drift is 0.136%; FP4-only saves 1.479 ms
against the midpoint and 1.447 ms against the faster control. This clears the
frozen 1.0 ms gate and authorizes collapsed-command engineering, not a speed or
production claim: the schedule remains one command per layer and sees only
513-515 rows.

The collapsed one-command experiment closes that authorization with a decisive
`KILL`. It retains one 512-ID slice and compact completion record per layer,
validates active and inactive slices before trace or commit, and distinguishes
instrumented and collapsed execution while preserving an execution-independent
selection payload digest. A two-layer one-encoder Metal differential proves the
shared scratch and retained outputs exact.

The current-asset product gate pays one lineage prefix to position 3,070, runs
one paired audit and eight unaudited collapsed tokens, then restores two exact
F16 controls from the pre-seal snapshot. All structural gates pass and exactly
189 CSA-layer selections are traced. Quality and speed do not: every audit mask
differs by 24-102 IDs; minimum cosine is 0.983313, maximum relative RMS 0.182029,
and maximum absolute error 3.19737. Candidate GPU/wall medians are
45.768/47.137 ms versus faster controls at 44.312/45.366 ms. Direct FP4-Q/K
selection, paged FP4 K, snapshot v2, and further same-design tuning are closed.
Reopen only for a materially new guarded/mixed premise that clears deep quality
before a product timing campaign.

The pinned llama.cpp depth command is not a free decode-only bracket. Its
`--n-depth` implementation executes `test_prompt(n_depth)` and serializes the
resulting state before timed generation on the first repetition at every new
depth. A 32K row would therefore reintroduce the long cold-prefill loop this
strategy deliberately retired. Defer that external bracket until a reusable
prepared state exists or the comparison itself becomes the gating uncertainty.

The nonperturbative whole-command and historical stage-attribution commands
are:

```bash
cargo test --release -p qwen-llm --test deepseek_v4_position_zero_live profile_native_deepseek_v4_whole_token_breakdown_at_128_and_512 -- --ignored --exact --nocapture
cargo test --release -p qwen-llm --test deepseek_v4_position_zero_live profile_native_deepseek_v4_stage_families_at_128_and_512 -- --ignored --exact --nocapture
```

Gate:

- Packed N=1 preserves position-zero argmax 201 at cosine 0.999999548 and
  relative RMS 0.000951217 against the full b10222 vector.
- Packed `[35, 201, 200, 34]` publishes the first CSA row and preserves argmax
  262 at cosine 0.999999903 / relative RMS 0.000448886. Ordinary singleton
  decode from that retained state preserves position-4 argmax 63,325 at
  0.999999961 / 0.000278771.
- Packed `[35, 201, 200, 34] * 32` publishes the first HCA row and preserves
  position-127 argmax 35 at 0.998357518 / 0.057293913. Ordinary position-128
  decode wraps raw slot zero, preserves argmax 201, and recovers to
  0.998725996 / 0.050462295.
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
- Starting from the durable position-2048 state, a packed four-token chunk
  crosses the first sparse boundary and a packed one-token continuation reaches
  position 2052. The continuation preserves argmax 201 and differs from the
  cooperative singleton path by only 0.000265079 relative RMS. The
  packed causal-state digest
  `03cea8187af6782fa374bf8ce441057d74924880eb6be46439b25982098476db`
  is pinned separately from the singleton reduction history.
- Starting from the durable position-2052 state, one 124-token packed chunk
  keeps sparse CSA active from 513 through 544 visible rows and publishes HCA
  row 16 on its final token. The packed boundary/continuation preserves
  argmaxes 35/201 at 0.995150607 / 0.098514095 and
  0.997212245 / 0.076768115; the immediate continuation recovers both measures.
  The complete checkpoint-producing live packet takes 23.324 seconds cold.
- Starting from the durable position-2176 state, seven 128-token chunks fill
  the third CSA slab and publish HCA row 23 in 26.7 seconds. Position 3071 and
  its continuation preserve argmaxes 35/201; fixed 64- and 128-token partitions
  are bit-identical. The pinned schedule-envelope and decision-transcript gates
  explain the larger singleton-relative drift without relaxing operation
  semantics. Layer-4 native/oracle CSA score cosines exceed 0.99999986 while
  each reversed-row perturbation exceeds both schedules' cutoff margins; the
  independent b10222 schedules repeat that mechanism at layer 6. Exact capture
  and repeat commands plus the complete llm and llama_core instrumentation
  patches are retained with the fixtures. The feature-enabled live producer
  must reproduce the pinned native transcript canonically before publication.
  Fresh durable restore reproduces the endpoint bits, and the release CLI
  reaches that historical 3,073-forward endpoint in 1.79 seconds end to end.
- At positions 2051, 2052, and 3071 the cooperative and legacy singleton
  attention reductions differ in only 28-46 of 32,768 output bits. Maximum
  absolute error is at most 5.83e-11 and relative RMS at most 9e-9. The induced
  deep-logit schedule is deterministic across fresh processes, remains inside
  the pinned b10222 schedule envelope, preserves every consumed CSA selection
  and MoE expert set, and restores the existing causal snapshots without an ABI
  change.
- The dense cooperative swap starts exactly at singleton position one.
  Singleton position zero keeps argmax 201, cosine 0.999999999, relative RMS
  0.000049306, and full-vector
  SHA-256 `33ec463aee992d3b557a58bd5710080d6a42f85ba14c9ceb07d5f772ab1cfd37`.
  GPU route publication later keeps argmax 201, improves relative RMS to
  0.000007494, and establishes its own deterministic full-vector SHA-256
  `dc2fd6f18c5ba761cef7cc39c3ecb364addf6ef6be3b469bf70db6ff27d6f630`.
  The position-4 singleton continuation and named SWA production shape exercise
  the new path; newest-row, local-only, wrapped-CSA, and invalid-count ablations
  re-pin visibility rather than assuming it from the packed implementation.
- Existing packed causal states remain bit-identical because packed attention
  already used this reduction. A newly captured singleton position-2052 state
  is published separately as `target/dsv4-position2052-dense-cooperative.ds4c`
  and changes digest to
  `4caf4cada32e320cabd86c6399e135abe8961c811a856d500443755c26d68fd6`:
  its prefix is unchanged, raw-cache changes are confined to newly executed
  positions 2049-2051 in layers 1-42, and published changes are confined to the
  newly emitted CSA row 512 with no older or HCA publication drift. Historical
  snapshots still restore under ABI v1, and packed row-16/position-3072 digests
  remain exact.
- Restoring that certified position-3072 state into a request-sized session,
  then crossing row 768 with one four-token packed chunk, preserves the existing
  endpoint hash, agrees with singleton execution within 0.001869 relative RMS at
  the boundary and 0.000724 on continuation, and restores that continuation
  bit-for-bit. This makes the former ceiling a tested allocator property rather
  than a new fixture ladder.
- At exact full-context capacities, far-index compressor publication,
  score/mask generation across 262,144 CSA rows, deterministic top-512
  selection, selected CSA attention, dense HCA across 8,192 rows, snapshot
  sizing, and admission execute at terminal row counts and physical strides.
  Production head counts and widths are covered by separate composition gates;
  together they avoid replaying 8,192 semantically redundant packed chunks.
- A real-weight session constructed directly at position 65,663 executes the
  selector inside every sparse CSA layer and the first tiled HCA query. Its
  explicitly initialized zero causal state remains exact across scalar/bitwise
  and cooperative/radix4 schedules at full-logit SHA-256
  `1c0f5e0475314e693bfe0664b5454a2ece26d9a5913f9a5218cdf39da59582d4`
  and causal SHA-256
  `03f15887db83e4baf0ad5ba66f95b3e2a7fe461858d92aebe9f0b3009f83254e`.
  The earlier synthetic hashes relied on uninitialized direct-jump state and
  are retired as fixture-undefined rather than interpreted as inference drift.
- A successful advance revokes the session's current logits and final hidden
  observation; vectors already copied to the host remain ordinary owned values.
  A callback unwind from retained position 2 leaves that position unchanged,
  keeps observations invalid, sets poison, and rejects subsequent decode.

Broader S6 work remains:

- The exact 32-group, 18-dispatch full selector clears its terminal gate. Mixed
  GPU/wall medians are 0.674/0.818 ms versus faster radix4 controls at
  1.874/2.040 ms; all-tied medians are 0.907/1.075 versus 2.025/2.203 ms.
  Deterministic partition quotas preserve lower-row ties and cache-order IDs;
  private mask/ID state is structurally validated before exact publication or
  first-K fallback. The crossover now freezes explicit-opt-in eligibility at
  Q=1/K=512, `visible <= capacity <= 262144`, `visible >= 196608`, and
  `visible >= capacity - capacity / 4`. At the boundary, all-tied GPU/wall
  savings remain 0.627/0.638 ms for max capacity and 0.641/0.635 ms for the
  decimal-million capacity. The bounded singleton integration owns admitted
  scratch and non-reusing generation state behind that seam. Preserve F16
  scoring and keep packed, shallow, and ranked-output routing current.
  The diagnostics-only collapsed FP4 implementation remains negative evidence.
- Fuse mHC split/Sinkhorn/collapse, compressor projection/store, and shared-KV
  sparse attention only after packed attribution identifies one of those seams.
  The first narrow packet kills chronological publication as a standalone
  N=128 target: its normalized 73.273 ms envelope has a 54.163 ms/5.099%
  uncertainty-adjusted lower bound, below the frozen 150 ms/15% floor. Next
  isolate attention body plus output from the post-row boundary through
  `encode_output`; split them only if the combined target clears. Then split the
  unchanged roughly 688 ms post-route span before widening another grouped
  format. Whole-token submission, per-layer route/selector failure records, and
  verified-prefix callbacks are promoted. An asynchronously immutable
  SSD-streaming ticket remains a separate product contract.
- Preserve promoted online singleton HCA and the exact packed/legacy tiled
  differential. Defer heads8/rows16 split-K while HCA remains below CSA; reopen
  HCA only if attribution returns it to the lead. The scalar scorer and bitwise
  selector remain executable differentials for the multi-group selection lane.

Gates:

- Decode throughput is at least current llama.cpp Metal on the same hashes,
  request, context, and memory policy.
- Fresh TTFT and warm decode have named phase attribution.
- Representative 32K/65,536 product workloads validate prompt throughput;
  durable or synthetic deep states measure decode at larger indices without
  making a million-token cold-replay gate part of ordinary development.
- Full-context memory and operation gates remain green while optimization
  replaces serial or recomputed work; the exact 1,048,576 ceiling does not move.
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
| Converter refreshes change routed dtypes under the same recipe name | Census every artifact by role; require generic packed and all-slot singleton coverage before admission |
| Existing Qwen session becomes branch-heavy | Keep DS4 model/session/forward types separate |
| CSA indexer dominates decode | Attribute full-history score and top-k before changing tile shapes |
| Generic Metal fallback hides memory blowups | Require explicit scratch accounting and resident-memory gates |
| Activation QAT differs across references | Freeze operation vectors and distinguish semantic BF16 from packed-cache promotion |
| No tiny official DS4 checkpoint exists | Build operation fixtures and a synthetic tiny family fixture; do not use passthrough layers as an exactness oracle |
| Prompt template differs across runtimes | Port the official 0731 encoder and compare rendered bytes, not rendered intent |
| Asset revisions drift | Pin all shard hashes and checkpoint metadata in every benchmark packet |
| DS4 variant churn | Freeze 0731; add another descriptor only for a measured quality gain and material schema change |

## Immediate next work

1. Treat 1,048,576 forwards as the closed structural correctness space. Keep the
   certified position-3072 checkpoint as the ordinary semantic full-model loop,
   the synthetic position-65,663 token as a mechanical integration gate, and
   terminal operation properties as the deep-index gate. Do not resume fixture
   ladders or require a cold million-token replay without a new discontinuity.
2. Preserve the completed representative dense, sparse, tiled-HCA, and terminal
   attribution packets. Use durable or constructed deep states to measure the
   mechanism under study directly; optimization, not another position unlock,
   remains the critical path to useful long-context inference.
3. Preserve the promoted dense-attention checkpoint: singleton position zero
   remains on its exact legacy lineage; SWA uses cooperative attention
   thereafter; CSA uses it through position 2050 before sparse selection starts
   at 2051; and HCA uses it through position 65,662 before online singleton
   attention starts at 65,663 under its explicit envelope. Packed HCA retains
   exact tiled attention above that boundary.
4. Treat short-context execution as bounded near-parity rather than the primary
   optimization lane. The one-command path is about 37-38 ms of GPU work versus
   llama.cpp's 36.1 ms total at depths 128/512; barrier, row-shape, geometry,
   and vector-decode probes all miss the 0.75 ms/token two-depth gate. Preserve
   the separate first-pruned-row complexity firewall: sparse CSA dispatches the
   exact radix4 selector from row 513 rather than entering the retired scalar
   positions-2,051-through-4,095 cliff.
5. Preserve the cooperative Lightning scorer and four-bit radix selector as
   executable differentials. Direct FP4-Q/K replacement is closed by the
   position-3,070 quality and speed KILL; keep its diagnostics and packed matrix
   shadow as negative evidence without migrating paged K or snapshot v2. The
    exact multi-group full selector now clears at 32 groups with exact masks,
    cache-order IDs, status fallback, and more than 1.11 ms/layer terminal GPU
    and wall saving. Its crossover freezes an experimental boundary at 196,608
    visible rows, at least three-quarters of no more than 262,144 physical rows.
    The bounded singleton integration now owns admitted session scratch and
    non-reusing generation state behind a hidden opt-in. A current-asset,
    real-weight synthetic zero-causal-state fixture at position 786,431 restores
    through snapshot-v1 and preserves exact fixture output/state plus a separate
    untimed 21-layer decision trace while saving 13.867/17.573 ms GPU/wall
    against the faster control. This is not real-prompt continuation evidence.
    Keep default, packed, and wider-device routing unchanged; do not repeat the
    5.43 GB restore campaign without implementation, asset, or device drift.
6. Keep the external depth bracket and packed-prompt optimization as independent
   lanes. `llama-bench --n-depth` performs the full cold prefix at each new
   depth, so do not pay that loop until a reusable state or gating cross-engine
   question justifies it. The deterministic packed GPU route/schedule topology
   clears its microproof but fails integrated promotion: all 86 same-input route
   ID records are exact, while small GPU weight deltas amplify to 4.3% packed
   relative RMS and a controlled Rust-weight hybrid restores output bits plus
   every recorded state digest. The uninstrumented candidate also regresses one
   warm comparison by 2.15%; the later audit-inclusive timing is not an isolated
   seam measurement. Keep the topology under diagnostics and preserve ordinary
   Rust routing. The independent grouped-compute lane now promotes the 25
   current-asset IQ2_XS/IQ2_XS/IQ3_XXS layers on Apple M4 Max: compact
   expert-major tiles feed grouped gate/up + SwiGLU and down/scatter kernels
   while preserving current-asset output bits and recorded state identity at
   N=12/32/128. N=128 R5 wall medians improve 30.409%, while traced R3 medians
   improve 35.91% post-route GPU and 21.64% summed packed-command GPU. One R3
   wall bracket misses at 12.2%, so this is not a per-run guarantee. Retain the
   isolated rollback, capability/dtype fallback, and 6,291,456-byte admitted
   scratch. The separate all-IQ3 widening is exact but remains `HOLD`: a mapped
   IQ3 projection plus exact gate/up arena aliases execute all 16 layers in four
   test-only dispatches without another allocation. Model-free stagewise and
   current-asset N=12/32/128 outputs and state are bit-exact. Two R5 campaigns
   are rejected for control drift; the sealed balanced R8 stabilizes controls
   at 0.721% but misses candidate stationarity at 6.050% against a 5% gate. A
   separate traced bracket records 9.929% total packed GPU saving and 56.534%
   in the affected layers with unchanged-layer time flat; it is attribution,
   not a balanced promotion gate. Retain the implementation under diagnostics,
   authorize no same-condition retry, and do not build fused IQ3 gate/up now.
   The first pre-expert attribution removes chronological row publication from
   the standalone queue. A three-pass all-layer packet preserves bit-exact
   logits/hidden and continuation logits plus matching causal identities,
   tokens, and 47,990 complete dispatch geometries. Controls are stable within
   2.868%; sampled topology perturbation is at most 1.806%. Chronological work
   occupies 6.992%/6.818% and normalizes to 74.620/71.925 ms; its frozen lower
   bound is only 54.163 ms/5.099%. The earlier broad profiler is retained only
   as negative evidence because adjacent Metal pass envelopes overlap and its
   layer-41 ambiguity gate fails. No eight-stage live lane remains. Next measure
   attention body plus output together, then attribute the unchanged roughly
   688 ms post-route span. Keep MXFP4 down and GPU route arithmetic unchanged
   and do not revive a monolithic FFN kernel.
7. Pursue streaming snapshots and the remaining DSML tool/developer encoder as
   independent product lanes, not blockers for inference optimization.
