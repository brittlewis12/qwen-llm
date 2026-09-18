# K2 Horizon implementation plan

Status: profile/binder, native tokenizer, and tiny CPU equation reference implemented;
no full-model K2 execution or runtime dispatch implemented.
Base: `main` at `4d8716ab`. Worktree: `/Users/tito/code/qwen-llm-k2-horizon`.
Branch: `feat/k2-horizon`. Decision date: 2026-09-18.

## Progress

First packet: explicit library-only 7B config and descriptor binding, stage-aware
context/theta, exact tensor inventory/storage checks, and logical F16/Q8 cache
sizing. Eight model-free CPU tests pass; the explicit CPU/header-only test also
binds the downloaded Q8 artifact. Family dispatch and all execution lanes remain
unchanged. Subsequent packets supply scalar math and tokenizer conformance.
Pre-commit adversarial review found no blockers (see review record).

Second packet: native K2 tokenization with NFC after added-token partitioning,
dedicated Unicode splitting, cumulative normalized-input bounds, and explicit
single-sequence BOS/pair-separator policy. Independent HF fixtures cover 35
inputs each for pinned pretraining and posttraining tokenizer revisions; both
vocabulary-only test containers pass all token-ID/BOS/decode comparisons, and
the real Q8 header passes the posttraining corpus. Five regular K2 tests include
256 Unicode splitter property cases; three existing Qwen/JoyAI splitter tests
also pass. No FFI dependency change. Tokenizer fixture generation downloads
metadata only and does not convert model checkpoints. Authored span mapping,
runtime profile identity, and application-lane integration remain later work.

Third packet: bounded synthetic CPU forward reference and independently generated
NumPy 2.2.6 batch-causal fixtures. Two layers exercise residual width different
from query width, four-group direct-gamma RMS, full split-half RoPE, contiguous
GQA, sequential residual SwiGLU, and untied grouped-norm readout. Three theta/base
pairs each cover F32/F16 cache semantics, every prefill/continuation split, and
single-token appends. Every attention read uses rounded stored K/V, including the
current token. Sessions borrow immutable weights and commit only after all
tokens/layers/readouts succeed. Seven CPU tests cover equation agreement,
validation, rollback on nonfinite readout/cache overflow, and fixture sensitivity
to seven independently corrupted equations (not Rust mutation testing).

The oracle uses F32 activations with F64 reduction/rotation/softmax intermediates;
it is not a CUDA BF16 or Metal numerical ABI. F16 values are round-tripped through
F32 containers, not an implemented GPU cache layout. Tiny hard limits prohibit
full-model CPU deployment or dequantization. Model-checkpoint parity and Metal
qualification remain separate outstanding gates.

The user subsequently authorized autonomous implementation, local commit
checkpoints after review, and one background GGUF download. GPU/shared-server
coordination restrictions are unchanged. The download completed with verified
size/SHA256 at `~/models/K2-Horizon-7B-Q8_0.gguf`; IFM currently offers BF16 only
and the inspected Unsloth inventory contains no K2 Horizon conversion.

Artifact: `abenzerps/K2-Horizon-7B-GGUF` at
`a5094087a5a55c2de80264c11504d8ca95a022ff`, 9,573,964,160 bytes, SHA256
`5a98a289aba5c8c99ef05c9261287f19e8f586fd679bab45b632eb86b47809bf`.
The publisher declares HF source revision
`2c9659a84c4eea6f9f60462221fe762c8c84d75c`. That declaration is provenance, not
independent proof of quantization fidelity. Header binding finds 254 Q8 matrices
and 73 F32 norms, 9,562,505,216 payload bytes; no weights are executed.

## User contract

- Bring up dense K2 Horizon 7B first, directly. MoVA 36B-A4B follows only after
  the dense integration is useful. Other sizes and Uno are not prerequisites.
- Treat compatible intermediate checkpoints as first-class models in `run`,
  `serve`, `bench`, and forward-only `lens`, not as a later release exception.
- Use a native Rust tokenizer. Reuse the existing BPE core; qualify against
  independently pinned HF tokenizer behavior. llama.cpp may be another oracle,
  but a production FFI dependency update is not part of the plan.
- Lens includes captures, readouts, externally fitted transport application,
  and forward interventions. No local fitting, backward passes, or per-family
  VJPs. Fitting happens separately on CUDA/BF16 safetensors.
- Checkpoint-to-GGUF conversion is outside current scope. Specify its required
  metadata/provenance without implementing a converter or acquiring the fleet.
- Improve KV representation without silently dropping context positions. Keep
  an F16 scientific control and identify any approximate cache execution.
- Autonomous implementation, reviewed local commits, and the one Q8 download
  are authorized by the subsequent implementation request. GPU execution,
  shared-server changes, additional checkpoint downloads, and pushes are not.
  Gate live validation under the existing coordination rules.

## Architectural boundary

Create a separate K2 family/config/binder and a shared K2 block implementation.
Do not stretch ordinary Qwen's GDN/gated-attention graph into a K2 alias. Reuse
existing tensor storage, quantized GEMV/GEMM, GQA, RoPE, grouped reduction,
sampling, retained-model ownership, and request infrastructure where their
contracts actually match. No universal graph framework or repository-wide
abstraction rewrite is a prerequisite.

The existing `loader::Model` and `Runtime::load_opened_gguf_with_intent` are
ordinary-Qwen paths, not the implementation base for K2 model/session ownership.
Follow the separate-family pattern already used by Muse/DeepSeek, sharing
low-level primitives rather than their model-specific runtime assumptions.

7B geometry: 36 layers, hidden 4096, FFN 12288, Q/KV heads 32/8, head dimension
128, four contiguous RMS groups, model vocabulary 250624, untied embedding/head.
Conventional sequential pre-norm residuals, full causal GQA at every layer,
SwiGLU, direct-gamma RMSNorm with epsilon 1e-6, no Q/K norm, no attention output
gate, no MoE, and no recurrent state. Final readout also uses grouped RMSNorm.

The inspected stage configurations differ in real execution parameters:

| Branch | Context | Default RoPE base |
| --- | ---: | ---: |
| `pretrain_1100000` | 8192 | 500000 |
| `mid_1_55000` | 32768 | 1000000 |
| `mid_2_25000` | 131072 | 10000000 |
| posttrained `main` | 524288 | 10000000 |

These observations do not establish every wildcard checkpoint's contract.
Resolve fixture/source branches to immutable revisions. Validate actual graph,
dimensions, positional fields, tokenizer, tensor roles, and storage. Do not
force final-model constants or gate compatible weights on a compiled hash list.

Separate architecture admission, artifact identity, and request capability.
Artifact identity binds the actual GGUF shards, source revision when known,
conversion lineage, config/tokenizer identity, and numerical profile. Same
geometry is not same model. Unknown provenance is not fabricated provenance.
Track declared checkpoint context separately from the backend's qualified
execution limit and the request's allocated capacity. A valid 512K config must
not imply that an incomplete short-context backend can execute 512K.

Raw/token-ID/teacher-forced operation must not require a chat template. Chat,
tools, and reasoning need a separately validated checkpoint-appropriate profile;
neither a model name nor a converter-inserted template proves capability.

## Milestones and acceptance evidence

### M0: Stage-aware profile, binder, and numerical contract

Deliver a 7B config/profile, GGUF tensor-role binding, checked memory plan, and
small model-free forward fixtures. Specify direct-gamma grouped normalization,
full-head RoPE coordinates, GQA grouping, residual order, activation/cache dtypes,
and accumulation/cast conventions. Name separate HF and GGUF oracle identities
rather than treating differing reference numerics as interchangeable.

Acceptance: synthetic pretrain/midtrain/final metadata all resolve the intended
context/RoPE without requiring chat. Malformed, missing, unsupported, oversized,
or inconsistent tensors/metadata fail before allocation. Fixture tests include
nonzero/different Q/K positions, final grouped norm, and optional-capability
rejection. Intermediate-stage coverage starts here, not after final release.

Do not register an architecture as runnable until its execution dispatch exists.
Discovery/inspection support and execution capability are distinct.
The first packet is only the profile/binder subset of M0, exposed through an
explicit library API and synthetic tests. It does not enable runtime discovery
or model selection. Nonzero-position and grouped-normalization math fixtures
belong to the subsequent CPU-math packet; profile completion is not all of M0.
Represent valid configuration, implemented execution capability, and unavailable
required facts separately. Missing provenance does not invent lineage; missing
required math metadata must not silently select a default.

### M1: Native tokenizer and input identity

Reuse BPE merges, byte decoding, and safe special-token handling. Add the exact
K2 Unicode splitting/normalization contract. HF fixtures cover NFC ordering,
added-token precedence, combining marks, ZWNJ/ZWJ, case-folded contractions,
digit groups, whitespace, padded vocabulary, BOS insertion, stop sets, and
token-piece bytes. Exercise pretraining and posttraining identities separately.

Acceptance: pinned independent input/token/byte fixtures pass; existing native
tokenizers remain unchanged. Lens authored-byte/normalized-byte/token spans have
an explicit mapping or precise rejection for unsupported bindings. Exact token
IDs may unblock M2 in parallel, but do not replace delivered text support.

### M2a: Dense F16 correctness slice, raw run and bench

Implement one K2 block path used by singleton decode and packed prefill. Keep
weights native for admitted formats, including row-local/native embedding
lookup; never silently expand large matrices. Start with F16 KV and correct
head-dim 128/group-4 GQA. Admission prices request capacity, scratch, weights,
and temporary state rather than allocating the declared maximum by default.

Deliver raw `run` and `bench` through the new family. Preserve fresh TTFT,
inter-token latency, memory, and cold/model-ready timing boundaries.

Acceptance: model-free primitive/layer fixtures first. In a separately cleared
live-validation window, require layerwise activation/logit/state agreement,
multi-position continuation, packed-versus-serial comparisons with declared
tolerances, and native residency reconciliation. Distinguish implementation
evidence from per-artifact qualification; absent converted checkpoints remain
an explicit validation gap, not falsely claimed support evidence.

Initially limit execution to the actual implemented attention capacity. Existing
ordinary Qwen v4 attention is head-dim256. Muse has head-dim128 kernels, but its
optimized online path requires Q32/KV2/group16, not K2's Q32/KV8/group4. A generic
materialized-score fallback has a bounded position limit and is only a
short-context correctness bridge. Reusing either family's host wrapper unchanged
does not establish K2 attention support.

### M2b: Bounded-scratch long-context and packed K2 attention

Adapt existing tiled/online attention algorithms behind a K2-compatible
head-dim128/group4 contract. Preserve causal absolute positions across packed
chunks, K/V strides, GQA mapping, masking, and reduction semantics. Reuse
primitives and scheduling ideas, not ordinary-Qwen packed eligibility rules.

Acceptance: nonzero-base packed fixtures, context/partition/tile boundary cases,
per-head output comparisons, scratch upper bounds, and multi-token continuation.
Keep unqualified lengths gated explicitly. Full declared-context admission and
numerical validation are distinct from a full-length performance run; report
which evidence exists instead of implying all three from one test.

### M3: Forward-only lens on the same execution

Wire selected-site captures, plain grouped-norm/head readouts, imported matrix
application, and ordered forward interventions into the same K2 block path.
No fitting fallback or backward interface dependency. Missing/incompatible
assets fail before GPU execution. Do not create a second research forward graph.

Bind assets to source checkpoint/site/coordinates/orientation/dimensions and
content digest. Record target artifact, weight/cache precision, positional and
tokenizer profile, and serial/packed topology. Intentional cross-checkpoint or
BF16-to-quantized transfer needs explicit recorded policy, not silent acceptance.

Define a bounded forward-only asset reader whose schema does not require a
local fit job or a Qwen3.6 release. Existing `full_lens.rs` contains fixed Qwen
geometry, and `lens_run/lenses.rs` consumes completed fit-token artifacts. Reuse
their safe I/O/validation techniques where appropriate, not those admission
contracts. Validate sizes, finite values, digest, matrix orientation, and
supported sites on the host before allocating/uploading selected matrices.

Acceptance: no-op controls, selected-site capture identity, final-readout checks,
intervention ordering, imported synthetic matrix orientation, invalid-asset
rejection, and raw inputs without chat. Tests must prove K2 fitting remains
unavailable. Do not make lens depend on completion of the chat/tool parser.

### M4: Serving and checkpoint-facing request integration

Deliver completion-style serving for base/intermediate checkpoints, then
checkpoint-appropriate chat/reasoning/tool rendering and incremental parsing.
Use the same tokenizer, forward path, artifact/profile identity, and stop policy
as run/bench/lens. Keep retained model ownership separate from fresh sequence
state. Define cancellation/failure cleanup and compatible snapshot identity.

Override both request rendering and output protocol: `GenerationBackend`
defaults to Qwen ChatML and Qwen tool parsing. Reusing HTTP transport alone must
not inherit those defaults. Preserve raw-input intent through request parsing;
unsupported instructions/messages/tools should receive explicit errors rather
than be silently wrapped or ignored. Token-ID inputs remain guaranteed for
local run/bench/lens fixtures; this plan does not invent an HTTP token-ID API.

Snapshot compatibility binds the actual checkpoint plus config/positional and
tokenizer identity, cache dtype/layout, committed prefix, and numerical ABI.
Logical versus physical capacity must be priced and validated independently;
topology-dependent numerical changes must not accidentally reuse incompatible
state. Snapshot/cancellation behavior must be tested for each advertised lane.

Acceptance: lane-equivalent rendered bytes/token IDs, raw checkpoint requests
without chat requirements, BOS exactly once, partial-tag parsing, stop handling,
request isolation, capability errors, and restore/continuation checks for any
advertised reuse feature. Checkpoints without validated tools must not advertise
tools merely because final-release templates are available.

M0-M4 together define the first complete 7B product milestone. A working CLI
completion alone is not completion of the user's requested scope.

### M5: Qualified compact KV representation

First adapt Q8_0 cache append and inline-dequantizing attention to K2. Existing
Q8 attention in `metal/attn.rs` hardcodes head-dim 256 and Q8 GQA groups 6/8;
K2 is 128/group 4. Do not bypass guards or enable `QWEN_KV_Q8` unchanged.
Cover packed prefill, continuation, snapshot ABI, and lens precision records.

Acceptance: exact memory accounting, no full-history float expansion, numerical
and activation/readout error checks, long-range tasks, and whole-request latency
versus F16. Qualify representative earlier checkpoints separately. Capacity
savings and speed wins are different claims. Promote only qualified defaults;
retain explicit scientific controls without proliferating experiment switches.

Cache-format design begins at M0; Q8 implementation need not block M3/M4. More
advanced K/V-specific grouping, rotations, codebooks, outlier treatment, or
redundancy exploitation are separate measured candidates, not prohibited by
choosing Q8 as the first baseline. No silent eviction/truncation/rolling window.

### M6: MoVA extension, after dense usefulness

Reuse the dense family's graph infrastructure and indexed expert primitives.
Add the 48-layer profile: first three dense, then independent FFN top-8/100 and
value top-4/64 routes, additive shared FFN expert, selection-only correction
bias, normalized sigmoid weights scaled by 2.5, and per-value-expert SiLU before
mixing. Cache mixed V, not each expert's V. Attention remains ordinary GQA.
Apply the beta-ln(2) softplus attention gate, including leading dense layers.

Resolve router partition/rounding provenance explicitly before selecting an
oracle. Acceptance adds route-level fixtures, mixed-V continuation, expert
residency, two independent packed routing paths, and lens site semantics. Do
not assume a selector tuned for another family's expert counts is appropriate.

## Memory and research schedule

For 7B, single-sequence retained K/V is
`36 * 2 * 8 * 128 = 73728 scalars/token`, or 144 KiB/token in F16.

| Context | F16 | Q8_0 (34 bytes/32) | Illustrative Q4_0 (18 bytes/32) |
| --- | ---: | ---: | ---: |
| 32768 | 4.50 GiB | 2.39 GiB | 1.27 GiB |
| 131072 | 18.00 GiB | 9.56 GiB | 5.06 GiB |
| 524288 | 72.00 GiB | 38.25 GiB | 20.25 GiB |

Block scales are included; page padding, tails, scratch, weights, captures, and
snapshots are not. These are capacity calculations, not measured allocations or
Q4 quality claims. MoVA has the same KV width and 48 layers: retained KV is 4/3
as large. F16 is a control, not a claim of bitwise CUDA BF16 equivalence.

Potential later lens optimization: fixed-input teacher-forced layer-major
execution can discard completed layers' K/V instead of retaining every layer
for future decode. It trades that state for full-sequence residual buffers and
possibly streaming/recomputation. At 512K, one layer's F16 K/V is 2 GiB and one
F32 residual matrix is 8 GiB; neither is a total-memory promise. Use chunked FFN
scratch and selected captures. This schedule is not required for initial lens,
must share block math, and cannot promise cheap continuation with discarded KV.

## Source and reuse anchors

- Local family registry: `crates/qwen-llm/src/model_family.rs`.
- Existing tokenizer/BPE: `crates/qwen-llm/src/tokenizer.rs`.
- Native quantized dispatch: `crates/qwen-llm/src/metal_forward/dispatch.rs`.
- Existing attention: `crates/qwen-llm/src/metal/attn.rs` and
  `crates/qwen-llm/src/muse_glimmer_metal.rs`.
- Existing lens contract: `docs/LENS-RUN.md`, `crates/qwen-cli/src/muse_lens_run.rs`.
- Existing server boundary: `crates/qwen-cli/src/serve/backend_muse.rs`.
- IFM donor revision: `42adf019f76013dac873b5b43950d54d5ab27216` in
  `MBZUAI-IFM/llama.cpp`, particularly `src/models/k2-horizon.cpp`,
  `conversion/k2_horizon.py`, `src/llama-vocab.cpp`, and `src/unicode.cpp`.
- HF configs: `IFM/K2-Horizon-7B` branches listed above. The audit inspected
  branch contents; immutable fixture pins are still M0 work.

## Review status

Fresh read-only review session: `01a0b4e8-1183-7780-98fd-4a2038bf2504`
(`luna`, high effort). Initial verdict: revise before implementation. The
source-backed loader, attention, lens-import, and serving-default objections
are incorporated above. See `docs/K2-HORIZON-REVIEW.md` for disposition and the
first bounded packet. Main's unrelated dirty files remain untouched.
Closure review: **proceed with the first packet**, not approval of implemented
runtime behavior. The remaining documentation/API boundaries are incorporated.
