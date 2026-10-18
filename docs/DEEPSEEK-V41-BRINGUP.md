# V4.1 Flash 1017 bring-up

## Current status (2026-10-18)

The separate `qwen deepseek-v41` runtime now executes old REAP-272E with the
original Engram shards. Routed experts remain MXFP4; arithmetic is NativeF32,
not reference activation-quantized FP8. This does not admit 1017 weights.

- Converted weights persist by default; measured cache-hit loading is about
  7 seconds with zero source payload reads, rather than rewriting 221 GiB.
- Opt-in `--gpu-step` retains two CPU Engram boundaries and uses three command
  submissions per token. On the 256 GiB M5, warm short-chat forwards take about
  59-60 ms; a 10K-token capture takes 862.7 seconds (87.4 ms median forward).
- Seven-token, 40-layer CPU comparison passes. Independent teacher-forced
  attention replay at positions 2046/2047 and 9998/9999 matches all 32 selected
  ID sets exactly; worst output absolute error is 2.89e-5. These positions exercise
  index selection, not candidate pruning, which starts at 16,385 tokens.
- Real JSON/SSE, multi-turn, reasoning, tools and disconnect recovery pass on
  the GPU path. Stats-enabled serving reports per-token process page-ins and IO.
  The 10K capture sees at most two page-ins per token; this is not a guarantee
  against eviction under other workloads. No residency set or wiring is enabled.
- The frozen eight-case pilot completes in 70.3 minutes: 12.7 input tokens/s
  and 11.8 generated tokens/s. Two independently blinded reviews score this
  configuration 17/32 and 16/32, versus Maya's 22/32 in both. Four cases exhaust
  the frozen output budget, three without a final answer. Completed-case scores
  are competitive, but this is not an established intelligence-ceiling win.
  Maya's retained run takes 3.0 minutes with substantially fewer generated tokens;
  these are whole-configuration observations, not a causal quantization test.
  Resolve the reasoning-policy/output-budget question before funding CED or more
  kernel optimization. The pilot observes 3,406 process page-ins over 51,390
  native steps, at most two per step, without a residency set or limit change.

Named fixtures and ignored short/long oracle witnesses live in
`crates/qwen-cli/tests/deepseek_v41_oracle.rs`; the long witness regenerates a
10K capture and therefore needs exclusive model execution time. Both short
replays, the tokenizer witness and the registered long replay pass on the M5.
Full contracts pass. The shared-family run passes 94 witnesses, with two missing
Qwen assets; three DS4 qualification performance facets still fail (route digest,
matrix timing spread, grouped-expert equality). This is not a blanket
qualification pass. The following sections retain the historical checkpoint
boundaries, not the current scope.

## Target and checkpoint boundary

Target: useful text sessions on a 256 GiB Apple Silicon worker with a resident
backbone and bounded disk-backed Engram reads. A potential 1017 REAP checkpoint
at roughly four-bit expert storage is the capacity opportunity, not an assumed
speedup or established quality win. No large checkpoint download, system memory
limit change, or V4.1 execution is part of this first checkpoint.

Implemented in this checkpoint:

- Separate model-free `deepseek_v41::TextConfig`, explicit base-384 and potential
  REAP-272/256 geometries, logical expert shapes and packed payload arithmetic.
- Strict pinned text metadata, quantization and special-token validation; all
  four cache owners, eight selection producers, and candidate ownership.
- Checked publication eligibility at pair/context boundaries. This is a plan,
  not committed runtime state or evidence that preceding pair inputs exist.
- Offline source integrity checks and executable ownership/activation-dispatch
  controls extracted from the pinned Python AST. Numerical operations are stubs.
- Fixture-identity selection in the existing model-witness runner, so completion
  gates do not depend on someone remembering relevant test-name substrings.

Not implemented by the first checkpoint: full-model tensor binding, converter
admission, Engram hashing or gathers, numerical kernels, protocol adapter, session execution, serving,
vision or NextN. No family registration or runtime support claim is added.
Potential REAP geometry is not pruned-artifact provenance or qualification.

## Immutable references

Numerical source: `deepseek-ai/DeepSeek-V4.1-Flash-1017` at
`1bae0e1088993b9b82de08f59de5be5a34e42170`.
`scripts/reference/deepseek_v41/sources.json` pins SHA256 for config, model,
kernel, Engram and license files; offline copies are under `fixtures/` there.
Capture never updates trust anchors automatically:

```sh
uv run --no-project python scripts/reference/deepseek_v41/capture.py
uv run --no-project python scripts/reference/deepseek_v41/capture.py --fetch
uv run --no-project python scripts/reference/deepseek_v41/check.py
cargo test --release -p qwen-llm --lib deepseek_v41::
```

The 1017 `Indexer.forward` publishes `shared_attn.index_k` whenever `owns_k`,
independently of whether the current compression pair completes. The ownership
test executes that source prefix with recording stubs. A negative control moves
publication inside the append guard and reproduces the stale decoder-20 pointer
at incomplete encoder pairs. Reindex and reuse dispatch are checked separately.
These tests do not execute PyTorch, validate compression mathematics, or establish
logit parity. Reduced-shape numerical oracles remain the next execution gate.

Protocol source: `deepseek-ai/deepseek-recipe` at
`8cadfede7063c896b944e7bae05daa3549ae97ea`, MIT; the inspected core, encoding and
protocol crate manifests are version `0.1.0`. This is a source pin, not a verified
crates.io package equivalence. It is not yet a dependency. Use the official
V4.1 renderer and streaming parser behind an adapter rather than recreating
Jinja/DSML semantics. At this pin, encoding and protocol crates enable the
tokenizers HTTP feature unconditionally: choose and review the dependency
boundary before integration, rather than importing image/server machinery by
accident. Test tool/reasoning history, byte and token output, split markers,
UTF-8 boundaries and truncated EOF. The protocol toolkit is not a model oracle.

## Arithmetic and storage contract

The first native expert baseline is **NativeF32**: F32 activations, accumulation
and outputs over decoded MXFP4 weights, without expanded persistent weight banks.
It does not emulate the reference's activation quantization or BF16 output.

The independent **ReferenceFp8** contract quantizes matmul activations to E4M3
in blocks of 32 with UE8M0 scales before FP4/FP8 GEMM; scaled products accumulate
in FP32 and return BF16. The offline source test verifies dispatch/arguments, not
quantizer or GEMM numerical behavior. NativeF32 will first compare against an
independent decoded-weight F32 oracle. Comparisons against ReferenceFp8 measure
a declared arithmetic difference; they must not silently relax a parity gate.

Source expert matrices, with logical dimensions [output, input]:

| Role | Source | Logical shape | Packed values | Separate E8M0 scales |
|---|---|---|---|---|
| gate | w1 | 2304 x 5120 | 2304 x 2560 | 2304 x 160 |
| up | w3 | 2304 x 5120 | 2304 x 2560 | 2304 x 160 |
| down | w2 | 5120 x 2304 | 5120 x 1152 | 5120 x 72 |

The source-layout checkpoint below verifies these expectations against selected
pinned safetensors headers, not every tensor in the checkpoint. HF's
separate scale tensors are not GGML's interleaved 17-byte MXFP4 blocks. Pin nibble
ordering, scale exponents (including edge values), signed zero, row/expert strides
and any repacking before admitting a converter. Equal payload sizes are not byte
compatibility. Default shared-expert weights are FP8, not routed-expert FP4.

Current executor inventory, before any V4.1 kernel work:

| Existing GGML-layout primitive | MXFP4 gate/up | MXFP4 down |
|---|---|---|
| Generic or indexed single-slot GEMV | available | available |
| Fused all-slot decode | missing | available |
| Native all-slot composition | missing decoder | missing decoder |
| Grouped/batch gate/up prefill | missing | not applicable |
| Dedicated routed down prefill | not applicable | matrix or row-GEMV |
| Generic mat-mat dispatch | missing | missing |

This is not V4.1 eligibility. Existing DS4 routing has a 256-expert bound; some
fast paths additionally require V4 artifact byte totals or K160/K216 geometry.
Do not reuse those policies for 384 or 272 experts. Add MXFP4 native composition
first; fast gate/up kernels are a later measured decision. Geometry/dtype and
GPU capability govern executable paths. Content identity remains appropriate
for snapshots and converter provenance, not accidental execution whitelists.

## Work order and integration

1. Pin contracts and tests without model allocation (this checkpoint).
2. Bind actual tensor descriptors and layout fixtures; implement independent
   decoded-weight arithmetic and source-faithful reduced-shape oracles. Prepare
   protocol integration independently of GPU work.
3. Add correct baseline execution: native MXFP4 composition, bounded Engram,
   CSA2 state ownership and lagged mHC, then full-forward text continuation.
4. Qualify commit/cancel/reset/restore behavior and whole-lifecycle admission.
   Retain the exact full-forward baseline while adding CED prefill.
5. Promote optimizations only with actual dispatch evidence and same-run
   incumbent comparisons. Defer approximate replay, vision and speculation.

Use a separate worktree and integrate current main before dependent shared-kernel
work. The concurrent maintainability lane owns device-gate removal and DS4
qualification predicates. Only the capability-gate landing is a dependency;
independent artifact-identity cleanup need not delay V4.1. Request a specific
primitive's conversion if it becomes necessary; do not edit it concurrently.

The M5 toolchain prerequisite is resolved: Xcode 27.1 and the separate Metal
Toolchain component are installed. The native baseline is recorded in
`bench/2026-10-17-m5-toolchain-baseline/README.md`: ordinary workspace tests,
available model witnesses, validated correctness replays and a GLM trunk control
pass. The qualification runner retains three failed facets (one duplicated HCA
speed gate and one missing K216 asset), not a blanket qualification pass. Its
first-light binaries remain unchanged. Keep these GPU/OS/compiler controls
separate from subsequent V4.1 architecture changes.

## Regression and review gates

Use name filters in the edit loop. At each completed shared-kernel change, run
the full annotated V4 0731, GLM and Qwen fixture witness union on the M4 host with
the assets, then qualification and full contracts. Do not select only tests that
seem related. The prefix selection is based on fixture annotations, independent
of test names; a requested prefix with no witnesses is an error.

```sh
uv run scripts/model-witnesses.py \
  --fixture-prefix deepseek-v4-flash-0731 \
  --fixture-prefix glm-5.3-flash \
  --fixture-prefix qwen \
  --fixture-prefix dflash
uv run scripts/qualification-witnesses.py
scripts/check-contracts.sh --full
```

Set `QWEN_MODEL_SEARCH_PATH` to the existing artifact search roots as documented in
`ENV.md`; do not encode machine paths in scripts. Missing required fixtures or
Metal are failures, not passes. The runner covers its annotated witness catalog,
not every GPU or unannotated test. Preserve the per-test execution counts and
report missing prerequisites explicitly. The model-free checkpoint changes no
shared kernel and does not claim to have run this GPU gate.

**Coverage gap found during this checkpoint:** the union resolves 99 annotated
witnesses (97 in qwen-llm, two in qwen-cli), but DS4 and GLM Metal integration
tests still use free-form ignore reasons that the runner does not discover.
Examples include `deepseek_v4_metal/tests.rs`'s current-asset packets and
`glm5_next_metal/tests.rs`'s checkpoint/oracle packets. The command above is not
yet the complete family execution regression set. Reconcile that catalog and
its required artifact/oracle prerequisites before completing a shared-kernel
change. Do not relabel timed experiments as correctness witnesses or treat
99 selected tests as proof that the missing execution packets ran.

A fast-path witness must prove the candidate actually ran with production
geometry and must be callable before the device is qualified. Compare against
the real incumbent in the same run, not absolute timings from another chip.
Update arithmetic declarations and their witnesses together. Keep immutable
oracle captures unless inputs/semantics change; full-prefix runs belong at actual
numerical, addressing, ownership or scheduling discontinuities.

Review small coherent checkpoints before integration and rerun affected checks
on the integrated tree. Share storage/dispatch mechanisms, not family-specific
state layouts or numerical policies. Retain negative results in evidence/history,
not indefinitely in production switches.

## First-checkpoint verification

Implementation commit `27efde5b`, tree
`e520658e466ebcbc61977d0225694fb28102582e`, passed
`scripts/check-contracts.sh --full`: all workspace test targets compiled, six
model-free V4.1 Rust tests, six offline source-contract Python tests, ten
comparison tripwire tests and the README family-table test passed. The existing
format, environment, path and witness-runner structural checks passed as well.
This was not workspace test execution or a GPU qualification run.

Independent read-only review identified two initially incomplete gates: Rust
tests were compiled but not executed by the contract script, and Python discovery
could succeed with zero tests. Both are fixed; Rust requires nonzero passes and
zero ignored tests, and Python names its required module and rejects empty
suites. Empty selection and missing-module controls are retained. Follow-up
review found no remaining blocker for this model-free checkpoint. The 99-witness
catalog was enumerated only, not executed. No model weights were downloaded.

## Source-layout and MXFP4 checkpoint

`deepseek_v41::source` now binds main-model expert gate/up/down weight-scale
pairs from bounded safetensors headers. It rejects duplicate keys, unsupported
storage types, mismatched shapes/extents, overlapping or out-of-file offsets,
invalid main-layer/expert indices and incomplete scale pairs. File labels and
declared lengths are caller inputs, not content authentication. This is not a
complete model binder or an accepted GGUF conversion contract.

`scripts/reference/deepseek_v41/tensor-headers.json` records 18 descriptors in
four pinned headers (shards 3, 44, 47 and 48). The captured files total 344,416
bytes including header-length prefixes; the source index is 7,470,294 bytes and
is not vendored. Only metadata was fetched, not weight payloads. Capture requires
206 responses with exact Content-Range/Content-Length, rejects oversized inputs
before reading, and is bounded to 16 MiB of response bodies total. Redirects are
limited to five HTTPS hops and closed without reading their bodies. Repeat
capture must match the
committed manifest rather than silently replace its trust anchors.

Observed routed main and MTP values are `I8` packed bytes with `F8_E8M0` scales;
the dense Q projection uses `F8_E4M3` with 32x32 weight scales. Engram uses
`F8_E4M3` rows of 256 values and eight `F8_E8M0` scales per row. The MTP headers
are inventory evidence only; the Rust expert binder accepts main-model names.
Headers establish neither numeric values nor nibble ordering.

`deepseek_v41::mxfp4` provides a caller-buffer row repacker and independent
mathematical E2M1/E8M0 decoding. HF adjacent low-even/high-odd codes are permuted
into GGML's low-first-16/high-second-16 layout, retaining all codes and scales
without requantization. Layout evidence also comes from DwarfStar's
`gguf-tools/deepseek41_quantize.py` at
`bd66c402070042bf0a79ad6ece8242de4c93680c`. Synthetic repacking is checked
against the separately linked GGML CPU decoder, not a new Metal implementation.

All 16 codes and 256 scale bytes are covered, including subnormal scales,
overflow, scale-255 refusal, unique nibble positions, multiple rows/blocks and
late-error output preservation. The source decoder preserves negative zero;
GGML canonicalizes code 8 to positive zero on decode. Packed bytes remain
lossless, but decoded zero bits are explicitly not claimed identical. A tiny
NativeF32 dot oracle uses sequential F32 multiply/add, not FP8 activation
quantization, BF16 outputs or a claim of parallel-GPU bit identity.

These are bounded correctness tools, not a whole-checkpoint converter or a
conversion-throughput result. The ordinary required contract runner executes
the source-control, header and Engram address suites, and the Rust V4.1 filter
includes the binder, arithmetic and bounded address-state tests.

```sh
uv run --no-project python scripts/reference/deepseek_v41/tensor_headers.py
uv run --no-project python scripts/reference/deepseek_v41/tensor_headers.py --capture
uv run --no-project python scripts/reference/deepseek_v41/check.py
cargo test --release -p qwen-llm --lib deepseek_v41::
```

### Regression prerequisite audit

Retained GLM checkpoint, qualification and sparse-v1/v2 captures were located.
Their sizes/record framing match the manifests under `scripts/reference/glm53`;
the execution tests still must check complete hashes before use. Bind
`GLM53_ORACLE_DIR`, `GLM53_QUAL_ORACLE_DIR`, `GLM53_SPARSE_ORACLE_DIR` and
`GLM53_SPARSE_V2_DIR` explicitly to those retained captures; do not regenerate
references merely because the current worktree lacks its own copy. Stock IQ3
is the oracle artifact; Maya/Q4 are not interchangeable oracle inputs.

Current DS4 and REAP K216/K160 assets were located. The current 0731 artifact
has 104,207,848,032 file bytes; the historical oracle artifact has
102,999,888,416 bytes and is not present locally. Their filenames coincide.
Use explicit current-artifact identity, not the existing legacy-named fixture
alias. Historical b10222 comparisons remain historical; they must not be run
against current weights or counted as current-model passes. Repo policy does
not require reprovisioning the historical archive for ordinary development.

The coverage repair must retain active correctness packets and their prerequisites,
not indiscriminately tag every ignored test. GLM has maintained oracle, sparse,
snapshot/reset, observational-capture and serving packets. Current DS4 has live
same-model differential and snapshot-continuation controls; mixed diagnostic
packets require correctness-only extraction. Some DS4 tests require the
`dsv4-diagnostics` feature, and a GLM observer uses a `#[path]` module whose
Rust test name differs from its filename. The existing name-derived runner
cannot discover those by changing ignore prose alone. Archived, sealed, timing
and acquisition packets must remain explicitly classified rather than quietly
added to the correctness set.

That catalog/identity repair is still pending before shared-kernel completion,
including Qwen's external-oracle and unannotated execution tests. It overlaps
files owned by the ongoing maintainability lane; the source-layout checkpoint
does not edit those files or claim full-family GPU regression coverage.

### Bounded Engram addressing

`deepseek_v41::engram::AddressState` maps already-compressed IDs to 24 rows per
Engram layer (48 total). It retains only the last three IDs plus absolute position
and compressed pad ID. Each step returns addresses and a staged next state without
mutating the committed state. Restore validates history length, ID ranges and
context position; this is a bounded hash-state component, not a session snapshot
or proof of prefix/content identity.

`engram-addresses.json` pins eight multipliers, 48 prime bucket moduli and 1,008
row IDs across 21 positions. Generation executes the SHA-verified upstream AST
using a NumPy int64 tensor facade, not real PyTorch. NumPy `2.2.6` generates the
reference RNG parameters; offline verification uses only the standard library.
The prime-search substitute is checked independently against a sieve. The Rust
implementation compares every captured address and covers all dead-token phases,
chunking, bounded restoration, context limits and discarded staged steps.

Inputs are compressed IDs, not raw tokenizer IDs. Pad ID 17 in these fixtures is
deliberately synthetic. Tokenizer normalization/mapping, the actual compressed
pad ID, disk-row reads and floating-point Engram injection remain unimplemented.
Do not use this component on raw token IDs or infer model execution from integer
address agreement. Generator/verification commands are documented in
`scripts/reference/deepseek_v41/engram_oracle.py`.

### Verification

Implementation `e3e662a3`, tree `d76b8a41646956aa1fa3b671299ca65bdf8b187b`,
passed `scripts/check-contracts.sh --full`: all workspace targets compiled,
22 V4.1 Rust tests and 26 offline Python tests executed without skips, and the
existing comparison/structural/README gates passed. Explicit bounded header
recapture and isolated NumPy Engram regeneration matched the pinned fixtures.

Read-only review found and resolved an unbounded redirect-body read in the
initial capture transport. The replacement is tested through the actual urllib
handler chain, including hop limits, rejected schemes and regenerated Host
headers. Follow-up review found no remaining blocker for this model-free scope;
an independent 100-case history/mask probe also matched the extracted Engram
source under multiple chunkings. These results are not GPU or full-model
qualification. No shared kernels, model weight payloads, system memory limits
or installed inference binaries were changed.
