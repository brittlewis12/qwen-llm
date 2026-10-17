# V4.1 Flash 1017 bring-up

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

Not implemented: checkpoint tensor binding, converter admission, Engram hashing
or gathers, numerical kernels, protocol adapter, session execution, serving,
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

These are source-derived expectations, not verified safetensors headers. HF's
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

The M5 worker lacks the Xcode Metal compiler. Its first-light tests used binaries
built elsewhere, not a locally built kernel suite. Xcode installation requires
the user's Apple ID. After installation, establish a workspace-test,
qualification-runner and model-witness baseline before debugging new kernels
there. Model-free work can proceed while this is blocked. Do not attribute a
new GPU/OS/toolchain failure to the new architecture without that control.

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

Set `QWEN_MODEL_PATH` to the existing artifact search roots as documented in
`ENV.md`; do not encode machine paths in scripts. Missing required fixtures or
Metal are failures, not passes. The runner covers its annotated witness catalog,
not every GPU or unannotated test. Preserve the per-test execution counts and
report missing prerequisites explicitly. The model-free checkpoint changes no
shared kernel and does not claim to have run this GPU gate.

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
