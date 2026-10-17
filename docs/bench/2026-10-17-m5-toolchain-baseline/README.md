# M5 Ultra native toolchain baseline

Source: `d47eeced1d77b2d780c14315363c27bd8623089a`, clean detached worker checkout.
No shared kernels or qualification policies changed for this run. The pending
capability-gate maintenance work was not yet on main.

## Results

| Check | Result | Scope |
|---|---|---|
| Native release workspace build | passed | Xcode 27.1, Metal 32023.921.6, SDK 27.0, Rust 1.99.1 |
| Ordinary workspace tests | 2,922 passed, 0 failed, 645 ignored | Metal required, API validation enabled; CPU and GPU tests combined |
| Qualification registry | exit 1: three failed facets | Two HCA entries share one failing speed gate; one missing model fixture |
| Correctness replay rerun | nine unique commands passed | API validation enabled; covers ten correctness facets |
| Available annotated model witnesses | 13 selected, 13 passed | Ten Qwen A3B checks and three GLM metadata/tokenizer/render checks |
| Additional GLM trunk gate | passed | 200-token native Exact/Fast comparison and three continuation steps |
| Offline V4.1 source contracts | 26 passed | No model execution claim |

The ordinary suite takes 100.864 seconds including Cargo overhead. Four nested
one-test child summaries are excluded from the top-level pass total. The 645
ignored tests were not executed. A successful harness result is not proof that
every optional branch ran; three foreign-device tests, for example, require a
second distinct Metal device. No count of executed GPU tests is claimed.

GLM uses the stock IQ3 artifact already retained on the worker. Prefill top-1
matches, prefill KL is `7.775e-7`, and the three continuation KL values are
`5.262e-5`, `3.777e-6`, `4.919e-6`, all below the existing `1e-3` gate. This is
native self-consistency, not an independent upstream full-model oracle or a
quality comparison. API-validation timings are not production throughput.

## Qualification failures retained

The registry reports 34 facets: 12 passed, three failed, five scope-only, seven
not applicable on this device, and seven with no witness. Fourteen paths decline
under the old exact-device policies. Replay success does not widen those policies.

`deepseek_v4.grouped_long_hca` and `deepseek_v4.splitk_hca` invoke the same
`profile_tiled_hca_at_far_context` performance test. Both fail its 2048-row
**ungrouped online versus legacy** requirement:

```text
online_ms <= baseline_midpoint_ms * 0.87
```

The online arm is approximately 0.960 ms versus 1.073 ms legacy, short of the
required 13% saving. The grouped arm is approximately 0.686 ms and split8 is
0.093 ms, but the test aborts before its 8192-row performance case. These are
not failed grouped/split8 timing assertions and not complete performance
qualification of those production paths. The separate numerical HCA witness
passes through 8192 rows, including grouped/split8 repeatability checks.

The MXFP4 matrix performance witness fails before measurement because the worker
lacks the K216 artifact. Its synthetic correctness witness passes. This is an
asset-blocked witness, not evidence that MXFP4 is slow or numerically incorrect.
No large model was transferred merely to make the report green.

One useful positive result is the K160 Q3_K/Q4_K all-slot replay: 80 fast
dispatches, approximately 0.065315 ms fast versus the 0.253619 ms generic
all-slot incumbent (serial indexed: 1.001413 ms), passing the existing relative
gate. It does not establish V4.1 or MXFP4 gate/up support.

The HCA measurements are diagnostic. Simulator downloads/installation were
allowed to continue; the machine was not treated as an isolated performance lab.
Do not loosen thresholds or convert these failures to passes. The maintenance
handoff is to separate production grouped/split8 evidence from the online
characterization gate, preserving the failed result and completing the remaining
case coverage. The inspected pending device-gate change does not itself repair
this witness or supply the absent K216 model.

## Toolchain and memory

The worker runs macOS 27.0.1 (`26A434`), Xcode 27.1 (`27A9275`), SDK 27.0 and
Metal Toolchain asset `27A266a`. `xcrun --find metal` initially found an Xcode
stub while actual invocation failed. Installing the separate component with
`xcodebuild -downloadComponent MetalToolchain` resolved that; actual compiler
and linker invocation and a native build were verified. Simulator runtimes are
not a prerequisite and were left alone.

This is a new compiler baseline, not a controlled M4/M5 hardware comparison.
Rust 1.99.1 reports existing `Atomic::fetch_update` deprecations; they do not
prevent compilation and were not mass-edited during baseline establishment.

`iogpu.wired_limit_mb` remains zero (automatic). Swap usage is zero at both
endpoints. Cumulative compressions remain 46,664; pre-existing compressor
occupancy decreases from 6,348 to 6,328 16-KiB pages (about 99 MiB). This is not
a continuous peak-memory measurement or a claim of zero compressed memory.

No persistent inference server remains. The installed first-light binary was
not replaced; its SHA256 remains
`5bf0b34c9470ed9bab2bd4447de5f547fb1a9e6a13f18d2534442bead36e9c55`.
The separate worker-native qwen build has SHA256
`730a3ed8a35be51e5e4cb60086f3e68814de904bffef4de9b1d9c68d4d10d808`.

## Reproduction and evidence

Run from the pinned checkout, with inherited family/experimental overrides
cleared. Build/test commands and fixture selection:

```sh
CARGO_BUILD_JOBS=4 cargo test --release --locked --workspace --no-run
env -u QWEN_MODEL_SEARCH_PATH MTL_DEBUG_LAYER=1 \
  QWEN_REQUIRE_METAL_TESTS=1 QWEN_METAL_LEASE_WAIT=1 \
  cargo test --release --locked --workspace --no-fail-fast -- \
  --test-threads=1 --nocapture
env -u MTL_DEBUG_LAYER QWEN_METAL_LEASE_WAIT=1 \
  uv run scripts/qualification-witnesses.py --qwen target/release/qwen \
  --out "$EVIDENCE/qualification.json"
MTL_DEBUG_LAYER=1 QWEN_METAL_LEASE_WAIT=1 \
  uv run scripts/model-witnesses.py \
  --fixture-prefix qwen3.6-35b-a3b-ud-q4_k_m \
  --fixture-prefix glm-5.3-flash-ud-iq3_xxs
```

Bind `QWEN_A3B_MOE_MODEL` and `GLM53_GGUF` explicitly to the existing assets;
use the canonical stock-IQ3 first shard for GLM. Do not substitute Maya/Q4 into
the stock-IQ3 census. No DS4 model-level execution is claimed on this worker.
Correctness replay commands are retained in `results.json`; each was additionally
run with `MTL_DEBUG_LAYER=1` and `--nocapture`, separately from performance.

The additional GLM gate is:

```sh
MTL_DEBUG_LAYER=1 QWEN_REQUIRE_METAL_TESTS=1 QWEN_METAL_LEASE_WAIT=1 \
  cargo test --release --locked -p qwen-llm --lib \
  glm5_next_metal::tests::packed_fast_matches_exact_over_a_grouped_block_and_tail \
  -- --ignored --exact --nocapture --test-threads=1
```

`results.json` contains path-free aggregate evidence, per-target counts, exact
witness commands/statuses and hashes of retained private raw logs. Raw logs and
launch wrappers remain outside Git because they contain machine-specific paths.
`summarize.py` regenerates the public result from that evidence directory;
`test_summary.py` guards nested-count inflation, omitted summaries and accidental
promotion of failed or unexecuted qualification facets:

```sh
uv run --no-project python -m unittest discover \
  -s docs/bench/2026-10-17-m5-toolchain-baseline -v
```
