# Joint GSQ Loader And IQ1 Expert Integration

This is a new independent composition, not a continuation of a historical
window worker. Base: `6e80eac2176d10fd73f44096ca6d541885610232`.
The base already includes the standard GSQ loader (`0079569a`), concurrent
IQ3 4096-row selector (`97c93dd5`), and parent GSQ qualification documents.
Their production loader, control/source handling, and selector are retained.
PERF-ROADMAP, PERF-LOG, and the parent's actual M5 data are not edited.

The standalone corrected IQ1 branch remains clean at
`791251c46dd42fa0e08bc9d429a773f5db7cf469`. Its old frozen roots, including
the parent M5 evidence at `0e263131`, remain unchanged. The new worktree uses
branch `feat/v41-gsq-iq1-integration`; it does not edit main, publish, download
models, use the network, or install a maintained executable.

## Reviewed Series Versus Composition Fixup

Only the IQ1 task series from `2647dac9` through the corrected standalone HEAD
is cherry-picked. Unrelated main documentation history is not replayed:

| Standalone commit | Joint cherry-pick | Purpose |
| --- | --- | --- |
| `463e4e07` | `6f454226` | Native/grouped/tensor existing-IQ1 adapters and fixtures |
| `47312602` | `b1259feb` | Original frozen-proof handoff |
| `b4691ebc` | `0f13b85a` | Uniform shared count-capacity guards |
| `33920980` | `93ff4649` | Assigned-only tensor comparison |
| `0e263131` | `b888ae53` | Qualification repair evidence |
| `1d9b2a14` | `70883e5b` | Independent native row capability expectations |
| `791251c4` | `721e0c74` | Corrected parent M5 and ignored-test census |

The subsequent joint fixup changes four test/fixture files plus the loader
status document and this handoff. It does not change production loader or
CLI capability logic:
- `gguf/tests/gsq_loader.rs`: both authentic quant variants must now produce
  valid plans for all 120 banks. IQ2_XXS includes 60 IQ1_M banks. The previous
  intentional IQ1 rejection is removed because executable native adapters
  are present, not because the loader gate is relaxed.
- `gguf/tests.rs`: include IQ1_S/M in positive native-bank metadata cases;
  retain Unknown/I32, row alignment, geometry, and storage-size refusals.
- `deepseek_v41_fixture.rs` in the CLI: parameterize the existing CPU-only
  header fixture with width and independently typed gate/up/down banks;
  the existing two-wide F32 fixture remains the default.
- `qwen/deepseek_v41.rs` tests: cover IQ1_S/M and the released mixed down
  combinations, without a fit gate; require rejection of actual unsupported
  types and malformed row descriptors. No Metal context is created.

All native capability remains backed by the real IQ1 kernel families. No
production model-name/hash allowlist, dtype gate relaxation, control default,
new format/codec, or memory-policy change is introduced. Existing shape,
finiteness, alias/conflict, source ownership, and budget checks remain active.

## Authentic Prefix Replay

The ignored `cpu_only_sparse_header_control_replay` test uses the original
saved headers, prefixes, and exact controls through `V41_GSQ_AUDIT_DIR`.
It creates temporary sparse files, not complete weights, and never calls
resident loading, Engram table-row gathering, or a GPU API.

Both IQ2_XS and IQ2_XXS validate 120 runtime banks with four backbone sources
and one separate Engram source. Parsed and planned typed controls, token map,
hash configuration, source groups, and packed-bank accounting agree. Exact
control budgets are retained for each variant:
- Payload bytes read: 517952.
- Cold normalization bound: 3105632 bytes.
- Retained token/hash bound: 519648 bytes.
- Disk-only Engram tables: 110595290400 bytes.

Source bookkeeping remains source-specific (4578/4586 bytes in these replay
logs); it is not incorrectly forced equal across different file names.
IQ2_XXS/2.00 is now serialization/native-capability eligible. It is not
downloaded or full-model qualified here. There is no fit/quality claim for
it or a smaller package. The parent's actual 2.25 Full/CED 128/2K/8K results
remain separate; no CDFwd executor or harness stack from `05edb66f` is imported.

## Local Verification

All compilation and CPU work uses the production lease with
`QWEN_METAL_LEASE_WAIT=1`. Cargo uses `--offline`. Optimized lib/CLI test
executables and all three metallibs are frozen before the build lease is
released. CPU fixtures run under a CPU/build lease. The optional M4 GPU
fixtures run frozen directly with `MTL_DEBUG_LAYER=1`, self-leasing rather
than using an external double wrapper.

| Check | Result |
| --- | --- |
| Lib `deepseek_v41::gguf::tests::`, excluding eight GPU fixtures | 63 passed, one ignored |
| Lib `cpu_only_sparse_header_control_replay --ignored` | One passed, both quant variants |
| CLI `deepseek_v41::tests::` | 15 passed |
| Lib `metal::expert_native::tests::rows::native_rows_arithmetic_source_and_capability_contract --exact` | One passed |
| Frozen M4 `iq1_actual_product_pso_preflight` | One passed |
| Frozen M4 `iq1_native_cpu_codec_real_k_and_route_contract` | One passed |
| `cargo check --offline --workspace --all-targets` | Passed, existing warnings |
| Task-file rustfmt/ASCII checks | Passed |

The CPU GGUF filter excludes exactly these GPU fixtures using `--skip`:

```text
nonzero_bf16_load_forward_multiple_steps
direct_views_retain_mapping_and_preserve_offsets
standard_q8_engram_wkv_alias_and_positional_rows
engram_standard_codecs_gather_selected_rows_with_exact_strides
unused_tensors_keep_inventory_without_binding_or_dtype_requirements
metadata_hash_survives_planning_and_loading
singleton_expert_axes_are_restored_on_direct_views
combined_table_stays_out_of_metal_residency
```

## Evidence And Source Bridges

Private evidence, relative to the requested IQ1 evidence root:
`joint-integration/freeze-joint-v3/`,
`joint-integration/freeze-joint-v3-cpu-results.json`,
`joint-integration/m4-pso-native-results.json`,
`joint-integration/source-bridge.json`, and
`joint-integration/workspace-all-targets.log`.

All 21 compiled task-source hashes equal the corrected standalone
`capability-census/freeze-v9` manifest. The IQ1 shader/host-adapter hashes
also agree with the previously preserved parent M5 source identities.
The main loader/control/companion files and the IQ3 selector sources are
byte-for-byte unchanged from the joint base. Only four additional compiled
test/fixture source files are needed for this composition.

| Frozen joint artifact | SHA-256 |
| --- | --- |
| Lib tests | `2fc2d534d0d3d9b833734bdb01495d6f0c3a944233e3c4030c07a40bf02e3794` |
| CLI tests | `d3338feacf7efa2789dcab19699c301748ceaf7034011ac823b858968cd46c0a` |
| Product metallib | `b623201c7cd35f5cb7d6ec190ee6349d8074f66dbbd863cd63d97aa74887121c` |
| Tensor metallib | `64da3a665759978f44b21101ceb8c98a20806753c78c9953fb915c2397fc4580` |
| Research metallib | `00506cb04df3d957d932d029abd35ea1ca8280be2862e82402a33cfb5caf4189` |

The local metallibs also match the standalone frozen M4 artifacts. This is
not an assertion that cross-toolchain M5 binaries equal local M4 binaries.
The parent will freshly rebuild the corrected standalone IQ1 matrix source
on M5, then integrate the reviewed joint tree after proof. No fresh joint
M5 execution, model quality, speed, or smaller-memory fit result is claimed.
