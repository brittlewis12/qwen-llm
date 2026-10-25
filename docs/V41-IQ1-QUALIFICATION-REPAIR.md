# IQ1 Qualification Repairs

The original feature/documentation commits `463e4e07` and `47312602` and
their `freeze-v7` evidence remain intact. Independent review identified two
qualification defects, not an IQ1 decoder regression. This repair changes
neither model payloads nor default policies, selectors, row strides,
dequantization, K loops, reductions, clamping, or epilogues for accepted
production counts. No main-worktree, remote, model, or frozen-oracle work is
part of this change.

## Shared Count Bounds

Bounds repair commit: `b4691ebc`.

Each repaired shared template returns uniformly for the entire threadgroup
before its first ID access or barrier when the count is negative, exceeds
the expert's ID-row capacity, falls outside the requested inclusive count
range, or has no slots in the current tile. The exact predicate is:

```cpp
if (count < 0 || uint(count) > args.N
    || count < int(args.min_count) || count > int(args.max_count) || r1 >= count) return;
```

Fused SwiGLU argument structs call the same capacity `args.n_tokens` rather
than `args.N`. The guard depends only on the count, constant arguments, and
threadgroup tile origin, never lane/thread indices. Seven bodies are covered:

| Source | Bodies / capacity |
| --- | --- |
| `kernels/moe.metal` | Generic MM (Down, UpSiluMul, UpClamped), `N`; generic n16 fused SwiGLU (unclamped/clamped), `n_tokens` |
| `kernels/tensor/moe_grouped_slots_tensor.metal` | Narrow tensor body, `N`; F16-only wide expert kernel, `N` |
| `kernels/moe_grouped_f32.metal` | F32-operand down, `N`; fused clamped SwiGLU, `n_tokens` |
| `kernels/moe_grouped_n32.metal` | Fused clamped n32 template, `n_tokens` |

The wide kernel also now honors its existing min/max count fields. Normal
wide callers request the full range, so valid production behavior is
unchanged. The narrow tensor body's dense case has `count == N`, hence the
new ceiling does not affect it. These shared bounds apply to every dtype
instantiated by these templates, not just IQ1. Other specialized kernel
families are not claimed to have received a comprehensive safety rewrite.
There is no CPU count readback or new weight/scratch representation.

The strengthened malformed expert is the last of three experts: count 36
with ID-row capacity 35. Its 35 in-payload IDs are unique valid slots 35..69,
disjoint from the active valid expert. They must remain unassigned because
the expert's count is invalid, not because its IDs are invalid. Neither
unchanged byte canaries nor output guards are described as read detectors.

Actual frozen M4 red/green evidence:
- `qualification-p2/count-red`: stronger fixture with the original kernels;
  its focused malformed-route test exits 101 after a supposedly unassigned
  `-777` output becomes `-0.0020306574`.
- `qualification-p2/count-green`: repaired kernels pass that same test, 32
  focused IQ1 checks, and the full regression filter set.
- The CPU structural test verifies the exact uniform predicate and its
  ordering before IDs/barriers in all seven bodies. The new M4 role test
  covers IQ1_S/M and IQ2_S/IQ3_S generic roles with valid IDs and oversized,
  negative, and excluded-range counts. Existing range-split regressions
  continue to pass.

## Active-Only Agreement

Agreement repair commit: `33920980`.

`assigned_projections` centralizes finite checks, exact unassigned-sentinel
checks, and extraction of assigned output/reference values. `Fixture::run`
still checks complete output guards and input immutability, but returns only
the assigned values. Both the real-K tensor witness and the mixed-release
M5 witness now pass those active-only vectors to `assert_tile_agreement`.
The M4 fixture also asserts the returned assigned-element count.

The original constants remain `2e-3 * ref_max + 1e-5`. A CPU negative control
uses two active values (reference peak 0.25) and 126 unchanged `-777`
sentinels. Increasing one active value by 0.25 must fail with the active
threshold 0.00051; guard values cannot inflate it to approximately 1.554.
An unchanged-active positive control passes, and a separately changed
sentinel is rejected exactly. Expected panics are caught inside this test.
No tolerance is weakened, and no NaN-hiding fold is added.

## Final Frozen M4 Proof

Evidence is relative to the requested private IQ1 evidence root. The new
bundle is `qualification-p2/freeze-v8`; results are in
`qualification-p2/freeze-v8-results.json` and per-filter logs alongside it.
Compilation and freezing of the executable, three metallibs, and source
manifest occur before releasing the production build lease. GPU fixtures
self-lease and run the frozen executable directly, with
`MTL_DEBUG_LAYER=1`, `QWEN_METAL_LEASE_WAIT=1`, and one test thread.

All following filters pass on Apple M4 Max. Except for the noted skip, each
is run as `"$FROZEN" FILTER --nocapture --test-threads=1` with those variables:

| Exact filter | Expected result |
| --- | --- |
| `iq1 --skip metal_forward::` | 33 passed, 8 ignored |
| `generic_grouped_moe_` | 51 passed |
| `native_capabilities_match_actual_dispatch_and_abi` | 1 passed |
| `native_all_down_formats_cpu_oracle_offsets_and_slot_isolation` | 1 passed |
| `native_mixed_gate_up_cpu_oracles_and_route_rejection` | 1 passed |
| `native_validation_rejects_before_any_dispatch` | 1 passed |
| `moe_grouped_generic_mapping_matches_metal_source` | 1 passed |
| `moe_grouped_tensor_mapping_matches_metal_source` | 1 passed, source mapping only |
| `moe_grouped_generic_layout_matches_ggml_storage` | 1 passed |
| `moe_grouped_generic_covers_every_resident_expert_dtype` | 1 passed |
| `clamped_recipes_cover_independent_layouts_and_real_entry_points` | 1 passed |
| `expert_capability_names_have_shader_implementations` | 1 passed |
| `tensor_grouped_moe_case_list_covers_mapping` | 1 passed, inventory only |
| `tensor_dense_case_list_covers_mapping` | 1 passed, inventory only |
| `selected_strategy_threshold_legacy_fallback_and_entrypoints` | 1 passed |

The three added focused tests are individually filterable (one pass each):
`iq1_shared_count_guards_precede_ids_and_barriers`,
`iq1_shared_generic_roles_reject_capacity_and_count_ranges_with_valid_ids`,
and `iq1_tensor_agreement_rejects_active_error_despite_large_sentinels`.
Test-file formatting and ASCII checks run under the production lease;
Clippy retains the repository's existing warning limitations.

| Frozen artifact | SHA-256 |
| --- | --- |
| Test executable | `2fbab2de657a238c29e9a6f1daa86177e57d6c2cbfd1750ddc470760ccb0169c` |
| Product metallib | `b623201c7cd35f5cb7d6ec190ee6349d8074f66dbbd863cd63d97aa74887121c` |
| Tensor metallib | `64da3a665759978f44b21101ceb8c98a20806753c78c9953fb915c2397fc4580` |
| Research metallib | `00506cb04df3d957d932d029abd35ea1ca8280be2862e82402a33cfb5caf4189` |

The manifest binds all 21 changed/new compiled source files to this binary.
The shared IQ1 decoder, dense entrypoint suffix, and existing decoder/grid
hashes remain unchanged. The original `freeze-v7` artifact hashes are checked
separately without replacing that bundle or its handoff index.

## M5 Proof Scope

The parent has now run the M5 proof at `0e263131`: the scoped default filter
passes 33 checks, and the unskipped ignored filter passes nine tests (eight
M5 tensor witnesses plus one legacy embedding loader/gather witness). The
runner's expected count of eight was stale; it was not a tensor failure.
The subsequent native-module expectation repair and fresh-build rerun plan
are in `docs/V41-IQ1-CAPABILITY-CENSUS.md`. The following command inventories
are corrected without changing any test's GPU requirements.

Use the newly frozen binary for the parent's fresh-build rerun:

```sh
MTL_DEBUG_LAYER=1 QWEN_METAL_LEASE_WAIT=1 "$FROZEN" \
  iq1 --ignored --nocapture --test-threads=1
```

Expected inventory: nine passes on M5: eight tensor execution tests plus
`metal_forward::role_tests::iq1_m_embedding_loader_and_gather_use_frozen_choice`.
Add `--skip metal_forward::` for exactly eight M5 tensor witnesses. The
scoped default command `iq1 --skip metal_forward::` continues to expect 33
passes and eight ignored tests. No GPU requirement is relaxed. The stronger
invalid-count IDs exercise the narrow
and wide tensor guards; both M5 witnesses now compare only assigned values.
Unavailable tensor hardware must still refuse the required-M5 assertion.
The deployment tier remains Metal 4.0 at macOS 26.0. The local M4 results
remain distinct from the parent's M5 evidence. Neither supplies full-model
held-text quality or speed qualification.
