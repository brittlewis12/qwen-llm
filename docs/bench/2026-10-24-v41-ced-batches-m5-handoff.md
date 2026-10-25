# Bounded CED M5 Handoff

Qualified bounded M4 source: `c661d593c346287773d94b5caf608101ba075695`, on
integrated base `41f4eb543177619ecb480a5b59f6754be5e42b59`. The M4 results do
not qualify M5 tensor paths. The filters below exist in the frozen test inventory
but have not been executed on M5 as part of this candidate.

Build and freeze the release core test executable from clean, pinned source under
the production build lease before the shared target can be overwritten. Set
`CORE_TEST` to that frozen executable. Each test below acquires the production
Metal lease itself: execute it directly, not through another lease wrapper.

```sh
while IFS= read -r filter; do
  MTL_DEBUG_LAYER=1 QWEN_METAL_LEASE_WAIT=1 "$CORE_TEST" \
    --exact "$filter" --ignored --nocapture --test-threads=1 || exit "$?"
done <<'FILTERS'
deepseek_v41::forward::ffn_gpu::arithmetic_tests::wide_full_vs_compacted_ffn_maya_bitwise_and_status
deepseek_v41::forward::gpu_step::ced::batches::tests::m5_small_and_wide_batch_lineage_preserves_full_tile_packed_singletons
metal::moe_grouped_generic::tests::wide_activation_type::wide_activation_type_f16_4096_keeps_wide_numeric_contract
metal::moe_grouped_generic::tests::wide_activation_type::wide_activation_type_f32_4096_preserves_narrow_numeric_contract
metal::moe_grouped_generic::tests::wide_activation_type::wide_activation_type_invalid_and_direct_probes_refuse_without_dispatch
metal::moe_grouped_generic::tests::wide_narrow_compacted_maya_gate_up_bitwise
FILTERS
```

Expected on supported M5: exactly one passed, zero failed per invocation; six
passed total. Preserve logs and verify the expected hardware, Metal validation
and lease acquisition. Zero matching tests or unsupported-hardware early returns
are not coverage. Carrying any later test-only FFN census correction requires a
new clean build/freeze and recording that source revision.

The automatic structural-fallback and selected 2K/4K geometry tests already pass
on bounded M4 and can also be rerun on M5. Neither this six-filter matrix nor
those bounded fixtures replace full-model qualification or native/current-control
timing. Normal production inclusive CPU/device transient peaks remain unmeasured;
the finite command allowance is not a formal driver-memory upper bound.
