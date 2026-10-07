//! CPU-only GSQ adapter contract regressions; no Metal device is created.
use super::*;

#[test]
fn gsq_shared_projections_fit_existing_singleton_scratch() {
    let g = Qwen4ExpMoeMetalGeometry::new(2560, 512, 10, 640, 640).unwrap();
    require_shared_projection_scratch(g).unwrap();
    assert_eq!(
        g.hidden_size * g.experts_per_token * size_of::<f32>(),
        102_400
    );
    assert_eq!(2 * g.shared_intermediate_size * size_of::<f32>(), 5_120);
}

#[test]
fn shared_scratch_capacity_is_checked_before_subrange_views() {
    let exact = Qwen4ExpMoeMetalGeometry::new(256, 2, 1, 32, 128).unwrap();
    require_shared_projection_scratch(exact).unwrap();
    let too_large = Qwen4ExpMoeMetalGeometry::new(256, 2, 1, 32, 160).unwrap();
    let error = require_shared_projection_scratch(too_large)
        .unwrap_err()
        .to_string();
    assert!(error.contains("320 scratch elements"), "{error}");
    assert!(error.contains("256"), "{error}");
}

#[test]
fn gsq_q2_down_uses_k640_without_superblock_padding() {
    let layout = crate::metal::moe_grouped_generic_layout(GgmlType::Q2_0).unwrap();
    // The expert's 640-wide row has ten 64-element Q2_0 blocks. A K-quant
    // 256-element assumption would reject it or read into the next row.
    assert_eq!(640 % layout.block_elems, 0);
    assert_eq!(640 / layout.block_elems * layout.block_bytes, 180);
    assert_eq!(GgmlType::Q2_0.storage_layout(), Some((64, 18)));
}
