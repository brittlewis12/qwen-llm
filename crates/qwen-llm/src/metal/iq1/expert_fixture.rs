use super::super::*;

pub(crate) fn iq1_test_scale(block: &mut [u8], dtype: GgmlType, scale: f32) {
    let bits = half::f16::from_f32(scale).to_bits();
    if dtype == GgmlType::IQ1_S {
        block[..2].copy_from_slice(&bits.to_le_bytes());
    } else {
        assert_eq!(dtype, GgmlType::IQ1_M);
        for i in 0..4 {
            let off = 48 + 2 * i;
            let word = u16::from_le_bytes([block[off], block[off + 1]]) & 0x0fff;
            let word = word | (((bits >> (4 * i)) & 15) << 12);
            block[off..off + 2].copy_from_slice(&word.to_le_bytes());
        }
    }
}

pub(crate) fn admit_iq1_expert_fixture(ctx: &MetalContext) {
    // A conservative 128 MiB fixture envelope, split so only CPU growth is
    // charged against CPU availability; shared buffers follow MWSS headroom.
    let admission = evaluate_metal_memory_admission_with_cpu_bytes(
        64 << 20,
        64 << 20,
        16 << 20,
        ctx.memory_signals(),
        true,
    );
    assert!(
        admission.is_admitted(),
        "bounded IQ1 fixture: {admission:?}"
    );
}
