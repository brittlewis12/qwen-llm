use super::*;
use crate::gguf::GgufFile;
use crate::metal::{
    rms_norm_mul_vjp_rows_f32_readback_for_test, silu_mul_vjp_f32_readback_for_test,
};
use crate::muse_glimmer::{MuseGlimmerChatTemplateProfile, MuseGlimmerConfig};
use crate::muse_glimmer_lens::muse_glimmer_selected_token_covectors;
use crate::muse_glimmer_metal::{
    encode_muse_glimmer_causal_gqa_vjp_bank_f32,
    encode_muse_glimmer_rope_adjacent_pair_in_place_f32,
};
use crate::muse_glimmer_residency::{MuseGlimmerMetalWeightPlan, MuseGlimmerMetalWeights};
use crate::muse_glimmer_text_session::{MuseGlimmerTextForward, MuseGlimmerTextSession};
use objc2_metal::{MTLCommandBuffer, MTLCommandQueue};

fn tiny_geometry() -> MuseAttentionGeometry {
    MuseAttentionGeometry {
        hidden: 4,
        feed_forward: 6,
        q_heads: 2,
        kv_heads: 1,
        head_dim: 2,
        query: 4,
        kv: 2,
        rope_theta: 500_000.0,
    }
}

fn dot(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right)
        .map(|(&left, &right)| left * right)
        .sum()
}

fn attention_geometry(q_heads: usize, kv_heads: usize, head_dim: usize) -> MuseAttentionGeometry {
    MuseAttentionGeometry {
        hidden: 6_656,
        feed_forward: 19_968,
        q_heads,
        kv_heads,
        head_dim,
        query: q_heads * head_dim,
        kv: kv_heads * head_dim,
        rope_theta: 500_000.0,
    }
}

fn attention_values(count: usize, multiplier: usize, modulus: usize, scale: f32) -> Vec<f32> {
    (0..count)
        .map(|index| {
            ((index * multiplier + 3) % modulus) as f32 * scale - (modulus / 2) as f32 * scale
        })
        .collect()
}

fn attention_differential(actual: &[f32], expected: &[f32]) -> (f64, f64, f64) {
    assert_eq!(actual.len(), expected.len());
    let mut difference_sq = 0.0f64;
    let mut actual_sq = 0.0f64;
    let mut expected_sq = 0.0f64;
    let mut dot = 0.0f64;
    let mut max_difference = 0.0f64;
    let mut max_expected = 0.0f64;
    for (&actual, &expected) in actual.iter().zip(expected) {
        let actual = f64::from(actual);
        let expected = f64::from(expected);
        let difference = actual - expected;
        difference_sq += difference * difference;
        actual_sq += actual * actual;
        expected_sq += expected * expected;
        dot += actual * expected;
        max_difference = max_difference.max(difference.abs());
        max_expected = max_expected.max(expected.abs());
    }
    (
        (difference_sq / expected_sq.max(f64::MIN_POSITIVE)).sqrt(),
        max_difference / max_expected.max(1.0),
        dot / (actual_sq * expected_sq).sqrt(),
    )
}

struct AttentionBankBuffers {
    query: MetalTensor,
    key: MetalTensor,
    value: MetalTensor,
    gate: MetalTensor,
    attention_output: MetalTensor,
    probabilities: MetalTensor,
    grad_gated: MetalTensor,
    grad_query: MetalTensor,
    partial_grad_key: MetalTensor,
    partial_grad_value: MetalTensor,
    grad_key: MetalTensor,
    grad_value: MetalTensor,
    grad_gate: MetalTensor,
    basis_count: usize,
    n_tokens: usize,
    geometry: MuseAttentionGeometry,
}

impl AttentionBankBuffers {
    #[allow(clippy::too_many_arguments)]
    fn new(
        ctx: &MetalContext,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        gate: &[f32],
        forward: &MuseAttentionForward,
        grad_gated: &[f32],
        basis_count: usize,
        n_tokens: usize,
        geometry: MuseAttentionGeometry,
    ) -> Self {
        let bank_rows = basis_count * n_tokens;
        let partial_shape = vec![
            geometry.head_dim as u64,
            n_tokens as u64,
            geometry.q_heads as u64,
            basis_count as u64,
        ];
        Self {
            query: from_f32(ctx, q, row_shape(geometry.query, n_tokens).unwrap()).unwrap(),
            key: from_f32(ctx, k, row_shape(geometry.kv, n_tokens).unwrap()).unwrap(),
            value: from_f32(ctx, v, row_shape(geometry.kv, n_tokens).unwrap()).unwrap(),
            gate: from_f32(ctx, gate, row_shape(geometry.query, n_tokens).unwrap()).unwrap(),
            attention_output: from_f32(
                ctx,
                &forward.attention_output,
                row_shape(geometry.query, n_tokens).unwrap(),
            )
            .unwrap(),
            probabilities: from_f32(
                ctx,
                &forward.probabilities,
                vec![n_tokens as u64, geometry.q_heads as u64, n_tokens as u64],
            )
            .unwrap(),
            grad_gated: from_f32(
                ctx,
                grad_gated,
                row_shape(geometry.query, bank_rows).unwrap(),
            )
            .unwrap(),
            grad_query: MetalTensor::zeros_f32(ctx, row_shape(geometry.query, bank_rows).unwrap())
                .unwrap(),
            partial_grad_key: MetalTensor::zeros_f32(ctx, partial_shape.clone()).unwrap(),
            partial_grad_value: MetalTensor::zeros_f32(ctx, partial_shape).unwrap(),
            grad_key: MetalTensor::zeros_f32(ctx, row_shape(geometry.kv, bank_rows).unwrap())
                .unwrap(),
            grad_value: MetalTensor::zeros_f32(ctx, row_shape(geometry.kv, bank_rows).unwrap())
                .unwrap(),
            grad_gate: MetalTensor::zeros_f32(ctx, row_shape(geometry.query, bank_rows).unwrap())
                .unwrap(),
            basis_count,
            n_tokens,
            geometry,
        }
    }

    fn run(&self, ctx: &MetalContext) -> f64 {
        let command = ctx.queue.commandBuffer().expect("attention VJP command");
        let encoder = KernelEncoder::begin(&command);
        encode_muse_glimmer_causal_gqa_vjp_bank_f32(
            ctx,
            &encoder,
            &self.query,
            &self.key,
            &self.value,
            &self.gate,
            &self.attention_output,
            &self.probabilities,
            &self.grad_gated,
            &self.grad_query,
            &self.partial_grad_key,
            &self.partial_grad_value,
            &self.grad_key,
            &self.grad_value,
            &self.grad_gate,
            self.basis_count,
            self.n_tokens,
            self.geometry.q_heads,
            self.geometry.kv_heads,
            self.geometry.head_dim,
        )
        .unwrap();
        encoder.end();
        command.commit();
        command.waitUntilCompleted();
        assert!(command.error().is_none(), "{:?}", command.error());
        let start = command.GPUStartTime();
        let end = command.GPUEndTime();
        assert!(start.is_finite() && end.is_finite() && start > 0.0 && end > start);
        (end - start) * 1.0e3
    }
}

#[test]
fn metal_causal_gqa_vjp_bank_matches_cpu() {
    let ctx = MetalContext::new().unwrap();
    for (geometry, n_tokens, basis_count) in [
        (attention_geometry(4, 2, 32), 3usize, 1usize),
        (attention_geometry(32, 2, 128), 16, 32),
    ] {
        let q = attention_values(n_tokens * geometry.query, 17, 101, 0.002);
        let k = attention_values(n_tokens * geometry.kv, 13, 89, 0.0025);
        let v = attention_values(n_tokens * geometry.kv, 19, 97, 0.003);
        let gate = attention_values(n_tokens * geometry.query, 11, 83, 0.02);
        let grad_gated = attention_values(basis_count * n_tokens * geometry.query, 23, 107, 0.004);
        let forward = cpu_causal_gqa_forward(&q, &k, &v, &gate, n_tokens, geometry).unwrap();
        let buffers = AttentionBankBuffers::new(
            &ctx,
            &q,
            &k,
            &v,
            &gate,
            &forward,
            &grad_gated,
            basis_count,
            n_tokens,
            geometry,
        );
        let gpu_ms = buffers.run(&ctx);

        let mut expected_q = Vec::with_capacity(grad_gated.len());
        let mut expected_k = Vec::with_capacity(basis_count * n_tokens * geometry.kv);
        let mut expected_v = Vec::with_capacity(basis_count * n_tokens * geometry.kv);
        let mut expected_gate = Vec::with_capacity(grad_gated.len());
        for basis in 0..basis_count {
            let start = basis * n_tokens * geometry.query;
            let vjp = cpu_causal_gqa_vjp(
                &q,
                &k,
                &v,
                &gate,
                &grad_gated[start..start + n_tokens * geometry.query],
                n_tokens,
                geometry,
                &forward,
            )
            .unwrap();
            expected_q.extend_from_slice(&vjp.grad_q);
            expected_k.extend_from_slice(&vjp.grad_k);
            expected_v.extend_from_slice(&vjp.grad_v);
            expected_gate.extend_from_slice(&vjp.grad_gate);
        }
        for (name, actual, expected) in [
            ("Q", read_f32(&buffers.grad_query), expected_q),
            ("K", read_f32(&buffers.grad_key), expected_k),
            ("V", read_f32(&buffers.grad_value), expected_v),
            ("gate", read_f32(&buffers.grad_gate), expected_gate),
        ] {
            let differential = attention_differential(&actual, &expected);
            eprintln!(
                "[muse-attn-vjp] B={basis_count} T={n_tokens} {name} differential={differential:?} gpu_ms={gpu_ms:.6}"
            );
            assert!(actual.iter().all(|value| value.is_finite()));
            assert!(differential.0 <= 5.0e-5, "{name}: {differential:?}");
            assert!(differential.1 <= 2.0e-4, "{name}: {differential:?}");
            assert!(differential.2 >= 0.999_999_9, "{name}: {differential:?}");
        }
    }
}

#[test]
#[ignore = "bounded model-free B32/T16 Muse attention VJP gate"]
fn profile_metal_causal_gqa_vjp_bank_b32_t16() {
    let ctx = MetalContext::new().unwrap();
    let geometry = attention_geometry(32, 2, 128);
    let n_tokens = 16;
    let basis_count = 32;
    let q = attention_values(n_tokens * geometry.query, 17, 101, 0.002);
    let k = attention_values(n_tokens * geometry.kv, 13, 89, 0.0025);
    let v = attention_values(n_tokens * geometry.kv, 19, 97, 0.003);
    let gate = attention_values(n_tokens * geometry.query, 11, 83, 0.02);
    let grad_gated = attention_values(basis_count * n_tokens * geometry.query, 23, 107, 0.004);
    let forward = cpu_causal_gqa_forward(&q, &k, &v, &gate, n_tokens, geometry).unwrap();
    let buffers = AttentionBankBuffers::new(
        &ctx,
        &q,
        &k,
        &v,
        &gate,
        &forward,
        &grad_gated,
        basis_count,
        n_tokens,
        geometry,
    );
    buffers.run(&ctx);
    let samples = (0..4).map(|_| buffers.run(&ctx)).collect::<Vec<_>>();
    let mean = samples.iter().sum::<f64>() / samples.len() as f64;
    let max = samples.iter().copied().fold(0.0f64, f64::max);
    eprintln!("[muse-attn-vjp] B32/T16 gpu_ms={samples:?} mean={mean:.6} max={max:.6}");
    assert!(max <= 5.0, "attention VJP exceeds 5 ms gate");
}

struct TinyMuseModel {
    config: MuseGlimmerConfig,
    token_embedding: MetalTensor,
    output_norm: MetalTensor,
    output: MetalTensor,
    attention_norm: MetalTensor,
    attention_query: MetalTensor,
    attention_key: MetalTensor,
    attention_value: MetalTensor,
    attention_gate: MetalTensor,
    attention_output: MetalTensor,
    query_norm: MetalTensor,
    key_norm: MetalTensor,
    post_attention_norm: MetalTensor,
    feed_forward_norm: MetalTensor,
    feed_forward_gate: MetalTensor,
    feed_forward_up: MetalTensor,
    feed_forward_down: MetalTensor,
    post_feed_forward_norm: MetalTensor,
}

impl TinyMuseModel {
    fn tensor(ctx: &MetalContext, shape: Vec<u64>, seed: usize) -> MetalTensor {
        let count = shape.iter().product::<u64>() as usize;
        let values = (0..count)
            .map(|index| ((index * (seed * 2 + 1) + seed) % 67) as f32 * 0.0015 - 0.0495)
            .collect::<Vec<_>>();
        from_f32(ctx, &values, shape).unwrap()
    }

    fn norm(ctx: &MetalContext, width: usize, seed: usize) -> MetalTensor {
        let values = (0..width)
            .map(|index| 0.75 + ((index * (seed + 1)) % 17) as f32 * 0.02)
            .collect::<Vec<_>>();
        from_f32(ctx, &values, vec![width as u64]).unwrap()
    }

    fn new(ctx: &MetalContext) -> Self {
        Self::new_with_sliding(ctx, vec![false, false])
    }

    fn new_with_sliding(ctx: &MetalContext, sliding_layers: Vec<bool>) -> Self {
        const H: usize = 64;
        const F: usize = 96;
        const QH: usize = 2;
        const KVH: usize = 1;
        const D: usize = 32;
        const Q: usize = QH * D;
        const KV: usize = KVH * D;
        const V: usize = 8;
        Self {
            config: MuseGlimmerConfig {
                layer_count: sliding_layers.len() as u32,
                context_length: 16,
                hidden_size: H as u32,
                feed_forward_size: F as u32,
                vocab_size: V as u32,
                query_head_count: QH as u32,
                kv_head_count: KVH as u32,
                key_head_dim: D as u32,
                value_head_dim: D as u32,
                rope_theta: 500_000.0,
                rms_epsilon: 1.0e-6,
                post_norm_epsilon: 1.0e-3,
                sliding_window: 2_048,
                sliding_layers,
                logit_scale: 1.0,
                final_logit_softcap: 20.0,
                tokenizer_model: "test".into(),
                tokenizer_pre: "test".into(),
                bos_token_id: 0,
                eos_token_id: 1,
                eot_token_id: 2,
                padding_token_id: 3,
                add_bos_token: true,
                add_sep_token: false,
                tokenizer_identity_sha256: [0; 32],
                chat_template_sha256: [0; 32],
                chat_template_profile: MuseGlimmerChatTemplateProfile::MetaFixed,
            },
            token_embedding: Self::tensor(ctx, vec![H as u64, V as u64], 1),
            output_norm: Self::norm(ctx, H, 2),
            output: Self::tensor(ctx, vec![H as u64, V as u64], 3),
            attention_norm: Self::norm(ctx, H, 4),
            attention_query: Self::tensor(ctx, vec![H as u64, Q as u64], 5),
            attention_key: Self::tensor(ctx, vec![H as u64, KV as u64], 6),
            attention_value: Self::tensor(ctx, vec![H as u64, KV as u64], 7),
            attention_gate: Self::tensor(ctx, vec![H as u64, Q as u64], 8),
            attention_output: Self::tensor(ctx, vec![Q as u64, H as u64], 9),
            query_norm: Self::norm(ctx, D, 10),
            key_norm: Self::norm(ctx, D, 11),
            post_attention_norm: Self::norm(ctx, H, 12),
            feed_forward_norm: Self::norm(ctx, H, 13),
            feed_forward_gate: Self::tensor(ctx, vec![H as u64, F as u64], 14),
            feed_forward_up: Self::tensor(ctx, vec![H as u64, F as u64], 15),
            feed_forward_down: Self::tensor(ctx, vec![F as u64, H as u64], 16),
            post_feed_forward_norm: Self::norm(ctx, H, 17),
        }
    }

    fn layer(&self, sliding_attention: bool) -> MuseGlimmerMetalLayerWeights<'_> {
        MuseGlimmerMetalLayerWeights {
            attention_norm: &self.attention_norm,
            attention_query: &self.attention_query,
            attention_key: &self.attention_key,
            attention_value: &self.attention_value,
            attention_gate: &self.attention_gate,
            attention_output: &self.attention_output,
            query_norm: &self.query_norm,
            key_norm: &self.key_norm,
            post_attention_norm: &self.post_attention_norm,
            feed_forward_norm: &self.feed_forward_norm,
            feed_forward_gate: &self.feed_forward_gate,
            feed_forward_up: &self.feed_forward_up,
            feed_forward_down: &self.feed_forward_down,
            post_feed_forward_norm: &self.post_feed_forward_norm,
            sliding_attention,
        }
    }

    fn weights(&self) -> MuseGlimmerMetalModelWeights<'_> {
        MuseGlimmerMetalModelWeights {
            config: &self.config,
            token_embedding: &self.token_embedding,
            output_norm: &self.output_norm,
            output: &self.output,
            layers: self
                .config
                .sliding_layers
                .iter()
                .map(|&sliding| self.layer(sliding))
                .collect(),
        }
    }
}

struct ZeroQ8MuseModel {
    config: MuseGlimmerConfig,
    token_embedding: MetalTensor,
    output_norm: MetalTensor,
    output: MetalTensor,
    attention_norm: MetalTensor,
    attention_query: MetalTensor,
    attention_key: MetalTensor,
    attention_value: MetalTensor,
    attention_gate: MetalTensor,
    attention_output: MetalTensor,
    query_norm: MetalTensor,
    key_norm: MetalTensor,
    post_attention_norm: MetalTensor,
    feed_forward_norm: MetalTensor,
    feed_forward_gate: MetalTensor,
    feed_forward_up: MetalTensor,
    feed_forward_down: MetalTensor,
    post_feed_forward_norm: MetalTensor,
}

impl ZeroQ8MuseModel {
    fn new(ctx: &MetalContext) -> Self {
        let config = MuseGlimmerConfig::release_reference();
        let hidden = config.hidden_size as usize;
        let feed_forward = config.feed_forward_size as usize;
        let query = config.query_width().unwrap() as usize;
        let kv = config.kv_width().unwrap() as usize;
        let head_dim = config.key_head_dim as usize;
        Self {
            config,
            token_embedding: MetalTensor::zeros_f32(ctx, vec![1]).unwrap(),
            output_norm: MetalTensor::zeros_f32(ctx, vec![hidden as u64]).unwrap(),
            output: MetalTensor::zeros_f32(ctx, vec![1]).unwrap(),
            attention_norm: MetalTensor::zeros_f32(ctx, vec![hidden as u64]).unwrap(),
            attention_query: MetalTensor::zeros_dtype(
                ctx,
                vec![hidden as u64, query as u64],
                GgmlType::Q8_0,
            )
            .unwrap(),
            attention_key: MetalTensor::zeros_dtype(
                ctx,
                vec![hidden as u64, kv as u64],
                GgmlType::Q8_0,
            )
            .unwrap(),
            attention_value: MetalTensor::zeros_dtype(
                ctx,
                vec![hidden as u64, kv as u64],
                GgmlType::Q8_0,
            )
            .unwrap(),
            attention_gate: MetalTensor::zeros_dtype(
                ctx,
                vec![hidden as u64, query as u64],
                GgmlType::Q8_0,
            )
            .unwrap(),
            attention_output: MetalTensor::zeros_dtype(
                ctx,
                vec![query as u64, hidden as u64],
                GgmlType::Q8_0,
            )
            .unwrap(),
            query_norm: MetalTensor::zeros_f32(ctx, vec![head_dim as u64]).unwrap(),
            key_norm: MetalTensor::zeros_f32(ctx, vec![head_dim as u64]).unwrap(),
            post_attention_norm: MetalTensor::zeros_f32(ctx, vec![hidden as u64]).unwrap(),
            feed_forward_norm: MetalTensor::zeros_f32(ctx, vec![hidden as u64]).unwrap(),
            feed_forward_gate: MetalTensor::zeros_dtype(
                ctx,
                vec![hidden as u64, feed_forward as u64],
                GgmlType::Q8_0,
            )
            .unwrap(),
            feed_forward_up: MetalTensor::zeros_dtype(
                ctx,
                vec![hidden as u64, feed_forward as u64],
                GgmlType::Q8_0,
            )
            .unwrap(),
            feed_forward_down: MetalTensor::zeros_dtype(
                ctx,
                vec![feed_forward as u64, hidden as u64],
                GgmlType::Q8_0,
            )
            .unwrap(),
            post_feed_forward_norm: MetalTensor::zeros_f32(ctx, vec![hidden as u64]).unwrap(),
        }
    }

    fn layer(&self, sliding_attention: bool) -> MuseGlimmerMetalLayerWeights<'_> {
        MuseGlimmerMetalLayerWeights {
            attention_norm: &self.attention_norm,
            attention_query: &self.attention_query,
            attention_key: &self.attention_key,
            attention_value: &self.attention_value,
            attention_gate: &self.attention_gate,
            attention_output: &self.attention_output,
            query_norm: &self.query_norm,
            key_norm: &self.key_norm,
            post_attention_norm: &self.post_attention_norm,
            feed_forward_norm: &self.feed_forward_norm,
            feed_forward_gate: &self.feed_forward_gate,
            feed_forward_up: &self.feed_forward_up,
            feed_forward_down: &self.feed_forward_down,
            post_feed_forward_norm: &self.post_feed_forward_norm,
            sliding_attention,
        }
    }

    fn weights(&self) -> MuseGlimmerMetalModelWeights<'_> {
        MuseGlimmerMetalModelWeights {
            config: &self.config,
            token_embedding: &self.token_embedding,
            output_norm: &self.output_norm,
            output: &self.output,
            layers: self
                .config
                .sliding_layers
                .iter()
                .map(|&sliding| self.layer(sliding))
                .collect(),
        }
    }
}

#[test]
fn one_full_attention_block_vjp_bank_matches_scalar() {
    let ctx = MetalContext::new().unwrap();
    let owned = TinyMuseModel::new(&ctx);
    let weights = owned.weights();
    let geometry = MuseAttentionGeometry::from_weights(&weights).unwrap();
    let layer = &weights.layers[1];

    for (n_tokens, basis_count) in [(3usize, 1usize), (5, 3), (16, 8)] {
        let input = attention_values(n_tokens * geometry.hidden, 29, 113, 0.004);
        let capture_replay =
            replay_state(&ctx, &weights, layer, &input, n_tokens, geometry).unwrap();
        let replay = replay_readback(&capture_replay).unwrap();
        let capture = MuseGlimmerLensCapture::new(
            1,
            (0..n_tokens as u32).collect(),
            geometry.hidden,
            input.clone(),
            replay.post_attention_residuals,
            replay.post_block_residuals,
        );
        let prepared =
            MuseGlimmerPreparedMetalAttentionBlock::new(&ctx, &weights, &capture).unwrap();
        let target = attention_values(basis_count * n_tokens * geometry.hidden, 31, 127, 0.003);
        let target_tensor = from_f32(
            &ctx,
            &target,
            row_shape(geometry.hidden, basis_count * n_tokens).unwrap(),
        )
        .unwrap();
        let grad_input = MetalTensor::zeros_f32(
            &ctx,
            row_shape(geometry.hidden, basis_count * n_tokens).unwrap(),
        )
        .unwrap();
        let mut workspace =
            MuseGlimmerMetalAttentionBankWorkspace::new(&ctx, &prepared, basis_count).unwrap();

        for rule in [MuseGlimmerLensRule::J, MuseGlimmerLensRule::R] {
            let command = ctx.queue.commandBuffer().unwrap();
            let encoder = KernelEncoder::begin(&command);
            prepared
                .encode_bank(
                    &ctx,
                    &encoder,
                    &target_tensor,
                    &grad_input,
                    &mut workspace,
                    rule,
                )
                .unwrap();
            encoder.end();
            command.commit();
            command.waitUntilCompleted();
            assert!(command.error().is_none(), "{:?}", command.error());

            let mut expected = Vec::with_capacity(target.len());
            for basis in 0..basis_count {
                let start = basis * n_tokens * geometry.hidden;
                let scalar = muse_glimmer_one_full_attention_block_vjp(
                    &ctx,
                    &weights,
                    &capture,
                    &target[start..start + n_tokens * geometry.hidden],
                    rule,
                )
                .unwrap();
                expected.extend_from_slice(&scalar.input_cotangent);
            }
            let actual = read_f32(&grad_input);
            let differential = attention_differential(&actual, &expected);
            eprintln!(
                "[muse-block-vjp] B={basis_count} T={n_tokens} rule={rule:?} differential={differential:?}"
            );
            assert!(actual.iter().all(|value| value.is_finite()));
            assert!(differential.0 <= 5.0e-5, "{differential:?}");
            assert!(differential.1 <= 2.0e-4, "{differential:?}");
            assert!(differential.2 >= 0.999_999_9, "{differential:?}");
        }
    }
}

#[test]
fn sliding_attention_metal_bank_matches_cpu_fallback_j_and_r() {
    let ctx = MetalContext::new().unwrap();
    let owned = TinyMuseModel::new_with_sliding(&ctx, vec![false, true]);
    let weights = owned.weights();
    let geometry = MuseAttentionGeometry::from_weights(&weights).unwrap();

    for (n_tokens, query_count) in [(3usize, 1usize), (5, 3), (16, 32)] {
        let input = attention_values(n_tokens * geometry.hidden, 59, 163, 0.004);
        let capture_replay = replay_state(
            &ctx,
            &weights,
            &weights.layers[1],
            &input,
            n_tokens,
            geometry,
        )
        .unwrap();
        let replay = replay_readback(&capture_replay).unwrap();
        let capture = MuseGlimmerLensCapture::new(
            1,
            (0..n_tokens as u32).collect(),
            geometry.hidden,
            input,
            replay.post_attention_residuals,
            replay.post_block_residuals,
        );
        let targets = attention_values(query_count * n_tokens * geometry.hidden, 61, 167, 0.003);

        for rule in [MuseGlimmerLensRule::J, MuseGlimmerLensRule::R] {
            let cpu = muse_glimmer_one_attention_block_vjp_query_batch(
                &ctx,
                &weights,
                &capture,
                &targets,
                query_count,
                rule,
            )
            .unwrap();
            let metal = muse_glimmer_one_sliding_attention_block_vjp_metal_query_batch(
                &ctx,
                &weights,
                &capture,
                &targets,
                query_count,
                rule,
            )
            .unwrap();
            let differential =
                attention_differential(&metal.input_cotangents, &cpu.input_cotangents);
            eprintln!(
                "[muse-sliding-bank] B={query_count} T={n_tokens} rule={rule:?} differential={differential:?} metal={:?} cpu={:?}",
                metal.timings, cpu.timings,
            );
            assert_eq!(
                metal.post_attention_replay_max_abs_error,
                cpu.post_attention_replay_max_abs_error
            );
            assert_eq!(
                metal.post_block_replay_max_abs_error,
                cpu.post_block_replay_max_abs_error
            );
            assert!(differential.0 <= 5.0e-5, "{differential:?}");
            assert!(differential.1 <= 2.0e-4, "{differential:?}");
            assert!(differential.2 >= 0.999_999_9, "{differential:?}");
            assert!(metal.timings.sliding_attention_bank_command > Duration::ZERO);
            assert!(cpu.timings.attention_cpu_reverse > Duration::ZERO);
        }
    }
}

#[test]
fn adjacent_row_slab_matches_scalar_basis_and_reduction() {
    let ctx = MetalContext::new().unwrap();
    let owned = TinyMuseModel::new(&ctx);
    let weights = owned.weights();
    let geometry = MuseAttentionGeometry::from_weights(&weights).unwrap();
    let n_tokens = 5;
    let input = attention_values(n_tokens * geometry.hidden, 41, 137, 0.004);
    let replay = replay_state(
        &ctx,
        &weights,
        &weights.layers[1],
        &input,
        n_tokens,
        geometry,
    )
    .unwrap();
    let replay = replay_readback(&replay).unwrap();
    let capture = MuseGlimmerLensCapture::new(
        1,
        (0..n_tokens as u32).collect(),
        geometry.hidden,
        input,
        replay.post_attention_residuals,
        replay.post_block_residuals,
    );
    let rows = 3..14;
    let valid_positions = adjacent_fit_position_range(n_tokens, 1).unwrap();
    let positions = valid_positions.clone().collect::<Vec<_>>();

    for rule in [MuseGlimmerLensRule::J, MuseGlimmerLensRule::R] {
        let slab = muse_glimmer_fit_adjacent_full_attention_rows_batched(
            &ctx,
            &weights,
            &capture,
            rows.clone(),
            1,
            4,
            rule,
        )
        .unwrap();
        let mut expected = Vec::new();
        for row in rows.clone() {
            let mut basis = vec![0.0_f32; geometry.hidden];
            basis[row as usize] = 1.0;
            let target = muse_glimmer_positioned_target_cotangent(
                &basis,
                n_tokens,
                geometry.hidden,
                &positions,
            )
            .unwrap();
            let scalar =
                muse_glimmer_one_full_attention_block_vjp(&ctx, &weights, &capture, &target, rule)
                    .unwrap();
            expected.extend(
                mean_reduce_position_rows(
                    &scalar.input_cotangent,
                    n_tokens,
                    geometry.hidden,
                    valid_positions.clone(),
                )
                .unwrap(),
            );
        }
        let differential = attention_differential(&slab.values, &expected);
        assert_eq!(slab.source_block, 0);
        assert_eq!(slab.target_block, 1);
        assert_eq!(slab.rule, rule);
        assert_eq!((slab.row_start, slab.row_end), (rows.start, rows.end));
        assert_eq!(slab.n_tokens, n_tokens);
        assert_eq!(slab.n_valid_positions, valid_positions.len());
        assert_eq!(slab.hidden_size, geometry.hidden);
        assert_eq!(
            slab.row_values(rows.start).unwrap(),
            &slab.values[..geometry.hidden]
        );
        assert!(slab.row_values(rows.end).is_none());
        assert!(differential.0 <= 5.0e-5, "{differential:?}");
        assert!(differential.1 <= 2.0e-4, "{differential:?}");
        assert!(differential.2 >= 0.999_999_9, "{differential:?}");
    }
}

#[allow(clippy::too_many_arguments)]
fn chunk_major_full_transport_reference(
    ctx: &MetalContext,
    weights: &MuseGlimmerMetalModelWeights<'_>,
    captures: &MuseGlimmerLensCaptureBank,
    target_block: u32,
    source_layers: &[u32],
    output_row_ids: &[u32],
    skip_first: usize,
    query_batch_size: usize,
    rule: MuseGlimmerLensRule,
) -> MuseGlimmerBatchedFullTransportRowFit {
    let hidden_size = captures.hidden_size();
    let valid_positions = adjacent_fit_position_range(captures.n_tokens(), skip_first).unwrap();
    let source_elements = output_row_ids.len() * hidden_size;
    let query_elements = captures.n_tokens() * hidden_size;
    let traversal = composition_traversal(target_block, source_layers).unwrap();
    let mut values = vec![0.0_f32; source_layers.len() * source_elements];
    let mut diagnostics = composition_diagnostics(weights, &traversal);
    let mut timings = MuseGlimmerQueryBatchVjpTimings::default();
    for (chunk_slot, rows) in output_row_ids.chunks(query_batch_size).enumerate() {
        let mut targets = vec![0.0_f32; rows.len() * query_elements];
        for (query, &row) in rows.iter().enumerate() {
            for position in valid_positions.clone() {
                targets[query * query_elements + position * hidden_size + row as usize] = 1.0;
            }
        }
        let composed = muse_glimmer_composed_vjp_query_batch(
            ctx,
            weights,
            captures,
            target_block,
            source_layers,
            &targets,
            rows.len(),
            rule,
        )
        .unwrap();
        timings.add_assign(composed.timings);
        for (destination, observed) in diagnostics.iter_mut().zip(&composed.diagnostics) {
            destination.post_attention_replay_max_abs_error = destination
                .post_attention_replay_max_abs_error
                .max(observed.post_attention_replay_max_abs_error);
            destination.post_block_replay_max_abs_error = destination
                .post_block_replay_max_abs_error
                .max(observed.post_block_replay_max_abs_error);
        }
        let first_row_slot = chunk_slot * query_batch_size;
        for source_slot in 0..source_layers.len() {
            for query in 0..rows.len() {
                let source_query = composed.source_query_values(source_slot, query).unwrap();
                let reduced = mean_reduce_position_rows(
                    source_query,
                    captures.n_tokens(),
                    hidden_size,
                    valid_positions.clone(),
                )
                .unwrap();
                let row_slot = first_row_slot + query;
                let destination = source_slot * source_elements + row_slot * hidden_size;
                values[destination..destination + hidden_size].copy_from_slice(&reduced);
            }
        }
    }
    MuseGlimmerBatchedFullTransportRowFit {
        source_layers: source_layers.to_vec(),
        target_block,
        method: rule,
        output_row_ids: output_row_ids.to_vec(),
        n_valid_positions: valid_positions.len(),
        hidden_size,
        query_batch_size,
        values,
        diagnostics,
        timings,
    }
}

#[test]
fn composed_query_batch_uses_full_bank_and_sliding_fallback() {
    let ctx = MetalContext::new().unwrap();
    let owned = TinyMuseModel::new_with_sliding(&ctx, vec![false, true, false]);
    let weights = owned.weights();
    let geometry = MuseAttentionGeometry::from_weights(&weights).unwrap();
    let n_tokens = 5;
    let input_1 = attention_values(n_tokens * geometry.hidden, 43, 139, 0.004);
    let state_1 = replay_state(
        &ctx,
        &weights,
        &weights.layers[1],
        &input_1,
        n_tokens,
        geometry,
    )
    .unwrap();
    let replay_1 = replay_readback(&state_1).unwrap();
    let input_2 = replay_1.post_block_residuals.clone();
    let state_2 = replay_state(
        &ctx,
        &weights,
        &weights.layers[2],
        &input_2,
        n_tokens,
        geometry,
    )
    .unwrap();
    let replay_2 = replay_readback(&state_2).unwrap();
    let mut inputs = input_1;
    inputs.extend_from_slice(&input_2);
    let mut post_attention = replay_1.post_attention_residuals;
    post_attention.extend_from_slice(&replay_2.post_attention_residuals);
    let mut post_block = replay_1.post_block_residuals;
    post_block.extend_from_slice(&replay_2.post_block_residuals);
    let captures = MuseGlimmerLensCaptureBank::new(
        vec![1, 2],
        (0..n_tokens as u32).collect(),
        geometry.hidden,
        inputs,
        post_attention,
        post_block,
    );
    let query_count = 3;
    let query_elements = n_tokens * geometry.hidden;
    let targets = attention_values(query_count * query_elements, 47, 149, 0.003);
    let capture_1 = captures.block_capture(1).unwrap();
    let capture_2 = captures.block_capture(2).unwrap();

    for rule in [MuseGlimmerLensRule::J, MuseGlimmerLensRule::R] {
        let composed = muse_glimmer_composed_vjp_query_batch(
            &ctx,
            &weights,
            &captures,
            2,
            &[0, 1],
            &targets,
            query_count,
            rule,
        )
        .unwrap();
        assert_eq!(
            composed
                .diagnostics
                .iter()
                .map(|diagnostic| diagnostic.kind)
                .collect::<Vec<_>>(),
            [
                MuseGlimmerAttentionBlockKind::Full,
                MuseGlimmerAttentionBlockKind::Sliding,
            ]
        );
        assert!(composed.timings.full_attention_bank_command > Duration::ZERO);
        assert!(composed.timings.attention_cpu_reverse > Duration::ZERO);
        for query in 0..query_count {
            let target = &targets[query * query_elements..(query + 1) * query_elements];
            let through_2 =
                muse_glimmer_one_attention_block_vjp(&ctx, &weights, &capture_2, target, rule)
                    .unwrap();
            let through_1 = muse_glimmer_one_attention_block_vjp(
                &ctx,
                &weights,
                &capture_1,
                &through_2.input_cotangent,
                rule,
            )
            .unwrap();
            for (source_slot, expected) in [
                (1, through_2.input_cotangent.as_slice()),
                (0, through_1.input_cotangent.as_slice()),
            ] {
                let differential = attention_differential(
                    composed.source_query_values(source_slot, query).unwrap(),
                    expected,
                );
                assert!(differential.0 <= 5.0e-5, "{differential:?}");
                assert!(differential.1 <= 2.0e-4, "{differential:?}");
                assert!(differential.2 >= 0.999_999_9, "{differential:?}");
            }
        }
    }
}

#[test]
fn block_major_full_transport_matches_chunk_major_j_and_r() {
    let ctx = MetalContext::new().unwrap();
    let owned = TinyMuseModel::new_with_sliding(&ctx, vec![false, true, false]);
    let weights = owned.weights();
    let geometry = MuseAttentionGeometry::from_weights(&weights).unwrap();
    let n_tokens = 5;
    let input_1 = attention_values(n_tokens * geometry.hidden, 53, 157, 0.004);
    let state_1 = replay_state(
        &ctx,
        &weights,
        &weights.layers[1],
        &input_1,
        n_tokens,
        geometry,
    )
    .unwrap();
    let replay_1 = replay_readback(&state_1).unwrap();
    let input_2 = replay_1.post_block_residuals.clone();
    let state_2 = replay_state(
        &ctx,
        &weights,
        &weights.layers[2],
        &input_2,
        n_tokens,
        geometry,
    )
    .unwrap();
    let replay_2 = replay_readback(&state_2).unwrap();
    let mut inputs = input_1;
    inputs.extend_from_slice(&input_2);
    let mut post_attention = replay_1.post_attention_residuals;
    post_attention.extend_from_slice(&replay_2.post_attention_residuals);
    let mut post_block = replay_1.post_block_residuals;
    post_block.extend_from_slice(&replay_2.post_block_residuals);
    let captures = MuseGlimmerLensCaptureBank::new(
        vec![1, 2],
        (0..n_tokens as u32).collect(),
        geometry.hidden,
        inputs,
        post_attention,
        post_block,
    );
    let output_rows = (0..33).collect::<Vec<_>>();

    for rule in [MuseGlimmerLensRule::J, MuseGlimmerLensRule::R] {
        let reference = chunk_major_full_transport_reference(
            &ctx,
            &weights,
            &captures,
            2,
            &[0, 1],
            &output_rows,
            1,
            32,
            rule,
        );
        let block_major = muse_glimmer_fit_full_transport_rows_to_sources_batched(
            &ctx,
            &weights,
            &captures,
            2,
            &[0, 1],
            &output_rows,
            1,
            32,
            rule,
        )
        .unwrap();
        let differential = attention_differential(&block_major.values, &reference.values);
        assert_eq!(block_major.source_layers, reference.source_layers);
        assert_eq!(block_major.output_row_ids, reference.output_row_ids);
        assert_eq!(block_major.n_valid_positions, reference.n_valid_positions);
        assert_eq!(block_major.hidden_size, reference.hidden_size);
        assert_eq!(block_major.query_batch_size, 32);
        assert_eq!(block_major.diagnostics, reference.diagnostics);
        assert!(block_major.source_row_values(0, 32).is_some());
        assert!(block_major.source_row_values(1, 33).is_none());
        assert!(differential.0 <= 5.0e-5, "{rule:?}: {differential:?}");
        assert!(differential.1 <= 2.0e-4, "{rule:?}: {differential:?}");
        assert!(differential.2 >= 0.999_999_9, "{rule:?}: {differential:?}");
        let components = block_major.timings.replay
            + block_major.timings.full_attention_bank_command
            + block_major.timings.sliding_attention_bank_command
            + block_major.timings.feed_forward_reverse
            + block_major.timings.attention_output_reverse
            + block_major.timings.attention_cpu_reverse
            + block_major.timings.attention_input_reverse;
        assert!(components <= block_major.timings.total);
    }
}

#[test]
#[ignore = "bounded released-Q8 R256 block-major falsifier"]
fn real_q8_block_major_r256_falsifier() {
    let mode = std::env::var("MUSE_GLIMMER_RUN_R256_FALSIFIER").unwrap_or_default();
    if mode != "1" && mode != "candidate" {
        eprintln!("set MUSE_GLIMMER_RUN_R256_FALSIFIER=1 for A/B or =candidate for candidate-only");
        return;
    }
    let path = crate::test_fixtures::MUSE_GLIMMER_Q8_0.required();
    let gguf = GgufFile::open(path).expect("open Muse Q8 target");
    let ctx = MetalContext::new().expect("open Metal context");
    let plan =
        MuseGlimmerMetalWeightPlan::for_release(&ctx, &gguf).expect("qualify Muse Q8 target");
    let admitted = plan
        .admit(ctx.memory_signals())
        .expect("admit Muse Q8 weights");
    let realized =
        MuseGlimmerMetalWeights::realize(&ctx, &gguf, admitted).expect("realize Muse Q8 weights");
    let weights = realized.into_weights();
    let tokens = (1..=MUSE_GLIMMER_LENS_MAX_PROMPT_TOKENS as u32).collect::<Vec<_>>();
    let block_ids = (1..=51).collect::<Vec<_>>();
    let mut session = MuseGlimmerTextSession::new(&ctx, weights.config(), tokens.len())
        .expect("allocate bounded Muse session");
    let forward = MuseGlimmerTextForward::new(&ctx, &weights).expect("bind Muse forward");
    let capture_started = Instant::now();
    let captures = forward
        .capture_fresh_lens_prompt_blocks(&tokens, &block_ids, &mut session)
        .expect("capture all traversed blocks");
    let capture_elapsed = capture_started.elapsed();
    let bound = MuseGlimmerMetalModelWeights::bind(&weights).expect("bind resident weights");
    let source_layers = (0..=50).collect::<Vec<_>>();
    let output_rows = (0..256).collect::<Vec<_>>();

    let candidate_started = Instant::now();
    let candidate = muse_glimmer_fit_full_transport_rows_to_sources_batched(
        &ctx,
        &bound,
        &captures,
        51,
        &source_layers,
        &output_rows,
        4,
        32,
        MuseGlimmerLensRule::R,
    )
    .expect("run block-major R256 candidate");
    let candidate_elapsed = candidate_started.elapsed();
    if mode == "candidate" {
        let components = candidate.timings.replay
            + candidate.timings.full_attention_bank_command
            + candidate.timings.sliding_attention_bank_command
            + candidate.timings.feed_forward_reverse
            + candidate.timings.attention_output_reverse
            + candidate.timings.attention_cpu_reverse
            + candidate.timings.attention_input_reverse;
        eprintln!(
            "[muse-full-r-r256-candidate] capture={capture_elapsed:?} candidate={candidate_elapsed:?} timings={:?}",
            candidate.timings,
        );
        assert_eq!(candidate.values.len(), 51 * 256 * 6_656);
        assert!(components <= candidate.timings.total);
        assert!(candidate.timings.full_attention_bank_command > Duration::ZERO);
        assert!(candidate.timings.sliding_attention_bank_command > Duration::ZERO);
        assert_eq!(candidate.timings.attention_cpu_reverse, Duration::ZERO);
        return;
    }
    let control_started = Instant::now();
    let control = chunk_major_full_transport_reference(
        &ctx,
        &bound,
        &captures,
        51,
        &source_layers,
        &output_rows,
        4,
        32,
        MuseGlimmerLensRule::R,
    );
    let control_elapsed = control_started.elapsed();

    let mut max_abs = 0.0_f64;
    let mut squared_error = 0.0_f64;
    for (&actual, &expected) in candidate.values.iter().zip(&control.values) {
        let difference = f64::from(actual) - f64::from(expected);
        max_abs = max_abs.max(difference.abs());
        squared_error += difference * difference;
    }
    let rms = (squared_error / candidate.values.len() as f64).sqrt();
    let speedup = control_elapsed.as_secs_f64() / candidate_elapsed.as_secs_f64();
    eprintln!(
        "[muse-full-r-r256] capture={capture_elapsed:?} candidate={candidate_elapsed:?} control={control_elapsed:?} speedup={speedup:.6} max_abs={max_abs:.9e} rms={rms:.9e} candidate_timings={:?} control_timings={:?}",
        candidate.timings, control.timings,
    );
    assert_eq!(candidate.diagnostics, control.diagnostics);
    assert_eq!(candidate.values.len(), 51 * 256 * 6_656);
    assert!(max_abs <= 1.862_645_15e-7, "max_abs={max_abs:.9e}");
    assert!(rms <= 1.030_751_24e-8, "rms={rms:.9e}");
    assert!(
        candidate_elapsed.as_secs_f64() <= control_elapsed.as_secs_f64() * 0.85,
        "candidate {candidate_elapsed:?} did not clear the 15% wall gate against {control_elapsed:?}"
    );
}

#[test]
#[ignore = "bounded released-Q8 sliding block-50 Metal falsifier"]
fn real_q8_sliding_block_50_metal_falsifier() {
    if std::env::var("MUSE_GLIMMER_RUN_SLIDING_FALSIFIER").as_deref() != Ok("1") {
        eprintln!("set MUSE_GLIMMER_RUN_SLIDING_FALSIFIER=1 to run the bounded Q8 falsifier");
        return;
    }
    let test_started = Instant::now();
    let path = crate::test_fixtures::MUSE_GLIMMER_Q8_0.required();
    let gguf = GgufFile::open(path).expect("open Muse Q8 target");
    let ctx = MetalContext::new().expect("open Metal context");
    let plan =
        MuseGlimmerMetalWeightPlan::for_release(&ctx, &gguf).expect("qualify Muse Q8 target");
    let admitted = plan
        .admit(ctx.memory_signals())
        .expect("admit Muse Q8 weights");
    let realized =
        MuseGlimmerMetalWeights::realize(&ctx, &gguf, admitted).expect("realize Muse Q8 weights");
    let weights = realized.into_weights();
    let tokens = (1..=MUSE_GLIMMER_LENS_MAX_PROMPT_TOKENS as u32).collect::<Vec<_>>();
    let mut session = MuseGlimmerTextSession::new(&ctx, weights.config(), tokens.len())
        .expect("allocate bounded Muse session");
    let forward = MuseGlimmerTextForward::new(&ctx, &weights).expect("bind Muse forward");
    let captures = forward
        .capture_fresh_lens_prompt_blocks(&tokens, &[50], &mut session)
        .expect("capture sliding block 50");
    let capture = captures
        .block_capture(50)
        .expect("extract sliding block 50");
    let bound = MuseGlimmerMetalModelWeights::bind(&weights).expect("bind resident weights");
    let prepared = MuseGlimmerPreparedMetalAttentionBlock::new_sliding(&ctx, &bound, &capture)
        .expect("prepare shared sliding replay");
    let query_count = MUSE_GLIMMER_QUERY_BATCH_MAX;
    let targets = attention_values(
        query_count * captures.n_tokens() * captures.hidden_size(),
        67,
        173,
        0.003,
    );
    let (post_attention_error, post_block_error) = prepared.replay_max_abs_errors();
    let run_candidate = || {
        reverse_prepared_metal_attention_block_vjp_query_batch(
            &ctx,
            &prepared,
            &targets,
            query_count,
            MuseGlimmerLensRule::R,
        )
        .expect("run sliding Metal candidate")
    };
    let run_control = || {
        reverse_attention_block_vjp_query_batch_from_replay(
            &ctx,
            prepared.config,
            prepared.target_block,
            &prepared.layer,
            &prepared.replay,
            post_attention_error,
            post_block_error,
            &targets,
            query_count,
            MuseGlimmerLensRule::R,
        )
        .expect("run sliding CPU control")
    };

    let warm_candidate = run_candidate();
    let warm_control = run_control();
    let warm_differential = attention_differential(
        &warm_candidate.input_cotangents,
        &warm_control.input_cotangents,
    );
    assert!(warm_differential.0 <= 5.0e-5, "{warm_differential:?}");
    assert!(warm_differential.1 <= 2.0e-4, "{warm_differential:?}");
    assert!(warm_differential.2 >= 0.999_999_9, "{warm_differential:?}");

    let mut candidate_samples = Vec::with_capacity(5);
    let mut control_samples = Vec::with_capacity(5);
    let mut worst_differential = (0.0_f64, 0.0_f64, 1.0_f64);
    for pair in 0..5 {
        let (candidate, control) = if pair % 2 == 0 {
            (run_candidate(), run_control())
        } else {
            let control = run_control();
            let candidate = run_candidate();
            (candidate, control)
        };
        let differential =
            attention_differential(&candidate.input_cotangents, &control.input_cotangents);
        worst_differential.0 = worst_differential.0.max(differential.0);
        worst_differential.1 = worst_differential.1.max(differential.1);
        worst_differential.2 = worst_differential.2.min(differential.2);
        candidate_samples.push(candidate.timings.total.as_secs_f64());
        control_samples.push(control.timings.total.as_secs_f64());
    }
    candidate_samples.sort_by(f64::total_cmp);
    control_samples.sort_by(f64::total_cmp);
    let candidate_median = candidate_samples[2];
    let control_median = control_samples[2];
    let speedup = control_median / candidate_median;
    eprintln!(
        "[muse-sliding-block50] candidate={candidate_samples:?} control={control_samples:?} medians={candidate_median:.9}/{control_median:.9}s speedup={speedup:.6} differential={worst_differential:?} total={:?}",
        test_started.elapsed(),
    );
    assert_eq!(warm_candidate.input_cotangents.len(), 3_407_872);
    assert!(worst_differential.0 <= 5.0e-5, "{worst_differential:?}");
    assert!(worst_differential.1 <= 2.0e-4, "{worst_differential:?}");
    assert!(
        worst_differential.2 >= 0.999_999_9,
        "{worst_differential:?}"
    );
    assert!(
        candidate_median <= control_median * 0.85,
        "candidate median {candidate_median:.9}s did not clear the 15% gate against {control_median:.9}s"
    );
    assert!(test_started.elapsed() < Duration::from_secs(180));
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct MuseBankDispatchShape {
    kernel: String,
    grid: [u64; 3],
    threads: [u64; 3],
}

#[derive(Clone, Debug)]
struct MuseBankStageProfile {
    stage_ms: [f64; 4],
    raw_coverage: f64,
    transition_ambiguity: f64,
    gap_ms: f64,
    overlap_ms: f64,
}

#[derive(Debug)]
struct MuseBankProfileArm {
    gpu_ms: f64,
    output_bits: Vec<u32>,
    trace: crate::metal::KernelTraceCounters,
    dispatches: Vec<MuseBankDispatchShape>,
    stages: Option<MuseBankStageProfile>,
}

fn resolve_muse_bank_stage_profile(
    ctx: &MetalContext,
    samples: &crate::metal::MetalTimestampSampleBuffer,
    command_gpu_ms: f64,
) -> MuseBankStageProfile {
    let timestamps = ctx.resolve_timestamp_samples(samples, 8).unwrap();
    let first = timestamps[0];
    let last = timestamps[7];
    let span_ticks = last.checked_sub(first).expect("ordered sampled span");
    assert!(span_ticks > 0);

    let mut stage_ticks = 0u64;
    let mut gap_ticks = 0u64;
    let mut overlap_ticks = 0u64;
    let mut previous_start = None;
    let mut previous_end = None;
    let mut durations = [0u64; 4];
    for stage in 0..4 {
        let start = timestamps[stage * 2];
        let end = timestamps[stage * 2 + 1];
        assert!(end >= start, "stage {stage} has inverted timestamps");
        assert!(previous_start.is_none_or(|previous| start >= previous));
        assert!(previous_end.is_none_or(|previous| end >= previous));
        if let Some(previous) = previous_end {
            if start >= previous {
                gap_ticks += start - previous;
            } else {
                overlap_ticks += previous - start;
            }
        }
        durations[stage] = end - start;
        stage_ticks += durations[stage];
        previous_start = Some(start);
        previous_end = Some(end);
    }
    assert_eq!(
        stage_ticks as i128 + gap_ticks as i128 - overlap_ticks as i128,
        span_ticks as i128,
    );

    let scale_ms_per_tick = command_gpu_ms / span_ticks as f64;
    MuseBankStageProfile {
        stage_ms: durations.map(|ticks| ticks as f64 * scale_ms_per_tick),
        raw_coverage: span_ticks as f64 * 1.0e-6 / command_gpu_ms,
        transition_ambiguity: (gap_ticks + overlap_ticks) as f64 / span_ticks as f64,
        gap_ms: gap_ticks as f64 * scale_ms_per_tick,
        overlap_ms: overlap_ticks as f64 * scale_ms_per_tick,
    }
}

fn run_zero_q8_muse_bank_profile_arm(
    ctx: &MetalContext,
    prepared: &MuseGlimmerPreparedMetalAttentionBlock<'_>,
    current: &MetalTensor,
    next: &MetalTensor,
    workspace: &mut MuseGlimmerMetalAttentionBankWorkspace,
    sampled: bool,
) -> MuseBankProfileArm {
    let samples = sampled.then(|| {
        ctx.timestamp_sample_buffer(8)
            .expect("stage-boundary sample buffer")
    });
    crate::metal::dispatch_census_begin();
    let trace_guard = crate::metal::kernel_trace_begin();
    let command = ctx.queue.commandBuffer().unwrap();
    if let Some(samples) = &samples {
        let geometry = prepared.replay.geometry;
        let n_tokens = prepared.replay.n_tokens;
        let basis_count = workspace.scratch.basis_count;
        let bank_rows = checked_mul(basis_count, n_tokens, "profile bank rows").unwrap();
        let tensors = &prepared.replay.tensors;

        let encoder = KernelEncoder::try_begin_sampled(&command, samples, 0, 1, false).unwrap();
        encode_muse_glimmer_one_attention_block_vjp_bank_feed_forward(
            ctx,
            &encoder,
            prepared.config,
            &prepared.layer,
            tensors,
            geometry,
            n_tokens,
            bank_rows,
            current,
            &workspace.scratch,
            MuseGlimmerLensRule::R,
        )
        .unwrap();
        encoder.end();

        let encoder = KernelEncoder::try_begin_sampled(&command, samples, 2, 3, false).unwrap();
        encode_muse_glimmer_one_attention_block_vjp_bank_attention_output(
            ctx,
            &encoder,
            prepared.config,
            &prepared.layer,
            tensors,
            geometry,
            n_tokens,
            bank_rows,
            &workspace.scratch,
            MuseGlimmerLensRule::R,
        )
        .unwrap();
        encoder.end();

        let encoder = KernelEncoder::try_begin_sampled(&command, samples, 4, 5, false).unwrap();
        encode_muse_glimmer_one_attention_block_vjp_bank_causal_gqa(
            ctx,
            &encoder,
            &prepared.layer,
            tensors,
            geometry,
            basis_count,
            n_tokens,
            bank_rows,
            &workspace.scratch,
        )
        .unwrap();
        encoder.end();

        let encoder = KernelEncoder::try_begin_sampled(&command, samples, 6, 7, false).unwrap();
        encode_muse_glimmer_one_attention_block_vjp_bank_attention_input(
            ctx,
            &encoder,
            prepared.config,
            &prepared.layer,
            tensors,
            geometry,
            n_tokens,
            bank_rows,
            next,
            &workspace.scratch,
            MuseGlimmerLensRule::R,
        )
        .unwrap();
        encoder.end();
    } else {
        let encoder = KernelEncoder::begin(&command);
        prepared
            .encode_bank(
                ctx,
                &encoder,
                current,
                next,
                workspace,
                MuseGlimmerLensRule::R,
            )
            .unwrap();
        encoder.end();
    }
    let trace = crate::metal::kernel_trace_snapshot();
    let dispatches = crate::metal::dispatch_census_take()
        .into_iter()
        .map(|row| MuseBankDispatchShape {
            kernel: row.kernel,
            grid: [row.grid_width, row.grid_height, row.grid_depth],
            threads: [row.threads_width, row.threads_height, row.threads_depth],
        })
        .collect::<Vec<_>>();
    drop(trace_guard);

    command.commit();
    command.waitUntilCompleted();
    assert!(command.error().is_none(), "{:?}", command.error());
    let start = command.GPUStartTime();
    let end = command.GPUEndTime();
    assert!(start.is_finite() && end.is_finite() && start > 0.0 && end > start);
    let gpu_ms = (end - start) * 1.0e3;
    let stages = samples
        .as_ref()
        .map(|samples| resolve_muse_bank_stage_profile(ctx, samples, gpu_ms));
    MuseBankProfileArm {
        gpu_ms,
        output_bits: read_f32(next).into_iter().map(f32::to_bits).collect(),
        trace,
        dispatches,
        stages,
    }
}

#[test]
#[ignore = "bounded zero-Q8 Muse full-block B8/B32 timing and attribution gate"]
fn profile_one_full_attention_block_vjp_bank_zero_q8() {
    use std::time::Instant;

    fn drift(values: &[f64]) -> f64 {
        let minimum = values.iter().copied().reduce(f64::min).unwrap();
        let maximum = values.iter().copied().reduce(f64::max).unwrap();
        2.0 * (maximum - minimum) / (maximum + minimum)
    }

    fn median(values: impl IntoIterator<Item = f64>) -> f64 {
        let mut values = values.into_iter().collect::<Vec<_>>();
        values.sort_by(f64::total_cmp);
        values[values.len() / 2]
    }

    let test_started = Instant::now();
    let ctx = MetalContext::new().unwrap();
    let model_prepare_start = Instant::now();
    let owned = ZeroQ8MuseModel::new(&ctx);
    let weights = owned.weights();
    const TARGET_BLOCK: u32 = 51;
    const N_TOKENS: usize = 16;
    let hidden = weights.config.hidden_size as usize;
    assert!(!weights.layers[TARGET_BLOCK as usize].sliding_attention);
    let input = vec![0.0f32; N_TOKENS * hidden];
    let capture = MuseGlimmerLensCapture::new(
        TARGET_BLOCK,
        (0..N_TOKENS as u32).collect(),
        hidden,
        input.clone(),
        input.clone(),
        input,
    );
    let prepared = MuseGlimmerPreparedMetalAttentionBlock::new(&ctx, &weights, &capture).unwrap();
    let model_prepare_seconds = model_prepare_start.elapsed().as_secs_f64();
    assert!(
        model_prepare_seconds <= 30.0,
        "model preparation took {model_prepare_seconds:.3} s"
    );

    let mut results = Vec::new();
    for (basis_count, hard_stop_ms) in [(8usize, 100.0f64), (32, 300.0)] {
        let current_values = attention_values(basis_count * N_TOKENS * hidden, 37, 131, 0.002);
        let current_bits = current_values
            .iter()
            .copied()
            .map(f32::to_bits)
            .collect::<Vec<_>>();
        let current = from_f32(
            &ctx,
            &current_values,
            row_shape(hidden, basis_count * N_TOKENS).unwrap(),
        )
        .unwrap();
        let next = MetalTensor::zeros_f32(&ctx, row_shape(hidden, basis_count * N_TOKENS).unwrap())
            .unwrap();
        let mut workspace =
            MuseGlimmerMetalAttentionBankWorkspace::new(&ctx, &prepared, basis_count).unwrap();

        let warm_control = run_zero_q8_muse_bank_profile_arm(
            &ctx,
            &prepared,
            &current,
            &next,
            &mut workspace,
            false,
        );
        assert_eq!(warm_control.output_bits, current_bits);
        assert_eq!(warm_control.trace.encoders, 1);
        assert_eq!(warm_control.trace.concurrent_encoders, 0);
        assert_eq!(
            warm_control.trace.dispatches as usize,
            warm_control.dispatches.len()
        );
        assert!(warm_control.stages.is_none());

        let mut run = |sampled| {
            let arm = run_zero_q8_muse_bank_profile_arm(
                &ctx,
                &prepared,
                &current,
                &next,
                &mut workspace,
                sampled,
            );
            assert_eq!(arm.output_bits, current_bits);
            assert_eq!(arm.dispatches, warm_control.dispatches);
            assert_eq!(arm.trace.dispatches, warm_control.trace.dispatches);
            assert_eq!(arm.trace.encoders, if sampled { 4 } else { 1 });
            assert_eq!(arm.trace.concurrent_encoders, 0);
            assert_eq!(arm.trace.dispatches as usize, arm.dispatches.len());
            assert_eq!(arm.stages.is_some(), sampled);
            arm
        };

        if basis_count == 8 {
            let controls = [run(false), run(false), run(false)];
            let first = controls[0].gpu_ms;
            assert!(
                first <= hard_stop_ms,
                "B={basis_count} first command {first:.3} ms exceeds {hard_stop_ms:.0} ms"
            );
            let samples = controls.map(|arm| arm.gpu_ms);
            let mean = samples.iter().sum::<f64>() / samples.len() as f64;
            eprintln!(
                "[muse-block-vjp-zero-q8] B={basis_count} samples_ms={samples:?} mean_ms={mean:.6}"
            );
            results.push((basis_count, mean));
            continue;
        }

        let warm_sampled = run(true);
        assert_eq!(warm_sampled.output_bits, warm_control.output_bits);
        let mut controls = Vec::with_capacity(4);
        let mut sampled = Vec::with_capacity(3);
        for sampled_arm in [false, true, false, true, false, true, false] {
            let arm = run(sampled_arm);
            if sampled_arm {
                sampled.push(arm);
            } else {
                controls.push(arm);
            }
        }

        let control_gpu_ms = controls.iter().map(|arm| arm.gpu_ms).collect::<Vec<_>>();
        let sampled_gpu_ms = sampled.iter().map(|arm| arm.gpu_ms).collect::<Vec<_>>();
        assert!(
            control_gpu_ms[0] <= hard_stop_ms,
            "B={basis_count} first command {:.3} ms exceeds {hard_stop_ms:.0} ms",
            control_gpu_ms[0],
        );
        assert!(
            drift(&control_gpu_ms) <= 0.05,
            "control drift {control_gpu_ms:?}"
        );
        assert!(
            drift(&sampled_gpu_ms) <= 0.05,
            "sampled drift {sampled_gpu_ms:?}"
        );

        let interpolated = control_gpu_ms
            .windows(2)
            .map(|pair| (pair[0] + pair[1]) * 0.5)
            .collect::<Vec<_>>();
        let perturbation = sampled_gpu_ms
            .iter()
            .zip(&interpolated)
            .map(|(sampled, control)| sampled / control - 1.0)
            .collect::<Vec<_>>();
        let sample_is_valid = sampled
            .iter()
            .zip(&perturbation)
            .map(|(arm, perturbation)| {
                let profile = arm.stages.as_ref().unwrap();
                (profile.raw_coverage - 1.0).abs() <= 0.005
                    && profile.transition_ambiguity <= 0.025
                    && perturbation.abs() <= 0.10
            })
            .collect::<Vec<_>>();
        let accepted = sample_is_valid
            .iter()
            .enumerate()
            .filter_map(|(index, &valid)| valid.then_some(index))
            .collect::<Vec<_>>();
        assert!(accepted.len() >= 2, "sample validity {sample_is_valid:?}");

        let stage_shares = sampled
            .iter()
            .map(|arm| {
                let profile = arm.stages.as_ref().unwrap();
                profile.stage_ms.map(|stage| stage / arm.gpu_ms)
            })
            .collect::<Vec<_>>();
        let stage_repeat_delta: [f64; 4] = std::array::from_fn(|stage| {
            let values = accepted.iter().map(|&index| stage_shares[index][stage]);
            let minimum = values.clone().reduce(f64::min).unwrap();
            let maximum = values.reduce(f64::max).unwrap();
            maximum - minimum
        });
        assert!(
            stage_repeat_delta.iter().all(|&delta| delta <= 0.02),
            "stage-share repeat deltas {stage_repeat_delta:?}"
        );

        let stage_median_ms: [f64; 4] = std::array::from_fn(|stage| {
            median(
                accepted
                    .iter()
                    .map(|&index| sampled[index].stages.as_ref().unwrap().stage_ms[stage]),
            )
        });
        let raw_coverage = sampled
            .iter()
            .map(|arm| arm.stages.as_ref().unwrap().raw_coverage)
            .collect::<Vec<_>>();
        let ambiguity = sampled
            .iter()
            .map(|arm| arm.stages.as_ref().unwrap().transition_ambiguity)
            .collect::<Vec<_>>();
        let gap_ms = sampled
            .iter()
            .map(|arm| arm.stages.as_ref().unwrap().gap_ms)
            .collect::<Vec<_>>();
        let overlap_ms = sampled
            .iter()
            .map(|arm| arm.stages.as_ref().unwrap().overlap_ms)
            .collect::<Vec<_>>();
        eprintln!(
            "[muse-block-vjp-attribution] control_ms={control_gpu_ms:?} sampled_ms={sampled_gpu_ms:?} perturbation={perturbation:?} coverage={raw_coverage:?} ambiguity={ambiguity:?} gap_ms={gap_ms:?} overlap_ms={overlap_ms:?} valid={sample_is_valid:?} stage_median_ms={stage_median_ms:?} stage_repeat_delta={stage_repeat_delta:?}"
        );

        let mean = control_gpu_ms.iter().sum::<f64>() / control_gpu_ms.len() as f64;
        results.push((basis_count, mean));
    }
    let throughput_ratio = results[1].1 / results[0].1;
    let selected_basis = if throughput_ratio > 5.0 { 8 } else { 32 };
    eprintln!(
        "[muse-block-vjp-zero-q8] model_prepare_s={model_prepare_seconds:.3} B32/B8={throughput_ratio:.3} selected_B={selected_basis}"
    );
    assert!(test_started.elapsed() <= Duration::from_secs(180));
}

#[test]
fn cpu_causal_gqa_vjp_matches_directional_finite_difference() {
    let geometry = tiny_geometry();
    let tokens = 3;
    let q = (0..tokens * geometry.query)
        .map(|index| index as f32 * 0.07 - 0.3)
        .collect::<Vec<_>>();
    let k = (0..tokens * geometry.kv)
        .map(|index| index as f32 * -0.09 + 0.2)
        .collect::<Vec<_>>();
    let v = (0..tokens * geometry.kv)
        .map(|index| index as f32 * 0.11 - 0.1)
        .collect::<Vec<_>>();
    let gate = (0..tokens * geometry.query)
        .map(|index| index as f32 * 0.05 - 0.2)
        .collect::<Vec<_>>();
    let cotangent = (0..tokens * geometry.query)
        .map(|index| (index as f32 * 0.31).sin())
        .collect::<Vec<_>>();
    let forward = cpu_causal_gqa_forward(&q, &k, &v, &gate, tokens, geometry).unwrap();
    let vjp =
        cpu_causal_gqa_vjp(&q, &k, &v, &gate, &cotangent, tokens, geometry, &forward).unwrap();
    let epsilon = 1e-3_f32;
    for (primal, gradient, label) in [
        (q.as_slice(), vjp.grad_q.as_slice(), "Q"),
        (k.as_slice(), vjp.grad_k.as_slice(), "K"),
        (v.as_slice(), vjp.grad_v.as_slice(), "V"),
        (gate.as_slice(), vjp.grad_gate.as_slice(), "gate"),
    ] {
        let direction = (0..primal.len())
            .map(|index| (index as f32 * 0.73 + 0.2).cos())
            .collect::<Vec<_>>();
        let plus = primal
            .iter()
            .zip(&direction)
            .map(|(&value, &direction)| value + epsilon * direction)
            .collect::<Vec<_>>();
        let minus = primal
            .iter()
            .zip(&direction)
            .map(|(&value, &direction)| value - epsilon * direction)
            .collect::<Vec<_>>();
        let evaluate = |replacement: &[f32]| {
            let (q_value, k_value, v_value, gate_value) = match label {
                "Q" => (replacement, k.as_slice(), v.as_slice(), gate.as_slice()),
                "K" => (q.as_slice(), replacement, v.as_slice(), gate.as_slice()),
                "V" => (q.as_slice(), k.as_slice(), replacement, gate.as_slice()),
                _ => (q.as_slice(), k.as_slice(), v.as_slice(), replacement),
            };
            dot(
                &cpu_causal_gqa_forward(q_value, k_value, v_value, gate_value, tokens, geometry)
                    .unwrap()
                    .gated_output,
                &cotangent,
            )
        };
        let finite_difference = (evaluate(&plus) - evaluate(&minus)) / (2.0 * epsilon);
        let reverse = dot(gradient, &direction);
        assert!(
            (finite_difference - reverse).abs() < 3e-4,
            "{label}: finite difference {finite_difference}, reverse {reverse}"
        );
    }
}

#[test]
fn adjacent_pair_rope_matches_production_kernel_and_inverse() {
    const TOKENS: usize = 3;
    const Q_HEADS: usize = 2;
    const KV_HEADS: usize = 1;
    const HEAD_DIM: usize = 4;
    const THETA: f32 = 10_000.0;
    let q_original = (0..TOKENS * Q_HEADS * HEAD_DIM)
        .map(|index| index as f32 * 0.07 - 0.5)
        .collect::<Vec<_>>();
    let k_original = (0..TOKENS * KV_HEADS * HEAD_DIM)
        .map(|index| index as f32 * -0.09 + 0.4)
        .collect::<Vec<_>>();
    let mut q_cpu = q_original.clone();
    let mut k_cpu = k_original.clone();
    adjacent_pair_rope_rows_in_place(&mut q_cpu, TOKENS, Q_HEADS, HEAD_DIM, THETA, false).unwrap();
    adjacent_pair_rope_rows_in_place(&mut k_cpu, TOKENS, KV_HEADS, HEAD_DIM, THETA, false).unwrap();

    let ctx = MetalContext::new().unwrap();
    let q_gpu = from_f32(
        &ctx,
        &q_original,
        row_shape(Q_HEADS * HEAD_DIM, TOKENS).unwrap(),
    )
    .unwrap();
    let k_gpu = from_f32(
        &ctx,
        &k_original,
        row_shape(KV_HEADS * HEAD_DIM, TOKENS).unwrap(),
    )
    .unwrap();
    run_command(&ctx, |encoder| {
        for token in 0..TOKENS {
            encode_muse_glimmer_rope_adjacent_pair_in_place_f32(
                &ctx,
                encoder,
                &row_view(&q_gpu, token, Q_HEADS * HEAD_DIM),
                &row_view(&k_gpu, token, KV_HEADS * HEAD_DIM),
                Q_HEADS,
                KV_HEADS,
                HEAD_DIM,
                token as u32,
                THETA,
            )?;
        }
        Ok(())
    })
    .unwrap();
    for (&cpu, gpu) in q_cpu.iter().zip(read_f32(&q_gpu)) {
        assert!((cpu - gpu).abs() < 2e-6);
    }
    for (&cpu, gpu) in k_cpu.iter().zip(read_f32(&k_gpu)) {
        assert!((cpu - gpu).abs() < 2e-6);
    }
    adjacent_pair_rope_rows_in_place(&mut q_cpu, TOKENS, Q_HEADS, HEAD_DIM, THETA, true).unwrap();
    adjacent_pair_rope_rows_in_place(&mut k_cpu, TOKENS, KV_HEADS, HEAD_DIM, THETA, true).unwrap();
    for (&actual, &expected) in q_cpu.iter().zip(&q_original) {
        assert!((actual - expected).abs() < 2e-6);
    }
    for (&actual, &expected) in k_cpu.iter().zip(&k_original) {
        assert!((actual - expected).abs() < 2e-6);
    }
}

#[test]
fn r_rules_differ_from_j_at_residual_norm_and_swiglu_only() {
    assert_ne!(
        MuseGlimmerLensRule::J.rms_norm_rule(MuseGlimmerRmsNormSite::ResidualPreAttention),
        MuseGlimmerLensRule::R.rms_norm_rule(MuseGlimmerRmsNormSite::ResidualPreAttention)
    );
    assert_ne!(
        MuseGlimmerLensRule::J.swiglu_rule(),
        MuseGlimmerLensRule::R.swiglu_rule()
    );
    assert_eq!(
        MuseGlimmerLensRule::J.rms_norm_rule(MuseGlimmerRmsNormSite::AttentionQuery),
        MuseGlimmerLensRule::R.rms_norm_rule(MuseGlimmerRmsNormSite::AttentionQuery)
    );
    assert_eq!(
        MuseGlimmerLensRule::J.attention_rule(),
        MuseGlimmerLensRule::R.attention_rule()
    );
}

#[test]
fn r_kernels_match_detached_rms_and_identity_half_references() {
    let ctx = MetalContext::new().unwrap();
    let x = [1.5_f32, -0.5];
    let weight = [0.75_f32, 1.25];
    let incoming = [0.4_f32, -0.8];
    let epsilon = 1e-5_f32;
    let j_rms = rms_norm_mul_vjp_rows_f32_readback_for_test(
        &ctx,
        &x,
        &weight,
        &incoming,
        1,
        2,
        epsilon,
        MuseGlimmerLensRule::J.rms_norm_rule(MuseGlimmerRmsNormSite::ResidualPreAttention),
    )
    .unwrap();
    let r_rms = rms_norm_mul_vjp_rows_f32_readback_for_test(
        &ctx,
        &x,
        &weight,
        &incoming,
        1,
        2,
        epsilon,
        MuseGlimmerLensRule::R.rms_norm_rule(MuseGlimmerRmsNormSite::ResidualPreAttention),
    )
    .unwrap();
    let scale = ((x[0] * x[0] + x[1] * x[1]) / 2.0 + epsilon).sqrt().recip();
    let r_rms_reference = [
        incoming[0] * weight[0] * scale,
        incoming[1] * weight[1] * scale,
    ];
    for (&actual, &reference) in r_rms.iter().zip(&r_rms_reference) {
        assert!((actual - reference).abs() < 1e-6);
    }
    assert!(
        j_rms
            .iter()
            .zip(&r_rms)
            .any(|(&j, &r)| (j - r).abs() > 1e-3)
    );

    let gate = [0.7_f32, -1.1];
    let up = [1.3_f32, -0.4];
    let grad = [0.6_f32, 0.9];
    let (j_gate, j_up) = silu_mul_vjp_f32_readback_for_test(
        &ctx,
        &gate,
        &up,
        &grad,
        1,
        2,
        MuseGlimmerLensRule::J.swiglu_rule(),
    )
    .unwrap();
    let (r_gate, r_up) = silu_mul_vjp_f32_readback_for_test(
        &ctx,
        &gate,
        &up,
        &grad,
        1,
        2,
        MuseGlimmerLensRule::R.swiglu_rule(),
    )
    .unwrap();
    for index in 0..gate.len() {
        let sigmoid_gate = sigmoid(gate[index]);
        assert!((r_gate[index] - 0.5 * grad[index] * up[index] * sigmoid_gate).abs() < 1e-6);
        assert!((r_up[index] - 0.5 * grad[index] * gate[index] * sigmoid_gate).abs() < 1e-6);
    }
    assert!(
        j_gate
            .iter()
            .zip(&r_gate)
            .any(|(&j, &r)| (j - r).abs() > 1e-3)
    );
    assert!(j_up.iter().zip(&r_up).any(|(&j, &r)| (j - r).abs() > 1e-3));
}

#[test]
fn request_validation_rejects_bad_shapes_and_non_finite_values() {
    let capture =
        MuseGlimmerLensCapture::new(3, vec![1], 2, vec![0.0; 2], vec![0.0; 2], vec![0.0; 2]);
    validate_request(&capture, &[1.0, 2.0], 2).unwrap();
    assert!(validate_request(&capture, &[1.0], 2).is_err());
    assert!(validate_request(&capture, &[1.0, f32::NAN], 2).is_err());
    let block_zero =
        MuseGlimmerLensCapture::new(0, vec![1], 2, vec![0.0; 2], vec![0.0; 2], vec![0.0; 2]);
    assert!(validate_request(&block_zero, &[1.0, 2.0], 2).is_err());
}

#[test]
fn query_batch_layout_and_cpu_attention_preserve_isolation() {
    let geometry = tiny_geometry();
    let n_tokens = 3;
    let query_count = 3;
    let q = (0..n_tokens * geometry.query)
        .map(|index| index as f32 * 0.07 - 0.3)
        .collect::<Vec<_>>();
    let k = (0..n_tokens * geometry.kv)
        .map(|index| index as f32 * -0.09 + 0.2)
        .collect::<Vec<_>>();
    let v = (0..n_tokens * geometry.kv)
        .map(|index| index as f32 * 0.11 - 0.1)
        .collect::<Vec<_>>();
    let gate = (0..n_tokens * geometry.query)
        .map(|index| index as f32 * 0.05 - 0.2)
        .collect::<Vec<_>>();
    let query_major = (0..query_count * n_tokens * geometry.query)
        .map(|index| (index as f32 * 0.173 + 0.4).sin())
        .collect::<Vec<_>>();
    let token_query =
        query_major_to_token_query(&query_major, query_count, n_tokens, geometry.query).unwrap();
    assert_eq!(
        token_query_to_query_major(&token_query, query_count, n_tokens, geometry.query).unwrap(),
        query_major
    );
    let forward = cpu_causal_gqa_forward(&q, &k, &v, &gate, n_tokens, geometry).unwrap();
    let batch = cpu_causal_gqa_vjp_query_batch(
        &q,
        &k,
        &v,
        &gate,
        &token_query,
        query_count,
        n_tokens,
        geometry,
        &forward,
        false,
    )
    .unwrap();
    for query_slot in 0..query_count {
        let query_elements = n_tokens * geometry.query;
        let scalar = cpu_causal_gqa_vjp(
            &q,
            &k,
            &v,
            &gate,
            &query_major[query_slot * query_elements..(query_slot + 1) * query_elements],
            n_tokens,
            geometry,
            &forward,
        )
        .unwrap();
        for token in 0..n_tokens {
            for head in 0..geometry.q_heads {
                let scalar_start = (token * geometry.q_heads + head) * geometry.head_dim;
                let batch_start = ((token * geometry.q_heads + head) * query_count + query_slot)
                    * geometry.head_dim;
                assert_eq!(
                    &batch.grad_q_head_query[batch_start..batch_start + geometry.head_dim],
                    &scalar.grad_q[scalar_start..scalar_start + geometry.head_dim]
                );
            }
            for head in 0..geometry.kv_heads {
                let scalar_start = (token * geometry.kv_heads + head) * geometry.head_dim;
                let batch_start = ((token * geometry.kv_heads + head) * query_count + query_slot)
                    * geometry.head_dim;
                assert_eq!(
                    &batch.grad_k_head_query[batch_start..batch_start + geometry.head_dim],
                    &scalar.grad_k[scalar_start..scalar_start + geometry.head_dim]
                );
            }
            for (width, scalar_values, batch_values) in [
                (
                    geometry.kv,
                    scalar.grad_v.as_slice(),
                    batch.grad_v_token_query.as_slice(),
                ),
                (
                    geometry.query,
                    scalar.grad_gate.as_slice(),
                    batch.grad_gate_token_query.as_slice(),
                ),
            ] {
                let scalar_start = token * width;
                let batch_start = (token * query_count + query_slot) * width;
                assert_eq!(
                    &batch_values[batch_start..batch_start + width],
                    &scalar_values[scalar_start..scalar_start + width]
                );
            }
        }
    }
}

#[test]
fn query_batch_validation_rejects_malformed_shapes_counts_and_values() {
    let capture =
        MuseGlimmerLensCapture::new(3, vec![1, 2], 2, vec![0.0; 4], vec![0.0; 4], vec![0.0; 4]);
    validate_query_batch_request(&capture, &[1.0; 8], 2, 2).unwrap();
    assert!(validate_query_batch_request(&capture, &[], 0, 2).is_err());
    assert!(
        validate_query_batch_request(
            &capture,
            &[1.0; 4 * (MUSE_GLIMMER_QUERY_BATCH_MAX + 1)],
            MUSE_GLIMMER_QUERY_BATCH_MAX + 1,
            2,
        )
        .is_err()
    );
    assert!(validate_query_batch_request(&capture, &[1.0; 7], 2, 2).is_err());
    let mut non_finite = [1.0; 8];
    non_finite[7] = f32::NAN;
    assert!(validate_query_batch_request(&capture, &non_finite, 2, 2).is_err());
}

#[test]
fn replay_difference_rejects_non_finite_operands() {
    assert_eq!(max_abs_difference(&[1.0, -2.0], &[1.5, -1.0]).unwrap(), 1.0);
    assert!(max_abs_difference(&[f32::NAN], &[0.0]).is_err());
    assert!(max_abs_difference(&[0.0], &[f32::INFINITY]).is_err());
}

#[test]
fn positioned_target_cotangent_places_rows_and_rejects_bad_requests() {
    let values = muse_glimmer_positioned_target_cotangent(&[1.0, -2.0], 3, 2, &[1, 2]).unwrap();
    assert_eq!(values, [0.0, 0.0, 1.0, -2.0, 1.0, -2.0]);
    assert!(muse_glimmer_positioned_target_cotangent(&[1.0], 3, 2, &[1]).is_err());
    assert!(muse_glimmer_positioned_target_cotangent(&[1.0, 2.0], 3, 2, &[3]).is_err());
    assert!(muse_glimmer_positioned_target_cotangent(&[1.0, 2.0], 3, 2, &[1, 1]).is_err());
    assert!(muse_glimmer_positioned_target_cotangent(&[1.0, f32::NAN], 3, 2, &[1]).is_err());
}

#[test]
fn adjacent_fit_positions_exclude_final_and_honor_skip_first() {
    assert_eq!(adjacent_fit_position_range(5, 0).unwrap(), 0..4);
    assert_eq!(adjacent_fit_position_range(5, 2).unwrap(), 2..4);
    assert!(adjacent_fit_position_range(1, 0).is_err());
    assert!(adjacent_fit_position_range(5, 4).is_err());
}

#[test]
fn adjacent_fit_reduction_means_only_valid_source_rows() {
    let rows = [1.0, 10.0, 3.0, 30.0, 5.0, 50.0, 100.0, 1000.0];
    assert_eq!(
        mean_reduce_position_rows(&rows, 4, 2, 1..3).unwrap(),
        [4.0, 40.0]
    );
    assert!(mean_reduce_position_rows(&rows, 4, 2, 3..3).is_err());
    assert!(mean_reduce_position_rows(&rows[..6], 4, 2, 1..3).is_err());
}

#[test]
fn composition_traversal_is_reverse_and_checkpoints_after_each_block() {
    let traversal = composition_traversal(51, &[49, 50]).unwrap();
    assert_eq!(traversal, [51, 50]);
    let checkpoints = traversal
        .iter()
        .map(|&block| (block, block - 1))
        .collect::<Vec<_>>();
    assert_eq!(checkpoints, [(51, 50), (50, 49)]);
    assert_eq!(composition_traversal(51, &[50]).unwrap(), [51]);
    assert!(composition_traversal(51, &[]).is_err());
    assert!(composition_traversal(51, &[50, 49]).is_err());
    assert!(composition_traversal(51, &[50, 50]).is_err());
    assert!(composition_traversal(51, &[51]).is_err());
}

#[test]
fn full_transport_rows_validate_and_expose_source_row_layout() {
    validate_output_row_ids(&[0, 2], 3).unwrap();
    validate_output_row_ids(
        &(0..MUSE_GLIMMER_FULL_TRANSPORT_MAX_ROWS_PER_SHARD as u32).collect::<Vec<_>>(),
        MUSE_GLIMMER_FULL_TRANSPORT_MAX_ROWS_PER_SHARD,
    )
    .unwrap();
    assert!(validate_output_row_ids(&[], 3).is_err());
    assert!(validate_output_row_ids(&[1, 1], 3).is_err());
    assert!(validate_output_row_ids(&[2, 1], 3).is_err());
    assert!(validate_output_row_ids(&[0, 3], 3).is_err());
    assert!(
        validate_output_row_ids(
            &(0..=MUSE_GLIMMER_FULL_TRANSPORT_MAX_ROWS_PER_SHARD as u32).collect::<Vec<_>>(),
            MUSE_GLIMMER_FULL_TRANSPORT_MAX_ROWS_PER_SHARD + 1,
        )
        .is_err()
    );

    let fit = MuseGlimmerFullTransportRowFit {
        source_layers: vec![4, 9],
        target_block: 10,
        method: MuseGlimmerLensRule::R,
        output_row_ids: vec![0, 2],
        n_valid_positions: 2,
        hidden_size: 3,
        values: (0..12).map(|value| value as f32).collect(),
        diagnostics: Vec::new(),
    };
    assert_eq!(
        fit.source_values(1).unwrap(),
        [6.0, 7.0, 8.0, 9.0, 10.0, 11.0]
    );
    assert_eq!(fit.source_row_values(0, 1).unwrap(), [3.0, 4.0, 5.0]);
    assert_eq!(fit.source_row_values(1, 0).unwrap(), [6.0, 7.0, 8.0]);
    assert!(fit.source_row_values(0, 2).is_none());
    assert!(fit.source_values(2).is_none());
}

#[test]
fn basis_rows_match_generic_covectors_with_orientation_and_mean_reduction() {
    let source_layers = [0, 1];
    let traversal = [2, 1];
    let mut diagnostics = traversal
        .iter()
        .map(|&block| MuseGlimmerBlockReplayDiagnostic {
            block,
            kind: MuseGlimmerAttentionBlockKind::Full,
            post_attention_replay_max_abs_error: 0.0,
            post_block_replay_max_abs_error: 0.0,
        })
        .collect::<Vec<_>>();
    let reverse = |block: u32, current: &[f32]| {
        let mut input = Vec::with_capacity(current.len());
        for (position, row) in current.chunks_exact(3).enumerate() {
            if block == 2 {
                let scale = position as f32 + 1.0;
                input.extend([
                    scale * (row[0] + 2.0 * row[1]),
                    scale * (3.0 * row[0] + row[2]),
                    scale * 4.0 * row[2],
                ]);
            } else {
                input.extend([2.0 * row[0], 3.0 * row[1], 5.0 * row[2]]);
            }
        }
        Ok(MuseGlimmerOneBlockVjp {
            target_block: block,
            rule: MuseGlimmerLensRule::J,
            n_tokens: 3,
            hidden_size: 3,
            input_cotangent: input,
            post_attention_replay_max_abs_error: block as f32 * 0.01,
            post_block_replay_max_abs_error: block as f32 * 0.02,
        })
    };
    let basis_rows = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0];
    let fitted = compose_covector_rows_to_sources(
        &source_layers,
        &traversal,
        &basis_rows,
        2,
        3,
        3,
        0..2,
        &mut diagnostics,
        reverse,
    )
    .unwrap();
    assert_eq!(
        fitted,
        [3.0, 13.5, 0.0, 0.0, 4.5, 30.0, 1.5, 4.5, 0.0, 0.0, 1.5, 6.0]
    );

    let mut scalar_diagnostics = diagnostics
        .iter()
        .map(|diagnostic| MuseGlimmerBlockReplayDiagnostic {
            block: diagnostic.block,
            kind: diagnostic.kind,
            post_attention_replay_max_abs_error: 0.0,
            post_block_replay_max_abs_error: 0.0,
        })
        .collect::<Vec<_>>();
    let scalar = compose_covector_rows_to_sources(
        &source_layers,
        &traversal,
        &[0.0, 0.0, 1.0],
        1,
        3,
        3,
        0..2,
        &mut scalar_diagnostics,
        reverse,
    )
    .unwrap();
    assert_eq!(&fitted[3..6], &scalar[0..3]);
    assert_eq!(&fitted[9..12], &scalar[3..6]);
    assert_eq!(diagnostics, scalar_diagnostics);
}

#[test]
#[ignore = "requires the authenticated local Unsloth Muse Glimmer Q8_0 target"]
fn real_q8_block_51_replay_and_vjp_smoke() {
    let path = crate::test_fixtures::MUSE_GLIMMER_Q8_0.required();
    let gguf = GgufFile::open(path).expect("open Muse Q8 target");
    let ctx = MetalContext::new().expect("open Metal context");
    let plan =
        MuseGlimmerMetalWeightPlan::for_release(&ctx, &gguf).expect("qualify Muse Q8 target");
    let admitted = plan
        .admit(ctx.memory_signals())
        .expect("admit Muse Q8 weights");
    let realized =
        MuseGlimmerMetalWeights::realize(&ctx, &gguf, admitted).expect("realize Muse Q8 weights");
    let weights = realized.into_weights();
    let tokens = [weights.config().bos_token_id, 19_873, 24];
    let mut session = MuseGlimmerTextSession::new(&ctx, weights.config(), tokens.len())
        .expect("allocate bounded Muse session");
    let forward = MuseGlimmerTextForward::new(&ctx, &weights).expect("bind Muse forward");
    let capture = forward
        .capture_fresh_lens_prompt(&tokens, 51, &mut session)
        .expect("capture block 51");
    let covectors =
        muse_glimmer_selected_token_covectors(&ctx, &weights, &[weights.config().eos_token_id])
            .expect("extract EOS covector");
    let target_cotangent = muse_glimmer_positioned_target_cotangent(
        covectors.token_values(0).unwrap(),
        tokens.len(),
        weights.config().hidden_size as usize,
        &[2],
    )
    .expect("place EOS covector on final prompt row");
    let bound = MuseGlimmerMetalModelWeights::bind(&weights).expect("bind resident weights");
    let fit = muse_glimmer_fit_adjacent_full_attention_selected_tokens(
        &ctx,
        &bound,
        &capture,
        &covectors,
        0,
        MuseGlimmerLensRule::J,
    )
    .expect("fit adjacent selected-token direction");
    assert_eq!(fit.source_block, 50);
    assert_eq!(fit.target_block, 51);
    assert_eq!(fit.n_valid_positions, 2);
    assert_eq!(fit.token_ids, [weights.config().eos_token_id]);
    assert_eq!(fit.values.len(), weights.config().hidden_size as usize);
    assert!(fit.values.iter().all(|value| value.is_finite()));
    let j = muse_glimmer_one_full_attention_block_vjp(
        &ctx,
        &bound,
        &capture,
        &target_cotangent,
        MuseGlimmerLensRule::J,
    )
    .expect("J replay and reverse block 51");
    let r = muse_glimmer_one_full_attention_block_vjp(
        &ctx,
        &bound,
        &capture,
        &target_cotangent,
        MuseGlimmerLensRule::R,
    )
    .expect("R replay and reverse block 51");
    let direction = (0..capture.input_residuals().len())
        .map(|index| (index as f32 * 0.013_37 + 0.41).sin())
        .collect::<Vec<_>>();
    let epsilon = 0.04_f32;
    let plus = capture
        .input_residuals()
        .iter()
        .zip(&direction)
        .map(|(&value, &direction)| value + epsilon * direction)
        .collect::<Vec<_>>();
    let minus = capture
        .input_residuals()
        .iter()
        .zip(&direction)
        .map(|(&value, &direction)| value - epsilon * direction)
        .collect::<Vec<_>>();
    let plus_replay =
        muse_glimmer_replay_one_full_attention_block(&ctx, &bound, 51, &plus, tokens.len())
            .expect("positive directional replay");
    let minus_replay =
        muse_glimmer_replay_one_full_attention_block(&ctx, &bound, 51, &minus, tokens.len())
            .expect("negative directional replay");
    let finite_difference = (dot(&plus_replay.post_block_residuals, &target_cotangent)
        - dot(&minus_replay.post_block_residuals, &target_cotangent))
        / (2.0 * epsilon);
    let reverse_directional = dot(&j.input_cotangent, &direction);
    let absolute_error = (finite_difference - reverse_directional).abs();
    let relative_error = absolute_error
        / finite_difference
            .abs()
            .max(reverse_directional.abs())
            .max(1e-6);
    let jr_max_abs_difference = j
        .input_cotangent
        .iter()
        .zip(&r.input_cotangent)
        .map(|(&j, &r)| (j - r).abs())
        .fold(0.0_f32, f32::max);
    eprintln!(
        "Muse Q8 block 51 T=3: post_attention_max_abs={} post_block_max_abs={} epsilon={} finite_difference={} reverse={} absolute_error={} relative_error={} jr_grad_max_abs_difference={} fit_max_abs={}",
        j.post_attention_replay_max_abs_error,
        j.post_block_replay_max_abs_error,
        epsilon,
        finite_difference,
        reverse_directional,
        absolute_error,
        relative_error,
        jr_max_abs_difference,
        fit.values
            .iter()
            .map(|value| value.abs())
            .fold(0.0_f32, f32::max),
    );
    assert!(j.input_cotangent.iter().all(|value| value.is_finite()));
    assert!(r.input_cotangent.iter().all(|value| value.is_finite()));
    assert!(j.post_attention_replay_max_abs_error < 0.022);
    assert!(j.post_block_replay_max_abs_error < 0.022);
    assert!(relative_error < 0.005);
    assert!(jr_max_abs_difference > 1e-4);
}

#[test]
#[ignore = "requires the authenticated local Unsloth Muse Glimmer Q8_0 target"]
fn real_q8_sliding_block_50_replay_and_vjp_smoke() {
    let path = crate::test_fixtures::MUSE_GLIMMER_Q8_0.required();
    let gguf = GgufFile::open(path).expect("open Muse Q8 target");
    let ctx = MetalContext::new().expect("open Metal context");
    let plan =
        MuseGlimmerMetalWeightPlan::for_release(&ctx, &gguf).expect("qualify Muse Q8 target");
    let admitted = plan
        .admit(ctx.memory_signals())
        .expect("admit Muse Q8 weights");
    let realized =
        MuseGlimmerMetalWeights::realize(&ctx, &gguf, admitted).expect("realize Muse Q8 weights");
    let weights = realized.into_weights();
    let tokens = [weights.config().bos_token_id, 19_873, 24];
    let mut session = MuseGlimmerTextSession::new(&ctx, weights.config(), tokens.len())
        .expect("allocate bounded Muse session");
    let forward = MuseGlimmerTextForward::new(&ctx, &weights).expect("bind Muse forward");
    let capture = forward
        .capture_fresh_lens_prompt(&tokens, 50, &mut session)
        .expect("capture sliding block 50");
    assert_eq!(capture.target_block(), 50);
    let covectors =
        muse_glimmer_selected_token_covectors(&ctx, &weights, &[weights.config().eos_token_id])
            .expect("extract EOS covector");
    let target_cotangent = muse_glimmer_positioned_target_cotangent(
        covectors.token_values(0).unwrap(),
        tokens.len(),
        weights.config().hidden_size as usize,
        &[2],
    )
    .expect("place EOS covector on final prompt row");
    let bound = MuseGlimmerMetalModelWeights::bind(&weights).expect("bind resident weights");
    assert!(
        muse_glimmer_one_full_attention_block_vjp(
            &ctx,
            &bound,
            &capture,
            &target_cotangent,
            MuseGlimmerLensRule::J,
        )
        .is_err()
    );
    let j = muse_glimmer_one_attention_block_vjp(
        &ctx,
        &bound,
        &capture,
        &target_cotangent,
        MuseGlimmerLensRule::J,
    )
    .expect("J replay and reverse sliding block 50");
    let r = muse_glimmer_one_attention_block_vjp(
        &ctx,
        &bound,
        &capture,
        &target_cotangent,
        MuseGlimmerLensRule::R,
    )
    .expect("R replay and reverse sliding block 50");
    let direction = (0..capture.input_residuals().len())
        .map(|index| (index as f32 * 0.017_11 + 0.29).cos())
        .collect::<Vec<_>>();
    let epsilon = 0.04_f32;
    let plus = capture
        .input_residuals()
        .iter()
        .zip(&direction)
        .map(|(&value, &direction)| value + epsilon * direction)
        .collect::<Vec<_>>();
    let minus = capture
        .input_residuals()
        .iter()
        .zip(&direction)
        .map(|(&value, &direction)| value - epsilon * direction)
        .collect::<Vec<_>>();
    let plus_replay =
        muse_glimmer_replay_one_attention_block(&ctx, &bound, 50, &plus, tokens.len())
            .expect("positive sliding directional replay");
    let minus_replay =
        muse_glimmer_replay_one_attention_block(&ctx, &bound, 50, &minus, tokens.len())
            .expect("negative sliding directional replay");
    let finite_difference = (dot(&plus_replay.post_block_residuals, &target_cotangent)
        - dot(&minus_replay.post_block_residuals, &target_cotangent))
        / (2.0 * epsilon);
    let reverse_directional = dot(&j.input_cotangent, &direction);
    let absolute_error = (finite_difference - reverse_directional).abs();
    let relative_error = absolute_error
        / finite_difference
            .abs()
            .max(reverse_directional.abs())
            .max(1e-6);
    let jr_max_abs_difference = j
        .input_cotangent
        .iter()
        .zip(&r.input_cotangent)
        .map(|(&j, &r)| (j - r).abs())
        .fold(0.0_f32, f32::max);
    eprintln!(
        "Muse Q8 sliding block 50/source 49 T=3: post_attention_max_abs={} post_block_max_abs={} epsilon={} finite_difference={} reverse={} absolute_error={} relative_error={} jr_grad_max_abs_difference={}",
        j.post_attention_replay_max_abs_error,
        j.post_block_replay_max_abs_error,
        epsilon,
        finite_difference,
        reverse_directional,
        absolute_error,
        relative_error,
        jr_max_abs_difference,
    );
    assert!(j.input_cotangent.iter().all(|value| value.is_finite()));
    assert!(r.input_cotangent.iter().all(|value| value.is_finite()));
    assert!(j.post_attention_replay_max_abs_error < 0.015);
    assert!(j.post_block_replay_max_abs_error < 0.015);
    assert!(relative_error < 0.002);
    assert!(jr_max_abs_difference > 1e-4);
}

#[test]
#[ignore = "requires the authenticated local Unsloth Muse Glimmer Q8_0 target"]
fn real_q8_two_block_multi_source_composition_smoke() {
    let path = crate::test_fixtures::MUSE_GLIMMER_Q8_0.required();
    let gguf = GgufFile::open(path).expect("open Muse Q8 target");
    let ctx = MetalContext::new().expect("open Metal context");
    let plan =
        MuseGlimmerMetalWeightPlan::for_release(&ctx, &gguf).expect("qualify Muse Q8 target");
    let admitted = plan
        .admit(ctx.memory_signals())
        .expect("admit Muse Q8 weights");
    let realized =
        MuseGlimmerMetalWeights::realize(&ctx, &gguf, admitted).expect("realize Muse Q8 weights");
    let weights = realized.into_weights();
    let tokens = [weights.config().bos_token_id, 19_873, 24];
    let mut session = MuseGlimmerTextSession::new(&ctx, weights.config(), tokens.len())
        .expect("allocate bounded Muse session");
    let forward = MuseGlimmerTextForward::new(&ctx, &weights).expect("bind Muse forward");
    let captures = forward
        .capture_fresh_lens_prompt_blocks(&tokens, &[50, 51], &mut session)
        .expect("capture blocks 50 and 51 in one prompt pass");
    assert_eq!(captures.block_ids(), [50, 51]);
    let capture_50 = captures.block_capture(50).expect("extract block 50");
    let capture_51 = captures.block_capture(51).expect("extract block 51");
    assert_eq!(
        capture_51.input_residuals(),
        capture_50.post_block_residuals()
    );
    let covectors =
        muse_glimmer_selected_token_covectors(&ctx, &weights, &[weights.config().eos_token_id])
            .expect("extract EOS covector");
    let bound = MuseGlimmerMetalModelWeights::bind(&weights).expect("bind resident weights");
    let adjacent = muse_glimmer_fit_adjacent_full_attention_selected_tokens(
        &ctx,
        &bound,
        &capture_51,
        &covectors,
        0,
        MuseGlimmerLensRule::J,
    )
    .expect("fit adjacent source 50");
    let composed_j = muse_glimmer_fit_selected_tokens_to_sources(
        &ctx,
        &bound,
        &captures,
        51,
        &[49, 50],
        &covectors,
        0,
        MuseGlimmerLensRule::J,
    )
    .expect("fit J sources 49 and 50");
    let composed_r = muse_glimmer_fit_selected_tokens_to_sources(
        &ctx,
        &bound,
        &captures,
        51,
        &[49, 50],
        &covectors,
        0,
        MuseGlimmerLensRule::R,
    )
    .expect("fit R sources 49 and 50");
    let basis = muse_glimmer_fit_full_transport_rows_to_sources(
        &ctx,
        &bound,
        &captures,
        51,
        &[49, 50],
        &[0],
        0,
        MuseGlimmerLensRule::J,
    )
    .expect("fit hidden basis row zero");
    assert_eq!(composed_j.source_layers, [49, 50]);
    assert_eq!(composed_j.target_block, 51);
    assert_eq!(composed_j.n_valid_positions, 2);
    assert_eq!(composed_j.diagnostics.len(), 2);
    assert_eq!(composed_j.diagnostics[0].block, 51);
    assert_eq!(
        composed_j.diagnostics[0].kind,
        MuseGlimmerAttentionBlockKind::Full
    );
    assert_eq!(composed_j.diagnostics[1].block, 50);
    assert_eq!(
        composed_j.diagnostics[1].kind,
        MuseGlimmerAttentionBlockKind::Sliding
    );
    let source_50_difference = max_abs_difference(
        composed_j.source_token_values(1, 0).unwrap(),
        adjacent.token_values(0).unwrap(),
    )
    .unwrap();

    let valid_positions = [0, 1];
    let target_cotangent = muse_glimmer_positioned_target_cotangent(
        covectors.token_values(0).unwrap(),
        tokens.len(),
        weights.config().hidden_size as usize,
        &valid_positions,
    )
    .unwrap();
    let explicit_51 = muse_glimmer_one_attention_block_vjp(
        &ctx,
        &bound,
        &capture_51,
        &target_cotangent,
        MuseGlimmerLensRule::J,
    )
    .expect("explicit reverse block 51");
    let explicit_50 = muse_glimmer_one_attention_block_vjp(
        &ctx,
        &bound,
        &capture_50,
        &explicit_51.input_cotangent,
        MuseGlimmerLensRule::J,
    )
    .expect("explicit reverse block 50");
    let explicit_source_49 = mean_reduce_position_rows(
        &explicit_50.input_cotangent,
        tokens.len(),
        weights.config().hidden_size as usize,
        0..2,
    )
    .unwrap();
    let source_49_difference = max_abs_difference(
        composed_j.source_token_values(0, 0).unwrap(),
        &explicit_source_49,
    )
    .unwrap();

    let basis_target = muse_glimmer_positioned_target_cotangent(
        &[1.0]
            .into_iter()
            .chain(std::iter::repeat_n(
                0.0,
                weights.config().hidden_size as usize - 1,
            ))
            .collect::<Vec<_>>(),
        tokens.len(),
        weights.config().hidden_size as usize,
        &valid_positions,
    )
    .unwrap();
    let basis_explicit_51 = muse_glimmer_one_attention_block_vjp(
        &ctx,
        &bound,
        &capture_51,
        &basis_target,
        MuseGlimmerLensRule::J,
    )
    .unwrap();
    let basis_explicit_50 = muse_glimmer_one_attention_block_vjp(
        &ctx,
        &bound,
        &capture_50,
        &basis_explicit_51.input_cotangent,
        MuseGlimmerLensRule::J,
    )
    .unwrap();
    let basis_source_49 = mean_reduce_position_rows(
        &basis_explicit_50.input_cotangent,
        tokens.len(),
        weights.config().hidden_size as usize,
        0..2,
    )
    .unwrap();
    assert_eq!(basis.output_row_ids, [0]);
    assert_eq!(basis.source_row_values(0, 0).unwrap(), basis_source_49);

    let direction = (0..capture_50.input_residuals().len())
        .map(|index| (index as f32 * 0.019_31 + 0.17).sin())
        .collect::<Vec<_>>();
    let epsilon = 0.04_f32;
    let plus = capture_50
        .input_residuals()
        .iter()
        .zip(&direction)
        .map(|(&value, &direction)| value + epsilon * direction)
        .collect::<Vec<_>>();
    let minus = capture_50
        .input_residuals()
        .iter()
        .zip(&direction)
        .map(|(&value, &direction)| value - epsilon * direction)
        .collect::<Vec<_>>();
    let chain = |input: &[f32]| {
        let block_50 =
            muse_glimmer_replay_one_attention_block(&ctx, &bound, 50, input, tokens.len()).unwrap();
        muse_glimmer_replay_one_attention_block(
            &ctx,
            &bound,
            51,
            &block_50.post_block_residuals,
            tokens.len(),
        )
        .unwrap()
    };
    let finite_difference = (dot(&chain(&plus).post_block_residuals, &target_cotangent)
        - dot(&chain(&minus).post_block_residuals, &target_cotangent))
        / (2.0 * epsilon);
    let reverse_directional = dot(&explicit_50.input_cotangent, &direction);
    let absolute_error = (finite_difference - reverse_directional).abs();
    let relative_error = absolute_error
        / finite_difference
            .abs()
            .max(reverse_directional.abs())
            .max(1e-6);
    let jr_max_abs_difference = composed_j
        .values
        .iter()
        .zip(&composed_r.values)
        .map(|(&j, &r)| (j - r).abs())
        .fold(0.0_f32, f32::max);
    eprintln!(
        "Muse Q8 chain 51->49: source50_diff={} source49_diff={} epsilon={} finite_difference={} reverse={} absolute_error={} relative_error={} jr_max_abs_difference={} block51_drift={}/{} block50_drift={}/{}",
        source_50_difference,
        source_49_difference,
        epsilon,
        finite_difference,
        reverse_directional,
        absolute_error,
        relative_error,
        jr_max_abs_difference,
        composed_j.diagnostics[0].post_attention_replay_max_abs_error,
        composed_j.diagnostics[0].post_block_replay_max_abs_error,
        composed_j.diagnostics[1].post_attention_replay_max_abs_error,
        composed_j.diagnostics[1].post_block_replay_max_abs_error,
    );
    assert!(source_50_difference < 1e-7);
    assert!(source_49_difference < 1e-7);
    assert!(composed_j.values.iter().all(|value| value.is_finite()));
    assert!(composed_r.values.iter().all(|value| value.is_finite()));
    assert!(jr_max_abs_difference > 1e-4);
    assert!(relative_error < 0.01);
    assert!(composed_j.diagnostics.iter().all(|diagnostic| {
        diagnostic.post_attention_replay_max_abs_error < 0.025
            && diagnostic.post_block_replay_max_abs_error < 0.025
    }));
    run_real_q8_query_batch_proof(&ctx, &bound, &captures);

    let bank_tokens = (0..MUSE_GLIMMER_LENS_MAX_PROMPT_TOKENS)
        .map(|slot| {
            if slot == 0 {
                weights.config().bos_token_id
            } else {
                19_873 + slot as u32
            }
        })
        .collect::<Vec<_>>();
    let mut bank_session =
        MuseGlimmerTextSession::new(&ctx, weights.config(), MUSE_GLIMMER_LENS_MAX_PROMPT_TOKENS)
            .expect("allocate Q8 bank session");
    let bank_captures = forward
        .capture_fresh_lens_prompt_blocks(&bank_tokens, &[51], &mut bank_session)
        .expect("capture T=16 bank prompt");
    run_real_q8_large_query_bank_proof(&ctx, &bound, &bank_captures);
}

fn run_real_q8_large_query_bank_proof(
    ctx: &MetalContext,
    bound: &MuseGlimmerMetalModelWeights<'_>,
    captures: &MuseGlimmerLensCaptureBank,
) {
    let capture = captures.block_capture(51).unwrap();
    let hidden = captures.hidden_size();
    let n_tokens = captures.n_tokens();
    assert_eq!(n_tokens, MUSE_GLIMMER_LENS_MAX_PROMPT_TOKENS);
    let query_elements = n_tokens * hidden;
    let mut targets = vec![0.0_f32; MUSE_GLIMMER_QUERY_BATCH_MAX * query_elements];
    for query in 0..MUSE_GLIMMER_QUERY_BATCH_MAX {
        let position = query % n_tokens;
        targets[query * query_elements + position * hidden + query] = 0.25 + query as f32 * 0.03125;
    }
    let run = |query_count| {
        muse_glimmer_one_full_attention_block_vjp_query_batch(
            ctx,
            bound,
            &capture,
            &targets[..query_count * query_elements],
            query_count,
            MuseGlimmerLensRule::J,
        )
        .expect("run T=16 Q8 bank proof")
    };
    let q4 = run(4);
    let q8 = run(8);
    let q32 = run(32);
    eprintln!(
        "Muse Q8 T=16 bank Q=4/8/32 total={:?}/{:?}/{:?} full_bank={:?}/{:?}/{:?}",
        q4.timings.total,
        q8.timings.total,
        q32.timings.total,
        q4.timings.full_attention_bank_command,
        q8.timings.full_attention_bank_command,
        q32.timings.full_attention_bank_command,
    );
    assert!(
        q8.timings.full_attention_bank_command.as_secs_f64() / 8.0
            < q4.timings.full_attention_bank_command.as_secs_f64() / 4.0,
        "qualified full-attention bank did not reduce per-query reverse time"
    );
    assert!(
        max_abs_difference(
            &q4.input_cotangents,
            &q8.input_cotangents[..q4.input_cotangents.len()]
        )
        .unwrap()
            < 2e-5
    );
    assert!(
        max_abs_difference(
            &q8.input_cotangents,
            &q32.input_cotangents[..q8.input_cotangents.len()]
        )
        .unwrap()
            < 2e-5
    );
    for query in [0, 1, MUSE_GLIMMER_QUERY_BATCH_MAX - 1] {
        let scalar = muse_glimmer_one_attention_block_vjp(
            ctx,
            bound,
            &capture,
            &targets[query * query_elements..(query + 1) * query_elements],
            MuseGlimmerLensRule::J,
        )
        .expect("run scalar T=16 bank oracle");
        assert!(
            max_abs_difference(
                &scalar.input_cotangent,
                &q32.input_cotangents[query * query_elements..(query + 1) * query_elements]
            )
            .unwrap()
                < 2e-5
        );
    }
}

fn run_real_q8_query_batch_proof(
    ctx: &MetalContext,
    bound: &MuseGlimmerMetalModelWeights<'_>,
    captures: &MuseGlimmerLensCaptureBank,
) {
    let capture_50 = captures.block_capture(50).unwrap();
    let capture_51 = captures.block_capture(51).unwrap();
    let hidden = captures.hidden_size();
    let n_tokens = captures.n_tokens();
    let query_elements = n_tokens * hidden;
    let mut targets = vec![0.0_f32; MUSE_GLIMMER_QUERY_BATCH_MAX * query_elements];
    for query in 0..MUSE_GLIMMER_QUERY_BATCH_MAX {
        let position = query % n_tokens;
        targets[query * query_elements + position * hidden + query] = 0.25 + query as f32 * 0.125;
        targets[query * query_elements + ((position + 1) % n_tokens) * hidden + 31 - query] =
            -0.5 + query as f32 * 0.03125;
    }

    let difference = |left: &[f32], right: &[f32]| {
        max_abs_difference(left, right).expect("matching comparison shapes")
    };
    let mut prefix_results = Vec::new();
    for query_count in [1, 2, 4, 8] {
        let result = muse_glimmer_one_full_attention_block_vjp_query_batch(
            &ctx,
            &bound,
            &capture_51,
            &targets[..query_count * query_elements],
            query_count,
            MuseGlimmerLensRule::J,
        )
        .expect("full-block J query batch");
        eprintln!(
            "Muse Q8 query batch Q={query_count}: total={:?} replay={:?} full_bank_reverse={:?}",
            result.timings.total, result.timings.replay, result.timings.full_attention_bank_command,
        );
        assert_eq!(result.query_count, query_count);
        assert_eq!(result.input_cotangents.len(), query_count * query_elements);
        prefix_results.push(result);
    }
    for window in prefix_results.windows(2) {
        assert!(
            difference(
                &window[0].input_cotangents,
                &window[1].input_cotangents[..window[0].input_cotangents.len()]
            ) < 2e-6,
            "query result changed with batch size"
        );
    }
    for query in 0..2 {
        let scalar = muse_glimmer_one_attention_block_vjp(
            &ctx,
            &bound,
            &capture_51,
            &targets[query * query_elements..(query + 1) * query_elements],
            MuseGlimmerLensRule::J,
        )
        .expect("scalar full-block J reference");
        assert!(
            difference(
                &scalar.input_cotangent,
                &prefix_results[3].input_cotangents
                    [query * query_elements..(query + 1) * query_elements]
            ) < 2e-6,
            "Q=8 query does not agree with isolated scalar query"
        );
    }

    for (capture, kind) in [
        (&capture_50, MuseGlimmerAttentionBlockKind::Sliding),
        (&capture_51, MuseGlimmerAttentionBlockKind::Full),
    ] {
        for rule in [MuseGlimmerLensRule::J, MuseGlimmerLensRule::R] {
            let batch = muse_glimmer_one_attention_block_vjp_query_batch(
                &ctx,
                &bound,
                capture,
                &targets[..2 * query_elements],
                2,
                rule,
            )
            .expect("J/R full/sliding query batch");
            for query in 0..2 {
                let scalar = muse_glimmer_one_attention_block_vjp(
                    &ctx,
                    &bound,
                    capture,
                    &targets[query * query_elements..(query + 1) * query_elements],
                    rule,
                )
                .expect("J/R full/sliding scalar reference");
                assert!(
                    difference(
                        &scalar.input_cotangent,
                        &batch.input_cotangents
                            [query * query_elements..(query + 1) * query_elements]
                    ) < 2e-6,
                    "{kind:?}/{rule:?} batch differs from scalar"
                );
            }
        }
    }
    assert!(
        muse_glimmer_one_full_attention_block_vjp_query_batch(
            &ctx,
            &bound,
            &capture_50,
            &targets[..query_elements],
            1,
            MuseGlimmerLensRule::J,
        )
        .is_err()
    );

    let composed = muse_glimmer_composed_vjp_query_batch(
        &ctx,
        &bound,
        &captures,
        51,
        &[49, 50],
        &targets[..2 * query_elements],
        2,
        MuseGlimmerLensRule::J,
    )
    .expect("two-source composed query batch");
    assert_eq!(composed.query_count, 2);
    assert_eq!(composed.diagnostics.len(), 2);
    assert_eq!(
        composed.diagnostics[0].kind,
        MuseGlimmerAttentionBlockKind::Full
    );
    assert_eq!(
        composed.diagnostics[1].kind,
        MuseGlimmerAttentionBlockKind::Sliding
    );
    for query in 0..2 {
        let through_51 = muse_glimmer_one_attention_block_vjp(
            &ctx,
            &bound,
            &capture_51,
            &targets[query * query_elements..(query + 1) * query_elements],
            MuseGlimmerLensRule::J,
        )
        .unwrap();
        let through_50 = muse_glimmer_one_attention_block_vjp(
            &ctx,
            &bound,
            &capture_50,
            &through_51.input_cotangent,
            MuseGlimmerLensRule::J,
        )
        .unwrap();
        assert!(
            difference(
                composed.source_query_values(1, query).unwrap(),
                &through_51.input_cotangent
            ) < 2e-6
        );
        assert!(
            difference(
                composed.source_query_values(0, query).unwrap(),
                &through_50.input_cotangent
            ) < 2e-6
        );
    }

    let row_ids = [0, 7, 31];
    let scalar_rows = muse_glimmer_fit_full_transport_rows_to_sources(
        &ctx,
        &bound,
        &captures,
        51,
        &[49, 50],
        &row_ids,
        0,
        MuseGlimmerLensRule::J,
    )
    .expect("scalar full-row oracle");
    let batched_rows = muse_glimmer_fit_full_transport_rows_to_sources_batched(
        &ctx,
        &bound,
        &captures,
        51,
        &[49, 50],
        &row_ids,
        0,
        2,
        MuseGlimmerLensRule::J,
    )
    .expect("batched full-row fit");
    assert_eq!(batched_rows.n_valid_positions, 2);
    assert_eq!(batched_rows.output_row_ids, row_ids);
    assert!(difference(&scalar_rows.values, &batched_rows.values) < 2e-6);
    assert_eq!(scalar_rows.diagnostics, batched_rows.diagnostics);
    assert!(
        muse_glimmer_fit_full_transport_rows_to_sources_batched(
            &ctx,
            &bound,
            &captures,
            51,
            &[49, 50],
            &row_ids,
            0,
            MUSE_GLIMMER_QUERY_BATCH_MAX + 1,
            MuseGlimmerLensRule::J,
        )
        .is_err()
    );
    eprintln!(
        "Muse Q8 composed Q=2 total={:?}; full rows R=3/Q=2 total={:?}",
        composed.timings.total, batched_rows.timings.total
    );
}
