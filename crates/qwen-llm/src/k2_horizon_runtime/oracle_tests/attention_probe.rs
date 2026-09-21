use super::*;
use crate::metal::encode_attn_decode_f16kv_f32;

pub(super) struct Probe {
    layer: u32,
    query: Vec<f32>,
    keys: Vec<f32>,
    values: Vec<f32>,
    output: Vec<f32>,
}

pub(super) fn decode(bytes: &[u8], base: u32, tokens: &[u32]) -> Vec<Probe> {
    assert!((1..=256).contains(&tokens.len()));
    let expected = 32 + tokens.len() * 4 + 2 * (4 + (8192 + tokens.len() * 2048) * 4);
    assert_eq!(bytes.len(), expected);
    assert_eq!(&bytes[..8], b"K2ATN001");
    let mut words = bytes[8..]
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes(b.try_into().unwrap()));
    for value in [base, tokens.len() as u32, 2, 128, 32, 8] {
        assert_eq!(words.next(), Some(value));
    }
    for &token in tokens {
        assert_eq!(words.next(), Some(token));
    }
    let mut probes = Vec::new();
    for layer in [0, 20] {
        assert_eq!(words.next(), Some(layer));
        let mut values = |count| {
            (0..count)
                .map(|_| {
                    let value = f32::from_bits(words.next().unwrap());
                    assert!(value.is_finite());
                    value
                })
                .collect()
        };
        probes.push(Probe {
            layer,
            query: values(4096),
            keys: values(tokens.len() * 1024),
            values: values(tokens.len() * 1024),
            output: values(4096),
        });
    }
    assert_eq!(words.next(), None);
    probes
}

fn attention_f64(query: &[f32], keys: &[half::f16], values: &[half::f16]) -> Vec<f32> {
    let count = keys.len() / 1024;
    assert_eq!(query.len(), 4096);
    assert_eq!(keys.len(), values.len());
    assert_eq!(keys.len(), count * 1024);
    assert!(count > 0);
    let mut output = vec![0.; 4096];
    // Match the graph's F32 scale constant, but independently sum/exp/divide in F64.
    let scale = f64::from(1.0_f32 / 128.0_f32.sqrt());
    for head in 0..32 {
        let kv_head = head / 4;
        let scores = (0..count)
            .map(|p| {
                (0..128)
                    .map(|d| {
                        f64::from(query[head * 128 + d])
                            * f64::from(keys[p * 1024 + kv_head * 128 + d].to_f32())
                    })
                    .sum::<f64>()
                    * scale
            })
            .collect::<Vec<_>>();
        let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let numerators = scores
            .iter()
            .map(|s| (s - maximum).exp())
            .collect::<Vec<_>>();
        let denominator = numerators.iter().sum::<f64>();
        for d in 0..128 {
            output[head * 128 + d] = (numerators
                .iter()
                .enumerate()
                .map(|(p, probability)| {
                    probability * f64::from(values[p * 1024 + kv_head * 128 + d].to_f32())
                })
                .sum::<f64>()
                / denominator) as f32;
        }
    }
    output
}

pub(super) fn replay(
    ctx: &MetalContext,
    probe: &Probe,
    config: &K2HorizonConfig,
) -> serde_json::Value {
    let keys = probe
        .keys
        .iter()
        .map(|&v| half::f16::from_f32(v))
        .collect::<Vec<_>>();
    let values = probe
        .values
        .iter()
        .map(|&v| half::f16::from_f32(v))
        .collect::<Vec<_>>();
    assert!(keys.iter().chain(&values).all(|v| v.is_finite()));
    let expected = attention_f64(&probe.query, &keys, &values);
    let count = keys.len() / 1024;
    let plan = K2ShortContextPlan::new(config.clone(), 0, count as u32).unwrap();
    let planes = plan.layer_planes(probe.layer).unwrap();
    let mut arena_values = vec![half::f16::NAN; plan.arena_bytes() as usize / 2];
    arena_values[planes.key.start as usize / 2..planes.key.end as usize / 2].copy_from_slice(&keys);
    arena_values[planes.value.start as usize / 2..planes.value.end as usize / 2]
        .copy_from_slice(&values);
    let sizes = [4096 * 4, plan.arena_bytes(), 4096 * 4, 4096 * 4];
    let _transaction = ctx.begin_allocation_transaction();
    let price = price_buffers(ctx, &sizes).unwrap();
    admit(ctx, price).unwrap();
    let before = ctx.current_allocated_size();
    let query = MetalTensor::from_bytes(
        ctx,
        bytemuck::cast_slice(&probe.query),
        vec![128, 32],
        GgmlType::F32,
    )
    .unwrap();
    let arena = MetalTensor::from_bytes(
        ctx,
        bytemuck::cast_slice(&arena_values),
        vec![arena_values.len() as u64],
        GgmlType::F16,
    )
    .unwrap();
    let key = arena.view_subrange(planes.key.start / 2, vec![128, 8, count as u64]);
    let value = arena.view_subrange(planes.value.start / 2, vec![128, 8, count as u64]);
    let output = MetalTensor::zeros_f32(ctx, vec![128, 32]).unwrap();
    let online = MetalTensor::zeros_f32(ctx, vec![128, 32]).unwrap();
    for tensor in [&output, &online] {
        validate_cpu_layout(
            tensor.buffer.storageMode() == MTLStorageMode::Shared,
            tensor.offset,
            4096 * 4,
            tensor.buffer.length() as u64,
        )
        .unwrap();
    }
    reconcile(ctx, before, price).unwrap();
    let command = ctx.queue.commandBuffer().unwrap();
    let encoder = KernelEncoder::begin(&command);
    let result = encode_attn_decode_f16kv_f32(
        ctx, &encoder, &query, &key, &value, &output, 32, 8, 128, count,
    )
    .and_then(|()| {
        let append = plan.append(0, 0, count as u32).unwrap();
        let token = append.token(count as u32 - 1).unwrap();
        crate::k2_horizon_metal::encode_online_attention(
            ctx,
            &encoder,
            &token,
            probe.layer,
            &arena,
            &query,
            &online,
        )
    });
    encoder.end();
    result.unwrap();
    command.commit();
    command.waitUntilCompleted();
    assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
    let actual = read_f32(&output);
    let candidate = read_f32(&online);
    json!({"layer":probe.layer,"visible_length":count,"identical_reference_inputs":true,
        "native_vs_f64":error_metrics(actual, &expected),
        "ifm_vs_f64":error_metrics(&probe.output, &expected),
        "native_vs_ifm":error_metrics(actual, &probe.output),
        "online_vs_f64":error_metrics(candidate, &expected),
        "online_vs_ifm":error_metrics(candidate, &probe.output),
        "online_vs_materialized":error_metrics(candidate, actual)})
}

pub(super) fn compare_cache(session: &K2Session<'_, '_>, probe: &Probe) -> serde_json::Value {
    let tensor = &session.buffers.cache;
    assert_eq!(tensor.dtype, GgmlType::F16);
    assert_eq!(session.committed_len() as usize * 1024, probe.keys.len());
    assert!(!session.is_poisoned());
    validate_cpu_layout(
        tensor.buffer.storageMode() == MTLStorageMode::Shared,
        tensor.offset,
        tensor.n_bytes(),
        tensor.buffer.length() as u64,
    )
    .unwrap();
    // validate_cpu_layout enforces zero offset. The synchronous append has
    // completed, and the private arena cannot change during this shared borrow.
    let cache = unsafe {
        std::slice::from_raw_parts(
            tensor.buffer.contents().as_ptr().cast::<half::f16>(),
            tensor.n_elements() as usize,
        )
    };
    let planes = session.request.layer_planes(probe.layer).unwrap();
    let mut reports = Vec::new();
    for (kind, range, reference) in [
        ("key", planes.key, &probe.keys),
        ("value", planes.value, &probe.values),
    ] {
        let start = range.start as usize / 2;
        assert!(reference.len() <= (range.end - range.start) as usize / 2);
        let actual = &cache[start..start + reference.len()];
        let expected = reference
            .iter()
            .map(|&value| half::f16::from_f32(value))
            .collect::<Vec<_>>();
        let differing = actual
            .iter()
            .zip(&expected)
            .enumerate()
            .filter_map(|(i, (a, b))| (a.to_bits() != b.to_bits()).then_some(i))
            .collect::<Vec<_>>();
        reports.push(
            json!({"kind":kind,"elements":actual.len(),"different_f16_bits":differing.len(),
            "first_differing_position":differing.first().map(|i| i/1024),
            "metrics":error_metrics(&actual.iter().map(|v|v.to_f32()).collect::<Vec<_>>(),
                &expected.iter().map(|v|v.to_f32()).collect::<Vec<_>>())}),
        );
    }
    json!({"layer":probe.layer,"visible_length":session.committed_len(),"planes":reports})
}

#[test]
fn f64_attention_preserves_gqa_mapping_and_stable_softmax() {
    let mut q = vec![0.; 4096];
    let mut k = vec![half::f16::ZERO; 2048];
    let mut v = vec![half::f16::ZERO; 2048];
    for head in 0..8 {
        for d in 0..128 {
            v[head * 128 + d] = half::f16::from_f32(head as f32);
            v[1024 + head * 128 + d] = half::f16::from_f32(head as f32 + 2.);
        }
        k[1024 + head * 128] = half::f16::ONE;
    }
    let average = attention_f64(&q, &k, &v);
    for head in 0..32 {
        assert_eq!(average[head * 128], (head / 4) as f32 + 1.);
    }
    for head in 0..32 {
        q[head * 128] = 1e6;
    }
    let selected = attention_f64(&q, &k, &v);
    for head in 0..32 {
        assert_eq!(selected[head * 128], (head / 4) as f32 + 2.);
    }
}

#[test]
fn probe_protocol_rejects_coordinates_shapes_and_nonfinite_values() {
    let mut bytes = b"K2ATN001".to_vec();
    for value in [37u32, 1, 2, 128, 32, 8, 42] {
        bytes.extend(value.to_le_bytes());
    }
    for layer in [0u32, 20] {
        bytes.extend(layer.to_le_bytes());
        bytes.resize(bytes.len() + (8192 + 2048) * 4, 0);
    }
    assert_eq!(decode(&bytes, 37, &[42]).len(), 2);
    for offset in [0, 8, 12, 16, 20, 24, 28, 32, 36] {
        let mut corrupt = bytes.clone();
        corrupt[offset] ^= 1;
        assert!(std::panic::catch_unwind(|| decode(&corrupt, 37, &[42])).is_err());
    }
    assert!(std::panic::catch_unwind(|| decode(&bytes[..bytes.len() - 1], 37, &[42])).is_err());
    bytes[40..44].copy_from_slice(&f32::NAN.to_le_bytes());
    assert!(std::panic::catch_unwind(|| decode(&bytes, 37, &[42])).is_err());
}
