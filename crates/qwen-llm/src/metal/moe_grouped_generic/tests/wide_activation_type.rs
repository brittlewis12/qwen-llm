use super::*;

const K: usize = 32;
const M: usize = 64;
const EXPERTS: usize = 2;
const TOKENS: usize = 4096;
const ACTIVE: usize = 32;
const GUARD: usize = 16;
const CANARY: f32 = -7.25;
const CLAMP: f32 = 0.5;

fn context() -> MetalContext {
    assert_eq!(std::env::var("MTL_DEBUG_LAYER").as_deref(), Ok("1"));
    for name in ["QWEN_EXPERT_TENSOR_TILE", "QWEN_V41_EXPERT_WIDE"] {
        assert!(std::env::var_os(name).is_none(), "requires default {name}");
    }
    let ctx = MetalContext::new().expect("Metal context");
    assert!(
        moe_grouped_tensor_selected(&ctx),
        "M5 tensor coverage unavailable, not a skip/pass: {:?}",
        ctx.tensor_ops_status()
    );
    eprintln!("wide-type-fix device={}", ctx.describe());
    ctx
}

fn slot(i: usize) -> usize {
    if i + 1 == ACTIVE { TOKENS - 1 } else { i * 131 }
}

fn input(token: usize, lane: usize) -> f32 {
    0.125 + ((token * 7 + lane) % 16) as f32 / 32.0
}

fn coefficient(up: bool, row: usize) -> f32 {
    if up {
        (row % 3 + 1) as f32 / 128.0
    } else {
        (row % 4 + 1) as f32 / 64.0
    }
}

struct Fixture {
    gate: MetalTensor,
    up: MetalTensor,
    x: MetalTensor,
    counts: MetalTensor,
    ids: MetalTensor,
    out: MetalTensor,
}

impl Fixture {
    fn new(ctx: &MetalContext, activation: GgmlType) -> Self {
        let activation_bytes = match activation {
            GgmlType::F16 => 2,
            GgmlType::F32 => 4,
            _ => panic!("fixture activation"),
        };
        let bytes = [
            K * M * EXPERTS * 2,
            K * M * EXPERTS * 2,
            TOKENS * K * activation_bytes + 2 * GUARD,
            EXPERTS * 4 + 2 * GUARD,
            EXPERTS * TOKENS * 4 + 2 * GUARD,
            TOKENS * M * 4 + 2 * GUARD,
        ];
        let priced = bytes
            .iter()
            .map(|&n| {
                ctx.price_shared_buffer_upper(n as u64)
                    .unwrap()
                    .priced_upper_bytes
            })
            .sum::<u64>();
        // Host payload, offset_tensor staging/reallocation, and before/after
        // readbacks fit four copies of the complete explicit buffer inventory.
        let cpu_bytes = 4 * bytes.iter().sum::<usize>() as u64;
        assert!(priced + cpu_bytes <= 128 << 20);
        let _transaction = ctx.begin_allocation_transaction();
        let admission = evaluate_metal_memory_admission_with_cpu_bytes(
            priced,
            cpu_bytes,
            0,
            ctx.memory_signals(),
            true,
        );
        assert!(
            admission.is_admitted(),
            "tiny fixture admission: {admission:?}"
        );
        eprintln!(
            "wide-type-fix fixture activation={activation:?} priced_metal_bytes={priced} cpu_bytes={cpu_bytes} explicit_limit_bytes={}",
            128 << 20
        );

        let bank = |up| {
            let values: Vec<half::f16> = (0..EXPERTS * M * K)
                .map(|i| half::f16::from_f32(coefficient(up, (i / K) % M)))
                .collect();
            bank_tensor(
                ctx,
                GgmlType::F16,
                bytemuck::cast_slice(&values),
                K,
                M,
                EXPERTS,
            )
        };
        let x: Vec<f32> = (0..TOKENS * K).map(|i| input(i / K, i % K)).collect();
        let x = if activation == GgmlType::F16 {
            let half: Vec<half::f16> = x.iter().map(|&v| half::f16::from_f32(v)).collect();
            offset_tensor(
                ctx,
                GUARD,
                bytemuck::cast_slice(&half),
                GUARD,
                vec![(TOKENS * K) as u64],
                activation,
            )
        } else {
            offset_tensor(
                ctx,
                GUARD,
                bytemuck::cast_slice(&x),
                GUARD,
                vec![(TOKENS * K) as u64],
                activation,
            )
        };
        let counts = offset_tensor(
            ctx,
            GUARD,
            bytemuck::cast_slice(&[ACTIVE as i32, 0]),
            GUARD,
            vec![EXPERTS as u64],
            GgmlType::I32,
        );
        let mut ids = vec![-1i32; EXPERTS * TOKENS];
        for (i, id) in ids[..ACTIVE].iter_mut().enumerate() {
            *id = slot(i) as i32;
        }
        let ids = offset_tensor(
            ctx,
            GUARD,
            bytemuck::cast_slice(&ids),
            GUARD,
            vec![(EXPERTS * TOKENS) as u64],
            GgmlType::I32,
        );
        let out = offset_tensor(
            ctx,
            GUARD,
            bytemuck::cast_slice(&vec![CANARY; TOKENS * M]),
            GUARD,
            vec![(TOKENS * M) as u64],
            GgmlType::F32,
        );
        Self {
            gate: bank(false),
            up: bank(true),
            x,
            counts,
            ids,
            out,
        }
    }

    fn public(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        x: &MetalTensor,
    ) -> Result<(), MetalError> {
        encode_moe_swiglu_f32_grouped_slots_tensor(
            ctx,
            enc,
            &self.gate,
            &self.up,
            x,
            &self.counts,
            &self.ids,
            &self.out,
            K,
            M,
            EXPERTS,
            1,
            TOKENS,
            Some(CLAMP),
        )
    }
}

fn numeric_contract(activation: GgmlType) {
    let _lease = acquire_metal_benchmark_lease().expect("wide activation type production lease");
    let ctx = context();
    let f = Fixture::new(&ctx, activation);
    let counts_before = tensor_backing_bytes(&f.counts);
    let ids_before = tensor_backing_bytes(&f.ids);
    dispatch_census_begin();
    let command = ctx.queue.commandBuffer().unwrap();
    let enc = KernelEncoder::begin(&command);
    f.public(&ctx, &enc, &f.x).unwrap();
    enc.end();
    let census = dispatch_census_take();
    commit_and_wait(&command).unwrap();
    assert_eq!(tensor_backing_bytes(&f.counts), counts_before);
    assert_eq!(tensor_backing_bytes(&f.ids), ids_before);
    for tensor in [&f.x, &f.counts, &f.ids, &f.out] {
        assert_offset_guards(tensor, GUARD, GUARD);
    }
    let actual = tensor_f32_at_offset(&f.out);
    let mut expected = vec![CANARY; TOKENS * M];
    for i in 0..ACTIVE {
        let token = slot(i);
        // Both activation and bank values are exactly representable in half;
        // this is the original half-operand, F32-accumulation contract.
        let sum: f32 = (0..K).map(|lane| input(token, lane)).sum();
        for row in 0..M {
            let gate = (sum * coefficient(false, row)).min(CLAMP);
            let up = (sum * coefficient(true, row)).clamp(-CLAMP, CLAMP);
            expected[token * M + row] = gate / (1.0 + (-gate).exp()) * up;
        }
    }
    assert!(expected.iter().any(|&v| v != CANARY && v.abs() > 0.01));
    let mut max_error = 0.0f32;
    for (&a, &e) in actual.iter().zip(&expected) {
        assert!(a.is_finite());
        if e == CANARY {
            assert_eq!(a.to_bits(), CANARY.to_bits(), "inactive output changed");
        } else {
            max_error = max_error.max((a - e).abs());
        }
    }
    let kernels: Vec<_> = census.iter().map(|r| r.kernel.as_str()).collect();
    eprintln!(
        "wide-type-fix activation={activation:?} active={ACTIVE} max_abs_error={max_error} kernels={kernels:?}"
    );
    assert!(
        max_error <= 2e-5,
        "{activation:?} half-operand numeric contract: error={max_error}"
    );
    let expected_kernels = match activation {
        GgmlType::F32 => vec![
            "kernel_moe_down_f16_f32_grouped_slots_tensor",
            "kernel_moe_up_clamped_f16_f32_grouped_slots_tensor",
        ],
        GgmlType::F16 => vec!["kernel_moe_wide_probe_f16_f16act"; 2],
        _ => unreachable!(),
    };
    assert_eq!(kernels, expected_kernels);
    for row in census {
        assert_eq!(
            row.grid_width,
            (TOKENS / if activation == GgmlType::F16 { 64 } else { 32 }) as u64
        );
        assert_eq!(
            (row.grid_height, row.grid_depth, row.tg_threads),
            (1, EXPERTS as u64, 128)
        );
        assert!(!row.encoder_concurrent);
    }
}

#[test]
#[ignore = "self-leased M5 tensor dtype regression; MTL_DEBUG_LAYER=1; <=128 MiB complete fixture; no model"]
fn wide_activation_type_f32_4096_preserves_narrow_numeric_contract() {
    numeric_contract(GgmlType::F32);
}

#[test]
#[ignore = "self-leased M5 tensor dtype regression; MTL_DEBUG_LAYER=1; <=128 MiB complete fixture; no model"]
fn wide_activation_type_f16_4096_keeps_wide_numeric_contract() {
    numeric_contract(GgmlType::F16);
}

#[test]
#[ignore = "self-leased M5 tensor dtype refusal; MTL_DEBUG_LAYER=1; <=128 MiB complete fixture; no model"]
fn wide_activation_type_invalid_and_direct_probes_refuse_without_dispatch() {
    let _lease = acquire_metal_benchmark_lease().expect("wide activation type production lease");
    let ctx = context();
    let mut failures = Vec::new();
    for (role, dtype) in [
        ("public", GgmlType::I32),
        ("up", GgmlType::F32),
        ("up", GgmlType::I32),
        ("down", GgmlType::F32),
        ("down", GgmlType::I32),
    ] {
        let f = Fixture::new(&ctx, GgmlType::F32);
        let mut x = f.x.clone();
        x.dtype = dtype;
        let before: Vec<_> = [&f.counts, &f.ids, &f.out].map(tensor_backing_bytes).into();
        dispatch_census_begin();
        let command = ctx.queue.commandBuffer().unwrap();
        let enc = KernelEncoder::begin(&command);
        let result = match role {
            "public" => f.public(&ctx, &enc, &x),
            "up" => encode_moe_swiglu_wide_probe(
                &ctx, &enc, &f.gate, &f.up, &x, &f.counts, &f.ids, &f.out, K, M, EXPERTS, 1,
                TOKENS, CLAMP,
            ),
            "down" => encode_moe_down_wide_probe(
                &ctx, &enc, &f.gate, &x, &f.counts, &f.ids, &f.out, K, M, EXPERTS, TOKENS,
            ),
            _ => unreachable!(),
        };
        enc.end();
        let census = dispatch_census_take();
        commit_and_wait(&command).unwrap();
        let unchanged = [&f.counts, &f.ids, &f.out]
            .iter()
            .zip(&before)
            .all(|(tensor, bytes)| tensor_backing_bytes(tensor) == *bytes);
        let refused =
            matches!(&result, Err(MetalError::BadShape { detail, .. }) if detail.contains("F16"));
        eprintln!(
            "wide-type-fix refusal role={role} dtype={dtype:?} result={result:?} dispatches={} counts_ids_output_unchanged={unchanged}",
            census.len()
        );
        if !refused || !census.is_empty() || !unchanged {
            failures.push(format!(
                "{role}/{dtype:?}: refused={refused}, dispatches={}, unchanged={unchanged}",
                census.len()
            ));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("; "));
}
