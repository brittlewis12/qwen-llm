use super::*;

fn random(n: usize, seed: &mut u32) -> Vec<f32> {
    (0..n)
        .map(|_| {
            *seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            ((*seed >> 8) as f32 / 16777216.0 - 0.5) * 2.0
        })
        .collect()
}

fn check(ctx: &MetalContext, c: &Config, gate: &EngramGate, x: &[f32], kv: &[f32], rows: usize) {
    let bound = BoundGate::new(ctx, gate, c).unwrap();
    let mut expected = x.to_vec();
    for row in 0..rows {
        gate.inject(
            &mut expected[row * 4 * c.hidden..(row + 1) * 4 * c.hidden],
            &kv[row * 5 * c.hidden..(row + 1) * 5 * c.hidden],
            c.norm_eps,
        )
        .unwrap();
    }
    // Nonzero offsets and inactive capacity catch row-stride/offset overruns.
    let mut storage = vec![123.0];
    storage.extend_from_slice(x);
    storage.push(456.0);
    let backing = upload(ctx, &storage, vec![storage.len() as u64]).unwrap();
    let input = backing.view_subrange(1, vec![(x.len() + 1) as u64]);
    let mut projected_storage = vec![789.0];
    projected_storage.extend_from_slice(kv);
    let projected_backing = upload(
        ctx,
        &projected_storage,
        vec![projected_storage.len() as u64],
    )
    .unwrap();
    let projected = projected_backing.view_subrange(1, vec![kv.len() as u64]);
    let command = ctx.queue.commandBuffer().unwrap();
    let enc = KernelEncoder::begin(&command);
    bound.encode(ctx, &enc, &input, &projected, rows).unwrap();
    enc.end();
    metal::commit_and_wait(&command).unwrap();
    let actual = f32_slice(&backing).unwrap();
    assert_eq!(actual[0], 123.0);
    assert_eq!(actual[x.len() + 1], 456.0);
    for (i, (&a, &e)) in actual[1..=x.len()].iter().zip(&expected).enumerate() {
        assert!(
            (a - e).abs() <= 2e-6 * (1.0 + e.abs()),
            "hidden={} rows={rows} index={i}: GPU {a} CPU {e}",
            c.hidden
        );
    }
    assert_eq!(f32_slice(&projected_backing).unwrap(), projected_storage);
}

#[test]
fn random_both_layouts_tiny_and_real_hidden_batched() {
    let ctx = MetalContext::new().unwrap();
    let mut seed = 42;
    for hidden in [1, 7, 33, 5120] {
        let c = Config {
            hidden,
            norm_eps: 1e-6,
            ..Config::default()
        };
        let q = random(hidden * 4, &mut seed);
        let k = random(hidden * 4, &mut seed);
        let folded = EngramGate::Folded(q.iter().zip(&k).map(|(q, k)| q * k).collect());
        let separate = EngramGate::Separate { q, k };
        // 1 and 3 rows take the wide kernel, 17 the per-simdgroup batch one.
        for rows in [1, 3, 17] {
            let x = random(rows * hidden * 4, &mut seed);
            let kv = random(rows * hidden * 5, &mut seed);
            for gate in [&separate, &folded] {
                check(&ctx, &c, gate, &x, &kv, rows);
            }
        }
    }
}

#[test]
fn zeros_signed_zero_floor_and_both_rms_factors() {
    let ctx = MetalContext::new().unwrap();
    for hidden in [1, 7, 5120] {
        let c = Config {
            hidden,
            norm_eps: 1e-6,
            ..Config::default()
        };
        for folded in [false, true] {
            for scale in [0.0, -0.0, 1e-9, -1e-9, 1e-6, -1e-6, 1.0, -1.0] {
                let gate = if folded {
                    EngramGate::Folded(vec![scale; hidden * 4])
                } else {
                    EngramGate::Separate {
                        q: vec![scale; hidden * 4],
                        k: vec![1.0; hidden * 4],
                    }
                };
                let mut x = vec![0.0; hidden * 4 * 3];
                let mut kv = vec![0.0; hidden * 5 * 3];
                for row in 0..3 {
                    for stream in 0..4 {
                        let h = [0.0, -0.0, 0.125, -8.0][stream];
                        let key = [0.0, 0.25, 4.0][row];
                        x[(row * 4 + stream) * hidden..(row * 4 + stream + 1) * hidden].fill(h);
                        kv[(row * 5 + stream) * hidden..(row * 5 + stream + 1) * hidden].fill(key);
                    }
                    kv[(row * 5 + 4) * hidden..(row * 5 + 5) * hidden].fill(1.0);
                }
                check(&ctx, &c, &gate, &x, &kv, 3);
            }
        }
    }
}

#[test]
fn rejects_invalid_constants_and_activation_shapes() {
    let ctx = MetalContext::new().unwrap();
    let c = Config {
        hidden: 1,
        ..Config::default()
    };
    assert!(BoundGate::new(&ctx, &EngramGate::Folded(vec![1.0; 3]), &c).is_err());
    let gate = BoundGate::new(&ctx, &EngramGate::Folded(vec![1.0; 4]), &c).unwrap();
    let x = MetalTensor::zeros_f32(&ctx, vec![4]).unwrap();
    let kv = MetalTensor::zeros_f32(&ctx, vec![5]).unwrap();
    let half = MetalTensor::zeros_f16(&ctx, vec![5]).unwrap();
    let command = ctx.queue.commandBuffer().unwrap();
    let enc = KernelEncoder::begin(&command);
    assert!(gate.encode(&ctx, &enc, &x, &kv, 2).is_err());
    assert!(gate.encode(&ctx, &enc, &x, &kv, usize::MAX).is_err());
    assert!(gate.encode(&ctx, &enc, &x, &half, 1).is_err());
    gate.encode(&ctx, &enc, &x, &kv, 0).unwrap();
    enc.end();
}
