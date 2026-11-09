use super::*;

#[test]
fn parallel_pool_geometry_retains_unqualified_shapes() {
    assert!(parallel_pool_shape(4, 128));
    for (ratio, width) in [(0, 128), (2, 128), (8, 128), (4, 64), (4, 256)] {
        assert!(!parallel_pool_shape(ratio, width));
    }
}

#[test]
fn parallel_pool_matches_scalar_with_shifted_buffers() {
    let _lease = crate::metal::acquire_metal_test_lease().unwrap();
    let ctx = MetalContext::new().unwrap();
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        ratio: u32,
        head_dim: u32,
        rotary_dim: u32,
        block: u32,
        theta: f32,
        eps: f32,
    }
    for mode in 0..4 {
        let pending_storage = MetalTensor::zeros_f32(&ctx, vec![520]).unwrap();
        write_f32_tensor(&pending_storage, &vec![-777.; 520]);
        let pending = pending_storage.view_subrange(4, vec![512]);
        let values: Vec<f32> = (0..512)
            .map(|i| match mode {
                0 => {
                    if i % 2 == 0 {
                        0.0
                    } else {
                        -0.0
                    }
                }
                1 => ((i * 37 % 251) as f32 - 125.) / 64.,
                2 => ((i * 19 % 127) as f32 - 63.) * 64.125,
                _ => (i % 19) as f32 * 0.125 + (i % 3 + 1) as f32 / 2048.,
            })
            .collect();
        write_f32_tensor(&pending, &values);
        let weight_storage = MetalTensor::zeros_f32(&ctx, vec![136]).unwrap();
        write_f32_tensor(&weight_storage, &vec![-777.; 136]);
        let weight = weight_storage.view_subrange(4, vec![128]);
        write_f32_tensor(
            &weight,
            &(0..128)
                .map(|i| 0.5 + (i % 11) as f32 / 8.)
                .collect::<Vec<_>>(),
        );
        let frozen_input = read_tensor_bytes(&pending_storage);
        let frozen_weight = read_tensor_bytes(&weight_storage);
        for rotary_dim in [32, 64, 128] {
            for block in [0u32, 1, 19, 2048, 8192] {
                let count = (block as usize + 2) * 128;
                let old_store = MetalTensor::zeros_f16(&ctx, vec![(count + 16) as u64]).unwrap();
                let new_store = MetalTensor::zeros_f16(&ctx, vec![(count + 16) as u64]).unwrap();
                for storage in [&old_store, &new_store] {
                    write_f16_prefix(storage, &vec![-777.; count + 16]);
                }
                let old = old_store.view_subrange(8, vec![count as u64]);
                let new = new_store.view_subrange(8, vec![count as u64]);
                let cmd = ctx.queue.commandBuffer().unwrap();
                let enc = KernelEncoder::begin(&cmd);
                for (name, destination, threads) in [
                    ("kernel_qwen4exp_qsa_pool_publish_f16", &old, 1),
                    ("kernel_qwen4exp_qsa_pool_publish_parallel_f16", &new, 128),
                ] {
                    let pso = ctx.pipeline(name).unwrap();
                    enc.set_pipeline(&pso);
                    enc.set_bytes(
                        0,
                        &Args {
                            ratio: 4,
                            head_dim: 128,
                            rotary_dim,
                            block,
                            theta: 1_000_000.,
                            eps: 1e-6,
                        },
                    );
                    enc.set_tensor(1, &pending);
                    enc.set_tensor(2, &weight);
                    enc.set_tensor(3, destination);
                    enc.dispatch(
                        MTLSize {
                            width: 1,
                            height: 1,
                            depth: 1,
                        },
                        MTLSize {
                            width: threads,
                            height: 1,
                            depth: 1,
                        },
                    );
                }
                enc.end();
                cmd.commit();
                crate::metal::wait_completed(&cmd).unwrap();
                let expected = read_tensor_bytes(&old_store);
                assert_eq!(
                    expected,
                    read_tensor_bytes(&new_store),
                    "mode={mode} rotary={rotary_dim} block={block}"
                );
                let begin = (8 + block as usize * 128) * 2;
                let end = begin + 256;
                let guard = f16::from_f32(-777.).to_bits().to_ne_bytes();
                assert!(expected[..begin].chunks_exact(2).all(|x| x == guard));
                assert!(expected[end..].chunks_exact(2).all(|x| x == guard));
            }
        }
        assert_eq!(frozen_input, read_tensor_bytes(&pending_storage));
        assert_eq!(frozen_weight, read_tensor_bytes(&weight_storage));
    }
}
