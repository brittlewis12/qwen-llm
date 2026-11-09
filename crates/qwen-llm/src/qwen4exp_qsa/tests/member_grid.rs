use super::*;

#[test]
fn independent_selection_attention_preserve_private_state_and_partial_groups() {
    let _lease = crate::metal::acquire_metal_test_lease().expect("GPU test campaign lease");
    let ctx = MetalContext::new().unwrap();
    let g = selected_attention_test_geometry(4_096);
    let mut members: Vec<_> = (0..4)
        .map(|_| QwenSparseAttentionMetalWorkspace::new(&ctx, g).unwrap())
        .collect();
    let controls: Vec<_> = [2_052, 2_055, 3_073, 4_095]
        .into_iter()
        .map(|n| (n - 1, n))
        .collect();
    for (i, w) in members.iter().enumerate() {
        let sample = |n: usize, salt: usize| -> Vec<f32> {
            (0..n)
                .map(|j| (((j * 13 + salt + i * 31) % 127) as f32 - 63.) / 128.)
                .collect()
        };
        write_f32_tensor(&w.query, &sample(g.query_width(), 3));
        write_f32_tensor(&w.raw_gate, &sample(g.query_width(), 7));
        write_f32_tensor(&w.index_query, &sample(4 * g.index_head_dim, 11));
        write_f16_prefix(
            &w.compressed_index_keys,
            &sample(g.block_capacity() * g.index_head_dim, 13),
        );
        write_f16_prefix(&w.key_cache, &sample(g.capacity * g.kv_width(), 17));
        write_f16_prefix(&w.value_cache, &sample(g.capacity * g.kv_width(), 19));
        write_i32_tensor(&w.visible_blocks, &[(controls[i].1 / g.ratio) as i32]);
    }
    let caches: Vec<_> = members
        .iter()
        .map(|w| {
            (
                read_tensor_bytes(&w.key_cache),
                read_tensor_bytes(&w.value_cache),
                read_tensor_bytes(&w.compressed_index_keys),
            )
        })
        .collect();
    let complete = |cmd: &objc2::rc::Retained<
        objc2::runtime::ProtocolObject<dyn objc2_metal::MTLCommandBuffer>,
    >| {
        cmd.commit();
        crate::metal::wait_completed(cmd).unwrap();
    };
    let cmd = ctx.queue.commandBuffer().unwrap();
    let enc = KernelEncoder::begin(&cmd);
    for (w, &(p, n)) in members.iter().zip(&controls) {
        encode_index_scores(&ctx, &enc, w, n / g.ratio).unwrap();
        encode_select_blocks(&ctx, &enc, w, n / g.ratio).unwrap();
        encode_expand_ids(&ctx, &enc, w, n / g.ratio, n).unwrap();
        encode_attention_step(&ctx, &enc, w, p, n).unwrap();
    }
    enc.end();
    complete(&cmd);
    let expected: Vec<_> = members
        .iter()
        .map(|w| {
            [
                read_tensor_bytes(&w.selected_blocks),
                read_tensor_bytes(&w.selected_count),
                read_tensor_bytes(&w.selector_status),
                read_tensor_bytes(&w.token_ids),
                read_tensor_bytes(&w.attention),
            ]
        })
        .collect();
    // Exercise distinct lengths, every physical argument slot, B=3's unused
    // fourth binding and the changed member order without republishing caches.
    for width in [2, 3, 4] {
        for w in &members[..width] {
            write_i32_tensor(&w.token_ids, &vec![-77; g.output_width()]);
            write_i32_tensor(&w.selected_blocks, &vec![-77; g.block_budget()]);
            write_i32_tensor(&w.selected_count, &[-77]);
            write_i32_tensor(&w.selector_status, &[-77]);
            write_f32_tensor(&w.attention, &vec![f32::NAN; g.query_width()]);
        }
        let cmd = ctx.queue.commandBuffer().unwrap();
        let enc = KernelEncoder::begin(&cmd);
        let mut refs: Vec<_> = members[..width].iter_mut().rev().collect();
        let ctrls: Vec<_> = controls[..width].iter().rev().copied().collect();
        super::super::decode_batch::selection::encode(&ctx, &enc, &refs, &ctrls).unwrap();
        super::super::decode_batch::attention::encode(&ctx, &enc, &refs, &ctrls).unwrap();
        refs.clear();
        enc.end();
        complete(&cmd);
        for (i, w) in members[..width].iter().enumerate() {
            for (field, want) in [
                &w.selected_blocks,
                &w.selected_count,
                &w.selector_status,
                &w.token_ids,
                &w.attention,
            ]
            .into_iter()
            .zip(&expected[i])
            {
                assert_eq!(&read_tensor_bytes(field), want, "B={width} member={i}");
            }
        }
    }
    for (w, frozen) in members.iter().zip(caches) {
        assert_eq!(read_tensor_bytes(&w.key_cache), frozen.0);
        assert_eq!(read_tensor_bytes(&w.value_cache), frozen.1);
        assert_eq!(read_tensor_bytes(&w.compressed_index_keys), frozen.2);
    }
}
