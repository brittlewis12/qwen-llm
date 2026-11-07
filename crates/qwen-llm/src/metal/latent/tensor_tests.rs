use super::*;
use crate::metal::test_support::{max_abs_diff_finite, offset_tensor, tensor_f32_at_offset};

struct Fixture {
    q: MetalTensor,
    cache: MetalTensor,
    ids: MetalTensor,
    counts: MetalTensor,
    visible: MetalTensor,
    sinks: MetalTensor,
    out: MetalTensor,
    reference: MetalTensor,
    scores: MetalTensor,
    partials: MetalTensor,
    stats: MetalTensor,
    g: SelectedAttentionShape,
}
impl Fixture {
    fn new(ctx: &MetalContext, rows: usize, slots: usize) -> Self {
        let capacity = slots.max(37) + 249;
        let f = |values: &[f32], shape| {
            offset_tensor(
                ctx,
                32,
                bytemuck::cast_slice(values),
                32,
                shape,
                GgmlType::F32,
            )
        };
        let ints = |values: &[i32], shape| {
            offset_tensor(
                ctx,
                20,
                bytemuck::cast_slice(values),
                20,
                shape,
                GgmlType::I32,
            )
        };
        let q: Vec<f32> = (0..rows * 64 * 512)
            .map(|i| ((i * 29 + 7) % 101) as f32 * 0.006 - 0.3)
            .collect();
        let cache: Vec<half::f16> = (0..capacity * 512)
            .map(|i| half::f16::from_f32(((i * 37 + 13) % 211) as f32 * 0.005 - 0.525))
            .collect();
        let ids: Vec<i32> = (0..rows * slots)
            .map(|i| {
                if i % 13 == 0 {
                    -1
                } else if i % 17 == 0 {
                    capacity as i32
                } else {
                    ((i / slots * 73 + i % slots * 17) % capacity) as i32
                }
            })
            .collect();
        let mut counts = vec![slots as i32; rows];
        if rows > 1 {
            counts[1] = 0;
        }
        let mut visible = vec![capacity as i32; rows];
        visible[0] = (capacity / 2) as i32;
        let sink: Vec<f32> = (0..64)
            .map(|h| {
                if h == 0 {
                    LATENT_NO_SINK
                } else if h == 1 {
                    20.0
                } else {
                    (h % 3) as f32 - 1.
                }
            })
            .collect();
        let out = f(&vec![f32::NAN; rows * 32768], vec![32768, rows as u64]);
        let reference = f(&vec![f32::NAN; rows * 32768], vec![32768, rows as u64]);
        let units = 16 * 64 * selected_attention_splits(0, slots);
        let score_shape = score_layout(slots, rows).unwrap();
        let scores = f(
            &vec![f32::NAN; score_shape.into_iter().product::<u64>() as usize],
            score_shape.to_vec(),
        );
        Self {
            q: f(&q, vec![32768, rows as u64]),
            cache: offset_tensor(
                ctx,
                8,
                bytemuck::cast_slice(&cache),
                8,
                vec![512, capacity as u64],
                GgmlType::F16,
            ),
            ids: ints(&ids, vec![slots as u64, rows as u64]),
            counts: ints(&counts, vec![rows as u64]),
            visible: ints(&visible, vec![rows as u64]),
            sinks: f(&sink, vec![64]),
            out,
            reference,
            scores,
            partials: MetalTensor::zeros_f32(ctx, vec![512, units as u64]).unwrap(),
            stats: MetalTensor::zeros_f32(ctx, vec![2, units as u64]).unwrap(),
            g: SelectedAttentionShape {
                head_count: 64,
                query_count: rows,
                query_token_offset: 0,
                token_count: rows,
                chunk_start_position: capacity,
                window: 0,
                raw_cache_is_chunk: false,
                selected_slots: slots,
                compressed_capacity: capacity,
                scale: 0.0625,
                direct: true,
            },
        }
    }
    fn buffers<'a>(&'a self, out: &'a MetalTensor) -> SelectedAttention<'a> {
        SelectedAttention {
            queries: &self.q,
            raw_cache: &self.cache,
            raw_cache_before_chunk: &self.cache,
            compressed_cache: &self.cache,
            selected_ids: &self.ids,
            selected_counts: &self.counts,
            visible_counts: &self.visible,
            sinks: &self.sinks,
            output: out,
        }
    }
    fn run(&self, ctx: &MetalContext, first: usize, count: usize, tensor: bool) {
        let ids = self.ids.view_subrange(
            (first * self.g.selected_slots) as u64,
            vec![self.g.selected_slots as u64, count as u64],
        );
        let counts = self.counts.view_subrange(first as u64, vec![count as u64]);
        let visible = self.visible.view_subrange(first as u64, vec![count as u64]);
        let b = SelectedAttention {
            selected_ids: &ids,
            selected_counts: &counts,
            visible_counts: &visible,
            ..self.buffers(if tensor { &self.out } else { &self.reference })
        };
        let g = SelectedAttentionShape {
            query_count: count,
            query_token_offset: first,
            ..self.g
        };
        let command = ctx.queue.commandBuffer().unwrap();
        let enc = KernelEncoder::begin(&command);
        if tensor {
            let scores = self
                .scores
                .view_subrange(0, score_layout(g.selected_slots, count).unwrap().to_vec());
            assert!(try_encode_selected_attention_tensor_f16(ctx, &enc, &b, g, &scores).unwrap());
        } else {
            let units = count * 64 * selected_attention_splits(0, g.selected_slots);
            let values = self.partials.view_subrange(0, vec![512, units as u64]);
            let stats = self.stats.view_subrange(0, vec![2, units as u64]);
            encode_online_selected_attention_split_f16(
                ctx,
                &enc,
                &b,
                g,
                &SelectedAttentionPartials {
                    values: &values,
                    stats: &stats,
                },
            )
            .unwrap();
        }
        enc.end();
        command.commit();
        wait_completed(&command).unwrap();
    }
}
fn context() -> Option<MetalContext> {
    let ctx = crate::test_fixtures::metal_context_or_skip()?;
    if !ctx.tensor_ops_status().is_available() {
        return None;
    }
    Some(ctx)
}
fn assert_guards(t: &MetalTensor, prefix: usize, suffix: usize) {
    let b = unsafe {
        std::slice::from_raw_parts(t.buffer.contents().as_ptr().cast::<u8>(), t.buffer.length())
    };
    assert!(b[..prefix].iter().all(|&v| v == 0xA5));
    assert!(b[b.len() - suffix..].iter().all(|&v| v == 0x5A));
}
#[test]
fn selected_tensor_attention_matches_split_and_query_offsets() {
    let Some(ctx) = context() else {
        return;
    };
    for (rows, slots) in [(3, 33), (64, 2051)] {
        let f = Fixture::new(&ctx, rows, slots);
        for first in (0..rows).step_by(16) {
            f.run(&ctx, first, 16.min(rows - first), false);
        }
        f.run(&ctx, 0, rows, true);
        let expected = tensor_f32_at_offset(&f.reference);
        let full = tensor_f32_at_offset(&f.out);
        let worst = max_abs_diff_finite("tensor vs split", &full, &expected);
        assert!(worst < 2e-4, "rows {rows} slots {slots}: {worst}");
        for first in (0..rows).step_by(2) {
            f.run(&ctx, first, 2.min(rows - first), true);
        }
        let pieces = tensor_f32_at_offset(&f.out);
        assert_eq!(
            full.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            pieces.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        for t in [&f.q, &f.out, &f.reference, &f.scores, &f.sinks] {
            assert_guards(t, 32, 32);
        }
        assert_guards(&f.cache, 8, 8);
        for t in [&f.ids, &f.counts, &f.visible] {
            assert_guards(t, 20, 20);
        }
    }
}
#[test]
fn selected_tensor_attention_matches_f64_softmax_with_sinks_and_rejections() {
    let Some(ctx) = context() else {
        return;
    };
    let f = Fixture::new(&ctx, 3, 33);
    f.run(&ctx, 0, 3, true);
    let q = tensor_f32_at_offset(&f.q);
    let sink = tensor_f32_at_offset(&f.sinks);
    let got = tensor_f32_at_offset(&f.out);
    let cache = unsafe {
        std::slice::from_raw_parts(
            f.cache
                .buffer
                .contents()
                .as_ptr()
                .cast::<u8>()
                .add(f.cache.offset as usize)
                .cast::<half::f16>(),
            f.cache.n_elements() as usize,
        )
    };
    let ints = |t: &MetalTensor| unsafe {
        std::slice::from_raw_parts(
            t.buffer
                .contents()
                .as_ptr()
                .cast::<u8>()
                .add(t.offset as usize)
                .cast::<i32>(),
            t.n_elements() as usize,
        )
    };
    let (ids, counts, visible) = (ints(&f.ids), ints(&f.counts), ints(&f.visible));
    for r in 0..3 {
        for h in 0..64 {
            let mut weights = Vec::new();
            let mut maximum = f64::from(sink[h]);
            for &id in &ids[r * 33..r * 33 + counts[r].max(0).min(33) as usize] {
                if id < 0 || id >= visible[r] || id as usize >= f.g.compressed_capacity {
                    continue;
                }
                let key = &cache[id as usize * 512..][..512];
                let query = &q[(r * 64 + h) * 512..][..512];
                let score = key
                    .iter()
                    .zip(query)
                    .map(|(k, q)| k.to_f64() * f64::from(*q))
                    .sum::<f64>()
                    * f64::from(f.g.scale);
                maximum = maximum.max(score);
                weights.push((id as usize, score));
            }
            let denominator = (f64::from(sink[h]) - maximum).exp()
                + weights
                    .iter()
                    .map(|(_, s)| (s - maximum).exp())
                    .sum::<f64>();
            for d in 0..512 {
                let want = weights
                    .iter()
                    .map(|(id, s)| (s - maximum).exp() * cache[id * 512 + d].to_f64())
                    .sum::<f64>()
                    / denominator;
                let v = got[(r * 64 + h) * 512 + d];
                assert!(v.is_finite());
                assert!(
                    (f64::from(v) - want).abs() < 2e-4,
                    "r{r} h{h} d{d}: {v} vs {want}"
                );
            }
        }
    }
}
#[test]
fn selected_tensor_attention_masks_logits_equal_to_no_sink_and_rejects_bad_bindings() {
    let Some(ctx) = context() else {
        return;
    };
    let f = Fixture::new(&ctx, 1, 33);
    unsafe {
        std::slice::from_raw_parts_mut(
            f.q.buffer
                .contents()
                .as_ptr()
                .cast::<u8>()
                .add(f.q.offset as usize)
                .cast::<f32>(),
            32768,
        )
        .fill(-0.25);
        std::slice::from_raw_parts_mut(
            f.cache
                .buffer
                .contents()
                .as_ptr()
                .cast::<u8>()
                .add(f.cache.offset as usize)
                .cast::<half::f16>(),
            f.cache.n_elements() as usize,
        )
        .fill(half::f16::from_f32(0.5));
        let ids = std::slice::from_raw_parts_mut(
            f.ids
                .buffer
                .contents()
                .as_ptr()
                .cast::<u8>()
                .add(f.ids.offset as usize)
                .cast::<i32>(),
            33,
        );
        ids.fill(-1);
        ids[0] = 0;
        std::slice::from_raw_parts_mut(
            f.sinks
                .buffer
                .contents()
                .as_ptr()
                .cast::<u8>()
                .add(f.sinks.offset as usize)
                .cast::<f32>(),
            64,
        )
        .fill(LATENT_NO_SINK);
    }
    let b = f.buffers(&f.out);
    let g = SelectedAttentionShape {
        scale: 1e30 / 64.0,
        ..f.g
    };
    let command = ctx.queue.commandBuffer().unwrap();
    let enc = KernelEncoder::begin(&command);
    dispatch_census_begin();
    let bad = |scores: &MetalTensor, g| {
        try_encode_selected_attention_tensor_f16(&ctx, &enc, &b, g, scores)
    };
    let mut readonly = f.scores.clone();
    readonly.provenance = MetalTensorProvenance::RetainedGgufReadOnly;
    assert!(bad(&readonly, g).is_err());
    let mut past = f.scores.clone();
    past.offset = past.buffer.length() as u64 - 4;
    assert!(bad(&past, g).is_err());
    let mut misaligned = f.scores.clone();
    misaligned.offset = 4;
    assert!(bad(&misaligned, g).is_err());
    let mut aliased = f.out.clone();
    aliased.shape = f.scores.shape.clone();
    assert!(bad(&aliased, g).is_err());
    let mut wrong = f.scores.clone();
    wrong.dtype = GgmlType::F16;
    assert!(bad(&wrong, g).is_err());
    assert!(
        bad(
            &f.scores,
            SelectedAttentionShape {
                scale: f32::NAN,
                ..g
            }
        )
        .is_err()
    );
    assert!(
        bad(
            &f.scores,
            SelectedAttentionShape {
                query_count: 0,
                ..g
            }
        )
        .is_err()
    );
    assert!(
        bad(
            &f.scores,
            SelectedAttentionShape {
                query_token_offset: 1,
                ..g
            }
        )
        .is_err()
    );
    for unsupported in [
        SelectedAttentionShape {
            head_count: 32,
            ..g
        },
        SelectedAttentionShape { window: 1, ..g },
        SelectedAttentionShape { direct: false, ..g },
    ] {
        assert!(!bad(&f.scores, unsupported).unwrap());
    }
    assert!(dispatch_census_take().is_empty());
    assert!(bad(&f.scores, g).unwrap());
    enc.end();
    command.commit();
    wait_completed(&command).unwrap();
    let actual = tensor_f32_at_offset(&f.out);
    assert!(
        actual
            .iter()
            .all(|v| v.is_finite() && (*v - 0.25).abs() < 1e-6)
    );
}
