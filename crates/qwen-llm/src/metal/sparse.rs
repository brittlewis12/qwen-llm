//! Sparse-attention indexer primitives shared by DeepSeek-V4 and GLM-5.3:
//! lightning-indexer scores over F16 keys and exact top-k selection.
//!
//! Scores: for each query and visible key row,
//! `score = sum_h max(q_h . k, 0) * w_h` (heads in order), with F16 queries
//! and keys multiplied by simdgroup half matrices into F32, and F32 head
//! weights (callers fold any scale into them). Rows at or beyond a query's
//! visible count score `-inf`; there is no other mask.
//!
//! Selection: an exact radix select of each query's `top_k` largest visible
//! scores, emitted as ascending row ids padded with -1, with ties at the
//! threshold broken by lowest row id. Signed zeros and subnormals compare
//! equal. A non-finite visible score sets the query's status to 2 and the
//! kernel falls back to rows `0..count`: callers must check status before
//! trusting the selection.

use super::checks::{bad_shape, check_disjoint, check_tensor, require_serial, to_u32};
use super::*;

/// Selector status for a well-formed selection.
pub const SELECT_STATUS_OK: i32 = 0;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ScoreArgs {
    head_count: u32,
    head_dim: u32,
    row_capacity: u32,
    query_count: u32,
}

/// Buffers for [`encode_lightning_scores_f16_matrix`] over `query_count`
/// queries and `row_capacity` key rows.
pub struct LightningScores<'a> {
    /// F16 `[head_dim, heads, queries]`.
    pub queries: &'a MetalTensor,
    /// F32 `[heads, queries]`.
    pub head_weights: &'a MetalTensor,
    /// F16 `[head_dim, row_capacity]`.
    pub keys: &'a MetalTensor,
    /// I32 `[queries]`: visible key rows per query.
    pub visible_counts: &'a MetalTensor,
    /// F32 `[row_capacity, queries]`; only rows below `max_dispatched_rows`
    /// are written.
    pub scores: &'a MetalTensor,
}

/// Lightning-indexer scores with half-matrix dot products, for 32 or 64
/// heads of width 128, over the first `max_dispatched_rows` key rows (which
/// must cover every query's visible count).
pub fn encode_lightning_scores_f16_matrix(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    b: &LightningScores<'_>,
    head_count: usize,
    head_dim: usize,
    row_capacity: usize,
    max_dispatched_rows: usize,
    query_count: usize,
) -> Result<(), MetalError> {
    const K: &str = "lightning_scores_f16_matrix";
    require_serial(K, enc)?;
    let kernel = match (head_count, head_dim) {
        (64, 128) => "kernel_deepseek_v4_lightning_indexer_scores_f16_matrix_ceiling",
        (32, 128) => "kernel_lightning_indexer_scores_f16_matrix_h32",
        _ => {
            return Err(bad_shape(
                K,
                format!("needs 32 or 64 heads of width 128, got {head_count}x{head_dim}"),
            ));
        }
    };
    if row_capacity == 0
        || query_count == 0
        || max_dispatched_rows == 0
        || max_dispatched_rows > row_capacity
    {
        return Err(bad_shape(
            K,
            format!(
                "rows {max_dispatched_rows} of capacity {row_capacity} for {query_count} queries"
            ),
        ));
    }
    // Shader offsets are 32-bit.
    let overflow = || bad_shape(K, "element counts exceed 32-bit shader offsets");
    for elements in [
        row_capacity.checked_mul(query_count),
        head_count.checked_mul(query_count),
        (head_count * head_dim).checked_mul(query_count),
        row_capacity.checked_mul(head_dim),
    ] {
        let elements = elements.ok_or_else(overflow)?;
        if u32::try_from(elements).is_err() {
            return Err(overflow());
        }
    }
    let (h, d, r, q) = (
        head_count as u64,
        head_dim as u64,
        row_capacity as u64,
        query_count as u64,
    );
    check_tensor(K, b.queries, GgmlType::F16, &[d, h, q], false, "queries")?;
    check_tensor(
        K,
        b.head_weights,
        GgmlType::F32,
        &[h, q],
        false,
        "head weights",
    )?;
    check_tensor(K, b.keys, GgmlType::F16, &[d, r], false, "keys")?;
    check_tensor(
        K,
        b.visible_counts,
        GgmlType::I32,
        &[q],
        false,
        "visible counts",
    )?;
    check_tensor(K, b.scores, GgmlType::F32, &[r, q], true, "scores")?;
    check_disjoint(
        K,
        b.scores,
        &[
            (b.queries, "queries"),
            (b.head_weights, "head weights"),
            (b.keys, "keys"),
            (b.visible_counts, "visible counts"),
        ],
    )?;
    let threads = head_count / 8 * 32;
    let threadgroup_bytes = 8 * 128 * 2 + 8 * head_count * 4;
    let pso = ctx.pipeline(kernel)?;
    if pso.threadExecutionWidth() != 32
        || pso.maxTotalThreadsPerThreadgroup() < threads
        || ctx.device.maxThreadgroupMemoryLength() < threadgroup_bytes
    {
        return Err(bad_shape(
            K,
            format!("needs {threads} 32-lane threads and {threadgroup_bytes} threadgroup bytes"),
        ));
    }
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &ScoreArgs {
            head_count: to_u32(K, head_count, "heads")?,
            head_dim: to_u32(K, head_dim, "head dim")?,
            row_capacity: to_u32(K, row_capacity, "row capacity")?,
            query_count: to_u32(K, query_count, "queries")?,
        },
    );
    enc.set_tensor(1, b.queries);
    enc.set_tensor(2, b.head_weights);
    enc.set_tensor(3, b.keys);
    enc.set_tensor(4, b.visible_counts);
    enc.set_tensor(5, b.scores);
    enc.set_threadgroup_memory(0, 8 * 128 * 2);
    enc.set_threadgroup_memory(1, 8 * head_count * 4);
    enc.dispatch(
        MTLSize {
            width: max_dispatched_rows.div_ceil(8),
            height: query_count,
            depth: 1,
        },
        MTLSize {
            width: threads,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

/// Buffers for [`encode_select_top_k_ids`].
pub struct TopKSelection<'a> {
    /// F32 `[row_capacity, queries]`.
    pub scores: &'a MetalTensor,
    /// I32 `[queries]`.
    pub visible_counts: &'a MetalTensor,
    /// I32 `[top_k, queries]`: ascending selected row ids, -1 padded.
    pub ids: &'a MetalTensor,
    /// I32 `[queries]`: `min(visible, top_k)`.
    pub counts: &'a MetalTensor,
    /// I32 `[queries]`: [`SELECT_STATUS_OK`] or a failure code.
    pub status: &'a MetalTensor,
}

/// Exact top-`top_k` selection (radix select, 4-bit digits, one 256-thread
/// threadgroup per query). `max_visible_rows` bounds every query's visible
/// count and must exceed `top_k` (selection prunes something).
pub fn encode_select_top_k_ids(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    b: &TopKSelection<'_>,
    row_capacity: usize,
    max_visible_rows: usize,
    top_k: usize,
    query_count: usize,
) -> Result<(), MetalError> {
    const K: &str = "select_top_k_ids";
    const THREADS: usize = 256;
    require_serial(K, enc)?;
    if row_capacity == 0
        || top_k == 0
        || query_count == 0
        || max_visible_rows <= top_k
        || max_visible_rows > row_capacity
    {
        return Err(bad_shape(
            K,
            format!(
                "top {top_k} of at most {max_visible_rows} visible rows (capacity {row_capacity}) for {query_count} queries"
            ),
        ));
    }
    let overflow = || bad_shape(K, "element counts exceed 32-bit shader offsets");
    for elements in [
        row_capacity.checked_mul(query_count),
        top_k.checked_mul(query_count),
    ] {
        if u32::try_from(elements.ok_or_else(overflow)?).is_err() {
            return Err(overflow());
        }
    }
    let (r, k, q) = (row_capacity as u64, top_k as u64, query_count as u64);
    check_tensor(K, b.scores, GgmlType::F32, &[r, q], false, "scores")?;
    check_tensor(
        K,
        b.visible_counts,
        GgmlType::I32,
        &[q],
        false,
        "visible counts",
    )?;
    check_tensor(K, b.ids, GgmlType::I32, &[k, q], true, "ids")?;
    check_tensor(K, b.counts, GgmlType::I32, &[q], true, "counts")?;
    check_tensor(K, b.status, GgmlType::I32, &[q], true, "status")?;
    let inputs = [(b.scores, "scores"), (b.visible_counts, "visible counts")];
    for (output, name) in [(b.ids, "ids"), (b.counts, "counts"), (b.status, "status")] {
        check_disjoint(K, output, &inputs)
            .map_err(|_| bad_shape(K, format!("{name} overlaps an input")))?;
    }
    check_disjoint(K, b.ids, &[(b.counts, "counts"), (b.status, "status")])?;
    check_disjoint(K, b.counts, &[(b.status, "status")])?;
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        row_capacity: u32,
        top_k: u32,
        query_count: u32,
        emit_ranked: u32,
    }
    let pso = ctx.pipeline("kernel_deepseek_v4_select_top_k_radix4_ids_f32")?;
    if pso.threadExecutionWidth() != 32 || pso.maxTotalThreadsPerThreadgroup() < THREADS {
        return Err(bad_shape(K, "needs 256 threads of 32-lane simdgroups"));
    }
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &Args {
            row_capacity: to_u32(K, row_capacity, "row capacity")?,
            top_k: to_u32(K, top_k, "top k")?,
            query_count: to_u32(K, query_count, "queries")?,
            emit_ranked: 0,
        },
    );
    enc.set_tensor(1, b.scores);
    enc.set_tensor(2, b.visible_counts);
    enc.set_tensor(3, b.ids);
    enc.set_tensor(4, b.counts);
    enc.set_tensor(5, b.status);
    enc.set_threadgroup_memory(0, THREADS * 4);
    enc.set_threadgroup_memory(1, THREADS * 4);
    enc.dispatch(
        MTLSize {
            width: query_count,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: THREADS,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::test_support::{offset_tensor, tensor_backing_bytes};
    use super::*;
    use half::f16;

    fn tensor<T: bytemuck::Pod>(
        ctx: &MetalContext,
        values: &[T],
        shape: Vec<u64>,
        dtype: GgmlType,
    ) -> MetalTensor {
        offset_tensor(ctx, 16, bytemuck::cast_slice(values), 16, shape, dtype)
    }

    fn words(t: &MetalTensor) -> Vec<[u8; 4]> {
        let bytes = tensor_backing_bytes(t);
        let start = t.offset as usize;
        bytes[start..start + 4 * t.n_elements() as usize]
            .as_chunks::<4>()
            .0
            .to_vec()
    }

    fn read_f32(t: &MetalTensor) -> Vec<f32> {
        words(t).into_iter().map(f32::from_le_bytes).collect()
    }

    fn read_i32(t: &MetalTensor) -> Vec<i32> {
        words(t).into_iter().map(i32::from_le_bytes).collect()
    }

    fn run(ctx: &MetalContext, encode: impl FnOnce(&KernelEncoder)) {
        let command = ctx.queue.commandBuffer().expect("command buffer");
        let encoder = KernelEncoder::begin(&command);
        encode(&encoder);
        encoder.end();
        command.commit();
        wait_completed(&command).expect("command buffer failed");
    }

    fn noise(seed: u32, n: usize, scale: f32) -> Vec<f32> {
        let mut state = seed.wrapping_mul(0x9e37_79b9) | 1;
        (0..n)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                ((state >> 8) as f32 / 16_777_216.0 - 0.5) * 2.0 * scale
            })
            .collect()
    }

    const H: usize = 32;
    const D: usize = 128;

    /// Scores for one query: F16 operands, F32 sum in dimension order, ReLU,
    /// F32 head weights summed in head order; `-inf` past `visible`.
    fn cpu_scores(q: &[f16], w: &[f32], keys: &[f16], visible: usize, rows: usize) -> Vec<f32> {
        (0..rows)
            .map(|r| {
                if r >= visible {
                    return f32::NEG_INFINITY;
                }
                (0..H)
                    .map(|h| {
                        let dot: f32 = (0..D)
                            .map(|d| q[h * D + d].to_f32() * keys[r * D + d].to_f32())
                            .sum();
                        dot.max(0.0) * w[h]
                    })
                    .sum()
            })
            .collect()
    }

    /// 32-head half-matrix scores equal the CPU contract (F16 queries and
    /// keys, F32 accumulation, signed head weights, -inf past visibility),
    /// and each query's scores are bitwise independent of its batch (1, 7,
    /// 8 and 9 queries; neighbors with different values and visibility).
    #[test]
    fn h32_scores_match_cpu_contract_and_are_batch_independent() {
        let Some(ctx) = crate::test_fixtures::metal_context_or_skip() else {
            return;
        };
        const ROWS: usize = 1_031;
        let keys: Vec<f16> = noise(1, ROWS * D, 1.0)
            .into_iter()
            .map(f16::from_f32)
            .collect();
        let keys_t = tensor(&ctx, &keys, vec![D as u64, ROWS as u64], GgmlType::F16);
        let visible_all = [1_031i32, 600, 513, 1, 0, 1_030, 777, 8, 1_024];
        let queries: Vec<f16> = noise(2, 9 * H * D, 0.5)
            .into_iter()
            .map(f16::from_f32)
            .collect();
        let weights = noise(3, 9 * H, 0.03);
        let score = |q0: usize, count: usize| -> Vec<f32> {
            let q = tensor(
                &ctx,
                &queries[q0 * H * D..(q0 + count) * H * D],
                vec![D as u64, H as u64, count as u64],
                GgmlType::F16,
            );
            let w = tensor(
                &ctx,
                &weights[q0 * H..(q0 + count) * H],
                vec![H as u64, count as u64],
                GgmlType::F32,
            );
            let visible = &visible_all[q0..q0 + count];
            let v = tensor(&ctx, visible, vec![count as u64], GgmlType::I32);
            let s = tensor(
                &ctx,
                &vec![7.0f32; ROWS * count],
                vec![ROWS as u64, count as u64],
                GgmlType::F32,
            );
            let rows = visible.iter().copied().max().unwrap().max(1) as usize;
            run(&ctx, |enc| {
                encode_lightning_scores_f16_matrix(
                    &ctx,
                    enc,
                    &LightningScores {
                        queries: &q,
                        head_weights: &w,
                        keys: &keys_t,
                        visible_counts: &v,
                        scores: &s,
                    },
                    H,
                    D,
                    ROWS,
                    rows,
                    count,
                )
                .unwrap();
            });
            read_f32(&s)
        };
        let batch = score(0, 9);
        for query in 0..9 {
            let visible = visible_all[query].max(0) as usize;
            let got = &batch[query * ROWS..query * ROWS + visible];
            let want = cpu_scores(
                &queries[query * H * D..(query + 1) * H * D],
                &weights[query * H..(query + 1) * H],
                &keys,
                visible,
                visible,
            );
            for (r, (g, e)) in got.iter().zip(&want).enumerate() {
                assert!(
                    (g - e).abs() <= 1e-5 * e.abs().max(1.0),
                    "query {query} row {r}: {g} vs {e}"
                );
            }
            // Rows past visibility (up to the dispatched extent) are -inf.
            let dispatched = (visible_all.iter().copied().max().unwrap() as usize).div_ceil(8) * 8;
            let tail = &batch[query * ROWS + visible..query * ROWS + dispatched.min(ROWS)];
            assert!(
                tail.iter().all(|&s| s == f32::NEG_INFINITY),
                "query {query} tail"
            );
            let alone = score(query, 1);
            let bits = |s: &[f32]| s.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
            assert_eq!(
                bits(&alone[..visible]),
                bits(got),
                "query {query} depends on its batch"
            );
        }
        for (q0, count) in [(0, 7), (1, 8)] {
            let part = score(q0, count);
            for i in 0..count {
                let visible = visible_all[q0 + i].max(0) as usize;
                assert!(
                    part[i * ROWS..i * ROWS + visible]
                        .iter()
                        .zip(&batch[(q0 + i) * ROWS..])
                        .all(|(a, b)| a.to_bits() == b.to_bits()),
                    "batch of {count} from {q0}: query {i}"
                );
            }
        }
    }

    /// Selector order key: higher first, signed zeros and subnormals equal,
    /// ties by lowest row; the selection is emitted in ascending row order.
    fn cpu_select(scores: &[f32], visible: usize, k: usize) -> Vec<i32> {
        let key = |v: f32| {
            let bits = v.to_bits();
            let bits = if bits & 0x7f80_0000 == 0 { 0 } else { bits };
            if bits & 0x8000_0000 != 0 {
                !bits
            } else {
                bits ^ 0x8000_0000
            }
        };
        let mut rows: Vec<usize> = (0..visible).collect();
        rows.sort_by(|&a, &b| key(scores[b]).cmp(&key(scores[a])).then(a.cmp(&b)));
        let mut picked: Vec<i32> = rows[..k.min(visible)].iter().map(|&r| r as i32).collect();
        picked.sort_unstable();
        picked
    }

    /// Top-512 of GLM-sized pool scores: distinct values, heavy ties at the
    /// threshold, signed zeros and subnormals at the threshold, and the
    /// first sparse visibility (513 pools); a NaN in the visible prefix sets
    /// status 2.
    #[test]
    fn top_k_selection_matches_cpu_contract_at_glm_shapes() {
        let Some(ctx) = crate::test_fixtures::metal_context_or_skip() else {
            return;
        };
        const ROWS: usize = 1_100;
        const K: usize = 512;
        let visible = [1_100i32, 700, 513, 1_000, 900];
        let mut scores = vec![0.0f32; ROWS * visible.len()];
        let distinct = noise(5, ROWS, 10.0);
        for r in 0..ROWS {
            scores[r] = distinct[r];
            scores[ROWS + r] = (r % 7) as f32; // ties at every level
            scores[2 * ROWS + r] = distinct[r];
            scores[3 * ROWS + r] = match r % 5 {
                0 => 0.0,
                1 => -0.0,
                2 => f32::from_bits(1), // subnormal
                3 => -f32::from_bits(3),
                _ => 1.0 + r as f32,
            };
            scores[4 * ROWS + r] = distinct[r];
        }
        scores[4 * ROWS + 123] = f32::NAN;
        let q = visible.len();
        let s = tensor(&ctx, &scores, vec![ROWS as u64, q as u64], GgmlType::F32);
        let v = tensor(&ctx, &visible, vec![q as u64], GgmlType::I32);
        let ids = tensor(
            &ctx,
            &vec![-7i32; K * q],
            vec![K as u64, q as u64],
            GgmlType::I32,
        );
        let counts = tensor(&ctx, &vec![-7i32; q], vec![q as u64], GgmlType::I32);
        let status = tensor(&ctx, &vec![-7i32; q], vec![q as u64], GgmlType::I32);
        run(&ctx, |enc| {
            encode_select_top_k_ids(
                &ctx,
                enc,
                &TopKSelection {
                    scores: &s,
                    visible_counts: &v,
                    ids: &ids,
                    counts: &counts,
                    status: &status,
                },
                ROWS,
                ROWS,
                K,
                q,
            )
            .unwrap();
        });
        let (ids, counts, status) = (read_i32(&ids), read_i32(&counts), read_i32(&status));
        for query in 0..4 {
            let want = cpu_select(
                &scores[query * ROWS..(query + 1) * ROWS],
                visible[query] as usize,
                K,
            );
            assert_eq!(status[query], SELECT_STATUS_OK, "query {query}");
            assert_eq!(counts[query], K as i32, "query {query}");
            assert_eq!(&ids[query * K..(query + 1) * K], &want[..], "query {query}");
        }
        assert_ne!(status[4], SELECT_STATUS_OK, "NaN in the visible prefix");
    }

    /// Geometry the selector cannot honor is refused before encoding.
    #[test]
    fn top_k_selection_refuses_non_pruning_geometry() {
        let Some(ctx) = crate::test_fixtures::metal_context_or_skip() else {
            return;
        };
        let s = tensor(&ctx, &[0.0f32; 512], vec![512, 1], GgmlType::F32);
        let i1 = tensor(&ctx, &[0i32; 1], vec![1], GgmlType::I32);
        let ids = tensor(&ctx, &[0i32; 512], vec![512, 1], GgmlType::I32);
        let (c, st) = (
            tensor(&ctx, &[0i32; 1], vec![1], GgmlType::I32),
            tensor(&ctx, &[0i32; 1], vec![1], GgmlType::I32),
        );
        let command = ctx.queue.commandBuffer().unwrap();
        let enc = KernelEncoder::begin(&command);
        let b = TopKSelection {
            scores: &s,
            visible_counts: &i1,
            ids: &ids,
            counts: &c,
            status: &st,
        };
        // 512 visible rows cannot prune a top-512 selection.
        assert!(encode_select_top_k_ids(&ctx, &enc, &b, 512, 512, 512, 1).is_err());
        enc.end();
    }
}
