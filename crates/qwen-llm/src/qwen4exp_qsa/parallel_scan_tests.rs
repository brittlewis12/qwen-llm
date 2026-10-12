use super::*;
use objc2_metal::MTLCommandQueue;

const TOP_K: usize = 512;
const GUARD: u32 = 0x5a17_c3e9;
const PREFIX: usize = 5;
const SUFFIX: usize = 11;
const CASES: usize = 21;

// Numeric comparison is deliberately independent of the GPU's sortable-bit key.
fn oracle(bits: &[u32], visible: i32, top_k: usize) -> (Vec<i32>, i32, i32) {
    let valid = visible > 0 && visible as usize <= bits.len();
    let n = if valid { visible as usize } else { 0 };
    let count = n.min(top_k);
    let status = if !valid {
        1
    } else if bits[..n].iter().any(|&b| !f32::from_bits(b).is_finite()) {
        2
    } else {
        0
    };
    let mut indices: Vec<usize> = (0..n).collect();
    if status == 0 {
        let numeric = |b: u32| {
            let f = f32::from_bits(b);
            if f == 0.0 || f.is_subnormal() {
                0.0_f64
            } else {
                f as f64
            }
        };
        indices.sort_unstable_by(|&a, &b| {
            numeric(bits[b])
                .partial_cmp(&numeric(bits[a]))
                .unwrap()
                .then_with(|| a.cmp(&b))
        });
    }
    indices.truncate(count);
    indices.sort_unstable();
    let mut ids = vec![-1; top_k];
    for (out, id) in ids.iter_mut().zip(indices) {
        *out = id as i32;
    }
    (ids, count as i32, status)
}

fn score_case(capacity: usize, case: usize, seed: usize) -> (Vec<u32>, i32) {
    const VALUES: [u32; 14] = [
        0,
        0x8000_0000,
        1,
        0x8000_0001,
        0x007f_ffff,
        0x807f_ffff,
        0x0080_0000,
        0x8080_0000,
        0x3f80_0000,
        0xbf80_0000,
        0x7f7f_ffff,
        0xff7f_ffff,
        0x3f00_0000,
        0xbf00_0000,
    ];
    let mut bits: Vec<u32> = (0..capacity)
        .map(|i| VALUES[(i * 7 + i / 31 + seed) % VALUES.len()])
        .collect();
    let mut visible = capacity as i32;
    match case {
        0 => {}
        1 => bits.fill(1.0_f32.to_bits()),
        2 => {
            for (i, b) in bits.iter_mut().enumerate() {
                *b = VALUES[i % 6];
            }
        }
        3 => {
            // 256 winners span all SIMD groups. The remaining 256 selections
            // cut through ties spanning lane chunks 31/32 and later chunks.
            let chunk = capacity / 256;
            bits.fill((-1.0_f32).to_bits());
            bits[31 * chunk..43 * chunk].fill(0);
            for i in (0..capacity).step_by(chunk) {
                bits[i] = 1.0_f32.to_bits();
            }
        }
        4 => {
            visible -= 1;
            bits[capacity - 1] = 0x7fc0_0123;
        }
        5 => visible = 512,
        6 => visible = 511,
        7 => visible = 513,
        8 => visible = 1,
        9 => {
            visible = 0;
            bits.fill(0x7fc0_0123);
        }
        10 => {
            visible = -1;
            bits.fill(0xff80_0000);
        }
        11 => {
            visible += 1;
            bits.fill(0x7fc0_0123);
        }
        12 => bits[capacity - 1] = 0x7fc0_0123,
        13 => bits[capacity / 2] = 0x7f80_0000,
        14 => bits[31] = 0xff80_0000,
        15 => bits[32] = 0x7f80_0001,
        16 => visible = 257,
        17 => visible -= 33,
        18 => visible = 31,
        19 => visible = 32,
        20 => visible = 33,
        _ => unreachable!(),
    }
    if visible > 0 && (visible as usize) < capacity {
        bits[visible as usize..].fill(0x7fc0_0123);
    }
    (bits, visible)
}

#[test]
fn parallel_scan_cpu_oracle_canonical_bits_and_error_precedence() {
    let bits = [
        0,
        0x8000_0000,
        1,
        0x8000_0001,
        0x007f_ffff,
        0x807f_ffff,
        0x0080_0000,
        0x8080_0000,
    ];
    assert_eq!(oracle(&bits, 8, 4), (vec![0, 1, 2, 6], 4, 0));
    assert_eq!(oracle(&[0x7fc0_0001], 0, 3), (vec![-1; 3], 0, 1));
    for invalid in [0x7fc0_0001, 0x7f80_0001, 0x7f80_0000, 0xff80_0000] {
        assert_eq!(oracle(&[0, invalid, 0], 3, 2), (vec![0, 1], 2, 2));
        assert_eq!(oracle(&[0, invalid], 1, 2), (vec![0, -1], 1, 0));
    }
    for capacity in [8192, 16384] {
        let (bits, n) = score_case(capacity, 3, 0);
        let (ids, count, status) = oracle(&bits, n, TOP_K);
        assert_eq!((count, status), (512, 0));
        assert!(ids.windows(2).all(|w| w[0] < w[1]));
        for i in (0..capacity).step_by(capacity / 256) {
            assert!(ids.contains(&(i as i32)));
        }
    }
}

#[test]
fn parallel_scan_scope_default_nested_error_unwind_and_threads() {
    assert_eq!(packed_selector_kernel(), SELECTOR_KERNEL);
    let (result, count) = with_packed_qsa_parallel_scan(true, || {
        assert_eq!(packed_selector_kernel(), PARALLEL_SCAN_SELECTOR_KERNEL);
        // Simulate completed dispatch accounting without any Metal work.
        PACKED_SELECTOR_OVERRIDE.with(|s| {
            let mut v = s.get();
            v.substitutions = 3;
            s.set(v);
        });
        let (inner, n) = with_packed_qsa_parallel_scan(false, || {
            assert_eq!(packed_selector_kernel(), SELECTOR_KERNEL);
            Err::<(), _>("expected")
        });
        assert!(inner.is_err());
        assert_eq!(n, 0);
        let panic =
            std::panic::catch_unwind(|| with_packed_qsa_parallel_scan(false, || panic!("restore")));
        assert!(panic.is_err());
        assert_eq!(packed_selector_kernel(), PARALLEL_SCAN_SELECTOR_KERNEL);
        std::thread::spawn(|| {
            assert_eq!(packed_selector_kernel(), SELECTOR_KERNEL);
            with_packed_qsa_parallel_scan(true, || {
                assert_eq!(packed_selector_kernel(), PARALLEL_SCAN_SELECTOR_KERNEL)
            });
            assert_eq!(packed_selector_kernel(), SELECTOR_KERNEL);
        })
        .join()
        .unwrap();
        Err::<(), _>("outer")
    });
    assert!(result.is_err());
    assert_eq!(count, 3);
    assert_eq!(packed_selector_kernel(), SELECTOR_KERNEL);
    assert!(
        std::panic::catch_unwind(|| with_packed_qsa_parallel_scan(true, || panic!(
            "outer restore"
        )))
        .is_err()
    );
    assert_eq!(packed_selector_kernel(), SELECTOR_KERNEL);
}

#[test]
fn parallel_scan_pipeline_capability_rejects_without_fallback() {
    let check = |simd, threads, static_bytes, max_bytes| {
        validate_cooperative_pipeline(
            PARALLEL_SCAN_SELECTOR_KERNEL,
            simd,
            threads,
            static_bytes,
            SELECTOR_SCRATCH_BYTES,
            max_bytes,
        )
    };
    assert!(check(32, 256, 0, 2048).is_ok());
    assert!(check(32, 1024, 16, 2064).is_ok());
    for width in [0, 16, 64] {
        assert!(check(width, 256, 0, 2048).is_err());
    }
    assert!(check(32, 255, 0, 2048).is_err());
    assert!(check(32, 256, 16, 2048).is_err());
    assert!(check(32, 256, usize::MAX, usize::MAX).is_err());
}

struct Guarded {
    backing: MetalTensor,
    view: MetalTensor,
}
impl Guarded {
    fn new(ctx: &MetalContext, dtype: GgmlType, shape: Vec<u64>) -> Self {
        let count = shape.iter().product::<u64>() as usize;
        let full = vec![(PREFIX + count + SUFFIX) as u64];
        let backing = match dtype {
            GgmlType::F32 => MetalTensor::zeros_f32(ctx, full).unwrap(),
            GgmlType::I32 => MetalTensor::zeros_i32(ctx, full).unwrap(),
            _ => unreachable!(),
        };
        let view = backing.view_subrange(PREFIX as u64, shape);
        let result = Self { backing, view };
        result.reset();
        result
    }
    fn reset(&self) {
        write_words(
            &self.backing,
            &vec![GUARD; self.backing.n_elements() as usize],
        );
    }
    fn check_guards(&self) {
        let words = read_words(&self.backing);
        assert!(words[..PREFIX].iter().all(|&x| x == GUARD));
        assert!(words[words.len() - SUFFIX..].iter().all(|&x| x == GUARD));
    }
}
fn write_words(t: &MetalTensor, words: &[u32]) {
    assert!(t.is_writable());
    assert_eq!(t.n_elements() as usize, words.len());
    assert!(t.offset as usize + words.len() * 4 <= t.buffer.length());
    unsafe {
        std::ptr::copy_nonoverlapping(
            words.as_ptr(),
            t.buffer
                .contents()
                .as_ptr()
                .cast::<u8>()
                .add(t.offset as usize)
                .cast::<u32>(),
            words.len(),
        );
    }
}
fn read_words(t: &MetalTensor) -> Vec<u32> {
    unsafe {
        std::slice::from_raw_parts(
            t.buffer
                .contents()
                .as_ptr()
                .cast::<u8>()
                .add(t.offset as usize)
                .cast::<u32>(),
            t.n_elements() as usize,
        )
        .to_vec()
    }
}

#[test]
#[ignore = "leased Metal correctness only; owner runs with MTL_DEBUG_LAYER=1"]
fn parallel_scan_gpu_exact_ids_status_offsets_and_immutability() {
    let _lease = crate::metal::acquire_metal_benchmark_lease().unwrap();
    let ctx = MetalContext::new().unwrap();
    for capacity in [8192, 16384] {
        for queries in [1, 17] {
            let scores = Guarded::new(&ctx, GgmlType::F32, vec![capacity as u64, queries as u64]);
            let visible = Guarded::new(&ctx, GgmlType::I32, vec![queries as u64]);
            let outputs: Vec<_> = (0..2)
                .map(|_| {
                    [
                        Guarded::new(&ctx, GgmlType::I32, vec![TOP_K as u64, queries as u64]),
                        Guarded::new(&ctx, GgmlType::I32, vec![queries as u64]),
                        Guarded::new(&ctx, GgmlType::I32, vec![queries as u64]),
                    ]
                })
                .collect();
            let logical_bytes = scores.backing.n_bytes()
                + visible.backing.n_bytes()
                + outputs
                    .iter()
                    .flatten()
                    .map(|x| x.backing.n_bytes())
                    .sum::<u64>();
            let device_bytes = scores.backing.buffer.allocatedSize()
                + visible.backing.buffer.allocatedSize()
                + outputs
                    .iter()
                    .flatten()
                    .map(|x| x.backing.buffer.allocatedSize())
                    .sum::<usize>();
            eprintln!(
                "parallel_scan_correctness capacity={capacity} queries={queries} logical_bytes={logical_bytes} device_bytes={device_bytes} dynamic_tgm_bytes={SELECTOR_SCRATCH_BYTES}"
            );
            for case in 0..CASES {
                let mut score_bits = Vec::with_capacity(capacity * queries);
                let mut visibility = Vec::with_capacity(queries);
                let mut expected = [Vec::new(), Vec::new(), Vec::new()];
                for query in 0..queries {
                    let (bits, n) = score_case(capacity, (case + query) % CASES, query);
                    let (ids, count, status) = oracle(&bits, n, TOP_K);
                    score_bits.extend(bits);
                    visibility.push(n as u32);
                    expected[0].extend(ids.into_iter().map(|v| v as u32));
                    expected[1].push(count as u32);
                    expected[2].push(status as u32);
                }
                write_words(&scores.view, &score_bits);
                write_words(&visible.view, &visibility);
                for output in outputs.iter().flatten() {
                    output.reset();
                }
                let command = ctx.queue.commandBuffer().unwrap();
                let enc = KernelEncoder::begin(&command);
                crate::metal::dispatch_census_begin();
                // The scalar wrapper must ignore an active packed override.
                let (result, substitutions) = with_packed_qsa_parallel_scan(true, || {
                    encode_select_blocks_tensors(
                        &ctx,
                        &enc,
                        &scores.view,
                        &visible.view,
                        &outputs[0][0].view,
                        &outputs[0][1].view,
                        &outputs[0][2].view,
                        capacity,
                        TOP_K,
                        queries,
                    )?;
                    encode_select_blocks_packed_tensors(
                        &ctx,
                        &enc,
                        &scores.view,
                        &visible.view,
                        &outputs[1][0].view,
                        &outputs[1][1].view,
                        &outputs[1][2].view,
                        capacity,
                        TOP_K,
                        queries,
                    )
                });
                result.unwrap();
                assert_eq!(substitutions, 1);
                let census = crate::metal::dispatch_census_take();
                assert_eq!(
                    census.iter().map(|r| r.kernel.as_str()).collect::<Vec<_>>(),
                    [SELECTOR_KERNEL, PARALLEL_SCAN_SELECTOR_KERNEL]
                );
                enc.end();
                command.commit();
                crate::metal::wait_completed(&command).unwrap();
                for (arm, output) in outputs.iter().enumerate() {
                    for (kind, tensor) in output.iter().enumerate() {
                        assert_eq!(
                            read_words(&tensor.view),
                            expected[kind],
                            "capacity={capacity} queries={queries} case={case} arm={arm} field={kind}"
                        );
                        tensor.check_guards();
                    }
                }
                assert_eq!(read_words(&scores.view), score_bits);
                assert_eq!(read_words(&visible.view), visibility);
                scores.check_guards();
                visible.check_guards();
            }
            // Candidate validation rejects malformed physical bindings before any dispatch.
            let command = ctx.queue.commandBuffer().unwrap();
            let enc = KernelEncoder::begin(&command);
            for defect in 0..8 {
                let mut bad_scores = scores.view.clone();
                let mut bad_ids = outputs[1][0].view.clone();
                let mut bad_capacity = capacity;
                let mut bad_budget = TOP_K;
                let mut bad_queries = queries;
                match defect {
                    0 => bad_scores.offset += 1,
                    1 => bad_scores.offset = bad_scores.buffer.length() as u64,
                    2 => bad_ids = bad_scores.clone(),
                    3 => {
                        bad_ids = bad_scores.clone();
                        bad_ids.dtype = GgmlType::I32;
                        bad_ids.shape = vec![TOP_K as u64, queries as u64];
                    }
                    4 => bad_capacity = usize::MAX,
                    5 => bad_budget = 0,
                    6 => bad_budget = capacity + 1,
                    7 => bad_queries = 0,
                    _ => unreachable!(),
                }
                crate::metal::dispatch_census_begin();
                let (result, count) = with_packed_qsa_parallel_scan(true, || {
                    encode_select_blocks_packed_tensors(
                        &ctx,
                        &enc,
                        &bad_scores,
                        &visible.view,
                        &bad_ids,
                        &outputs[1][1].view,
                        &outputs[1][2].view,
                        bad_capacity,
                        bad_budget,
                        bad_queries,
                    )
                });
                assert!(result.is_err());
                assert_eq!(count, 0);
                assert!(crate::metal::dispatch_census_take().is_empty());
            }
            enc.end();
            // Deliberately never commit this validation-only command.
        }
    }
}
