//! Causal layer experiments, compiled only in the temporary research patch.
use super::*;
use std::collections::{BTreeMap, BTreeSet};

fn size(x: usize, y: usize) -> MTLSize {
    MTLSize {
        width: x,
        height: y,
        depth: 1,
    }
}
fn binding(s: &CapturedStep, slot: usize) -> (Buffer, usize) {
    let (_, Slot::Buffer(b, off)) = s.state.slots.iter().find(|(i, _)| *i == slot).unwrap() else {
        panic!("not buffer")
    };
    (b.clone(), *off)
}
fn words(s: &CapturedStep, slot: usize) -> Vec<u32> {
    let (_, Slot::Bytes(b)) = s.state.slots.iter().find(|(i, _)| *i == slot).unwrap() else {
        panic!("not bytes")
    };
    b.chunks_exact(4)
        .map(|v| u32::from_le_bytes(v.try_into().unwrap()))
        .collect()
}
fn float_values(s: &CapturedStep, slot: usize, count: usize) -> Vec<f32> {
    let (b, off) = binding(s, slot);
    unsafe {
        std::slice::from_raw_parts(
            b.contents().as_ptr().cast::<u8>().add(off).cast::<f32>(),
            count,
        )
    }
    .to_vec()
}
fn save_ranges(steps: &[CapturedStep], access: &AccessMap) -> Vec<SavedRange> {
    let mut ranges = BTreeMap::<usize, (Buffer, Vec<(usize, usize)>)>::new();
    for s in steps {
        for &(slot, _) in &access[&s.kernel] {
            let Some((_, Slot::Buffer(b, off))) = s.state.slots.iter().find(|(i, _)| *i == slot)
            else {
                continue;
            };
            if s.state.metadata.get(&slot).is_some_and(|m| !m.writable) {
                continue;
            }
            let len = s
                .state
                .metadata
                .get(&slot)
                .map(|m| m.bytes as usize)
                .unwrap_or(b.length() - off);
            assert!(len < 1024 * 1024 * 1024);
            ranges
                .entry(Retained::as_ptr(b) as *const () as usize)
                .or_insert_with(|| (b.clone(), Vec::new()))
                .1
                .push((*off, off + len));
        }
    }
    let mut saved = Vec::new();
    for (_, (b, mut spans)) in ranges {
        spans.sort_unstable();
        let mut merged = Vec::<(usize, usize)>::new();
        for (lo, hi) in spans {
            if let Some(last) = merged.last_mut() {
                if lo <= last.1 {
                    last.1 = last.1.max(hi);
                    continue;
                }
            }
            merged.push((lo, hi));
        }
        for (lo, hi) in merged {
            let bytes = unsafe {
                std::slice::from_raw_parts(b.contents().as_ptr().cast::<u8>().add(lo), hi - lo)
            }
            .to_vec();
            saved.push(SavedRange {
                buffer: b.clone(),
                offset: lo,
                bytes,
            });
        }
    }
    saved
}
fn restore(saved: &[SavedRange]) {
    for s in saved {
        unsafe {
            std::ptr::copy_nonoverlapping(
                s.bytes.as_ptr(),
                s.buffer.contents().as_ptr().cast::<u8>().add(s.offset),
                s.bytes.len(),
            );
        }
    }
}
fn run(ctx: &MetalContext, f: impl FnOnce(&KernelEncoder)) {
    let cmd = ctx.queue.commandBuffer().unwrap();
    let enc = KernelEncoder::begin(&cmd);
    f(&enc);
    enc.end();
    cmd.commit();
    wait_completed(&cmd).unwrap();
}
fn append(enc: &KernelEncoder, s: &CapturedStep) {
    s.state.apply(&enc.raw.borrow());
    enc.raw
        .borrow()
        .dispatchThreadgroups_threadsPerThreadgroup(s.grid, s.threads);
}
fn custom(
    name: &str,
    p: &Pipeline,
    grid: MTLSize,
    threads: MTLSize,
    refs: &[(usize, &CapturedStep, usize)],
    bytes: Option<&[u32]>,
) -> CapturedStep {
    let mut state = Replay::default();
    state.pipeline = Some(p.raw.clone());
    for &(to, s, from) in refs {
        let (_, slot) = s.state.slots.iter().find(|(i, _)| *i == from).unwrap();
        state.slots.push((to, slot.clone()));
        if let Some(m) = s.state.metadata.get(&from) {
            state.metadata.insert(to, m.clone());
        }
    }
    if let Some(b) = bytes {
        state.bytes(0, bytemuck::cast_slice(b));
    }
    CapturedStep {
        kernel: name.to_owned(),
        grid,
        threads,
        state,
        concurrent: false,
    }
}
pub struct LayerReplay {
    steps: Vec<CapturedStep>,
    entry: Vec<SavedRange>,
    expected: Vec<SavedRange>,
    access: AccessMap,
    start: usize,
    end: usize,
}
pub struct LayerVariant {
    steps: Vec<CapturedStep>,
    waves: Vec<Vec<usize>>,
    extra: Vec<SavedRange>,
}
pub struct OperatorOracle {
    reference: Vec<f64>,
    incumbent: Vec<f32>,
    output: (Buffer, usize, usize),
    state: Option<(Vec<f64>, Vec<f32>, Buffer, usize)>,
}
fn error_metrics(reference: &[f64], actual: &[f32]) -> serde_json::Value {
    let mut sq = 0f64;
    let mut max = 0f64;
    for (&r, &a) in reference.iter().zip(actual) {
        let d = f64::from(a) - r;
        sq += d * d;
        max = max.max(d.abs());
    }
    serde_json::json!({"rms":(sq/reference.len()as f64).sqrt(),"max_abs":max,"count":reference.len()})
}
impl OperatorOracle {
    pub fn evaluate(&self) -> serde_json::Value {
        let (b, o, n) = &self.output;
        let actual = unsafe {
            std::slice::from_raw_parts(b.contents().as_ptr().cast::<u8>().add(*o).cast::<f32>(), *n)
        };
        let base = error_metrics(&self.reference, &self.incumbent);
        let candidate = error_metrics(&self.reference, actual);
        let state=self.state.as_ref().map(|(reference,incumbent,b,o)|{
            let actual=unsafe{std::slice::from_raw_parts(b.contents().as_ptr().cast::<u8>().add(*o).cast::<f32>(),reference.len())};
            serde_json::json!({"incumbent":error_metrics(reference,incumbent),"candidate":error_metrics(reference,actual)})
        });
        serde_json::json!({"incumbent":base,"candidate":candidate,"rms_ratio":candidate["rms"].as_f64().unwrap()/base["rms"].as_f64().unwrap().max(1e-30),"state":state})
    }
}
impl DecodeGraph {
    pub fn feedback_probe(&self, enc: &KernelEncoder, greedy: &Pipeline) {
        let last = self.state.steps.last().unwrap();
        let first = &self.state.steps[0];
        let (logits, lo) = binding(last, 3);
        let (token, to) = binding(first, 2);
        let n = last.state.metadata[&3].shape[0] as u32;
        let threads = greedy.maxTotalThreadsPerThreadgroup().min(1024) / 32 * 32;
        enc.set_pipeline(greedy);
        enc.set_bytes(0, &[n, n, (threads / 32) as u32]);
        enc.set_buffer(1, &logits, lo as u64);
        enc.set_buffer(2, &token, to as u64);
        for slot in 0..3 {
            enc.set_threadgroup_memory(slot, (threads / 32) * 4);
        }
        enc.dispatch(size(1, 1), size(threads, 1));
        append(enc, first);
    }
    pub fn feedback_id(&self) -> i32 {
        let (b, o) = binding(&self.state.steps[0], 2);
        unsafe { *b.contents().as_ptr().cast::<u8>().add(o).cast::<i32>() }
    }
    pub fn prepare_layer(
        &self,
        ctx: &MetalContext,
        start: usize,
        end: usize,
        access: &AccessMap,
    ) -> LayerReplay {
        self.restore();
        run(ctx, |e| {
            for s in &self.state.steps[..start] {
                append(e, s)
            }
        });
        let steps = self.state.steps[start..end]
            .iter()
            .map(|s| CapturedStep {
                kernel: s.kernel.clone(),
                grid: s.grid,
                threads: s.threads,
                state: s.state.clone(),
                concurrent: s.concurrent,
            })
            .collect::<Vec<_>>();
        let entry = save_ranges(&steps, access);
        run(ctx, |e| {
            for s in &steps {
                append(e, s)
            }
        });
        // Final hyper residual and architectural recurrent/cache state only.
        // Intermediate buffers are intentionally elided by the fusion.
        let mut watched = Vec::new();
        let last = steps.last().unwrap();
        let (b, o) = binding(last, 3);
        watched.push((b, o, 40960));
        for s in &steps {
            let slots: Vec<usize> = if s.kernel == "kernel_gdn_step_decay_f32" {
                vec![6]
            } else if s.kernel == "kernel_qwen4exp_gdn_controls_prep_f32" {
                vec![10]
            } else if s.kernel == "kernel_qwen4exp_qsa_publish_kv_f16" {
                vec![3, 4]
            } else if s.kernel == "kernel_qwen4exp_qsa_write_pending_f32" {
                vec![2]
            } else {
                vec![]
            };
            for slot in slots {
                let (b, o) = binding(s, slot);
                watched.push((b, o, s.state.metadata[&slot].bytes as usize));
            }
        }
        let expected = watched
            .into_iter()
            .map(|(b, offset, n)| {
                let bytes = unsafe {
                    std::slice::from_raw_parts(b.contents().as_ptr().cast::<u8>().add(offset), n)
                }
                .to_vec();
                SavedRange {
                    buffer: b,
                    offset,
                    bytes,
                }
            })
            .collect();
        LayerReplay {
            steps,
            entry,
            expected,
            access: access.clone(),
            start,
            end,
        }
    }
    pub fn replay_interval(&self, enc: &KernelEncoder, start: usize, end: usize) {
        for s in &self.state.steps[start..end] {
            append(enc, s);
        }
    }
    pub fn step_count(&self) -> usize {
        self.state.steps.len()
    }
}
impl LayerReplay {
    pub fn operator_oracle(&self, ctx: &MetalContext) -> OperatorOracle {
        let gdn = self
            .steps
            .iter()
            .position(|s| s.kernel == "kernel_gdn_step_decay_f32");
        let i = gdn.unwrap_or_else(|| {
            self.steps
                .iter()
                .position(|s| s.kernel == "kernel_qwen4exp_qsa_decode_logits_reuse_f32")
                .unwrap()
        });
        self.restore();
        run(ctx, |e| {
            for s in &self.steps[..i] {
                append(e, s)
            }
        });
        if gdn.is_some() {
            let s = &self.steps[i];
            let norm = &self.steps[i + 1];
            let q = float_values(s, 1, 2048);
            let k = float_values(s, 2, 2048);
            let v = float_values(s, 3, 6144);
            let decay = float_values(s, 4, 48);
            let beta = float_values(s, 5, 48);
            let input = float_values(s, 6, 786432);
            let gate = float_values(norm, 3, 6144);
            let weight = float_values(norm, 2, 128);
            let eps = f32::from_bits(words(norm, 0)[1]) as f64;
            let mut state = vec![0f64; 786432];
            let mut reference = vec![0f64; 6144];
            for h in 0..48 {
                let hk = h % 16;
                for row in 0..128 {
                    let base = h * 16384 + row * 128;
                    let sk: f64 = (0..128)
                        .map(|d| {
                            f64::from(input[base + d])
                                * f64::from(decay[h])
                                * f64::from(k[hk * 128 + d])
                        })
                        .sum();
                    let delta = (f64::from(v[h * 128 + row]) - sk) * f64::from(beta[h]);
                    let mut out = 0f64;
                    for d in 0..128 {
                        let z = f64::from(input[base + d]) * f64::from(decay[h])
                            + delta * f64::from(k[hk * 128 + d]);
                        state[base + d] = z;
                        out += z * f64::from(q[hk * 128 + d]);
                    }
                    reference[h * 128 + row] = out;
                }
                let ss: f64 = reference[h * 128..(h + 1) * 128]
                    .iter()
                    .map(|x| x * x)
                    .sum();
                let scale = (ss / 128. + eps).sqrt().recip();
                for d in 0..128 {
                    reference[h * 128 + d] *=
                        scale * f64::from(weight[d]) / (1. + (-f64::from(gate[h * 128 + d])).exp());
                }
            }
            run(ctx, |e| {
                append(e, s);
                append(e, norm);
            });
            let (b, o) = binding(norm, 4);
            let (sb, so) = binding(s, 6);
            OperatorOracle {
                reference,
                incumbent: float_values(norm, 4, 6144),
                output: (b, o, 6144),
                state: Some((state, float_values(s, 6, 786432), sb, so)),
            }
        } else {
            let s = &self.steps[i];
            let pv = &self.steps[i + 2];
            let merge = &self.steps[i + 3];
            let a = words(s, 0);
            let count = a[3] as usize;
            let capacity = a[5] as usize;
            let scale = f32::from_bits(a[6]) as f64;
            let q = float_values(s, 1, 6144);
            let gate = float_values(merge, 2, 6144);
            let (ib, io) = binding(s, 3);
            let ids = unsafe {
                std::slice::from_raw_parts(
                    ib.contents().as_ptr().cast::<u8>().add(io).cast::<i32>(),
                    count,
                )
            };
            let (kb, ko) = binding(s, 2);
            let (vb, vo) = binding(pv, 3);
            let keys = unsafe {
                std::slice::from_raw_parts(
                    kb.contents()
                        .as_ptr()
                        .cast::<u8>()
                        .add(ko)
                        .cast::<half::f16>(),
                    capacity * 512,
                )
            };
            let values = unsafe {
                std::slice::from_raw_parts(
                    vb.contents()
                        .as_ptr()
                        .cast::<u8>()
                        .add(vo)
                        .cast::<half::f16>(),
                    capacity * 512,
                )
            };
            let mut reference = vec![0f64; 6144];
            for h in 0..24 {
                let kv = h / 12;
                let mut scores = vec![f64::NEG_INFINITY; count];
                for (j, &pos) in ids.iter().enumerate() {
                    if pos < 0 || pos as usize >= capacity {
                        continue;
                    }
                    let base = (pos as usize * 2 + kv) * 256;
                    let mut dot = 0f64;
                    for d in 0..256 {
                        dot += f64::from(q[h * 256 + d]) * keys[base + d].to_f64();
                    }
                    scores[j] = dot * scale;
                }
                let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let mut denom = 0f64;
                for x in &mut scores {
                    *x = (*x - max).exp();
                    denom += *x;
                }
                for (j, &pos) in ids.iter().enumerate() {
                    if pos < 0 || pos as usize >= capacity {
                        continue;
                    }
                    let base = (pos as usize * 2 + kv) * 256;
                    for d in 0..256 {
                        reference[h * 256 + d] += scores[j] * values[base + d].to_f64();
                    }
                }
                for d in 0..256 {
                    reference[h * 256 + d] /= denom * (1. + (-f64::from(gate[h * 256 + d])).exp());
                }
            }
            run(ctx, |e| {
                for s in &self.steps[i..i + 4] {
                    append(e, s)
                }
            });
            let (b, o) = binding(merge, 3);
            OperatorOracle {
                reference,
                incumbent: float_values(merge, 3, 6144),
                output: (b, o, 6144),
                state: None,
            }
        }
    }
    pub fn restore(&self) {
        restore(&self.entry);
    }
    pub fn replay(&self, enc: &KernelEncoder) {
        for s in &self.steps {
            append(enc, s);
        }
    }
    pub fn output(&self) -> Vec<f32> {
        float_values(self.steps.last().unwrap(), 3, 10240)
    }
    pub fn check(&self) -> serde_json::Value {
        let mut max_abs = 0f64;
        let mut sq = 0f64;
        let mut ref_sq = 0f64;
        let mut mismatches = 0usize;
        let mut finite = true;
        for s in &self.expected {
            let actual = unsafe {
                std::slice::from_raw_parts(
                    s.buffer.contents().as_ptr().cast::<u8>().add(s.offset),
                    s.bytes.len(),
                )
            };
            mismatches += actual.iter().zip(&s.bytes).filter(|(a, b)| a != b).count();
        }
        let reference: Vec<f32> = self.expected[0]
            .bytes
            .chunks_exact(4)
            .map(|x| f32::from_le_bytes(x.try_into().unwrap()))
            .collect();
        for (a, b) in self.output().iter().zip(reference) {
            finite &= a.is_finite();
            let d = f64::from(*a) - f64::from(b);
            max_abs = max_abs.max(d.abs());
            sq += d * d;
            ref_sq += f64::from(b).powi(2);
        }
        let statuses: Vec<i32> = self
            .steps
            .iter()
            .filter_map(|s| {
                let slot = if s.kernel == "kernel_qwen4exp_topk_guarded_simd_f32" {
                    4
                } else if s.kernel == "kernel_qwen4exp_qsa_select_scan_f32" {
                    5
                } else {
                    return None;
                };
                let (b, o) = binding(s, slot);
                Some(unsafe { *b.contents().as_ptr().cast::<u8>().add(o).cast::<i32>() })
            })
            .collect();
        serde_json::json!({"boundary_mismatched_bytes":mismatches,"hyper_max_abs":max_abs,"hyper_rel_l2":(sq/ref_sq.max(1e-30)).sqrt(),"finite":finite,"status_codes":statuses})
    }
    pub fn byte_floor(&self) -> f64 {
        self.metadata()["floor_us_at_1170_GBs"].as_f64().unwrap()
    }
    pub fn metadata(&self) -> serde_json::Value {
        let mut weights = 0f64;
        let mut persistent = 0usize;
        let mut by_kernel = BTreeMap::new();
        for s in &self.steps {
            let mut seen = BTreeSet::new();
            let mut bytes = 0f64;
            for &(slot, _) in &self.access[&s.kernel] {
                let Some(m) = s.state.metadata.get(&slot) else {
                    continue;
                };
                if m.writable {
                    continue;
                }
                let (b, o) = binding(s, slot);
                let id = Retained::as_ptr(&b) as *const () as usize;
                if !seen.insert((id, o, m.bytes)) {
                    continue;
                }
                let scale = if m.shape.len() == 3 && m.shape[2] == 512 {
                    10f64 / 512.
                } else {
                    1.
                };
                bytes += m.bytes as f64 * scale;
            }
            weights += bytes;
            by_kernel.insert(s.kernel.clone(), bytes);
            if s.kernel == "kernel_gdn_step_decay_f32" {
                persistent += 2 * 3145728;
            }
            if s.kernel == "kernel_qwen4exp_gdn_controls_prep_f32" {
                persistent += 2 * 122880;
            }
            if s.kernel == "kernel_qwen4exp_qsa_decode_logits_reuse_f32" {
                persistent += words(s, 0)[3] as usize * 2 * 256 * 2;
            }
            if s.kernel == "kernel_qwen4exp_qsa_split_f16" {
                persistent += words(s, 0)[0] as usize * 2 * 256 * 2;
            }
            if s.kernel == "kernel_qwen4exp_qsa_index_scores_4x128_f16" {
                persistent += words(s, 0)[0] as usize * 128 * 2;
            }
            if s.kernel == "kernel_qwen4exp_qsa_publish_kv_f16" {
                persistent += 2048;
            }
        }
        serde_json::json!({"start":self.start,"end":self.end,"dispatches":self.steps.len(),"weight_bytes":weights,"persistent_bytes":persistent,"logical_bytes":weights+persistent as f64,"floor_us_at_1170_GBs":(weights+persistent as f64)/1.17e6,"weight_bytes_by_kernel":by_kernel})
    }
    pub fn variant(
        &self,
        ctx: &MetalContext,
        pipes: &BTreeMap<String, Pipeline>,
        access: &AccessMap,
        parts: usize,
        splits: usize,
        fusions: u32,
    ) -> LayerVariant {
        let mut result = Vec::new();
        let mut extra = Vec::new();
        let mut shared_result: Option<CapturedStep> = None;
        let find = |name: &str| self.steps.iter().find(|s| s.kernel == name).unwrap();
        let ffn_inject = self
            .steps
            .iter()
            .rposition(|s| s.kernel == "kernel_qwen4exp_hc_inject_f32")
            .unwrap();
        let sum = find("kernel_moe_weighted_sum_f32");
        let axpy = find("kernel_axpy_scalar_f32");
        let shared = find("kernel_mat_vec_iq4_nl_f32_fast");
        let silu = find("kernel_silu_mul_f32");
        let mut i = 0;
        while i < self.steps.len() {
            let s = &self.steps[i];
            if fusions & 1 != 0 && s.kernel == "kernel_gdn_step_decay_f32" {
                let n = &self.steps[i + 1];
                assert_eq!(n.kernel, "kernel_qwen4exp_gdn_rmsnorm_sigmoid_hd128_r4_f32");
                self.restore();
                run(ctx, |e| {
                    for t in &self.steps[..i] {
                        append(e, t)
                    }
                });
                let values = float_values(s, 6, 786432);
                let old = ctx.buffer_from(&values).unwrap();
                let args = [parts as u32, words(n, 0)[1]];
                let mut step = custom(
                    "layer_gdn_fused",
                    &pipes["layer_gdn_fused"],
                    size(parts, 48),
                    size(128, 1),
                    &[
                        (1, s, 1),
                        (2, s, 2),
                        (3, s, 3),
                        (4, s, 4),
                        (5, s, 5),
                        (7, s, 6),
                        (8, n, 3),
                        (9, n, 2),
                        (10, n, 4),
                    ],
                    Some(&args),
                );
                step.state.buffer(6, &old, 0);
                step.state.metadata.insert(6, s.state.metadata[&6].clone());
                extra.push(SavedRange {
                    buffer: old,
                    offset: 0,
                    bytes: bytemuck::cast_slice(&values).to_vec(),
                });
                result.push(step);
                i += 2;
                continue;
            }
            if fusions & 2 != 0 && s.kernel == "kernel_qwen4exp_qsa_decode_logits_reuse_f32" {
                let pv = &self.steps[i + 2];
                let merge = &self.steps[i + 3];
                assert_eq!(pv.kernel, "kernel_qwen4exp_qsa_split_f16");
                let w = words(s, 0);
                let keys = (w[3] as usize).div_ceil(splits);
                assert!(keys <= 256);
                let args = [w[3], w[5], splits as u32, keys as u32, w[6]];
                let scratch = ctx.buffer_from(&vec![0f32; splits * 24 * 258]).unwrap();
                let mut first = custom(
                    "layer_qsa_flash",
                    &pipes["layer_qsa_flash"],
                    size(splits, 6),
                    size(128, 1),
                    &[(1, s, 1), (2, s, 2), (3, pv, 3), (4, s, 3)],
                    Some(&args),
                );
                let mut second = custom(
                    "layer_qsa_flash_merge",
                    &pipes["layer_qsa_flash_merge"],
                    size(24, 1),
                    size(32, 1),
                    &[(2, merge, 2), (3, merge, 3)],
                    Some(&args),
                );
                for (step, slot) in [(&mut first, 5), (&mut second, 1)] {
                    step.state.buffer(slot, &scratch, 0);
                    step.state.metadata.insert(
                        slot,
                        CaptureTensor {
                            dtype: "F32".into(),
                            shape: vec![(splits * 24 * 258) as u64],
                            bytes: (splits * 24 * 258 * 4) as u64,
                            writable: true,
                        },
                    );
                }
                result.push(first);
                result.push(second);
                i += 4;
                continue;
            }
            if fusions & 4 != 0 && s.kernel == "kernel_silu_mul_f32" {
                i += 1;
                continue;
            }
            if fusions & 4 != 0 && s.kernel == "kernel_mat_vec_iq4_nl_f32_fast" {
                let scratch = ctx.buffer_from(&vec![0f32; 2560]).unwrap();
                let mut step = custom(
                    "layer_shared_down",
                    &pipes["layer_shared_down"],
                    s.grid,
                    s.threads,
                    &[(0, s, 1), (1, silu, 1), (2, silu, 2)],
                    None,
                );
                step.state.buffer(3, &scratch, 0);
                step.state.metadata.insert(3, s.state.metadata[&3].clone());
                shared_result = Some(CapturedStep {
                    kernel: step.kernel.clone(),
                    grid: step.grid,
                    threads: step.threads,
                    state: step.state.clone(),
                    concurrent: false,
                });
                result.push(step);
                i += 1;
                continue;
            }
            if fusions & 8 != 0
                && (s.kernel == "kernel_moe_weighted_sum_f32"
                    || s.kernel == "kernel_axpy_scalar_f32")
            {
                i += 1;
                continue;
            }
            if fusions & 8 != 0 && i == ffn_inject {
                result.push(custom(
                    "layer_ffn_epilogue",
                    &pipes["layer_ffn_epilogue"],
                    size(10, 4),
                    size(256, 1),
                    &[
                        (0, sum, 1),
                        (1, sum, 2),
                        (2, shared_result.as_ref().unwrap_or(shared), 3),
                        (3, axpy, 2),
                        (4, s, 2),
                        (5, s, 3),
                    ],
                    None,
                ));
                i += 1;
                continue;
            }
            if fusions & 16 != 0
                && s.kernel == "kernel_copy_offset_f32"
                && i + 1 < self.steps.len()
                && self.steps[i + 1].kernel == "kernel_qwen4exp_hc_inject_f32"
            {
                if i + 1 == ffn_inject && fusions & 8 != 0 {
                    i += 1;
                    continue;
                }
                let mut inj = self.steps[i + 1].state.clone();
                let (b, o) = binding(s, 1);
                inj.buffer(1, &b, o as u64);
                inj.metadata.insert(1, s.state.metadata[&1].clone());
                let n = &self.steps[i + 1];
                result.push(CapturedStep {
                    kernel: n.kernel.clone(),
                    grid: n.grid,
                    threads: n.threads,
                    state: inj,
                    concurrent: false,
                });
                i += 2;
                continue;
            }
            result.push(CapturedStep {
                kernel: s.kernel.clone(),
                grid: s.grid,
                threads: s.threads,
                state: s.state.clone(),
                concurrent: false,
            });
            i += 1;
        }
        let spans: Vec<_> = result.iter().map(|s| step_spans(s, access)).collect();
        let mut successors = vec![Vec::new(); result.len()];
        let mut degree = vec![0usize; result.len()];
        for j in 0..result.len() {
            for k in j + 1..result.len() {
                if spans[j]
                    .iter()
                    .any(|a| spans[k].iter().any(|b| conflict(a, b)))
                {
                    successors[j].push(k);
                    degree[k] += 1;
                }
            }
        }
        let mut ready: Vec<_> = degree
            .iter()
            .enumerate()
            .filter_map(|(i, &d)| (d == 0).then_some(i))
            .collect();
        let mut waves = Vec::new();
        while !ready.is_empty() {
            let wave = std::mem::take(&mut ready);
            for &j in &wave {
                for &k in &successors[j] {
                    degree[k] -= 1;
                    if degree[k] == 0 {
                        ready.push(k);
                    }
                }
            }
            waves.push(wave);
        }
        assert_eq!(waves.iter().map(Vec::len).sum::<usize>(), result.len());
        LayerVariant {
            steps: result,
            waves,
            extra,
        }
    }
}
impl LayerVariant {
    pub fn restore(&self) {
        restore(&self.extra);
    }
    pub fn metadata(&self) -> serde_json::Value {
        serde_json::json!({"dispatches":self.steps.len(),"waves":self.waves,"private_state_bytes":self.extra.iter().map(|s|s.bytes.len()).sum::<usize>()})
    }
    pub fn replay(&self, enc: &KernelEncoder, concurrent: bool) {
        if concurrent {
            for (n, wave) in self.waves.iter().enumerate() {
                if n > 0 {
                    enc.raw
                        .borrow()
                        .memoryBarrierWithScope(objc2_metal::MTLBarrierScope::Buffers);
                }
                for &i in wave {
                    append(enc, &self.steps[i]);
                }
            }
        } else {
            for s in &self.steps {
                append(enc, s);
            }
        }
    }
}
