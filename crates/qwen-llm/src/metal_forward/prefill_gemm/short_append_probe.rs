//! Explicit model/Metal witness for short-append dispatch and row screens.
//! Never part of CPU acceptance or the default library test run.
use super::*;
use crate::metal_forward::{MetalForward, MetalModel, MetalSession};
use crate::{codec, gguf::GgufFile, loader::Model, metal::*, metal_dflash::*, tensor::TensorDesc};
use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLCommandQueue};
use serde_json::json;
use std::{cell::Cell, collections::BTreeMap};

/// Explicit ignored witness mode: trace, dense, q5, expert, quality (slice/document/pooled guards) or quality-document (the existing document/pooled NLL/KL protocol, with every slice reported).
const PROBE_MODE_ENV: &str = "QWEN_SHORT_PROBE_MODE";
/// External model fixture for the ignored short-append witness: 27b or a3b.
const PROBE_MODEL_ENV: &str = "QWEN_SHORT_PROBE_MODEL";
/// Optional comma-separated operator row counts for the ignored witness.
const PROBE_ROWS_ENV: &str = "QWEN_SHORT_PROBE_ROWS";
/// Frozen git revision supplying teacher-forcing texts; defaults to HEAD.
const PROBE_CORPUS_REV_ENV: &str = "QWEN_SHORT_PROBE_CORPUS_REV";
/// Optional prose/code selector to bound each value campaign; unset checks both.
const PROBE_DOC_ENV: &str = "QWEN_SHORT_PROBE_DOC";
/// Ignored quality witness only: override short expert roles with gateup, down or both; unset uses the production role selection. The original 2048-row route is unaffected.
const PROBE_EXPERT_ROLE_ENV: &str = "QWEN_SHORT_PROBE_EXPERT_ROLE";

pub(super) fn expert_roles_override() -> Option<[bool; 2]> {
    std::env::var(PROBE_EXPERT_ROLE_ENV)
        .ok()
        .map(|role| match role.as_str() {
            "gateup" => [true, false],
            "down" => [false, true],
            "both" => [true, true],
            _ => panic!("unknown expert role override"),
        })
}

thread_local! { static OVERRIDE: Cell<Option<bool>> = const { Cell::new(None) }; }
pub(super) fn override_value() -> Option<bool> {
    OVERRIDE.with(Cell::get)
}
pub(super) struct Scope(Option<bool>);
impl Scope {
    pub(super) fn new(on: bool) -> Self {
        Self(OVERRIDE.with(|v| v.replace(Some(on))))
    }
}
impl Drop for Scope {
    fn drop(&mut self) {
        OVERRIDE.with(|v| v.set(self.0));
    }
}
fn run_scope<T>(on: bool, f: impl FnOnce() -> T) -> T {
    let _s = Scope::new(on);
    f()
}
fn emit(v: serde_json::Value) {
    println!("{v}");
}
fn read_f32(t: &MetalTensor) -> &[f32] {
    unsafe {
        std::slice::from_raw_parts(
            t.buffer
                .contents()
                .as_ptr()
                .cast::<u8>()
                .add(t.offset as usize)
                .cast::<f32>(),
            t.n_elements() as usize,
        )
    }
}
fn read_i32(t: &MetalTensor, n: usize) -> Vec<i32> {
    unsafe {
        std::slice::from_raw_parts(
            t.buffer
                .contents()
                .as_ptr()
                .cast::<u8>()
                .add(t.offset as usize)
                .cast::<i32>(),
            n,
        )
        .to_vec()
    }
}
fn geometry(moe: bool, _append: usize, total: usize) -> (usize, PrefillScratchConfig) {
    if moe && total >= 1025 {
        (
            2048,
            PrefillScratchConfig {
                matrix_query_cap: Some(1024),
            },
        )
    } else {
        (total.clamp(1, 1024), PrefillScratchConfig::default())
    }
}
fn scratch(
    ctx: &MetalContext,
    mm: &MetalModel,
    append: usize,
    total: usize,
) -> MetalDFlashLayerMajorScratch {
    let (block, config) = geometry(mm.arch.kind == ArchKind::Moe, append, total);
    MetalDFlashLayerMajorScratch::fresh_prefill_with_matrix_max_pos_configured(
        ctx,
        mm,
        block as u32,
        total.max(block),
        config,
    )
    .unwrap()
}
fn ids(n: usize, vocab: usize) -> Vec<i32> {
    let mut s = 1u64;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s % vocab as u64) as i32
        })
        .collect()
}
fn census(rows: Vec<DispatchCensusRow>) -> Vec<serde_json::Value> {
    let mut agg: BTreeMap<(String, u64, u64, u64), (usize, u64)> = BTreeMap::new();
    for r in rows {
        let v = agg
            .entry((r.kernel, r.grid_width, r.grid_height, r.grid_depth))
            .or_default();
        v.0 += 1;
        v.1 += r.grid_tgs;
    }
    agg.into_iter().map(|((kernel,x,y,z),(count,tgs))|json!({"kernel":kernel,"grid":[x,y,z],"count":count,"threadgroups":tgs})).collect()
}
fn trace(ctx: &MetalContext, mm: &MetalModel) {
    let mf = MetalForward::new(ctx, mm);
    for n in [512, 1024, 1300] {
        let mut sc = scratch(ctx, mm, n, n);
        let mut session = MetalSession::fresh(ctx, mm, n + 1).unwrap();
        dispatch_census_begin();
        run_scope(false, || {
            prefill_tokens_with_multi_hidden(
                &mf,
                &ids(n, mm.arch.vocab_size as usize),
                0,
                &mut session,
                &mut sc,
                &[],
                None,
            )
        })
        .unwrap();
        let rows = dispatch_census_take();
        let counts = if mm.arch.kind == ArchKind::Moe {
            read_i32(&sc.moe_group_count_pack, 256)
        } else {
            vec![]
        };
        emit(
            json!({"record":"trace","tokens":n,"block":geometry(mm.arch.kind==ArchKind::Moe,n,n).0,"counts_last_layer":counts,"dispatches":census(rows)}),
        );
    }
}
fn metrics(a: &[f32], r: &[f64]) -> (f64, f64) {
    let mut p = 0f64;
    let mut e = 0.;
    let mut s = 0.;
    for (&a, &b) in a.iter().zip(r) {
        assert!(a.is_finite() && b.is_finite());
        let d = f64::from(a) - b;
        p = p.max(d.abs());
        e += d * d;
        s += b * b;
    }
    (p, (e / s).sqrt())
}
fn command(ctx: &MetalContext, f: impl FnOnce(&KernelEncoder)) -> f64 {
    let c = ctx.queue.commandBuffer().unwrap();
    let e = KernelEncoder::begin(&c);
    f(&e);
    e.end();
    c.commit();
    wait_completed(&c).unwrap();
    (c.GPUEndTime() - c.GPUStartTime()) * 1e3
}
fn thermal(arm: &str, where_: &str) {
    let o = std::process::Command::new("notifyutil")
        .args(["-g", "com.apple.system.thermalpressurelevel"])
        .output()
        .unwrap();
    let raw = String::from_utf8(o.stdout).unwrap();
    emit(json!({"record":"thermal","arm":arm,"where":where_,"value":raw.trim()}));
    assert!(
        o.status.success() && raw.split_whitespace().last() == Some("0"),
        "nonzero thermal; stop"
    );
}
struct Dense {
    m: usize,
    k: usize,
    n: usize,
    dtype: GgmlType,
    w: MetalTensor,
    x: MetalTensor,
    y: MetalTensor,
    spare: MetalTensor,
    reference: Vec<f64>,
    points: Vec<usize>,
}
impl Dense {
    fn new(ctx: &MetalContext, gg: &GgufFile, m: usize, tensor: &str) -> Self {
        let d = gg.find(tensor).unwrap();
        let (k, n) = (d.shape[0] as usize, d.shape[1] as usize);
        let bytes = match d.dtype {
            GgmlType::Q4_K => 144,
            GgmlType::Q5_K => 176,
            GgmlType::Q6_K => 210,
            _ => panic!("unsupported dense witness dtype"),
        };
        let stride = k / 256 * bytes;
        let points: Vec<_> = (0..32)
            .flat_map(|r| (0..32).map(move |c| (r * (m - 1) / 31) * n + c * (n - 1) / 31))
            .collect();
        let cols: Vec<_> = (0..32).map(|c| c * (n - 1) / 31).collect();
        let decoded: Vec<Vec<f32>> = cols
            .iter()
            .map(|&c| {
                codec::dequant_to_f32(
                    &TensorDesc {
                        name: "reference".into(),
                        shape: vec![k as u64],
                        dtype: d.dtype,
                        shard_idx: 0,
                        data_offset: 0,
                        n_bytes: stride as u64,
                    },
                    &gg.slice(d)[c * stride..(c + 1) * stride],
                )
                .unwrap()
            })
            .collect();
        let mut rng = 0x5a31_89efu64;
        let xv: Vec<f32> = (0..m * k)
            .map(|_| {
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                (rng >> 40) as f32 / (1u64 << 23) as f32 - 1.
            })
            .collect();
        let reference = points
            .iter()
            .enumerate()
            .map(|(i, &p)| {
                xv[p / n * k..(p / n + 1) * k]
                    .iter()
                    .zip(&decoded[i % 32])
                    .map(|(&x, &w)| f64::from(x) * f64::from(w))
                    .sum()
            })
            .collect();
        Self {
            m,
            k,
            n,
            dtype: d.dtype,
            w: MetalTensor::from_bytes(ctx, gg.slice(d), d.shape.clone(), d.dtype).unwrap(),
            x: MetalTensor::from_bytes(
                ctx,
                bytemuck::cast_slice(&xv),
                vec![(m * k) as u64],
                GgmlType::F32,
            )
            .unwrap(),
            y: MetalTensor::zeros_f32(ctx, vec![(m * n) as u64]).unwrap(),
            spare: MetalTensor::zeros_f32(ctx, vec![(m * k) as u64]).unwrap(),
            reference,
            points,
        }
    }
    fn run(&self, ctx: &MetalContext, on: bool) -> (f64, (f64, f64)) {
        unsafe {
            std::slice::from_raw_parts_mut(
                self.y.buffer.contents().as_ptr().cast::<f32>(),
                self.m * self.n,
            )
            .fill(f32::NAN);
        }
        let ms = run_scope(on, || {
            command(ctx, |e| {
                if self.dtype == GgmlType::Q5_K {
                    if on {
                        encode_mat_mat_tensor_f32_narrow(
                            ctx, e, &self.w, &self.x, &self.y, self.k, self.n, self.m,
                        )
                        .unwrap();
                    } else {
                        encode_mat_mat_dispatch(
                            ctx, e, &self.w, &self.x, &self.y, self.k, self.n, self.m,
                        )
                        .unwrap();
                    }
                    return;
                }
                encode_qwen_ffn_prefill_mat_mat(
                    ctx,
                    e,
                    &QWEN3_27B,
                    &self.w,
                    &self.x,
                    &self.y,
                    &self.spare,
                    self.k,
                    self.n,
                    self.m,
                )
                .unwrap()
            })
        });
        let data = read_f32(&self.y);
        assert!(data.iter().all(|v| v.is_finite()));
        let vals: Vec<f32> = self.points.iter().map(|&p| data[p]).collect();
        (ms, metrics(&vals, &self.reference))
    }
}
fn dense(ctx: &MetalContext, gg: &GgufFile, q5: bool) {
    let ms = std::env::var(PROBE_ROWS_ENV).unwrap_or(
        if q5 {
            "128,129,276,512,513,769,1000,1023,1024,1300"
        } else {
            "128,256,276,384,512,513,768,769,1000,1023,1024,1300"
        }
        .into(),
    );
    let mut fixtures = Vec::new();
    for m in ms.split(',').map(|s| s.parse::<usize>().unwrap()) {
        let names: &[&str] = if q5 {
            &["blk.0.ssm_out.weight"]
        } else {
            &[
                "blk.0.ffn_gate.weight",
                "blk.8.ffn_down.weight",
                "blk.0.ffn_down.weight",
            ]
        };
        for name in names {
            fixtures.push(Dense::new(ctx, gg, m, name));
        }
    }
    let mut bounds = BTreeMap::new();
    thermal("qualify", "before");
    for (i, f) in fixtures.iter().enumerate() {
        let (_, a) = f.run(ctx, false);
        let (_, b) = f.run(ctx, true);
        let pass = b.0 <= 1.5 * a.0 && b.1 <= 1.5 * a.1;
        emit(
            json!({"record":"qualify","m":f.m,"n":f.n,"k":f.k,"dtype":format!("{:?}",f.dtype),"base":a,"candidate":b,"ratios":[b.0/a.0,b.1/a.1],"pass":pass}),
        );
        assert!(pass, "F64 bound");
        bounds.insert(i, a);
    }
    thermal("qualify", "after");
    for (arm, on) in [false, true, true, false].into_iter().enumerate() {
        std::thread::sleep(std::time::Duration::from_secs(30));
        thermal(&format!("{arm}-{on}"), "before");
        for f in &fixtures {
            f.run(ctx, on);
            f.run(ctx, on);
        }
        for sample in 0..6 {
            for (i, f) in fixtures.iter().enumerate() {
                let (ms, err) = f.run(ctx, on);
                let a = bounds[&i];
                assert!(err.0 <= 1.5 * a.0 && err.1 <= 1.5 * a.1);
                emit(
                    json!({"record":"op","arm":arm,"candidate":on,"sample":sample,"m":f.m,"k":f.k,"n":f.n,"dtype":format!("{:?}",f.dtype),"gpu_ms":ms}),
                );
            }
        }
        thermal(&format!("{arm}-{on}"), "after");
    }
}

#[test]
#[ignore = "external model/Metal short-append witness; opt-in only"]
fn run_short_append_probe() {
    let _lease = crate::metal::acquire_metal_benchmark_lease().unwrap();
    let file = match std::env::var(PROBE_MODEL_ENV)
        .unwrap_or("27b".into())
        .as_str()
    {
        "27b" => "Qwen3.8-27B-Q4_K_M.gguf",
        "a3b" => "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
        _ => panic!("unknown model"),
    };
    let path = crate::test_fixtures::required_filename(file);
    let gg = GgufFile::open(path).unwrap();
    let model = Model::from_gguf(&gg).unwrap();
    let ctx = MetalContext::new().unwrap();
    let mode = std::env::var(PROBE_MODE_ENV).unwrap_or("trace".into());
    if mode == "dense" || mode == "q5" {
        dense(&ctx, &gg, mode == "q5");
        return;
    }
    let mm = MetalModel::load(&ctx, &gg, &model).unwrap();
    match mode.as_str() {
        "trace" => trace(&ctx, &mm),
        "expert" => expert(&ctx, &mm),
        "quality" => quality(&ctx, &mm, &gg, true),
        "quality-document" => quality(&ctx, &mm, &gg, false),
        _ => panic!("unknown mode"),
    }
}

fn moe_at(block: &crate::metal_forward::MetalBlock) -> Option<&crate::metal_forward::MetalMoeFfn> {
    match block {
        crate::metal_forward::MetalBlock::Gdn(b) => b.ffn_moe.as_ref(),
        crate::metal_forward::MetalBlock::Attn(b) => b.ffn_moe.as_ref(),
    }
}
fn copy_f32(ctx: &MetalContext, t: &MetalTensor, n: usize) -> MetalTensor {
    MetalTensor::from_bytes(
        ctx,
        bytemuck::cast_slice(&read_f32(t)[..n]),
        vec![n as u64],
        GgmlType::F32,
    )
    .unwrap()
}
fn packed_row(t: &MetalTensor, e: usize, col: usize) -> Vec<f32> {
    let (k, n) = (t.shape[0] as usize, t.shape[1] as usize);
    let (block, bytes) = t.dtype.storage_layout().unwrap();
    let stride = k / (block as usize) * (bytes as usize);
    let offset = (e * n + col) * stride;
    let raw = unsafe {
        std::slice::from_raw_parts(
            t.buffer
                .contents()
                .as_ptr()
                .cast::<u8>()
                .add(t.offset as usize + offset),
            stride,
        )
    };
    codec::dequant_to_f32(
        &TensorDesc {
            name: "expert-reference".into(),
            shape: vec![k as u64],
            dtype: t.dtype,
            shard_idx: 0,
            data_offset: 0,
            n_bytes: stride as u64,
        },
        raw,
    )
    .unwrap()
}
struct Expert<'a> {
    m: usize,
    role: bool,
    moe: &'a crate::metal_forward::MetalMoeFfn,
    x: MetalTensor,
    counts: MetalTensor,
    ids: MetalTensor,
    y: MetalTensor,
    points: Vec<usize>,
    reference: Vec<f64>,
}
impl Expert<'_> {
    fn run(&self, ctx: &MetalContext, on: bool) -> (f64, (f64, f64)) {
        unsafe {
            std::slice::from_raw_parts_mut(
                self.y.buffer.contents().as_ptr().cast::<f32>(),
                self.y.n_elements() as usize,
            )
            .fill(f32::NAN);
        }
        let tensor = on || self.m == 2048;
        let ms = command(ctx, |e| {
            if self.role {
                crate::metal_dflash::encode_prefill_moe_grouped_swiglu(
                    ctx,
                    e,
                    self.moe,
                    &self.x,
                    &self.counts,
                    &self.ids,
                    &self.y,
                    2048,
                    512,
                    256,
                    8,
                    self.m,
                    Some(48),
                    tensor,
                )
                .unwrap();
            } else {
                crate::metal_dflash::encode_prefill_moe_grouped_down(
                    ctx,
                    e,
                    &self.moe.down_exps,
                    &self.x,
                    &self.counts,
                    &self.ids,
                    &self.y,
                    512,
                    2048,
                    256,
                    self.m,
                    39,
                    tensor,
                )
                .unwrap();
            }
        });
        let data = read_f32(&self.y);
        assert!(data.iter().all(|v| v.is_finite()));
        (
            ms,
            metrics(
                &self.points.iter().map(|&p| data[p]).collect::<Vec<_>>(),
                &self.reference,
            ),
        )
    }
}
fn expert(ctx: &MetalContext, mm: &MetalModel) {
    let mf = MetalForward::new(ctx, mm);
    let mut fixtures = Vec::new();
    let ms = std::env::var(PROBE_ROWS_ENV).unwrap_or("512,513,1024,1300,2048".into());
    for m in ms.split(',').map(|s| s.parse::<usize>().unwrap()) {
        let mut sc = scratch(ctx, mm, m, m);
        let mut session = MetalSession::fresh(ctx, mm, m + 1).unwrap();
        run_scope(false, || {
            prefill_tokens_with_multi_hidden(
                &mf,
                &ids(m, mm.arch.vocab_size as usize),
                0,
                &mut session,
                &mut sc,
                &[],
                None,
            )
        })
        .unwrap();
        let counts = read_i32(&sc.moe_group_count_pack, 256);
        let bucket = read_i32(&sc.moe_group_ids_pack, 256 * m);
        assert_eq!(counts.iter().sum::<i32>() as usize, m * 8);
        let mut owner = vec![usize::MAX; m * 8];
        for (e, &count) in counts.iter().enumerate() {
            assert!((0..=m as i32).contains(&count));
            for j in 0..count as usize {
                let slot = bucket[e * m + j] as usize;
                assert!(slot < m * 8 && owner[slot] == usize::MAX);
                owner[slot] = e;
            }
        }
        assert!(owner.iter().all(|&e| e != usize::MAX));
        emit(json!({"record":"expert_distribution","m":m,"counts":counts}));
        let norm = copy_f32(ctx, &sc.h_pack, m * 2048);
        let inner = copy_f32(ctx, &sc.moe_group_inner_pack, m * 8 * 512);
        let ct =
            MetalTensor::from_bytes(ctx, bytemuck::cast_slice(&counts), vec![256], GgmlType::I32)
                .unwrap();
        let it = MetalTensor::from_bytes(
            ctx,
            bytemuck::cast_slice(&bucket),
            vec![(256 * m) as u64],
            GgmlType::I32,
        )
        .unwrap();
        for dtype in [GgmlType::Q5_K, GgmlType::Q6_K] {
            let moe = mm
                .blocks
                .iter()
                .filter_map(moe_at)
                .find(|q| q.down_exps.dtype == dtype)
                .unwrap();
            for role in [true, false] {
                let n = if role { 512 } else { 2048 };
                let input = if role { norm.clone() } else { inner.clone() };
                let points: Vec<_> = (0..32)
                    .flat_map(|r| {
                        (0..32).map(move |c| (r * (m * 8 - 1) / 31) * n + c * (n - 1) / 31)
                    })
                    .collect();
                let reference = points
                    .iter()
                    .map(|&p| {
                        let (slot, col) = (p / n, p % n);
                        let e = owner[slot];
                        if role {
                            let x = &read_f32(&input)[slot / 8 * 2048..(slot / 8 + 1) * 2048];
                            let g = packed_row(&moe.gate_exps, e, col);
                            let u = packed_row(&moe.up_exps, e, col);
                            let dot = |w: &[f32]| {
                                x.iter()
                                    .zip(w)
                                    .map(|(&x, &w)| f64::from(x) * f64::from(w))
                                    .sum::<f64>()
                            };
                            let (g, u) = (dot(&g), dot(&u));
                            g / (1. + (-g).exp()) * u
                        } else {
                            let x = &read_f32(&input)[slot * 512..(slot + 1) * 512];
                            let w = packed_row(&moe.down_exps, e, col);
                            x.iter()
                                .zip(&w)
                                .map(|(&x, &w)| f64::from(x) * f64::from(w))
                                .sum()
                        }
                    })
                    .collect();
                fixtures.push(Expert {
                    m,
                    role,
                    moe,
                    x: input,
                    counts: ct.clone(),
                    ids: it.clone(),
                    y: MetalTensor::zeros_f32(ctx, vec![(m * 8 * n) as u64]).unwrap(),
                    points,
                    reference,
                });
            }
        }
    }
    let mut limits = Vec::new();
    let mut passing = Vec::new();
    thermal("qualify", "before");
    for f in &fixtures {
        let (_, a) = f.run(ctx, false);
        let (_, b) = f.run(ctx, true);
        let pass = b.0 <= 1.5 * a.0 && b.1 <= 1.5 * a.1;
        limits.push(a);
        passing.push(pass);
        emit(
            json!({"record":"qualify_expert","m":f.m,"role":if f.role{"gateup"}else{"down"},"down_dtype":format!("{:?}",f.moe.down_exps.dtype),"base":a,"candidate":b,"ratios":[b.0/a.0,b.1/a.1],"pass":pass}),
        );
    }
    thermal("qualify", "after");
    for (arm, on) in [false, true, true, false].into_iter().enumerate() {
        std::thread::sleep(std::time::Duration::from_secs(30));
        thermal(&format!("{arm}-{on}"), "before");
        for f in &fixtures {
            f.run(ctx, on);
            f.run(ctx, on);
        }
        for sample in 0..6 {
            for (i, f) in fixtures.iter().enumerate() {
                let (ms, err) = f.run(ctx, on);
                let a = limits[i];
                emit(
                    json!({"record":"op_expert","arm":arm,"candidate":on,"sample":sample,"m":f.m,"role":if f.role{"gateup"}else{"down"},"down_dtype":format!("{:?}",f.moe.down_exps.dtype),"gpu_ms":ms,"qualified":passing[i],"errors":err,"ratios":[err.0/a.0,err.1/a.1]}),
                );
            }
        }
        thermal(&format!("{arm}-{on}"), "after");
    }
}

#[derive(Clone, Copy, Default)]
struct Score {
    n: usize,
    off: f64,
    on: f64,
    kl: f64,
    max_kl: f64,
    agree: usize,
}
impl Score {
    fn add(&mut self, s: Self) {
        self.n += s.n;
        self.off += s.off;
        self.on += s.on;
        self.kl += s.kl;
        self.max_kl = self.max_kl.max(s.max_kl);
        self.agree += s.agree;
    }
    fn report(&self) -> serde_json::Value {
        let n = self.n as f64;
        json!({"targets":self.n,"nll_off":self.off/n,"nll_on":self.on/n,"delta_nll":(self.on-self.off)/n,"mean_kl":self.kl/n,"max_kl":self.max_kl,"argmax_agree":self.agree,"pass":((self.on-self.off)/n).abs()<=0.005&&self.kl/n<=0.01})
    }
    fn check(&self) {
        assert!(
            ((self.on - self.off) / self.n as f64).abs() <= 0.005
                && self.kl / self.n as f64 <= 0.01,
            "quality gate: {}",
            self.report()
        );
    }
}
fn hash(data: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    format!("{:x}", Sha256::digest(data))
}
fn frozen(path: &str) -> String {
    let rev = std::env::var(PROBE_CORPUS_REV_ENV).unwrap_or_else(|_| "HEAD".into());
    let o = std::process::Command::new("git")
        .args(["show", &format!("{rev}:{path}")])
        .output()
        .unwrap();
    assert!(o.status.success());
    String::from_utf8(o.stdout).unwrap()
}
fn readout(
    ctx: &MetalContext,
    mm: &MetalModel,
    hidden: &MetalTensor,
    n: usize,
    endpoint: &[f32],
) -> MetalTensor {
    let h = mm.arch.hidden_size as usize;
    let v = mm.arch.vocab_size as usize;
    let norm = MetalTensor::zeros_f32(ctx, vec![h as u64]).unwrap();
    let out = MetalTensor::zeros_f32(ctx, vec![(n * v) as u64]).unwrap();
    unsafe {
        std::slice::from_raw_parts_mut(out.buffer.contents().as_ptr().cast::<f32>(), n * v)
            .fill(f32::NAN);
    }
    command(ctx, |e| {
        for row in 0..n {
            let x = hidden.view_subrange((row * h) as u64, vec![h as u64]);
            let y = out.view_subrange((row * v) as u64, vec![v as u64]);
            encode_rms_norm_mul_f32(
                ctx,
                e,
                &x,
                &mm.output_norm,
                &norm,
                crate::metal_forward::RMS_EPS,
            )
            .unwrap();
            crate::metal_forward::encode_mat_vec_dispatch(ctx, e, &mm.lm_head, &norm, &y, h, v)
                .unwrap();
        }
    });
    let values = read_f32(&out);
    assert!(values.iter().all(|v| v.is_finite()));
    let last = &values[(n - 1) * v..];
    let peak = crate::compare::assert_max_abs_diff_f32(
        "short-append native endpoint readout",
        last,
        endpoint,
    );
    emit(json!({"record":"native_readout_witness","rows":n,"endpoint_max_abs":peak}));
    assert!(peak <= 1e-5, "native readout mismatch");
    out
}
fn compare_distribution(a: &[f32], b: &[f32], targets: &[i32], v: usize) -> Score {
    let mut s = Score::default();
    for (row, &target) in targets.iter().enumerate() {
        let x = &a[row * v..(row + 1) * v];
        let y = &b[row * v..(row + 1) * v];
        let argmax = |z: &[f32]| {
            z.iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .unwrap()
                .0
        };
        assert!(x.iter().chain(y).all(|v| v.is_finite()));
        let (mx, my) = (f64::from(x[argmax(x)]), f64::from(y[argmax(y)]));
        let zx = x.iter().map(|&a| (f64::from(a) - mx).exp()).sum::<f64>();
        let zy = y.iter().map(|&a| (f64::from(a) - my).exp()).sum::<f64>();
        let (lx, ly) = (mx + zx.ln(), my + zy.ln());
        let kl = x
            .iter()
            .zip(y)
            .map(|(&a, &b)| (f64::from(a) - mx).exp() / zx * (f64::from(a) - f64::from(b)))
            .sum::<f64>()
            + ly
            - lx;
        assert!(kl >= -1e-7 && kl.is_finite());
        let kl = kl.max(0.);
        s.n += 1;
        s.off += lx - f64::from(x[target as usize]);
        s.on += ly - f64::from(y[target as usize]);
        s.kl += kl;
        s.max_kl = s.max_kl.max(kl);
        s.agree += usize::from(argmax(x) == argmax(y));
    }
    s
}
fn quality(ctx: &MetalContext, mm: &MetalModel, gg: &GgufFile, per_slice_gate: bool) {
    let selected_doc = std::env::var(PROBE_DOC_ENV).ok();
    assert!(
        selected_doc
            .as_deref()
            .is_none_or(|d| matches!(d, "code" | "prose"))
    );
    let tok = crate::tokenizer::Tokenizer::from_gguf(gg).unwrap();
    let mf = MetalForward::new(ctx, mm);
    let h = mm.arch.hidden_size as usize;
    let v = mm.arch.vocab_size as usize;
    let mut pooled = Score::default();
    for (doc, prefix_path, target_path) in [
        (
            "prose",
            "docs/PERF-LOG.md",
            "docs/APPLE-GPU-OPTIMIZATION.md",
        ),
        (
            "code",
            "crates/qwen-llm/src/metal_forward/mod.rs",
            "crates/qwen-llm/src/metal_dflash.rs",
        ),
    ] {
        if selected_doc.as_deref().is_some_and(|d| d != doc) {
            continue;
        }
        let prefix_text = frozen(prefix_path);
        let text = frozen(target_path);
        let prefix = tok.encode(&prefix_text, false).unwrap();
        let targets = tok.encode(&text, false).unwrap();
        let lengths = [512usize, 1024, 1300];
        let total: usize = lengths.iter().sum();
        assert!(targets.len() > total && !prefix.is_empty());
        let mut document = Score::default();
        emit(
            json!({"record":"corpus","doc":doc,"prefix_sha256":hash(prefix_text.as_bytes()),"target_sha256":hash(text.as_bytes()),"teacher_ids_sha256":hash(bytemuck::cast_slice(&targets[..total+1])),"targets":total}),
        );
        for depth in [0usize, 32768] {
            let extent = depth + total + 1;
            let mut off = MetalSession::fresh(ctx, mm, extent).unwrap();
            let mut on = MetalSession::fresh(ctx, mm, extent).unwrap();
            if depth > 0 {
                let input: Vec<_> = prefix.iter().copied().cycle().take(depth).collect();
                for session in [&mut off, &mut on] {
                    let mut sc = scratch(ctx, mm, depth, depth);
                    run_scope(false, || {
                        prefill_tokens_with_multi_hidden(
                            &mf,
                            &input,
                            0,
                            session,
                            &mut sc,
                            &[],
                            None,
                        )
                    })
                    .unwrap();
                }
            }
            let mut consumed = 0;
            for n in lengths {
                let score = objc2::rc::autoreleasepool(|_| {
                    let mut outputs = Vec::new();
                    for (candidate, session) in [(false, &mut off), (true, &mut on)] {
                        let mut sc = scratch(ctx, mm, n, depth + consumed + n);
                        let hidden = MetalTensor::zeros_f32(ctx, vec![(n * h) as u64]).unwrap();
                        dispatch_census_begin();
                        let endpoint = run_scope(candidate, || {
                            prefill_tokens_with_multi_hidden(
                                &mf,
                                &targets[consumed..consumed + n],
                                (depth + consumed) as u32,
                                session,
                                &mut sc,
                                &[mm.arch.n_layer - 1],
                                Some(&hidden),
                            )
                        })
                        .unwrap();
                        let launches = census(dispatch_census_take());
                        emit(
                            json!({"record":"quality_dispatches","doc":doc,"depth":depth,"position":depth+consumed,"rows":n,"candidate":candidate,"dispatches":launches}),
                        );
                        outputs.push(readout(ctx, mm, &hidden, n, &endpoint));
                    }
                    compare_distribution(
                        read_f32(&outputs[0]),
                        read_f32(&outputs[1]),
                        &targets[consumed + 1..consumed + n + 1],
                        v,
                    )
                });
                emit(
                    json!({"record":"quality_case","doc":doc,"depth":depth,"position":depth+consumed,"rows":n,"gate":if per_slice_gate{"slice"}else{"informational"},"score":score.report()}),
                );
                if per_slice_gate {
                    score.check();
                }
                document.add(score);
                consumed += n;
            }
        }
        emit(
            json!({"record":"quality_document","doc":doc,"gate":"document","score":document.report()}),
        );
        document.check();
        pooled.add(document);
    }
    emit(json!({"record":"quality_pooled","gate":"pooled","score":pooled.report()}));
    pooled.check();
}
