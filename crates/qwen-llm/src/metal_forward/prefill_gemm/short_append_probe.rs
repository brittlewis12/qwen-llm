//! Explicit model/Metal witness for short-append dispatch and row screens.
//! Never part of CPU acceptance or the default library test run.
use super::*;
use crate::metal_forward::{MetalForward, MetalModel, MetalSession};
use crate::{codec, gguf::GgufFile, loader::Model, metal::*, metal_dflash::*, tensor::TensorDesc};
use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLCommandQueue};
use serde_json::json;
use std::{cell::Cell, collections::BTreeMap};

thread_local! { static OVERRIDE: Cell<Option<bool>> = const { Cell::new(None) }; }
pub(super) fn override_value() -> Option<bool> {
    OVERRIDE.with(Cell::get)
}
struct Scope(Option<bool>);
impl Scope {
    fn new(on: bool) -> Self {
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
fn geometry(moe: bool, append: usize, total: usize) -> (usize, PrefillScratchConfig) {
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
        let bytes = if d.dtype == GgmlType::Q4_K { 144 } else { 210 };
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
fn dense(ctx: &MetalContext, gg: &GgufFile) {
    let ms = std::env::var("QWEN_SHORT_PROBE_ROWS")
        .unwrap_or("128,256,276,384,512,768,1000,1023,1024,1300".into());
    let mut fixtures = Vec::new();
    for m in ms.split(',').map(|s| s.parse::<usize>().unwrap()) {
        for name in [
            "blk.0.ffn_gate.weight",
            "blk.8.ffn_down.weight",
            "blk.0.ffn_down.weight",
        ] {
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
    let file = match std::env::var("QWEN_SHORT_PROBE_MODEL")
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
    let mode = std::env::var("QWEN_SHORT_PROBE_MODE").unwrap_or("trace".into());
    if mode == "dense" {
        dense(&ctx, &gg);
        return;
    }
    let mm = MetalModel::load(&ctx, &gg, &model).unwrap();
    match mode.as_str() {
        "trace" => trace(&ctx, &mm),
        _ => panic!("unknown mode"),
    }
}
