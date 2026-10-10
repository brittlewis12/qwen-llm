//! CPU-only dot references over retained native BF16 words and captured inputs.
use super::*;
use crate::qwen4exp_metal::{GatedResidualMetalReadWeights, frontier_hc};
use objc2_metal::{MTLResource, MTLStorageMode};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum HcMode {
    Off,
    Production,
    F32DownUp,
}

impl HcMode {
    pub(super) fn parse(value: Option<&str>) -> Result<Self, &'static str> {
        match value {
            None | Some("off") => Ok(Self::Off),
            Some("production") => Ok(Self::Production),
            Some("f32downup") => Ok(Self::F32DownUp),
            _ => Err("FLASH_FRONTIER_LAYER0_HC must be off, production or f32downup"),
        }
    }
    pub(super) fn label(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Production => "production",
            Self::F32DownUp => "f32downup",
        }
    }
    fn policy(self) -> Option<frontier_hc::Policy> {
        match self {
            Self::Off => None,
            Self::Production => Some(frontier_hc::Policy::Production),
            Self::F32DownUp => Some(frontier_hc::Policy::F32DownUp),
        }
    }
}

pub(super) fn scoped<R>(
    probe: Option<frontier_hc::Probe>,
    mode: HcMode,
    candidate: bool,
    capture: bool,
    work: impl FnOnce() -> R,
) -> (R, Option<frontier_hc::Probe>) {
    match (probe, mode.policy()) {
        (Some(p), Some(policy)) => {
            let (r, p) = frontier_hc::with_frontier_hc_probe(p, policy, candidate, capture, work);
            (r, Some(p))
        }
        (None, None) => (work(), None),
        _ => panic!("HC mode/bank mismatch"),
    }
}

pub(super) fn witness(
    probe: &frontier_hc::Probe,
    rows: &[DispatchCensusRow],
    candidate: bool,
    label: &str,
    out: &mut std::fs::File,
) -> PacketResult<()> {
    let targeted: Vec<_> = rows
        .iter()
        .filter(|r| {
            r.tag
                .as_deref()
                .is_some_and(|t| t.starts_with(frontier_hc::PROJECTION_TAG))
        })
        .collect();
    let expected = if candidate { 2 } else { 4 };
    let mut valid = probe.records.len() == expected && targeted.len() == expected;
    let records:Vec<_>=probe.records.iter().zip(&targeted).map(|(r,k)| {
        let tag=format!("{}{}.absolute{}.N{}",frontier_hc::PROJECTION_TAG,r.role.label(),r.start,r.tokens);
        let expected_kernel=if r.policy==frontier_hc::Policy::F32DownUp || r.tokens==3 {"kernel_mat_mat_bf16_f32"} else {"kernel_mat_mat_bf16_bfloat_act_f32"};
        valid &= k.tag.as_deref()==Some(tag.as_str()) && k.kernel==expected_kernel;
        json!({"start":r.start,"tokens":r.tokens,"role":r.role.label(),"policy":format!("{:?}",r.policy),
            "kernel":k.kernel,"expected_kernel":expected_kernel,"grid":[k.grid_width,k.grid_height,k.grid_depth],
            "threads":[k.threads_width,k.threads_height,k.threads_depth]})
    }).collect();
    let other_bfloat = rows
        .iter()
        .filter(|r| {
            r.kernel == "kernel_mat_mat_bf16_bfloat_act_f32"
                && !r
                    .tag
                    .as_deref()
                    .is_some_and(|t| t.starts_with(frontier_hc::PROJECTION_TAG))
        })
        .count();
    emit(
        out,
        json!({"event":"hc_target_witness","label":label,"expected_calls":expected,"records":records,
        "other_bfloat_calls":other_bfloat,"valid":valid && other_bfloat>0}),
    );
    require(
        valid && other_bfloat > 0,
        "HC target policy/count mismatch or other bfloat projections disabled",
    )
}

fn output_path(suffix: &str) -> PacketResult<std::path::PathBuf> {
    let mut p = std::env::var_os("FLASH_PREFILL_OUT").ok_or("missing output path")?;
    p.push(suffix);
    Ok(p.into())
}

fn round_bf16(x: f32) -> f32 {
    let bits = x.to_bits();
    if bits & 0x7f80_0000 == 0x7f80_0000 {
        return x;
    }
    f32::from_bits(bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) & 0xffff_0000)
}

fn decode_word(bytes: &[u8]) -> f64 {
    f64::from(f32::from_bits(
        u32::from(u16::from_le_bytes([bytes[0], bytes[1]])) << 16,
    ))
}

fn weight_range(offset: u64, bytes: u64, physical: u64) -> PacketResult<std::ops::Range<usize>> {
    let end = offset
        .checked_add(bytes)
        .ok_or("HC weight range overflow")?;
    require(end <= physical, "HC weight outside physical buffer")?;
    Ok(usize::try_from(offset)?..usize::try_from(end)?)
}

fn dots(weights: &[u8], input: &[f32], k: usize, m: usize) -> [Vec<f64>; 2] {
    assert_eq!(weights.len(), k * m * 2);
    assert_eq!(input.len() % k, 0);
    let n = input.len() / k;
    let rounded: Vec<_> = input.iter().copied().map(round_bf16).collect();
    let mut original = vec![0.0f64; n * m];
    let mut bf16 = vec![0.0f64; n * m];
    for t in 0..n {
        for row in 0..m {
            let (mut a, mut b) = (0.0, 0.0);
            for j in 0..k {
                let offset = 2 * (row * k + j);
                let w = decode_word(&weights[offset..offset + 2]);
                a += w * f64::from(input[t * k + j]);
                b += w * f64::from(rounded[t * k + j]);
            }
            original[t * m + row] = a;
            bf16[t * m + row] = b;
        }
    }
    [original, bf16]
}

fn error(a: &[f64], b: &[f64]) -> Value {
    assert_eq!(a.len(), b.len());
    let (mut e, mut n, mut max) = (0.0f64, 0.0f64, 0.0f64);
    for (&a, &b) in a.iter().zip(b) {
        let d = a - b;
        e += d * d;
        n += b * b;
        max = max.max(d.abs());
    }
    json!({"rms_error":(e/a.len().max(1) as f64).sqrt(),"max_abs":max,"relative_l2":(n>0.0).then(||(e/n).sqrt())})
}

pub(super) struct Oracle {
    down: Vec<u8>,
    up: Vec<u8>,
}

impl Oracle {
    pub(super) const WEIGHT_CPU_BYTES: u64 =
        2 * (frontier_hc::HYPER * frontier_hc::LOW * 2) as u64 + (frontier_hc::HYPER * 4) as u64;

    /// Call while the runner is idle, after combined capture/CPU admission.
    pub(super) fn new(
        w: GatedResidualMetalReadWeights<'_>,
        out: &mut std::fs::File,
    ) -> PacketResult<Self> {
        let path = output_path(".hc.weights.bin")?;
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)?;
        let mut offset = 0u64;
        let mut retained = Vec::new();
        for (name, t, dtype, shape) in [
            ("down", w.down, GgmlType::BF16, vec![10240, 320]),
            ("up", w.up, GgmlType::BF16, vec![320, 10240]),
            ("norm", w.norm, GgmlType::F32, vec![10240]),
        ] {
            require(
                t.dtype == dtype && t.shape == shape,
                "HC oracle weight shape/dtype mismatch",
            )?;
            require(
                t.buffer.storageMode() == MTLStorageMode::Shared,
                "HC oracle requires CPU-readable shared weights",
            )?;
            let range = weight_range(t.offset, t.n_bytes(), t.buffer.length() as u64)?;
            // Native read-only weights; no command is in flight during setup.
            let bytes = unsafe {
                std::slice::from_raw_parts(
                    t.buffer.contents().as_ptr().cast::<u8>().add(range.start),
                    range.len(),
                )
            };
            let finite = if dtype == GgmlType::BF16 {
                bytes.chunks_exact(2).all(|b| decode_word(b).is_finite())
            } else {
                bytes
                    .chunks_exact(4)
                    .all(|b| f32::from_le_bytes(b.try_into().unwrap()).is_finite())
            };
            file.write_all(bytes)?;
            emit(
                out,
                json!({"event":"hc_oracle_weight","name":format!("blk.0.hc_attn_{name}.weight"),"dtype":format!("{dtype:?}"),
                "shape":shape,"resident_offset":t.offset,"path":path,"byte_offset":offset,"bytes":bytes.len(),
                "sha256":sha256_bytes(bytes),"all_finite":finite,"encoding":"native little-endian BF16 words; norm is F32"}),
            );
            require(finite, "nonfinite HC oracle weight")?;
            offset += bytes.len() as u64;
            if name != "norm" {
                retained.push(bytes.to_vec());
            }
        }
        file.sync_all()?;
        emit(
            out,
            json!({"event":"hc_weights_complete","path":path,"bytes":offset,"sha256":sha256_file(&path)}),
        );
        let up = retained.pop().unwrap();
        let down = retained.pop().unwrap();
        Ok(Self { down, up })
    }

    pub(super) fn retain(
        &self,
        probe: &frontier_hc::Probe,
        label: &str,
        reference: Option<&[Vec<f32>]>,
        out: &mut std::fs::File,
    ) -> PacketResult<Vec<Vec<f32>>> {
        let path = output_path(&format!(".{label}.hc.f32le"))?;
        let file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)?;
        let mut file = std::io::BufWriter::with_capacity(64 << 10, file);
        let mut values = Vec::new();
        let mut offset = 0u64;
        let mut finite = true;
        for (i, (name, t)) in probe.buffers.iter().enumerate() {
            // Both ordinary commands, including capture copies, have completed.
            let v = read_f32_tensor(t);
            let mut hash = Sha256::new();
            for &x in &v {
                let b = x.to_bits().to_le_bytes();
                file.write_all(&b)?;
                hash.update(b);
            }
            let all_finite = v.iter().all(|v| v.is_finite());
            finite &= all_finite;
            emit(
                out,
                json!({"event":"hc_capture_tensor","schedule":label,"name":name,"shape":t.shape,
                "absolute_range_half_open":[2048,2056],"path":path,"byte_offset":offset,"bytes":v.len()*4,
                "sha256":format!("{:x}",hash.finalize()),"all_finite":all_finite,
                "stage":match *name {"hc.hyper"=>"raw hyper input before per-branch RMS norm","hc.normalized"=>"actual down input",
                    "hc.down"=>"down output before in-place SiLU(x/4)","hc.low"=>"actual up input after SiLU(x/4)",_=>"up output before sigmoid/gated mean"}}),
            );
            offset += (v.len() * 4) as u64;
            if let Some(a) = reference {
                if all_finite {
                    let width = v.len() / 8;
                    emit(
                        out,
                        json!({"event":"hc_cross_schedule","name":name,"aggregate":metrics(&a[i],&v),
                    "rows":(0..8).map(|t|json!({"position":2048+t,"metrics":metrics(&a[i][t*width..(t+1)*width],&v[t*width..(t+1)*width])})).collect::<Vec<_>>(),
                    "interpretation":"input differences reported separately; up uses each arm's own captured activated low input"}),
                    );
                }
            }
            values.push(v);
        }
        file.flush()?;
        file.get_ref().sync_all()?;
        emit(
            out,
            json!({"event":"hc_capture_complete","schedule":label,"path":path,"bytes":offset,"sha256":sha256_file(&path),"all_finite":finite}),
        );
        require(finite, "nonfinite HC capture retained")?;
        let oracle_path = output_path(&format!(".{label}.hc.oracle.f64le"))?;
        let oracle_file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&oracle_path)?;
        let mut oracle_file = std::io::BufWriter::with_capacity(64 << 10, oracle_file);
        let mut oracle_offset = 0u64;
        for (role, w, x, y, k, m) in [
            ("down", &self.down, &values[1], &values[2], 10240, 320),
            ("up", &self.up, &values[3], &values[4], 320, 10240),
        ] {
            let refs = dots(w, x, k, m);
            let actual: Vec<_> = y.iter().copied().map(f64::from).collect();
            for (kind, r) in ["original_f32_activation", "rne_bf16_activation"]
                .into_iter()
                .zip(&refs)
            {
                let finite = r.iter().all(|v| v.is_finite());
                let mut hash = Sha256::new();
                for &value in r {
                    let b = value.to_le_bytes();
                    oracle_file.write_all(&b)?;
                    hash.update(b);
                }
                emit(
                    out,
                    json!({"event":"hc_f64_oracle","schedule":label,"role":role,"activation":kind,
                    "path":oracle_path,"byte_offset":oracle_offset,"bytes":r.len()*8,"shape":[m,8],
                    "sha256":format!("{:x}",hash.finalize()),"all_finite":finite,
                    "gpu_error":error(&actual,r),"rows":(0..8).map(|t|json!({"position":2048+t,"gpu_error":error(&actual[t*m..(t+1)*m],&r[t*m..(t+1)*m])})).collect::<Vec<_>>(),
                    "contract":"f64 sum of actual BF16 weight times this arm's captured projection input; no GPU bit-equivalence or quality gate"}),
                );
                oracle_offset += (r.len() * 8) as u64;
                require(finite, "nonfinite HC f64 oracle")?;
            }
            emit(
                out,
                json!({"event":"hc_rounding_reference_delta","schedule":label,"role":role,"metrics":error(&refs[1],&refs[0]),
                "interpretation":"same captured input in both references; not a recurrent-state or decoder-correctness verdict"}),
            );
        }
        oracle_file.flush()?;
        oracle_file.get_ref().sync_all()?;
        emit(
            out,
            json!({"event":"hc_oracle_complete","schedule":label,"path":oracle_path,"bytes":oracle_offset,"sha256":sha256_file(&oracle_path)}),
        );
        Ok(values)
    }
}

#[test]
fn frontier_hc_rne_known_bit_vectors() {
    for (a, b) in [
        (0, 0),
        (0x80000000, 0x80000000),
        (0x3f808000, 0x3f800000),
        (0x3f818000, 0x3f820000),
        (0xbf808000, 0xbf800000),
        (0xbf818000, 0xbf820000),
        (0x00008000, 0),
        (0x00018000, 0x00020000),
        (0x3f807fff, 0x3f800000),
        (0x3f808001, 0x3f810000),
    ] {
        assert_eq!(round_bf16(f32::from_bits(a)).to_bits(), b);
    }
}

#[test]
fn frontier_hc_f64_oracle_known_weights_and_activations() {
    let weights = [0x80, 0x3f, 0x00, 0xc0]; // exactly +1, -2, not a GPU decoder
    assert_eq!(
        dots(&weights, &[1.00390625, 2.0], 2, 1),
        [vec![-2.99609375], vec![-3.0]]
    );
    assert_eq!(HcMode::parse(None), Ok(HcMode::Off));
    assert_eq!(HcMode::parse(Some("off")), Ok(HcMode::Off));
    assert_eq!(HcMode::parse(Some("production")), Ok(HcMode::Production));
    assert_eq!(HcMode::parse(Some("f32downup")), Ok(HcMode::F32DownUp));
    assert!(HcMode::parse(Some("f32")).is_err());
}

#[test]
fn frontier_hc_native_weight_physical_bounds() {
    assert_eq!(weight_range(64, 66, 130).unwrap(), 64..130);
    assert_eq!(weight_range(64, 66, 4096).unwrap(), 64..130);
    assert!(weight_range(64, 66, 129).is_err());
    assert!(weight_range(u64::MAX, 2, u64::MAX).is_err());
}
