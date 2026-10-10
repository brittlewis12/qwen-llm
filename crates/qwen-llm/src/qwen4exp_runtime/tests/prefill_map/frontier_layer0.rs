//! Untimed layer-zero attribution with unchanged ordinary suffix command widths.
//! FLASH_FRONTIER_LAYER0_HC=off (default), production, or f32downup. Enabled
//! modes retain HC inputs/outputs and native weights and run bounded CPU oracles;
//! they require FLASH_FRONTIER_LAYER0_BF16_ACT=production.
use super::*;
use crate::qwen4exp_gdn::GatedDeltaNetMetalWeights;
use crate::qwen4exp_gdn::frontier_capture::{
    COPY_TAG, FrontierGdnCapture, GDN_TAG, with_frontier_gdn_capture,
};
use crate::qwen4exp_metal::frontier_hc;
#[path = "frontier_hc_oracle.rs"]
mod hc_oracle;
use hc_oracle::{HcMode, Oracle};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Bf16Act {
    Production,
    F32,
}

impl Bf16Act {
    fn parse(value: Option<&str>) -> Result<Self, &'static str> {
        match value {
            None | Some("production") => Ok(Self::Production),
            Some("f32") => Ok(Self::F32),
            _ => Err("FLASH_FRONTIER_LAYER0_BF16_ACT must be production or f32"),
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Production => "production",
            Self::F32 => "f32",
        }
    }

    fn suffix<R>(self, work: impl FnOnce() -> R) -> R {
        match self {
            Self::Production => work(),
            Self::F32 => crate::metal_forward::with_matmat_bf16_bfloat_act_override(false, work),
        }
    }
}

fn bf16_witness(
    out: &mut std::fs::File,
    label: &str,
    mode: Bf16Act,
    rows: &[DispatchCensusRow],
) -> PacketResult<()> {
    let bfloat = rows
        .iter()
        .filter(|r| r.kernel == "kernel_mat_mat_bf16_bfloat_act_f32")
        .count();
    let f32 = rows
        .iter()
        .filter(|r| r.kernel == "kernel_mat_mat_bf16_f32")
        .count();
    emit(
        out,
        json!({"event":"layer0_bf16_dispatch_witness","label":label,"mode":mode.label(),
        "bfloat_activation_kernel_count":bfloat,"f32_activation_kernel_count":f32,
        "uniform_f32_valid":(mode==Bf16Act::F32).then_some(bfloat==0)}),
    );
    require(
        mode != Bf16Act::F32 || bfloat == 0,
        "uniform F32 suffix dispatched BF16 activation kernel",
    )
}

fn allocation(
    r: &Qwen4ExpTextRunner<'_, '_, '_>,
    out: &mut std::fs::File,
    hc_mode: HcMode,
) -> PacketResult<(FrontierGdnCapture, Option<frontier_hc::Probe>)> {
    let weights = r.weights.zero_one.layer_zero.gdn;
    let mut specs = FrontierGdnCapture::specs(weights.geometry);
    let hc_enabled = hc_mode != HcMode::Off;
    if hc_enabled {
        specs.extend(frontier_hc::Probe::specs());
    }
    let mut logical = 0u64;
    let mut priced = 0u64;
    let mut rows = Vec::new();
    for (name, shape) in &specs {
        let bytes = shape
            .iter()
            .try_fold(4u64, |n, &d| n.checked_mul(d))
            .ok_or("capture bytes overflow")?;
        let price = r.ctx.price_shared_buffer_upper(bytes)?;
        logical = logical.checked_add(bytes).ok_or("capture sum overflow")?;
        priced = priced
            .checked_add(price.priced_upper_bytes)
            .ok_or("capture price overflow")?;
        rows.push(json!({"name":name,"shape":shape,"logical_bytes":bytes,
            "priced_upper_bytes":price.priced_upper_bytes,"alignment":price.alignment}));
    }
    let checkpoint_bytes = r
        .workspace
        .persistent_state_tensors()
        .iter()
        .map(|t| t.n_bytes())
        .sum::<u64>()
        + (r.weights.geometry.hyper_width() as u64 + r.weights.geometry.vocab_size() as u64) * 4;
    // Retain A, at most one B bank's worth of CPU values, endpoints/census and
    // serialization overhead. The existing runner gate already reserves the
    // checkpoint; include it again in this combined check before either exists.
    let cpu_bytes = checkpoint_bytes
        .checked_add(logical.checked_mul(2).ok_or("CPU capture overflow")?)
        .and_then(|b| {
            b.checked_add(if hc_enabled {
                Oracle::WEIGHT_CPU_BYTES + (4 << 20)
            } else {
                0
            })
        })
        .and_then(|b| b.checked_add(MARGIN))
        .ok_or("CPU bound overflow")?;
    require(
        cpu_bytes <= 512 << 20,
        "layer-zero CPU retention exceeds 512 MiB bound",
    )?;
    let _transaction = r.ctx.begin_allocation_transaction();
    let before = r.ctx.memory_signals();
    let admission = crate::metal::evaluate_metal_memory_admission_with_cpu_bytes(
        priced,
        cpu_bytes,
        crate::qwen4exp_text_session::QWEN4EXP_TEXT_SESSION_DYNAMIC_RESERVE_BYTES,
        before,
        true,
    );
    emit(
        out,
        json!({"event":"layer0_capture_admission","buffers":rows,
        "logical_bytes":logical,"priced_upper_bytes":priced,"cpu_upper_bytes":cpu_bytes,
        "checkpoint_bytes":checkpoint_bytes,"cpu_capture_copies":2,"cpu_other_margin":MARGIN,
        "hc_weight_retention_bytes":if hc_enabled {Oracle::WEIGHT_CPU_BYTES} else {0},
        "hc_oracle_working_bytes":if hc_enabled {4<<20} else {0},
        "admitted":admission.admitted,"admission":format!("{admission:?}")}),
    );
    require(
        admission.admitted,
        "capture admission refused; no diagnostic execution",
    )?;
    let bank = FrontierGdnCapture::new(r.ctx, weights)?;
    let hc = if hc_enabled {
        Some(frontier_hc::Probe::new(
            r.ctx,
            r.weights.zero_one.layer_zero.attention_residual.read,
        )?)
    } else {
        None
    };
    let observed = r
        .ctx
        .memory_signals()
        .current_allocated_bytes
        .saturating_sub(before.current_allocated_bytes);
    emit(
        out,
        json!({"event":"layer0_capture_allocation","observed_bytes":observed,"priced_upper_bytes":priced}),
    );
    require(observed <= priced, "capture allocation exceeded price")?;
    Ok((bank, hc))
}

struct Observation {
    endpoint: Endpoint,
    census: Vec<DispatchCensusRow>,
}

fn suffix(
    r: &mut Qwen4ExpTextRunner<'_, '_, '_>,
    tokens: &[u32],
    candidate: bool,
    bf16_act: Bf16Act,
    label: &str,
    out: &mut std::fs::File,
) -> PacketResult<Observation> {
    with_qwen4exp_frontier_schedule(candidate, || {
        require(
            r.next_position() == PREFIX,
            "probe must start at position2048",
        )?;
        let plan = normal_plan(r, END - PREFIX)?;
        require(
            plan.packed_ranges == ranges(PREFIX, candidate) && plan.scalar_start == END,
            "probe changed ordinary command widths",
        )?;
        require(
            !crate::metal::dispatch_census_is_active(),
            "external census active",
        )?;
        crate::metal::dispatch_census_begin();
        let result = bf16_act.suffix(|| {
            r.prefill_continuation_with_command_checkpoint(&tokens[PREFIX..END], || Ok(()))
                .map(|_| ())
        });
        let census = crate::metal::dispatch_census_take();
        result?;
        bf16_witness(out, label, bf16_act, &census)?;
        let timing = r.last_prefill_timing().ok_or("missing suffix timing")?;
        require(
            r.next_position() == END
                && timing.command_count == plan.packed_ranges.len()
                && timing.packed_token_count == END - PREFIX
                && timing.contains_selection,
            "suffix publication or command count mismatch",
        )?;
        witness(out, label, &plan, &census)?;
        let endpoint = endpoint(r, Some(if candidate { 2048 } else { 2045 }))?;
        emit_endpoint(out, label, &endpoint);
        require(
            !endpoint.logits.is_empty() && endpoint.logits.iter().all(|v| v.is_finite()),
            "nonfinite endpoint",
        )?;
        emit(
            out,
            json!({"event":"layer0_suffix_complete","label":label,"timing_eligible":false,
            "absolute_ranges":plan.packed_ranges.iter().map(|r|[r.start,r.end]).collect::<Vec<_>>(),
            "command_count":timing.command_count,"position":r.next_position()}),
        );
        Ok(Observation { endpoint, census })
    })
}

fn is_copy(row: &DispatchCensusRow) -> bool {
    row.tag
        .as_deref()
        .is_some_and(|t| t.starts_with(COPY_TAG) || t.starts_with(frontier_hc::COPY_TAG))
}

fn same_dispatch(a: &DispatchCensusRow, b: &DispatchCensusRow) -> bool {
    a.family == b.family
        && a.kernel == b.kernel
        && a.encoder_ordinal == b.encoder_ordinal
        && a.encoder_concurrent == b.encoder_concurrent
        && (
            a.grid_width,
            a.grid_height,
            a.grid_depth,
            a.threads_width,
            a.threads_height,
            a.threads_depth,
            a.grid_tgs,
            a.tg_threads,
        ) == (
            b.grid_width,
            b.grid_height,
            b.grid_depth,
            b.threads_width,
            b.threads_height,
            b.threads_depth,
            b.grid_tgs,
            b.tg_threads,
        )
}

fn concordance(
    out: &mut std::fs::File,
    label: &str,
    a: &Observation,
    b: &Observation,
    candidate: bool,
    hc_enabled: bool,
) -> PacketResult<()> {
    let bits_equal = a.endpoint.logits.len() == b.endpoint.logits.len()
        && a.endpoint
            .logits
            .iter()
            .zip(&b.endpoint.logits)
            .all(|(a, b)| a.to_bits() == b.to_bits());
    let state_equal = a.endpoint.state == b.endpoint.state;
    let original: Vec<_> = b.census.iter().filter(|r| !is_copy(r)).collect();
    let census_equal = a.census.len() == original.len()
        && a.census
            .iter()
            .zip(original)
            .all(|(a, b)| same_dispatch(a, b));
    let copies: Vec<_> = b.census.iter().filter(|r| is_copy(r)).collect();
    let copies_valid = copies.len()
        == 2 + (14 + if hc_enabled { 5 } else { 0 }) * if candidate { 1 } else { 2 }
        && copies.iter().all(|r| r.kernel == "kernel_copy_offset_f32");
    emit(
        out,
        json!({"event":"layer0_observer_concordance","schedule":label,
        "endpoint_bits_equal":bits_equal,"persistent_and_hyper_state_equal":state_equal,
        "original_dispatch_sequence_equal":census_equal,"capture_copies_valid":copies_valid,
        "copy_dispatches":copies.len(),"comparison":comparison(&a.endpoint,&b.endpoint),
        "contract":"exact within each schedule only; checkpoint recurrence keeps kernel/grid; capture copies excluded by dedicated tag"}),
    );
    require(
        bits_equal && state_equal && census_equal && copies_valid,
        "layer-zero observer changed its schedule's result or dispatches",
    )
}

fn metrics(a: &[f32], b: &[f32]) -> Value {
    assert_eq!(a.len(), b.len());
    let mut error = 0.0f64;
    let mut norm = 0.0f64;
    let mut maximum = 0.0f64;
    let mut different = 0usize;
    for (&a, &b) in a.iter().zip(b) {
        let d = f64::from(a) - f64::from(b);
        error += d * d;
        norm += f64::from(a).powi(2);
        maximum = maximum.max(d.abs());
        different += usize::from(a.to_bits() != b.to_bits());
    }
    json!({"elements":a.len(),"different_bits":different,"max_abs":maximum,
        "rms_error":(error/a.len().max(1) as f64).sqrt(),
        "relative_l2":(norm>0.0).then(|| (error/norm).sqrt()),
        "relative_l2_defined":norm>0.0})
}

fn retain(
    bank: &FrontierGdnCapture,
    label: &str,
    out: &mut std::fs::File,
    reference: Option<&[Vec<f32>]>,
) -> PacketResult<Vec<Vec<f32>>> {
    require(
        bank.complete,
        "layer-zero capture did not see the expected commands",
    )?;
    let mut path = std::env::var_os("FLASH_PREFILL_OUT").ok_or("missing output path")?;
    path.push(format!(".{label}.f32le"));
    let path = std::path::PathBuf::from(path);
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&path)?;
    emit(
        out,
        json!({"event":"layer0_binary_begin","schedule":label,"path":path,
        "format":"contiguous IEEE754 F32 little endian, tensor sections described below"}),
    );
    let mut offset = 0u64;
    let mut all_finite = true;
    let mut retained = Vec::new();
    for (index, (name, tensor)) in bank.buffers.iter().enumerate() {
        // The ordinary suffix call completed and released all state owners.
        let values = read_f32_tensor(tensor);
        let finite = values.iter().all(|v| v.is_finite());
        let mut hash = Sha256::new();
        for chunk in values.chunks(1024) {
            let mut bytes = [0u8; 4096];
            for (dst, v) in bytes.chunks_exact_mut(4).zip(chunk) {
                dst.copy_from_slice(&v.to_bits().to_le_bytes());
            }
            let bytes = &bytes[..chunk.len() * 4];
            file.write_all(bytes)?;
            hash.update(bytes);
        }
        let hash = format!("{:x}", hash.finalize());
        let bytes = values.len() as u64 * 4;
        emit(
            out,
            json!({"event":"layer0_capture_tensor","schedule":label,"name":name,"shape":tensor.shape,
            "path":path,"byte_offset":offset,"bytes":bytes,"sha256":hash,"all_finite":finite,
            "absolute_range_half_open":if index<14 {Some([2048,2056])} else {None},
            "stage_semantics":match *name {"input"=>"mixed GDN input", "qkv"=>"projection before convolution",
                "beta"=>"after sigmoid", "alpha"=>"alpha projection", "decay"=>"after decay chain",
                "query"|"key"|"value"=>"after serial convolution and SiLU",
                "query_norm"|"key_norm"=>"after L2 normalization", "recurrent"=>"before gate normalization",
                "gate"=>"gate projection", "normalized"=>"after gated RMS normalization", "output"=>"final projection",
                _=>"persistent recurrent state"},
            "state_semantics":match *name {"initial_delta"|"initial_conv"=>Some("before absolute position2048"),
                "state_checkpoints"=>Some("three post-token delta states, positions2048/2049/2050; third is state after first3"),_=>None},
            "layout":"first shape dimension contiguous; recurrent state [key_component,value_component,value_head,checkpoint]"}),
        );
        offset += bytes;
        all_finite &= finite;
        if let Some(reference) = reference {
            let a = &reference[index];
            require(a.len() == values.len(), "capture shape mismatch")?;
            if !finite || !a.iter().all(|v| v.is_finite()) {
                continue;
            }
            let slices = if index < 14 {
                8
            } else if *name == "state_checkpoints" {
                3
            } else {
                1
            };
            let width = values.len() / slices;
            emit(
                out,
                json!({"event":"layer0_cross_schedule","name":name,"aggregate":metrics(a,&values),
                "slices":(0..slices).map(|s|json!({"slice":s,
                    "absolute_position":if index<14 || *name=="state_checkpoints" {Some(2048+s)} else {None},
                    "metrics":metrics(&a[s*width..(s+1)*width],&values[s*width..(s+1)*width])})).collect::<Vec<_>>(),
                "policy":"report only; no cross-schedule bit or numerical threshold"}),
            );
        } else {
            retained.push(values);
        }
    }
    file.sync_all()?;
    emit(
        out,
        json!({"event":"layer0_binary_complete","schedule":label,"path":path,"bytes":offset,
        "sha256":sha256_file(&path),"all_finite":all_finite}),
    );
    require(
        all_finite,
        "nonfinite captured tensor; full raw bank retained",
    )?;
    Ok(retained)
}

fn weight_metadata(weights: GatedDeltaNetMetalWeights<'_>) -> Value {
    json!([("qkv",weights.qkv),("gate",weights.gate),("beta",weights.beta),
        ("alpha",weights.alpha),("conv",weights.conv),("a",weights.a),("dt_bias",weights.dt_bias),
        ("norm",weights.norm),("output",weights.output)].map(|(name,t)|json!({"name":name,
            "dtype":format!("{:?}",t.dtype),"shape":t.shape,"offset":t.offset,"bytes":t.n_bytes()})))
}

fn hc_weight_metadata(
    weights: &crate::qwen4exp_layer_zero::Qwen4ExpLayerZeroMetalWeights<'_>,
) -> Value {
    let rows: Vec<_> = [("attn",weights.attention_residual),("ffn",weights.ffn_residual)]
        .into_iter().flat_map(|(role,w)|[("norm",w.read.norm),("down",w.read.down),("up",w.read.up),("inject",w.inject)]
            .map(|(part,t)|json!({"name":format!("blk.0.hc_{role}_{part}.weight"),
                "dtype":format!("{:?}",t.dtype),"shape":t.shape,"offset":t.offset,"bytes":t.n_bytes()})))
        .collect();
    json!(rows)
}

fn packet(out: &mut std::fs::File, bf16_act: Bf16Act, hc_mode: HcMode) -> PacketResult<()> {
    with_native_artifact(out, |ctx, gguf, out| {
        let prompts = prompts(gguf, false, out)?;
        let tokens = &prompts[0].1;
        with_native_runner(ctx, gguf, CAPACITY, MARGIN, out, |r, out| {
            require(
                r.packed_qsa_dense_end()? == 2051,
                "probe requires frontier2051",
            )?;
            let (mut bank, mut hc) = allocation(r, out, hc_mode)?;
            let oracle = if hc_mode != HcMode::Off {
                Some(Oracle::new(
                    r.weights.zero_one.layer_zero.attention_residual.read,
                    out,
                )?)
            } else {
                None
            };
            emit(
                out,
                json!({"event":"layer0_weights","weights":weight_metadata(r.weights.zero_one.layer_zero.gdn),
                "hc_weights":hc_weight_metadata(&r.weights.zero_one.layer_zero),
                "geometry":format!("{:?}",r.weights.zero_one.layer_zero.gdn.geometry),
                "identity":"QKV Metal buffer object plus byte offset, scoped to suffix only"}),
            );
            r.reset()?;
            zero_persistent_state(r);
            require(
                !crate::metal::dispatch_census_is_active(),
                "external prefix census active",
            )?;
            crate::metal::dispatch_census_begin();
            let prefix =
                with_qwen4exp_frontier_schedule(false, || r.prefill(&tokens[..PREFIX]).map(|_| ()));
            let prefix_census = crate::metal::dispatch_census_take();
            prefix?;
            bf16_witness(
                out,
                "prefix2048/production",
                Bf16Act::Production,
                &prefix_census,
            )?;
            drop(prefix_census);
            let checkpoint = r.workspace.checkpoint_for_tests();
            let mut reference = Vec::new();
            let mut hc_reference = Vec::new();
            let mut a_endpoint = None;
            for (label, candidate) in [("A", false), ("B", true)] {
                r.workspace.restore_checkpoint_for_tests(&checkpoint);
                let (off, returned) =
                    hc_oracle::scoped(hc.take(), hc_mode, candidate, false, || {
                        suffix(r, tokens, candidate, bf16_act, &format!("{label}/off"), out)
                    });
                hc = returned;
                let off = off?;
                if let Some(probe) = &hc {
                    hc_oracle::witness(
                        probe,
                        &off.census,
                        candidate,
                        &format!("{label}/off"),
                        out,
                    )?;
                }
                r.workspace.restore_checkpoint_for_tests(&checkpoint);
                bank.candidate = candidate;
                let ((on, returned), returned_hc) =
                    hc_oracle::scoped(hc.take(), hc_mode, candidate, true, || {
                        with_frontier_gdn_capture(bank, || {
                            suffix(r, tokens, candidate, bf16_act, &format!("{label}/on"), out)
                        })
                    });
                bank = returned;
                hc = returned_hc;
                let on = on?;
                if let Some(probe) = &hc {
                    hc_oracle::witness(probe, &on.census, candidate, &format!("{label}/on"), out)?;
                }
                require(bank.complete, "missing layer-zero capture")?;
                let gdn_rows: Vec<_> = on.census.iter().filter(|r|r.tag.as_deref().is_some_and(|s|s.starts_with(GDN_TAG)))
                    .map(|r|json!({"tag":r.tag,"kernel":r.kernel,"encoder":r.encoder_ordinal,
                        "concurrent":r.encoder_concurrent,"grid":[r.grid_width,r.grid_height,r.grid_depth],
                        "threads":[r.threads_width,r.threads_height,r.threads_depth]})).collect();
                emit(
                    out,
                    json!({"event":"layer0_dispatches","schedule":label,"rows":gdn_rows}),
                );
                // Retain evidence even if observer concordance subsequently fails.
                if candidate {
                    retain(&bank, label, out, Some(&reference))?;
                } else {
                    reference = retain(&bank, label, out, None)?;
                }
                if let (Some(oracle), Some(probe)) = (&oracle, &hc) {
                    if candidate {
                        oracle.retain(probe, label, Some(&hc_reference), out)?;
                    } else {
                        hc_reference = oracle.retain(probe, label, None, out)?;
                    }
                }
                concordance(out, label, &off, &on, candidate, hc_mode != HcMode::Off)?;
                if let Some(a) = &a_endpoint {
                    emit(
                        out,
                        json!({"event":"layer0_endpoint_cross_schedule","comparison":comparison(a,&on.endpoint),
                        "kl_a_b":kl(&a.logits,&on.endpoint.logits),"kl_b_a":kl(&on.endpoint.logits,&a.logits),
                        "policy":"reported, no cross-schedule quality gate"}),
                    );
                } else {
                    a_endpoint = Some(on.endpoint);
                }
            }
            Ok(())
        })
    })
}

#[test]
fn frontier_layer0_bf16_mode_defaults_and_rejects_unknown_values() {
    assert_eq!(Bf16Act::parse(None), Ok(Bf16Act::Production));
    assert_eq!(Bf16Act::parse(Some("production")), Ok(Bf16Act::Production));
    assert_eq!(Bf16Act::parse(Some("f32")), Ok(Bf16Act::F32));
    for value in ["", "false", "bfloat", "F32"] {
        assert!(Bf16Act::parse(Some(value)).is_err());
    }
}

#[test]
fn frontier_layer0_metrics_report_drift_without_a_gate() {
    let same = metrics(&[0.0, 2.0], &[0.0, 2.0]);
    assert_eq!(same["different_bits"], 0);
    assert_eq!(same["max_abs"], 0.0);
    let different = metrics(&[0.0, 2.0], &[0.0, 3.0]);
    assert_eq!(different["different_bits"], 1);
    assert_eq!(different["relative_l2"], 0.5);
    assert!(metrics(&[0.0], &[1.0])["relative_l2"].is_null());
}

#[test]
fn frontier_layer0_census_checks_kernel_geometry_and_order_metadata() {
    let row = DispatchCensusRow {
        family: "gdn",
        tag: None,
        encoder_ordinal: 0,
        encoder_concurrent: false,
        kernel: "kernel_gdn_step_decay_packed_nsg4_f32".into(),
        grid_width: 32,
        grid_height: 48,
        grid_depth: 1,
        threads_width: 32,
        threads_height: 4,
        threads_depth: 1,
        grid_tgs: 1536,
        tg_threads: 128,
    };
    let mut tagged = row.clone();
    tagged.tag = Some(format!("{GDN_TAG}absolute2048.N2048"));
    assert!(same_dispatch(&row, &tagged));
    assert!(!is_copy(&tagged));
    tagged.grid_width += 1;
    assert!(!same_dispatch(&row, &tagged));
    tagged = row.clone();
    tagged.kernel.push_str("_different");
    assert!(!same_dispatch(&row, &tagged));
    tagged = row.clone();
    tagged.encoder_ordinal += 1;
    assert!(!same_dispatch(&row, &tagged));
}

#[test]
#[ignore = "release; normal native admission/lease; FLASH_PREFILL_MODEL and NEW persistent FLASH_PREFILL_OUT"]
fn native_frontier_layer0() {
    let bf16_act = Bf16Act::parse(
        std::env::var("FLASH_FRONTIER_LAYER0_BF16_ACT")
            .ok()
            .as_deref(),
    )
    .unwrap();
    let hc_mode = HcMode::parse(std::env::var("FLASH_FRONTIER_LAYER0_HC").ok().as_deref()).unwrap();
    assert!(
        hc_mode == HcMode::Off || bf16_act == Bf16Act::Production,
        "HC probe excludes the broad suffix BF16 override"
    );
    assert!(
        hc_mode == HcMode::Off || crate::env_flag::read_default_on("QWEN_MATMAT_BF16_BFLOAT_ACT"),
        "HC attribution requires production BF16 activation policy enabled"
    );
    let environment: BTreeMap<_, _> = [
        "QWEN_MATMAT_Q4_K_N64",
        "QWEN_MATMAT_Q5_K_N64",
        "QWEN_MATMAT_Q6_K_N64",
        "QWEN_MATMAT_Q5_K_N64_MIN_N",
        "QWEN_MATMAT_Q6_K_N64_MIN_N",
        "QWEN_MATMAT_Q5_K_N2_SEQ",
        "QWEN_MATMAT_SMALLN_TABLE",
        "QWEN_MATMAT_BF16_BFLOAT_ACT",
    ]
    .map(|key| (key, std::env::var(key).ok()))
    .into_iter()
    .collect();
    run_packet(
        "flash.frontier_layer0.v1",
        include_bytes!("frontier_layer0.rs"),
        json!({
            "scope":"cfg(test) capture of layer0 only; eight absolute rows2048..2055; unchanged N3/2045 versus N2048; optional BF16 intervention covers all suffix layers",
            "work":"one N2048 prefix, restore checkpoint before Aoff/Aon/Boff/Bon; four complete ordinary suffixes; no continuation or timing claim",
            "corpus":"retained prose; GSQ artifact selected by FLASH_PREFILL_MODEL; current production router in both arms",
            "observer_gate":"exact same-schedule endpoint/state/census; cross-schedule metrics report only",
            "bf16_activation_mode":bf16_act.label(),
            "hc_mode":hc_mode.label(),
            "hc_scope":"off preserves the original probe; production/f32downup enable HC captures and oracle. Policy is active identically observer-off/on, only for both-identity-qualified layer0 attention down/up at actual suffix ranges. Prefix, injection, FFN HC, other layers and projection policies remain production. Existing broad HC overrides are rejected.",
            "hc_source_sha256":sha256_bytes(include_bytes!("../../../qwen4exp_metal/frontier_hc.rs")),
            "hc_integration_source_sha256":sha256_bytes(include_bytes!("../../../qwen4exp_metal.rs")),
            "hc_oracle_source_sha256":sha256_bytes(include_bytes!("frontier_hc_oracle.rs")),
            "hc_outputs":".hc.weights.bin (BF16 down/up and F32 norm); .A.hc.f32le/.B.hc.f32le; .A.hc.oracle.f64le/.B.hc.oracle.f64le, all create_new",
            "bf16_override_scope":"f32 disables the shared dense BF16 bfloat-activation path only inside each complete ordinary suffix call, equally for Aoff/Aon/Boff/Bon and all layers; production adds no override. Prefix2048 and its checkpoint always use production settings. No router, width, quantized-kernel or weight-dtype override.",
            "capture_source_sha256":sha256_bytes(include_bytes!("../../../qwen4exp_gdn/frontier_capture.rs")),
            "schedule_source_sha256":sha256_bytes(include_bytes!("frontier_schedule.rs")),
            "environment": environment,
            "binary_outputs":"FLASH_PREFILL_OUT.A.f32le and .B.f32le, create_new; per-tensor offsets/shapes/hashes in JSONL"
        }),
        |out| packet(out, bf16_act, hc_mode),
    );
}
