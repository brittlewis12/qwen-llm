//! `qwen-bench` — throughput harness and microbenchmark surface for qwen-llm.
//!
//! Product-facing modes are `decode`, `pp`, `tg`, `suite`, `ctx-sweep`, and
//! `phase`; the remaining subcommands are bounded probes that live in one
//! module each under `bench/`. Designed so "measure → change → measure" is
//! `cargo run --release -p qwen-cli --bin qwen-bench -- decode -m ... --tokens 64`,
//! not "run the right ignored test by name".

#[path = "bench/args.rs"]
mod args;
mod attn_capture;
#[path = "bench/attn_micro.rs"]
mod attn_micro;
mod attn_stage_floor;
#[cfg(feature = "dsv4-diagnostics")]
mod batch_probe;
#[path = "bench/block_slice.rs"]
mod block_slice;
#[path = "bench/decode_window.rs"]
mod decode_window;
mod dense_block_batch;
mod dense_whole_batch;
#[path = "bench/dflash.rs"]
mod dflash;
mod dflash_e0_lockstep;
#[cfg(feature = "dflash-k0s-diagnostics")]
mod dflash_k0s;
mod dflash_sampled_oracle;
#[path = "bench/diagnostics.rs"]
mod diagnostics;
#[cfg(feature = "dsv4-diagnostics")]
mod dsv4_mhc_delete;
#[cfg(feature = "dsv4-diagnostics")]
mod dsv4_prefill;
#[path = "bench/gdn_replay.rs"]
mod gdn_replay;
mod gguf_arena_floor;
mod grammar_lm_head_row_floor;
mod grammar_row_runtime;
mod host_validity;
#[path = "bench/identity.rs"]
mod identity;
mod integrated_grammar_row;
mod lm_head_screening_oracle;
mod messages;
#[allow(dead_code)]
mod model_request;
mod moe_gdn_repair;
#[path = "bench/moe_micro.rs"]
mod moe_micro;
#[path = "bench/mtp.rs"]
mod mtp;
mod muse_glimmer_request_bench;
mod open_responses;
#[path = "bench/pld.rs"]
mod pld;
#[path = "bench/power.rs"]
mod power;
#[path = "bench/pp.rs"]
mod pp;
#[path = "bench/prefix_cache.rs"]
mod prefix_cache;
mod prefix_cache_vt_ab;
#[path = "bench/proj_micro.rs"]
mod proj_micro;
#[allow(dead_code)]
mod prompt_template;
mod q4_mma_ceiling;
mod response_shape_runtime;
#[path = "bench/roofline.rs"]
mod roofline;
mod rope_micro;
mod shutdown;
#[path = "../source_identity.rs"]
mod source_identity;
#[path = "bench/suite.rs"]
mod suite;
#[path = "bench/tg.rs"]
mod tg;
#[path = "bench/timing.rs"]
mod timing;
#[path = "bench/tok.rs"]
mod tok;
#[path = "bench/topology.rs"]
mod topology;
mod tracing_init;
#[path = "bench/vocab_audit.rs"]
mod vocab_audit;

use anyhow::{Context, Result, anyhow};
use args::*;
use attn_micro::*;
use block_slice::*;
use clap::{Parser, Subcommand, ValueEnum};
use decode_window::*;
use dflash::*;
use diagnostics::*;
use gdn_replay::*;
use identity::*;
#[cfg(test)]
use messages::{
    ChatMessage, messages_auto_preserve_thinking, parse_messages_input,
    render_qwen_messages_prompt, strip_think,
};
use messages::{load_messages_prompt, messages_thinking_mode};
use moe_micro::*;
use mtp::*;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::{NSError, NSString};
use objc2_metal::{
    MTLAllocation, MTLBuffer, MTLCommandBuffer, MTLCommandQueue, MTLComputePipelineState,
    MTLCreateSystemDefaultDevice, MTLDevice, MTLResidencySet, MTLResidencySetDescriptor, MTLSize,
};
use pld::*;
use power::*;
use pp::*;
use prefix_cache::*;
use proj_micro::*;
use qwen_llm::{
    forward::mat_vec_pub,
    gguf::GgufFile,
    loader::{Model, open_dflash_drafter},
    metal::{
        BlitEncoder, KernelEncoder, KernelTraceCounters, MetalContext, MetalTensor,
        RetainedStorageDisposition, attn_v4_choose_group_tile, attn_v4_choose_nwg,
        attn_v4_choose_tile_c, encode_add_inplace_f32, encode_attn_decode_v4_f32,
        encode_attn_decode_v4_main_only_f32, encode_attn_decode_v4_reduce_only_f32,
        encode_attn_prefill_v4_g6_q2_c32_f32, encode_attn_prefill_v4_g8_t2_q2_c64_f32,
        encode_attn_prefill_v4_g8_t2_q4_c64_f32, encode_attn_prefill_v4_g16_t4_q2_c64_f32,
        encode_attn_prefill_v4_g16_t4_q4_c64_f32, encode_fill_f32, encode_gdn_decay_chain_f32,
        encode_get_rows_f32, encode_mat_vec_f32_sigmoid, encode_mat_vec_q8_0_batch_f32,
        encode_moe_down_weighted_sum_q5_K_f32_packed_slots,
        encode_moe_down_weighted_sum_q5_K_f32_packed_slots_k512_r2,
        encode_moe_fused_routed_q4q5_token_f32, encode_moe_swiglu_q4_K_f32,
        encode_moe_swiglu_q4_K_f32_packed_slots, encode_mul_f32,
        encode_qk_rms_norm_rope_f32_packed_consecutive, encode_residual_rms_norm_mul_f32,
        encode_rms_norm_batched_f32, encode_rms_norm_batched_src_strided_f32,
        encode_rms_norm_mul_f32, encode_roofline_fma_f32, encode_roofline_stream_f32,
        encode_rope_neox_f32, encode_rope_neox_f32_packed_consecutive, encode_rope_neox_pair_f32,
        encode_scatter_offset_f32_to_f16, encode_scatter_offset_f32_to_f16_kv,
        encode_scatter_offset_f32_to_q8_0_kv, encode_sigmoid_f32,
        encode_sigmoid_mul_gate_strided_f32, encode_split_q_gate_f32, encode_touch_bytes_f32,
        host_page_size_bytes, kernel_trace_begin, kernel_trace_snapshot, plan_retained_storage,
        with_attn_v4_group_tile_override,
    },
    metal_dflash::{
        DFlashDecoder, MetalDFlashHead, MetalDFlashLayerMajorScratch, MetalDFlashSession,
        MetalDFlashVerifyScratch, prefill_tokens_prompt_only_profiled,
        prefill_tokens_with_multi_hidden, prefill_tokens_with_multi_hidden_profiled,
        with_prefill_dense_ffn_fused_swiglu_q4_override,
    },
    metal_forward::{
        MetalBlock, MetalForward, MetalModel, MetalMoeFfn, MetalSession, ModelWeightStorageKind,
        MoeRouteReplayRow, RMS_EPS, gguf_descriptor_layout_digest, model_weight_storage_requests,
        mtp_weight_source_descriptors, native_quant_embedding_storage_supported,
        production_native_quant_embedding_storage_enabled,
    },
    metal_forward::{encode_mat_mat_dispatch, encode_mat_vec_dispatch},
    metal_mtp::{
        DecodeOutput, MetalMtpHead, MetalMtpSession, MtpBaseHiddenVariant, MtpHistoryMode,
        MtpRankRow, MtpRecursiveHiddenVariant, PackedDraftPlan, PackedStepProbe,
        PackedStepProbePhase, RecordedDraftStep, RecordedMtpWork, SpeculativeDecoder,
        quantize_lm_head_to_affine_q4_gs64, quantize_lm_head_to_q4_0, quantize_lm_head_to_q4_1,
    },
    prompt_lookup::{
        DRAFT_TOKENS, PromptLookupProposer, PromptLookupTerminalCause, ProposalSource,
        terminal_draft_window,
    },
    runtime::{LoadedModel, Runtime, SequenceConfig},
    tensor::GgmlType,
    tokenizer::{LlamaCppTokenizer, NativeTokenizer, Tokenizer, token_ids_sha256_i32le},
};
use roofline::*;
use std::collections::{BTreeMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::time::{Duration, Instant};
use suite::*;
use tg::*;
use timing::*;
use tok::*;
use topology::*;
use vocab_audit::*;

type MetalQueue = Retained<ProtocolObject<dyn MTLCommandQueue>>;
type MetalCommand = Retained<ProtocolObject<dyn MTLCommandBuffer>>;
type CapturedDownRouteTensors = (usize, Vec<(MetalTensor, MetalTensor, MetalTensor)>);

fn env_flag_enabled(name: &str) -> bool {
    matches!(
        std::env::var(name).as_deref(),
        Ok("1") | Ok("true") | Ok("TRUE") | Ok("yes") | Ok("YES")
    )
}

fn env_flag_default_on(name: &str) -> bool {
    !matches!(
        std::env::var(name).as_deref(),
        Ok("0") | Ok("false") | Ok("FALSE") | Ok("no") | Ok("NO")
    )
}

/// Resolve the effective stop-token set: CLI override if provided,
/// otherwise the GGUF's declared set. Errors when the GGUF declares
/// nothing AND no override is given. No heuristic fallback — silent
/// defaults are exactly the bug this is fixing.
fn resolve_stop_tokens(
    g: &qwen_llm::gguf::GgufFile,
    override_set: Option<Vec<i32>>,
) -> Result<Vec<i32>> {
    if let Some(s) = override_set {
        return Ok(s);
    }
    g.stop_token_ids()
        .map_err(|e| anyhow!("resolving stop tokens from GGUF: {e}"))
}

#[cfg(test)]
mod stop_token_cli_tests {
    use super::*;

    #[test]
    fn mtp_stop_tokens_accept_single_and_comma_delimited_values() {
        let single =
            MtpArgs::try_parse_from(["mtp", "--model", "model.gguf", "--stop-tokens", "0"])
                .expect("single stop token");
        assert_eq!(single.stop_tokens, Some(vec![0]));

        let multiple = MtpArgs::try_parse_from([
            "mtp",
            "--model",
            "model.gguf",
            "--stop-tokens",
            "248046,248044",
        ])
        .expect("comma-delimited stop tokens");
        assert_eq!(multiple.stop_tokens, Some(vec![248046, 248044]));
    }

    #[test]
    fn bench_parses_dflash_sampled_oracle_subcommand() {
        let parsed = Args::try_parse_from([
            "qwen-bench",
            "dflash-sampled-oracle",
            "--model",
            "target.gguf",
            "--drafter",
            "draft.gguf",
            "--prompt",
            "hello",
            "--output",
            "evidence.jsonl",
        ])
        .expect("sampled oracle command");
        assert!(matches!(parsed.cmd, Cmd::DflashSampledOracle(_)));
    }

    #[test]
    fn bench_parses_dflash_e0_lockstep_subcommand() {
        let parsed = Args::try_parse_from([
            "qwen-bench",
            "dflash-e0-lockstep",
            "--model",
            "target.gguf",
            "--drafter",
            "draft.gguf",
            "--binding-manifest",
            "e0-binding.json",
            "--prompt",
            "hello",
            "--output",
            "e0.jsonl",
            "--state-sidecar",
            "e0-state.bin",
        ])
        .expect("E0 lockstep command");
        assert!(matches!(parsed.cmd, Cmd::DflashE0Lockstep(_)));
    }

    #[cfg(feature = "dflash-k0s-diagnostics")]
    #[test]
    fn bench_parses_hidden_dflash_k0s_lattice_subcommand() {
        let parsed = Args::try_parse_from([
            "qwen-bench",
            "dflash-k0s-lattice",
            "--attempt-id",
            "attempt-1",
            "--model",
            "target.gguf",
            "--drafter",
            "draft.gguf",
            "--prompt",
            "hello",
            "--carry-token",
            "42",
            "--continuation-carry-token",
            "43",
            "--manifest",
            "manifest.json",
            "--manifest-sha256",
            &"a".repeat(64),
            "--command-manifest",
            "command.json",
            "--fixture",
            "fixture.json",
            "--temperature",
            "0.7",
            "--fixed-chain",
            "alternate:42:0,1,2,3,4,5,6",
            "--trace-output",
            "trace.jsonl",
            "--sidecar-output",
            "rows.bin",
        ])
        .expect("K0-S lattice command");
        assert!(matches!(parsed.cmd, Cmd::DflashK0sLattice(_)));
    }

    #[cfg(feature = "dflash-k0s-diagnostics")]
    #[test]
    fn bench_parses_hidden_dflash_k0s_inventory_subcommand() {
        let parsed = Args::try_parse_from([
            "qwen-bench",
            "dflash-k0s-inventory",
            "--model",
            "target.gguf",
            "--drafter",
            "draft.gguf",
            "--prompt",
            "hello",
            "--carry-token",
            "42",
            "--inventory-spec",
            "inventory-spec.json",
            "--inventory-spec-sha256",
            &"a".repeat(64),
            "--output",
            "inventory.json",
        ])
        .expect("K0-S inventory command");
        assert!(matches!(parsed.cmd, Cmd::DflashK0sInventory(_)));
    }

    #[cfg(feature = "dflash-k0s-diagnostics")]
    #[test]
    fn feature_on_links_hidden_dflash_k0s_staged_call_sites() {
        assert!(std::mem::size_of::<dflash_k0s::DflashK0sArgs>() > 0);
        let producer = include_str!("dflash_k0s.rs");
        let runtime = producer
            .split("fn run_arm(")
            .nth(1)
            .unwrap()
            .split("fn projection_parts(")
            .next()
            .unwrap();
        assert_eq!(
            runtime.matches("draft_block_with_k0s_observation(").count(),
            2
        );
        assert_eq!(runtime.matches("extract_k0s_observation(").count(), 1);
        assert_eq!(
            runtime
                .matches("finish_k0s_observation_without_extraction(")
                .count(),
            1
        );
        assert!(!runtime.contains(&["draft_block_with_k0s", "_diagnostic("].concat()));
        assert!(!include_str!("main.rs").contains(&["DFlash", "K0s"].concat()));
    }

    #[cfg(not(feature = "dflash-k0s-diagnostics"))]
    #[test]
    fn default_parser_has_no_dflash_k0s_lattice_subcommand() {
        let error = Args::try_parse_from(["qwen-bench", "dflash-k0s-lattice"])
            .expect_err("feature-off parser must reject K0-S");
        assert!(error.to_string().contains("unrecognized subcommand"));
        let error = Args::try_parse_from(["qwen-bench", "dflash-k0s-inventory"])
            .expect_err("feature-off parser must reject K0-S inventory");
        assert!(error.to_string().contains("unrecognized subcommand"));
    }

    #[cfg(not(feature = "dflash-k0s-diagnostics"))]
    #[test]
    fn default_bench_source_gates_every_k0s_reference() {
        let source = concat!(
            include_str!("bench.rs"),
            "\n",
            include_str!("bench/args.rs")
        );
        for line in source.lines().filter(|line| line.contains("dflash_k0s")) {
            assert!(
                line.contains("mod dflash_k0s")
                    || line.contains("dflash_k0s::")
                    || line.contains("filter(|line|")
                    || line.contains("let producer = include_str!")
                    || line.contains("feature_on_links_hidden")
                    || line.contains("bench_parses_hidden_dflash_k0s")
                    || line.contains("k0s_reference")
                    || line.contains("k0s_lattice")
                    || line.trim_start().starts_with("#[cfg(feature"),
                "unexpected feature-off K0-S source reference: {line}"
            );
        }
        assert!(!source.contains(&["draft_block_with_k0s", "_diagnostic("].concat()));
        assert!(!source.contains(&["DFlash", "K0sCapture"].concat()));
        let product_sources = [
            ("main.rs", include_str!("main.rs")),
            ("cli.rs", include_str!("cli.rs")),
            (
                "execution_selector.rs",
                include_str!("execution_selector.rs"),
            ),
            (
                "response_shape_runtime.rs",
                include_str!("response_shape_runtime.rs"),
            ),
            (
                "grammar_row_runtime.rs",
                include_str!("grammar_row_runtime.rs"),
            ),
            ("serve/mod.rs", include_str!("serve/mod.rs")),
            ("serve/backend.rs", include_str!("serve/backend.rs")),
            ("serve/backend_ds4.rs", include_str!("serve/backend_ds4.rs")),
            ("serve/events.rs", include_str!("serve/events.rs")),
            ("serve/http.rs", include_str!("serve/http.rs")),
            (
                "open_responses/items.rs",
                include_str!("open_responses/items.rs"),
            ),
            ("serve/partition.rs", include_str!("serve/partition.rs")),
            (
                "open_responses/render.rs",
                include_str!("open_responses/render.rs"),
            ),
            ("serve/render_ds4.rs", include_str!("serve/render_ds4.rs")),
            (
                "open_responses/tool_parse.rs",
                include_str!("open_responses/tool_parse.rs"),
            ),
            ("serve/utf8.rs", include_str!("serve/utf8.rs")),
            (
                "qwen-llm/runtime.rs",
                include_str!("../../qwen-llm/src/runtime.rs"),
            ),
            (
                "qwen-llm/metal_mtp.rs",
                include_str!("../../qwen-llm/src/metal_mtp.rs"),
            ),
            ("qwen-llm/lib.rs", include_str!("../../qwen-llm/src/lib.rs")),
        ];
        for (name, product) in product_sources {
            assert!(
                !product.contains(&["draft_block_with_k0s", "_diagnostic"].concat()),
                "product source {name} links the diagnostic call"
            );
            assert!(
                !product.contains(&["DFlash", "K0s"].concat()),
                "product source {name} names a K0-S library type"
            );
        }
    }
}

/// JSON schema version for `BenchRow`. Bump when fields are renamed,
/// removed, or have their semantics changed. Adding new optional fields
/// (always-null on old emitters) does NOT require a bump.
const BENCH_SCHEMA_VERSION: u32 = 2;

/// One bench result row. Field names match `llama-bench`'s JSON schema where
/// the meaning is the same; engine-specific fields are `Option<T>` and
/// serialized as explicit `null` (NOT omitted) so downstream consumers can
/// rely on a stable field set.
#[derive(Debug, Clone, serde::Serialize)]
struct BenchRow {
    schema_version: u32,
    engine: &'static str,
    build_commit: &'static str,
    /// `1` if source changes or hidden index flags were observed at build time
    /// or runtime.
    build_dirty: u8,
    build_identity: BuildIdentity,
    test_time: String,
    model_filename: String,
    model_size: u64,
    model_n_params: u64,
    arch_kind: &'static str,
    /// `pp<N>` or `tg<N>`, matching `llama-bench`'s shape vocabulary.
    test: String,
    n_tokens: usize,
    n_repetitions: usize,
    avg_ts: f64,
    stddev_ts: f64,
    samples_ts: Vec<f64>,
    samples_ns: Vec<u64>,
    avg_ns: u64,
    avg_compute_ns: Option<u64>,
    avg_session_alloc_ns: Option<u64>,
    avg_scratch_alloc_ns: Option<u64>,
    avg_gpu_ns: Option<u64>,
    kernel_trace_command_buffers_per_token: Option<f64>,
    kernel_trace_encoders_per_token: Option<f64>,
    kernel_trace_concurrent_encoders_per_token: Option<f64>,
    kernel_trace_dispatches_per_token: Option<f64>,
    /// Effective decode bandwidth (GB/s). `None` for `pp<N>` rows and for
    /// MoE `tg<N>` rows — MoE active-param accounting is out of scope here,
    /// and the naive `model_size × t/s` overstates by ~10x for MoE.
    decode_gb_per_s: Option<f64>,
    prefill_chunk: Option<usize>,
    decode_mode: Option<&'static str>,
    prefill_mode: Option<&'static str>,
    power: Option<PowerSnapshot>,
    qwen_env: std::collections::BTreeMap<String, String>,
}

#[derive(Debug, Clone, Default, serde::Serialize)]
struct PowerSnapshot {
    source: Option<String>,
    battery_percent: Option<u8>,
    battery_state: Option<String>,
    battery_warning: Option<String>,
    powermode_battery: Option<i32>,
    powermode_ac: Option<i32>,
    thermal_warning_recorded: Option<bool>,
    performance_warning_recorded: Option<bool>,
    cpu_power_status_recorded: Option<bool>,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, clap::ValueEnum, Default)]
enum OutputFormat {
    #[default]
    Text,
    /// Suppresses stderr text so `qwen-bench ... -o json | jq` works clean.
    Json,
}

/// `YYYY-MM-DDTHH:MM:SSZ`, matching lcpp's `test_time` shape. Uses
/// Hinnant's days_from_civil so we don't pull in chrono.
fn utc_iso8601_now() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0) as i64;
    let days = secs.div_euclid(86_400);
    let sod = secs.rem_euclid(86_400);
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = mp + if mp < 10 { 3 } else { -9 };
    let year = y + if m <= 2 { 1 } else { 0 };
    let hour = sod / 3600;
    let minute = (sod % 3600) / 60;
    let second = sod % 60;
    format!("{year:04}-{m:02}-{d:02}T{hour:02}:{minute:02}:{second:02}Z")
}

/// Sum of weight-tensor bytes (matches lcpp's `model_size` = `llama_model_size`).
/// NOT file size — GGUF metadata and alignment padding are excluded so
/// derived bandwidth numbers are apples-to-apples with lcpp.
fn model_weight_bytes(g: &qwen_llm::gguf::GgufFile) -> u64 {
    g.tensors.iter().map(|t| t.n_bytes).sum()
}

fn main() -> std::process::ExitCode {
    shutdown::finish(run())
}

fn run() -> Result<()> {
    shutdown::install()?;
    tracing_init::install_default_subscriber();

    let args = Args::parse();
    let policy = BuildIdentityPolicy {
        allow_dirty: args.allow_dirty,
        allow_unverifiable: args.allow_unverifiable_build,
    };
    let _ = BUILD_IDENTITY_POLICY.set(policy);
    if !matches!(
        &args.cmd,
        Cmd::BuildInfo(_) | Cmd::LmHeadScreeningOracle(_) | Cmd::DflashE0Lockstep(_)
    ) {
        validate_build_identity(qwen_build_identity_packet(), policy)?;
    }
    match args.cmd {
        Cmd::AttnCapture(a) => {
            attn_capture::run(a, serde_json::to_value(qwen_build_identity_packet())?)
        }
        Cmd::AttnStageFloor(a) => {
            attn_stage_floor::run(a, serde_json::to_value(qwen_build_identity_packet())?)
        }
        #[cfg(feature = "dsv4-diagnostics")]
        Cmd::QueueOverlapProbe(a) => {
            batch_probe::run(a, serde_json::to_value(qwen_build_identity_packet())?)
        }
        Cmd::DecodeDenseBlockBatch(a) => dense_block_batch::run(a),
        Cmd::DecodeDenseAttnBatch(a) => dense_block_batch::run_attention(a),
        Cmd::DecodeDenseWholeBatch(a) => dense_whole_batch::run(a),
        Cmd::DecodeMoeGdnRepair(a) => moe_gdn_repair::run(a),
        Cmd::BuildInfo(a) => run_build_info(a),
        Cmd::GgufStoragePlan(a) => run_gguf_storage_plan(a),
        Cmd::GgufArenaFloor(a) => {
            gguf_arena_floor::run(a, serde_json::to_value(qwen_build_identity_packet())?)
        }
        Cmd::PrefixCache(a) => run_prefix_cache(a),
        Cmd::PrefixCacheVtAb(a) => {
            prefix_cache_vt_ab::run(a, serde_json::to_value(qwen_build_identity_packet())?)
        }
        Cmd::VocabAudit(a) => run_vocab_audit(a),
        Cmd::Decode(a) => run_decode(a),
        Cmd::MuseRequest(a) => muse_glimmer_request_bench::run(a),
        Cmd::Pp(a) => run_pp(a),
        #[cfg(feature = "dsv4-diagnostics")]
        Cmd::Dsv4Prefill(a) => {
            dsv4_prefill::run(a, serde_json::to_value(qwen_build_identity_packet())?)
        }
        #[cfg(feature = "dsv4-diagnostics")]
        Cmd::Dsv4MhcDelete(a) => dsv4_mhc_delete::run(
            a,
            serde_json::to_value(recorded_build_identity())?,
            capture_qwen_env(),
        ),
        Cmd::Tg(a) => run_tg(a),
        Cmd::Suite(a) => run_suite(a),
        Cmd::CtxSweep(a) => run_ctx_sweep(a),
        Cmd::Phase(a) => run_phase(a),
        Cmd::AttnIntra(a) => run_attn_intra(a),
        Cmd::GdnProjMicro(a) => run_gdn_proj_micro(a),
        Cmd::MatmatSmallnMicro(a) => run_matmat_smalln_micro(a),
        Cmd::DecodeProjBatch(a) => run_decode_proj_batch(a),
        Cmd::DecodeGdnLayerReplay(a) => run_decode_gdn_layer_replay(a),
        Cmd::DecodeGdnChainReplay(a) => run_decode_gdn_chain_replay(a),
        Cmd::DecodeBlockSliceReplay(a) => run_decode_block_slice_replay(a),
        Cmd::DecodeBlockSliceTrace(a) => run_decode_block_slice_trace(a),
        Cmd::DecodeBlockSliceMarginSweep(a) => run_decode_block_slice_margin_sweep(a),
        Cmd::DecodeBlockSliceRealMargin(a) => run_decode_block_slice_real_margin(a),
        Cmd::DecodeMoeRouterRepackCheck(a) => run_decode_moe_router_repack_check(a),
        Cmd::MoeDownMicro(a) => run_moe_down_micro(a),
        Cmd::MoeGateupMicro(a) => run_moe_gateup_micro(a),
        Cmd::MoeBatchSweep(a) => run_moe_batch_sweep(a),
        Cmd::Roofline(a) => run_roofline(a),
        Cmd::RopeMicro(a) => rope_micro::run(a, serde_json::to_value(recorded_build_identity())?),
        Cmd::Q4MmaCeiling(a) => {
            q4_mma_ceiling::run(a, serde_json::to_value(recorded_build_identity())?)
        }
        Cmd::GrammarLmHeadRowFloor(a) => grammar_lm_head_row_floor::run(
            a,
            serde_json::to_value(recorded_build_identity())?,
            capture_qwen_env(),
        ),
        Cmd::IntegratedGrammarRow(a) => integrated_grammar_row::run(
            a,
            serde_json::to_value(recorded_build_identity())?,
            capture_qwen_env(),
        ),
        Cmd::LmHeadScreeningOracle(a) => lm_head_screening_oracle::run(a),
        Cmd::MetalCounters(a) => run_metal_counters(a),
        Cmd::MetalPipelines(a) => run_metal_pipelines(a),
        Cmd::TopologyProbe(a) => run_topology_probe(a),
        Cmd::DispatchCensus(a) => run_dispatch_census(a),
        Cmd::DecodeWindow(a) => run_decode_window(a),
        Cmd::Mtp(a) => run_mtp(a),
        Cmd::Pld(a) => run_pld(a),
        Cmd::DflashLazy(a) => run_dflash_lazy(a),
        Cmd::DflashSampledOracle(a) => dflash_sampled_oracle::run(
            a,
            serde_json::to_value(recorded_build_identity())?,
            capture_qwen_env(),
        ),
        Cmd::DflashE0Lockstep(a) => dflash_e0_lockstep::run(
            a,
            serde_json::to_value(recorded_build_identity())?,
            capture_qwen_env(),
        ),
        #[cfg(feature = "dflash-k0s-diagnostics")]
        Cmd::DflashK0sLattice(a) => {
            dflash_k0s::run(a, serde_json::to_value(recorded_build_identity())?)
        }
        #[cfg(feature = "dflash-k0s-diagnostics")]
        Cmd::DflashK0sInventory(a) => {
            dflash_k0s::run_inventory(a, serde_json::to_value(recorded_build_identity())?)
        }
        Cmd::Dflash(a) => run_dflash(a),
        Cmd::Tok(a) => run_tok(a),
        Cmd::AttnPrefillMicro(a) => run_attn_prefill_micro(a),
        Cmd::AttnFrontMicro(a) => run_attn_front_micro(a),
        Cmd::AttnLayerMicro(a) => run_attn_layer_micro(a),
        Cmd::PpFfnAb(a) => run_pp_ffn_ab(a),
        Cmd::PpWait(a) => run_pp_wait(a),
    }
}

fn run_build_info(args: BuildInfoArgs) -> Result<()> {
    let identity = recorded_build_identity();
    match args.output {
        OutputFormat::Json => println!("{}", serde_json::to_string_pretty(&identity)?),
        OutputFormat::Text => {
            println!("status\t{}", identity.status);
            println!("build_commit\t{}", identity.build_commit);
            println!(
                "runtime_commit\t{}",
                identity.runtime_commit.as_deref().unwrap_or("unknown")
            );
            println!(
                "build_source_state\t{}",
                identity.build_source_state.as_deref().unwrap_or("unknown")
            );
            println!(
                "runtime_source_state\t{}",
                identity
                    .runtime_source_state
                    .as_deref()
                    .unwrap_or("unknown")
            );
            println!(
                "dirty\tbuild={} runtime={}",
                identity
                    .build_dirty
                    .map(|v| v.to_string())
                    .unwrap_or_else(|| "unknown".to_string()),
                identity
                    .runtime_dirty
                    .map(|v| v.to_string())
                    .unwrap_or_else(|| "unknown".to_string())
            );
            println!("stamp_source\t{}", identity.stamp_source);
            println!("problems\t{}", identity.problems.join(","));
        }
    }
    Ok(())
}

fn run_gguf_storage_plan(args: GgufStoragePlanArgs) -> Result<()> {
    let gguf = GgufFile::open(&args.model).context("open GGUF")?;
    let model = Model::from_gguf(&gguf).context("bind model")?;
    let native_embedding = match args.embedding_policy {
        GgufStorageEmbeddingPolicy::ProductionAuto => {
            production_native_quant_embedding_storage_enabled(&model)
        }
        GgufStorageEmbeddingPolicy::ForceNativeIfSupported => {
            native_quant_embedding_storage_supported(&model)
        }
        GgufStorageEmbeddingPolicy::Disabled => false,
    };
    let requests = model_weight_storage_requests(&model, native_embedding, args.router_f16)?;
    let direct = requests
        .iter()
        .filter(|request| request.kind == ModelWeightStorageKind::Direct)
        .map(|request| request.desc)
        .collect::<Vec<_>>();
    let page_size = host_page_size_bytes()?;
    let device = MTLCreateSystemDefaultDevice().ok_or_else(|| anyhow!("no Metal device"))?;
    let device_max_buffer_length = device.maxBufferLength();
    let max_buffer_length = args.max_buffer_length.unwrap_or(device_max_buffer_length);
    if max_buffer_length > device_max_buffer_length {
        return Err(anyhow!(concat!(
            "requested maxBufferLength {max_buffer_length} exceeds device limit ",
            "{device_max_buffer_length}"
        )));
    }
    let shard_mapped_lengths = gguf.shard_mapped_lengths();
    let plan = plan_retained_storage(
        &shard_mapped_lengths,
        &direct,
        page_size,
        max_buffer_length,
        args.alignment,
    )?;

    let mut direct_logical_bytes = 0u64;
    let mut converted_source_bytes = 0u64;
    let mut converted_resident_bytes = 0u64;
    let mut converted_count = 0usize;
    let mut current_base_weight_private_bytes = 0u64;
    for request in &requests {
        current_base_weight_private_bytes = current_base_weight_private_bytes
            .checked_add(request.resident_bytes)
            .ok_or_else(|| anyhow!("current resident byte accounting overflow"))?;
        if request.kind == ModelWeightStorageKind::Direct {
            direct_logical_bytes = direct_logical_bytes
                .checked_add(request.desc.n_bytes)
                .ok_or_else(|| anyhow!("direct byte accounting overflow"))?;
        } else {
            converted_count += 1;
            converted_source_bytes = converted_source_bytes
                .checked_add(request.desc.n_bytes)
                .ok_or_else(|| anyhow!("converted source byte accounting overflow"))?;
            converted_resident_bytes = converted_resident_bytes
                .checked_add(request.resident_bytes)
                .ok_or_else(|| anyhow!("converted resident byte accounting overflow"))?;
        }
    }
    let planned_base_weight_private_bytes = converted_resident_bytes
        .checked_add(plan.unique_fallback_bytes)
        .ok_or_else(|| anyhow!("planned resident byte accounting overflow"))?;
    let estimated_base_weight_private_bytes_removed = current_base_weight_private_bytes
        .checked_sub(planned_base_weight_private_bytes)
        .ok_or_else(|| anyhow!("planned residency exceeds copied residency"))?;
    let mapped_window_bytes = plan.windows.iter().try_fold(0u64, |total, window| {
        total
            .checked_add(window.length as u64)
            .ok_or_else(|| anyhow!("window byte accounting overflow"))
    })?;
    let alias_count = plan
        .entries
        .iter()
        .filter(|entry| matches!(entry.disposition, RetainedStorageDisposition::Alias { .. }))
        .count();
    let unique_view_count = plan
        .entries
        .iter()
        .filter(|entry| matches!(entry.disposition, RetainedStorageDisposition::View { .. }))
        .count();
    let fallback_rows = plan
        .entries
        .iter()
        .filter_map(|entry| match entry.disposition {
            RetainedStorageDisposition::CopyFallback { reason } => Some(serde_json::json!({
                "name": entry.name,
                "shard_idx": entry.shard_idx,
                "data_offset": entry.data_offset,
                "n_bytes": entry.n_bytes,
                "reason": format!("{reason:?}"),
            })),
            _ => None,
        })
        .collect::<Vec<_>>();
    let windows = plan
        .windows
        .iter()
        .enumerate()
        .map(|(index, window)| {
            serde_json::json!({
                "index": index,
                "shard_idx": window.shard_idx,
                "mmap_offset": window.mmap_offset,
                "length": window.length,
            })
        })
        .collect::<Vec<_>>();
    let bound_sources = requests
        .iter()
        .map(|request| {
            (
                request.desc.shard_idx,
                request.desc.data_offset,
                request.desc.n_bytes,
            )
        })
        .collect::<HashSet<_>>();
    let mtp_descriptors = mtp_weight_source_descriptors(&model);
    let mtp_sources = mtp_descriptors
        .iter()
        .map(|desc| (desc.shard_idx, desc.data_offset, desc.n_bytes))
        .collect::<HashSet<_>>();
    let mtp_descriptor_bytes = mtp_descriptors.iter().try_fold(0u64, |total, desc| {
        total
            .checked_add(desc.n_bytes)
            .ok_or_else(|| anyhow!("MTP descriptor byte accounting overflow"))
    })?;
    let unbound = gguf
        .tensors
        .iter()
        .filter(|desc| {
            let key = (desc.shard_idx, desc.data_offset, desc.n_bytes);
            !bound_sources.contains(&key) && !mtp_sources.contains(&key)
        })
        .collect::<Vec<_>>();
    let unbound_bytes = unbound.iter().try_fold(0u64, |total, desc| {
        total
            .checked_add(desc.n_bytes)
            .ok_or_else(|| anyhow!("unbound byte accounting overflow"))
    })?;
    let row = serde_json::json!({
        "schema_version": 1,
        "model": args.model,
        "architecture": gguf.architecture(),
        "descriptor_layout_digest": format!("{:#018x}", gguf_descriptor_layout_digest(&gguf)),
        "shard_mapped_lengths": shard_mapped_lengths,
        "tensor_descriptors": gguf.tensors.len(),
        "base_weight_requests": requests.len(),
        "mtp_present": model.mtp.is_some(),
        "mtp_descriptor_count": mtp_descriptors.len(),
        "mtp_descriptor_bytes": mtp_descriptor_bytes,
        "native_quant_embedding": native_embedding,
        "embedding_policy": args.embedding_policy.label(),
        "router_f16": args.router_f16,
        "page_size": page_size,
        "required_alignment": args.alignment,
        "device_max_buffer_length": device_max_buffer_length,
        "planned_max_buffer_length": max_buffer_length,
        "usable_window_length": plan.usable_window_length,
        "windows": windows,
        "sum_window_lengths": mapped_window_bytes,
        "direct_logical_count": direct.len(),
        "direct_logical_bytes": direct_logical_bytes,
        "direct_unique_view_count": unique_view_count,
        "direct_unique_view_bytes": plan.unique_view_bytes,
        "direct_logical_view_bytes": plan.logical_view_bytes,
        "direct_alias_count": alias_count,
        "direct_alias_bytes": plan.alias_bytes,
        "fallback_unique_count": fallback_rows.len(),
        "fallback_unique_bytes": plan.unique_fallback_bytes,
        "fallbacks": fallback_rows,
        "converted_count": converted_count,
        "converted_source_bytes": converted_source_bytes,
        "converted_resident_bytes": converted_resident_bytes,
        "copied_base_weight_private_bytes_under_policy": current_base_weight_private_bytes,
        "planned_base_weight_private_bytes_under_policy": planned_base_weight_private_bytes,
        "estimated_base_weight_private_bytes_removed_under_policy":
            estimated_base_weight_private_bytes_removed,
        "unbound_descriptor_count": unbound.len(),
        "unbound_descriptor_bytes": unbound_bytes,
        "build_identity": qwen_build_identity_packet(),
    });
    match args.output {
        OutputFormat::Json => println!("{}", serde_json::to_string_pretty(&row)?),
        OutputFormat::Text => {
            println!("model\t{}", args.model.display());
            println!("shards\t{}", gguf.shard_count());
            println!("windows\t{}", plan.windows.len());
            println!("direct_view_bytes\t{}", plan.logical_view_bytes);
            println!("fallback_bytes\t{}", plan.unique_fallback_bytes);
            println!("converted_resident_bytes\t{converted_resident_bytes}");
            println!(
                "estimated_base_weight_private_bytes_removed_under_policy\t{}",
                estimated_base_weight_private_bytes_removed
            );
        }
    }
    Ok(())
}

fn argmax_i32(logits: &[f32]) -> i32 {
    let mut best = (0usize, f32::NEG_INFINITY);
    for (i, &v) in logits.iter().enumerate() {
        if v > best.1 {
            best = (i, v);
        }
    }
    best.0 as i32
}
