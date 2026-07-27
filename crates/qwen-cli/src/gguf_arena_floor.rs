use crate::OutputFormat;
use anyhow::{Context, Result, anyhow};
use clap::{Parser, ValueEnum};
use objc2::{
    rc::{Retained, Weak, autoreleasepool},
    runtime::ProtocolObject,
};
use objc2_metal::{
    MTLBuffer, MTLCPUCacheMode, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandQueue,
    MTLDevice, MTLHazardTrackingMode, MTLResource, MTLStorageMode,
};
use qwen_llm::{
    gguf::GgufFile,
    loader::Model,
    metal::{
        BlitEncoder, Buffer, DiagnosticGgufBlitReleaseProbe, MetalContext,
        RetainedStorageDisposition, RetainedStorageFallback, RetainedStoragePlan,
        host_page_size_bytes, plan_retained_storage,
    },
    metal_forward::{
        ModelWeightStorageKind, gguf_descriptor_layout_digest,
        model_weight_storage_inventory_digest, model_weight_storage_requests,
        native_quant_embedding_storage_supported,
        production_native_quant_embedding_storage_enabled, retained_storage_plan_digest,
    },
    model::{Arch, ArchKind},
    tensor::TensorDesc,
};
use serde_json::{Value, json};
use std::collections::HashSet;
use std::mem::MaybeUninit;
use std::path::PathBuf;
use std::time::{Duration, Instant};

const REQUIRED_ALIGNMENT: usize = 32;
const EXPECTED_PAGE_SIZE: usize = 16_384;
const EXPECTED_MAX_BUFFER_LENGTH: usize = 77_309_411_328;
const FROZEN_PARALLEL_COPY_WORKERS: usize = 4;
const PARALLEL_COPY_ALGORITHM: &str = "minimax-contiguous-v1";
const ALLOWED_DIAGNOSTIC_WORKERS: [usize; 6] = [1, 2, 4, 6, 8, 12];
const A3B_BLIT_PLAN_DIGEST: &str =
    "fa2685e223ad8ea6271c6061041fe8d996b4e6cc70e060588b750732577c92af";
const A3B_BLIT_WINDOW_OFFSET: u64 = 10_977_280;
const A3B_BLIT_WINDOW_BYTES: usize = 22_123_544_576;
const A3B_BLIT_WINDOW_LOGICAL_BYTES: u64 = 22_123_530_752;
const A3B_BLIT_WINDOW_GAP_BYTES: u64 = 13_824;
const A3B_BLIT_FALLBACK_REQUEST: usize = 721;
const A3B_BLIT_FALLBACK_BYTES: u64 = 8_192;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, ValueEnum)]
enum FloorProfileId {
    #[value(name = "a3b-q4km-v1")]
    A3bQ4kmV1,
    #[value(name = "dense27b-q4km-v1")]
    Dense27bQ4kmV1,
}

impl FloorProfileId {
    fn label(self) -> &'static str {
        match self {
            Self::A3bQ4kmV1 => "a3b-q4km-v1",
            Self::Dense27bQ4kmV1 => "dense27b-q4km-v1",
        }
    }
}

#[derive(Clone, Copy)]
struct ScheduleIdentity {
    request_index: usize,
    name: &'static str,
    shard_idx: usize,
    source_offset: u64,
    n_bytes: u64,
}

#[derive(Clone, Copy)]
struct ScheduleBoundary {
    first: ScheduleIdentity,
    last: ScheduleIdentity,
}

struct FloorProfile {
    id: FloorProfileId,
    architecture: &'static str,
    arch: Arch,
    tied_embeddings: bool,
    mtp_present: bool,
    shard_mapped_lengths: &'static [usize],
    descriptor_digest: &'static str,
    inventory_digest: &'static str,
    request_count: usize,
    logical_copy_bytes: u64,
    device_name: &'static str,
    cuts: [usize; FROZEN_PARALLEL_COPY_WORKERS - 1],
    task_counts: [usize; FROZEN_PARALLEL_COPY_WORKERS],
    worker_bytes: [u64; FROZEN_PARALLEL_COPY_WORKERS],
    boundaries: [ScheduleBoundary; FROZEN_PARALLEL_COPY_WORKERS],
}

const A3B_ARCH: Arch = Arch {
    kind: ArchKind::Moe,
    n_layer: 40,
    hidden_size: 2048,
    intermediate_size: 0,
    vocab_size: 248_320,
    full_attention_interval: 4,
    n_q_heads: 16,
    n_kv_heads: 2,
    attn_head_dim: 256,
    rope_theta: 10_000_000.0,
    partial_rotary_factor: 0.25,
    gdn_n_v_heads: 32,
    gdn_n_k_heads: 16,
    gdn_head_dim: 128,
    gdn_conv_kernel: 4,
    expert_count: 256,
    expert_used_count: 8,
    expert_feed_forward_length: 512,
    expert_shared_feed_forward_length: 512,
    mtp_n_hidden_layers: 0,
};

const DENSE27B_ARCH: Arch = Arch {
    kind: ArchKind::Dense,
    n_layer: 64,
    hidden_size: 5120,
    intermediate_size: 17_408,
    vocab_size: 248_320,
    full_attention_interval: 4,
    n_q_heads: 24,
    n_kv_heads: 4,
    attn_head_dim: 256,
    rope_theta: 10_000_000.0,
    partial_rotary_factor: 0.25,
    gdn_n_v_heads: 48,
    gdn_n_k_heads: 16,
    gdn_head_dim: 128,
    gdn_conv_kernel: 4,
    expert_count: 0,
    expert_used_count: 0,
    expert_feed_forward_length: 0,
    expert_shared_feed_forward_length: 0,
    mtp_n_hidden_layers: 0,
};

const FLOOR_PROFILES: [FloorProfile; 2] = [
    FloorProfile {
        id: FloorProfileId::A3bQ4kmV1,
        architecture: "qwen35moe",
        arch: A3B_ARCH,
        tied_embeddings: false,
        mtp_present: false,
        shard_mapped_lengths: &[22_134_528_992],
        descriptor_digest: "0x5ae645df5cf7d568",
        inventory_digest: "f57153febec22463c7789b892d4d084041d722483a93191c81c40ab86be7d9e5",
        request_count: 733,
        logical_copy_bytes: 22_123_538_944,
        device_name: "Apple M4 Max",
        cuts: [155, 359, 539],
        task_counts: [155, 204, 180, 194],
        worker_bytes: [5_532_746_240, 5_462_315_776, 5_595_522_304, 5_532_954_624],
        boundaries: [
            ScheduleBoundary {
                first: ScheduleIdentity {
                    request_index: 2,
                    name: "output.weight",
                    shard_idx: 0,
                    source_offset: 10_990_048,
                    n_bytes: 417_177_600,
                },
                last: ScheduleIdentity {
                    request_index: 164,
                    name: "blk.8.ffn_gate_exps.weight",
                    shard_idx: 0,
                    source_offset: 5_392_741_344,
                    n_bytes: 150_994_944,
                },
            },
            ScheduleBoundary {
                first: ScheduleIdentity {
                    request_index: 163,
                    name: "blk.8.ffn_gate_inp.weight",
                    shard_idx: 0,
                    source_offset: 5_543_736_288,
                    n_bytes: 2_097_152,
                },
                last: ScheduleIdentity {
                    request_index: 354,
                    name: "blk.19.attn_v.weight",
                    shard_idx: 0,
                    source_offset: 11_004_937_952,
                    n_bytes: 1_114_112,
                },
            },
            ScheduleBoundary {
                first: ScheduleIdentity {
                    request_index: 366,
                    name: "blk.19.ffn_down_exps.weight",
                    shard_idx: 0,
                    source_offset: 11_006_052_064,
                    n_bytes: 184_549_376,
                },
                last: ScheduleIdentity {
                    request_index: 548,
                    name: "blk.29.ffn_gate_exps.weight",
                    shard_idx: 0,
                    source_offset: 16_450_579_424,
                    n_bytes: 150_994_944,
                },
            },
            ScheduleBoundary {
                first: ScheduleIdentity {
                    request_index: 547,
                    name: "blk.29.ffn_gate_inp.weight",
                    shard_idx: 0,
                    source_offset: 16_601_574_368,
                    n_bytes: 2_097_152,
                },
                last: ScheduleIdentity {
                    request_index: 721,
                    name: "blk.39.post_attention_norm.weight",
                    shard_idx: 0,
                    source_offset: 22_134_520_800,
                    n_bytes: 8_192,
                },
            },
        ],
    },
    FloorProfile {
        id: FloorProfileId::Dense27bQ4kmV1,
        architecture: "qwen35",
        arch: DENSE27B_ARCH,
        tied_embeddings: false,
        mtp_present: false,
        shard_mapped_lengths: &[16_817_244_384],
        descriptor_digest: "0xd116405fd99f54d9",
        inventory_digest: "50e9af4e4f590fc85687a71f5602ce035e7fdf0e2a31e928b2c7a2be10458a07",
        request_count: 851,
        logical_copy_bytes: 16_806_250_496,
        device_name: "Apple M4 Max",
        cuts: [136, 377, 618],
        task_counts: [136, 241, 241, 233],
        worker_bytes: [4_194_110_464, 4_214_375_808, 4_204_933_376, 4_192_830_848],
        boundaries: [
            ScheduleBoundary {
                first: ScheduleIdentity {
                    request_index: 2,
                    name: "output.weight",
                    shard_idx: 0,
                    source_offset: 10_993_888,
                    n_bytes: 1_042_944_000,
                },
                last: ScheduleIdentity {
                    request_index: 135,
                    name: "blk.9.ssm_norm.weight",
                    shard_idx: 0,
                    source_offset: 4_205_103_840,
                    n_bytes: 512,
                },
            },
            ScheduleBoundary {
                first: ScheduleIdentity {
                    request_index: 136,
                    name: "blk.9.ssm_out.weight",
                    shard_idx: 0,
                    source_offset: 4_205_104_352,
                    n_bytes: 21_626_880,
                },
                last: ScheduleIdentity {
                    request_index: 379,
                    name: "blk.28.attn_qkv.weight",
                    shard_idx: 0,
                    source_offset: 8_376_472_160,
                    n_bytes: 43_008_000,
                },
            },
            ScheduleBoundary {
                first: ScheduleIdentity {
                    request_index: 378,
                    name: "blk.28.ffn_down.weight",
                    shard_idx: 0,
                    source_offset: 8_419_480_160,
                    n_bytes: 73_113_600,
                },
                last: ScheduleIdentity {
                    request_index: 618,
                    name: "blk.46.ffn_down.weight",
                    shard_idx: 0,
                    source_offset: 12_551_299_936,
                    n_bytes: 73_113_600,
                },
            },
            ScheduleBoundary {
                first: ScheduleIdentity {
                    request_index: 616,
                    name: "blk.46.ffn_gate.weight",
                    shard_idx: 0,
                    source_offset: 12_624_413_536,
                    n_bytes: 50_135_040,
                },
                last: ScheduleIdentity {
                    request_index: 844,
                    name: "blk.63.post_attention_norm.weight",
                    shard_idx: 0,
                    source_offset: 16_817_223_904,
                    n_bytes: 20_480,
                },
            },
        ],
    },
];

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum ArenaFloorArm {
    Copied,
    ParallelCopied,
    ParallelPread,
    TransientMmapBlit,
    ArenaSerial,
    ArenaFour,
}

impl ArenaFloorArm {
    fn label(self) -> &'static str {
        match self {
            Self::Copied => "copied",
            Self::ParallelCopied => "parallel-copied",
            Self::ParallelPread => "parallel-pread",
            Self::TransientMmapBlit => "transient-mmap-blit",
            Self::ArenaSerial => "arena-serial",
            Self::ArenaFour => "arena-four",
        }
    }

    fn uses_parallel_schedule(self) -> bool {
        matches!(self, Self::ParallelCopied | Self::ParallelPread)
    }
}

#[derive(Parser, Debug)]
pub(crate) struct GgufArenaFloorArgs {
    /// Path to the first GGUF shard.
    #[arg(short = 'm', long)]
    model: PathBuf,
    /// Exact authenticated materialization profile.
    #[arg(long, value_enum, required_unless_present = "describe")]
    profile: Option<FloorProfileId>,
    /// Materialization arm.
    #[arg(long, value_enum, required_unless_present = "describe")]
    arm: Option<ArenaFloorArm>,
    /// Emit authenticated geometry without touching payload bytes.
    #[arg(long)]
    describe: bool,
    /// Diagnostic population workers: exactly one of 1, 2, 4, 6, 8, or 12.
    #[arg(long, default_value_t = FROZEN_PARALLEL_COPY_WORKERS, value_parser = parse_worker_count)]
    workers: usize,
    /// `text` or `json`.
    #[arg(short = 'o', long, value_enum, default_value = "json")]
    output: OutputFormat,
}

fn parse_worker_count(value: &str) -> std::result::Result<usize, String> {
    let workers = value
        .parse::<usize>()
        .map_err(|_| format!("invalid worker count {value:?}"))?;
    if ALLOWED_DIAGNOSTIC_WORKERS.contains(&workers) {
        Ok(workers)
    } else {
        Err(format!(
            "worker count must be one of 1, 2, 4, 6, 8, or 12 (got {workers})"
        ))
    }
}

fn validate_worker_scope(
    workers: usize,
    describe: bool,
    arm: Option<ArenaFloorArm>,
    profile: Option<FloorProfileId>,
) -> Result<()> {
    if !ALLOWED_DIAGNOSTIC_WORKERS.contains(&workers) {
        return Err(anyhow!("unsupported diagnostic worker count {workers}"));
    }
    if !describe
        && arm == Some(ArenaFloorArm::TransientMmapBlit)
        && profile != Some(FloorProfileId::A3bQ4kmV1)
    {
        return Err(anyhow!(
            "transient mmap-blit requires --profile a3b-q4km-v1"
        ));
    }
    if workers == FROZEN_PARALLEL_COPY_WORKERS || describe {
        return Ok(());
    }
    if arm == Some(ArenaFloorArm::ParallelPread) && profile == Some(FloorProfileId::A3bQ4kmV1) {
        Ok(())
    } else {
        Err(anyhow!(
            "non-default --workers is diagnostic-only and requires --arm parallel-pread --profile a3b-q4km-v1 or authenticated A3B describe geometry"
        ))
    }
}

fn validate_describe_worker_scope(
    workers: usize,
    authenticated_profile: Option<FloorProfileId>,
) -> Result<()> {
    if workers == FROZEN_PARALLEL_COPY_WORKERS
        || authenticated_profile == Some(FloorProfileId::A3bQ4kmV1)
    {
        Ok(())
    } else {
        Err(anyhow!(
            "non-default --workers with --describe requires geometry authenticated as a3b-q4km-v1"
        ))
    }
}

#[derive(Clone, Copy)]
struct Usage {
    minor_faults: i64,
    major_faults: i64,
    user_time_us: i64,
    system_time_us: i64,
}

#[derive(Clone, Copy)]
struct ProcUsage {
    instructions: u64,
    cycles: u64,
    billed_energy: u64,
    serviced_energy: u64,
}

#[derive(Clone)]
struct Binding {
    buffer: Buffer,
    resource_index: usize,
    offset: usize,
    length: usize,
}

struct Materialized {
    resources: Vec<Buffer>,
    bindings: Vec<Binding>,
    allocation_wall: Option<Duration>,
    source_resolution_wall: Option<Duration>,
    copy_wall: Option<Duration>,
    source_release_wall: Option<Duration>,
    binding_wall: Duration,
    worker_count: usize,
    schedule: Option<ParallelCopySchedule>,
    blit_population: Option<BlitPopulation>,
    ready_wall: Duration,
}

#[derive(Clone, Copy)]
struct BlitPopulation {
    order_count: usize,
    order_first_request: usize,
    order_last_request: usize,
    source_window_count: usize,
    source_window_bytes: u64,
    window_blit_count: usize,
    window_blit_bytes: u64,
    source_window_gap_bytes: u64,
    fallback_source_count: usize,
    fallback_source_bytes: u64,
    cpu_staging_copy_bytes: u64,
    blit_count: usize,
    blit_bytes: u64,
    command_buffer_count: usize,
    blit_encoder_count: usize,
    commit_count: usize,
    wait_count: usize,
    command_error_count: usize,
    source_window_deallocator_calls: usize,
    source_window_deallocator_mismatches: usize,
    source_buffers_alive: usize,
    retained_references: bool,
    command_status: MTLCommandBufferStatus,
    gpu_start_time: Option<f64>,
    gpu_end_time: Option<f64>,
    gpu_wall_ms: Option<f64>,
    allocated_after_destinations: u64,
    allocated_with_sources: u64,
    allocated_after_source_release: u64,
}

struct Correctness {
    payload_bytes_checked: u64,
    entries_checked: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ParallelCopyPartition {
    start: usize,
    end: usize,
    bytes: u64,
    first_shard: usize,
    first_source_offset: u64,
    last_shard: usize,
    last_source_offset: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ParallelCopySchedule {
    sorted_request_indices: Vec<usize>,
    cuts: Vec<usize>,
    partitions: Vec<ParallelCopyPartition>,
}

struct ParallelCopyTask<'a> {
    source: &'a [u8],
    destination: &'a mut [u8],
}

struct ParallelPreadTask<'a> {
    shard_idx: usize,
    source_offset: u64,
    destination: &'a mut [u8],
}

fn capture_usage() -> Result<Usage> {
    let mut usage = MaybeUninit::<libc::rusage>::zeroed();
    let rc = unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) };
    if rc != 0 {
        return Err(std::io::Error::last_os_error()).context("getrusage");
    }
    let usage = unsafe { usage.assume_init() };
    Ok(Usage {
        minor_faults: usage.ru_minflt,
        major_faults: usage.ru_majflt,
        user_time_us: timeval_us(usage.ru_utime)?,
        system_time_us: timeval_us(usage.ru_stime)?,
    })
}

fn timeval_us(value: libc::timeval) -> Result<i64> {
    value
        .tv_sec
        .checked_mul(1_000_000)
        .and_then(|seconds| seconds.checked_add(i64::from(value.tv_usec)))
        .ok_or_else(|| anyhow!("getrusage time overflow"))
}

fn capture_proc_usage() -> Result<ProcUsage> {
    let mut usage = MaybeUninit::<libc::rusage_info_v4>::zeroed();
    let rc = unsafe {
        libc::proc_pid_rusage(
            libc::getpid(),
            libc::RUSAGE_INFO_V4,
            usage.as_mut_ptr().cast::<libc::rusage_info_t>(),
        )
    };
    if rc != 0 {
        return Err(std::io::Error::last_os_error()).context("proc_pid_rusage v4");
    }
    let usage = unsafe { usage.assume_init() };
    Ok(ProcUsage {
        instructions: usage.ri_instructions,
        cycles: usage.ri_cycles,
        billed_energy: usage.ri_billed_energy,
        serviced_energy: usage.ri_serviced_energy,
    })
}

fn duration_ms(value: Duration) -> f64 {
    value.as_secs_f64() * 1e3
}

fn duration_us(value: Duration) -> Result<u64> {
    u64::try_from(value.as_micros()).map_err(|_| anyhow!("duration does not fit u64 microseconds"))
}

fn minimum_contiguous_groups(lengths: &[u64], capacity: u64) -> Option<usize> {
    let mut groups = 0usize;
    let mut group_bytes = 0u64;
    for &length in lengths {
        if length == 0 || length > capacity {
            return None;
        }
        if group_bytes > capacity - length {
            groups = groups.checked_add(1)?;
            group_bytes = length;
        } else {
            group_bytes += length;
        }
    }
    groups.checked_add(usize::from(group_bytes > 0))
}

fn can_partition_exactly(lengths: &[u64], groups: usize, capacity: u64) -> bool {
    groups > 0
        && lengths.len() >= groups
        && minimum_contiguous_groups(lengths, capacity).is_some_and(|minimum| minimum <= groups)
}

fn minimax_partition_cuts(lengths: &[u64], workers: usize) -> Result<Vec<usize>> {
    if workers == 0 || lengths.len() < workers || lengths.contains(&0) {
        return Err(anyhow!(
            "parallel copy requires at least {workers} nonempty tasks and at least one worker"
        ));
    }
    let mut total = 0u64;
    let mut lower = 0u64;
    for &length in lengths {
        total = total
            .checked_add(length)
            .ok_or_else(|| anyhow!("parallel-copy byte total overflow"))?;
        lower = lower.max(length);
    }
    let mut upper = total;
    while lower < upper {
        let midpoint = lower + (upper - lower) / 2;
        if can_partition_exactly(lengths, workers, midpoint) {
            upper = midpoint;
        } else {
            lower = midpoint + 1;
        }
    }

    let capacity = lower;
    let mut cuts = vec![0usize; workers - 1];
    let mut start = 0usize;
    for (cut_index, cut) in cuts.iter_mut().enumerate() {
        let remaining_groups = workers - cut_index - 1;
        let latest_cut = lengths.len() - remaining_groups;
        let mut group_bytes = 0u64;
        let mut selected = None;
        for candidate in start + 1..=latest_cut {
            let length = lengths[candidate - 1];
            if group_bytes > capacity - length {
                break;
            }
            group_bytes += length;
            if can_partition_exactly(&lengths[candidate..], remaining_groups, capacity) {
                selected = Some(candidate);
                break;
            }
        }
        *cut = selected.ok_or_else(|| anyhow!("parallel-copy cut is unavailable"))?;
        start = *cut;
    }
    if !can_partition_exactly(&lengths[start..], 1, capacity) {
        return Err(anyhow!("parallel-copy final partition exceeds optimum"));
    }
    Ok(cuts)
}

fn source_order(direct: &[&TensorDesc]) -> Vec<usize> {
    let mut request_indices = (0..direct.len()).collect::<Vec<_>>();
    request_indices.sort_by_key(|&request_index| {
        let desc = direct[request_index];
        (desc.shard_idx, desc.data_offset, request_index)
    });
    request_indices
}

fn parallel_copy_schedule(direct: &[&TensorDesc], workers: usize) -> Result<ParallelCopySchedule> {
    let sorted_request_indices = source_order(direct);
    let lengths = sorted_request_indices
        .iter()
        .map(|&request_index| direct[request_index].n_bytes)
        .collect::<Vec<_>>();
    let cuts = minimax_partition_cuts(&lengths, workers)?;
    let boundaries = std::iter::once(0)
        .chain(cuts.iter().copied())
        .chain(std::iter::once(direct.len()))
        .collect::<Vec<_>>();
    let mut partitions = Vec::with_capacity(workers);
    for worker in 0..workers {
        let start = boundaries[worker];
        let end = boundaries[worker + 1];
        let bytes = lengths[start..end]
            .iter()
            .try_fold(0u64, |total, &length| {
                total
                    .checked_add(length)
                    .ok_or_else(|| anyhow!("parallel-copy partition byte overflow"))
            })?;
        let first = direct[sorted_request_indices[start]];
        let last = direct[sorted_request_indices[end - 1]];
        partitions.push(ParallelCopyPartition {
            start,
            end,
            bytes,
            first_shard: first.shard_idx,
            first_source_offset: first.data_offset,
            last_shard: last.shard_idx,
            last_source_offset: last.data_offset,
        });
    }
    let schedule = ParallelCopySchedule {
        sorted_request_indices,
        cuts,
        partitions,
    };
    validate_parallel_copy_schedule(&schedule, direct, workers)?;
    Ok(schedule)
}

fn validate_parallel_copy_schedule(
    schedule: &ParallelCopySchedule,
    direct: &[&TensorDesc],
    expected_workers: usize,
) -> Result<()> {
    if !ALLOWED_DIAGNOSTIC_WORKERS.contains(&expected_workers)
        || schedule.partitions.len() != expected_workers
        || schedule.cuts.len() != expected_workers - 1
        || direct.len() < expected_workers
        || schedule.sorted_request_indices.len() != direct.len()
    {
        return Err(anyhow!(
            "parallel-copy schedule worker or task count drifted"
        ));
    }

    let expected_order = source_order(direct);
    if schedule.sorted_request_indices != expected_order {
        return Err(anyhow!(
            "parallel-copy schedule is not the deterministic request permutation"
        ));
    }

    let mut boundaries = Vec::with_capacity(expected_workers + 1);
    boundaries.push(0);
    boundaries.extend(schedule.cuts.iter().copied());
    boundaries.push(direct.len());
    if boundaries.iter().any(|&boundary| boundary > direct.len())
        || boundaries.windows(2).any(|pair| pair[0] >= pair[1])
    {
        return Err(anyhow!(
            "parallel-copy schedule has an empty or unordered partition"
        ));
    }

    let mut total_bytes = 0u64;
    for (worker, (partition, pair)) in schedule
        .partitions
        .iter()
        .zip(boundaries.windows(2))
        .enumerate()
    {
        let start = pair[0];
        let end = pair[1];
        if partition.start != start || partition.end != end {
            return Err(anyhow!(
                "parallel-copy worker {worker} partition has an overlap or gap"
            ));
        }
        let request_indices = schedule
            .sorted_request_indices
            .get(start..end)
            .ok_or_else(|| anyhow!("parallel-copy worker {worker} partition is out of range"))?;
        let bytes = request_indices
            .iter()
            .try_fold(0u64, |total, &request_index| {
                total
                    .checked_add(direct[request_index].n_bytes)
                    .ok_or_else(|| anyhow!("parallel-copy partition byte overflow"))
            })?;
        let first = direct[request_indices[0]];
        let last = direct[*request_indices.last().expect("partition is nonempty")];
        if bytes == 0
            || partition.bytes != bytes
            || partition.first_shard != first.shard_idx
            || partition.first_source_offset != first.data_offset
            || partition.last_shard != last.shard_idx
            || partition.last_source_offset != last.data_offset
        {
            return Err(anyhow!(
                "parallel-copy worker {worker} bytes or boundaries drifted"
            ));
        }
        total_bytes = total_bytes
            .checked_add(bytes)
            .ok_or_else(|| anyhow!("parallel-copy schedule byte overflow"))?;
    }
    let expected_bytes = direct.iter().try_fold(0u64, |total, desc| {
        total
            .checked_add(desc.n_bytes)
            .ok_or_else(|| anyhow!("parallel-copy request byte overflow"))
    })?;
    if total_bytes != expected_bytes {
        return Err(anyhow!("parallel-copy schedule task union is incomplete"));
    }
    Ok(())
}

fn schedule_identity_matches(
    expected: ScheduleIdentity,
    request_index: usize,
    desc: &TensorDesc,
) -> bool {
    expected.request_index == request_index
        && expected.name == desc.name
        && expected.shard_idx == desc.shard_idx
        && expected.source_offset == desc.data_offset
        && expected.n_bytes == desc.n_bytes
}

fn frozen_parallel_copy_schedule(
    profile: &FloorProfile,
    direct: &[&TensorDesc],
) -> Result<ParallelCopySchedule> {
    if direct.len() != profile.request_count || direct.iter().any(|desc| desc.n_bytes == 0) {
        return Err(anyhow!("floor profile request count or length drifted"));
    }
    let sorted_request_indices = source_order(direct);
    let mut permutation = sorted_request_indices.clone();
    permutation.sort_unstable();
    if permutation.iter().copied().ne(0..direct.len()) {
        return Err(anyhow!("floor profile schedule is not a permutation"));
    }

    let boundaries = [
        0,
        profile.cuts[0],
        profile.cuts[1],
        profile.cuts[2],
        direct.len(),
    ];
    if boundaries[FROZEN_PARALLEL_COPY_WORKERS] != direct.len()
        || boundaries.iter().any(|&boundary| boundary > direct.len())
        || boundaries.windows(2).any(|pair| pair[0] >= pair[1])
    {
        return Err(anyhow!("floor profile schedule has an empty partition"));
    }
    let mut partitions = Vec::with_capacity(FROZEN_PARALLEL_COPY_WORKERS);
    let mut total_bytes = 0u64;
    for worker in 0..FROZEN_PARALLEL_COPY_WORKERS {
        let start = boundaries[worker];
        let end = boundaries[worker + 1];
        let partition = sorted_request_indices
            .get(start..end)
            .ok_or_else(|| anyhow!("floor profile partition {worker} is out of range"))?;
        let bytes = partition.iter().try_fold(0u64, |total, &request_index| {
            total
                .checked_add(direct[request_index].n_bytes)
                .ok_or_else(|| anyhow!("floor profile partition bytes overflow"))
        })?;
        let first_request_index = partition[0];
        let last_request_index = *partition.last().expect("partition is nonempty");
        if end - start != profile.task_counts[worker]
            || bytes != profile.worker_bytes[worker]
            || !schedule_identity_matches(
                profile.boundaries[worker].first,
                first_request_index,
                direct[first_request_index],
            )
            || !schedule_identity_matches(
                profile.boundaries[worker].last,
                last_request_index,
                direct[last_request_index],
            )
        {
            return Err(anyhow!("floor profile partition {worker} drifted"));
        }
        total_bytes = total_bytes
            .checked_add(bytes)
            .ok_or_else(|| anyhow!("floor profile schedule bytes overflow"))?;
        let first = direct[first_request_index];
        let last = direct[last_request_index];
        partitions.push(ParallelCopyPartition {
            start,
            end,
            bytes,
            first_shard: first.shard_idx,
            first_source_offset: first.data_offset,
            last_shard: last.shard_idx,
            last_source_offset: last.data_offset,
        });
    }
    if total_bytes != profile.logical_copy_bytes {
        return Err(anyhow!("floor profile schedule byte total drifted"));
    }
    let schedule = ParallelCopySchedule {
        sorted_request_indices,
        cuts: profile.cuts.to_vec(),
        partitions,
    };
    validate_parallel_copy_schedule(&schedule, direct, FROZEN_PARALLEL_COPY_WORKERS)?;
    Ok(schedule)
}

struct ProfileFacts<'a> {
    gguf: &'a GgufFile,
    model: &'a Model<'a>,
    native_embedding: bool,
    direct: &'a [&'a TensorDesc],
    page_size: usize,
    max_buffer_length: usize,
    device_name: &'a str,
    unified_memory: bool,
    descriptor_digest: &'a str,
    inventory_digest: &'a str,
    logical_copy_bytes: u64,
}

fn profile_metadata_matches(profile: &FloorProfile, facts: &ProfileFacts<'_>) -> bool {
    facts.gguf.architecture().as_deref() == Some(profile.architecture)
        && facts.model.arch == profile.arch
        && facts.model.tied_embeddings == profile.tied_embeddings
        && facts.model.mtp.is_some() == profile.mtp_present
        && facts.native_embedding
        && facts.gguf.shard_mapped_lengths() == profile.shard_mapped_lengths
        && facts.descriptor_digest == profile.descriptor_digest
        && facts.inventory_digest == profile.inventory_digest
        && facts.direct.len() == profile.request_count
        && facts.direct.iter().all(|desc| desc.n_bytes > 0)
        && facts.logical_copy_bytes == profile.logical_copy_bytes
        && facts.page_size == EXPECTED_PAGE_SIZE
        && facts.max_buffer_length == EXPECTED_MAX_BUFFER_LENGTH
        && facts.device_name == profile.device_name
        && facts.unified_memory
        && REQUIRED_ALIGNMENT == 32
}

fn authenticated_profile_matches(
    facts: &ProfileFacts<'_>,
    direct: &[&TensorDesc],
) -> Result<Vec<(&'static FloorProfile, ParallelCopySchedule)>> {
    let mut ids = HashSet::with_capacity(FLOOR_PROFILES.len());
    let mut matches = Vec::new();
    for profile in &FLOOR_PROFILES {
        if !ids.insert(profile.id) {
            return Err(anyhow!("floor profile table contains a duplicate ID"));
        }
        if profile_metadata_matches(profile, facts) {
            let schedule = frozen_parallel_copy_schedule(profile, direct)?;
            matches.push((profile, schedule));
        }
    }
    Ok(matches)
}

fn validate_source_endpoints(gguf: &GgufFile, direct: &[&TensorDesc]) -> Result<()> {
    let shard_lengths = gguf.shard_mapped_lengths();
    for (request_index, desc) in direct.iter().enumerate() {
        let shard_length = *shard_lengths
            .get(desc.shard_idx)
            .ok_or_else(|| anyhow!("source request {request_index} shard is unavailable"))?
            as u64;
        let endpoint = desc
            .data_offset
            .checked_add(desc.n_bytes)
            .ok_or_else(|| anyhow!("source request {request_index} endpoint overflow"))?;
        if desc.n_bytes == 0 || endpoint > shard_length {
            return Err(anyhow!(
                "source request {request_index} exceeds its mapped shard"
            ));
        }
    }
    Ok(())
}

fn schedule_boundary_json(request_index: usize, desc: &TensorDesc) -> Value {
    json!({
        "request_index": request_index,
        "name": desc.name,
        "shard_idx": desc.shard_idx,
        "source_offset": desc.data_offset,
        "n_bytes": desc.n_bytes,
    })
}

fn parallel_copy_schedule_json(
    schedule: &ParallelCopySchedule,
    direct: &[&TensorDesc],
) -> Result<Value> {
    let workers = schedule.partitions.len();
    validate_parallel_copy_schedule(schedule, direct, workers)?;
    let total_bytes = schedule
        .partitions
        .iter()
        .try_fold(0u64, |total, partition| {
            total
                .checked_add(partition.bytes)
                .ok_or_else(|| anyhow!("parallel-copy schedule byte overflow"))
        })?;
    let min_bytes = schedule
        .partitions
        .iter()
        .map(|partition| partition.bytes)
        .min()
        .expect("validated schedule has partitions");
    let max_bytes = schedule
        .partitions
        .iter()
        .map(|partition| partition.bytes)
        .max()
        .expect("validated schedule has partitions");
    let partitions = schedule
        .partitions
        .iter()
        .map(|partition| {
            let first_request_index = *schedule
                .sorted_request_indices
                .get(partition.start)
                .ok_or_else(|| anyhow!("parallel-copy first boundary is unavailable"))?;
            let last_request_index = *schedule
                .sorted_request_indices
                .get(
                    partition
                        .end
                        .checked_sub(1)
                        .ok_or_else(|| anyhow!("parallel-copy partition is empty"))?,
                )
                .ok_or_else(|| anyhow!("parallel-copy last boundary is unavailable"))?;
            let first = direct
                .get(first_request_index)
                .ok_or_else(|| anyhow!("parallel-copy first request is unavailable"))?;
            let last = direct
                .get(last_request_index)
                .ok_or_else(|| anyhow!("parallel-copy last request is unavailable"))?;
            Ok(json!({
                "start": partition.start,
                "end": partition.end,
                "task_count": partition.end - partition.start,
                "bytes": partition.bytes,
                "first_shard": partition.first_shard,
                "first_source_offset": partition.first_source_offset,
                "last_shard": partition.last_shard,
                "last_source_offset": partition.last_source_offset,
                "first": schedule_boundary_json(first_request_index, first),
                "last": schedule_boundary_json(last_request_index, last),
            }))
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(json!({
        "algorithm": PARALLEL_COPY_ALGORITHM,
        "workers": workers,
        "cuts": schedule.cuts,
        "task_counts": schedule.partitions.iter().map(|partition| {
            partition.end - partition.start
        }).collect::<Vec<_>>(),
        "worker_bytes": schedule.partitions.iter().map(|partition| {
            partition.bytes
        }).collect::<Vec<_>>(),
        "max_to_min": max_bytes as f64 / min_bytes as f64,
        "max_to_ideal": max_bytes as f64
            / (total_bytes as f64 / workers as f64),
        "partitions": partitions,
    }))
}

fn validate_copied_topology(
    profile: &FloorProfile,
    direct: &[&TensorDesc],
    resources: &[Buffer],
    bindings: &[Binding],
) -> Result<()> {
    if direct.len() != profile.request_count
        || resources.len() != profile.request_count
        || bindings.len() != profile.request_count
    {
        return Err(anyhow!("copied topology count drifted"));
    }
    let mut identities = HashSet::with_capacity(resources.len());
    let mut resource_bytes = 0u64;
    for (index, ((desc, resource), binding)) in
        direct.iter().zip(resources).zip(bindings).enumerate()
    {
        let identity = Retained::as_ptr(resource) as *const () as usize;
        resource_bytes = resource_bytes
            .checked_add(resource.length() as u64)
            .ok_or_else(|| anyhow!("copied topology byte overflow"))?;
        if !identities.insert(identity)
            || resource.length() as u64 != desc.n_bytes
            || resource.length() == 0
            || resource.storageMode() != MTLStorageMode::Shared
            || resource.cpuCacheMode() != MTLCPUCacheMode::DefaultCache
            || resource.hazardTrackingMode() != MTLHazardTrackingMode::Tracked
            || binding.resource_index != index
            || binding.offset != 0
            || binding.length != resource.length()
            || Retained::as_ptr(&binding.buffer) != Retained::as_ptr(resource)
        {
            return Err(anyhow!("copied topology resource {index} drifted"));
        }
    }
    if resource_bytes != profile.logical_copy_bytes {
        return Err(anyhow!("copied topology byte total drifted"));
    }
    Ok(())
}

unsafe fn exclusive_buffer_bytes_mut(buffer: &mut Buffer) -> &mut [u8] {
    // SAFETY: the caller proves the buffer is nonempty and CPU-accessible,
    // all destination ranges are pairwise disjoint and source-disjoint, and
    // this exclusive borrow outlives every mutable slice created from it.
    unsafe {
        std::slice::from_raw_parts_mut(buffer.contents().as_ptr().cast::<u8>(), buffer.length())
    }
}

fn allocate_copied_resources(ctx: &MetalContext, direct: &[&TensorDesc]) -> Result<Vec<Buffer>> {
    let mut resources = Vec::with_capacity(direct.len());
    for desc in direct {
        let length = usize::try_from(desc.n_bytes)
            .map_err(|_| anyhow!("tensor byte length does not fit usize"))?;
        resources.push(ctx.buffer_uninit(length)?);
    }
    Ok(resources)
}

fn validate_copied_destinations(
    resources: &[Buffer],
    direct: &[&TensorDesc],
    operation: &str,
) -> Result<Vec<(usize, usize)>> {
    if resources.len() != direct.len() {
        return Err(anyhow!("{operation} destination count drifted"));
    }
    let mut resource_identities = HashSet::with_capacity(resources.len());
    let mut destination_ranges = Vec::with_capacity(resources.len());
    for (request_index, buffer) in resources.iter().enumerate() {
        let identity = Retained::as_ptr(buffer) as *const () as usize;
        if buffer.length() == 0
            || buffer.storageMode() != MTLStorageMode::Shared
            || buffer.cpuCacheMode() != MTLCPUCacheMode::DefaultCache
            || buffer.hazardTrackingMode() != MTLHazardTrackingMode::Tracked
            || buffer.length() as u64 != direct[request_index].n_bytes
        {
            return Err(anyhow!(
                "{operation} destination resource {request_index} mode or length drifted"
            ));
        }
        let start = buffer.contents().as_ptr().cast::<u8>() as usize;
        let end = start
            .checked_add(buffer.length())
            .ok_or_else(|| anyhow!("{operation} destination range overflow"))?;
        if !resource_identities.insert(identity) || start == 0 {
            return Err(anyhow!(
                "{operation} destination resource {request_index} drifted"
            ));
        }
        destination_ranges.push((start, end));
    }
    let mut ranges_by_address = destination_ranges.clone();
    ranges_by_address.sort_unstable();
    if ranges_by_address
        .windows(2)
        .any(|pair| pair[0].1 > pair[1].0)
    {
        return Err(anyhow!("{operation} destination resources overlap"));
    }
    Ok(destination_ranges)
}

fn order_parallel_tasks<T>(
    tasks_by_request: &mut [Option<T>],
    schedule: &ParallelCopySchedule,
    operation: &str,
) -> Result<Vec<T>> {
    let mut tasks = Vec::with_capacity(tasks_by_request.len());
    for &request_index in &schedule.sorted_request_indices {
        tasks.push(
            tasks_by_request
                .get_mut(request_index)
                .ok_or_else(|| anyhow!("{operation} request {request_index} is out of range"))?
                .take()
                .ok_or_else(|| {
                    anyhow!("{operation} request {request_index} was assigned more than once")
                })?,
        );
    }
    if tasks_by_request.iter().any(Option::is_some) {
        return Err(anyhow!("{operation} task union is incomplete"));
    }
    Ok(tasks)
}

fn partition_parallel_tasks<'a, T>(
    tasks: &'a mut [T],
    schedule: &ParallelCopySchedule,
    operation: &str,
) -> Result<Vec<&'a mut [T]>> {
    let mut task_tail = tasks;
    let mut worker_partitions = Vec::with_capacity(schedule.partitions.len());
    let mut consumed = 0usize;
    for (worker, partition) in schedule.partitions.iter().enumerate() {
        if partition.start != consumed || partition.end > schedule.sorted_request_indices.len() {
            return Err(anyhow!("{operation} worker {worker} partition is invalid"));
        }
        let count = partition
            .end
            .checked_sub(partition.start)
            .ok_or_else(|| anyhow!("{operation} worker {worker} partition underflow"))?;
        if count == 0 || count > task_tail.len() {
            return Err(anyhow!(
                "{operation} worker {worker} partition extent is invalid"
            ));
        }
        let (worker_tasks, remaining) = task_tail.split_at_mut(count);
        worker_partitions.push(worker_tasks);
        task_tail = remaining;
        consumed = partition.end;
    }
    if consumed != schedule.sorted_request_indices.len() || !task_tail.is_empty() {
        return Err(anyhow!("{operation} partitions do not consume every task"));
    }
    Ok(worker_partitions)
}

fn build_copied_bindings(
    profile: &FloorProfile,
    direct: &[&TensorDesc],
    resources: &[Buffer],
) -> Result<Vec<Binding>> {
    let bindings = resources
        .iter()
        .enumerate()
        .zip(direct)
        .map(|((resource_index, buffer), desc)| {
            Ok(Binding {
                buffer: buffer.clone(),
                resource_index,
                offset: 0,
                length: usize::try_from(desc.n_bytes)
                    .map_err(|_| anyhow!("tensor byte length does not fit usize"))?,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    validate_copied_topology(profile, direct, resources, &bindings)?;
    Ok(bindings)
}

fn materialize_copied(
    ctx: &MetalContext,
    gguf: &GgufFile,
    direct: &[&TensorDesc],
    profile: &FloorProfile,
) -> Result<Materialized> {
    let ready_started = Instant::now();
    let mut resources = Vec::with_capacity(direct.len());
    for desc in direct {
        resources.push(ctx.buffer_from(gguf.try_slice(desc)?)?);
    }
    let binding_started = Instant::now();
    let bindings = resources
        .iter()
        .enumerate()
        .zip(direct)
        .map(|((resource_index, buffer), desc)| {
            Ok(Binding {
                buffer: buffer.clone(),
                resource_index,
                offset: 0,
                length: usize::try_from(desc.n_bytes)
                    .map_err(|_| anyhow!("tensor byte length does not fit usize"))?,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    validate_copied_topology(profile, direct, &resources, &bindings)?;
    let binding_wall = binding_started.elapsed();
    let ready_wall = ready_started.elapsed();
    Ok(Materialized {
        resources,
        bindings,
        allocation_wall: None,
        source_resolution_wall: None,
        copy_wall: None,
        source_release_wall: None,
        binding_wall,
        worker_count: 0,
        schedule: None,
        blit_population: None,
        ready_wall,
    })
}

fn materialize_parallel_copied(
    ctx: &MetalContext,
    gguf: &GgufFile,
    direct: &[&TensorDesc],
    schedule: &ParallelCopySchedule,
    profile: &FloorProfile,
) -> Result<Materialized> {
    if schedule.partitions.len() != FROZEN_PARALLEL_COPY_WORKERS {
        return Err(anyhow!("parallel-copy materialization is frozen at W4"));
    }
    let ready_started = Instant::now();
    let allocation_started = ready_started;
    let mut resources = allocate_copied_resources(ctx, direct)?;
    let allocation_finished = Instant::now();

    let timed_schedule = frozen_parallel_copy_schedule(profile, direct)?;
    if schedule != &timed_schedule {
        return Err(anyhow!("parallel-copy timed W4 schedule proof drifted"));
    }
    let destination_ranges = validate_copied_destinations(&resources, direct, "parallel-copy")?;

    let shard_lengths = gguf.shard_mapped_lengths();
    let mut sources = Vec::with_capacity(direct.len());
    for (request_index, desc) in direct.iter().enumerate() {
        let endpoint = desc
            .data_offset
            .checked_add(desc.n_bytes)
            .ok_or_else(|| anyhow!("parallel-copy source endpoint overflow"))?;
        if endpoint
            > *shard_lengths
                .get(desc.shard_idx)
                .ok_or_else(|| anyhow!("parallel-copy source shard is unavailable"))?
                as u64
        {
            return Err(anyhow!(
                "parallel-copy source {request_index} exceeds its shard"
            ));
        }
        let source = gguf.try_slice(desc)?;
        let (start, end) = destination_ranges[request_index];
        let source_start = source.as_ptr() as usize;
        let source_end = source_start
            .checked_add(source.len())
            .ok_or_else(|| anyhow!("parallel-copy source address overflow"))?;
        if source.is_empty()
            || source_start == 0
            || source.len() != end - start
            || destination_ranges
                .iter()
                .any(|&(left, right)| source_start < right && left < source_end)
        {
            return Err(anyhow!(
                "parallel-copy source {request_index} is invalid or overlaps a destination"
            ));
        }
        sources.push(source);
    }

    let mut tasks_by_request = resources
        .iter_mut()
        .zip(&sources)
        .map(|(resource, &source)| {
            Some(ParallelCopyTask {
                source,
                // SAFETY: all complete source and destination range, mode,
                // identity, and exclusivity prerequisites were proven above.
                destination: unsafe { exclusive_buffer_bytes_mut(resource) },
            })
        })
        .collect::<Vec<_>>();
    let mut tasks = order_parallel_tasks(&mut tasks_by_request, schedule, "parallel-copy")?;
    let worker_partitions = partition_parallel_tasks(&mut tasks, schedule, "parallel-copy")?;

    let (copy_result, source_finished) = std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(schedule.partitions.len());
        let mut spawn_error = None;
        let source_finished = Instant::now();
        for (worker, worker_tasks) in worker_partitions.into_iter().enumerate() {
            match std::thread::Builder::new().spawn_scoped(scope, move || {
                for task in worker_tasks {
                    task.destination.copy_from_slice(task.source);
                }
            }) {
                Ok(handle) => handles.push((worker, handle)),
                Err(error) => {
                    spawn_error = Some((worker, error));
                    break;
                }
            }
        }
        let mut panicked_worker = None;
        for (worker, handle) in handles {
            if handle.join().is_err() && panicked_worker.is_none() {
                panicked_worker = Some(worker);
            }
        }
        let result = if let Some((worker, error)) = spawn_error {
            Err(anyhow!(
                "parallel-copy worker {worker} spawn failed: {error}"
            ))
        } else if let Some(worker) = panicked_worker {
            Err(anyhow!("parallel-copy worker {worker} panicked"))
        } else {
            Ok(())
        };
        (result, source_finished)
    });
    copy_result?;
    let copy_finished = Instant::now();
    drop(tasks);
    drop(tasks_by_request);
    drop(sources);

    let bindings = build_copied_bindings(profile, direct, &resources)?;
    let binding_finished = Instant::now();
    let allocation_wall = allocation_finished.duration_since(allocation_started);
    let source_resolution_wall = source_finished.duration_since(allocation_finished);
    let copy_wall = copy_finished.duration_since(source_finished);
    let binding_wall = binding_finished.duration_since(copy_finished);
    let ready_wall = binding_finished.duration_since(ready_started);
    let phase_wall = allocation_wall
        .checked_add(source_resolution_wall)
        .and_then(|value| value.checked_add(copy_wall))
        .and_then(|value| value.checked_add(binding_wall))
        .ok_or_else(|| anyhow!("parallel-copy phase duration overflow"))?;
    if ready_wall.as_micros().abs_diff(phase_wall.as_micros()) > 4 {
        return Err(anyhow!("parallel-copy phase timing does not reconcile"));
    }
    Ok(Materialized {
        resources,
        bindings,
        allocation_wall: Some(allocation_wall),
        source_resolution_wall: Some(source_resolution_wall),
        copy_wall: Some(copy_wall),
        source_release_wall: None,
        binding_wall,
        worker_count: schedule.partitions.len(),
        schedule: Some(schedule.clone()),
        blit_population: None,
        ready_wall,
    })
}

fn materialize_parallel_pread(
    ctx: &MetalContext,
    gguf: &GgufFile,
    direct: &[&TensorDesc],
    schedule: &ParallelCopySchedule,
    profile: &FloorProfile,
) -> Result<Materialized> {
    let workers = schedule.partitions.len();
    if workers != FROZEN_PARALLEL_COPY_WORKERS {
        let dynamic_schedule = parallel_copy_schedule(direct, workers)?;
        if schedule != &dynamic_schedule {
            return Err(anyhow!("parallel-pread dynamic schedule proof drifted"));
        }
    }
    let ready_started = Instant::now();
    let allocation_started = ready_started;
    let mut resources = allocate_copied_resources(ctx, direct)?;
    let allocation_finished = Instant::now();

    if workers == FROZEN_PARALLEL_COPY_WORKERS
        && schedule != &frozen_parallel_copy_schedule(profile, direct)?
    {
        return Err(anyhow!("parallel-pread timed W4 schedule proof drifted"));
    }
    let destination_ranges = validate_copied_destinations(&resources, direct, "parallel-pread")?;
    let shard_lengths = gguf.shard_mapped_lengths();
    let mut tasks_by_request = Vec::with_capacity(direct.len());
    for (request_index, (resource, desc)) in resources.iter_mut().zip(direct).enumerate() {
        let endpoint = desc
            .data_offset
            .checked_add(desc.n_bytes)
            .ok_or_else(|| anyhow!("parallel-pread source endpoint overflow"))?;
        if endpoint
            > *shard_lengths
                .get(desc.shard_idx)
                .ok_or_else(|| anyhow!("parallel-pread source shard is unavailable"))?
                as u64
        {
            return Err(anyhow!(
                "parallel-pread source {request_index} exceeds its shard"
            ));
        }
        let (start, end) = destination_ranges[request_index];
        if end - start != resource.length() || resource.length() as u64 != desc.n_bytes {
            return Err(anyhow!(
                "parallel-pread destination {request_index} length drifted"
            ));
        }
        tasks_by_request.push(Some(ParallelPreadTask {
            shard_idx: desc.shard_idx,
            source_offset: desc.data_offset,
            // SAFETY: complete destination range, mode, identity, and
            // exclusivity prerequisites were proven above.
            destination: unsafe { exclusive_buffer_bytes_mut(resource) },
        }));
    }
    let mut tasks = order_parallel_tasks(&mut tasks_by_request, schedule, "parallel-pread")?;
    let worker_partitions = partition_parallel_tasks(&mut tasks, schedule, "parallel-pread")?;

    let (copy_result, source_finished) = std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(schedule.partitions.len());
        let mut spawn_error = None;
        let source_finished = Instant::now();
        for (worker, worker_tasks) in worker_partitions.into_iter().enumerate() {
            match std::thread::Builder::new().spawn_scoped(scope, move || -> Result<()> {
                for task in worker_tasks {
                    gguf.read_shard_exact_at(task.shard_idx, task.source_offset, task.destination)?;
                }
                Ok(())
            }) {
                Ok(handle) => handles.push((worker, handle)),
                Err(error) => {
                    spawn_error = Some((worker, error));
                    break;
                }
            }
        }
        let mut worker_error = None;
        let mut panicked_worker = None;
        for (worker, handle) in handles {
            match handle.join() {
                Ok(Ok(())) => {}
                Ok(Err(error)) if worker_error.is_none() => {
                    worker_error = Some((worker, error));
                }
                Ok(Err(_)) => {}
                Err(_) if panicked_worker.is_none() => panicked_worker = Some(worker),
                Err(_) => {}
            }
        }
        let result = if let Some((worker, error)) = spawn_error {
            Err(anyhow!(
                "parallel-pread worker {worker} spawn failed: {error}"
            ))
        } else if let Some((worker, error)) = worker_error {
            Err(error.context(format!("parallel-pread worker {worker} failed")))
        } else if let Some(worker) = panicked_worker {
            Err(anyhow!("parallel-pread worker {worker} panicked"))
        } else {
            Ok(())
        };
        (result, source_finished)
    });
    copy_result?;
    let copy_finished = Instant::now();
    drop(tasks);
    drop(tasks_by_request);

    let bindings = build_copied_bindings(profile, direct, &resources)?;
    let binding_finished = Instant::now();
    let allocation_wall = allocation_finished.duration_since(allocation_started);
    let source_resolution_wall = source_finished.duration_since(allocation_finished);
    let copy_wall = copy_finished.duration_since(source_finished);
    let binding_wall = binding_finished.duration_since(copy_finished);
    let ready_wall = binding_finished.duration_since(ready_started);
    let phase_wall = allocation_wall
        .checked_add(source_resolution_wall)
        .and_then(|value| value.checked_add(copy_wall))
        .and_then(|value| value.checked_add(binding_wall))
        .ok_or_else(|| anyhow!("parallel-pread phase duration overflow"))?;
    if ready_wall.as_micros().abs_diff(phase_wall.as_micros()) > 4 {
        return Err(anyhow!("parallel-pread phase timing does not reconcile"));
    }
    Ok(Materialized {
        resources,
        bindings,
        allocation_wall: Some(allocation_wall),
        source_resolution_wall: Some(source_resolution_wall),
        copy_wall: Some(copy_wall),
        source_release_wall: None,
        binding_wall,
        worker_count: schedule.partitions.len(),
        schedule: Some(schedule.clone()),
        blit_population: None,
        ready_wall,
    })
}

fn validate_transient_mmap_blit_plan(
    profile: &FloorProfile,
    direct: &[&TensorDesc],
    plan: &RetainedStoragePlan,
) -> Result<()> {
    if profile.id != FloorProfileId::A3bQ4kmV1
        || retained_storage_plan_digest(plan) != A3B_BLIT_PLAN_DIGEST
        || plan.page_size != EXPECTED_PAGE_SIZE
        || plan.max_buffer_length != EXPECTED_MAX_BUFFER_LENGTH
        || plan.required_alignment != REQUIRED_ALIGNMENT
        || plan.windows.len() != 1
        || plan.entries.len() != direct.len()
        || plan.unique_view_bytes != A3B_BLIT_WINDOW_LOGICAL_BYTES
        || plan.logical_view_bytes != A3B_BLIT_WINDOW_LOGICAL_BYTES
        || plan.unique_fallback_bytes != A3B_BLIT_FALLBACK_BYTES
        || plan.alias_bytes != 0
    {
        return Err(anyhow!("transient mmap-blit plan identity drifted"));
    }
    let window = &plan.windows[0];
    let window_gap_bytes = (window.length as u64)
        .checked_sub(plan.unique_view_bytes)
        .ok_or_else(|| anyhow!("transient mmap-blit window byte accounting underflow"))?;
    if window.shard_idx != 0
        || window.mmap_offset != A3B_BLIT_WINDOW_OFFSET
        || window.length != A3B_BLIT_WINDOW_BYTES
        || window_gap_bytes != A3B_BLIT_WINDOW_GAP_BYTES
    {
        return Err(anyhow!("transient mmap-blit source window drifted"));
    }

    let mut view_count = 0usize;
    let mut view_bytes = 0u64;
    let mut fallback_count = 0usize;
    for (request_index, (entry, desc)) in plan.entries.iter().zip(direct).enumerate() {
        if entry.request_index != request_index
            || entry.name != desc.name
            || entry.shard_idx != desc.shard_idx
            || entry.data_offset != desc.data_offset
            || entry.n_bytes != desc.n_bytes
        {
            return Err(anyhow!(
                "transient mmap-blit entry {request_index} identity drifted"
            ));
        }
        match entry.disposition {
            RetainedStorageDisposition::View {
                window_index,
                buffer_offset,
            } => {
                if window_index != 0
                    || buffer_offset
                        != desc
                            .data_offset
                            .checked_sub(window.mmap_offset)
                            .ok_or_else(|| anyhow!("transient mmap-blit view underflow"))?
                {
                    return Err(anyhow!("transient mmap-blit view {request_index} drifted"));
                }
                view_count += 1;
                view_bytes = view_bytes
                    .checked_add(desc.n_bytes)
                    .ok_or_else(|| anyhow!("transient mmap-blit view bytes overflow"))?;
            }
            RetainedStorageDisposition::CopyFallback { reason } => {
                if request_index != A3B_BLIT_FALLBACK_REQUEST
                    || reason != RetainedStorageFallback::FinalPartialPage
                    || desc.n_bytes != A3B_BLIT_FALLBACK_BYTES
                {
                    return Err(anyhow!(
                        "transient mmap-blit fallback {request_index} drifted"
                    ));
                }
                fallback_count += 1;
            }
            RetainedStorageDisposition::Alias { .. } => {
                return Err(anyhow!("transient mmap-blit aliases are unsupported"));
            }
        }
    }
    if view_count != 732 || view_bytes != A3B_BLIT_WINDOW_LOGICAL_BYTES || fallback_count != 1 {
        return Err(anyhow!("transient mmap-blit plan accounting drifted"));
    }
    Ok(())
}

fn materialize_transient_mmap_blit(
    ctx: &MetalContext,
    gguf: &GgufFile,
    direct: &[&TensorDesc],
    blit_order: &[usize],
    plan: &RetainedStoragePlan,
    profile: &FloorProfile,
) -> Result<Materialized> {
    validate_transient_mmap_blit_plan(profile, direct, plan)?;
    if blit_order != source_order(direct)
        || blit_order.first().copied() != Some(2)
        || blit_order.last().copied() != Some(A3B_BLIT_FALLBACK_REQUEST)
    {
        return Err(anyhow!("transient mmap-blit source order drifted"));
    }

    let ready_started = Instant::now();
    let allocation_started = ready_started;
    let resources = allocate_copied_resources(ctx, direct)?;
    let allocation_finished = Instant::now();
    validate_copied_destinations(&resources, direct, "transient mmap-blit")?;
    let allocated_after_destinations = ctx.current_allocated_size();

    let (
        source_finished,
        copy_finished,
        window_probes,
        fallback_weaks,
        retained_references,
        command_status,
        gpu_start_time,
        gpu_end_time,
        gpu_wall_ms,
        allocated_with_sources,
    ) = autoreleasepool(|_| -> Result<_> {
        let mut source_windows = Vec::with_capacity(plan.windows.len());
        for window in &plan.windows {
            source_windows.push(ctx.diagnostic_gguf_blit_source_window(
                gguf,
                window,
                plan.required_alignment,
            )?);
        }
        let window_probes = source_windows
            .iter()
            .map(|window| window.release_probe())
            .collect::<Vec<DiagnosticGgufBlitReleaseProbe>>();

        let mut fallback_sources = std::iter::repeat_with(|| None)
            .take(direct.len())
            .collect::<Vec<Option<Buffer>>>();
        for entry in &plan.entries {
            if matches!(
                entry.disposition,
                RetainedStorageDisposition::CopyFallback { .. }
            ) {
                let desc = direct[entry.request_index];
                fallback_sources[entry.request_index] =
                    Some(ctx.buffer_from(gguf.try_slice(desc)?)?);
            }
        }
        let fallback_weaks = fallback_sources
            .iter()
            .filter_map(Option::as_ref)
            .map(Weak::from_retained)
            .collect::<Vec<Weak<ProtocolObject<dyn MTLBuffer>>>>();
        if fallback_weaks.len() != 1 {
            return Err(anyhow!("transient mmap-blit fallback source count drifted"));
        }
        let source_finished = Instant::now();
        let allocated_with_sources = ctx.current_allocated_size();

        let command = ctx
            .queue
            .commandBuffer()
            .context("transient mmap-blit command buffer")?;
        let retained_references = command.retainedReferences();
        if !retained_references {
            return Err(anyhow!(
                "transient mmap-blit command buffer does not retain references"
            ));
        }
        let blit = BlitEncoder::try_begin(&command).context("transient mmap-blit encoder")?;
        let encode_result = (|| -> Result<()> {
            for &request_index in blit_order {
                let desc = direct[request_index];
                let destination = &resources[request_index];
                match plan.entries[request_index].disposition {
                    RetainedStorageDisposition::View { window_index, .. } => {
                        source_windows[window_index].encode_copy_to(
                            &blit,
                            desc.shard_idx,
                            desc.data_offset,
                            destination,
                            0,
                            desc.n_bytes,
                        )?;
                    }
                    RetainedStorageDisposition::CopyFallback { .. } => {
                        let source = fallback_sources[request_index].as_ref().ok_or_else(|| {
                            anyhow!("transient mmap-blit fallback source is missing")
                        })?;
                        if Retained::as_ptr(source) == Retained::as_ptr(destination)
                            || source.length() as u64 != desc.n_bytes
                            || destination.length() as u64 != desc.n_bytes
                        {
                            return Err(anyhow!(
                                "transient mmap-blit fallback resource {request_index} drifted"
                            ));
                        }
                        blit.copy_buffer(source, 0, destination, 0, desc.n_bytes);
                    }
                    RetainedStorageDisposition::Alias { .. } => unreachable!(),
                }
            }
            Ok(())
        })();
        blit.end();
        encode_result?;
        command.commit();
        command.waitUntilCompleted();
        let command_status = command.status();
        let command_error = command.error();
        if command_status != MTLCommandBufferStatus::Completed || command_error.is_some() {
            return Err(anyhow!(
                "transient mmap-blit command failed: status={command_status:?} error={command_error:?}"
            ));
        }
        let gpu_start = command.GPUStartTime();
        let gpu_end = command.GPUEndTime();
        let (gpu_start_time, gpu_end_time, gpu_wall_ms) = if gpu_start.is_finite()
            && gpu_end.is_finite()
            && gpu_start > 0.0
            && gpu_end > gpu_start
        {
            (
                Some(gpu_start),
                Some(gpu_end),
                Some((gpu_end - gpu_start) * 1e3),
            )
        } else {
            (None, None, None)
        };
        let copy_finished = Instant::now();
        drop(command);
        drop(fallback_sources);
        drop(source_windows);
        Ok((
            source_finished,
            copy_finished,
            window_probes,
            fallback_weaks,
            retained_references,
            command_status,
            gpu_start_time,
            gpu_end_time,
            gpu_wall_ms,
            allocated_with_sources,
        ))
    })?;

    let mut source_window_deallocator_calls = 0usize;
    let mut source_window_deallocator_mismatches = 0usize;
    let mut source_buffers_alive = 0usize;
    for probe in window_probes {
        let report = probe.report();
        source_window_deallocator_calls += report.deallocator_calls;
        source_window_deallocator_mismatches += report.deallocator_mismatches;
        source_buffers_alive += usize::from(report.source_alive);
    }
    source_buffers_alive += fallback_weaks
        .iter()
        .filter(|weak| weak.load().is_some())
        .count();
    if source_window_deallocator_calls != 1
        || source_window_deallocator_mismatches != 0
        || source_buffers_alive != 0
    {
        return Err(anyhow!(
            "transient mmap-blit source release failed: calls={source_window_deallocator_calls} mismatches={source_window_deallocator_mismatches} alive={source_buffers_alive}"
        ));
    }
    let source_release_finished = Instant::now();
    let allocated_after_source_release = ctx.current_allocated_size();

    let bindings = build_copied_bindings(profile, direct, &resources)?;
    let binding_finished = Instant::now();
    let allocation_wall = allocation_finished.duration_since(allocation_started);
    let source_resolution_wall = source_finished.duration_since(allocation_finished);
    let copy_wall = copy_finished.duration_since(source_finished);
    let source_release_wall = source_release_finished.duration_since(copy_finished);
    let binding_wall = binding_finished.duration_since(source_release_finished);
    let ready_wall = binding_finished.duration_since(ready_started);
    let phase_wall = allocation_wall
        .checked_add(source_resolution_wall)
        .and_then(|value| value.checked_add(copy_wall))
        .and_then(|value| value.checked_add(source_release_wall))
        .and_then(|value| value.checked_add(binding_wall))
        .ok_or_else(|| anyhow!("transient mmap-blit phase duration overflow"))?;
    if ready_wall.as_micros().abs_diff(phase_wall.as_micros()) > 4 {
        return Err(anyhow!(
            "transient mmap-blit phase timing does not reconcile"
        ));
    }

    let source_window_bytes = plan.windows.iter().try_fold(0u64, |total, window| {
        total
            .checked_add(window.length as u64)
            .ok_or_else(|| anyhow!("transient mmap-blit source window bytes overflow"))
    })?;
    let window_blit_count = plan
        .entries
        .iter()
        .filter(|entry| matches!(entry.disposition, RetainedStorageDisposition::View { .. }))
        .count();
    let fallback_source_count = plan
        .entries
        .iter()
        .filter(|entry| {
            matches!(
                entry.disposition,
                RetainedStorageDisposition::CopyFallback { .. }
            )
        })
        .count();
    let blit_bytes = direct.iter().try_fold(0u64, |total, desc| {
        total
            .checked_add(desc.n_bytes)
            .ok_or_else(|| anyhow!("transient mmap-blit total bytes overflow"))
    })?;
    let source_window_gap_bytes = source_window_bytes
        .checked_sub(plan.unique_view_bytes)
        .ok_or_else(|| anyhow!("transient mmap-blit source window gap underflow"))?;

    Ok(Materialized {
        resources,
        bindings,
        allocation_wall: Some(allocation_wall),
        source_resolution_wall: Some(source_resolution_wall),
        copy_wall: Some(copy_wall),
        source_release_wall: Some(source_release_wall),
        binding_wall,
        worker_count: 0,
        schedule: None,
        blit_population: Some(BlitPopulation {
            order_count: blit_order.len(),
            order_first_request: blit_order[0],
            order_last_request: *blit_order.last().expect("blit order is nonempty"),
            source_window_count: plan.windows.len(),
            source_window_bytes,
            window_blit_count,
            window_blit_bytes: plan.logical_view_bytes,
            source_window_gap_bytes,
            fallback_source_count,
            fallback_source_bytes: plan.unique_fallback_bytes,
            cpu_staging_copy_bytes: plan.unique_fallback_bytes,
            blit_count: blit_order.len(),
            blit_bytes,
            command_buffer_count: 1,
            blit_encoder_count: 1,
            commit_count: 1,
            wait_count: 1,
            command_error_count: 0,
            source_window_deallocator_calls,
            source_window_deallocator_mismatches,
            source_buffers_alive,
            retained_references,
            command_status,
            gpu_start_time,
            gpu_end_time,
            gpu_wall_ms,
            allocated_after_destinations,
            allocated_with_sources,
            allocated_after_source_release,
        }),
        ready_wall,
    })
}

fn buffer_bytes(buffer: &Buffer, offset: usize, length: usize) -> Result<&[u8]> {
    let end = offset
        .checked_add(length)
        .ok_or_else(|| anyhow!("buffer view endpoint overflow"))?;
    if end > buffer.length() {
        return Err(anyhow!(
            "buffer view endpoint {end} exceeds length {}",
            buffer.length()
        ));
    }
    Ok(unsafe {
        std::slice::from_raw_parts(buffer.contents().as_ptr().cast::<u8>().add(offset), length)
    })
}

fn verify_materialized(
    gguf: &GgufFile,
    direct: &[&TensorDesc],
    materialized: &Materialized,
    profile: &FloorProfile,
) -> Result<Correctness> {
    validate_copied_topology(
        profile,
        direct,
        &materialized.resources,
        &materialized.bindings,
    )?;
    let mut payload_bytes_checked = 0u64;
    for (request_index, (desc, binding)) in direct.iter().zip(&materialized.bindings).enumerate() {
        if binding.offset % REQUIRED_ALIGNMENT != 0 {
            return Err(anyhow!("binding {request_index} is misaligned"));
        }
        let source = gguf.try_slice(desc)?;
        let actual = buffer_bytes(&binding.buffer, binding.offset, binding.length)?;
        if actual != source {
            return Err(anyhow!("binding {request_index} differs from source"));
        }
        payload_bytes_checked = payload_bytes_checked
            .checked_add(desc.n_bytes)
            .ok_or_else(|| anyhow!("verified payload bytes overflow"))?;
    }
    if payload_bytes_checked != profile.logical_copy_bytes {
        return Err(anyhow!("verified payload byte total drifted"));
    }
    Ok(Correctness {
        payload_bytes_checked,
        entries_checked: direct.len(),
    })
}

pub(crate) fn run(args: GgufArenaFloorArgs, build_identity: Value) -> Result<()> {
    validate_worker_scope(args.workers, args.describe, args.arm, args.profile)?;
    let gguf = GgufFile::open(&args.model).context("open GGUF")?;
    let model = Model::from_gguf(&gguf).context("bind model")?;
    let native_embedding_supported = native_quant_embedding_storage_supported(&model);
    let native_embedding = production_native_quant_embedding_storage_enabled(&model);
    if !native_embedding {
        return Err(anyhow!(
            "GGUF floor requires the production native embedding policy"
        ));
    }
    let requests = model_weight_storage_requests(&model, native_embedding, false)?;
    if requests
        .iter()
        .any(|request| request.kind != ModelWeightStorageKind::Direct)
    {
        return Err(anyhow!("GGUF floor requires an all-direct inventory"));
    }
    let direct = requests
        .iter()
        .map(|request| request.desc)
        .collect::<Vec<_>>();
    let page_size = host_page_size_bytes()?;
    let ctx = MetalContext::new()?;
    let max_buffer_length = ctx.max_buffer_length();
    let logical_copy_bytes = direct.iter().try_fold(0u64, |total, desc| {
        total
            .checked_add(desc.n_bytes)
            .ok_or_else(|| anyhow!("logical byte accounting overflow"))
    })?;
    let descriptor_digest = format!("{:#018x}", gguf_descriptor_layout_digest(&gguf));
    let inventory_digest = model_weight_storage_inventory_digest(&requests);
    let device_name = ctx.device.name().to_string();
    let unified_memory = ctx.device.hasUnifiedMemory();
    let architecture_tuple = json!({
        "kind": format!("{:?}", model.arch.kind).to_lowercase(),
        "n_layer": model.arch.n_layer,
        "hidden_size": model.arch.hidden_size,
        "intermediate_size": model.arch.intermediate_size,
        "vocab_size": model.arch.vocab_size,
        "full_attention_interval": model.arch.full_attention_interval,
        "n_q_heads": model.arch.n_q_heads,
        "n_kv_heads": model.arch.n_kv_heads,
        "attn_head_dim": model.arch.attn_head_dim,
        "rope_theta": model.arch.rope_theta,
        "partial_rotary_factor": model.arch.partial_rotary_factor,
        "gdn_n_v_heads": model.arch.gdn_n_v_heads,
        "gdn_n_k_heads": model.arch.gdn_n_k_heads,
        "gdn_head_dim": model.arch.gdn_head_dim,
        "gdn_conv_kernel": model.arch.gdn_conv_kernel,
        "expert_count": model.arch.expert_count,
        "expert_used_count": model.arch.expert_used_count,
        "expert_feed_forward_length": model.arch.expert_feed_forward_length,
        "expert_shared_feed_forward_length": model.arch.expert_shared_feed_forward_length,
        "mtp_n_hidden_layers": model.arch.mtp_n_hidden_layers,
    });
    let facts = ProfileFacts {
        gguf: &gguf,
        model: &model,
        native_embedding,
        direct: &direct,
        page_size,
        max_buffer_length,
        device_name: &device_name,
        unified_memory,
        descriptor_digest: &descriptor_digest,
        inventory_digest: &inventory_digest,
        logical_copy_bytes,
    };

    if args.describe {
        let usage_capability = capture_usage()?;
        let proc_capability = capture_proc_usage()?;
        let computed_schedule = parallel_copy_schedule(&direct, args.workers)?;
        let planner_descriptive = (|| -> Result<Value> {
            let plan = qwen_llm::metal::plan_retained_storage(
                &gguf.shard_mapped_lengths(),
                &direct,
                page_size,
                max_buffer_length,
                REQUIRED_ALIGNMENT,
            )?;
            let window_bytes = plan.windows.iter().try_fold(0u64, |total, window| {
                total
                    .checked_add(window.length as u64)
                    .ok_or_else(|| anyhow!("window byte accounting overflow"))
            })?;
            let arena_copy_bytes = window_bytes
                .checked_add(plan.unique_fallback_bytes)
                .ok_or_else(|| anyhow!("arena copy byte accounting overflow"))?;
            let planner_gap_bytes = window_bytes
                .checked_sub(plan.unique_view_bytes)
                .ok_or_else(|| anyhow!("planned view bytes exceed window bytes"))?;
            let alias_count = plan
                .entries
                .iter()
                .filter(|entry| {
                    matches!(entry.disposition, RetainedStorageDisposition::Alias { .. })
                })
                .count();
            let fallback_count = plan
                .entries
                .iter()
                .filter(|entry| {
                    matches!(
                        entry.disposition,
                        RetainedStorageDisposition::CopyFallback { .. }
                    )
                })
                .count();
            let view_count = plan
                .entries
                .iter()
                .filter(|entry| {
                    matches!(entry.disposition, RetainedStorageDisposition::View { .. })
                })
                .count();
            Ok(json!({
                "status": "ok",
                "planner_digest": retained_storage_plan_digest(&plan),
                "view_count": view_count,
                "window_count": plan.windows.len(),
                "fallback_count": fallback_count,
                "alias_count": alias_count,
                "unique_view_bytes": plan.unique_view_bytes,
                "logical_view_bytes": plan.logical_view_bytes,
                "fallback_bytes": plan.unique_fallback_bytes,
                "window_bytes": window_bytes,
                "planner_gap_bytes": planner_gap_bytes,
                "arena_copy_bytes": arena_copy_bytes,
                "fallback_reasons": plan.entries.iter().filter_map(|entry| {
                    match entry.disposition {
                        RetainedStorageDisposition::CopyFallback { reason } => {
                            Some(format!("{reason:?}"))
                        }
                        _ => None,
                    }
                }).collect::<Vec<_>>(),
            }))
        })()
        .unwrap_or_else(|error| {
            json!({
                "status": "error",
                "error": error.to_string(),
            })
        });
        let matching = authenticated_profile_matches(&facts, &direct)?;
        if matching.len() > 1 {
            return Err(anyhow!("GGUF floor geometry matches multiple profiles"));
        }
        let matched_profile = matching.first().map(|(profile, _)| *profile);
        validate_describe_worker_scope(args.workers, matched_profile.map(|profile| profile.id))?;
        let frozen_schedule = matching.first().map(|(_, schedule)| schedule);
        let computed_schedule_json = parallel_copy_schedule_json(&computed_schedule, &direct)?;
        let embedding_environment_absent = std::env::var_os("QWEN_NATIVE_QUANT_EMBED").is_none();
        let usage_capability_json = json!({
            "getrusage": true,
            "proc_pid_rusage_v4": true,
            "sample_minor_faults": usage_capability.minor_faults,
            "sample_major_faults": usage_capability.major_faults,
            "sample_instructions_raw": proc_capability.instructions,
            "sample_cycles_raw": proc_capability.cycles,
            "sample_billed_energy_raw": proc_capability.billed_energy,
            "sample_serviced_energy_raw": proc_capability.serviced_energy,
        });
        let row = json!({
            "schema_version": 2,
            "mode": "describe",
            "model": args.model,
            "materialization_supported": matched_profile.is_some(),
            "materialization_environment_admissible": embedding_environment_absent,
            "matched_profile": matched_profile.map(|profile| profile.id.label()),
            "architecture": gguf.architecture(),
            "descriptor_layout_digest": descriptor_digest,
            "inventory_digest": inventory_digest,
            "retained_planner": planner_descriptive,
            "native_quant_embedding": native_embedding,
            "native_quant_embedding_supported": native_embedding_supported,
            "native_quant_embedding_selection": if embedding_environment_absent {
                "production-auto-promoted"
            } else {
                "environment-present-unadmitted"
            },
            "page_size": page_size,
            "required_alignment": REQUIRED_ALIGNMENT,
            "max_buffer_length": max_buffer_length,
            "request_count": direct.len(),
            "logical_copy_bytes": logical_copy_bytes,
            "shard_mapped_lengths": gguf.shard_mapped_lengths(),
            "architecture_tuple": architecture_tuple,
            "tied_embeddings": model.tied_embeddings,
            "mtp_present": model.mtp.is_some(),
            "device_name": device_name,
            "unified_memory": unified_memory,
            "parallel_copy_schedule": computed_schedule_json.clone(),
            "computed_schedule": computed_schedule_json,
            "frozen_schedule": frozen_schedule.map(|schedule| {
                parallel_copy_schedule_json(schedule, &direct)
            }).transpose()?,
            "usage_capability": usage_capability_json,
            "build_identity": build_identity,
        });
        match args.output {
            OutputFormat::Json => println!("{}", serde_json::to_string_pretty(&row)?),
            OutputFormat::Text => println!("{}", serde_json::to_string_pretty(&row)?),
        }
        return Ok(());
    }
    let arm = args
        .arm
        .ok_or_else(|| anyhow!("materialization arm is required"))?;
    if matches!(arm, ArenaFloorArm::ArenaSerial | ArenaFloorArm::ArenaFour) {
        return Err(anyhow!(
            "arena materialization arms are retired; use copied, parallel-copied, parallel-pread, or transient-mmap-blit"
        ));
    }
    if std::env::var_os("QWEN_NATIVE_QUANT_EMBED").is_some() {
        return Err(anyhow!(
            "materialization profiles require QWEN_NATIVE_QUANT_EMBED to be absent"
        ));
    }
    let profile_id = args
        .profile
        .ok_or_else(|| anyhow!("materialization profile is required"))?;
    let mut matching = authenticated_profile_matches(&facts, &direct)?;
    if matching.len() != 1 {
        return Err(anyhow!(
            "loaded geometry matches {} floor profiles; exactly one is required",
            matching.len()
        ));
    }
    let (profile, frozen_parallel_schedule) = matching.pop().expect("one profile match exists");
    if !native_embedding_supported || !native_embedding || profile.id != profile_id {
        return Err(anyhow!(
            "requested floor profile {} does not match loaded geometry",
            profile_id.label()
        ));
    }
    validate_source_endpoints(&gguf, &direct)?;
    if arm == ArenaFloorArm::TransientMmapBlit && profile.id != FloorProfileId::A3bQ4kmV1 {
        return Err(anyhow!(
            "transient mmap-blit is diagnostic-only for a3b-q4km-v1"
        ));
    }
    let parallel_schedule = if arm.uses_parallel_schedule() {
        let schedule = parallel_copy_schedule(&direct, args.workers)?;
        if args.workers == FROZEN_PARALLEL_COPY_WORKERS && schedule != frozen_parallel_schedule {
            return Err(anyhow!(
                "dynamic W4 schedule does not reproduce the frozen floor schedule"
            ));
        }
        Some(schedule)
    } else {
        None
    };
    let transient_blit_plan = if arm == ArenaFloorArm::TransientMmapBlit {
        let plan = plan_retained_storage(
            &gguf.shard_mapped_lengths(),
            &direct,
            page_size,
            max_buffer_length,
            REQUIRED_ALIGNMENT,
        )?;
        validate_transient_mmap_blit_plan(profile, &direct, &plan)?;
        Some(plan)
    } else {
        None
    };
    let blit_order = (arm == ArenaFloorArm::TransientMmapBlit).then(|| source_order(&direct));

    let allocated_before = ctx.current_allocated_size();
    let usage_before = capture_usage()?;
    let proc_before = capture_proc_usage()?;
    let materialized = match arm {
        ArenaFloorArm::Copied => materialize_copied(&ctx, &gguf, &direct, profile)?,
        ArenaFloorArm::ParallelCopied => materialize_parallel_copied(
            &ctx,
            &gguf,
            &direct,
            parallel_schedule
                .as_ref()
                .expect("parallel-copy arm has a schedule"),
            profile,
        )?,
        ArenaFloorArm::ParallelPread => materialize_parallel_pread(
            &ctx,
            &gguf,
            &direct,
            parallel_schedule
                .as_ref()
                .expect("parallel-pread arm has a schedule"),
            profile,
        )?,
        ArenaFloorArm::TransientMmapBlit => materialize_transient_mmap_blit(
            &ctx,
            &gguf,
            &direct,
            blit_order.as_deref().expect("blit arm has an order"),
            transient_blit_plan.as_ref().expect("blit arm has a plan"),
            profile,
        )?,
        ArenaFloorArm::ArenaSerial | ArenaFloorArm::ArenaFour => unreachable!(),
    };
    let ready_wall = materialized.ready_wall;
    let usage_after = capture_usage()?;
    let proc_after = capture_proc_usage()?;
    let allocated_ready = ctx.current_allocated_size();
    if (arm.uses_parallel_schedule()
        && materialized.schedule.as_ref() != parallel_schedule.as_ref())
        || (!arm.uses_parallel_schedule() && materialized.schedule.is_some())
    {
        return Err(anyhow!("materialized parallel-population schedule drifted"));
    }

    let user_cpu_us = usage_after
        .user_time_us
        .checked_sub(usage_before.user_time_us)
        .ok_or_else(|| anyhow!("user CPU time regressed"))?;
    let system_cpu_us = usage_after
        .system_time_us
        .checked_sub(usage_before.system_time_us)
        .ok_or_else(|| anyhow!("system CPU time regressed"))?;
    let total_cpu_us = user_cpu_us
        .checked_add(system_cpu_us)
        .ok_or_else(|| anyhow!("total CPU time overflow"))?;
    let timer_minor_faults = usage_after
        .minor_faults
        .checked_sub(usage_before.minor_faults)
        .ok_or_else(|| anyhow!("minor-fault delta overflow"))?;
    let timer_major_faults = usage_after
        .major_faults
        .checked_sub(usage_before.major_faults)
        .ok_or_else(|| anyhow!("major-fault delta overflow"))?;
    if user_cpu_us < 0
        || system_cpu_us < 0
        || total_cpu_us < 0
        || timer_minor_faults < 0
        || timer_major_faults < 0
    {
        return Err(anyhow!(
            "getrusage counter regressed: user={}..{} system={}..{} minor={}..{} major={}..{}",
            usage_before.user_time_us,
            usage_after.user_time_us,
            usage_before.system_time_us,
            usage_after.system_time_us,
            usage_before.minor_faults,
            usage_after.minor_faults,
            usage_before.major_faults,
            usage_after.major_faults,
        ));
    }
    let ready_us = duration_us(ready_wall)?;
    if ready_us == 0 {
        return Err(anyhow!("ready wall is zero"));
    }
    let proc_instructions = proc_after
        .instructions
        .checked_sub(proc_before.instructions)
        .ok_or_else(|| anyhow!("proc instructions regressed"))?;
    let proc_cycles = proc_after
        .cycles
        .checked_sub(proc_before.cycles)
        .ok_or_else(|| anyhow!("proc cycles regressed"))?;
    let proc_billed_energy = proc_after
        .billed_energy
        .checked_sub(proc_before.billed_energy)
        .ok_or_else(|| anyhow!("proc billed energy regressed"))?;
    let proc_serviced_energy = proc_after
        .serviced_energy
        .checked_sub(proc_before.serviced_energy)
        .ok_or_else(|| anyhow!("proc serviced energy regressed"))?;
    let cpu_per_wall = total_cpu_us as f64 / ready_us as f64;
    if !cpu_per_wall.is_finite() || cpu_per_wall < 0.0 {
        return Err(anyhow!("CPU per wall is invalid"));
    }

    let correctness = verify_materialized(&gguf, &direct, &materialized, profile)?;
    let resource_count = materialized.resources.len();
    let binding_count = materialized.bindings.len();
    let blit_population = materialized.blit_population;
    let unattributed_wall = match (
        materialized.allocation_wall,
        materialized.source_resolution_wall,
        materialized.copy_wall,
        materialized.source_release_wall,
    ) {
        (Some(allocation), Some(source_resolution), Some(copy), source_release) => {
            let accounted = allocation
                .checked_add(source_resolution)
                .and_then(|value| value.checked_add(copy))
                .and_then(|value| {
                    source_release.map_or(Some(value), |release| value.checked_add(release))
                })
                .and_then(|value| value.checked_add(materialized.binding_wall))
                .ok_or_else(|| anyhow!("subinterval duration overflow"))?;
            Some(
                ready_wall
                    .checked_sub(accounted)
                    .ok_or_else(|| anyhow!("subintervals exceed ready wall"))?,
            )
        }
        (None, None, None, None) => None,
        _ => return Err(anyhow!("partial subinterval timing is invalid")),
    };
    let allocation_wall = materialized.allocation_wall.map(duration_ms);
    let source_resolution_wall = materialized.source_resolution_wall.map(duration_ms);
    let copy_wall = materialized.copy_wall.map(duration_ms);
    let source_release_wall = materialized.source_release_wall.map(duration_ms);
    let binding_wall = duration_ms(materialized.binding_wall);
    let allocation_us = materialized.allocation_wall.map(duration_us).transpose()?;
    let source_resolution_us = materialized
        .source_resolution_wall
        .map(duration_us)
        .transpose()?;
    let copy_us = materialized.copy_wall.map(duration_us).transpose()?;
    let source_release_us = materialized
        .source_release_wall
        .map(duration_us)
        .transpose()?;
    let binding_us = duration_us(materialized.binding_wall)?;
    let unattributed_us = unattributed_wall.map(duration_us).transpose()?;
    let worker_count = materialized.worker_count;
    let teardown_started = Instant::now();
    drop(materialized);
    let teardown_wall = teardown_started.elapsed();
    let allocated_after_drop = ctx.current_allocated_size();

    let physical_bytes = logical_copy_bytes;
    let ready_gbps = physical_bytes as f64 / ready_wall.as_secs_f64() / 1e9;
    let copy_gbps = copy_wall.map(|wall_ms| physical_bytes as f64 / (wall_ms / 1e3) / 1e9);
    let resource_modes_json = json!({
        "creation_storage": "shared",
        "creation_cpu_cache": "default_cache",
        "creation_hazard_tracking": "default",
        "observed_storage": "shared",
        "observed_cpu_cache": "default_cache",
        "observed_hazard_tracking": "tracked",
    });
    let mut timing_json = json!({
        "ready_wall_ms": duration_ms(ready_wall),
        "ready_us": ready_us,
        "allocation_wall_ms": allocation_wall,
        "allocation_us": allocation_us,
        "source_resolution_wall_ms": source_resolution_wall,
        "source_us": source_resolution_us,
        "source_resolution_us": source_resolution_us,
        "copy_wall_ms": copy_wall,
        "copy_us": copy_us,
        "binding_wall_ms": binding_wall,
        "binding_us": binding_us,
        "unattributed_wall_ms": unattributed_wall.map(duration_ms),
        "unattributed_us": unattributed_us,
        "teardown_wall_ms": duration_ms(teardown_wall),
        "teardown_us": duration_us(teardown_wall)?,
    });
    if let (Some(wall_ms), Some(wall_us)) = (source_release_wall, source_release_us) {
        let timing = timing_json
            .as_object_mut()
            .expect("timing JSON is an object");
        timing.insert("source_release_wall_ms".to_string(), json!(wall_ms));
        timing.insert("source_release_us".to_string(), json!(wall_us));
    }
    let rusage_json = json!({
        "timer_minor_faults": timer_minor_faults,
        "timer_major_faults": timer_major_faults,
        "user_cpu_us": user_cpu_us,
        "system_cpu_us": system_cpu_us,
        "total_cpu_us": total_cpu_us,
        "cpu_per_wall": cpu_per_wall,
    });
    let proc_rusage_json = json!({
        "instructions_delta_raw": proc_instructions,
        "cycles_delta_raw": proc_cycles,
        "billed_energy_delta_raw": proc_billed_energy,
        "serviced_energy_delta_raw": proc_serviced_energy,
    });
    let blit_population_json = blit_population.map(|blit| {
        json!({
            "schema_version": 1,
            "order": {
                "algorithm": "shard-offset-request-v1",
                "count": blit.order_count,
                "first_request_index": blit.order_first_request,
                "last_request_index": blit.order_last_request,
            },
            "sources": {
                "window_count": blit.source_window_count,
                "window_bytes": blit.source_window_bytes,
                "window_gap_bytes": blit.source_window_gap_bytes,
                "fallback_count": blit.fallback_source_count,
                "fallback_bytes": blit.fallback_source_bytes,
                "cpu_staging_copy_bytes": blit.cpu_staging_copy_bytes,
            },
            "copies": {
                "window_count": blit.window_blit_count,
                "window_bytes": blit.window_blit_bytes,
                "total_count": blit.blit_count,
                "total_bytes": blit.blit_bytes,
            },
            "command": {
                "buffer_count": blit.command_buffer_count,
                "encoder_count": blit.blit_encoder_count,
                "commit_count": blit.commit_count,
                "wait_count": blit.wait_count,
                "error_count": blit.command_error_count,
                "status": if blit.command_status == MTLCommandBufferStatus::Completed {
                    "completed"
                } else {
                    "unexpected"
                },
                "status_code": blit.command_status.0,
                "retained_references": blit.retained_references,
                "gpu_start_time": blit.gpu_start_time,
                "gpu_end_time": blit.gpu_end_time,
                "gpu_wall_ms": blit.gpu_wall_ms,
            },
            "release": {
                "window_deallocator_calls": blit.source_window_deallocator_calls,
                "window_deallocator_mismatches": blit.source_window_deallocator_mismatches,
                "source_buffers_alive": blit.source_buffers_alive,
                "allocated_after_destinations": blit.allocated_after_destinations,
                "allocated_with_sources": blit.allocated_with_sources,
                "allocated_after_source_release": blit.allocated_after_source_release,
            },
        })
    });
    let reported_parallel_schedule = if arm == ArenaFloorArm::TransientMmapBlit {
        None
    } else {
        parallel_schedule
            .as_ref()
            .or(Some(&frozen_parallel_schedule))
    };
    let parallel_copy_schedule_json = reported_parallel_schedule
        .map(|schedule| parallel_copy_schedule_json(schedule, &direct))
        .transpose()?;
    let mut row = json!({
        "schema_version": 2,
        "arm": arm.label(),
        "profile": profile.id.label(),
        "model": args.model,
        "architecture": gguf.architecture(),
        "architecture_tuple": architecture_tuple,
        "tied_embeddings": model.tied_embeddings,
        "mtp_present": model.mtp.is_some(),
        "shard_mapped_lengths": gguf.shard_mapped_lengths(),
        "descriptor_layout_digest": descriptor_digest,
        "inventory_digest": inventory_digest,
        "native_quant_embedding": native_embedding,
        "native_quant_embedding_supported": native_embedding_supported,
        "native_quant_embedding_selection": "production-auto-promoted",
        "page_size": page_size,
        "required_alignment": REQUIRED_ALIGNMENT,
        "max_buffer_length": max_buffer_length,
        "device_name": device_name,
        "unified_memory": unified_memory,
        "request_count": direct.len(),
        "resource_count": resource_count,
        "binding_count": binding_count,
        "logical_copy_bytes": logical_copy_bytes,
        "physical_copy_bytes": physical_bytes,
        "resource_modes": resource_modes_json,
        "parallel_copy_schedule": parallel_copy_schedule_json,
        "timing": timing_json,
        "throughput": {
            "ready_gbps_decimal": ready_gbps,
            "copy_gbps_decimal": copy_gbps,
        },
        "rusage": rusage_json,
        "proc_rusage_v4": proc_rusage_json,
        "metal_allocated_bytes": {
            "before": allocated_before,
            "ready": allocated_ready,
            "after_drop": allocated_after_drop,
        },
        "correctness": {
            "passed": true,
            "payload_bytes_checked": correctness.payload_bytes_checked,
            "entries_checked": correctness.entries_checked,
        },
        "worker_count": worker_count,
        "build_identity": build_identity,
    });
    if let Some(blit_population_json) = blit_population_json {
        row.as_object_mut()
            .expect("floor row JSON is an object")
            .insert("blit_population".to_string(), blit_population_json);
    }
    match args.output {
        OutputFormat::Json => println!("{}", serde_json::to_string_pretty(&row)?),
        OutputFormat::Text => {
            println!("arm\t{}", arm.label());
            println!("ready_wall_ms\t{:.3}", duration_ms(ready_wall));
            println!("ready_gbps_decimal\t{ready_gbps:.3}");
            println!("timer_major_faults\t{}", timer_major_faults);
            println!("correctness\tpass");
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        ALLOWED_DIAGNOSTIC_WORKERS, ArenaFloorArm, FROZEN_PARALLEL_COPY_WORKERS, FloorProfileId,
        GgufArenaFloorArgs, PARALLEL_COPY_ALGORITHM, minimax_partition_cuts,
        parallel_copy_schedule, parallel_copy_schedule_json, parse_worker_count, source_order,
        validate_describe_worker_scope, validate_parallel_copy_schedule, validate_worker_scope,
    };
    use clap::Parser;
    use qwen_llm::tensor::{GgmlType, TensorDesc};

    fn descriptors(lengths: &[u64]) -> Vec<TensorDesc> {
        lengths
            .iter()
            .enumerate()
            .map(|(index, &n_bytes)| TensorDesc {
                name: format!("tensor.{index}"),
                shape: vec![n_bytes],
                dtype: GgmlType::F32,
                shard_idx: index % 2,
                data_offset: (index as u64) * 100,
                n_bytes,
            })
            .collect()
    }

    fn descriptor_refs(descriptors: &[TensorDesc]) -> Vec<&TensorDesc> {
        descriptors.iter().collect()
    }

    #[test]
    fn worker_parser_accepts_exact_bounded_set() {
        for workers in ALLOWED_DIAGNOSTIC_WORKERS {
            assert_eq!(parse_worker_count(&workers.to_string()), Ok(workers));
            let args = GgufArenaFloorArgs::try_parse_from([
                "qwen",
                "--model",
                "fixture.gguf",
                "--describe",
                "--workers",
                &workers.to_string(),
            ])
            .expect("allowed worker count");
            assert_eq!(args.workers, workers);
        }
        let default =
            GgufArenaFloorArgs::try_parse_from(["qwen", "--model", "fixture.gguf", "--describe"])
                .expect("default worker count");
        assert_eq!(default.workers, FROZEN_PARALLEL_COPY_WORKERS);
    }

    #[test]
    fn worker_parser_rejects_other_and_malformed_values() {
        for workers in [0, 3, 5, 7, 9, 10, 11, 13] {
            let value = workers.to_string();
            assert!(parse_worker_count(&value).is_err());
            assert!(
                GgufArenaFloorArgs::try_parse_from([
                    "qwen",
                    "--model",
                    "fixture.gguf",
                    "--describe",
                    "--workers",
                    &value,
                ])
                .is_err()
            );
        }
        for malformed in ["", "abc", "1.0", "-1", " 4"] {
            assert!(parse_worker_count(malformed).is_err());
            assert!(
                GgufArenaFloorArgs::try_parse_from([
                    "qwen",
                    "--model",
                    "fixture.gguf",
                    "--describe",
                    "--workers",
                    malformed,
                ])
                .is_err()
            );
        }
    }

    #[test]
    fn transient_mmap_blit_parser_and_scope_are_a3b_only() {
        let args = GgufArenaFloorArgs::try_parse_from([
            "qwen",
            "--model",
            "fixture.gguf",
            "--profile",
            "a3b-q4km-v1",
            "--arm",
            "transient-mmap-blit",
        ])
        .expect("transient mmap-blit arguments");
        assert_eq!(args.arm, Some(ArenaFloorArm::TransientMmapBlit));
        assert!(validate_worker_scope(args.workers, args.describe, args.arm, args.profile).is_ok());
        assert!(
            validate_worker_scope(
                args.workers,
                false,
                args.arm,
                Some(FloorProfileId::Dense27bQ4kmV1),
            )
            .is_err()
        );
    }

    #[test]
    fn minimax_partition_supports_every_allowed_worker_count() {
        let lengths = [1; 24];
        for workers in ALLOWED_DIAGNOSTIC_WORKERS {
            let cuts = minimax_partition_cuts(&lengths, workers).expect("partition");
            let expected = (1..workers)
                .map(|partition| partition * lengths.len() / workers)
                .collect::<Vec<_>>();
            assert_eq!(cuts, expected, "W{workers}");
        }
        assert!(
            minimax_partition_cuts(&lengths, 1)
                .expect("W1 partition")
                .is_empty()
        );
    }

    #[test]
    fn minimax_partition_uses_lexicographically_first_equal_cuts() {
        assert_eq!(
            minimax_partition_cuts(&[1; 8], 4).expect("partition"),
            [2, 4, 6]
        );
        assert_eq!(
            minimax_partition_cuts(&[1, 1, 1, 2, 3], 4).expect("global tie partition"),
            [1, 2, 4]
        );
    }

    #[test]
    fn minimax_partition_keeps_a_giant_task_whole() {
        let lengths = [100, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1];
        for workers in ALLOWED_DIAGNOSTIC_WORKERS {
            let cuts = minimax_partition_cuts(&lengths, workers).expect("partition");
            if workers > 1 {
                assert_eq!(cuts[0], 1, "W{workers}");
            }
        }
    }

    #[test]
    fn minimax_partition_rejects_invalid_inputs() {
        assert!(minimax_partition_cuts(&[1], 0).is_err());
        assert!(minimax_partition_cuts(&[1, 1, 1], 4).is_err());
        assert!(minimax_partition_cuts(&[1, 1, 1, 0], 4).is_err());
        assert!(minimax_partition_cuts(&[u64::MAX, 1, 1, 1], 4).is_err());
    }

    #[test]
    fn dynamic_w1_schedule_covers_every_task_and_serializes() {
        let descriptors = descriptors(&[1, 2, 3, 4, 5]);
        let direct = descriptor_refs(&descriptors);
        let schedule = parallel_copy_schedule(&direct, 1).expect("W1 schedule");

        assert!(schedule.cuts.is_empty());
        assert_eq!(schedule.partitions.len(), 1);
        assert_eq!(schedule.partitions[0].start, 0);
        assert_eq!(schedule.partitions[0].end, direct.len());
        assert_eq!(schedule.partitions[0].bytes, 15);
        validate_parallel_copy_schedule(&schedule, &direct, 1).expect("valid W1 schedule");

        let encoded = parallel_copy_schedule_json(&schedule, &direct).expect("W1 schedule JSON");
        assert_eq!(encoded["algorithm"].as_str(), Some(PARALLEL_COPY_ALGORITHM));
        assert_eq!(encoded["workers"].as_u64(), Some(1));
        assert!(encoded["cuts"].as_array().expect("cuts array").is_empty());
        assert_eq!(
            encoded["worker_bytes"].as_array().expect("worker bytes")[0].as_u64(),
            Some(15)
        );
    }

    #[test]
    fn dynamic_w4_reproduces_legacy_synthetic_shape() {
        let descriptors = descriptors(&[1; 8]);
        let direct = descriptor_refs(&descriptors);
        let schedule = parallel_copy_schedule(&direct, 4).expect("W4 schedule");
        assert_eq!(schedule.cuts, [2, 4, 6]);
        assert_eq!(schedule.partitions.len(), 4);
        assert_eq!(
            schedule
                .partitions
                .iter()
                .map(|partition| partition.end - partition.start)
                .collect::<Vec<_>>(),
            [2, 2, 2, 2]
        );
    }

    #[test]
    fn source_order_is_shard_offset_then_request_index() {
        let mut descriptors = descriptors(&[4, 4, 4, 4]);
        descriptors[0].shard_idx = 1;
        descriptors[0].data_offset = 0;
        descriptors[1].shard_idx = 0;
        descriptors[1].data_offset = 100;
        descriptors[2].shard_idx = 0;
        descriptors[2].data_offset = 50;
        descriptors[3].shard_idx = 0;
        descriptors[3].data_offset = 100;
        let direct = descriptor_refs(&descriptors);
        assert_eq!(source_order(&direct), [2, 1, 3, 0]);
    }

    #[test]
    fn schedule_validation_rejects_structural_and_accounting_drift() {
        let descriptors = descriptors(&[1, 2, 3, 4, 5, 6]);
        let direct = descriptor_refs(&descriptors);
        let schedule = parallel_copy_schedule(&direct, 2).expect("schedule");

        let mut drifted = schedule.clone();
        drifted.cuts.clear();
        assert!(validate_parallel_copy_schedule(&drifted, &direct, 2).is_err());

        let mut drifted = schedule.clone();
        drifted.partitions.pop();
        assert!(validate_parallel_copy_schedule(&drifted, &direct, 2).is_err());

        let mut drifted = schedule.clone();
        drifted.cuts[0] = 0;
        assert!(validate_parallel_copy_schedule(&drifted, &direct, 2).is_err());

        let mut drifted = schedule.clone();
        drifted.partitions[1].start += 1;
        assert!(validate_parallel_copy_schedule(&drifted, &direct, 2).is_err());

        let mut drifted = schedule.clone();
        drifted.partitions[1].start -= 1;
        assert!(validate_parallel_copy_schedule(&drifted, &direct, 2).is_err());

        let mut drifted = schedule.clone();
        drifted.sorted_request_indices[0] = drifted.sorted_request_indices[1];
        assert!(validate_parallel_copy_schedule(&drifted, &direct, 2).is_err());

        let mut drifted = schedule;
        drifted.partitions[0].bytes += 1;
        assert!(validate_parallel_copy_schedule(&drifted, &direct, 2).is_err());
    }

    #[test]
    fn non_default_execution_scope_is_a3b_parallel_pread_only() {
        for workers in ALLOWED_DIAGNOSTIC_WORKERS {
            assert!(
                validate_worker_scope(
                    workers,
                    false,
                    Some(ArenaFloorArm::ParallelPread),
                    Some(FloorProfileId::A3bQ4kmV1),
                )
                .is_ok()
            );
        }

        for workers in ALLOWED_DIAGNOSTIC_WORKERS
            .into_iter()
            .filter(|&workers| workers != FROZEN_PARALLEL_COPY_WORKERS)
        {
            for arm in [
                ArenaFloorArm::Copied,
                ArenaFloorArm::ParallelCopied,
                ArenaFloorArm::TransientMmapBlit,
                ArenaFloorArm::ArenaSerial,
                ArenaFloorArm::ArenaFour,
            ] {
                assert!(
                    validate_worker_scope(
                        workers,
                        false,
                        Some(arm),
                        Some(FloorProfileId::A3bQ4kmV1),
                    )
                    .is_err()
                );
            }
            assert!(
                validate_worker_scope(
                    workers,
                    false,
                    Some(ArenaFloorArm::ParallelPread),
                    Some(FloorProfileId::Dense27bQ4kmV1),
                )
                .is_err()
            );
            assert!(validate_worker_scope(workers, false, None, None).is_err());
        }

        assert!(
            validate_worker_scope(
                FROZEN_PARALLEL_COPY_WORKERS,
                false,
                Some(ArenaFloorArm::TransientMmapBlit),
                Some(FloorProfileId::A3bQ4kmV1),
            )
            .is_ok()
        );
        assert!(
            validate_worker_scope(
                FROZEN_PARALLEL_COPY_WORKERS,
                false,
                Some(ArenaFloorArm::TransientMmapBlit),
                Some(FloorProfileId::Dense27bQ4kmV1),
            )
            .is_err()
        );
        assert!(
            validate_worker_scope(
                FROZEN_PARALLEL_COPY_WORKERS,
                false,
                Some(ArenaFloorArm::TransientMmapBlit),
                None,
            )
            .is_err()
        );

        for arm in [
            ArenaFloorArm::Copied,
            ArenaFloorArm::ParallelCopied,
            ArenaFloorArm::ParallelPread,
            ArenaFloorArm::ArenaSerial,
            ArenaFloorArm::ArenaFour,
        ] {
            assert!(
                validate_worker_scope(
                    FROZEN_PARALLEL_COPY_WORKERS,
                    false,
                    Some(arm),
                    Some(FloorProfileId::Dense27bQ4kmV1),
                )
                .is_ok()
            );
        }
    }

    #[test]
    fn non_default_describe_scope_requires_authenticated_a3b_geometry() {
        for workers in ALLOWED_DIAGNOSTIC_WORKERS {
            assert!(
                validate_describe_worker_scope(workers, Some(FloorProfileId::A3bQ4kmV1)).is_ok()
            );
        }

        for workers in ALLOWED_DIAGNOSTIC_WORKERS
            .into_iter()
            .filter(|&workers| workers != FROZEN_PARALLEL_COPY_WORKERS)
        {
            assert!(
                validate_describe_worker_scope(workers, Some(FloorProfileId::Dense27bQ4kmV1))
                    .is_err()
            );
            assert!(validate_describe_worker_scope(workers, None).is_err());
        }

        assert!(
            validate_describe_worker_scope(
                FROZEN_PARALLEL_COPY_WORKERS,
                Some(FloorProfileId::Dense27bQ4kmV1),
            )
            .is_ok()
        );
        assert!(validate_describe_worker_scope(FROZEN_PARALLEL_COPY_WORKERS, None).is_ok());
    }
}
