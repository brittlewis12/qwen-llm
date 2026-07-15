use crate::OutputFormat;
use anyhow::{Context, Result, anyhow};
use clap::{Parser, ValueEnum};
use objc2::rc::Retained;
use objc2_metal::{MTLBuffer, MTLCPUCacheMode, MTLHazardTrackingMode, MTLResource, MTLStorageMode};
use qwen_llm::{
    gguf::GgufFile,
    loader::Model,
    metal::{Buffer, MetalContext, RetainedStorageDisposition, host_page_size_bytes},
    metal_forward::{
        ModelWeightStorageKind, gguf_descriptor_layout_digest,
        model_weight_storage_inventory_digest, model_weight_storage_requests,
        production_native_quant_embedding_storage_enabled, retained_storage_plan_digest,
    },
    tensor::TensorDesc,
};
use serde_json::{Value, json};
use std::collections::{HashMap, HashSet};
use std::mem::MaybeUninit;
use std::path::PathBuf;
use std::time::{Duration, Instant};

const REQUIRED_ALIGNMENT: usize = 32;
const EXPECTED_PAGE_SIZE: usize = 16_384;
const EXPECTED_MAX_BUFFER_LENGTH: usize = 77_309_411_328;
const EXPECTED_REQUEST_COUNT: usize = 733;
const EXPECTED_VIEW_COUNT: usize = 732;
const EXPECTED_LOGICAL_COPY_BYTES: u64 = 22_123_538_944;
const EXPECTED_UNIQUE_VIEW_BYTES: u64 = 22_123_530_752;
const EXPECTED_FALLBACK_BYTES: u64 = 8_192;
const EXPECTED_WINDOW_BYTES: u64 = 22_123_544_576;
const EXPECTED_GAP_BYTES: u64 = 13_824;
const EXPECTED_ARENA_COPY_BYTES: u64 = 22_123_552_768;
const EXPECTED_ARCHITECTURE: &str = "qwen35moe";
const EXPECTED_DESCRIPTOR_DIGEST: &str = "0x5ae645df5cf7d568";
const EXPECTED_INVENTORY_DIGEST: &str =
    "f57153febec22463c7789b892d4d084041d722483a93191c81c40ab86be7d9e5";
const EXPECTED_PLANNER_DIGEST: &str =
    "fa2685e223ad8ea6271c6061041fe8d996b4e6cc70e060588b750732577c92af";
const PARALLEL_COPY_WORKERS: usize = 4;
const EXPECTED_PARALLEL_CUTS: [usize; 3] = [155, 359, 539];
const EXPECTED_PARALLEL_TASK_COUNTS: [usize; 4] = [155, 204, 180, 194];
const EXPECTED_PARALLEL_BYTES: [u64; 4] =
    [5_532_746_240, 5_462_315_776, 5_595_522_304, 5_532_954_624];

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum ArenaFloorArm {
    Copied,
    ParallelCopied,
    ArenaSerial,
    ArenaFour,
}

impl ArenaFloorArm {
    fn label(self) -> &'static str {
        match self {
            Self::Copied => "copied",
            Self::ParallelCopied => "parallel-copied",
            Self::ArenaSerial => "arena-serial",
            Self::ArenaFour => "arena-four",
        }
    }

    fn has_copied_topology(self) -> bool {
        matches!(self, Self::Copied | Self::ParallelCopied)
    }

    fn has_arena_topology(self) -> bool {
        matches!(self, Self::ArenaSerial | Self::ArenaFour)
    }
}

#[derive(Parser, Debug)]
pub(crate) struct GgufArenaFloorArgs {
    /// Path to the first GGUF shard.
    #[arg(short = 'm', long)]
    model: PathBuf,
    /// Materialization arm.
    #[arg(long, value_enum, required_unless_present = "describe")]
    arm: Option<ArenaFloorArm>,
    /// Emit authenticated geometry without touching payload bytes.
    #[arg(long)]
    describe: bool,
    /// `text` or `json`.
    #[arg(short = 'o', long, value_enum, default_value = "json")]
    output: OutputFormat,
}

#[derive(Clone, Copy)]
struct Usage {
    minor_faults: i64,
    major_faults: i64,
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
    binding_wall: Duration,
    worker_count: usize,
    schedule: Option<ParallelCopySchedule>,
}

struct Correctness {
    full_windows_checked: usize,
    fallback_bytes_checked: u64,
    entries_checked: usize,
    aliases_checked: usize,
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
    cuts: [usize; PARALLEL_COPY_WORKERS - 1],
    partitions: Vec<ParallelCopyPartition>,
}

struct ParallelCopyTask<'a> {
    source: &'a [u8],
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
    })
}

fn duration_ms(value: Duration) -> f64 {
    value.as_secs_f64() * 1e3
}

fn copy_into_buffer(buffer: &Buffer, offset: usize, source: &[u8]) -> Result<()> {
    let end = offset
        .checked_add(source.len())
        .ok_or_else(|| anyhow!("destination endpoint overflow"))?;
    if end > buffer.length() {
        return Err(anyhow!(
            "copy endpoint {end} exceeds buffer length {}",
            buffer.length()
        ));
    }
    let destination = unsafe {
        std::slice::from_raw_parts_mut(
            buffer.contents().as_ptr().cast::<u8>().add(offset),
            source.len(),
        )
    };
    destination.copy_from_slice(source);
    Ok(())
}

fn copy_four_workers(source: &[u8], buffer: &Buffer, page_size: usize) -> Result<usize> {
    if source.len() % page_size != 0 {
        return Err(anyhow!(
            "window length {} is not page aligned to {page_size}",
            source.len()
        ));
    }
    let pages = source.len() / page_size;
    if pages < 4 {
        return Err(anyhow!("window has only {pages} pages for four workers"));
    }
    if source.len() > buffer.length() {
        return Err(anyhow!("source exceeds destination buffer"));
    }
    let destination = unsafe {
        std::slice::from_raw_parts_mut(buffer.contents().as_ptr().cast::<u8>(), source.len())
    };
    let boundaries = four_worker_boundaries(source.len(), page_size)?;

    std::thread::scope(|scope| {
        let mut source_tail = source;
        let mut destination_tail = destination;
        let mut previous = 0usize;
        for &boundary in boundaries.iter().skip(1) {
            let length = boundary - previous;
            let (source_chunk, next_source) = source_tail.split_at(length);
            let (destination_chunk, next_destination) = destination_tail.split_at_mut(length);
            scope.spawn(move || destination_chunk.copy_from_slice(source_chunk));
            source_tail = next_source;
            destination_tail = next_destination;
            previous = boundary;
        }
    });
    Ok(4)
}

fn four_worker_boundaries(length: usize, page_size: usize) -> Result<[usize; 5]> {
    if page_size == 0 || length % page_size != 0 {
        return Err(anyhow!("four-worker range is not page aligned"));
    }
    let pages = length / page_size;
    if pages < 4 {
        return Err(anyhow!("four-worker range has fewer than four pages"));
    }
    let boundaries = std::array::from_fn(|worker| page_size * (worker * pages / 4));
    if boundaries[0] != 0 || boundaries[4] != length {
        return Err(anyhow!("four-worker boundaries do not cover the range"));
    }
    if boundaries.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err(anyhow!("four-worker boundaries overlap or are empty"));
    }
    Ok(boundaries)
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

fn minimax_partition_cuts(lengths: &[u64]) -> Result<[usize; PARALLEL_COPY_WORKERS - 1]> {
    if lengths.len() < PARALLEL_COPY_WORKERS || lengths.contains(&0) {
        return Err(anyhow!(
            "parallel copy requires at least four nonempty tasks"
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
        if can_partition_exactly(lengths, PARALLEL_COPY_WORKERS, midpoint) {
            upper = midpoint;
        } else {
            lower = midpoint + 1;
        }
    }

    let capacity = lower;
    let mut cuts = [0usize; PARALLEL_COPY_WORKERS - 1];
    let mut start = 0usize;
    for (cut_index, cut) in cuts.iter_mut().enumerate() {
        let remaining_groups = PARALLEL_COPY_WORKERS - cut_index - 1;
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

fn parallel_copy_schedule(direct: &[&TensorDesc]) -> Result<ParallelCopySchedule> {
    let mut sorted_request_indices = (0..direct.len()).collect::<Vec<_>>();
    sorted_request_indices.sort_by_key(|&request_index| {
        let desc = direct[request_index];
        (desc.shard_idx, desc.data_offset, request_index)
    });
    let lengths = sorted_request_indices
        .iter()
        .map(|&request_index| direct[request_index].n_bytes)
        .collect::<Vec<_>>();
    let cuts = minimax_partition_cuts(&lengths)?;
    let boundaries = [0, cuts[0], cuts[1], cuts[2], direct.len()];
    let mut partitions = Vec::with_capacity(PARALLEL_COPY_WORKERS);
    for worker in 0..PARALLEL_COPY_WORKERS {
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
    Ok(ParallelCopySchedule {
        sorted_request_indices,
        cuts,
        partitions,
    })
}

fn parallel_copy_schedule_json(schedule: &ParallelCopySchedule) -> Result<Value> {
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
        .expect("schedule has four partitions");
    let max_bytes = schedule
        .partitions
        .iter()
        .map(|partition| partition.bytes)
        .max()
        .expect("schedule has four partitions");
    Ok(json!({
        "workers": PARALLEL_COPY_WORKERS,
        "cuts": schedule.cuts,
        "task_counts": schedule.partitions.iter().map(|partition| {
            partition.end - partition.start
        }).collect::<Vec<_>>(),
        "worker_bytes": schedule.partitions.iter().map(|partition| {
            partition.bytes
        }).collect::<Vec<_>>(),
        "max_to_min": max_bytes as f64 / min_bytes as f64,
        "max_to_ideal": max_bytes as f64
            / (total_bytes as f64 / PARALLEL_COPY_WORKERS as f64),
        "partitions": schedule.partitions.iter().map(|partition| json!({
            "start": partition.start,
            "end": partition.end,
            "task_count": partition.end - partition.start,
            "bytes": partition.bytes,
            "first_shard": partition.first_shard,
            "first_source_offset": partition.first_source_offset,
            "last_shard": partition.last_shard,
            "last_source_offset": partition.last_source_offset,
        })).collect::<Vec<_>>(),
    }))
}

fn materialize_copied(
    ctx: &MetalContext,
    gguf: &GgufFile,
    direct: &[&TensorDesc],
) -> Result<Materialized> {
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
    Ok(Materialized {
        resources,
        bindings,
        allocation_wall: None,
        source_resolution_wall: None,
        copy_wall: None,
        binding_wall: binding_started.elapsed(),
        worker_count: 0,
        schedule: None,
    })
}

fn materialize_parallel_copied(
    ctx: &MetalContext,
    gguf: &GgufFile,
    direct: &[&TensorDesc],
    schedule: &ParallelCopySchedule,
) -> Result<Materialized> {
    let allocation_started = Instant::now();
    let resources = direct
        .iter()
        .map(|desc| {
            usize::try_from(desc.n_bytes)
                .map_err(|_| anyhow!("tensor byte length does not fit usize"))
                .and_then(|length| ctx.buffer_uninit(length).map_err(anyhow::Error::from))
        })
        .collect::<Result<Vec<_>>>()?;
    let allocation_wall = allocation_started.elapsed();

    let mut resource_identities = HashSet::with_capacity(resources.len());
    let mut destination_ranges = Vec::with_capacity(resources.len());
    for (request_index, buffer) in resources.iter().enumerate() {
        let identity = Retained::as_ptr(buffer) as *const () as usize;
        if !resource_identities.insert(identity) {
            return Err(anyhow!(
                "parallel-copy resource {request_index} aliases a prior resource"
            ));
        }
        let start = buffer.contents().as_ptr().cast::<u8>() as usize;
        let end = start
            .checked_add(buffer.length())
            .ok_or_else(|| anyhow!("parallel-copy destination range overflow"))?;
        destination_ranges.push((start, end));
    }
    let mut ranges_by_address = destination_ranges.clone();
    ranges_by_address.sort_unstable();
    if ranges_by_address
        .windows(2)
        .any(|pair| pair[0].1 > pair[1].0)
    {
        return Err(anyhow!("parallel-copy destination resources overlap"));
    }

    let source_started = Instant::now();
    let mut tasks = Vec::with_capacity(direct.len());
    for &request_index in &schedule.sorted_request_indices {
        let desc = direct[request_index];
        let source = gguf.try_slice(desc)?;
        let (start, end) = destination_ranges[request_index];
        if source.len() != end - start {
            return Err(anyhow!(
                "parallel-copy request {request_index} source/destination mismatch"
            ));
        }
        let destination = unsafe {
            // SAFETY: every exact-sized buffer has a unique Objective-C identity,
            // the destination ranges were proven pairwise disjoint, and each range
            // is assigned to exactly one task before any worker starts.
            std::slice::from_raw_parts_mut(start as *mut u8, end - start)
        };
        tasks.push(ParallelCopyTask {
            source,
            destination,
        });
    }
    let source_resolution_wall = source_started.elapsed();

    let copy_started = Instant::now();
    std::thread::scope(|scope| {
        let mut task_tail = tasks.as_mut_slice();
        let mut consumed = 0usize;
        for partition in &schedule.partitions {
            assert_eq!(partition.start, consumed);
            let count = partition.end - partition.start;
            let (worker_tasks, remaining) = task_tail.split_at_mut(count);
            scope.spawn(move || {
                for task in worker_tasks {
                    task.destination.copy_from_slice(task.source);
                }
            });
            task_tail = remaining;
            consumed = partition.end;
        }
        assert_eq!(consumed, schedule.sorted_request_indices.len());
        assert!(task_tail.is_empty());
    });
    let copy_wall = copy_started.elapsed();
    drop(tasks);

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
    Ok(Materialized {
        resources,
        bindings,
        allocation_wall: Some(allocation_wall),
        source_resolution_wall: Some(source_resolution_wall),
        copy_wall: Some(copy_wall),
        binding_wall: binding_started.elapsed(),
        worker_count: PARALLEL_COPY_WORKERS,
        schedule: Some(schedule.clone()),
    })
}

fn materialize_arena(
    ctx: &MetalContext,
    gguf: &GgufFile,
    direct: &[&TensorDesc],
    plan: &qwen_llm::metal::RetainedStoragePlan,
    four_workers: bool,
) -> Result<Materialized> {
    let allocation_started = Instant::now();
    let mut resources = Vec::with_capacity(plan.windows.len() + plan.entries.len());
    for window in &plan.windows {
        resources.push(ctx.buffer_uninit(window.length)?);
    }
    let mut fallback_resources = HashMap::new();
    for entry in &plan.entries {
        if matches!(
            entry.disposition,
            RetainedStorageDisposition::CopyFallback { .. }
        ) {
            let length = usize::try_from(entry.n_bytes)
                .map_err(|_| anyhow!("fallback byte length does not fit usize"))?;
            let resource_index = resources.len();
            resources.push(ctx.buffer_uninit(length)?);
            fallback_resources.insert(entry.request_index, resource_index);
        }
    }
    let allocation_wall = allocation_started.elapsed();

    let mut source_resolution_wall = Duration::ZERO;
    let mut copy_wall = Duration::ZERO;
    let mut worker_count = 0usize;
    for (window_index, window) in plan.windows.iter().enumerate() {
        let source_started = Instant::now();
        let source = gguf.try_shard_range(window.shard_idx, window.mmap_offset, window.length)?;
        source_resolution_wall += source_started.elapsed();
        let copy_started = Instant::now();
        if four_workers {
            worker_count += copy_four_workers(source, &resources[window_index], plan.page_size)?;
        } else {
            copy_into_buffer(&resources[window_index], 0, source)?;
        }
        copy_wall += copy_started.elapsed();
    }
    for entry in &plan.entries {
        if let RetainedStorageDisposition::CopyFallback { .. } = entry.disposition {
            let desc = direct
                .get(entry.request_index)
                .ok_or_else(|| anyhow!("fallback request index is out of bounds"))?;
            let source_started = Instant::now();
            let source = gguf.try_slice(desc)?;
            source_resolution_wall += source_started.elapsed();
            let resource_index = *fallback_resources
                .get(&entry.request_index)
                .ok_or_else(|| anyhow!("fallback resource is missing"))?;
            let copy_started = Instant::now();
            copy_into_buffer(&resources[resource_index], 0, source)?;
            copy_wall += copy_started.elapsed();
        }
    }

    let binding_started = Instant::now();
    let mut bindings = Vec::with_capacity(plan.entries.len());
    for entry in &plan.entries {
        let (resource_index, offset) = match entry.disposition {
            RetainedStorageDisposition::View {
                window_index,
                buffer_offset,
            } => (
                window_index,
                usize::try_from(buffer_offset)
                    .map_err(|_| anyhow!("planned buffer offset does not fit usize"))?,
            ),
            RetainedStorageDisposition::CopyFallback { .. } => (
                *fallback_resources
                    .get(&entry.request_index)
                    .ok_or_else(|| anyhow!("fallback resource is missing"))?,
                0,
            ),
            RetainedStorageDisposition::Alias {
                source_request_index,
            } => {
                let source: &Binding = bindings
                    .get(source_request_index)
                    .ok_or_else(|| anyhow!("alias source binding is unavailable"))?;
                (source.resource_index, source.offset)
            }
        };
        let buffer = resources
            .get(resource_index)
            .ok_or_else(|| anyhow!("planned resource index is out of bounds"))?
            .clone();
        bindings.push(Binding {
            buffer,
            resource_index,
            offset,
            length: usize::try_from(entry.n_bytes)
                .map_err(|_| anyhow!("entry byte length does not fit usize"))?,
        });
    }
    Ok(Materialized {
        resources,
        bindings,
        allocation_wall: Some(allocation_wall),
        source_resolution_wall: Some(source_resolution_wall),
        copy_wall: Some(copy_wall),
        binding_wall: binding_started.elapsed(),
        worker_count,
        schedule: None,
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
    arm: ArenaFloorArm,
    gguf: &GgufFile,
    direct: &[&TensorDesc],
    plan: &qwen_llm::metal::RetainedStoragePlan,
    materialized: &Materialized,
    alignment: usize,
) -> Result<Correctness> {
    if materialized.bindings.len() != direct.len() {
        return Err(anyhow!("binding count does not match direct requests"));
    }
    let resource_bytes = materialized
        .resources
        .iter()
        .try_fold(0u64, |total, buffer| {
            total
                .checked_add(buffer.length() as u64)
                .ok_or_else(|| anyhow!("resource byte accounting overflow"))
        })?;
    let binding_bytes = materialized
        .bindings
        .iter()
        .try_fold(0u64, |total, binding| {
            total
                .checked_add(binding.length as u64)
                .ok_or_else(|| anyhow!("binding byte accounting overflow"))
        })?;
    if binding_bytes != EXPECTED_LOGICAL_COPY_BYTES {
        return Err(anyhow!("binding byte ledger drifted"));
    }
    for buffer in &materialized.resources {
        if buffer.storageMode() != MTLStorageMode::Shared
            || buffer.cpuCacheMode() != MTLCPUCacheMode::DefaultCache
            || buffer.hazardTrackingMode() != MTLHazardTrackingMode::Tracked
        {
            return Err(anyhow!("materialized resource mode drifted"));
        }
    }

    if arm.has_copied_topology() {
        let resource_identities = materialized
            .resources
            .iter()
            .map(|buffer| Retained::as_ptr(buffer) as *const () as usize)
            .collect::<HashSet<_>>();
        if materialized.resources.len() != EXPECTED_REQUEST_COUNT
            || resource_identities.len() != EXPECTED_REQUEST_COUNT
            || resource_bytes != EXPECTED_LOGICAL_COPY_BYTES
        {
            return Err(anyhow!("copied-topology resource ledger drifted"));
        }
        for (index, ((buffer, desc), binding)) in materialized
            .resources
            .iter()
            .zip(direct)
            .zip(&materialized.bindings)
            .enumerate()
        {
            if buffer.length() as u64 != desc.n_bytes
                || binding.resource_index != index
                || binding.offset != 0
                || binding.length != buffer.length()
                || Retained::as_ptr(&binding.buffer) != Retained::as_ptr(buffer)
            {
                return Err(anyhow!(
                    "copied-topology resource or binding {index} drifted"
                ));
            }
        }
    } else {
        match arm {
            ArenaFloorArm::ArenaSerial | ArenaFloorArm::ArenaFour => {
                if materialized.resources.len() != 2 || resource_bytes != EXPECTED_ARENA_COPY_BYTES
                {
                    return Err(anyhow!("arena resource ledger drifted"));
                }
            }
            ArenaFloorArm::Copied | ArenaFloorArm::ParallelCopied => unreachable!(),
        }
    }

    let mut full_windows_checked = 0usize;
    if arm.has_arena_topology() {
        for (window_index, window) in plan.windows.iter().enumerate() {
            let source =
                gguf.try_shard_range(window.shard_idx, window.mmap_offset, window.length)?;
            let actual = buffer_bytes(&materialized.resources[window_index], 0, window.length)?;
            if actual != source {
                return Err(anyhow!("arena window {window_index} differs from source"));
            }
            if materialized.resources[window_index].length() != window.length {
                return Err(anyhow!("arena window {window_index} length drifted"));
            }
            full_windows_checked += 1;
        }
    }

    let mut fallback_bytes_checked = 0u64;
    let mut aliases_checked = 0usize;
    for (request_index, ((desc, entry), binding)) in direct
        .iter()
        .zip(&plan.entries)
        .zip(&materialized.bindings)
        .enumerate()
    {
        if entry.request_index != request_index
            || entry.name != desc.name
            || entry.shard_idx != desc.shard_idx
            || entry.data_offset != desc.data_offset
            || entry.n_bytes != desc.n_bytes
        {
            return Err(anyhow!(
                "planner entry {request_index} drifted from request"
            ));
        }
        if binding.offset % alignment != 0 {
            return Err(anyhow!("binding {request_index} is misaligned"));
        }
        let source = gguf.try_slice(desc)?;
        let actual = buffer_bytes(&binding.buffer, binding.offset, binding.length)?;
        if actual != source {
            return Err(anyhow!("binding {request_index} differs from source"));
        }
        match entry.disposition {
            RetainedStorageDisposition::CopyFallback { .. } => {
                if materialized.resources[binding.resource_index].length() as u64 != entry.n_bytes {
                    return Err(anyhow!("fallback resource length drifted"));
                }
                fallback_bytes_checked = fallback_bytes_checked
                    .checked_add(entry.n_bytes)
                    .ok_or_else(|| anyhow!("fallback verification bytes overflow"))?;
            }
            RetainedStorageDisposition::Alias {
                source_request_index,
            } => {
                let source_binding = materialized
                    .bindings
                    .get(source_request_index)
                    .ok_or_else(|| anyhow!("alias source binding is unavailable"))?;
                if Retained::as_ptr(&binding.buffer) != Retained::as_ptr(&source_binding.buffer)
                    || binding.offset != source_binding.offset
                {
                    return Err(anyhow!(
                        "alias {request_index} does not share its source view"
                    ));
                }
                aliases_checked += 1;
            }
            RetainedStorageDisposition::View { .. } => {}
        }
    }
    Ok(Correctness {
        full_windows_checked,
        fallback_bytes_checked,
        entries_checked: direct.len(),
        aliases_checked,
    })
}

pub(crate) fn run(args: GgufArenaFloorArgs, build_identity: Value) -> Result<()> {
    let gguf = GgufFile::open(&args.model).context("open GGUF")?;
    let model = Model::from_gguf(&gguf).context("bind model")?;
    let native_embedding = production_native_quant_embedding_storage_enabled(&model);
    if !native_embedding {
        return Err(anyhow!(
            "owned-arena floor requires the production native embedding policy"
        ));
    }
    let requests = model_weight_storage_requests(&model, native_embedding, false)?;
    if requests
        .iter()
        .any(|request| request.kind != ModelWeightStorageKind::Direct)
    {
        return Err(anyhow!(
            "owned-arena floor requires an all-direct inventory"
        ));
    }
    let direct = requests
        .iter()
        .map(|request| request.desc)
        .collect::<Vec<_>>();
    let parallel_schedule = parallel_copy_schedule(&direct)?;
    let page_size = host_page_size_bytes()?;
    let ctx = MetalContext::new()?;
    let max_buffer_length = ctx.max_buffer_length();
    let plan = qwen_llm::metal::plan_retained_storage(
        &gguf.shard_mapped_lengths(),
        &direct,
        page_size,
        max_buffer_length,
        REQUIRED_ALIGNMENT,
    )?;

    let logical_copy_bytes = direct.iter().try_fold(0u64, |total, desc| {
        total
            .checked_add(desc.n_bytes)
            .ok_or_else(|| anyhow!("logical byte accounting overflow"))
    })?;
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
        .filter(|entry| matches!(entry.disposition, RetainedStorageDisposition::Alias { .. }))
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
        .filter(|entry| matches!(entry.disposition, RetainedStorageDisposition::View { .. }))
        .count();
    let descriptor_digest = format!("{:#018x}", gguf_descriptor_layout_digest(&gguf));
    let inventory_digest = model_weight_storage_inventory_digest(&requests);
    let planner_digest = retained_storage_plan_digest(&plan);

    if args.describe {
        let row = json!({
            "schema_version": 1,
            "mode": "describe",
            "model": args.model,
            "architecture": gguf.architecture(),
            "descriptor_layout_digest": descriptor_digest,
            "inventory_digest": inventory_digest,
            "planner_digest": planner_digest,
            "native_quant_embedding": native_embedding,
            "page_size": page_size,
            "required_alignment": REQUIRED_ALIGNMENT,
            "max_buffer_length": max_buffer_length,
            "request_count": direct.len(),
            "view_count": view_count,
            "window_count": plan.windows.len(),
            "fallback_count": fallback_count,
            "alias_count": alias_count,
            "logical_copy_bytes": logical_copy_bytes,
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
            "parallel_copy_schedule": parallel_copy_schedule_json(&parallel_schedule)?,
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

    if page_size != EXPECTED_PAGE_SIZE
        || gguf.architecture().as_deref() != Some(EXPECTED_ARCHITECTURE)
        || descriptor_digest != EXPECTED_DESCRIPTOR_DIGEST
        || inventory_digest != EXPECTED_INVENTORY_DIGEST
        || planner_digest != EXPECTED_PLANNER_DIGEST
        || max_buffer_length != EXPECTED_MAX_BUFFER_LENGTH
        || direct.len() != EXPECTED_REQUEST_COUNT
        || view_count != EXPECTED_VIEW_COUNT
        || alias_count != 0
        || fallback_count != 1
        || plan.windows.len() != 1
        || logical_copy_bytes != EXPECTED_LOGICAL_COPY_BYTES
        || plan.unique_view_bytes != EXPECTED_UNIQUE_VIEW_BYTES
        || plan.logical_view_bytes != EXPECTED_UNIQUE_VIEW_BYTES
        || plan.unique_fallback_bytes != EXPECTED_FALLBACK_BYTES
        || window_bytes != EXPECTED_WINDOW_BYTES
        || planner_gap_bytes != EXPECTED_GAP_BYTES
        || arena_copy_bytes != EXPECTED_ARENA_COPY_BYTES
        || parallel_schedule.cuts != EXPECTED_PARALLEL_CUTS
        || parallel_schedule
            .partitions
            .iter()
            .map(|partition| partition.end - partition.start)
            .collect::<Vec<_>>()
            .as_slice()
            != EXPECTED_PARALLEL_TASK_COUNTS.as_slice()
        || parallel_schedule
            .partitions
            .iter()
            .map(|partition| partition.bytes)
            .collect::<Vec<_>>()
            .as_slice()
            != EXPECTED_PARALLEL_BYTES.as_slice()
    {
        return Err(anyhow!("owned-arena frozen geometry drifted"));
    }
    let fallback = plan
        .entries
        .iter()
        .find(|entry| {
            matches!(
                entry.disposition,
                RetainedStorageDisposition::CopyFallback { .. }
            )
        })
        .ok_or_else(|| anyhow!("owned-arena fallback is missing"))?;
    if fallback.n_bytes != EXPECTED_FALLBACK_BYTES
        || !matches!(
            fallback.disposition,
            RetainedStorageDisposition::CopyFallback {
                reason: qwen_llm::metal::RetainedStorageFallback::FinalPartialPage
            }
        )
    {
        return Err(anyhow!("owned-arena fallback provenance drifted"));
    }

    let allocated_before = ctx.current_allocated_size();
    let usage_before = capture_usage()?;
    let ready_started = Instant::now();
    let materialized = match arm {
        ArenaFloorArm::Copied => materialize_copied(&ctx, &gguf, &direct)?,
        ArenaFloorArm::ParallelCopied => {
            materialize_parallel_copied(&ctx, &gguf, &direct, &parallel_schedule)?
        }
        ArenaFloorArm::ArenaSerial => materialize_arena(&ctx, &gguf, &direct, &plan, false)?,
        ArenaFloorArm::ArenaFour => materialize_arena(&ctx, &gguf, &direct, &plan, true)?,
    };
    let ready_wall = ready_started.elapsed();
    let usage_after = capture_usage()?;
    let allocated_ready = ctx.current_allocated_size();
    if (arm == ArenaFloorArm::ParallelCopied
        && materialized.schedule.as_ref() != Some(&parallel_schedule))
        || (arm != ArenaFloorArm::ParallelCopied && materialized.schedule.is_some())
    {
        return Err(anyhow!("materialized parallel-copy schedule drifted"));
    }

    let correctness = verify_materialized(
        arm,
        &gguf,
        &direct,
        &plan,
        &materialized,
        REQUIRED_ALIGNMENT,
    )?;
    let resource_count = materialized.resources.len();
    let binding_count = materialized.bindings.len();
    let unattributed_wall = match (
        materialized.allocation_wall,
        materialized.source_resolution_wall,
        materialized.copy_wall,
    ) {
        (Some(allocation), Some(source_resolution), Some(copy)) => {
            let accounted = allocation
                .checked_add(source_resolution)
                .and_then(|value| value.checked_add(copy))
                .and_then(|value| value.checked_add(materialized.binding_wall))
                .ok_or_else(|| anyhow!("subinterval duration overflow"))?;
            Some(
                ready_wall
                    .checked_sub(accounted)
                    .ok_or_else(|| anyhow!("subintervals exceed ready wall"))?,
            )
        }
        (None, None, None) => None,
        _ => return Err(anyhow!("partial subinterval timing is invalid")),
    };
    let allocation_wall = materialized.allocation_wall.map(duration_ms);
    let source_resolution_wall = materialized.source_resolution_wall.map(duration_ms);
    let copy_wall = materialized.copy_wall.map(duration_ms);
    let binding_wall = duration_ms(materialized.binding_wall);
    let worker_count = materialized.worker_count;
    let teardown_started = Instant::now();
    drop(materialized);
    let teardown_wall = teardown_started.elapsed();
    let allocated_after_drop = ctx.current_allocated_size();

    let physical_bytes = match arm {
        ArenaFloorArm::Copied | ArenaFloorArm::ParallelCopied => logical_copy_bytes,
        ArenaFloorArm::ArenaSerial | ArenaFloorArm::ArenaFour => arena_copy_bytes,
    };
    let ready_gbps = physical_bytes as f64 / ready_wall.as_secs_f64() / 1e9;
    let copy_gbps = copy_wall.map(|wall_ms| physical_bytes as f64 / (wall_ms / 1e3) / 1e9);
    let row = json!({
        "schema_version": 1,
        "arm": arm.label(),
        "model": args.model,
        "architecture": gguf.architecture(),
        "descriptor_layout_digest": descriptor_digest,
        "inventory_digest": inventory_digest,
        "planner_digest": planner_digest,
        "native_quant_embedding": native_embedding,
        "page_size": page_size,
        "required_alignment": REQUIRED_ALIGNMENT,
        "max_buffer_length": max_buffer_length,
        "request_count": direct.len(),
        "view_count": view_count,
        "window_count": plan.windows.len(),
        "fallback_count": fallback_count,
        "alias_count": alias_count,
        "resource_count": resource_count,
        "binding_count": binding_count,
        "logical_copy_bytes": logical_copy_bytes,
        "unique_view_bytes": plan.unique_view_bytes,
        "logical_view_bytes": plan.logical_view_bytes,
        "fallback_bytes": plan.unique_fallback_bytes,
        "window_bytes": window_bytes,
        "planner_gap_bytes": planner_gap_bytes,
        "arena_copy_bytes": arena_copy_bytes,
        "physical_copy_bytes": physical_bytes,
        "fallback_reasons": plan.entries.iter().filter_map(|entry| {
            match entry.disposition {
                RetainedStorageDisposition::CopyFallback { reason } => {
                    Some(format!("{reason:?}"))
                }
                _ => None,
            }
        }).collect::<Vec<_>>(),
        "resource_modes": {
            "creation_storage": "shared",
            "creation_cpu_cache": "default_cache",
            "creation_hazard_tracking": "default",
            "observed_storage": "shared",
            "observed_cpu_cache": "default_cache",
            "observed_hazard_tracking": "tracked",
        },
        "parallel_copy_schedule": parallel_copy_schedule_json(&parallel_schedule)?,
        "timing": {
            "ready_wall_ms": duration_ms(ready_wall),
            "allocation_wall_ms": allocation_wall,
            "source_resolution_wall_ms": source_resolution_wall,
            "copy_wall_ms": copy_wall,
            "binding_wall_ms": binding_wall,
            "unattributed_wall_ms": unattributed_wall.map(duration_ms),
            "teardown_wall_ms": duration_ms(teardown_wall),
        },
        "throughput": {
            "ready_gbps_decimal": ready_gbps,
            "copy_gbps_decimal": copy_gbps,
        },
        "rusage": {
            "timer_minor_faults": usage_after.minor_faults - usage_before.minor_faults,
            "timer_major_faults": usage_after.major_faults - usage_before.major_faults,
        },
        "metal_allocated_bytes": {
            "before": allocated_before,
            "ready": allocated_ready,
            "after_drop": allocated_after_drop,
        },
        "correctness": {
            "passed": true,
            "full_windows_checked": correctness.full_windows_checked,
            "fallback_bytes_checked": correctness.fallback_bytes_checked,
            "entries_checked": correctness.entries_checked,
            "aliases_checked": correctness.aliases_checked,
        },
        "worker_count": worker_count,
        "build_identity": build_identity,
    });
    match args.output {
        OutputFormat::Json => println!("{}", serde_json::to_string_pretty(&row)?),
        OutputFormat::Text => {
            println!("arm\t{}", arm.label());
            println!("ready_wall_ms\t{:.3}", duration_ms(ready_wall));
            println!("ready_gbps_decimal\t{ready_gbps:.3}");
            println!(
                "timer_major_faults\t{}",
                usage_after.major_faults - usage_before.major_faults
            );
            println!("correctness\tpass");
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{four_worker_boundaries, minimax_partition_cuts};

    #[test]
    fn four_worker_boundaries_cover_nondivisible_page_counts() {
        let page = 16_384;
        let boundaries = four_worker_boundaries(11 * page, page).expect("valid boundaries");
        assert_eq!(boundaries, [0, 2 * page, 5 * page, 8 * page, 11 * page]);
        assert!(four_worker_boundaries(3 * page, page).is_err());
        assert!(four_worker_boundaries(4 * page + 1, page).is_err());
    }

    #[test]
    fn minimax_partition_uses_lexicographically_first_equal_cuts() {
        assert_eq!(
            minimax_partition_cuts(&[1; 8]).expect("partition"),
            [2, 4, 6]
        );
        assert_eq!(
            minimax_partition_cuts(&[1, 1, 1, 2, 3]).expect("global tie partition"),
            [1, 2, 4]
        );
    }

    #[test]
    fn minimax_partition_keeps_a_giant_task_whole() {
        assert_eq!(
            minimax_partition_cuts(&[10, 1, 1, 1, 1, 1]).expect("partition"),
            [1, 2, 3]
        );
    }

    #[test]
    fn minimax_partition_rejects_invalid_inputs() {
        assert!(minimax_partition_cuts(&[1, 1, 1]).is_err());
        assert!(minimax_partition_cuts(&[1, 1, 1, 0]).is_err());
        assert!(minimax_partition_cuts(&[u64::MAX, 1, 1, 1]).is_err());
    }
}
