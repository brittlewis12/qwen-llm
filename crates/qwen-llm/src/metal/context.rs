//! MetalContext: device, queue, pipeline cache, memory admission, process lease.

use super::*;

pub(crate) const METAL_PROCESS_LEASE_WAIT_ENV: &str = "QWEN_METAL_LEASE_WAIT";

pub(crate) const METAL_PROCESS_LEASE_SKIP_WIRED_GATE_ENV: &str = "QWEN_METAL_LEASE_SKIP_WIRED_GATE";

pub(crate) const UNSAFE_WIRED_MEMORY_DIVISOR: u64 = 2;

pub(crate) const WIRED_MEMORY_STABILIZATION_POLLS: usize = 30;

pub(crate) const WIRED_MEMORY_STABILIZATION_INTERVAL: std::time::Duration =
    std::time::Duration::from_millis(500);

pub(crate) struct MetalProcessLease {
    pub(crate) _file: File,
    pub(crate) _path: PathBuf,
    pub(crate) owner_pid: u32,
}

pub(crate) static METAL_PROCESS_LEASE: OnceLock<Arc<MetalProcessLease>> = OnceLock::new();

pub(crate) static METAL_PROCESS_LEASE_ACQUIRE: OnceLock<Mutex<()>> = OnceLock::new();

pub(crate) fn metal_process_lease_path() -> Result<PathBuf, MetalError> {
    // SAFETY: geteuid has no preconditions and cannot fail.
    let uid = unsafe { libc::geteuid() };
    #[cfg(test)]
    let directory =
        std::env::temp_dir().join(format!("qwen-llm-metal-tests-{}", std::process::id()));
    #[cfg(not(test))]
    let directory = PathBuf::from("/tmp").join(format!("qwen-llm-{uid}"));
    secure_metal_process_lease_path(directory, uid)
}

fn secure_metal_process_lease_path(
    directory: PathBuf,
    uid: libc::uid_t,
) -> Result<PathBuf, MetalError> {
    match std::fs::symlink_metadata(&directory) {
        Ok(metadata) => {
            if !metadata.is_dir() || metadata.uid() != uid {
                return Err(MetalError::ProcessLease(format!(
                    "lease directory {} is not a user-owned directory",
                    directory.display()
                )));
            }
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            std::fs::create_dir_all(&directory).map_err(|error| {
                MetalError::ProcessLease(format!(
                    "create lease directory {}: {error}",
                    directory.display()
                ))
            })?;
        }
        Err(error) => {
            return Err(MetalError::ProcessLease(format!(
                "inspect lease directory {}: {error}",
                directory.display()
            )));
        }
    }
    std::fs::set_permissions(&directory, std::fs::Permissions::from_mode(0o700)).map_err(
        |error| {
            MetalError::ProcessLease(format!(
                "secure lease directory {}: {error}",
                directory.display()
            ))
        },
    )?;
    let secured = std::fs::symlink_metadata(&directory).map_err(|error| {
        MetalError::ProcessLease(format!(
            "verify lease directory {}: {error}",
            directory.display()
        ))
    })?;
    if !secured.is_dir() || secured.uid() != uid || secured.mode() & 0o077 != 0 {
        return Err(MetalError::ProcessLease(format!(
            "lease directory {} did not retain private ownership and mode",
            directory.display()
        )));
    }
    Ok(directory.join("metal.lock"))
}

pub(crate) fn flock_file(file: &File, operation: libc::c_int) -> std::io::Result<()> {
    // SAFETY: file owns a valid descriptor and flock accepts this operation.
    if unsafe { libc::flock(file.as_raw_fd(), operation) } == 0 {
        Ok(())
    } else {
        // Do not hide EINTR: a queue-managed process must be able to unwind
        // after its cooperative SIGINT/SIGTERM handler interrupts a wait.
        Err(std::io::Error::last_os_error())
    }
}

pub(crate) fn lease_owner_field(value: &str) -> String {
    value
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_' | '.' | '/') {
                ch
            } else {
                '_'
            }
        })
        .take(512)
        .collect()
}

pub(crate) fn lease_owner_display(value: &str) -> String {
    value
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || matches!(ch, ' ' | '=' | '-' | '_' | '.' | '/' | ':') {
                ch
            } else {
                '_'
            }
        })
        .take(512)
        .collect()
}

pub(crate) fn metal_process_lease_owner() -> String {
    let executable = std::env::current_exe()
        .ok()
        .and_then(|path| path.file_name().map(|name| name.to_owned()))
        .and_then(|name| name.into_string().ok())
        .unwrap_or_else(|| "unknown".to_string());
    let session = std::env::var("OPENCODE_CALLER_SESSION_ID")
        .map(|value| lease_owner_field(&value))
        .unwrap_or_else(|_| "none".to_string());
    format!(
        "pid={} executable={} opencode_session={}\n",
        std::process::id(),
        lease_owner_field(&executable),
        session,
    )
}

pub(crate) fn read_metal_process_lease_owner(file: &mut File) -> String {
    if file.seek(SeekFrom::Start(0)).is_err() {
        return "owner metadata unavailable".to_string();
    }
    let mut owner = String::new();
    if file.take(4096).read_to_string(&mut owner).is_err() || owner.trim().is_empty() {
        "owner metadata unavailable".to_string()
    } else {
        lease_owner_display(owner.trim())
    }
}

pub(crate) fn validate_metal_process_lease_file(
    file: &File,
    path: &Path,
) -> Result<(), MetalError> {
    // SAFETY: geteuid has no preconditions and cannot fail.
    let uid = unsafe { libc::geteuid() };
    let opened = file.metadata().map_err(|error| {
        MetalError::ProcessLease(format!("inspect open lease {}: {error}", path.display()))
    })?;
    let linked = std::fs::symlink_metadata(path).map_err(|error| {
        MetalError::ProcessLease(format!("inspect linked lease {}: {error}", path.display()))
    })?;
    if !opened.is_file()
        || !linked.is_file()
        || opened.uid() != uid
        || linked.uid() != uid
        || opened.mode() & 0o077 != 0
        || linked.mode() & 0o077 != 0
        || opened.nlink() != 1
        || opened.dev() != linked.dev()
        || opened.ino() != linked.ino()
    {
        return Err(MetalError::ProcessLease(format!(
            "lease {} is not one user-owned regular inode",
            path.display()
        )));
    }
    Ok(())
}

pub(crate) fn open_metal_process_lease(
    path: &Path,
    wait: bool,
) -> Result<MetalProcessLease, MetalError> {
    let mut file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .mode(0o600)
        .custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW)
        .open(path)
        .map_err(|error| MetalError::ProcessLease(format!("open {}: {error}", path.display())))?;
    file.set_permissions(std::fs::Permissions::from_mode(0o600))
        .map_err(|error| {
            MetalError::ProcessLease(format!("secure lease {}: {error}", path.display()))
        })?;
    validate_metal_process_lease_file(&file, path)?;
    match flock_file(&file, libc::LOCK_EX | libc::LOCK_NB) {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
            let owner = read_metal_process_lease_owner(&mut file);
            if !wait {
                return Err(MetalError::ProcessLease(format!(
                    "another qwen process owns {} ({owner}); queue the work or set {METAL_PROCESS_LEASE_WAIT_ENV}=1 to wait",
                    path.display(),
                )));
            }
            // No profile script anchors this line, so choosing the
            // `qwen_diag` target (bare body, no `WARN` badge) is a
            // stylistic call rather than a byte-preservation requirement:
            // operators have hit `metal: waiting for process lease` at
            // column zero for years, and preserving that muscle-memory
            // anchor for the "why is it hanging?" case reads better than
            // burying it under a timestamp+level+target prefix. Level is
            // `warn` rather than `info` so it survives `RUST_LOG=warn`
            // while an operator investigates the hang; severity here is a
            // filter directive, not a visual marker. Users who want it
            // gone can `RUST_LOG=qwen_diag=off`.
            tracing::warn!(
                target: "qwen_diag",
                "metal: waiting for process lease {} ({owner})",
                path.display(),
            );
            flock_file(&file, libc::LOCK_EX).map_err(|error| {
                MetalError::ProcessLease(format!("wait for {}: {error}", path.display()))
            })?;
        }
        Err(error) => {
            return Err(MetalError::ProcessLease(format!(
                "lock {}: {error}",
                path.display()
            )));
        }
    }
    validate_metal_process_lease_file(&file, path)?;
    file.set_len(0).map_err(|error| {
        MetalError::ProcessLease(format!("truncate {}: {error}", path.display()))
    })?;
    file.seek(SeekFrom::Start(0))
        .map_err(|error| MetalError::ProcessLease(format!("rewind {}: {error}", path.display())))?;
    // This record is diagnostic attribution, not lock authority. The flock is
    // authoritative, so do not put a device durability barrier on acquisition;
    // a crash may leave stale or partial text that the next owner overwrites.
    file.write_all(metal_process_lease_owner().as_bytes())
        .and_then(|()| file.flush())
        .map_err(|error| {
            MetalError::ProcessLease(format!("write owner to {}: {error}", path.display()))
        })?;
    Ok(MetalProcessLease {
        _file: file,
        _path: path.to_owned(),
        owner_pid: std::process::id(),
    })
}

pub(crate) fn host_wired_memory_is_unsafe(wired_bytes: u64, physical_bytes: u64) -> bool {
    physical_bytes > 0 && wired_bytes >= physical_bytes / UNSAFE_WIRED_MEMORY_DIVISOR
}

pub(crate) fn ensure_host_wired_memory_is_safe() -> Result<(), MetalError> {
    if cfg!(test) || crate::env_flag::read_default_off(METAL_PROCESS_LEASE_SKIP_WIRED_GATE_ENV) {
        if !cfg!(test) {
            // Not routed through `qwen_diag`: no profile script parses this
            // line, so we prefer the default `Full` formatter — the visible
            // `WARN` badge matters for a safety-gate bypass and the extra
            // module-path prefix is fine.
            tracing::warn!(
                "metal: bypassing wired-memory poison gate via {METAL_PROCESS_LEASE_SKIP_WIRED_GATE_ENV}=1",
            );
        }
        return Ok(());
    }
    check_host_wired_memory()
}

fn check_host_wired_memory() -> Result<(), MetalError> {
    let physical_bytes = host_physical_memory_bytes().ok_or_else(|| {
        MetalError::HostMemoryTelemetry("physical memory size is unavailable".to_string())
    })?;
    for poll in 0..=WIRED_MEMORY_STABILIZATION_POLLS {
        let wired_bytes = crate::cache_probe::wired_memory_bytes()
            .map_err(|error| MetalError::HostMemoryTelemetry(error.to_string()))?;
        if !host_wired_memory_is_unsafe(wired_bytes, physical_bytes) {
            return Ok(());
        }
        if poll == WIRED_MEMORY_STABILIZATION_POLLS {
            return Err(MetalError::UnsafeHostWiredMemory {
                wired_bytes,
                physical_bytes,
            });
        }
        std::thread::sleep(WIRED_MEMORY_STABILIZATION_INTERVAL);
    }
    unreachable!("wired-memory stabilization loop returns on every terminal poll")
}

/// Unit-test contexts have isolated locks. Model/performance tests must also
/// retain this production lease until their contexts and GPU resources drop.
/// Like production, it waits for a busy lease only when
/// `QWEN_METAL_LEASE_WAIT=1`; otherwise it refuses at once.
#[cfg(test)]
pub(crate) fn acquire_metal_benchmark_lease() -> Result<MetalProcessLease, MetalError> {
    let uid = unsafe { libc::geteuid() };
    let path = secure_metal_process_lease_path(
        PathBuf::from("/tmp").join(format!("qwen-llm-{uid}")),
        uid,
    )?;
    let wait = crate::env_flag::read_default_off(METAL_PROCESS_LEASE_WAIT_ENV);
    let lease = open_metal_process_lease(&path, wait)?;
    check_host_wired_memory()?;
    Ok(lease)
}

pub(crate) fn acquire_metal_process_lease() -> Result<Arc<MetalProcessLease>, MetalError> {
    if let Some(lease) = METAL_PROCESS_LEASE.get() {
        if lease.owner_pid != std::process::id() {
            return Err(MetalError::ProcessLease(
                "Metal lease cannot be inherited across fork".to_string(),
            ));
        }
        return Ok(lease.clone());
    }
    let acquisition = METAL_PROCESS_LEASE_ACQUIRE.get_or_init(|| Mutex::new(()));
    let _guard = acquisition.lock();
    if let Some(lease) = METAL_PROCESS_LEASE.get() {
        if lease.owner_pid != std::process::id() {
            return Err(MetalError::ProcessLease(
                "Metal lease cannot be inherited across fork".to_string(),
            ));
        }
        return Ok(lease.clone());
    }
    let wait = crate::env_flag::read_default_off(METAL_PROCESS_LEASE_WAIT_ENV);
    let lease = Arc::new(open_metal_process_lease(
        &metal_process_lease_path()?,
        wait,
    )?);
    ensure_host_wired_memory_is_safe()?;
    let _ = METAL_PROCESS_LEASE.set(lease.clone());
    Ok(lease)
}

pub struct MetalContext {
    pub device: Device,
    device_facts: Arc<super::DeviceFacts>,
    pub queue: Queue,
    pub library: Library,
    tensor_library: Arc<Option<Library>>,
    tensor_ops_status: super::TensorOpsStatus,
    /// Research/bench probe kernels, loaded on first miss (see `pipeline`).
    research_library: Arc<Mutex<Option<Library>>>,
    pub(crate) pso_cache: Arc<Mutex<MetalPipelineCache>>,
    pub(crate) _process_lease: Arc<MetalProcessLease>,
}

/// Device facts, pricing, and allocation signals without a command queue,
/// metallib, or process lease.
pub struct MetalDeviceProbe {
    device: Device,
    facts: super::DeviceFacts,
}

impl MetalDeviceProbe {
    pub fn new() -> Result<Self, MetalError> {
        let device = MTLCreateSystemDefaultDevice().ok_or(MetalError::NoDevice)?;
        let facts = super::DeviceFacts::from_device(&device);
        Ok(Self { device, facts })
    }

    pub fn device_facts(&self) -> &super::DeviceFacts {
        &self.facts
    }

    pub fn max_buffer_length(&self) -> u64 {
        self.device.maxBufferLength() as u64
    }

    pub fn memory_signals(&self) -> MetalMemorySignals {
        let _transaction = METAL_ALLOCATION_TRANSACTION.lock();
        MetalMemorySignals {
            recommended_max_bytes: self.facts.recommended_max_working_set_bytes,
            current_allocated_bytes: self.device.currentAllocatedSize() as u64,
            process_limit_remaining_bytes: MetalContext::process_limit_bytes_remaining(),
            host_budget: capture_host_memory_budget(),
        }
    }

    pub fn shared_buffer_size_and_align(
        &self,
        logical_bytes: u64,
    ) -> Result<MetalBufferSizeAndAlign, MetalError> {
        let length = usize::try_from(logical_bytes.max(1)).map_err(|_| MetalError::BadShape {
            kernel: "device_shared_buffer_size_and_align",
            detail: format!("logical byte request {logical_bytes} does not fit usize"),
        })?;
        let priced = self.device.heapBufferSizeAndAlignWithLength_options(
            length,
            MTLResourceOptions::StorageModeShared,
        );
        Ok(MetalBufferSizeAndAlign {
            size: u64::try_from(priced.size).map_err(|_| MetalError::BadShape {
                kernel: "device_shared_buffer_size_and_align",
                detail: "priced size does not fit u64".into(),
            })?,
            alignment: u64::try_from(priced.align).map_err(|_| MetalError::BadShape {
                kernel: "device_shared_buffer_size_and_align",
                detail: "priced alignment does not fit u64".into(),
            })?,
        })
    }
}

/// Host-side projection of the ordinary model load plan from already-bound
/// GGUF metadata and device facts. Unlike a model load, this prices requests
/// without creating a context or acquiring its process lease.
pub fn probe_metal_device() -> Result<MetalDeviceProbe, MetalError> {
    MetalDeviceProbe::new()
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MetalBufferSizeAndAlign {
    pub size: u64,
    pub alignment: u64,
}

/// Allocation upper bound for one planned shared buffer (see
/// [`MetalContext::price_shared_buffer_upper`]).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PricedSharedBuffer {
    /// Device heap size rounded up to `alignment`.
    pub priced_upper_bytes: u64,
    /// `max(device alignment, host page)`; always a power of two.
    pub alignment: u64,
}

/// Why a planned shared buffer could not be priced. Rendered without a
/// buffer name so each planner can prefix its own.
#[derive(Debug, thiserror::Error)]
pub enum SharedBufferPricingError {
    #[error("has zero bytes")]
    ZeroBytes,
    #[error("Metal maximum buffer length exceeds u64")]
    DeviceMaximumUnrepresentable,
    #[error("requires {logical_bytes} bytes, beyond device maximum {max_buffer_length}")]
    ExceedsDeviceMaximum {
        logical_bytes: u64,
        max_buffer_length: u64,
    },
    #[error(
        "invalid Metal pricing: logical={logical_bytes} priced={priced_size} alignment={alignment} host_page={host_page}"
    )]
    InvalidPricing {
        logical_bytes: u64,
        priced_size: u64,
        alignment: u64,
        host_page: u64,
    },
    #[error("aligned Metal pricing overflows u64")]
    Overflow,
    #[error(transparent)]
    Metal(MetalError),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MetalMemorySignals {
    pub recommended_max_bytes: u64,
    pub current_allocated_bytes: u64,
    pub process_limit_remaining_bytes: Option<u64>,
    pub host_budget: Option<HostMemoryBudget>,
}

impl MetalMemorySignals {
    pub fn effective_device_usage(self) -> Option<u64> {
        self.host_budget?
            .effective_device_usage(self.current_allocated_bytes)
    }

    pub fn host_remaining_bytes(self) -> Option<u64> {
        self.host_budget?
            .remaining_bytes(self.current_allocated_bytes)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MetalMemoryAdmissionReason {
    AdmittedWithProcessBudget,
    AdmittedWithHostBudget,
    RequiredBytesOverflow,
    InvalidWorkingSetSignal,
    ProcessSignalUnavailable,
    HostSignalUnavailable,
    InvalidHostSignal,
    WorkingSetInsufficient,
    ProcessInsufficient,
    BothInsufficient,
    HostInsufficient,
    HostAvailabilityInsufficient,
}

impl MetalMemoryAdmissionReason {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::AdmittedWithProcessBudget => "admitted_with_process_budget",
            Self::AdmittedWithHostBudget => "admitted_with_host_budget",
            Self::RequiredBytesOverflow => "required_bytes_overflow",
            Self::InvalidWorkingSetSignal => "invalid_working_set_signal",
            Self::ProcessSignalUnavailable => "process_signal_unavailable",
            Self::HostSignalUnavailable => "host_signal_unavailable",
            Self::InvalidHostSignal => "invalid_host_signal",
            Self::WorkingSetInsufficient => "working_set_insufficient",
            Self::ProcessInsufficient => "process_insufficient",
            Self::BothInsufficient => "both_insufficient",
            Self::HostInsufficient => "host_insufficient",
            Self::HostAvailabilityInsufficient => "host_availability_insufficient",
        }
    }

    pub fn is_admitted(self) -> bool {
        matches!(
            self,
            Self::AdmittedWithProcessBudget | Self::AdmittedWithHostBudget
        )
    }

    /// Memory pressure: the signals were valid and the request does not fit
    /// (it may fit once memory frees). Missing or invalid telemetry and a
    /// size that overflows are refusals of another kind.
    pub fn is_pressure(self) -> bool {
        matches!(
            self,
            Self::WorkingSetInsufficient
                | Self::ProcessInsufficient
                | Self::BothInsufficient
                | Self::HostInsufficient
                | Self::HostAvailabilityInsufficient
        )
    }
}

/// A refused [`MetalMemoryAdmission`], kept typed through error chains so
/// callers can tell memory pressure ([`MetalMemoryAdmissionReason::is_pressure`])
/// from unavailable or invalid memory telemetry and from a size that
/// overflows. Built only from a decision that did not admit
/// ([`MetalMemoryAdmission::refusal`]).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MemoryAdmissionDenied {
    reason: MetalMemoryAdmissionReason,
    /// `None` when the required size overflows.
    pub required_bytes: Option<u64>,
    pub signals: MetalMemorySignals,
    pub working_set_headroom_bytes: Option<u64>,
}

impl std::fmt::Display for MemoryAdmissionDenied {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let bytes = |value: Option<u64>| value.map_or("unknown".to_string(), |b| b.to_string());
        write!(
            f,
            "memory admission denied ({}): required {} bytes, working-set headroom {} bytes, process remaining {} bytes, host remaining {} bytes, host budget {:?}",
            self.reason.as_str(),
            bytes(self.required_bytes),
            bytes(self.working_set_headroom_bytes),
            bytes(self.signals.process_limit_remaining_bytes),
            bytes(self.signals.host_remaining_bytes()),
            self.signals.host_budget,
        )
    }
}

impl std::error::Error for MemoryAdmissionDenied {}

impl MemoryAdmissionDenied {
    /// Construct a refusal for a refusing reason; admitted reasons return
    /// `None`.
    pub fn new(
        reason: MetalMemoryAdmissionReason,
        required_bytes: Option<u64>,
        signals: MetalMemorySignals,
        working_set_headroom_bytes: Option<u64>,
    ) -> Option<Self> {
        (!reason.is_admitted()).then_some(Self {
            reason,
            required_bytes,
            signals,
            working_set_headroom_bytes,
        })
    }

    /// The refusal reason. This type can never expose an admitted reason.
    pub fn reason(&self) -> MetalMemoryAdmissionReason {
        self.reason
    }
}

/// ```compile_fail
/// use qwen_llm::metal::{MetalMemoryAdmission, MetalMemoryAdmissionReason};
/// # let signals = qwen_llm::metal::MetalMemorySignals {
/// #     recommended_max_bytes: 1,
/// #     current_allocated_bytes: 0,
/// #     process_limit_remaining_bytes: Some(1),
/// #     host_budget: Some(qwen_llm::metal::HostMemoryBudget {
/// #         physical_bytes: 1024,
/// #         conservative_available_bytes: 1024,
/// #         external_gap_bytes: 0,
/// #         private_shadow_bytes: 0,
/// #     }),
/// # };
/// let _ = MetalMemoryAdmission {
///     admitted: false,
///     reason: MetalMemoryAdmissionReason::AdmittedWithProcessBudget,
///     scratch_upper_bytes: 1,
///     reserve_bytes: 0,
///     required_bytes: Some(1),
///     signals,
///     working_set_headroom_bytes: Some(1),
/// };
/// ```
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub struct MetalMemoryAdmission {
    pub reason: MetalMemoryAdmissionReason,
    pub scratch_upper_bytes: u64,
    pub reserve_bytes: u64,
    pub required_bytes: Option<u64>,
    pub signals: MetalMemorySignals,
    pub working_set_headroom_bytes: Option<u64>,
}

impl MetalMemoryAdmission {
    /// Build a reason-authoritative admission decision with its diagnostics.
    pub fn new(
        reason: MetalMemoryAdmissionReason,
        scratch_upper_bytes: u64,
        reserve_bytes: u64,
        required_bytes: Option<u64>,
        signals: MetalMemorySignals,
        working_set_headroom_bytes: Option<u64>,
    ) -> Self {
        Self {
            reason,
            scratch_upper_bytes,
            reserve_bytes,
            required_bytes,
            signals,
            working_set_headroom_bytes,
        }
    }

    pub fn is_admitted(&self) -> bool {
        self.reason.is_admitted()
    }

    /// The typed refusal of a decision that did not admit; `None` exactly
    /// when its reason admits.
    pub fn refusal(&self) -> Option<MemoryAdmissionDenied> {
        (!self.is_admitted()).then_some(MemoryAdmissionDenied {
            reason: self.reason,
            required_bytes: self.required_bytes,
            signals: self.signals,
            working_set_headroom_bytes: self.working_set_headroom_bytes,
        })
    }
}

pub fn evaluate_metal_memory_admission_with_cpu_bytes(
    metal_upper_bytes: u64,
    cpu_upper_bytes: u64,
    reserve_bytes: u64,
    signals: MetalMemorySignals,
    allow_zero_process_budget: bool,
) -> MetalMemoryAdmission {
    evaluate_memory_admission(
        metal_upper_bytes,
        cpu_upper_bytes,
        reserve_bytes,
        signals,
        allow_zero_process_budget,
        false,
    )
}

fn evaluate_memory_admission(
    metal_upper_bytes: u64,
    cpu_upper_bytes: u64,
    reserve_bytes: u64,
    signals: MetalMemorySignals,
    allow_zero_process_budget: bool,
    allocation_complete: bool,
) -> MetalMemoryAdmission {
    let required_bytes = metal_upper_bytes
        .checked_add(cpu_upper_bytes)
        .and_then(|bytes| bytes.checked_add(reserve_bytes));
    let metal_required_bytes = metal_upper_bytes.checked_add(reserve_bytes);
    let working_set_headroom_bytes = signals.recommended_max_bytes.checked_sub(
        signals
            .effective_device_usage()
            .unwrap_or(signals.current_allocated_bytes),
    );
    let reason = match (
        required_bytes,
        metal_required_bytes,
        working_set_headroom_bytes,
    ) {
        (None, _, _) | (_, None, _) => MetalMemoryAdmissionReason::RequiredBytesOverflow,
        _ if signals.host_budget.is_none() => MetalMemoryAdmissionReason::HostSignalUnavailable,
        _ if signals.host_remaining_bytes().is_none() => {
            MetalMemoryAdmissionReason::InvalidHostSignal
        }
        _ if signals.host_budget.is_some_and(|host| {
            host.within_ceiling(signals.current_allocated_bytes)
                .is_none()
                || host.within_availability().is_none()
        }) =>
        {
            MetalMemoryAdmissionReason::InvalidHostSignal
        }
        (Some(_), Some(_), _) if signals.recommended_max_bytes == 0 => {
            MetalMemoryAdmissionReason::InvalidWorkingSetSignal
        }
        (Some(_), Some(_), None) => MetalMemoryAdmissionReason::InvalidWorkingSetSignal,
        (Some(required), Some(metal_required), Some(working_set_headroom)) => {
            let working_set_fits = (allocation_complete || working_set_headroom > 0)
                && metal_required <= working_set_headroom;
            let host_remaining = signals
                .host_remaining_bytes()
                .expect("validated host signal");
            let host = signals.host_budget.expect("validated host signal");
            let host_ceiling_fits = host.within_ceiling(signals.current_allocated_bytes)
                == Some(true)
                && required
                    <= host
                        .ceiling_remaining_bytes(signals.current_allocated_bytes)
                        .expect("validated host signal");
            let host_availability_fits = host.within_availability() == Some(true)
                && (allocation_complete || host_remaining > 0)
                && required <= host_remaining;
            match signals.process_limit_remaining_bytes {
                None => MetalMemoryAdmissionReason::ProcessSignalUnavailable,
                Some(0) if !allow_zero_process_budget => {
                    MetalMemoryAdmissionReason::ProcessSignalUnavailable
                }
                Some(_) if !host_ceiling_fits => MetalMemoryAdmissionReason::HostInsufficient,
                Some(_) if !host_availability_fits && working_set_fits => {
                    MetalMemoryAdmissionReason::HostAvailabilityInsufficient
                }
                Some(_) if !host_availability_fits => {
                    MetalMemoryAdmissionReason::WorkingSetInsufficient
                }
                Some(0) => {
                    if working_set_fits {
                        MetalMemoryAdmissionReason::AdmittedWithHostBudget
                    } else {
                        MetalMemoryAdmissionReason::WorkingSetInsufficient
                    }
                }
                Some(process_available) => {
                    let process_fits = required <= process_available;
                    match (working_set_fits, process_fits) {
                        (true, true) => MetalMemoryAdmissionReason::AdmittedWithProcessBudget,
                        (false, true) => MetalMemoryAdmissionReason::WorkingSetInsufficient,
                        (true, false) => MetalMemoryAdmissionReason::ProcessInsufficient,
                        (false, false) => MetalMemoryAdmissionReason::BothInsufficient,
                    }
                }
            }
        }
    };
    MetalMemoryAdmission::new(
        reason,
        metal_upper_bytes.saturating_add(cpu_upper_bytes),
        reserve_bytes,
        required_bytes,
        signals,
        working_set_headroom_bytes,
    )
}

pub fn evaluate_metal_memory_admission(
    scratch_upper_bytes: u64,
    reserve_bytes: u64,
    signals: MetalMemorySignals,
    allow_zero_process_budget: bool,
) -> MetalMemoryAdmission {
    evaluate_metal_memory_admission_with_cpu_bytes(
        scratch_upper_bytes,
        0,
        reserve_bytes,
        signals,
        allow_zero_process_budget,
    )
}

fn completed_external_import_admission(signals: MetalMemorySignals) -> MetalMemoryAdmission {
    // No new demand remains. Exact exhaustion is valid, but a saturated
    // remaining-byte counter alone cannot distinguish it from overshoot.
    evaluate_memory_admission(0, 0, 0, signals, true, true)
}

unsafe impl Send for MetalContext {}
unsafe impl Sync for MetalContext {}

impl MetalContext {
    pub(crate) fn begin_allocation_transaction(&self) -> MetalAllocationTransactionGuard {
        MetalAllocationTransactionGuard {
            _guard: METAL_ALLOCATION_TRANSACTION.lock(),
        }
    }

    pub fn recommended_max_working_set_size(&self) -> u64 {
        self.device.recommendedMaxWorkingSetSize()
    }

    pub fn process_limit_bytes_remaining() -> Option<u64> {
        let mut info = std::mem::MaybeUninit::<TaskVmInfo>::zeroed();
        let mut count =
            u32::try_from(std::mem::size_of::<TaskVmInfo>() / std::mem::size_of::<u32>()).ok()?;
        let capacity = count;
        // SAFETY: `info` is writable and large enough for `count` natural_t
        // words. The Mach call initializes the returned revision on success.
        let result = unsafe {
            task_info(
                mach_task_self_,
                TASK_VM_INFO,
                info.as_mut_ptr().cast::<i32>(),
                &mut count,
            )
        };
        let limit_count = u32::try_from(
            (std::mem::offset_of!(TaskVmInfo, limit_bytes_remaining) + std::mem::size_of::<u64>())
                / std::mem::size_of::<u32>(),
        )
        .ok()?;
        if result != 0 || count < limit_count || count > capacity {
            return None;
        }
        // SAFETY: A successful rev4 response initialized the packed field.
        Some(unsafe { std::ptr::addr_of!((*info.as_ptr()).limit_bytes_remaining).read_unaligned() })
    }

    pub fn memory_signals(&self) -> MetalMemorySignals {
        let _transaction = METAL_ALLOCATION_TRANSACTION.lock();
        MetalMemorySignals {
            recommended_max_bytes: self.recommended_max_working_set_size(),
            current_allocated_bytes: self.current_allocated_size(),
            process_limit_remaining_bytes: Self::process_limit_bytes_remaining(),
            host_budget: capture_host_memory_budget(),
        }
    }

    pub fn shared_buffer_size_and_align(
        &self,
        logical_bytes: u64,
    ) -> Result<MetalBufferSizeAndAlign, MetalError> {
        let length = usize::try_from(logical_bytes.max(1)).map_err(|_| MetalError::BadShape {
            kernel: "shared_buffer_size_and_align",
            detail: format!("logical byte request {logical_bytes} does not fit usize"),
        })?;
        let priced = self.device.heapBufferSizeAndAlignWithLength_options(
            length,
            MTLResourceOptions::StorageModeShared,
        );
        Ok(MetalBufferSizeAndAlign {
            size: u64::try_from(priced.size).map_err(|_| MetalError::BadShape {
                kernel: "shared_buffer_size_and_align",
                detail: "priced buffer size does not fit u64".into(),
            })?,
            alignment: u64::try_from(priced.align).map_err(|_| MetalError::BadShape {
                kernel: "shared_buffer_size_and_align",
                detail: "priced buffer alignment does not fit u64".into(),
            })?,
        })
    }

    /// Price one planned shared buffer at its allocation upper bound (see
    /// [`price_shared_buffer_upper`]), reading the device maximum, the heap
    /// pricing, and the host page from this context.
    pub fn price_shared_buffer_upper(
        &self,
        logical_bytes: u64,
    ) -> Result<PricedSharedBuffer, SharedBufferPricingError> {
        if logical_bytes == 0 {
            return Err(SharedBufferPricingError::ZeroBytes);
        }
        let max_buffer_length = u64::try_from(self.max_buffer_length())
            .map_err(|_| SharedBufferPricingError::DeviceMaximumUnrepresentable)?;
        if logical_bytes > max_buffer_length {
            return Err(SharedBufferPricingError::ExceedsDeviceMaximum {
                logical_bytes,
                max_buffer_length,
            });
        }
        let priced = self
            .shared_buffer_size_and_align(logical_bytes)
            .map_err(SharedBufferPricingError::Metal)?;
        let host_page = host_page_size_bytes().map_err(SharedBufferPricingError::Metal)? as u64;
        price_shared_buffer_upper(logical_bytes, priced, host_page, max_buffer_length)
    }

    /// Initialize a Metal context backed by the embedded `kernels.metallib`.
    pub fn new() -> Result<Self, MetalError> {
        let process_lease = acquire_metal_process_lease()?;
        let device = MTLCreateSystemDefaultDevice().ok_or(MetalError::NoDevice)?;
        let device_facts = Arc::new(super::DeviceFacts::from_device(&device));
        if device_facts.os_supports_metallib() == Some(false) {
            return Err(MetalError::UnsupportedOs {
                required: device_facts.product_metallib_deployment_target,
                actual: device_facts.os_version.clone().unwrap_or_default(),
            });
        }
        let queue = device.newCommandQueue().ok_or(MetalError::NoQueue)?;

        let metallib_bytes = crate::KERNELS_METALLIB;
        if metallib_bytes.is_empty() {
            return Err(MetalError::EmptyLibrary);
        }
        let library = load_library(&device, metallib_bytes)?;
        let (tensor_library, tensor_ops_status) =
            match super::tensor_ops::load_tensor_library(&device, &device_facts) {
                Ok(library) => (Some(library), super::TensorOpsStatus::Available),
                Err(status) => (None, status),
            };

        Ok(Self {
            device,
            device_facts,
            queue,
            library,
            tensor_library: Arc::new(tensor_library),
            tensor_ops_status,
            research_library: Arc::new(Mutex::new(None)),
            pso_cache: Arc::new(Mutex::new(MetalPipelineCache::default())),
            _process_lease: process_lease,
        })
    }

    /// Create another command-queue view over the same device, library,
    /// pipeline cache, and process lease. This is the host-side primitive for
    /// independent sequence concurrency; it does not duplicate model state.
    pub fn with_new_command_queue(&self) -> Result<Self, MetalError> {
        let queue = self.device.newCommandQueue().ok_or(MetalError::NoQueue)?;
        Ok(Self {
            device: self.device.clone(),
            device_facts: self.device_facts.clone(),
            queue,
            library: self.library.clone(),
            tensor_library: self.tensor_library.clone(),
            tensor_ops_status: self.tensor_ops_status.clone(),
            research_library: self.research_library.clone(),
            pso_cache: self.pso_cache.clone(),
            _process_lease: self._process_lease.clone(),
        })
    }

    pub fn device_facts(&self) -> &super::DeviceFacts {
        &self.device_facts
    }

    pub fn tensor_ops_status(&self) -> &super::TensorOpsStatus {
        &self.tensor_ops_status
    }

    /// Pipeline from the Metal 4 tensor library, cached alongside the main
    /// library's pipelines (tensor kernel names carry their own suffix, so
    /// the one cache cannot alias). `NoFunction` when the tier is absent.
    pub(crate) fn tensor_pipeline(&self, name: &str) -> Result<Pipeline, MetalError> {
        let Some(library) = self.tensor_library.as_ref() else {
            return Err(MetalError::NoFunction(name.to_string()));
        };
        if let Some(p) = self.pso_cache.lock().pipelines.get(name) {
            return Ok(p.clone());
        }
        let function_name = NSString::from_str(name);
        let function = library
            .newFunctionWithName(&function_name)
            .ok_or_else(|| MetalError::NoFunction(name.to_string()))?;
        let pso = self
            .device
            .newComputePipelineStateWithFunction_error(&function)
            .map_err(|error| {
                MetalError::Pipeline(name.to_string(), error.localizedDescription().to_string())
            })?;
        let pso = Pipeline {
            raw: pso,
            name: Arc::from(name),
        };
        self.pso_cache
            .lock()
            .pipelines
            .insert(Arc::clone(&pso.name), pso.clone());
        Ok(pso)
    }

    /// Resolve a function from the research metallib, loading it on first use.
    fn research_function(
        &self,
        name: &NSString,
    ) -> Result<Option<Retained<ProtocolObject<dyn MTLFunction>>>, MetalError> {
        let mut slot = self.research_library.lock();
        if slot.is_none() {
            let bytes = crate::KERNELS_RESEARCH_METALLIB;
            if bytes.is_empty() {
                return Ok(None);
            }
            *slot = Some(load_library(&self.device, bytes)?);
        }
        Ok(slot
            .as_ref()
            .and_then(|library| library.newFunctionWithName(name)))
    }

    /// Look up a kernel function by name, compiling its pipeline state
    /// object on first request and caching it thereafter. Product kernels
    /// come from the embedded product metallib; bench/research probes fall
    /// back to the research metallib.
    pub fn pipeline(&self, name: &str) -> Result<Pipeline, MetalError> {
        let metrics_enabled = {
            let mut cache = self.pso_cache.lock();
            if let Some(p) = cache.pipelines.get(name) {
                return Ok(p.clone());
            }
            if cache.metrics_enabled {
                cache.metrics.misses = cache.metrics.misses.saturating_add(1);
            }
            cache.metrics_enabled
        };
        let miss_t0 = metrics_enabled.then(std::time::Instant::now);
        let func_name = NSString::from_str(name);
        let function = match self.library.newFunctionWithName(&func_name) {
            Some(function) => function,
            None => match self.research_function(&func_name)? {
                Some(function) => function,
                None => {
                    if let Some(start) = miss_t0 {
                        self.record_pipeline_miss_wall(start.elapsed(), None);
                    }
                    return Err(MetalError::NoFunction(name.to_string()));
                }
            },
        };
        let compiler_t0 = metrics_enabled.then(std::time::Instant::now);
        let pso_result = self
            .device
            .newComputePipelineStateWithFunction_error(&function);
        if let Some(start) = miss_t0 {
            self.record_pipeline_miss_wall(
                start.elapsed(),
                compiler_t0.map(|compiler_start| compiler_start.elapsed()),
            );
        }
        let pso = pso_result.map_err(|e: Retained<NSError>| {
            MetalError::Pipeline(name.to_string(), e.localizedDescription().to_string())
        })?;
        let pso = Pipeline {
            raw: pso,
            name: Arc::from(name),
        };
        self.pso_cache
            .lock()
            .pipelines
            .insert(Arc::clone(&pso.name), pso.clone());
        Ok(pso)
    }

    pub fn set_pipeline_cache_metrics_enabled(&self, enabled: bool) {
        let mut cache = self.pso_cache.lock();
        cache.metrics_enabled = enabled;
        if enabled {
            cache.metrics = MetalPipelineCacheMetrics::default();
        }
    }

    pub fn pipeline_cache_metrics(&self) -> MetalPipelineCacheMetrics {
        self.pso_cache.lock().metrics
    }

    fn record_pipeline_miss_wall(
        &self,
        miss_wall: std::time::Duration,
        compiler_wall: Option<std::time::Duration>,
    ) {
        let mut cache = self.pso_cache.lock();
        if !cache.metrics_enabled {
            return;
        }
        cache.metrics.miss_wall_ns = cache
            .metrics
            .miss_wall_ns
            .saturating_add(duration_ns_saturating(miss_wall));
        if let Some(compiler_wall) = compiler_wall {
            cache.metrics.compiler_wall_ns = cache
                .metrics
                .compiler_wall_ns
                .saturating_add(duration_ns_saturating(compiler_wall));
        }
    }

    pub fn pipeline_info(&self, name: &str) -> Result<MetalPipelineInfo, MetalError> {
        let pso = self.pipeline(name)?;
        Ok(MetalPipelineInfo {
            name: name.to_string(),
            thread_execution_width: pso.threadExecutionWidth(),
            max_total_threads_per_threadgroup: pso.maxTotalThreadsPerThreadgroup(),
            static_threadgroup_memory_length: pso.staticThreadgroupMemoryLength(),
            supports_indirect_command_buffers: pso.supportIndirectCommandBuffers(),
        })
    }

    /// Bytes currently allocated through this Metal device.
    pub fn current_allocated_size(&self) -> u64 {
        self.device.currentAllocatedSize() as u64
    }

    pub fn max_buffer_length(&self) -> usize {
        self.device.maxBufferLength()
    }

    /// Allocate a buffer populated from a `bytemuck::Pod` slice.
    /// Uses `StorageModeShared` (unified memory).
    pub fn buffer_from<T: bytemuck::Pod>(&self, data: &[T]) -> Result<Buffer, MetalError> {
        let _allocation_transaction = self.begin_allocation_transaction();
        let bytes = bytemuck::cast_slice::<T, u8>(data);
        let n = bytes.len();
        if n == 0 {
            let buffer = self
                .device
                .newBufferWithLength_options(1, MTLResourceOptions::StorageModeShared)
                .ok_or(MetalError::NoBuffer(1))?;
            record_buffer_allocation(1, &buffer);
            return Ok(buffer);
        }
        let ptr = std::ptr::NonNull::new(bytes.as_ptr() as *mut std::ffi::c_void)
            .ok_or(MetalError::NoBuffer(n))?;
        // SAFETY: `bytes` is valid for `n` bytes for the duration of the call;
        // Metal copies the data into the new buffer.
        let buf = unsafe {
            self.device.newBufferWithBytes_length_options(
                ptr,
                n,
                MTLResourceOptions::StorageModeShared,
            )
        }
        .ok_or(MetalError::NoBuffer(n))?;
        record_buffer_allocation(n, &buf);
        Ok(buf)
    }

    /// The caller must pass an immutable MAP_SHARED mapping. Private mappings
    /// must use gguf_no_copy_window_with_ownership and declare their full span.
    pub(crate) fn gguf_no_copy_window(
        &self,
        mmap: Arc<Mmap>,
        shard_idx: usize,
        mmap_offset: usize,
        length: usize,
        required_alignment: usize,
    ) -> Result<MetalGgufBacking, MetalError> {
        self.gguf_no_copy_window_with_observer(
            mmap,
            shard_idx,
            mmap_offset,
            length,
            required_alignment,
            GgufMappingOwnership::Shared,
            |_pointer, _length| {},
        )
    }

    pub(crate) fn gguf_no_copy_window_with_ownership(
        &self,
        mmap: Arc<Mmap>,
        shard_idx: usize,
        mmap_offset: usize,
        length: usize,
        required_alignment: usize,
        ownership: GgufMappingOwnership,
    ) -> Result<MetalGgufBacking, MetalError> {
        self.gguf_no_copy_window_with_observer(
            mmap,
            shard_idx,
            mmap_offset,
            length,
            required_alignment,
            ownership,
            |_pointer, _length| {},
        )
    }

    #[doc(hidden)]
    pub fn diagnostic_gguf_blit_source_window(
        &self,
        gguf: &GgufFile,
        window: &RetainedStorageWindow,
        required_alignment: usize,
    ) -> Result<DiagnosticGgufBlitSourceWindow, MetalError> {
        let mmap = gguf.retained_shard_mmap(window.shard_idx).ok_or_else(|| {
            MetalError::GgufNoCopy(format!(
                "diagnostic blit source shard {} is unavailable",
                window.shard_idx
            ))
        })?;
        let mmap_offset = usize::try_from(window.mmap_offset).map_err(|_| {
            MetalError::GgufNoCopy(format!(
                "diagnostic blit source offset {} does not fit usize",
                window.mmap_offset
            ))
        })?;
        let geometry = GgufBackingGeometry::new_window(
            window.shard_idx,
            mmap.len(),
            mmap_offset,
            window.length,
            host_page_size()?,
            required_alignment,
        )?;
        // SAFETY: geometry construction proved the nonempty window starts within
        // this retained mmap.
        let expected_pointer = unsafe { mmap.as_ptr().add(mmap_offset) } as usize;
        let expected_length = window.length;
        let deallocator_calls = Arc::new(AtomicUsize::new(0));
        let deallocator_mismatches = Arc::new(AtomicUsize::new(0));
        let calls = Arc::clone(&deallocator_calls);
        let mismatches = Arc::clone(&deallocator_mismatches);
        let backing = self.gguf_no_copy_geometry_with_observer(
            mmap,
            geometry,
            GgufMappingOwnership::Shared,
            move |pointer, length| {
                if pointer.as_ptr() as usize != expected_pointer || length != expected_length {
                    mismatches.fetch_add(1, Ordering::Relaxed);
                }
                calls.fetch_add(1, Ordering::Release);
            },
        )?;
        let probe = DiagnosticGgufBlitReleaseProbe {
            weak: Weak::from_retained(&backing.buffer),
            deallocator_calls,
            deallocator_mismatches,
        };
        Ok(DiagnosticGgufBlitSourceWindow { backing, probe })
    }

    #[cfg(test)]
    pub(crate) fn gguf_no_copy_backing_with_observer<F>(
        &self,
        mmap: Arc<Mmap>,
        shard_idx: usize,
        required_alignment: usize,
        observer: F,
    ) -> Result<MetalGgufBacking, MetalError>
    where
        F: Fn(NonNull<c_void>, usize) + Send + Sync + 'static,
    {
        let page_size = host_page_size()?;
        let mapped_len = mmap.len();
        let geometry =
            GgufBackingGeometry::new(shard_idx, mapped_len, page_size, required_alignment)?;
        self.gguf_no_copy_geometry_with_observer(
            mmap,
            geometry,
            GgufMappingOwnership::Shared,
            observer,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn gguf_no_copy_window_with_observer<F>(
        &self,
        mmap: Arc<Mmap>,
        shard_idx: usize,
        mmap_offset: usize,
        length: usize,
        required_alignment: usize,
        ownership: GgufMappingOwnership,
        observer: F,
    ) -> Result<MetalGgufBacking, MetalError>
    where
        F: Fn(NonNull<c_void>, usize) + Send + Sync + 'static,
    {
        let geometry = GgufBackingGeometry::new_window(
            shard_idx,
            mmap.len(),
            mmap_offset,
            length,
            host_page_size()?,
            required_alignment,
        )?;
        self.gguf_no_copy_geometry_with_observer(mmap, geometry, ownership, observer)
    }

    pub(crate) fn gguf_no_copy_geometry_with_observer<F>(
        &self,
        mmap: Arc<Mmap>,
        geometry: GgufBackingGeometry,
        ownership: GgufMappingOwnership,
        observer: F,
    ) -> Result<MetalGgufBacking, MetalError>
    where
        F: Fn(NonNull<c_void>, usize) + Send + Sync + 'static,
    {
        let _allocation_transaction = self.begin_allocation_transaction();
        let ptr = NonNull::new(
            // SAFETY: geometry construction proves mmap_offset is within the
            // mapping and starts a non-empty window.
            unsafe { mmap.as_ptr().add(geometry.mmap_offset()) } as *mut c_void,
        )
        .ok_or_else(|| MetalError::GgufNoCopy("mmap pointer is null".to_string()))?;
        if !(ptr.as_ptr() as usize).is_multiple_of(geometry.page_size()) {
            return Err(MetalError::GgufNoCopy(format!(
                "window pointer {:p} is not aligned to host page size {}",
                ptr.as_ptr(),
                geometry.page_size(),
            )));
        }
        let max_len = self.device.maxBufferLength();
        if geometry.exposed_len() > max_len {
            return Err(MetalError::GgufNoCopy(format!(
                "page-aligned length {} exceeds Metal maxBufferLength {max_len}",
                geometry.exposed_len()
            )));
        }

        let length = u64::try_from(geometry.exposed_len())
            .map_err(|_| MetalError::GgufNoCopy("import length exceeds u64".into()))?;
        let shadow = ownership.shadow_bytes(length)?;
        let before = self.memory_signals();
        admit_external_import(length, shadow, before)?;
        let charge = ExternalMemoryCharge::reserve(length, shadow)?;
        let resource_charge = Arc::clone(&charge);

        fn assert_send_sync<T: Send + Sync>(_: &T) {}
        assert_send_sync(&mmap);
        let keepalive = Arc::clone(&mmap);
        let deallocator: RcBlock<dyn Fn(NonNull<c_void>, usize)> =
            RcBlock::new(move |pointer: NonNull<c_void>, length: usize| {
                resource_charge.release();
                observer(pointer, length);
                let _ = &keepalive;
            });
        // SAFETY: `ptr` starts a read-only, page-aligned window within one mmap
        // VM region. Its length is page aligned and within both the mapping and
        // Metal's per-buffer limit. The copied deallocator block retains the
        // Arc<Mmap> until this MTLBuffer is destroyed. Other overlapping
        // windows retain independent Arc clones and do not unmap this region.
        let buffer = unsafe {
            self.device
                .newBufferWithBytesNoCopy_length_options_deallocator(
                    ptr,
                    geometry.exposed_len(),
                    MTLResourceOptions::StorageModeShared,
                    Some(&deallocator),
                )
        };
        let Some(buffer) = buffer else {
            charge.release();
            return Err(MetalError::NoBuffer(geometry.exposed_len()));
        };
        // The M4 tiny witness matched these counters, not all drivers or sizes.
        // Contradictory observations receive no accounting credit. This ledger
        // corrects API coverage; it does not measure distinct physical pages.
        let allocated = (buffer.allocatedSize() as u64).min(length);
        let observed = self
            .current_allocated_size()
            .saturating_sub(before.current_allocated_bytes);
        if observed >= allocated {
            charge.credit_allocated(allocated);
        }
        if let Some(denied) = completed_external_import_admission(self.memory_signals()).refusal() {
            return Err(denied.into());
        }
        Ok(MetalGgufBacking { buffer, geometry })
    }

    /// Allocate an uninitialized output buffer of `n_bytes`.
    pub fn buffer_uninit(&self, n_bytes: usize) -> Result<Buffer, MetalError> {
        let _allocation_transaction = self.begin_allocation_transaction();
        let n = n_bytes.max(1);
        let buffer = self
            .device
            .newBufferWithLength_options(n, MTLResourceOptions::StorageModeShared)
            .ok_or(MetalError::NoBuffer(n))?;
        record_buffer_allocation(n, &buffer);
        Ok(buffer)
    }

    pub fn describe(&self) -> String {
        let name = self.device.name().to_string();
        let max_tg = self.device.maxThreadgroupMemoryLength();
        let unified = self.device.hasUnifiedMemory();
        format!("{name} | unified_memory={unified} | max_threadgroup_memory={max_tg} bytes")
    }

    pub fn counter_capabilities(&self) -> MetalCounterCapabilities {
        let mut sets = Vec::new();
        if let Some(counter_sets) = self.device.counterSets() {
            for i in 0..counter_sets.len() {
                let set = counter_sets.objectAtIndex(i);
                let counters_obj = set.counters();
                let mut counters = Vec::with_capacity(counters_obj.len());
                for j in 0..counters_obj.len() {
                    counters.push(counters_obj.objectAtIndex(j).name().to_string());
                }
                let desc = MTLCounterSampleBufferDescriptor::new();
                desc.setCounterSet(Some(&set));
                // SAFETY: the descriptor owns the sample-count field and `2` is
                // the minimum useful before/after probe for future dispatch tests.
                unsafe { desc.setSampleCount(2) };
                let sample_buffer_status = match self
                    .device
                    .newCounterSampleBufferWithDescriptor_error(&desc)
                {
                    Ok(_) => "ok".to_string(),
                    Err(e) => e.localizedDescription().to_string(),
                };
                sets.push(MetalCounterSetInfo {
                    name: set.name().to_string(),
                    counters,
                    sample_buffer_status,
                });
            }
        }
        MetalCounterCapabilities {
            supports_stage_boundary: self
                .device
                .supportsCounterSampling(MTLCounterSamplingPoint::AtStageBoundary),
            supports_dispatch_boundary: self
                .device
                .supportsCounterSampling(MTLCounterSamplingPoint::AtDispatchBoundary),
            supports_blit_boundary: self
                .device
                .supportsCounterSampling(MTLCounterSamplingPoint::AtBlitBoundary),
            sets,
        }
    }

    fn timestamp_counter_set(
        &self,
    ) -> Result<Retained<ProtocolObject<dyn MTLCounterSet>>, MetalError> {
        let counter_sets = self
            .device
            .counterSets()
            .ok_or_else(|| MetalError::Counter("device exposes no counter sets".to_string()))?;
        for i in 0..counter_sets.len() {
            let set = counter_sets.objectAtIndex(i);
            let set_name = set.name().to_string();
            if set_name.eq_ignore_ascii_case("timestamp") {
                return Ok(set);
            }
            let counters = set.counters();
            for j in 0..counters.len() {
                let counter_name = counters.objectAtIndex(j).name().to_string();
                if counter_name.eq_ignore_ascii_case("timestamp") {
                    return Ok(set);
                }
            }
        }
        Err(MetalError::Counter(
            "device exposes no timestamp counter set".to_string(),
        ))
    }

    pub fn timestamp_sample_buffer(
        &self,
        sample_count: usize,
    ) -> Result<MetalTimestampSampleBuffer, MetalError> {
        self.timestamp_sample_buffer_at(sample_count, MTLCounterSamplingPoint::AtStageBoundary)
    }

    pub fn timestamp_dispatch_sample_buffer(
        &self,
        sample_count: usize,
    ) -> Result<MetalTimestampSampleBuffer, MetalError> {
        self.timestamp_sample_buffer_at(sample_count, MTLCounterSamplingPoint::AtDispatchBoundary)
    }

    fn timestamp_sample_buffer_at(
        &self,
        sample_count: usize,
        sampling_point: MTLCounterSamplingPoint,
    ) -> Result<MetalTimestampSampleBuffer, MetalError> {
        if sample_count == 0 {
            return Err(MetalError::Counter(
                "timestamp sample count must be non-zero".to_string(),
            ));
        }
        if !self.device.supportsCounterSampling(sampling_point) {
            let label = match sampling_point {
                MTLCounterSamplingPoint::AtStageBoundary => "stage-boundary",
                MTLCounterSamplingPoint::AtDispatchBoundary => "dispatch-boundary",
                MTLCounterSamplingPoint::AtBlitBoundary => "blit-boundary",
                _ => "requested",
            };
            return Err(MetalError::Counter(format!(
                "device does not support {label} counter sampling"
            )));
        }
        let set = self.timestamp_counter_set()?;
        let desc = MTLCounterSampleBufferDescriptor::new();
        desc.setCounterSet(Some(&set));
        desc.setLabel(&NSString::from_str("qwen decode stage timestamps"));
        desc.setStorageMode(MTLStorageMode::Shared);
        unsafe { desc.setSampleCount(sample_count) };
        let raw = self
            .device
            .newCounterSampleBufferWithDescriptor_error(&desc)
            .map_err(|e| MetalError::Counter(e.localizedDescription().to_string()))?;
        Ok(MetalTimestampSampleBuffer { raw, sample_count })
    }

    pub fn resolve_timestamp_samples(
        &self,
        samples: &MetalTimestampSampleBuffer,
        sample_count: usize,
    ) -> Result<Vec<u64>, MetalError> {
        if sample_count > samples.sample_count {
            return Err(MetalError::Counter(format!(
                "resolve requested {sample_count} samples from {}-sample buffer",
                samples.sample_count
            )));
        }
        let data = unsafe {
            samples
                .raw
                .resolveCounterRange(NSRange::new(0, sample_count))
        }
        .ok_or_else(|| MetalError::Counter("resolveCounterRange returned nil".to_string()))?;
        let bytes = unsafe { data.as_bytes_unchecked() };
        let stride = std::mem::size_of::<MTLCounterResultTimestamp>();
        let needed = sample_count
            .checked_mul(stride)
            .ok_or_else(|| MetalError::Counter("timestamp resolve size overflow".to_string()))?;
        if bytes.len() < needed {
            return Err(MetalError::Counter(format!(
                "timestamp resolve returned {} bytes, need {needed}",
                bytes.len()
            )));
        }
        let mut out = Vec::with_capacity(sample_count);
        for chunk in bytes[..needed].chunks_exact(stride) {
            let mut raw = [0u8; 8];
            raw.copy_from_slice(&chunk[..8]);
            out.push(u64::from_ne_bytes(raw));
        }
        Ok(out)
    }
}

pub(crate) fn load_library(device: &Device, bytes: &[u8]) -> Result<Library, MetalError> {
    // Strategy: write the embedded `.metallib` to a temp file and load via
    // `newLibraryWithURL:`. The library bytes are small (≲100 KB even with
    // all kernels) so I/O is negligible vs pipeline-state compilation.
    // Alternative `newLibraryWithData:` wants `dispatch_data_t`, which
    // objc2-metal 0.3 doesn't expose ergonomically yet.
    use std::io::Write;
    let mut tmp = std::env::temp_dir();
    tmp.push(format!(
        "qwen-kernels-{}-{}.metallib",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    ));
    {
        let mut f = std::fs::File::create(&tmp)
            .map_err(|e| MetalError::LoadLibrary(format!("temp file create: {e}")))?;
        f.write_all(bytes)
            .map_err(|e| MetalError::LoadLibrary(format!("temp file write: {e}")))?;
    }
    let path_str = tmp
        .to_str()
        .ok_or_else(|| MetalError::LoadLibrary("temp path not UTF-8".to_string()))?;
    let url = NSURL::fileURLWithPath(&NSString::from_str(path_str));
    let lib = device
        .newLibraryWithURL_error(&url)
        .map_err(|e: Retained<NSError>| {
            MetalError::LoadLibrary(e.localizedDescription().to_string())
        })?;
    let _ = std::fs::remove_file(&tmp);
    Ok(lib)
}

pub(crate) fn host_page_size() -> Result<usize, MetalError> {
    // SAFETY: getpagesize has no preconditions and returns a process constant.
    let page_size = unsafe { getpagesize() };
    usize::try_from(page_size)
        .ok()
        .filter(|size| size.is_power_of_two())
        .ok_or_else(|| MetalError::GgufNoCopy(format!("invalid host page size {page_size}")))
}

pub fn host_page_size_bytes() -> Result<usize, MetalError> {
    host_page_size()
}

/// Allocation upper bound for one planned shared buffer: the device heap size
/// rounded up to the larger of the device alignment and the host page, with
/// every step checked. This is the arithmetic every family's residency and
/// session planner performs before admission; it carries no policy
/// (reserves, budgets, and which buffers to plan stay with the caller).
/// Pure over its inputs so planners can test it without a device.
pub fn price_shared_buffer_upper(
    logical_bytes: u64,
    priced: MetalBufferSizeAndAlign,
    host_page: u64,
    max_buffer_length: u64,
) -> Result<PricedSharedBuffer, SharedBufferPricingError> {
    if logical_bytes == 0 {
        return Err(SharedBufferPricingError::ZeroBytes);
    }
    if logical_bytes > max_buffer_length {
        return Err(SharedBufferPricingError::ExceedsDeviceMaximum {
            logical_bytes,
            max_buffer_length,
        });
    }
    if priced.size < logical_bytes
        || priced.alignment == 0
        || !priced.alignment.is_power_of_two()
        || host_page == 0
        || !host_page.is_power_of_two()
    {
        return Err(SharedBufferPricingError::InvalidPricing {
            logical_bytes,
            priced_size: priced.size,
            alignment: priced.alignment,
            host_page,
        });
    }
    let alignment = priced.alignment.max(host_page);
    let priced_upper_bytes = priced
        .size
        .checked_add(alignment - 1)
        .map(|bytes| bytes / alignment * alignment)
        .ok_or(SharedBufferPricingError::Overflow)?;
    Ok(PricedSharedBuffer {
        priced_upper_bytes,
        alignment,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metal::test_support::*;

    fn priced(size: u64, alignment: u64) -> MetalBufferSizeAndAlign {
        MetalBufferSizeAndAlign { size, alignment }
    }

    #[test]
    fn shared_buffer_pricing_rounds_to_the_larger_of_device_and_host_alignment() {
        let out = price_shared_buffer_upper(4_097, priced(4_352, 256), 4_096, 4_097).unwrap();
        assert_eq!(
            out,
            PricedSharedBuffer {
                priced_upper_bytes: 8_192,
                alignment: 4_096
            }
        );
        let out = price_shared_buffer_upper(10, priced(16, 16_384), 4_096, u64::MAX).unwrap();
        assert_eq!(out.alignment, 16_384);
        assert_eq!(out.priced_upper_bytes, 16_384);
    }

    #[test]
    fn shared_buffer_pricing_fails_closed_at_every_boundary() {
        assert!(matches!(
            price_shared_buffer_upper(0, priced(1, 1), 4_096, u64::MAX),
            Err(SharedBufferPricingError::ZeroBytes)
        ));
        assert!(matches!(
            price_shared_buffer_upper(4_098, priced(4_098, 2), 4_096, 4_097),
            Err(SharedBufferPricingError::ExceedsDeviceMaximum { .. })
        ));
        for (logical, size, alignment, host_page) in [
            (4_097, 4_096, 256, 4_096), // underpriced
            (1, 1, 3, 4_096),           // device alignment not a power of two
            (1, 1, 0, 4_096),           // zero device alignment
            (1, 1, 256, 0),             // zero host page
            (1, 1, 256, 3),             // host page not a power of two
        ] {
            assert!(
                matches!(
                    price_shared_buffer_upper(
                        logical,
                        priced(size, alignment),
                        host_page,
                        u64::MAX
                    ),
                    Err(SharedBufferPricingError::InvalidPricing { .. })
                ),
                "{logical} {size} {alignment} {host_page}"
            );
        }
        assert!(matches!(
            price_shared_buffer_upper(u64::MAX, priced(u64::MAX, 2), 4_096, u64::MAX),
            Err(SharedBufferPricingError::Overflow)
        ));
    }

    #[test]
    fn shared_buffer_pricing_errors_render_without_a_name_so_callers_prefix_theirs() {
        let error = price_shared_buffer_upper(9, priced(9, 2), 4_096, 8).unwrap_err();
        assert_eq!(
            format!("planned Metal buffer \"x\" {error}"),
            "planned Metal buffer \"x\" requires 9 bytes, beyond device maximum 8"
        );
    }

    #[test]
    fn process_lease_rejects_overlap_and_recovers_after_release() {
        let fixture = LeaseFixture::new();
        let first = open_metal_process_lease(&fixture.0, false).expect("acquire first lease");
        let error = open_metal_process_lease(&fixture.0, false)
            .err()
            .expect("overlapping lease must fail");
        let MetalError::ProcessLease(detail) = error else {
            panic!("unexpected overlap error: {error}");
        };
        assert!(detail.contains("another qwen process owns"));
        assert!(detail.contains(&format!("pid={}", std::process::id())));
        drop(first);
        let second = open_metal_process_lease(&fixture.0, false)
            .expect("lease must recover after owner release");
        drop(second);
    }

    #[test]
    fn process_lease_owner_fields_are_single_line_and_bounded() {
        let dirty = format!("bad value\n{}", "x".repeat(1_024));
        let cleaned = lease_owner_field(&dirty);
        assert!(!cleaned.contains('\n'));
        assert!(!cleaned.contains(' '));
        assert_eq!(cleaned.len(), 512);
    }

    #[test]
    fn process_lease_owner_display_cannot_inject_lines() {
        let dirty = "pid=1\nforged=owner\t\u{1b}[31m";
        let cleaned = lease_owner_display(dirty);
        assert_eq!(cleaned, "pid=1_forged=owner___31m");
        assert!(!cleaned.contains('\n'));
        assert!(!cleaned.contains('\t'));
    }

    #[test]
    fn wired_memory_guard_rejects_half_of_physical_memory() {
        assert!(!host_wired_memory_is_unsafe(0, 128));
        assert!(!host_wired_memory_is_unsafe(63, 128));
        assert!(host_wired_memory_is_unsafe(64, 128));
        assert!(!host_wired_memory_is_unsafe(u64::MAX, 0));
    }

    #[test]
    fn cpu_only_admission_bytes_do_not_consume_metal_headroom() {
        let signals = MetalMemorySignals {
            host_budget: Some(crate::metal::HostMemoryBudget {
                physical_bytes: 1024 << 30,
                conservative_available_bytes: 1024 << 30,
                external_gap_bytes: 0,
                private_shadow_bytes: 0,
            }),
            recommended_max_bytes: 1_000,
            current_allocated_bytes: 700,
            process_limit_remaining_bytes: Some(1_000),
        };
        let decision = evaluate_metal_memory_admission_with_cpu_bytes(200, 400, 50, signals, false);
        assert!(decision.is_admitted());
        assert_eq!(decision.required_bytes, Some(650));
        assert_eq!(decision.working_set_headroom_bytes, Some(300));

        let denied = evaluate_metal_memory_admission_with_cpu_bytes(200, 800, 50, signals, false);
        assert!(!denied.is_admitted());
        assert_eq!(
            denied.reason,
            MetalMemoryAdmissionReason::ProcessInsufficient
        );
    }

    #[test]
    fn metal_memory_admission_requires_both_advisory_budgets() {
        let exact =
            evaluate_metal_memory_admission(400, 100, memory_signals(1500, 1000, Some(500)), false);
        assert!(exact.is_admitted());
        assert_eq!(
            exact.reason,
            MetalMemoryAdmissionReason::AdmittedWithProcessBudget
        );
        assert_eq!(exact.required_bytes, Some(500));
        assert_eq!(exact.working_set_headroom_bytes, Some(500));

        for (signals, reason) in [
            (
                memory_signals(1499, 1000, Some(500)),
                MetalMemoryAdmissionReason::WorkingSetInsufficient,
            ),
            (
                memory_signals(1500, 1000, Some(499)),
                MetalMemoryAdmissionReason::ProcessInsufficient,
            ),
            (
                memory_signals(1499, 1000, Some(499)),
                MetalMemoryAdmissionReason::BothInsufficient,
            ),
            (
                memory_signals(1000, 1000, Some(500)),
                MetalMemoryAdmissionReason::WorkingSetInsufficient,
            ),
            (
                memory_signals(999, 1000, Some(500)),
                MetalMemoryAdmissionReason::InvalidWorkingSetSignal,
            ),
            (
                memory_signals(1500, 1000, Some(0)),
                MetalMemoryAdmissionReason::ProcessSignalUnavailable,
            ),
            (
                memory_signals(1500, 1000, None),
                MetalMemoryAdmissionReason::ProcessSignalUnavailable,
            ),
        ] {
            let decision = evaluate_metal_memory_admission(400, 100, signals, false);
            assert!(!decision.is_admitted());
            assert_eq!(decision.reason, reason);
        }
    }

    #[test]
    fn metal_memory_admission_fails_closed_on_required_overflow() {
        let decision = evaluate_metal_memory_admission(
            u64::MAX,
            1,
            memory_signals(u64::MAX, 1, Some(u64::MAX)),
            false,
        );
        assert!(!decision.is_admitted());
        assert_eq!(decision.required_bytes, None);
        assert_eq!(
            decision.reason,
            MetalMemoryAdmissionReason::RequiredBytesOverflow
        );
    }

    #[test]
    fn metal_memory_admission_uses_host_fallback_for_zero_process_budget_only_when_allowed() {
        let signals = memory_signals(1500, 1000, Some(0));
        let admitted = evaluate_metal_memory_admission(400, 100, signals, true);
        assert!(admitted.is_admitted());
        assert_eq!(
            admitted.reason,
            MetalMemoryAdmissionReason::AdmittedWithHostBudget
        );
        let denied = evaluate_metal_memory_admission(400, 100, signals, false);
        assert!(!denied.is_admitted());
        assert_eq!(
            denied.reason,
            MetalMemoryAdmissionReason::ProcessSignalUnavailable
        );
    }

    #[test]
    fn metal_memory_admission_rejects_zero_required_with_zero_headroom() {
        let decision =
            evaluate_metal_memory_admission(0, 0, memory_signals(1000, 1000, Some(1)), false);
        assert!(!decision.is_admitted());
        assert_eq!(
            decision.reason,
            MetalMemoryAdmissionReason::WorkingSetInsufficient
        );
    }

    #[test]
    fn metal_memory_probes_are_available_on_product_host() {
        let Some(ctx) = crate::test_fixtures::metal_context_or_skip() else {
            return;
        };
        let signals = ctx.memory_signals();
        assert!(signals.recommended_max_bytes > 0);
        eprintln!("[metal-memory-signals] {signals:?}");
    }

    #[test]
    #[ignore = "M4 Max host memory-probe characterization; measured values are host-specific"]
    fn metal_memory_probes_match_local_m4_max() {
        let ctx = MetalContext::new().expect("Metal context");
        let signals = ctx.memory_signals();
        assert_eq!(signals.recommended_max_bytes, 103_079_215_104);
        assert_eq!(signals.process_limit_remaining_bytes, Some(0));
    }

    #[test]
    fn metal_context_initializes() {
        let Some(ctx) = crate::test_fixtures::metal_context_or_skip() else {
            return;
        };
        eprintln!("[metal] {}", ctx.describe());
    }
}

#[cfg(test)]
mod admission_refusal_tests {
    use super::*;

    fn signals(process: Option<u64>) -> MetalMemorySignals {
        MetalMemorySignals {
            host_budget: Some(crate::metal::HostMemoryBudget {
                physical_bytes: 1024 << 30,
                conservative_available_bytes: 1024 << 30,
                external_gap_bytes: 0,
                private_shadow_bytes: 0,
            }),
            recommended_max_bytes: 100,
            current_allocated_bytes: 40,
            process_limit_remaining_bytes: process,
        }
    }

    #[test]
    fn completed_import_accepts_exact_bounds_but_not_saturated_overshoot() {
        use MetalMemoryAdmissionReason as R;
        let mut before = MetalMemorySignals {
            recommended_max_bytes: 800,
            current_allocated_bytes: 100,
            process_limit_remaining_bytes: Some(0),
            host_budget: Some(HostMemoryBudget {
                physical_bytes: 1_000,
                conservative_available_bytes: 900,
                external_gap_bytes: 0,
                private_shadow_bytes: 0,
            }),
        };
        assert!(evaluate_metal_memory_admission(700, 0, before, true).is_admitted());
        before.current_allocated_bytes = 800;
        before
            .host_budget
            .as_mut()
            .unwrap()
            .conservative_available_bytes = 200;
        assert_eq!(before.host_remaining_bytes(), Some(0));
        assert!(completed_external_import_admission(before).is_admitted());
        assert!(!evaluate_metal_memory_admission(0, 0, before, true).is_admitted());

        let mut after = before;
        after.current_allocated_bytes = 801;
        assert!(!completed_external_import_admission(after).is_admitted());
        after.recommended_max_bytes = 900;
        assert_eq!(
            completed_external_import_admission(after).reason,
            R::HostInsufficient
        );
        after = before;
        after
            .host_budget
            .as_mut()
            .unwrap()
            .conservative_available_bytes = 199;
        assert_eq!(after.host_remaining_bytes(), Some(0));
        assert_eq!(
            completed_external_import_admission(after).reason,
            R::HostAvailabilityInsufficient
        );
        after = before;
        after.host_budget.as_mut().unwrap().private_shadow_bytes = 1;
        assert_eq!(
            completed_external_import_admission(after).reason,
            R::HostInsufficient
        );
    }

    /// An admitted decision never becomes a refusal; every refused decision
    /// keeps its typed reason, and only valid-signal shortfalls are pressure.
    #[test]
    fn refusals_are_typed_and_only_shortfalls_are_pressure() {
        use MetalMemoryAdmissionReason as R;
        let cases = [
            (
                evaluate_metal_memory_admission(10, 0, signals(Some(50)), true),
                None,
            ),
            (
                evaluate_metal_memory_admission(10, 0, signals(Some(0)), true),
                None,
            ),
            (
                evaluate_metal_memory_admission(70, 0, signals(Some(500)), true),
                Some((R::WorkingSetInsufficient, true)),
            ),
            (
                evaluate_metal_memory_admission(30, 0, signals(Some(20)), true),
                Some((R::ProcessInsufficient, true)),
            ),
            (
                evaluate_metal_memory_admission(70, 0, signals(Some(20)), true),
                Some((R::BothInsufficient, true)),
            ),
            (
                evaluate_metal_memory_admission(10, 0, signals(None), true),
                Some((R::ProcessSignalUnavailable, false)),
            ),
            (
                evaluate_metal_memory_admission(u64::MAX, 1, signals(Some(50)), true),
                Some((R::RequiredBytesOverflow, false)),
            ),
            (
                evaluate_metal_memory_admission(
                    10,
                    0,
                    MetalMemorySignals {
                        host_budget: Some(crate::metal::HostMemoryBudget {
                            physical_bytes: 1024 << 30,
                            conservative_available_bytes: 1024 << 30,
                            external_gap_bytes: 0,
                            private_shadow_bytes: 0,
                        }),
                        recommended_max_bytes: 0,
                        ..signals(Some(50))
                    },
                    true,
                ),
                Some((R::InvalidWorkingSetSignal, false)),
            ),
        ];
        for (admission, expected) in cases {
            let refusal = admission.refusal();
            assert_eq!(refusal.is_none(), admission.is_admitted(), "{admission:?}");
            match (refusal, expected) {
                (None, None) => assert!(admission.reason.is_admitted()),
                (Some(denied), Some((reason, pressure))) => {
                    assert_eq!(denied.reason, reason);
                    assert_eq!(denied.reason.is_pressure(), pressure);
                    assert_eq!(denied.required_bytes, admission.required_bytes);
                    assert!(denied.to_string().contains(reason.as_str()));
                }
                other => panic!("{other:?} for {admission:?}"),
            }
        }
        // A denial cannot be constructed from an admitted reason, and refusal
        // presence always matches the single reason-derived outcome.
        for reason in [
            R::AdmittedWithProcessBudget,
            R::AdmittedWithHostBudget,
            R::RequiredBytesOverflow,
            R::InvalidWorkingSetSignal,
            R::ProcessSignalUnavailable,
            R::WorkingSetInsufficient,
            R::ProcessInsufficient,
            R::BothInsufficient,
            R::HostInsufficient,
            R::HostAvailabilityInsufficient,
            R::HostSignalUnavailable,
            R::InvalidHostSignal,
        ] {
            let decision =
                MetalMemoryAdmission::new(reason, 10, 0, Some(10), signals(Some(50)), Some(50));
            assert_eq!(decision.refusal().is_none(), reason.is_admitted());
            assert_eq!(
                MemoryAdmissionDenied::new(reason, Some(10), signals(Some(50)), Some(50)).is_some(),
                !reason.is_admitted(),
            );
        }
    }
}

#[cfg(test)]
mod mapping_import_tests;
