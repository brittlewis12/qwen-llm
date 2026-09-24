//! Weight residency, memory plans, capacity, and allocation requests.

use super::*;

/// Engine-owned evidence ceiling through the model's exact context length.
/// A request may allocate less, but allocation never authorizes execution past
/// this independently promoted boundary.
pub const DEEPSEEK_V4_PROMOTED_FORWARD_CAPACITY: usize = 1_048_576;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DeepSeekV4SessionCapacity {
    pub(super) forward_limit: u32,
    pub(super) csa_physical_rows: usize,
    pub(super) hca_physical_rows: usize,
}

impl DeepSeekV4SessionCapacity {
    pub fn for_forward_limit(
        forward_limit: usize,
        model_context_length: u32,
    ) -> Result<Self, DeepSeekV4MetalError> {
        if forward_limit == 0 {
            return invalid("DeepSeek V4 session capacity requires at least one forward");
        }
        if forward_limit > DEEPSEEK_V4_PROMOTED_FORWARD_CAPACITY {
            return invalid(format!(
                "DeepSeek V4 request requires {forward_limit} forwards, beyond promoted evidence capacity {DEEPSEEK_V4_PROMOTED_FORWARD_CAPACITY}"
            ));
        }
        if forward_limit > model_context_length as usize {
            return invalid(format!(
                "DeepSeek V4 request requires {forward_limit} forwards, beyond model context length {model_context_length}"
            ));
        }
        let forward_limit = u32::try_from(forward_limit).map_err(|_| {
            DeepSeekV4MetalError::Invalid("DeepSeek V4 forward capacity exceeds u32".into())
        })?;
        let csa_physical_rows = rounded_history_capacity(
            forward_limit as usize / 4,
            DEEPSEEK_V4_CSA_HISTORY_CAPACITY_ROWS,
            "CSA",
        )?;
        let hca_physical_rows = rounded_history_capacity(
            forward_limit as usize / 128,
            DEEPSEEK_V4_HCA_HISTORY_CAPACITY_ROWS,
            "HCA",
        )?;
        Ok(Self {
            forward_limit,
            csa_physical_rows,
            hca_physical_rows,
        })
    }

    pub fn forward_limit(self) -> usize {
        self.forward_limit as usize
    }

    pub fn csa_physical_rows(self) -> usize {
        self.csa_physical_rows
    }

    pub fn hca_physical_rows(self) -> usize {
        self.hca_physical_rows
    }

    #[doc(hidden)]
    pub fn qualify_multigroup_selector_experiment(
        self,
    ) -> Result<DeepSeekV4MultigroupSelectorGeometry, DeepSeekV4MetalError> {
        let max_visible_rows = self.forward_limit() / 4;
        if !deepseek_v4_multigroup_selector_eligible(self.csa_physical_rows(), max_visible_rows) {
            return invalid(format!(
                "DeepSeek V4 session cannot reach the qualified multi-group selector band: forwards={} physical_capacity_rows={} max_visible_rows={} requires capacity={}..={} visible>={} and visible>=capacity-capacity/4",
                self.forward_limit(),
                self.csa_physical_rows(),
                max_visible_rows,
                DEEPSEEK_V4_MULTIGROUP_SELECTOR_MIN_VISIBLE_ROWS,
                DEEPSEEK_V4_MULTIGROUP_SELECTOR_MAX_CAPACITY_ROWS,
                DEEPSEEK_V4_MULTIGROUP_SELECTOR_MIN_VISIBLE_ROWS,
            ));
        }
        Ok(DeepSeekV4MultigroupSelectorGeometry {
            forward_limit: self.forward_limit(),
            physical_capacity_rows: self.csa_physical_rows(),
            max_visible_rows,
        })
    }

    pub(super) fn validate_position(self, position: u32) -> Result<(), DeepSeekV4MetalError> {
        if position >= self.forward_limit {
            return invalid(format!(
                "DeepSeek V4 session capacity is {} forwards; next position is {position}",
                self.forward_limit
            ));
        }
        Ok(())
    }

    pub(super) fn validate_next_position(
        self,
        next_position: u32,
    ) -> Result<(), DeepSeekV4MetalError> {
        if next_position > self.forward_limit {
            return invalid(format!(
                "DeepSeek V4 state position {next_position} exceeds session capacity {}",
                self.forward_limit
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DeepSeekV4ResidencyReport {
    pub tensor_count: usize,
    pub source_bytes: u64,
    pub window_count: usize,
    pub window_bytes: u64,
    pub view_count: usize,
    pub unique_view_bytes: u64,
    pub logical_view_bytes: u64,
    pub alias_count: usize,
    pub alias_bytes: u64,
    pub fallback_count: usize,
    pub fallback_bytes: u64,
    pub resident_bytes: u64,
    pub page_size: usize,
    pub max_buffer_length: usize,
    pub required_alignment: usize,
}

impl fmt::Display for DeepSeekV4ResidencyReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "tensors={} source_bytes={} windows={}/{} views={}/{} logical_view_bytes={} aliases={}/{} fallbacks={}/{} resident_bytes={} page={} max_buffer={} alignment={}",
            self.tensor_count,
            self.source_bytes,
            self.window_count,
            self.window_bytes,
            self.view_count,
            self.unique_view_bytes,
            self.logical_view_bytes,
            self.alias_count,
            self.alias_bytes,
            self.fallback_count,
            self.fallback_bytes,
            self.resident_bytes,
            self.page_size,
            self.max_buffer_length,
            self.required_alignment,
        )
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DeepSeekV4SessionAllocation {
    pub name: String,
    pub logical_bytes: u64,
    pub priced_bytes: u64,
    pub alignment: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DeepSeekV4MemoryPlan {
    pub(super) residency_buffer_count: usize,
    pub(super) residency_logical_bytes: u64,
    pub(super) residency_priced_upper_bytes: u64,
    pub(super) session_logical_bytes: u64,
    pub(super) session_priced_upper_bytes: u64,
    pub(super) total_priced_upper_bytes: u64,
    pub(super) session_allocations: Vec<DeepSeekV4SessionAllocation>,
}

impl DeepSeekV4MemoryPlan {
    pub fn residency_buffer_count(&self) -> usize {
        self.residency_buffer_count
    }

    pub fn residency_logical_bytes(&self) -> u64 {
        self.residency_logical_bytes
    }

    pub fn residency_priced_upper_bytes(&self) -> u64 {
        self.residency_priced_upper_bytes
    }

    pub fn session_logical_bytes(&self) -> u64 {
        self.session_logical_bytes
    }

    pub fn session_priced_upper_bytes(&self) -> u64 {
        self.session_priced_upper_bytes
    }

    pub fn total_priced_upper_bytes(&self) -> u64 {
        self.total_priced_upper_bytes
    }

    pub fn session_allocations(&self) -> &[DeepSeekV4SessionAllocation] {
        &self.session_allocations
    }

    pub fn admission(&self, signals: MetalMemorySignals) -> MetalMemoryAdmission {
        self.admission_for_sessions(signals, 1)
            .expect("the validated one-session memory plan must not overflow")
    }

    pub fn priced_upper_bytes_for_sessions(
        &self,
        session_count: usize,
    ) -> Result<u64, DeepSeekV4MetalError> {
        if session_count == 0 {
            return invalid("DeepSeek V4 memory admission requires at least one session");
        }
        let session_count = u64::try_from(session_count).map_err(|_| {
            DeepSeekV4MetalError::Invalid("DeepSeek V4 session count exceeds u64".into())
        })?;
        let sessions = self
            .session_priced_upper_bytes
            .checked_mul(session_count)
            .ok_or_else(|| {
                DeepSeekV4MetalError::Invalid(
                    "DeepSeek V4 multi-session priced byte total overflow".into(),
                )
            })?;
        self.residency_priced_upper_bytes
            .checked_add(sessions)
            .ok_or_else(|| {
                DeepSeekV4MetalError::Invalid(
                    "DeepSeek V4 residency plus multi-session byte total overflow".into(),
                )
            })
    }

    pub fn admission_for_sessions(
        &self,
        signals: MetalMemorySignals,
        session_count: usize,
    ) -> Result<MetalMemoryAdmission, DeepSeekV4MetalError> {
        Ok(evaluate_metal_memory_admission(
            self.priced_upper_bytes_for_sessions(session_count)?,
            DEEPSEEK_V4_DYNAMIC_MEMORY_RESERVE_BYTES,
            signals,
            true,
        ))
    }

    pub fn required_with_reserve_bytes(&self) -> Result<u64, DeepSeekV4MetalError> {
        self.required_with_reserve_bytes_for_sessions(1)
    }

    pub fn required_with_reserve_bytes_for_sessions(
        &self,
        session_count: usize,
    ) -> Result<u64, DeepSeekV4MetalError> {
        self.priced_upper_bytes_for_sessions(session_count)?
            .checked_add(DEEPSEEK_V4_DYNAMIC_MEMORY_RESERVE_BYTES)
            .ok_or_else(|| {
                DeepSeekV4MetalError::Invalid(
                    "DeepSeek V4 priced memory plus dynamic reserve overflow".into(),
                )
            })
    }

    pub(super) fn observed_delta(before_residency_bytes: u64, observed_bytes: u64) -> u64 {
        observed_bytes.saturating_sub(before_residency_bytes)
    }

    pub fn reconcile_residency(
        &self,
        before_residency_bytes: u64,
        after_residency_bytes: u64,
    ) -> Result<u64, DeepSeekV4MetalError> {
        let observed = Self::observed_delta(before_residency_bytes, after_residency_bytes);
        let limit = self.residency_priced_upper_bytes;
        if observed > limit {
            return invalid(format!(
                "observed DeepSeek V4 residency delta {observed} exceeds priced residency {limit}"
            ));
        }
        Ok(observed)
    }

    pub fn reconcile_session(
        &self,
        before_residency_bytes: u64,
        after_residency_bytes: u64,
        after_session_bytes: u64,
    ) -> Result<u64, DeepSeekV4MetalError> {
        let observed_total = Self::observed_delta(before_residency_bytes, after_session_bytes);
        let observed_session = after_session_bytes.saturating_sub(after_residency_bytes);
        if observed_total > self.total_priced_upper_bytes {
            return invalid(format!(
                "observed DeepSeek V4 session total {observed_total} exceeds priced model plus session {}",
                self.total_priced_upper_bytes
            ));
        }
        if observed_session > self.session_priced_upper_bytes {
            return invalid(format!(
                "observed DeepSeek V4 session increment {observed_session} exceeds priced session inventory {}",
                self.session_priced_upper_bytes
            ));
        }
        Ok(observed_total)
    }

    pub fn reconcile_first_forward(
        &self,
        before_residency_bytes: u64,
        after_first_forward_bytes: u64,
    ) -> Result<u64, DeepSeekV4MetalError> {
        self.reconcile_total_phase(
            before_residency_bytes,
            after_first_forward_bytes,
            "first forward",
        )
    }

    pub(super) fn reconcile_total_phase(
        &self,
        before_residency_bytes: u64,
        observed_bytes: u64,
        phase: &str,
    ) -> Result<u64, DeepSeekV4MetalError> {
        let observed = Self::observed_delta(before_residency_bytes, observed_bytes);
        let limit = self.required_with_reserve_bytes()?;
        if observed > limit {
            return invalid(format!(
                "observed DeepSeek V4 {phase} delta {observed} exceeds planned total plus reserve {limit}"
            ));
        }
        Ok(observed)
    }

    pub fn reconcile(
        &self,
        samples: DeepSeekV4MemorySamples,
    ) -> Result<DeepSeekV4MemoryReconciliation, DeepSeekV4MetalError> {
        let observed_residency_delta_bytes = self.reconcile_residency(
            samples.before_residency_bytes,
            samples.after_residency_bytes,
        )?;
        let observed_session_delta_bytes = self.reconcile_session(
            samples.before_residency_bytes,
            samples.after_residency_bytes,
            samples.after_session_bytes,
        )?;
        let observed_first_forward_delta_bytes = self.reconcile_first_forward(
            samples.before_residency_bytes,
            samples.after_first_forward_bytes,
        )?;
        let total_limit = self.required_with_reserve_bytes()?;
        let sampled_peak_bytes = samples
            .after_residency_bytes
            .max(samples.after_session_bytes)
            .max(samples.after_first_forward_bytes);
        let sampled_peak_delta_bytes =
            sampled_peak_bytes.saturating_sub(samples.before_residency_bytes);
        Ok(DeepSeekV4MemoryReconciliation {
            samples,
            observed_residency_delta_bytes,
            observed_session_delta_bytes,
            observed_first_forward_delta_bytes,
            sampled_peak_delta_bytes,
            planned_limit_bytes: total_limit,
        })
    }
}

impl fmt::Display for DeepSeekV4MemoryPlan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "residency_buffers={} residency_logical={} residency_priced={} session_buffers={} session_logical={} session_priced={} total_priced={} reserve={} required={}",
            self.residency_buffer_count,
            self.residency_logical_bytes,
            self.residency_priced_upper_bytes,
            self.session_allocations.len(),
            self.session_logical_bytes,
            self.session_priced_upper_bytes,
            self.total_priced_upper_bytes,
            DEEPSEEK_V4_DYNAMIC_MEMORY_RESERVE_BYTES,
            self.required_with_reserve_bytes().unwrap_or(u64::MAX),
        )
    }
}

pub struct DeepSeekV4MetalLoadPlan {
    pub(super) config: DeepSeekV4Config,
    pub(super) session_capacity: DeepSeekV4SessionCapacity,
    pub(super) retained: RetainedStoragePlan,
    pub(super) descriptors: Vec<DeepSeekV4DescriptorFingerprint>,
    pub(super) report: DeepSeekV4ResidencyReport,
    pub(super) memory: DeepSeekV4MemoryPlan,
    pub(super) device_registry_id: u64,
}

impl DeepSeekV4MetalLoadPlan {
    pub fn config(&self) -> &DeepSeekV4Config {
        &self.config
    }

    pub fn residency_report(&self) -> &DeepSeekV4ResidencyReport {
        &self.report
    }

    pub fn memory_plan(&self) -> &DeepSeekV4MemoryPlan {
        &self.memory
    }

    pub fn session_capacity(&self) -> DeepSeekV4SessionCapacity {
        self.session_capacity
    }

    pub fn admit(
        self,
        signals: MetalMemorySignals,
    ) -> Result<DeepSeekV4AdmittedLoadPlan, DeepSeekV4MetalError> {
        self.admit_for_sessions(signals, 1)
    }

    pub fn admit_for_sessions(
        self,
        signals: MetalMemorySignals,
        session_count: usize,
    ) -> Result<DeepSeekV4AdmittedLoadPlan, DeepSeekV4MetalError> {
        let admission = self.memory.admission_for_sessions(signals, session_count)?;
        if !admission.admitted {
            return invalid(format!(
                "DeepSeek V4 {session_count}-session memory admission denied before Metal residency: reason={} required={:?} working_set_headroom={:?} process_remaining={:?}",
                admission.reason.as_str(),
                admission.required_bytes,
                admission.working_set_headroom_bytes,
                admission.signals.process_limit_remaining_bytes,
            ));
        }
        Ok(DeepSeekV4AdmittedLoadPlan {
            plan: self,
            admission,
            session_count,
        })
    }
}

pub struct DeepSeekV4AdmittedLoadPlan {
    pub(super) plan: DeepSeekV4MetalLoadPlan,
    pub(super) admission: MetalMemoryAdmission,
    pub(super) session_count: usize,
}

impl DeepSeekV4AdmittedLoadPlan {
    pub fn admission(&self) -> MetalMemoryAdmission {
        self.admission
    }

    pub fn memory_plan(&self) -> &DeepSeekV4MemoryPlan {
        &self.plan.memory
    }

    pub fn session_count(&self) -> usize {
        self.session_count
    }

    pub fn residency_report(&self) -> &DeepSeekV4ResidencyReport {
        &self.plan.report
    }
}

/// Exact, read-only Metal realization of every tensor in a strict DeepSeek V4
/// Flash-0731 GGUF binding.
pub struct DeepSeekV4MetalResidency {
    pub(super) config: DeepSeekV4Config,
    pub(super) session_capacity: DeepSeekV4SessionCapacity,
    pub(super) _residency_set: Option<DeepSeekV4ResidencySetGuard>,
    pub(super) tensors: BTreeMap<String, MetalTensor>,
    pub(super) report: DeepSeekV4ResidencyReport,
    pub(super) device_registry_id: u64,
}

unsafe impl Send for DeepSeekV4MetalResidency {}
unsafe impl Sync for DeepSeekV4MetalResidency {}

pub(crate) struct DeepSeekV4ResidencySetGuard {
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    set: Retained<ProtocolObject<dyn MTLResidencySet>>,
}

impl Drop for DeepSeekV4ResidencySetGuard {
    fn drop(&mut self) {
        let started = std::time::Instant::now();
        let allocations = self.set.allocationCount();
        let committed_allocation_bytes = self.set.allocatedSize();
        self.queue.removeResidencySet(&self.set);
        self.set.endResidency();
        let _ = std::io::Write::write_fmt(
            &mut std::io::stderr().lock(),
            format_args!(
                "deepseek_v4: model residency set API teardown returned allocations={allocations} committed_allocation_bytes={committed_allocation_bytes} elapsed_ms={:.3}\n",
                started.elapsed().as_secs_f64() * 1e3,
            ),
        );
    }
}

pub(super) fn buffer_as_allocation(
    buffer: &ProtocolObject<dyn MTLBuffer>,
) -> &ProtocolObject<dyn MTLAllocation> {
    ProtocolObject::from_ref(buffer)
}

pub(super) fn deepseek_v4_residency_set_scope_qualified(
    enabled: bool,
    device_name: &str,
    layer_count: u32,
    expert_count: u32,
    report: &DeepSeekV4ResidencyReport,
) -> bool {
    enabled
        && device_name == "Apple M4 Max"
        && layer_count as usize == DEEPSEEK_V4_LAYER_COUNT
        && report.tensor_count == DEEPSEEK_V4_FLASH_0731_TENSOR_COUNT
        && ((expert_count == 160 && report.source_bytes == DEEPSEEK_V4_REAP_K160_SOURCE_BYTES)
            || (expert_count == 216 && report.source_bytes == DEEPSEEK_V4_REAP_K216_SOURCE_BYTES)
            || (expert_count == 256 && report.source_bytes == DEEPSEEK_V4_FRESH_SOURCE_BYTES))
}

/// Which qualified artifact this is, if any. Several prefill, MoE, indexer
/// and attention fast paths are enabled only for these exact artifacts on an
/// M4 Max because their numerics were validated there (see the 2026-09-24
/// DS4 gate audit); say so once at load instead of declining silently.
fn log_deepseek_v4_artifact_qualification(
    ctx: &MetalContext,
    config: &DeepSeekV4Config,
    report: &DeepSeekV4ResidencyReport,
) {
    let device = ctx.device.name().to_string();
    let profile = match (report.tensor_count, config.expert_count, report.source_bytes) {
        (DEEPSEEK_V4_FLASH_0731_TENSOR_COUNT, 256, DEEPSEEK_V4_FRESH_SOURCE_BYTES) => "fresh",
        (DEEPSEEK_V4_FLASH_0731_TENSOR_COUNT, 160, DEEPSEEK_V4_REAP_K160_SOURCE_BYTES) => {
            "reap_k160"
        }
        (DEEPSEEK_V4_FLASH_0731_TENSOR_COUNT, 216, DEEPSEEK_V4_REAP_K216_SOURCE_BYTES) => {
            "reap_k216"
        }
        _ => "none",
    };
    let qualified = profile != "none" && device == "Apple M4 Max";
    eprintln!(
        "deepseek_v4: artifact profile={profile} tensor_count={} experts={} source_bytes={} device={device:?} identity_gated_fast_paths={}",
        report.tensor_count,
        config.expert_count,
        report.source_bytes,
        if qualified {
            "on"
        } else {
            "off (packed Q8 matrices, grouped experts, all-slot Q3/Q4, multigroup selector, long HCA are validated only for the qualified artifacts on M4 Max)"
        },
    );
}

pub(super) fn create_deepseek_v4_residency_set(
    ctx: &MetalContext,
    tensors: &BTreeMap<String, MetalTensor>,
    config: &DeepSeekV4Config,
    report: &DeepSeekV4ResidencyReport,
) -> Option<DeepSeekV4ResidencySetGuard> {
    if !deepseek_v4_residency_set_scope_qualified(
        deepseek_v4_residency_set_enabled(),
        &ctx.device.name().to_string(),
        config.layer_count,
        config.expert_count,
        report,
    ) {
        return None;
    }
    let mut seen = HashSet::new();
    let mut buffers = Vec::new();
    for tensor in tensors.values() {
        let ptr = Retained::as_ptr(&tensor.buffer) as *const _ as usize;
        if seen.insert(ptr) {
            buffers.push(&*tensor.buffer);
        }
    }
    if buffers.is_empty() {
        eprintln!("deepseek_v4: model residency set skipped because no buffers were realized");
        return None;
    }

    let descriptor = MTLResidencySetDescriptor::new();
    descriptor.setLabel(Some(&NSString::from_str("qwen-dsv4-model-weights")));
    // SAFETY: initialCapacity is advisory and equals the unique allocation count.
    unsafe { descriptor.setInitialCapacity(buffers.len()) };
    let set = match ctx.device.newResidencySetWithDescriptor_error(&descriptor) {
        Ok(set) => set,
        Err(error) => {
            let error: Retained<NSError> = error;
            eprintln!(
                "deepseek_v4: model residency set unavailable; continuing without it: {}",
                error.localizedDescription()
            );
            return None;
        }
    };
    for buffer in buffers {
        set.addAllocation(buffer_as_allocation(buffer));
    }
    set.commit();
    set.requestResidency();
    ctx.queue.addResidencySet(&set);
    eprintln!(
        "deepseek_v4: model residency set active allocations={} committed_allocation_bytes={} device_registry_id={} opt_in=QWEN_DSV4_RESIDENCY_SET=1 warning=forced_termination_may_strand_wired_memory",
        set.allocationCount(),
        set.allocatedSize(),
        ctx.device.registryID(),
    );
    Some(DeepSeekV4ResidencySetGuard {
        queue: ctx.queue.clone(),
        set,
    })
}

impl DeepSeekV4MetalResidency {
    pub(super) fn validate_context(&self, ctx: &MetalContext) -> Result<(), DeepSeekV4MetalError> {
        if self.device_registry_id != ctx.device.registryID() {
            return invalid(format!(
                "DeepSeek V4 residency belongs to Metal device registry {}, context is {}",
                self.device_registry_id,
                ctx.device.registryID()
            ));
        }
        if let Some(guard) = &self._residency_set {
            let required_queue = Retained::as_ptr(&guard.queue).cast::<()>() as usize;
            let context_queue = Retained::as_ptr(&ctx.queue).cast::<()>() as usize;
            if required_queue != context_queue {
                return invalid(
                    "DeepSeek V4 residency set is attached to a different Metal command queue",
                );
            }
        }
        Ok(())
    }

    pub fn plan(
        ctx: &MetalContext,
        gguf: &GgufFile,
    ) -> Result<DeepSeekV4MetalLoadPlan, DeepSeekV4MetalError> {
        Self::plan_for_forward_limit(ctx, gguf, DEEPSEEK_V4_PROMOTED_FORWARD_CAPACITY)
    }

    pub fn plan_for_forward_limit(
        ctx: &MetalContext,
        gguf: &GgufFile,
        forward_limit: usize,
    ) -> Result<DeepSeekV4MetalLoadPlan, DeepSeekV4MetalError> {
        let model = DeepSeekV4Model::from_gguf_flash_0731(gguf)?;
        validate_strict_binding(gguf, &model)?;
        validate_session_config(&model.config)?;
        validate_session_lookup_storage(gguf)?;
        let session_capacity = DeepSeekV4SessionCapacity::for_forward_limit(
            forward_limit,
            model.config.context_length,
        )?;

        let requests = gguf.tensors.iter().collect::<Vec<_>>();
        let retained = plan_retained_storage(
            &gguf.shard_mapped_lengths(),
            &requests,
            host_page_size_bytes()?,
            ctx.max_buffer_length(),
            GGUF_BINDING_ALIGNMENT,
        )?;
        validate_fallback_policy(&retained)?;
        let report = report_for_plan(&retained)?;
        let memory = build_memory_plan(ctx, &retained, &report, &model.config, session_capacity)?;
        let descriptors = gguf
            .tensors
            .iter()
            .map(DeepSeekV4DescriptorFingerprint::from)
            .collect();
        Ok(DeepSeekV4MetalLoadPlan {
            config: model.config,
            session_capacity,
            retained,
            descriptors,
            report,
            memory,
            device_registry_id: ctx.device.registryID(),
        })
    }

    pub fn load_from_plan(
        ctx: &MetalContext,
        gguf: &GgufFile,
        admitted: DeepSeekV4AdmittedLoadPlan,
    ) -> Result<DeepSeekV4RealizedLoad, DeepSeekV4MetalError> {
        let admitted_session_count = admitted.session_count;
        let plan = admitted.plan;
        if plan.device_registry_id != ctx.device.registryID() {
            return invalid(format!(
                "DeepSeek V4 load plan belongs to Metal device registry {}, load context is {}",
                plan.device_registry_id,
                ctx.device.registryID()
            ));
        }
        let model = DeepSeekV4Model::from_gguf_flash_0731(gguf)?;
        validate_strict_binding(gguf, &model)?;
        if model.config != plan.config {
            return invalid("DeepSeek V4 load-plan configuration differs from the GGUF binding");
        }
        validate_descriptor_fingerprints(&gguf.tensors, &plan.descriptors)?;
        validate_fallback_policy(&plan.retained)?;
        validate_retained_plan_against_gguf(ctx, gguf, &plan.retained)?;
        if report_for_plan(&plan.retained)? != plan.report {
            return invalid("DeepSeek V4 retained-plan report changed before realization");
        }
        if build_memory_plan(
            ctx,
            &plan.retained,
            &plan.report,
            &plan.config,
            plan.session_capacity,
        )? != plan.memory
        {
            return invalid("DeepSeek V4 memory plan changed before realization");
        }
        let _allocation_transaction = ctx.begin_allocation_transaction();
        let refreshed_admission = plan
            .memory
            .admission_for_sessions(ctx.memory_signals(), admitted_session_count)?;
        if !refreshed_admission.admitted {
            return invalid(format!(
                "DeepSeek V4 {admitted_session_count}-session memory admission changed before realization: reason={} required={:?} working_set_headroom={:?} process_remaining={:?}",
                refreshed_admission.reason.as_str(),
                refreshed_admission.required_bytes,
                refreshed_admission.working_set_headroom_bytes,
                refreshed_admission.signals.process_limit_remaining_bytes,
            ));
        }
        let windows = realize_windows(ctx, gguf, &plan.retained)?;
        let tensors = realize_tensors(ctx, gguf, &plan.retained, &windows)?;
        validate_realization(gguf, &tensors, &plan.report)?;
        let residency_set =
            {
                log_deepseek_v4_artifact_qualification(ctx, &plan.config, &plan.report);
                create_deepseek_v4_residency_set(ctx, &tensors, &plan.config, &plan.report)
            };
        let after_residency_bytes = ctx.current_allocated_size();
        plan.memory.reconcile_residency(
            refreshed_admission.signals.current_allocated_bytes,
            after_residency_bytes,
        )?;

        Ok(DeepSeekV4RealizedLoad {
            residency: Self {
                config: plan.config,
                session_capacity: plan.session_capacity,
                _residency_set: residency_set,
                tensors,
                report: plan.report,
                device_registry_id: ctx.device.registryID(),
            },
            admission: refreshed_admission,
            after_residency_bytes,
        })
    }

    pub fn config(&self) -> &DeepSeekV4Config {
        &self.config
    }

    pub fn session_capacity(&self) -> DeepSeekV4SessionCapacity {
        self.session_capacity
    }

    pub fn tensor(&self, name: &str) -> Option<&MetalTensor> {
        self.tensors.get(name)
    }

    /// Look up an exact GGUF tensor name without silently accepting aliases.
    pub fn require_tensor(&self, name: &str) -> Result<&MetalTensor, DeepSeekV4MetalError> {
        self.tensor(name).ok_or_else(|| {
            DeepSeekV4MetalError::Invalid(format!(
                "strict DeepSeek V4 residency is missing required tensor {name:?}"
            ))
        })
    }

    pub fn tensors(&self) -> impl ExactSizeIterator<Item = (&str, &MetalTensor)> {
        self.tensors
            .iter()
            .map(|(name, tensor)| (name.as_str(), tensor))
    }

    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    pub fn report(&self) -> &DeepSeekV4ResidencyReport {
        &self.report
    }

    pub fn device_registry_id(&self) -> u64 {
        self.device_registry_id
    }
}

pub(super) fn rounded_history_capacity(
    required_rows: usize,
    floor_rows: usize,
    name: &str,
) -> Result<usize, DeepSeekV4MetalError> {
    if floor_rows == 0 || !floor_rows.is_multiple_of(DEEPSEEK_V4_COMPRESSED_HISTORY_SLAB_ROWS) {
        return invalid(format!(
            "DeepSeek V4 {name} history floor {floor_rows} is not a nonzero slab multiple"
        ));
    }
    let slabs = required_rows
        .checked_add(DEEPSEEK_V4_COMPRESSED_HISTORY_SLAB_ROWS - 1)
        .ok_or_else(|| {
            DeepSeekV4MetalError::Invalid(format!(
                "DeepSeek V4 {name} history row rounding overflow"
            ))
        })?
        / DEEPSEEK_V4_COMPRESSED_HISTORY_SLAB_ROWS;
    let rounded = slabs
        .checked_mul(DEEPSEEK_V4_COMPRESSED_HISTORY_SLAB_ROWS)
        .ok_or_else(|| {
            DeepSeekV4MetalError::Invalid(format!("DeepSeek V4 {name} history capacity overflow"))
        })?;
    Ok(rounded.max(floor_rows))
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum DeepSeekV4PrepareTestPolicy {
    Production,
    Paired,
    Composed,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) struct DeepSeekV4PairedPrepareCapabilities {
    pub(super) q6_projection: bool,
    pub(super) q8_projection: bool,
    pub(super) norm_and_rope: bool,
}

impl DeepSeekV4PairedPrepareCapabilities {
    pub(super) fn probe(ctx: &MetalContext) -> Self {
        let supports = |kernel: &str, threads: usize| {
            ctx.pipeline(kernel)
                .is_ok_and(|pipeline| pipeline.maxTotalThreadsPerThreadgroup() >= threads)
        };
        let norm_threads = ctx
            .pipeline("kernel_rms_norm_mul_f32")
            .ok()
            .map(|pipeline| pipeline.maxTotalThreadsPerThreadgroup().min(1024))
            .filter(|&threads| threads > 0);
        Self {
            q6_projection: supports("kernel_ds4_prepare_projection_pair_q6_q8_f32", 128),
            q8_projection: supports("kernel_ds4_prepare_projection_pair_q8_q8_f32", 128),
            norm_and_rope: norm_threads.is_some_and(|threads| {
                supports("kernel_ds4_prepare_norm_pair_f32", threads)
                    && supports("kernel_deepseek_v4_rope_pair_in_place", 256)
            }),
        }
    }

    pub(super) fn supports(self, q_dtype: GgmlType) -> bool {
        self.norm_and_rope
            && match q_dtype {
                GgmlType::Q6_K => self.q6_projection,
                GgmlType::Q8_0 => self.q8_projection,
                _ => false,
            }
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn encode_ds4_prepare_projection_pair(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    q_weight: &MetalTensor,
    kv_weight: &MetalTensor,
    input: &MetalTensor,
    q_output: &MetalTensor,
    kv_output: &MetalTensor,
    n_in: usize,
    q_out: usize,
    kv_out: usize,
) -> Result<(), DeepSeekV4MetalError> {
    require_serial(enc, "paired prepare projection")?;
    validate_matvec_weight(q_weight, n_in, q_out, "paired Q A weight")?;
    validate_matvec_weight(kv_weight, n_in, kv_out, "paired KV weight")?;
    validate_f32(input, &[n_in as u64], false, "paired projection input")?;
    validate_f32(q_output, &[q_out as u64], true, "paired Q A output")?;
    validate_f32(kv_output, &[kv_out as u64], true, "paired KV output")?;
    if n_in == 0
        || q_out == 0
        || kv_out == 0
        || kv_weight.dtype != GgmlType::Q8_0
        || !matches!(q_weight.dtype, GgmlType::Q6_K | GgmlType::Q8_0)
        || metal_tensor_ranges_overlap(q_output, kv_output)
        || [q_output, kv_output].iter().any(|output| {
            metal_tensor_ranges_overlap(output, q_weight)
                || metal_tensor_ranges_overlap(output, kv_weight)
                || metal_tensor_ranges_overlap(output, input)
        })
        || u32::try_from(n_in).is_err()
        || u32::try_from(q_out).is_err()
        || u32::try_from(kv_out).is_err()
    {
        return invalid(format!(
            "paired prepare projection requires Q6_K/Q8_0 Q A and Q8_0 KV weights over nonzero, distinct F32 outputs; got {:?}/{:?} {n_in} -> {q_out}/{kv_out}",
            q_weight.dtype, kv_weight.dtype,
        ));
    }
    let (kernel, q_rows_per_group) = match q_weight.dtype {
        GgmlType::Q6_K => ("kernel_ds4_prepare_projection_pair_q6_q8_f32", 4usize),
        GgmlType::Q8_0 => ("kernel_ds4_prepare_projection_pair_q8_q8_f32", 2usize),
        _ => unreachable!("paired Q A dtype was qualified"),
    };
    let pso = ctx.pipeline(kernel)?;
    if pso.maxTotalThreadsPerThreadgroup() < 128 {
        return invalid(format!(
            "paired prepare projection pipeline supports {} threads, requires 128",
            pso.maxTotalThreadsPerThreadgroup()
        ));
    }
    enc.note_read(q_weight);
    enc.note_read(kv_weight);
    enc.note_read(input);
    enc.note_write(q_output);
    enc.note_write(kv_output);
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n_in: u32,
        q_out: u32,
        kv_out: u32,
    }
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &Args {
            n_in: n_in as u32,
            q_out: q_out as u32,
            kv_out: kv_out as u32,
        },
    );
    enc.set_tensor(1, q_weight);
    enc.set_tensor(2, kv_weight);
    enc.set_tensor(3, input);
    enc.set_tensor(4, q_output);
    enc.set_tensor(5, kv_output);
    enc.set_threadgroup_memory(0, 32 * 2 * std::mem::size_of::<f32>());
    enc.dispatch(
        MTLSize {
            width: q_out.div_ceil(q_rows_per_group).max(kv_out.div_ceil(2)),
            height: 1,
            depth: 2,
        },
        MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

pub(super) fn validate_retained_plan_against_gguf(
    ctx: &MetalContext,
    gguf: &GgufFile,
    plan: &RetainedStoragePlan,
) -> Result<(), DeepSeekV4MetalError> {
    validate_retained_plan_geometry(
        host_page_size_bytes()?,
        ctx.max_buffer_length(),
        &gguf.shard_mapped_lengths(),
        &gguf.tensors,
        plan,
    )
}

pub(super) fn validate_retained_plan_geometry(
    page_size: usize,
    max_buffer_length: usize,
    shard_lengths: &[usize],
    tensors: &[crate::tensor::TensorDesc],
    plan: &RetainedStoragePlan,
) -> Result<(), DeepSeekV4MetalError> {
    if page_size == 0 {
        return invalid("DeepSeek V4 retained-plan page size is zero");
    }
    let usable_window_length = max_buffer_length / page_size * page_size;
    if usable_window_length == 0
        || plan.page_size != page_size
        || plan.max_buffer_length != max_buffer_length
        || plan.usable_window_length != usable_window_length
        || plan.required_alignment != GGUF_BINDING_ALIGNMENT
        || plan.entries.len() != tensors.len()
    {
        return invalid("DeepSeek V4 retained-plan geometry changed before realization");
    }
    for (index, window) in plan.windows.iter().enumerate() {
        let shard_length = shard_lengths
            .get(window.shard_idx)
            .copied()
            .ok_or_else(|| {
                DeepSeekV4MetalError::Invalid(format!(
                    "retained window {index} references missing shard {}",
                    window.shard_idx
                ))
            })?;
        let mmap_offset = usize::try_from(window.mmap_offset).map_err(|_| {
            DeepSeekV4MetalError::Invalid(format!("retained window {index} offset exceeds usize"))
        })?;
        let end = mmap_offset.checked_add(window.length).ok_or_else(|| {
            DeepSeekV4MetalError::Invalid(format!("retained window {index} endpoint overflow"))
        })?;
        if window.length == 0
            || window.length > plan.usable_window_length
            || !mmap_offset.is_multiple_of(page_size)
            || !window.length.is_multiple_of(page_size)
            || end > shard_length
        {
            return invalid(format!(
                "retained window {index} is outside the planned shard/page/buffer geometry"
            ));
        }
    }
    for (index, (entry, desc)) in plan.entries.iter().zip(tensors).enumerate() {
        if entry.request_index != index
            || entry.name != desc.name
            || entry.shard_idx != desc.shard_idx
            || entry.data_offset != desc.data_offset
            || entry.n_bytes != desc.n_bytes
        {
            return invalid(format!("planner descriptor drift at index {index}"));
        }
        match entry.disposition {
            RetainedStorageDisposition::View {
                window_index,
                buffer_offset,
            } => {
                let window = plan.windows.get(window_index).ok_or_else(|| {
                    DeepSeekV4MetalError::Invalid(format!(
                        "retained entry {index} references missing window {window_index}"
                    ))
                })?;
                let expected_data_offset = window
                    .mmap_offset
                    .checked_add(buffer_offset)
                    .ok_or_else(|| {
                        DeepSeekV4MetalError::Invalid(format!(
                            "retained entry {index} source offset overflow"
                        ))
                    })?;
                let buffer_end = buffer_offset.checked_add(entry.n_bytes).ok_or_else(|| {
                    DeepSeekV4MetalError::Invalid(format!(
                        "retained entry {index} buffer endpoint overflow"
                    ))
                })?;
                if entry.shard_idx != window.shard_idx
                    || entry.data_offset != expected_data_offset
                    || buffer_end > window.length as u64
                    || !buffer_offset.is_multiple_of(plan.required_alignment as u64)
                {
                    return invalid(format!(
                        "retained entry {index} differs from its planned window binding"
                    ));
                }
            }
            RetainedStorageDisposition::Alias {
                source_request_index,
            } => {
                if source_request_index >= index {
                    return invalid(format!(
                        "retained alias {index} has non-prior source {source_request_index}"
                    ));
                }
                let source_entry = &plan.entries[source_request_index];
                let source_desc = &tensors[source_request_index];
                if source_entry.shard_idx != entry.shard_idx
                    || source_entry.data_offset != entry.data_offset
                    || source_entry.n_bytes != entry.n_bytes
                    || source_desc.shape != desc.shape
                    || source_desc.dtype != desc.dtype
                    || matches!(
                        source_entry.disposition,
                        RetainedStorageDisposition::Alias { .. }
                    )
                {
                    return invalid(format!(
                        "retained alias {index} differs from source {source_request_index}"
                    ));
                }
            }
            RetainedStorageDisposition::CopyFallback { reason } => {
                let shard_length =
                    shard_lengths.get(entry.shard_idx).copied().ok_or_else(|| {
                        DeepSeekV4MetalError::Invalid(format!(
                            "retained fallback {index} references missing shard {}",
                            entry.shard_idx
                        ))
                    })?;
                let start = usize::try_from(entry.data_offset).map_err(|_| {
                    DeepSeekV4MetalError::Invalid(format!(
                        "retained fallback {index} offset exceeds usize"
                    ))
                })?;
                let length = usize::try_from(entry.n_bytes).map_err(|_| {
                    DeepSeekV4MetalError::Invalid(format!(
                        "retained fallback {index} length exceeds usize"
                    ))
                })?;
                let end = start.checked_add(length).ok_or_else(|| {
                    DeepSeekV4MetalError::Invalid(format!(
                        "retained fallback {index} endpoint overflow"
                    ))
                })?;
                let rounded_end = end
                    .checked_add(page_size - 1)
                    .map(|value| value / page_size * page_size)
                    .ok_or_else(|| {
                        DeepSeekV4MetalError::Invalid(format!(
                            "retained fallback {index} page endpoint overflow"
                        ))
                    })?;
                let window_start = start / page_size * page_size;
                let candidate_window_length =
                    rounded_end.checked_sub(window_start).ok_or_else(|| {
                        DeepSeekV4MetalError::Invalid(format!(
                            "retained fallback {index} window range underflow"
                        ))
                    })?;
                let full_page_end = shard_length / page_size * page_size;
                if reason != RetainedStorageFallback::FinalPartialPage
                    || length == 0
                    || start % plan.required_alignment != 0
                    || end > shard_length
                    || end <= full_page_end
                    || candidate_window_length > plan.usable_window_length
                {
                    return invalid(format!(
                        "retained entry {index} differs from a final-partial-page fallback"
                    ));
                }
            }
        }
    }
    let requests = tensors.iter().collect::<Vec<_>>();
    let rebuilt = plan_retained_storage(
        shard_lengths,
        &requests,
        page_size,
        max_buffer_length,
        GGUF_BINDING_ALIGNMENT,
    )?;
    if rebuilt != *plan {
        return invalid("DeepSeek V4 retained plan differs from deterministic planner output");
    }
    Ok(())
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct DeepSeekV4SessionAllocationRequest {
    pub(super) name: String,
    pub(super) logical_bytes: u64,
}

pub(super) fn push_session_allocation(
    requests: &mut Vec<DeepSeekV4SessionAllocationRequest>,
    name: impl Into<String>,
    elements: usize,
    element_bytes: usize,
) -> Result<(), DeepSeekV4MetalError> {
    let logical_bytes = checked_mul(elements, element_bytes, "session allocation bytes")?;
    requests.push(DeepSeekV4SessionAllocationRequest {
        name: name.into(),
        logical_bytes: u64::try_from(logical_bytes).map_err(|_| {
            DeepSeekV4MetalError::Invalid("session allocation bytes exceed u64".into())
        })?,
    });
    Ok(())
}

pub(super) fn deepseek_v4_session_allocation_requests(
    config: &DeepSeekV4Config,
    capacity: DeepSeekV4SessionCapacity,
) -> Result<Vec<DeepSeekV4SessionAllocationRequest>, DeepSeekV4MetalError> {
    validate_session_config(config)?;
    deepseek_v4_session_allocation_requests_for_kinds(
        &config.attention_kinds,
        config.expert_count as usize,
        capacity,
    )
}

pub(super) fn deepseek_v4_session_allocation_requests_for_kinds(
    attention_kinds: &[AttentionKind],
    expert_count: usize,
    capacity: DeepSeekV4SessionCapacity,
) -> Result<Vec<DeepSeekV4SessionAllocationRequest>, DeepSeekV4MetalError> {
    let sliding = attention_kinds
        .iter()
        .filter(|&&kind| kind == AttentionKind::SlidingWindow)
        .count();
    let csa = attention_kinds
        .iter()
        .filter(|&&kind| kind == AttentionKind::CompressedSparse)
        .count();
    let hca = attention_kinds
        .iter()
        .filter(|&&kind| kind == AttentionKind::HeavilyCompressed)
        .count();
    if attention_kinds.len() != DEEPSEEK_V4_LAYER_COUNT || (sliding, csa, hca) != (2, 21, 20) {
        return invalid(format!(
            "session allocation inventory requires 43 layers split 2/21/20, got {}/{sliding}/{csa}/{hca}",
            attention_kinds.len()
        ));
    }
    let attention = deepseek_v4_session_attention_config();
    let attention_dims = attention.checked()?;
    let moe = DeepSeekV4MoeConfig {
        hidden_size: DEEPSEEK_V4_HIDDEN_SIZE,
        ffn_size: 2_048,
        expert_count,
        top_k: 6,
        routed_scale: 1.0,
    };
    moe.checked()?;
    let mut requests = Vec::with_capacity(521);
    let f32_bytes = std::mem::size_of::<f32>();
    let i32_bytes = std::mem::size_of::<i32>();
    let f16_bytes = std::mem::size_of::<u16>();
    let residual_elements = residual_len(DEEPSEEK_V4_HIDDEN_SIZE)?;

    push_session_allocation(&mut requests, "token_id", 1, i32_bytes)?;
    push_session_allocation(
        &mut requests,
        "embedding",
        DEEPSEEK_V4_HIDDEN_SIZE,
        f32_bytes,
    )?;
    for name in ["residual_primary", "residual_secondary"] {
        push_session_allocation(&mut requests, name, residual_elements, f32_bytes)?;
    }

    for name in ["hyper.ones", "hyper.normalized"] {
        push_session_allocation(&mut requests, name, residual_elements, f32_bytes)?;
    }
    push_session_allocation(
        &mut requests,
        "hyper.mixes",
        DEEPSEEK_V4_HC_PARAMETER_COUNT,
        f32_bytes,
    )?;
    for name in ["hyper.pre", "hyper.post"] {
        push_session_allocation(&mut requests, name, DEEPSEEK_V4_CONNECTION_COUNT, f32_bytes)?;
    }
    push_session_allocation(
        &mut requests,
        "hyper.combination",
        checked_mul(
            DEEPSEEK_V4_CONNECTION_COUNT,
            DEEPSEEK_V4_CONNECTION_COUNT,
            "hyper combination elements",
        )?,
        f32_bytes,
    )?;
    push_session_allocation(
        &mut requests,
        "hyper.collapsed",
        DEEPSEEK_V4_HIDDEN_SIZE,
        f32_bytes,
    )?;
    for name in ["hyper.head_mixes", "hyper.head_gates"] {
        push_session_allocation(&mut requests, name, DEEPSEEK_V4_CONNECTION_COUNT, f32_bytes)?;
    }

    push_session_allocation(
        &mut requests,
        "attention.normalized_input",
        attention.hidden_size,
        f32_bytes,
    )?;
    for name in ["attention.q_lora_raw", "attention.q_lora"] {
        push_session_allocation(&mut requests, name, attention.q_lora_rank, f32_bytes)?;
    }
    for name in ["attention.queries_raw", "attention.queries"] {
        push_session_allocation(&mut requests, name, attention_dims.query_width, f32_bytes)?;
    }
    for name in ["attention.kv_raw", "attention.kv", "attention.cached_kv"] {
        push_session_allocation(&mut requests, name, attention.head_dim, f32_bytes)?;
    }
    push_session_allocation(
        &mut requests,
        "attention.heads",
        attention_dims.query_width,
        f32_bytes,
    )?;
    push_session_allocation(
        &mut requests,
        "attention.hca_partial_output",
        checked_mul(
            attention_dims.query_width,
            DEEPSEEK_V4_SPLITK_HCA_PARTITIONS,
            "split-K HCA partial output elements",
        )?,
        f32_bytes,
    )?;
    push_session_allocation(
        &mut requests,
        "attention.hca_partial_ml",
        checked_mul(
            checked_mul(
                2,
                attention.head_count,
                "split-K HCA max/mass per partition",
            )?,
            DEEPSEEK_V4_SPLITK_HCA_PARTITIONS,
            "split-K HCA partial max/mass elements",
        )?,
        f32_bytes,
    )?;
    push_session_allocation(
        &mut requests,
        "attention.low_rank",
        attention_dims.low_rank_width,
        f32_bytes,
    )?;
    push_session_allocation(
        &mut requests,
        "attention.output",
        attention.hidden_size,
        f32_bytes,
    )?;
    push_session_allocation(
        &mut requests,
        "attention.head_norm_ones",
        attention.head_dim,
        f32_bytes,
    )?;
    push_session_allocation(
        &mut requests,
        "sparse_csa.index_queries",
        64 * 128,
        f32_bytes,
    )?;
    push_session_allocation(
        &mut requests,
        "sparse_csa.matrix_queries_f16",
        64 * 128,
        f16_bytes,
    )?;
    push_session_allocation(&mut requests, "sparse_csa.head_weights", 64, f32_bytes)?;
    push_session_allocation(&mut requests, "sparse_csa.visible_counts", 1, i32_bytes)?;
    push_session_allocation(
        &mut requests,
        "sparse_csa.scores",
        capacity.csa_physical_rows(),
        f32_bytes,
    )?;
    push_session_allocation(
        &mut requests,
        "sparse_csa.selected_mask",
        capacity.csa_physical_rows(),
        i32_bytes,
    )?;
    push_session_allocation(
        &mut requests,
        "sparse_csa.cache_order_ids",
        DEEPSEEK_V4_CSA_TOP_K,
        i32_bytes,
    )?;
    for name in ["sparse_csa.selected_counts", "sparse_csa.status"] {
        push_session_allocation(&mut requests, name, 1, i32_bytes)?;
    }
    if deepseek_v4_multigroup_selector_capacity_supported(capacity.csa_physical_rows()) {
        push_session_allocation(
            &mut requests,
            "sparse_csa.multigroup.records",
            checked_mul(
                DEEPSEEK_V4_MULTIGROUP_SELECTOR_RECORD_WORDS,
                DEEPSEEK_V4_MULTIGROUP_SELECTOR_GROUPS,
                "multi-group selector record elements",
            )?,
            i32_bytes,
        )?;
        push_session_allocation(
            &mut requests,
            "sparse_csa.multigroup.partition_plan",
            checked_mul(
                DEEPSEEK_V4_MULTIGROUP_SELECTOR_PLAN_WORDS,
                DEEPSEEK_V4_MULTIGROUP_SELECTOR_GROUPS,
                "multi-group selector plan elements",
            )?,
            i32_bytes,
        )?;
        push_session_allocation(
            &mut requests,
            "sparse_csa.multigroup.state",
            DEEPSEEK_V4_MULTIGROUP_SELECTOR_STATE_WORDS,
            i32_bytes,
        )?;
        push_session_allocation(
            &mut requests,
            "sparse_csa.multigroup.private_mask",
            capacity.csa_physical_rows(),
            std::mem::size_of::<u8>(),
        )?;
        push_session_allocation(
            &mut requests,
            "sparse_csa.multigroup.private_ids",
            DEEPSEEK_V4_CSA_TOP_K,
            i32_bytes,
        )?;
    }
    #[cfg(feature = "dsv4-diagnostics")]
    {
        push_session_allocation(
            &mut requests,
            "fp4_shadow.query_values",
            crate::deepseek_v4_oracle::INDEXER_FP4_VALUE_BYTES * 64,
            std::mem::size_of::<u8>(),
        )?;
        push_session_allocation(
            &mut requests,
            "fp4_shadow.query_scales",
            crate::deepseek_v4_oracle::INDEXER_FP4_SCALE_BYTES * 64,
            std::mem::size_of::<u8>(),
        )?;
        push_session_allocation(&mut requests, "fp4_shadow.query_status", 64, i32_bytes)?;
        push_session_allocation(&mut requests, "fp4_shadow.query_units", 128 * 64, f16_bytes)?;
        push_session_allocation(&mut requests, "fp4_shadow.eligible_visible", 1, i32_bytes)?;
        push_session_allocation(&mut requests, "fp4_shadow.eligibility_record", 3, i32_bytes)?;
        push_session_allocation(
            &mut requests,
            "fp4_shadow.scores",
            capacity.csa_physical_rows(),
            f32_bytes,
        )?;
        push_session_allocation(
            &mut requests,
            "fp4_shadow.selected_mask",
            capacity.csa_physical_rows(),
            i32_bytes,
        )?;
        push_session_allocation(
            &mut requests,
            "fp4_shadow.cache_order_ids",
            DEEPSEEK_V4_CSA_TOP_K,
            i32_bytes,
        )?;
        for name in ["fp4_shadow.selected_count", "fp4_shadow.status"] {
            push_session_allocation(&mut requests, name, 1, i32_bytes)?;
        }
        push_session_allocation(
            &mut requests,
            "fp4_collapsed_selections.cache_order_ids",
            checked_mul(
                DEEPSEEK_V4_CSA_TOP_K,
                DEEPSEEK_V4_LAYER_COUNT,
                "collapsed FP4 layer-selection ID elements",
            )?,
            i32_bytes,
        )?;
        push_session_allocation(
            &mut requests,
            "fp4_collapsed_selections.integers",
            checked_mul(
                DEEPSEEK_V4_FP4_COMPLETION_RECORD_I32_WIDTH,
                DEEPSEEK_V4_LAYER_COUNT,
                "collapsed FP4 completion-record elements",
            )?,
            i32_bytes,
        )?;
    }
    push_session_allocation(
        &mut requests,
        "layer_selections.integers",
        checked_mul(
            DEEPSEEK_V4_SELECTION_RECORD_I32_WIDTH,
            DEEPSEEK_V4_LAYER_COUNT,
            "layer-selection record elements",
        )?,
        i32_bytes,
    )?;
    push_session_allocation(
        &mut requests,
        "raw_cache",
        checked_mul(
            checked_mul(
                attention.head_dim,
                DEEPSEEK_V4_LOCAL_WINDOW,
                "raw-cache layer elements",
            )?,
            DEEPSEEK_V4_LAYER_COUNT,
            "raw-cache session elements",
        )?,
        f16_bytes,
    )?;

    let attention_dim = 512;
    let indexer_dim = 128;
    for (layer, kind) in attention_kinds.iter().copied().enumerate() {
        match kind {
            AttentionKind::SlidingWindow => {}
            AttentionKind::CompressedSparse => {
                append_compressor_frontier_allocations(
                    &mut requests,
                    &format!("compressor.{layer}.attention"),
                    4,
                    attention_dim,
                    DeepSeekV4CompressorPublication::Attention,
                    capacity.csa_physical_rows(),
                )?;
                append_compressor_frontier_allocations(
                    &mut requests,
                    &format!("compressor.{layer}.indexer"),
                    4,
                    indexer_dim,
                    DeepSeekV4CompressorPublication::IndexerHadamard,
                    capacity.csa_physical_rows(),
                )?;
            }
            AttentionKind::HeavilyCompressed => append_compressor_frontier_allocations(
                &mut requests,
                &format!("compressor.{layer}.attention"),
                128,
                attention_dim,
                DeepSeekV4CompressorPublication::Attention,
                capacity.hca_physical_rows(),
            )?,
        }
    }

    push_session_allocation(
        &mut requests,
        "moe.normalized_input",
        moe.hidden_size,
        f32_bytes,
    )?;
    push_session_allocation(&mut requests, "moe.logits", moe.expert_count, f32_bytes)?;
    push_session_allocation(&mut requests, "moe.expert_ids", moe.top_k, i32_bytes)?;
    push_session_allocation(&mut requests, "moe.weights", moe.top_k, f32_bytes)?;
    push_session_allocation(&mut requests, "moe.route_status", 1, i32_bytes)?;
    push_session_allocation(
        &mut requests,
        "layer_routes.integers",
        checked_mul(
            DEEPSEEK_V4_ROUTE_RECORD_I32_WIDTH,
            DEEPSEEK_V4_LAYER_COUNT,
            "layer-route integer record elements",
        )?,
        i32_bytes,
    )?;
    push_session_allocation(
        &mut requests,
        "layer_routes.weights",
        checked_mul(
            moe.top_k,
            DEEPSEEK_V4_LAYER_COUNT,
            "layer-route weight record elements",
        )?,
        f32_bytes,
    )?;
    push_session_allocation(
        &mut requests,
        "moe.gate",
        checked_mul(moe.ffn_size, 3, "MoE gate and fused Q6 scratch elements")?,
        f32_bytes,
    )?;
    for name in ["moe.up", "moe.inner"] {
        push_session_allocation(&mut requests, name, moe.ffn_size, f32_bytes)?;
    }
    push_session_allocation(
        &mut requests,
        "moe.routed_inner",
        checked_mul(moe.ffn_size, moe.top_k, "MoE all-slot inner elements")?,
        f32_bytes,
    )?;
    push_session_allocation(
        &mut requests,
        "moe.expert_outputs",
        checked_mul(moe.hidden_size, moe.top_k, "MoE expert-output elements")?,
        f32_bytes,
    )?;
    for name in ["moe.routed_output", "moe.shared_output", "moe.final_output"] {
        push_session_allocation(&mut requests, name, moe.hidden_size, f32_bytes)?;
    }

    for name in ["final_hidden", "final_normalized_hidden"] {
        push_session_allocation(&mut requests, name, DEEPSEEK_V4_HIDDEN_SIZE, f32_bytes)?;
    }
    push_session_allocation(&mut requests, "logits", DEEPSEEK_V4_VOCAB_SIZE, f32_bytes)?;
    prefill::append_session_allocation_requests(&mut requests, capacity.csa_physical_rows())?;
    Ok(requests)
}

pub(super) fn build_memory_plan(
    ctx: &MetalContext,
    retained: &RetainedStoragePlan,
    report: &DeepSeekV4ResidencyReport,
    config: &DeepSeekV4Config,
    capacity: DeepSeekV4SessionCapacity,
) -> Result<DeepSeekV4MemoryPlan, DeepSeekV4MetalError> {
    let mut residency_priced_upper_bytes = 0_u64;
    let mut residency_buffer_count = 0_usize;
    let mut residency_logical_bytes = 0_u64;
    for (index, window) in retained.windows.iter().enumerate() {
        let logical = u64::try_from(window.length).map_err(|_| {
            DeepSeekV4MetalError::Invalid("retained window length exceeds u64".into())
        })?;
        let (priced, _) = price_shared_buffer(ctx, logical, &format!("weight_window[{index}]"))?;
        residency_priced_upper_bytes = residency_priced_upper_bytes
            .checked_add(priced)
            .ok_or_else(|| {
                DeepSeekV4MetalError::Invalid("priced residency byte count overflow".into())
            })?;
        residency_logical_bytes =
            residency_logical_bytes
                .checked_add(logical)
                .ok_or_else(|| {
                    DeepSeekV4MetalError::Invalid("logical residency byte count overflow".into())
                })?;
        residency_buffer_count += 1;
    }
    for entry in &retained.entries {
        if matches!(
            entry.disposition,
            RetainedStorageDisposition::CopyFallback { .. }
        ) {
            let (priced, _) = price_shared_buffer(ctx, entry.n_bytes, &entry.name)?;
            residency_priced_upper_bytes = residency_priced_upper_bytes
                .checked_add(priced)
                .ok_or_else(|| {
                    DeepSeekV4MetalError::Invalid("priced fallback byte count overflow".into())
                })?;
            residency_logical_bytes = residency_logical_bytes
                .checked_add(entry.n_bytes)
                .ok_or_else(|| {
                    DeepSeekV4MetalError::Invalid("logical fallback byte count overflow".into())
                })?;
            residency_buffer_count += 1;
        }
    }
    if residency_logical_bytes != report.resident_bytes {
        return invalid(format!(
            "memory-plan residency bytes {residency_logical_bytes} differ from report {}",
            report.resident_bytes
        ));
    }

    let requests = deepseek_v4_session_allocation_requests(config, capacity)?;
    let mut session_allocations = Vec::with_capacity(requests.len());
    let mut session_logical_bytes = 0_u64;
    let mut session_priced_upper_bytes = 0_u64;
    for request in requests {
        let (priced_bytes, alignment) =
            price_shared_buffer(ctx, request.logical_bytes, &request.name)?;
        session_logical_bytes = session_logical_bytes
            .checked_add(request.logical_bytes)
            .ok_or_else(|| {
                DeepSeekV4MetalError::Invalid("session logical byte count overflow".into())
            })?;
        session_priced_upper_bytes = session_priced_upper_bytes
            .checked_add(priced_bytes)
            .ok_or_else(|| {
                DeepSeekV4MetalError::Invalid("session priced byte count overflow".into())
            })?;
        session_allocations.push(DeepSeekV4SessionAllocation {
            name: request.name,
            logical_bytes: request.logical_bytes,
            priced_bytes,
            alignment,
        });
    }
    let total_priced_upper_bytes = residency_priced_upper_bytes
        .checked_add(session_priced_upper_bytes)
        .ok_or_else(|| {
            DeepSeekV4MetalError::Invalid("total priced Metal byte count overflow".into())
        })?;
    total_priced_upper_bytes
        .checked_add(DEEPSEEK_V4_DYNAMIC_MEMORY_RESERVE_BYTES)
        .ok_or_else(|| {
            DeepSeekV4MetalError::Invalid(
                "total priced Metal bytes plus dynamic reserve overflow".into(),
            )
        })?;
    Ok(DeepSeekV4MemoryPlan {
        residency_buffer_count,
        residency_logical_bytes,
        residency_priced_upper_bytes,
        session_logical_bytes,
        session_priced_upper_bytes,
        total_priced_upper_bytes,
        session_allocations,
    })
}

pub(super) fn report_for_plan(
    plan: &RetainedStoragePlan,
) -> Result<DeepSeekV4ResidencyReport, DeepSeekV4MetalError> {
    let source_bytes = plan.entries.iter().try_fold(0_u64, |total, entry| {
        total
            .checked_add(entry.n_bytes)
            .ok_or_else(|| DeepSeekV4MetalError::Invalid("source byte count overflow".into()))
    })?;
    let window_bytes = plan.windows.iter().try_fold(0_u64, |total, window| {
        total
            .checked_add(window.length as u64)
            .ok_or_else(|| DeepSeekV4MetalError::Invalid("window byte count overflow".into()))
    })?;
    let resident_bytes = window_bytes
        .checked_add(plan.unique_fallback_bytes)
        .ok_or_else(|| DeepSeekV4MetalError::Invalid("resident byte count overflow".into()))?;
    let view_count = plan
        .entries
        .iter()
        .filter(|entry| matches!(entry.disposition, RetainedStorageDisposition::View { .. }))
        .count();
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
    Ok(DeepSeekV4ResidencyReport {
        tensor_count: plan.entries.len(),
        source_bytes,
        window_count: plan.windows.len(),
        window_bytes,
        view_count,
        unique_view_bytes: plan.unique_view_bytes,
        logical_view_bytes: plan.logical_view_bytes,
        alias_count,
        alias_bytes: plan.alias_bytes,
        fallback_count,
        fallback_bytes: plan.unique_fallback_bytes,
        resident_bytes,
        page_size: plan.page_size,
        max_buffer_length: plan.max_buffer_length,
        required_alignment: plan.required_alignment,
    })
}
