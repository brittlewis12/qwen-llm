//! Prompt scratch can be absent without changing the session's arithmetic.
//! Every re-acquisition admits the original width before any state advances.

use super::*;
use objc2_metal::MTLResource;
use std::cell::Cell;

crate::env_flag!(
    /// Commit admitted GLM prompt-phase pages before warming weights on memory-tight non-Apple10 devices. Preserves width and arithmetic; zero restores lazy allocation and startup prefetch order. Experimental until the residency gate passes.
    default_off peak_requested,
    "QWEN_GLM53_PREFILL_PEAK_FIRST"
);

thread_local! { static COMMIT_PAGE: Cell<usize> = const { Cell::new(0) }; }

pub(super) struct CommitScope(usize);

impl CommitScope {
    pub(super) fn new(ctx: &MetalContext, ledger: &Glm5NextMemoryLedger) -> Result<Self> {
        let page = if Glm5NextSession::prefill_peak_first(ctx, ledger) {
            crate::metal::host_page_size_bytes()?
        } else {
            0
        };
        Ok(Self(COMMIT_PAGE.with(|value| value.replace(page))))
    }
}

impl Drop for CommitScope {
    fn drop(&mut self) {
        COMMIT_PAGE.with(|value| value.set(self.0));
    }
}

/// Called only on fresh, initialized, owned shared buffers before publication.
pub(super) fn commit_new_tensor(tensor: &MetalTensor) {
    COMMIT_PAGE.with(|page| {
        let page = page.get();
        if page == 0 {
            return;
        }
        let pointer = tensor.buffer.contents().as_ptr().cast::<u8>();
        let bytes = tensor.buffer.length();
        // Writing the existing byte faults in a private physical page without
        // changing zero or nonzero initialization. No GPU can own this tensor.
        for offset in (0..bytes).step_by(page) {
            unsafe {
                let address = pointer.add(offset);
                address.write_volatile(address.read_volatile());
            }
        }
    });
}

crate::env_flag!(
    /// Default-on release of GLM prompt workspace before ordinary decode when the session peak reaches 75% of physical memory and independent batching is unsupported. The configured prefill width, lineage and F32/Half operand policy survive release; later appends re-admit the same buffers before mutation. Zero retains prompt scratch. Supported M5 independent batches keep their existing workspace and F32-weight contract.
    default_on release_requested,
    "QWEN_GLM53_RELEASE_PREFILL"
);

#[derive(Clone, Copy)]
pub(super) struct PrefillConfig {
    pub rows: usize,
    pub arith: packed::Arith,
}

fn memory_tight(peak: u64, physical: Option<u64>, batch_supported: bool) -> bool {
    !batch_supported
        && physical.is_some_and(|physical| {
            physical > 0 && u128::from(peak) * 4 >= u128::from(physical) * 3
        })
}

pub(super) fn release_policy(
    ctx: &MetalContext,
    weights: &Glm5NextWeights,
    ledger: &Glm5NextMemoryLedger,
) -> bool {
    release_requested()
        && ledger.prefill_rows() > 0
        && memory_tight(
            ledger.peak_bytes(),
            ctx.device_facts().physical_memory_bytes,
            Glm5NextBatch::supported(ctx, weights),
        )
}

impl Glm5NextSession<'_> {
    /// Whether ordinary prompt setup commits its admitted physical peak first.
    /// Apple10 paths retain their independent batch allocation contract.
    pub fn prefill_peak_first(ctx: &MetalContext, ledger: &Glm5NextMemoryLedger) -> bool {
        peak_requested()
            && ledger.prefill_rows() > 0
            && memory_tight(
                ledger.peak_bytes(),
                ctx.device_facts().physical_memory_bytes,
                ctx.device_facts().gpu_families.apple10,
            )
    }
    /// The configured prefill ceiling, including while its buffers are released.
    pub fn prefill_rows(&self) -> usize {
        self.prefill_config.map_or(0, |config| config.rows)
    }

    /// Allocated prompt rows; zero means ordinary decode owns no prompt scratch.
    pub fn prefill_workspace_rows(&self) -> usize {
        self.packed.as_ref().map_or(0, |packed| packed.rows())
    }

    pub(super) fn prefill_arith(&self) -> Option<packed::Arith> {
        self.packed
            .as_ref()
            .map(|packed| packed.arith())
            .or(self.prefill_config.map(|config| config.arith))
    }

    /// Release prompt-only buffers on eligible memory-tight devices. No state,
    /// routes, logits, arithmetic policy or snapshot identity changes. Calls
    /// are synchronous, so no command can still be using these buffers.
    /// Returns the ledger's released bytes, not a device-counter estimate.
    pub fn prepare_decode(&mut self, ctx: &MetalContext) -> Result<u64> {
        if self.poisoned {
            return Err(Glm5NextMetalError::Poisoned);
        }
        if !self.release_prefill_for_decode || self.packed.is_none() {
            return Ok(0);
        }
        let allocated = ctx.current_allocated_size();
        let released = self.release_workspace(ctx)?;
        let device_released = allocated.saturating_sub(ctx.current_allocated_size());
        tracing::info!(target: "qwen_diag",
            "glm workspace: phase=decode configured_rows={} allocated_rows=0 released_bytes={released} device_released_bytes={device_released}",
            self.prefill_rows());
        Ok(released)
    }

    fn workspace_ledger(&self, ctx: &MetalContext, rows: usize) -> Result<Glm5NextMemoryLedger> {
        if self.weights.embedding.buffer.device().registryID() != ctx.device.registryID() {
            return invalid("prompt workspace belongs to another device");
        }
        Ok(Glm5NextMemoryLedger::new_with_options(
            &self.weights.config,
            self.weights.retained_bytes,
            self.capacity as u64,
            rows as u64,
            self.ledger.scratch_options(),
            &device_price(ctx),
        )?)
    }

    pub(super) fn release_workspace(&mut self, ctx: &MetalContext) -> Result<u64> {
        let Some(packed) = &self.packed else {
            return Ok(0);
        };
        let arith = packed.arith();
        let ledger = self.workspace_ledger(ctx, 0)?;
        let released = self.ledger.session_buffer_bytes() - ledger.session_buffer_bytes();
        self.prefill_config.as_mut().expect("packed policy").arith = arith;
        self.packed = None;
        self.ledger = ledger;
        Ok(released)
    }

    /// Re-admit the configured prompt workspace before an append. Refusal or
    /// allocation failure leaves causal state and policy untouched, with no
    /// serial fallback or change to the command width. The caller's future CPU
    /// allowance (such as serve transport) is charged in the same decision.
    pub fn prepare_prefill(&mut self, ctx: &MetalContext, cpu_reserve: u64) -> Result<()> {
        if self.poisoned {
            return Err(Glm5NextMetalError::Poisoned);
        }
        let Some(config) = self.prefill_config else {
            return Ok(());
        };
        if self.packed.is_some() {
            return Ok(());
        }
        let ledger = self.workspace_ledger(ctx, config.rows)?;
        let extra = ledger.session_buffer_bytes() - self.ledger.session_buffer_bytes();
        let _allocation = ctx.begin_allocation_transaction();
        let admission = evaluate_metal_memory_admission_with_cpu_bytes(
            extra,
            cpu_reserve,
            memory::DYNAMIC_RESERVE_BYTES,
            ctx.memory_signals(),
            true,
        );
        if let Some(denied) = admission.refusal() {
            return Err(Glm5NextMetalError::MemoryAdmission {
                denied,
                budget_bytes: admission_budget(&admission),
                advice: CapacityAdvice::NotEvaluated,
            });
        }
        let _commit = CommitScope::new(ctx, &ledger)?;
        let mut packed = packed::PackedScratch::new_with_options(
            ctx,
            &self.weights.config,
            config.rows,
            self.capacity,
            ledger.scratch_options(),
        )?;
        packed.lineage = config.arith.lineage;
        packed.precision = config.arith.precision;
        self.packed = Some(packed);
        self.ledger = ledger;
        tracing::info!(target: "qwen_diag",
            "glm workspace: phase=prefill configured_rows={} allocated_rows={} admitted_bytes={extra}",
            config.rows, config.rows);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_workspace_policy_requires_tight_memory_and_excludes_independent_batches() {
        assert!(!memory_tight(74, Some(100), false));
        assert!(memory_tight(75, Some(100), false));
        assert!(memory_tight(u64::MAX, Some(u64::MAX), false));
        assert!(!memory_tight(u64::MAX, Some(u64::MAX), true));
        assert!(!memory_tight(u64::MAX, None, false));
        assert!(!memory_tight(u64::MAX, Some(0), false));
        // The lane's actual memory geometry, without a device-name gate.
        assert!(memory_tight(116 << 30, Some(128 << 30), false));
        assert!(!memory_tight(120 << 30, Some(256 << 30), false));
    }

    /// Matched native timing: one weights instance and persistent state, the
    /// same captured prefix, capacity and teacher-forced inputs. Only unused
    /// prompt storage changes. The one-row comparator never runs an append;
    /// production keeps its original 2048-row command ceiling.
    #[test]
    #[ignore = "model characterization: glm-5.3-flash-maya-m-iq2_s"]
    fn m4_native_decode_matches_one_row_reference() {
        assert_eq!(std::env::var("MTL_DEBUG_LAYER").as_deref(), Ok("0"));
        let _lease = crate::metal::acquire_metal_benchmark_lease().unwrap();
        let fixture = crate::test_fixtures::Fixture {
            id: "glm-5.3-flash-maya-m-iq2_s",
            env: &[],
            filename: "GLM-5.3-Flash-Maya-M-IQ2_S-00001-of-00003.gguf",
        };
        let gguf = GgufFile::open(fixture.required()).unwrap();
        let ctx = MetalContext::new().unwrap();
        let weights = Glm5NextWeights::load(&ctx, &gguf).unwrap();
        let tokenizer = crate::tokenizer::Tokenizer::from_gguf(&gguf).unwrap();
        let text = format!(
            "[gMASK]<sop>Review this Rust server.\n{}\n{}",
            include_str!("../../../qwen-cli/src/serve/backend_glm5_next.rs"),
            super::super::tests::long_qualification_text()
        );
        let tokens: Vec<u32> = tokenizer
            .encode(&text, false)
            .unwrap()
            .into_iter()
            .map(|t| t as u32)
            .collect();
        assert!(tokens.len() >= 8256);
        // One upfront allowance for the future CPU snapshot, two complete
        // logit streams and the tiny reference workspace. Both GPU shapes
        // are normally admitted before prefill; no timing arm allocates.
        let cpu_reserve = 512 << 20;
        let logits_reserve = 2 * 64 * u64::from(weights.config.vocab_size) * 4;
        let mut session = Glm5NextSession::with_prefill_rows_and_cpu_reserve(
            &ctx,
            &weights,
            10000,
            2048,
            cpu_reserve,
        )
        .unwrap();
        assert!(
            session.release_prefill_for_decode,
            "requires the eligible memory-tight default path"
        );
        session.set_packed_lineage(PackedLineage::Fast).unwrap();
        session.set_fast_precision(FastPrecision::Half).unwrap();
        let decode_ledger = session.workspace_ledger(&ctx, 0).unwrap();
        let reference_ledger = session.workspace_ledger(&ctx, 1).unwrap();
        let reference_bytes =
            reference_ledger.session_buffer_bytes() - decode_ledger.session_buffer_bytes();
        let mut parked_reference = {
            let _allocation = ctx.begin_allocation_transaction();
            // Carve the already planned GPU bytes out of the remaining
            // future allowance; the total charge and dynamic reserve do
            // not decrease. CPU snapshot/logits still fit the remainder.
            let admission = evaluate_metal_memory_admission_with_cpu_bytes(
                reference_bytes,
                cpu_reserve.checked_sub(reference_bytes).unwrap(),
                memory::DYNAMIC_RESERVE_BYTES,
                ctx.memory_signals(),
                true,
            );
            assert!(
                admission.refusal().is_none(),
                "reference admission: {admission:?}"
            );
            let mut reference = packed::PackedScratch::new_with_options(
                &ctx,
                &weights.config,
                1,
                10000,
                reference_ledger.scratch_options(),
            )
            .unwrap();
            reference.lineage = PackedLineage::Fast;
            reference.precision = FastPrecision::Half;
            Some(reference)
        };
        session.prefill_packed(&ctx, &tokens[..8192]).unwrap();
        let prefix = session.capture_snapshot().unwrap();
        assert!(prefix.bytes() as u64 + logits_reserve + reference_bytes < cpu_reserve);
        session.prepare_decode(&ctx).unwrap();

        let thermal = || {
            let status = std::process::Command::new("notifyutil")
                .args(["-g", "com.apple.system.thermalpressurelevel"])
                .output()
                .unwrap();
            assert!(status.status.success());
            let text = String::from_utf8(status.stdout).unwrap();
            assert_eq!(text.split_whitespace().last(), Some("0"), "{text}");
        };
        let mut oracle: Option<Vec<Vec<f32>>> = None;
        let mut samples = Vec::new();
        let mut records = Vec::new();
        for (label, resident) in [
            ("stock1", false),
            ("resident1", true),
            ("resident2", true),
            ("stock2", false),
        ] {
            // Keep the same pre-admitted device allocation in both arms.
            // Stock's session owns no prompt workspace; the tiny comparator
            // is parked outside it until the reference arm borrows it.
            if resident && session.packed.is_none() {
                session.packed = parked_reference.take();
            } else if !resident && session.packed.is_some() {
                parked_reference = session.packed.take();
            }
            session.prefill_config.as_mut().unwrap().rows = if resident { 1 } else { 2048 };
            session.ledger = if resident {
                reference_ledger
            } else {
                decode_ledger
            };
            session.release_prefill_for_decode = !resident;
            std::thread::sleep(std::time::Duration::from_secs(30));
            thermal();
            session.restore_snapshot(&prefix).unwrap();
            // Priming follows cooling; restoring after this warm-up is
            // untimed, as in the original resident roofline measurement.
            for &token in &tokens[8192..8256] {
                session.forward(&ctx, token).unwrap();
            }
            session.restore_snapshot(&prefix).unwrap();
            thermal();
            let start = std::time::Instant::now();
            let logits: Vec<Vec<f32>> = tokens[8192..8256]
                .iter()
                .map(|&token| session.forward(&ctx, token).unwrap())
                .collect();
            let ms = start.elapsed().as_secs_f64() * 1e3;
            thermal();
            assert!(ms.is_finite() && ms > 0.0);
            let kl = oracle.as_ref().map_or(0.0, |reference| {
                reference
                    .iter()
                    .zip(&logits)
                    .map(|(a, b)| super::super::tests::kl_divergence(a, b))
                    .sum::<f64>()
                    / 64.0
            });
            assert!(kl <= 1e-6, "matched-state KL {kl}");
            if oracle.is_none() {
                oracle = Some(logits);
            }
            samples.push((resident, ms));
            let record = serde_json::json!({"label":label, "native_ms":ms,
                "tps":64000.0/ms, "configured_rows":session.prefill_rows(),
                "allocated_rows":session.prefill_workspace_rows(), "kl":kl,
                "nonweight_device_bytes":ctx.current_allocated_size().saturating_sub(weights.retained_bytes)});
            eprintln!("matched native arm: {record}");
            records.push(record);
        }
        let stock: Vec<f64> = samples
            .iter()
            .filter(|(r, _)| !r)
            .map(|(_, ms)| *ms)
            .collect();
        let resident: Vec<f64> = samples
            .iter()
            .filter(|(r, _)| *r)
            .map(|(_, ms)| *ms)
            .collect();
        let stock_ms = (stock[0] + stock[1]) / 2.0;
        let resident_ms = (resident[0] + resident[1]) / 2.0;
        let delta_ms_per_token = (stock_ms - resident_ms) / 64.0;
        let noise_ms_per_token = (stock[0] - stock[1])
            .abs()
            .max((resident[0] - resident[1]).abs())
            / 64.0;
        // Characterization reports the decision; it does not make CI depend
        // on a speed threshold. The preregistered floor is 0.10 ms/token.
        eprintln!(
            "matched native summary: {}",
            serde_json::json!({
            "context":8192, "tokens":64, "capacity":10000,
            "reference_workspace_bytes":reference_bytes, "reference_preadmitted":true,
            "snapshot_bytes":prefix.bytes(), "arms":records,
            "stock_tps":64000.0/stock_ms, "resident_tps":64000.0/resident_ms,
            "delta_ms_per_token":delta_ms_per_token,
            "noise_ms_per_token":noise_ms_per_token, "floor_ms_per_token":0.10,
            "within_noise":delta_ms_per_token <= noise_ms_per_token.max(0.10)})
        );
    }

    #[test]
    #[ignore = "model witness: glm-5.3-flash-maya-m-iq2_s"]
    fn append_snapshot_release_preserves_distribution_and_policy() {
        assert_eq!(std::env::var("MTL_DEBUG_LAYER").as_deref(), Ok("1"));
        let _lease = crate::metal::acquire_metal_benchmark_lease().unwrap();
        let fixture = crate::test_fixtures::Fixture {
            id: "glm-5.3-flash-maya-m-iq2_s",
            env: &[],
            filename: "GLM-5.3-Flash-Maya-M-IQ2_S-00001-of-00003.gguf",
        };
        let gguf = GgufFile::open(fixture.required()).unwrap();
        let ctx = MetalContext::new().unwrap();
        let weights = Glm5NextWeights::load(&ctx, &gguf).unwrap();
        let tokenizer = crate::tokenizer::Tokenizer::from_gguf(&gguf).unwrap();
        let text = format!(
            "[gMASK]<sop>Review this Rust server for memory lifetime bugs.\n{}\n{}",
            include_str!("../../../qwen-cli/src/serve/backend_glm5_next.rs"),
            super::super::tests::long_qualification_text(),
        );
        let tokens: Vec<u32> = tokenizer
            .encode(&text, false)
            .unwrap()
            .into_iter()
            .map(|t| t as u32)
            .collect();
        assert!(tokens.len() >= 8192 + 512 + 65);
        let mut session = Glm5NextSession::with_prefill_rows(&ctx, &weights, 9000, 2048).unwrap();
        session.set_packed_lineage(PackedLineage::Fast).unwrap();
        session.set_fast_precision(FastPrecision::Half).unwrap();
        // Explicit incumbent control, independent of the process-wide knob.
        session.release_prefill_for_decode = false;
        objc2::rc::autoreleasepool(|_| session.prefill_packed(&ctx, &tokens[..8192])).unwrap();
        let prefix = session.capture_snapshot().unwrap();
        let append_logits =
            objc2::rc::autoreleasepool(|_| session.prefill_packed(&ctx, &tokens[8192..8704]))
                .unwrap();
        let appended = session.capture_snapshot().unwrap();
        let full = *session.ledger();
        let targets = &tokens[8704..8769];
        let rollout = |session: &mut Glm5NextSession<'_>| -> Vec<Vec<f32>> {
            targets[..64]
                .iter()
                .map(|&t| session.forward(&ctx, t).unwrap())
                .collect()
        };
        let reference = objc2::rc::autoreleasepool(|_| rollout(&mut session));
        session.restore_snapshot(&appended).unwrap();
        let noise = objc2::rc::autoreleasepool(|_| rollout(&mut session));
        session.restore_snapshot(&appended).unwrap();
        session.release_prefill_for_decode = true;
        let released = session.prepare_decode(&ctx).unwrap();
        assert!(released > 0);
        assert_eq!(session.prefill_workspace_rows(), 0);
        assert_eq!(session.prefill_rows(), 2048);
        assert_eq!(session.ledger().prefill_rows(), 0);
        assert_eq!(
            full.session_buffer_bytes() - session.ledger().session_buffer_bytes(),
            released
        );
        assert_eq!(session.packed_lineage(), Some(PackedLineage::Fast));
        assert_eq!(session.fast_precision(), Some(FastPrecision::Half));
        assert!(session.set_fast_precision(FastPrecision::F32).is_err());
        assert!(session.set_packed_lineage(PackedLineage::Exact).is_err());
        assert!(session.capture_snapshot().unwrap().same_state(&appended));
        let candidate = rollout(&mut session);
        let score = |rows: &[Vec<f32>]| -> (f64, f64) {
            let mut dnll = 0.0;
            let mut kl = 0.0;
            let logp = |row: &[f32], target: u32| {
                let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
                f64::from(row[target as usize])
                    - max
                    - row
                        .iter()
                        .map(|&x| (f64::from(x) - max).exp())
                        .sum::<f64>()
                        .ln()
            };
            for (i, (a, b)) in reference.iter().zip(rows).enumerate() {
                dnll += logp(a, targets[i + 1]) - logp(b, targets[i + 1]);
                kl += super::super::tests::kl_divergence(a, b);
            }
            (dnll / rows.len() as f64, kl / rows.len() as f64)
        };
        let noise = score(&noise);
        let candidate = score(&candidate);
        assert!(
            candidate.0.abs() <= noise.0.abs() + 1e-6,
            "signed dNLL {candidate:?}, noise {noise:?}"
        );
        assert!(
            candidate.1 <= noise.1 + 1e-6,
            "KL {candidate:?}, noise {noise:?}"
        );

        // A later append must re-admit the original width and operand policy,
        // even though snapshot restore itself needs no prompt scratch.
        session.restore_snapshot(&prefix).unwrap();
        let replay = session.prefill_packed(&ctx, &tokens[8192..8704]).unwrap();
        assert_eq!(session.prefill_workspace_rows(), 2048);
        assert_eq!(*session.ledger(), full);
        assert!(session.capture_snapshot().unwrap().same_state(&appended));
        let append_kl = super::super::tests::kl_divergence(&append_logits, &replay);
        assert!(append_kl <= 1e-6);
        let weak = objc2::rc::Weak::from_retained(
            &session.packed.as_ref().unwrap().route_ids()[0].1.buffer,
        );
        let allocated = ctx.current_allocated_size();
        let released_again = session.prepare_decode(&ctx).unwrap();
        let device_released = allocated.saturating_sub(ctx.current_allocated_size());
        assert!(
            weak.load().is_none(),
            "completed prompt command retained its route buffer"
        );
        assert!(
            device_released >= released_again.saturating_sub(16 << 20),
            "ledger freed {released_again} bytes but device freed only {device_released}"
        );
        session.poisoned = true;
        assert!(matches!(
            session.prepare_prefill(&ctx, 0),
            Err(Glm5NextMetalError::Poisoned)
        ));
        assert!(matches!(
            session.prepare_decode(&ctx),
            Err(Glm5NextMetalError::Poisoned)
        ));
        assert_eq!(session.prefill_workspace_rows(), 0);
        eprintln!(
            "workspace witness: {}",
            serde_json::json!({
                "context":8192, "append":512, "teacher_forced":64,
                "released_bytes":released, "snapshot_bytes":appended.bytes(),
                "device_released_bytes":device_released,
                "signed_dnll":candidate.0, "kl":candidate.1,
                "noise_signed_dnll":noise.0, "noise_kl":noise.1, "append_kl":append_kl,
            })
        );
    }
}
