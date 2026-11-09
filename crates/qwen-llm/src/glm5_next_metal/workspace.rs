//! Prompt scratch can be absent without changing the session's arithmetic.
//! Every re-acquisition admits the original width before any state advances.

use super::*;
use objc2_metal::MTLResource;

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
        let released = self.release_workspace(ctx)?;
        tracing::info!(target: "qwen_diag",
            "glm workspace: phase=decode configured_rows={} allocated_rows=0 released_bytes={released}",
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
        session.prefill_packed(&ctx, &tokens[..8192]).unwrap();
        let prefix = session.capture_snapshot().unwrap();
        let append_logits = session.prefill_packed(&ctx, &tokens[8192..8704]).unwrap();
        let appended = session.capture_snapshot().unwrap();
        let full = *session.ledger();
        let targets = &tokens[8704..8769];
        let rollout = |session: &mut Glm5NextSession<'_>| -> Vec<Vec<f32>> {
            targets[..64]
                .iter()
                .map(|&t| session.forward(&ctx, t).unwrap())
                .collect()
        };
        let reference = rollout(&mut session);
        session.restore_snapshot(&appended).unwrap();
        let noise = rollout(&mut session);
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
        session.prepare_decode(&ctx).unwrap();
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
                "signed_dnll":candidate.0, "kl":candidate.1,
                "noise_signed_dnll":noise.0, "noise_kl":noise.1, "append_kl":append_kl,
            })
        );
    }
}
