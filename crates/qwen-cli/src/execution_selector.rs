use super::{ExplicitCliOptions, GreedyGpuArgmaxMode, PrefillChunkArg};
use qwen_llm::model::{Arch, ArchKind};
use qwen_llm::model_family::ModelFamily;
use qwen_llm::moe_batch16::{HeadMode, MoeBatch16PlanTelemetry};
use serde::Serialize;

#[derive(Clone, Copy, Debug, Eq, PartialEq, clap::ValueEnum)]
pub(super) enum ExecutionModeArg {
    Serial,
    Auto,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum SelectedExecution {
    Serial,
    Concurrency2,
    DenseBatch8,
    MoeBatch16,
}

impl SelectedExecution {
    pub(super) fn as_str(self) -> &'static str {
        match self {
            Self::Serial => "serial",
            Self::Concurrency2 => "concurrency2",
            Self::DenseBatch8 => "dense_batch8",
            Self::MoeBatch16 => "moe_batch16",
        }
    }

    pub(super) fn width(self) -> usize {
        match self {
            Self::Serial => 1,
            Self::Concurrency2 => 2,
            Self::DenseBatch8 => 8,
            Self::MoeBatch16 => 16,
        }
    }

    pub(super) fn accelerated(self) -> bool {
        self != Self::Serial
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum QwenMoeEconomicsProfile {
    A3bQ4,
    A3bIq4,
    Unqualified,
}

impl QwenMoeEconomicsProfile {
    fn as_str(self) -> &'static str {
        match self {
            Self::A3bQ4 => "a3b_q4_non_mtp",
            Self::A3bIq4 => "a3b_iq4_non_mtp",
            Self::Unqualified => "unqualified",
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) struct QwenSelectionInput {
    pub mode: Option<ExecutionModeArg>,
    pub family: Option<ModelFamily>,
    pub arch: Arch,
    pub request_count: usize,
    pub all_requests_accelerable: bool,
    pub fixed_prefill_chunk: bool,
    pub acceleration_blocker: Option<&'static str>,
    pub dense_full_cohorts: usize,
    pub dense_serial_remainders: usize,
    pub moe_full_cohorts: usize,
    pub moe_serial_remainders: usize,
    pub fixed_cohort_economics_rejected: bool,
    pub concurrency2_memory_admitted: bool,
    pub dense_batch8_memory_admitted: bool,
    pub moe_batch16_memory_admitted: bool,
    pub moe_plan: Option<MoeBatch16PlanTelemetry>,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct DeepSeekSelectionInput {
    pub mode: Option<ExecutionModeArg>,
    pub request_count: usize,
    pub stdin: bool,
    pub residency_set: bool,
    pub two_session_memory_admitted: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct ExecutionSelection {
    pub selected: SelectedExecution,
    pub reason: &'static str,
    pub profile: &'static str,
    pub full_cohorts: usize,
    pub serial_remainder_requests: usize,
    pub moe_plan: Option<MoeBatch16PlanTelemetry>,
}

#[derive(Debug, Serialize)]
struct MoePlanRecord {
    q8_batched_gdn_blocks: usize,
    packed_q4_gate_up_blocks: usize,
    per_lane_gdn_blocks: usize,
    per_lane_gate_up_blocks: usize,
    per_lane_iq3_gate_up_blocks: usize,
    per_lane_other_gate_up_blocks: usize,
    head_mode: &'static str,
}

impl From<MoeBatch16PlanTelemetry> for MoePlanRecord {
    fn from(plan: MoeBatch16PlanTelemetry) -> Self {
        Self {
            q8_batched_gdn_blocks: plan.q8_batched_gdn_blocks,
            packed_q4_gate_up_blocks: plan.packed_q4_gate_up_blocks,
            per_lane_gdn_blocks: plan.per_lane_gdn_blocks,
            per_lane_gate_up_blocks: plan.per_lane_gate_up_blocks,
            per_lane_iq3_gate_up_blocks: plan.per_lane_iq3_gate_up_blocks,
            per_lane_other_gate_up_blocks: plan.per_lane_other_gate_up_blocks,
            head_mode: match plan.head_mode {
                HeadMode::BatchedQ6 => "batched_q6",
                HeadMode::BatchedQ8 => "batched_q8",
                HeadMode::PerLane => "per_lane",
            },
        }
    }
}

#[derive(Debug, Serialize)]
pub(super) struct ExecutionSelectionRecord {
    schema_version: u32,
    backend: &'static str,
    requested_mode: &'static str,
    family: &'static str,
    selected_mode: &'static str,
    width: usize,
    reason: &'static str,
    profile: &'static str,
    requests: Option<usize>,
    full_cohorts: usize,
    serial_remainder_requests: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    ragged_prompt_policy: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    ragged_prompt_plan_decision: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    refill_policy: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    planned_refill_arenas: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    planned_refill_requests: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    moe_plan: Option<MoePlanRecord>,
}

impl ExecutionSelectionRecord {
    pub(super) fn new(
        family: Option<ModelFamily>,
        requests: Option<usize>,
        selection: ExecutionSelection,
        ragged_prompt_policy: Option<&'static str>,
        ragged_prompt_plan_decision: Option<&'static str>,
        refill_policy: Option<&'static str>,
        planned_refill_arenas: Option<usize>,
        planned_refill_requests: Option<usize>,
    ) -> Self {
        Self {
            schema_version: 3,
            backend: "execution_selector_v3",
            requested_mode: "auto",
            family: match family {
                Some(ModelFamily::Qwen35) => "qwen35",
                Some(ModelFamily::Qwen35Moe) => "qwen35moe",
                Some(ModelFamily::Qwen4Exp) => "qwen4exp",
                Some(ModelFamily::DeepSeek4) => "deepseek4",
                Some(ModelFamily::MuseGlimmer) => "muse-glimmer",
                Some(ModelFamily::K2Horizon) => "k2-horizon",
                None => "unknown",
            },
            selected_mode: selection.selected.as_str(),
            width: selection.selected.width(),
            reason: selection.reason,
            profile: selection.profile,
            requests,
            full_cohorts: selection.full_cohorts,
            serial_remainder_requests: selection.serial_remainder_requests,
            ragged_prompt_policy,
            ragged_prompt_plan_decision,
            refill_policy,
            planned_refill_arenas,
            planned_refill_requests,
            moe_plan: selection.moe_plan.map(Into::into),
        }
    }
}

fn serial(
    reason: &'static str,
    profile: &'static str,
    moe_plan: Option<MoeBatch16PlanTelemetry>,
) -> ExecutionSelection {
    ExecutionSelection {
        selected: SelectedExecution::Serial,
        reason,
        profile,
        full_cohorts: 0,
        serial_remainder_requests: 0,
        moe_plan,
    }
}

fn is_measured_a3b_arch(arch: Arch) -> bool {
    arch.kind == ArchKind::Moe
        && arch.n_layer == 40
        && arch.hidden_size == 2_048
        && arch.vocab_size == 248_320
        && arch.full_attention_interval == 4
        && arch.n_q_heads == 16
        && arch.n_kv_heads == 2
        && arch.attn_head_dim == 256
        && arch.rope_theta.to_bits() == 10_000_000.0f32.to_bits()
        && arch.partial_rotary_factor.to_bits() == 0.25f32.to_bits()
        && arch.gdn_n_v_heads == 32
        && arch.gdn_n_k_heads == 16
        && arch.gdn_head_dim == 128
        && arch.gdn_conv_kernel == 4
        && arch.expert_count == 256
        && arch.expert_used_count == 8
        && arch.expert_feed_forward_length == 512
        && arch.expert_shared_feed_forward_length == 512
        && arch.mtp_n_hidden_layers == 0
}

fn qwen_moe_profile(arch: Arch, plan: Option<MoeBatch16PlanTelemetry>) -> QwenMoeEconomicsProfile {
    let Some(plan) = plan else {
        return QwenMoeEconomicsProfile::Unqualified;
    };
    if !is_measured_a3b_arch(arch)
        || plan.q8_batched_gdn_blocks != 30
        || plan.per_lane_gdn_blocks != 0
        || plan.head_mode != HeadMode::BatchedQ6
    {
        return QwenMoeEconomicsProfile::Unqualified;
    }
    if plan.packed_q4_gate_up_blocks == 40 && plan.per_lane_gate_up_blocks == 0 {
        QwenMoeEconomicsProfile::A3bQ4
    } else if plan.packed_q4_gate_up_blocks == 0
        && plan.per_lane_gate_up_blocks == 40
        && plan.per_lane_iq3_gate_up_blocks == 40
        && plan.per_lane_other_gate_up_blocks == 0
    {
        QwenMoeEconomicsProfile::A3bIq4
    } else {
        QwenMoeEconomicsProfile::Unqualified
    }
}

pub(super) fn select_qwen(input: QwenSelectionInput) -> ExecutionSelection {
    if input.mode != Some(ExecutionModeArg::Auto) {
        return serial("explicit_serial", "not_evaluated", input.moe_plan);
    }
    if let Some(reason) = input.acceleration_blocker {
        return serial(reason, "not_evaluated", input.moe_plan);
    }
    if !input.all_requests_accelerable {
        return serial(
            "request_contract_incompatible",
            "not_evaluated",
            input.moe_plan,
        );
    }

    match input.family {
        Some(ModelFamily::Qwen35) => {
            if input.fixed_prefill_chunk
                && input.dense_full_cohorts > 0
                && input.dense_batch8_memory_admitted
            {
                ExecutionSelection {
                    selected: SelectedExecution::DenseBatch8,
                    reason: "supported_family_default",
                    profile: "dense_qwen",
                    full_cohorts: input.dense_full_cohorts,
                    serial_remainder_requests: input.dense_serial_remainders,
                    moe_plan: None,
                }
            } else if input.request_count >= 2 && input.concurrency2_memory_admitted {
                ExecutionSelection {
                    selected: SelectedExecution::Concurrency2,
                    reason: if input.dense_full_cohorts > 0 {
                        "memory_narrowed_to_pair"
                    } else if input.fixed_cohort_economics_rejected {
                        "utilization_narrowed_to_pair"
                    } else {
                        "independent_pair_fallback"
                    },
                    profile: "dense_qwen",
                    full_cohorts: input.request_count / 2,
                    serial_remainder_requests: input.request_count % 2,
                    moe_plan: None,
                }
            } else {
                serial(
                    if input.request_count >= 2 {
                        "parallel_memory_denied"
                    } else {
                        "no_ready_parallel_work"
                    },
                    "dense_qwen",
                    None,
                )
            }
        }
        Some(ModelFamily::Qwen35Moe) => {
            let profile = qwen_moe_profile(input.arch, input.moe_plan);
            if profile == QwenMoeEconomicsProfile::A3bQ4
                && input.fixed_prefill_chunk
                && input.moe_full_cohorts > 0
                && input.moe_batch16_memory_admitted
            {
                ExecutionSelection {
                    selected: SelectedExecution::MoeBatch16,
                    reason: "measured_width_preference",
                    profile: profile.as_str(),
                    full_cohorts: input.moe_full_cohorts,
                    serial_remainder_requests: input.moe_serial_remainders,
                    moe_plan: input.moe_plan,
                }
            } else if matches!(
                profile,
                QwenMoeEconomicsProfile::A3bQ4 | QwenMoeEconomicsProfile::A3bIq4
            ) && input.request_count >= 2
                && input.concurrency2_memory_admitted
            {
                ExecutionSelection {
                    selected: SelectedExecution::Concurrency2,
                    reason: if profile == QwenMoeEconomicsProfile::A3bQ4
                        && input.moe_full_cohorts > 0
                    {
                        "memory_narrowed_to_pair"
                    } else if profile == QwenMoeEconomicsProfile::A3bQ4
                        && input.fixed_cohort_economics_rejected
                    {
                        "utilization_narrowed_to_pair"
                    } else {
                        "measured_width_preference"
                    },
                    profile: profile.as_str(),
                    full_cohorts: input.request_count / 2,
                    serial_remainder_requests: input.request_count % 2,
                    moe_plan: input.moe_plan,
                }
            } else if profile == QwenMoeEconomicsProfile::Unqualified {
                serial("profile_unmeasured", profile.as_str(), input.moe_plan)
            } else {
                serial(
                    if input.request_count >= 2 {
                        "parallel_memory_denied"
                    } else {
                        "no_ready_parallel_work"
                    },
                    profile.as_str(),
                    input.moe_plan,
                )
            }
        }
        Some(
            ModelFamily::Qwen4Exp
            | ModelFamily::DeepSeek4
            | ModelFamily::MuseGlimmer
            | ModelFamily::K2Horizon,
        )
        | None => serial("unsupported_family", "not_evaluated", input.moe_plan),
    }
}

pub(super) fn select_deepseek(input: DeepSeekSelectionInput) -> ExecutionSelection {
    if input.mode != Some(ExecutionModeArg::Auto) {
        return serial("explicit_serial", "deepseek_v4", None);
    }
    if input.stdin {
        return serial("streaming_input", "deepseek_v4", None);
    }
    if input.request_count < 2 {
        return serial("no_ready_parallel_work", "deepseek_v4", None);
    }
    if input.residency_set {
        return serial("queue_scoped_residency", "deepseek_v4", None);
    }
    if !input.two_session_memory_admitted {
        return serial("two_session_memory_denied", "deepseek_v4", None);
    }
    ExecutionSelection {
        selected: SelectedExecution::Concurrency2,
        reason: "supported_family_default",
        profile: "deepseek_v4",
        full_cohorts: input.request_count / 2,
        serial_remainder_requests: input.request_count % 2,
        moe_plan: None,
    }
}

pub(super) fn qwen_acceleration_blocker(
    args: &super::Args,
    explicit: ExplicitCliOptions,
    greedy_gpu_mode: GreedyGpuArgmaxMode,
) -> Option<&'static str> {
    if args.prompt_lookup {
        return Some("prompt_lookup_requested");
    }
    if args.request_stats.is_some() || args.request_stats_jsonl.is_some() {
        return Some("request_sidecar_requested");
    }
    if args.trace_request.is_some() {
        return Some("request_trace_requested");
    }
    if args.cache_prefix_tokens.is_some()
        || explicit.prefix_cache_max_mib
        || explicit.cache_prefix_auto_min_tokens
    {
        return Some("prefix_cache_configuration_requested");
    }
    if args.durable_prefix_cache.is_some()
        || explicit.durable_prefix_cache_max_mib
        || explicit.durable_prefix_cache_max_entry_mib
        || explicit.durable_prefix_cache_min_tokens
    {
        return Some("durable_cache_configuration_requested");
    }
    if greedy_gpu_mode == GreedyGpuArgmaxMode::ExplicitRollback {
        return Some("gpu_greedy_explicitly_disabled");
    }
    None
}

pub(super) fn fixed_prefill_chunk(prefill_chunk: PrefillChunkArg) -> bool {
    matches!(prefill_chunk, PrefillChunkArg::Fixed(_))
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    fn a3b_arch() -> Arch {
        Arch {
            kind: ArchKind::Moe,
            n_layer: 40,
            hidden_size: 2_048,
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
        }
    }

    fn q4_plan() -> MoeBatch16PlanTelemetry {
        MoeBatch16PlanTelemetry {
            q8_batched_gdn_blocks: 30,
            packed_q4_gate_up_blocks: 40,
            per_lane_gdn_blocks: 0,
            per_lane_gate_up_blocks: 0,
            per_lane_iq3_gate_up_blocks: 0,
            per_lane_other_gate_up_blocks: 0,
            head_mode: HeadMode::BatchedQ6,
        }
    }

    fn qwen_input(plan: MoeBatch16PlanTelemetry) -> QwenSelectionInput {
        QwenSelectionInput {
            mode: Some(ExecutionModeArg::Auto),
            family: Some(ModelFamily::Qwen35Moe),
            arch: a3b_arch(),
            request_count: 16,
            all_requests_accelerable: true,
            fixed_prefill_chunk: true,
            acceleration_blocker: None,
            dense_full_cohorts: 0,
            dense_serial_remainders: 0,
            moe_full_cohorts: 1,
            moe_serial_remainders: 0,
            fixed_cohort_economics_rejected: false,
            concurrency2_memory_admitted: true,
            dense_batch8_memory_admitted: true,
            moe_batch16_memory_admitted: true,
            moe_plan: Some(plan),
        }
    }

    #[test]
    fn q4_a3b_prefers_ready_b16_then_b2() {
        let input = qwen_input(q4_plan());
        assert_eq!(select_qwen(input).selected, SelectedExecution::MoeBatch16);
        assert_eq!(select_qwen(input).profile, "a3b_q4_non_mtp");

        let input = QwenSelectionInput {
            moe_full_cohorts: 0,
            moe_serial_remainders: 8,
            request_count: 8,
            ..input
        };
        assert_eq!(select_qwen(input).selected, SelectedExecution::Concurrency2);

        let input = QwenSelectionInput {
            fixed_cohort_economics_rejected: true,
            ..input
        };
        assert_eq!(select_qwen(input).reason, "utilization_narrowed_to_pair");

        let input = QwenSelectionInput {
            moe_full_cohorts: 1,
            moe_batch16_memory_admitted: false,
            ..input
        };
        assert_eq!(select_qwen(input).reason, "memory_narrowed_to_pair");
    }

    #[test]
    fn iq4_a3b_prefers_b2_even_with_ready_b16() {
        let input = qwen_input(MoeBatch16PlanTelemetry {
            packed_q4_gate_up_blocks: 0,
            per_lane_gate_up_blocks: 40,
            per_lane_iq3_gate_up_blocks: 40,
            ..q4_plan()
        });
        let selected = select_qwen(input);
        assert_eq!(selected.selected, SelectedExecution::Concurrency2);
        assert_eq!(selected.profile, "a3b_iq4_non_mtp");

        let input = QwenSelectionInput {
            fixed_cohort_economics_rejected: true,
            ..input
        };
        assert_eq!(select_qwen(input).reason, "measured_width_preference");
    }

    #[test]
    fn mtp_and_unknown_compositions_remain_serial() {
        let mut input = qwen_input(q4_plan());
        input.arch.mtp_n_hidden_layers = 1;
        assert_eq!(select_qwen(input).selected, SelectedExecution::Serial);

        let input = QwenSelectionInput {
            moe_plan: None,
            ..qwen_input(q4_plan())
        };
        assert_eq!(select_qwen(input).selected, SelectedExecution::Serial);
    }

    #[test]
    fn dense_uses_ready_b8_or_pair_fallback() {
        let mut arch = a3b_arch();
        arch.kind = ArchKind::Dense;
        let input = QwenSelectionInput {
            mode: Some(ExecutionModeArg::Auto),
            family: Some(ModelFamily::Qwen35),
            arch,
            request_count: 9,
            all_requests_accelerable: true,
            fixed_prefill_chunk: true,
            acceleration_blocker: None,
            dense_full_cohorts: 1,
            dense_serial_remainders: 1,
            moe_full_cohorts: 0,
            moe_serial_remainders: 0,
            fixed_cohort_economics_rejected: false,
            concurrency2_memory_admitted: true,
            dense_batch8_memory_admitted: true,
            moe_batch16_memory_admitted: true,
            moe_plan: None,
        };
        assert_eq!(select_qwen(input).selected, SelectedExecution::DenseBatch8);
        assert_eq!(select_qwen(input).serial_remainder_requests, 1);

        let input = QwenSelectionInput {
            dense_full_cohorts: 0,
            dense_serial_remainders: 7,
            request_count: 7,
            ..input
        };
        assert_eq!(select_qwen(input).selected, SelectedExecution::Concurrency2);

        let input = QwenSelectionInput {
            fixed_cohort_economics_rejected: true,
            ..input
        };
        assert_eq!(select_qwen(input).reason, "utilization_narrowed_to_pair");

        let input = QwenSelectionInput {
            dense_full_cohorts: 1,
            dense_batch8_memory_admitted: false,
            ..input
        };
        assert_eq!(select_qwen(input).reason, "memory_narrowed_to_pair");
    }

    #[test]
    fn selector_v3_scopes_dense_refill_fields() {
        let dense = ExecutionSelectionRecord::new(
            Some(ModelFamily::Qwen35),
            Some(32),
            ExecutionSelection {
                selected: SelectedExecution::DenseBatch8,
                reason: "supported_family_default",
                profile: "dense_qwen",
                full_cohorts: 4,
                serial_remainder_requests: 0,
                moe_plan: None,
            },
            Some("automatic_dense_charged"),
            Some("automatic_dense_refill_admitted"),
            Some("automatic_charged_ragged"),
            Some(2),
            Some(32),
        );
        let dense = serde_json::to_value(dense).unwrap();
        assert_eq!(dense["schema_version"], 3);
        assert_eq!(dense["backend"], "execution_selector_v3");
        assert_eq!(dense["refill_policy"], "automatic_charged_ragged");
        assert_eq!(dense["planned_refill_arenas"], 2);
        assert_eq!(dense["planned_refill_requests"], 32);

        let moe_selection = select_qwen(qwen_input(q4_plan()));
        let moe = serde_json::to_value(ExecutionSelectionRecord::new(
            Some(ModelFamily::Qwen35Moe),
            Some(16),
            moe_selection,
            Some("disabled"),
            Some("disabled"),
            None,
            None,
            None,
        ))
        .unwrap();
        for key in [
            "refill_policy",
            "planned_refill_arenas",
            "planned_refill_requests",
        ] {
            assert!(moe.get(key).is_none(), "MoE selector retained {key}");
        }

        let deepseek_selection = select_deepseek(DeepSeekSelectionInput {
            mode: Some(ExecutionModeArg::Auto),
            request_count: 2,
            stdin: false,
            residency_set: false,
            two_session_memory_admitted: true,
        });
        let deepseek = serde_json::to_value(ExecutionSelectionRecord::new(
            Some(ModelFamily::DeepSeek4),
            Some(2),
            deepseek_selection,
            None,
            None,
            None,
            None,
            None,
        ))
        .unwrap();
        for key in [
            "ragged_prompt_policy",
            "ragged_prompt_plan_decision",
            "refill_policy",
            "planned_refill_arenas",
            "planned_refill_requests",
        ] {
            assert!(
                deepseek.get(key).is_none(),
                "DeepSeek selector retained {key}"
            );
        }
    }

    #[test]
    fn deepseek_auto_falls_back_before_residency() {
        let base = DeepSeekSelectionInput {
            mode: Some(ExecutionModeArg::Auto),
            request_count: 3,
            stdin: false,
            residency_set: false,
            two_session_memory_admitted: true,
        };
        assert_eq!(
            select_deepseek(base).selected,
            SelectedExecution::Concurrency2
        );
        assert_eq!(
            select_deepseek(DeepSeekSelectionInput {
                two_session_memory_admitted: false,
                ..base
            })
            .selected,
            SelectedExecution::Serial
        );
        assert_eq!(
            select_deepseek(DeepSeekSelectionInput {
                stdin: true,
                ..base
            })
            .selected,
            SelectedExecution::Serial
        );
    }

    #[test]
    fn incompatible_request_contract_preserves_serial_features() {
        let input = QwenSelectionInput {
            acceleration_blocker: Some("request_sidecar_requested"),
            ..qwen_input(q4_plan())
        };
        let selected = select_qwen(input);
        assert_eq!(selected.selected, SelectedExecution::Serial);
        assert_eq!(selected.reason, "request_sidecar_requested");
    }

    #[test]
    fn memory_denial_narrows_before_falling_back_to_serial() {
        let input = QwenSelectionInput {
            moe_batch16_memory_admitted: false,
            ..qwen_input(q4_plan())
        };
        let selected = select_qwen(input);
        assert_eq!(selected.selected, SelectedExecution::Concurrency2);
        assert_eq!(selected.reason, "memory_narrowed_to_pair");

        let input = QwenSelectionInput {
            concurrency2_memory_admitted: false,
            ..input
        };
        let selected = select_qwen(input);
        assert_eq!(selected.selected, SelectedExecution::Serial);
        assert_eq!(selected.reason, "parallel_memory_denied");
    }

    #[test]
    fn cli_mode_is_opt_in_and_conflicts_with_explicit_widths() {
        let default = super::super::Args::try_parse_from([
            "qwen",
            "--model",
            "model.gguf",
            "--requests-jsonl",
            "requests.jsonl",
        ])
        .unwrap();
        assert_eq!(default.execution_mode, None);

        let automatic = super::super::Args::try_parse_from([
            "qwen",
            "--model",
            "model.gguf",
            "--requests-jsonl",
            "requests.jsonl",
            "--execution-mode",
            "auto",
        ])
        .unwrap();
        assert_eq!(automatic.execution_mode, Some(ExecutionModeArg::Auto));

        assert!(
            super::super::Args::try_parse_from([
                "qwen",
                "--model",
                "model.gguf",
                "--requests-jsonl",
                "requests.jsonl",
                "--execution-mode",
                "auto",
                "--concurrency",
                "2",
            ])
            .is_err()
        );
        assert!(
            super::super::Args::try_parse_from([
                "qwen",
                "--model",
                "model.gguf",
                "--execution-mode",
                "auto",
            ])
            .is_err()
        );
    }
}
