import ast
import hashlib
import json
import re
import statistics
from pathlib import Path


LOG = Path(__file__).with_name("attention-split.log")
EXPECTED_LOG_SHA256 = "3fd96776530bd90903513aa790488cf314a81ce5a1ccfe96158b9d18a589278b"
STAGE_NAMES = [
    "BeforeAttentionBody",
    "AttentionBody",
    "AttentionOutputProjections",
    "AfterAttentionOutput",
]
COHORT_NAMES = ["SlidingWindow", "CompressedSparse", "HeavilyCompressed"]
ORIGINAL_FLOOR_MS = 150.0
CURRENT_FLOOR_MS = 158.3
SHARE_FLOOR = 0.15
COHORT_SHARE_FLOOR = 0.10
STAGE_REPEAT_LIMIT = 0.02
COHORT_REPEAT_LIMIT = 0.03


def named_array(line: str, name: str) -> list[float]:
    match = re.search(rf"{name}=(\[[^\]]*\])", line)
    if match is None:
        raise ValueError(f"missing {name}")
    return ast.literal_eval(match.group(1))


def sample_record(lines: list[str], index: int) -> dict[str, object]:
    marker = f"packed_attention_split_profile_sample index={index} "
    line = next(line for line in lines if marker in line)
    stage_match = re.search(r"stage_ms=(\[[^\]]*\])", line)
    cohort_match = re.search(r"cohort_stage_ms=(\[\[.*?\]\]) cohort_command_ms=", line)
    command_match = re.search(r"cohort_command_ms=(\[[^\]]*\])", line)
    if stage_match is None or cohort_match is None or command_match is None:
        raise ValueError(f"sample {index} is incomplete")
    return {
        "stage_ms": ast.literal_eval(stage_match.group(1)),
        "cohort_stage_ms": ast.literal_eval(cohort_match.group(1)),
        "cohort_command_ms": ast.literal_eval(command_match.group(1)),
    }


def analyze_stage(
    stage_index: int,
    samples: list[dict[str, object]],
    sampled_gpu_ms: list[float],
    interpolated_control_gpu_ms: list[float],
    ordinary_gpu_median_ms: float,
    common_uncertainty: float,
) -> dict[str, object]:
    raw_ms = [samples[index]["stage_ms"][stage_index] for index in range(2)]
    shares = [raw_ms[index] / sampled_gpu_ms[index] for index in range(2)]
    repeat_delta = abs(shares[0] - shares[1])
    observer_uncertainty = max(common_uncertainty, repeat_delta)
    normalized_ms = [
        shares[index] * interpolated_control_gpu_ms[index] for index in range(2)
    ]
    normalized_median_ms = statistics.median(normalized_ms)
    lower_share = max(0.0, statistics.mean(shares) - observer_uncertainty)
    lower_ms = max(
        0.0,
        normalized_median_ms - observer_uncertainty * ordinary_gpu_median_ms,
    )
    cohort_shares = [
        [
            samples[index]["cohort_stage_ms"][cohort][stage_index]
            / samples[index]["cohort_command_ms"][cohort]
            for cohort in range(3)
        ]
        for index in range(2)
    ]
    mean_cohort_shares = [
        statistics.mean(cohort_shares[index][cohort] for index in range(2))
        for cohort in range(3)
    ]
    cohort_repeat_delta = [
        abs(cohort_shares[0][cohort] - cohort_shares[1][cohort]) for cohort in range(3)
    ]
    authorized = (
        lower_share >= SHARE_FLOOR
        and lower_ms >= CURRENT_FLOOR_MS
        and repeat_delta <= STAGE_REPEAT_LIMIT
        and mean_cohort_shares[1] >= COHORT_SHARE_FLOOR
        and mean_cohort_shares[2] >= COHORT_SHARE_FLOOR
        and cohort_repeat_delta[1] <= COHORT_REPEAT_LIMIT
        and cohort_repeat_delta[2] <= COHORT_REPEAT_LIMIT
    )
    return {
        "raw_ms": raw_ms,
        "shares": shares,
        "repeat_delta": repeat_delta,
        "observer_uncertainty": observer_uncertainty,
        "normalized_ms": normalized_ms,
        "normalized_median_ms": normalized_median_ms,
        "lower_share": lower_share,
        "lower_ms": lower_ms,
        "cohort_shares": {
            COHORT_NAMES[cohort]: [cohort_shares[index][cohort] for index in range(2)]
            for cohort in range(3)
        },
        "mean_cohort_shares": {
            COHORT_NAMES[cohort]: mean_cohort_shares[cohort] for cohort in range(3)
        },
        "cohort_repeat_delta": {
            COHORT_NAMES[cohort]: cohort_repeat_delta[cohort] for cohort in range(3)
        },
        "authorized_for_decomposition": authorized,
    }


payload = LOG.read_bytes()
log_sha256 = hashlib.sha256(payload).hexdigest()
if log_sha256 != EXPECTED_LOG_SHA256:
    raise ValueError(f"unexpected log SHA-256 {log_sha256}")

lines = payload.decode().splitlines()
summary = next(line for line in lines if "packed_attention_split_profile n=" in line)
control_gpu_ms = named_array(summary, "control_gpu_ms")
interpolated_control_gpu_ms = named_array(summary, "interpolated_control_gpu_ms")
sampled_gpu_ms = named_array(summary, "sampled_gpu_ms")
sampled_perturbation = named_array(summary, "sampled_perturbation")
transition_match = re.search(r"transition_uncertainty=([0-9.]+)", summary)
if transition_match is None:
    raise ValueError("missing transition uncertainty")
transition_uncertainty = float(transition_match.group(1))
topology_uncertainty = max(abs(value) for value in sampled_perturbation)
common_uncertainty = max(transition_uncertainty, topology_uncertainty)
ordinary_gpu_median_ms = statistics.median(control_gpu_ms)
samples = [sample_record(lines, index) for index in range(2)]

stages = {
    STAGE_NAMES[index]: analyze_stage(
        index,
        samples,
        sampled_gpu_ms,
        interpolated_control_gpu_ms,
        ordinary_gpu_median_ms,
        common_uncertainty,
    )
    for index in [0, 3]
}

expected = {
    "BeforeAttentionBody": (490.732109210340, 477.569940299229, True),
    "AfterAttentionOutput": (65.995094718050, 56.583301633612, False),
}
for name, (median_ms, lower_ms, authorized) in expected.items():
    result = stages[name]
    if abs(result["normalized_median_ms"] - median_ms) > 1e-9:
        raise ValueError(f"{name} normalized median changed")
    if abs(result["lower_ms"] - lower_ms) > 1e-9:
        raise ValueError(f"{name} lower bound changed")
    if result["authorized_for_decomposition"] is not authorized:
        raise ValueError(f"{name} decision changed")

print(
    json.dumps(
        {
            "log_sha256": log_sha256,
            "control_gpu_ms": control_gpu_ms,
            "interpolated_control_gpu_ms": interpolated_control_gpu_ms,
            "sampled_gpu_ms": sampled_gpu_ms,
            "ordinary_gpu_median_ms": ordinary_gpu_median_ms,
            "transition_uncertainty": transition_uncertainty,
            "topology_uncertainty": topology_uncertainty,
            "common_uncertainty": common_uncertainty,
            "gates": {
                "original_floor_ms": ORIGINAL_FLOOR_MS,
                "current_floor_ms": CURRENT_FLOOR_MS,
                "share_floor": SHARE_FLOOR,
                "csa_hca_share_floor": COHORT_SHARE_FLOOR,
                "stage_repeat_limit": STAGE_REPEAT_LIMIT,
                "csa_hca_repeat_limit": COHORT_REPEAT_LIMIT,
            },
            "stages": stages,
        },
        indent=2,
        sort_keys=True,
    )
)
