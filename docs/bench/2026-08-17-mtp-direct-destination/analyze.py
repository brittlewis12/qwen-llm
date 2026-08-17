#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# ///

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
import struct
from datetime import UTC, datetime, timedelta
from pathlib import Path


MODEL = (
    "/Users/tito/models/unsloth-Qwen3.6-35B-A3B-MTP-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"
)
PROMPT = "The quick brown fox jumps over the lazy dog"
PROMPT_IDS = [760, 3841, 13477, 37550, 33075, 888, 279, 15217, 5388]
EXPECTED_BANKS = {
    "policy": "F32",
    "gate_dtype": "F32",
    "up_dtype": "F32",
    "down_dtype": "F32",
    "bytes": 3_221_225_472,
}
EXPECTED_FEATURES = {
    "packed_base_prefill": True,
    "q5_k_n2_seq": True,
    "iq2_s_n2_nc2": True,
    "iq3_s_n2_nc2": True,
    "skip_final_checkpoint": True,
    "shared_kv_q2_requested": False,
    "shared_kv_q2_min_position": 16_384,
}
LAZY_SEMANTICS = {
    "verify_mode": "lazy_mtp1",
    "sampler": "greedy_argmax",
    "correction_accounting": "deferred_next_step_carry",
    "equivalence": "target_greedy_sequence",
}
PACKED_SEMANTICS = {
    "verify_mode": "packed_n",
    "sampler": "greedy_argmax",
    "correction_accounting": "deferred_next_step_carry",
    "equivalence": "target_greedy_sequence_and_terminal_resume_audit",
}
EXPECTED_ORACLE_BANKS = [
    ("blk.40.ffn_gate_exps.weight", "Q4_K", [2048, 512, 256]),
    ("blk.40.ffn_up_exps.weight", "Q4_K", [2048, 512, 256]),
    ("blk.40.ffn_down_exps.weight", "Q5_K", [512, 2048, 256]),
]
ARM_SCHEDULE = [
    (1, 1, "ab", "control", False),
    (2, 1, "ab", "direct", True),
    (3, 2, "ba", "direct", True),
    (4, 2, "ba", "control", False),
    (5, 3, "ab", "control", False),
    (6, 3, "ab", "direct", True),
    (7, 4, "ba", "direct", True),
    (8, 4, "ba", "control", False),
    (9, 5, "ab", "control", False),
    (10, 5, "ab", "direct", True),
    (11, 6, "ba", "direct", True),
    (12, 6, "ba", "control", False),
]
GLOBAL_SCHEDULE = [
    (1, "build-info-before", None, None, None, None, 0, "build-info-before.json"),
    (2, "packed-control", None, None, "control", False, 1, "packed-control.json"),
    (3, "byte-oracle", None, None, None, None, 0, "byte-oracle.json"),
    (
        4,
        "build-info-after-oracle",
        None,
        None,
        None,
        None,
        0,
        "build-info-after-oracle.json",
    ),
]
GLOBAL_SCHEDULE.extend(
    (
        4 + campaign_sequence,
        "arm",
        pair,
        order,
        arm,
        direct,
        0,
        f"pair{pair:02d}-{order}-{arm}.json",
    )
    for campaign_sequence, pair, order, arm, direct in ARM_SCHEDULE
)
GLOBAL_SCHEDULE.append(
    (
        17,
        "build-info-after-campaign",
        None,
        None,
        None,
        None,
        0,
        "build-info-after-campaign.json",
    )
)
NAME_RE = re.compile(
    r"pair(?P<pair>\d+)-(?P<order>ab|ba)-(?P<arm>control|direct)\.json"
)
DIGEST_RE = re.compile(r"[0-9a-f]{64}")
SOURCE_STATE_RE = re.compile(r"git-source-sha256-v2:[0-9a-f]{64}")


def median(values: list[float | int]) -> float:
    return float(statistics.median(values))


def token_digest(values: list[int]) -> str:
    payload = b"".join(struct.pack("<i", value) for value in values)
    return hashlib.sha256(payload).hexdigest()


def parse_utc(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)


def parse_time(path: Path) -> dict[str, float | int]:
    text = path.read_text()
    real = re.search(r"^\s*([0-9.]+) real", text, re.MULTILINE)
    rss = re.search(r"^\s*(\d+)\s+maximum resident set size$", text, re.MULTILINE)
    footprint = re.search(r"^\s*(\d+)\s+peak memory footprint$", text, re.MULTILINE)
    if not (real and rss and footprint):
        raise ValueError(f"incomplete time packet: {path}")
    return {
        "real_s": float(real.group(1)),
        "max_rss_bytes": int(rss.group(1)),
        "peak_footprint_bytes": int(footprint.group(1)),
    }


def validate_build_identity(identity: dict) -> tuple[str, str]:
    if identity["status"] != "dirty" or identity["problems"] != ["dirty"]:
        raise ValueError(f"invalid build identity status: {identity}")
    if identity["overrides"] != ["allow_dirty"]:
        raise ValueError(f"invalid build identity override: {identity}")
    if identity["build_dirty"] is not True or identity["runtime_dirty"] is not True:
        raise ValueError(f"invalid dirty-state provenance: {identity}")
    if identity["build_commit"] != identity["runtime_commit"]:
        raise ValueError(f"build/runtime commit mismatch: {identity}")
    if identity["build_source_state"] != identity["runtime_source_state"]:
        raise ValueError(f"build/runtime source mismatch: {identity}")
    source_state = identity["build_source_state"]
    if not SOURCE_STATE_RE.fullmatch(source_state):
        raise ValueError(f"invalid source state: {source_state}")
    return identity["build_commit"], source_state


def load_chronology(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        rows = list(reader)
    expected_fields = [
        "event_sequence",
        "event",
        "pair",
        "order",
        "arm",
        "direct",
        "started_utc",
        "ended_utc",
        "exit_status",
        "artifact",
    ]
    if reader.fieldnames != expected_fields or len(rows) != len(GLOBAL_SCHEDULE):
        raise ValueError("chronology schema or event count mismatch")

    normalized = []
    previous_end = None
    for raw, expected in zip(rows, GLOBAL_SCHEDULE, strict=True):
        if raw["direct"] not in {"", "0", "1"}:
            raise ValueError(f"invalid chronology treatment value: {raw}")
        row = {
            "event_sequence": int(raw["event_sequence"]),
            "event": raw["event"],
            "pair": int(raw["pair"]) if raw["pair"] else None,
            "order": raw["order"] or None,
            "arm": raw["arm"] or None,
            "direct": None if raw["direct"] == "" else raw["direct"] == "1",
            "started_utc": raw["started_utc"],
            "ended_utc": raw["ended_utc"],
            "exit_status": int(raw["exit_status"]),
            "artifact": raw["artifact"],
        }
        actual = (
            row["event_sequence"],
            row["event"],
            row["pair"],
            row["order"],
            row["arm"],
            row["direct"],
            row["exit_status"],
            row["artifact"],
        )
        if actual != expected:
            raise ValueError(f"chronology mismatch: {actual} != {expected}")
        started = parse_utc(row["started_utc"])
        ended = parse_utc(row["ended_utc"])
        if ended < started or (previous_end is not None and started < previous_end):
            raise ValueError(f"non-serial chronology timestamps: {row}")
        previous_end = ended
        normalized.append(row)
    return normalized


def validate_prompt_and_tokens(packet: dict) -> str:
    if packet["model"] != MODEL or packet["prompt"] != PROMPT:
        raise ValueError("model or prompt mismatch")
    if (
        packet["qwen_chat"]
        or packet["system"] is not None
        or packet["disable_thinking"]
    ):
        raise ValueError("chat rendering controls drifted")
    if packet["prompt_tokens"] != len(PROMPT_IDS):
        raise ValueError("prompt token count mismatch")
    if packet["prompt_token_sha256"] != token_digest(PROMPT_IDS):
        raise ValueError("prompt token digest mismatch")
    fixture = packet["token_fixture"]
    if fixture["schema_version"] != 1 or fixture["prompt_token_ids"] != PROMPT_IDS:
        raise ValueError("prompt token fixture mismatch")
    target_ids = fixture["target_generated_token_ids"]
    target_sha256 = token_digest(target_ids)
    if packet["target_generated_token_sha256"] != target_sha256:
        raise ValueError("target token digest does not authenticate token fixture")
    return target_sha256


def validate_common_mtp(packet: dict, direct: bool) -> None:
    if packet["generated_requested"] != 16 or packet["spec_tokens"] != 1:
        raise ValueError("generation shape mismatch")
    if packet["probe"] != "Normal" or not packet["no_warmup"]:
        raise ValueError("probe or warmup mismatch")
    if packet["stop_tokens"] != [248_046]:
        raise ValueError("stop-token set mismatch")
    if (
        packet["base_hidden"] != "PostNorm"
        or packet["recursive_hidden"] != "PostNorm"
        or packet["mtp_history"] != "Committed"
    ):
        raise ValueError("MTP state contract mismatch")
    if (
        packet["single_cb_draft"]
        or packet["draft_token_embd_head"]
        or packet["draft_lm_head_q4_1"]
        or packet["draft_lm_head_q4_0"]
        or packet["draft_lm_head_q4_affine64"]
        or packet["rank_topk"] is not None
    ):
        raise ValueError("draft override leaked into campaign")
    expected_features = dict(EXPECTED_FEATURES)
    expected_features["direct_mtp_f32_destination"] = direct
    if packet["execution_features"] != expected_features:
        raise ValueError(f"execution feature mismatch: {packet['execution_features']}")
    expected_env = {
        "QWEN_MTP_DIRECT_F32_DEST": str(int(direct)),
        "QWEN_MTP_MOE_NATIVE_BANKS": "0",
    }
    if packet["qwen_env"] != expected_env:
        raise ValueError(f"QWEN environment mismatch: {packet['qwen_env']}")
    if packet["mtp_moe_banks"] != EXPECTED_BANKS:
        raise ValueError("MTP bank ledger mismatch")
    if not packet["identical"]:
        raise ValueError("target stream mismatch")
    if packet["reference"]["emitted"] != 16 or packet["speculative"]["emitted"] != 16:
        raise ValueError("emitted token count mismatch")
    if not math.isfinite(packet["mtp_load_ms"]) or packet["mtp_load_ms"] <= 0:
        raise ValueError("invalid MTP load timing")


def validate_arm(packet: dict, direct: bool) -> tuple[str, str, str]:
    target_digest = validate_prompt_and_tokens(packet)
    validate_common_mtp(packet, direct)
    if (
        packet["logical_verify_n"] is not None
        or packet["physical_verify_n"] is not None
    ):
        raise ValueError("packed verification leaked into lazy-D1 campaign")
    if packet["semantics"] != LAZY_SEMANTICS:
        raise ValueError("lazy-D1 semantic contract mismatch")
    if packet["speculative"]["target_state"] is not None:
        raise ValueError("unexpected terminal-state assay in lazy-D1 campaign")
    commit, source_state = validate_build_identity(packet["build_identity"])
    return commit, source_state, target_digest


def load_byte_oracle(path: Path) -> tuple[dict, bool, str, str]:
    packet = json.loads(path.read_text())
    if set(packet) != {
        "schema",
        "fixture",
        "test_passed",
        "staged_equals_direct",
        "source_identity",
        "recorded_unix_ms",
        "test_binary_sha256",
        "banks",
    }:
        raise ValueError("byte-oracle schema fields mismatch")
    if packet["schema"] != "qwen-mtp-direct-f32-byte-oracle/v2":
        raise ValueError("byte-oracle schema mismatch")
    if packet["fixture"] != MODEL or packet["test_passed"] is not True:
        raise ValueError("byte-oracle fixture or test status mismatch")
    if (
        not isinstance(packet["recorded_unix_ms"], int)
        or packet["recorded_unix_ms"] <= 0
    ):
        raise ValueError("byte-oracle timestamp mismatch")
    if not DIGEST_RE.fullmatch(packet["test_binary_sha256"]):
        raise ValueError("byte-oracle test binary identity mismatch")
    if len(packet["banks"]) != len(EXPECTED_ORACLE_BANKS):
        raise ValueError("byte-oracle bank count mismatch")

    equal = True
    for bank, expected in zip(packet["banks"], EXPECTED_ORACLE_BANKS, strict=True):
        name, source_dtype, shape = expected
        if set(bank) != {
            "name",
            "source_dtype",
            "shape",
            "bytes",
            "staged_blake3",
            "direct_blake3",
        }:
            raise ValueError(f"byte-oracle bank schema mismatch: {bank}")
        if (
            bank["name"] != name
            or bank["source_dtype"] != source_dtype
            or bank["shape"] != shape
            or bank["bytes"] != 1_073_741_824
        ):
            raise ValueError(f"byte-oracle bank geometry mismatch: {bank}")
        staged = bank["staged_blake3"]
        direct = bank["direct_blake3"]
        if not DIGEST_RE.fullmatch(staged) or not DIGEST_RE.fullmatch(direct):
            raise ValueError(f"invalid byte-oracle digest: {bank}")
        equal &= staged == direct
    if packet["staged_equals_direct"] is not equal:
        raise ValueError("byte-oracle aggregate equality mismatch")
    commit, source_state = validate_build_identity(packet["source_identity"])
    return packet, equal, commit, source_state


def validate_excluded_control(path: Path) -> tuple[dict, str, str, str]:
    if path.name != "packed-control.json":
        raise ValueError("excluded control artifact name mismatch")
    packet = json.loads(path.read_text())
    target_digest = validate_prompt_and_tokens(packet)
    validate_common_mtp(packet, False)
    target_state = packet["speculative"]["target_state"]
    if (
        packet["logical_verify_n"] != 2
        or packet["physical_verify_n"] != 2
        or packet["semantics"] != PACKED_SEMANTICS
        or target_state is None
        or target_state["resume_audit_pass"] is not False
        or target_state["continuation_steps"] != 16
        or target_state["kv_payload_cosine"] >= 0.99999
        or target_state["continuation_logits_max_abs"] <= 5e-2
        or target_state["gdn_state_max_abs"] <= 1e-2
        or target_state["continuation_argmax_equal"] is not True
    ):
        raise ValueError("excluded packed-N2 control assay mismatch")
    commit, source_state = validate_build_identity(packet["build_identity"])
    excluded = {
        "reason": "staged packed-N2 baseline failed its preexisting terminal-state assay",
        "time": parse_time(path.with_suffix(".time")),
        "packet": packet,
    }
    return excluded, commit, source_state, target_digest


def load_arms(root: Path, chronology: list[dict]) -> list[dict]:
    paths = list(root.glob("pair*.json"))
    if len(paths) != len(ARM_SCHEDULE):
        raise ValueError(f"expected 12 arm packets, found {len(paths)}")
    by_key = {}
    for path in paths:
        match = NAME_RE.fullmatch(path.name)
        if match is None:
            raise ValueError(f"unexpected arm packet name: {path.name}")
        key = (int(match.group("pair")), match.group("arm"))
        if key in by_key:
            raise ValueError(f"duplicate arm packet: {key}")
        by_key[key] = (match.group("order"), path)

    arm_events = [row for row in chronology if row["event"] == "arm"]
    arms = []
    for campaign_sequence, event in enumerate(arm_events, start=1):
        key = (event["pair"], event["arm"])
        if key not in by_key:
            raise ValueError(f"chronology arm has no packet: {key}")
        order, path = by_key.pop(key)
        if order != event["order"] or path.name != event["artifact"]:
            raise ValueError(f"arm packet/chronology mismatch: {path}")
        packet = json.loads(path.read_text())
        validate_arm(packet, bool(event["direct"]))
        arms.append(
            {
                "event_sequence": event["event_sequence"],
                "campaign_sequence": campaign_sequence,
                "pair": event["pair"],
                "order": event["order"],
                "arm": event["arm"],
                "time": parse_time(path.with_suffix(".time")),
                "packet": packet,
            }
        )
    if by_key:
        raise ValueError(f"packets absent from chronology: {sorted(by_key)}")
    return arms


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--byte-oracle", type=Path, required=True)
    parser.add_argument("--invalid-packed-control", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", args.expected_commit):
        raise ValueError("expected commit must be a full lowercase object id")

    chronology = load_chronology(args.manifest)
    build_before = json.loads((args.input / "build-info-before.json").read_text())
    build_after_oracle = json.loads(
        (args.input / "build-info-after-oracle.json").read_text()
    )
    build_after_campaign = json.loads(
        (args.input / "build-info-after-campaign.json").read_text()
    )
    validate_build_identity(build_before)
    if build_after_oracle != build_before or build_after_campaign != build_before:
        raise ValueError("source identity drifted across chronological campaign")

    byte_oracle, byte_oracle_equal, oracle_commit, oracle_source_state = (
        load_byte_oracle(args.byte_oracle)
    )
    if byte_oracle["source_identity"] != build_before:
        raise ValueError("byte oracle is not bound to campaign build identity")
    oracle_event = next(row for row in chronology if row["event"] == "byte-oracle")
    oracle_recorded = datetime.fromtimestamp(
        byte_oracle["recorded_unix_ms"] / 1000, tz=UTC
    )
    oracle_started = parse_utc(oracle_event["started_utc"])
    oracle_ended_exclusive = parse_utc(oracle_event["ended_utc"]) + timedelta(seconds=1)
    if not (oracle_started <= oracle_recorded < oracle_ended_exclusive):
        raise ValueError("byte-oracle packet timestamp lies outside its event")

    excluded, excluded_commit, excluded_source_state, excluded_target_digest = (
        validate_excluded_control(args.invalid_packed_control)
    )
    arms = load_arms(args.input, chronology)
    campaign_identity = arms[0]["packet"]["build_identity"]
    campaign_commit, campaign_source_state = validate_build_identity(campaign_identity)
    target_digest = arms[0]["packet"]["target_generated_token_sha256"]
    if not all(arm["packet"]["build_identity"] == build_before for arm in arms):
        raise ValueError("arm build identities differ from chronological build packet")
    if not all(
        arm["packet"]["target_generated_token_sha256"] == target_digest for arm in arms
    ):
        raise ValueError("target token stream drifted across arms")
    if (
        campaign_identity != build_before
        or campaign_commit != args.expected_commit
        or oracle_commit != campaign_commit
        or excluded_commit != campaign_commit
        or oracle_source_state != campaign_source_state
        or excluded_source_state != campaign_source_state
        or excluded_target_digest != target_digest
    ):
        raise ValueError(
            "oracle, excluded control, and admitted arms are not source-bound"
        )

    by_pair: dict[int, dict[str, dict]] = {}
    for arm in arms:
        by_pair.setdefault(arm["pair"], {})[arm["arm"]] = arm
    if sorted(by_pair) != list(range(1, 7)) or any(
        set(pair) != {"control", "direct"} for pair in by_pair.values()
    ):
        raise ValueError("incomplete pair ledger")

    pairs = []
    for pair_id, pair_arms in sorted(by_pair.items()):
        control = pair_arms["control"]
        direct = pair_arms["direct"]
        if control["order"] != direct["order"]:
            raise ValueError(f"pair order mismatch: {pair_id}")
        pairs.append(
            {
                "pair": pair_id,
                "order": control["order"],
                "mtp_load_saving_ms": control["packet"]["mtp_load_ms"]
                - direct["packet"]["mtp_load_ms"],
                "max_rss_deleted_bytes": control["time"]["max_rss_bytes"]
                - direct["time"]["max_rss_bytes"],
                "peak_footprint_deleted_bytes": control["time"]["peak_footprint_bytes"]
                - direct["time"]["peak_footprint_bytes"],
                "whole_process_wall_saving_s": control["time"]["real_s"]
                - direct["time"]["real_s"],
            }
        )

    control_arms = [arm for arm in arms if arm["arm"] == "control"]
    direct_arms = [arm for arm in arms if arm["arm"] == "direct"]
    control_load = [arm["packet"]["mtp_load_ms"] for arm in control_arms]
    direct_load = [arm["packet"]["mtp_load_ms"] for arm in direct_arms]
    load_savings = [pair["mtp_load_saving_ms"] for pair in pairs]
    rss_deletions = [pair["max_rss_deleted_bytes"] for pair in pairs]
    ab_savings = [pair["mtp_load_saving_ms"] for pair in pairs if pair["order"] == "ab"]
    ba_savings = [pair["mtp_load_saving_ms"] for pair in pairs if pair["order"] == "ba"]
    gates = {
        "full_byte_oracle_equal": byte_oracle_equal,
        "chronological_source_binding_valid": True,
        "full_protocol_manifest_valid": True,
        "all_semantics_environments_and_ledgers_equal": True,
        "candidate_wins_at_least_5_of_6": sum(value > 0 for value in load_savings) >= 5,
        "ab_median_load_saving_positive": median(ab_savings) > 0,
        "ba_median_load_saving_positive": median(ba_savings) > 0,
        "overall_median_load_saving_positive": median(load_savings) > 0,
        "median_rss_deletion_at_least_512_mib": median(rss_deletions)
        >= 512 * 1024 * 1024,
    }

    summary = {
        "control_mtp_load_ms": control_load,
        "direct_mtp_load_ms": direct_load,
        "control_median_mtp_load_ms": median(control_load),
        "direct_median_mtp_load_ms": median(direct_load),
        "median_paired_mtp_load_saving_ms": median(load_savings),
        "median_speedup": median(control_load) / median(direct_load),
        "paired_saving_over_control_median": median(load_savings)
        / median(control_load),
        "marginal_median_load_reduction_fraction": 1
        - median(direct_load) / median(control_load),
        "candidate_wins": sum(value > 0 for value in load_savings),
        "pairs": len(pairs),
        "ab_median_mtp_load_saving_ms": median(ab_savings),
        "ba_median_mtp_load_saving_ms": median(ba_savings),
        "control_median_max_rss_bytes": median(
            [arm["time"]["max_rss_bytes"] for arm in control_arms]
        ),
        "direct_median_max_rss_bytes": median(
            [arm["time"]["max_rss_bytes"] for arm in direct_arms]
        ),
        "median_max_rss_deleted_bytes": median(rss_deletions),
        "control_median_peak_footprint_bytes": median(
            [arm["time"]["peak_footprint_bytes"] for arm in control_arms]
        ),
        "direct_median_peak_footprint_bytes": median(
            [arm["time"]["peak_footprint_bytes"] for arm in direct_arms]
        ),
        "median_peak_footprint_deleted_bytes": median(
            [pair["peak_footprint_deleted_bytes"] for pair in pairs]
        ),
        "control_median_whole_process_wall_s": median(
            [arm["time"]["real_s"] for arm in control_arms]
        ),
        "direct_median_whole_process_wall_s": median(
            [arm["time"]["real_s"] for arm in direct_arms]
        ),
        "median_whole_process_wall_saving_s": median(
            [pair["whole_process_wall_saving_s"] for pair in pairs]
        ),
    }
    result = {
        "schema": "qwen-mtp-direct-f32-destination/v3",
        "protocol": {
            "model": MODEL,
            "prompt": PROMPT,
            "prompt_token_ids": PROMPT_IDS,
            "prompt_token_sha256": token_digest(PROMPT_IDS),
            "generated_requested": 16,
            "spec_tokens": 1,
            "probe": "Normal",
            "verify_mode": "lazy_mtp1",
            "no_warmup": True,
            "benchmarked_commit": args.expected_commit,
            "arm_schedule": [
                {
                    "campaign_sequence": sequence,
                    "pair": pair,
                    "order": order,
                    "arm": arm,
                    "direct": direct,
                }
                for sequence, pair, order, arm, direct in ARM_SCHEDULE
            ],
        },
        "source_identity": {
            "commit": campaign_commit,
            "source_state": campaign_source_state,
        },
        "target_generated_token_sha256": target_digest,
        "bank_ledger": EXPECTED_BANKS,
        "byte_oracle": byte_oracle,
        "execution_chronology": chronology,
        "arms": arms,
        "pairs": pairs,
        "summary": summary,
        "gates": gates,
        "disposition": "GO" if all(gates.values()) else "KILL",
        "excluded_setup_samples": [excluded],
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps({"gates": gates, "disposition": result["disposition"]}, indent=2))


if __name__ == "__main__":
    main()
