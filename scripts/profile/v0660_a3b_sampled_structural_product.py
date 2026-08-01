#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shlex
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/profile/v0660_a3b_sampled_structural_product.py"
PREREG = ROOT / "docs/bench/v0660-a3b-sampled-structural-product.md"
MODEL = Path("/Users/tito/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")
PROMPT = ROOT / "docs/bench/tokenizer-prompts/current-reva-n8-interactive-qwen36.txt"
QWEN = ROOT / "target/release/qwen"
QWEN_BENCH = ROOT / "target/release/qwen-bench"
PACKET = ROOT / "target/profiles/v0660-a3b-sampled-structural-product-p1"

PREDECESSOR_PREREG = (
    ROOT / "docs/bench/v0659-positive-temperature-sampling-parser-repair.md"
)
PREDECESSOR_SCRIPT = (
    ROOT / "scripts/profile/v0659_positive_temperature_sampling_parser_repair.py"
)
PREDECESSOR_DECISION = (
    ROOT
    / "target/profiles/v0659-positive-temperature-sampling-parser-repair-p1/decision.json"
)

SUCCESS_SCHEMA = "qwen-v0660-a3b-sampled-structural-product/v1"
FAILURE_SCHEMA = "qwen-v0660-a3b-sampled-structural-product-failure/v1"
PREREG_SHA256 = "2716f29180e076a3c0598050015906b99e0dc87c50ac1178c7ec36f51ea7e983"
PREDECESSOR_PREREG_SHA256 = (
    "89a15f05ff7cc87b8fe22270522350e2c883fd86f45b6f1db5d01f17b9730af1"
)
PREDECESSOR_SCRIPT_SHA256 = (
    "adde8b1ca275f9fa66795959bfe9f5eef72b292d07fa3586227dfd10456e57d5"
)
PREDECESSOR_DECISION_SHA256 = (
    "ae21f8944eae07d950127961f29b6e2814efbc6970af60994fee175655af8534"
)
MODEL_BYTES = 22_134_528_992
MODEL_SHA256 = "ac0e2c1189e055faa36eff361580e79c5bd6f8e76bffb4ce547f167d53e31a61"
PROMPT_BYTES = 1_891
PROMPT_SHA256 = "e265de9742d1b22e566fc108ae26331ccf46166c6f071e73f211e0a1a7e8b474"
PROMPT_TOKENS = 419
PROMPT_TOKEN_SHA256 = "fb4bbb4dc66ca7d219099e2974e787ef976f80789cde3e48b8a905dceece1f9f"
RUNTIME_TOKENIZER_ID = "a4b0b26f8a8c9917"
PAIR_ORDERS = ("AB", "BA", "BA", "AB", "AB", "BA")
COOLDOWN_SECONDS = 5.0
CHILD_TIMEOUT_SECONDS = 30 * 60
CPU_TEST_FLOORS = {"cpu-sampling": 17, "cpu-cli": 58}
MODEL_CONFORMANCE_MARKER = (
    "[sampled-structural-a3b] exact rows/state PASS "
    f"prompt_token_sha256={PROMPT_TOKEN_SHA256}"
)

CHILD_ENV_ALLOWLIST = {
    "CARGO_HOME",
    "COMMAND_MODE",
    "DEVELOPER_DIR",
    "HOME",
    "LANG",
    "LOGNAME",
    "MACOSX_DEPLOYMENT_TARGET",
    "PATH",
    "RUSTUP_HOME",
    "SDKROOT",
    "SHELL",
    "TERM",
    "TMPDIR",
    "USER",
}
LOGGING_CONTROLS = {
    "DEBUG",
    "LOG_LEVEL",
    "RUST_BACKTRACE",
    "RUST_LOG",
    "RUST_LOG_STYLE",
}
KNOWN_COMPETITORS = {
    "llama-bench",
    "llama-cli",
    "llama-server",
    "mlx_lm",
    "ollama",
    "qwen",
    "qwen-bench",
}
METAL_BENCHMARK_MARKERS = (
    "gpu-bench",
    "gpu_bench",
    "metal-bench",
    "metal_bench",
    "metal-capture",
    "metal_capture",
)
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
RUNTIME_ID_RE = re.compile(r"[0-9a-f]{16}\Z")
CPU_IDLE_RE = re.compile(r"CPU usage:.*?([0-9]+(?:\.[0-9]+)?)% idle")
MEMORY_RE = re.compile(r"System-wide memory free percentage:\s*([0-9]+)%")
SWAP_USED_RE = re.compile(r"used\s*=\s*([0-9.]+)([KMG])")
TEST_RESULT_RE = re.compile(
    r"^test result: ok\. ([0-9]+) passed; ([0-9]+) failed; "
    r"([0-9]+) ignored; ([0-9]+) measured; ([0-9]+) filtered out; "
    r"finished in [^\r\n]+$",
    re.MULTILINE,
)
PSO_PHASE_KEYS = {"prefill", "generation", "total"}
PSO_METRIC_KEYS = {"misses", "miss_wall_ns", "compiler_wall_ns"}
METAL_SAMPLE_KEYS = {
    "current_bytes",
    "delta_from_model_ready_bytes",
    "delta_from_request_start_bytes",
}
METAL_SAMPLE_NAMES = {
    "process_model_ready",
    "request_start",
    "after_scratch",
    "after_sequence",
    "after_prefill",
    "after_first_stdout_flush",
    "request_end_before_state_drop",
    "after_request_state_drop",
}
METAL_ALLOCATED_KEYS = METAL_SAMPLE_NAMES | {"current_allocated_sampled_max_bytes"}

SCHEMA10_KEYS = {
    "schema_version",
    "request_epoch",
    "request_index",
    "tokenizer_reused",
    "pair_requested",
    "pair_id",
    "pair_request_equal",
    "pair_generated_tokens_equal",
    "prefix_cache_used",
    "build_commit",
    "build_dirty",
    "build_source_state",
    "model",
    "runtime_identity_kind",
    "runtime_model_id",
    "runtime_tokenizer_id",
    "greedy_gpu_selection_reason",
    "request_start_unix_ms",
    "runtime_and_model_load_ms",
    "stdout_sink",
    "ttft_endpoint",
    "prompt_source",
    "prompt_bytes",
    "prompt_tokens",
    "requested_tokens",
    "generated_tokens",
    "generated_token_sha256",
    "stop_reason",
    "decode_policy",
    "sampling",
    "terminal_token_target_transition_consumed",
    "no_special_tokens",
    "prefill_chunk_requested",
    "prefill_chunk_effective",
    "max_context_tokens",
    "prompt_acquisition_ms",
    "tokenizer_init_ms",
    "tokenization_ms",
    "capacity_validation_ms",
    "scratch_allocation_ms",
    "sequence_allocation_ms",
    "prefill_ms",
    "first_token_selection_ms",
    "first_token_callback_duration_ms",
    "first_token_ready_ms",
    "ttft_ms",
    "generation_ms",
    "transition_count",
    "transition_ms",
    "transition_tps",
    "inference_complete_ms",
    "total_request_ms",
    "pso_cache",
    "metal_allocated",
}
STRUCTURAL_KEYS = {
    "version",
    "algorithm_version",
    "path",
    "prompt_owned_bounded_calls",
    "borrowed_transition_calls",
    "resident_head_wait_calls",
    "validated_shared_row_calls",
    "fallback_calls",
    "input_logits_total",
    "input_logits_min",
    "input_logits_max",
    "retained_top_k_total",
    "retained_top_k_min",
    "retained_top_k_max",
    "max_heap_len",
    "max_heap_capacity",
    "full_candidate_vector_allocations",
    "transition_logits_copy_bytes",
    "extra_command_buffers",
    "gpu_sampling_dispatches",
}
SAMPLING_KEYS = {
    "algorithm_version",
    "temperature",
    "top_k",
    "top_p",
    "min_p",
    "effective_seed",
    "draws",
}
EQUALITY_FIELDS = (
    "request_epoch",
    "request_index",
    "tokenizer_reused",
    "pair_requested",
    "pair_id",
    "pair_request_equal",
    "pair_generated_tokens_equal",
    "prefix_cache_used",
    "build_commit",
    "build_dirty",
    "build_source_state",
    "model",
    "runtime_identity_kind",
    "runtime_model_id",
    "runtime_tokenizer_id",
    "greedy_gpu_selection_reason",
    "stdout_sink",
    "ttft_endpoint",
    "prompt_source",
    "prompt_bytes",
    "prompt_tokens",
    "requested_tokens",
    "generated_tokens",
    "generated_token_sha256",
    "stop_reason",
    "decode_policy",
    "sampling",
    "terminal_token_target_transition_consumed",
    "no_special_tokens",
    "prefill_chunk_requested",
    "prefill_chunk_effective",
    "max_context_tokens",
    "transition_count",
)


class PacketFailure(RuntimeError):
    def __init__(self, message: str, *, kind: str) -> None:
        super().__init__(message)
        self.kind = kind


class TerminalPublicationFailure(PacketFailure):
    def __init__(
        self,
        message: str,
        *,
        decision_published: bool,
        failure_alias_published: bool,
    ) -> None:
        super().__init__(message, kind="infrastructure")
        self.decision_published = decision_published
        self.failure_alias_published = failure_alias_published


def require(condition: bool, message: str, *, kind: str = "candidate") -> None:
    if not condition:
        raise PacketFailure(message, kind=kind)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()


def complete_write(descriptor: int, payload: bytes) -> None:
    view = memoryview(payload)
    written = 0
    while written < len(view):
        count = os.write(descriptor, view[written:])
        if count <= 0:
            raise OSError("write made no forward progress")
        written += count


def write_new_fsynced(path: Path, payload: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        complete_write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def publish_terminal(payload: bytes, *, failure: bool) -> None:
    decision = PACKET / "decision.json"
    failure_path = PACKET / "failure.json"
    require(
        not decision.exists() and not failure_path.exists(),
        "refusing contradictory terminal state",
        kind="infrastructure",
    )
    decision_temporary = decision.with_name(f".{decision.name}.terminal-{os.getpid()}")
    failure_temporary = failure_path.with_name(
        f".{failure_path.name}.terminal-{os.getpid()}"
    )
    temporaries = [decision_temporary] + ([failure_temporary] if failure else [])
    require(
        not any(path.exists() for path in temporaries),
        "stale terminal temporary exists",
        kind="infrastructure",
    )
    decision_published = False
    failure_alias_published = False
    try:
        write_new_fsynced(decision_temporary, payload)
        require(
            decision_temporary.stat().st_size == len(payload)
            and sha256_file(decision_temporary) == sha256_bytes(payload),
            "staged decision payload verification failed",
            kind="infrastructure",
        )
        if failure:
            os.link(decision_temporary, failure_temporary)
            require(
                failure_temporary.stat().st_size == len(payload)
                and sha256_file(failure_temporary) == sha256_bytes(payload),
                "staged failure alias verification failed",
                kind="infrastructure",
            )
        fsync_directory(PACKET)
        os.replace(decision_temporary, decision)
        decision_published = True
        if failure:
            os.replace(failure_temporary, failure_path)
            failure_alias_published = True
        fsync_directory(PACKET)
    except BaseException as error:
        if not decision_published:
            for temporary in temporaries:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass
        raise TerminalPublicationFailure(
            f"terminal publication failed: {type(error).__name__}; "
            f"decision_published={decision_published}; "
            f"failure_alias_published={failure_alias_published}",
            decision_published=decision_published,
            failure_alias_published=failure_alias_published,
        ) from error


def write_json(path: Path, value: Any, *, replace: bool = False) -> None:
    payload = json_bytes(value)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    require(
        not temporary.exists(),
        f"stale temporary JSON {temporary}",
        kind="infrastructure",
    )
    if not replace:
        require(
            not path.exists(), f"artifact already exists: {path}", kind="infrastructure"
        )
    try:
        write_new_fsynced(temporary, payload)
        os.replace(temporary, path)
        fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def append_journal(value: dict[str, Any]) -> None:
    payload = (
        json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":")) + "\n"
    )
    path = PACKET / "journal.jsonl"
    created = not path.exists()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        complete_write(descriptor, payload.encode())
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    if created:
        fsync_directory(PACKET)


def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def reject_constant(value: str) -> None:
    raise PacketFailure(f"non-finite JSON constant {value!r}", kind="candidate")


def parse_finite_float(value: str) -> float:
    try:
        parsed = float(value)
    except (ValueError, OverflowError) as error:
        raise PacketFailure(
            f"invalid JSON float {value!r}", kind="candidate"
        ) from error
    require(math.isfinite(parsed), f"non-finite JSON float {value!r}")
    return parsed


def parse_json(text: str) -> Any:
    try:
        return json.loads(
            text,
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_constant,
            parse_float=parse_finite_float,
        )
    except PacketFailure:
        raise
    except (json.JSONDecodeError, ValueError, OverflowError) as error:
        raise PacketFailure(
            f"malformed JSON: {type(error).__name__}", kind="candidate"
        ) from error


def parse_json_bytes(payload: bytes, label: str) -> Any:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise PacketFailure(f"{label}: invalid UTF-8", kind="candidate") from error
    return parse_json(text)


def read_jsonl_one(path: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    require(payload.endswith(b"\n"), f"{path}: JSONL row lacks newline")
    lines = payload.splitlines()
    require(
        len(lines) == 1 and bool(lines[0]), f"{path}: expected one physical JSONL row"
    )
    value = parse_json_bytes(lines[0], str(path))
    require(isinstance(value, dict), f"{path}: JSONL row is not an object")
    return value


def environment_commitments(environment: dict[str, str]) -> dict[str, dict[str, Any]]:
    return {
        key: {
            "value_bytes": len(value.encode()),
            "value_sha256": sha256_bytes(value.encode()),
        }
        for key, value in sorted(environment.items())
    }


def is_control_name(name: str) -> bool:
    upper = name.upper()
    return (
        upper.startswith(("QWEN", "MTL", "METAL"))
        or upper in LOGGING_CONTROLS
        or (upper.startswith("LC_") and "LOG" in upper)
    )


def child_env() -> dict[str, str]:
    return {
        key: value
        for key, value in os.environ.items()
        if (key in CHILD_ENV_ALLOWLIST or key.startswith("LC_"))
        and not is_control_name(key)
    }


def command_commitment(argv: list[str]) -> dict[str, Any]:
    encoded = b"\0".join(item.encode() for item in argv)
    return {
        "argc": len(argv),
        "argv_bytes": len(encoded),
        "argv_sha256": sha256_bytes(encoded),
    }


def output_commitment(payload: bytes) -> dict[str, Any]:
    return {"bytes": len(payload), "sha256": sha256_bytes(payload)}


def run_text(argv: list[str], *, timeout: int = 120) -> tuple[str, dict[str, Any]]:
    payload, commitment = run_bytes(argv, timeout=timeout)
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise PacketFailure(
            f"helper output invalid UTF-8: command={command_commitment(argv)} "
            f"output={commitment}",
            kind="infrastructure",
        ) from error
    return text, commitment


def run_bytes(argv: list[str], *, timeout: int = 120) -> tuple[bytes, dict[str, Any]]:
    try:
        result = subprocess.run(
            argv,
            cwd=ROOT,
            env=child_env(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        payload = error.output or b""
        raise PacketFailure(
            f"helper timed out: command={command_commitment(argv)} output={output_commitment(payload)}",
            kind="infrastructure",
        ) from error
    require(
        result.returncode == 0,
        f"helper failed: command={command_commitment(argv)} returncode={result.returncode} "
        f"output={output_commitment(result.stdout)}",
        kind="infrastructure",
    )
    return result.stdout, output_commitment(result.stdout)


def run_environment_text(
    argv: list[str], *, timeout: int = 120
) -> tuple[str, dict[str, Any]]:
    try:
        return run_text(argv, timeout=timeout)
    except PacketFailure as error:
        raise PacketFailure(str(error), kind="environment") from error


def git_text(*arguments: str) -> str:
    text, _ = run_text(["git", *arguments])
    return text.strip()


def runner_contract_checks() -> dict[str, bool]:
    duplicate = nonfinite = malformed = overflow = False
    try:
        parse_json('{"x":1,"x":2}')
    except PacketFailure:
        duplicate = True
    try:
        parse_json('{"x":NaN}')
    except PacketFailure:
        nonfinite = True
    try:
        parse_json('{"x":}')
    except PacketFailure:
        malformed = True
    try:
        parse_json('{"x":1e9999}')
    except PacketFailure:
        overflow = True
    require(
        duplicate and nonfinite and malformed and overflow,
        "strict JSON self-test failed",
    )
    committed = environment_commitments({"TOKEN": "plaintext-secret"})
    require(
        "plaintext-secret" not in json.dumps(committed), "environment redaction failed"
    )
    require(
        not any(is_control_name(key) for key in child_env()), "child controls survived"
    )
    require(PAIR_ORDERS == ("AB", "BA", "BA", "AB", "AB", "BA"), "pair order changed")
    sample = [9.0, 1.0, 3.0, 7.0, 5.0, 11.0]
    require(median_six(sample) == 6.0, "even-six median changed")
    require(
        disposition_for("infrastructure", False) == "CONSUMED_NO_AUTHORITY"
        and disposition_for("candidate", False) == "KILL"
        and disposition_for("environment", True) == "INVALID"
        and disposition_for("candidate", True) == "KILL",
        "terminal disposition partition changed",
    )
    a = product_argv(Path("unique.jsonl"), "A")
    b = product_argv(Path("unique.jsonl"), "B")
    require(a[0] == "target/release/qwen", "product executable spelling changed")
    require(b == a + ["--sampled-structural"], "B argv is not A plus one switch")
    require(len(SCHEMA10_KEYS) == 54, "schema-10 key count changed")
    require(len(STRUCTURAL_KEYS) == 20, "structural key count changed")
    require(
        len(expected_nonterminal_artifact_names()) == 79, "artifact-name set changed"
    )
    timestamp_bool_rejected = False
    try:
        require_int(True, "self-test timestamp")
    except PacketFailure:
        timestamp_bool_rejected = True
    require(timestamp_bool_rejected, "timestamp validator accepted bool")
    allocation_fixture: dict[str, Any] = {
        "current_allocated_sampled_max_bytes": 180,
    }
    for index, name in enumerate(sorted(METAL_SAMPLE_NAMES)):
        allocation_fixture[name] = {"current_bytes": 110 + index * 10}
    model_ready = allocation_fixture["process_model_ready"]["current_bytes"]
    request_start = allocation_fixture["request_start"]["current_bytes"]
    for name in METAL_SAMPLE_NAMES:
        item = allocation_fixture[name]
        item["delta_from_model_ready_bytes"] = item["current_bytes"] - model_ready
        item["delta_from_request_start_bytes"] = item["current_bytes"] - request_start
    validate_metal_allocated(allocation_fixture)
    require(
        parse_model_conformance_marker(MODEL_CONFORMANCE_MARKER + "\n")
        == PROMPT_TOKEN_SHA256,
        "model marker self-test failed",
    )
    return {
        "duplicate_json_rejected": duplicate,
        "nonfinite_json_rejected": nonfinite,
        "malformed_json_rejected": malformed,
        "float_overflow_rejected": overflow,
        "environment_redaction_checked": True,
        "pairing_and_median_checked": True,
        "dispositions_checked": True,
        "argv_checked": True,
        "schemas_checked": True,
        "strict_timestamp_checked": True,
        "metal_allocation_deltas_checked": True,
        "model_marker_checked": True,
    }


def check_only() -> dict[str, Any]:
    required = (SCRIPT, PREREG, PREDECESSOR_PREREG, PREDECESSOR_SCRIPT, PROMPT)
    require(all(path.is_file() for path in required), "required static file is absent")
    require(sha256_file(PREREG) == PREREG_SHA256, "v0.660 preregistration hash changed")
    require(
        sha256_file(PREDECESSOR_PREREG) == PREDECESSOR_PREREG_SHA256,
        "v0.659 preregistration hash changed",
    )
    require(
        sha256_file(PREDECESSOR_SCRIPT) == PREDECESSOR_SCRIPT_SHA256,
        "v0.659 runner hash changed",
    )
    require(PROMPT.stat().st_size == PROMPT_BYTES, "prompt size changed")
    require(sha256_file(PROMPT) == PROMPT_SHA256, "prompt hash changed")
    tracked = set(
        git_text(
            "ls-files", str(SCRIPT.relative_to(ROOT)), str(PREREG.relative_to(ROOT))
        ).splitlines()
    )
    return {
        "phase": "check-only",
        "packet_root_created": False,
        "model_hashed": False,
        "model_or_gpu_work_run": False,
        "runner_sha256": sha256_file(SCRIPT),
        "prereg_sha256": PREREG_SHA256,
        "predecessor_prereg_sha256": PREDECESSOR_PREREG_SHA256,
        "predecessor_runner_sha256": PREDECESSOR_SCRIPT_SHA256,
        "prompt_sha256": PROMPT_SHA256,
        "runner_tracked": str(SCRIPT.relative_to(ROOT)) in tracked,
        "prereg_tracked": str(PREREG.relative_to(ROOT)) in tracked,
        "release_files_present": {
            "qwen": QWEN.is_file(),
            "qwen_bench": QWEN_BENCH.is_file(),
        },
        "runner_contract_checks": runner_contract_checks(),
    }


def authenticate_authority() -> dict[str, Any]:
    for path in (PREREG, PREDECESSOR_PREREG, PREDECESSOR_SCRIPT, PREDECESSOR_DECISION):
        require(path.is_file(), f"authority input missing: {path}")
    require(
        sha256_file(PREREG) == PREREG_SHA256, "v0.660 preregistration hash mismatch"
    )
    require(
        sha256_file(PREDECESSOR_PREREG) == PREDECESSOR_PREREG_SHA256,
        "v0.659 preregistration hash mismatch",
    )
    require(
        sha256_file(PREDECESSOR_SCRIPT) == PREDECESSOR_SCRIPT_SHA256,
        "v0.659 runner hash mismatch",
    )
    decision_payload = PREDECESSOR_DECISION.read_bytes()
    require(
        sha256_bytes(decision_payload) == PREDECESSOR_DECISION_SHA256,
        "v0.659 canonical decision hash mismatch",
    )
    decision = parse_json_bytes(decision_payload, "v0.659 decision")
    require(isinstance(decision, dict), "v0.659 decision is not an object")
    reduction = decision.get("reduction")
    require(isinstance(reduction, dict), "v0.659 reduction absent")
    require_exact(
        reduction.get("disposition"),
        "GO_BOUNDED_IMPLEMENTATION",
        "authority.disposition",
    )
    require(
        type(reduction.get("authority")) is list
        and len(reduction["authority"]) == 1
        and type(reduction["authority"][0]) is str
        and reduction["authority"][0] == "structural",
        "authority scope changed",
    )
    require_exact(
        reduction.get("max_authorized_packets"), 1, "authority.max_authorized_packets"
    )
    return {
        "v0660_prereg_sha256": PREREG_SHA256,
        "v0659_prereg_sha256": PREDECESSOR_PREREG_SHA256,
        "v0659_runner_sha256": PREDECESSOR_SCRIPT_SHA256,
        "v0659_decision_sha256": PREDECESSOR_DECISION_SHA256,
        "disposition": "GO_BOUNDED_IMPLEMENTATION",
        "authority": ["structural"],
        "max_authorized_packets": 1,
        "predecessor_artifacts_read": [
            str(PREDECESSOR_PREREG.relative_to(ROOT)),
            str(PREDECESSOR_SCRIPT.relative_to(ROOT)),
            str(PREDECESSOR_DECISION.relative_to(ROOT)),
        ],
    }


def static_acquire_checks() -> dict[str, Any]:
    qwen_controls = sorted(key for key in os.environ if key.startswith("QWEN_"))
    require(
        not qwen_controls,
        f"inherited QWEN_* controls: names={qwen_controls}",
        kind="environment",
    )
    for path in (SCRIPT, PREREG, PROMPT, MODEL, QWEN, QWEN_BENCH):
        require(path.is_file(), f"required acquisition file absent: {path}")
    require(MODEL.stat().st_size == MODEL_BYTES, "model size mismatch")
    require(PROMPT.stat().st_size == PROMPT_BYTES, "prompt size mismatch")
    require(sha256_file(MODEL) == MODEL_SHA256, "model hash mismatch")
    require(sha256_file(PROMPT) == PROMPT_SHA256, "prompt hash mismatch")
    require(
        sha256_file(PREDECESSOR_PREREG) == PREDECESSOR_PREREG_SHA256,
        "v0.659 preregistration hash mismatch during static capture",
    )
    require(
        sha256_file(PREDECESSOR_SCRIPT) == PREDECESSOR_SCRIPT_SHA256,
        "v0.659 runner hash mismatch during static capture",
    )
    require(
        sha256_file(PREDECESSOR_DECISION) == PREDECESSOR_DECISION_SHA256,
        "v0.659 decision hash mismatch during static capture",
    )
    commit = git_text("rev-parse", "HEAD")
    require(
        not git_text("status", "--porcelain=v1"),
        "acquisition requires a clean worktree",
    )
    for path in (SCRIPT, PREREG):
        relative = str(path.relative_to(ROOT))
        require(
            git_text("ls-files", "--error-unmatch", relative) == relative,
            f"not tracked: {relative}",
        )
        blob, _ = run_bytes(["git", "show", f"HEAD:{relative}"])
        require(
            blob == path.read_bytes(),
            f"working file differs from HEAD: {relative}",
        )
    build_text, build_output = run_text(
        [str(QWEN_BENCH), "build-info", "--output", "json"]
    )
    build = parse_json(build_text)
    require(isinstance(build, dict), "build-info is not an object")
    require(build.get("status") == "match", "build/runtime identity mismatch")
    require(
        build.get("build_commit") == commit and build.get("runtime_commit") == commit,
        "build commit mismatch",
    )
    require(
        build.get("build_dirty") is False and build.get("runtime_dirty") is False,
        "dirty build identity",
    )
    require(
        build.get("build_source_state") == build.get("runtime_source_state"),
        "source-state mismatch",
    )
    require(
        build.get("overrides") == [] and build.get("problems") == [],
        "build identity exceptions",
    )
    model_stat = MODEL.stat()
    return {
        "source_commit": commit,
        "source_tree": git_text("rev-parse", "HEAD^{tree}"),
        "runner_sha256": sha256_file(SCRIPT),
        "prereg_sha256": PREREG_SHA256,
        "predecessor_prereg_sha256": sha256_file(PREDECESSOR_PREREG),
        "predecessor_runner_sha256": sha256_file(PREDECESSOR_SCRIPT),
        "predecessor_decision_sha256": sha256_file(PREDECESSOR_DECISION),
        "prompt_sha256": PROMPT_SHA256,
        "prompt_token_sha256": PROMPT_TOKEN_SHA256,
        "model_sha256": MODEL_SHA256,
        "model_identity": {
            "device": model_stat.st_dev,
            "inode": model_stat.st_ino,
            "bytes": model_stat.st_size,
            "mtime_ns": model_stat.st_mtime_ns,
        },
        "qwen_sha256": sha256_file(QWEN),
        "qwen_bench_sha256": sha256_file(QWEN_BENCH),
        "build_identity": build,
        "build_info_output": build_output,
        "parent_environment": environment_commitments(dict(os.environ)),
        "child_environment": environment_commitments(child_env()),
        "parent_environment_names": sorted(os.environ),
        "child_environment_names": sorted(child_env()),
        "runner_contract_checks": runner_contract_checks(),
    }


def process_census() -> tuple[
    list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]
]:
    text, helper = run_environment_text(["ps", "-axo", "pid=,comm=,args="])
    census: list[dict[str, Any]] = []
    competitors: list[dict[str, Any]] = []
    for line in text.splitlines():
        fields = line.strip().split(None, 2)
        if len(fields) < 2 or not fields[0].isdigit():
            continue
        pid = int(fields[0])
        comm = Path(fields[1]).name
        args = fields[2] if len(fields) == 3 else ""
        try:
            argv = shlex.split(args)
        except ValueError:
            argv = args.split()
        names = {comm, *(Path(token).name for token in argv)}
        observation = {
            "pid": pid,
            "comm": comm,
            "argv0": Path(argv[0]).name if argv else comm,
            "args_bytes": len(args.encode()),
            "args_sha256": sha256_bytes(args.encode()),
        }
        census.append(observation)
        python_mlx = any(
            token == "-m"
            and index + 1 < len(argv)
            and argv[index + 1].startswith("mlx_lm")
            for index, token in enumerate(argv)
        )
        metal_bench = any(
            marker in name.lower()
            for name in names
            for marker in METAL_BENCHMARK_MARKERS
        )
        if pid != os.getpid() and (
            bool(names & KNOWN_COMPETITORS)
            or python_mlx
            or metal_bench
            or any(name.startswith(("qwen-", "llama-")) for name in names)
        ):
            competitors.append(observation)
    return census, competitors, helper


def competitor_capture(label: str) -> dict[str, Any]:
    census, competitors, helper = process_census()
    encoded = json.dumps(census, sort_keys=True, separators=(",", ":")).encode()
    result = {
        "label": label,
        "process_census": census,
        "process_census_sha256": sha256_bytes(encoded),
        "competitors": competitors,
        "helper_output": helper,
    }
    return result


def validate_competitor_capture(result: dict[str, Any]) -> None:
    require(
        not result["competitors"],
        f"{result['label']}: competing inference process detected",
        kind="environment",
    )


def cpu_idle_sample() -> tuple[float, dict[str, Any]]:
    text, helper = run_environment_text(
        ["top", "-l", "2", "-s", "1", "-n", "0"], timeout=30
    )
    matches = CPU_IDLE_RE.findall(text)
    require(
        len(matches) >= 2, "cannot parse one-second CPU idle sample", kind="environment"
    )
    return float(matches[-1]), helper


def host_gate(label: str, attested: bool) -> dict[str, Any]:
    batt, batt_output = run_environment_text(["pmset", "-g", "batt"])
    therm, therm_output = run_environment_text(["pmset", "-g", "therm"])
    memory, memory_output = run_environment_text(["memory_pressure", "-Q"])
    memory_match = MEMORY_RE.search(memory)
    require(
        memory_match is not None,
        f"{label}: cannot parse memory availability",
        kind="environment",
    )
    idle_pairs = [cpu_idle_sample() for _ in range(3)]
    idle = [item[0] for item in idle_pairs]
    census, competitors, ps_output = process_census()
    result = {
        "label": label,
        "ac_power": "AC Power" in batt,
        "thermal_warning": "No thermal warning level has been recorded" not in therm,
        "performance_warning": "No performance warning level has been recorded"
        not in therm,
        "memory_available_percent": int(memory_match.group(1)),
        "cpu_idle_samples_percent": idle,
        "cpu_idle_median_percent": statistics.median(idle),
        "competitors": competitors,
        "process_census": census,
        "operator_no_other_user_gpu_attestation": attested,
        "helper_outputs": {
            "pmset_batt": batt_output,
            "pmset_therm": therm_output,
            "memory_pressure": memory_output,
            "cpu_idle": [item[1] for item in idle_pairs],
            "ps": ps_output,
        },
    }
    return result


def validate_host_gate(result: dict[str, Any]) -> None:
    label = result["label"]
    require(result["ac_power"], f"{label}: AC power required", kind="environment")
    require(
        not result["thermal_warning"], f"{label}: thermal warning", kind="environment"
    )
    require(
        not result["performance_warning"],
        f"{label}: performance warning",
        kind="environment",
    )
    require(
        result["memory_available_percent"] >= 50,
        f"{label}: memory below 50%",
        kind="environment",
    )
    require(
        result["cpu_idle_median_percent"] >= 75.0,
        f"{label}: CPU idle below 75%",
        kind="environment",
    )
    require(
        not result["competitors"],
        f"{label}: competing inference process",
        kind="environment",
    )
    require(
        result["operator_no_other_user_gpu_attestation"],
        f"{label}: GPU attestation absent",
        kind="environment",
    )


def parse_vm_stat(text: str) -> dict[str, int]:
    lines = text.splitlines()
    require(bool(lines), "empty vm_stat output", kind="environment")
    page = re.search(r"page size of ([0-9]+) bytes", lines[0])
    require(page is not None, "cannot parse vm_stat page size", kind="environment")
    values = {"page_size": int(page.group(1))}
    for line in lines[1:]:
        if ":" not in line:
            continue
        key, raw = line.split(":", 1)
        canonical = raw.strip().rstrip(".").replace(".", "")
        if canonical.isdigit():
            values[key.strip()] = int(canonical)
    return values


def vm_snapshot(label: str) -> dict[str, Any]:
    vm_text, vm_output = run_environment_text(["vm_stat"])
    swap_text, swap_output = run_environment_text(["sysctl", "-n", "vm.swapusage"])
    vm = parse_vm_stat(vm_text)
    swap = SWAP_USED_RE.search(swap_text)
    require(swap is not None, "cannot parse swap occupancy", kind="environment")
    required = {
        "Pageouts": "pageouts",
        "Compressions": "compressions",
        "Swapouts": "swapouts",
        "Pages stored in compressor": "compressor_stored_pages",
        "Pages occupied by compressor": "compressor_occupied_pages",
    }
    result: dict[str, Any] = {
        "label": label,
        "helper_outputs": {"vm_stat": vm_output, "swapusage": swap_output},
    }
    for source, target in required.items():
        require(source in vm, f"vm_stat missing {source}", kind="environment")
        result[target] = vm[source]
    scale = {"K": 1024, "M": 1024**2, "G": 1024**3}[swap.group(2)]
    result["swap_occupancy_bytes"] = round(float(swap.group(1)) * scale)
    result["page_size"] = vm["page_size"]
    return result


def vm_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    names = (
        "swap_occupancy_bytes",
        "compressor_stored_pages",
        "compressor_occupied_pages",
        "pageouts",
        "compressions",
        "swapouts",
    )
    deltas = {name: after[name] - before[name] for name in names}
    require(
        all(deltas[name] >= 0 for name in ("pageouts", "compressions", "swapouts")),
        "VM cumulative counter regressed",
        kind="environment",
    )
    fatal = [
        name
        for name in (
            "swap_occupancy_bytes",
            "compressor_stored_pages",
            "compressor_occupied_pages",
        )
        if deltas[name] > 0
    ]
    return {
        "deltas": deltas,
        "fatal_occupancy_growth": fatal,
        "advisory_growth": {
            name: deltas[name] for name in ("pageouts", "compressions", "swapouts")
        },
    }


def product_argv(timing: Path, arm: str) -> list[str]:
    require(arm in ("A", "B"), f"unknown arm {arm}")
    argv = [
        "target/release/qwen",
        "--model",
        str(MODEL),
        "--prompt-file",
        str(PROMPT.relative_to(ROOT)),
        "--tokens",
        "128",
        "--temp",
        "0.7",
        "--top-k",
        "200",
        "--top-p",
        "1.0",
        "--min-p",
        "0.05",
        "--seed",
        "42",
        "--prefill-chunk",
        "1024",
        "--max-context-tokens",
        "1024",
        "--prefix-cache-max-mib",
        "0",
        "--cache-prefix-auto-min-tokens",
        "0",
        "--request-timings",
        str(timing),
    ]
    if arm == "B":
        argv.append("--sampled-structural")
    return argv


def process_group_exists(group: int) -> bool:
    try:
        os.killpg(group, 0)
        return True
    except ProcessLookupError:
        return False


def wait_group_absent(group: int, seconds: float) -> bool:
    deadline = time.monotonic() + seconds
    while process_group_exists(group):
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.05)
    return True


def terminate_exact_group(
    process: subprocess.Popen[bytes], group: int, label: str, *, group_verified: bool
) -> tuple[list[dict[str, Any]], list[str], bool]:
    records: list[dict[str, Any]] = []
    errors: list[str] = []

    def record(event: str, **fields: Any) -> bool:
        item = {
            "event": f"cleanup-{event}",
            "label": label,
            "pid": process.pid,
            "process_group": group,
            "monotonic_ns": time.monotonic_ns(),
            **fields,
        }
        records.append(item)
        try:
            append_journal(item)
            item["journaled"] = True
            return True
        except BaseException as error:
            item["journaled"] = False
            item["journal_error_type"] = type(error).__name__
            errors.append(f"journal-{event}:{type(error).__name__}")
            return False

    def terminate_pid_fallback(reason: str) -> bool:
        errors.append(f"group-unverified:{reason}")
        intent_journaled = record("pid-fallback-intent", action="TERM", reason=reason)
        try:
            process.terminate()
            record(
                "pid-fallback-action",
                action="TERM",
                status="sent",
                intent_journaled=intent_journaled,
            )
        except ProcessLookupError:
            record(
                "pid-fallback-action",
                action="TERM",
                status="already-absent",
                intent_journaled=intent_journaled,
            )
        except BaseException as error:
            errors.append(f"pid-TERM:{type(error).__name__}")
            record(
                "pid-fallback-action",
                action="TERM",
                status="error",
                error_type=type(error).__name__,
                intent_journaled=intent_journaled,
            )
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            kill_intent_journaled = record(
                "pid-fallback-intent", action="KILL", reason="TERM-timeout"
            )
            try:
                process.kill()
                record(
                    "pid-fallback-action",
                    action="KILL",
                    status="sent",
                    intent_journaled=kill_intent_journaled,
                )
            except ProcessLookupError:
                record(
                    "pid-fallback-action",
                    action="KILL",
                    status="already-absent",
                    intent_journaled=kill_intent_journaled,
                )
            except BaseException as error:
                errors.append(f"pid-KILL:{type(error).__name__}")
                record(
                    "pid-fallback-action",
                    action="KILL",
                    status="error",
                    error_type=type(error).__name__,
                    intent_journaled=kill_intent_journaled,
                )
            try:
                process.wait(timeout=10)
            except BaseException as error:
                errors.append(f"pid-KILL-wait:{type(error).__name__}")
        except BaseException as error:
            errors.append(f"pid-TERM-wait:{type(error).__name__}")
        child_live = process.poll() is None
        try:
            unverified_group_live = process_group_exists(group)
        except BaseException as error:
            errors.append(f"unverified-group-check:{type(error).__name__}")
            unverified_group_live = True
        contained = not child_live and not unverified_group_live
        record(
            "pid-fallback-complete",
            child_live=child_live,
            unverified_group_live=unverified_group_live,
            contained=contained,
        )
        return contained

    if not group_verified:
        try:
            if os.getpgid(process.pid) != group:
                record("refused", reason="unexpected-process-group")
                contained = terminate_pid_fallback("unexpected-process-group")
                return records, errors, contained
            group_verified = True
            record("validation-result", status="verified")
        except BaseException as error:
            record("validation-error", error_type=type(error).__name__)
            contained = terminate_pid_fallback(f"validation-{type(error).__name__}")
            return records, errors, contained

    def signal_group(name: str, number: signal.Signals) -> None:
        intent_journaled = record("signal-intent", signal=name)
        try:
            os.killpg(group, number)
            record(
                "signal-action",
                signal=name,
                status="sent",
                intent_journaled=intent_journaled,
            )
        except ProcessLookupError:
            record(
                "signal-action",
                signal=name,
                status="already-absent",
                intent_journaled=intent_journaled,
            )
        except BaseException as error:
            errors.append(f"{name}:{type(error).__name__}")
            record(
                "signal-action",
                signal=name,
                status="error",
                error_type=type(error).__name__,
                intent_journaled=intent_journaled,
            )

    try:
        exists = process_group_exists(group)
    except BaseException as error:
        exists = True
        errors.append(f"pre-TERM-check:{type(error).__name__}")
        record("check-error", phase="pre-TERM", error_type=type(error).__name__)
    if exists:
        signal_group("TERM", signal.SIGTERM)
    record("wait-intent", phase="after-TERM", timeout_seconds=10)
    try:
        process.wait(timeout=10)
        record("wait-result", phase="after-TERM", status="reaped")
    except subprocess.TimeoutExpired:
        record("wait-result", phase="after-TERM", status="timeout")
    except BaseException as error:
        errors.append(f"TERM-wait:{type(error).__name__}")
        record(
            "wait-result",
            phase="after-TERM",
            status="error",
            error_type=type(error).__name__,
        )
    try:
        absent = wait_group_absent(group, 2.0)
    except BaseException as error:
        absent = False
        errors.append(f"post-TERM-check:{type(error).__name__}")
        record("check-error", phase="post-TERM", error_type=type(error).__name__)
    if not absent:
        signal_group("KILL", signal.SIGKILL)
        record("wait-intent", phase="after-KILL", timeout_seconds=10)
        try:
            process.wait(timeout=10)
            record("wait-result", phase="after-KILL", status="reaped")
        except BaseException as error:
            errors.append(f"KILL-wait:{type(error).__name__}")
            record(
                "wait-result",
                phase="after-KILL",
                status="error",
                error_type=type(error).__name__,
            )
    try:
        survived = not wait_group_absent(group, 2.0)
    except BaseException as error:
        survived = True
        errors.append(f"final-check:{type(error).__name__}")
        record("check-error", phase="final", error_type=type(error).__name__)
    if survived:
        errors.append("process-group-survived")
    record("complete", survived=survived, errors=list(errors))
    return records, errors, not survived and process.poll() is not None


def terminate_exact_child_without_group(
    process: subprocess.Popen[bytes], label: str
) -> tuple[list[dict[str, Any]], list[str], bool]:
    records: list[dict[str, Any]] = []
    errors: list[str] = []

    def record(event: str, **fields: Any) -> bool:
        item = {
            "event": f"cleanup-{event}",
            "label": label,
            "pid": process.pid,
            "process_group": None,
            "monotonic_ns": time.monotonic_ns(),
            **fields,
        }
        records.append(item)
        try:
            append_journal(item)
            item["journaled"] = True
            return True
        except BaseException as error:
            item["journaled"] = False
            item["journal_error_type"] = type(error).__name__
            errors.append(f"journal-{event}:{type(error).__name__}")
            return False

    for action, method in (("TERM", process.terminate), ("KILL", process.kill)):
        if process.poll() is not None:
            break
        intent_journaled = record(
            "pid-only-intent", action=action, reason="group-not-recorded"
        )
        try:
            method()
            record(
                "pid-only-action",
                action=action,
                status="sent",
                intent_journaled=intent_journaled,
            )
        except ProcessLookupError:
            record(
                "pid-only-action",
                action=action,
                status="already-absent",
                intent_journaled=intent_journaled,
            )
        except BaseException as error:
            errors.append(f"pid-only-{action}:{type(error).__name__}")
            record(
                "pid-only-action",
                action=action,
                status="error",
                error_type=type(error).__name__,
                intent_journaled=intent_journaled,
            )
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            continue
        except BaseException as error:
            errors.append(f"pid-only-{action}-wait:{type(error).__name__}")
    contained = process.poll() is not None
    record("pid-only-complete", contained=contained, errors=list(errors))
    if not contained:
        errors.append("exact-child-still-live")
    return records, errors, contained


def file_identity(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"artifact absent: {path}", kind="infrastructure")
    return {
        "name": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def run_attempt(
    label: str,
    argv: list[str],
    stdout: Path,
    stderr: Path,
    extra_artifacts: tuple[Path, ...] = (),
    *,
    pair_index: int | None = None,
    orientation: str | None = None,
    position: int | None = None,
    arm: str | None = None,
    expose_b: Callable[[], None] | None = None,
    frozen_argv: bool = False,
) -> dict[str, Any]:
    paths = (stdout, stderr, *extra_artifacts)
    require(
        not any(path.exists() for path in paths),
        f"{label}: child artifact path reused",
        kind="infrastructure",
    )
    launch = {
        "event": "launch-intent",
        "label": label,
        "pair_index": pair_index,
        "orientation": orientation,
        "position": position,
        "arm": arm,
        "unix_ms": time.time_ns() // 1_000_000,
        "executable": argv[0],
        "argv_commitment": command_commitment(argv),
        "environment_names": sorted(child_env()),
        "environment_commitments": environment_commitments(child_env()),
        "artifacts": [path.name for path in paths],
    }
    if frozen_argv:
        launch["argv"] = argv
    append_journal(launch)
    process: subprocess.Popen[bytes] | None = None
    group: int | None = None
    group_verified = False
    timed_out = False
    leaked = False
    actions: list[dict[str, Any]] = []
    cleanup_errors: list[str] = []
    child_contained = True
    start_ns: int | None = None
    end_ns: int | None = None
    spawn_error: BaseException | None = None
    try:
        with (
            stdout.open("xb", buffering=0) as out_handle,
            stderr.open("xb", buffering=0) as err_handle,
        ):
            if arm == "B" and expose_b is not None:
                expose_b()
            start_ns = time.monotonic_ns()
            try:
                process = subprocess.Popen(
                    argv,
                    cwd=ROOT,
                    env=child_env(),
                    stdout=out_handle,
                    stderr=err_handle,
                    start_new_session=True,
                )
                group = process.pid
                require(
                    os.getpgid(process.pid) == group,
                    f"{label}: process group isolation failed",
                    kind="infrastructure",
                )
                group_verified = True
                append_journal(
                    {
                        "event": "spawned",
                        "label": label,
                        "pair_index": pair_index,
                        "orientation": orientation,
                        "position": position,
                        "arm": arm,
                        "pid": process.pid,
                        "process_group": group,
                        "unix_ms": time.time_ns() // 1_000_000,
                    }
                )
                try:
                    process.wait(timeout=CHILD_TIMEOUT_SECONDS)
                    end_ns = time.monotonic_ns()
                except subprocess.TimeoutExpired:
                    timed_out = True
                    actions, cleanup_errors, child_contained = terminate_exact_group(
                        process, group, label, group_verified=group_verified
                    )
                if process_group_exists(group):
                    leaked = True
                    more_actions, more_errors, more_contained = terminate_exact_group(
                        process, group, label, group_verified=group_verified
                    )
                    actions.extend(more_actions)
                    cleanup_errors.extend(more_errors)
                    child_contained = more_contained
            except BaseException as error:
                spawn_error = error
                if process is not None and group is not None:
                    more_actions, more_errors, more_contained = terminate_exact_group(
                        process, group, label, group_verified=group_verified
                    )
                    actions.extend(more_actions)
                    cleanup_errors.extend(more_errors)
                    child_contained = more_contained
            if (
                end_ns is None
                and process is not None
                and process.returncode is not None
                and not timed_out
            ):
                end_ns = time.monotonic_ns()
            os.fsync(out_handle.fileno())
            os.fsync(err_handle.fileno())
    except BaseException as error:
        if spawn_error is None:
            spawn_error = error
        if process is not None and group is not None:
            more_actions, more_errors, more_contained = terminate_exact_group(
                process, group, label, group_verified=group_verified
            )
            actions.extend(more_actions)
            cleanup_errors.extend(more_errors)
            child_contained = more_contained
        elif process is not None:
            more_actions, more_errors, more_contained = (
                terminate_exact_child_without_group(process, label)
            )
            actions.extend(more_actions)
            cleanup_errors.extend(more_errors)
            child_contained = child_contained and more_contained
    if process is not None and process.poll() is None:
        if group is not None:
            more_actions, more_errors, more_contained = terminate_exact_group(
                process, group, label, group_verified=group_verified
            )
        else:
            more_actions, more_errors, more_contained = (
                terminate_exact_child_without_group(process, label)
            )
        actions.extend(more_actions)
        cleanup_errors.extend(more_errors)
        child_contained = more_contained
    artifact_errors: list[str] = []
    for artifact in extra_artifacts:
        if artifact.is_file():
            try:
                with artifact.open("rb", buffering=0) as handle:
                    os.fsync(handle.fileno())
            except OSError as error:
                artifact_errors.append(f"{artifact.name}:fsync:{type(error).__name__}")
        else:
            artifact_errors.append(f"{artifact.name}:absent")
    identities = []
    for path in paths:
        try:
            identities.append(file_identity(path))
        except (OSError, PacketFailure) as error:
            artifact_errors.append(f"{path.name}:identity:{type(error).__name__}")
    try:
        fsync_directory(PACKET)
    except OSError as error:
        artifact_errors.append(f"packet-directory:fsync:{type(error).__name__}")
    record = {
        "event": "completion",
        "label": label,
        "pair_index": pair_index,
        "orientation": orientation,
        "position": position,
        "arm": arm,
        "pid": None if process is None else process.pid,
        "process_group": group,
        "returncode": None if process is None else process.returncode,
        "timed_out": timed_out,
        "leaked_process_group": leaked,
        "termination_actions": actions,
        "cleanup_errors": cleanup_errors,
        "child_contained": child_contained,
        "spawn_to_exit_start_monotonic_ns": start_ns,
        "spawn_to_exit_end_monotonic_ns": end_ns,
        "spawn_to_exit_ms": None
        if start_ns is None or end_ns is None
        else (end_ns - start_ns) / 1e6,
        "artifacts": identities,
        "artifact_errors": artifact_errors,
        "error_type": None if spawn_error is None else type(spawn_error).__name__,
        "unix_ms": time.time_ns() // 1_000_000,
    }
    try:
        append_journal(record)
    except BaseException as error:
        artifact_errors.append(f"completion-journal:{type(error).__name__}")
    require(
        child_contained,
        f"{label}: child containment could not be established",
        kind="live-child",
    )
    require(
        spawn_error is None,
        f"{label}: spawn/wait/I/O failure ({record['error_type']})",
        kind="infrastructure",
    )
    require(not timed_out, f"{label}: child timed out", kind="infrastructure")
    require(not leaked, f"{label}: child leaked process group", kind="infrastructure")
    require(
        not cleanup_errors,
        f"{label}: cleanup errors: {cleanup_errors}",
        kind="infrastructure",
    )
    require(
        process is not None and process.returncode == 0,
        f"{label}: child exit failure",
        kind="candidate",
    )
    require(
        start_ns is not None and end_ns is not None and end_ns >= start_ns,
        f"{label}: invalid spawn clock",
    )
    require(
        not artifact_errors,
        f"{label}: child artifact capture failure: {artifact_errors}",
        kind="infrastructure",
    )
    require(
        len(identities) == len(paths),
        f"{label}: child artifact capture failure",
        kind="infrastructure",
    )
    return record


def parse_test_results(
    stdout: Path,
    stderr: Path,
    label: str,
    floor: int,
    *,
    exact_summary_count: int | None = 1,
) -> dict[str, Any]:
    text = stdout.read_text(encoding="utf-8", errors="strict") + stderr.read_text(
        encoding="utf-8", errors="strict"
    )
    matches = [
        tuple(int(item) for item in match) for match in TEST_RESULT_RE.findall(text)
    ]
    if exact_summary_count is None:
        require(matches, f"{label}: no cargo test summary found")
    else:
        require(
            len(matches) == exact_summary_count,
            f"{label}: expected exactly {exact_summary_count} cargo test summary",
        )
    discovered = sum(sum(row) for row in matches)
    passed = sum(row[0] for row in matches)
    failed = sum(row[1] for row in matches)
    ignored = sum(row[2] for row in matches)
    measured = sum(row[3] for row in matches)
    filtered = sum(row[4] for row in matches)
    require(
        failed == 0 and passed >= floor, f"{label}: test floor or success gate failed"
    )
    return {
        "summaries": len(matches),
        "per_summary": [
            {
                "passed": row[0],
                "failed": row[1],
                "ignored": row[2],
                "measured": row[3],
                "filtered_out": row[4],
            }
            for row in matches
        ],
        "passed": passed,
        "failed": failed,
        "ignored": ignored,
        "measured": measured,
        "filtered_out": filtered,
        "discovered": discovered,
        "required_passing_floor": floor,
    }


def require_int(value: Any, label: str, expected: int | None = None) -> int:
    require(type(value) is int and value >= 0, f"{label}: expected nonnegative integer")
    if expected is not None:
        require(value == expected, f"{label}: {value} != {expected}")
    return value


def require_signed_int(value: Any, label: str) -> int:
    require(type(value) is int, f"{label}: expected exact integer")
    return value


def require_exact(value: Any, expected: Any, label: str) -> None:
    if expected is None:
        require(value is None, f"{label}: expected None")
    else:
        require(
            type(value) is type(expected) and value == expected,
            f"{label}: exact type/value changed",
        )


def require_number(value: Any, label: str, *, positive: bool = False) -> float:
    require(type(value) in (int, float), f"{label}: expected number")
    try:
        number = float(value)
    except (ValueError, OverflowError) as error:
        raise PacketFailure(f"{label}: numeric overflow", kind="candidate") from error
    require(math.isfinite(number), f"{label}: non-finite")
    require(number >= 0.0, f"{label}: negative")
    if positive:
        require(number > 0.0, f"{label}: not positive")
    return number


def validate_all_timing_numbers(value: Any, path: str = "row") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            child = f"{path}.{key}"
            if key.endswith(("_ms", "_tps", "_ns")):
                require_number(item, child)
            validate_all_timing_numbers(item, child)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            validate_all_timing_numbers(item, f"{path}[{index}]")


def validate_pso_cache(value: Any) -> None:
    require(
        type(value) is dict and set(value) == PSO_PHASE_KEYS, "pso_cache keys changed"
    )
    for phase in sorted(PSO_PHASE_KEYS):
        metrics = value[phase]
        require(
            type(metrics) is dict and set(metrics) == PSO_METRIC_KEYS,
            f"pso_cache.{phase} keys changed",
        )
        for key in PSO_METRIC_KEYS:
            require_int(metrics[key], f"pso_cache.{phase}.{key}")


def validate_metal_allocated(value: Any) -> None:
    require(
        type(value) is dict and set(value) == METAL_ALLOCATED_KEYS,
        "metal_allocated keys changed",
    )
    maximum = require_int(
        value["current_allocated_sampled_max_bytes"],
        "metal_allocated.current_allocated_sampled_max_bytes",
    )
    currents = []
    for name in sorted(METAL_SAMPLE_NAMES):
        sample = value[name]
        require(
            type(sample) is dict and set(sample) == METAL_SAMPLE_KEYS,
            f"metal_allocated.{name} keys changed",
        )
        currents.append(
            require_int(
                sample["current_bytes"], f"metal_allocated.{name}.current_bytes"
            )
        )
        require_signed_int(
            sample["delta_from_model_ready_bytes"],
            f"metal_allocated.{name}.delta_from_model_ready_bytes",
        )
        require_signed_int(
            sample["delta_from_request_start_bytes"],
            f"metal_allocated.{name}.delta_from_request_start_bytes",
        )
    require(
        maximum == max(currents), "metal_allocated sampled maximum does not reconcile"
    )
    model_ready = value["process_model_ready"]["current_bytes"]
    request_start = value["request_start"]["current_bytes"]
    for name in sorted(METAL_SAMPLE_NAMES):
        sample = value[name]
        require(
            sample["delta_from_model_ready_bytes"]
            == sample["current_bytes"] - model_ready,
            f"metal_allocated.{name}.delta_from_model_ready_bytes does not reconcile",
        )
        require(
            sample["delta_from_request_start_bytes"]
            == sample["current_bytes"] - request_start,
            f"metal_allocated.{name}.delta_from_request_start_bytes does not reconcile",
        )


def validate_build_and_fixture(row: dict[str, Any], static: dict[str, Any]) -> None:
    build = static["build_identity"]
    require_exact(row.get("build_commit"), build["build_commit"], "row.build_commit")
    require_exact(row.get("build_dirty"), "0", "row.build_dirty")
    require_exact(
        row.get("build_source_state"),
        build["build_source_state"],
        "row.build_source_state",
    )
    require(
        sha256_file(QWEN) == static["qwen_sha256"],
        "qwen binary changed during acquisition",
    )
    require(
        sha256_file(QWEN_BENCH) == static["qwen_bench_sha256"],
        "qwen-bench changed during acquisition",
    )
    stat = MODEL.stat()
    require(
        {
            "device": stat.st_dev,
            "inode": stat.st_ino,
            "bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
        == static["model_identity"],
        "model identity changed during acquisition",
    )


def validate_common_row(row: dict[str, Any], static: dict[str, Any]) -> None:
    validate_build_and_fixture(row, static)
    expected = {
        "request_epoch": "first_post_model_load",
        "request_index": 0,
        "tokenizer_reused": False,
        "pair_requested": False,
        "pair_id": None,
        "pair_request_equal": None,
        "pair_generated_tokens_equal": None,
        "prefix_cache_used": False,
        "model": str(MODEL),
        "runtime_identity_kind": "metadata_compatibility_v1",
        "runtime_tokenizer_id": RUNTIME_TOKENIZER_ID,
        "greedy_gpu_selection_reason": "ineligible_request",
        "stdout_sink": "redirected",
        "ttft_endpoint": "stdout_flush_complete",
        "prompt_source": "file",
        "prompt_bytes": PROMPT_BYTES,
        "prompt_tokens": PROMPT_TOKENS,
        "requested_tokens": 128,
        "generated_tokens": 128,
        "stop_reason": "token_limit",
        "decode_policy": "sampled_cpu",
        "terminal_token_target_transition_consumed": False,
        "no_special_tokens": False,
        "prefill_chunk_requested": 1024,
        "prefill_chunk_effective": PROMPT_TOKENS,
        "max_context_tokens": 1024,
        "transition_count": 127,
    }
    for key, expected_value in expected.items():
        require_exact(row.get(key), expected_value, f"row.{key}")
    require(
        type(row.get("runtime_model_id")) is str
        and RUNTIME_ID_RE.fullmatch(row["runtime_model_id"]) is not None,
        "runtime model identity invalid",
    )
    require(
        type(row.get("generated_token_sha256")) is str
        and SHA256_RE.fullmatch(row["generated_token_sha256"]) is not None,
        "generated token digest invalid",
    )
    sampling = row.get("sampling")
    require(
        type(sampling) is dict and set(sampling) == SAMPLING_KEYS,
        "sampling key set changed",
    )
    require_exact(sampling["algorithm_version"], 1, "sampling.algorithm_version")
    require_exact(sampling["temperature"], 0.7, "sampling.temperature")
    require_exact(sampling["top_k"], 200, "sampling.top_k")
    require_exact(sampling["top_p"], 1.0, "sampling.top_p")
    require_exact(sampling["min_p"], 0.05, "sampling.min_p")
    require_exact(sampling["effective_seed"], 42, "sampling.effective_seed")
    require_exact(sampling["draws"], 128, "sampling.draws")
    require_int(row.get("request_start_unix_ms"), "request_start_unix_ms")
    validate_pso_cache(row.get("pso_cache"))
    validate_metal_allocated(row.get("metal_allocated"))
    validate_all_timing_numbers(row)
    first_selection = require_number(
        row.get("first_token_selection_ms"), "first_token_selection_ms"
    )
    first_ready = require_number(
        row.get("first_token_ready_ms"), "first_token_ready_ms"
    )
    ttft = require_number(row.get("ttft_ms"), "ttft_ms")
    inference = require_number(
        row.get("inference_complete_ms"), "inference_complete_ms"
    )
    total = require_number(row.get("total_request_ms"), "total_request_ms")
    require(
        first_selection <= first_ready <= ttft <= inference <= total,
        "request milestones out of order",
    )
    generation = require_number(
        row.get("generation_ms"), "generation_ms", positive=True
    )
    transition = require_number(
        row.get("transition_ms"), "transition_ms", positive=True
    )
    transition_tps = require_number(
        row.get("transition_tps"), "transition_tps", positive=True
    )
    require(transition <= generation, "transition wall exceeds generation wall")
    expected_tps = 127_000.0 / transition
    require(
        abs(transition_tps - expected_tps) <= max(1e-9, expected_tps * 1e-9),
        "transition_tps does not reconcile",
    )


def validate_structural(structural: Any) -> None:
    require(
        type(structural) is dict and set(structural) == STRUCTURAL_KEYS,
        "structural key set changed",
    )
    for key in STRUCTURAL_KEYS - {"path"}:
        require_int(structural.get(key), f"sampled_structural.{key}")
    require_exact(
        structural.get("path"),
        "bounded_topk_borrowed_transitions",
        "sampled_structural.path",
    )
    expected = {
        "version": 1,
        "algorithm_version": 1,
        "path": "bounded_topk_borrowed_transitions",
        "prompt_owned_bounded_calls": 1,
        "borrowed_transition_calls": 127,
        "resident_head_wait_calls": 127,
        "validated_shared_row_calls": 127,
        "fallback_calls": 0,
        "input_logits_total": 31_784_960,
        "input_logits_min": 248_320,
        "input_logits_max": 248_320,
        "retained_top_k_total": 25_600,
        "retained_top_k_min": 200,
        "retained_top_k_max": 200,
        "max_heap_len": 200,
        "full_candidate_vector_allocations": 0,
        "transition_logits_copy_bytes": 0,
        "extra_command_buffers": 0,
        "gpu_sampling_dispatches": 0,
    }
    for key, value in expected.items():
        require(structural.get(key) == value, f"sampled_structural.{key} changed")
    require(
        200 <= structural["max_heap_capacity"] < 248_320,
        "heap allocator capacity outside contract",
    )


def validate_scored_row(row: dict[str, Any], arm: str, static: dict[str, Any]) -> None:
    require(type(row) is dict, "timing row must be an exact object")
    if arm == "A":
        require(set(row) == SCHEMA10_KEYS, "A schema-10 top-level key set changed")
        require_int(row.get("schema_version"), "A schema_version", 10)
        require(
            "sampled_structural" not in row and "sampling_attribution" not in row,
            "A has candidate telemetry",
        )
    else:
        require(
            set(row) == SCHEMA10_KEYS | {"sampled_structural"},
            "B schema-12 top-level key set changed",
        )
        require_int(row.get("schema_version"), "B schema_version", 12)
        require("sampling_attribution" not in row, "B has sampling attribution")
        validate_structural(row.get("sampled_structural"))
    validate_common_row(row, static)


def run_scored_child(
    pair_index: int,
    orientation: str,
    position: int,
    arm: str,
    static: dict[str, Any],
    expose_b: Callable[[], None],
) -> dict[str, Any]:
    stem = f"pair-{pair_index:02d}-{orientation.lower()}-{position}-{arm.lower()}"
    timing = PACKET / f"{stem}.timing.jsonl"
    stdout = PACKET / f"{stem}.stdout"
    stderr = PACKET / f"{stem}.stderr"
    argv = product_argv(timing, arm)
    require_static_unchanged(static)
    record = run_attempt(
        stem,
        argv,
        stdout,
        stderr,
        (timing,),
        pair_index=pair_index,
        orientation=orientation,
        position=position,
        arm=arm,
        expose_b=expose_b,
        frozen_argv=True,
    )
    row = read_jsonl_one(timing)
    validate_scored_row(row, arm, static)
    record["timing"] = row
    record["stdout_identity"] = file_identity(stdout)
    record["stderr_identity"] = file_identity(stderr)
    record["timing_identity"] = file_identity(timing)
    write_json(PACKET / f"{stem}.json", record)
    return record


def median_six(values: list[float]) -> float:
    require(len(values) == 6, "median requires exactly six values")
    ordered = sorted(values)
    value = (ordered[2] + ordered[3]) / 2.0
    require(math.isfinite(value), "median is non-finite")
    return value


def median_three(values: list[float]) -> float:
    require(len(values) == 3, "stratum requires exactly three values")
    return sorted(values)[1]


def reduce_pairs(children: list[dict[str, Any]]) -> dict[str, Any]:
    require(len(children) == 12, "reduction requires 12 unique children")
    pairs: list[dict[str, Any]] = []
    for pair_index, orientation in enumerate(PAIR_ORDERS, start=1):
        selected = [child for child in children if child["pair_index"] == pair_index]
        selected.sort(key=lambda child: child["position"])
        require(
            len(selected) == 2
            and [child["position"] for child in selected] == [1, 2]
            and all(child["orientation"] == orientation for child in selected)
            and "".join(child["arm"] for child in selected) == orientation,
            "pair incomplete",
        )
        by_arm = {child["arm"]: child for child in selected}
        a = by_arm["A"]
        b = by_arm["B"]
        ar = a["timing"]
        br = b["timing"]
        generation_a = require_number(
            ar["generation_ms"], "A generation", positive=True
        )
        generation_b = require_number(
            br["generation_ms"], "B generation", positive=True
        )
        values = {
            "generation_fraction": (generation_a - generation_b) / generation_a,
            "request_saving_ms": require_number(ar["total_request_ms"], "A request")
            - require_number(br["total_request_ms"], "B request"),
            "ttft_delta_ms": require_number(br["ttft_ms"], "B ttft")
            - require_number(ar["ttft_ms"], "A ttft"),
            "spawn_saving_ms": require_number(a["spawn_to_exit_ms"], "A spawn")
            - require_number(b["spawn_to_exit_ms"], "B spawn"),
        }
        require(
            all(math.isfinite(value) for value in values.values()),
            "paired metric is non-finite",
        )
        pairs.append({"pair_index": pair_index, "orientation": orientation, **values})
    names = (
        "generation_fraction",
        "request_saving_ms",
        "ttft_delta_ms",
        "spawn_saving_ms",
    )
    raw = {name: [pair[name] for pair in pairs] for name in names}
    medians = {name: median_six(raw[name]) for name in names}
    strata = {
        orientation: {
            name: median_three(
                [pair[name] for pair in pairs if pair["orientation"] == orientation]
            )
            for name in names
        }
        for orientation in ("AB", "BA")
    }
    generation_wins = sum(pair["generation_fraction"] > 0 for pair in pairs)
    request_wins = sum(pair["request_saving_ms"] > 0 for pair in pairs)
    gates = {
        "median_generation_fraction_at_least_0_05": medians["generation_fraction"]
        >= 0.05,
        "generation_wins_at_least_5": generation_wins >= 5,
        "median_request_saving_ms_at_least_5": medians["request_saving_ms"] >= 5.0,
        "request_wins_at_least_5": request_wins >= 5,
        "ab_generation_median_positive": strata["AB"]["generation_fraction"] > 0,
        "ba_generation_median_positive": strata["BA"]["generation_fraction"] > 0,
        "ab_request_median_positive": strata["AB"]["request_saving_ms"] > 0,
        "ba_request_median_positive": strata["BA"]["request_saving_ms"] > 0,
        "median_ttft_delta_ms_at_most_10": medians["ttft_delta_ms"] <= 10.0,
        "median_spawn_saving_ms_nonnegative": medians["spawn_saving_ms"] >= 0.0,
    }
    return {
        "pairs": pairs,
        "raw_ordered_values": raw,
        "median_rule": "arithmetic mean of sorted values 3 and 4",
        "medians": medians,
        "strata_medians": strata,
        "generation_wins": generation_wins,
        "request_wins": request_wins,
        "gates": gates,
        "all_performance_gates_pass": all(gates.values()),
    }


def validate_cross_child_equality(children: list[dict[str, Any]]) -> dict[str, Any]:
    require(len(children) == 12, "expected exactly 12 scored children")
    labels = [child["label"] for child in children]
    require(len(set(labels)) == 12, "scored child identity duplicated")
    reference = children[0]
    reference_stdout = (PACKET / f"{reference['label']}.stdout").read_bytes()
    reference_row = reference["timing"]
    for child in children[1:]:
        stdout = (PACKET / f"{child['label']}.stdout").read_bytes()
        require(stdout == reference_stdout, f"{child['label']}: stdout differs")
        row = child["timing"]
        for key in EQUALITY_FIELDS:
            require(
                row.get(key) == reference_row.get(key),
                f"{child['label']}: equality field differs: {key}",
            )
    return {
        "children": labels,
        "stdout": output_commitment(reference_stdout),
        "generated_token_sha256": reference_row["generated_token_sha256"],
        "generated_tokens": 128,
        "transition_count": 127,
        "stop_reason": "token_limit",
        "terminal_token_target_transition_consumed": False,
        "sampling": reference_row["sampling"],
        "runtime_tokenizer_id": RUNTIME_TOKENIZER_ID,
        "all_equal": True,
    }


def expected_nonterminal_artifact_names() -> set[str]:
    names = {
        "attempt.json",
        "journal.jsonl",
        "early-competitor-census.json",
        "authority.json",
        "static.json",
        "vm-before.json",
        "vm-after.json",
        "vm-delta.json",
        "host-gate-00-before-model-conformance.json",
        "host-gate-13-after-scoring.json",
    }
    for label in ("cpu-sampling", "cpu-cli", "model-conformance"):
        names.update({f"{label}.stdout", f"{label}.stderr", f"{label}.json"})
    for child_number in range(1, 13):
        names.add(f"host-gate-{child_number:02d}-before-child.json")
    for pair_index, orientation in enumerate(PAIR_ORDERS, start=1):
        for position, arm in enumerate(orientation, start=1):
            stem = (
                f"pair-{pair_index:02d}-{orientation.lower()}-{position}-{arm.lower()}"
            )
            names.update(
                {
                    f"{stem}.stdout",
                    f"{stem}.stderr",
                    f"{stem}.timing.jsonl",
                    f"{stem}.json",
                }
            )
    return names


def strict_success_inventory() -> list[dict[str, Any]]:
    expected = expected_nonterminal_artifact_names()
    entries = list(PACKET.iterdir())
    actual = {path.name for path in entries}
    require(
        actual == expected,
        f"nonterminal artifact names changed: missing={sorted(expected - actual)} foreign={sorted(actual - expected)}",
    )
    require(
        all(path.is_file() for path in entries),
        "nonterminal inventory contains non-file entry",
    )
    return [file_identity(PACKET / name) for name in sorted(expected)]


def failure_inventory() -> list[dict[str, Any]]:
    try:
        entries = sorted(PACKET.iterdir(), key=lambda path: path.name)
    except BaseException as error:
        return [{"name": None, "inventory_error": f"iterdir:{type(error).__name__}"}]
    inventory = []
    for path in entries:
        if path.name in {"decision.json", "failure.json"}:
            continue
        item: dict[str, Any] = {"name": path.name}
        try:
            item["is_file"] = path.is_file()
            if item["is_file"]:
                item["bytes"] = path.stat().st_size
                item["sha256"] = sha256_file(path)
            else:
                item["inventory_error"] = "not-a-file"
        except BaseException as error:
            item["inventory_error"] = type(error).__name__
        inventory.append(item)
    return inventory


def disposition_for(kind: str, b_exposed: bool) -> str:
    if not b_exposed and kind in {"environment", "infrastructure", "process-capture"}:
        return "CONSUMED_NO_AUTHORITY"
    if kind in {"environment", "process-capture"}:
        return "INVALID"
    return "KILL"


def acquire(attested: bool) -> str:
    require(
        not PACKET.exists(),
        f"packet root already exists: {PACKET}",
        kind="infrastructure",
    )
    require(
        attested,
        "acquisition requires --attest-no-other-user-gpu-workload",
        kind="environment",
    )
    require(
        PACKET.parent.is_dir(),
        f"packet parent absent: {PACKET.parent}",
        kind="infrastructure",
    )
    stage = "packet-reservation"
    root_created = False
    b_state = {"exposed": False, "exposure_unix_ms": None}
    host_gates: list[dict[str, Any]] = []

    def expose_b() -> None:
        if not b_state["exposed"]:
            b_state["exposed"] = True
            b_state["exposure_unix_ms"] = time.time_ns() // 1_000_000
            append_journal(
                {
                    "event": "candidate-B-exposure",
                    "state": "exposed-before-first-spawn-attempt",
                    "unix_ms": b_state["exposure_unix_ms"],
                }
            )

    try:
        os.mkdir(PACKET)
        root_created = True
        fsync_directory(PACKET.parent)
        write_json(
            PACKET / "attempt.json",
            {
                "schema": "qwen-v0660-reserved-attempt/v1",
                "state": "reserved",
                "pid": os.getpid(),
                "unix_ms": time.time_ns() // 1_000_000,
                "packet": str(PACKET.relative_to(ROOT)),
                "user_gpu_attestation": True,
            },
        )
        stage = "early-competitor-census"
        early = competitor_capture("early-after-reservation")
        write_json(PACKET / "early-competitor-census.json", early)
        validate_competitor_capture(early)
        stage = "authority-authentication"
        authority = authenticate_authority()
        write_json(PACKET / "authority.json", authority)
        stage = "static-identity"
        static = static_acquire_checks()
        write_json(PACKET / "static.json", static)
        stage = "vm-before-cpu-conformance"
        vm_before = vm_snapshot("before-cpu-conformance")
        write_json(PACKET / "vm-before.json", vm_before)
        stage = "cpu-conformance"
        cpu = run_conformance_cpu_only()
        stage = "host-before-model-conformance"
        gate = host_gate("before-model-conformance", attested)
        host_gates.append(gate)
        write_json(PACKET / "host-gate-00-before-model-conformance.json", gate)
        validate_host_gate(gate)
        stage = "model-conformance"
        model_conformance = run_model_conformance_only()
        stage = "post-conformance-identity"
        require_static_unchanged(static)

        children: list[dict[str, Any]] = []
        child_number = 0
        for pair_index, orientation in enumerate(PAIR_ORDERS, start=1):
            for position, arm in enumerate(orientation, start=1):
                if child_number:
                    stage = f"cooldown-before-child-{child_number + 1:02d}"
                    time.sleep(COOLDOWN_SECONDS)
                child_number += 1
                stage = f"host-before-child-{child_number:02d}"
                gate = host_gate(f"before-scored-child-{child_number:02d}", attested)
                host_gates.append(gate)
                write_json(
                    PACKET / f"host-gate-{child_number:02d}-before-child.json", gate
                )
                validate_host_gate(gate)
                stage = f"scored-child-{child_number:02d}-{arm}"
                child = run_scored_child(
                    pair_index, orientation, position, arm, static, expose_b
                )
                children.append(child)

        stage = "host-after-scoring"
        gate = host_gate("after-scoring", attested)
        host_gates.append(gate)
        write_json(PACKET / "host-gate-13-after-scoring.json", gate)
        validate_host_gate(gate)
        require(len(host_gates) == 14, "full host gate count changed")
        stage = "vm-after-scoring"
        vm_after = vm_snapshot("after-scoring")
        write_json(PACKET / "vm-after.json", vm_after)
        interval = vm_delta(vm_before, vm_after)
        write_json(PACKET / "vm-delta.json", interval)
        require(
            not interval["fatal_occupancy_growth"],
            "fatal VM occupancy growth",
            kind="environment",
        )
        stage = "cross-child-correctness"
        equality = validate_cross_child_equality(children)
        stage = "final-static-and-model-reauthentication"
        require_static_unchanged(static, rehash_model=True)
        stage = "paired-reduction"
        reduction = reduce_pairs(children)
        disposition = "GO" if reduction["all_performance_gates_pass"] else "NO-GO/PARK"
        inventory = strict_success_inventory()
        stage = "terminal-publication"
        decision = {
            "schema": SUCCESS_SCHEMA,
            "disposition": disposition,
            "authority": []
            if disposition != "GO"
            else ["exact-frozen-fixture-policy-review"],
            "candidate_B_exposed": b_state["exposed"],
            "candidate_B_exposure_unix_ms": b_state["exposure_unix_ms"],
            "user_gpu_attestation": True,
            "authority_authentication": authority,
            "static_identity": static,
            "early_competitor_census": early,
            "cpu_conformance": cpu,
            "model_conformance": model_conformance,
            "host_gates": host_gates,
            "vm_before": vm_before,
            "vm_after": vm_after,
            "vm_delta": interval,
            "children": children,
            "cross_child_equality": equality,
            "reduction": reduction,
            "inventory_scope": "all nonterminal packet files immediately before decision publication",
            "inventory": inventory,
        }
        human_output = json.dumps(
            {"disposition": disposition, "reduction": reduction},
            indent=2,
            sort_keys=True,
        )
        decision_payload = json_bytes(decision)
        stage = "terminal-static-reauthentication"
        require_static_unchanged(static)
        stage = "terminal-publication"
        publish_terminal(decision_payload, failure=False)
        return human_output
    except BaseException as error:
        if not root_created:
            raise
        if isinstance(error, TerminalPublicationFailure) and error.decision_published:
            raise
        kind = error.kind if isinstance(error, PacketFailure) else "infrastructure"
        if kind == "live-child":
            raise
        disposition = disposition_for(kind, bool(b_state["exposed"]))
        failure = {
            "schema": FAILURE_SCHEMA,
            "disposition": disposition,
            "authority": [],
            "stage": stage,
            "failure_kind": kind,
            "error_type": type(error).__name__,
            "error": str(error),
            "candidate_B_exposed": b_state["exposed"],
            "candidate_B_exposure_unix_ms": b_state["exposure_unix_ms"],
            "user_gpu_attestation": True,
            "host_gate_count_completed": len(host_gates),
            "inventory_scope": "all nonterminal packet files immediately before terminal publication",
            "inventory": failure_inventory(),
        }
        try:
            publish_terminal(json_bytes(failure), failure=True)
        except TerminalPublicationFailure as publication_error:
            if publication_error.decision_published:
                raise publication_error from error
        except BaseException:
            pass
        raise


def run_conformance_cpu_only() -> list[dict[str, Any]]:
    commands = (
        (
            "cpu-sampling",
            ["cargo", "test", "--locked", "-p", "qwen-llm", "sampling::", "--lib"],
        ),
        ("cpu-cli", ["cargo", "test", "--locked", "-p", "qwen-cli", "--bin", "qwen"]),
    )
    results = []
    for label, argv in commands:
        stdout = PACKET / f"{label}.stdout"
        stderr = PACKET / f"{label}.stderr"
        record = run_attempt(label, argv, stdout, stderr)
        record["test_results"] = parse_test_results(
            stdout, stderr, label, CPU_TEST_FLOORS[label]
        )
        write_json(PACKET / f"{label}.json", record)
        results.append(record)
    return results


def parse_model_conformance_marker(text: str) -> str:
    prefix = "[sampled-structural-a3b] exact rows/state PASS prompt_token_sha256="
    markers = [line for line in text.splitlines() if line.startswith(prefix)]
    require(
        markers == [MODEL_CONFORMANCE_MARKER],
        "model conformance marker absent, duplicated, or changed",
    )
    digest = markers[0].removeprefix(prefix)
    require(
        SHA256_RE.fullmatch(digest) is not None and digest == PROMPT_TOKEN_SHA256,
        "model conformance marker prompt-token digest changed",
    )
    return digest


def run_model_conformance_only() -> dict[str, Any]:
    label = "model-conformance"
    argv = [
        "cargo",
        "test",
        "--locked",
        "--release",
        "-p",
        "qwen-llm",
        "metal_sampled_structural_matches_copied_a3b",
        "--",
        "--ignored",
        "--nocapture",
    ]
    stdout = PACKET / f"{label}.stdout"
    stderr = PACKET / f"{label}.stderr"
    record = run_attempt(label, argv, stdout, stderr)
    text = stdout.read_text(encoding="utf-8") + stderr.read_text(encoding="utf-8")
    marker_prompt_digest = parse_model_conformance_marker(text)
    summary = parse_test_results(stdout, stderr, label, 1, exact_summary_count=None)
    require(summary["passed"] == 1, "model conformance did not run exactly one test")
    record["test_results"] = summary
    record["marker"] = MODEL_CONFORMANCE_MARKER
    record["authenticated_prompt_token_sha256"] = marker_prompt_digest
    write_json(PACKET / f"{label}.json", record)
    return record


def require_static_unchanged(
    static: dict[str, Any], *, rehash_model: bool = False
) -> None:
    require(
        sha256_file(SCRIPT) == static["runner_sha256"],
        "runner changed during conformance",
    )
    require(
        sha256_file(PREREG) == static["prereg_sha256"],
        "preregistration changed during conformance",
    )
    require(
        sha256_file(PROMPT) == static["prompt_sha256"],
        "prompt changed during conformance",
    )
    require(
        sha256_file(PREDECESSOR_PREREG) == static["predecessor_prereg_sha256"],
        "v0.659 preregistration changed",
    )
    require(
        sha256_file(PREDECESSOR_SCRIPT) == static["predecessor_runner_sha256"],
        "v0.659 runner changed",
    )
    require(
        sha256_file(PREDECESSOR_DECISION) == static["predecessor_decision_sha256"],
        "v0.659 decision changed",
    )
    require(
        sha256_file(QWEN) == static["qwen_sha256"],
        "qwen binary changed during conformance",
    )
    require(
        sha256_file(QWEN_BENCH) == static["qwen_bench_sha256"],
        "qwen-bench changed during conformance",
    )
    require(
        git_text("rev-parse", "HEAD") == static["source_commit"],
        "source commit changed",
    )
    require(
        git_text("rev-parse", "HEAD^{tree}") == static["source_tree"],
        "source tree changed",
    )
    require(
        not git_text("status", "--porcelain=v1"), "worktree changed during conformance"
    )
    stat = MODEL.stat()
    require(
        {
            "device": stat.st_dev,
            "inode": stat.st_ino,
            "bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
        == static["model_identity"],
        "model stat identity changed",
    )
    if rehash_model:
        require(
            sha256_file(MODEL) == static["model_sha256"],
            "model hash changed after scoring",
        )


def emit_stdout_unbuffered(text: str) -> None:
    try:
        complete_write(1, (text + "\n").encode())
    except OSError:
        pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("check-only", "acquire"), required=True)
    parser.add_argument("--attest-no-other-user-gpu-workload", action="store_true")
    args = parser.parse_args()
    if args.phase == "check-only":
        print(json.dumps(check_only(), indent=2, sort_keys=True))
        return 0
    output = acquire(args.attest_no_other_user_gpu_workload)
    emit_stdout_unbuffered(output)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"v0.660 runner failed: {error}", file=sys.stderr)
        raise
