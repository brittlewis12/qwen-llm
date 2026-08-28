#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# ///

"""Analyze the frozen Qwen3.8 Flash-Next selected-prefill quality packet."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import struct
import subprocess
import sys
from pathlib import Path

import qwen4exp_selected_quality_llama as llama_support
from qwen4exp_selected_quality_llama import (
    LOCAL_OPERATION_COUNT,
    MANIFEST_SHA256,
    PACKET_ID,
    PRODUCER_STOP_TOKEN_ID,
    TOTAL_OPERATION_COUNT,
    VOCAB_SIZE,
    canonical_json,
    file_stamp,
    is_lower_hex,
    json_equal_exact,
    parse_json_strict,
    require_bool,
    require_close,
    require_dict,
    require_exact_keys,
    require_int,
    require_list,
    require_number,
    require_string,
    sha256,
    sha256_file,
    validate_core_report,
    validate_fixtures,
    validate_score_row,
)


ANALYSIS_SCHEMA = "qwen4exp-selected-quality-analysis"
ANALYSIS_BINDING_SCHEMA = "qwen4exp-selected-quality-analysis-binding-v1"
SELECTED_START = 2051
ARMS = ("A", "B", "C")
ARM_LABELS = {
    "A": "A_default_safe",
    "B": "B_generic_selected",
    "C": "C_f32_hc_down",
}
CONTRASTS = ("B-A", "C-A", "C-B")
NLL_UPPER_LIMIT = 0.010
SHAPE_NLL_LIMIT = 0.020
DOCUMENT_NLL_LIMIT = 0.050
ANSWER_NLL_LIMIT = 0.050
C_VS_B_SUPERIORITY_LIMIT = -0.005


def read_stable(path: Path, role: str) -> tuple[bytes, dict[str, int | str]]:
    canonical = path.resolve(strict=True)
    before = file_stamp(canonical)
    data = canonical.read_bytes()
    after = file_stamp(canonical)
    if before != after:
        raise RuntimeError(f"{role} changed while reading")
    if len(data) != after["bytes"]:
        raise RuntimeError(f"{role} byte count changed while reading")
    return data, after


def git_blob(repository: Path, commit: str, relative_path: str) -> bytes:
    completed = subprocess.run(
        ["/usr/bin/git", "-C", str(repository), "show", f"{commit}:{relative_path}"],
        check=True,
        stdout=subprocess.PIPE,
    )
    return completed.stdout


def validate_common_sources(
    repository: Path,
    local: dict[str, object],
    llama: dict[str, object],
    source_snapshots: dict[str, dict[str, object]],
) -> dict[str, object]:
    local_source = local["report"]["implementation"]["source"]
    llama_source = llama["report"]["source"]
    local_commit = require_string(local_source["source_commit"], "local source commit")
    llama_commit = require_string(
        llama_source["qwen_source_commit"], "llama source commit"
    )
    if local_commit != llama_commit or not is_lower_hex(local_commit, 40):
        raise RuntimeError("local and llama evidence do not share one source commit")
    paths = {
        "analyzer": "scripts/bench/qwen4exp_selected_quality_analyze.py",
        "llama_wrapper": "scripts/bench/qwen4exp_selected_quality_llama.py",
        "llama_main": "scripts/bench/qwen4exp_selected_quality_llama/main.cpp",
        "llama_cmake": "scripts/bench/qwen4exp_selected_quality_llama/CMakeLists.txt",
        "local_scorer": "crates/qwen-llm/src/qwen4exp_selected_quality.rs",
    }
    local_required = set(local_source["required_head_paths"])
    llama_required = set(llama_source["required_head_paths"])
    if (
        not {paths["analyzer"], paths["llama_wrapper"]} <= local_required
        or not {
            paths["analyzer"],
            paths["llama_wrapper"],
            paths["llama_main"],
            paths["llama_cmake"],
        }
        <= llama_required
    ):
        raise RuntimeError(
            "analysis sources are absent from an acquisition source scope"
        )
    runner_rows = {str(row["path"]): row for row in llama_source["runner_sources"]}
    committed: dict[str, dict[str, object]] = {}
    for role, relative in paths.items():
        blob = git_blob(repository, local_commit, relative)
        row = {
            "path": relative,
            "bytes": len(blob),
            "sha256": sha256(blob),
        }
        committed[role] = row
        if role in source_snapshots:
            snapshot = source_snapshots[role]
            if snapshot["bytes"] != len(blob) or snapshot["sha256"] != row["sha256"]:
                raise RuntimeError(f"executed {role} source differs from common commit")
        if role in {"analyzer", "llama_wrapper", "llama_main", "llama_cmake"}:
            runner_row = runner_rows.get(relative)
            if runner_row is None or (
                runner_row.get("bytes") != len(blob)
                or runner_row.get("sha256") != row["sha256"]
            ):
                raise RuntimeError(f"D source manifest does not bind {role}")
    scorer = local["report"]["implementation"]["scorer_source"]
    if (
        scorer.get("path") != paths["local_scorer"]
        or scorer.get("bytes") != committed["local_scorer"]["bytes"]
        or scorer.get("sha256") != committed["local_scorer"]["sha256"]
    ):
        raise RuntimeError("local scorer source differs from common commit")
    core_source_domain = (
        "qwen4exp-selected-quality-llama-core-sources-v1\n"
        f"main.cpp={committed['llama_main']['sha256']}\n"
        f"CMakeLists.txt={committed['llama_cmake']['sha256']}\n"
    )
    expected_core_source = {
        "main_cpp_sha256": committed["llama_main"]["sha256"],
        "cmake_lists_sha256": committed["llama_cmake"]["sha256"],
        "source_manifest_schema": "qwen4exp-selected-quality-llama-core-sources-v1",
        "source_manifest_domain_utf8": core_source_domain,
        "source_manifest_sha256": sha256(core_source_domain.encode()),
        "build_type": "Release",
    }
    if not json_equal_exact(llama_source["core_build_expected"], expected_core_source):
        raise RuntimeError("D outer source does not bind the committed core harness")
    isolated_build = llama["report"]["isolated_build"]
    expected_isolated_core = {
        **expected_core_source,
        "llama_cpp_tree": llama_source["llama_cpp"]["tree"],
        "llama_cpp_export_manifest_sha256": isolated_build["llama_cpp_export"][
            "tree_manifest"
        ]["manifest_sha256"],
        "build_policy_sha256": isolated_build["policy_sha256"],
    }
    if not json_equal_exact(
        isolated_build["core_build_expected"], expected_isolated_core
    ):
        raise RuntimeError("D isolated core root diverges from committed harness")

    kernel_manifest = local["report"]["implementation"]["source"][
        "kernel_source_manifest"
    ]
    kernel_rows = {str(row["path"]): row for row in kernel_manifest["files"]}
    committed_kernel_paths = subprocess.run(
        [
            "/usr/bin/git",
            "-C",
            str(repository),
            "ls-tree",
            "-r",
            "--name-only",
            local_commit,
            "--",
            "kernels",
        ],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout.splitlines()
    committed_kernel_paths = [
        path
        for path in committed_kernel_paths
        if Path(path).parent.as_posix() == "kernels" and path.endswith(".metal")
    ]
    if set(kernel_rows) != set(committed_kernel_paths):
        raise RuntimeError(
            "local kernel manifest differs from committed direct-child inventory"
        )
    for relative in committed_kernel_paths:
        blob = git_blob(repository, local_commit, relative)
        row = kernel_rows[relative]
        if row["bytes"] != len(blob) or row["sha256"] != sha256(blob):
            raise RuntimeError(f"local kernel differs from common commit: {relative}")
    return {
        "commit": local_commit,
        "sources": committed,
        "core_build_expected": expected_core_source,
        "isolated_core_build_expected": expected_isolated_core,
        "committed_kernel_files": len(committed_kernel_paths),
    }


def sha256_u32(domain: bytes, values: list[int]) -> str:
    digest = hashlib.sha256(domain)
    for value in values:
        digest.update(struct.pack("<I", value))
    return digest.hexdigest()


def hash_treatment(arm: str, prompt_tokens: int) -> str:
    digest = hashlib.sha256(b"qwen4exp-selected-quality-hc-treatment-records-v1\0")
    count = 96 if arm == "C" and prompt_tokens > SELECTED_START else 0
    for _ in range(count):
        digest.update(struct.pack("<I", 1))
        for value in (SELECTED_START, prompt_tokens - SELECTED_START, 10_240, 320):
            digest.update(struct.pack("<Q", value))
    return digest.hexdigest()


def replay_binding_sha256(replay: dict[str, object]) -> str:
    endpoint = require_dict(replay["endpoint"], "replay.endpoint")
    terminal = require_dict(replay["terminal"], "replay.terminal")
    digest = hashlib.sha256(b"qwen4exp-selected-quality-replay-identity-v1\0")
    values = (
        ("endpoint_logits", endpoint["logits_sha256_f32le"]),
        ("endpoint_state", endpoint["persistent_state_sha256"]),
        ("endpoint_qsa", endpoint["qsa_lengths_sha256"]),
        ("endpoint_ple", endpoint["ple_history_sha256_u32le"]),
        ("terminal_logits", terminal["logits_sha256_f32le"]),
        ("terminal_state", terminal["persistent_state_sha256"]),
        ("terminal_qsa", terminal["qsa_lengths_sha256"]),
        ("terminal_ple", terminal["ple_history_sha256_u32le"]),
        ("logit_trace", replay["logit_trace_sha256"]),
        ("treatment", replay["treatment_sha256"]),
    )
    for name, value in values:
        encoded_name = name.encode()
        encoded_value = require_string(value, f"replay.{name}").encode()
        digest.update(struct.pack("<Q", len(encoded_name)))
        digest.update(encoded_name)
        digest.update(encoded_value)
    for value in (
        endpoint["committed_length"],
        terminal["committed_length"],
        replay["logit_trace_rows"],
        replay["treatment_records"],
    ):
        digest.update(struct.pack("<Q", require_int(value, "replay binding integer")))
    return digest.hexdigest()


def run_binding_root(rows: list[dict[str, object]]) -> str:
    digest = hashlib.sha256(b"qwen4exp-selected-quality-ordered-run-bindings-v1\0")
    for ordinal, row in enumerate(rows):
        require_exact_keys(
            row,
            {
                "operation_ordinal",
                "fixture_id",
                "phase",
                "mode",
                "arm",
                "run_binding_sha256",
            },
            f"ordered run binding {ordinal}",
        )
        if require_int(row["operation_ordinal"], "ordered binding ordinal") != ordinal:
            raise RuntimeError("ordered local run binding ordinal")
        digest.update(struct.pack("<Q", ordinal))
        digest.update(require_string(row["run_binding_sha256"], "run binding").encode())
    return digest.hexdigest()


class OutputReservation:
    def __init__(self, output: Path) -> None:
        if output.exists():
            raise RuntimeError(f"output already exists: {output}")
        self.output = output
        self.lock = (
            output.parent / f".{output.name}.qwen4exp-selected-quality-analysis.lock"
        )
        descriptor = os.open(self.lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w") as stream:
            stream.write(f"pid={os.getpid()}\noutput={output}\n")
            stream.flush()
            os.fsync(stream.fileno())
        self.active = True
        self.sync_directory()

    def sync_directory(self) -> None:
        descriptor = os.open(self.output.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def publish(self, data: bytes) -> str:
        report_sha256 = sha256(data)
        temporary = self.output.parent / (
            f".{self.output.name}.tmp.{os.getpid()}.{report_sha256[:16]}"
        )
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        published = False
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.link(temporary, self.output)
            published = True
            self.sync_directory()
            temporary.unlink()
            self.lock.unlink()
            self.sync_directory()
            self.active = False
            return report_sha256
        except BaseException:
            if published:
                self.output.unlink(missing_ok=True)
            temporary.unlink(missing_ok=True)
            self.sync_directory()
            raise

    def close(self) -> None:
        if self.active:
            self.lock.unlink(missing_ok=True)
            self.sync_directory()
            self.active = False

    def __enter__(self) -> OutputReservation:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def validate_snapshot(
    value: object, committed_length: int, path: str
) -> dict[str, object]:
    snapshot = require_dict(value, path)
    require_exact_keys(
        snapshot,
        {
            "logits_sha256_f32le",
            "persistent_state_sha256",
            "qsa_lengths_sha256",
            "ple_history_sha256_u32le",
            "committed_length",
        },
        path,
    )
    for key in (
        "logits_sha256_f32le",
        "persistent_state_sha256",
        "qsa_lengths_sha256",
        "ple_history_sha256_u32le",
    ):
        if not is_lower_hex(snapshot[key]):
            raise RuntimeError(f"{path}.{key}")
    if (
        require_int(snapshot["committed_length"], f"{path}.committed_length")
        != committed_length
    ):
        raise RuntimeError(f"{path}.committed_length")
    return snapshot


def validate_replay(
    value: object,
    arm: str,
    prompt_tokens: int,
    terminal_tokens: int,
    trace_rows: int,
    path: str,
) -> dict[str, object]:
    replay = require_dict(value, path)
    require_exact_keys(
        replay,
        {
            "endpoint",
            "terminal",
            "logit_trace_sha256",
            "logit_trace_rows",
            "treatment_sha256",
            "treatment_records",
            "binding_sha256",
        },
        path,
    )
    validate_snapshot(replay["endpoint"], prompt_tokens, f"{path}.endpoint")
    validate_snapshot(replay["terminal"], terminal_tokens, f"{path}.terminal")
    if not is_lower_hex(replay["logit_trace_sha256"]):
        raise RuntimeError(f"{path}.logit_trace_sha256")
    if (
        require_int(replay["logit_trace_rows"], f"{path}.logit_trace_rows")
        != trace_rows
    ):
        raise RuntimeError(f"{path}.logit_trace_rows")
    treatment_records = 96 if arm == "C" and prompt_tokens > SELECTED_START else 0
    if (
        require_int(replay["treatment_records"], f"{path}.treatment_records")
        != treatment_records
    ):
        raise RuntimeError(f"{path}.treatment_records")
    if replay["treatment_sha256"] != hash_treatment(arm, prompt_tokens):
        raise RuntimeError(f"{path}.treatment_sha256")
    expected_binding = replay_binding_sha256(replay)
    if replay["binding_sha256"] != expected_binding:
        raise RuntimeError(f"{path}.binding_sha256")
    return replay


def validate_prefill(value: object, arm: str, prompt_tokens: int, path: str) -> None:
    prefill = require_dict(value, path)
    require_exact_keys(
        prefill,
        {
            "token_count",
            "packed_token_count",
            "contains_selection",
            "command_count",
            "encode_cpu_ms",
            "completion_wait_ms",
            "gpu_ms",
            "gpu_samples",
            "total_wall_ms",
            "outside_gpu_ms",
        },
        path,
    )
    selected = arm != "A" and prompt_tokens > SELECTED_START
    expected_packed = prompt_tokens if selected else min(prompt_tokens, SELECTED_START)
    if prompt_tokens == SELECTED_START:
        expected_commands = 2
    elif selected:
        expected_commands = 3
    else:
        expected_commands = 2 + prompt_tokens - SELECTED_START
    expected = {
        "token_count": prompt_tokens,
        "packed_token_count": expected_packed,
        "contains_selection": selected,
        "command_count": expected_commands,
    }
    for key, expected_value in expected.items():
        if prefill[key] != expected_value:
            raise RuntimeError(f"{path}.{key}")
    for key in (
        "encode_cpu_ms",
        "completion_wait_ms",
        "total_wall_ms",
    ):
        require_number(prefill[key], f"{path}.{key}")
    gpu_samples = require_int(prefill["gpu_samples"], f"{path}.gpu_samples")
    if not 0 <= gpu_samples <= expected_commands:
        raise RuntimeError(f"{path}.gpu_samples")
    if gpu_samples == expected_commands:
        require_number(prefill["gpu_ms"], f"{path}.gpu_ms")
        require_number(prefill["outside_gpu_ms"], f"{path}.outside_gpu_ms")
    elif prefill["gpu_ms"] is not None or prefill["outside_gpu_ms"] is not None:
        raise RuntimeError(f"{path} incomplete GPU telemetry")


def validate_local_binding(
    observation: dict[str, object],
    operation: dict[str, object],
    input_sha256: str,
    evidence_binding_sha256: str,
    path: str,
) -> str:
    binding = require_dict(observation["binding"], f"{path}.binding")
    require_exact_keys(
        binding,
        {
            "semantic_payload_encoding",
            "semantic_payload_bytes",
            "semantic_payload_sha256",
            "semantic_payload_json_compact",
            "domain_utf8",
            "sha256",
        },
        f"{path}.binding",
    )
    if binding["semantic_payload_encoding"] != (
        "serde_json compact UTF-8 over the observation before its binding field"
    ):
        raise RuntimeError(f"{path}.binding.semantic_payload_encoding")
    compact = require_string(
        binding["semantic_payload_json_compact"],
        f"{path}.binding.semantic_payload_json_compact",
    )
    compact_bytes = compact.encode()
    payload = dict(observation)
    del payload["binding"]
    if not json_equal_exact(parse_json_strict(compact_bytes), payload):
        raise RuntimeError(f"{path}.binding semantic payload content")
    if require_int(
        binding["semantic_payload_bytes"], f"{path}.binding.semantic_payload_bytes"
    ) != len(compact_bytes):
        raise RuntimeError(f"{path}.binding.semantic_payload_bytes")
    semantic_sha256 = sha256(compact_bytes)
    if binding["semantic_payload_sha256"] != semantic_sha256:
        raise RuntimeError(f"{path}.binding.semantic_payload_sha256")
    replay = require_dict(observation["replay"], f"{path}.replay")
    domain = (
        "qwen4exp-selected-quality-run-binding-v1\0"
        f"evidence_binding_sha256={evidence_binding_sha256}\n"
        f"fixture_manifest_sha256={MANIFEST_SHA256}\n"
        f"fixture_id={operation['fixture_id']}\n"
        f"input_sha256={input_sha256}\n"
        f"operation_ordinal={operation['ordinal']}\n"
        f"phase={operation['phase']}\n"
        f"mode={operation['mode']}\n"
        f"arm={operation['arm']}\n"
        f"replay_identity_sha256={replay['binding_sha256']}\n"
        f"semantic_payload_sha256={semantic_sha256}\n"
    )
    if binding["domain_utf8"] != domain or binding["sha256"] != sha256(domain.encode()):
        raise RuntimeError(f"{path}.binding domain")
    return require_string(binding["sha256"], f"{path}.binding.sha256")


def validate_local_score_rows(
    continuation: dict[str, object],
    expected_tokens: list[int],
    path: str,
) -> dict[str, object]:
    require_exact_keys(
        continuation,
        {"tokens", "nll_sum_f64", "mean_nll_f64", "top1_hits", "scored_rows"},
        path,
    )
    if require_int(continuation["tokens"], f"{path}.tokens") != len(expected_tokens):
        raise RuntimeError(f"{path}.tokens")
    rows = require_list(continuation["scored_rows"], f"{path}.scored_rows")
    if len(rows) != len(expected_tokens):
        raise RuntimeError(f"{path}.scored_rows count")
    nll_values: list[float] = []
    predictions: list[int] = []
    top1_hits = 0
    for ordinal, (row, target) in enumerate(zip(rows, expected_tokens, strict=True)):
        nll, top1, prediction = validate_score_row(
            row,
            ordinal,
            target,
            f"{path}.scored_rows[{ordinal}]",
        )
        nll_values.append(nll)
        predictions.append(prediction)
        top1_hits += int(top1)
    nll_sum = sum(nll_values)
    require_close(
        require_number(continuation["nll_sum_f64"], f"{path}.nll_sum_f64"),
        nll_sum,
        f"{path}.nll_sum_f64",
    )
    require_close(
        require_number(continuation["mean_nll_f64"], f"{path}.mean_nll_f64"),
        nll_sum / len(expected_tokens),
        f"{path}.mean_nll_f64",
    )
    if require_int(continuation["top1_hits"], f"{path}.top1_hits") != top1_hits:
        raise RuntimeError(f"{path}.top1_hits")
    return {
        "nll_values": nll_values,
        "nll_sum": nll_sum,
        "mean_nll": nll_sum / len(expected_tokens),
        "top1_hits": top1_hits,
        "predictions": predictions,
    }


def validate_retrieval_answer(
    answer: dict[str, object],
    fixture: dict[str, object],
    path: str,
) -> dict[str, object]:
    require_exact_keys(
        answer,
        {
            "expected_token_ids",
            "tokens",
            "nll_sum_f64",
            "mean_nll_f64",
            "scored_rows",
            "greedy_prefix_through_first_mismatch_or_stop",
            "greedy_prefix_contract",
            "exact_pass",
        },
        path,
    )
    targets = [int(value) for value in fixture["answer_token_ids"]]
    if answer["expected_token_ids"] != targets:
        raise RuntimeError(f"{path}.expected_token_ids")
    if require_int(answer["tokens"], f"{path}.tokens") != len(targets):
        raise RuntimeError(f"{path}.tokens")
    rows = require_list(answer["scored_rows"], f"{path}.scored_rows")
    if len(rows) != len(targets):
        raise RuntimeError(f"{path}.scored_rows count")
    nll_values: list[float] = []
    predictions: list[int] = []
    for ordinal, (row, target) in enumerate(zip(rows, targets, strict=True)):
        nll, _, prediction = validate_score_row(
            row,
            ordinal,
            target,
            f"{path}.scored_rows[{ordinal}]",
        )
        nll_values.append(nll)
        predictions.append(prediction)
    nll_sum = sum(nll_values)
    require_close(
        require_number(answer["nll_sum_f64"], f"{path}.nll_sum_f64"),
        nll_sum,
        f"{path}.nll_sum_f64",
    )
    require_close(
        require_number(answer["mean_nll_f64"], f"{path}.mean_nll_f64"),
        nll_sum / len(targets),
        f"{path}.mean_nll_f64",
    )
    if answer["greedy_prefix_contract"] != (
        "true free-running prefix; after a mismatch, remaining expected answer tokens "
        "are teacher-forced only"
    ):
        raise RuntimeError(f"{path}.greedy_prefix_contract")
    prefix = require_list(
        answer["greedy_prefix_through_first_mismatch_or_stop"],
        f"{path}.greedy_prefix_through_first_mismatch_or_stop",
    )
    if not all(
        not isinstance(token, bool)
        and isinstance(token, int)
        and 0 <= token < VOCAB_SIZE
        for token in prefix
    ):
        raise RuntimeError(f"{path} greedy prefix token range")
    expected_prefix: list[int] = []
    all_match = True
    for prediction, target in zip(predictions, targets, strict=True):
        if all_match:
            expected_prefix.append(prediction)
            all_match = prediction == target
    exact_pass = require_bool(answer["exact_pass"], f"{path}.exact_pass")
    if all_match:
        if len(prefix) != len(targets) + 1 or prefix[:-1] != expected_prefix:
            raise RuntimeError(f"{path} greedy stop row")
        expected_exact = prefix[-1] in fixture["producer_stop_token_ids"]
    else:
        if prefix != expected_prefix:
            raise RuntimeError(f"{path} greedy mismatch prefix")
        expected_exact = False
    if (
        len(prefix) > int(fixture["max_generated_tokens"])
        or exact_pass != expected_exact
    ):
        raise RuntimeError(f"{path}.exact_pass")
    return {
        "nll_values": nll_values,
        "nll_sum": nll_sum,
        "mean_nll": nll_sum / len(targets),
        "predictions": predictions,
        "greedy_prefix": prefix,
        "exact_pass": exact_pass,
    }


def validate_model_rows(
    rows: object,
    lock: dict[str, object],
    path: str,
) -> list[dict[str, object]]:
    observed = require_list(rows, path)
    expected_rows = sorted(lock["shards"], key=lambda row: int(row["index"]))
    if len(observed) != len(expected_rows):
        raise RuntimeError(f"{path} count")
    normalized: list[dict[str, object]] = []
    for index, (value, expected) in enumerate(
        zip(observed, expected_rows, strict=True)
    ):
        row = require_dict(value, f"{path}[{index}]")
        file_name = require_string(row.get("file_name"), f"{path}[{index}].file_name")
        if (
            require_int(row.get("index"), f"{path}[{index}].index") != expected["index"]
            or file_name != expected["filename"]
            or require_int(row.get("bytes"), f"{path}[{index}].bytes")
            != expected["bytes"]
            or require_string(row.get("sha256"), f"{path}[{index}].sha256")
            != expected["sha256"]
        ):
            raise RuntimeError(f"{path}[{index}] model identity")
        normalized.append(
            {
                "index": index,
                "file_name": file_name,
                "bytes": expected["bytes"],
                "sha256": expected["sha256"],
            }
        )
    return normalized


def validate_file_stamp(value: object, path: str) -> dict[str, object]:
    stamp = require_dict(value, path)
    require_exact_keys(
        stamp,
        {"path", "device", "inode", "bytes", "mtime_ns", "ctime_ns", "mode"},
        path,
    )
    require_string(stamp["path"], f"{path}.path")
    for key in ("device", "inode", "bytes", "mtime_ns", "ctime_ns", "mode"):
        if require_int(stamp[key], f"{path}.{key}") < 0:
            raise RuntimeError(f"{path}.{key}")
    return stamp


def validate_command_record(value: object, path: str) -> None:
    record = require_dict(value, path)
    require_exact_keys(
        record,
        {"command", "stdout_bytes", "stdout_sha256", "returncode"},
        path,
    )
    command = require_list(record["command"], f"{path}.command")
    if not command or not all(isinstance(argument, str) for argument in command):
        raise RuntimeError(f"{path}.command")
    if require_int(record["stdout_bytes"], f"{path}.stdout_bytes") < 0:
        raise RuntimeError(f"{path}.stdout_bytes")
    if not is_lower_hex(record["stdout_sha256"]):
        raise RuntimeError(f"{path}.stdout_sha256")
    if require_int(record["returncode"], f"{path}.returncode") != 0:
        raise RuntimeError(f"{path}.returncode")


def validate_d_build_details(
    isolated_build: dict[str, object],
    source: dict[str, object],
) -> None:
    policy = require_dict(isolated_build["policy"], "llama build policy")
    require_exact_keys(
        policy,
        {
            "schema",
            "llama_cpp_commit",
            "llama_cpp_tree",
            "llama_cpp_export_manifest_sha256",
            "generator",
            "parallel_jobs",
            "tools",
            "definitions",
            "environment",
        },
        "llama build policy",
    )
    if (
        policy["schema"] != "qwen4exp-selected-quality-llama-isolated-build-v1"
        or policy["llama_cpp_commit"] != source["llama_cpp"]["commit"]
        or policy["llama_cpp_tree"] != source["llama_cpp"]["tree"]
        or policy["generator"] != "Ninja"
        or require_int(policy["parallel_jobs"], "llama build jobs") != 8
    ):
        raise RuntimeError("llama frozen build policy")
    tools = require_dict(policy["tools"], "llama build tools")
    require_exact_keys(
        tools,
        {"cmake", "ninja", "clang", "clangxx", "macos_sdk", "git", "tar", "otool"},
        "llama build tools",
    )
    for name in ("cmake", "ninja"):
        tool = require_dict(tools[name], f"llama build tool {name}")
        require_exact_keys(tool, {"path", "version"}, f"llama build tool {name}")
        if not require_string(tool["path"], f"{name} path").startswith(
            "/"
        ) or not require_string(tool["version"], f"{name} version"):
            raise RuntimeError(f"llama build tool {name}")
    for name, expected_path in (
        ("clang", "/usr/bin/clang"),
        ("clangxx", "/usr/bin/clang++"),
    ):
        tool = require_dict(tools[name], f"llama build tool {name}")
        require_exact_keys(tool, {"path", "version_sha256"}, f"llama build tool {name}")
        if tool["path"] != expected_path or not is_lower_hex(tool["version_sha256"]):
            raise RuntimeError(f"llama build tool {name}")
    if (
        tools["git"] != "/usr/bin/git"
        or tools["tar"] != "/usr/bin/tar"
        or tools["otool"] != "/usr/bin/otool"
        or not require_string(tools["macos_sdk"], "macOS SDK").startswith("/")
    ):
        raise RuntimeError("llama system build tools")
    definitions = require_dict(policy["definitions"], "llama build definitions")
    expected_static_definitions = {
        "CMAKE_ASM_COMPILER": "/usr/bin/clang",
        "CMAKE_BUILD_TYPE": "Release",
        "CMAKE_C_COMPILER": "/usr/bin/clang",
        "CMAKE_CXX_COMPILER": "/usr/bin/clang++",
        "CMAKE_C_FLAGS": "-fno-fast-math -fno-unsafe-math-optimizations -ffp-contract=off",
        "CMAKE_C_FLAGS_RELEASE": "-O3 -DNDEBUG",
        "CMAKE_CXX_FLAGS": "-fno-fast-math -fno-unsafe-math-optimizations -ffp-contract=off",
        "CMAKE_CXX_FLAGS_RELEASE": "-O3 -DNDEBUG",
        "CMAKE_EXE_LINKER_FLAGS": "",
        "CMAKE_EXPORT_COMPILE_COMMANDS": "ON",
        "CMAKE_OSX_ARCHITECTURES": "arm64",
        "GGML_ACCELERATE": "ON",
        "GGML_BACKEND_DL": "OFF",
        "GGML_BLAS": "OFF",
        "GGML_CCACHE": "OFF",
        "GGML_CPU": "ON",
        "GGML_CPU_ALL_VARIANTS": "OFF",
        "GGML_CPU_KLEIDIAI": "OFF",
        "GGML_LLAMAFILE": "OFF",
        "GGML_METAL": "ON",
        "GGML_METAL_EMBED_LIBRARY": "ON",
        "GGML_METAL_NDEBUG": "OFF",
        "GGML_METAL_SHADER_DEBUG": "OFF",
        "GGML_NATIVE": "OFF",
        "GGML_OPENMP": "OFF",
    }
    if set(definitions) != set(expected_static_definitions) | {
        "CMAKE_MAKE_PROGRAM",
        "CMAKE_OSX_SYSROOT",
    }:
        raise RuntimeError("llama build definition key set")
    for key, expected in expected_static_definitions.items():
        if definitions[key] != expected:
            raise RuntimeError(f"llama build definition {key}")
    if (
        definitions["CMAKE_MAKE_PROGRAM"] != tools["ninja"]["path"]
        or definitions["CMAKE_OSX_SYSROOT"] != tools["macos_sdk"]
    ):
        raise RuntimeError("llama build tool-definition cross-link")
    expected_environment = {
        "HOME": "<private-build-home>",
        "LC_ALL": "C",
        "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
        "TMPDIR": "<private-build-tmp>",
        "ZERO_AR_DATE": "1",
    }
    if not json_equal_exact(policy["environment"], expected_environment):
        raise RuntimeError("llama build environment")

    export = require_dict(isolated_build["llama_cpp_export"], "llama source export")
    require_exact_keys(
        export,
        {
            "archive_bytes",
            "archive_sha256",
            "tree_manifest",
            "read_only_during_build",
            "manifest_revalidated_after_build",
        },
        "llama source export",
    )
    if (
        require_int(export["archive_bytes"], "llama archive bytes") <= 0
        or not is_lower_hex(export["archive_sha256"])
        or export["read_only_during_build"] is not True
        or export["manifest_revalidated_after_build"] is not True
    ):
        raise RuntimeError("llama source archive")
    tree_manifest = require_dict(export["tree_manifest"], "llama export manifest")
    require_exact_keys(
        tree_manifest,
        {"schema", "files", "bytes", "manifest_sha256", "rows"},
        "llama export manifest",
    )
    rows = require_list(tree_manifest["rows"], "llama export rows")
    domain = bytearray(b"qwen4exp-selected-quality-llama-export-v1\0")
    prior_path = ""
    total_bytes = 0
    for index, value in enumerate(rows):
        row = require_dict(value, f"llama export row {index}")
        require_exact_keys(
            row,
            {"path", "git_mode", "kind", "bytes", "sha256", "git_blob_id"},
            f"llama export row {index}",
        )
        relative = require_string(row["path"], "llama export path")
        if relative <= prior_path:
            raise RuntimeError("llama export path order")
        prior_path = relative
        mode = require_string(row["git_mode"], "llama export mode")
        kind = require_string(row["kind"], "llama export kind")
        if (mode, kind) not in {
            ("100644", "regular"),
            ("100755", "regular"),
            ("120000", "symlink"),
        }:
            raise RuntimeError("llama export mode")
        size = require_int(row["bytes"], "llama export bytes")
        if size < 0 or not is_lower_hex(row["sha256"]):
            raise RuntimeError("llama export content identity")
        blob_id = require_string(row["git_blob_id"], "llama export Git blob")
        expected_blob_length = (
            40 if source["llama_cpp"]["object_format"] == "sha1" else 64
        )
        if not is_lower_hex(blob_id, expected_blob_length):
            raise RuntimeError("llama export Git blob identity")
        total_bytes += size
        domain.extend(
            f"{relative}\t{mode}\t{kind}\t{size}\t{row['sha256']}\t{blob_id}\n".encode()
        )
    manifest_sha256 = sha256(bytes(domain))
    if (
        tree_manifest["schema"] != "qwen4exp-selected-quality-llama-export-v1"
        or require_int(tree_manifest["files"], "llama export file count") != len(rows)
        or require_int(tree_manifest["bytes"], "llama export total bytes")
        != total_bytes
        or tree_manifest["manifest_sha256"] != manifest_sha256
        or policy["llama_cpp_export_manifest_sha256"] != manifest_sha256
    ):
        raise RuntimeError("llama export manifest binding")

    validate_command_record(isolated_build["configure"], "llama configure")
    validate_command_record(isolated_build["build"], "llama build")
    cache = require_dict(isolated_build["cmake_cache"], "llama CMake cache")
    require_exact_keys(
        cache, {"bytes", "sha256", "enforced_values"}, "llama CMake cache"
    )
    if require_int(cache["bytes"], "llama CMake cache bytes") <= 0 or not is_lower_hex(
        cache["sha256"]
    ):
        raise RuntimeError("llama CMake cache identity")
    enforced = require_dict(cache["enforced_values"], "llama enforced CMake values")
    for key, value in definitions.items():
        if enforced.get(key) != value:
            raise RuntimeError(f"llama enforced CMake value {key}")
    expected_enforced = {
        "BUILD_SHARED_LIBS": "OFF",
        "LLAMA_BUILD_EXAMPLES": "OFF",
        "LLAMA_BUILD_SERVER": "OFF",
        "LLAMA_BUILD_TESTS": "OFF",
        "LLAMA_BUILD_TOOLS": "OFF",
        "LLAMA_OPENSSL": "OFF",
        "LLAMA_SUBPROCESS": "OFF",
        "QWEN4EXP_BUILD_POLICY_SHA256": isolated_build["policy_sha256"],
        "QWEN4EXP_LLAMA_EXPORT_MANIFEST_SHA256": manifest_sha256,
        "QWEN4EXP_LLAMA_TREE": source["llama_cpp"]["tree"],
    }
    for key, expected in expected_enforced.items():
        if enforced.get(key) != expected:
            raise RuntimeError(f"llama enforced CMake value {key}")
    if set(enforced) != set(definitions) | set(expected_enforced):
        raise RuntimeError("llama enforced CMake key set")

    compile_commands = require_dict(
        isolated_build["compile_commands"], "llama compile commands"
    )
    require_exact_keys(
        compile_commands,
        {
            "path",
            "bytes",
            "sha256",
            "commands",
            "language_commands_with_safe_fp_policy",
            "forbidden_flags_absent",
            "forced_includes_absent",
        },
        "llama compile commands",
    )
    command_count = require_int(compile_commands["commands"], "compile command count")
    safe_command_count = require_int(
        compile_commands["language_commands_with_safe_fp_policy"],
        "safe FP command count",
    )
    if (
        compile_commands["path"] != "compile_commands.json"
        or require_int(compile_commands["bytes"], "compile command bytes") <= 0
        or not is_lower_hex(compile_commands["sha256"])
        or command_count < 100
        or not 100 <= safe_command_count <= command_count
        or compile_commands["forced_includes_absent"] is not True
    ):
        raise RuntimeError("llama compile command policy")
    expected_forbidden = sorted(
        {
            "-Ofast",
            "-fassociative-math",
            "-ffast-math",
            "-ffinite-math-only",
            "-fno-signed-zeros",
            "-fno-trapping-math",
            "-freciprocal-math",
            "-funsafe-math-optimizations",
            "-march=native",
            "-mtune=native",
        }
    )
    if compile_commands["forbidden_flags_absent"] != expected_forbidden:
        raise RuntimeError("llama forbidden compile flag census")
    for key, required_keys in (
        ("ninja_commands", {"bytes", "sha256", "unsafe_options_absent"}),
        ("build_ninja", {"bytes", "sha256"}),
    ):
        record = require_dict(isolated_build[key], f"llama {key}")
        require_exact_keys(record, required_keys, f"llama {key}")
        if require_int(record["bytes"], f"llama {key} bytes") <= 0 or not is_lower_hex(
            record["sha256"]
        ):
            raise RuntimeError(f"llama {key} identity")
        if key == "ninja_commands" and record["unsafe_options_absent"] is not True:
            raise RuntimeError("llama unsafe Ninja options")
    linkage = require_dict(isolated_build["dynamic_linkage"], "llama dynamic linkage")
    require_exact_keys(
        linkage,
        {
            "policy",
            "dependencies",
            "otool_L_sha256",
            "otool_l_bytes",
            "otool_l_sha256",
            "lc_rpath_absent",
        },
        "llama dynamic linkage",
    )
    dependencies = require_list(linkage["dependencies"], "llama dynamic dependencies")
    if (
        linkage["policy"] != "only /usr/lib and /System/Library/Frameworks dependencies"
        or not dependencies
        or not all(
            isinstance(dependency, str)
            and dependency.startswith(("/usr/lib/", "/System/Library/Frameworks/"))
            for dependency in dependencies
        )
        or not is_lower_hex(linkage["otool_L_sha256"])
        or require_int(linkage["otool_l_bytes"], "otool load command bytes") <= 0
        or not is_lower_hex(linkage["otool_l_sha256"])
        or linkage["lc_rpath_absent"] is not True
    ):
        raise RuntimeError("llama dynamic linkage policy")


def input_sha256_for_operation(
    fixture: dict[str, object],
) -> str:
    return str(fixture["tokens"]["sha256_raw_i32le"])


def validate_compact_payload(
    compact_value: object,
    payload: object,
    expected_sha256: object,
    path: str,
) -> str:
    compact = require_string(compact_value, f"{path} compact JSON")
    encoded = compact.encode()
    if not json_equal_exact(parse_json_strict(encoded), payload):
        raise RuntimeError(f"{path} compact JSON content")
    observed_sha256 = sha256(encoded)
    if expected_sha256 != observed_sha256:
        raise RuntimeError(f"{path} compact JSON SHA-256")
    return observed_sha256


def validate_local_global_provenance(
    report: dict[str, object],
    manifest: dict[str, object],
) -> str:
    implementation = require_dict(report["implementation"], "local.implementation")
    source = require_dict(implementation["source"], "local source")
    require_exact_keys(
        source,
        {
            "source_commit",
            "tracked_diff_definition",
            "tracked_diff_bytes",
            "tracked_diff_sha256",
            "required_head_paths",
            "required_head_path_count",
            "scoped_status_bytes",
            "scoped_status_sha256",
            "full_worktree_status_bytes",
            "full_worktree_status_sha256",
            "kernel_source_manifest",
            "kernel_source_manifest_sha256",
            "path_dependencies",
            "path_dependencies_json_compact",
            "path_dependencies_sha256",
            "cleanliness_scope",
        },
        "local source",
    )
    if (
        not is_lower_hex(source["source_commit"], 40)
        or source["tracked_diff_definition"]
        != "sha256(git diff --binary --no-ext-diff HEAD --)"
        or source["tracked_diff_bytes"] != 0
        or source["tracked_diff_sha256"]
        != "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        or source["scoped_status_bytes"] != 0
        or source["scoped_status_sha256"]
        != "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        or source["cleanliness_scope"]
        != (
            "all tracked files plus untracked files under the crate, kernels, packet, "
            "preparation script, and workspace manifests; both external path dependencies "
            "have pinned Git trees and clean relevant scopes"
        )
    ):
        raise RuntimeError("local source cleanliness contract")
    required_paths = require_list(source["required_head_paths"], "local required paths")
    if (
        source["required_head_path_count"] != len(required_paths)
        or required_paths != sorted(set(required_paths))
        or "scripts/bench/qwen4exp_selected_quality_analyze.py" not in required_paths
    ):
        raise RuntimeError("local required source path contract")
    if require_int(
        source["full_worktree_status_bytes"], "local full status bytes"
    ) < 0 or not is_lower_hex(source["full_worktree_status_sha256"]):
        raise RuntimeError("local full worktree status identity")

    kernels = require_dict(source["kernel_source_manifest"], "local kernel manifest")
    require_exact_keys(
        kernels,
        {"schema", "discovery", "domain_utf8", "sha256", "files"},
        "local kernel manifest",
    )
    files = require_list(kernels["files"], "local kernel files")
    kernel_domain = "qwen4exp-selected-quality-kernel-sources-v1\0"
    prior_path = ""
    for index, value in enumerate(files):
        row = require_dict(value, f"local kernel file {index}")
        require_exact_keys(row, {"path", "bytes", "sha256"}, "local kernel file")
        kernel_path = require_string(row["path"], "local kernel path")
        if kernel_path <= prior_path or not kernel_path.endswith(".metal"):
            raise RuntimeError("local kernel path order")
        prior_path = kernel_path
        byte_count = require_int(row["bytes"], "local kernel bytes")
        if byte_count <= 0 or not is_lower_hex(row["sha256"]):
            raise RuntimeError("local kernel file identity")
        kernel_domain += f"{kernel_path}\t{byte_count}\t{row['sha256']}\n"
    kernel_sha256 = sha256(kernel_domain.encode())
    if (
        kernels["schema"] != "qwen4exp-selected-quality-kernel-sources-v1"
        or kernels["discovery"]
        != (
            "sorted direct children of workspace kernels/ with extension .metal, "
            "matching crates/qwen-llm/build.rs"
        )
        or kernels["domain_utf8"] != kernel_domain
        or kernels["sha256"] != kernel_sha256
        or source["kernel_source_manifest_sha256"] != kernel_sha256
    ):
        raise RuntimeError("local kernel source manifest binding")

    dependencies = require_dict(source["path_dependencies"], "local path dependencies")
    require_exact_keys(
        dependencies, {"gguf_rs", "llama_cpp_sys_2"}, "local path dependencies"
    )
    expected_dependencies = {
        "gguf_rs": {
            "commit": "3a92c518bce43959686bef1093b31b4067502d2b",
            "tree_spec": "HEAD^{tree}",
            "tree": "c4fd676301c9b9ac58a8ef1441717384afa4fac9",
            "clean_scopes": ["."],
        },
        "llama_cpp_sys_2": {
            "commit": "0f1868b3b52dea227c16b5707578f935042cf668",
            "tree_spec": "HEAD:llama-cpp-sys-2",
            "tree": "bf14ff07169dc0b52112146c1e35918a400234b5",
            "clean_scopes": ["Cargo.toml", "llama-cpp-sys-2"],
        },
    }
    for name, expected in expected_dependencies.items():
        dependency = require_dict(dependencies[name], f"local dependency {name}")
        require_exact_keys(
            dependency,
            {
                "repository_path",
                "commit",
                "tree_spec",
                "tree",
                "clean_scopes",
                "scoped_status_sha256",
                "full_status_bytes",
                "full_status_sha256",
                "qualification",
            },
            f"local dependency {name}",
        )
        for key, expected_value in expected.items():
            if dependency[key] != expected_value:
                raise RuntimeError(f"local dependency {name}.{key}")
        if (
            dependency["scoped_status_sha256"]
            != "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
            or require_int(dependency["full_status_bytes"], "dependency status bytes")
            < 0
            or not is_lower_hex(dependency["full_status_sha256"])
            or dependency["qualification"]
            != (
                "the pinned Git tree plus an empty scoped status identify every "
                "path-dependency source byte used by this workspace"
            )
        ):
            raise RuntimeError(f"local dependency {name} status identity")
    path_dependencies_sha256 = validate_compact_payload(
        source["path_dependencies_json_compact"],
        dependencies,
        source["path_dependencies_sha256"],
        "local path dependencies",
    )

    expected_arithmetic = {
        "undeclared_qwen_environment_overrides_rejected": True,
        "allowed_qwen_environment_names": [
            "QWEN4EXP_Q3_K_XL_RUNTIME_GGUF",
            "QWEN4EXP_SELECTED_QUALITY_OUT",
            "QWEN4EXP_SELECTED_QUALITY_SOURCE",
            "QWEN4EXP_SELECTED_QUALITY_DIFF_SHA256",
        ],
        "qwen4exp_moe_iq3_fast": {
            "effective": True,
            "authority": "cfg(test) thread-local override around all 78 local operations",
        },
        "qwen4exp_packed_router_e8p32_strict": {
            "effective": True,
            "authority": "cfg(test) thread-local override around all 78 local operations",
        },
        "qwen_matvec_q8_0_lcpp": {
            "effective": True,
            "authority": "default-on source policy after rejecting its environment override",
        },
        "all_other_kernel_policies": (
            "source-bound defaults in the exact test executable and embedded metallib"
        ),
    }
    arithmetic = require_dict(
        implementation["arithmetic_policy"], "local arithmetic policy"
    )
    if not json_equal_exact(arithmetic, expected_arithmetic):
        raise RuntimeError("local arithmetic policy")
    arithmetic_sha256 = validate_compact_payload(
        implementation["arithmetic_policy_json_compact"],
        arithmetic,
        implementation["arithmetic_policy_sha256"],
        "local arithmetic policy",
    )

    test_executable = require_dict(
        implementation["test_executable"], "local test executable"
    )
    require_exact_keys(
        test_executable, {"path", "bytes", "sha256"}, "local test executable"
    )
    if require_int(
        test_executable["bytes"], "local test executable bytes"
    ) <= 0 or not is_lower_hex(test_executable["sha256"]):
        raise RuntimeError("local test executable identity")
    scorer = require_dict(implementation["scorer_source"], "local scorer source")
    require_exact_keys(scorer, {"path", "bytes", "sha256"}, "local scorer source")
    if require_int(scorer["bytes"], "local scorer bytes") <= 0 or not is_lower_hex(
        scorer["sha256"]
    ):
        raise RuntimeError("local scorer identity")
    metallib = require_dict(implementation["embedded_metallib"], "local metallib")
    require_exact_keys(metallib, {"bytes", "sha256"}, "local metallib")
    if require_int(metallib["bytes"], "local metallib bytes") <= 0 or not is_lower_hex(
        metallib["sha256"]
    ):
        raise RuntimeError("local metallib identity")

    model = require_dict(report["model"], "local.model")
    require_exact_keys(
        model,
        {
            "repository",
            "revision",
            "quant",
            "qualification",
            "shard_manifest_domain_utf8",
            "shard_manifest_sha256",
            "retained_descriptor_stamp_domain_utf8",
            "retained_descriptor_stamp_sha256",
            "mapped_bytes_hashed_directly",
            "retained_stamps_revalidated_after_hashing",
            "shards",
            "config_equals_flash_next_reference",
        },
        "local.model",
    )
    shard_domain = "qwen4exp-release-model-shard-manifest-v1\0"
    retained_domain = "qwen4exp-selected-quality-model-stamps-v1\0"
    for index, value in enumerate(require_list(model["shards"], "local model shards")):
        row = require_dict(value, f"local model shard {index}")
        require_exact_keys(
            row,
            {
                "index",
                "path",
                "file_name",
                "bytes",
                "sha256",
                "retained_descriptor_stamp",
            },
            f"local model shard {index}",
        )
        if require_int(row["index"], "local model shard index") != index:
            raise RuntimeError("local model shard index")
        require_string(row["path"], "local model shard path")
        require_string(row["file_name"], "local model shard file name")
        if require_int(
            row["bytes"], "local model shard bytes"
        ) <= 0 or not is_lower_hex(row["sha256"]):
            raise RuntimeError("local model shard identity")
        stamp = require_dict(row["retained_descriptor_stamp"], "local retained stamp")
        require_exact_keys(
            stamp,
            {"device", "inode", "mtime_sec", "mtime_nsec", "ctime_sec", "ctime_nsec"},
            "local retained stamp",
        )
        for key in (
            "device",
            "inode",
            "mtime_sec",
            "mtime_nsec",
            "ctime_sec",
            "ctime_nsec",
        ):
            if require_int(stamp[key], f"local retained stamp {key}") < 0:
                raise RuntimeError(f"local retained stamp {key}")
        retained_domain += (
            f"{index}\t{stamp['device']}\t{stamp['inode']}\t{row['bytes']}\t"
            f"{stamp['mtime_sec']}\t{stamp['mtime_nsec']}\t"
            f"{stamp['ctime_sec']}\t{stamp['ctime_nsec']}\n"
        )
        shard_domain += (
            f"{index}\t{row['file_name']}\t{row['bytes']}\t{row['sha256']}\n"
        )
    shard_sha256 = sha256(shard_domain.encode())
    retained_sha256 = sha256(retained_domain.encode())
    if (
        model.get("shard_manifest_domain_utf8") != shard_domain
        or model.get("shard_manifest_sha256") != shard_sha256
        or shard_sha256 != manifest["acquisition_model_lock"]["shard_manifest_sha256"]
        or model.get("retained_descriptor_stamp_domain_utf8") != retained_domain
        or model.get("retained_descriptor_stamp_sha256") != retained_sha256
    ):
        raise RuntimeError("local retained model stamp binding")

    device = require_dict(report["device"], "local.device")
    require_exact_keys(
        device,
        {"name", "registry_id", "max_threadgroup_memory_bytes"},
        "local.device",
    )
    device_name = require_string(device["name"], "local device name")
    registry_id = require_int(device["registry_id"], "local device registry ID")
    if (
        not device_name
        or registry_id < 0
        or require_int(
            device["max_threadgroup_memory_bytes"], "local threadgroup memory"
        )
        <= 0
    ):
        raise RuntimeError("local device identity")
    runtime = require_dict(report["runtime"], "local.runtime")
    require_exact_keys(
        runtime,
        {
            "forward_limit",
            "qsa_physical_capacity",
            "observed_weight_bytes",
            "observed_session_bytes",
            "admission",
        },
        "local.runtime",
    )
    if (
        require_int(runtime["forward_limit"], "local runtime forward limit") != 4195
        or min(
            require_int(runtime[key], f"local runtime {key}")
            for key in (
                "qsa_physical_capacity",
                "observed_weight_bytes",
                "observed_session_bytes",
            )
        )
        <= 0
    ):
        raise RuntimeError("local runtime capacity")
    admission = require_dict(runtime["admission"], "local admission")
    require_exact_keys(
        admission,
        {"aggregate_admitted", "weights_admitted", "session_admitted"},
        "local admission",
    )
    for value in admission.values():
        require_bool(value, "local admission flag")
    expected_scoring = {
        "vocab_size": VOCAB_SIZE,
        "logits": "all F32 values must be finite",
        "nll": "max-subtracted F64 logsumexp over the complete row",
        "argmax_tie_policy": "lowest token ID",
        "natural_forwards": (
            "score each of 96 current rows, then feed that target exactly once; terminal row is unscored"
        ),
        "retrieval_exact": (
            "answer from generated token zero followed immediately by a producer stop"
        ),
    }
    if not json_equal_exact(report["scoring_contract"], expected_scoring):
        raise RuntimeError("local scoring contract")

    expected_domain = (
        "qwen4exp-selected-quality-local-abc-evidence-v1\0"
        f"packet_id={PACKET_ID}\n"
        f"fixture_manifest_sha256={MANIFEST_SHA256}\n"
        f"source_commit={source['source_commit']}\n"
        f"tracked_diff_sha256={source['tracked_diff_sha256']}\n"
        f"kernel_source_manifest_sha256={kernel_sha256}\n"
        f"path_dependencies_sha256={path_dependencies_sha256}\n"
        f"test_executable_sha256={test_executable['sha256']}\n"
        f"scorer_source_sha256={scorer['sha256']}\n"
        f"embedded_metallib_sha256={metallib['sha256']}\n"
        f"arithmetic_policy_sha256={arithmetic_sha256}\n"
        f"model_shard_manifest_sha256={model['shard_manifest_sha256']}\n"
        f"model_retained_stamps_sha256={retained_sha256}\n"
        f"device_name={device_name}\n"
        f"device_registry_id={registry_id}\n"
    )
    if implementation["evidence_domain_utf8"] != expected_domain or implementation[
        "evidence_binding_sha256"
    ] != sha256(expected_domain.encode()):
        raise RuntimeError("local global evidence domain")
    return sha256(expected_domain.encode())


def validate_local_report(
    value: object,
    manifest: dict[str, object],
    fixtures: dict[str, dict[str, object]],
    tokens: dict[str, list[int]],
) -> dict[str, object]:
    report = require_dict(value, "local")
    require_exact_keys(
        report,
        {
            "schema",
            "schema_version",
            "packet_id",
            "status",
            "disposition",
            "fixture_manifest",
            "implementation",
            "model",
            "tokenizer",
            "device",
            "runtime",
            "scoring_contract",
            "observations",
        },
        "local",
    )
    if (
        report["schema"] != "qwen4exp-selected-quality-local-abc-evidence"
        or require_int(report["schema_version"], "local schema version") != 1
        or report["packet_id"] != PACKET_ID
        or report["status"] != "local_abc_acquired_unanalyzed"
        or report["disposition"] is not None
    ):
        raise RuntimeError("local evidence identity")
    fixture_manifest = require_dict(
        report["fixture_manifest"], "local.fixture_manifest"
    )
    require_exact_keys(
        fixture_manifest, {"path", "bytes", "sha256"}, "local.fixture_manifest"
    )
    if (
        fixture_manifest["path"]
        != "docs/bench/2026-08-28-qwen4exp-selected-quality-prereg/fixtures.json"
        or require_int(fixture_manifest["bytes"], "local fixture bytes") <= 0
        or fixture_manifest["sha256"] != MANIFEST_SHA256
    ):
        raise RuntimeError("local fixture manifest identity")
    implementation = require_dict(report["implementation"], "local.implementation")
    require_exact_keys(
        implementation,
        {
            "source",
            "test_executable",
            "scorer_source",
            "embedded_metallib",
            "arithmetic_policy",
            "arithmetic_policy_json_compact",
            "arithmetic_policy_sha256",
            "evidence_domain_utf8",
            "evidence_binding_sha256",
            "ordered_run_bindings",
            "ordered_run_binding_root_sha256",
        },
        "local.implementation",
    )
    evidence_binding = validate_local_global_provenance(report, manifest)
    model = require_dict(report["model"], "local.model")
    if (
        model.get("repository") != manifest["acquisition_model_lock"]["repository"]
        or model.get("revision") != manifest["acquisition_model_lock"]["revision"]
        or model.get("quant") != manifest["acquisition_model_lock"]["quant"]
        or model.get("shard_manifest_sha256")
        != manifest["acquisition_model_lock"]["shard_manifest_sha256"]
        or model.get("mapped_bytes_hashed_directly") is not True
        or model.get("retained_stamps_revalidated_after_hashing") is not True
        or model.get("config_equals_flash_next_reference") is not True
    ):
        raise RuntimeError("local model contract")
    model_rows = validate_model_rows(
        model.get("shards"), manifest["acquisition_model_lock"], "local.model.shards"
    )
    tokenizer = require_dict(report["tokenizer"], "local.tokenizer")
    require_exact_keys(
        tokenizer,
        {
            "identity_schema",
            "identity_sha256",
            "vocab_size",
            "producer_stop_token_ids",
            "model",
            "pretokenizer",
            "chat_template_sha256",
            "qualification_scope",
        },
        "local.tokenizer",
    )
    expected_tokenizer = {
        "identity_schema": manifest["tokenizer"]["identity_schema"],
        "identity_sha256": manifest["tokenizer"]["identity_sha256"],
        "vocab_size": VOCAB_SIZE,
        "producer_stop_token_ids": [PRODUCER_STOP_TOKEN_ID],
        "model": manifest["tokenizer"]["model"],
        "pretokenizer": manifest["tokenizer"]["pretokenizer"],
        "chat_template_sha256": manifest["tokenizer"]["chat_template_sha256"],
    }
    for key, expected in expected_tokenizer.items():
        if not json_equal_exact(tokenizer.get(key), expected):
            raise RuntimeError(f"local.tokenizer.{key}")
    if tokenizer["qualification_scope"] != (
        "released-artifact evidence only; not a production admission rule"
    ):
        raise RuntimeError("local.tokenizer.qualification_scope")

    observations = require_dict(report["observations"], "local.observations")
    expected_counts = {
        "scope_control": 3,
        "natural_semantic": 36,
        "open_greedy": 12,
        "retrieval_semantic": 24,
        "reverse_replay": 3,
    }
    require_exact_keys(observations, set(expected_counts), "local.observations")
    by_ordinal: dict[int, dict[str, object]] = {}
    phase_for_ordinal: dict[int, str] = {}
    for phase, expected_count in expected_counts.items():
        rows = require_list(observations[phase], f"local.observations.{phase}")
        if len(rows) != expected_count:
            raise RuntimeError(f"local.observations.{phase} count")
        for value in rows:
            row = require_dict(value, f"local observation {phase}")
            ordinal = require_int(
                row.get("operation_ordinal"), "local operation ordinal"
            )
            if ordinal in by_ordinal:
                raise RuntimeError(f"duplicate local operation ordinal {ordinal}")
            by_ordinal[ordinal] = row
            phase_for_ordinal[ordinal] = phase
    if set(by_ordinal) != set(range(LOCAL_OPERATION_COUNT)):
        raise RuntimeError("local operation ordinal set")

    plan = manifest["execution"]["operation_plan"]
    if len(plan) != TOTAL_OPERATION_COUNT:
        raise RuntimeError("fixture operation plan count")
    semantic: dict[str, dict[str, dict[str, object]]] = {}
    retrieval: dict[str, dict[str, dict[str, object]]] = {}
    open_greedy: dict[str, dict[str, dict[str, object]]] = {}
    scope: dict[str, dict[str, object]] = {}
    reverse: dict[str, dict[str, object]] = {}
    observed_bindings: list[dict[str, object]] = []

    for ordinal, operation_value in enumerate(plan[:LOCAL_OPERATION_COUNT]):
        operation = require_dict(operation_value, f"operation plan {ordinal}")
        row = by_ordinal[ordinal]
        phase = str(operation["phase"])
        if phase_for_ordinal[ordinal] != phase:
            raise RuntimeError(f"local operation {ordinal} phase")
        fixture_id = str(operation["fixture_id"])
        fixture = fixtures[fixture_id]
        arm = str(operation["arm"])
        if arm not in ARMS:
            raise RuntimeError(f"local operation {ordinal} arm")
        common = {
            "operation_ordinal": ordinal,
            "fixture_id": fixture_id,
            "arm": arm,
            "arm_label": ARM_LABELS[arm],
            "mode": operation["mode"],
        }
        for key, expected in common.items():
            if row.get(key) != expected:
                raise RuntimeError(f"local operation {ordinal}.{key}")
        prompt_count = (
            len(tokens[fixture_id])
            if phase == "retrieval_semantic"
            else int(fixture["prompt_token_count"])
        )
        validate_prefill(
            row["prefill"], arm, prompt_count, f"local operation {ordinal}.prefill"
        )

        if phase == "scope_control":
            require_exact_keys(
                row,
                {
                    "operation_ordinal",
                    "fixture_id",
                    "arm",
                    "arm_label",
                    "mode",
                    "prefill",
                    "replay",
                    "binding",
                },
                f"local operation {ordinal}",
            )
            replay = validate_replay(
                row["replay"],
                arm,
                prompt_count,
                prompt_count,
                1,
                f"local operation {ordinal}.replay",
            )
            scope[arm] = {"observation": row, "replay": replay}
        elif phase in {"natural_semantic", "reverse_replay"}:
            keys = {
                "operation_ordinal",
                "fixture_id",
                "document_source_ordinal",
                "prompt_token_count",
                "selected_suffix_tokens",
                "arm",
                "arm_label",
                "mode",
                "prefill",
                "continuation",
                "replay",
                "binding",
            }
            if phase == "reverse_replay":
                keys.add("matches_initial_semantic_replay")
            require_exact_keys(row, keys, f"local operation {ordinal}")
            if (
                row["document_source_ordinal"] != fixture["document"]["source_ordinal"]
                or row["prompt_token_count"] != prompt_count
                or row["selected_suffix_tokens"] != fixture["selected_suffix_tokens"]
            ):
                raise RuntimeError(f"local operation {ordinal} natural fixture scalars")
            expected_tokens = tokens[fixture_id][prompt_count:]
            scored = validate_local_score_rows(
                require_dict(
                    row["continuation"], f"local operation {ordinal}.continuation"
                ),
                expected_tokens,
                f"local operation {ordinal}.continuation",
            )
            replay = validate_replay(
                row["replay"],
                arm,
                prompt_count,
                prompt_count + 96,
                97,
                f"local operation {ordinal}.replay",
            )
            record = {"observation": row, "scores": scored, "replay": replay}
            if phase == "natural_semantic":
                semantic.setdefault(fixture_id, {})[arm] = record
            else:
                if row["matches_initial_semantic_replay"] is not True:
                    raise RuntimeError(f"local operation {ordinal} reverse replay flag")
                reverse[arm] = record
        elif phase == "open_greedy":
            require_exact_keys(
                row,
                {
                    "operation_ordinal",
                    "fixture_id",
                    "prompt_token_count",
                    "arm",
                    "arm_label",
                    "mode",
                    "prefill",
                    "greedy",
                    "replay",
                    "binding",
                },
                f"local operation {ordinal}",
            )
            greedy = require_dict(row["greedy"], f"local operation {ordinal}.greedy")
            require_exact_keys(
                greedy,
                {
                    "tie_policy",
                    "maximum_tokens",
                    "generated_token_ids",
                    "generated_token_ids_sha256_u32le",
                    "fed_non_stop_tokens",
                    "stopped_on_producer_token",
                },
                f"local operation {ordinal}.greedy",
            )
            generated = require_list(greedy["generated_token_ids"], "open greedy IDs")
            if (
                greedy["tie_policy"] != "lowest_token_id"
                or greedy["maximum_tokens"] != 32
                or not 1 <= len(generated) <= 32
                or not all(
                    not isinstance(token, bool)
                    and isinstance(token, int)
                    and 0 <= token < VOCAB_SIZE
                    for token in generated
                )
            ):
                raise RuntimeError(f"local operation {ordinal} greedy contract")
            if greedy["generated_token_ids_sha256_u32le"] != sha256_u32(
                b"qwen4exp-selected-quality-open-greedy-u32le-v1\0", generated
            ):
                raise RuntimeError(f"local operation {ordinal} greedy hash")
            stopped = require_bool(
                greedy["stopped_on_producer_token"], "open greedy stop"
            )
            fed = require_int(greedy["fed_non_stop_tokens"], "open greedy fed tokens")
            expected_fed = len(generated) - int(stopped)
            stop_positions = [
                index
                for index, token in enumerate(generated)
                if token == PRODUCER_STOP_TOKEN_ID
            ]
            expected_stop_positions = [len(generated) - 1] if stopped else []
            if fed != expected_fed or stop_positions != expected_stop_positions:
                raise RuntimeError(f"local operation {ordinal} greedy feed contract")
            replay = validate_replay(
                row["replay"],
                arm,
                prompt_count,
                prompt_count + fed,
                fed + 1,
                f"local operation {ordinal}.replay",
            )
            open_greedy.setdefault(fixture_id, {})[arm] = {
                "observation": row,
                "generated": generated,
                "replay": replay,
            }
        elif phase == "retrieval_semantic":
            require_exact_keys(
                row,
                {
                    "operation_ordinal",
                    "fixture_id",
                    "kind",
                    "document_source_ordinal",
                    "prompt_token_count",
                    "selected_suffix_tokens",
                    "arm",
                    "arm_label",
                    "mode",
                    "prefill",
                    "answer",
                    "replay",
                    "binding",
                },
                f"local operation {ordinal}",
            )
            if (
                row["kind"] != fixture["kind"]
                or row["document_source_ordinal"]
                != fixture["document"]["source_ordinal"]
                or row["prompt_token_count"] != 4099
                or row["selected_suffix_tokens"] != 2048
            ):
                raise RuntimeError(
                    f"local operation {ordinal} retrieval fixture scalars"
                )
            answer = validate_retrieval_answer(
                require_dict(row["answer"], f"local operation {ordinal}.answer"),
                fixture,
                f"local operation {ordinal}.answer",
            )
            replay = validate_replay(
                row["replay"],
                arm,
                4099,
                4099 + len(fixture["answer_token_ids"]),
                len(fixture["answer_token_ids"]) + 1,
                f"local operation {ordinal}.replay",
            )
            retrieval.setdefault(fixture_id, {})[arm] = {
                "observation": row,
                "answer": answer,
                "replay": replay,
            }
        else:
            raise RuntimeError(f"unknown local phase {phase}")

        binding_sha256 = validate_local_binding(
            row,
            operation,
            input_sha256_for_operation(fixture),
            evidence_binding,
            f"local operation {ordinal}",
        )
        observed_bindings.append(
            {
                "operation_ordinal": ordinal,
                "fixture_id": fixture_id,
                "phase": phase,
                "mode": operation["mode"],
                "arm": arm,
                "run_binding_sha256": binding_sha256,
            }
        )

    ordered = require_list(
        implementation["ordered_run_bindings"], "local ordered run bindings"
    )
    if (
        not json_equal_exact(ordered, observed_bindings)
        or run_binding_root(observed_bindings)
        != implementation["ordered_run_binding_root_sha256"]
    ):
        raise RuntimeError("local ordered run binding root")
    if len(semantic) != 12 or any(set(arms) != set(ARMS) for arms in semantic.values()):
        raise RuntimeError("local natural semantic arm coverage")
    if len(retrieval) != 8 or any(
        set(arms) != set(ARMS) for arms in retrieval.values()
    ):
        raise RuntimeError("local retrieval arm coverage")
    if len(open_greedy) != 4 or any(
        set(arms) != set(ARMS) for arms in open_greedy.values()
    ):
        raise RuntimeError("local open greedy arm coverage")
    if set(scope) != set(ARMS) or set(reverse) != set(ARMS):
        raise RuntimeError("local scope/reverse arm coverage")
    scope_replays = [scope[arm]["replay"] for arm in ARMS]
    if not all(replay == scope_replays[0] for replay in scope_replays[1:]):
        raise RuntimeError("scope control is not bit-identical")
    reverse_fixture = next(
        str(fixture["fixture_id"])
        for fixture in manifest["natural_fixtures"]
        if fixture["reverse_replay"]
    )
    for arm in ARMS:
        if reverse[arm]["replay"] != semantic[reverse_fixture][arm]["replay"]:
            raise RuntimeError(f"reverse replay mismatch for arm {arm}")
    for fixture_id, arms in open_greedy.items():
        for arm in ARMS:
            if (
                arms[arm]["replay"]["endpoint"]
                != semantic[fixture_id][arm]["replay"]["endpoint"]
            ):
                raise RuntimeError(
                    f"open greedy endpoint mismatch for {fixture_id} {arm}"
                )
    return {
        "report": report,
        "semantic": semantic,
        "retrieval": retrieval,
        "open_greedy": open_greedy,
        "scope": scope,
        "reverse": reverse,
        "model_rows": model_rows,
        "ordered_run_binding_root_sha256": implementation[
            "ordered_run_binding_root_sha256"
        ],
    }


def validate_llama_evidence(
    value: object,
    manifest: dict[str, object],
    fixtures: dict[str, dict[str, object]],
    tokens: dict[str, list[int]],
    expected_operations: list[tuple[str, str, str]],
) -> dict[str, object]:
    report = require_dict(value, "llama evidence")
    require_exact_keys(
        report,
        {
            "schema",
            "schema_version",
            "packet_id",
            "status",
            "disposition",
            "fixture_manifest",
            "environment",
            "source",
            "isolated_build",
            "core_execution",
            "model",
            "host",
            "evidence_domain_utf8",
            "evidence_binding_sha256",
            "core_output",
        },
        "llama evidence",
    )
    if (
        report["schema"] != "qwen4exp-selected-quality-llama-evidence"
        or require_int(report["schema_version"], "llama schema version") != 1
        or report["packet_id"] != PACKET_ID
        or report["status"] != "llama_cpp_d_acquired_unanalyzed"
        or report["disposition"] is not None
    ):
        raise RuntimeError("llama evidence identity")
    fixture_manifest = require_dict(
        report["fixture_manifest"], "llama fixture manifest"
    )
    require_exact_keys(
        fixture_manifest, {"path", "bytes", "sha256"}, "llama fixture manifest"
    )
    if (
        not require_string(fixture_manifest["path"], "llama fixture path").endswith(
            "/docs/bench/2026-08-28-qwen4exp-selected-quality-prereg/fixtures.json"
        )
        or require_int(fixture_manifest["bytes"], "llama fixture bytes") <= 0
        or fixture_manifest["sha256"] != MANIFEST_SHA256
    ):
        raise RuntimeError("llama fixture manifest identity")
    source = require_dict(report["source"], "llama source")
    isolated_build = require_dict(report["isolated_build"], "llama isolated build")
    require_exact_keys(
        source,
        {
            "qwen_source_commit",
            "qwen_tracked_diff_sha256",
            "required_head_paths",
            "runner_sources",
            "runner_source_manifest_sha256",
            "core_build_expected",
            "llama_cpp",
        },
        "llama source",
    )
    if (
        not is_lower_hex(source["qwen_source_commit"], 40)
        or source["qwen_tracked_diff_sha256"]
        != "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        or "scripts/bench/qwen4exp_selected_quality_analyze.py"
        not in source["required_head_paths"]
    ):
        raise RuntimeError("llama source identity")
    runner_sources = require_list(source["runner_sources"], "llama runner sources")
    source_domain = bytearray(b"qwen4exp-selected-quality-llama-runner-sources-v1\0")
    for index, value in enumerate(runner_sources):
        row = require_dict(value, f"llama runner source {index}")
        require_exact_keys(
            row,
            {"path", "bytes", "sha256", "stamp"},
            f"llama runner source {index}",
        )
        if require_int(
            row["bytes"], "llama runner source bytes"
        ) <= 0 or not is_lower_hex(row["sha256"]):
            raise RuntimeError("llama runner source identity")
        stamp = validate_file_stamp(row["stamp"], f"llama runner source {index} stamp")
        if stamp["bytes"] != row["bytes"]:
            raise RuntimeError("llama runner source stamp bytes")
        source_domain.extend(
            f"{row['path']}\t{row['bytes']}\t{row['sha256']}\n".encode()
        )
    if source["runner_source_manifest_sha256"] != sha256(bytes(source_domain)):
        raise RuntimeError("llama runner source manifest")
    llama_cpp_source = require_dict(source["llama_cpp"], "llama pinned source")
    require_exact_keys(
        llama_cpp_source,
        {
            "repository",
            "commit",
            "tree",
            "object_format",
            "support_pull_request",
            "scoped_status_sha256",
        },
        "llama pinned source",
    )
    if (
        llama_cpp_source["repository"] != "ggml-org/llama.cpp"
        or llama_cpp_source["commit"] != "6c84c7d5d8833c6e0df69628f75a0f599797934e"
        or llama_cpp_source["support_pull_request"] != 27742
        or llama_cpp_source["object_format"] not in {"sha1", "sha256"}
        or llama_cpp_source["scoped_status_sha256"]
        != "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    ):
        raise RuntimeError("llama pinned source contract")
    require_exact_keys(
        isolated_build,
        {
            "schema",
            "policy",
            "policy_bytes",
            "policy_sha256",
            "llama_cpp_export",
            "configure",
            "build",
            "cmake_cache",
            "compile_commands",
            "ninja_commands",
            "build_ninja",
            "dynamic_linkage",
            "core_build_expected",
            "core_executable",
            "source_and_core_revalidated_after_acquisition",
        },
        "llama isolated build",
    )
    policy = require_dict(isolated_build["policy"], "llama build policy")
    policy_bytes = canonical_json(policy)
    if (
        isolated_build["schema"] != "qwen4exp-selected-quality-llama-isolated-build-v1"
        or require_int(isolated_build["policy_bytes"], "llama policy bytes")
        != len(policy_bytes)
        or isolated_build["policy_sha256"] != sha256(policy_bytes)
        or isolated_build["source_and_core_revalidated_after_acquisition"] is not True
    ):
        raise RuntimeError("llama isolated build policy binding")
    validate_d_build_details(isolated_build, source)
    core_executable = require_dict(
        isolated_build["core_executable"], "llama core executable"
    )
    require_exact_keys(
        core_executable,
        {"path", "bytes", "sha256", "stamp", "linkage"},
        "llama core executable",
    )
    core_executable_path = require_string(
        core_executable["path"], "llama core executable path"
    )
    core_executable_stamp = validate_file_stamp(
        core_executable["stamp"], "llama core executable stamp"
    )
    if require_int(
        core_executable["bytes"], "llama core executable bytes"
    ) <= 0 or not is_lower_hex(core_executable["sha256"]):
        raise RuntimeError("llama core executable identity")
    if (
        core_executable_stamp["path"] != core_executable_path
        or core_executable_stamp["bytes"] != core_executable["bytes"]
        or core_executable["linkage"]
        != "static llama.cpp/ggml; allowlisted system libraries only"
    ):
        raise RuntimeError("llama core executable stamp")
    core_execution = require_dict(report["core_execution"], "llama core execution")
    require_exact_keys(
        core_execution,
        {
            "strategy",
            "source_path",
            "source_stamp",
            "bytes",
            "sha256",
            "stamp_before_execution",
            "stamp_after_execution",
            "revalidated_after_execution",
        },
        "llama core execution",
    )
    if (
        core_execution["bytes"] != core_executable["bytes"]
        or core_execution["sha256"] != core_executable["sha256"]
        or core_execution["revalidated_after_execution"] is not True
        or core_execution["strategy"]
        != "exclusive byte copy, fsync, chmod 0500, execute copied inode"
        or core_execution["source_path"] != core_executable_path
        or not json_equal_exact(core_execution["source_stamp"], core_executable_stamp)
        or not json_equal_exact(
            core_execution["stamp_before_execution"],
            core_execution["stamp_after_execution"],
        )
    ):
        raise RuntimeError("llama executed core binding")
    validate_file_stamp(
        core_execution["stamp_before_execution"], "llama staged core stamp"
    )
    environment = require_dict(report["environment"], "llama environment")
    require_exact_keys(
        environment,
        {
            "forbidden_prefixes",
            "forbidden_build_variables",
            "undeclared_overrides_rejected",
        },
        "llama environment",
    )
    if (
        environment["undeclared_overrides_rejected"] is not True
        or environment["forbidden_prefixes"]
        != ["CMAKE_", "DYLD_", "GGML_", "LLAMA_", "METAL_", "MTL_", "QWEN"]
        or environment["forbidden_build_variables"]
        != sorted(llama_support.FORBIDDEN_BUILD_ENVIRONMENT)
    ):
        raise RuntimeError("llama environment override policy")
    host = require_dict(report["host"], "llama host")
    require_exact_keys(host, {"platform", "machine", "python"}, "llama host")
    for key, host_value in host.items():
        if not require_string(host_value, f"llama host {key}"):
            raise RuntimeError("empty llama host identity")
    model = require_dict(report["model"], "llama model")
    require_exact_keys(
        model,
        {
            "repository",
            "revision",
            "quant",
            "shard_manifest_sha256",
            "qualification",
            "locked_first_shard_path",
            "shards",
        },
        "llama model",
    )
    if (
        model.get("repository") != manifest["acquisition_model_lock"]["repository"]
        or model.get("revision") != manifest["acquisition_model_lock"]["revision"]
        or model.get("quant") != manifest["acquisition_model_lock"]["quant"]
        or model.get("shard_manifest_sha256")
        != manifest["acquisition_model_lock"]["shard_manifest_sha256"]
    ):
        raise RuntimeError("llama model contract")
    model_rows = validate_model_rows(
        model.get("shards"), manifest["acquisition_model_lock"], "llama model shards"
    )
    shard_values = require_list(model["shards"], "llama model shards")
    for index, value in enumerate(shard_values):
        shard = require_dict(value, f"llama model shard {index}")
        require_exact_keys(
            shard,
            {"index", "file_name", "bytes", "sha256", "stamp"},
            f"llama model shard {index}",
        )
        stamp = validate_file_stamp(shard["stamp"], f"llama model shard {index} stamp")
        if stamp["bytes"] != shard["bytes"]:
            raise RuntimeError(f"llama model shard {index} stamp bytes")
    canonical_paths = [
        Path(
            require_string(
                require_dict(
                    require_dict(row, "llama model shard")["stamp"],
                    "llama model shard stamp",
                )["path"],
                "llama model shard path",
            )
        )
        for row in shard_values
    ]
    if (
        model["locked_first_shard_path"] != str(canonical_paths[0])
        or any(not path.is_absolute() for path in canonical_paths)
        or model["qualification"]
        != "every local shard was hashed in full before llama.cpp execution"
    ):
        raise RuntimeError("llama model path binding")
    core_output = require_dict(report["core_output"], "llama core output")
    require_exact_keys(
        core_output,
        {"bytes", "sha256", "raw_json_utf8", "report"},
        "llama core output",
    )
    if require_int(
        core_output["bytes"], "llama core output bytes"
    ) <= 0 or not is_lower_hex(core_output["sha256"]):
        raise RuntimeError("llama core output identity")
    raw_core = require_string(
        core_output["raw_json_utf8"], "llama raw core JSON"
    ).encode()
    if (
        len(raw_core) != core_output["bytes"]
        or sha256(raw_core) != core_output["sha256"]
        or not json_equal_exact(parse_json_strict(raw_core), core_output["report"])
    ):
        raise RuntimeError("llama raw core JSON binding")
    core = validate_core_report(
        core_output["report"],
        manifest,
        tokens,
        fixtures,
        expected_operations,
        isolated_build,
        canonical_paths,
    )
    expected_domain = (
        "qwen4exp-selected-quality-llama-evidence-v1\0"
        f"packet_id={PACKET_ID}\n"
        f"fixture_manifest_sha256={MANIFEST_SHA256}\n"
        f"qwen_source_commit={source['qwen_source_commit']}\n"
        f"runner_source_manifest_sha256={source['runner_source_manifest_sha256']}\n"
        f"llama_cpp_tree={source['llama_cpp']['tree']}\n"
        f"llama_cpp_export_manifest_sha256={isolated_build['llama_cpp_export']['tree_manifest']['manifest_sha256']}\n"
        f"build_policy_sha256={isolated_build['policy_sha256']}\n"
        f"core_executable_sha256={isolated_build['core_executable']['sha256']}\n"
        f"llama_cpp_commit={source['llama_cpp']['commit']}\n"
        f"model_shard_manifest_sha256={model['shard_manifest_sha256']}\n"
        f"core_output_sha256={core_output['sha256']}\n"
        f"core_semantic_payload_sha256={core['binding']['semantic_payload_sha256']}\n"
    )
    if report["evidence_domain_utf8"] != expected_domain or report[
        "evidence_binding_sha256"
    ] != sha256(expected_domain.encode()):
        raise RuntimeError("llama outer evidence binding")
    natural: dict[str, dict[str, object]] = {}
    retrieval: dict[str, dict[str, object]] = {}
    for operation_value in core["operations"]:
        operation = require_dict(operation_value, "llama core operation")
        fixture_id = str(operation["fixture_id"])
        if operation["mode"] == "teacher_forced_nll_96":
            continuation = require_dict(operation["continuation"], "llama continuation")
            natural[fixture_id] = {
                "operation": operation,
                "nll_values": [
                    require_number(row["nll_f64"], "llama natural NLL")
                    for row in continuation["scored_rows"]
                ],
                "nll_sum": require_number(
                    continuation["nll_sum_f64"], "llama natural NLL sum"
                ),
                "mean_nll": require_number(
                    continuation["mean_nll_f64"], "llama natural mean NLL"
                ),
                "top1_hits": require_int(
                    continuation["top1_hits"], "llama natural top1 hits"
                ),
                "predictions": [
                    require_int(row["argmax_token_id"], "llama natural prediction")
                    for row in continuation["scored_rows"]
                ],
            }
        else:
            answer = require_dict(operation["answer"], "llama retrieval answer")
            retrieval[fixture_id] = {
                "operation": operation,
                "nll_values": [
                    require_number(row["nll_f64"], "llama retrieval NLL")
                    for row in answer["scored_rows"]
                ],
                "nll_sum": require_number(
                    answer["nll_sum_f64"], "llama answer NLL sum"
                ),
                "mean_nll": require_number(
                    answer["mean_nll_f64"], "llama answer mean NLL"
                ),
                "predictions": [
                    require_int(row["argmax_token_id"], "llama retrieval prediction")
                    for row in answer["scored_rows"]
                ],
                "greedy_prefix": answer["greedy_prefix_through_first_mismatch_or_stop"],
                "exact_pass": require_bool(answer["exact_pass"], "llama exact pass"),
            }
    if len(natural) != 12 or len(retrieval) != 8:
        raise RuntimeError("llama semantic fixture coverage")
    return {
        "report": report,
        "core": core,
        "natural": natural,
        "retrieval": retrieval,
        "model_rows": model_rows,
    }


class SplitMix64:
    def __init__(self, seed: int) -> None:
        self.state = seed & ((1 << 64) - 1)

    def next(self) -> int:
        mask = (1 << 64) - 1
        self.state = (self.state + 0x9E3779B97F4A7C15) & mask
        value = self.state
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
        return (value ^ (value >> 31)) & mask

    def bounded_three(self) -> int:
        acceptance_limit = (1 << 64) - 1
        while True:
            value = self.next()
            if value < acceptance_limit:
                return value % 3


def nearest_rank(values: list[float], probability: float) -> float:
    if not values:
        raise RuntimeError("empty bootstrap distribution")
    ordered = sorted(values)
    index = math.ceil(probability * len(ordered)) - 1
    return ordered[index]


def compute_bootstrap(
    document_rows: list[dict[str, object]],
    manifest: dict[str, object],
) -> dict[str, object]:
    probe = SplitMix64(0)
    if [probe.next() for _ in range(4)] != [
        0xE220A8397B1DCDAF,
        0x6E789E6AA1B965F4,
        0x06C45D188009454F,
        0xF88BB8A8724C81EC,
    ]:
        raise RuntimeError("SplitMix64 implementation does not match golden vectors")
    contract = manifest["execution"]["bootstrap"]
    shapes = [int(shape) for shape in contract["strata"]]
    draws = int(contract["draws"])
    seed = int(contract["seed_u64"])
    by_shape: dict[int, list[dict[str, object]]] = {shape: [] for shape in shapes}
    for row in document_rows:
        by_shape[int(row["prompt_token_count"])].append(row)
    if any(len(by_shape[shape]) != 3 for shape in shapes):
        raise RuntimeError("bootstrap document strata")

    generator = SplitMix64(seed)
    matrix_digest = hashlib.sha256(
        b"qwen4exp-selected-quality-bootstrap-indices-u8-v1\0"
    )
    draw_digest = hashlib.sha256(
        b"qwen4exp-selected-quality-bootstrap-contrasts-f64le-v1\0"
    )
    distributions: dict[str, list[float]] = {contrast: [] for contrast in CONTRASTS}
    for _ in range(draws):
        sums = {contrast: 0.0 for contrast in CONTRASTS}
        sampled = 0
        for shape in shapes:
            for _ in range(3):
                index = generator.bounded_three()
                matrix_digest.update(bytes([index]))
                row = by_shape[shape][index]
                for contrast in CONTRASTS:
                    sums[contrast] += float(row["deltas"][contrast])
                sampled += 1
        if sampled != 12:
            raise RuntimeError("bootstrap sampled document count")
        for contrast in CONTRASTS:
            value = sums[contrast] / sampled
            distributions[contrast].append(value)
            draw_digest.update(struct.pack("<d", value))

    quantiles = {
        "B-A": 0.975,
        "C-A": 0.975,
        "C-B": 0.95,
    }
    matrix_sha256 = matrix_digest.hexdigest()
    if (
        matrix_sha256
        != "d27fd59792dac2d61b33452cd4143bf555578c11ab73d54e5f0dde5bb93e41b3"
    ):
        raise RuntimeError("frozen bootstrap resample matrix drift")
    return {
        "draws": draws,
        "seed_u64": seed,
        "rng": contract["rng"],
        "draw_order": contract["draw_order"],
        "bounded_mapping": contract["bounded_mapping"],
        "quantile_convention": contract["quantile_convention"],
        "contrast_order": list(CONTRASTS),
        "shared_resample_matrix_sha256_u8": matrix_sha256,
        "ordered_contrast_draws_sha256_f64le": draw_digest.hexdigest(),
        "contrasts": {
            contrast: {
                "one_sided_probability": quantiles[contrast],
                "upper_bound_nats_per_token": nearest_rank(
                    distributions[contrast], quantiles[contrast]
                ),
                "minimum_draw": min(distributions[contrast]),
                "maximum_draw": max(distributions[contrast]),
            }
            for contrast in CONTRASTS
        },
    }


def first_divergence(left: list[int], right: list[int]) -> int | None:
    for index, (left_token, right_token) in enumerate(zip(left, right, strict=False)):
        if left_token != right_token:
            return index
    if len(left) != len(right):
        return min(len(left), len(right))
    return None


def compute_natural_metrics(
    local: dict[str, object],
    llama: dict[str, object],
    manifest: dict[str, object],
) -> dict[str, object]:
    documents: list[dict[str, object]] = []
    shape_accumulator: dict[int, list[dict[str, object]]] = {}
    total_prediction_agreements = 0
    total_predictions = 0
    d_minus_a_nll = 0.0
    for fixture in manifest["natural_fixtures"]:
        fixture_id = str(fixture["fixture_id"])
        prompt_tokens = int(fixture["prompt_token_count"])
        local_arms = local["semantic"][fixture_id]
        d = llama["natural"][fixture_id]
        arms = {
            arm: {
                "mean_nll_f64": local_arms[arm]["scores"]["mean_nll"],
                "nll_sum_f64": local_arms[arm]["scores"]["nll_sum"],
                "top1_hits": local_arms[arm]["scores"]["top1_hits"],
            }
            for arm in ARMS
        }
        arms["D"] = {
            "mean_nll_f64": d["mean_nll"],
            "nll_sum_f64": d["nll_sum"],
            "top1_hits": d["top1_hits"],
        }
        deltas = {
            "B-A": arms["B"]["mean_nll_f64"] - arms["A"]["mean_nll_f64"],
            "C-A": arms["C"]["mean_nll_f64"] - arms["A"]["mean_nll_f64"],
            "C-B": arms["C"]["mean_nll_f64"] - arms["B"]["mean_nll_f64"],
            "D-A": arms["D"]["mean_nll_f64"] - arms["A"]["mean_nll_f64"],
        }
        top1_net = {
            "B-A": arms["B"]["top1_hits"] - arms["A"]["top1_hits"],
            "C-A": arms["C"]["top1_hits"] - arms["A"]["top1_hits"],
            "D-A": arms["D"]["top1_hits"] - arms["A"]["top1_hits"],
        }
        a_predictions = local_arms["A"]["scores"]["predictions"]
        d_predictions = d["predictions"]
        agreement = sum(
            left == right
            for left, right in zip(a_predictions, d_predictions, strict=True)
        )
        total_prediction_agreements += agreement
        total_predictions += len(a_predictions)
        d_minus_a_nll += float(deltas["D-A"])
        row = {
            "fixture_id": fixture_id,
            "document_source_ordinal": fixture["document"]["source_ordinal"],
            "prompt_token_count": prompt_tokens,
            "selected_suffix_tokens": fixture["selected_suffix_tokens"],
            "continuation_tokens": 96,
            "arms": arms,
            "deltas": deltas,
            "top1_net": top1_net,
            "llama_vs_a_argmax_agreement": {
                "hits": agreement,
                "tokens": len(a_predictions),
            },
        }
        documents.append(row)
        shape_accumulator.setdefault(prompt_tokens, []).append(row)

    shapes = []
    for shape in sorted(shape_accumulator):
        rows = shape_accumulator[shape]
        shapes.append(
            {
                "prompt_token_count": shape,
                "documents": len(rows),
                "arms": {
                    arm: {
                        "mean_nll_f64": sum(
                            float(row["arms"][arm]["mean_nll_f64"]) for row in rows
                        )
                        / len(rows),
                        "top1_hits": sum(
                            int(row["arms"][arm]["top1_hits"]) for row in rows
                        ),
                    }
                    for arm in (*ARMS, "D")
                },
                "deltas": {
                    contrast: sum(float(row["deltas"][contrast]) for row in rows)
                    / len(rows)
                    for contrast in (*CONTRASTS, "D-A")
                },
                "top1_net": {
                    contrast: sum(int(row["top1_net"][contrast]) for row in rows)
                    for contrast in ("B-A", "C-A", "D-A")
                },
            }
        )
    overall = {
        "documents": len(documents),
        "scored_tokens": len(documents) * 96,
        "arms": {
            arm: {
                "mean_nll_f64": sum(
                    float(row["arms"][arm]["mean_nll_f64"]) for row in documents
                )
                / len(documents),
                "top1_hits": sum(
                    int(row["arms"][arm]["top1_hits"]) for row in documents
                ),
            }
            for arm in (*ARMS, "D")
        },
        "deltas": {
            contrast: sum(float(row["deltas"][contrast]) for row in documents)
            / len(documents)
            for contrast in (*CONTRASTS, "D-A")
        },
        "top1_net": {
            contrast: sum(int(row["top1_net"][contrast]) for row in documents)
            for contrast in ("B-A", "C-A", "D-A")
        },
    }
    bootstrap = compute_bootstrap(documents, manifest)
    return {
        "documents": documents,
        "shapes": shapes,
        "overall": overall,
        "bootstrap": bootstrap,
        "llama_cpp_triangulation": {
            "authority": "descriptive same-token same-quant triangulation; not numerical truth",
            "mean_document_nll_delta_d_minus_a": d_minus_a_nll / len(documents),
            "argmax_agreement_hits": total_prediction_agreements,
            "argmax_agreement_tokens": total_predictions,
            "argmax_agreement_rate": total_prediction_agreements / total_predictions,
        },
    }


def compute_retrieval_metrics(
    local: dict[str, object],
    llama: dict[str, object],
    manifest: dict[str, object],
) -> dict[str, object]:
    tasks = []
    aggregate = {
        arm: {"nll_sum": 0.0, "tokens": 0, "task_means": [], "exact_passes": 0}
        for arm in (*ARMS, "D")
    }
    for fixture in manifest["retrieval_fixtures"]:
        fixture_id = str(fixture["fixture_id"])
        local_arms = local["retrieval"][fixture_id]
        d = llama["retrieval"][fixture_id]
        arms: dict[str, dict[str, object]] = {}
        for arm in ARMS:
            answer = local_arms[arm]["answer"]
            arms[arm] = {
                "tokens": len(answer["nll_values"]),
                "nll_sum_f64": answer["nll_sum"],
                "mean_nll_f64": answer["mean_nll"],
                "exact_pass": answer["exact_pass"],
                "greedy_prefix": answer["greedy_prefix"],
            }
        arms["D"] = {
            "tokens": len(d["nll_values"]),
            "nll_sum_f64": d["nll_sum"],
            "mean_nll_f64": d["mean_nll"],
            "exact_pass": d["exact_pass"],
            "greedy_prefix": d["greedy_prefix"],
        }
        for arm, values in arms.items():
            aggregate[arm]["nll_sum"] += float(values["nll_sum_f64"])
            aggregate[arm]["tokens"] += int(values["tokens"])
            aggregate[arm]["task_means"].append(float(values["mean_nll_f64"]))
            aggregate[arm]["exact_passes"] += int(values["exact_pass"])
        tasks.append(
            {
                "fixture_id": fixture_id,
                "kind": fixture["kind"],
                "answer_token_ids": fixture["answer_token_ids"],
                "arms": arms,
                "answer_nll_deltas": {
                    "B-A": arms["B"]["mean_nll_f64"] - arms["A"]["mean_nll_f64"],
                    "C-A": arms["C"]["mean_nll_f64"] - arms["A"]["mean_nll_f64"],
                    "C-B": arms["C"]["mean_nll_f64"] - arms["B"]["mean_nll_f64"],
                    "D-A": arms["D"]["mean_nll_f64"] - arms["A"]["mean_nll_f64"],
                },
            }
        )
    summaries = {}
    for arm, values in aggregate.items():
        summaries[arm] = {
            "answer_tokens": values["tokens"],
            "token_weighted_mean_nll_f64": values["nll_sum"] / values["tokens"],
            "unweighted_task_mean_nll_f64": sum(values["task_means"])
            / len(values["task_means"]),
            "exact_passes": values["exact_passes"],
            "tasks": len(tasks),
        }
    return {
        "tasks": tasks,
        "aggregate": summaries,
        "token_weighted_deltas": {
            "B-A": summaries["B"]["token_weighted_mean_nll_f64"]
            - summaries["A"]["token_weighted_mean_nll_f64"],
            "C-A": summaries["C"]["token_weighted_mean_nll_f64"]
            - summaries["A"]["token_weighted_mean_nll_f64"],
            "C-B": summaries["C"]["token_weighted_mean_nll_f64"]
            - summaries["B"]["token_weighted_mean_nll_f64"],
            "D-A": summaries["D"]["token_weighted_mean_nll_f64"]
            - summaries["A"]["token_weighted_mean_nll_f64"],
        },
    }


def compute_open_greedy(local: dict[str, object]) -> dict[str, object]:
    rows = []
    for fixture_id in sorted(local["open_greedy"]):
        arms = local["open_greedy"][fixture_id]
        generated = {arm: arms[arm]["generated"] for arm in ARMS}
        rows.append(
            {
                "fixture_id": fixture_id,
                "generated_token_ids": generated,
                "first_divergence": {
                    "B-A": first_divergence(generated["B"], generated["A"]),
                    "C-A": first_divergence(generated["C"], generated["A"]),
                    "C-B": first_divergence(generated["C"], generated["B"]),
                },
            }
        )
    return {
        "authority": "descriptive only",
        "fixtures": rows,
    }


def evaluate_candidate(
    arm: str,
    natural: dict[str, object],
    retrieval: dict[str, object],
) -> dict[str, object]:
    contrast = f"{arm}-A"
    overall_delta = float(natural["overall"]["deltas"][contrast])
    shape_deltas = {
        str(row["prompt_token_count"]): float(row["deltas"][contrast])
        for row in natural["shapes"]
    }
    document_deltas = {
        str(row["fixture_id"]): float(row["deltas"][contrast])
        for row in natural["documents"]
    }
    upper = float(
        natural["bootstrap"]["contrasts"][contrast]["upper_bound_nats_per_token"]
    )
    shapes_pass = all(delta <= SHAPE_NLL_LIMIT for delta in shape_deltas.values())
    documents_pass = all(
        delta <= DOCUMENT_NLL_LIMIT for delta in document_deltas.values()
    )
    point_guardrails_pass = shapes_pass and documents_pass
    confidence_pass = upper <= NLL_UPPER_LIMIT
    if not point_guardrails_pass:
        nll_status = "FAIL"
    elif not confidence_pass:
        nll_status = "HOLD"
    else:
        nll_status = "PASS"

    aggregate = retrieval["aggregate"]
    exact_total_pass = aggregate[arm]["exact_passes"] >= aggregate["A"]["exact_passes"]
    answer_delta = float(retrieval["token_weighted_deltas"][contrast])
    answer_nll_pass = answer_delta <= ANSWER_NLL_LIMIT
    hard_kill_tasks = [
        str(task["fixture_id"])
        for task in retrieval["tasks"]
        if task["arms"]["A"]["exact_pass"]
        and task["arms"]["D"]["exact_pass"]
        and not task["arms"][arm]["exact_pass"]
    ]
    retrieval_pass = exact_total_pass and answer_nll_pass and not hard_kill_tasks
    if not retrieval_pass:
        status = "FAIL"
    elif nll_status == "HOLD":
        status = "HOLD"
    else:
        status = nll_status
    return {
        "arm": arm,
        "status": status,
        "nll": {
            "status": nll_status,
            "overall_point_estimate": overall_delta,
            "overall_point_estimate_authority": (
                "descriptive; no separate preregistered threshold"
            ),
            "point_guardrails_pass": point_guardrails_pass,
            "one_sided_97_5_upper_bound": upper,
            "upper_bound_limit": NLL_UPPER_LIMIT,
            "confidence_pass": confidence_pass,
            "shape_deltas": shape_deltas,
            "shape_limit": SHAPE_NLL_LIMIT,
            "shapes_pass": shapes_pass,
            "document_deltas": document_deltas,
            "document_limit": DOCUMENT_NLL_LIMIT,
            "documents_pass": documents_pass,
        },
        "retrieval": {
            "status": "PASS" if retrieval_pass else "FAIL",
            "candidate_exact_passes": aggregate[arm]["exact_passes"],
            "incumbent_exact_passes": aggregate["A"]["exact_passes"],
            "exact_total_pass": exact_total_pass,
            "token_weighted_answer_nll_delta": answer_delta,
            "answer_nll_limit": ANSWER_NLL_LIMIT,
            "answer_nll_pass": answer_nll_pass,
            "hard_kill_tasks_where_a_and_d_pass": hard_kill_tasks,
        },
    }


def decide_disposition(
    b_gate: dict[str, object],
    c_gate: dict[str, object],
    c_vs_b: dict[str, object],
) -> dict[str, object]:
    b_status = str(b_gate["status"])
    c_status = str(c_gate["status"])
    if b_status == "PASS":
        code = "B_CORRECTNESS_GO"
        selected = "B"
        explanation = (
            "Generic selected-packed prefill passes correctness. It still requires an "
            "uninstrumented performance packet before any production default change."
        )
    elif b_status == "FAIL" and c_status == "PASS":
        code = "C_COST_GATE_ONLY"
        selected = None
        explanation = (
            "Generic selected-packed fails while the F32-HC-down challenger passes. "
            "C may advance only to a separate performance and cost gate."
        )
    elif "HOLD" in {b_status, c_status}:
        code = "HOLD_FROZEN_COHORT_EXTENSION_REQUIRED"
        selected = None
        explanation = (
            "At least one viable point estimate misses its confidence gate. Any cohort "
            "extension requires a new preregistration and fresh document-disjoint inputs."
        )
    else:
        code = "SELECTED_PACKED_DEFAULT_OFF"
        selected = None
        explanation = (
            "Neither selected-packed candidate passes every frozen gate; retain the "
            "default-safe packed-dense plus scalar-selected path."
        )
    return {
        "code": code,
        "correctness_go_arm": selected,
        "production_default_changed": False,
        "explanation": explanation,
        "c_cost_gate_eligible": bool(c_vs_b["eligible_for_cost_gate"]),
        "c_cost_gate_completed": False,
    }


def attach_analysis_binding(report: dict[str, object]) -> None:
    compact = json.dumps(
        report,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    encoded = compact.encode()
    report["binding"] = {
        "schema": ANALYSIS_BINDING_SCHEMA,
        "semantic_payload_encoding": "sorted compact Python JSON before binding",
        "semantic_payload_bytes": len(encoded),
        "semantic_payload_sha256": sha256(encoded),
        "semantic_payload_json_compact": compact,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", type=Path, required=True)
    parser.add_argument("--llama", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repository = Path(__file__).resolve().parents[2]
    analyzer_path = Path(__file__).resolve()
    wrapper_path = analyzer_path.with_name("qwen4exp_selected_quality_llama.py")
    imported_wrapper_path = Path(
        require_string(llama_support.__file__, "imported wrapper __file__")
    ).resolve(strict=True)
    imported_wrapper_origin = Path(
        require_string(llama_support.__spec__.origin, "imported wrapper origin")
    ).resolve(strict=True)
    if imported_wrapper_path != wrapper_path or imported_wrapper_origin != wrapper_path:
        raise RuntimeError(
            "imported wrapper module does not match the committed sibling"
        )
    analyzer_bytes, analyzer_stamp = read_stable(analyzer_path, "analyzer source")
    wrapper_bytes, wrapper_stamp = read_stable(wrapper_path, "llama wrapper source")
    source_snapshots = {
        "analyzer": {
            "bytes": len(analyzer_bytes),
            "sha256": sha256(analyzer_bytes),
        },
        "llama_wrapper": {
            "bytes": len(wrapper_bytes),
            "sha256": sha256(wrapper_bytes),
        },
    }
    output_parent = args.output.parent.resolve(strict=True)
    if not output_parent.is_dir():
        raise RuntimeError(f"output parent is not a directory: {output_parent}")
    output = output_parent / args.output.name
    fixtures_path = args.fixtures.resolve(strict=True)
    local_path = args.local.resolve(strict=True)
    llama_path = args.llama.resolve(strict=True)
    fixture_bytes, fixture_stamp = read_stable(fixtures_path, "fixture manifest")
    if sha256(fixture_bytes) != MANIFEST_SHA256:
        raise RuntimeError("fixture manifest SHA-256")
    local_bytes, local_stamp = read_stable(local_path, "local evidence")
    llama_bytes, llama_stamp = read_stable(llama_path, "llama evidence")
    manifest, tokens, fixtures, expected_operations = validate_fixtures(fixtures_path)
    local = validate_local_report(
        parse_json_strict(local_bytes), manifest, fixtures, tokens
    )
    llama = validate_llama_evidence(
        parse_json_strict(llama_bytes),
        manifest,
        fixtures,
        tokens,
        expected_operations,
    )
    if local["model_rows"] != llama["model_rows"]:
        raise RuntimeError("local and llama evidence use different model bytes")
    common_source = validate_common_sources(
        repository,
        local,
        llama,
        source_snapshots,
    )

    natural = compute_natural_metrics(local, llama, manifest)
    retrieval = compute_retrieval_metrics(local, llama, manifest)
    open_greedy = compute_open_greedy(local)
    b_gate = evaluate_candidate("B", natural, retrieval)
    c_gate = evaluate_candidate("C", natural, retrieval)
    c_vs_b_upper = float(
        natural["bootstrap"]["contrasts"]["C-B"]["upper_bound_nats_per_token"]
    )
    c_passes_b_tasks = all(
        not task["arms"]["B"]["exact_pass"] or task["arms"]["C"]["exact_pass"]
        for task in retrieval["tasks"]
    )
    c_vs_b = {
        "one_sided_95_upper_bound_nats_per_token": c_vs_b_upper,
        "required_strict_upper_bound": C_VS_B_SUPERIORITY_LIMIT,
        "nll_superiority_pass": c_vs_b_upper < C_VS_B_SUPERIORITY_LIMIT,
        "c_passes_every_retrieval_task_b_passes": c_passes_b_tasks,
        "both_candidates_pass": b_gate["status"] == "PASS"
        and c_gate["status"] == "PASS",
        "eligible_for_cost_gate": c_gate["status"] == "PASS"
        and (
            b_gate["status"] == "FAIL"
            or (
                b_gate["status"] == "PASS"
                and c_vs_b_upper < C_VS_B_SUPERIORITY_LIMIT
                and c_passes_b_tasks
            )
        ),
    }
    disposition = decide_disposition(b_gate, c_gate, c_vs_b)

    input_domain = (
        "qwen4exp-selected-quality-analysis-inputs-v1\0"
        f"packet_id={PACKET_ID}\n"
        f"fixture_manifest_sha256={MANIFEST_SHA256}\n"
        f"local_evidence_sha256={sha256(local_bytes)}\n"
        f"llama_evidence_sha256={sha256(llama_bytes)}\n"
        f"common_source_commit={common_source['commit']}\n"
        f"analyzer_source_sha256={sha256(analyzer_bytes)}\n"
        f"llama_wrapper_source_sha256={sha256(wrapper_bytes)}\n"
    )
    report: dict[str, object] = {
        "schema": ANALYSIS_SCHEMA,
        "schema_version": 1,
        "packet_id": PACKET_ID,
        "status": "analyzed",
        "inputs": {
            "fixture_manifest": {
                "path": str(fixtures_path),
                "bytes": len(fixture_bytes),
                "sha256": MANIFEST_SHA256,
                "stamp": fixture_stamp,
            },
            "local_evidence": {
                "path": str(local_path),
                "bytes": len(local_bytes),
                "sha256": sha256(local_bytes),
                "stamp": local_stamp,
                "ordered_run_binding_root_sha256": local[
                    "ordered_run_binding_root_sha256"
                ],
            },
            "llama_evidence": {
                "path": str(llama_path),
                "bytes": len(llama_bytes),
                "sha256": sha256(llama_bytes),
                "stamp": llama_stamp,
                "evidence_binding_sha256": llama["report"]["evidence_binding_sha256"],
            },
            "analyzer": {
                "path": str(analyzer_path),
                "bytes": len(analyzer_bytes),
                "sha256": sha256(analyzer_bytes),
                "stamp": analyzer_stamp,
            },
            "llama_wrapper": {
                "path": str(wrapper_path),
                "bytes": len(wrapper_bytes),
                "sha256": sha256(wrapper_bytes),
                "stamp": wrapper_stamp,
            },
            "common_source": common_source,
            "domain_utf8": input_domain,
            "binding_sha256": sha256(input_domain.encode()),
        },
        "validation": {
            "local_operations": LOCAL_OPERATION_COUNT,
            "llama_operations": TOTAL_OPERATION_COUNT - LOCAL_OPERATION_COUNT,
            "natural_documents": 12,
            "natural_scored_tokens_per_arm": 1152,
            "retrieval_tasks": 8,
            "same_locked_model_bytes": True,
            "structural_validation": {
                "status": "PASS",
                "checks": [
                    "finite complete-vocabulary score rows",
                    "frozen operation order and topology",
                    "persistent replay identities",
                    "HC treatment counts and hashes",
                    "same-commit source and kernel roots",
                    "same locked model shards",
                ],
            },
        },
        "contract": {
            "candidate_nll_upper_limit": NLL_UPPER_LIMIT,
            "shape_nll_limit": SHAPE_NLL_LIMIT,
            "document_nll_limit": DOCUMENT_NLL_LIMIT,
            "answer_nll_limit": ANSWER_NLL_LIMIT,
            "c_vs_b_superiority_limit": C_VS_B_SUPERIORITY_LIMIT,
            "llama_cpp_authority": "secondary same-quant triangulation only",
            "upstream_bf16_required": False,
        },
        "natural": natural,
        "retrieval": retrieval,
        "open_greedy": open_greedy,
        "gates": {
            "B": b_gate,
            "C": c_gate,
            "C_vs_B": c_vs_b,
        },
        "disposition": disposition,
    }
    attach_analysis_binding(report)
    if (
        file_stamp(fixtures_path) != fixture_stamp
        or file_stamp(local_path) != local_stamp
        or file_stamp(llama_path) != llama_stamp
        or file_stamp(analyzer_path) != analyzer_stamp
        or file_stamp(wrapper_path) != wrapper_stamp
    ):
        raise RuntimeError("analysis input changed before publication")
    report_bytes = canonical_json(report)
    with OutputReservation(output) as reservation:
        report_sha256 = reservation.publish(report_bytes)
    print(
        json.dumps(
            {
                "output": str(output),
                "bytes": len(report_bytes),
                "sha256": report_sha256,
                "disposition": disposition["code"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
