#!/usr/bin/env python3

"""v0.625 post-edit A3B prefetch-suppression confirmation packet."""

import argparse
import hashlib
import json
import os
import re
import signal
import subprocess
import time
from pathlib import Path

import v0602_a3b_parallel_copied_loader as protocol
import v0620_a3b_parallel_pread_product as product
import v0621_a3b_parallel_pread_auto as auto

ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = ROOT / "target/profiles/v0625-a3b-prefetch-suppression-confirmation-p1"
PREREG = ROOT / "docs/bench/v0625-a3b-prefetch-suppression-confirmation.md"
BASE_PROTOCOL = ROOT / "scripts/profile/v0602_a3b_parallel_copied_loader.py"
PREAD_PROTOCOL = ROOT / "scripts/profile/v0619_a3b_parallel_pread_loader.py"
PRODUCT_PROTOCOL = ROOT / "scripts/profile/v0620_a3b_parallel_pread_product.py"
AUTO_PROTOCOL = ROOT / "scripts/profile/v0621_a3b_parallel_pread_auto.py"
PREDECESSOR_PROTOCOL = ROOT / "scripts/profile/v0624_a3b_pread_prefetch_suppression.py"
COMMON_RUNNER = ROOT / "scripts/profile/v0593_demand_paged_no_copy.py"
EXAMPLE_BINARY = ROOT / "target/release/examples/first_byte_spike"
PREDECESSOR = ROOT / "target/profiles/v0624-a3b-pread-prefetch-suppression-p1"
IMPLEMENTATION_PARENT = "4f2117c39740f3b9187ae8a7b0e6749094b7393d"
PREDECESSOR_COMPLETE_SHA256 = (
    "5e8f35ab6747ce520a112aebf0d178a2099e37377c59b67e81b8f58dabe24d03"
)
PREDECESSOR_DECISION_SHA256 = (
    "451982802bd1b5348c0cab23d69cfd85a7b7c9bf6806f512e89c7d8152ddf3f8"
)
PREDECESSOR_INVENTORY_SHA256 = (
    "0ad18a2886e085aa7805807ac85c3d6fc74987120dfc834eb379e5f5e0858d0e"
)
MODEL_PAGES = auto.MODEL_PAGES
PHYSICAL_READ_MIN_GIB = 20.50
PHYSICAL_READ_MAX_GIB = 20.75
EXPECTED_FIRST_TOKEN = '11751 piece=" Paris"'
CHILDREN = (("S", "cold-only"), ("A", "always"), ("O", "off"))
SELECTOR_TESTS = (
    "runtime::tests::prefetch_action_selector_table_is_fail_closed",
    "metal_forward::tests::prepared_auto_prefetch_advice_selector_table_is_fail_closed",
)
SUPPRESSION_LINE = (
    "[runtime-prefetch] schema=1 configured=cold-only action=suppressed "
    "reason=authenticated-disposable-auto-a3b-direct-pread "
    "profile=a3b-q4km-v1 population=pread"
)
SUPPRESSION_STDOUT = (
    "prefetch:    suppressed (authenticated disposable Auto A3B direct pread)"
)
DECIMAL_PATTERN = r"[0-9]+\.[0-9]+"
SIGNED_DECIMAL_PATTERN = r"[+-][0-9]+\.[0-9]+"


class EvidenceDurabilityError(RuntimeError):
    pass


class EvidenceInvalid(RuntimeError):
    pass


class ContractMiss(RuntimeError):
    pass


def parse_json_file(path: Path) -> dict[str, object]:
    value = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=protocol.reject_json_constant,
    )
    if not isinstance(value, dict):
        raise RuntimeError(f"{path} is not a JSON object")
    return value


def verify_predecessor() -> dict[str, object]:
    manifest, decision = auto.verify_inventory(
        PREDECESSOR,
        PREDECESSOR_COMPLETE_SHA256,
        PREDECESSOR_DECISION_SHA256,
        PREDECESSOR_INVENTORY_SHA256,
    )
    cold = decision.get("stages", {}).get("cold")
    if (
        decision.get("status") != "go"
        or decision.get("authority")
        != "one-a3b-coldonly-suppression-selector-implementation"
        or decision.get("source_commit") != manifest.get("source_commit")
        or not isinstance(cold, dict)
        or cold.get("passes") is not True
    ):
        raise RuntimeError("v0.624 predecessor authority drifted")
    return {
        "packet_complete_sha256": PREDECESSOR_COMPLETE_SHA256,
        "decision_sha256": PREDECESSOR_DECISION_SHA256,
        "inventory_sha256": PREDECESSOR_INVENTORY_SHA256,
        "source_commit": decision["source_commit"],
        "authority": decision["authority"],
    }


def required_manifest_paths() -> tuple[Path, ...]:
    return (
        Path(__file__).resolve(),
        PREREG,
        BASE_PROTOCOL,
        PREAD_PROTOCOL,
        PRODUCT_PROTOCOL,
        AUTO_PROTOCOL,
        PREDECESSOR_PROTOCOL,
        COMMON_RUNNER,
        protocol.MODEL,
        protocol.CLI_BINARY,
        protocol.BENCH_BINARY,
        EXAMPLE_BINARY,
        PREDECESSOR / "packet-complete.json",
        PREDECESSOR / "decision.json",
        PREDECESSOR / "artifact-inventory.sha256",
    )


def source_and_build_identity() -> tuple[str, dict[str, object]]:
    tracked = (Path(__file__).resolve(), PREREG)
    for path in tracked:
        protocol.command_text(
            ["git", "ls-files", "--error-unmatch", str(path.relative_to(ROOT))]
        )
    commit = protocol.command_text(["git", "rev-parse", "HEAD"]).strip()
    dirty = protocol.command_text(["git", "status", "--porcelain=v1"]).strip()
    if dirty:
        raise RuntimeError(f"source is dirty: {dirty!r}")
    parent = protocol.command_text(["git", "rev-parse", "HEAD^"]).strip()
    changed = set(
        protocol.command_text(
            ["git", "diff", "--name-only", f"{IMPLEMENTATION_PARENT}..{commit}"]
        ).splitlines()
    )
    expected = {str(path.relative_to(ROOT)) for path in tracked}
    if parent != IMPLEMENTATION_PARENT or changed != expected:
        raise RuntimeError(
            f"packet source boundary drifted: parent={parent} changed={sorted(changed)}"
        )
    build = protocol.parse_json(
        protocol.command_text(
            [str(protocol.BENCH_BINARY), "build-info", "--output", "json"]
        )
    )
    if not isinstance(build, dict):
        raise RuntimeError("build identity is not an object")
    if (
        build.get("build_commit") != commit
        or build.get("runtime_commit") != commit
        or build.get("status") != "match"
        or build.get("build_dirty") is not False
        or build.get("runtime_dirty") is not False
        or build.get("build_source_state") != build.get("runtime_source_state")
    ):
        raise RuntimeError(f"source/build identity mismatch: {build}")
    return commit, build


def build_manifest(
    removed_environment: list[str], base_env: dict[str, str]
) -> dict[str, object]:
    imported = verify_predecessor()
    commit, build = source_and_build_identity()
    paths = required_manifest_paths()
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError(f"missing packet inputs: {missing}")
    hashes = {str(path): protocol.common.sha256_file(path) for path in paths}
    if hashes[str(protocol.MODEL)] != protocol.EXPECTED_MODEL_SHA256:
        raise RuntimeError("model SHA-256 drifted")
    device = protocol.command_text(
        [str(protocol.CLI_BINARY), "--info"], env=base_env
    ).strip()
    macos = protocol.command_text(["sw_vers", "-productVersion"], env=base_env).strip()
    hw_memsize = int(
        protocol.command_text(["sysctl", "-n", "hw.memsize"], env=base_env)
    )
    if (
        device != protocol.EXPECTED_DEVICE
        or not macos.startswith("15.")
        or hw_memsize != protocol.EXPECTED_HW_MEMSIZE
    ):
        raise RuntimeError("host boundary drifted")
    return {
        "schema": 1,
        "created_unix_ms": time.time_ns() // 1_000_000,
        "source_commit": commit,
        "build_identity": build,
        "device": device,
        "macos": macos,
        "hw_memsize": hw_memsize,
        "removed_environment": removed_environment,
        "child_environment": protocol.child_environment_record(base_env),
        "time_resource_probe": protocol.preflight_time_resources(base_env),
        "sha256": hashes,
        "model_size_bytes": protocol.MODEL.stat().st_size,
        "children": [list(child) for child in CHILDREN],
        "cooldown_s": protocol.COOLDOWN_S,
        "host_sample_limit": protocol.HOST_SAMPLE_LIMIT,
        "host_sample_interval_s": protocol.HOST_SAMPLE_INTERVAL_S,
        "child_retry_count": 0,
        "imported_predecessor": imported,
        "physical_read_window_gib": [PHYSICAL_READ_MIN_GIB, PHYSICAL_READ_MAX_GIB],
        "selector_test_filters": list(SELECTOR_TESTS),
        "marker_parser_negative_controls": verify_marker_parser_negative_controls(),
    }


def unique_match(pattern: str, text: str, label: str) -> re.Match[str]:
    matches = list(re.finditer(pattern, text, re.MULTILINE))
    if len(matches) != 1:
        raise EvidenceInvalid(f"{label} occurrence count drifted: {len(matches)}")
    return matches[0]


def parse_stdout(stdout: str, arm: str) -> dict[str, object]:
    model = unique_match(r"^model:\s+(.+)$", stdout, "model")
    size = unique_match(rf"^size:\s+({DECIMAL_PATTERN}) GiB$", stdout, "model size")
    intent = unique_match(r"^intent:\s+(.+)$", stdout, "intent")
    invalidate_flag = unique_match(
        r"^invalidate:\s+(true|false)$", stdout, "invalidate flag"
    )
    prompt = unique_match(r"^prompt:\s+(.+)$", stdout, "prompt")
    encoded = unique_match(
        r"^\s+prompt encoded to (\d+) tokens: (.+)$", stdout, "encoded prompt"
    )
    pre_arm = unique_match(
        rf"^pre-arm residency: (\d+)/(\d+) pages "
        rf"\(({DECIMAL_PATTERN})%\)$",
        stdout,
        "pre-arm residency",
    )
    invalidation = unique_match(
        r"^invalidate: (\d+)/(\d+) -> (\d+)/(\d+)$", stdout, "invalidation"
    )
    load = unique_match(
        rf"^load:\s+({DECIMAL_PATTERN}) s\s+pageins=\s*(\d+)\s+"
        rf"diskR=\s*({DECIMAL_PATTERN}) GiB$",
        stdout,
        "load",
    )
    first = unique_match(
        rf"^FIRST BYTE:\s+({DECIMAL_PATTERN}) s$", stdout, "first byte"
    )
    total = unique_match(
        rf"^rusage total: pageins=\s*(\d+)\s+"
        rf"diskR=\s*({DECIMAL_PATTERN}) GiB\s+"
        rf"diskW=\s*({DECIMAL_PATTERN}) MiB\s+"
        rf"\u0394RSS=\s*({SIGNED_DECIMAL_PATTERN}) GiB$",
        stdout,
        "total resources",
    )
    token = unique_match(r"^first token: id=(\d+) piece=(.*)$", stdout, "first token")
    post = unique_match(
        rf"^post-arm residency: (\d+)/(\d+) pages "
        rf"\(({DECIMAL_PATTERN})%\)$",
        stdout,
        "post-arm residency",
    )
    policy = unique_match(r"^policy:\s+(.+)$", stdout, "policy")
    if (
        model.group(1) != str(protocol.MODEL)
        or float(size.group(1)) != 20.61
        or intent.group(1) != "DisposableSingleTurn"
        or invalidate_flag.group(1) != "true"
        or prompt.group(1) != '"The capital of France is"'
        or encoded.group(1) != "5"
        or encoded.group(2) != "[760, 6511, 314, 9338, 369]"
    ):
        raise EvidenceInvalid("product-shape output drifted")
    pre_resident, pre_total, pre_percent = pre_arm.groups()
    before, total_before, after, total_after = map(int, invalidation.groups())
    post_resident, post_total, post_percent = post.groups()
    if min(int(pre_total), total_before, total_after, int(post_total)) <= 0:
        raise EvidenceInvalid("residency evidence has a zero page total")
    expected_pre_percent = round(int(pre_resident) / int(pre_total) * 100.0, 1)
    expected_post_percent = round(int(post_resident) / int(post_total) * 100.0, 1)
    if (
        int(pre_resident) != MODEL_PAGES
        or int(pre_total) != MODEL_PAGES
        or float(pre_percent) != expected_pre_percent
        or before != MODEL_PAGES
        or total_before != MODEL_PAGES
        or after != 0
        or total_after != MODEL_PAGES
        or int(post_total) != MODEL_PAGES
        or int(post_resident) / int(post_total) < 0.99
        or float(post_percent) != expected_post_percent
    ):
        raise EvidenceInvalid("residency contract drifted")
    disk_gib = float(total.group(2))
    if not PHYSICAL_READ_MIN_GIB <= disk_gib <= PHYSICAL_READ_MAX_GIB:
        raise EvidenceInvalid("physical-read window missed")
    contract_errors = []
    expected_policy = {
        "S": "ColdOnly { threshold: ResidencyThreshold(0.9) }",
        "A": "Always",
        "O": "Off",
    }[arm]
    if policy.group(1) != expected_policy:
        contract_errors.append("prefetch policy output drifted")

    prefetch_lines = [
        line.strip()
        for line in stdout.splitlines()
        if line.strip().startswith("prefetch:")
    ]
    prefetch = None
    if arm == "S":
        if prefetch_lines != [SUPPRESSION_STDOUT]:
            contract_errors.append("suppressed prefetch output drifted")
        else:
            prefetch = {"action": "suppressed"}
    elif arm == "A":
        matches = re.findall(
            rf"^\s+prefetch:\s+({DECIMAL_PATTERN}) s\s+"
            rf"(\d+) shards prefetched, (\d+) skipped, "
            rf"({DECIMAL_PATTERN}) GiB returned$",
            stdout,
            re.MULTILINE,
        )
        if len(prefetch_lines) != 1:
            contract_errors.append("Always prefetch occurrence count drifted")
        elif len(matches) != 1:
            raise EvidenceInvalid("Always prefetch evidence is malformed")
        else:
            wall_s, prefetched, skipped, returned_gib = matches[0]
            if (
                int(prefetched) != 1
                or int(skipped) != 0
                or float(returned_gib) != 20.61
            ):
                contract_errors.append("Always did not prefetch the complete shard")
            else:
                prefetch = {
                    "action": "configured",
                    "wall_s": float(wall_s),
                    "shards_prefetched": int(prefetched),
                    "shards_skipped": int(skipped),
                    "returned_gib": float(returned_gib),
                }
    elif prefetch_lines:
        contract_errors.append("Off unexpectedly emitted a prefetch phase")

    first_token = f"{token.group(1)} piece={token.group(2)}"
    if first_token != EXPECTED_FIRST_TOKEN:
        contract_errors.append(f"first token drifted: {first_token!r}")
    if contract_errors:
        raise ContractMiss("; ".join(contract_errors))
    return {
        "pre_arm_resident_pages": int(pre_resident),
        "invalidated_before_pages": before,
        "invalidated_after_pages": after,
        "load_s": float(load.group(1)),
        "load_pageins": int(load.group(2)),
        "load_disk_gib": float(load.group(3)),
        "first_byte_s": float(first.group(1)),
        "total_pageins": int(total.group(1)),
        "total_disk_gib": disk_gib,
        "total_disk_write_mib": float(total.group(3)),
        "total_rss_delta_gib": float(total.group(4)),
        "first_token": first_token,
        "post_resident_pages": int(post_resident),
        "post_resident_fraction": int(post_resident) / int(post_total),
        "prefetch": prefetch,
    }


def recognized_load_lines(stderr: str) -> list[str]:
    prefixes = (
        "[metal-load] native quantized token embedding policy:",
        "[metal-gguf-parallel-policy]",
        "[runtime-prefetch]",
        auto.PREAD_MARKER,
        "[metal-load-ledger]",
    )
    return [line for line in stderr.splitlines() if line.startswith(prefixes)]


def all_load_family_lines(stderr: str) -> list[str]:
    return [
        line
        for line in stderr.splitlines()
        if line.startswith(
            (
                "[metal-load]",
                "[metal-load-",
                "[metal-gguf-",
                "[runtime-prefetch]",
                "[runtime-prefetch-",
            )
        )
    ]


def verify_marker_parser_negative_controls() -> list[str]:
    unknown = [
        "[metal-load] future-policy: x",
        "[metal-gguf-future] x",
        "[runtime-prefetch-v2] x",
        "[metal-load-ledger-v2] x",
    ]
    for line in unknown:
        text = f"{protocol.POLICY_LINE}\n{line}\n"
        if all_load_family_lines(text) == recognized_load_lines(text):
            raise RuntimeError(f"unknown marker control escaped: {line}")
    return unknown


def parse_load_contract(stderr: str, arm: str) -> dict[str, object]:
    expected = [
        protocol.POLICY_LINE,
        auto.AUTO_POLICY_LINE,
    ]
    if arm == "S":
        expected.append(SUPPRESSION_LINE)
    expected.extend((auto.PREAD_MARKER, protocol.LEDGER_LINE))
    recognized = recognized_load_lines(stderr)
    all_load_lines = all_load_family_lines(stderr)
    pread_lines = [
        line for line in all_load_lines if line.startswith(auto.PREAD_MARKER)
    ]
    phase_values = []
    for line in pread_lines:
        try:
            phase_values.append(product.parse_marker(line, auto.PREAD_MARKER))
        except Exception as error:
            raise EvidenceInvalid(
                f"{arm} direct-pread marker is malformed: {error}"
            ) from error

    contract_errors = []
    if all_load_lines != recognized:
        contract_errors.append(f"{arm} unrecognized load line: {all_load_lines!r}")
    if len(recognized) != len(expected):
        contract_errors.append(f"{arm} load-line count drifted: {recognized!r}")
    if stderr.count(SUPPRESSION_LINE) != (1 if arm == "S" else 0):
        contract_errors.append(f"{arm} suppression marker count drifted")
    if stderr.count(auto.PREAD_MARKER) != 1:
        contract_errors.append(f"{arm} direct-pread marker count drifted")
    if stderr.count(auto.AUTO_POLICY_LINE) != 1:
        contract_errors.append(f"{arm} Auto marker count drifted")
    if len(recognized) == len(expected):
        for line, required in zip(recognized, expected, strict=True):
            if required != auto.PREAD_MARKER and line != required:
                contract_errors.append(
                    f"{arm} load-line ordering drifted: {recognized!r}"
                )
                break
    if contract_errors:
        raise ContractMiss("; ".join(contract_errors))
    return {
        "storage": "auto-parallel-pread",
        "suppressed": arm == "S",
        "recognized_lines": recognized,
        "phase_us": phase_values[0],
    }


def run_selector_tests(base_env: dict[str, str]) -> dict[str, object]:
    records = []
    env = base_env.copy()
    env["CARGO_TERM_COLOR"] = "never"
    for index, test_filter in enumerate(SELECTOR_TESTS, 1):
        stem = f"selector-{index:02d}"
        stdout_path = ARTIFACT / f"{stem}.out"
        stderr_path = ARTIFACT / f"{stem}.err"
        command = [
            "cargo",
            "test",
            "-p",
            "qwen-llm",
            "--lib",
            test_filter,
            "--",
            "--exact",
        ]
        started = time.perf_counter()
        process = None
        wait_errors: list[str] = []
        try:
            with (
                stdout_path.open("xb") as stdout_file,
                stderr_path.open("xb") as stderr_file,
            ):
                prior_sigint = signal.signal(signal.SIGINT, signal.SIG_IGN)
                try:
                    process = subprocess.Popen(
                        command,
                        cwd=ROOT,
                        env=env,
                        stdout=stdout_file,
                        stderr=stderr_file,
                    )
                    returncode, wait_errors = protocol.wait_for_child(process)
                finally:
                    signal.signal(signal.SIGINT, prior_sigint)
            fsync_raw_artifacts((stdout_path, stderr_path))
        except (OSError, KeyboardInterrupt) as error:
            error_text = f"{type(error).__name__}:{error}"
            if process is not None:
                returncode, deferred = protocol.wait_for_child(process)
                wait_errors.extend(deferred)
                wait_errors.append(error_text)
            fsync_raw_artifacts((stdout_path, stderr_path))
            if process is None:
                raise protocol.InconclusivePacket(
                    "selector",
                    stem,
                    [f"selector_test_spawn_failed={error_text}"],
                ) from error
        stdout_bytes = stdout_path.read_bytes()
        stderr_bytes = stderr_path.read_bytes()
        decode_errors = []
        try:
            stdout = stdout_bytes.decode("utf-8")
        except UnicodeDecodeError as error:
            stdout = stdout_bytes.decode("utf-8", errors="replace")
            decode_errors.append(f"stdout:{error}")
        try:
            stderr_bytes.decode("utf-8")
        except UnicodeDecodeError as error:
            decode_errors.append(f"stderr:{error}")
        passed_line = f"test {test_filter} ... ok"
        failed_line = f"test {test_filter} ... FAILED"
        summary_matches = re.findall(
            r"^test result: (ok|FAILED)\. (\d+) passed; (\d+) failed; "
            r"\d+ ignored; \d+ measured; \d+ filtered out; "
            r"finished in [0-9.]+s$",
            stdout,
            re.MULTILINE,
        )
        evidence_valid = (
            not decode_errors
            and not wait_errors
            and len(summary_matches) == 1
            and stdout.count(passed_line) + stdout.count(failed_line) == 1
        )
        passed = (
            evidence_valid
            and returncode == 0
            and stdout.count(passed_line) == 1
            and summary_matches[0][0] == "ok"
            and summary_matches[0][1:] == ("1", "0")
        )
        records.append(
            {
                "test_filter": test_filter,
                "command": command,
                "returncode": returncode,
                "wall_ms": (time.perf_counter() - started) * 1e3,
                "stdout_sha256": hashlib.sha256(stdout_bytes).hexdigest(),
                "stderr_sha256": hashlib.sha256(stderr_bytes).hexdigest(),
                "decode_errors": decode_errors,
                "wait_errors": wait_errors,
                "evidence_valid": evidence_valid,
                "passed": passed,
            }
        )
    stage = {
        "records": records,
        "evidence_valid": all(record["evidence_valid"] for record in records),
        "passes": all(record["passed"] for record in records),
    }
    result_path = ARTIFACT / "selector-tests.json"
    try:
        with result_path.open("xb") as output:
            output.write((protocol.json_text(stage, pretty=True) + "\n").encode())
            output.flush()
            os.fsync(output.fileno())
        protocol.fsync_directory(ARTIFACT)
    except OSError as error:
        raise EvidenceDurabilityError(
            f"selector result fsync failed: {type(error).__name__}: {error}"
        ) from error
    if not stage["evidence_valid"]:
        raise protocol.InconclusivePacket(
            "selector", "selector-tests", ["selector_test_evidence_invalid"]
        )
    return stage


def fsync_raw_artifacts(paths: tuple[Path, ...]) -> None:
    try:
        for path in paths:
            if not path.exists():
                continue
            with path.open("rb+") as artifact:
                artifact.flush()
                os.fsync(artifact.fileno())
        protocol.fsync_directory(ARTIFACT)
    except OSError as error:
        raise EvidenceDurabilityError(
            f"raw child artifact fsync failed: {type(error).__name__}: {error}"
        ) from error


def run_child(
    arm: str,
    policy: str,
    child_index: int,
    base_env: dict[str, str],
    manifest: dict[str, object],
) -> dict[str, object]:
    stem = f"confirm-{child_index:02d}-{arm.lower()}"
    stdout_path = ARTIFACT / f"{stem}.out"
    stderr_path = ARTIFACT / f"{stem}.err"
    for path in (stdout_path, stderr_path):
        if path.exists():
            raise RuntimeError(f"refusing to reuse child artifact {path}")
    conditioning = protocol.condition_for_child(stem, manifest)
    env = base_env.copy()
    command = [
        "/usr/bin/time",
        "-l",
        str(EXAMPLE_BINARY),
        str(protocol.MODEL),
        "--policy",
        policy,
        "--invalidate",
        "--prompt",
        "The capital of France is",
        "--tokens",
        "0",
        "--intent",
        "disposable",
    ]
    protocol.record_launch("confirm", stem, command, arm, child_index, child_index)
    process = None
    wait_errors: list[str] = []
    started = time.perf_counter()
    try:
        with (
            stdout_path.open("xb") as stdout_file,
            stderr_path.open("xb") as stderr_file,
        ):
            prior_sigint = signal.signal(signal.SIGINT, signal.SIG_IGN)
            try:
                process = subprocess.Popen(
                    command,
                    cwd=ROOT,
                    env=env,
                    stdout=stdout_file,
                    stderr=stderr_file,
                )
                returncode, wait_errors = protocol.wait_for_child(process)
                process_wall_ms = (time.perf_counter() - started) * 1e3
            finally:
                signal.signal(signal.SIGINT, prior_sigint)
        fsync_raw_artifacts((stdout_path, stderr_path))
    except (OSError, KeyboardInterrupt) as error:
        error_text = f"{type(error).__name__}:{error}"
        if process is None:
            fsync_raw_artifacts((stdout_path, stderr_path))
            protocol.record_completion("confirm", stem, None, error_text)
            if isinstance(error, OSError):
                protocol.record_spawn_failure_evidence(
                    "confirm", stem, conditioning, error
                )
            raise protocol.InconclusivePacket(
                "confirm", stem, [f"child_spawn_or_pipe_failed={error_text}"]
            ) from error
        returncode, deferred = protocol.wait_for_child(process)
        process_wall_ms = (time.perf_counter() - started) * 1e3
        wait_errors.extend(deferred)
        wait_errors.append(error_text)
        fsync_raw_artifacts((stdout_path, stderr_path))
    protocol.record_completion(
        "confirm",
        stem,
        returncode,
        ";".join(wait_errors) if wait_errors else None,
    )
    post_exit = protocol.capture_post_exit_state(conditioning)
    protocol.record_post_exit_state("confirm", stem, returncode, post_exit)
    stdout_bytes = stdout_path.read_bytes()
    stderr_bytes = stderr_path.read_bytes()
    decode_errors = []
    try:
        stdout = stdout_bytes.decode("utf-8")
    except UnicodeDecodeError as error:
        stdout = stdout_bytes.decode("utf-8", errors="replace")
        decode_errors.append(f"stdout:{error}")
    try:
        stderr = stderr_bytes.decode("utf-8")
    except UnicodeDecodeError as error:
        stderr = stderr_bytes.decode("utf-8", errors="replace")
        decode_errors.append(f"stderr:{error}")
    reasons = list(post_exit["child_interval"]["failure_reasons"])
    reasons.extend(f"child_utf8_invalid={error}" for error in decode_errors)
    if not post_exit["host_after_exit"]["valid"]:
        reasons.append("post_exit_host_invalid")
    if wait_errors:
        reasons.append(f"child_wait_interrupted={';'.join(wait_errors)}")
    if returncode != 0:
        reasons.append(f"child_nonzero_exit={returncode}")
    try:
        resources = protocol.process_resources(stderr)
    except Exception as error:
        resources = None
        reasons.append(
            f"child_process_resource_parse_invalid={type(error).__name__}:{error}"
        )
    if resources is not None:
        if resources["swaps"] != 0:
            reasons.append("child_swaps")
        if resources["block_input_operations"] != 0:
            reasons.append("child_block_input")
    try:
        times = auto.parse_process_times(stderr)
    except Exception as error:
        times = {
            "real_s": None,
            "user_s": None,
            "system_s": None,
            "total_cpu_s": None,
        }
        reasons.append(
            f"child_process_time_parse_invalid={type(error).__name__}:{error}"
        )

    parsed = None
    load_contract = None
    contract_error = None
    if returncode == 0 and not decode_errors:
        evidence_errors = []
        contract_errors = []
        try:
            parsed = parse_stdout(stdout, arm)
        except EvidenceInvalid as error:
            evidence_errors.append(str(error))
        except ContractMiss as error:
            contract_errors.append(str(error))
        try:
            load_contract = parse_load_contract(stderr, arm)
        except EvidenceInvalid as error:
            evidence_errors.append(str(error))
        except ContractMiss as error:
            contract_errors.append(str(error))
        if evidence_errors:
            reasons.append(f"product_evidence_invalid={'; '.join(evidence_errors)}")
        elif contract_errors:
            contract_error = f"ContractMiss:{'; '.join(contract_errors)}"
    validity = {**post_exit, "process_resources": resources}
    protocol.record_post_exit_evidence("confirm", stem, returncode, validity, reasons)
    return {
        "stage": "confirm",
        "artifact_stem": stem,
        "arm": arm,
        "child_index": child_index,
        "command": command,
        "arm_environment": {
            "QWEN_GGUF_PARALLEL_COPY": None,
            "prefetch_policy": policy,
        },
        **conditioning,
        **validity,
        **times,
        "process_wall_ms": process_wall_ms,
        "stdout_sha256": hashlib.sha256(stdout_bytes).hexdigest(),
        "stderr_sha256": hashlib.sha256(stderr_bytes).hexdigest(),
        "load_contract": load_contract,
        "product": parsed,
        "contract_pass": contract_error is None,
        "contract_error": contract_error,
        "valid": not reasons,
        "validity_reasons": reasons,
    }


def run_stage(
    base_env: dict[str, str],
    manifest: dict[str, object],
    attempts_path: Path,
) -> list[dict[str, object]]:
    rows = []
    for child_index, (arm, policy) in enumerate(CHILDREN, 1):
        row = run_child(arm, policy, child_index, base_env, manifest)
        protocol.append_row(attempts_path, row)
        rows.append(row)
        if not row["valid"]:
            raise protocol.InconclusivePacket(
                "confirm", row["artifact_stem"], list(row["validity_reasons"])
            )
    if len(rows) != len(CHILDREN):
        raise RuntimeError("confirmation child count drifted")
    return rows


def analyze(rows: list[dict[str, object]]) -> dict[str, object]:
    if len(rows) != len(CHILDREN):
        raise RuntimeError("confirmation analysis child count drifted")
    expected_arms = [arm for arm, _ in CHILDREN]
    observed_arms = [str(row["arm"]) for row in rows]
    if observed_arms != expected_arms:
        raise RuntimeError("confirmation arm order drifted")
    contracts = {
        str(row["arm"]): {
            "artifact_stem": row["artifact_stem"],
            "contract_pass": row["contract_pass"],
            "contract_error": row["contract_error"],
            "load_contract": row["load_contract"],
            "product": row["product"],
        }
        for row in rows
    }
    gates = {
        "three_unique_children": len({row["artifact_stem"] for row in rows}) == 3,
        "all_infrastructure_valid": all(bool(row["valid"]) for row in rows),
        "all_product_contracts_pass": all(bool(row["contract_pass"]) for row in rows),
    }
    return {"contracts": contracts, "gates": gates, "passes": all(gates.values())}


def configure_protocol() -> None:
    protocol.ARTIFACT = ARTIFACT
    protocol.PREREG = PREREG
    protocol.required_manifest_paths = required_manifest_paths
    protocol.source_and_build_identity = source_and_build_identity
    protocol.build_manifest = build_manifest


def write_decision(decision: dict[str, object], manifest: dict[str, object]) -> None:
    if decision.get("status") == "go":
        imported = verify_predecessor()
        selector = decision.get("stages", {}).get("selector")
        confirm = decision.get("stages", {}).get("confirm")
        if (
            decision.get("authority")
            != "authenticated-disposable-a3b-auto-pread-coldonly-always-off"
            or decision.get("source_commit") != manifest.get("source_commit")
            or decision.get("imported_predecessor")
            != manifest.get("imported_predecessor")
            or imported != manifest.get("imported_predecessor")
            or not isinstance(selector, dict)
            or selector.get("passes") is not True
            or not isinstance(confirm, dict)
            or confirm.get("passes") is not True
        ):
            raise RuntimeError("v0.625 GO authority conjunction is incomplete")
    elif decision.get("authority") != "none":
        raise RuntimeError("non-GO v0.625 decision carries authority")
    protocol.write_identity_checked_decision(decision, manifest)


def run(*, preflight_only: bool) -> None:
    if ARTIFACT.exists():
        raise RuntimeError(f"refusing to reuse packet directory {ARTIFACT}")
    base_env, removed_environment = protocol.common.normalized_environment()
    manifest = build_manifest(removed_environment, base_env)
    if preflight_only:
        vm_state = protocol.capture_vm_state()
        if vm_state["capture_errors"]:
            raise RuntimeError(
                f"VM preflight capture failed: {vm_state['capture_errors']}"
            )
        print(
            protocol.json_text(
                {
                    "status": "preflight-passed",
                    "source_commit": manifest["source_commit"],
                    "manifest_sha256": hashlib.sha256(
                        (protocol.json_text(manifest, pretty=True) + "\n").encode()
                    ).hexdigest(),
                },
                pretty=True,
            )
        )
        return
    protocol.reserve_artifact(manifest)
    attempts_path = ARTIFACT / "attempts.jsonl"
    stages = {}
    try:
        stages["selector"] = run_selector_tests(base_env)
        if not stages["selector"]["passes"]:
            write_decision(
                {
                    "schema": 1,
                    "status": "kill",
                    "authority": "none",
                    "stopped_after": "selector",
                    "source_commit": manifest["source_commit"],
                    "imported_predecessor": manifest["imported_predecessor"],
                    "stages": stages,
                },
                manifest,
            )
            return
        rows = run_stage(base_env, manifest, attempts_path)
        stages["confirm"] = analyze(rows)
        status = "go" if stages["confirm"]["passes"] else "kill"
        authority = (
            "authenticated-disposable-a3b-auto-pread-coldonly-always-off"
            if status == "go"
            else "none"
        )
        write_decision(
            {
                "schema": 1,
                "status": status,
                "authority": authority,
                "stopped_after": "confirm",
                "source_commit": manifest["source_commit"],
                "imported_predecessor": manifest["imported_predecessor"],
                "stages": stages,
            },
            manifest,
        )
    except protocol.InconclusivePacket as error:
        write_decision(
            {
                "schema": 1,
                "status": "inconclusive",
                "authority": "none",
                "stopped_after": error.stage,
                "failed_child": error.child,
                "reasons": error.reasons,
                "source_commit": manifest["source_commit"],
                "imported_predecessor": manifest["imported_predecessor"],
                "stages": stages,
            },
            manifest,
        )
    except KeyboardInterrupt:
        write_decision(
            {
                "schema": 1,
                "status": "inconclusive",
                "authority": "none",
                "stopped_after": "operator-interrupt",
                "reasons": ["operator_interrupt_after_child_cleanup"],
                "source_commit": manifest["source_commit"],
                "imported_predecessor": manifest["imported_predecessor"],
                "stages": stages,
            },
            manifest,
        )
    except EvidenceDurabilityError:
        raise
    except Exception as error:
        if protocol.decision_publication_started():
            raise
        write_decision(
            {
                "schema": 1,
                "status": "implementation_or_contract_defect",
                "authority": "none",
                "stopped_after": "confirm",
                "error_type": type(error).__name__,
                "error": str(error),
                "source_commit": manifest["source_commit"],
                "imported_predecessor": manifest["imported_predecessor"],
                "stages": stages,
            },
            manifest,
        )
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true")
    arguments = parser.parse_args()
    configure_protocol()
    run(preflight_only=arguments.preflight_only)
