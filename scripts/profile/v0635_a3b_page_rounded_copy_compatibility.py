#!/usr/bin/env python3
"""v0.635 loaded-only A3B page-rounded-copy compatibility packet."""

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import statistics
import subprocess
import time

import v0630_dense27b_pread_loaded_stability as proven


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = ROOT / "target/profiles/v0635-a3b-page-rounded-copy-compatibility-p1"
PREREG = ROOT / "docs/bench/v0635-a3b-page-rounded-copy-compatibility.md"
RUNNER = Path(__file__).resolve()
METAL_SOURCE = ROOT / "crates/qwen-llm/src/metal_forward.rs"
MODEL = Path("/Users/tito/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")
PROMPT = ROOT / "docs/bench/tokenizer-prompts/current-reva-n8-interactive-qwen36.txt"
BENCH = ROOT / "target/release/qwen-bench"
CLI = ROOT / "target/release/qwen"
V0593 = ROOT / "scripts/profile/v0593_demand_paged_no_copy.py"
V0602 = ROOT / "scripts/profile/v0602_a3b_parallel_copied_loader.py"
V0630 = ROOT / "scripts/profile/v0630_dense27b_pread_loaded_stability.py"

BASE_COMMIT = "0619c925d488a9f4b47b64c509b6231f4ab6bfb1"
PREREG_COMMIT = "6fe16d5c878314b093799b551658eb651fdedbdc"
IMPLEMENTATION_COMMIT = "94e819b47d56553d8b298600b45a01b151ce5ddf"
FINAL_METAL_SHA256 = "0c7fa44527365eb791fd0edf8e581d88c5a7de849059b8aefd2c251fa3d24ce8"
IMPLEMENTATION_PATCH_SHA256 = (
    "e14a5a08a105d7cbaeb76e789a097a238e2b37127d07de73c10ec5d02bf05dab"
)
GGUF_COMMIT = "c7369fd4868a6f613459fff355477f53bf4ee2f1"
LLAMA_COMMIT = "fe4fb533d1ed2855b6ac5492e56c42007d410409"
LLAMA_UNTRACKED = (".claude/settings.local.json",)
HELPER_HASHES = {
    V0593: "4c1b73a2897d8bf882988407f45b59be7b19d236c06a612991491c57af0d7be4",
    V0602: "e5775489802dddc2d88dc9b950940417f325ebcc4ece29194550b7e430f851e3",
    V0630: "aaf3cfa79c33d781c0c3d2455ce1d7df5f264305fe19b5c8522a495f647f57dd",
}
MODEL_SHA256 = "ac0e2c1189e055faa36eff361580e79c5bd6f8e76bffb4ce547f167d53e31a61"
MODEL_SIZE = 22_134_528_992
PROMPT_SHA256 = "e265de9742d1b22e566fc108ae26331ccf46166c6f071e73f211e0a1a7e8b474"
PROMPT_BYTES = 1_891
PROMPT_TOKENS = 419
RUNTIME_MODEL_ID = "e6024ce53109fdf7"
RUNTIME_TOKENIZER_ID = "a4b0b26f8a8c9917"
MACOS_VERSION = "15.6.1"
MACOS_BUILD = "24G90"
HW_MEMSIZE = 137_438_953_472
DEVICE = (
    "device: Apple M4 Max | unified_memory=true | max_threadgroup_memory=32768 bytes"
)

PAIR_ORDERS = ("AB", "BA", "BA", "AB", "AB", "BA")
RUNS = 5
TRANSITIONS = 127
TRACE_IDS = 128
COOLDOWN_S = 30.0
HOST_SAMPLE_LIMIT = 6
HOST_SAMPLE_INTERVAL_S = 30.0
U64_MAX = 2**64 - 1
CORRECTNESS_TEST = (
    "metal_forward::tests::gguf_parallel_page_rounded_a3b_q4_is_bit_exact"
)

POLICY_LINE = (
    "[metal-load] native quantized token embedding policy: "
    "auto-promoted (Q8_0 [2048, 248320])"
)
LEDGER_LINE = (
    "[metal-load-ledger] source=733/22123538944 "
    "direct_copy=733/22123538944 direct_view=0/0 direct_alias=0/0 "
    "tail_fallback=0/0 converted=0/0/0 derived=0/0"
)
COMMON_MARKER = (
    "workers=4 cuts=155,359,539 tasks=155,204,180,194 "
    "worker_bytes=5532746240,5462315776,5595522304,5532954624 "
    "first_offsets=10990048,5543736288,11006052064,16601574368 "
    "last_offsets=5392741344,11004937952,16450579424,22134520800 "
    "create=shared,default_cache,default observed=shared,default_cache,tracked "
    "page=16384 alignment=32 max_buffer=77309411328 mapped=22134528992 "
    "layout=0x5ae645df5cf7d568 "
    "inventory=f57153febec22463c7789b892d4d084041d722483a93191c81c40ab86be7d9e5 "
    "plan=fa2685e223ad8ea6271c6061041fe8d996b4e6cc70e060588b750732577c92af "
)
EXACT_MARKER_PREFIX = (
    "[metal-gguf-parallel-copied] schema=1 resources=733 bytes=22123538944 "
    + COMMON_MARKER
)
ROUNDED_MARKER_PREFIX = (
    "[metal-gguf-parallel-page-rounded] schema=2 resources=733 "
    "logical_bytes=22123538944 allocated_bytes=22126297088 "
    "padding_bytes=2758144 padded_resources=232 " + COMMON_MARKER
)
TIMING_NAMES = ("allocation_us", "source_us", "copy_us", "binding_us", "ready_us")


class ContractDefect(RuntimeError):
    pass


class ChildContractDefect(ContractDefect):
    def __init__(self, child: str, reason: str) -> None:
        super().__init__(reason)
        self.child = child


class Inconclusive(RuntimeError):
    def __init__(self, stage: str, child: str | None, reasons: list[str]) -> None:
        super().__init__(f"{stage}:{child}:{','.join(reasons)}")
        self.stage = stage
        self.child = child
        self.reasons = reasons


Unsealed = proven.UnsealedPacket
sha256 = proven.sha256
write_json = proven.write_json
write_fsynced = proven.write_fsynced
append_jsonl = proven.append_jsonl
fsync_directory = proven.fsync_directory
json_text = proven.json_text
strict_equal = proven.strict_equal


def command(command: list[str], *, env: dict[str, str] | None = None) -> str:
    return subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    ).stdout


def git(args: list[str], repo: Path = ROOT) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    ).stdout.strip()


def patch_hash(left: str, right: str) -> str:
    result = subprocess.run(
        [
            "git",
            "diff",
            "--binary",
            f"{left}..{right}",
            "--",
            str(METAL_SOURCE.relative_to(ROOT)),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    return hashlib.sha256(result.stdout).hexdigest()


def parent(commit: str) -> str:
    fields = git(["rev-list", "--parents", "-n", "1", commit]).split()
    if len(fields) != 2 or fields[0] != commit:
        raise ContractDefect(f"commit is not single-parent: {commit}")
    return fields[1]


def sibling(
    path: Path, expected: str, allowed: tuple[str, ...] = ()
) -> dict[str, object]:
    if git(["rev-parse", "HEAD"], path) != expected:
        raise ContractDefect(f"sibling commit drifted: {path}")
    status = git(["status", "--porcelain=v1", "--untracked-files=all"], path)
    lines = status.splitlines() if status else []
    if any(not line.startswith("?? ") for line in lines):
        raise ContractDefect(f"sibling tracked state is dirty: {path}")
    untracked = sorted(line[3:] for line in lines)
    if untracked != sorted(allowed):
        raise ContractDefect(f"sibling untracked state drifted: {path}: {untracked}")
    return {"path": str(path), "commit": expected, "allowed_untracked": untracked}


def verify_source() -> dict[str, object]:
    for path in (PREREG, RUNNER, METAL_SOURCE):
        git(["ls-files", "--error-unmatch", str(path.relative_to(ROOT))])
    dirty = git(["status", "--porcelain=v1", "--untracked-files=all"])
    if dirty:
        raise ContractDefect(f"worktree is not clean: {dirty!r}")
    head = git(["rev-parse", "HEAD"])
    implementation = parent(head)
    prereg = parent(implementation)
    base = parent(prereg)
    if (
        base != BASE_COMMIT
        or prereg != PREREG_COMMIT
        or implementation != IMPLEMENTATION_COMMIT
    ):
        raise ContractDefect(
            f"frozen base/R/H chain drifted: {base}/{prereg}/{implementation}"
        )
    prereg_paths = sorted(
        (str(PREREG.relative_to(ROOT)), str(RUNNER.relative_to(ROOT)))
    )
    r_status = git(["diff", "--name-status", f"{base}..{prereg}"]).splitlines()
    if sorted(r_status) != sorted(f"A\t{path}" for path in prereg_paths):
        raise ContractDefect("R must add exactly the preregistration and runner")
    metal_rel = str(METAL_SOURCE.relative_to(ROOT))
    if git(["diff", "--name-status", f"{prereg}..{implementation}"]).splitlines() != [
        f"M\t{metal_rel}"
    ]:
        raise ContractDefect("H must modify only metal_forward.rs")
    if sorted(
        git(["diff", "--name-status", f"{implementation}..{head}"]).splitlines()
    ) != sorted(f"M\t{path}" for path in prereg_paths):
        raise ContractDefect("R2 must modify only the preregistration and runner")
    if sha256(METAL_SOURCE) != FINAL_METAL_SHA256:
        raise ContractDefect("final metal_forward.rs digest drifted")
    if patch_hash(prereg, implementation) != IMPLEMENTATION_PATCH_SHA256:
        raise ContractDefect("R..H file-only binary patch digest drifted")
    build = json.loads(command([str(BENCH), "build-info", "--output", "json"]))
    if not isinstance(build, dict) or any(
        (
            build.get("build_commit") != head,
            build.get("runtime_commit") != head,
            build.get("status") != "match",
            build.get("build_dirty") is not False,
            build.get("runtime_dirty") is not False,
            build.get("build_source_state") != build.get("runtime_source_state"),
        )
    ):
        raise ContractDefect(f"build/runtime R2 identity drifted: {build}")
    return {
        "head": head,
        "preregistration_commit": prereg,
        "implementation_commit": implementation,
        "preflight_repair_commit": head,
        "base_commit": base,
        "metal_source_sha256": FINAL_METAL_SHA256,
        "implementation_patch_sha256": IMPLEMENTATION_PATCH_SHA256,
        "build_identity": build,
        "siblings": [
            sibling(ROOT.parent / "gguf", GGUF_COMMIT),
            sibling(ROOT.parent / "llama-cpp-rs", LLAMA_COMMIT, LLAMA_UNTRACKED),
        ],
    }


def file_identity(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {
        "device": stat.st_dev,
        "inode": stat.st_ino,
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def safe_environments() -> tuple[dict[str, str], dict[str, str], list[str]]:
    return proven.safe_environments()


def capture_host() -> dict[str, object]:
    return proven.capture_host_state()


def capture_vm() -> dict[str, object]:
    return proven.capture_vm_state()


def vm_interval(
    label: str, before: dict[str, object], after: dict[str, object]
) -> dict[str, object]:
    return proven.vm_interval(label, before, after)


def interrupt_reasons(packet_signals: list[int]) -> list[str]:
    return [f"operator_signal_deferred={signum}" for signum in packet_signals]


def check_interrupt(
    packet_signals: list[int], stage: str, child: str | None = None
) -> None:
    if packet_signals:
        raise Inconclusive(stage, child, interrupt_reasons(packet_signals))


def verify_tool_protocol(env: dict[str, str]) -> dict[str, object]:
    probe = subprocess.run(
        ["/usr/bin/time", "-l", "/usr/bin/true"],
        cwd=ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    if probe.returncode or probe.stdout:
        raise ContractDefect("/usr/bin/time protocol drifted")
    resources = proven.process_resources(probe.stderr)
    vm = capture_vm()
    host = capture_host()
    if vm.get("errors"):
        raise ContractDefect("VM capture/parser protocol drifted")
    if host.get("valid") is not True:
        raise Inconclusive("preflight", None, ["host_invalid_at_preflight"])
    return {"time_resources": resources, "vm": vm, "host": host}


def build_manifest(
    child_env: dict[str, str], test_env: dict[str, str], removed: list[str]
) -> dict[str, object]:
    source = verify_source()
    for path, expected in HELPER_HASHES.items():
        if sha256(path) != expected:
            raise ContractDefect(f"proven helper digest drifted: {path}")
    if (
        command(["sw_vers", "-productVersion"]).strip() != MACOS_VERSION
        or command(["sw_vers", "-buildVersion"]).strip() != MACOS_BUILD
    ):
        raise ContractDefect("macOS identity drifted")
    if int(command(["sysctl", "-n", "hw.memsize"])) != HW_MEMSIZE:
        raise ContractDefect("host memory identity drifted")
    cli_sha256_before = sha256(CLI)
    device_output = command([str(CLI), "--info"]).strip()
    cli_sha256_after = sha256(CLI)
    if cli_sha256_before != cli_sha256_after or device_output != DEVICE:
        raise ContractDefect("Metal device identity drifted")
    model_id = file_identity(MODEL)
    model_sha256 = sha256(MODEL)
    prompt_sha256 = sha256(PROMPT)
    if model_id["size_bytes"] != MODEL_SIZE or model_sha256 != MODEL_SHA256:
        raise ContractDefect("model size or SHA-256 drifted")
    if PROMPT.stat().st_size != PROMPT_BYTES or prompt_sha256 != PROMPT_SHA256:
        raise ContractDefect("prompt size or SHA-256 drifted")
    paths = (
        PREREG,
        RUNNER,
        METAL_SOURCE,
        MODEL,
        PROMPT,
        BENCH,
        CLI,
        V0593,
        V0602,
        V0630,
    )
    hashes = {
        str(path): (
            model_sha256
            if path == MODEL
            else prompt_sha256
            if path == PROMPT
            else cli_sha256_after
            if path == CLI
            else sha256(path)
        )
        for path in paths
    }
    return {
        "schema": 1,
        "created_unix_ms": time.time_ns() // 1_000_000,
        "source": source,
        "source_commit": source["head"],
        "model_file_identity": model_id,
        "v0602_observed_runtime_identity": {
            "model_id": RUNTIME_MODEL_ID,
            "tokenizer_id": RUNTIME_TOKENIZER_ID,
            "independently_remeasured": False,
            "used_as_gate": False,
        },
        "macos_product_version": MACOS_VERSION,
        "macos_build_version": MACOS_BUILD,
        "device_probe": {
            "command": [str(CLI), "--info"],
            "output": device_output,
            "sha256": cli_sha256_after,
            "source_authoritative": False,
        },
        "child_environment": child_env,
        "test_environment": test_env,
        "removed_environment": removed,
        "tool_protocol": verify_tool_protocol(child_env),
        "sha256": hashes,
        "cell": {
            "pair_orders": list(PAIR_ORDERS),
            "children": 12,
            "runs": RUNS,
            "scored_repetitions": [3, 4, 5],
            "prompt_bytes": PROMPT_BYTES,
            "prompt_tokens": PROMPT_TOKENS,
            "transitions": TRANSITIONS,
            "trace_ids": TRACE_IDS,
            "prefill_chunk": 1024,
            "kv_capacity": 1024,
            "full_logits_decode": True,
            "generated_token_trace": True,
            "cooldown_s": COOLDOWN_S,
            "retry_count": 0,
        },
        "claim_scope": {
            "v0602_performance_observations_imported": 0,
            "v0602_timed_children_imported": 0,
            "correctness_imported_as_gate": False,
            "loaded_gate_definition_imported": True,
        },
    }


def verify_non_model(manifest: dict[str, object]) -> None:
    if not strict_equal(verify_source(), manifest.get("source")):
        raise ContractDefect("source/build/sibling identity changed")
    child, tests, removed = safe_environments()
    if not strict_equal(
        (child, tests, removed),
        (
            manifest.get("child_environment"),
            manifest.get("test_environment"),
            manifest.get("removed_environment"),
        ),
    ):
        raise ContractDefect("normalized environments changed")
    for text, expected in manifest["sha256"].items():
        path = Path(text)
        if path != MODEL and (not path.is_file() or sha256(path) != expected):
            raise ContractDefect(f"non-model input changed: {path}")


def parse_marker(line: str, arm: str) -> dict[str, int]:
    if arm not in ("A", "B"):
        raise ContractDefect(f"invalid marker arm: {arm!r}")
    prefix = EXACT_MARKER_PREFIX if arm == "A" else ROUNDED_MARKER_PREFIX
    suffix = line.removeprefix(prefix)
    if suffix == line:
        raise ContractDefect(f"{arm} marker static grammar drifted")
    fields = suffix.split(" ")
    if len(fields) != len(TIMING_NAMES):
        raise ContractDefect(f"{arm} marker timing field count drifted")
    values: dict[str, int] = {}
    for field, name in zip(fields, TIMING_NAMES, strict=True):
        value = field.removeprefix(name + "=")
        if value == field or not value.isascii() or not value.isdecimal():
            raise ContractDefect(f"{arm} marker {name} is not unsigned decimal")
        parsed = int(value)
        if str(parsed) != value or parsed > U64_MAX:
            raise ContractDefect(f"{arm} marker {name} is not canonical u64")
        values[name] = parsed
    if abs(values["ready_us"] - sum(values[name] for name in TIMING_NAMES[:4])) > 4:
        raise ContractDefect(f"{arm} marker timing does not reconcile")
    return values


def parse_load(stderr: str, arm: str) -> dict[str, object]:
    if arm not in ("A", "B"):
        raise ContractDefect(f"invalid load arm: {arm!r}")
    expected_tag = (
        "[metal-gguf-parallel-copied]"
        if arm == "A"
        else "[metal-gguf-parallel-page-rounded]"
    )
    if stderr.count("[metal-gguf-") != 1 or stderr.count(expected_tag) != 1:
        raise ContractDefect(f"{arm} storage marker count drifted")
    if stderr.count(POLICY_LINE) != 1 or stderr.count(LEDGER_LINE) != 1:
        raise ContractDefect(f"{arm} policy or ledger count drifted")
    prefixes = (
        "[metal-load] native quantized token embedding policy:",
        "[metal-gguf-",
        "[metal-load-ledger]",
    )
    lines = [line for line in stderr.splitlines() if line.startswith(prefixes)]
    for line in stderr.splitlines():
        if ("[metal-load" in line or "[metal-gguf-" in line) and line not in lines:
            raise ContractDefect(f"unrecognized load line: {line!r}")
    if len(lines) != 3 or lines[0] != POLICY_LINE or lines[2] != LEDGER_LINE:
        raise ContractDefect(f"{arm} load line order drifted")
    return {"recognized_lines": lines, "marker_phase_us": parse_marker(lines[1], arm)}


def recognize_correctness(text: str) -> tuple[list[str], dict[str, object]]:
    prefixes = (
        "[metal-load] native quantized token embedding policy:",
        "[metal-gguf-parallel-copied]",
        "[metal-gguf-parallel-page-rounded]",
        "[metal-load-ledger]",
    )
    harness = f"test {CORRECTNESS_TEST} ... "
    lines: list[str] = []
    for line in text.splitlines():
        if not lines and line == harness + POLICY_LINE:
            lines.append(POLICY_LINE)
        elif line.startswith(prefixes):
            lines.append(line)
        elif "[metal-load" in line or "[metal-gguf-" in line:
            raise ContractDefect(f"malformed correctness load line: {line!r}")
    if (
        len(lines) != 6
        or lines[0] != POLICY_LINE
        or lines[2] != LEDGER_LINE
        or lines[3] != POLICY_LINE
        or lines[5] != LEDGER_LINE
    ):
        raise ContractDefect(f"correctness six-line A/B protocol drifted: {lines!r}")
    return lines, {"A": parse_marker(lines[1], "A"), "B": parse_marker(lines[4], "B")}


def run_token_protocol(env: dict[str, str]) -> dict[str, object]:
    result = subprocess.run(
        [str(BENCH), "decode", "--help"],
        cwd=ROOT,
        env=env,
        check=False,
        capture_output=True,
        start_new_session=True,
    )
    if result.returncode or result.stderr:
        raise ContractDefect("qwen-bench decode help failed")
    text = result.stdout.decode("utf-8")
    if (
        text.count("--generated-token-trace") != 1
        or text.count(
            "Print the exact initial token plus every timed transition result"
        )
        != 1
    ):
        raise ContractDefect("generated-token-trace help protocol drifted")
    path = ARTIFACT / "token-protocol.out"
    write_fsynced(path, result.stdout)
    row = {
        "schema": 1,
        "command": [str(BENCH), "decode", "--help"],
        "returncode": 0,
        "output_sha256": sha256(path),
        "passed": True,
    }
    write_json(ARTIFACT / "token-protocol.json", row)
    return row


def run_correctness(env: dict[str, str]) -> dict[str, object]:
    cmd = [
        "cargo",
        "test",
        "--release",
        "-p",
        "qwen-llm",
        CORRECTNESS_TEST,
        "--",
        "--ignored",
        "--exact",
        "--nocapture",
        "--test-threads=1",
    ]
    path = ARTIFACT / "correctness.out"
    result: subprocess.CompletedProcess[bytes] | None = None
    spawn_error: OSError | None = None
    try:
        with path.open("xb") as output:
            try:
                result = subprocess.run(
                    cmd,
                    cwd=ROOT,
                    env=env,
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    check=False,
                    start_new_session=True,
                )
            except OSError as error:
                spawn_error = error
            output.flush()
            os.fsync(output.fileno())
    except OSError as error:
        raise Unsealed(f"correctness output durability failed: {error}") from error
    if spawn_error is not None:
        raise Inconclusive(
            "correctness",
            None,
            [f"correctness_spawn={type(spawn_error).__name__}:{spawn_error}"],
        ) from spawn_error
    if result is None:
        raise ContractDefect("correctness process returned no result")
    text = path.read_text(encoding="utf-8")
    if result.returncode != 0:
        raise ContractDefect("release bit-exact correctness test failed")
    proven.recognize_test(text, CORRECTNESS_TEST, interleaved=True)
    lines, markers = recognize_correctness(text)
    row = {
        "schema": 1,
        "command": cmd,
        "returncode": 0,
        "output_sha256": sha256(path),
        "recognized_load_lines": lines,
        "marker_phase_us": markers,
        "passed": True,
    }
    write_json(ARTIFACT / "correctness.json", row)
    return row


def warm_model(expected_identity: object) -> dict[str, object]:
    if file_identity(MODEL) != expected_identity:
        raise ContractDefect("model identity drifted before reread")
    digest = hashlib.sha256()
    total = 0
    buffer = bytearray(8 * 1024 * 1024)
    started = time.perf_counter_ns()
    with MODEL.open("rb", buffering=0) as handle:
        while True:
            count = handle.readinto(buffer)
            if not count:
                break
            total += count
            digest.update(memoryview(buffer)[:count])
    if (
        total != MODEL_SIZE
        or digest.hexdigest() != MODEL_SHA256
        or file_identity(MODEL) != expected_identity
    ):
        raise ContractDefect("full model reread identity or SHA-256 drifted")
    return {
        "bytes_read": total,
        "sha256": digest.hexdigest(),
        "wall_ns": time.perf_counter_ns() - started,
        "buffer_bytes": len(buffer),
    }


def condition(
    stem: str, manifest: dict[str, object], packet_signals: list[int]
) -> dict[str, object]:
    verify_non_model(manifest)
    host_before = capture_host()
    vm_before = capture_vm()
    try:
        cache = warm_model(manifest["model_file_identity"])
    except ContractDefect:
        raise
    except OSError as error:
        reasons = [f"model_reread_io={type(error).__name__}:{error}"]
        write_json(
            ARTIFACT / f"{stem}.conditioning.json",
            {
                "schema": 1,
                "artifact_stem": stem,
                "monotonic_ns": time.monotonic_ns(),
                "host_before_cache": host_before,
                "vm_before_cache": vm_before,
                "cache": None,
                "validity_reasons": reasons,
            },
        )
        raise Inconclusive("conditioning", stem, reasons) from error
    time.sleep(COOLDOWN_S)
    samples = []
    for index in range(HOST_SAMPLE_LIMIT):
        verify_non_model(manifest)
        samples.append(capture_host())
        if samples[-1].get("valid") is True:
            break
        if index + 1 < HOST_SAMPLE_LIMIT:
            time.sleep(HOST_SAMPLE_INTERVAL_S)
    vm_spawn = capture_vm()
    interval = vm_interval("conditioning", vm_before, vm_spawn)
    reasons = list(interval["failure_reasons"])
    if not samples or samples[-1].get("valid") is not True:
        reasons.append("host_sampler_exhausted_before_child")
    reasons.extend(interrupt_reasons(packet_signals))
    evidence = {
        "schema": 1,
        "artifact_stem": stem,
        "monotonic_ns": time.monotonic_ns(),
        "host_before_cache": host_before,
        "vm_before_cache": vm_before,
        "cache": cache,
        "cooldown_s": COOLDOWN_S,
        "host_samples": samples,
        "host_before_spawn": samples[-1] if samples else None,
        "vm_before_spawn": vm_spawn,
        "interval": interval,
        "validity_reasons": reasons,
    }
    write_json(ARTIFACT / f"{stem}.conditioning.json", evidence)
    if reasons:
        raise Inconclusive("conditioning", stem, reasons)
    return evidence


def arm_environment(
    base: dict[str, str], arm: str
) -> tuple[dict[str, str], dict[str, str]]:
    if arm not in ("A", "B"):
        raise ContractDefect(f"invalid arm: {arm}")
    delta = {
        "QWEN_GGUF_PARALLEL_COPY": "1" if arm == "A" else "page-rounded-copy",
        "QWEN_GGUF_OWNED_ARENA": "0",
        "QWEN_GGUF_NO_COPY": "0",
    }
    env = base.copy()
    env.update(delta)
    return env, delta


def child_command(prompt: str) -> list[str]:
    return [
        "/usr/bin/time",
        "-l",
        str(BENCH),
        "decode",
        "--model",
        str(MODEL),
        "--prompt",
        prompt,
        "--tokens",
        str(TRANSITIONS),
        "--runs",
        str(RUNS),
        "--prefill-chunk",
        "1024",
        "--kv-capacity",
        "1024",
        "--full-logits-decode",
        "--generated-token-trace",
    ]


def decimal(text: str, label: str) -> float:
    if re.fullmatch(r"(?:0|[1-9][0-9]*)(?:\.[0-9]+)?", text) is None:
        raise ContractDefect(f"{label} is not canonical unsigned decimal")
    value = float(text)
    if not math.isfinite(value) or value <= 0:
        raise ContractDefect(f"{label} is not finite positive")
    return value


def parse_bench(stderr: str, prompt: str) -> dict[str, object]:
    header = (
        f"[bench] model={MODEL} prompt={json.dumps(prompt, ensure_ascii=False)} "
        f"({PROMPT_TOKENS} tokens), gen={TRANSITIONS} tokens, kv_capacity=1024"
    )
    if re.findall(r"^\[bench\] model=.*$", stderr, re.MULTILINE) != [header]:
        raise ContractDefect("benchmark shape drifted")
    for line in (
        f"[bench] {DEVICE}",
        "[bench] === results (5 runs) ===",
        "[bench] decode mode: full-logits",
        "[bench] prefill mode: packed layer-major",
        "[bench] prefill chunk: 1024",
    ):
        if stderr.splitlines().count(line) != 1:
            raise ContractDefect(f"benchmark protocol line drifted: {line}")
    matches = re.findall(
        r"^\[bench\] rep\s+(\d+): prefill\s+([0-9]+(?:\.[0-9]+)?) ms "
        r"\(([0-9]+(?:\.[0-9]+)?) t/s\)\s+decode\s+([0-9]+(?:\.[0-9]+)?) ms "
        r"\(([0-9]+(?:\.[0-9]+)?) t/s\)$",
        stderr,
        re.MULTILINE,
    )
    requests = re.findall(
        r"^\[bench\] rep\s+(\d+) request\s+([0-9]+(?:\.[0-9]+)?) ms$",
        stderr,
        re.MULTILINE,
    )
    if (
        len(matches) != RUNS
        or [int(row[0]) for row in matches] != list(range(1, 6))
        or len(requests) != RUNS
        or [int(row[0]) for row in requests] != list(range(1, 6))
        or len([line for line in stderr.splitlines() if line.startswith("[bench] rep")])
        != 2 * RUNS
    ):
        raise ContractDefect("five-repetition protocol drifted")
    repetitions = []
    for metric, request_row in zip(matches, requests, strict=True):
        rep, p_ms, p_tps, d_ms, d_tps = metric
        row = {
            "rep": int(rep),
            "prefill_ms": decimal(p_ms, "prefill_ms"),
            "prefill_tps_reported": decimal(p_tps, "prefill_tps"),
            "decode_ms": decimal(d_ms, "decode_ms"),
            "decode_tps_reported": decimal(d_tps, "decode_tps"),
            "request_wall_ms": decimal(request_row[1], "request_ms"),
        }
        if (
            abs(PROMPT_TOKENS * 1000 / row["prefill_ms"] - row["prefill_tps_reported"])
            > 1.0
            or abs(TRANSITIONS * 1000 / row["decode_ms"] - row["decode_tps_reported"])
            > 0.2
        ):
            raise ContractDefect("reported throughput does not reconcile")
        if row["request_wall_ms"] + 0.2 < row["prefill_ms"] + row["decode_ms"]:
            raise ContractDefect("request wall excludes phase work")
        repetitions.append(row)
    averages = re.findall(
        r"^\[bench\] request wall: ([0-9]+(?:\.[0-9]+)?) ms avg$", stderr, re.MULTILINE
    )
    if (
        len(averages) != 1
        or abs(
            statistics.mean(row["request_wall_ms"] for row in repetitions)
            - decimal(averages[0], "request_average")
        )
        > 0.11
    ):
        raise ContractDefect("request average drifted")
    traces = re.findall(
        r"^\[bench\] generated token trace: (\[.*\])$", stderr, re.MULTILINE
    )
    protocol_lines = [
        line for line in stderr.splitlines() if "[bench] generated token" in line
    ]
    if len(traces) != 1 or protocol_lines != [
        f"[bench] generated token trace: {traces[0]}"
    ]:
        raise ContractDefect("generated-token-trace line drifted")
    tokens = json.loads(traces[0])
    if (
        not isinstance(tokens, list)
        or len(tokens) != TRACE_IDS
        or any(
            type(token) is not int or token < 0 or token >= 248_320 for token in tokens
        )
    ):
        raise ContractDefect("token trace shape drifted")
    canonical = json.dumps(tokens, separators=(",", ":"))
    if traces[0] != canonical:
        raise ContractDefect("token trace is not compact canonical JSON")
    generated = re.findall(r"^\[bench\] generated: (.*)$", stderr, re.MULTILINE)
    generated_protocol = [
        line for line in stderr.splitlines() if line.startswith("[bench] generated")
    ]
    if len(generated) != 1 or generated_protocol != [
        f"[bench] generated token trace: {traces[0]}",
        f"[bench] generated: {generated[0]}",
    ]:
        raise ContractDefect("generated output line drifted")
    return {
        "repetitions": repetitions,
        "token_trace": tokens,
        "token_trace_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
        "generated_debug": generated[0],
        "generated_debug_sha256": hashlib.sha256(generated[0].encode()).hexdigest(),
    }


def record_launch(
    stem: str,
    arm: str,
    pair: int,
    order: str,
    position: int,
    cmd: list[str],
    delta: dict[str, str],
    manifest: dict[str, object],
) -> None:
    append_jsonl(
        ARTIFACT / "launch-seal.jsonl",
        {
            "schema": 1,
            "event": "launch",
            "unix_ms": time.time_ns() // 1_000_000,
            "monotonic_ns": time.monotonic_ns(),
            "artifact_stem": stem,
            "arm": arm,
            "pair_index": pair,
            "pair_order": order,
            "position": position,
            "command": cmd,
            "environment_delta": delta,
            "source_commit": manifest["source_commit"],
            "conditioning_sha256": sha256(ARTIFACT / f"{stem}.conditioning.json"),
        },
    )


def record_completion(stem: str, returncode: int | None, error: str | None) -> None:
    append_jsonl(
        ARTIFACT / "launch-seal.jsonl",
        {
            "schema": 1,
            "event": "completion",
            "unix_ms": time.time_ns() // 1_000_000,
            "monotonic_ns": time.monotonic_ns(),
            "artifact_stem": stem,
            "returncode": returncode,
            "error": error,
        },
    )


def record_golden(row: dict[str, object]) -> None:
    append_jsonl(
        ARTIFACT / "launch-seal.jsonl",
        {
            "schema": 1,
            "event": "golden-token-trace",
            "unix_ms": time.time_ns() // 1_000_000,
            "monotonic_ns": time.monotonic_ns(),
            "source_artifact_stem": row["artifact_stem"],
            "token_trace": row["bench"]["token_trace"],
            "token_trace_sha256": row["bench"]["token_trace_sha256"],
            "generated_debug_sha256": row["bench"]["generated_debug_sha256"],
        },
    )


def run_child(
    base_env: dict[str, str],
    manifest: dict[str, object],
    prompt: str,
    pair: int,
    order: str,
    position: int,
    arm: str,
    packet_signals: list[int],
) -> dict[str, object]:
    stem = f"loaded-p{pair:02d}-{order.lower()}-r{position}-{arm.lower()}"
    out_path, err_path = ARTIFACT / f"{stem}.out", ARTIFACT / f"{stem}.err"
    conditioning = condition(stem, manifest, packet_signals)
    check_interrupt(packet_signals, "conditioning", stem)
    env, delta = arm_environment(base_env, arm)
    cmd = child_command(prompt)
    process: subprocess.Popen[bytes] | None = None
    returncode: int | None = None
    errors: list[str] = []
    durability: OSError | None = None
    lifecycle = True
    launched = False
    started = time.monotonic_ns()
    try:
        with out_path.open("xb") as stdout, err_path.open("xb") as stderr:
            fsync_directory(ARTIFACT)
            record_launch(stem, arm, pair, order, position, cmd, delta, manifest)
            launched = True
            try:
                process = subprocess.Popen(
                    cmd,
                    cwd=ROOT,
                    env=env,
                    stdout=stdout,
                    stderr=stderr,
                    start_new_session=True,
                )
            except Exception as error:  # noqa: BLE001 - preserve launch evidence
                errors.append(f"spawn={type(error).__name__}:{error}")
            if process is not None:
                returncode, wait_errors, lifecycle = proven.wait_for_child_process(
                    process
                )
                errors.extend(wait_errors)
            try:
                stdout.flush()
                os.fsync(stdout.fileno())
                stderr.flush()
                os.fsync(stderr.fileno())
            except OSError as error:
                durability = error
    except OSError as error:
        durability = error
    finally:
        if launched:
            if not lifecycle:
                errors.append("child_lifecycle_not_reaped")
            if durability:
                errors.append(f"durability={durability}")
            record_completion(stem, returncode, ";".join(errors) if errors else None)
    ended = time.monotonic_ns()
    if durability:
        raise Unsealed(f"child raw durability failed: {stem}: {durability}")
    if not lifecycle:
        raise Unsealed(f"exact child PID could not be reaped: {stem}")
    host_after, vm_after = capture_host(), capture_vm()
    interval = vm_interval("child", conditioning["vm_before_spawn"], vm_after)
    load = bench = resources = None
    contract_error = None
    child_io_error = None
    try:
        if returncode == 0 and not out_path.read_bytes():
            stderr_text = err_path.read_text(encoding="utf-8")
            load = parse_load(stderr_text, arm)
            bench = parse_bench(stderr_text, prompt)
            resources = proven.process_resources(stderr_text)
        elif returncode == 0:
            raise ContractDefect("qwen-bench emitted unexpected stdout")
    except OSError as error:
        child_io_error = f"child_output_io={type(error).__name__}:{error}"
    except Exception as error:  # noqa: BLE001 - normalize parser boundary
        contract_error = f"{type(error).__name__}:{error}"
    reasons = list(interval["failure_reasons"])
    reasons.extend(interrupt_reasons(packet_signals))
    if errors:
        reasons.extend(errors)
    if child_io_error:
        reasons.append(child_io_error)
    if returncode != 0:
        reasons.append(f"child_returncode={returncode}")
    if host_after.get("valid") is not True:
        reasons.append("host_invalid_after_child")
    if resources is not None:
        if resources["block_input_operations"] != 0:
            reasons.append("child_block_input")
        if resources["swaps"] != 0:
            reasons.append("child_swaps")
    post = {
        "schema": 1,
        "artifact_stem": stem,
        "host_after_exit": host_after,
        "vm_after_exit": vm_after,
        "interval": interval,
        "process_resources": resources,
        "loaded_major_faults_advisory": resources.get("page_faults")
        if resources
        else None,
        "validity_reasons": reasons,
        "contract_error": contract_error,
    }
    post_path = ARTIFACT / f"{stem}.post-exit.json"
    write_json(post_path, post)
    fsync_directory(ARTIFACT)
    row = {
        "schema": 1,
        "stage": "loaded",
        "artifact_stem": stem,
        "arm": arm,
        "pair_index": pair,
        "pair_order": order,
        "position": position,
        "command": cmd,
        "environment_delta": delta,
        "returncode": returncode,
        "process_wall_ns": ended - started,
        "stdout_sha256": sha256(out_path),
        "stderr_sha256": sha256(err_path),
        "conditioning_sha256": sha256(ARTIFACT / f"{stem}.conditioning.json"),
        "post_exit_sha256": sha256(post_path),
        "load_contract": load,
        "bench": bench,
        "process_resources": resources,
        "loaded_major_faults_advisory": post["loaded_major_faults_advisory"],
        "valid": not reasons and contract_error is None,
        "validity_reasons": reasons,
        "contract_error": contract_error,
    }
    append_jsonl(ARTIFACT / "attempts.jsonl", row)
    if contract_error:
        raise ChildContractDefect(stem, contract_error)
    if reasons:
        raise Inconclusive("loaded", stem, reasons)
    return row


def positive_ratio(left: float, right: float, label: str) -> float:
    if not all(math.isfinite(value) and value > 0 for value in (left, right)):
        raise ContractDefect(f"invalid ratio operands: {label}")
    return left / right


def late_metrics(row: dict[str, object]) -> dict[str, object]:
    reps = row["bench"]["repetitions"]
    if len(reps) != 5 or [rep["rep"] for rep in reps] != [1, 2, 3, 4, 5]:
        raise ContractDefect("analysis repetition identity drifted")
    late = reps[2:]
    tps = [
        positive_ratio(TRANSITIONS * 1000.0, rep["decode_ms"], "decode_tps")
        for rep in reps
    ]
    late_tps = tps[2:]
    return {
        "scored_repetitions": [3, 4, 5],
        "excluded_repetitions": [1, 2],
        "prefill_ms": statistics.median(rep["prefill_ms"] for rep in late),
        "decode_ms": statistics.median(rep["decode_ms"] for rep in late),
        "request_ms": statistics.median(rep["request_wall_ms"] for rep in late),
        "decode_tps": tps,
        "rep5_over_rep3_tps": tps[4] / tps[2],
        "late_tps_relative_range": (max(late_tps) - min(late_tps))
        / statistics.median(late_tps),
    }


def analyze(rows: list[dict[str, object]]) -> dict[str, object]:
    if len(rows) != 12 or any(row.get("valid") is not True for row in rows):
        raise ContractDefect("performance analysis requires all 12 valid children")
    traces = [row["bench"]["token_trace"] for row in rows]
    if any(not strict_equal(trace, traces[0]) for trace in traces[1:]):
        raise ContractDefect("canonical token-ID traces differ")
    outputs = {row["bench"]["generated_debug_sha256"] for row in rows}
    if len(outputs) != 1:
        raise ContractDefect("generated output hashes differ")
    pairs = []
    for index, order in enumerate(PAIR_ORDERS, 1):
        selected = [row for row in rows if row["pair_index"] == index]
        selected.sort(key=lambda row: row["position"])
        if (
            len(selected) != 2
            or "".join(row["arm"] for row in selected) != order
            or any(row["pair_order"] != order for row in selected)
        ):
            raise ContractDefect(f"pair membership drifted: {index}")
        arm_rows = {row["arm"]: row for row in selected}
        a, b = late_metrics(arm_rows["A"]), late_metrics(arm_rows["B"])
        metric = {
            "pair_index": index,
            "pair_order": order,
            "prefill_a_over_b": positive_ratio(a["prefill_ms"], b["prefill_ms"], "P"),
            "decode_a_over_b": positive_ratio(a["decode_ms"], b["decode_ms"], "D"),
            "request_b_over_a": positive_ratio(b["request_ms"], a["request_ms"], "R"),
            "rss_b_over_a": positive_ratio(
                float(arm_rows["B"]["process_resources"]["maximum_resident_set_size"]),
                float(arm_rows["A"]["process_resources"]["maximum_resident_set_size"]),
                "rss",
            ),
            "footprint_b_over_a": positive_ratio(
                float(arm_rows["B"]["process_resources"]["peak_memory_footprint"]),
                float(arm_rows["A"]["process_resources"]["peak_memory_footprint"]),
                "footprint",
            ),
            "a_late": a,
            "b_late": b,
        }
        metric["stability_gates"] = {
            "a_rep5_over_rep3": 0.98 <= a["rep5_over_rep3_tps"] <= 1.02,
            "b_rep5_over_rep3": 0.98 <= b["rep5_over_rep3_tps"] <= 1.02,
            "a_late_range": a["late_tps_relative_range"] <= 0.03,
            "b_late_range": b["late_tps_relative_range"] <= 0.03,
        }
        metric["performance_gates"] = {
            "prefill_parity": metric["prefill_a_over_b"] >= 0.99,
            "decode_parity": metric["decode_a_over_b"] >= 0.99,
            "request_nonregression": metric["request_b_over_a"] <= 1.01,
            "rss_ratio": metric["rss_b_over_a"] <= 1.05,
            "footprint_ratio": metric["footprint_b_over_a"] <= 1.05,
        }
        pairs.append(metric)
    stable = all(all(row["stability_gates"].values()) for row in pairs)
    performance = all(all(row["performance_gates"].values()) for row in pairs)
    classification = (
        "inconclusive-instability" if not stable else ("go" if performance else "kill")
    )
    return {
        "schema": 1,
        "stage": "loaded-only",
        "pairs": pairs,
        "stable": stable,
        "performance_passes": performance,
        "classification": classification,
        "token_trace": traces[0],
        "token_trace_sha256": rows[0]["bench"]["token_trace_sha256"],
        "generated_debug_sha256": next(iter(outputs)),
        "scored_repetition_count": 36,
        "excluded_repetition_count": 24,
    }


def expected_children() -> list[tuple[int, str, int, str, str]]:
    values = []
    for pair, order in enumerate(PAIR_ORDERS, 1):
        for position, arm in enumerate(order, 1):
            values.append(
                (
                    pair,
                    order,
                    position,
                    arm,
                    f"loaded-p{pair:02d}-{order.lower()}-r{position}-{arm.lower()}",
                )
            )
    return values


def read_attempts() -> list[dict[str, object]]:
    return proven.read_jsonl(ARTIFACT / "attempts.jsonl")


def verify_attempt_artifacts(rows: list[dict[str, object]]) -> None:
    expected = expected_children()
    manifest = json.loads((ARTIFACT / "manifest.json").read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ContractDefect("manifest is malformed during artifact verification")
    if len(rows) > 12:
        raise ContractDefect("attempt ledger exceeds frozen schedule")
    for row, (pair, order, position, arm, stem) in zip(
        rows, expected[: len(rows)], strict=True
    ):
        if (
            row.get("pair_index"),
            row.get("pair_order"),
            row.get("position"),
            row.get("arm"),
            row.get("artifact_stem"),
        ) != (pair, order, position, arm, stem):
            raise ContractDefect("attempt order drifted")
        for suffix, key in (
            ("out", "stdout_sha256"),
            ("err", "stderr_sha256"),
            ("conditioning.json", "conditioning_sha256"),
            ("post-exit.json", "post_exit_sha256"),
        ):
            path = ARTIFACT / f"{stem}.{suffix}"
            if not path.is_file() or sha256(path) != row.get(key):
                raise ContractDefect(f"attempt artifact binding drifted: {path.name}")
    events = proven.read_jsonl(ARTIFACT / "launch-seal.jsonl")
    golden = bool(rows and rows[0].get("valid") is True)
    if len(events) != 2 * len(rows) + int(golden):
        raise ContractDefect("launch/completion/golden event count drifted")
    cursor = 0
    for index, row in enumerate(rows):
        launch, completion = events[cursor : cursor + 2]
        cursor += 2
        if (
            launch.get("schema") != 1
            or launch.get("event") != "launch"
            or launch.get("arm") != row["arm"]
            or launch.get("pair_index") != row["pair_index"]
            or launch.get("pair_order") != row["pair_order"]
            or launch.get("position") != row["position"]
            or not strict_equal(launch.get("command"), row["command"])
            or not strict_equal(
                launch.get("environment_delta"), row["environment_delta"]
            )
            or launch.get("source_commit") != manifest.get("source_commit")
            or launch.get("conditioning_sha256") != row["conditioning_sha256"]
            or completion.get("schema") != 1
            or completion.get("event") != "completion"
            or launch.get("artifact_stem") != row["artifact_stem"]
            or completion.get("artifact_stem") != row["artifact_stem"]
            or completion.get("returncode") != row["returncode"]
            or (row.get("valid") is True and completion.get("error") is not None)
        ):
            raise ContractDefect("launch/completion event order drifted")
        if index == 0 and golden:
            event = events[cursor]
            cursor += 1
            if (
                event.get("schema") != 1
                or event.get("event") != "golden-token-trace"
                or event.get("source_artifact_stem") != row["artifact_stem"]
                or not strict_equal(
                    event.get("token_trace"), row["bench"]["token_trace"]
                )
                or event.get("token_trace_sha256") != row["bench"]["token_trace_sha256"]
                or event.get("generated_debug_sha256")
                != row["bench"]["generated_debug_sha256"]
            ):
                raise ContractDefect("golden trace event drifted")
    if cursor != len(events):
        raise ContractDefect("launch ledger has unconsumed events")


def final_identity(manifest: dict[str, object]) -> dict[str, object]:
    result = {
        "schema": 1,
        "matches": False,
        "model_file_identity": None,
        "model_sha256": None,
        "source": None,
        "error": None,
        "unix_ms": time.time_ns() // 1_000_000,
    }
    try:
        verify_non_model(manifest)
        identity, digest, source = file_identity(MODEL), sha256(MODEL), verify_source()
        result.update(
            {
                "model_file_identity": identity,
                "model_sha256": digest,
                "source": source,
                "matches": identity == manifest["model_file_identity"]
                and digest == MODEL_SHA256,
            }
        )
    except Exception as error:  # noqa: BLE001 - persist final identity failure
        result["error"] = f"{type(error).__name__}:{error}"
    write_json(ARTIFACT / "final-identity.json", result)
    if result["matches"] is not True:
        raise ContractDefect(f"final identity failed: {result['error']}")
    return result


def make_decision(
    manifest: dict[str, object],
    correctness: object,
    token: object,
    stage: object,
    status: str,
    stopped: str,
    failed: str | None,
    reasons: list[str],
) -> dict[str, object]:
    if status not in (
        "implementation_or_contract_defect",
        "inconclusive",
        "kill",
        "go",
    ):
        raise ContractDefect(f"invalid status: {status}")
    go = status == "go"
    return {
        "schema": 1,
        "status": status,
        "authority": "no-production-authority",
        "force_authorized": False,
        "successor_authorization": "preregister-one-bench-only-page-aligned-a3b-sidecar-image"
        if go
        else "none",
        "profile_disposition": (
            "page-rounded-compatible-sidecar-oracle-authorized"
            if go
            else "closed-page-rounded-independent-resource-image-route"
            if status == "kill"
            else "unresolved-no-authority"
        ),
        "stopped_after": stopped,
        "failed_child": failed,
        "reasons": reasons,
        "source_commit": manifest["source_commit"],
        "correctness": correctness,
        "token_protocol": token,
        "stage": stage,
        "attempts_sha256": sha256(ARTIFACT / "attempts.jsonl"),
        "claim_scope": {
            **manifest["claim_scope"],
            "comparison": "exact-parallel-copy-vs-page-rounded-parallel-copy",
            "fresh_or_cold_timing": False,
            "converter_authorized": False,
            "product_authorized": False,
            "default_authorized": False,
            "no_copy_conclusion": False,
            "kill_scope": "733-independent-resource-page-rounded-image-construction"
            if status == "kill"
            else None,
        },
    }


def publish(decision: dict[str, object]) -> None:
    rows = read_attempts()
    verify_attempt_artifacts(rows)
    write_json(ARTIFACT / "decision.json", decision)
    members = sorted(
        path
        for path in ARTIFACT.iterdir()
        if path.name not in ("artifact-inventory.sha256", "packet-complete.json")
    )
    if any(not path.is_file() or path.is_symlink() for path in members):
        raise Unsealed("artifact directory contains non-regular member")
    if len(rows) == 12:
        expected_names = {
            "manifest.json",
            "token-protocol.out",
            "token-protocol.json",
            "correctness.out",
            "correctness.json",
            "attempts.jsonl",
            "launch-seal.jsonl",
            "final-identity.json",
            "decision.json",
        }
        for _pair, _order, _position, _arm, stem in expected_children():
            expected_names.update(
                {
                    f"{stem}.out",
                    f"{stem}.err",
                    f"{stem}.conditioning.json",
                    f"{stem}.post-exit.json",
                }
            )
        if {path.name for path in members} != expected_names:
            raise Unsealed("complete packet member names drifted")
    if (
        decision.get("status") in ("go", "kill", "inconclusive")
        and len(rows) == 12
        and len(members) != 57
    ):
        raise Unsealed(f"complete packet member arithmetic drifted: {len(members)}")
    inventory = ARTIFACT / "artifact-inventory.sha256"
    write_fsynced(
        inventory,
        "".join(
            f"{sha256(path)}  {path.relative_to(ROOT)}\n" for path in members
        ).encode(),
    )
    write_json(
        ARTIFACT / "packet-complete.json",
        {
            "schema": 1,
            "decision_sha256": sha256(ARTIFACT / "decision.json"),
            "inventory_sha256": sha256(inventory),
            "inventory_members": len(members),
            "final_files": len(members) + 2,
        },
    )
    fsync_directory(ARTIFACT)
    if len(rows) == 12 and len(list(ARTIFACT.iterdir())) != 59:
        raise Unsealed("final packet file arithmetic drifted")


def execute(
    manifest: dict[str, object],
    child_env: dict[str, str],
    test_env: dict[str, str],
    prompt: str,
    packet_signals: list[int],
) -> dict[str, object]:
    correctness = token = stage = None
    rows: list[dict[str, object]] = []
    status, stopped, failed, reasons = (
        "implementation_or_contract_defect",
        "preflight",
        None,
        [],
    )
    try:
        stopped = "token-protocol"
        token = run_token_protocol(child_env)
        check_interrupt(packet_signals, stopped)
        stopped = "correctness"
        correctness = run_correctness(test_env)
        check_interrupt(packet_signals, stopped)
        stopped = "loaded"
        golden = None
        for pair, order, position, arm, _stem in expected_children():
            row = run_child(
                child_env,
                manifest,
                prompt,
                pair,
                order,
                position,
                arm,
                packet_signals,
            )
            rows.append(row)
            if golden is None:
                golden = row["bench"]["token_trace"]
                record_golden(row)
            elif not strict_equal(row["bench"]["token_trace"], golden):
                raise ChildContractDefect(
                    row["artifact_stem"], "timed token identity differs"
                )
            check_interrupt(packet_signals, stopped, row["artifact_stem"])
        stage = analyze(rows)
        status = stage["classification"]
        if status == "inconclusive-instability":
            status, reasons = "inconclusive", ["inconclusive-instability"]
        stopped = "loaded-only"
    except Inconclusive as error:
        status, stopped, failed, reasons = (
            "inconclusive",
            error.stage,
            error.child,
            error.reasons,
        )
    except ChildContractDefect as error:
        status, failed, reasons = (
            "implementation_or_contract_defect",
            error.child,
            [str(error)],
        )
    except ContractDefect as error:
        status, failed, reasons = (
            "implementation_or_contract_defect",
            rows[-1]["artifact_stem"] if rows else None,
            [str(error)],
        )
    except Unsealed:
        raise
    except Exception as error:  # noqa: BLE001 - seal classifiable defect
        status, failed, reasons = (
            "implementation_or_contract_defect",
            rows[-1]["artifact_stem"] if rows else None,
            [f"{type(error).__name__}:{error}"],
        )
    try:
        final_identity(manifest)
    except ContractDefect as error:
        status, stopped, failed, reasons = (
            "implementation_or_contract_defect",
            "final-identity",
            None,
            [str(error)],
        )
    return make_decision(
        manifest, correctness, token, stage, status, stopped, failed, reasons
    )


def synthetic_rows(
    *, stability: bool = True, performance: bool = True
) -> list[dict[str, object]]:
    rows = []
    trace = list(range(TRACE_IDS))
    for pair, order, position, arm, stem in expected_children():
        base_ms = 100.0
        factor = 1.0 if arm == "A" else (1.005 if performance else 1.03)
        decode = [
            base_ms * factor * value
            for value in (
                (1.2, 1.1, 1.0, 1.0, 1.0) if stability else (1.2, 1.1, 1.0, 1.04, 1.08)
            )
        ]
        reps = [
            {
                "rep": index + 1,
                "prefill_ms": 50.0 * factor,
                "decode_ms": value,
                "request_wall_ms": 160.0 * factor,
            }
            for index, value in enumerate(decode)
        ]
        rows.append(
            {
                "valid": True,
                "stage": "loaded",
                "pair_index": pair,
                "pair_order": order,
                "position": position,
                "arm": arm,
                "artifact_stem": stem,
                "bench": {
                    "repetitions": reps,
                    "token_trace": trace,
                    "token_trace_sha256": "t",
                    "generated_debug_sha256": "g",
                },
                "process_resources": {
                    "maximum_resident_set_size": 1000
                    if arm == "A"
                    else (1040 if performance else 1060),
                    "peak_memory_footprint": 2000
                    if arm == "A"
                    else (2080 if performance else 2120),
                },
            }
        )
    return rows


def self_test() -> None:
    expected = expected_children()
    assert len(expected) == 12 and [row[1] for row in expected[::2]] == list(
        PAIR_ORDERS
    )
    assert 12 * 3 == 36 and 9 + 12 * 4 == 57 and 57 + 2 == 59
    exact = EXACT_MARKER_PREFIX + " ".join(
        f"{name}={value}"
        for name, value in zip(TIMING_NAMES, (1, 2, 3, 4, 10), strict=True)
    )
    rounded = ROUNDED_MARKER_PREFIX + " ".join(
        f"{name}={value}"
        for name, value in zip(TIMING_NAMES, (1, 2, 3, 4, 10), strict=True)
    )
    assert parse_marker(exact, "A")["ready_us"] == 10
    assert parse_marker(rounded, "B")["ready_us"] == 10
    correctness_text = "\n".join(
        (
            f"test {CORRECTNESS_TEST} ... {POLICY_LINE}",
            exact,
            LEDGER_LINE,
            POLICY_LINE,
            rounded,
            LEDGER_LINE,
        )
    )
    recognized, phases = recognize_correctness(correctness_text)
    assert len(recognized) == 6 and phases["A"]["ready_us"] == 10
    try:
        parse_marker(
            rounded.replace("allocated_bytes=22126297088", "allocated_bytes=1"), "B"
        )
    except ContractDefect:
        pass
    else:
        raise AssertionError("rounded marker mutation accepted")
    go = analyze(synthetic_rows())
    assert go["classification"] == "go" and go["scored_repetition_count"] == 36
    assert all(pair["a_late"]["excluded_repetitions"] == [1, 2] for pair in go["pairs"])
    excluded_mutation = copy.deepcopy(synthetic_rows())
    for row in excluded_mutation:
        for index, factor in ((0, 1000.0), (1, 0.001)):
            repetition = row["bench"]["repetitions"][index]
            repetition["prefill_ms"] *= factor
            repetition["decode_ms"] *= factor
            repetition["request_wall_ms"] *= factor
    mutated = analyze(excluded_mutation)
    scored_keys = (
        "prefill_a_over_b",
        "decode_a_over_b",
        "request_b_over_a",
        "rss_b_over_a",
        "footprint_b_over_a",
        "stability_gates",
        "performance_gates",
    )
    for baseline_pair, mutated_pair in zip(go["pairs"], mutated["pairs"], strict=True):
        assert all(
            strict_equal(baseline_pair[key], mutated_pair[key]) for key in scored_keys
        )
        for arm in ("a_late", "b_late"):
            for key in (
                "prefill_ms",
                "decode_ms",
                "request_ms",
                "rep5_over_rep3_tps",
                "late_tps_relative_range",
            ):
                assert baseline_pair[arm][key] == mutated_pair[arm][key]
    assert (
        analyze(synthetic_rows(stability=False))["classification"]
        == "inconclusive-instability"
    )
    assert analyze(synthetic_rows(performance=False))["classification"] == "kill"
    print(
        json_text(
            {
                "status": "self-test-pass",
                "children": 12,
                "scored_repetitions": 36,
                "packet_members": 57,
                "final_files": 59,
            },
            pretty=True,
        )
    )


def apply_late_interrupt(
    decision: dict[str, object], packet_signals: list[int]
) -> dict[str, object]:
    if (
        not packet_signals
        or decision.get("status") == "implementation_or_contract_defect"
    ):
        return decision
    updated = copy.deepcopy(decision)
    updated.update(
        {
            "status": "inconclusive",
            "successor_authorization": "none",
            "profile_disposition": "unresolved-no-authority",
            "stopped_after": "operator-signal-before-publication",
            "failed_child": None,
            "reasons": interrupt_reasons(packet_signals),
        }
    )
    return updated


def run_packet(*, preflight_only: bool) -> None:
    if ARTIFACT.exists():
        raise RuntimeError(f"refusing to reuse artifact directory: {ARTIFACT}")
    child_env, test_env, removed = safe_environments()
    manifest = build_manifest(child_env, test_env, removed)
    prompt = PROMPT.read_text(encoding="utf-8")
    if len(prompt.encode("utf-8")) != PROMPT_BYTES:
        raise ContractDefect("prompt runtime bytes drifted")
    if preflight_only:
        print(
            json_text(
                {
                    "status": "preflight-pass",
                    "source_commit": manifest["source_commit"],
                    "children": 12,
                    "model_sha256": MODEL_SHA256,
                },
                pretty=True,
            )
        )
        return
    packet_signals: list[int] = []

    def defer_sigint(signum: int, _frame: object) -> None:
        packet_signals.append(signum)

    prior_handler = signal.signal(signal.SIGINT, defer_sigint)
    try:
        ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
        ARTIFACT.mkdir()
        fsync_directory(ARTIFACT.parent)
        write_json(ARTIFACT / "manifest.json", manifest)
        write_fsynced(ARTIFACT / "attempts.jsonl", b"")
        write_fsynced(ARTIFACT / "launch-seal.jsonl", b"")
        fsync_directory(ARTIFACT)
        decision = execute(manifest, child_env, test_env, prompt, packet_signals)
        previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT})
        try:
            if signal.SIGINT in signal.sigpending():
                packet_signals.append(signal.SIGINT)
            decision = apply_late_interrupt(decision, packet_signals)
            publish(decision)
        finally:
            signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
    finally:
        signal.signal(signal.SIGINT, prior_handler)
    print(json_text(decision, pretty=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--self-test", action="store_true")
    group.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        run_packet(preflight_only=args.preflight_only)


if __name__ == "__main__":
    main()
