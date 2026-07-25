#!/usr/bin/env python3
"""v0.628 dense-27B default-ColdOnly target-file-cold composition guard."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import time

import v0605_dense27b_parallel_copied_loader as mechanics


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = ROOT / "target/profiles/v0628-dense27b-pread-cold-composition-p1"
PREREG = ROOT / "docs/bench/v0628-dense27b-pread-cold-composition.md"
RUNNER = Path(__file__).resolve()
EXAMPLE_SOURCE = ROOT / "crates/qwen-llm/examples/first_byte_spike.rs"
EXAMPLE_BINARY = ROOT / "target/release/examples/first_byte_spike"
MODEL = Path("/Users/tito/models/Qwen3.6-27B-Q4_K_M.gguf")
QWEN = ROOT / "target/release/qwen"
QWEN_BENCH = ROOT / "target/release/qwen-bench"
V0627_ROOT = ROOT / "target/profiles/v0627-dense27b-parallel-pread-fresh-repair-p1"
V0627_DECISION = V0627_ROOT / "decision.json"
V0627_INVENTORY = V0627_ROOT / "artifact-inventory.sha256"
V0627_COMPLETE = V0627_ROOT / "packet-complete.json"
V0627_PROMPT = (
    ROOT / "docs/bench/tokenizer-prompts/current-reva-n8-interactive-qwen36.txt"
)

BASE_COMMIT = "7fc4c6b2e08379bd45be8b9db8dc742aa5e7a04e"
DENSE_COMMIT = "7fb0488a5c075d8c62c8f9802352f717ec95b085"
V0627_SOURCE = "26d805a911b55bdc5bf141637d9c7d8cd676f251"
EXPECTED_PREREG_SHA256 = (
    "7b12eb5d8820b156c5cc02b414c9e20910511543a9f4b98069f08e33f833cd6a"
)
EXPECTED_MODEL_SHA256 = (
    "5ed60d0af4650a854b1755bd392f9aef4872643dc25a254bc68043fa638392a0"
)
EXPECTED_V0627_PROMPT_SHA256 = (
    "e265de9742d1b22e566fc108ae26331ccf46166c6f071e73f211e0a1a7e8b474"
)
EXPECTED_MODEL_SIZE = 16_817_244_384
EXPECTED_PAGE_SIZE = 16_384
EXPECTED_MODEL_PAGES = 1_026_444
EXPECTED_DEVICE = (
    "device: Apple M4 Max | unified_memory=true | max_threadgroup_memory=32768 bytes"
)
EXPECTED_HW_MEMSIZE = 137_438_953_472
EXPECTED_V0627_DECISION = (
    "637184029bab7484c7185ff62bda13d2dbb775628471e031fe1951c25ab5f749"
)
EXPECTED_V0627_INVENTORY = (
    "110941aaac80583f54a8122a94af15936114cef54e5c3d10f08b8c9d4c6ec9d8"
)
EXPECTED_V0627_COMPLETE = (
    "c9e6d1b25bc74534de88a6ae4726c5167c50b5e5e819a2ca7afae9e865b875af"
)
V0627_SUCCESSOR = (
    "preregister-and-execute-separate-default-coldonly-target-file-cold-guard-only"
)
PASS_SUCCESSOR = "preregister-separate-short-period-loaded-stability-packet-only"
CPU_TEST = "tests::default_policy_selection_is_exact"
PAIR_ORDERS = ("AB", "BA")
PROMPT = "The capital of France is"
PROMPT_IDS = "[760, 6511, 314, 9338, 369]"
READ_MIN_GIB = 12.50
READ_MAX_GIB = 17.25
READ_MIN_BYTES = 13_421_772_800
READ_MAX_BYTES = 18_522_046_464
VALID_STATUSES = {
    "implementation_or_contract_defect",
    "inconclusive",
    "kill",
    "cold_composition_pass",
}
POLICY_LINE = (
    "[metal-load] native quantized token embedding policy: "
    "auto-promoted (Q4_K [5120, 248320])"
)
LEDGER_LINE = (
    "[metal-load-ledger] source=851/16806250496 "
    "direct_copy=851/16806250496 direct_view=0/0 direct_alias=0/0 "
    "tail_fallback=0/0 converted=0/0/0 derived=0/0"
)
MARKER_PREFIX = (
    "[metal-gguf-parallel-pread] schema=2 profile=dense27b-q4km-v1 "
    "resources=851 bytes=16806250496 workers=4 cuts=136,377,618 "
    "tasks=136,241,241,233 "
    "worker_bytes=4194110464,4214375808,4204933376,4192830848 "
    "w0_first=2,output.weight,0,10993888,1042944000 "
    "w0_last=135,blk.9.ssm_norm.weight,0,4205103840,512 "
    "w1_first=136,blk.9.ssm_out.weight,0,4205104352,21626880 "
    "w1_last=379,blk.28.attn_qkv.weight,0,8376472160,43008000 "
    "w2_first=378,blk.28.ffn_down.weight,0,8419480160,73113600 "
    "w2_last=618,blk.46.ffn_down.weight,0,12551299936,73113600 "
    "w3_first=616,blk.46.ffn_gate.weight,0,12624413536,50135040 "
    "w3_last=844,blk.63.post_attention_norm.weight,0,16817223904,20480 "
    "create=shared,default_cache,default observed=shared,default_cache,tracked "
    "page=16384 alignment=32 max_buffer=77309411328 "
    "mapped=16817244384 layout=0xd116405fd99f54d9 "
    "inventory=50e9af4e4f590fc85687a71f5602ce035e7fdf0e2a31e928b2c7a2be10458a07 "
)
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
STABLE_STDOUT_TOKENS = (
    "harness_source_sha256=",
    "policy_source=",
    "policy_configured=",
    "target_identity_before_invalidate=",
    "target_identity_after_invalidate=",
    "policy_observed=",
    "target_identity_after_load=",
    "prefetch_exact=",
    "timing_exact=",
    "rusage_exact=",
)


UnsealedPacket = mechanics.UnsealedPacket


class InconclusivePacket(RuntimeError):
    def __init__(self, stage: str, child: str, reasons: list[str]) -> None:
        super().__init__(f"{stage}:{child}: {reasons}")
        self.stage = stage
        self.child = child
        self.reasons = reasons


def sha256(path: Path) -> str:
    return mechanics.common.sha256_file(path)


def json_value(path: Path) -> dict[str, object]:
    value = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=mechanics.reject_json_constant,
    )
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def strict_equal(left: object, right: object) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            strict_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(
            strict_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    return left == right


def json_text(value: object, *, pretty: bool = False) -> str:
    return mechanics.json_text(value, pretty=pretty)


def command_text(command: list[str], env: dict[str, str] | None = None) -> str:
    return mechanics.command_text(command, env)


def write_fsynced_json(path: Path, value: object) -> None:
    try:
        with path.open("x", encoding="utf-8") as output:
            output.write(json_text(value, pretty=True) + "\n")
            output.flush()
            os.fsync(output.fileno())
        mechanics.fsync_directory(path.parent)
    except OSError as error:
        raise UnsealedPacket(
            f"durable JSON write failed: {path.name}: {error}"
        ) from error


def append_fsynced(path: Path, value: object) -> None:
    try:
        with path.open("a", encoding="utf-8") as output:
            output.write(json_text(value) + "\n")
            output.flush()
            os.fsync(output.fileno())
        mechanics.fsync_directory(path.parent)
    except OSError as error:
        raise UnsealedPacket(f"durable append failed: {path.name}: {error}") from error


def inventory_members(path: Path) -> dict[Path, str]:
    members: dict[Path, str] = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split("  ", 1)
        if len(fields) != 2 or re.fullmatch(r"[0-9a-f]{64}", fields[0]) is None:
            raise RuntimeError(f"v0.627 inventory row {number} is malformed")
        relative = Path(fields[1])
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError(f"v0.627 inventory row {number} is unsafe")
        member = ROOT / relative
        if member in members or not member.is_file() or member.is_symlink():
            raise RuntimeError(
                f"v0.627 member is missing, duplicate, or unsafe: {member}"
            )
        if sha256(member) != fields[0]:
            raise RuntimeError(f"v0.627 member drifted: {member}")
        members[member] = fields[0]
    return members


def all_true_gates(value: object, label: str) -> None:
    if not isinstance(value, dict) or not value:
        raise RuntimeError(f"v0.627 {label} gates are missing")
    if any(type(item) is not bool or item is not True for item in value.values()):
        raise RuntimeError(f"v0.627 {label} gates are not all true")


def verify_v0627_bridge() -> dict[Path, str]:
    seals = {
        V0627_DECISION: EXPECTED_V0627_DECISION,
        V0627_INVENTORY: EXPECTED_V0627_INVENTORY,
        V0627_COMPLETE: EXPECTED_V0627_COMPLETE,
    }
    for path, digest in seals.items():
        if not path.is_file() or path.is_symlink() or sha256(path) != digest:
            raise RuntimeError(f"v0.627 seal drifted: {path}")
    complete = json_value(V0627_COMPLETE)
    if not strict_equal(
        complete,
        {
            "schema": 1,
            "decision_sha256": EXPECTED_V0627_DECISION,
            "inventory_sha256": EXPECTED_V0627_INVENTORY,
        },
    ):
        raise RuntimeError("v0.627 completion binding drifted")
    members = inventory_members(V0627_INVENTORY)
    if len(members) != 84 or members.get(V0627_DECISION) != EXPECTED_V0627_DECISION:
        raise RuntimeError("v0.627 inventory membership drifted")
    observed = list(V0627_ROOT.iterdir())
    expected = set(members) | {V0627_INVENTORY, V0627_COMPLETE}
    if len(observed) != 86 or set(observed) != expected:
        raise RuntimeError("v0.627 final 86-file directory drifted")
    if any(not path.is_file() or path.is_symlink() for path in observed):
        raise RuntimeError("v0.627 directory contains a non-regular member")

    decision = json_value(V0627_DECISION)
    if (
        decision.get("schema") != 1
        or decision.get("status") != "fresh_effect_pass"
        or decision.get("authority") != "none"
        or decision.get("force_authorized") is not False
        or decision.get("successor_authorization") != V0627_SUCCESSOR
        or decision.get("source_commit") != V0627_SOURCE
    ):
        raise RuntimeError("v0.627 terminal bridge drifted")
    selectors = decision.get("cpu_selectors")
    correctness = decision.get("correctness")
    stages = decision.get("stages")
    if (
        not isinstance(selectors, dict)
        or selectors.get("passed") is not True
        or not isinstance(selectors.get("tests"), list)
        or len(selectors["tests"]) != 4
        or any(row.get("passed") is not True for row in selectors["tests"])
        or not isinstance(correctness, dict)
        or correctness.get("passed") is not True
        or not isinstance(stages, dict)
        or not isinstance(stages.get("fresh_128"), dict)
    ):
        raise RuntimeError("v0.627 selector/correctness bridge drifted")
    fresh = stages["fresh_128"]
    all_true_gates(fresh.get("gates"), "fresh")
    if fresh.get("passes") is not True:
        raise RuntimeError("v0.627 fresh result is not passed")

    attempts = V0627_ROOT / "attempts.jsonl"
    rows = [
        json.loads(line, parse_constant=mechanics.reject_json_constant)
        for line in attempts.read_text(encoding="utf-8").splitlines()
    ]
    stems = [row.get("artifact_stem") for row in rows if isinstance(row, dict)]
    expected_orders = ("AB", "BA", "BA", "AB", "AB", "BA")
    expected_stems = [
        f"fresh-p{pair_index:02d}-{order.lower()}-r{position}-{arm.lower()}"
        for pair_index, order in enumerate(expected_orders, 1)
        for position, arm in enumerate(order, 1)
    ]
    if (
        len(rows) != 12
        or stems != expected_stems
        or len(set(stems)) != 12
        or any(
            row.get("valid") is not True
            or row.get("pair_index") != pair_index
            or row.get("pair_order") != order
            or row.get("position") != position
            or row.get("arm") != arm
            for row, (pair_index, order, position, arm) in zip(
                rows,
                (
                    (pair_index, order, position, arm)
                    for pair_index, order in enumerate(expected_orders, 1)
                    for position, arm in enumerate(order, 1)
                ),
                strict=True,
            )
        )
        or decision.get("attempts_sha256") != members.get(attempts)
    ):
        raise RuntimeError("v0.627 12-attempt bridge drifted")
    launch = V0627_ROOT / "launch-seal.jsonl"
    events = [
        json.loads(line, parse_constant=mechanics.reject_json_constant)
        for line in launch.read_text(encoding="utf-8").splitlines()
    ]
    if len(events) != 24:
        raise RuntimeError("v0.627 launch/completion count drifted")
    for index, stem in enumerate(stems):
        first, second = events[index * 2 : index * 2 + 2]
        if (
            first.get("event") != "launch"
            or second.get("event") != "completion"
            or first.get("artifact_stem") != stem
            or second.get("artifact_stem") != stem
            or second.get("returncode") != 0
        ):
            raise RuntimeError("v0.627 launch/completion sequence drifted")

    manifest = json_value(V0627_ROOT / "manifest.json")
    build = manifest.get("build_identity")
    hashes = manifest.get("sha256")
    if (
        manifest.get("source_commit") != V0627_SOURCE
        or manifest.get("authority") != "none"
        or not isinstance(build, dict)
        or build.get("build_commit") != V0627_SOURCE
        or build.get("runtime_commit") != V0627_SOURCE
        or build.get("build_dirty") is not False
        or build.get("runtime_dirty") is not False
        or build.get("build_source_state") != build.get("runtime_source_state")
        or not isinstance(hashes, dict)
        or hashes.get(str(MODEL)) != EXPECTED_MODEL_SHA256
        or hashes.get(str(V0627_PROMPT)) != EXPECTED_V0627_PROMPT_SHA256
        or manifest.get("expected_stdout_sha256")
        != "c94cb4d2661b181e91ab5db8c65ff252fcd97064073ff332d60715052b784cd0"
        or manifest.get("prompt_tokens") != 419
        or manifest.get("output_tokens") != 128
    ):
        raise RuntimeError("v0.627 source/build/model/prompt/golden bridge drifted")
    qwen = build.get("qwen_identity")
    embedded = qwen.get("embedded") if isinstance(qwen, dict) else None
    if not isinstance(embedded, dict) or embedded != {
        "commit": V0627_SOURCE,
        "build_source_state": build.get("build_source_state"),
    }:
        raise RuntimeError("v0.627 embedded qwen identity drifted")
    final = json_value(V0627_ROOT / "final-model-sha256.json")
    if (
        final.get("matches") is not True
        or final.get("actual_sha256") != EXPECTED_MODEL_SHA256
        or final.get("expected_sha256") != EXPECTED_MODEL_SHA256
        or final.get("observed_file_identity_before")
        != final.get("expected_file_identity")
        or final.get("observed_file_identity_after")
        != final.get("expected_file_identity")
    ):
        raise RuntimeError("v0.627 final model identity drifted")
    return {
        **members,
        V0627_INVENTORY: seals[V0627_INVENTORY],
        V0627_COMPLETE: seals[V0627_COMPLETE],
    }


def changed_paths(revision: str, pathspecs: list[str] | None = None) -> list[str]:
    command = ["git", "diff", "--name-only", revision]
    if pathspecs:
        command.extend(["--", *pathspecs])
    return [line for line in command_text(command).splitlines() if line]


def validate_build(value: object, commit: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise RuntimeError("qwen-bench build-info is not an object")
    if (
        value.get("build_commit") != commit
        or value.get("runtime_commit") != commit
        or value.get("status") != "match"
        or value.get("build_dirty") is not False
        or value.get("runtime_dirty") is not False
        or value.get("build_source_state") != value.get("runtime_source_state")
    ):
        raise RuntimeError("qwen-bench build/runtime identity mismatch")
    return value


def source_and_build_identity() -> tuple[str, dict[str, object]]:
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError("runner cwd is not repository root")
    for path in (PREREG, RUNNER, EXAMPLE_SOURCE):
        command_text(
            ["git", "ls-files", "--error-unmatch", str(path.relative_to(ROOT))]
        )
    if command_text(["git", "status", "--porcelain=v1"]).strip():
        raise RuntimeError("source worktree is dirty")
    commit = command_text(["git", "rev-parse", "HEAD"]).strip()
    if command_text(["git", "rev-parse", "HEAD^^"]).strip() != BASE_COMMIT:
        raise RuntimeError("implementation HEAD is not two commits above certification")
    packet = sorted(changed_paths(f"{BASE_COMMIT}..HEAD^"))
    expected_packet = sorted(str(path.relative_to(ROOT)) for path in (PREREG, RUNNER))
    if packet != expected_packet:
        raise RuntimeError(f"packet commit boundary drifted: {packet}")
    implementation = changed_paths("HEAD^..HEAD")
    if implementation != [str(EXAMPLE_SOURCE.relative_to(ROOT))]:
        raise RuntimeError(f"harness implementation boundary drifted: {implementation}")
    protected = [
        "Cargo.lock",
        ":(top,glob)**/Cargo.toml",
        ":(top,glob).cargo/**",
        ":(top,glob)crates/**",
        ":(top,glob)kernels/**",
    ]
    dense_drift = changed_paths(f"{DENSE_COMMIT}..HEAD", protected)
    if dense_drift != [str(EXAMPLE_SOURCE.relative_to(ROOT))]:
        raise RuntimeError(
            f"production tree differs from dense implementation: {dense_drift}"
        )
    build = validate_build(
        json.loads(command_text([str(QWEN_BENCH), "build-info", "--output", "json"])),
        commit,
    )
    qwen_bytes = QWEN.read_bytes()
    embedded = {
        "commit": commit,
        "build_source_state": str(build["build_source_state"]),
    }
    if any(value.encode() not in qwen_bytes for value in embedded.values()):
        raise RuntimeError("release qwen lacks exact embedded H identity")
    return commit, {
        **build,
        "qwen_bench_sha256": sha256(QWEN_BENCH),
        "qwen_identity": {
            "sha256": hashlib.sha256(qwen_bytes).hexdigest(),
            "embedded": embedded,
            "verified": True,
        },
        "example_sha256": sha256(EXAMPLE_BINARY),
    }


def file_identity(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {
        "device": stat.st_dev,
        "inode": stat.st_ino,
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def valid_file_identity(value: object) -> bool:
    return (
        isinstance(value, dict)
        and set(value) == {"device", "inode", "size_bytes", "mtime_ns"}
        and all(type(item) is int for item in value.values())
        and all(value[key] >= 0 for key in ("device", "inode", "size_bytes"))
    )


def required_paths(bridge: dict[Path, str]) -> tuple[Path, ...]:
    return tuple(
        dict.fromkeys(
            (
                RUNNER,
                PREREG,
                EXAMPLE_SOURCE,
                EXAMPLE_BINARY,
                QWEN,
                QWEN_BENCH,
                MODEL,
                Path(mechanics.__file__).resolve(),
                Path(mechanics.common.__file__).resolve(),
                *bridge,
            )
        )
    )


def build_manifest(removed: list[str], base_env: dict[str, str]) -> dict[str, object]:
    bridge = verify_v0627_bridge()
    commit, build = source_and_build_identity()
    paths = required_paths(bridge)
    if any(not path.is_file() for path in paths):
        raise RuntimeError("required manifest input is missing")
    hashes = {str(path): sha256(path) for path in paths}
    if (
        hashes[str(PREREG)] != EXPECTED_PREREG_SHA256
        or hashes[str(MODEL)] != EXPECTED_MODEL_SHA256
        or MODEL.stat().st_size != EXPECTED_MODEL_SIZE
    ):
        raise RuntimeError("preregistration or model identity drifted")
    device = command_text([str(QWEN), "--info"], env=base_env).strip()
    product = command_text(["sw_vers", "-productVersion"], env=base_env).strip()
    os_build = command_text(["sw_vers", "-buildVersion"], env=base_env).strip()
    memory = int(command_text(["sysctl", "-n", "hw.memsize"], env=base_env))
    if (
        device != EXPECTED_DEVICE
        or memory != EXPECTED_HW_MEMSIZE
        or not product
        or not os_build
    ):
        raise RuntimeError("host identity drifted")
    identity = file_identity(MODEL)
    page_size = os.sysconf("SC_PAGE_SIZE")
    pages = (identity["size_bytes"] + page_size - 1) // page_size
    if (
        identity["size_bytes"] != EXPECTED_MODEL_SIZE
        or page_size != EXPECTED_PAGE_SIZE
        or pages != EXPECTED_MODEL_PAGES
    ):
        raise RuntimeError("model size/page derivation drifted")
    return {
        "schema": 1,
        "protocol": "v0.628-dense27b-pread-cold-composition",
        "created_unix_ms": time.time_ns() // 1_000_000,
        "source_commit": commit,
        "base_commit": BASE_COMMIT,
        "dense_implementation_commit": DENSE_COMMIT,
        "build_identity": build,
        "device": device,
        "macos_product_version": product,
        "macos_build_version": os_build,
        "hw_memsize": memory,
        "removed_environment": removed,
        "child_environment": mechanics.child_environment_record(base_env),
        "time_resource_probe": mechanics.preflight_time_resources(base_env),
        "sha256": hashes,
        "model_file_identity": identity,
        "model_size_bytes": EXPECTED_MODEL_SIZE,
        "page_size": EXPECTED_PAGE_SIZE,
        "model_pages": EXPECTED_MODEL_PAGES,
        "prompt": PROMPT,
        "prompt_sha256": hashlib.sha256(PROMPT.encode()).hexdigest(),
        "prompt_tokens": 5,
        "output_tokens_after_first_byte": 0,
        "pair_orders": list(PAIR_ORDERS),
        "attempt_count": 4,
        "child_retry_count": 0,
        "physical_read_window_gib": [READ_MIN_GIB, READ_MAX_GIB],
        "physical_read_window_bytes": [READ_MIN_BYTES, READ_MAX_BYTES],
        "v0627_bridge": {
            "decision_sha256": EXPECTED_V0627_DECISION,
            "inventory_sha256": EXPECTED_V0627_INVENTORY,
            "completion_sha256": EXPECTED_V0627_COMPLETE,
            "inventory_members": len(bridge) - 2,
            "authority_imported": False,
        },
        "authority": "none",
        "force_authorized": False,
    }


def verify_non_model_identity(manifest: dict[str, object]) -> None:
    commit, build = source_and_build_identity()
    if commit != manifest.get("source_commit") or build != manifest.get(
        "build_identity"
    ):
        raise RuntimeError("source/build/runtime identity changed")
    expected = manifest.get("sha256")
    if not isinstance(expected, dict):
        raise RuntimeError("manifest hash map is malformed")
    actual = {name: sha256(Path(name)) for name in expected if name != str(MODEL)}
    wanted = {name: digest for name, digest in expected.items() if name != str(MODEL)}
    if actual != wanted:
        raise RuntimeError("non-model packet input changed")
    verify_v0627_bridge()


def cpu_test_command() -> list[str]:
    return [
        "cargo",
        "test",
        "--release",
        "-p",
        "qwen-llm",
        "--example",
        "first_byte_spike",
        CPU_TEST,
        "--",
        "--exact",
        "--nocapture",
        "--test-threads=1",
    ]


def recognize_cpu_test(text: str) -> None:
    escaped = re.escape(CPU_TEST)
    result = re.findall(rf"^test {escaped} \.\.\. ok\r?$", text, re.MULTILINE)
    summary = re.findall(
        r"^test result: ok\. 1 passed; 0 failed; 0 ignored; 0 measured; "
        r"\d+ filtered out; finished in [0-9]+(?:\.[0-9]+)?s\r?$",
        text,
        re.MULTILINE,
    )
    if len(result) != 1 or len(summary) != 1:
        raise RuntimeError("exact CPU-only example test was not uniquely passed")


def run_cpu_test(base_env: dict[str, str]) -> dict[str, object]:
    path = ARTIFACT / "cpu-default-policy.out"
    command = cpu_test_command()
    deferred_sigint: list[int] = []

    def defer_sigint(signum: int, _frame: object) -> None:
        deferred_sigint.append(signum)

    started = time.perf_counter()
    result = None
    execution_error = None
    durability_error = None
    prior = signal.signal(signal.SIGINT, defer_sigint)
    try:
        try:
            with path.open("xb") as output:
                try:
                    result = subprocess.run(
                        command,
                        cwd=ROOT,
                        env=base_env,
                        stdout=output,
                        stderr=subprocess.STDOUT,
                        check=False,
                    )
                except BaseException as error:
                    execution_error = error
                try:
                    output.flush()
                    os.fsync(output.fileno())
                except OSError as error:
                    durability_error = error
        except OSError as error:
            durability_error = error
        try:
            mechanics.fsync_directory(ARTIFACT)
        except OSError as error:
            durability_error = error
    finally:
        signal.signal(signal.SIGINT, prior)
    if durability_error is not None:
        raise UnsealedPacket(f"CPU test output durability failed: {durability_error}")
    if deferred_sigint:
        raise InconclusivePacket(
            "cpu-default-policy",
            CPU_TEST,
            [f"operator_sigint_deferred={len(deferred_sigint)}"],
        )
    if execution_error is not None or result is None:
        raise UnsealedPacket(f"CPU test execution did not return: {execution_error}")
    text = path.read_text(encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError("CPU-only default policy test returned nonzero")
    recognize_cpu_test(text)
    row = {
        "schema": 1,
        "test": CPU_TEST,
        "command": command,
        "returncode": result.returncode,
        "wall_ms": (time.perf_counter() - started) * 1e3,
        "output_sha256": sha256(path),
        "passed": True,
    }
    write_fsynced_json(ARTIFACT / "cpu-default-policy.json", row)
    return row


def arm_environment(arm: str) -> dict[str, str]:
    if arm not in ("A", "B"):
        raise ValueError(f"unknown arm: {arm}")
    return {"QWEN_GGUF_PARALLEL_COPY": "0" if arm == "A" else "pread"}


def child_command() -> list[str]:
    return [
        "/usr/bin/time",
        "-l",
        str(EXAMPLE_BINARY),
        str(MODEL),
        "--policy",
        "default",
        "--invalidate",
        "--prompt",
        PROMPT,
        "--tokens",
        "0",
        "--intent",
        "force-only",
    ]


def unique_match(pattern: str, text: str, label: str) -> re.Match[str]:
    matches = list(re.finditer(pattern, text, re.MULTILINE))
    if len(matches) != 1:
        raise RuntimeError(f"{label} occurrence count drifted: {len(matches)}")
    return matches[0]


def parse_stable_identity(value: str) -> dict[str, int]:
    match = re.fullmatch(r"dev:(\d+),ino:(\d+),size:(\d+),mtime_ns:(-?\d+)", value)
    if match is None:
        raise RuntimeError("stable target identity grammar drifted")
    raw = match.groups()
    device, inode, size, mtime = map(int, raw)
    if any(
        str(parsed) != encoded
        for parsed, encoded in zip((device, inode, size, mtime), raw, strict=True)
    ):
        raise RuntimeError("stable target identity is not canonical decimal")
    if (
        any(value > 2**64 - 1 for value in (device, inode, size))
        or mtime < -(2**127)
        or mtime > 2**127 - 1
    ):
        raise RuntimeError("stable target identity is outside frozen integer widths")
    return {
        "device": device,
        "inode": inode,
        "size_bytes": size,
        "mtime_ns": mtime,
    }


def stable_stdout_protocol(
    stdout: str, manifest: dict[str, object]
) -> dict[str, object]:
    lines = stdout.splitlines()
    positioned: list[tuple[int, str]] = []
    by_token: dict[str, str] = {}
    for token in STABLE_STDOUT_TOKENS:
        occurrences = stdout.count(token)
        containing = [line for line in lines if token in line]
        if occurrences != len(containing) or occurrences > 1:
            raise RuntimeError(f"stable stdout token occurrence drifted: {token}")
        if containing:
            line = containing[0]
            if not line.startswith(token):
                raise RuntimeError(f"stable stdout token is not a line prefix: {token}")
            positioned.append((lines.index(line), token))
            by_token[token] = line
    observed = [token for _, token in sorted(positioned)]
    expected_prefix = list(STABLE_STDOUT_TOKENS[: len(observed)])
    if observed != expected_prefix:
        raise RuntimeError("stable stdout protocol is not an exact emitted prefix")

    hashes = manifest.get("sha256")
    expected_source = (
        hashes.get(str(EXAMPLE_SOURCE)) if isinstance(hashes, dict) else None
    )
    if (
        not isinstance(expected_source, str)
        or re.fullmatch(r"[0-9a-f]{64}", expected_source) is None
    ):
        raise RuntimeError("manifest example source identity is malformed")
    if (
        "harness_source_sha256=" in by_token
        and by_token["harness_source_sha256="]
        != f"harness_source_sha256={expected_source}"
    ):
        raise RuntimeError("executed example source identity drifted")
    exact = {
        "policy_source=": "policy_source=default",
        "policy_configured=": (
            "policy_configured=cold-only threshold=0.9 workers=0 chunk_bytes=0"
        ),
        "policy_observed=": (
            "policy_observed=cold-only threshold=0.9 action=configured-policy "
            "suppressed=false"
        ),
        "prefetch_exact=": (
            "prefetch_exact=events:1,prefetched:1,skipped:0,bytes:16817244384"
        ),
    }
    for token, wanted in exact.items():
        if token in by_token and by_token[token] != wanted:
            raise RuntimeError(f"stable stdout field drifted: {token}")

    identities: dict[str, dict[str, int]] = {}
    for label in ("before_invalidate", "after_invalidate", "after_load"):
        token = f"target_identity_{label}="
        if token in by_token:
            value = parse_stable_identity(by_token[token].removeprefix(token))
            if value != manifest.get("model_file_identity"):
                raise RuntimeError(f"target identity drifted: {label}")
            identities[label] = value

    timing = None
    if "timing_exact=" in by_token:
        match = re.fullmatch(
            r"timing_exact=load_us:(\d+),first_byte_us:(\d+)",
            by_token["timing_exact="],
        )
        if match is None:
            raise RuntimeError("exact timing grammar drifted")
        raw = match.groups()
        values = tuple(map(int, raw))
        if any(
            str(value) != encoded for value, encoded in zip(values, raw, strict=True)
        ):
            raise RuntimeError("exact timing is not canonical decimal")
        timing = {"load_us": values[0], "first_byte_us": values[1]}
        if any(value <= 0 or value > 2**128 - 1 for value in timing.values()):
            raise RuntimeError("exact timing value is nonpositive")

    rusage = None
    if "rusage_exact=" in by_token:
        match = re.fullmatch(
            r"rusage_exact=pageins:(\d+),disk_read_bytes:(\d+),"
            r"disk_write_bytes:(\d+),rss_delta_bytes:(-?\d+),"
            r"footprint_delta_bytes:(-?\d+)",
            by_token["rusage_exact="],
        )
        if match is None:
            raise RuntimeError("exact rusage grammar drifted")
        raw = match.groups()
        values = list(map(int, raw))
        if any(
            str(value) != encoded for value, encoded in zip(values, raw, strict=True)
        ):
            raise RuntimeError("exact rusage is not canonical decimal")
        rusage = {
            "pageins": values[0],
            "disk_read_bytes": values[1],
            "disk_write_bytes": values[2],
            "rss_delta_bytes": values[3],
            "footprint_delta_bytes": values[4],
        }
        if any(value > 2**64 - 1 for value in values[:3]) or any(
            value < -(2**63) or value > 2**63 - 1 for value in values[3:]
        ):
            raise RuntimeError("exact rusage value is outside frozen integer widths")
    return {
        "status": (
            "complete"
            if len(observed) == len(STABLE_STDOUT_TOKENS)
            else "incomplete-valid-prefix"
        ),
        "observed_tokens": observed,
        "lines": by_token,
        "target_identities": identities,
        "timing": timing,
        "rusage": rusage,
    }


def parse_cold_stdout(stdout: str, manifest: dict[str, object]) -> dict[str, object]:
    stable = stable_stdout_protocol(stdout, manifest)
    if stable["status"] != "complete":
        raise RuntimeError("successful child lacks complete stable stdout protocol")
    model = unique_match(r"^model:\s+(.+)$", stdout, "model")
    prompt = unique_match(r"^prompt:\s+(.+)$", stdout, "prompt")
    intent = unique_match(r"^intent:\s+(.+)$", stdout, "intent")
    invalidate_flag = unique_match(
        r"^invalidate:\s+(true|false)$", stdout, "invalidate"
    )
    encoded = unique_match(
        r"^\s+prompt encoded to (\d+) tokens: (.+)$", stdout, "encoded prompt"
    )
    pre = unique_match(
        r"^pre-arm residency: (\d+)/(\d+) pages \(([0-9]+\.[0-9])%\)$",
        stdout,
        "pre-arm residency",
    )
    invalidation = unique_match(
        r"^invalidate: (\d+)/(\d+) -> (\d+)/(\d+)$", stdout, "invalidation"
    )
    load = unique_match(
        r"^load:\s+([0-9.]+) s\s+pageins=\s*(\d+)\s+diskR=\s*([0-9.]+) GiB$",
        stdout,
        "load",
    )
    first = unique_match(r"^FIRST BYTE:\s+([0-9.]+) s$", stdout, "first byte")
    total = unique_match(
        r"^rusage total: pageins=\s*(\d+)\s+diskR=\s*([0-9.]+) GiB\s+"
        r"diskW=\s*([0-9.]+) MiB\s+.+RSS=\s*([+-][0-9.]+) GiB$",
        stdout,
        "process rusage",
    )
    token = unique_match(r"^first token: id=(\d+) piece=(.*)$", stdout, "first token")
    post = unique_match(
        r"^post-arm residency: (\d+)/(\d+) pages \(([0-9]+\.[0-9])%\)$",
        stdout,
        "post-arm residency",
    )
    if (
        model.group(1) != str(MODEL)
        or prompt.group(1) != json.dumps(PROMPT)
        or intent.group(1) != "ForceOnly"
        or invalidate_flag.group(1) != "true"
        or encoded.group(1) != "5"
        or encoded.group(2) != PROMPT_IDS
    ):
        raise RuntimeError("short first-byte cell shape drifted")
    pre_resident, pre_total, pre_pct = pre.groups()
    before, total_before, after, total_after = map(int, invalidation.groups())
    post_resident, post_total, post_pct = post.groups()
    if (
        int(pre_resident) != EXPECTED_MODEL_PAGES
        or int(pre_total) != EXPECTED_MODEL_PAGES
        or float(pre_pct) != 100.0
        or (before, total_before, after, total_after)
        != (EXPECTED_MODEL_PAGES, EXPECTED_MODEL_PAGES, 0, EXPECTED_MODEL_PAGES)
        or int(post_total) != EXPECTED_MODEL_PAGES
        or int(post_resident) / int(post_total) < 0.99
        or float(post_pct) != round(int(post_resident) / int(post_total) * 100.0, 1)
    ):
        raise RuntimeError("cold residency/invalidation contract drifted")

    identity_values = stable["target_identities"]
    if (
        len(
            set(json.dumps(value, sort_keys=True) for value in identity_values.values())
        )
        != 1
    ):
        raise RuntimeError("target identity changed during child")
    prefetch_occurrences = stdout.count("prefetch:")
    prefetch_lines = [line for line in stdout.splitlines() if "prefetch:" in line]
    if prefetch_occurrences != 1 or len(prefetch_lines) != 1:
        raise RuntimeError("human prefetch occurrence count drifted")
    human_prefetch = unique_match(
        r"^\s+prefetch:\s+[0-9.]+ s\s+1 shards prefetched, 0 skipped, "
        r"15\.66 GiB returned$",
        stdout,
        "human prefetch summary",
    )
    del human_prefetch
    timing = stable["timing"]
    rusage = stable["rusage"]
    if not isinstance(timing, dict) or not isinstance(rusage, dict):
        raise RuntimeError("complete stable timing/rusage evidence is missing")
    disk_gib = rusage["disk_read_bytes"] / (1 << 30)
    if (
        abs(float(load.group(1)) * 1e6 - timing["load_us"]) > 750.0
        or abs(float(first.group(1)) * 1e6 - timing["first_byte_us"]) > 750.0
        or abs(float(total.group(2)) - disk_gib) > 0.006
        or int(total.group(1)) != rusage["pageins"]
        or abs(float(total.group(3)) - rusage["disk_write_bytes"] / (1 << 20)) > 0.051
        or abs(float(total.group(4)) - rusage["rss_delta_bytes"] / (1 << 30)) > 0.006
    ):
        raise RuntimeError("human and exact timing/rusage evidence disagree")
    return {
        "page_size": EXPECTED_PAGE_SIZE,
        "total_pages": EXPECTED_MODEL_PAGES,
        "pre_arm_resident_pages": int(pre_resident),
        "invalidated_before_pages": before,
        "invalidated_after_pages": after,
        "load_s": float(load.group(1)),
        "load_us": timing["load_us"],
        "load_pageins": int(load.group(2)),
        "load_disk_gib": float(load.group(3)),
        "first_byte_s": float(first.group(1)),
        "first_byte_us": timing["first_byte_us"],
        "total_pageins": int(total.group(1)),
        "total_disk_gib": disk_gib,
        "total_disk_read_bytes": rusage["disk_read_bytes"],
        "total_disk_write_mib": float(total.group(3)),
        "total_rss_delta_gib": float(total.group(4)),
        "first_token": f"id={token.group(1)} piece={token.group(2)}",
        "post_resident_pages": int(post_resident),
        "post_resident_fraction": int(post_resident) / int(post_total),
        "target_identities": identity_values,
        "prefetch_bytes": EXPECTED_MODEL_SIZE,
        "stable_protocol": stable,
    }


def parse_dense_marker(line: str) -> dict[str, int]:
    if not line.startswith(MARKER_PREFIX):
        raise RuntimeError("dense schema-2 pread marker prefix drifted")
    dynamic_names = (
        "allocation_us",
        "source_us",
        "copy_us",
        "binding_us",
        "ready_us",
        "user_cpu_us",
        "system_cpu_us",
        "total_cpu_us",
        "timer_minor_faults",
        "timer_major_faults",
        "instructions_delta_raw",
        "cycles_delta_raw",
    )
    suffix = " ".join(rf"{name}=(\d+)" for name in dynamic_names)
    match = re.fullmatch(re.escape(MARKER_PREFIX) + suffix, line)
    if match is None:
        raise RuntimeError("dense schema-2 marker field order/grammar drifted")
    parsed = {}
    for name, raw in zip(dynamic_names, match.groups(), strict=True):
        value = int(raw)
        if str(value) != raw or value > 2**64 - 1:
            raise RuntimeError(f"dense marker {name} is not canonical u64")
        parsed[name] = value
    if parsed["timer_major_faults"] != 0:
        raise RuntimeError("dense marker major faults are nonzero")
    phases = ("allocation_us", "source_us", "copy_us", "binding_us")
    if (
        parsed["ready_us"] == 0
        or abs(parsed["ready_us"] - sum(parsed[name] for name in phases)) > 4
    ):
        raise RuntimeError("dense marker phase accounting does not reconcile")
    if parsed["total_cpu_us"] != parsed["user_cpu_us"] + parsed["system_cpu_us"]:
        raise RuntimeError("dense marker CPU accounting does not reconcile")
    return parsed


def reject_runtime_prefetch_diagnostics(plain: str) -> None:
    occurrences = plain.count("prefetch:")
    containing = [line for line in plain.splitlines() if "prefetch:" in line]
    if occurrences != len(containing):
        raise RuntimeError("runtime prefetch token is not line-aligned")
    if containing:
        raise RuntimeError("unexpected runtime prefetch diagnostic appeared")


def exact_load_token_lines(plain: str, arm: str) -> list[str]:
    expected_marker_count = 0 if arm == "A" else 1
    counts = {
        "policy": plain.count("[metal-load] native quantized token embedding policy:"),
        "ledger": plain.count("[metal-load-ledger]"),
        "all_markers": plain.count("[metal-gguf-"),
        "pread_marker": plain.count("[metal-gguf-parallel-pread]"),
    }
    if counts != {
        "policy": 1,
        "ledger": 1,
        "all_markers": expected_marker_count,
        "pread_marker": expected_marker_count,
    }:
        raise RuntimeError(f"{arm} load-token occurrence counts drifted: {counts}")
    prefixes = (
        "[metal-load] native quantized token embedding policy:",
        "[metal-gguf-",
        "[metal-load-ledger]",
    )
    containing = [
        line for line in plain.splitlines() if any(p in line for p in prefixes)
    ]
    recognized = [line for line in containing if line.startswith(prefixes)]
    if containing != recognized:
        raise RuntimeError(f"{arm} load token is not an exact line prefix")
    return recognized


def parse_load_contract(stderr: str, arm: str) -> dict[str, object]:
    plain = ANSI_RE.sub("", stderr)
    forbidden = (
        "suppressed",
        "prefetch: skipped (already warm)",
        "prefetch: residency probe failed",
        "prefetch: shard failed",
        "[runtime-prefetch]",
        "[metal-gguf-parallel-copied]",
        "[metal-gguf-owned]",
        "[metal-gguf-retained]",
        "[metal-gguf-no-copy]",
    )
    if any(value in plain for value in forbidden):
        raise RuntimeError("forbidden prefetch/storage evidence appeared")
    reject_runtime_prefetch_diagnostics(plain)
    recognized = exact_load_token_lines(plain, arm)
    markers = [line for line in recognized if line.startswith("[metal-gguf-")]
    expected_markers = 0 if arm == "A" else 1
    if len(markers) != expected_markers:
        raise RuntimeError(f"{arm} storage marker count drifted")
    marker = parse_dense_marker(markers[0]) if markers else None
    expected = [POLICY_LINE, LEDGER_LINE]
    if arm == "B":
        expected = [POLICY_LINE, None, LEDGER_LINE]
    if len(recognized) != len(expected):
        raise RuntimeError(f"{arm} load-line count drifted")
    for observed, wanted in zip(recognized, expected, strict=True):
        if wanted is not None and observed != wanted:
            raise RuntimeError(f"{arm} native policy/ledger order drifted")
    return {
        "storage": "copied" if arm == "A" else "parallel-pread",
        "marker": markers[0] if markers else None,
        "marker_fields": marker,
        "recognized_lines": recognized,
    }


def parse_observed_contract(stderr: str, arm: str) -> dict[str, object] | None:
    plain = ANSI_RE.sub("", stderr)
    prefixes = (
        "[metal-load] native quantized token embedding policy:",
        "[metal-gguf-",
        "[metal-load-ledger]",
    )
    forbidden = (
        "suppressed",
        "prefetch: skipped (already warm)",
        "prefetch: residency probe failed",
        "prefetch: shard failed",
        "[runtime-prefetch]",
        "[metal-gguf-parallel-copied]",
        "[metal-gguf-owned]",
        "[metal-gguf-retained]",
        "[metal-gguf-no-copy]",
    )
    if any(value in plain for value in forbidden):
        raise RuntimeError(
            "failed child emitted contradictory prefetch/storage evidence"
        )
    reject_runtime_prefetch_diagnostics(plain)
    occurrences = {
        prefix: plain.count(prefix)
        for prefix in (
            "[metal-load] native quantized token embedding policy:",
            "[metal-load-ledger]",
            "[metal-gguf-",
            "[metal-gguf-parallel-pread]",
        )
    }
    maximum = {
        "[metal-load] native quantized token embedding policy:": 1,
        "[metal-load-ledger]": 1,
        "[metal-gguf-": 0 if arm == "A" else 1,
        "[metal-gguf-parallel-pread]": 0 if arm == "A" else 1,
    }
    if any(occurrences[token] > limit for token, limit in maximum.items()):
        raise RuntimeError("failed child load-token occurrence count drifted")
    containing = [
        line for line in plain.splitlines() if any(p in line for p in prefixes)
    ]
    recognized = [line for line in containing if line.startswith(prefixes)]
    if containing != recognized:
        raise RuntimeError("failed child load token is not an exact line prefix")
    if not recognized:
        return None
    expected: list[str | None] = [POLICY_LINE, LEDGER_LINE]
    if arm == "B":
        expected = [POLICY_LINE, None, LEDGER_LINE]
    if len(recognized) > len(expected):
        raise RuntimeError("failed child load contract has extra lines")
    for observed, wanted in zip(recognized, expected, strict=False):
        if wanted is None:
            parse_dense_marker(observed)
        elif observed != wanted:
            raise RuntimeError("failed child load contract contradicts prefix")
    if len(recognized) == len(expected):
        return {"status": "complete", "contract": parse_load_contract(stderr, arm)}
    return {
        "status": "incomplete-valid-prefix",
        "recognized_lines": recognized,
    }


def fsync_raw(paths: tuple[Path, ...]) -> None:
    try:
        for path in paths:
            if path.exists():
                with path.open("rb+") as artifact:
                    artifact.flush()
                    os.fsync(artifact.fileno())
        mechanics.fsync_directory(ARTIFACT)
    except OSError as error:
        raise UnsealedPacket(f"raw artifact fsync failed: {error}") from error


def record_launch(
    stem: str,
    command: list[str],
    arm: str,
    pair_index: int,
    order: str,
    position: int,
    manifest: dict[str, object],
) -> None:
    append_fsynced(
        ARTIFACT / "launch-seal.jsonl",
        {
            "event": "launch",
            "unix_ms": time.time_ns() // 1_000_000,
            "stage": "cold-composition",
            "artifact_stem": stem,
            "command": command,
            "arm": arm,
            "arm_environment": arm_environment(arm),
            "normalized_base_environment": manifest["child_environment"],
            "pair_index": pair_index,
            "pair_order": order,
            "position": position,
            "source_commit": manifest["source_commit"],
            "build_identity": manifest["build_identity"],
        },
    )


def record_completion(stem: str, returncode: int | None, error: str | None) -> None:
    append_fsynced(
        ARTIFACT / "launch-seal.jsonl",
        {
            "event": "completion",
            "unix_ms": time.time_ns() // 1_000_000,
            "stage": "cold-composition",
            "artifact_stem": stem,
            "returncode": returncode,
            "error": error,
        },
    )


def process_validity(
    stderr: str, post: dict[str, object]
) -> tuple[dict[str, object], list[str]]:
    reasons = list(post["child_interval"]["failure_reasons"])
    if post["host_after_exit"].get("valid") is not True:
        reasons.append("post_exit_host_invalid")
    try:
        resources = mechanics.process_resources(stderr)
    except Exception as error:
        resources = None
        reasons.append(f"process_resources_invalid={type(error).__name__}:{error}")
    if resources is not None:
        if resources.get("page_faults") != 0:
            reasons.append("child_major_faults_nonzero")
        if resources.get("swaps") != 0:
            reasons.append("child_swaps_nonzero")
    return {**post, "process_resources": resources}, reasons


def run_child(
    arm: str,
    pair_index: int,
    order: str,
    position: int,
    base_env: dict[str, str],
    manifest: dict[str, object],
) -> dict[str, object]:
    stem = f"cold-p{pair_index:02d}-{order.lower()}-r{position}-{arm.lower()}"
    stdout_path = ARTIFACT / f"{stem}.out"
    stderr_path = ARTIFACT / f"{stem}.err"
    try:
        conditioning = mechanics.condition_for_child(stem, manifest)
    except mechanics.InconclusivePacket as error:
        raise InconclusivePacket(error.stage, error.child, error.reasons) from error
    conditioning_record = {
        "schema": 1,
        "artifact_stem": stem,
        "evidence": conditioning,
    }
    write_fsynced_json(ARTIFACT / f"{stem}.conditioning.json", conditioning_record)
    verify_non_model_identity(manifest)
    if file_identity(MODEL) != manifest.get("model_file_identity"):
        raise RuntimeError("model identity drifted before launch")
    command = child_command()
    env = base_env.copy()
    env.update(arm_environment(arm))
    process = None
    deferred: list[str] = []
    deferred_sigint: list[int] = []

    def defer_sigint(signum: int, _frame: object) -> None:
        deferred_sigint.append(signum)

    started = time.perf_counter()
    prior = signal.signal(signal.SIGINT, defer_sigint)
    try:
        record_launch(stem, command, arm, pair_index, order, position, manifest)
        try:
            with (
                stdout_path.open("xb") as stdout_file,
                stderr_path.open("xb") as stderr_file,
            ):
                process = subprocess.Popen(
                    command,
                    cwd=ROOT,
                    env=env,
                    stdout=stdout_file,
                    stderr=stderr_file,
                )
                returncode, deferred = mechanics.wait_for_child(process)
            wall_ms = (time.perf_counter() - started) * 1e3
            fsync_raw((stdout_path, stderr_path))
        except (OSError, KeyboardInterrupt) as error:
            if process is None:
                fsync_raw((stdout_path, stderr_path))
                record_completion(stem, None, f"{type(error).__name__}:{error}")
                if isinstance(error, OSError):
                    mechanics.record_spawn_failure_evidence(
                        "cold-composition", stem, conditioning, error
                    )
                raise InconclusivePacket(
                    "cold-composition",
                    stem,
                    [f"spawn_failed={type(error).__name__}:{error}"],
                ) from error
            returncode, more = mechanics.wait_for_child(process)
            deferred.extend(more)
            deferred.append(f"{type(error).__name__}:{error}")
            wall_ms = (time.perf_counter() - started) * 1e3
            fsync_raw((stdout_path, stderr_path))
        if deferred_sigint:
            deferred.append(f"SIGINTx{len(deferred_sigint)}")
        included_sigints = len(deferred_sigint)
        record_completion(stem, returncode, ";".join(deferred) if deferred else None)
        post = mechanics.capture_post_exit_state(conditioning)
        write_fsynced_json(
            ARTIFACT / f"{stem}.post-exit-state.json",
            {
                "schema": 1,
                "artifact_stem": stem,
                "returncode": returncode,
                **post,
            },
        )
        stdout_bytes = stdout_path.read_bytes()
        stderr_bytes = stderr_path.read_bytes()
        try:
            stdout = stdout_bytes.decode("utf-8")
            stderr = stderr_bytes.decode("utf-8")
            decode_error = None
        except UnicodeDecodeError as error:
            stdout = stderr = ""
            decode_error = f"raw_utf8_invalid={error}"
        validity, reasons = process_validity(stderr, post)
        if deferred:
            reasons.append(f"child_wait_interrupted={';'.join(deferred)}")
        if returncode != 0:
            reasons.append(f"child_nonzero_exit={returncode}")
        if decode_error is not None:
            reasons.append(decode_error)
        post_record = {
            "schema": 1,
            "artifact_stem": stem,
            "returncode": returncode,
            "evidence": validity,
            "reasons": reasons,
        }
        write_fsynced_json(ARTIFACT / f"{stem}.post-exit.json", post_record)
    finally:
        signal.signal(signal.SIGINT, prior)
    if len(deferred_sigint) != included_sigints:
        raise InconclusivePacket(
            "cold-composition",
            stem,
            [f"operator_sigint_after_completion={len(deferred_sigint)}"],
        )
    if decode_error is not None:
        raise RuntimeError(decode_error)
    if returncode == 0:
        cold = parse_cold_stdout(stdout, manifest)
        load_contract = parse_load_contract(stderr, arm)
        observed_stdout = None
    else:
        cold = None
        load_contract = parse_observed_contract(stderr, arm)
        observed_stdout = stable_stdout_protocol(stdout, manifest)
    return {
        "stage": "cold-composition",
        "artifact_stem": stem,
        "arm": arm,
        "pair_index": pair_index,
        "pair_order": order,
        "position": position,
        "command": command,
        "arm_environment": arm_environment(arm),
        "process_wall_ms": wall_ms,
        "returncode": returncode,
        "stdout_sha256": hashlib.sha256(stdout_bytes).hexdigest(),
        "stderr_sha256": hashlib.sha256(stderr_bytes).hexdigest(),
        "conditioning": conditioning,
        **validity,
        "cold": cold,
        "load_contract": load_contract,
        "observed_stdout": observed_stdout,
        "valid": not reasons,
        "validity_reasons": reasons,
    }


def run_stage(
    base_env: dict[str, str], manifest: dict[str, object]
) -> list[dict[str, object]]:
    attempts = ARTIFACT / "attempts.jsonl"
    rows = []
    for pair_index, order in enumerate(PAIR_ORDERS, 1):
        for position, arm in enumerate(order, 1):
            row = run_child(arm, pair_index, order, position, base_env, manifest)
            append_fsynced(attempts, row)
            rows.append(row)
            if not row["valid"]:
                raise InconclusivePacket(
                    "cold-composition",
                    str(row["artifact_stem"]),
                    list(row["validity_reasons"]),
                )
    if len(rows) != 4:
        raise RuntimeError("cold composition did not complete four children")
    return rows


def ratio(numerator: float, denominator: float, label: str) -> float:
    if not all(
        math.isfinite(value) and value > 0 for value in (numerator, denominator)
    ):
        raise RuntimeError(f"{label} ratio inputs are invalid")
    return numerator / denominator


def analyze(rows: list[dict[str, object]]) -> dict[str, object]:
    if len(rows) != 4:
        raise RuntimeError("analysis requires exactly four children")
    tokens = {row["cold"]["first_token"] for row in rows}
    if len(tokens) != 1:
        raise RuntimeError("first token ID/piece does not agree across all children")
    pairs = []
    for pair_index, order in enumerate(PAIR_ORDERS, 1):
        pair = [row for row in rows if row["pair_index"] == pair_index]
        if [row["arm"] for row in pair] != list(order):
            raise RuntimeError(f"pair {pair_index} order drifted")
        by_arm = {row["arm"]: row for row in pair}
        a, b = by_arm["A"], by_arm["B"]
        a_resources = a.get("process_resources")
        b_resources = b.get("process_resources")
        if not isinstance(a_resources, dict) or not isinstance(b_resources, dict):
            raise RuntimeError("pair process resources are malformed")
        load_delta_us = b["cold"]["load_us"] - a["cold"]["load_us"]
        first_delta_us = b["cold"]["first_byte_us"] - a["cold"]["first_byte_us"]
        load_delta_ms = load_delta_us / 1e3
        first_delta_ms = first_delta_us / 1e3
        a_read_bytes = a["cold"]["total_disk_read_bytes"]
        b_read_bytes = b["cold"]["total_disk_read_bytes"]
        read_ratio = b_read_bytes / a_read_bytes if a_read_bytes > 0 else None
        cpu_ratio = ratio(
            b_resources["total_cpu_ms"], a_resources["total_cpu_ms"], "CPU"
        )
        footprint_ratio = ratio(
            b_resources["peak_memory_footprint"],
            a_resources["peak_memory_footprint"],
            "footprint",
        )
        gates = {
            "load_b_minus_a_nonpositive": load_delta_ms <= 0.0,
            "first_byte_b_minus_a_nonpositive": first_delta_ms <= 0.0,
            "physical_read_b_over_a": (
                a_read_bytes > 0 and 10 * b_read_bytes <= 11 * a_read_bytes
            ),
            "complete_cpu_b_over_a": cpu_ratio <= 1.10,
            "peak_footprint_b_over_a": footprint_ratio <= 1.05,
            "a_absolute_read_window": READ_MIN_BYTES <= a_read_bytes <= READ_MAX_BYTES,
            "b_absolute_read_window": READ_MIN_BYTES <= b_read_bytes <= READ_MAX_BYTES,
        }
        pairs.append(
            {
                "pair_index": pair_index,
                "pair_order": order,
                "load_b_minus_a_us": load_delta_us,
                "load_b_minus_a_ms": load_delta_ms,
                "first_byte_b_minus_a_us": first_delta_us,
                "first_byte_b_minus_a_ms": first_delta_ms,
                "physical_read_b_over_a": read_ratio,
                "complete_cpu_b_over_a": cpu_ratio,
                "peak_footprint_b_over_a": footprint_ratio,
                "rss_diagnostic": {
                    "a_delta_gib": a["cold"]["total_rss_delta_gib"],
                    "b_delta_gib": b["cold"]["total_rss_delta_gib"],
                },
                "exit_diagnostic": {
                    "a_process_wall_ms": a["process_wall_ms"],
                    "b_process_wall_ms": b["process_wall_ms"],
                },
                "gates": gates,
                "passes": all(gates.values()),
            }
        )
    return {
        "schema": 1,
        "stage": "cold-composition",
        "pairs": pairs,
        "first_token_agreement": tokens.pop(),
        "physical_read_attribution": (
            "process-wide proc_pid_rusage; bounded jointly by exact target identity, "
            "in-child invalidation, residency, exact prefetch bytes, and markers"
        ),
        "effect_size_or_mde_claim": False,
        "passes": all(pair["passes"] for pair in pairs),
    }


def expected_children() -> list[dict[str, object]]:
    return [
        {
            "artifact_stem": (
                f"cold-p{index:02d}-{order.lower()}-r{position}-{arm.lower()}"
            ),
            "arm": arm,
            "pair_index": index,
            "pair_order": order,
            "position": position,
        }
        for index, order in enumerate(PAIR_ORDERS, 1)
        for position, arm in enumerate(order, 1)
    ]


def expected_stems() -> list[str]:
    return [str(child["artifact_stem"]) for child in expected_children()]


def read_attempts() -> list[dict[str, object]]:
    path = ARTIFACT / "attempts.jsonl"
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = json.loads(line, parse_constant=mechanics.reject_json_constant)
        if not isinstance(value, dict):
            raise RuntimeError("attempt row is malformed")
        rows.append(value)
    return rows


def record_final_identity(manifest: dict[str, object]) -> dict[str, object]:
    path = ARTIFACT / "final-model-sha256.json"
    non_model_error = None
    try:
        verify_non_model_identity(manifest)
    except Exception as error:
        non_model_error = f"{type(error).__name__}: {error}"
    before = after = actual = None
    model_error = None
    try:
        before = file_identity(MODEL)
        actual = sha256(MODEL)
        after = file_identity(MODEL)
    except Exception as error:
        model_error = f"{type(error).__name__}: {error}"
    expected_hash = manifest["sha256"][str(MODEL)]
    expected_identity = manifest["model_file_identity"]
    matches = (
        non_model_error is None
        and model_error is None
        and actual == expected_hash
        and before == expected_identity
        and after == expected_identity
    )
    payload = {
        "schema": 1,
        "model": str(MODEL),
        "expected_size_bytes": EXPECTED_MODEL_SIZE,
        "expected_sha256": expected_hash,
        "actual_sha256": actual,
        "expected_file_identity": expected_identity,
        "observed_file_identity_before": before,
        "observed_file_identity_after": after,
        "non_model_identity_error": non_model_error,
        "model_identity_error": model_error,
        "matches": matches,
    }
    if path.exists():
        if path.is_symlink() or not strict_equal(json_value(path), payload):
            raise UnsealedPacket(
                "existing final identity report is not exactly reusable"
            )
    else:
        write_fsynced_json(path, payload)
    return payload


def seal_file(path: Path, sealed: dict[Path, str]) -> None:
    if not path.is_file() or path.is_symlink() or path in sealed:
        raise RuntimeError(
            f"required artifact is missing/duplicate/unsafe: {path.name}"
        )
    sealed[path] = sha256(path)


def json_lines(path: Path) -> list[dict[str, object]]:
    values = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = json.loads(line, parse_constant=mechanics.reject_json_constant)
        if not isinstance(value, dict):
            raise RuntimeError(f"JSONL member is not an object: {path.name}")
        values.append(value)
    return values


def verify_attempt_row(
    row: dict[str, object],
    expected: dict[str, object],
    launch: dict[str, object],
    completion: dict[str, object],
    manifest: dict[str, object],
    sealed: dict[Path, str],
) -> None:
    stem = str(expected["artifact_stem"])
    exact = {
        "stage": "cold-composition",
        "artifact_stem": stem,
        "arm": expected["arm"],
        "pair_index": expected["pair_index"],
        "pair_order": expected["pair_order"],
        "position": expected["position"],
        "command": child_command(),
        "arm_environment": arm_environment(str(expected["arm"])),
    }
    if any(not strict_equal(row.get(key), value) for key, value in exact.items()):
        raise RuntimeError(f"attempt metadata drifted: {stem}")
    if not strict_equal(completion.get("returncode"), row.get("returncode")):
        raise RuntimeError(f"completion return code drifted: {stem}")

    conditioning_path = ARTIFACT / f"{stem}.conditioning.json"
    stdout_path = ARTIFACT / f"{stem}.out"
    stderr_path = ARTIFACT / f"{stem}.err"
    post_state_path = ARTIFACT / f"{stem}.post-exit-state.json"
    post_path = ARTIFACT / f"{stem}.post-exit.json"
    for path in (
        conditioning_path,
        stdout_path,
        stderr_path,
        post_state_path,
        post_path,
    ):
        if path not in sealed:
            raise RuntimeError(f"completed attempt artifact is unsealed: {path.name}")
    conditioning = json_value(conditioning_path)
    if not strict_equal(
        conditioning,
        {
            "schema": 1,
            "artifact_stem": stem,
            "evidence": row.get("conditioning"),
        },
    ):
        raise RuntimeError(f"conditioning/attempt binding drifted: {stem}")
    if (
        row.get("stdout_sha256") != sealed[stdout_path]
        or row.get("stderr_sha256") != sealed[stderr_path]
    ):
        raise RuntimeError(f"raw attempt binding drifted: {stem}")

    post_state = json_value(post_state_path)
    post = json_value(post_path)
    post_keys = ("host_after_exit", "vm_after_exit", "child_interval")
    if (
        not strict_equal(post_state.get("schema"), 1)
        or post_state.get("artifact_stem") != stem
        or not strict_equal(post_state.get("returncode"), row.get("returncode"))
        or any(not strict_equal(post_state.get(key), row.get(key)) for key in post_keys)
    ):
        raise RuntimeError(f"post-exit-state/attempt binding drifted: {stem}")
    evidence = post.get("evidence")
    if (
        not strict_equal(post.get("schema"), 1)
        or post.get("artifact_stem") != stem
        or not strict_equal(post.get("returncode"), row.get("returncode"))
        or not isinstance(evidence, dict)
        or any(not strict_equal(evidence.get(key), row.get(key)) for key in post_keys)
        or not strict_equal(
            evidence.get("process_resources"), row.get("process_resources")
        )
        or not strict_equal(post.get("reasons"), row.get("validity_reasons"))
        or row.get("valid") is not (not row.get("validity_reasons"))
    ):
        raise RuntimeError(f"post-exit/attempt binding drifted: {stem}")

    stdout = stdout_path.read_text(encoding="utf-8")
    stderr = stderr_path.read_text(encoding="utf-8")
    stored_post = {key: post_state[key] for key in post_keys}
    recomputed_validity, recomputed_reasons = process_validity(stderr, stored_post)
    completion_error = completion.get("error")
    if isinstance(completion_error, str):
        recomputed_reasons.append(f"child_wait_interrupted={completion_error}")
    if completion.get("returncode") != 0:
        recomputed_reasons.append(f"child_nonzero_exit={completion.get('returncode')}")
    if (
        not strict_equal(recomputed_validity, evidence)
        or any(
            not strict_equal(recomputed_validity.get(key), row.get(key))
            for key in (*post_keys, "process_resources")
        )
        or not strict_equal(recomputed_reasons, row.get("validity_reasons"))
        or not strict_equal(recomputed_reasons, post.get("reasons"))
    ):
        raise RuntimeError(f"attempt validity recomputation drifted: {stem}")
    arm = str(expected["arm"])
    if row.get("returncode") == 0:
        if (
            not strict_equal(row.get("cold"), parse_cold_stdout(stdout, manifest))
            or not strict_equal(
                row.get("load_contract"), parse_load_contract(stderr, arm)
            )
            or row.get("observed_stdout") is not None
        ):
            raise RuntimeError(f"successful attempt parse binding drifted: {stem}")
    else:
        if (
            row.get("cold") is not None
            or not strict_equal(
                row.get("load_contract"), parse_observed_contract(stderr, arm)
            )
            or not strict_equal(
                row.get("observed_stdout"), stable_stdout_protocol(stdout, manifest)
            )
        ):
            raise RuntimeError(f"failed attempt parse binding drifted: {stem}")
    process_wall = row.get("process_wall_ms")
    if (
        isinstance(process_wall, bool)
        or not isinstance(process_wall, (int, float))
        or not math.isfinite(float(process_wall))
        or float(process_wall) <= 0
    ):
        raise RuntimeError(f"attempt process wall is invalid: {stem}")


def verify_attempt_artifacts(
    decision: dict[str, object],
    manifest: dict[str, object],
    expected_final_identity: dict[str, object],
) -> dict[Path, str]:
    sealed: dict[Path, str] = {}
    manifest_path = ARTIFACT / "manifest.json"
    seal_file(manifest_path, sealed)
    if not strict_equal(json_value(manifest_path), manifest):
        raise RuntimeError("reserved manifest differs from live packet manifest")

    status = decision.get("status")
    if status not in VALID_STATUSES:
        raise RuntimeError("decision status is outside frozen set")
    expected_successor = PASS_SUCCESSOR if status == "cold_composition_pass" else "none"
    if (
        not strict_equal(decision.get("schema"), 1)
        or decision.get("source_commit") != manifest.get("source_commit")
        or not strict_equal(
            decision.get("imported_v0627"), manifest.get("v0627_bridge")
        )
        or decision.get("authority") != "none"
        or decision.get("force_authorized") is not False
        or decision.get("successor_authorization") != expected_successor
    ):
        raise RuntimeError("decision identity/authority boundary drifted")

    cpu = decision.get("cpu_test")
    output = ARTIFACT / "cpu-default-policy.out"
    metadata = ARTIFACT / "cpu-default-policy.json"
    if isinstance(cpu, dict):
        if (
            not strict_equal(json_value(metadata), cpu)
            or cpu.get("test") != CPU_TEST
            or not strict_equal(cpu.get("command"), cpu_test_command())
            or not strict_equal(cpu.get("returncode"), 0)
            or cpu.get("passed") is not True
            or cpu.get("output_sha256") != sha256(output)
        ):
            raise RuntimeError("CPU test metadata binding drifted")
        recognize_cpu_test(output.read_text(encoding="utf-8"))
        seal_file(output, sealed)
        seal_file(metadata, sealed)
    elif output.is_file():
        seal_file(output, sealed)
    if metadata.is_file() and metadata not in sealed:
        raise RuntimeError("orphan CPU test metadata exists")

    attempts = read_attempts()
    expected = expected_children()
    stems = [row.get("artifact_stem") for row in attempts]
    if stems != expected_stems()[: len(stems)] or len(set(stems)) != len(stems):
        raise RuntimeError("attempts are not the frozen unique prefix")
    attempts_path = ARTIFACT / "attempts.jsonl"
    if attempts_path.is_file():
        seal_file(attempts_path, sealed)

    launch_path = ARTIFACT / "launch-seal.jsonl"
    events: list[dict[str, object]] = []
    if launch_path.is_file():
        seal_file(launch_path, sealed)
        events = json_lines(launch_path)
    if len(events) % 2 or len(events) // 2 < len(attempts):
        raise RuntimeError("launch/completion evidence is incomplete")
    launch_count = len(events) // 2
    if launch_count > len(expected) or launch_count - len(attempts) > 1:
        raise RuntimeError("launch/attempt prefix cardinality drifted")
    if launch_count and not isinstance(cpu, dict):
        raise RuntimeError("model child launched without passed CPU contract test")

    launches: list[tuple[dict[str, object], dict[str, object]]] = []
    for index in range(launch_count):
        launch, completion = events[index * 2 : index * 2 + 2]
        child = expected[index]
        stem = str(child["artifact_stem"])
        launch_exact = {
            "event": "launch",
            "stage": "cold-composition",
            "artifact_stem": stem,
            "command": child_command(),
            "arm": child["arm"],
            "arm_environment": arm_environment(str(child["arm"])),
            "normalized_base_environment": manifest["child_environment"],
            "pair_index": child["pair_index"],
            "pair_order": child["pair_order"],
            "position": child["position"],
            "source_commit": manifest["source_commit"],
            "build_identity": manifest["build_identity"],
        }
        if (
            any(
                not strict_equal(launch.get(key), value)
                for key, value in launch_exact.items()
            )
            or type(launch.get("unix_ms")) is not int
            or launch["unix_ms"] <= 0
            or completion.get("event") != "completion"
            or completion.get("stage") != "cold-composition"
            or completion.get("artifact_stem") != stem
            or type(completion.get("unix_ms")) is not int
            or completion["unix_ms"] < launch["unix_ms"]
            or not (
                completion.get("returncode") is None
                or type(completion.get("returncode")) is int
            )
            or not (
                completion.get("error") is None
                or isinstance(completion.get("error"), str)
            )
        ):
            raise RuntimeError(f"launch/completion exact prefix drifted: {stem}")
        launches.append((launch, completion))

        conditioning_path = ARTIFACT / f"{stem}.conditioning.json"
        seal_file(conditioning_path, sealed)
        conditioning = json_value(conditioning_path)
        if (
            not strict_equal(conditioning.get("schema"), 1)
            or conditioning.get("artifact_stem") != stem
            or not isinstance(conditioning.get("evidence"), dict)
        ):
            raise RuntimeError(f"conditioning record is malformed: {stem}")
        if completion.get("returncode") is not None:
            for suffix in ("out", "err", "post-exit-state.json", "post-exit.json"):
                seal_file(ARTIFACT / f"{stem}.{suffix}", sealed)
            post_state = json_value(ARTIFACT / f"{stem}.post-exit-state.json")
            post = json_value(ARTIFACT / f"{stem}.post-exit.json")
            evidence = post.get("evidence")
            post_keys = ("host_after_exit", "vm_after_exit", "child_interval")
            if (
                not strict_equal(post_state.get("schema"), 1)
                or post_state.get("artifact_stem") != stem
                or not strict_equal(
                    post_state.get("returncode"), completion.get("returncode")
                )
                or not strict_equal(post.get("schema"), 1)
                or post.get("artifact_stem") != stem
                or not strict_equal(
                    post.get("returncode"), completion.get("returncode")
                )
                or not isinstance(evidence, dict)
                or any(
                    not strict_equal(evidence.get(key), post_state.get(key))
                    for key in post_keys
                )
                or not isinstance(post.get("reasons"), list)
            ):
                raise RuntimeError(
                    f"completed launch post-exit binding drifted: {stem}"
                )
        else:
            for suffix in ("out", "err"):
                path = ARTIFACT / f"{stem}.{suffix}"
                if path.is_file():
                    seal_file(path, sealed)
            spawn = ARTIFACT / f"{stem}.spawn-failure.json"
            seal_file(spawn, sealed)
            spawn_value = json_value(spawn)
            if (
                spawn_value.get("stage") != "cold-composition"
                or spawn_value.get("artifact_stem") != stem
            ):
                raise RuntimeError(f"spawn failure binding drifted: {stem}")

    for index, row in enumerate(attempts):
        launch, completion = launches[index]
        verify_attempt_row(row, expected[index], launch, completion, manifest, sealed)

    failed = decision.get("failed_child")
    if isinstance(failed, str):
        for suffix in ("prelaunch-failure.json", "spawn-failure.json"):
            path = ARTIFACT / f"{failed}.{suffix}"
            if path.is_file() and path not in sealed:
                seal_file(path, sealed)
        if decision.get("stopped_after") == "prelaunch":
            prelaunch = ARTIFACT / f"{failed}.prelaunch-failure.json"
            if prelaunch not in sealed:
                raise RuntimeError(
                    "prelaunch inconclusive lacks durable failure evidence"
                )
            value = json_value(prelaunch)
            if (
                value.get("artifact_stem") != failed
                or not strict_equal(
                    value.get("failure_reasons"), decision.get("reasons")
                )
                or not isinstance(value.get("evidence"), dict)
            ):
                raise RuntimeError("prelaunch failure/decision binding drifted")
    if launch_count > len(attempts):
        extra_stem = expected_stems()[len(attempts)]
        if extra_stem != expected_stems()[launch_count - 1]:
            raise RuntimeError("extra completed launch is not the next frozen child")
        if status not in ("implementation_or_contract_defect", "inconclusive"):
            raise RuntimeError(
                "successful terminal decision has an unbound extra launch"
            )
    if launch_count == len(attempts) and len(attempts) < len(expected):
        orphan = ARTIFACT / f"{expected_stems()[len(attempts)]}.conditioning.json"
        if orphan.is_file():
            if status != "implementation_or_contract_defect":
                raise RuntimeError("non-defect decision has orphan conditioning")
            value = json_value(orphan)
            if (
                not strict_equal(value.get("schema"), 1)
                or value.get("artifact_stem") != expected_stems()[len(attempts)]
                or not isinstance(value.get("evidence"), dict)
            ):
                raise RuntimeError("orphan conditioning record is malformed")
            seal_file(orphan, sealed)

    if decision.get("attempts_sha256") != sealed.get(attempts_path):
        raise RuntimeError("decision attempt hash drifted")
    if status in ("kill", "cold_composition_pass"):
        if (
            len(attempts) != 4
            or launch_count != 4
            or any(row.get("valid") is not True for row in attempts)
        ):
            raise RuntimeError(
                "terminal composition decision lacks four valid attempts"
            )
        recomputed = analyze(attempts)
        expected_status = "cold_composition_pass" if recomputed["passes"] else "kill"
        if (
            not strict_equal(decision.get("stage"), recomputed)
            or status != expected_status
        ):
            raise RuntimeError("terminal composition analysis/status drifted")
        if decision.get("stopped_after") != "cold-composition":
            raise RuntimeError("terminal composition stop stage drifted")
    elif isinstance(decision.get("stage"), dict):
        raise RuntimeError("early terminal decision carries aggregate stage analysis")
    if status == "inconclusive" and attempts and attempts[-1].get("valid") is not True:
        if failed != attempts[-1].get("artifact_stem") or not strict_equal(
            decision.get("reasons"), attempts[-1].get("validity_reasons")
        ):
            raise RuntimeError("inconclusive attempt binding drifted")

    final = ARTIFACT / "final-model-sha256.json"
    seal_file(final, sealed)
    final_value = json_value(final)
    if not strict_equal(final_value, expected_final_identity):
        raise RuntimeError(
            "sealed final identity differs from freshly computed payload"
        )
    final_expected = {
        "schema": 1,
        "model": str(MODEL),
        "expected_size_bytes": EXPECTED_MODEL_SIZE,
        "expected_sha256": manifest["sha256"][str(MODEL)],
        "expected_file_identity": manifest["model_file_identity"],
    }
    final_keys = {
        *final_expected,
        "actual_sha256",
        "observed_file_identity_before",
        "observed_file_identity_after",
        "non_model_identity_error",
        "model_identity_error",
        "matches",
    }
    if set(final_value) != final_keys or any(
        not strict_equal(final_value.get(key), value)
        for key, value in final_expected.items()
    ):
        raise RuntimeError("final identity report does not bind the manifest")
    actual_hash = final_value.get("actual_sha256")
    before = final_value.get("observed_file_identity_before")
    after = final_value.get("observed_file_identity_after")
    non_model_error = final_value.get("non_model_identity_error")
    model_error = final_value.get("model_identity_error")
    if (
        not (
            actual_hash is None
            or isinstance(actual_hash, str)
            and re.fullmatch(r"[0-9a-f]{64}", actual_hash) is not None
        )
        or not (before is None or valid_file_identity(before))
        or not (after is None or valid_file_identity(after))
        or not (non_model_error is None or isinstance(non_model_error, str))
        or not (model_error is None or isinstance(model_error, str))
        or type(final_value.get("matches")) is not bool
    ):
        raise RuntimeError("final identity report field type drifted")
    recomputed_matches = (
        non_model_error is None
        and model_error is None
        and actual_hash == final_expected["expected_sha256"]
        and strict_equal(before, final_expected["expected_file_identity"])
        and strict_equal(after, final_expected["expected_file_identity"])
    )
    if final_value["matches"] is not recomputed_matches:
        raise RuntimeError("final identity report consistency drifted")
    successful_final = {
        **final_expected,
        "actual_sha256": manifest["sha256"][str(MODEL)],
        "observed_file_identity_before": manifest["model_file_identity"],
        "observed_file_identity_after": manifest["model_file_identity"],
        "non_model_identity_error": None,
        "model_identity_error": None,
        "matches": True,
    }
    decision_identity_error = decision.get("identity_error")
    if decision_identity_error is None and not strict_equal(
        final_value, successful_final
    ):
        raise RuntimeError("non-defect decision lacks exact successful final identity")
    if decision_identity_error is not None and (
        status != "implementation_or_contract_defect"
        or decision_identity_error
        != "RuntimeError: final packet identity did not match manifest"
        or final_value["matches"] is not False
    ):
        raise RuntimeError("identity-defect decision/report binding drifted")

    publications = {
        "decision.json",
        "artifact-inventory.sha256",
        "packet-complete.json",
        ".decision.json.tmp",
        ".artifact-inventory.sha256.tmp",
        ".packet-complete.json.tmp",
    }
    unsafe = [
        path.name
        for path in ARTIFACT.iterdir()
        if path.is_symlink() or not path.is_file()
    ]
    if unsafe:
        raise RuntimeError(f"unsafe packet artifacts exist: {sorted(unsafe)}")
    unknown = {
        path.name
        for path in ARTIFACT.iterdir()
        if path.is_file() and path not in sealed and path.name not in publications
    }
    if unknown:
        raise RuntimeError(f"unverified packet artifacts exist: {sorted(unknown)}")
    return sealed


def attempts_sha256() -> str | None:
    path = ARTIFACT / "attempts.jsonl"
    return sha256(path) if path.is_file() else None


def make_decision(
    status: str,
    manifest: dict[str, object],
    cpu_test: dict[str, object] | None,
    stage: dict[str, object] | None,
    stopped_after: str,
    **extra: object,
) -> dict[str, object]:
    if status not in VALID_STATUSES:
        raise RuntimeError(f"status is not frozen: {status}")
    return {
        "schema": 1,
        "status": status,
        "authority": "none",
        "force_authorized": False,
        "successor_authorization": PASS_SUCCESSOR
        if status == "cold_composition_pass"
        else "none",
        "stopped_after": stopped_after,
        "source_commit": manifest["source_commit"],
        "imported_v0627": manifest["v0627_bridge"],
        "cpu_test": cpu_test,
        "stage": stage,
        "attempts_sha256": attempts_sha256(),
        **extra,
    }


def publish(decision: dict[str, object], manifest: dict[str, object]) -> None:
    prior = signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        identity_error = None
        try:
            final_identity = record_final_identity(manifest)
            if final_identity.get("matches") is not True:
                identity_error = (
                    "RuntimeError: final packet identity did not match manifest"
                )
        except UnsealedPacket:
            raise
        except Exception as error:
            identity_error = f"{type(error).__name__}: {error}"
            raise UnsealedPacket(
                f"final identity could not produce a complete payload: {identity_error}"
            ) from error
        if identity_error is not None:
            decision = {
                **decision,
                "reported_status_before_identity_check": decision.get("status"),
                "completed_stage_before_identity_defect": decision.get("stage"),
                "status": "implementation_or_contract_defect",
                "authority": "none",
                "force_authorized": False,
                "successor_authorization": "none",
                "stage": None,
                "identity_error": identity_error,
            }
        sealed = verify_attempt_artifacts(decision, manifest, final_identity)
        mechanics.write_decision(decision, sealed)
    finally:
        signal.signal(signal.SIGINT, prior)


def configure_mechanics() -> None:
    mechanics.ARTIFACT = ARTIFACT
    mechanics.PREREG = PREREG
    mechanics.MODEL = MODEL
    mechanics.CLI_BINARY = QWEN
    mechanics.BENCH_BINARY = QWEN_BENCH
    mechanics.PAIR_ORDERS = PAIR_ORDERS
    mechanics.arm_environment = arm_environment
    mechanics.verify_non_model_identity = verify_non_model_identity


def main(*, preflight_only: bool) -> None:
    if ARTIFACT.exists():
        raise RuntimeError(f"refusing to reuse packet directory {ARTIFACT}")
    configure_mechanics()
    base_env, removed = mechanics.common.normalized_environment()
    manifest = build_manifest(removed, base_env)
    if preflight_only:
        vm = mechanics.capture_vm_state()
        if vm["capture_errors"]:
            raise RuntimeError(f"VM preflight capture failed: {vm['capture_errors']}")
        print(
            json_text(
                {"status": "preflight-passed", "source": manifest["source_commit"]}
            )
        )
        return
    mechanics.reserve_artifact(manifest)
    cpu_test = None
    stage = None
    launched_stage = "cpu-default-policy"
    try:
        cpu_test = run_cpu_test(base_env)
        verify_non_model_identity(manifest)
        launched_stage = "cold-composition"
        rows = run_stage(base_env, manifest)
        stage = analyze(rows)
        status = "cold_composition_pass" if stage["passes"] else "kill"
        publish(
            make_decision(status, manifest, cpu_test, stage, launched_stage), manifest
        )
    except UnsealedPacket:
        raise
    except InconclusivePacket as error:
        publish(
            make_decision(
                "inconclusive",
                manifest,
                cpu_test,
                stage,
                error.stage,
                failed_child=error.child,
                reasons=error.reasons,
            ),
            manifest,
        )
    except KeyboardInterrupt:
        publish(
            make_decision(
                "inconclusive",
                manifest,
                cpu_test,
                stage,
                launched_stage,
                reasons=["operator_interrupt_after_child_cleanup"],
            ),
            manifest,
        )
    except Exception as error:
        if mechanics.decision_publication_started():
            raise
        publish(
            make_decision(
                "implementation_or_contract_defect",
                manifest,
                cpu_test,
                None,
                launched_stage,
                completed_stage_before_defect=stage,
                error_type=type(error).__name__,
                error=str(error),
            ),
            manifest,
        )
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true")
    arguments = parser.parse_args()
    main(preflight_only=arguments.preflight_only)
