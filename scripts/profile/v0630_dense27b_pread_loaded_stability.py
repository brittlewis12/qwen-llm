#!/usr/bin/env python3
"""v0.630 dense-27B direct-pread short-period loaded stability."""

import argparse
import ctypes
import hashlib
import json
import math
import mmap
import os
import re
import signal
import statistics
import subprocess
import time
from decimal import Decimal, InvalidOperation
from itertools import pairwise
from pathlib import Path

import v0593_demand_paged_no_copy as common

ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = ROOT / "target/profiles/v0630-dense27b-pread-loaded-stability-p1"
PREREG = ROOT / "docs/bench/v0630-dense27b-pread-loaded-stability.md"
RUNNER = Path(__file__).resolve()
BENCH_SOURCE = ROOT / "crates/qwen-cli/src/bench.rs"
MODEL = Path("/Users/tito/models/Qwen3.6-27B-Q4_K_M.gguf")
PROMPT = ROOT / "docs/bench/tokenizer-prompts/current-reva-n8-interactive-qwen36.txt"
BENCH_BINARY = ROOT / "target/release/qwen-bench"
V0629_ROOT = ROOT / "target/profiles/v0629-dense27b-pread-cold-composition-repair-p1"
V0629_DECISION = V0629_ROOT / "decision.json"
V0629_INVENTORY = V0629_ROOT / "artifact-inventory.sha256"
V0629_COMPLETE = V0629_ROOT / "packet-complete.json"

BASE_COMMIT = "cc47190654b898e2004117b6d03be1006a89d1fa"
EXPECTED_MACOS_PRODUCT_VERSION = "15.6.1"
EXPECTED_MACOS_BUILD_VERSION = "24G90"
EXPECTED_GGUF_COMMIT = "c7369fd4868a6f613459fff355477f53bf4ee2f1"
EXPECTED_LLAMA_CPP_RS_COMMIT = "fe4fb533d1ed2855b6ac5492e56c42007d410409"
EXPECTED_LLAMA_CPP_RS_UNTRACKED = (".claude/settings.local.json",)
EXPECTED_BENCH_SOURCE_SHA256 = (
    "5efea85fc599f8efe1b5ce2fd7e930a779065555df84b81d6de08370021e05ba"
)
EXPECTED_IMPLEMENTATION_PATCH_SHA256 = (
    "bf6ad710294af218b802ddb6631f8381c639eca419d332fd26c87cb96f5bc14e"
)
EXPECTED_MODEL_SHA256 = (
    "5ed60d0af4650a854b1755bd392f9aef4872643dc25a254bc68043fa638392a0"
)
EXPECTED_MODEL_SIZE = 16_817_244_384
EXPECTED_PROMPT_SHA256 = (
    "e265de9742d1b22e566fc108ae26331ccf46166c6f071e73f211e0a1a7e8b474"
)
EXPECTED_PROMPT_BYTES = 1_891
EXPECTED_PROMPT_TOKENS = 419
EXPECTED_TRANSITIONS = 31
EXPECTED_TRACE_TOKENS = EXPECTED_TRANSITIONS + 1
EXPECTED_VOCAB_SIZE = 248_320
EXPECTED_DEVICE = (
    "device: Apple M4 Max | unified_memory=true | max_threadgroup_memory=32768 bytes"
)
EXPECTED_HW_MEMSIZE = 137_438_953_472
V0629_SEALS = {
    V0629_DECISION: "949f8066c3ab7261a2ee7916ca4d3e8ce92b56e6a6cf4396b55474020be21aa9",
    V0629_INVENTORY: "6a94d45e1b8ccb170ac77226a65033d26e6441fbe415298828288dc09a2c4e5c",
    V0629_COMPLETE: "11c2a86d958ee4d935a87edfa37e335616232a9e63ed2c8b713c7b08c39a8ba5",
}
QUARTET_ORDERS = ("ABBA", "BAAB") * 4
CORRECTNESS_TEST = "metal_forward::tests::gguf_parallel_pread_dense27b_q4_is_bit_exact"
VALID_STATUSES = {
    "implementation_or_contract_defect",
    "inconclusive",
    "kill",
    "go",
}
U64_MAX = 2**64 - 1
MAX_CONDITIONING_TO_LAUNCH_NS = 2_000_000_000
MAX_CONSECUTIVE_START_NS = 12_000_000_000
MAX_QUARTET_SPAN_NS = 45_000_000_000
MAX_REVERSAL_SPAN_NS = 90_000_000_000
MAX_ARM_RELATIVE_RANGE = 0.05
SAFE_ENVIRONMENT_KEYS = (
    "CARGO_HOME",
    "HOME",
    "LANG",
    "LC_ALL",
    "LOGNAME",
    "MISE_CACHE_DIR",
    "MISE_CONFIG_DIR",
    "MISE_DATA_DIR",
    "PATH",
    "RUSTUP_HOME",
    "SHELL",
    "TERM",
    "TMPDIR",
    "USER",
)

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


class ContractDefect(RuntimeError):
    pass


class ChildContractDefect(ContractDefect):
    def __init__(self, child: str, reason: str) -> None:
        super().__init__(reason)
        self.child = child


class InconclusivePacket(RuntimeError):
    def __init__(self, stage: str, child: str | None, reasons: list[str]) -> None:
        super().__init__(f"{stage}: {child}: {', '.join(reasons)}")
        self.stage = stage
        self.child = child
        self.reasons = reasons


class UnsealedPacket(RuntimeError):
    pass


def sha256(path: Path) -> str:
    return common.sha256_file(path)


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
    if pretty:
        return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False)
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def parse_json(text: str, label: str) -> object:
    try:
        return json.loads(text)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ContractDefect(f"{label} is not valid JSON: {error}") from error


def write_fsynced(path: Path, data: bytes, *, exclusive: bool = True) -> None:
    mode = "xb" if exclusive else "wb"
    try:
        with path.open(mode) as output:
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
    except OSError as error:
        raise UnsealedPacket(f"durable write failed for {path}: {error}") from error


def write_json(path: Path, value: object, *, exclusive: bool = True) -> None:
    write_fsynced(
        path,
        (json_text(value, pretty=True) + "\n").encode("utf-8"),
        exclusive=exclusive,
    )


def append_jsonl(path: Path, value: object) -> None:
    try:
        with path.open("ab") as output:
            output.write((json_text(value) + "\n").encode("utf-8"))
            output.flush()
            os.fsync(output.fileno())
    except OSError as error:
        raise UnsealedPacket(f"durable append failed for {path}: {error}") from error


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def command_output(command: list[str], *, env: dict[str, str] | None = None) -> str:
    return subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    ).stdout


def git_output(arguments: list[str], *, repo: Path = ROOT) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=repo,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    ).stdout.strip()


def changed_paths(left: str, right: str) -> list[str]:
    value = git_output(["diff", "--name-only", f"{left}..{right}"])
    return [] if not value else value.splitlines()


def commit_parents(commit: str) -> list[str]:
    fields = git_output(["rev-list", "--parents", "-n", "1", commit]).split()
    if not fields or fields[0] != commit:
        raise ContractDefect(f"cannot resolve exact commit parents: {commit}")
    return fields[1:]


def name_status(left: str, right: str) -> list[str]:
    value = git_output(["diff", "--name-status", f"{left}..{right}"])
    return [] if not value else value.splitlines()


def implementation_patch_sha256(left: str, right: str) -> str:
    result = subprocess.run(
        [
            "git",
            "diff",
            "--binary",
            f"{left}..{right}",
            "--",
            str(BENCH_SOURCE.relative_to(ROOT)),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    return hashlib.sha256(result.stdout).hexdigest()


def sibling_identity(
    path: Path, expected_commit: str, allowed_untracked: tuple[str, ...] = ()
) -> dict[str, object]:
    if not path.is_dir():
        raise ContractDefect(f"sibling dependency is missing: {path}")
    commit = git_output(["rev-parse", "HEAD"], repo=path)
    status = git_output(
        ["status", "--porcelain=v1", "--untracked-files=all"], repo=path
    )
    lines = [] if not status else status.splitlines()
    tracked_dirty = [line for line in lines if not line.startswith("?? ")]
    untracked = sorted(line[3:] for line in lines if line.startswith("?? "))
    if tracked_dirty:
        raise ContractDefect(
            f"sibling dependency has tracked changes: {path}: {tracked_dirty!r}"
        )
    if untracked != sorted(allowed_untracked):
        raise ContractDefect(
            f"sibling dependency untracked allowlist drifted: {path}: {untracked!r}"
        )
    if commit != expected_commit:
        raise ContractDefect(f"sibling dependency commit drifted: {path}: {commit}")
    return {
        "path": str(path.resolve()),
        "commit": commit,
        "tracked_dirty": False,
        "allowed_untracked": untracked,
    }


def verify_source_topology() -> dict[str, object]:
    for path in (PREREG, RUNNER, BENCH_SOURCE):
        relative = path.relative_to(ROOT)
        git_output(["ls-files", "--error-unmatch", str(relative)])
    dirty = git_output(["status", "--porcelain=v1", "--untracked-files=all"])
    if dirty:
        raise ContractDefect(f"worktree is not clean: {dirty!r}")
    head = git_output(["rev-parse", "HEAD"])
    prereg = git_output(["rev-parse", "HEAD^"])
    base = git_output(["rev-parse", "HEAD^^"])
    if base != BASE_COMMIT:
        raise ContractDefect(f"v0.630 base commit drifted: {base}")
    if commit_parents(prereg) != [base] or commit_parents(head) != [prereg]:
        raise ContractDefect(
            "v0.630 preregistration or implementation is a merge commit"
        )
    expected_prereg = sorted(
        [str(PREREG.relative_to(ROOT)), str(RUNNER.relative_to(ROOT))]
    )
    expected_impl = [str(BENCH_SOURCE.relative_to(ROOT))]
    if sorted(changed_paths(base, prereg)) != expected_prereg:
        raise ContractDefect("preregistration commit has an unexpected diff")
    if changed_paths(prereg, head) != expected_impl:
        raise ContractDefect("implementation commit has an unexpected diff")
    if sorted(name_status(base, prereg)) != sorted(
        f"A\t{path}" for path in expected_prereg
    ):
        raise ContractDefect("preregistration files were not exact additions")
    if name_status(prereg, head) != [f"M\t{expected_impl[0]}"]:
        raise ContractDefect("bench implementation was not an exact modification")
    if sha256(BENCH_SOURCE) != EXPECTED_BENCH_SOURCE_SHA256:
        raise ContractDefect("complete bench.rs implementation digest drifted")
    if (
        implementation_patch_sha256(prereg, head)
        != EXPECTED_IMPLEMENTATION_PATCH_SHA256
    ):
        raise ContractDefect("frozen R..H implementation patch drifted")
    build = parse_json(
        command_output([str(BENCH_BINARY), "build-info", "--output", "json"]),
        "qwen-bench build-info",
    )
    if not isinstance(build, dict):
        raise ContractDefect("qwen-bench build identity is not an object")
    if (
        build.get("build_commit") != head
        or build.get("runtime_commit") != head
        or build.get("status") != "match"
        or build.get("build_dirty") is not False
        or build.get("runtime_dirty") is not False
        or build.get("build_source_state") != build.get("runtime_source_state")
    ):
        raise ContractDefect(f"source/build/runtime identity mismatch: {build}")
    dependencies = [
        sibling_identity(ROOT.parent / "gguf", EXPECTED_GGUF_COMMIT),
        sibling_identity(
            ROOT.parent / "llama-cpp-rs",
            EXPECTED_LLAMA_CPP_RS_COMMIT,
            EXPECTED_LLAMA_CPP_RS_UNTRACKED,
        ),
    ]
    return {
        "head": head,
        "preregistration_commit": prereg,
        "base_commit": base,
        "build_identity": build,
        "sibling_dependencies": dependencies,
    }


def inventory_members(path: Path) -> dict[Path, str]:
    members: dict[Path, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if match is None:
            raise ContractDefect(f"malformed inventory line: {line!r}")
        member = ROOT / match.group(2)
        if member in members:
            raise ContractDefect(f"duplicate inventory member: {member}")
        members[member] = match.group(1)
    return members


def verify_v0629_authority() -> dict[str, object]:
    for path, expected in V0629_SEALS.items():
        if not path.is_file() or path.is_symlink() or sha256(path) != expected:
            raise ContractDefect(f"v0.629 seal drifted: {path}")
    complete = parse_json(
        V0629_COMPLETE.read_text(encoding="utf-8"), "v0.629 completion"
    )
    if not strict_equal(
        complete,
        {
            "schema": 1,
            "decision_sha256": V0629_SEALS[V0629_DECISION],
            "inventory_sha256": V0629_SEALS[V0629_INVENTORY],
        },
    ):
        raise ContractDefect("v0.629 completion binding drifted")
    members = inventory_members(V0629_INVENTORY)
    if len(members) != 27 or members.get(V0629_DECISION) != V0629_SEALS[V0629_DECISION]:
        raise ContractDefect("v0.629 inventory membership drifted")
    for path, expected in members.items():
        if not path.is_file() or path.is_symlink() or sha256(path) != expected:
            raise ContractDefect(f"v0.629 inventory member drifted: {path}")
    observed = set(V0629_ROOT.iterdir())
    if observed != set(members) | {V0629_INVENTORY, V0629_COMPLETE}:
        raise ContractDefect("v0.629 final artifact set drifted")
    decision = parse_json(V0629_DECISION.read_text(encoding="utf-8"), "v0.629 decision")
    if not isinstance(decision, dict):
        raise ContractDefect("v0.629 decision is malformed")
    terminal = {
        "schema": 1,
        "status": "cold_composition_pass",
        "authority": "none",
        "force_authorized": False,
        "stopped_after": "cold-composition",
        "successor_authorization": (
            "preregister-separate-short-period-loaded-stability-packet-only"
        ),
        "source_commit": "fd089c7d40f0d9391048a9051ed31d64df0716d8",
    }
    for key, expected in terminal.items():
        if not strict_equal(decision.get(key), expected):
            raise ContractDefect(f"v0.629 authority field drifted: {key}")
    stage = decision.get("stage")
    if not isinstance(stage, dict) or stage.get("passes") is not True:
        raise ContractDefect("v0.629 cold-composition result drifted")
    return {
        "decision_sha256": V0629_SEALS[V0629_DECISION],
        "inventory_sha256": V0629_SEALS[V0629_INVENTORY],
        "completion_sha256": V0629_SEALS[V0629_COMPLETE],
        "inventory_members": len(members),
        "authority_imported": True,
        "performance_observations_imported": 0,
        "successor_authorization": terminal["successor_authorization"],
    }


def file_identity(path: Path) -> dict[str, int]:
    value = path.stat()
    return {
        "device": value.st_dev,
        "inode": value.st_ino,
        "size_bytes": value.st_size,
        "mtime_ns": value.st_mtime_ns,
    }


def safe_environments() -> tuple[dict[str, str], dict[str, str], list[str]]:
    inherited = os.environ
    child = {key: inherited[key] for key in SAFE_ENVIRONMENT_KEYS if inherited.get(key)}
    for required in ("HOME", "PATH", "TMPDIR"):
        if required not in child:
            raise ContractDefect(f"required safe environment key is absent: {required}")
    child["RUST_BACKTRACE"] = "0"
    tests = child.copy()
    tests["RUST_TEST_THREADS"] = "1"
    removed = sorted(key for key in inherited if key not in tests)
    return child, tests, removed


def build_manifest(
    child_env: dict[str, str], test_env: dict[str, str], removed: list[str]
) -> dict[str, object]:
    source = verify_source_topology()
    authority = verify_v0629_authority()
    if command_output(["sysctl", "-n", "hw.memsize"]).strip() != str(
        EXPECTED_HW_MEMSIZE
    ):
        raise ContractDefect("host memory size drifted")
    if command_output([str(BENCH_BINARY), "metal-info"]).strip() != EXPECTED_DEVICE:
        raise ContractDefect("Metal device identity drifted")
    model_identity = file_identity(MODEL)
    if model_identity["size_bytes"] != EXPECTED_MODEL_SIZE:
        raise ContractDefect("model size drifted")
    prompt_bytes = PROMPT.read_bytes()
    if len(prompt_bytes) != EXPECTED_PROMPT_BYTES:
        raise ContractDefect("prompt byte count drifted")
    paths = (PREREG, RUNNER, BENCH_SOURCE, MODEL, PROMPT, BENCH_BINARY)
    hashes = {str(path): sha256(path) for path in paths}
    if hashes[str(MODEL)] != EXPECTED_MODEL_SHA256:
        raise ContractDefect("model SHA-256 drifted")
    if hashes[str(PROMPT)] != EXPECTED_PROMPT_SHA256:
        raise ContractDefect("prompt SHA-256 drifted")
    tool_dialects = verify_tool_dialects(child_env)
    product_version = command_output(["sw_vers", "-productVersion"]).strip()
    build_version = command_output(["sw_vers", "-buildVersion"]).strip()
    if (
        product_version != EXPECTED_MACOS_PRODUCT_VERSION
        or build_version != EXPECTED_MACOS_BUILD_VERSION
    ):
        raise ContractDefect("frozen macOS product/build identity drifted")
    return {
        "schema": 1,
        "created_unix_ms": time.time_ns() // 1_000_000,
        "source": source,
        "source_commit": source["head"],
        "build_identity": source["build_identity"],
        "imported_v0629": authority,
        "model_file_identity": model_identity,
        "macos_product_version": product_version,
        "macos_build_version": build_version,
        "removed_environment": removed,
        "child_environment": child_env,
        "test_environment": test_env,
        "tool_dialects": tool_dialects,
        "sha256": hashes,
        "cell": {
            "quartet_orders": list(QUARTET_ORDERS),
            "prompt_bytes": EXPECTED_PROMPT_BYTES,
            "prompt_tokens": EXPECTED_PROMPT_TOKENS,
            "transitions": EXPECTED_TRANSITIONS,
            "runs": 1,
            "prefill_chunk": 1024,
            "kv_capacity": 1024,
            "full_logits_decode": True,
            "generated_token_trace": True,
        },
    }


def verify_non_model_identity(manifest: dict[str, object]) -> None:
    source = verify_source_topology()
    if not strict_equal(source, manifest.get("source")):
        raise ContractDefect("source/build/dependency identity changed")
    child_env, test_env, removed = safe_environments()
    if (
        not strict_equal(child_env, manifest.get("child_environment"))
        or not strict_equal(test_env, manifest.get("test_environment"))
        or not strict_equal(removed, manifest.get("removed_environment"))
    ):
        raise ContractDefect("safe environment contract changed")
    hashes = manifest.get("sha256")
    if not isinstance(hashes, dict):
        raise ContractDefect("manifest hash map is malformed")
    for path_text, expected in hashes.items():
        path = Path(path_text)
        if path == MODEL:
            continue
        if not path.is_file() or sha256(path) != expected:
            raise ContractDefect(f"non-model input changed: {path}")
    if command_output(["sw_vers", "-productVersion"]).strip() != manifest.get(
        "macos_product_version"
    ) or command_output(["sw_vers", "-buildVersion"]).strip() != manifest.get(
        "macos_build_version"
    ):
        raise ContractDefect("OS identity changed")


def capture_host_state() -> dict[str, object]:
    try:
        return common.capture_host_state()
    except (OSError, RuntimeError, subprocess.SubprocessError) as error:
        return {
            "thermal": None,
            "battery": None,
            "memory_pressure": None,
            "memory_available_percent": None,
            "valid": False,
            "capture_error": f"{type(error).__name__}:{error}",
        }


def parse_swap_bytes(text: str) -> int:
    match = re.search(r"\bused\s*=\s*([0-9.]+)([KMG])", text)
    if match is None:
        raise RuntimeError(f"cannot parse swap usage: {text!r}")
    scale = {"K": 1024, "M": 1024**2, "G": 1024**3}[match.group(2)]
    try:
        value = float(match.group(1))
    except ValueError as error:
        raise RuntimeError(f"cannot parse swap decimal: {text!r}") from error
    if not math.isfinite(value) or value < 0:
        raise RuntimeError(f"invalid swap decimal: {text!r}")
    return round(value * scale)


def parse_vm_counter(text: str, label: str) -> int:
    match = re.search(rf"^{re.escape(label)}:\s+(\d+)\.$", text, re.MULTILINE)
    if match is None:
        raise RuntimeError(f"cannot parse vm_stat {label}")
    return int(match.group(1))


def capture_vm_state() -> dict[str, object]:
    errors: list[str] = []
    try:
        vm_stat = command_output(["vm_stat"])
    except (OSError, RuntimeError, subprocess.SubprocessError) as error:
        vm_stat = None
        errors.append(f"vm_stat={type(error).__name__}:{error}")
    try:
        swapusage = command_output(["sysctl", "-n", "vm.swapusage"])
    except (OSError, RuntimeError, subprocess.SubprocessError) as error:
        swapusage = None
        errors.append(f"swapusage={type(error).__name__}:{error}")
    parsed: dict[str, int | None] = {
        "pageouts": None,
        "compressions": None,
        "swapouts": None,
        "compressor_stored_pages": None,
        "compressor_occupied_pages": None,
        "swap_used_bytes": None,
    }
    if vm_stat is not None:
        for key, label in (
            ("pageouts", "Pageouts"),
            ("compressions", "Compressions"),
            ("swapouts", "Swapouts"),
            ("compressor_stored_pages", "Pages stored in compressor"),
            ("compressor_occupied_pages", "Pages occupied by compressor"),
        ):
            try:
                parsed[key] = parse_vm_counter(vm_stat, label)
            except RuntimeError as error:
                errors.append(f"{key}={type(error).__name__}:{error}")
    if swapusage is not None:
        try:
            parsed["swap_used_bytes"] = parse_swap_bytes(swapusage)
        except RuntimeError as error:
            errors.append(f"swap_used_bytes={type(error).__name__}:{error}")
    return {**parsed, "vm_stat": vm_stat, "swapusage": swapusage, "errors": errors}


def vm_interval(
    label: str, before: dict[str, object], after: dict[str, object]
) -> dict[str, object]:
    keys = (
        "pageouts",
        "compressions",
        "swapouts",
        "compressor_stored_pages",
        "compressor_occupied_pages",
        "swap_used_bytes",
    )
    deltas: dict[str, int | None] = {}
    for key in keys:
        left = before.get(key)
        right = after.get(key)
        if type(left) is int and type(right) is int:
            deltas[key] = right - left
        else:
            deltas[key] = None
    reasons: list[str] = []
    if before.get("errors") or after.get("errors"):
        reasons.append(f"{label}_vm_capture_invalid")
    if any(value is None for value in deltas.values()) and not reasons:
        reasons.append(f"{label}_vm_delta_unavailable")
    for key in ("pageouts", "compressions", "swapouts"):
        value = deltas[key]
        if value is not None and value < 0:
            reasons.append(f"{label}_{key}_counter_regressed")
    for key in ("swapouts", "swap_used_bytes"):
        if deltas[key] is not None and deltas[key] > 0:
            reasons.append(f"{label}_{key}_growth")
    return {
        "label": label,
        "before": before,
        "after": after,
        "deltas": deltas,
        "advisory": {
            "pageouts_growth": max(deltas["pageouts"] or 0, 0),
            "compressions_growth": max(deltas["compressions"] or 0, 0),
            "compressor_stored_pages_delta": deltas["compressor_stored_pages"],
            "compressor_occupied_pages_delta": deltas["compressor_occupied_pages"],
        },
        "failure_reasons": reasons,
    }


def probe_model_residency(expected_identity: object) -> dict[str, object]:
    if (
        not isinstance(expected_identity, dict)
        or expected_identity.get("size_bytes") != EXPECTED_MODEL_SIZE
    ):
        raise ContractDefect("manifest model identity is malformed")
    size = EXPECTED_MODEL_SIZE
    page_size = os.sysconf("SC_PAGE_SIZE")
    page_count = (size + page_size - 1) // page_size
    libc = ctypes.CDLL(None, use_errno=True)
    libc.mmap.restype = ctypes.c_void_p
    libc.mmap.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_longlong,
    ]
    libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
    libc.mincore.restype = ctypes.c_int
    libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    libc.munmap.restype = ctypes.c_int
    descriptor = os.open(MODEL, os.O_RDONLY)
    primary: BaseException | None = None
    try:
        if (
            file_identity(MODEL) != expected_identity
            or {
                "device": os.fstat(descriptor).st_dev,
                "inode": os.fstat(descriptor).st_ino,
                "size_bytes": os.fstat(descriptor).st_size,
                "mtime_ns": os.fstat(descriptor).st_mtime_ns,
            }
            != expected_identity
        ):
            raise ContractDefect("model identity drifted before mincore")
        address = libc.mmap(None, size, mmap.PROT_READ, mmap.MAP_SHARED, descriptor, 0)
        if address == ctypes.c_void_p(-1).value:
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error))
        mapping_error: BaseException | None = None
        try:
            vector = (ctypes.c_ubyte * page_count)()
            if libc.mincore(address, size, vector) != 0:
                error = ctypes.get_errno()
                raise OSError(error, os.strerror(error))
            resident = sum(1 for value in vector if value & 1)
            if file_identity(MODEL) != expected_identity:
                raise ContractDefect("model identity drifted during mincore")
        except BaseException as error:
            mapping_error = error
            raise
        finally:
            if libc.munmap(address, size) != 0:
                cleanup = OSError(ctypes.get_errno(), os.strerror(ctypes.get_errno()))
                if mapping_error is None:
                    raise cleanup
                mapping_error.add_note(f"munmap failed: {cleanup}")
    except BaseException as error:
        primary = error
        raise
    finally:
        try:
            os.close(descriptor)
        except OSError as cleanup:
            if primary is None:
                raise
            primary.add_note(f"close failed: {cleanup}")
    return {
        "page_size": page_size,
        "total_pages": page_count,
        "resident_pages": resident,
        "resident_fraction": resident / page_count,
        "all_pages_resident": resident == page_count,
        "file_identity": expected_identity,
    }


def warm_model_file(expected_identity: object) -> dict[str, object]:
    if file_identity(MODEL) != expected_identity:
        raise ContractDefect("model identity drifted before cache conditioning")
    started = time.perf_counter_ns()
    total = 0
    buffer = bytearray(8 * 1024 * 1024)
    with MODEL.open("rb", buffering=0) as handle:
        while True:
            count = handle.readinto(buffer)
            if count == 0:
                break
            total += count
    if total != EXPECTED_MODEL_SIZE or file_identity(MODEL) != expected_identity:
        raise ContractDefect("cache conditioning did not preserve exact model identity")
    return {"wall_ns": time.perf_counter_ns() - started, "bytes_read": total}


def condition_child(stem: str, manifest: dict[str, object]) -> dict[str, object]:
    verify_non_model_identity(manifest)
    host_before = capture_host_state()
    vm_before = capture_vm_state()
    if host_before.get("valid") is not True:
        write_json(
            ARTIFACT / f"{stem}.conditioning.json",
            {
                "schema": 1,
                "artifact_stem": stem,
                "monotonic_ns": time.monotonic_ns(),
                "host_before_cache": host_before,
                "vm_before_cache": vm_before,
                "validity_reasons": ["host_invalid_before_cache"],
            },
        )
        raise InconclusivePacket("conditioning", stem, ["host_invalid_before_cache"])
    try:
        cache = warm_model_file(manifest.get("model_file_identity"))
        residency = probe_model_residency(manifest.get("model_file_identity"))
    except ContractDefect:
        raise
    except Exception as error:
        reason = f"cache_or_residency_failed={type(error).__name__}:{error}"
        write_json(
            ARTIFACT / f"{stem}.conditioning.json",
            {
                "schema": 1,
                "artifact_stem": stem,
                "monotonic_ns": time.monotonic_ns(),
                "host_before_cache": host_before,
                "vm_before_cache": vm_before,
                "validity_reasons": [reason],
            },
        )
        raise InconclusivePacket(
            "conditioning",
            stem,
            [reason],
        ) from error
    host_after = capture_host_state()
    vm_after = capture_vm_state()
    interval = vm_interval("conditioning", vm_before, vm_after)
    reasons = list(interval["failure_reasons"])
    if host_after.get("valid") is not True:
        reasons.append("host_invalid_before_spawn")
    if residency.get("all_pages_resident") is not True:
        reasons.append("target_file_not_fully_resident")
    evidence = {
        "schema": 1,
        "artifact_stem": stem,
        "monotonic_ns": time.monotonic_ns(),
        "host_before_cache": host_before,
        "host_before_spawn": host_after,
        "cache": cache,
        "residency": residency,
        "interval": interval,
    }
    path = ARTIFACT / f"{stem}.conditioning.json"
    write_json(path, evidence)
    if reasons:
        raise InconclusivePacket("conditioning", stem, reasons)
    return evidence


def arm_environment(
    base_env: dict[str, str], arm: str
) -> tuple[dict[str, str], dict[str, str]]:
    if arm not in ("A", "B"):
        raise ContractDefect(f"invalid arm: {arm}")
    delta = {
        "QWEN_GGUF_PARALLEL_COPY": "0" if arm == "A" else "pread",
        "QWEN_GGUF_OWNED_ARENA": "0",
        "QWEN_GGUF_NO_COPY": "0",
    }
    env = base_env.copy()
    env.update(delta)
    return env, delta


def child_command(prompt: str) -> list[str]:
    return [
        "/usr/bin/time",
        "-l",
        str(BENCH_BINARY),
        "decode",
        "--model",
        str(MODEL),
        "--prompt",
        prompt,
        "--tokens",
        str(EXPECTED_TRANSITIONS),
        "--runs",
        "1",
        "--prefill-chunk",
        "1024",
        "--kv-capacity",
        "1024",
        "--full-logits-decode",
        "--generated-token-trace",
    ]


def parse_marker(line: str) -> dict[str, int | float]:
    if " plan=" in line:
        raise ContractDefect("dense direct-pread marker contains a planner field")
    suffix = line.removeprefix(MARKER_PREFIX)
    if suffix == line:
        raise ContractDefect("dense direct-pread marker prefix drifted")
    names = (
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
    fields = suffix.split(" ")
    if len(fields) != len(names):
        raise ContractDefect("dense marker dynamic field count drifted")
    values: dict[str, int | float] = {}
    for field, name in zip(fields, names, strict=True):
        value = field.removeprefix(f"{name}=")
        if value == field or not value.isascii() or not value.isdecimal():
            raise ContractDefect(f"dense marker {name} is not unsigned decimal")
        parsed = int(value)
        if str(parsed) != value or parsed > U64_MAX:
            raise ContractDefect(f"dense marker {name} is not canonical")
        values[name] = parsed
    phase = sum(int(values[name]) for name in names[:4])
    if int(values["ready_us"]) == 0 or abs(int(values["ready_us"]) - phase) > 4:
        raise ContractDefect("dense marker timing does not reconcile")
    if int(values["total_cpu_us"]) != int(values["user_cpu_us"]) + int(
        values["system_cpu_us"]
    ):
        raise ContractDefect("dense marker CPU does not reconcile")
    if values["timer_major_faults"] != 0:
        raise ContractDefect("dense direct-pread marker reports major faults")
    values["cpu_per_wall"] = int(values["total_cpu_us"]) / int(values["ready_us"])
    return values


def parse_load_contract(stderr: str, arm: str) -> dict[str, object]:
    marker_count = 1 if arm == "B" else 0
    if stderr.count("[metal-gguf-") != marker_count:
        raise ContractDefect(f"{arm} total storage-marker count drifted")
    if stderr.count("[metal-gguf-parallel-pread]") != marker_count:
        raise ContractDefect(f"{arm} direct-pread marker count drifted")
    if "[metal-gguf-parallel-copied]" in stderr:
        raise ContractDefect("parallel mmap-copy marker appeared")
    forbidden = ("[metal-gguf-owned]", "[metal-gguf-retained]", "[metal-gguf-no-copy]")
    if any(value in stderr for value in forbidden):
        raise ContractDefect("unrequested storage marker appeared")
    if stderr.count(POLICY_LINE) != 1 or stderr.count(LEDGER_LINE) != 1:
        raise ContractDefect(f"{arm} policy or ledger count drifted")
    prefixes = (
        "[metal-load] native quantized token embedding policy:",
        "[metal-gguf-parallel-pread]",
        "[metal-load-ledger]",
    )
    lines = [line for line in stderr.splitlines() if line.startswith(prefixes)]
    for line in stderr.splitlines():
        if ("[metal-load" in line or "[metal-gguf-" in line) and line not in lines:
            raise ContractDefect(f"unrecognized load/storage line: {line!r}")
    if arm == "A":
        if lines != [POLICY_LINE, LEDGER_LINE]:
            raise ContractDefect(f"A load protocol drifted: {lines!r}")
        return {"storage": "ordinary-copied", "recognized_lines": lines, "marker": None}
    if len(lines) != 3 or lines[0] != POLICY_LINE or lines[2] != LEDGER_LINE:
        raise ContractDefect(f"B load protocol drifted: {lines!r}")
    return {
        "storage": "parallel-pread",
        "recognized_lines": lines,
        "marker": parse_marker(lines[1]),
    }


def finite_positive(value: float, label: str) -> float:
    if not math.isfinite(value) or value <= 0:
        raise ContractDefect(f"{label} is not finite positive")
    return value


def parse_positive_decimal(value: str, label: str) -> float:
    if re.fullmatch(r"(?:0|[1-9][0-9]*)(?:\.[0-9]+)?", value) is None:
        raise ContractDefect(f"{label} is not canonical unsigned decimal")
    try:
        parsed = float(value)
    except ValueError as error:
        raise ContractDefect(f"{label} cannot be parsed") from error
    return finite_positive(parsed, label)


def parse_bench(stderr: str, prompt: str) -> dict[str, object]:
    expected_header = (
        f"[bench] model={MODEL} "
        f"prompt={json.dumps(prompt, ensure_ascii=False)} "
        f"({EXPECTED_PROMPT_TOKENS} tokens), "
        f"gen={EXPECTED_TRANSITIONS} tokens, kv_capacity=1024"
    )
    headers = re.findall(r"^\[bench\] model=.*$", stderr, re.MULTILINE)
    if headers != [expected_header]:
        raise ContractDefect("loaded benchmark shape drifted")
    for line in (
        f"[bench] {EXPECTED_DEVICE}",
        "[bench] === results (1 run) ===",
        "[bench] decode mode: full-logits",
        "[bench] prefill mode: packed layer-major",
        "[bench] prefill chunk: 1024",
    ):
        if stderr.splitlines().count(line) != 1:
            raise ContractDefect(f"missing unique benchmark protocol line: {line}")
    matches = re.findall(
        r"^\[bench\] rep\s+(\d+): prefill\s+([0-9]+(?:\.[0-9]+)?) ms "
        r"\(([0-9]+(?:\.[0-9]+)?) t/s\)\s+decode\s+"
        r"([0-9]+(?:\.[0-9]+)?) ms \(([0-9]+(?:\.[0-9]+)?) t/s\)$",
        stderr,
        re.MULTILINE,
    )
    if len(matches) != 1 or matches[0][0] != "1":
        raise ContractDefect("timed repetition protocol drifted")
    _, prefill_ms, prefill_tps, decode_ms, decode_tps = matches[0]
    prefill = parse_positive_decimal(prefill_ms, "prefill_ms")
    decode = parse_positive_decimal(decode_ms, "decode_ms")
    reported_prefill = parse_positive_decimal(prefill_tps, "prefill_tps")
    reported_decode = parse_positive_decimal(decode_tps, "decode_tps")
    if abs(EXPECTED_PROMPT_TOKENS * 1000.0 / prefill - reported_prefill) > 1.0:
        raise ContractDefect("prefill throughput does not reconcile")
    if abs(EXPECTED_TRANSITIONS * 1000.0 / decode - reported_decode) > 0.2:
        raise ContractDefect("decode throughput does not reconcile")
    requests = re.findall(
        r"^\[bench\] rep\s+1 request\s+([0-9]+(?:\.[0-9]+)?) ms$",
        stderr,
        re.MULTILINE,
    )
    averages = re.findall(
        r"^\[bench\] request wall: ([0-9]+(?:\.[0-9]+)?) ms avg$",
        stderr,
        re.MULTILINE,
    )
    if len(requests) != 1 or len(averages) != 1:
        raise ContractDefect("request-wall protocol drifted")
    request = parse_positive_decimal(requests[0], "request_ms")
    average = parse_positive_decimal(averages[0], "request_average_ms")
    if abs(request - average) > 0.11 or request + 0.2 < prefill + decode:
        raise ContractDefect("request wall does not reconcile")
    trace_lines = re.findall(
        r"^\[bench\] generated token trace: (\[.*\])$", stderr, re.MULTILINE
    )
    token_protocol_lines = [
        line for line in stderr.splitlines() if "[bench] generated token" in line
    ]
    if len(trace_lines) != 1 or token_protocol_lines != [
        f"[bench] generated token trace: {trace_lines[0]}"
    ]:
        raise ContractDefect("exact generated-token-trace line drifted")
    tokens = parse_json(trace_lines[0], "generated token trace")
    if (
        not isinstance(tokens, list)
        or len(tokens) != EXPECTED_TRACE_TOKENS
        or any(
            type(value) is not int or value < 0 or value >= EXPECTED_VOCAB_SIZE
            for value in tokens
        )
    ):
        raise ContractDefect("generated token trace is not the exact canonical shape")
    canonical_tokens = json_text(tokens)
    if trace_lines[0] != canonical_tokens:
        raise ContractDefect("generated token trace is not compact canonical JSON")
    generated = re.findall(r"^\[bench\] generated: (.*)$", stderr, re.MULTILINE)
    if len(generated) != 1:
        raise ContractDefect("decoded generated-text line drifted")
    return {
        "prefill_ms": prefill,
        "decode_ms": decode,
        "request_ms": request,
        "prefill_tps_reported": reported_prefill,
        "decode_tps_reported": reported_decode,
        "token_trace": tokens,
        "token_trace_sha256": hashlib.sha256(canonical_tokens.encode()).hexdigest(),
        "generated_debug": generated[0],
        "generated_debug_sha256": hashlib.sha256(generated[0].encode()).hexdigest(),
    }


def parse_time_resource(stderr: str, label: str) -> int:
    values = re.findall(rf"^\s*(\d+)\s+{re.escape(label)}$", stderr, re.MULTILINE)
    if len(values) != 1:
        raise ContractDefect(f"/usr/bin/time label drifted: {label}")
    return int(values[0])


def seconds_to_ms(value: str, label: str) -> int:
    try:
        milliseconds = Decimal(value) * 1000
    except InvalidOperation as error:
        raise ContractDefect(f"invalid /usr/bin/time {label}") from error
    if milliseconds != milliseconds.to_integral_value():
        raise ContractDefect(f"non-integral /usr/bin/time {label}")
    return int(milliseconds)


def process_resources(stderr: str) -> dict[str, int]:
    summaries = re.findall(
        r"^\s*([0-9]+\.[0-9]{2}) real\s+([0-9]+\.[0-9]{2}) user\s+"
        r"([0-9]+\.[0-9]{2}) sys$",
        stderr,
        re.MULTILINE,
    )
    if len(summaries) != 1:
        raise ContractDefect("/usr/bin/time summary drifted")
    real, user, system = summaries[0]
    result = {
        "real_ms": seconds_to_ms(real, "real"),
        "user_cpu_ms": seconds_to_ms(user, "user"),
        "system_cpu_ms": seconds_to_ms(system, "system"),
    }
    result["total_cpu_ms"] = result["user_cpu_ms"] + result["system_cpu_ms"]
    for label in (
        "maximum resident set size",
        "page reclaims",
        "page faults",
        "swaps",
        "block input operations",
        "block output operations",
        "instructions retired",
        "cycles elapsed",
        "peak memory footprint",
    ):
        result[label.replace(" ", "_")] = parse_time_resource(stderr, label)
    return result


def verify_tool_dialects(child_env: dict[str, str]) -> dict[str, object]:
    result = subprocess.run(
        ["/usr/bin/time", "-l", "/usr/bin/true"],
        cwd=ROOT,
        env=child_env,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or result.stdout:
        raise ContractDefect("/usr/bin/time preflight command drifted")
    resources = process_resources(result.stderr)
    host = capture_host_state()
    vm = capture_vm_state()
    if host.get("valid") is not True:
        raise InconclusivePacket("preflight", None, ["host_invalid_at_preflight"])
    if vm.get("errors"):
        raise ContractDefect("vm_stat or swapusage dialect drifted")
    return {
        "time_resources": resources,
        "host": host,
        "vm_keys": {
            key: vm[key]
            for key in (
                "pageouts",
                "compressions",
                "swapouts",
                "compressor_stored_pages",
                "compressor_occupied_pages",
                "swap_used_bytes",
            )
        },
    }


def cargo_test_command(package: str, test: str, *, ignored: bool) -> list[str]:
    command = ["cargo", "test", "--release", "-p", package, test, "--"]
    if ignored:
        command.append("--ignored")
    command.extend(["--exact", "--nocapture", "--test-threads=1"])
    return command


def recognize_test(text: str, test: str, *, interleaved: bool) -> None:
    escaped = re.escape(test)
    starts = list(re.finditer(rf"^test {escaped} \.\.\. ", text, re.MULTILINE))
    summaries = list(
        re.finditer(
            r"^test result: ok\. 1 passed; 0 failed; 0 ignored; 0 measured; "
            r"\d+ filtered out; finished in [0-9]+(?:\.[0-9]+)?s\r?$",
            text,
            re.MULTILINE,
        )
    )
    contiguous = list(
        re.finditer(rf"^test {escaped} \.\.\. ok\r?$", text, re.MULTILINE)
    )
    if (
        len(starts) != 1
        or len(summaries) != 1
        or starts[0].start() >= summaries[0].start()
    ):
        raise ContractDefect(f"Cargo test identity drifted: {test}")
    if not interleaved:
        if len(contiguous) != 1:
            raise ContractDefect(f"CPU test result drifted: {test}")
        return
    standalone = list(re.finditer(r"^ok\r?$", text, re.MULTILINE))
    valid = (len(contiguous) == 1 and not standalone) or (
        not contiguous
        and len(standalone) == 1
        and starts[0].end() <= standalone[0].start() < summaries[0].start()
    )
    if not valid:
        raise ContractDefect(f"interleaved correctness result drifted: {test}")


def run_test(
    base_env: dict[str, str],
    package: str,
    test: str,
    output_name: str,
    *,
    ignored: bool,
) -> dict[str, object]:
    path = ARTIFACT / output_name
    command = cargo_test_command(package, test, ignored=ignored)
    started = time.perf_counter_ns()
    result: subprocess.CompletedProcess[bytes] | None = None
    execution_error: Exception | None = None
    durability_error: OSError | None = None
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
            except Exception as error:  # noqa: BLE001 - classify test launch failures
                execution_error = error
            try:
                output.flush()
                os.fsync(output.fileno())
            except OSError as error:
                durability_error = error
    except OSError as error:
        durability_error = error
    if durability_error is not None:
        raise UnsealedPacket(f"test output durability failed: {durability_error}")
    if execution_error is not None:
        raise ContractDefect(
            f"Cargo test launch failed: {type(execution_error).__name__}:{execution_error}"
        ) from execution_error
    if result is None:
        raise ContractDefect("Cargo test returned no process result")
    text = path.read_text(encoding="utf-8")
    if result.returncode != 0:
        raise ContractDefect(f"Cargo test failed: {test}")
    recognize_test(text, test, interleaved=ignored)
    return {
        "schema": 1,
        "test": test,
        "command": command,
        "returncode": result.returncode,
        "wall_ns": time.perf_counter_ns() - started,
        "output_sha256": sha256(path),
        "passed": True,
    }


def run_correctness(base_env: dict[str, str]) -> dict[str, object]:
    row = run_test(
        base_env,
        "qwen-llm",
        CORRECTNESS_TEST,
        "correctness.out",
        ignored=True,
    )
    text = (ARTIFACT / "correctness.out").read_text(encoding="utf-8")
    prefixes = (
        "[metal-load] native quantized token embedding policy:",
        "[metal-gguf-parallel-pread]",
        "[metal-load-ledger]",
    )
    recognized: list[str] = []
    harness_prefix = f"test {CORRECTNESS_TEST} ... "
    for line in text.splitlines():
        if not recognized and line == harness_prefix + POLICY_LINE:
            recognized.append(POLICY_LINE)
        elif line.startswith(prefixes):
            recognized.append(line)
        elif "[metal-load" in line or "[metal-gguf-" in line:
            raise ContractDefect(f"malformed correctness load line: {line!r}")
    if (
        len(recognized) != 5
        or recognized[:3] != [POLICY_LINE, LEDGER_LINE, POLICY_LINE]
        or recognized[4] != LEDGER_LINE
    ):
        raise ContractDefect(f"correctness A/B load protocol drifted: {recognized!r}")
    marker = parse_marker(recognized[3])
    row.update({"recognized_load_lines": recognized, "candidate_marker": marker})
    write_json(ARTIFACT / "correctness.json", row)
    return row


def run_token_protocol_check(base_env: dict[str, str]) -> dict[str, object]:
    command = [str(BENCH_BINARY), "decode", "--help"]
    started = time.perf_counter_ns()
    result = subprocess.run(
        command,
        cwd=ROOT,
        env=base_env,
        check=False,
        capture_output=True,
    )
    if result.returncode != 0 or result.stderr:
        raise ContractDefect("qwen-bench decode help protocol failed")
    output = result.stdout.decode("utf-8")
    if (
        output.count("--generated-token-trace") != 1
        or output.count(
            "Print the exact initial token plus every timed transition result"
        )
        != 1
    ):
        raise ContractDefect("generated-token-trace help protocol drifted")
    path = ARTIFACT / "token-protocol.out"
    write_fsynced(path, result.stdout)
    row = {
        "schema": 1,
        "check": "qwen-bench-decode-help-generated-token-trace",
        "command": command,
        "returncode": result.returncode,
        "wall_ns": time.perf_counter_ns() - started,
        "output_sha256": sha256(path),
        "passed": True,
    }
    write_json(ARTIFACT / "token-protocol.json", row)
    return row


def record_launch(
    stem: str,
    arm: str,
    quartet_index: int,
    order: str,
    position: int,
    command: list[str],
    environment_delta: dict[str, str],
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
            "quartet_index": quartet_index,
            "quartet_order": order,
            "position": position,
            "command": command,
            "environment_delta": environment_delta,
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


def wait_for_child_process(
    process: subprocess.Popen[bytes],
) -> tuple[int | None, list[str], bool]:
    errors: list[str] = []
    while True:
        try:
            return process.wait(), errors, True
        except InterruptedError:
            continue
        except Exception as error:  # noqa: BLE001 - exact-PID fallback below
            errors.append(f"wait={type(error).__name__}:{error}")
            break
    while True:
        try:
            pid, status = os.waitpid(process.pid, 0)
            if pid != process.pid:
                errors.append(f"waitpid_returned_unexpected_pid={pid}")
                return None, errors, False
            returncode = os.waitstatus_to_exitcode(status)
            process.returncode = returncode
            return returncode, errors, True
        except InterruptedError:
            continue
        except ChildProcessError as error:
            errors.append(f"waitpid={type(error).__name__}:{error}")
            try:
                returncode = process.poll()
            except Exception as poll_error:  # noqa: BLE001 - lifecycle evidence
                errors.append(f"poll={type(poll_error).__name__}:{poll_error}")
                return None, errors, False
            return returncode, errors, returncode is not None
        except Exception as error:  # noqa: BLE001 - lifecycle evidence
            errors.append(f"waitpid={type(error).__name__}:{error}")
            return None, errors, False


def run_child(
    base_env: dict[str, str],
    manifest: dict[str, object],
    prompt: str,
    quartet_index: int,
    order: str,
    position: int,
    arm: str,
) -> dict[str, object]:
    stem = f"loaded-q{quartet_index:02d}-{order.lower()}-r{position}-{arm.lower()}"
    stdout_path = ARTIFACT / f"{stem}.out"
    stderr_path = ARTIFACT / f"{stem}.err"
    conditioning = condition_child(stem, manifest)
    env, delta = arm_environment(base_env, arm)
    command = child_command(prompt)
    started_wall = time.time_ns()
    started_mono = time.monotonic_ns()
    process: subprocess.Popen[bytes] | None = None
    returncode: int | None = None
    spawn_error: Exception | None = None
    spawn_started_mono: int | None = None
    spawn_completed_mono: int | None = None
    wait_errors: list[str] = []
    lifecycle_reaped = True
    durability_error: OSError | None = None
    launch_recorded = False
    deferred_signals: list[int] = []

    def defer_sigint(signum: int, _frame: object) -> None:
        deferred_signals.append(signum)

    prior = signal.signal(signal.SIGINT, defer_sigint)
    try:
        try:
            with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
                record_launch(
                    stem,
                    arm,
                    quartet_index,
                    order,
                    position,
                    command,
                    delta,
                    manifest,
                )
                launch_recorded = True
                try:
                    spawn_started_mono = time.monotonic_ns()
                    process = subprocess.Popen(
                        command,
                        cwd=ROOT,
                        env=env,
                        stdout=stdout,
                        stderr=stderr,
                    )
                except Exception as error:  # noqa: BLE001 - seal spawn failure evidence
                    spawn_error = error
                finally:
                    spawn_completed_mono = time.monotonic_ns()
                if process is not None:
                    returncode, wait_errors, lifecycle_reaped = wait_for_child_process(
                        process
                    )
                try:
                    stdout.flush()
                    os.fsync(stdout.fileno())
                    stderr.flush()
                    os.fsync(stderr.fileno())
                except OSError as error:
                    durability_error = error
        except OSError as error:
            durability_error = error
    finally:
        signal.signal(signal.SIGINT, prior)
        if launch_recorded:
            errors = []
            if spawn_error is not None:
                errors.append(f"{type(spawn_error).__name__}:{spawn_error}")
            errors.extend(wait_errors)
            if not lifecycle_reaped:
                errors.append("child_lifecycle_not_reaped")
            if durability_error is not None:
                errors.append(f"{type(durability_error).__name__}:{durability_error}")
            record_completion(
                stem,
                returncode,
                ";".join(errors) if errors else None,
            )
    ended_mono = time.monotonic_ns()
    ended_wall = time.time_ns()
    if durability_error is not None:
        raise UnsealedPacket(f"child raw durability failed: {stem}: {durability_error}")
    if not lifecycle_reaped:
        raise UnsealedPacket(f"child lifecycle could not be reaped exactly: {stem}")
    if spawn_started_mono is None or spawn_completed_mono is None:
        raise UnsealedPacket(f"child spawn bracket was not reached: {stem}")
    host_after = capture_host_state()
    vm_after = capture_vm_state()
    interval = vm_interval("child", conditioning["interval"]["after"], vm_after)
    base_row = {
        "schema": 1,
        "artifact_stem": stem,
        "arm": arm,
        "quartet_index": quartet_index,
        "quartet_order": order,
        "position": position,
        "started_unix_ns": started_wall,
        "ended_unix_ns": ended_wall,
        "conditioning_monotonic_ns": conditioning["monotonic_ns"],
        "spawn_started_monotonic_ns": spawn_started_mono,
        "spawn_completed_monotonic_ns": spawn_completed_mono,
        "started_monotonic_ns": started_mono,
        "ended_monotonic_ns": ended_mono,
        "process_wall_ns": ended_mono - started_mono,
        "spawn_to_exit_ns": ended_mono - spawn_completed_mono,
        "command": command,
        "environment_delta": delta,
        "returncode": returncode,
        "stdout_sha256": sha256(stdout_path),
        "stderr_sha256": sha256(stderr_path),
        "conditioning_sha256": sha256(ARTIFACT / f"{stem}.conditioning.json"),
    }
    if deferred_signals or spawn_error is not None or wait_errors or returncode != 0:
        reasons = list(interval["failure_reasons"])
        if deferred_signals:
            reasons.append(f"operator_sigint_deferred={len(deferred_signals)}")
        if spawn_error is not None:
            reasons.append(
                f"child_spawn_failed={type(spawn_error).__name__}:{spawn_error}"
            )
        if wait_errors:
            reasons.extend(f"child_{error}" for error in wait_errors)
        if spawn_error is None and returncode != 0:
            reasons.append(f"child_returncode={returncode}")
        post = {
            "schema": 1,
            "artifact_stem": stem,
            "host_after_exit": host_after,
            "vm_after_exit": vm_after,
            "interval": interval,
            "process_resources": None,
            "validity_reasons": reasons,
        }
        post_path = ARTIFACT / f"{stem}.post-exit.json"
        write_json(post_path, post)
        row = {
            **base_row,
            "post_exit_sha256": sha256(post_path),
            "load_contract": None,
            "bench": None,
            "process_resources": None,
            "valid": False,
            "validity_reasons": reasons,
            "failure_class": "invalid-child",
        }
        append_jsonl(ARTIFACT / "attempts.jsonl", row)
        raise InconclusivePacket("loaded", stem, reasons)

    parse_error: ContractDefect | None = None
    load: dict[str, object] | None = None
    bench: dict[str, object] | None = None
    resources: dict[str, int] | None = None
    try:
        stdout = stdout_path.read_bytes()
        stderr = stderr_path.read_text(encoding="utf-8")
        if stdout:
            raise ContractDefect(f"qwen-bench emitted unexpected stdout: {stem}")
        load = parse_load_contract(stderr, arm)
        bench = parse_bench(stderr, prompt)
        resources = process_resources(stderr)
    except ContractDefect as error:
        parse_error = error
    except Exception as error:  # noqa: BLE001 - normalize child parser failures
        parse_error = ContractDefect(
            f"child parser failed: {type(error).__name__}:{error}"
        )
    if parse_error is not None:
        post = {
            "schema": 1,
            "artifact_stem": stem,
            "host_after_exit": host_after,
            "vm_after_exit": vm_after,
            "interval": interval,
            "process_resources": resources,
            "validity_reasons": [],
            "contract_error": str(parse_error),
        }
        post_path = ARTIFACT / f"{stem}.post-exit.json"
        write_json(post_path, post)
        row = {
            **base_row,
            "post_exit_sha256": sha256(post_path),
            "load_contract": load,
            "bench": bench,
            "process_resources": resources,
            "valid": False,
            "validity_reasons": [],
            "failure_class": "implementation_or_contract_defect",
            "contract_error": str(parse_error),
        }
        append_jsonl(ARTIFACT / "attempts.jsonl", row)
        raise ChildContractDefect(stem, str(parse_error))

    assert load is not None and bench is not None and resources is not None
    reasons = list(interval["failure_reasons"])
    if host_after.get("valid") is not True:
        reasons.append("host_invalid_after_child")
    if resources["block_input_operations"] != 0:
        reasons.append("child_block_input")
    if resources["page_faults"] != 0:
        reasons.append("child_major_faults")
    if resources["swaps"] != 0:
        reasons.append("child_swaps")
    post = {
        "schema": 1,
        "artifact_stem": stem,
        "host_after_exit": host_after,
        "vm_after_exit": vm_after,
        "interval": interval,
        "process_resources": resources,
        "validity_reasons": reasons,
    }
    post_path = ARTIFACT / f"{stem}.post-exit.json"
    write_json(post_path, post)
    row = {
        **base_row,
        "post_exit_sha256": sha256(post_path),
        "load_contract": load,
        "bench": bench,
        "process_resources": resources,
        "valid": not reasons,
        "validity_reasons": reasons,
        "failure_class": None,
    }
    append_jsonl(ARTIFACT / "attempts.jsonl", row)
    if reasons:
        raise InconclusivePacket("loaded", stem, reasons)
    return row


def geometric_pair(left: float, right: float, label: str) -> float:
    finite_positive(left, label)
    finite_positive(right, label)
    return math.sqrt(left * right)


def ratio(numerator: float, denominator: float, label: str) -> float:
    finite_positive(numerator, f"{label}.numerator")
    finite_positive(denominator, f"{label}.denominator")
    value = numerator / denominator
    if not math.isfinite(value) or value <= 0:
        raise ContractDefect(f"invalid ratio: {label}")
    return value


def stratum_median(rows: list[dict[str, object]], order: str, field: str) -> float:
    values = [float(row[field]) for row in rows if row["quartet_order"] == order]
    if len(values) != 4:
        raise ContractDefect(f"{order} stratum membership drifted for {field}")
    return statistics.median(values)


def relative_range(values: list[float], label: str) -> float:
    if not values:
        raise ContractDefect(f"empty stability series: {label}")
    for value in values:
        finite_positive(value, label)
    return (max(values) - min(values)) / statistics.median(values)


def analyze(rows: list[dict[str, object]]) -> dict[str, object]:
    if len(rows) != 32:
        raise ContractDefect("complete analysis requires exactly 32 children")
    token_arrays = [row["bench"]["token_trace"] for row in rows]
    if any(not strict_equal(tokens, token_arrays[0]) for tokens in token_arrays[1:]):
        raise ContractDefect("generated token-ID arrays differ")
    debug_hashes = {row["bench"]["generated_debug_sha256"] for row in rows}
    if len(debug_hashes) != 1:
        raise ContractDefect("decoded generated-text lines differ")
    quartets: list[dict[str, object]] = []
    for index, expected_order in enumerate(QUARTET_ORDERS, 1):
        selected = [row for row in rows if row["quartet_index"] == index]
        selected.sort(key=lambda row: row["position"])
        if (
            len(selected) != 4
            or "".join(row["arm"] for row in selected) != expected_order
            or any(row["quartet_order"] != expected_order for row in selected)
        ):
            raise ContractDefect(f"quartet {index} membership drifted")
        arms = {
            arm: [row for row in selected if row["arm"] == arm] for arm in ("A", "B")
        }
        metrics: dict[str, float] = {}
        for field in ("prefill_ms", "decode_ms", "request_ms"):
            a = geometric_pair(
                float(arms["A"][0]["bench"][field]),
                float(arms["A"][1]["bench"][field]),
                f"q{index}.{field}.A",
            )
            b = geometric_pair(
                float(arms["B"][0]["bench"][field]),
                float(arms["B"][1]["bench"][field]),
                f"q{index}.{field}.B",
            )
            metrics[f"{field}_gm_a"] = a
            metrics[f"{field}_gm_b"] = b
        complete_cpu_a = geometric_pair(
            *(float(row["process_resources"]["total_cpu_ms"]) for row in arms["A"]),
            f"q{index}.cpu.A",
        )
        complete_cpu_b = geometric_pair(
            *(float(row["process_resources"]["total_cpu_ms"]) for row in arms["B"]),
            f"q{index}.cpu.B",
        )
        rss_a = max(
            row["process_resources"]["maximum_resident_set_size"] for row in arms["A"]
        )
        rss_b = max(
            row["process_resources"]["maximum_resident_set_size"] for row in arms["B"]
        )
        footprint_a = max(
            row["process_resources"]["peak_memory_footprint"] for row in arms["A"]
        )
        footprint_b = max(
            row["process_resources"]["peak_memory_footprint"] for row in arms["B"]
        )
        quartet = {
            "quartet_index": index,
            "quartet_order": expected_order,
            "prefill_b_over_a": ratio(
                metrics["prefill_ms_gm_b"], metrics["prefill_ms_gm_a"], f"q{index}.P"
            ),
            "decode_b_over_a": ratio(
                metrics["decode_ms_gm_b"], metrics["decode_ms_gm_a"], f"q{index}.D"
            ),
            "request_b_over_a": ratio(
                metrics["request_ms_gm_b"], metrics["request_ms_gm_a"], f"q{index}.R"
            ),
            "complete_cpu_b_over_a": ratio(
                complete_cpu_b, complete_cpu_a, f"q{index}.CPU"
            ),
            "rss_b_over_a": ratio(float(rss_b), float(rss_a), f"q{index}.RSS"),
            "footprint_b_over_a": ratio(
                float(footprint_b), float(footprint_a), f"q{index}.footprint"
            ),
            "arm_geometric_means": metrics,
            "child_stems": [row["artifact_stem"] for row in selected],
            "child_spawn_completed_monotonic_ns": [
                row["spawn_completed_monotonic_ns"] for row in selected
            ],
            "child_ended_monotonic_ns": [row["ended_monotonic_ns"] for row in selected],
            "simple_wins": {
                "prefill": metrics["prefill_ms_gm_b"] <= metrics["prefill_ms_gm_a"],
                "decode": metrics["decode_ms_gm_b"] <= metrics["decode_ms_gm_a"],
                "request": metrics["request_ms_gm_b"] <= metrics["request_ms_gm_a"],
            },
        }
        quartet["memory_gates"] = {
            "rss": quartet["rss_b_over_a"] <= 1.05,
            "footprint": quartet["footprint_b_over_a"] <= 1.05,
        }
        quartet["performance_gates"] = {
            "prefill": quartet["prefill_b_over_a"] <= 1.01,
            "decode": quartet["decode_b_over_a"] <= 1.01,
            "request": quartet["request_b_over_a"] <= 1.01,
            "complete_cpu": quartet["complete_cpu_b_over_a"] <= 1.10,
            **quartet["memory_gates"],
        }
        quartet["passes"] = all(quartet["performance_gates"].values())
        quartets.append(quartet)
    medians: dict[str, dict[str, float]] = {}
    for field in (
        "prefill_b_over_a",
        "decode_b_over_a",
        "request_b_over_a",
        "complete_cpu_b_over_a",
        "rss_b_over_a",
        "footprint_b_over_a",
    ):
        medians[field] = {
            "all": statistics.median(float(row[field]) for row in quartets),
            "ABBA": stratum_median(quartets, "ABBA", field),
            "BAAB": stratum_median(quartets, "BAAB", field),
        }
    median_gates = {
        "prefill_all": medians["prefill_b_over_a"]["all"] <= 1.01,
        "prefill_abba": medians["prefill_b_over_a"]["ABBA"] <= 1.01,
        "prefill_baab": medians["prefill_b_over_a"]["BAAB"] <= 1.01,
        "decode_all": medians["decode_b_over_a"]["all"] <= 1.01,
        "decode_abba": medians["decode_b_over_a"]["ABBA"] <= 1.01,
        "decode_baab": medians["decode_b_over_a"]["BAAB"] <= 1.01,
        "request_all": medians["request_b_over_a"]["all"] <= 1.01,
        "request_abba": medians["request_b_over_a"]["ABBA"] <= 1.01,
        "request_baab": medians["request_b_over_a"]["BAAB"] <= 1.01,
        "cpu_all": medians["complete_cpu_b_over_a"]["all"] <= 1.10,
        "cpu_abba": medians["complete_cpu_b_over_a"]["ABBA"] <= 1.10,
        "cpu_baab": medians["complete_cpu_b_over_a"]["BAAB"] <= 1.10,
        "rss_all": medians["rss_b_over_a"]["all"] <= 1.05,
        "rss_abba": medians["rss_b_over_a"]["ABBA"] <= 1.05,
        "rss_baab": medians["rss_b_over_a"]["BAAB"] <= 1.05,
        "footprint_all": medians["footprint_b_over_a"]["all"] <= 1.05,
        "footprint_abba": medians["footprint_b_over_a"]["ABBA"] <= 1.05,
        "footprint_baab": medians["footprint_b_over_a"]["BAAB"] <= 1.05,
    }
    start_ns = min(row["spawn_completed_monotonic_ns"] for row in rows)
    end_ns = max(row["ended_monotonic_ns"] for row in rows)
    consecutive_start_ns = [
        right["spawn_completed_monotonic_ns"] - left["spawn_completed_monotonic_ns"]
        for left, right in pairwise(rows)
    ]
    conditioning_to_start_ns = [
        row["spawn_completed_monotonic_ns"] - row["conditioning_monotonic_ns"]
        for row in rows
    ]
    quartet_span_ns = []
    for index in range(8):
        group = rows[index * 4 : index * 4 + 4]
        quartet_span_ns.append(
            group[-1]["ended_monotonic_ns"] - group[0]["spawn_completed_monotonic_ns"]
        )
    reversal_span_ns = []
    for index in range(4):
        group = rows[index * 8 : index * 8 + 8]
        reversal_span_ns.append(
            group[-1]["ended_monotonic_ns"] - group[0]["spawn_completed_monotonic_ns"]
        )
    nearest_cross_arm_idle_ns = []
    for left, right in pairwise(rows):
        if left["arm"] != right["arm"]:
            nearest_cross_arm_idle_ns.append(
                right["spawn_completed_monotonic_ns"] - left["ended_monotonic_ns"]
            )
    temporal_gates = {
        "conditioning_to_start": max(conditioning_to_start_ns)
        <= MAX_CONDITIONING_TO_LAUNCH_NS,
        "consecutive_start": max(consecutive_start_ns) <= MAX_CONSECUTIVE_START_NS,
        "quartet_span": max(quartet_span_ns) <= MAX_QUARTET_SPAN_NS,
        "reversal_span": max(reversal_span_ns) <= MAX_REVERSAL_SPAN_NS,
    }
    trajectories: dict[str, dict[str, object]] = {}
    trajectory_gates: dict[str, bool] = {}
    for arm in ("A", "B"):
        selected = [row for row in rows if row["arm"] == arm]
        if len(selected) != 16:
            raise ContractDefect(f"{arm} trajectory membership drifted")
        for field in ("prefill_ms", "decode_ms", "request_ms"):
            values = [float(row["bench"][field]) for row in selected]
            key = f"{arm}.{field}"
            spread = relative_range(values, key)
            trajectories[key] = {
                "values": values,
                "minimum": min(values),
                "median": statistics.median(values),
                "maximum": max(values),
                "relative_range": spread,
                "last_over_first": ratio(values[-1], values[0], f"{key}.last_first"),
            }
            trajectory_gates[key] = spread <= MAX_ARM_RELATIVE_RANGE
    stability_gates = {**temporal_gates, **trajectory_gates}
    every_quartet_passes = all(row["passes"] for row in quartets)
    performance_gates = {
        **median_gates,
        "every_quartet": every_quartet_passes,
    }

    miss_specs = {
        "prefill": ("prefill_b_over_a", 1.01),
        "decode": ("decode_b_over_a", 1.01),
        "request": ("request_b_over_a", 1.01),
        "complete_cpu": ("complete_cpu_b_over_a", 1.10),
        "rss": ("rss_b_over_a", 1.05),
        "footprint": ("footprint_b_over_a", 1.05),
    }
    consistent_misses: dict[str, dict[str, object]] = {}
    for name, (field, threshold) in miss_specs.items():
        count = sum(float(row[field]) > threshold for row in quartets)
        medians_miss = all(value > threshold for value in medians[field].values())
        consistent_misses[name] = {
            "quartets_over_limit": count,
            "aggregate_and_strata_over_limit": medians_miss,
            "kill_consistent": count >= 7 and medians_miss,
        }
    stable = all(stability_gates.values())
    performance_passes = all(performance_gates.values())
    kill_consistent = any(row["kill_consistent"] for row in consistent_misses.values())
    if not stable:
        classification = "inconclusive-instability"
    elif performance_passes:
        classification = "go"
    elif kill_consistent:
        classification = "kill"
    else:
        classification = "inconclusive-heterogeneous"
    return {
        "schema": 1,
        "stage": "loaded-stability",
        "quartets": quartets,
        "medians": medians,
        "stability": {
            "gates": stability_gates,
            "passes": stable,
            "trajectories": trajectories,
        },
        "performance": {
            "gates": performance_gates,
            "passes": performance_passes,
            "consistent_misses": consistent_misses,
        },
        "classification": classification,
        "token_trace": token_arrays[0],
        "token_trace_sha256": rows[0]["bench"]["token_trace_sha256"],
        "generated_debug_sha256": next(iter(debug_hashes)),
        "simple_win_counts": {
            field: sum(bool(row["simple_wins"][field]) for row in quartets)
            for field in ("prefill", "decode", "request")
        },
        "timing_diagnostics": {
            "complete_stage_wall_s": (end_ns - start_ns) / 1e9,
            "conditioning_to_start_s": [
                value / 1e9 for value in conditioning_to_start_ns
            ],
            "consecutive_start_s": [value / 1e9 for value in consecutive_start_ns],
            "quartet_span_s": [value / 1e9 for value in quartet_span_ns],
            "reversal_span_s": [value / 1e9 for value in reversal_span_ns],
            "nearest_cross_arm_idle_gap_s": [
                value / 1e9 for value in nearest_cross_arm_idle_ns
            ],
            "child_wall_s": [row["spawn_to_exit_ns"] / 1e9 for row in rows],
            "child_midpoint_monotonic_ns": [
                (row["spawn_completed_monotonic_ns"] + row["ended_monotonic_ns"]) // 2
                for row in rows
            ],
            "first_child": rows[0]["artifact_stem"],
            "last_child": rows[-1]["artifact_stem"],
        },
        "claim": "narrow engineering admission, not confidence-interval noninferiority",
    }


def expected_children() -> list[tuple[int, str, int, str, str]]:
    rows = []
    for quartet_index, order in enumerate(QUARTET_ORDERS, 1):
        for position, arm in enumerate(order, 1):
            stem = (
                f"loaded-q{quartet_index:02d}-{order.lower()}-r{position}-{arm.lower()}"
            )
            rows.append((quartet_index, order, position, arm, stem))
    return rows


def read_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        return []
    rows = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        value = parse_json(line, f"{path.name}:{index}")
        if not isinstance(value, dict):
            raise ContractDefect(f"{path.name}:{index} is not an object")
        rows.append(value)
    return rows


def verify_attempt_artifacts(rows: list[dict[str, object]]) -> None:
    expected = expected_children()
    if len(rows) > len(expected):
        raise ContractDefect("attempt ledger exceeds frozen child list")
    for row, (quartet, order, position, arm, stem) in zip(
        rows, expected[: len(rows)], strict=True
    ):
        if (
            row.get("artifact_stem") != stem
            or row.get("quartet_index") != quartet
            or row.get("quartet_order") != order
            or row.get("position") != position
            or row.get("arm") != arm
        ):
            raise ContractDefect("attempt order or identity drifted")
        for suffix, field in (
            ("out", "stdout_sha256"),
            ("err", "stderr_sha256"),
            ("conditioning.json", "conditioning_sha256"),
            ("post-exit.json", "post_exit_sha256"),
        ):
            path = ARTIFACT / f"{stem}.{suffix}"
            if not path.is_file() or sha256(path) != row.get(field):
                raise ContractDefect(
                    f"attempt raw artifact binding drifted: {path.name}"
                )
    events = read_jsonl(ARTIFACT / "launch-seal.jsonl")
    has_golden = bool(
        rows and rows[0].get("valid") is True and isinstance(rows[0].get("bench"), dict)
    )
    expected_events = 2 * len(rows) + int(has_golden)
    if len(events) != expected_events:
        raise ContractDefect("launch/completion/golden event count drifted")
    cursor = 0
    for index, row in enumerate(rows):
        launch = events[cursor]
        completion = events[cursor + 1]
        cursor += 2
        if (
            launch.get("event") != "launch"
            or launch.get("artifact_stem") != row["artifact_stem"]
        ):
            raise ContractDefect("launch event order drifted")
        if (
            completion.get("event") != "completion"
            or completion.get("artifact_stem") != row["artifact_stem"]
            or completion.get("returncode") != row["returncode"]
        ):
            raise ContractDefect("completion event order drifted")
        if index == 0 and has_golden:
            golden = events[cursor]
            cursor += 1
            if (
                golden.get("event") != "golden-token-trace"
                or golden.get("source_artifact_stem") != row["artifact_stem"]
                or not strict_equal(
                    golden.get("token_trace"), row["bench"]["token_trace"]
                )
            ):
                raise ContractDefect("golden token event drifted")
    if cursor != len(events):
        raise ContractDefect("unconsumed launch-seal events")


def final_identity(manifest: dict[str, object]) -> dict[str, object]:
    result = {
        "schema": 1,
        "model_file_identity": None,
        "model_sha256": None,
        "expected_file_identity": manifest["model_file_identity"],
        "expected_sha256": manifest["sha256"][str(MODEL)],
        "matches": False,
        "source": None,
        "error": None,
        "unix_ms": time.time_ns() // 1_000_000,
    }
    try:
        verify_non_model_identity(manifest)
        identity = file_identity(MODEL)
        digest = sha256(MODEL)
        source = verify_source_topology()
        result.update(
            {
                "model_file_identity": identity,
                "model_sha256": digest,
                "matches": identity == manifest["model_file_identity"]
                and digest == manifest["sha256"][str(MODEL)],
                "source": source,
            }
        )
    except UnsealedPacket:
        raise
    except Exception as error:
        result["error"] = f"{type(error).__name__}:{error}"
        write_json(ARTIFACT / "final-identity.json", result)
        raise ContractDefect(f"final identity failed: {result['error']}") from error
    write_json(ARTIFACT / "final-identity.json", result)
    if result["matches"] is not True:
        raise ContractDefect("final model identity drifted")
    return result


def make_decision(
    manifest: dict[str, object],
    correctness: dict[str, object] | None,
    token_protocol: dict[str, object] | None,
    stage: dict[str, object] | None,
    status: str,
    stopped_after: str,
    failed_child: str | None,
    reasons: list[str],
) -> dict[str, object]:
    if status not in VALID_STATUSES:
        raise ContractDefect(f"invalid decision status: {status}")
    attempts_path = ARTIFACT / "attempts.jsonl"
    attempts_digest = sha256(attempts_path) if attempts_path.is_file() else None
    go = status == "go"
    return {
        "schema": 1,
        "status": status,
        "authority": ("explicit-force-only-dense27b-direct-pread" if go else "none"),
        "force_authorized": go,
        "successor_authorization": (
            "separate-dense-auto-selector-preregistration-only" if go else "none"
        ),
        "stopped_after": stopped_after,
        "failed_child": failed_child,
        "reasons": reasons,
        "source_commit": manifest["source_commit"],
        "imported_v0629": manifest["imported_v0629"],
        "correctness": correctness,
        "token_protocol": token_protocol,
        "stage": stage,
        "attempts_sha256": attempts_digest,
        "claim_scope": {
            "profile": "dense27b-q4km-v1",
            "selection": "explicit-QWEN_GGUF_PARALLEL_COPY=pread",
            "load_intent": "ForceOnly/reusable structural profile",
            "auto_default_authorized": False,
            "performance_imports_used_for_scoring": 0,
        },
    }


def publish(decision: dict[str, object]) -> None:
    rows = read_jsonl(ARTIFACT / "attempts.jsonl")
    verify_attempt_artifacts(rows)
    decision_path = ARTIFACT / "decision.json"
    write_json(decision_path, decision)
    candidates = sorted(
        path
        for path in ARTIFACT.iterdir()
        if path.name not in {"artifact-inventory.sha256", "packet-complete.json"}
    )
    if any(not path.is_file() or path.is_symlink() for path in candidates):
        raise UnsealedPacket("artifact directory contains a non-regular member")
    inventory_path = ARTIFACT / "artifact-inventory.sha256"
    inventory_lines = [
        f"{sha256(path)}  {path.relative_to(ROOT)}\n" for path in candidates
    ]
    write_fsynced(inventory_path, "".join(inventory_lines).encode("utf-8"))
    complete = {
        "schema": 1,
        "decision_sha256": sha256(decision_path),
        "inventory_sha256": sha256(inventory_path),
    }
    write_json(ARTIFACT / "packet-complete.json", complete)
    fsync_directory(ARTIFACT)
    final_set = set(ARTIFACT.iterdir())
    expected_set = set(candidates) | {inventory_path, ARTIFACT / "packet-complete.json"}
    if final_set != expected_set:
        raise UnsealedPacket("final artifact membership changed during publication")


def seal_preflight_failure(
    status: str, stage: str, failed_child: str | None, reasons: list[str]
) -> None:
    failure = {
        "schema": 1,
        "status": status,
        "stage": stage,
        "failed_child": failed_child,
        "reasons": reasons,
    }
    write_json(ARTIFACT / "preflight-failure.json", failure)
    decision = {
        "schema": 1,
        "status": status,
        "authority": "none",
        "force_authorized": False,
        "successor_authorization": "none",
        "stopped_after": stage,
        "failed_child": failed_child,
        "reasons": reasons,
        "source_commit": None,
        "imported_v0629": None,
        "correctness": None,
        "token_protocol": None,
        "stage": None,
        "attempts_sha256": sha256(ARTIFACT / "attempts.jsonl"),
        "claim_scope": None,
    }
    publish(decision)
    print(json_text(decision, pretty=True))


def run_packet(*, preflight_only: bool) -> None:
    if ARTIFACT.exists():
        raise RuntimeError(f"refusing to reuse artifact directory: {ARTIFACT}")
    if preflight_only:
        child_env, test_env, removed = safe_environments()
        manifest = build_manifest(child_env, test_env, removed)
        prompt = PROMPT.read_text(encoding="utf-8")
        if len(prompt.encode("utf-8")) != EXPECTED_PROMPT_BYTES:
            raise ContractDefect("prompt text decode changed byte count")
        print(
            json_text(
                {
                    "status": "preflight_pass",
                    "source_commit": manifest["source_commit"],
                    "quartets": len(QUARTET_ORDERS),
                    "children": len(expected_children()),
                    "model_sha256": manifest["sha256"][str(MODEL)],
                },
                pretty=True,
            )
        )
        return
    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT.mkdir()
    fsync_directory(ARTIFACT.parent)
    write_fsynced(ARTIFACT / "attempts.jsonl", b"")
    write_fsynced(ARTIFACT / "launch-seal.jsonl", b"")
    fsync_directory(ARTIFACT)
    try:
        child_env, test_env, removed = safe_environments()
        manifest = build_manifest(child_env, test_env, removed)
        prompt = PROMPT.read_text(encoding="utf-8")
        if len(prompt.encode("utf-8")) != EXPECTED_PROMPT_BYTES:
            raise ContractDefect("prompt text decode changed byte count")
    except InconclusivePacket as error:
        seal_preflight_failure("inconclusive", error.stage, error.child, error.reasons)
        return
    except ContractDefect as error:
        seal_preflight_failure(
            "implementation_or_contract_defect", "preflight", None, [str(error)]
        )
        return
    except UnsealedPacket:
        raise
    except Exception as error:  # noqa: BLE001 - seal classifiable preflight failures
        seal_preflight_failure(
            "implementation_or_contract_defect",
            "preflight",
            None,
            [f"{type(error).__name__}:{error}"],
        )
        return
    write_json(ARTIFACT / "manifest.json", manifest)
    fsync_directory(ARTIFACT)
    correctness: dict[str, object] | None = None
    token_protocol: dict[str, object] | None = None
    stage: dict[str, object] | None = None
    status = "implementation_or_contract_defect"
    stopped_after = "preflight"
    failed_child: str | None = None
    reasons: list[str] = []
    rows: list[dict[str, object]] = []
    try:
        stopped_after = "token-protocol"
        token_protocol = run_token_protocol_check(child_env)
        stopped_after = "correctness"
        correctness = run_correctness(test_env)
        stopped_after = "loaded-stability"
        golden: list[int] | None = None
        for quartet_index, order in enumerate(QUARTET_ORDERS, 1):
            for position, arm in enumerate(order, 1):
                row = run_child(
                    child_env,
                    manifest,
                    prompt,
                    quartet_index,
                    order,
                    position,
                    arm,
                )
                rows.append(row)
                if golden is None:
                    if arm != "A" or quartet_index != 1 or position != 1:
                        raise ChildContractDefect(
                            row["artifact_stem"],
                            "first valid child is not frozen A golden",
                        )
                    golden = row["bench"]["token_trace"]
                    record_golden(row)
                elif not strict_equal(row["bench"]["token_trace"], golden):
                    raise ChildContractDefect(
                        row["artifact_stem"],
                        f"timed token identity differs: {row['artifact_stem']}",
                    )
        stage = analyze(rows)
        stopped_after = "loaded-stability"
        classification = stage["classification"]
        if classification == "go":
            status = "go"
        elif classification == "kill":
            status = "kill"
        else:
            status = "inconclusive"
            reasons = [classification]
    except InconclusivePacket as error:
        status = "inconclusive"
        stopped_after = error.stage
        failed_child = error.child
        reasons = error.reasons
    except ChildContractDefect as error:
        status = "implementation_or_contract_defect"
        failed_child = error.child
        reasons = [str(error)]
    except ContractDefect as error:
        status = "implementation_or_contract_defect"
        failed_child = rows[-1]["artifact_stem"] if rows else None
        reasons = [str(error)]
    except UnsealedPacket:
        raise
    except Exception as error:  # noqa: BLE001 - seal classifiable packet failures
        status = "implementation_or_contract_defect"
        failed_child = rows[-1]["artifact_stem"] if rows else None
        reasons = [f"{type(error).__name__}:{error}"]
    try:
        final_identity(manifest)
    except ContractDefect as error:
        status = "implementation_or_contract_defect"
        stopped_after = "final-identity"
        failed_child = None
        reasons = [str(error)]
    except UnsealedPacket:
        raise
    except Exception as error:  # noqa: BLE001 - seal classifiable identity failures
        status = "implementation_or_contract_defect"
        stopped_after = "final-identity"
        failed_child = None
        reasons = [f"{type(error).__name__}:{error}"]
    decision = make_decision(
        manifest,
        correctness,
        token_protocol,
        stage,
        status,
        stopped_after,
        failed_child,
        reasons,
    )
    publish(decision)
    print(json_text(decision, pretty=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    run_packet(preflight_only=args.preflight_only)


if __name__ == "__main__":
    main()
