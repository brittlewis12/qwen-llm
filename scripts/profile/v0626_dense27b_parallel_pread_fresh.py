#!/usr/bin/env python3
"""v0.626 dense-27B direct-pread cache-warm fresh-effect packet."""

import argparse
import ctypes
import hashlib
import json
import mmap
import os
from pathlib import Path
import re
import signal
import subprocess
import time

import v0605_dense27b_parallel_copied_loader as mechanics


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = ROOT / "target/profiles/v0626-dense27b-parallel-pread-fresh-p1"
PREREG = ROOT / "docs/bench/v0626-dense27b-parallel-pread-fresh.md"
MECHANICS = ROOT / "scripts/profile/v0605_dense27b_parallel_copied_loader.py"
COMMON_RUNNER = ROOT / "scripts/profile/v0593_demand_paged_no_copy.py"
MODEL = Path("/Users/tito/models/Qwen3.6-27B-Q4_K_M.gguf")
PROMPT = ROOT / "docs/bench/tokenizer-prompts/current-reva-n8-interactive-qwen36.txt"
CLI_BINARY = ROOT / "target/release/qwen"
BENCH_BINARY = ROOT / "target/release/qwen-bench"
V0603_ROOT = ROOT / "target/profiles/v0603-dense27b-parallel-copied-floor-p1"
V0603_DECISION = V0603_ROOT / "decision.json"
V0603_INVENTORY = V0603_ROOT / "artifact-inventory.sha256"
V0603_COMPLETE = V0603_ROOT / "packet-complete.json"
V0593_ROOT = ROOT / "target/profiles/v0593-demand-paged-no-copy-p1"
V0593_MANIFEST = V0593_ROOT / "manifest.json"
V0593_ATTEMPTS = V0593_ROOT / "attempts.jsonl"
V0593_PREREG = ROOT / "docs/bench/v0593-demand-paged-no-copy.md"
V0593_RUNNER = ROOT / "scripts/profile/v0593_demand_paged_no_copy.py"

BASE_COMMIT = "129379f1c1918e30ab4725856fe9699a7f54d0ed"
V0603_SOURCE_COMMIT = "e6e964ffad896a480ac280de8ff07634cdd344bd"
EXPECTED_MODEL_SHA256 = (
    "5ed60d0af4650a854b1755bd392f9aef4872643dc25a254bc68043fa638392a0"
)
EXPECTED_PROMPT_SHA256 = (
    "e265de9742d1b22e566fc108ae26331ccf46166c6f071e73f211e0a1a7e8b474"
)
EXPECTED_STDOUT_SHA256 = (
    "c94cb4d2661b181e91ab5db8c65ff252fcd97064073ff332d60715052b784cd0"
)
EXPECTED_PREREG_SHA256 = (
    "090c33f68ad57cd809a2b3edebee8da8e4de0bc3cd69f77524f652fe233d9e16"
)
EXPECTED_V0603 = {
    V0603_DECISION: "a0361b55d93cee769fc8d4db44eecdf83f3d1c63bdda5ecec07fd3d2edd279ac",
    V0603_INVENTORY: "fa87cf2788d58b2f988ce07760ad3619bad835f4abb77dedd238b3dd563e3a9e",
    V0603_COMPLETE: "bcb19bc3f8776fe16e6459abb5930e020d89f2bd8cf20e3a31a7e567959282ab",
}
EXPECTED_V0593_STATIC = {
    V0593_MANIFEST: "378393cacc1ad4e840fc5a6f905c1d463200f8fd5a08a7e09bfab6e621757a51",
    V0593_ATTEMPTS: "59cbd3e5ae95d93dd337b379c25090e623c5c51462c2b2ded036b47a12adca70",
    V0593_PREREG: "5b316a595f1949cebb564182ab48a12b3798c007e1aea26a1f06dba482b273b3",
    V0593_RUNNER: "4c1b73a2897d8bf882988407f45b59be7b19d236c06a612991491c57af0d7be4",
}
EXPECTED_V0593_TIMINGS = {
    "n128-p01-a01-r1-b": "d41202ea26cfa5f96460c92a3097bfbca0366893d5c226c5f1a7b3ef1a22b18f",
    "n128-p01-a01-r2-a": "8516122874ff8fb27e3bdc3fc8c8dddc75958dc0005821712d2377dfc8856709",
    "n128-p02-a01-r1-a": "33d2f44d940e74a70194389453bb804d33669cc5ec4f3eda8527ed4b988a2453",
    "n128-p02-a01-r2-b": "c8b44790094f53b3f352f77bb281ef99ac3fbb055bb8f9b2a89148f54bcc5db3",
}
EXPECTED_MODEL_SIZE = 16_817_244_384
EXPECTED_DEVICE = (
    "device: Apple M4 Max | unified_memory=true | max_threadgroup_memory=32768 bytes"
)
EXPECTED_HW_MEMSIZE = 137_438_953_472
PAIR_ORDERS = ("AB", "BA", "BA", "AB", "AB", "BA")
CPU_SELECTOR_TESTS = (
    "metal_forward::tests::gguf_parallel_pread_capability_table_is_exact",
    "metal_forward::tests::gguf_parallel_pread_dense_marker_table_is_exact",
    "metal_forward::tests::gguf_parallel_pread_dense_auto_remains_none",
    "metal_forward::tests::gguf_parallel_pread_dense_advice_preserves_configured_policy",
)
CORRECTNESS_TEST = "metal_forward::tests::gguf_parallel_pread_dense27b_q4_is_bit_exact"
VALID_DECISION_STATUSES = {
    "implementation_or_contract_defect",
    "inconclusive",
    "kill",
    "fresh_effect_pass",
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
PREFETCH_SKIP_RE = re.compile(
    r"^.*INFO qwen_llm::runtime: prefetch: skipped \(already warm\) "
    + rf"shard={re.escape(str(MODEL))} resident_pct=100\.0$",
    re.MULTILINE,
)

common = mechanics.common
BASE_PARSE_MARKER = mechanics.parse_marker
BASE_VALIDATE_FRESH_TIMING = mechanics.validate_fresh_timing
BASE_RECORD_LAUNCH = mechanics.record_launch
PENDING_CONDITIONING: dict[str, dict[str, object]] = {}
for name, value in {
    "ARTIFACT": ARTIFACT,
    "PREREG": PREREG,
    "MODEL": MODEL,
    "PROMPT": PROMPT,
    "CLI_BINARY": CLI_BINARY,
    "BENCH_BINARY": BENCH_BINARY,
    "PAIR_ORDERS": PAIR_ORDERS,
    "POLICY_LINE": POLICY_LINE,
    "LEDGER_LINE": LEDGER_LINE,
    "MARKER_PREFIX": MARKER_PREFIX,
    "CORRECTNESS_HARNESS_PREFIX": f"test {CORRECTNESS_TEST} ... ",
}.items():
    setattr(mechanics, name, value)
common.MODEL = MODEL


def command_text(command: list[str], env: dict[str, str] | None = None) -> str:
    return mechanics.command_text(command, env)


def parse_json(text: str) -> object:
    return mechanics.parse_json(text)


def json_text(value: object, *, pretty: bool = False) -> str:
    return mechanics.json_text(value, pretty=pretty)


def sha256(path: Path) -> str:
    return common.sha256_file(path)


def inventory_members(path: Path, label: str) -> dict[Path, str]:
    members: dict[Path, str] = {}
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        fields = line.split("  ", 1)
        if len(fields) != 2 or re.fullmatch(r"[0-9a-f]{64}", fields[0]) is None:
            raise RuntimeError(f"{label} inventory row {line_number} is malformed")
        relative = Path(fields[1])
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError(f"{label} inventory row {line_number} escapes root")
        member = ROOT / relative
        if member in members or not member.is_file():
            raise RuntimeError(
                f"{label} inventory member is missing or duplicate: {member}"
            )
        if sha256(member) != fields[0]:
            raise RuntimeError(f"{label} inventory member drifted: {member}")
        members[member] = fields[0]
    return members


def verify_v0603_floor() -> dict[Path, str]:
    for path, expected in EXPECTED_V0603.items():
        if sha256(path) != expected:
            raise RuntimeError(f"sealed v0.603 evidence drifted: {path}")
    complete = parse_json(V0603_COMPLETE.read_text(encoding="utf-8"))
    if complete != {
        "schema": 1,
        "decision_sha256": EXPECTED_V0603[V0603_DECISION],
        "inventory_sha256": EXPECTED_V0603[V0603_INVENTORY],
    }:
        raise RuntimeError("v0.603 completion seal drifted")
    members = inventory_members(V0603_INVENTORY, "v0.603")
    if members.get(V0603_DECISION) != EXPECTED_V0603[V0603_DECISION]:
        raise RuntimeError("v0.603 inventory does not bind its decision")
    if len(members) != 29:
        raise RuntimeError("v0.603 sealed inventory member count drifted")
    decision = parse_json(V0603_DECISION.read_text(encoding="utf-8"))
    if not isinstance(decision, dict):
        raise RuntimeError("v0.603 decision is not an object")
    expected = {
        "schema": 1,
        "status": "go",
        "authority": "implement-force-only-dense27b-loader-pilot",
        "source_commit": V0603_SOURCE_COMMIT,
        "completed_blocks": 6,
    }
    if any(decision.get(key) != value for key, value in expected.items()):
        raise RuntimeError("v0.603 floor decision identity drifted")
    result = decision.get("result")
    if (
        not isinstance(result, dict)
        or result.get("wins") != 6
        or result.get("median_saving_ms") != 965.0405000000001
        or result.get("passes") is not True
        or not isinstance(result.get("gates"), dict)
        or not all(result["gates"].values())
    ):
        raise RuntimeError("v0.603 floor result drifted")
    return members


def verify_v0593_stdout_golden() -> dict[Path, str]:
    members = dict(EXPECTED_V0593_STATIC)
    for stem, digest in EXPECTED_V0593_TIMINGS.items():
        members[V0593_ROOT / f"{stem}.timing.jsonl"] = digest
        members[V0593_ROOT / f"{stem}.out"] = EXPECTED_STDOUT_SHA256
    for path, expected in members.items():
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"retrospective v0.593 golden evidence drifted: {path}")

    manifest = parse_json(V0593_MANIFEST.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise RuntimeError("v0.593 golden manifest is malformed")
    build = manifest.get("build_identity")
    expected_manifest_hashes = {
        str(MODEL): EXPECTED_MODEL_SHA256,
        str(PROMPT): EXPECTED_PROMPT_SHA256,
        str(V0593_PREREG): EXPECTED_V0593_STATIC[V0593_PREREG],
        str(V0593_RUNNER): EXPECTED_V0593_STATIC[V0593_RUNNER],
    }
    if (
        not isinstance(build, dict)
        or build.get("status") != "match"
        or build.get("build_commit") != manifest.get("source_commit")
        or build.get("runtime_commit") != manifest.get("source_commit")
        or build.get("build_dirty") is not False
        or build.get("runtime_dirty") is not False
        or build.get("build_source_state") != build.get("runtime_source_state")
        or not isinstance(manifest.get("sha256"), dict)
        or any(
            manifest["sha256"].get(path) != digest
            for path, digest in expected_manifest_hashes.items()
        )
    ):
        raise RuntimeError("v0.593 golden manifest identity drifted")

    rows = [
        parse_json(line)
        for line in V0593_ATTEMPTS.read_text(encoding="utf-8").splitlines()
    ]
    selected_rows = [
        row
        for row in rows
        if isinstance(row, dict) and row.get("artifact_stem") in EXPECTED_V0593_TIMINGS
    ]
    selected = {row.get("artifact_stem"): row for row in selected_rows}
    if len(selected_rows) != 4 or set(selected) != set(EXPECTED_V0593_TIMINGS):
        raise RuntimeError("v0.593 golden attempt set drifted")
    if sorted(row.get("arm") for row in selected.values()) != ["A", "A", "B", "B"]:
        raise RuntimeError("v0.593 golden arm balance drifted")
    for stem, row in selected.items():
        timing_path = V0593_ROOT / f"{stem}.timing.jsonl"
        timing_rows = [
            parse_json(line)
            for line in timing_path.read_text(encoding="utf-8").splitlines()
            if line
        ]
        if (
            row.get("valid") is not True
            or row.get("validity_reasons") != []
            or row.get("tokens") != 128
            or row.get("stdout_sha256") != EXPECTED_STDOUT_SHA256
            or len(timing_rows) != 1
            or timing_rows[0] != row.get("timing")
            or not isinstance(timing_rows[0], dict)
            or timing_rows[0].get("model") != str(MODEL)
            or timing_rows[0].get("prompt_tokens") != 419
            or timing_rows[0].get("requested_tokens") != 128
            or timing_rows[0].get("generated_tokens") != 128
            or timing_rows[0].get("transition_count") != 127
            or timing_rows[0].get("decode_policy") != "greedy_argmax"
        ):
            raise RuntimeError(f"v0.593 golden row drifted: {stem}")
    return members


def git_changed_paths(revision: str) -> list[str]:
    text = command_text(["git", "diff", "--name-only", revision])
    return [line for line in text.splitlines() if line]


def validate_build_identity(
    value: object, commit: str, label: str
) -> dict[str, object]:
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} build identity is not an object")
    if (
        value.get("build_commit") != commit
        or value.get("runtime_commit") != commit
        or value.get("status") != "match"
        or value.get("build_dirty") is not False
        or value.get("runtime_dirty") is not False
        or value.get("build_source_state") != value.get("runtime_source_state")
    ):
        raise RuntimeError(f"{label} source/build/runtime identity mismatch: {value}")
    return value


def source_and_build_identity() -> tuple[str, dict[str, object]]:
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError("runner working directory is not repository root")
    for path in (Path(__file__).resolve(), PREREG, MECHANICS, COMMON_RUNNER, PROMPT):
        command_text(
            ["git", "ls-files", "--error-unmatch", str(path.relative_to(ROOT))]
        )
    if command_text(["git", "status", "--porcelain=v1"]).strip():
        raise RuntimeError("source worktree is dirty")
    commit = command_text(["git", "rev-parse", "HEAD"]).strip()
    if command_text(["git", "rev-parse", "HEAD^^"]).strip() != BASE_COMMIT:
        raise RuntimeError("implementation HEAD is not two commits above frozen main")
    packet_paths = sorted(git_changed_paths(f"{BASE_COMMIT}..HEAD^"))
    if packet_paths != sorted(
        [str(PREREG.relative_to(ROOT)), str(Path(__file__).resolve().relative_to(ROOT))]
    ):
        raise RuntimeError(f"packet commit path boundary drifted: {packet_paths}")
    implementation_paths = git_changed_paths("HEAD^..HEAD")
    if implementation_paths != ["crates/qwen-llm/src/metal_forward.rs"]:
        raise RuntimeError(
            f"implementation commit path boundary drifted: {implementation_paths}"
        )
    bench = validate_build_identity(
        parse_json(command_text([str(BENCH_BINARY), "build-info", "--output", "json"])),
        commit,
        "qwen-bench",
    )
    qwen_bytes = CLI_BINARY.read_bytes()
    embedded = {
        "commit": commit,
        "build_source_state": str(bench["build_source_state"]),
    }
    if any(value.encode() not in qwen_bytes for value in embedded.values()):
        raise RuntimeError("release qwen identity is not embedded")
    return commit, {
        **bench,
        "qwen_identity": {
            "sha256": hashlib.sha256(qwen_bytes).hexdigest(),
            "embedded": embedded,
            "verified": True,
        },
        "qwen_bench_sha256": sha256(BENCH_BINARY),
    }


def required_manifest_paths(
    v0603_members: dict[Path, str], v0593_members: dict[Path, str]
) -> tuple[Path, ...]:
    return tuple(
        dict.fromkeys(
            (
                Path(__file__).resolve(),
                PREREG,
                MECHANICS,
                COMMON_RUNNER,
                MODEL,
                PROMPT,
                CLI_BINARY,
                BENCH_BINARY,
                V0603_DECISION,
                V0603_INVENTORY,
                V0603_COMPLETE,
                *v0603_members,
                *v0593_members,
            )
        )
    )


def build_manifest(
    removed_environment: list[str], base_env: dict[str, str]
) -> dict[str, object]:
    v0603_members = verify_v0603_floor()
    v0593_members = verify_v0593_stdout_golden()
    commit, build = source_and_build_identity()
    paths = required_manifest_paths(v0603_members, v0593_members)
    if any(not path.is_file() for path in paths):
        raise RuntimeError("a required packet input is missing")
    hashes = {str(path): sha256(path) for path in paths}
    if (
        hashes[str(MODEL)] != EXPECTED_MODEL_SHA256
        or MODEL.stat().st_size != EXPECTED_MODEL_SIZE
    ):
        raise RuntimeError("model identity drifted")
    if hashes[str(PROMPT)] != EXPECTED_PROMPT_SHA256 or PROMPT.stat().st_size != 1_891:
        raise RuntimeError("prompt identity drifted")
    if hashes[str(PREREG)] != EXPECTED_PREREG_SHA256:
        raise RuntimeError("preregistration document drifted")
    device = command_text([str(CLI_BINARY), "--info"], env=base_env).strip()
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
    return {
        "schema": 1,
        "protocol": "v0.626-dense27b-parallel-pread-fresh",
        "created_unix_ms": time.time_ns() // 1_000_000,
        "source_commit": commit,
        "base_commit": BASE_COMMIT,
        "build_identity": build,
        "device": device,
        "macos_product_version": product,
        "macos_build_version": os_build,
        "hw_memsize": memory,
        "removed_environment": removed_environment,
        "child_environment": mechanics.child_environment_record(base_env),
        "time_resource_probe": mechanics.preflight_time_resources(base_env),
        "sha256": hashes,
        "v0603_inventory_members": {
            str(path): digest for path, digest in v0603_members.items()
        },
        "v0593_stdout_golden_members": {
            str(path): digest for path, digest in v0593_members.items()
        },
        "model_size_bytes": EXPECTED_MODEL_SIZE,
        "model_file_identity": model_file_identity(),
        "prompt_bytes": 1_891,
        "prompt_tokens": 419,
        "output_tokens": 128,
        "transition_count": 127,
        "expected_stdout_sha256": EXPECTED_STDOUT_SHA256,
        "time_page_fault_semantics": {
            "page_reclaims": "minor_page_faults",
            "page_faults": "major_page_faults",
        },
        "pair_orders": list(PAIR_ORDERS),
        "stages": ["cpu-selectors", "full-state-correctness", "fresh-128"],
        "child_retry_count": 0,
        "loaded_runs": 0,
        "target_file_cold_stages": 0,
        "authority": "none",
    }


def verify_non_model_identity(manifest: dict[str, object]) -> None:
    commit, build = source_and_build_identity()
    if commit != manifest.get("source_commit") or build != manifest.get(
        "build_identity"
    ):
        raise RuntimeError("source/build identity changed during packet")
    expected = manifest.get("sha256")
    if not isinstance(expected, dict):
        raise RuntimeError("manifest hashes are malformed")
    actual = {path: sha256(Path(path)) for path in expected if path != str(MODEL)}
    expected_non_model = {
        path: digest for path, digest in expected.items() if path != str(MODEL)
    }
    if actual != expected_non_model:
        raise RuntimeError("non-model packet input changed")
    if command_text(["sw_vers", "-productVersion"]).strip() != manifest.get(
        "macos_product_version"
    ) or command_text(["sw_vers", "-buildVersion"]).strip() != manifest.get(
        "macos_build_version"
    ):
        raise RuntimeError("OS identity changed during packet")


def verify_packet_identity(manifest: dict[str, object]) -> None:
    verify_non_model_identity(manifest)
    if sha256(MODEL) != manifest["sha256"][str(MODEL)]:
        raise RuntimeError("model changed during packet")


def arm_environment(arm: str) -> dict[str, str | None]:
    if arm not in ("A", "B"):
        raise ValueError(f"unknown arm {arm!r}")
    return {"QWEN_GGUF_PARALLEL_COPY": "pread" if arm == "B" else "0"}


def environment_for_arm(base_env: dict[str, str], arm: str) -> dict[str, str]:
    env = base_env.copy()
    for key, value in arm_environment(arm).items():
        if value is not None:
            env[key] = value
    return env


def file_identity(stat: os.stat_result) -> dict[str, int]:
    return {
        "device": stat.st_dev,
        "inode": stat.st_ino,
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def model_file_identity() -> dict[str, int]:
    return file_identity(MODEL.stat())


def probe_model_residency(expected_identity: object) -> dict[str, object]:
    if not isinstance(expected_identity, dict):
        raise RuntimeError("manifest model file identity is malformed")
    size = expected_identity.get("size_bytes")
    if type(size) is not int or size != EXPECTED_MODEL_SIZE:
        raise RuntimeError("manifest model size identity is malformed")
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
    descriptor_primary_error = None
    try:
        before = file_identity(os.fstat(descriptor))
        path_before = model_file_identity()
        if before != expected_identity or path_before != expected_identity:
            raise RuntimeError("target file identity drifted before residency proof")
        address = libc.mmap(None, size, mmap.PROT_READ, mmap.MAP_SHARED, descriptor, 0)
        failed = ctypes.c_void_p(-1).value
        if address == failed:
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error))
        primary_error = None
        try:
            vector = (ctypes.c_ubyte * page_count)()
            if libc.mincore(address, size, vector) != 0:
                error = ctypes.get_errno()
                raise OSError(error, os.strerror(error))
            resident = sum(1 for value in vector if value & 1)
            after = file_identity(os.fstat(descriptor))
            path_after = model_file_identity()
            if after != expected_identity or path_after != expected_identity:
                raise RuntimeError(
                    "target file identity drifted during residency proof"
                )
        except BaseException as error:
            primary_error = error
            raise
        finally:
            if libc.munmap(address, size) != 0:
                error_number = ctypes.get_errno()
                cleanup_error = OSError(error_number, os.strerror(error_number))
                if primary_error is None:
                    raise cleanup_error
                primary_error.add_note(f"munmap cleanup failed: {cleanup_error}")
    except BaseException as error:
        descriptor_primary_error = error
        raise
    finally:
        try:
            os.close(descriptor)
        except OSError as cleanup_error:
            if descriptor_primary_error is None:
                raise
            descriptor_primary_error.add_note(f"close cleanup failed: {cleanup_error}")
    return {
        "page_size": page_size,
        "total_pages": page_count,
        "resident_pages": resident,
        "resident_fraction": resident / page_count,
        "all_pages_resident": resident == page_count,
        "file_identity": expected_identity,
    }


BASE_CONDITION_FOR_CHILD = mechanics.condition_for_child


def condition_for_child(stem: str, manifest: dict[str, object]) -> dict[str, object]:
    evidence = BASE_CONDITION_FOR_CHILD(stem, manifest)
    try:
        residency = probe_model_residency(manifest.get("model_file_identity"))
        host = mechanics.capture_host_state()
        vm_after = mechanics.capture_vm_state()
        residency_interval = mechanics.vm_interval(
            "residency_proof", evidence["vm_before_spawn"], vm_after
        )
    except Exception as error:
        reasons = [f"target_residency_proof_failed={type(error).__name__}:{error}"]
        mechanics.record_prelaunch_failure(stem, evidence, reasons)
        raise mechanics.InconclusivePacket("prelaunch", stem, reasons) from error
    reasons = list(residency_interval["failure_reasons"])
    if residency["all_pages_resident"] is not True:
        reasons.append("target_file_not_fully_resident")
    if host.get("valid") is not True:
        reasons.append("host_invalid_at_residency_proof")
    evidence.update(
        {
            "pre_spawn_residency": residency,
            "host_before_spawn": host,
            "vm_before_spawn": vm_after,
            "residency_interval": residency_interval,
        }
    )
    if reasons:
        mechanics.record_prelaunch_failure(stem, evidence, reasons)
        raise mechanics.InconclusivePacket("prelaunch", stem, reasons)
    if stem in PENDING_CONDITIONING:
        raise RuntimeError(f"duplicate pending conditioning evidence for {stem}")
    PENDING_CONDITIONING[stem] = evidence
    return evidence


def record_launch(
    stage: str,
    stem: str,
    command: list[str],
    arm: str,
    pair_index: int,
    position: int,
    manifest: dict[str, object],
) -> None:
    if stage != "fresh-128":
        raise RuntimeError(f"v0.626 received an unexpected launch stage: {stage}")
    evidence = PENDING_CONDITIONING.pop(stem, None)
    if evidence is None:
        raise RuntimeError(f"launch lacks pending conditioning evidence: {stem}")
    write_fsynced_json(
        ARTIFACT / f"{stem}.conditioning.json",
        {
            "schema": 1,
            "artifact_stem": stem,
            "evidence": evidence,
        },
    )
    BASE_RECORD_LAUNCH(stage, stem, command, arm, pair_index, position, manifest)


def parse_marker(line: str) -> dict[str, int | float]:
    values = BASE_PARSE_MARKER(line)
    if values["timer_major_faults"] != 0:
        raise RuntimeError("direct-pread marker reports target major faults")
    return values


def validate_prefetch_contract(stderr: str) -> None:
    plain = ANSI_RE.sub("", stderr)
    if "[runtime-prefetch]" in plain or "suppressed" in plain:
        raise RuntimeError("prefetch suppression marker appeared")
    matches = PREFETCH_SKIP_RE.findall(plain)
    if len(matches) != 1:
        raise RuntimeError(
            "default ColdOnly did not skip once at 100% target residency"
        )
    if "prefetch: shard warmed" in plain or "residency probe failed" in plain:
        raise RuntimeError("default ColdOnly attempted target-file warming")


def parse_load_contract(stderr: str, arm: str) -> dict[str, object]:
    validate_prefetch_contract(stderr)
    expected_markers = 1 if arm == "B" else 0
    if stderr.count("[metal-gguf-") != expected_markers:
        raise RuntimeError(f"{arm} total storage-marker count drifted")
    if stderr.count("[metal-gguf-parallel-pread]") != expected_markers:
        raise RuntimeError(f"{arm} direct-pread marker count drifted")
    if "[metal-gguf-parallel-copied]" in stderr:
        raise RuntimeError("copied candidate marker appeared")
    forbidden = ("[metal-gguf-owned]", "[metal-gguf-retained]", "[metal-gguf-no-copy]")
    if any(value in stderr for value in forbidden):
        raise RuntimeError("unrequested storage marker appeared")
    recognized = [
        line
        for line in stderr.splitlines()
        if line.startswith(
            (
                "[metal-load] native quantized token embedding policy:",
                "[metal-gguf-parallel-pread]",
                "[metal-load-ledger]",
            )
        )
    ]
    expected = (
        [POLICY_LINE, LEDGER_LINE] if arm == "A" else [POLICY_LINE, None, LEDGER_LINE]
    )
    if len(recognized) != len(expected):
        raise RuntimeError(f"{arm} load-line count drifted: {recognized!r}")
    if recognized[0] != POLICY_LINE or recognized[-1] != LEDGER_LINE:
        raise RuntimeError(f"{arm} policy or copied ledger drifted")
    if arm == "A":
        return {"storage": "copied", "marker": None}
    timings = parse_marker(recognized[1])
    return {"storage": "parallel-pread", "marker": recognized[1], "phase_us": timings}


def parse_observed_load_contract(stderr: str, arm: str) -> dict[str, object] | None:
    tokens = (
        "[metal-load] native quantized token embedding policy:",
        "[metal-gguf-parallel-pread]",
        "[metal-load-ledger]",
    )
    expected_markers = 1 if arm == "B" else 0
    total_markers = stderr.count("[metal-gguf-")
    pread_markers = stderr.count("[metal-gguf-parallel-pread]")
    if total_markers > expected_markers or total_markers != pread_markers:
        raise RuntimeError("observed storage-marker set contradicts contract")
    marker_lines = [line for line in stderr.splitlines() if "[metal-gguf-" in line]
    if len(marker_lines) != total_markers or any(
        not line.startswith("[metal-gguf-parallel-pread]") for line in marker_lines
    ):
        raise RuntimeError("observed storage marker is not an exact recognized line")
    recognized = [line for line in stderr.splitlines() if line.startswith(tokens)]
    if not recognized:
        return None
    expected = (
        [POLICY_LINE, LEDGER_LINE] if arm == "A" else [POLICY_LINE, None, LEDGER_LINE]
    )
    if len(recognized) > len(expected):
        raise RuntimeError("observed load contract has extra lines")
    for index, line in enumerate(recognized):
        if arm == "B" and index == 1:
            parse_marker(line)
        elif line != expected[index]:
            raise RuntimeError("observed load contract contradicts expected prefix")
    if len(recognized) < len(expected):
        return {"status": "incomplete-valid-prefix", "recognized_lines": recognized}
    return {"status": "complete", "contract": parse_load_contract(stderr, arm)}


def run_test_command(
    command: list[str], output_path: Path, base_env: dict[str, str], test_name: str
) -> dict[str, object]:
    deferred_sigint: list[int] = []

    def defer_sigint(signum: int, _frame: object) -> None:
        deferred_sigint.append(signum)

    prior = signal.signal(signal.SIGINT, defer_sigint)
    started = time.perf_counter()
    result = None
    execution_error = None
    durability_error = None
    try:
        try:
            with output_path.open("xb") as output:
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
    finally:
        signal.signal(signal.SIGINT, prior)
    if durability_error is not None:
        raise mechanics.UnsealedPacket(
            f"test output durability failed: {durability_error}"
        )
    if deferred_sigint:
        raise mechanics.InconclusivePacket(
            "cpu-selectors" if "--ignored" not in command else "correctness",
            test_name,
            [f"operator_sigint_deferred={len(deferred_sigint)}"],
        )
    if execution_error is not None or result is None:
        raise mechanics.UnsealedPacket(
            f"test execution did not return: {execution_error}"
        )
    text = output_path.read_text(encoding="utf-8")
    if (
        result.returncode != 0
        or f"test {test_name} ... ok" not in text
        or "test result: ok. 1 passed; 0 failed;" not in text
        or text.count(f"test {test_name} ... ok") != 1
    ):
        raise RuntimeError(
            f"exact release test failed or was not uniquely selected: {test_name}"
        )
    return {
        "test": test_name,
        "command": command,
        "wall_ms": (time.perf_counter() - started) * 1e3,
        "output_path": str(output_path),
        "output_sha256": sha256(output_path),
        "passed": True,
    }


def cargo_test_command(test_name: str, *, ignored: bool) -> list[str]:
    command = ["cargo", "test", "--release", "-p", "qwen-llm", "--lib", test_name, "--"]
    if ignored:
        command.append("--ignored")
    command.extend(["--exact", "--nocapture", "--test-threads=1"])
    return command


def write_fsynced_json(path: Path, value: object) -> None:
    try:
        with path.open("x", encoding="utf-8") as output:
            output.write(json_text(value, pretty=True) + "\n")
            output.flush()
            os.fsync(output.fileno())
    except OSError as error:
        raise mechanics.UnsealedPacket(
            f"metadata write failed for {path.name}: {error}"
        ) from error


def run_cpu_selectors(base_env: dict[str, str]) -> dict[str, object]:
    rows = []
    for index, test_name in enumerate(CPU_SELECTOR_TESTS, 1):
        rows.append(
            run_test_command(
                cargo_test_command(test_name, ignored=False),
                ARTIFACT / f"cpu-selector-{index:02d}.out",
                base_env,
                test_name,
            )
        )
    result = {"schema": 1, "tests": rows, "passed": True}
    write_fsynced_json(ARTIFACT / "cpu-selectors.json", result)
    return result


def extract_correctness_load_lines(text: str) -> list[str]:
    tokens = (
        "[metal-load] native quantized token embedding policy:",
        "[metal-gguf-parallel-pread]",
        "[metal-load-ledger]",
    )
    recognized = []
    harness_prefix = f"test {CORRECTNESS_TEST} ... "
    for line in text.splitlines():
        if not recognized and line == harness_prefix + POLICY_LINE:
            recognized.append(POLICY_LINE)
        elif line.startswith(tokens):
            recognized.append(line)
        elif "[metal-load" in line or "[metal-gguf-" in line:
            raise RuntimeError(f"malformed correctness load line: {line!r}")
    return recognized


def run_correctness(base_env: dict[str, str]) -> dict[str, object]:
    row = run_test_command(
        cargo_test_command(CORRECTNESS_TEST, ignored=True),
        ARTIFACT / "correctness.out",
        base_env,
        CORRECTNESS_TEST,
    )
    text = (ARTIFACT / "correctness.out").read_text(encoding="utf-8")
    if text.count("[metal-gguf-") != 1:
        raise RuntimeError("correctness total storage-marker count drifted")
    if text.count("[metal-gguf-parallel-pread]") != 1:
        raise RuntimeError("correctness direct-pread marker count drifted")
    if text.count("[metal-load] native quantized token embedding policy:") != 2:
        raise RuntimeError("correctness native policy count drifted")
    if text.count("[metal-load-ledger]") != 2 or "[runtime-prefetch]" in text:
        raise RuntimeError("correctness ledger or suppression contract drifted")
    lines = extract_correctness_load_lines(text)
    if (
        len(lines) != 5
        or lines[:3] != [POLICY_LINE, LEDGER_LINE, POLICY_LINE]
        or lines[4] != LEDGER_LINE
    ):
        raise RuntimeError(f"correctness A/B load order drifted: {lines!r}")
    marker = parse_marker(lines[3])
    result = {**row, "recognized_load_lines": lines, "candidate_phase_us": marker}
    write_fsynced_json(ARTIFACT / "correctness.json", result)
    return result


def validate_fresh_timing(
    timing: dict[str, object], manifest: dict[str, object]
) -> None:
    BASE_VALIDATE_FRESH_TIMING(timing, manifest)
    commit = manifest["source_commit"]
    build = manifest["build_identity"]
    if (
        timing.get("build_commit") != commit
        or timing.get("build_dirty") not in (False, "false", "0")
        or timing.get("build_source_state") != build.get("build_source_state")
    ):
        raise RuntimeError("release qwen source/build/runtime identity drifted")


BASE_RUN_FRESH_CHILD = mechanics.run_fresh_child


def run_fresh_child(*args: object, **kwargs: object) -> dict[str, object]:
    row = BASE_RUN_FRESH_CHILD(*args, **kwargs)
    for field in ("load_contract", "timing"):
        observed = row.get(field)
        if (
            isinstance(observed, dict)
            and observed.get("status") == "failed-child-unparsed"
        ):
            raise RuntimeError(f"failed child has contradictory complete {field}")
    if (
        row.get("returncode") == 0
        and row.get("stdout_sha256") != EXPECTED_STDOUT_SHA256
    ):
        raise RuntimeError("fresh stdout differs from frozen exact-request golden")
    resources = row.get("process_resources")
    if isinstance(resources, dict):
        resources["major_page_faults"] = resources.get("page_faults")
    if row.get("valid") is True:
        marker = (
            row.get("load_contract", {}).get("phase_us")
            if row.get("arm") == "B"
            else None
        )
        if not isinstance(resources, dict):
            raise RuntimeError("fresh process resources are malformed")
        if (
            resources.get("block_input_operations") != 0
            or resources.get("page_faults") != 0
        ):
            raise RuntimeError("target physical I/O or major faults were nonzero")
        if isinstance(marker, dict) and marker.get("timer_major_faults") != 0:
            raise RuntimeError("candidate marker major faults were nonzero")
        residency = row.get("pre_spawn_residency")
        if (
            not isinstance(residency, dict)
            or residency.get("all_pages_resident") is not True
        ):
            raise RuntimeError("fresh row lacks complete target residency proof")
    return row


BASE_ANALYZE_FRESH = mechanics.analyze_fresh


def analyze_fresh(rows: list[dict[str, object]]) -> dict[str, object]:
    result = BASE_ANALYZE_FRESH(rows)
    if result.get("global_output_sha256") != EXPECTED_STDOUT_SHA256:
        raise RuntimeError("fresh analysis stdout golden drifted")
    return result


def expected_child_stems() -> list[str]:
    return [
        f"fresh-p{pair_index:02d}-{order.lower()}-r{position}-{arm.lower()}"
        for pair_index, order in enumerate(PAIR_ORDERS, 1)
        for position, arm in enumerate(order, 1)
    ]


def verify_attempt_artifacts(decision: dict[str, object]) -> dict[Path, str]:
    sealed: dict[Path, str] = {}

    def seal(path: Path) -> None:
        if not path.is_file() or path in sealed:
            raise RuntimeError(
                f"required artifact is missing or duplicate: {path.name}"
            )
        sealed[path] = sha256(path)

    manifest_path = ARTIFACT / "manifest.json"
    seal(manifest_path)
    manifest = parse_json(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise RuntimeError("manifest is malformed")
    cpu = decision.get("cpu_selectors")
    if isinstance(cpu, dict):
        metadata = ARTIFACT / "cpu-selectors.json"
        if parse_json(metadata.read_text(encoding="utf-8")) != cpu:
            raise RuntimeError("CPU selector metadata drifted")
        seal(metadata)
        tests = cpu.get("tests")
        if not isinstance(tests, list) or len(tests) != len(CPU_SELECTOR_TESTS):
            raise RuntimeError("CPU selector set drifted")
        for index, (row, name) in enumerate(
            zip(tests, CPU_SELECTOR_TESTS, strict=True), 1
        ):
            path = ARTIFACT / f"cpu-selector-{index:02d}.out"
            if (
                not isinstance(row, dict)
                or row.get("test") != name
                or row.get("output_sha256") != sha256(path)
            ):
                raise RuntimeError("CPU selector output binding drifted")
            seal(path)
    else:
        for index in range(1, len(CPU_SELECTOR_TESTS) + 1):
            path = ARTIFACT / f"cpu-selector-{index:02d}.out"
            if path.is_file():
                seal(path)
    correctness = decision.get("correctness")
    if isinstance(correctness, dict):
        output = ARTIFACT / "correctness.out"
        metadata = ARTIFACT / "correctness.json"
        if correctness.get("output_sha256") != sha256(output):
            raise RuntimeError("correctness output binding drifted")
        if parse_json(metadata.read_text(encoding="utf-8")) != correctness:
            raise RuntimeError("correctness metadata drifted")
        seal(output)
        seal(metadata)
    else:
        output = ARTIFACT / "correctness.out"
        if output.is_file():
            seal(output)

    attempts_path = ARTIFACT / "attempts.jsonl"
    attempt_rows = []
    if attempts_path.is_file():
        seal(attempts_path)
        for line in attempts_path.read_text(encoding="utf-8").splitlines():
            value = parse_json(line)
            if not isinstance(value, dict):
                raise RuntimeError("attempt row is malformed")
            attempt_rows.append(value)
    launch_path = ARTIFACT / "launch-seal.jsonl"
    events = []
    if launch_path.is_file():
        seal(launch_path)
        for line in launch_path.read_text(encoding="utf-8").splitlines():
            value = parse_json(line)
            if not isinstance(value, dict):
                raise RuntimeError("launch event is malformed")
            events.append(value)
    if len(events) % 2:
        raise RuntimeError("launch/completion ledger is incomplete")
    launches = []
    for index in range(0, len(events), 2):
        launch, completion = events[index : index + 2]
        if launch.get("event") != "launch" or completion.get("event") != "completion":
            raise RuntimeError("launch/completion event order drifted")
        if launch.get("artifact_stem") != completion.get("artifact_stem"):
            raise RuntimeError("launch/completion child identity drifted")
        launches.append((launch, completion))
    stems = expected_child_stems()
    launch_stems = [row[0].get("artifact_stem") for row in launches]
    attempt_stems = [row.get("artifact_stem") for row in attempt_rows]
    if (
        launch_stems != stems[: len(launch_stems)]
        or attempt_stems != launch_stems[: len(attempt_stems)]
    ):
        raise RuntimeError("child execution is not the frozen sole-attempt prefix")
    if len(set(launch_stems)) != len(launch_stems) or len(set(attempt_stems)) != len(
        attempt_stems
    ):
        raise RuntimeError("duplicate child attempt identity")
    if len(launch_stems) - len(attempt_stems) > 1:
        raise RuntimeError("more than one launch lacks an attempt row")
    conditioning_by_stem: dict[str, dict[str, object]] = {}
    for index, (launch, completion) in enumerate(launches):
        order = PAIR_ORDERS[index // 2]
        stem = stems[index]
        arm = order[index % 2]
        conditioning_path = ARTIFACT / f"{stem}.conditioning.json"
        conditioning = parse_json(conditioning_path.read_text(encoding="utf-8"))
        if (
            not isinstance(conditioning, dict)
            or conditioning.get("schema") != 1
            or conditioning.get("artifact_stem") != stem
            or not isinstance(conditioning.get("evidence"), dict)
        ):
            raise RuntimeError(f"conditioning evidence is malformed for {stem}")
        conditioning_by_stem[stem] = conditioning["evidence"]
        seal(conditioning_path)
        command = mechanics.fresh_command(ARTIFACT / f"{stem}.timing.jsonl")
        if (
            launch.get("stage") != "fresh-128"
            or launch.get("pair_order") != order
            or launch.get("pair_index") != index // 2 + 1
            or launch.get("position") != index % 2 + 1
            or launch.get("arm") != arm
            or launch.get("command") != command
            or launch.get("arm_environment") != arm_environment(arm)
            or launch.get("source_commit") != manifest.get("source_commit")
            or launch.get("build_identity") != manifest.get("build_identity")
        ):
            raise RuntimeError(f"launch identity drifted for {stem}")
        if completion.get("stage") != "fresh-128":
            raise RuntimeError("completion stage drifted")
        for suffix in ("out", "err"):
            path = ARTIFACT / f"{stem}.{suffix}"
            if completion.get("returncode") is None:
                if path.is_file():
                    seal(path)
            else:
                seal(path)
        timing_path = ARTIFACT / f"{stem}.timing.jsonl"
        if timing_path.is_file():
            seal(timing_path)
        if completion.get("returncode") is None:
            path = ARTIFACT / f"{stem}.spawn-failure.json"
            seal(path)
        else:
            for suffix in ("post-exit-state.json", "post-exit.json"):
                seal(ARTIFACT / f"{stem}.{suffix}")
    for index, row in enumerate(attempt_rows):
        launch, completion = launches[index]
        stem = stems[index]
        if any(
            row.get(key) != launch.get(key)
            for key in (
                "stage",
                "pair_order",
                "pair_index",
                "position",
                "arm",
                "command",
            )
        ):
            raise RuntimeError(f"attempt metadata drifted for {stem}")
        if row.get("returncode") != completion.get("returncode"):
            raise RuntimeError(f"attempt completion drifted for {stem}")
        if any(
            row.get(key) != value for key, value in conditioning_by_stem[stem].items()
        ):
            raise RuntimeError(f"attempt conditioning binding drifted for {stem}")
        for suffix, key in (
            ("out", "stdout_sha256"),
            ("err", "stderr_sha256"),
            ("timing.jsonl", "timing_sha256"),
        ):
            path = ARTIFACT / f"{stem}.{suffix}"
            if row.get(key) != sealed.get(path):
                raise RuntimeError(f"attempt raw binding drifted for {path.name}")
        if row.get("valid") is True and row.get("timing_sha256") is None:
            raise RuntimeError(f"valid attempt lacks timing evidence for {stem}")

    status = decision.get("status")
    if status not in VALID_DECISION_STATUSES or decision.get("authority") != "none":
        raise RuntimeError("decision status or authority is outside the frozen set")
    if status in ("kill", "fresh_effect_pass") and (
        len(launches) != 12 or len(attempt_rows) != 12
    ):
        raise RuntimeError("performance decision lacks all 12 sole attempts")
    expected_successor = (
        "preregister-and-execute-separate-default-coldonly-target-file-cold-guard-only"
        if status == "fresh_effect_pass"
        else "none"
    )
    if (
        decision.get("force_authorized") is not False
        or decision.get("successor_authorization") != expected_successor
    ):
        raise RuntimeError("decision successor boundary drifted")
    if decision.get("attempts_sha256") != sealed.get(attempts_path):
        raise RuntimeError("decision attempt ledger hash drifted")
    failed = decision.get("failed_child")
    if isinstance(failed, str):
        prelaunch = ARTIFACT / f"{failed}.prelaunch-failure.json"
        spawn = ARTIFACT / f"{failed}.spawn-failure.json"
        if prelaunch.is_file():
            seal(prelaunch)
        elif spawn.is_file() and spawn not in sealed:
            seal(spawn)
    final_model = ARTIFACT / "final-model-sha256.json"
    seal(final_model)
    final = parse_json(final_model.read_text(encoding="utf-8"))
    if not isinstance(final, dict):
        raise RuntimeError("final identity report is malformed")
    expected_hash = manifest["sha256"][str(MODEL)]
    expected_file_identity = manifest["model_file_identity"]
    fixed_identity = (
        final.get("schema") == 1
        and final.get("model") == str(MODEL)
        and final.get("expected_size_bytes") == EXPECTED_MODEL_SIZE
        and final.get("expected_sha256") == expected_hash
        and final.get("expected_file_identity") == expected_file_identity
    )
    identity_evidence_matches = (
        fixed_identity
        and final.get("actual_sha256") == expected_hash
        and final.get("observed_file_identity_before") == expected_file_identity
        and final.get("observed_file_identity_after") == expected_file_identity
        and final.get("non_model_identity_error") is None
        and final.get("model_identity_error") is None
    )
    if final.get("matches") is not identity_evidence_matches:
        raise RuntimeError("final identity match bit is inconsistent")
    exact_match = identity_evidence_matches and final.get("matches") is True
    if not fixed_identity or (
        status != "implementation_or_contract_defect" and not exact_match
    ):
        raise RuntimeError("final identity report contradicts decision")
    if status == "implementation_or_contract_defect" and final.get("matches") is True:
        if not exact_match:
            raise RuntimeError("successful defect-time identity report is inconsistent")

    publication_names = {
        "decision.json",
        "artifact-inventory.sha256",
        "packet-complete.json",
        ".decision.json.tmp",
        ".artifact-inventory.sha256.tmp",
        ".packet-complete.json.tmp",
    }
    unknown = {
        path.name
        for path in ARTIFACT.iterdir()
        if path.is_file() and path not in sealed and path.name not in publication_names
    }
    if unknown:
        raise RuntimeError(f"unverified packet artifacts exist: {sorted(unknown)}")
    return sealed


def record_final_identity_report(manifest: dict[str, object]) -> None:
    report_path = ARTIFACT / "final-model-sha256.json"
    if report_path.exists():
        existing = parse_json(report_path.read_text(encoding="utf-8"))
        if not isinstance(existing, dict) or existing.get("schema") != 1:
            raise mechanics.UnsealedPacket(
                "existing final identity report is malformed"
            )
        if existing.get("matches") is True:
            return
        raise RuntimeError("existing final packet identity report records a mismatch")

    non_model_error = None
    try:
        verify_non_model_identity(manifest)
    except Exception as error:
        non_model_error = f"{type(error).__name__}: {error}"

    observed_before = None
    observed_after = None
    actual_sha256 = None
    model_error = None
    try:
        observed_before = model_file_identity()
        actual_sha256 = sha256(MODEL)
        observed_after = model_file_identity()
    except Exception as error:
        model_error = f"{type(error).__name__}: {error}"

    expected_sha256 = manifest["sha256"][str(MODEL)]
    expected_identity = manifest["model_file_identity"]
    matches = (
        non_model_error is None
        and model_error is None
        and actual_sha256 == expected_sha256
        and observed_before == expected_identity
        and observed_after == expected_identity
    )
    report = {
        "schema": 1,
        "model": str(MODEL),
        "expected_size_bytes": EXPECTED_MODEL_SIZE,
        "expected_sha256": expected_sha256,
        "actual_sha256": actual_sha256,
        "expected_file_identity": expected_identity,
        "observed_file_identity_before": observed_before,
        "observed_file_identity_after": observed_after,
        "non_model_identity_error": non_model_error,
        "model_identity_error": model_error,
        "matches": matches,
    }
    write_fsynced_json(report_path, report)
    if not matches:
        raise RuntimeError("final packet identity did not match the manifest")


def write_identity_checked_decision(
    decision: dict[str, object], manifest: dict[str, object]
) -> None:
    prior = signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        try:
            record_final_identity_report(manifest)
            identity_error = None
        except mechanics.UnsealedPacket:
            raise
        except Exception as error:
            identity_error = f"{type(error).__name__}: {error}"
        if identity_error is not None:
            decision = {
                **decision,
                "reported_status_before_identity_check": decision.get("status"),
                "status": "implementation_or_contract_defect",
                "authority": "none",
                "force_authorized": False,
                "successor_authorization": "none",
                "identity_error": identity_error,
            }
        sealed = verify_attempt_artifacts(decision)
        mechanics.write_decision(decision, sealed)
    finally:
        signal.signal(signal.SIGINT, prior)


def make_decision(
    status: str,
    manifest: dict[str, object],
    cpu_selectors: dict[str, object] | None,
    correctness: dict[str, object] | None,
    stages: dict[str, object],
    stopped_after: str,
    **extra: object,
) -> dict[str, object]:
    if status not in VALID_DECISION_STATUSES:
        raise RuntimeError(f"v0.626 status is not frozen: {status!r}")
    reserved = {
        "schema",
        "status",
        "authority",
        "force_authorized",
        "successor_authorization",
        "stopped_after",
        "source_commit",
        "cpu_selectors",
        "correctness",
        "stages",
        "attempts_sha256",
    }
    collisions = reserved.intersection(extra)
    if collisions:
        raise RuntimeError(
            f"decision extras collide with reserved keys: {sorted(collisions)}"
        )
    return {
        "schema": 1,
        "status": status,
        "authority": "none",
        "force_authorized": False,
        "successor_authorization": (
            "preregister-and-execute-separate-default-coldonly-target-file-cold-guard-only"
            if status == "fresh_effect_pass"
            else "none"
        ),
        "stopped_after": stopped_after,
        "source_commit": manifest["source_commit"],
        "cpu_selectors": cpu_selectors,
        "correctness": correctness,
        "stages": stages,
        "attempts_sha256": mechanics.attempts_sha256(),
        **extra,
    }


def main(*, preflight_only: bool) -> None:
    if ARTIFACT.exists():
        raise RuntimeError(f"refusing to reuse packet directory {ARTIFACT}")
    base_env, removed = common.normalized_environment()
    manifest = build_manifest(removed, base_env)
    if preflight_only:
        vm = mechanics.capture_vm_state()
        if vm["capture_errors"]:
            raise RuntimeError(f"VM preflight capture failed: {vm['capture_errors']}")
        print(
            json_text(
                {
                    "preflight": "passed",
                    "source_commit": manifest["source_commit"],
                },
                pretty=True,
            )
        )
        return
    mechanics.reserve_artifact(manifest)
    attempts_path = ARTIFACT / "attempts.jsonl"
    cpu_selectors = None
    correctness = None
    stages: dict[str, object] = {}
    launched_stage = "cpu-selectors"
    try:
        cpu_selectors = run_cpu_selectors(base_env)
        verify_non_model_identity(manifest)
        launched_stage = "correctness"
        correctness = run_correctness(base_env)
        verify_packet_identity(manifest)
        launched_stage = "fresh-128"
        rows = mechanics.run_stage("fresh-128", base_env, manifest, attempts_path)
        stages["fresh_128"] = mechanics.analyze_fresh(rows)
        status = "fresh_effect_pass" if stages["fresh_128"]["passes"] else "kill"
        write_identity_checked_decision(
            make_decision(
                status, manifest, cpu_selectors, correctness, stages, "fresh-128"
            ),
            manifest,
        )
    except mechanics.UnsealedPacket:
        raise
    except mechanics.InconclusivePacket as error:
        write_identity_checked_decision(
            make_decision(
                "inconclusive",
                manifest,
                cpu_selectors,
                correctness,
                stages,
                error.stage,
                failed_child=error.child,
                reasons=error.reasons,
            ),
            manifest,
        )
    except KeyboardInterrupt:
        write_identity_checked_decision(
            make_decision(
                "inconclusive",
                manifest,
                cpu_selectors,
                correctness,
                stages,
                launched_stage,
                reasons=["operator_interrupt_after_child_cleanup"],
            ),
            manifest,
        )
    except Exception as error:
        if mechanics.decision_publication_started():
            raise
        write_identity_checked_decision(
            make_decision(
                "implementation_or_contract_defect",
                manifest,
                cpu_selectors,
                correctness,
                stages,
                launched_stage,
                error_type=type(error).__name__,
                error=str(error),
            ),
            manifest,
        )
        raise


for name, function in {
    "source_and_build_identity": source_and_build_identity,
    "verify_non_model_identity": verify_non_model_identity,
    "verify_packet_identity": verify_packet_identity,
    "arm_environment": arm_environment,
    "environment_for_arm": environment_for_arm,
    "condition_for_child": condition_for_child,
    "record_launch": record_launch,
    "parse_marker": parse_marker,
    "parse_load_contract": parse_load_contract,
    "parse_observed_load_contract": parse_observed_load_contract,
    "validate_fresh_timing": validate_fresh_timing,
    "run_fresh_child": run_fresh_child,
    "analyze_fresh": analyze_fresh,
    "expected_child_stems": expected_child_stems,
    "verify_attempt_artifacts": verify_attempt_artifacts,
    "write_identity_checked_decision": write_identity_checked_decision,
}.items():
    setattr(mechanics, name, function)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    main(preflight_only=args.preflight_only)
