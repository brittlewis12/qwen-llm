#!/usr/bin/env python3

import argparse
from decimal import Decimal, InvalidOperation
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

import v0593_demand_paged_no_copy as common


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = ROOT / "target/profiles/v0604-dense27b-parallel-copied-loader-p1"
PREREG = ROOT / "docs/bench/v0604-dense27b-parallel-copied-loader.md"
COMMON_RUNNER = ROOT / "scripts/profile/v0593_demand_paged_no_copy.py"
MODEL = Path("/Users/tito/models/Qwen3.6-27B-Q4_K_M.gguf")
PROMPT = ROOT / "docs/bench/tokenizer-prompts/current-reva-n8-interactive-qwen36.txt"
CLI_BINARY = ROOT / "target/release/qwen"
BENCH_BINARY = ROOT / "target/release/qwen-bench"
V0603_DENSE_DESCRIBE = ROOT / "docs/bench/v0603-dense27b-floor-describe.json"
V0603_A3B_DESCRIBE = ROOT / "docs/bench/v0603-a3b-floor-describe.json"
V0603_ROOT = ROOT / "target/profiles/v0603-dense27b-parallel-copied-floor-p1"
V0603_DECISION = V0603_ROOT / "decision.json"
V0603_COMPLETE = V0603_ROOT / "packet-complete.json"
V0603_INVENTORY = V0603_ROOT / "artifact-inventory.sha256"

EXPECTED_MODEL_SHA256 = (
    "5ed60d0af4650a854b1755bd392f9aef4872643dc25a254bc68043fa638392a0"
)
EXPECTED_PROMPT_SHA256 = (
    "e265de9742d1b22e566fc108ae26331ccf46166c6f071e73f211e0a1a7e8b474"
)
EXPECTED_RUNTIME_MODEL_ID = "6247cb71b536c975"
EXPECTED_RUNTIME_TOKENIZER_ID = "a4b0b26f8a8c9917"
EXPECTED_EVIDENCE_HASHES = {
    str(V0603_DENSE_DESCRIBE): (
        "4cfeccc3a8110c6e2632e7886eb73c425d815f74f2becd2c8df8c6e76453f7b7"
    ),
    str(V0603_A3B_DESCRIBE): (
        "833d63fcc628b41cff2691680de562301da6bd1812a7222c9e8b76cc6bb98180"
    ),
    str(V0603_DECISION): (
        "a0361b55d93cee769fc8d4db44eecdf83f3d1c63bdda5ecec07fd3d2edd279ac"
    ),
    str(V0603_COMPLETE): (
        "bcb19bc3f8776fe16e6459abb5930e020d89f2bd8cf20e3a31a7e567959282ab"
    ),
    str(V0603_INVENTORY): (
        "fa87cf2788d58b2f988ce07760ad3619bad835f4abb77dedd238b3dd563e3a9e"
    ),
}
EXPECTED_DEVICE = (
    "device: Apple M4 Max | unified_memory=true | max_threadgroup_memory=32768 bytes"
)
EXPECTED_HW_MEMSIZE = 137_438_953_472
EXPECTED_PROMPT_BYTES = 1_891
EXPECTED_PROMPT_TOKENS = 419
EXPECTED_OUTPUT_TOKENS = 128
EXPECTED_TRANSITIONS = 127
PAIR_ORDERS = ("AB", "BA", "BA", "AB", "AB", "BA")
RUNS = 5
COOLDOWN_S = 30.0
HOST_SAMPLE_LIMIT = 6
HOST_SAMPLE_INTERVAL_S = 30.0
READ_CHUNK_BYTES = 64 * 1024
U64_MAX = 2**64 - 1

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
    "[metal-gguf-parallel-copied] schema=2 profile=dense27b-q4km-v1 "
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
CORRECTNESS_HARNESS_PREFIX = (
    "test metal_forward::tests::gguf_parallel_copied_dense27b_q4_is_bit_exact ... "
)

common.MODEL = MODEL
common.EXPECTED_RUNTIME_MODEL_ID = EXPECTED_RUNTIME_MODEL_ID
common.EXPECTED_RUNTIME_TOKENIZER_ID = EXPECTED_RUNTIME_TOKENIZER_ID


class InconclusivePacket(RuntimeError):
    def __init__(self, stage: str, child: str, reasons: list[str]) -> None:
        super().__init__(f"{stage} {child}: {', '.join(reasons)}")
        self.stage = stage
        self.child = child
        self.reasons = reasons


class UnsealedPacket(RuntimeError):
    pass


def command_text(command: list[str], env: dict[str, str] | None = None) -> str:
    return subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    ).stdout


def parse_json(text: str) -> object:
    return json.loads(text, parse_constant=reject_json_constant)


def reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value!r}")


def json_text(value: object, *, pretty: bool = False) -> str:
    return json.dumps(
        value,
        indent=2 if pretty else None,
        sort_keys=True,
        allow_nan=False,
    )


def finite_number(
    value: object,
    label: str,
    *,
    positive: bool = False,
    nonnegative: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"{label} is not numeric")
    number = float(value)
    if not math.isfinite(number):
        raise RuntimeError(f"{label} is not finite")
    if positive and number <= 0:
        raise RuntimeError(f"{label} is not positive")
    if nonnegative and number < 0:
        raise RuntimeError(f"{label} is negative")
    return number


def positive_ratio(numerator: object, denominator: object, label: str) -> float:
    left = finite_number(numerator, f"{label}.numerator", positive=True)
    right = finite_number(denominator, f"{label}.denominator", positive=True)
    return finite_number(left / right, label, positive=True)


def finite_difference(left: object, right: object, label: str) -> float:
    minuend = finite_number(left, f"{label}.left", positive=True)
    subtrahend = finite_number(right, f"{label}.right", positive=True)
    return finite_number(minuend - subtrahend, label)


def required_manifest_paths() -> tuple[Path, ...]:
    return (
        Path(__file__).resolve(),
        PREREG,
        COMMON_RUNNER,
        MODEL,
        PROMPT,
        CLI_BINARY,
        BENCH_BINARY,
        V0603_DENSE_DESCRIBE,
        V0603_A3B_DESCRIBE,
        V0603_DECISION,
        V0603_COMPLETE,
        V0603_INVENTORY,
    )


def source_and_build_identity() -> tuple[str, dict[str, object]]:
    for path in (Path(__file__).resolve(), PREREG, COMMON_RUNNER, PROMPT):
        command_text(
            ["git", "ls-files", "--error-unmatch", str(path.relative_to(ROOT))]
        )
    commit = command_text(["git", "rev-parse", "HEAD"]).strip()
    dirty = command_text(["git", "status", "--porcelain=v1"]).strip()
    if dirty:
        raise RuntimeError(f"source is dirty: {dirty!r}")
    build = parse_json(
        command_text([str(BENCH_BINARY), "build-info", "--output", "json"])
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


def child_environment_record(env: dict[str, str]) -> dict[str, object]:
    digest = hashlib.sha256()
    for key, value in sorted(env.items()):
        key_bytes = key.encode()
        value_bytes = value.encode()
        digest.update(len(key_bytes).to_bytes(8, "little"))
        digest.update(key_bytes)
        digest.update(len(value_bytes).to_bytes(8, "little"))
        digest.update(value_bytes)
    controls = {
        key: value
        for key, value in sorted(env.items())
        if key.startswith(("QWEN_", "METAL_", "MTL_")) or key == "RUST_LOG"
    }
    if controls:
        raise RuntimeError(f"normalized child environment retains controls: {controls}")
    return {
        "schema": 1,
        "complete_sha256": digest.hexdigest(),
        "keys": sorted(env),
        "performance_controls": controls,
    }


def preflight_time_resources(base_env: dict[str, str]) -> dict[str, object]:
    output = command_text(
        ["/usr/bin/time", "-l", "/usr/bin/true"],
        env=base_env,
    )
    resources = process_resources(output)
    return {
        "command": ["/usr/bin/time", "-l", "/usr/bin/true"],
        "output_sha256": hashlib.sha256(output.encode()).hexdigest(),
        "resources": resources,
    }


def build_manifest(
    removed_environment: list[str], base_env: dict[str, str]
) -> dict[str, object]:
    commit, build = source_and_build_identity()
    paths = required_manifest_paths()
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError(f"missing packet inputs: {missing}")
    hashes = {str(path): common.sha256_file(path) for path in paths}
    if hashes[str(MODEL)] != EXPECTED_MODEL_SHA256:
        raise RuntimeError("model SHA-256 drifted")
    if hashes[str(PROMPT)] != EXPECTED_PROMPT_SHA256:
        raise RuntimeError("prompt SHA-256 drifted")
    for path, expected_hash in EXPECTED_EVIDENCE_HASHES.items():
        if hashes.get(path) != expected_hash:
            raise RuntimeError(f"sealed evidence SHA-256 drifted: {path}")
    v0603_complete = parse_json(V0603_COMPLETE.read_text(encoding="utf-8"))
    if not isinstance(v0603_complete, dict) or v0603_complete != {
        "schema": 1,
        "decision_sha256": EXPECTED_EVIDENCE_HASHES[str(V0603_DECISION)],
        "inventory_sha256": EXPECTED_EVIDENCE_HASHES[str(V0603_INVENTORY)],
    }:
        raise RuntimeError("v0.603 completion seal drifted")
    device = command_text([str(CLI_BINARY), "--info"], env=base_env).strip()
    macos_product = command_text(["sw_vers", "-productVersion"], env=base_env).strip()
    macos_build = command_text(["sw_vers", "-buildVersion"], env=base_env).strip()
    hw_memsize = int(command_text(["sysctl", "-n", "hw.memsize"], env=base_env))
    if device != EXPECTED_DEVICE:
        raise RuntimeError(f"device boundary drifted: {device!r}")
    if not macos_product or not macos_build:
        raise RuntimeError("macOS product/build identity is empty")
    if hw_memsize != EXPECTED_HW_MEMSIZE:
        raise RuntimeError(f"memory boundary drifted: {hw_memsize}")
    time_resource_probe = preflight_time_resources(base_env)
    return {
        "schema": 1,
        "created_unix_ms": time.time_ns() // 1_000_000,
        "source_commit": commit,
        "build_identity": build,
        "device": device,
        "macos_product_version": macos_product,
        "macos_build_version": macos_build,
        "hw_memsize": hw_memsize,
        "removed_environment": removed_environment,
        "child_environment": child_environment_record(base_env),
        "time_resource_probe": time_resource_probe,
        "sha256": hashes,
        "model_size_bytes": MODEL.stat().st_size,
        "prompt_bytes": PROMPT.stat().st_size,
        "prompt_tokens": EXPECTED_PROMPT_TOKENS,
        "output_tokens": EXPECTED_OUTPUT_TOKENS,
        "transition_count": EXPECTED_TRANSITIONS,
        "pair_orders": list(PAIR_ORDERS),
        "loaded_runs": RUNS,
        "cooldown_s": COOLDOWN_S,
        "host_sample_limit": HOST_SAMPLE_LIMIT,
        "host_sample_interval_s": HOST_SAMPLE_INTERVAL_S,
        "child_retry_count": 0,
        "sealed_evidence_hashes": EXPECTED_EVIDENCE_HASHES,
    }


def verify_packet_identity(manifest: dict[str, object]) -> None:
    verify_non_model_identity(manifest)
    actual_model = common.sha256_file(MODEL)
    if actual_model != manifest["sha256"][str(MODEL)]:
        raise RuntimeError("packet completion model hash drifted")


def verify_non_model_identity(manifest: dict[str, object]) -> None:
    commit, build = source_and_build_identity()
    if commit != manifest["source_commit"] or build != manifest["build_identity"]:
        raise RuntimeError("packet completion source/build identity drifted")
    hashes = {
        str(path): common.sha256_file(path)
        for path in required_manifest_paths()
        if path != MODEL
    }
    expected = {
        path: digest
        for path, digest in manifest["sha256"].items()
        if path != str(MODEL)
    }
    if hashes != expected:
        raise RuntimeError("packet completion non-model hashes drifted")
    product = command_text(["sw_vers", "-productVersion"]).strip()
    build_version = command_text(["sw_vers", "-buildVersion"]).strip()
    if (
        product != manifest["macos_product_version"]
        or build_version != manifest["macos_build_version"]
    ):
        raise RuntimeError("packet macOS product/build identity drifted")


def record_final_model_identity(manifest: dict[str, object]) -> None:
    verify_non_model_identity(manifest)
    actual_hash = common.sha256_file(MODEL)
    expected_hash = manifest["sha256"][str(MODEL)]
    payload = {
        "model": str(MODEL),
        "size_bytes": MODEL.stat().st_size,
        "expected_sha256": expected_hash,
        "actual_sha256": actual_hash,
        "matches": actual_hash == expected_hash,
    }
    path = ARTIFACT / "final-model-sha256.json"
    try:
        with path.open("x", encoding="utf-8") as output:
            output.write(json_text(payload, pretty=True) + "\n")
            output.flush()
            os.fsync(output.fileno())
    except OSError as error:
        raise UnsealedPacket(f"final model identity write failed: {error}") from error
    if not payload["matches"]:
        raise RuntimeError("packet completion model hash drifted")


def parse_swap_bytes(text: str) -> int:
    match = re.search(r"\bused\s*=\s*([0-9.]+)([KMG])", text)
    if match is None:
        raise RuntimeError(f"could not parse swap usage: {text!r}")
    scale = {"K": 1024, "M": 1024**2, "G": 1024**3}[match.group(2)]
    return round(float(match.group(1)) * scale)


def parse_vm_counter(text: str, label: str) -> int:
    match = re.search(rf"^{re.escape(label)}:\s+(\d+)\.$", text, re.MULTILINE)
    if match is None:
        raise RuntimeError(f"could not parse vm_stat {label}")
    return int(match.group(1))


def capture_vm_state() -> dict[str, object]:
    errors = []
    try:
        vm_stat = command_text(["vm_stat"])
    except Exception as error:
        vm_stat = None
        errors.append(f"vm_stat_capture={type(error).__name__}:{error}")
    try:
        swapusage = command_text(["sysctl", "-n", "vm.swapusage"])
    except Exception as error:
        swapusage = None
        errors.append(f"swapusage_capture={type(error).__name__}:{error}")

    parsed = {
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
            except Exception as error:
                errors.append(f"{key}_parse={type(error).__name__}:{error}")
    if swapusage is not None:
        try:
            parsed["swap_used_bytes"] = parse_swap_bytes(swapusage)
        except Exception as error:
            errors.append(f"swap_used_bytes_parse={type(error).__name__}:{error}")
    return {
        **parsed,
        "vm_stat": vm_stat,
        "swapusage": swapusage,
        "capture_errors": errors,
    }


def capture_host_state() -> dict[str, object]:
    try:
        return common.capture_host_state()
    except Exception as error:
        return {
            "thermal": None,
            "battery": None,
            "memory_pressure": None,
            "memory_available_percent": None,
            "valid": False,
            "capture_error": f"{type(error).__name__}: {error}",
        }


def vm_interval(
    label: str, before: dict[str, object], after: dict[str, object]
) -> dict[str, object]:
    sources = {
        "pageouts": "pageouts",
        "compressions": "compressions",
        "swapouts": "swapouts",
        "swap_occupancy_bytes": "swap_used_bytes",
        "compressor_stored_pages": "compressor_stored_pages",
        "compressor_occupied_pages": "compressor_occupied_pages",
    }
    deltas = {}
    for output, source in sources.items():
        left = before.get(source)
        right = after.get(source)
        if (
            isinstance(left, bool)
            or not isinstance(left, int)
            or isinstance(right, bool)
            or not isinstance(right, int)
        ):
            deltas[output] = None
        else:
            deltas[output] = right - left

    reasons = []
    if before.get("capture_errors"):
        reasons.append(f"{label}_before_vm_capture_or_parse_invalid")
    if after.get("capture_errors"):
        reasons.append(f"{label}_after_vm_capture_or_parse_invalid")
    if any(value is None for value in deltas.values()) and not reasons:
        reasons.append(f"{label}_vm_delta_unavailable")
    for counter in ("pageouts", "compressions", "swapouts"):
        value = deltas[counter]
        if value is not None and value < 0:
            reasons.append(f"{label}_{counter}_counter_regressed")
    if deltas["swapouts"] is not None and deltas["swapouts"] > 0:
        reasons.append(f"{label}_swapouts_growth")
    if (
        deltas["swap_occupancy_bytes"] is not None
        and deltas["swap_occupancy_bytes"] > 0
    ):
        reasons.append(f"{label}_swap_occupancy_growth")
    for gauge in ("compressor_stored_pages", "compressor_occupied_pages"):
        if deltas[gauge] is not None and deltas[gauge] > 0:
            reasons.append(f"{label}_{gauge}_growth")
    return {
        "label": label,
        "before": before,
        "after": after,
        "deltas": deltas,
        "advisory": {
            "pageouts_growth": max(deltas["pageouts"] or 0, 0),
            "compressions_growth": max(deltas["compressions"] or 0, 0),
        },
        "failure_reasons": reasons,
    }


def warm_model_file() -> tuple[float, int, str]:
    started = time.perf_counter()
    total = 0
    digest = hashlib.sha256()
    buffer = bytearray(8 * 1024 * 1024)
    with MODEL.open("rb", buffering=0) as handle:
        while True:
            count = handle.readinto(buffer)
            if count == 0:
                break
            total += count
            digest.update(memoryview(buffer)[:count])
    return (time.perf_counter() - started) * 1e3, total, digest.hexdigest()


def record_prelaunch_failure(
    stem: str, evidence: dict[str, object], reasons: list[str]
) -> None:
    path = ARTIFACT / f"{stem}.prelaunch-failure.json"
    payload = {
        "artifact_stem": stem,
        "evidence": evidence,
        "failure_reasons": reasons,
    }
    try:
        with path.open("xb") as output:
            output.write((json_text(payload, pretty=True) + "\n").encode())
            output.flush()
            os.fsync(output.fileno())
    except OSError as error:
        raise UnsealedPacket(f"prelaunch evidence write failed: {error}") from error


def condition_for_child(stem: str, manifest: dict[str, object]) -> dict[str, object]:
    host_before_cache = capture_host_state()
    vm_before_cache = capture_vm_state()
    if host_before_cache.get("capture_error") is not None:
        reasons = ["initial_host_capture_failed"]
        record_prelaunch_failure(
            stem,
            {
                "host_before_cache": host_before_cache,
                "vm_before_cache": vm_before_cache,
            },
            reasons,
        )
        raise InconclusivePacket("prelaunch", stem, reasons)
    try:
        cache_ms, cache_bytes, cache_sha256 = warm_model_file()
        model_size = MODEL.stat().st_size
    except OSError as error:
        reasons = [f"cache_read_or_stat_failed={type(error).__name__}:{error}"]
        record_prelaunch_failure(
            stem,
            {
                "host_before_cache": host_before_cache,
                "vm_before_cache": vm_before_cache,
            },
            reasons,
        )
        raise InconclusivePacket("prelaunch", stem, reasons) from error
    if cache_bytes != model_size:
        raise RuntimeError("cache precondition did not read the exact model size")
    if cache_sha256 != manifest["sha256"][str(MODEL)]:
        raise RuntimeError("cache precondition model hash drifted")
    time.sleep(COOLDOWN_S)

    host_samples = []
    for sample_index in range(HOST_SAMPLE_LIMIT):
        verify_non_model_identity(manifest)
        sample = capture_host_state()
        host_samples.append(sample)
        if sample["valid"]:
            break
        if sample_index + 1 < HOST_SAMPLE_LIMIT:
            time.sleep(HOST_SAMPLE_INTERVAL_S)
    if not host_samples[-1]["valid"]:
        reasons = ["host_sampler_exhausted_before_child"]
        record_prelaunch_failure(
            stem,
            {
                "host_before_cache": host_before_cache,
                "vm_before_cache": vm_before_cache,
                "cache_ms": cache_ms,
                "cache_bytes": cache_bytes,
                "cache_sha256": cache_sha256,
                "host_samples": host_samples,
            },
            reasons,
        )
        raise InconclusivePacket(
            "prelaunch",
            stem,
            reasons,
        )
    vm_before_spawn = capture_vm_state()
    cache_interval = vm_interval("cache", vm_before_cache, vm_before_spawn)
    if cache_interval["failure_reasons"]:
        reasons = list(cache_interval["failure_reasons"])
        record_prelaunch_failure(
            stem,
            {
                "host_before_cache": host_before_cache,
                "vm_before_cache": vm_before_cache,
                "cache_ms": cache_ms,
                "cache_bytes": cache_bytes,
                "cache_sha256": cache_sha256,
                "host_samples": host_samples,
                "vm_before_spawn": vm_before_spawn,
                "cache_interval": cache_interval,
            },
            reasons,
        )
        raise InconclusivePacket(
            "prelaunch",
            stem,
            reasons,
        )
    return {
        "host_before_cache": host_before_cache,
        "vm_before_cache": vm_before_cache,
        "cache_ms": cache_ms,
        "cache_bytes": cache_bytes,
        "cache_sha256": cache_sha256,
        "host_samples": host_samples,
        "host_before_spawn": host_samples[-1],
        "vm_before_spawn": vm_before_spawn,
        "cache_interval": cache_interval,
    }


def arm_environment(arm: str) -> dict[str, str | None]:
    if arm not in ("A", "B"):
        raise ValueError(f"unknown arm {arm!r}")
    return {
        "QWEN_GGUF_PARALLEL_COPY": "1" if arm == "B" else "0",
        "QWEN_GGUF_OWNED_ARENA": "0",
        "QWEN_GGUF_NO_COPY": "0",
        "QWEN_GGUF_NO_COPY_PREFAULT": None,
        "QWEN_NATIVE_QUANT_EMBED": None,
        "QWEN_MOE_ROUTER_F16": None,
    }


def environment_for_arm(base_env: dict[str, str], arm: str) -> dict[str, str]:
    env = base_env.copy()
    for key, value in arm_environment(arm).items():
        if value is not None:
            env[key] = value
    return env


def parse_marker(line: str) -> dict[str, int | float]:
    if " plan=" in line:
        raise RuntimeError("dense parallel-copy marker contains a planner field")
    suffix = line.removeprefix(MARKER_PREFIX)
    if suffix == line:
        raise RuntimeError("parallel-copy marker prefix or field order drifted")
    fields = suffix.split(" ")
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
    if len(fields) != len(names):
        raise RuntimeError("parallel-copy marker timing field count drifted")
    values = {}
    for field, name in zip(fields, names, strict=True):
        prefix = f"{name}="
        value = field.removeprefix(prefix)
        if value == field or not value or not value.isascii() or not value.isdecimal():
            raise RuntimeError(f"parallel-copy marker {name} is not unsigned decimal")
        parsed = int(value)
        if str(parsed) != value or parsed > U64_MAX:
            raise RuntimeError(f"parallel-copy marker {name} is not canonical")
        values[name] = parsed
    phase_sum = sum(values[name] for name in names[:4])
    if values["ready_us"] == 0 or abs(values["ready_us"] - phase_sum) > 4:
        raise RuntimeError("parallel-copy marker timing does not reconcile")
    if values["total_cpu_us"] != values["user_cpu_us"] + values["system_cpu_us"]:
        raise RuntimeError("parallel-copy marker CPU does not reconcile")
    values["cpu_per_wall"] = values["total_cpu_us"] / values["ready_us"]
    return values


def parse_load_contract(stderr: str, arm: str) -> dict[str, object]:
    expected_markers = 1 if arm == "B" else 0
    if stderr.count("[metal-gguf-parallel-copied]") != expected_markers:
        raise RuntimeError(f"{arm} candidate marker occurrence count drifted")
    if stderr.count("[metal-load] native quantized token embedding policy:") != 1:
        raise RuntimeError(f"{arm} native-policy occurrence count drifted")
    if stderr.count("[metal-load-ledger]") != 1:
        raise RuntimeError(f"{arm} load-ledger occurrence count drifted")
    prefixes = (
        "[metal-load] native quantized token embedding policy:",
        "[metal-gguf-parallel-copied]",
        "[metal-load-ledger]",
    )
    recognized = [line for line in stderr.splitlines() if line.startswith(prefixes)]
    forbidden = (
        "[metal-gguf-owned]",
        "[metal-gguf-retained]",
        "[metal-gguf-no-copy]",
    )
    if any(marker in stderr for marker in forbidden):
        raise RuntimeError("unrequested storage marker appeared")
    if arm == "A":
        if recognized != [POLICY_LINE, LEDGER_LINE]:
            raise RuntimeError(f"A load contract drifted: {recognized!r}")
        return {"storage": "copied", "marker": None}
    if len(recognized) != 3 or recognized[0] != POLICY_LINE:
        raise RuntimeError(f"B load-line count or policy drifted: {recognized!r}")
    if recognized[2] != LEDGER_LINE:
        raise RuntimeError("B copied ledger drifted")
    timings = parse_marker(recognized[1])
    return {
        "storage": "parallel-copied",
        "marker": recognized[1],
        "phase_us": timings,
    }


def parse_observed_load_contract(stderr: str, arm: str) -> dict[str, object] | None:
    tokens = (
        "[metal-load] native quantized token embedding policy:",
        "[metal-gguf-parallel-copied]",
        "[metal-load-ledger]",
    )
    occurrence_count = sum(stderr.count(token) for token in tokens)
    if occurrence_count == 0:
        return None
    recognized = [line for line in stderr.splitlines() if line.startswith(tokens)]
    if occurrence_count != len(recognized):
        raise RuntimeError("observed load token is not an exact line prefix")
    if any(
        marker in stderr
        for marker in (
            "[metal-gguf-owned]",
            "[metal-gguf-retained]",
            "[metal-gguf-no-copy]",
        )
    ):
        raise RuntimeError("observed unrequested storage marker")
    expected_kinds = (
        ("policy", "ledger") if arm == "A" else ("policy", "marker", "ledger")
    )
    if len(recognized) > len(expected_kinds):
        raise RuntimeError("observed load contract has extra lines")
    marker_timings = None
    for line, kind in zip(recognized, expected_kinds, strict=False):
        if kind == "policy" and line != POLICY_LINE:
            raise RuntimeError("observed native policy contradicts contract")
        if kind == "ledger" and line != LEDGER_LINE:
            raise RuntimeError("observed copied ledger contradicts contract")
        if kind == "marker":
            marker_timings = parse_marker(line)
    if len(recognized) < len(expected_kinds):
        return {
            "status": "incomplete-valid-prefix",
            "recognized_lines": recognized,
            "marker_phase_us": marker_timings,
        }
    complete = parse_load_contract(stderr, arm)
    return {"status": "complete", "contract": complete}


def parse_unique_time_resource(stderr: str, label: str) -> int:
    matches = re.findall(
        rf"^\s*(\d+)\s+{re.escape(label)}$",
        stderr,
        re.MULTILINE,
    )
    if len(matches) != 1:
        raise RuntimeError(f"expected one exact /usr/bin/time label {label!r}")
    return int(matches[0])


def decimal_seconds_to_ms(value: str, label: str) -> int:
    try:
        milliseconds = Decimal(value) * 1000
    except InvalidOperation as error:
        raise RuntimeError(f"invalid /usr/bin/time {label} seconds") from error
    if milliseconds != milliseconds.to_integral_value():
        raise RuntimeError(f"/usr/bin/time {label} is not exact milliseconds")
    return int(milliseconds)


def process_resources(stderr: str) -> dict[str, int]:
    summaries = re.findall(
        r"^\s*([0-9]+\.[0-9]{2}) real\s+"
        r"([0-9]+\.[0-9]{2}) user\s+"
        r"([0-9]+\.[0-9]{2}) sys$",
        stderr,
        re.MULTILINE,
    )
    if len(summaries) != 1:
        raise RuntimeError("expected one exact /usr/bin/time summary")
    real_s, user_s, system_s = summaries[0]
    real_ms = decimal_seconds_to_ms(real_s, "real")
    user_cpu_ms = decimal_seconds_to_ms(user_s, "user")
    system_cpu_ms = decimal_seconds_to_ms(system_s, "system")
    labels = (
        "maximum resident set size",
        "page reclaims",
        "page faults",
        "swaps",
        "block input operations",
        "block output operations",
        "instructions retired",
        "cycles elapsed",
        "peak memory footprint",
    )
    return {
        "real_ms": real_ms,
        "user_cpu_ms": user_cpu_ms,
        "system_cpu_ms": system_cpu_ms,
        "total_cpu_ms": user_cpu_ms + system_cpu_ms,
        **{
            label.replace(" ", "_"): parse_unique_time_resource(stderr, label)
            for label in labels
        },
    }


def capture_post_exit_state(
    conditioning: dict[str, object],
) -> dict[str, object]:
    host_after_exit = capture_host_state()
    vm_after_exit = capture_vm_state()
    return {
        "host_after_exit": host_after_exit,
        "vm_after_exit": vm_after_exit,
        "child_interval": vm_interval(
            "child",
            conditioning["vm_before_spawn"],
            vm_after_exit,
        ),
    }


def finish_child_validity(
    stderr: str,
    post_exit: dict[str, object],
    *,
    gate_major_faults: bool,
) -> tuple[dict[str, object], list[str]]:
    host_after_exit = post_exit["host_after_exit"]
    child_interval = post_exit["child_interval"]
    reasons = list(child_interval["failure_reasons"])
    try:
        resources = process_resources(stderr)
    except Exception as error:
        resources = None
        reasons.append(
            f"child_process_resource_parse_invalid={type(error).__name__}:{error}"
        )
    if not host_after_exit["valid"]:
        reasons.append("post_exit_host_invalid")
    if resources is not None:
        if resources["block_input_operations"] != 0:
            reasons.append("child_block_input")
        if gate_major_faults and resources["page_faults"] != 0:
            reasons.append("fresh_child_major_faults")
    return (
        {
            **post_exit,
            "process_resources": resources,
            "loaded_major_faults_advisory": (
                resources["page_faults"]
                if not gate_major_faults and resources is not None
                else None
            ),
        },
        reasons,
    )


def require_unique_line(stderr: str, line: str, label: str) -> None:
    if stderr.splitlines().count(line) != 1:
        raise RuntimeError(f"expected one exact {label} line")


def parse_loaded_bench(stderr: str) -> dict[str, object]:
    expected_header = (
        f"[bench] model={MODEL} "
        f"prompt={json.dumps(PROMPT.read_text(encoding='utf-8'), ensure_ascii=False)} "
        f"({EXPECTED_PROMPT_TOKENS} tokens), "
        f"gen={EXPECTED_TRANSITIONS} tokens, kv_capacity=1024"
    )
    header_lines = re.findall(r"^\[bench\] model=.*$", stderr, re.MULTILINE)
    if header_lines != [expected_header]:
        raise RuntimeError("loaded benchmark shape drifted")
    require_unique_line(stderr, "[bench] decode mode: full-logits", "decode mode")
    require_unique_line(
        stderr,
        "[bench] prefill mode: packed layer-major",
        "prefill mode",
    )
    require_unique_line(stderr, "[bench] prefill chunk: 1024", "prefill chunk")
    pattern = re.compile(
        r"^\[bench\] rep\s+(\d+): prefill\s+([0-9.]+) ms "
        r"\(([0-9.]+) t/s\)\s+decode\s+([0-9.]+) ms "
        r"\(([0-9.]+) t/s\)$",
        re.MULTILINE,
    )
    matches = pattern.findall(stderr)
    if len(matches) != RUNS or [int(row[0]) for row in matches] != list(
        range(1, RUNS + 1)
    ):
        raise RuntimeError("loaded repetition rows drifted")
    repetitions = []
    for rep, prefill_ms, prefill_tps, decode_ms, decode_tps in matches:
        row = {
            "rep": int(rep),
            "prefill_ms": float(prefill_ms),
            "prefill_tps_reported": float(prefill_tps),
            "decode_ms": float(decode_ms),
            "decode_tps_reported": float(decode_tps),
        }
        for key, value in row.items():
            if key != "rep":
                finite_number(value, f"loaded.{key}", positive=True)
        calculated_prefill = EXPECTED_PROMPT_TOKENS * 1000.0 / row["prefill_ms"]
        calculated_decode = EXPECTED_TRANSITIONS * 1000.0 / row["decode_ms"]
        if abs(calculated_prefill - row["prefill_tps_reported"]) > 1.0:
            raise RuntimeError("loaded prefill throughput is inconsistent")
        if abs(calculated_decode - row["decode_tps_reported"]) > 0.2:
            raise RuntimeError("loaded decode throughput is inconsistent")
        repetitions.append(row)

    request_matches = re.findall(
        r"^\[bench\] rep\s+(\d+) request\s+([0-9.]+) ms$",
        stderr,
        re.MULTILINE,
    )
    if len(request_matches) != RUNS or [int(row[0]) for row in request_matches] != list(
        range(1, RUNS + 1)
    ):
        raise RuntimeError("loaded request-wall rows drifted")
    request_walls = [float(row[1]) for row in request_matches]
    for repetition, request_wall in zip(repetitions, request_walls, strict=True):
        finite_number(request_wall, "loaded.request_wall_ms", positive=True)
        if request_wall + 0.2 < repetition["prefill_ms"] + repetition["decode_ms"]:
            raise RuntimeError("loaded request wall excludes phase work")
        repetition["request_wall_ms"] = request_wall
    average = re.findall(
        r"^\[bench\] request wall: ([0-9.]+) ms avg$",
        stderr,
        re.MULTILINE,
    )
    if (
        len(average) != 1
        or abs(statistics.mean(request_walls) - float(average[0])) > 0.11
    ):
        raise RuntimeError("loaded request-wall average drifted")
    generated = re.findall(r"^\[bench\] generated: (.*)$", stderr, re.MULTILINE)
    if len(generated) != 1:
        raise RuntimeError("loaded generated output line drifted")
    return {
        "repetitions": repetitions,
        "request_wall_average_reported_ms": float(average[0]),
        "generated_debug": generated[0],
        "generated_sha256": hashlib.sha256(generated[0].encode()).hexdigest(),
    }


def append_fsync(path: Path, row: dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as output:
        output.write(json_text(row) + "\n")
        output.flush()
        os.fsync(output.fileno())


def record_launch(
    stage: str,
    stem: str,
    command: list[str],
    arm: str,
    pair_index: int,
    position: int,
    manifest: dict[str, object],
) -> None:
    try:
        append_fsync(
            ARTIFACT / "launch-seal.jsonl",
            {
                "event": "launch",
                "unix_ms": time.time_ns() // 1_000_000,
                "stage": stage,
                "artifact_stem": stem,
                "command": command,
                "arm": arm,
                "arm_environment": arm_environment(arm),
                "normalized_base_environment": manifest["child_environment"],
                "pair_index": pair_index,
                "pair_order": PAIR_ORDERS[pair_index - 1],
                "position": position,
                "source_commit": manifest["source_commit"],
                "build_identity": manifest["build_identity"],
            },
        )
    except OSError as error:
        raise UnsealedPacket(f"launch seal failed for {stem}: {error}") from error


def record_completion(
    stage: str,
    stem: str,
    returncode: int | None,
    error: str | None,
) -> None:
    try:
        append_fsync(
            ARTIFACT / "launch-seal.jsonl",
            {
                "event": "completion",
                "unix_ms": time.time_ns() // 1_000_000,
                "stage": stage,
                "artifact_stem": stem,
                "returncode": returncode,
                "error": error,
            },
        )
    except OSError as write_error:
        raise UnsealedPacket(
            f"completion seal failed for {stem}: {write_error}"
        ) from write_error


def wait_for_child(process: subprocess.Popen[bytes]) -> tuple[int, list[str]]:
    deferred_errors = []
    while True:
        try:
            return process.wait(), deferred_errors
        except InterruptedError:
            deferred_errors.append("InterruptedError")
            continue
        except KeyboardInterrupt:
            deferred_errors.append("KeyboardInterrupt")
            continue
        except OSError as error:
            deferred_errors.append(f"{type(error).__name__}:{error}")
            try:
                returncode = process.poll()
            except KeyboardInterrupt:
                deferred_errors.append("poll_KeyboardInterrupt")
                returncode = None
            except OSError as poll_error:
                deferred_errors.append(f"poll_{type(poll_error).__name__}:{poll_error}")
                returncode = None
            if returncode is not None:
                return returncode, deferred_errors
            try:
                time.sleep(0.01)
            except KeyboardInterrupt:
                deferred_errors.append("sleep_KeyboardInterrupt")


def record_spawn_failure_evidence(
    stage: str,
    stem: str,
    conditioning: dict[str, object],
    error: OSError,
) -> None:
    host_after = capture_host_state()
    vm_after = capture_vm_state()
    evidence = {
        "stage": stage,
        "artifact_stem": stem,
        "error_type": type(error).__name__,
        "error": str(error),
        "host_after_failure": host_after,
        "vm_after_failure": vm_after,
        "child_interval": vm_interval(
            "child",
            conditioning["vm_before_spawn"],
            vm_after,
        ),
    }
    path = ARTIFACT / f"{stem}.spawn-failure.json"
    try:
        with path.open("xb") as output:
            output.write((json_text(evidence, pretty=True) + "\n").encode())
            output.flush()
            os.fsync(output.fileno())
    except OSError as write_error:
        raise UnsealedPacket(
            f"spawn failure evidence write failed: {write_error}"
        ) from write_error


def record_post_exit_evidence(
    stage: str,
    stem: str,
    returncode: int,
    evidence: dict[str, object],
    reasons: list[str],
) -> None:
    path = ARTIFACT / f"{stem}.post-exit.json"
    payload = {
        "stage": stage,
        "artifact_stem": stem,
        "returncode": returncode,
        "evidence": evidence,
        "validity_reasons_at_capture": reasons,
    }
    try:
        with path.open("xb") as output:
            output.write((json_text(payload, pretty=True) + "\n").encode())
            output.flush()
            os.fsync(output.fileno())
    except OSError as error:
        raise UnsealedPacket(f"post-exit evidence write failed: {error}") from error


def record_post_exit_state(
    stage: str,
    stem: str,
    returncode: int,
    post_exit: dict[str, object],
) -> None:
    path = ARTIFACT / f"{stem}.post-exit-state.json"
    payload = {
        "stage": stage,
        "artifact_stem": stem,
        "returncode": returncode,
        "post_exit": post_exit,
    }
    try:
        with path.open("xb") as output:
            output.write((json_text(payload, pretty=True) + "\n").encode())
            output.flush()
            os.fsync(output.fileno())
    except OSError as error:
        raise UnsealedPacket(f"post-exit state write failed: {error}") from error


def extract_correctness_load_lines(text: str) -> list[str]:
    tokens = (
        "[metal-load] native quantized token embedding policy:",
        "[metal-gguf-parallel-copied]",
        "[metal-load-ledger]",
    )
    recognized = []
    for line in text.splitlines():
        if not recognized and line == CORRECTNESS_HARNESS_PREFIX + POLICY_LINE:
            recognized.append(POLICY_LINE)
            continue
        matching = [token for token in tokens if line.startswith(token)]
        if len(matching) == 1:
            recognized.append(line)
            continue
        if "[metal-load" in line or "[metal-gguf-parallel-copied" in line:
            raise RuntimeError(f"malformed correctness load line: {line!r}")
    return recognized


def validate_correctness_load_text(
    text: str,
) -> tuple[list[str], dict[str, int | float]]:
    if text.count("[metal-gguf-parallel-copied]") != 1:
        raise RuntimeError("correctness candidate marker occurrence count drifted")
    if text.count("[metal-load] native quantized token embedding policy:") != 2:
        raise RuntimeError("correctness native-policy occurrence count drifted")
    if text.count("[metal-load-ledger]") != 2:
        raise RuntimeError("correctness ledger occurrence count drifted")
    recognized = extract_correctness_load_lines(text)
    if len(recognized) != 5:
        raise RuntimeError(f"correctness load-line count drifted: {recognized!r}")
    if recognized[:3] != [POLICY_LINE, LEDGER_LINE, POLICY_LINE]:
        raise RuntimeError("correctness A/B policy or A ledger ordering drifted")
    if recognized[4] != LEDGER_LINE:
        raise RuntimeError("correctness B ledger drifted")
    marker = parse_marker(recognized[3])
    return recognized, marker


def run_correctness(base_env: dict[str, str]) -> dict[str, object]:
    output_path = ARTIFACT / "correctness.out"
    command = [
        "cargo",
        "test",
        "--release",
        "-p",
        "qwen-llm",
        "--lib",
        "metal_forward::tests::gguf_parallel_copied_dense27b_q4_is_bit_exact",
        "--",
        "--ignored",
        "--exact",
        "--nocapture",
        "--test-threads=1",
    ]
    deferred_sigint: list[int] = []

    def defer_sigint(signum: int, _frame: object) -> None:
        deferred_sigint.append(signum)

    prior_sigint = signal.signal(signal.SIGINT, defer_sigint)
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
        signal.signal(signal.SIGINT, prior_sigint)
    if durability_error is not None:
        raise UnsealedPacket(
            f"correctness artifact write failed: {durability_error}"
        ) from durability_error
    if deferred_sigint:
        raise InconclusivePacket(
            "correctness",
            "correctness",
            [f"operator_sigint_deferred={len(deferred_sigint)}"],
        )
    if execution_error is not None:
        raise UnsealedPacket(
            f"correctness execution did not return: {execution_error}"
        ) from execution_error
    if result is None:
        raise UnsealedPacket("correctness execution returned no result")
    wall_ms = (time.perf_counter() - started) * 1e3
    text = output_path.read_text(encoding="utf-8")
    if result.returncode != 0 or "test result: ok. 1 passed;" not in text:
        raise RuntimeError("release full-state correctness gate failed")
    recognized, marker = validate_correctness_load_text(text)
    return {
        "command": command,
        "wall_ms": wall_ms,
        "output_path": str(output_path),
        "output_sha256": common.sha256_file(output_path),
        "recognized_load_lines": recognized,
        "candidate_phase_us": marker,
        "passed": True,
    }


def loaded_command(prompt_text: str) -> list[str]:
    return [
        "/usr/bin/time",
        "-l",
        str(BENCH_BINARY),
        "decode",
        "--model",
        str(MODEL),
        "--prompt",
        prompt_text,
        "--tokens",
        str(EXPECTED_TRANSITIONS),
        "--runs",
        str(RUNS),
        "--prefill-chunk",
        "1024",
        "--kv-capacity",
        "1024",
        "--full-logits-decode",
    ]


def fresh_command(timing_path: Path) -> list[str]:
    return [
        "/usr/bin/time",
        "-l",
        str(CLI_BINARY),
        "--model",
        str(MODEL),
        "--prompt-file",
        str(PROMPT),
        "--tokens",
        str(EXPECTED_OUTPUT_TOKENS),
        "--prefill-chunk",
        "1024",
        "--max-context-tokens",
        "1024",
        "--prefix-cache-max-mib",
        "0",
        "--request-timings",
        str(timing_path),
    ]


def run_loaded_child(
    arm: str,
    pair_index: int,
    order: str,
    position: int,
    prompt_text: str,
    base_env: dict[str, str],
    manifest: dict[str, object],
) -> dict[str, object]:
    stem = f"loaded-p{pair_index:02d}-{order.lower()}-r{position}-{arm.lower()}"
    stdout_path = ARTIFACT / f"{stem}.out"
    stderr_path = ARTIFACT / f"{stem}.err"
    for path in (stdout_path, stderr_path):
        if path.exists():
            raise RuntimeError(f"refusing to reuse child artifact {path}")
    conditioning = condition_for_child(stem, manifest)
    env = environment_for_arm(base_env, arm)
    command = loaded_command(prompt_text)
    record_launch("loaded", stem, command, arm, pair_index, position, manifest)
    process = None
    returncode = None
    wait_errors: list[str] = []
    completion_error = None
    deferred_sigint: list[int] = []

    def defer_sigint(signum: int, _frame: object) -> None:
        deferred_sigint.append(signum)

    prior_sigint = signal.signal(signal.SIGINT, defer_sigint)
    started = time.perf_counter()
    try:
        try:
            with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
                process = subprocess.Popen(
                    command,
                    cwd=ROOT,
                    env=env,
                    stdout=stdout,
                    stderr=stderr,
                )
                returncode, wait_errors = wait_for_child(process)
                stdout.flush()
                os.fsync(stdout.fileno())
                stderr.flush()
                os.fsync(stderr.fileno())
        except OSError as error:
            completion_error = f"{type(error).__name__}:{error}"
    finally:
        if process is not None and returncode is None:
            returncode, deferred_wait = wait_for_child(process)
            wait_errors.extend(deferred_wait)
        signal.signal(signal.SIGINT, prior_sigint)
        completion_parts = [
            part
            for part in (
                completion_error,
                ";".join(wait_errors) if wait_errors else None,
                (
                    f"operator_sigint_deferred={len(deferred_sigint)}"
                    if deferred_sigint
                    else None
                ),
            )
            if part
        ]
        record_completion(
            "loaded",
            stem,
            returncode,
            ";".join(completion_parts) if completion_parts else None,
        )
    process_wall_ms = (time.perf_counter() - started) * 1e3
    if process is None:
        if isinstance(completion_error, str):
            error = OSError(completion_error)
            record_spawn_failure_evidence("loaded", stem, conditioning, error)
        raise InconclusivePacket(
            "loaded",
            stem,
            [f"child_spawn_or_file_open_failed={completion_error}"],
        )
    if completion_error is not None:
        raise UnsealedPacket(f"loaded raw artifact flush failed: {completion_error}")
    post_exit = capture_post_exit_state(conditioning)
    record_post_exit_state("loaded", stem, returncode, post_exit)
    try:
        stderr_bytes = stderr_path.read_bytes()
    except OSError as error:
        reasons = [f"child_stderr_read_failed={type(error).__name__}:{error}"]
        record_post_exit_evidence(
            "loaded",
            stem,
            returncode,
            post_exit,
            reasons,
        )
        raise InconclusivePacket("loaded", stem, reasons) from error
    try:
        stderr = stderr_bytes.decode("utf-8")
        decode_error = None
    except UnicodeDecodeError as error:
        stderr = ""
        decode_error = f"child_stderr_utf8_invalid={error}"
    validity, reasons = finish_child_validity(
        stderr,
        post_exit,
        gate_major_faults=False,
    )
    if decode_error is not None:
        reasons.append(decode_error)
    if wait_errors:
        reasons.append(f"child_wait_or_io_interrupted={';'.join(wait_errors)}")
    if deferred_sigint:
        reasons.append(f"operator_sigint_deferred={len(deferred_sigint)}")
    if returncode != 0:
        reasons.append(f"child_nonzero_exit={returncode}")
    record_post_exit_evidence(
        "loaded",
        stem,
        returncode,
        validity,
        reasons,
    )
    if returncode == 0 and validity["process_resources"] is None:
        raise RuntimeError("successful loaded child process accounting is malformed")
    if decode_error is not None and returncode == 0:
        raise RuntimeError("loaded stderr is not valid UTF-8")
    if returncode != 0:
        try:
            load_contract = parse_observed_load_contract(stderr, arm)
        except Exception as error:
            load_contract = {
                "status": "failed-child-unparsed",
                "parse_error": f"{type(error).__name__}:{error}",
            }
        bench = None
    else:
        try:
            stdout_size = stdout_path.stat().st_size
        except OSError as error:
            reasons.append(f"child_stdout_stat_failed={type(error).__name__}:{error}")
            stdout_size = None
        if stdout_size is not None and stdout_size != 0:
            raise RuntimeError(f"loaded child {stem} emitted unexpected stdout")
        load_contract = parse_load_contract(stderr, arm)
        bench = parse_loaded_bench(stderr)
    return {
        "stage": "loaded",
        "artifact_stem": stem,
        "arm": arm,
        "pair_index": pair_index,
        "pair_order": order,
        "position": position,
        "command": command,
        "arm_environment": arm_environment(arm),
        **conditioning,
        **validity,
        "process_wall_ms": process_wall_ms,
        "returncode": returncode,
        "deferred_sigint": deferred_sigint,
        "stdout_sha256": hashlib.sha256(b"").hexdigest(),
        "stderr_sha256": hashlib.sha256(stderr_bytes).hexdigest(),
        "load_contract": load_contract,
        "bench": bench,
        "valid": not reasons,
        "validity_reasons": reasons,
    }


def validate_fresh_timing(
    timing: dict[str, object], manifest: dict[str, object]
) -> None:
    common.validate_timing(timing, EXPECTED_OUTPUT_TOKENS, manifest)
    expected = {
        "request_epoch": "first_post_model_load",
        "request_index": 0,
        "prefix_cache_used": False,
        "prefill_chunk_requested": 1024,
        "prefill_chunk_effective": EXPECTED_PROMPT_TOKENS,
        "prompt_tokens": EXPECTED_PROMPT_TOKENS,
        "max_context_tokens": 1024,
        "ttft_endpoint": "stdout_flush_complete",
        "decode_policy": "greedy_argmax",
        "stop_reason": "token_limit",
        "generated_tokens": EXPECTED_OUTPUT_TOKENS,
        "transition_count": EXPECTED_TRANSITIONS,
    }
    for key, value in expected.items():
        if timing.get(key) != value:
            raise RuntimeError(f"fresh timing {key} drifted: {timing.get(key)!r}")


def parse_observed_fresh_timing(
    timing_path: Path, manifest: dict[str, object]
) -> dict[str, object] | None:
    if not timing_path.is_file() or timing_path.stat().st_size == 0:
        return None
    data = timing_path.read_bytes()
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise RuntimeError("observed timing is not valid UTF-8") from error
    if not text.endswith("\n"):
        complete_lines = text.split("\n")[:-1]
        if any(line for line in complete_lines):
            raise RuntimeError("observed timing has complete rows plus a truncated row")
        return {
            "status": "incomplete-valid-prefix",
            "bytes": len(data),
        }
    lines = [line for line in text.splitlines() if line]
    if len(lines) != 1:
        raise RuntimeError("observed complete timing row count contradicts contract")
    timing = parse_json(lines[0])
    if not isinstance(timing, dict):
        raise RuntimeError("observed complete timing is not an object")
    validate_fresh_timing(timing, manifest)
    return {"status": "complete", "timing": timing}


def run_fresh_child(
    arm: str,
    pair_index: int,
    order: str,
    position: int,
    base_env: dict[str, str],
    manifest: dict[str, object],
) -> dict[str, object]:
    stem = f"fresh-p{pair_index:02d}-{order.lower()}-r{position}-{arm.lower()}"
    timing_path = ARTIFACT / f"{stem}.timing.jsonl"
    stdout_path = ARTIFACT / f"{stem}.out"
    stderr_path = ARTIFACT / f"{stem}.err"
    for path in (timing_path, stdout_path, stderr_path):
        if path.exists():
            raise RuntimeError(f"refusing to reuse child artifact {path}")
    conditioning = condition_for_child(stem, manifest)
    env = environment_for_arm(base_env, arm)
    command = fresh_command(timing_path)
    first_byte_ms = None
    last_byte_ms = None
    stdout_bytes = 0
    stdout_digest = hashlib.sha256()
    record_launch("fresh-128", stem, command, arm, pair_index, position, manifest)
    process = None
    returncode = None
    wait_errors: list[str] = []
    io_error = None
    spawn_error = None
    durability_error = None
    deferred_sigint: list[int] = []
    started = None

    def defer_sigint(signum: int, _frame: object) -> None:
        deferred_sigint.append(signum)

    prior_sigint = signal.signal(signal.SIGINT, defer_sigint)
    try:
        try:
            with stderr_path.open("xb") as stderr, stdout_path.open("xb") as stdout:
                started = time.perf_counter()
                try:
                    process = subprocess.Popen(
                        command,
                        cwd=ROOT,
                        env=env,
                        stdout=subprocess.PIPE,
                        stderr=stderr,
                    )
                except OSError as error:
                    spawn_error = f"{type(error).__name__}:{error}"
                if process is not None:
                    assert process.stdout is not None
                    try:
                        while True:
                            chunk = os.read(process.stdout.fileno(), READ_CHUNK_BYTES)
                            if not chunk:
                                break
                            observed_ms = (time.perf_counter() - started) * 1e3
                            if first_byte_ms is None:
                                first_byte_ms = observed_ms
                            last_byte_ms = observed_ms
                            stdout.write(chunk)
                            stdout_digest.update(chunk)
                            stdout_bytes += len(chunk)
                    except OSError as error:
                        io_error = f"{type(error).__name__}:{error}"
                        try:
                            process.stdout.close()
                        except OSError as close_error:
                            io_error += (
                                f";close_{type(close_error).__name__}:{close_error}"
                            )
                    returncode, wait_errors = wait_for_child(process)
                    exit_ms = (time.perf_counter() - started) * 1e3
                    if wait_errors:
                        wait_error = ";".join(wait_errors)
                        io_error = (
                            f"{io_error};{wait_error}" if io_error else wait_error
                        )
                stdout.flush()
                os.fsync(stdout.fileno())
                stderr.flush()
                os.fsync(stderr.fileno())
                if timing_path.is_file():
                    with timing_path.open("rb") as timing_file:
                        os.fsync(timing_file.fileno())
        except OSError as error:
            if process is None:
                spawn_error = f"{type(error).__name__}:{error}"
            else:
                durability_error = f"{type(error).__name__}:{error}"
    finally:
        if process is not None and returncode is None:
            returncode, deferred_wait = wait_for_child(process)
            wait_errors.extend(deferred_wait)
            if started is not None:
                exit_ms = (time.perf_counter() - started) * 1e3
        signal.signal(signal.SIGINT, prior_sigint)
        completion_parts = [
            part
            for part in (
                spawn_error,
                durability_error,
                io_error,
                ";".join(wait_errors) if wait_errors else None,
                (
                    f"operator_sigint_deferred={len(deferred_sigint)}"
                    if deferred_sigint
                    else None
                ),
            )
            if part
        ]
        record_completion(
            "fresh-128",
            stem,
            returncode,
            ";".join(completion_parts) if completion_parts else None,
        )
    if process is None:
        error = OSError(spawn_error or "child spawn failed")
        record_spawn_failure_evidence("fresh-128", stem, conditioning, error)
        raise InconclusivePacket(
            "fresh-128",
            stem,
            [f"child_spawn_or_file_open_failed={spawn_error}"],
        )
    if durability_error is not None:
        raise UnsealedPacket(f"fresh raw artifact flush failed: {durability_error}")
    post_exit = capture_post_exit_state(conditioning)
    record_post_exit_state("fresh-128", stem, returncode, post_exit)
    try:
        stderr_bytes = stderr_path.read_bytes()
    except OSError as error:
        reasons = [f"child_stderr_read_failed={type(error).__name__}:{error}"]
        record_post_exit_evidence(
            "fresh-128",
            stem,
            returncode,
            post_exit,
            reasons,
        )
        raise InconclusivePacket("fresh-128", stem, reasons) from error
    try:
        stderr = stderr_bytes.decode("utf-8")
        decode_error = None
    except UnicodeDecodeError as error:
        stderr = ""
        decode_error = f"child_stderr_utf8_invalid={error}"
    validity, reasons = finish_child_validity(
        stderr,
        post_exit,
        gate_major_faults=True,
    )
    if decode_error is not None:
        reasons.append(decode_error)
    if io_error is not None:
        reasons.append(f"child_stdout_io_failed={io_error}")
    if deferred_sigint:
        reasons.append(f"operator_sigint_deferred={len(deferred_sigint)}")
    if returncode != 0:
        reasons.append(f"child_nonzero_exit={returncode}")
    record_post_exit_evidence(
        "fresh-128",
        stem,
        returncode,
        validity,
        reasons,
    )
    if returncode == 0 and validity["process_resources"] is None:
        raise RuntimeError("successful fresh child process accounting is malformed")
    if decode_error is not None and returncode == 0:
        raise RuntimeError("fresh stderr is not valid UTF-8")
    if returncode != 0:
        try:
            load_contract = parse_observed_load_contract(stderr, arm)
        except Exception as error:
            load_contract = {
                "status": "failed-child-unparsed",
                "parse_error": f"{type(error).__name__}:{error}",
            }
        try:
            timing = parse_observed_fresh_timing(timing_path, manifest)
        except Exception as error:
            timing = {
                "status": "failed-child-unparsed",
                "parse_error": f"{type(error).__name__}:{error}",
            }
        final_to_exit_ms = None
        outer_residual_ms = None
    else:
        if first_byte_ms is None or last_byte_ms is None or stdout_bytes == 0:
            raise RuntimeError(f"fresh child {stem} emitted no stdout")
        load_contract = parse_load_contract(stderr, arm)
        final_to_exit_ms = finite_number(
            exit_ms - last_byte_ms,
            "last_stdout_byte_to_exit_ms",
            nonnegative=True,
        )
        if not timing_path.is_file():
            raise RuntimeError(f"fresh child {stem} timing artifact is missing")
        try:
            timing_text = timing_path.read_text(encoding="utf-8")
        except OSError as error:
            raise RuntimeError("successful fresh child timing read failed") from error
        else:
            timing_rows = [
                parse_json(line) for line in timing_text.splitlines() if line
            ]
            if len(timing_rows) != 1 or not isinstance(timing_rows[0], dict):
                raise RuntimeError(f"fresh child {stem} timing row count drifted")
            timing = timing_rows[0]
            validate_fresh_timing(timing, manifest)
            outer_residual_ms = finite_difference(
                first_byte_ms - float(timing["runtime_and_model_load_ms"]),
                timing["ttft_ms"],
                "outer_residual_ms",
            )
    try:
        timing_sha256 = (
            common.sha256_file(timing_path) if timing_path.is_file() else None
        )
    except OSError as error:
        reasons.append(f"child_timing_hash_failed={type(error).__name__}:{error}")
        timing_sha256 = None
    return {
        "stage": "fresh-128",
        "artifact_stem": stem,
        "arm": arm,
        "pair_index": pair_index,
        "pair_order": order,
        "position": position,
        "command": command,
        "arm_environment": arm_environment(arm),
        **conditioning,
        **validity,
        "spawn_to_first_byte_ms": first_byte_ms,
        "spawn_to_last_stdout_byte_ms": last_byte_ms,
        "spawn_to_exit_ms": exit_ms,
        "last_stdout_byte_to_exit_ms": final_to_exit_ms,
        "outer_residual_ms": outer_residual_ms,
        "deferred_sigint": deferred_sigint,
        "returncode": returncode,
        "stdout_bytes": stdout_bytes,
        "stdout_sha256": stdout_digest.hexdigest(),
        "stderr_sha256": hashlib.sha256(stderr_bytes).hexdigest(),
        "timing_sha256": timing_sha256,
        "load_contract": load_contract,
        "timing": timing,
        "valid": not reasons,
        "validity_reasons": reasons,
    }


def run_stage(
    stage: str,
    base_env: dict[str, str],
    manifest: dict[str, object],
    attempts_path: Path,
) -> list[dict[str, object]]:
    prompt_text = PROMPT.read_text(encoding="utf-8")
    rows = []
    for pair_index, order in enumerate(PAIR_ORDERS, 1):
        for position, arm in enumerate(order, 1):
            if stage == "loaded":
                row = run_loaded_child(
                    arm,
                    pair_index,
                    order,
                    position,
                    prompt_text,
                    base_env,
                    manifest,
                )
            elif stage == "fresh-128":
                row = run_fresh_child(
                    arm,
                    pair_index,
                    order,
                    position,
                    base_env,
                    manifest,
                )
            else:
                raise ValueError(f"unknown stage {stage!r}")
            try:
                append_fsync(attempts_path, row)
            except OSError as error:
                raise UnsealedPacket(
                    f"attempt row write failed for {row['artifact_stem']}: {error}"
                ) from error
            rows.append(row)
            if not row["valid"]:
                raise InconclusivePacket(
                    stage,
                    row["artifact_stem"],
                    list(row["validity_reasons"]),
                )
    if len(rows) != 12:
        raise RuntimeError(f"{stage} child count drifted")
    return rows


def paired_rows(
    rows: list[dict[str, object]], stage: str
) -> list[tuple[dict[str, object], dict[str, object]]]:
    if len(rows) != 12:
        raise RuntimeError(f"{stage} requires exactly 12 rows")
    pairs = []
    for pair_index, order in enumerate(PAIR_ORDERS, 1):
        selected = [
            row
            for row in rows
            if row["stage"] == stage
            and row["pair_index"] == pair_index
            and row["pair_order"] == order
        ]
        if len(selected) != 2 or {row["arm"] for row in selected} != {"A", "B"}:
            raise RuntimeError(f"{stage} pair {pair_index} membership drifted")
        if [
            row["arm"] for row in sorted(selected, key=lambda row: row["position"])
        ] != list(order):
            raise RuntimeError(f"{stage} pair {pair_index} order drifted")
        pairs.append(
            (
                next(row for row in selected if row["arm"] == "A"),
                next(row for row in selected if row["arm"] == "B"),
            )
        )
    return pairs


def late_metrics(row: dict[str, object]) -> dict[str, object]:
    repetitions = row["bench"]["repetitions"]
    if len(repetitions) != RUNS:
        raise RuntimeError("loaded repetition count drifted during analysis")
    late = repetitions[2:5]
    decode_tps = [
        positive_ratio(
            EXPECTED_TRANSITIONS * 1000.0,
            repetition["decode_ms"],
            "loaded.decode_tps",
        )
        for repetition in repetitions
    ]
    late_tps = decode_tps[2:5]
    prefill_ms = finite_number(
        statistics.median(item["prefill_ms"] for item in late),
        "loaded.prefill_median",
        positive=True,
    )
    decode_ms = finite_number(
        statistics.median(item["decode_ms"] for item in late),
        "loaded.decode_median",
        positive=True,
    )
    request_ms = finite_number(
        statistics.median(item["request_wall_ms"] for item in late),
        "loaded.request_median",
        positive=True,
    )
    relative_range = finite_number(
        (max(late_tps) - min(late_tps)) / statistics.median(late_tps),
        "loaded.late_tps_relative_range",
        nonnegative=True,
    )
    return {
        "prefill_ms": prefill_ms,
        "decode_ms": decode_ms,
        "request_ms": request_ms,
        "decode_tps": decode_tps,
        "rep5_over_rep3_tps": positive_ratio(
            decode_tps[4], decode_tps[2], "loaded.rep5_over_rep3_tps"
        ),
        "late_tps_relative_range": relative_range,
    }


def analyze_loaded(rows: list[dict[str, object]]) -> dict[str, object]:
    pairs = paired_rows(rows, "loaded")
    output_hashes = {row["bench"]["generated_sha256"] for pair in pairs for row in pair}
    if len(output_hashes) != 1:
        raise RuntimeError("loaded generated output identity differs")
    metrics = []
    for pair_index, (a_row, b_row) in enumerate(pairs, 1):
        a = late_metrics(a_row)
        b = late_metrics(b_row)
        row = {
            "pair_index": pair_index,
            "pair_order": PAIR_ORDERS[pair_index - 1],
            "prefill_a_over_b": positive_ratio(
                a["prefill_ms"], b["prefill_ms"], "loaded.P"
            ),
            "decode_a_over_b": positive_ratio(
                a["decode_ms"], b["decode_ms"], "loaded.D"
            ),
            "request_b_over_a": positive_ratio(
                b["request_ms"], a["request_ms"], "loaded.R"
            ),
            "a_rep5_over_rep3_tps": a["rep5_over_rep3_tps"],
            "b_rep5_over_rep3_tps": b["rep5_over_rep3_tps"],
            "a_late_tps_relative_range": a["late_tps_relative_range"],
            "b_late_tps_relative_range": b["late_tps_relative_range"],
            "rss_b_over_a": positive_ratio(
                b_row["process_resources"]["maximum_resident_set_size"],
                a_row["process_resources"]["maximum_resident_set_size"],
                "loaded.rss",
            ),
            "footprint_b_over_a": positive_ratio(
                b_row["process_resources"]["peak_memory_footprint"],
                a_row["process_resources"]["peak_memory_footprint"],
                "loaded.footprint",
            ),
            "complete_cpu_delta_ms": finite_difference(
                b_row["process_resources"]["total_cpu_ms"],
                a_row["process_resources"]["total_cpu_ms"],
                "loaded.complete_cpu_delta_ms",
            ),
            "complete_cpu_b_over_a": positive_ratio(
                b_row["process_resources"]["total_cpu_ms"],
                a_row["process_resources"]["total_cpu_ms"],
                "loaded.complete_cpu_b_over_a",
            ),
            "a_late": a,
            "b_late": b,
        }
        row["stability_gates"] = {
            "a_rep5_over_rep3": 0.98 <= row["a_rep5_over_rep3_tps"] <= 1.02,
            "b_rep5_over_rep3": 0.98 <= row["b_rep5_over_rep3_tps"] <= 1.02,
            "a_late_range": row["a_late_tps_relative_range"] <= 0.03,
            "b_late_range": row["b_late_tps_relative_range"] <= 0.03,
        }
        row["performance_gates"] = {
            "prefill_parity": row["prefill_a_over_b"] >= 0.99,
            "decode_parity": row["decode_a_over_b"] >= 0.99,
            "request_nonregression": row["request_b_over_a"] <= 1.01,
            "rss_ratio": row["rss_b_over_a"] <= 1.05,
            "footprint_ratio": row["footprint_b_over_a"] <= 1.05,
        }
        metrics.append(row)
    stable = all(all(row["stability_gates"].values()) for row in metrics)
    endpoint_cpu = [
        (
            row["pair_order"],
            finite_number(
                b_row["load_contract"]["phase_us"]["total_cpu_us"] / 1000.0,
                "loaded.endpoint_cpu_ms",
                positive=True,
            ),
        )
        for row, (_, b_row) in zip(metrics, pairs, strict=True)
    ]
    endpoint_cpu_ab = stratum(endpoint_cpu, "AB")
    endpoint_cpu_ba = stratum(endpoint_cpu, "BA")
    endpoint_cpu_gates = {
        "median_at_most_2450_ms": (
            statistics.median(value for _, value in endpoint_cpu) <= 2450.0
        ),
        "ab_median_at_most_2550_ms": statistics.median(endpoint_cpu_ab) <= 2550.0,
        "ba_median_at_most_2550_ms": statistics.median(endpoint_cpu_ba) <= 2550.0,
    }
    performance_passes = all(
        all(row["performance_gates"].values()) for row in metrics
    ) and all(endpoint_cpu_gates.values())
    rss_ratios = [row["rss_b_over_a"] for row in metrics]
    footprint_ratios = [row["footprint_b_over_a"] for row in metrics]
    return {
        "stage": "loaded",
        "global_output_sha256": next(iter(output_hashes)),
        "pairs": metrics,
        "rss_b_over_a": {
            "values": rss_ratios,
            "maximum": max(rss_ratios),
        },
        "footprint_b_over_a": {
            "values": footprint_ratios,
            "maximum": max(footprint_ratios),
        },
        "candidate_endpoint_cpu_ms": {
            "values": [value for _, value in endpoint_cpu],
            "ab": endpoint_cpu_ab,
            "ba": endpoint_cpu_ba,
            "median": statistics.median(value for _, value in endpoint_cpu),
            "gates": endpoint_cpu_gates,
        },
        "complete_process_cpu": {
            "paired_delta_ms": [row["complete_cpu_delta_ms"] for row in metrics],
            "b_over_a": [row["complete_cpu_b_over_a"] for row in metrics],
            "gating": "descriptive",
        },
        "stable": stable,
        "performance_passes": performance_passes,
        "passes": stable and performance_passes,
    }


def stratum(values: list[tuple[str, float]], order: str) -> list[float]:
    selected = [value for pair_order, value in values if pair_order == order]
    if len(selected) != 3:
        raise RuntimeError(f"expected three {order} stratum values")
    return selected


def analyze_fresh(rows: list[dict[str, object]]) -> dict[str, object]:
    pairs = paired_rows(rows, "fresh-128")
    output_hashes = {row["stdout_sha256"] for pair in pairs for row in pair}
    if len(output_hashes) != 1:
        raise RuntimeError("fresh generated output identity differs")
    metrics = []
    for pair_index, (a, b) in enumerate(pairs, 1):
        timing_a = a["timing"]
        timing_b = b["timing"]
        timing_fields = (
            "runtime_and_model_load_ms",
            "prefill_ms",
            "ttft_ms",
            "generation_ms",
            "total_request_ms",
        )
        for field in timing_fields:
            finite_number(timing_a[field], f"fresh.A.{field}", positive=True)
            finite_number(timing_b[field], f"fresh.B.{field}", positive=True)
        runtime_a = timing_a["runtime_and_model_load_ms"]
        runtime_b = timing_b["runtime_and_model_load_ms"]
        metrics.append(
            {
                "pair_index": pair_index,
                "pair_order": PAIR_ORDERS[pair_index - 1],
                "first_byte_speedup": positive_ratio(
                    a["spawn_to_first_byte_ms"],
                    b["spawn_to_first_byte_ms"],
                    "fresh.F",
                ),
                "exit_speedup": positive_ratio(
                    a["spawn_to_exit_ms"], b["spawn_to_exit_ms"], "fresh.E"
                ),
                "runtime_saving_ms": finite_difference(
                    runtime_a, runtime_b, "fresh.runtime_saving_ms"
                ),
                "runtime_b_over_a": positive_ratio(runtime_b, runtime_a, "fresh.Q"),
                "first_prefill_b_over_a": positive_ratio(
                    timing_b["prefill_ms"], timing_a["prefill_ms"], "fresh.PF"
                ),
                "ttft_b_over_a": positive_ratio(
                    timing_b["ttft_ms"], timing_a["ttft_ms"], "fresh.TT"
                ),
                "generation_b_over_a": positive_ratio(
                    timing_b["generation_ms"],
                    timing_a["generation_ms"],
                    "fresh.G",
                ),
                "request_b_over_a": positive_ratio(
                    timing_b["total_request_ms"],
                    timing_a["total_request_ms"],
                    "fresh.RQ",
                ),
                "complete_cpu_delta_ms": finite_difference(
                    b["process_resources"]["total_cpu_ms"],
                    a["process_resources"]["total_cpu_ms"],
                    "fresh.CPU",
                ),
                "complete_cpu_b_over_a": positive_ratio(
                    b["process_resources"]["total_cpu_ms"],
                    a["process_resources"]["total_cpu_ms"],
                    "fresh.complete_cpu_b_over_a",
                ),
                "candidate_endpoint_cpu_ms": finite_number(
                    b["load_contract"]["phase_us"]["total_cpu_us"] / 1000.0,
                    "fresh.endpoint_cpu_ms",
                    positive=True,
                ),
                "first_byte_b_wins": (
                    b["spawn_to_first_byte_ms"] < a["spawn_to_first_byte_ms"]
                ),
                "exit_b_wins": b["spawn_to_exit_ms"] < a["spawn_to_exit_ms"],
                "rss_b_over_a": positive_ratio(
                    b["process_resources"]["maximum_resident_set_size"],
                    a["process_resources"]["maximum_resident_set_size"],
                    "fresh.rss",
                ),
                "footprint_b_over_a": positive_ratio(
                    b["process_resources"]["peak_memory_footprint"],
                    a["process_resources"]["peak_memory_footprint"],
                    "fresh.footprint",
                ),
            }
        )
    first = [(row["pair_order"], row["first_byte_speedup"]) for row in metrics]
    exits = [(row["pair_order"], row["exit_speedup"]) for row in metrics]
    savings = [(row["pair_order"], row["runtime_saving_ms"]) for row in metrics]
    ratios = [(row["pair_order"], row["runtime_b_over_a"]) for row in metrics]
    prefills = [(row["pair_order"], row["first_prefill_b_over_a"]) for row in metrics]
    ttfts = [(row["pair_order"], row["ttft_b_over_a"]) for row in metrics]
    generations = [(row["pair_order"], row["generation_b_over_a"]) for row in metrics]
    requests = [(row["pair_order"], row["request_b_over_a"]) for row in metrics]
    cpu_deltas = [(row["pair_order"], row["complete_cpu_delta_ms"]) for row in metrics]
    endpoint_cpu = [
        (row["pair_order"], row["candidate_endpoint_cpu_ms"]) for row in metrics
    ]
    first_ab = stratum(first, "AB")
    first_ba = stratum(first, "BA")
    exit_ab = stratum(exits, "AB")
    exit_ba = stratum(exits, "BA")
    saving_ab = stratum(savings, "AB")
    saving_ba = stratum(savings, "BA")
    ratio_ab = stratum(ratios, "AB")
    ratio_ba = stratum(ratios, "BA")
    prefill_ab = stratum(prefills, "AB")
    prefill_ba = stratum(prefills, "BA")
    ttft_ab = stratum(ttfts, "AB")
    ttft_ba = stratum(ttfts, "BA")
    generation_ab = stratum(generations, "AB")
    generation_ba = stratum(generations, "BA")
    request_ab = stratum(requests, "AB")
    request_ba = stratum(requests, "BA")
    cpu_delta_ab = stratum(cpu_deltas, "AB")
    cpu_delta_ba = stratum(cpu_deltas, "BA")
    endpoint_cpu_ab = stratum(endpoint_cpu, "AB")
    endpoint_cpu_ba = stratum(endpoint_cpu, "BA")
    first_wins = sum(row["first_byte_b_wins"] for row in metrics)
    exit_wins = sum(row["exit_b_wins"] for row in metrics)
    first_ab_wins = sum(
        row["first_byte_b_wins"] for row in metrics if row["pair_order"] == "AB"
    )
    first_ba_wins = sum(
        row["first_byte_b_wins"] for row in metrics if row["pair_order"] == "BA"
    )
    exit_ab_wins = sum(
        row["exit_b_wins"] for row in metrics if row["pair_order"] == "AB"
    )
    exit_ba_wins = sum(
        row["exit_b_wins"] for row in metrics if row["pair_order"] == "BA"
    )
    gates = {
        "first_byte_median": statistics.median(value for _, value in first) >= 1.25,
        "first_byte_ab": statistics.median(first_ab) >= 1.20,
        "first_byte_ba": statistics.median(first_ba) >= 1.20,
        "first_byte_wins": first_wins >= 5,
        "first_byte_ab_wins": first_ab_wins >= 2,
        "first_byte_ba_wins": first_ba_wins >= 2,
        "exit_median": statistics.median(value for _, value in exits) >= 1.08,
        "exit_ab": statistics.median(exit_ab) >= 1.05,
        "exit_ba": statistics.median(exit_ba) >= 1.05,
        "exit_wins": exit_wins >= 5,
        "exit_ab_wins": exit_ab_wins >= 2,
        "exit_ba_wins": exit_ba_wins >= 2,
        "runtime_saving_median": (
            statistics.median(value for _, value in savings) >= 750.0
        ),
        "runtime_saving_ab": statistics.median(saving_ab) >= 600.0,
        "runtime_saving_ba": statistics.median(saving_ba) >= 600.0,
        "runtime_ratio_median": (
            statistics.median(value for _, value in ratios) <= 0.70
        ),
        "runtime_ratio_ab": statistics.median(ratio_ab) <= 0.80,
        "runtime_ratio_ba": statistics.median(ratio_ba) <= 0.80,
        "first_prefill_median": (
            statistics.median(value for _, value in prefills) <= 1.02
        ),
        "first_prefill_ab": statistics.median(prefill_ab) <= 1.03,
        "first_prefill_ba": statistics.median(prefill_ba) <= 1.03,
        "ttft_median": statistics.median(value for _, value in ttfts) <= 1.02,
        "ttft_ab": statistics.median(ttft_ab) <= 1.03,
        "ttft_ba": statistics.median(ttft_ba) <= 1.03,
        "generation_median": (
            statistics.median(value for _, value in generations) <= 1.01
        ),
        "generation_ab": statistics.median(generation_ab) <= 1.02,
        "generation_ba": statistics.median(generation_ba) <= 1.02,
        "request_median": statistics.median(value for _, value in requests) <= 1.01,
        "request_ab": statistics.median(request_ab) <= 1.02,
        "request_ba": statistics.median(request_ba) <= 1.02,
        "endpoint_cpu_median": (
            statistics.median(value for _, value in endpoint_cpu) <= 2450.0
        ),
        "endpoint_cpu_ab": statistics.median(endpoint_cpu_ab) <= 2550.0,
        "endpoint_cpu_ba": statistics.median(endpoint_cpu_ba) <= 2550.0,
        "complete_cpu_delta_median": (
            statistics.median(value for _, value in cpu_deltas) <= 900.0
        ),
        "complete_cpu_delta_ab": statistics.median(cpu_delta_ab) <= 1000.0,
        "complete_cpu_delta_ba": statistics.median(cpu_delta_ba) <= 1000.0,
        "rss_ratio": max(row["rss_b_over_a"] for row in metrics) <= 1.05,
        "footprint_ratio": (max(row["footprint_b_over_a"] for row in metrics) <= 1.05),
    }
    return {
        "stage": "fresh-128",
        "global_output_sha256": next(iter(output_hashes)),
        "pairs": metrics,
        "first_byte": {
            "values": [value for _, value in first],
            "ab": first_ab,
            "ba": first_ba,
            "median": statistics.median(value for _, value in first),
            "wins": first_wins,
            "ab_wins": first_ab_wins,
            "ba_wins": first_ba_wins,
        },
        "exit": {
            "values": [value for _, value in exits],
            "ab": exit_ab,
            "ba": exit_ba,
            "median": statistics.median(value for _, value in exits),
            "wins": exit_wins,
            "ab_wins": exit_ab_wins,
            "ba_wins": exit_ba_wins,
        },
        "runtime_saving_ms": {
            "values": [value for _, value in savings],
            "ab": saving_ab,
            "ba": saving_ba,
            "median": statistics.median(value for _, value in savings),
        },
        "runtime_b_over_a": {
            "values": [value for _, value in ratios],
            "ab": ratio_ab,
            "ba": ratio_ba,
            "median": statistics.median(value for _, value in ratios),
        },
        "model_ready": {
            "first_prefill_b_over_a": {
                "values": [value for _, value in prefills],
                "ab": prefill_ab,
                "ba": prefill_ba,
                "median": statistics.median(value for _, value in prefills),
            },
            "ttft_b_over_a": {
                "values": [value for _, value in ttfts],
                "ab": ttft_ab,
                "ba": ttft_ba,
                "median": statistics.median(value for _, value in ttfts),
            },
            "generation_b_over_a": {
                "values": [value for _, value in generations],
                "ab": generation_ab,
                "ba": generation_ba,
                "median": statistics.median(value for _, value in generations),
            },
            "request_b_over_a": {
                "values": [value for _, value in requests],
                "ab": request_ab,
                "ba": request_ba,
                "median": statistics.median(value for _, value in requests),
            },
        },
        "cpu": {
            "candidate_endpoint_ms": {
                "values": [value for _, value in endpoint_cpu],
                "ab": endpoint_cpu_ab,
                "ba": endpoint_cpu_ba,
                "median": statistics.median(value for _, value in endpoint_cpu),
            },
            "complete_process_delta_ms": {
                "values": [value for _, value in cpu_deltas],
                "ab": cpu_delta_ab,
                "ba": cpu_delta_ba,
                "median": statistics.median(value for _, value in cpu_deltas),
            },
            "complete_process_b_over_a": [
                row["complete_cpu_b_over_a"] for row in metrics
            ],
            "instructions_retired": {
                "A": [a["process_resources"]["instructions_retired"] for a, _ in pairs],
                "B": [b["process_resources"]["instructions_retired"] for _, b in pairs],
            },
            "cycles_elapsed": {
                "A": [a["process_resources"]["cycles_elapsed"] for a, _ in pairs],
                "B": [b["process_resources"]["cycles_elapsed"] for _, b in pairs],
            },
        },
        "rss_b_over_a": {
            "values": [row["rss_b_over_a"] for row in metrics],
            "maximum": max(row["rss_b_over_a"] for row in metrics),
        },
        "footprint_b_over_a": {
            "values": [row["footprint_b_over_a"] for row in metrics],
            "maximum": max(row["footprint_b_over_a"] for row in metrics),
        },
        "gates": gates,
        "passes": all(gates.values()),
    }


def expected_child_stems() -> list[str]:
    return [
        f"{stage}-p{pair_index:02d}-{order.lower()}-r{position}-{arm.lower()}"
        for stage in ("loaded", "fresh")
        for pair_index, order in enumerate(PAIR_ORDERS, 1)
        for position, arm in enumerate(order, 1)
    ]


def attempts_sha256() -> str | None:
    path = ARTIFACT / "attempts.jsonl"
    return common.sha256_file(path) if path.is_file() else None


def verify_attempt_artifacts(decision: dict[str, object]) -> dict[Path, str]:
    manifest_path = ARTIFACT / "manifest.json"
    manifest = parse_json(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise RuntimeError("packet manifest is not an object")
    attempts_path = ARTIFACT / "attempts.jsonl"
    sealed: dict[Path, str] = {}
    sealed[manifest_path] = common.sha256_file(manifest_path)
    correctness = decision.get("correctness")
    correctness_path = ARTIFACT / "correctness.out"
    correctness_metadata_path = ARTIFACT / "correctness.json"
    if correctness_path.is_file():
        correctness_digest = common.sha256_file(correctness_path)
        sealed[correctness_path] = correctness_digest
    else:
        correctness_digest = None
    if isinstance(correctness, dict):
        if correctness.get("output_sha256") != correctness_digest:
            raise RuntimeError("correctness artifact hash drifted")
        metadata = parse_json(correctness_metadata_path.read_text(encoding="utf-8"))
        if metadata != correctness:
            raise RuntimeError("correctness metadata drifted")
        sealed[correctness_metadata_path] = common.sha256_file(
            correctness_metadata_path
        )
    elif correctness_metadata_path.exists():
        raise RuntimeError("correctness metadata exists without a completed gate")
    attempt_rows = []
    if attempts_path.is_file():
        sealed[attempts_path] = common.sha256_file(attempts_path)
        lines = attempts_path.read_text(encoding="utf-8").splitlines()
    else:
        lines = []
    for line_number, line in enumerate(lines, 1):
        row = parse_json(line)
        if not isinstance(row, dict):
            raise RuntimeError(f"attempt row {line_number} is not an object")
        attempt_rows.append(row)
        stem = row.get("artifact_stem")
        if type(stem) is not str:
            raise RuntimeError(f"attempt row {line_number} lacks a stem")
        for suffix, key in (("out", "stdout_sha256"), ("err", "stderr_sha256")):
            digest = row.get(key)
            if type(digest) is not str or not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise RuntimeError(f"attempt row {line_number} has invalid {key}")
            path = ARTIFACT / f"{stem}.{suffix}"
            if path in sealed or common.sha256_file(path) != digest:
                raise RuntimeError(f"raw artifact {path.name} changed or duplicated")
            sealed[path] = digest
        timing_digest = row.get("timing_sha256")
        if timing_digest is not None:
            if type(timing_digest) is not str or not re.fullmatch(
                r"[0-9a-f]{64}", timing_digest
            ):
                raise RuntimeError(f"attempt row {line_number} timing hash is invalid")
            timing_path = ARTIFACT / f"{stem}.timing.jsonl"
            if common.sha256_file(timing_path) != timing_digest:
                raise RuntimeError(f"timing artifact {timing_path.name} changed")
            sealed[timing_path] = timing_digest

    launch_path = ARTIFACT / "launch-seal.jsonl"
    launch_rows = []
    if launch_path.is_file():
        sealed[launch_path] = common.sha256_file(launch_path)
        for line_number, line in enumerate(
            launch_path.read_text(encoding="utf-8").splitlines(), 1
        ):
            value = parse_json(line)
            if not isinstance(value, dict):
                raise RuntimeError(f"launch row {line_number} is not an object")
            launch_rows.append(value)
    if len(launch_rows) % 2 != 0:
        raise RuntimeError("launch/completion ledger is incomplete")
    launches = []
    for index in range(0, len(launch_rows), 2):
        launch = launch_rows[index]
        completion = launch_rows[index + 1]
        if launch.get("event") != "launch" or completion.get("event") != "completion":
            raise RuntimeError("launch/completion event order drifted")
        stem = launch.get("artifact_stem")
        if type(stem) is not str or completion.get("artifact_stem") != stem:
            raise RuntimeError("launch/completion stem drifted")
        if completion.get("stage") != launch.get("stage"):
            raise RuntimeError("launch/completion stage drifted")
        launches.append((launch, completion))

    expected_stems = expected_child_stems()
    launched_stems = [launch["artifact_stem"] for launch, _ in launches]
    attempt_stems = [row.get("artifact_stem") for row in attempt_rows]
    if len(set(launched_stems)) != len(launched_stems):
        raise RuntimeError("duplicate launched child identity")
    if len(set(attempt_stems)) != len(attempt_stems):
        raise RuntimeError("duplicate attempt child identity")
    if launched_stems != expected_stems[: len(launched_stems)]:
        raise RuntimeError("launched children are not the frozen packet prefix")
    if attempt_stems != launched_stems[: len(attempt_stems)]:
        raise RuntimeError("attempt rows do not match the launch prefix")
    unmatched_launches = len(launched_stems) - len(attempt_stems)
    if unmatched_launches > 1:
        raise RuntimeError("more than one launched child lacks an attempt row")

    prompt_text = PROMPT.read_text(encoding="utf-8")
    for index, (launch, completion) in enumerate(launches):
        expected_order = PAIR_ORDERS[index // 2 % len(PAIR_ORDERS)]
        expected_stage = "loaded" if index < 12 else "fresh-128"
        expected_pair = index // 2 % len(PAIR_ORDERS) + 1
        expected_position = index % 2 + 1
        expected_arm = expected_order[index % 2]
        expected_stem = expected_stems[index]
        if expected_stage == "loaded":
            expected_command = loaded_command(prompt_text)
        else:
            expected_command = fresh_command(ARTIFACT / f"{expected_stem}.timing.jsonl")
        if (
            launch.get("stage") != expected_stage
            or launch.get("artifact_stem") != expected_stem
            or launch.get("pair_order") != expected_order
            or launch.get("pair_index") != expected_pair
            or launch.get("position") != expected_position
            or launch.get("arm") != expected_arm
            or launch.get("command") != expected_command
            or launch.get("arm_environment") != arm_environment(expected_arm)
            or launch.get("normalized_base_environment")
            != manifest.get("child_environment")
            or launch.get("source_commit") != manifest.get("source_commit")
            or launch.get("build_identity") != manifest.get("build_identity")
        ):
            raise RuntimeError(f"launch row {index} frozen identity drifted")
        if (
            completion.get("returncode") is not None
            and type(completion.get("returncode")) is not int
        ):
            raise RuntimeError(f"completion row {index} return code is invalid")

    if unmatched_launches:
        launch, completion = launches[-1]
        unmatched_stem = launch["artifact_stem"]
        stopped_after = decision.get("stopped_after")
        bound_to_failed_child = decision.get("failed_child") == unmatched_stem
        bound_to_defect = (
            decision.get("status") == "implementation_or_contract_defect"
            and stopped_after == launch["stage"]
        )
        reasons = decision.get("reasons")
        bound_to_interrupt = (
            decision.get("status") == "inconclusive"
            and stopped_after == launch["stage"]
            and isinstance(reasons, list)
            and "operator_interrupt_after_child_cleanup" in reasons
        )
        if (
            decision.get("authority") != "none"
            or decision.get("status")
            not in (
                "inconclusive",
                "implementation_or_contract_defect",
            )
            or not (bound_to_failed_child or bound_to_defect or bound_to_interrupt)
        ):
            raise RuntimeError(
                "rowless launch is not bound to a non-authoritative stop"
            )
        returncode = completion.get("returncode")
        required_evidence = (
            [ARTIFACT / f"{unmatched_stem}.spawn-failure.json"]
            if returncode is None
            else [
                ARTIFACT / f"{unmatched_stem}.post-exit-state.json",
                ARTIFACT / f"{unmatched_stem}.post-exit.json",
            ]
        )
        for path in required_evidence:
            if not path.is_file():
                raise RuntimeError(f"rowless launch evidence is missing: {path.name}")
            sealed[path] = common.sha256_file(path)
        for suffix in ("out", "err", "timing.jsonl"):
            path = ARTIFACT / f"{unmatched_stem}.{suffix}"
            if path.is_file():
                sealed[path] = common.sha256_file(path)

    required_children = None
    if decision.get("status") in ("go", "kill"):
        required_children = (
            12 if decision.get("stopped_after") == "loaded-noninferiority" else 24
        )
    elif decision.get("stopped_after") == "loaded-instability":
        required_children = 12
    if required_children is not None and (
        len(launched_stems) != required_children
        or len(attempt_stems) != required_children
    ):
        raise RuntimeError("decision lacks its complete frozen child set")

    actual_attempts_sha = sealed.get(attempts_path)
    if decision.get("attempts_sha256") != actual_attempts_sha:
        raise RuntimeError("decision attempt ledger hash drifted")

    for index, row in enumerate(attempt_rows):
        launch, completion = launches[index]
        expected_order = PAIR_ORDERS[index // 2 % len(PAIR_ORDERS)]
        expected_stage = "loaded" if index < 12 else "fresh-128"
        if (
            row.get("stage") != expected_stage
            or row.get("pair_order") != expected_order
            or row.get("pair_index") != index // 2 % len(PAIR_ORDERS) + 1
            or row.get("position") != index % 2 + 1
            or row.get("arm") != expected_order[index % 2]
        ):
            raise RuntimeError(f"attempt row {index} frozen metadata drifted")
        for key in ("stage", "pair_order", "pair_index", "position", "arm", "command"):
            if row.get(key) != launch.get(key):
                raise RuntimeError(f"attempt row {index} launch {key} drifted")
        if row.get("returncode") != completion.get("returncode"):
            raise RuntimeError(f"attempt row {index} return code drifted")

    final_identity_path = ARTIFACT / "final-model-sha256.json"
    final_identity = parse_json(final_identity_path.read_text(encoding="utf-8"))
    expected_model_hash = manifest["sha256"][str(MODEL)]
    if final_identity != {
        "model": str(MODEL),
        "size_bytes": manifest["model_size_bytes"],
        "expected_sha256": expected_model_hash,
        "actual_sha256": expected_model_hash,
        "matches": True,
    }:
        raise RuntimeError("final model identity evidence drifted")
    sealed[final_identity_path] = common.sha256_file(final_identity_path)

    launched_set = set(launched_stems)
    actual_out_err = {
        path
        for pattern in ("loaded-*.out", "loaded-*.err", "fresh-*.out", "fresh-*.err")
        for path in ARTIFACT.glob(pattern)
        if path.is_file()
    }
    allowed_out_err = {
        ARTIFACT / f"{stem}.{suffix}"
        for stem in launched_set
        for suffix in ("out", "err")
    }
    if not actual_out_err.issubset(allowed_out_err):
        raise RuntimeError("raw artifact set contains an unlaunched child")
    actual_timing = {
        path for path in ARTIFACT.glob("fresh-*.timing.jsonl") if path.is_file()
    }
    allowed_timing = {
        ARTIFACT / f"{stem}.timing.jsonl"
        for stem in launched_set
        if stem.startswith("fresh-")
    }
    if not actual_timing.issubset(allowed_timing):
        raise RuntimeError("timing artifact set contains an unlaunched child")
    return sealed


def publish_decision(
    decision: dict[str, object], sealed_artifacts: dict[Path, str]
) -> None:
    decision_path = ARTIFACT / "decision.json"
    inventory_path = ARTIFACT / "artifact-inventory.sha256"
    complete_path = ARTIFACT / "packet-complete.json"
    decision_tmp = ARTIFACT / ".decision.json.tmp"
    inventory_tmp = ARTIFACT / ".artifact-inventory.sha256.tmp"
    complete_tmp = ARTIFACT / ".packet-complete.json.tmp"
    publication_paths = (
        decision_path,
        inventory_path,
        complete_path,
        decision_tmp,
        inventory_tmp,
        complete_tmp,
    )
    if any(path.exists() for path in publication_paths):
        raise RuntimeError("decision publication already started")

    decision_bytes = (json_text(decision, pretty=True) + "\n").encode()
    with decision_tmp.open("xb") as output:
        output.write(decision_bytes)
        output.flush()
        os.fsync(output.fileno())
    entries = []
    encountered_sealed = set()
    for artifact in sorted(ARTIFACT.iterdir()):
        if artifact.is_file() and artifact not in publication_paths:
            digest = common.sha256_file(artifact)
            if artifact in sealed_artifacts and digest != sealed_artifacts[artifact]:
                raise RuntimeError(f"sealed artifact {artifact.name} changed")
            if artifact in sealed_artifacts:
                encountered_sealed.add(artifact)
            entries.append(f"{digest}  {artifact.relative_to(ROOT)}")
    if encountered_sealed != set(sealed_artifacts):
        raise RuntimeError("sealed artifact disappeared during publication")
    decision_sha256 = hashlib.sha256(decision_bytes).hexdigest()
    entries.append(f"{decision_sha256}  {decision_path.relative_to(ROOT)}")
    inventory_bytes = ("\n".join(entries) + "\n").encode()
    with inventory_tmp.open("xb") as output:
        output.write(inventory_bytes)
        output.flush()
        os.fsync(output.fileno())
    inventory_sha256 = hashlib.sha256(inventory_bytes).hexdigest()
    complete = {
        "schema": 1,
        "decision_sha256": decision_sha256,
        "inventory_sha256": inventory_sha256,
    }
    with complete_tmp.open("xb") as output:
        output.write((json_text(complete, pretty=True) + "\n").encode())
        output.flush()
        os.fsync(output.fileno())
    os.replace(decision_tmp, decision_path)
    fsync_directory(ARTIFACT)
    os.replace(inventory_tmp, inventory_path)
    fsync_directory(ARTIFACT)
    os.replace(complete_tmp, complete_path)
    fsync_directory(ARTIFACT)
    try:
        print(json_text(decision, pretty=True))
    except BrokenPipeError:
        pass


def write_decision(
    decision: dict[str, object], sealed_artifacts: dict[Path, str]
) -> None:
    prior_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        publish_decision(decision, sealed_artifacts)
    finally:
        signal.signal(signal.SIGINT, prior_handler)


def decision_publication_started() -> bool:
    names = (
        "decision.json",
        "artifact-inventory.sha256",
        "packet-complete.json",
        ".decision.json.tmp",
        ".artifact-inventory.sha256.tmp",
        ".packet-complete.json.tmp",
    )
    return any((ARTIFACT / name).exists() for name in names)


def write_identity_checked_decision(
    decision: dict[str, object], manifest: dict[str, object]
) -> None:
    prior_sigint = signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        try:
            record_final_model_identity(manifest)
            identity_error = None
        except UnsealedPacket:
            raise
        except Exception as error:
            identity_error = f"{type(error).__name__}: {error}"
        if identity_error is not None:
            decision = {
                **decision,
                "reported_status_before_identity_check": decision.get("status"),
                "status": "implementation_or_contract_defect",
                "authority": "none",
                "identity_error": identity_error,
            }
        sealed_artifacts = verify_attempt_artifacts(decision)
        write_decision(decision, sealed_artifacts)
    finally:
        signal.signal(signal.SIGINT, prior_sigint)


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def reserve_artifact(manifest: dict[str, object]) -> None:
    preparing = ARTIFACT.with_name(f"{ARTIFACT.name}.preparing")
    if preparing.exists():
        raise RuntimeError(f"stale packet preparation directory {preparing}")
    preparing.mkdir(parents=True, exist_ok=False)
    try:
        with (preparing / "manifest.json").open("x", encoding="utf-8") as output:
            output.write(json_text(manifest, pretty=True) + "\n")
            output.flush()
            os.fsync(output.fileno())
        fsync_directory(preparing)
        os.replace(preparing, ARTIFACT)
        fsync_directory(ARTIFACT.parent)
    except Exception:
        raise


def main(*, preflight_only: bool) -> None:
    if ARTIFACT.exists():
        raise RuntimeError(f"refusing to reuse packet directory {ARTIFACT}")
    base_env, removed_environment = common.normalized_environment()
    manifest = build_manifest(removed_environment, base_env)
    if preflight_only:
        vm_state = capture_vm_state()
        if vm_state["capture_errors"]:
            raise RuntimeError(
                f"VM preflight capture failed: {vm_state['capture_errors']}"
            )
        print(
            json_text(
                {
                    "status": "preflight-passed",
                    "source_commit": manifest["source_commit"],
                    "manifest_sha256": hashlib.sha256(
                        (json_text(manifest, pretty=True) + "\n").encode()
                    ).hexdigest(),
                },
                pretty=True,
            )
        )
        return
    reserve_artifact(manifest)
    attempts_path = ARTIFACT / "attempts.jsonl"
    correctness = None
    stages = {}
    launched_stage = "none"
    try:
        correctness = run_correctness(base_env)
        try:
            with (ARTIFACT / "correctness.json").open("x", encoding="utf-8") as output:
                output.write(json_text(correctness, pretty=True) + "\n")
                output.flush()
                os.fsync(output.fileno())
        except OSError as error:
            raise UnsealedPacket(
                f"correctness metadata write failed: {error}"
            ) from error
        verify_packet_identity(manifest)

        launched_stage = "loaded"
        loaded_rows = run_stage("loaded", base_env, manifest, attempts_path)
        stages["loaded"] = analyze_loaded(loaded_rows)
        if not stages["loaded"]["stable"]:
            write_identity_checked_decision(
                {
                    "schema": 1,
                    "status": "inconclusive",
                    "authority": "none",
                    "stopped_after": "loaded-instability",
                    "source_commit": manifest["source_commit"],
                    "correctness": correctness,
                    "stages": stages,
                    "attempts_sha256": attempts_sha256(),
                },
                manifest,
            )
            return
        if not stages["loaded"]["performance_passes"]:
            write_identity_checked_decision(
                {
                    "schema": 1,
                    "status": "kill",
                    "authority": "none",
                    "stopped_after": "loaded-noninferiority",
                    "source_commit": manifest["source_commit"],
                    "correctness": correctness,
                    "stages": stages,
                    "attempts_sha256": attempts_sha256(),
                },
                manifest,
            )
            return

        launched_stage = "fresh-128"
        fresh_rows = run_stage("fresh-128", base_env, manifest, attempts_path)
        stages["fresh_128"] = analyze_fresh(fresh_rows)
        status = "go" if stages["fresh_128"]["passes"] else "kill"
        authority = (
            "force-only-dense27b-structural-profile" if status == "go" else "none"
        )
        write_identity_checked_decision(
            {
                "schema": 1,
                "status": status,
                "authority": authority,
                "stopped_after": "fresh-128",
                "source_commit": manifest["source_commit"],
                "correctness": correctness,
                "stages": stages,
                "attempts_sha256": attempts_sha256(),
            },
            manifest,
        )
    except UnsealedPacket:
        raise
    except InconclusivePacket as error:
        write_identity_checked_decision(
            {
                "schema": 1,
                "status": "inconclusive",
                "authority": "none",
                "stopped_after": error.stage,
                "failed_child": error.child,
                "reasons": error.reasons,
                "source_commit": manifest["source_commit"],
                "correctness": correctness,
                "stages": stages,
                "attempts_sha256": attempts_sha256(),
            },
            manifest,
        )
    except KeyboardInterrupt:
        write_identity_checked_decision(
            {
                "schema": 1,
                "status": "inconclusive",
                "authority": "none",
                "stopped_after": launched_stage,
                "reasons": ["operator_interrupt_after_child_cleanup"],
                "source_commit": manifest["source_commit"],
                "correctness": correctness,
                "stages": stages,
                "attempts_sha256": attempts_sha256(),
            },
            manifest,
        )
    except Exception as error:
        if decision_publication_started():
            raise
        write_identity_checked_decision(
            {
                "schema": 1,
                "status": "implementation_or_contract_defect",
                "authority": "none",
                "stopped_after": launched_stage,
                "error_type": type(error).__name__,
                "error": str(error),
                "source_commit": manifest["source_commit"],
                "correctness": correctness,
                "stages": stages,
                "attempts_sha256": attempts_sha256(),
            },
            manifest,
        )
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true")
    arguments = parser.parse_args()
    main(preflight_only=arguments.preflight_only)
