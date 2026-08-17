#!/usr/bin/env python3
from __future__ import annotations

import base64
import datetime as dt
import hashlib
import json
import os
import re
import shutil
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any, Callable

from analyze import analyze_failure_packet, analyze_packet, validate_child


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RAW = HERE / "raw"
CHRONOLOGY = RAW / "chronology.json"
RESULTS = HERE / "results.json"
TARGET_ARTIFACTS = ROOT / "target" / "vt-rebuild-campaign"
SCREEN_TEST = "metal::tests::attn_matrix_vt_rebuild_screen"
PROBE_TEST = "metal::tests::attn_matrix_vt_environment_probe"
CORRECTNESS_TESTS = [
    "metal::tests::attn_matrix_vt_dispatch_groups_cover_exact_thread_range",
    "metal::tests::attn_matrix_vt_compact_dispatch_matches_legacy_nonzero_span",
    "metal::tests::attn_matrix_vt_prefix_rebuild_preserves_scattered_suffix",
]
RECORD_MARKER = "VT_REBUILD_JSON "
PROBE_MARKER = "VT_ENV_JSON "
FAILURE_PATTERN = re.compile(r"V_T command failed arm=(D0|D1|D2|PREP_D2)")


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def encode_stream(data: bytes) -> dict[str, Any]:
    return {
        "base64": base64.b64encode(data).decode("ascii"),
        "sha256": sha256_bytes(data),
        "length": len(data),
    }


def decode_stream(packet: dict[str, Any]) -> bytes:
    data = base64.b64decode(packet["base64"], validate=True)
    if len(data) != packet["length"] or sha256_bytes(data) != packet["sha256"]:
        raise ValueError("stream authentication failed")
    return data


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    fsync_directory(path.parent)


def initialize_campaign(packet: dict[str, Any]) -> None:
    if RAW.exists() or RESULTS.exists():
        raise RuntimeError("refusing to overwrite existing campaign evidence")
    staging = HERE / f".raw-init-{os.getpid()}"
    staging.mkdir(parents=False, exist_ok=False)
    atomic_json(staging / "chronology.json", packet)
    os.replace(staging, RAW)
    fsync_directory(HERE)


def popen_packet(
    command: list[str],
    *,
    env: dict[str, str] | None = None,
    environment_overrides: dict[str, str] | None = None,
    environment_removed: list[str] | None = None,
    spawned: Callable[[subprocess.Popen[bytes]], None] | None = None,
) -> dict[str, Any]:
    started_at = now()
    start = time.monotonic()
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    spawn_callback_error = None
    if spawned is not None:
        try:
            spawned(process)
        except Exception as error:
            spawn_callback_error = f"{type(error).__name__}: {error}"
    stdout, stderr = process.communicate()
    return {
        "command": command,
        "environment_overrides": environment_overrides or {},
        "environment_removed": environment_removed or [],
        "pid": process.pid,
        "started_at": started_at,
        "finished_at": now(),
        "elapsed_s": time.monotonic() - start,
        "returncode": process.returncode,
        "termination_signal": -process.returncode if process.returncode < 0 else None,
        "spawn_callback_error": spawn_callback_error,
        "stdout": encode_stream(stdout),
        "stderr": encode_stream(stderr),
    }


def output_text(packet: dict[str, Any]) -> str:
    return (
        decode_stream(packet["stdout"]) + b"\n" + decode_stream(packet["stderr"])
    ).decode("utf-8", errors="strict")


def git_value(*arguments: str) -> str:
    packet = popen_packet(["git", *arguments])
    if packet["returncode"] != 0:
        raise RuntimeError(f"git {' '.join(arguments)} failed")
    return decode_stream(packet["stdout"]).decode().strip()


def source_identity(binary: Path) -> dict[str, Any]:
    status = git_value("status", "--porcelain", "--untracked-files=no")
    return {
        "head": git_value("rev-parse", "HEAD"),
        "tree": git_value("rev-parse", "HEAD^{tree}"),
        "tracked_clean": status == "",
        "binary_path": str(binary.resolve()),
        "binary_sha256": sha256_file(binary),
    }


def require_identity(actual: dict[str, Any], expected: dict[str, Any]) -> None:
    if actual != expected:
        raise RuntimeError(
            f"source/binary identity drift: expected={expected} actual={actual}"
        )


def build_test_binary() -> tuple[Path, dict[str, Any]]:
    command = [
        "cargo",
        "test",
        "-p",
        "qwen-llm",
        "--release",
        "--lib",
        "--no-run",
        "--message-format=json",
    ]
    packet = popen_packet(command)
    if packet["returncode"] != 0:
        raise RuntimeError("release test build failed")
    executables: list[Path] = []
    for line in decode_stream(packet["stdout"]).splitlines():
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (
            message.get("reason") == "compiler-artifact"
            and message.get("target", {}).get("name") == "qwen_llm"
            and message.get("profile", {}).get("test") is True
            and message.get("executable")
        ):
            executables.append(Path(message["executable"]))
    if len(executables) != 1 or not executables[0].is_file():
        raise RuntimeError(f"expected one qwen_llm test binary, got {executables}")
    source = executables[0].resolve()
    source_hash = sha256_file(source)
    destination_dir = TARGET_ARTIFACTS / source_hash
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / "qwen_llm_test"
    if destination.exists():
        if sha256_file(destination) != source_hash:
            raise RuntimeError("campaign artifact path contains different bytes")
    else:
        temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
        shutil.copyfile(source, temporary)
        os.chmod(temporary, 0o500)
        os.replace(temporary, destination)
        fsync_directory(destination_dir)
    if sha256_file(destination) != source_hash:
        raise RuntimeError("copied test binary hash mismatch")
    return destination.resolve(), packet


def sanitized_qwen_environment(
    overrides: dict[str, str],
) -> tuple[dict[str, str], list[str]]:
    removed = sorted(name for name in os.environ if name.startswith("QWEN_"))
    environment = {
        name: value for name, value in os.environ.items() if name not in removed
    }
    environment.update(overrides)
    return environment, removed


def exact_test(
    binary: Path,
    test_name: str,
    *,
    ignored: bool = False,
    require_metal: bool = False,
) -> dict[str, Any]:
    command = [str(binary), test_name]
    if ignored:
        command.append("--ignored")
    command.extend(["--exact", "--nocapture", "--test-threads=1"])
    overrides = {"QWEN_REQUIRE_METAL_TESTS": "1"} if require_metal else {}
    environment, removed = sanitized_qwen_environment(overrides)
    packet = popen_packet(
        command,
        env=environment,
        environment_overrides=overrides,
        environment_removed=removed,
    )
    text = output_text(packet)
    if packet["returncode"] != 0:
        raise RuntimeError(f"exact test failed: {test_name}")
    if (
        f"test {test_name} ..." not in text
        or "running 1 test" not in text
        or "1 passed" not in text
    ):
        raise RuntimeError(f"exact test identity/count drift: {test_name}")
    return packet


def parse_marker(packet: dict[str, Any], marker: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    stdout = decode_stream(packet["stdout"]).decode("utf-8", errors="strict")
    stderr = decode_stream(packet["stderr"]).decode("utf-8", errors="strict")
    if marker in stderr:
        raise ValueError(f"{marker.strip()} appeared on stderr")
    for line in stdout.splitlines():
        if marker not in line:
            continue
        records.append(json.loads(line.split(marker, 1)[1]))
    return records


def environment_probe(binary: Path) -> dict[str, Any]:
    packet = exact_test(binary, PROBE_TEST, ignored=True)
    records = parse_marker(packet, PROBE_MARKER)
    if len(records) != 1:
        raise RuntimeError("environment probe record count drift")
    return packet


def capture_system_environment() -> dict[str, Any]:
    commands = {
        "sw_vers": ["sw_vers"],
        "uname": ["uname", "-a"],
        "physical_memory": ["sysctl", "-n", "hw.memsize"],
        "power": ["pmset", "-g", "batt"],
        "memory_pressure": ["memory_pressure"],
        "thermal": ["pmset", "-g", "therm"],
    }
    packets = {name: popen_packet(command) for name, command in commands.items()}
    failed = [name for name, packet in packets.items() if packet["returncode"] != 0]
    if failed:
        raise RuntimeError(f"environment commands failed: {failed}")
    return packets


def protected_pid_packet() -> dict[str, Any]:
    packet = popen_packet(["ps", "-p", "8770", "-o", "pid=,state=,etime=,command="])
    stdout = decode_stream(packet["stdout"]).strip()
    if packet["returncode"] not in {0, 1}:
        raise RuntimeError("protected PID probe failed")
    packet["active"] = packet["returncode"] == 0 and bool(stdout)
    return packet


def provenance_snapshot(
    binary: Path, expected_identity: dict[str, Any]
) -> dict[str, Any]:
    identity = source_identity(binary)
    require_identity(identity, expected_identity)
    protected = protected_pid_packet()
    if protected["active"]:
        raise RuntimeError("protected PID 8770 is active; aborting without touching it")
    return {
        "captured_at": now(),
        "identity": identity,
        "system": capture_system_environment(),
        "protected_pid": protected,
        "metal_probe": environment_probe(binary),
    }


def classify_failure(packet: dict[str, Any]) -> str | None:
    match = FAILURE_PATTERN.search(output_text(packet))
    return match.group(1) if match else None


def run_cell(
    chronology: dict[str, Any],
    binary: Path,
    expected_identity: dict[str, Any],
    mode: str,
    prefix: int,
    chunk: int,
) -> dict[str, Any]:
    identity_before = source_identity(binary)
    require_identity(identity_before, expected_identity)
    command = [
        str(binary),
        SCREEN_TEST,
        "--ignored",
        "--exact",
        "--nocapture",
        "--test-threads=1",
    ]
    overrides = {
        "QWEN_VT_REBUILD_MODE": mode,
        "QWEN_VT_REBUILD_PREFIX": str(prefix),
        "QWEN_VT_REBUILD_CHUNK": str(chunk),
    }
    env, removed = sanitized_qwen_environment(overrides)
    child: dict[str, Any] = {
        "mode": mode,
        "prefix": prefix,
        "chunk": chunk,
        "state": "prepared",
        "command": command,
        "identity_before": identity_before,
    }
    chronology["children"].append(child)
    atomic_json(CHRONOLOGY, chronology)

    def spawned(process: subprocess.Popen[bytes]) -> None:
        child["state"] = "running"
        child["pid"] = process.pid
        child["spawned_at"] = now()
        atomic_json(CHRONOLOGY, chronology)

    execution = popen_packet(
        command,
        env=env,
        environment_overrides=overrides,
        environment_removed=removed,
        spawned=spawned,
    )
    child["state"] = "completed_unparsed"
    child["execution"] = execution
    atomic_json(CHRONOLOGY, chronology)
    if execution["spawn_callback_error"] is not None:
        raise RuntimeError(
            f"spawn persistence failed: {execution['spawn_callback_error']}"
        )
    child["identity_after"] = source_identity(binary)
    child["failure_arm"] = classify_failure(execution)
    atomic_json(CHRONOLOGY, chronology)
    require_identity(child["identity_after"], expected_identity)

    try:
        child["records"] = parse_marker(execution, RECORD_MARKER)
        child["state"] = "parsed"
    except Exception as error:
        child["state"] = "parse_failed"
        child["parse_error"] = f"{type(error).__name__}: {error}"
        atomic_json(CHRONOLOGY, chronology)
        raise
    atomic_json(CHRONOLOGY, chronology)
    return child


def execution_returncode(child: dict[str, Any]) -> int:
    return int(child.get("execution", {}).get("returncode", -999))


def d0_wall_times(child: dict[str, Any]) -> list[float]:
    return [
        float(record["wall_ms"])
        for record in child["records"]
        if record.get("kind") == "arm" and record.get("arm") == "D0"
    ]


def finish_failed_campaign(
    chronology: dict[str, Any],
    binary: Path,
    expected_identity: dict[str, Any],
    detail: str,
) -> None:
    chronology["failure"] = {"at": now(), "detail": detail}
    try:
        chronology["environment_after"] = provenance_snapshot(binary, expected_identity)
    except Exception as error:
        chronology["postflight_failure"] = f"{type(error).__name__}: {error}"
    chronology["finished_at"] = now()
    atomic_json(CHRONOLOGY, chronology)
    atomic_json(RESULTS, analyze_failure_packet(chronology))


def main() -> None:
    if RAW.exists() or RESULTS.exists():
        raise SystemExit("refusing to overwrite existing chronology or results")
    build_head = git_value("rev-parse", "HEAD")
    build_tree = git_value("rev-parse", "HEAD^{tree}")
    if git_value("status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("tracked worktree must be clean before preflight")
    binary, build = build_test_binary()
    expected_identity = source_identity(binary)
    if (
        expected_identity["head"] != build_head
        or expected_identity["tree"] != build_tree
    ):
        raise SystemExit("source identity drifted during build")

    correctness = [
        exact_test(binary, name, require_metal=True) for name in CORRECTNESS_TESTS
    ]
    environment_before = provenance_snapshot(binary, expected_identity)
    chronology: dict[str, Any] = {
        "schema_version": 2,
        "started_at": now(),
        "build_identity": {"head": build_head, "tree": build_tree},
        "binary_path": str(binary),
        "binary_sha256": expected_identity["binary_sha256"],
        "tracked_tree_clean": True,
        "ambient_qwen_environment": sorted(
            name for name in os.environ if name.startswith("QWEN_")
        ),
        "expected_identity": expected_identity,
        "build": build,
        "correctness": correctness,
        "environment_before": environment_before,
        "children": [],
        "safety": {},
    }
    initialize_campaign(chronology)

    try:
        for mode, prefix, chunk in [("dispatch", 512, 128), ("dispatch", 2048, 128)]:
            child = run_cell(chronology, binary, expected_identity, mode, prefix, chunk)
            if execution_returncode(child) != 0:
                raise RuntimeError(f"cell failed: {mode} P{prefix}/C{chunk}")
            validate_child(
                child, expected_identity, chronology["ambient_qwen_environment"]
            )

        walls = d0_wall_times(chronology["children"][1])
        if len(walls) != 6:
            raise RuntimeError("P2048 did not publish six D0 wall samples")
        run_dispatch_p8192 = (
            max(walls) <= 1000.0 and 4.0 * statistics.median(walls) <= 2000.0
        )
        chronology["safety"] = {
            "p2048_d0_wall_ms": walls,
            "p2048_d0_wall_median_ms": statistics.median(walls),
            "p2048_d0_wall_max_ms": max(walls),
            "run_dispatch_p8192": run_dispatch_p8192,
        }
        atomic_json(CHRONOLOGY, chronology)

        closing = [
            ("dispatch" if run_dispatch_p8192 else "compact", 8192, 128),
            ("compact", 16384, 128),
            ("compact", 32768, 128),
            ("overlap", 32768, 1024),
        ]
        for mode, prefix, chunk in closing:
            child = run_cell(chronology, binary, expected_identity, mode, prefix, chunk)
            if execution_returncode(child) != 0:
                raise RuntimeError(f"cell failed: {mode} P{prefix}/C{chunk}")
            validate_child(
                child, expected_identity, chronology["ambient_qwen_environment"]
            )

        environment_after = provenance_snapshot(binary, expected_identity)
        chronology["environment_after"] = environment_after
        chronology["finished_at"] = now()
        atomic_json(CHRONOLOGY, chronology)
        result = analyze_packet(chronology)
        atomic_json(RESULTS, result)
        print(json.dumps(result["disposition"], sort_keys=True))
    except Exception as error:
        finish_failed_campaign(
            chronology, binary, expected_identity, f"{type(error).__name__}: {error}"
        )
        raise


if __name__ == "__main__":
    main()
