#!/usr/bin/env python3
from __future__ import annotations

import base64
import datetime as dt
import hashlib
import json
import math
import os
import re
import shlex
import shutil
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RAW = HERE / "raw"
CHRONOLOGY = RAW / "chronology.json"
RESULTS = HERE / "results.json"
PREREGISTRATION = HERE / "README.md"
PREREGISTRATION_SHA256 = (
    "2ced683c74ebd5a287b16c804550e73ba86ee2a176efb8265e198567b01d21f7"
)
MODEL = Path("/Users/tito/models/Qwen3.8-27B-Q4_K_M.gguf")
MODEL_BYTES = 17_106_773_984
MODEL_SHA256 = "7b2aec3b9ababdfd75aa17552ee95607d866e44decf547f6f12fcef85cc89f1b"
PROTECTED_PID = 8770
MARKER = b"VT_PRODUCT_JSON "
ALLOWED_UNTRACKED = {"docs/H6-VISION.md", "docs/dsv4-paper.md"}
ARTIFACTS = ROOT / "target" / "vt-product-campaign"
CELLS = [(8192, 128), (16_384, 128), (16_384, 1024)]
FORBIDDEN_ENVIRONMENT = re.compile(r"^(?:QWEN|MTL|METAL)|^RUST_LOG$")
SWAP_USED = re.compile(r"used\s*=\s*([0-9.]+)([KMGTP])", re.IGNORECASE)


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stream_packet(data: bytes) -> dict[str, Any]:
    return {
        "base64": base64.b64encode(data).decode("ascii"),
        "bytes": len(data),
        "sha256": sha256_bytes(data),
    }


def stream_bytes(packet: dict[str, Any]) -> bytes:
    data = base64.b64decode(packet["base64"], validate=True)
    if len(data) != packet["bytes"] or sha256_bytes(data) != packet["sha256"]:
        raise RuntimeError("recorded stream authentication failed")
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
    if temporary.exists():
        raise RuntimeError(f"refusing stale temporary file {temporary}")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def initialize_chronology(value: dict[str, Any]) -> None:
    if RAW.exists() or RESULTS.exists():
        raise RuntimeError("refusing to overwrite an existing raw packet or results")
    staging = HERE / f".raw-init-{os.getpid()}"
    staging.mkdir(mode=0o700)
    try:
        atomic_json(staging / CHRONOLOGY.name, value)
        os.replace(staging, RAW)
        fsync_directory(HERE)
    finally:
        if staging.exists():
            staging.rmdir()


def command_packet(
    argv: list[str], *, cwd: Path = ROOT, env: dict[str, str] | None = None
) -> dict[str, Any]:
    started_at = now()
    started = time.monotonic()
    process = subprocess.Popen(
        argv, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    stdout, stderr = process.communicate()
    return {
        "argv": argv,
        "pid": process.pid,
        "started_at": started_at,
        "completed_at": now(),
        "elapsed_seconds": time.monotonic() - started,
        "returncode": process.returncode,
        "termination_signal": -process.returncode if process.returncode < 0 else None,
        "stdout": stream_packet(stdout),
        "stderr": stream_packet(stderr),
    }


def checked_command(argv: list[str], *, cwd: Path = ROOT) -> dict[str, Any]:
    packet = command_packet(argv, cwd=cwd)
    if packet["returncode"] != 0:
        raise RuntimeError(f"command failed: {argv!r}")
    return packet


def stdout_text(packet: dict[str, Any]) -> str:
    return stream_bytes(packet["stdout"]).decode("utf-8", errors="strict")


def git_text(*args: str) -> str:
    return stdout_text(checked_command(["git", *args])).strip()


def untracked_inventory() -> list[dict[str, Any]]:
    packet = checked_command(
        ["git", "status", "--porcelain=v1", "-z", "--untracked-files=all"]
    )
    entries = stream_bytes(packet["stdout"]).split(b"\0")
    inventory: list[dict[str, Any]] = []
    for raw in entries:
        if not raw:
            continue
        entry = raw.decode("utf-8", errors="surrogateescape")
        status, path_text = entry[:2], entry[3:]
        if status != "??":
            raise RuntimeError(f"tracked worktree is not clean: {entry!r}")
        if path_text not in ALLOWED_UNTRACKED:
            raise RuntimeError(f"unapproved untracked path: {path_text}")
        path = ROOT / path_text
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(
                f"allowed untracked path is not a regular file: {path_text}"
            )
        stat = path.stat()
        inventory.append(
            {"path": path_text, "bytes": stat.st_size, "sha256": sha256_file(path)}
        )
    return sorted(inventory, key=lambda item: item["path"])


def source_identity(
    binary: Path, expected_untracked: list[dict[str, Any]]
) -> dict[str, Any]:
    inventory = untracked_inventory()
    if inventory != expected_untracked:
        raise RuntimeError("allowed untracked-file inventory drifted")
    stat = binary.stat()
    return {
        "head": git_text("rev-parse", "HEAD"),
        "tree": git_text("rev-parse", "HEAD^{tree}"),
        "untracked": inventory,
        "binary": {
            "canonical_path": str(binary.resolve(strict=True)),
            "bytes": stat.st_size,
            "device": stat.st_dev,
            "inode": stat.st_ino,
            "mode": stat.st_mode,
            "sha256": sha256_file(binary),
        },
    }


def require_identity(actual: dict[str, Any], expected: dict[str, Any]) -> None:
    if actual != expected:
        raise RuntimeError(
            f"source/binary identity drift: expected={expected} actual={actual}"
        )


def authenticate_model() -> dict[str, Any]:
    canonical = MODEL.resolve(strict=True)
    if canonical != MODEL:
        raise RuntimeError(f"model path is not canonical: {MODEL} -> {canonical}")
    if MODEL.is_symlink() or not MODEL.is_file():
        raise RuntimeError("pinned model is not a regular file")
    stat = MODEL.stat()
    digest = sha256_file(MODEL)
    if stat.st_size != MODEL_BYTES or digest != MODEL_SHA256:
        raise RuntimeError("pinned model size or full SHA-256 mismatch")
    return {
        "canonical_path": str(canonical),
        "bytes": stat.st_size,
        "sha256": digest,
        "device": stat.st_dev,
        "inode": stat.st_ino,
        "modified_ns": stat.st_mtime_ns,
    }


def build_binary() -> tuple[Path, dict[str, Any]]:
    argv = ["cargo", "build", "--release", "-p", "qwen-cli", "--bin", "qwen-bench"]
    build_environment = {
        name: value
        for name, value in os.environ.items()
        if not FORBIDDEN_ENVIRONMENT.match(name)
    }
    build = command_packet(argv, env=build_environment)
    build["environment_removed"] = sorted(
        name for name in os.environ if FORBIDDEN_ENVIRONMENT.match(name)
    )
    if build["returncode"] != 0:
        raise RuntimeError("release qwen-bench build failed")
    source = (ROOT / "target" / "release" / "qwen-bench").resolve(strict=True)
    digest = sha256_file(source)
    destination_directory = ARTIFACTS / digest
    destination = destination_directory / "qwen-bench"
    destination_directory.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if not destination.is_file() or sha256_file(destination) != digest:
            raise RuntimeError("immutable artifact path contains unexpected bytes")
    else:
        temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
        shutil.copyfile(source, temporary)
        os.chmod(temporary, 0o500)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        fsync_directory(destination_directory)
    if sha256_file(destination) != digest:
        raise RuntimeError("copied qwen-bench digest mismatch")
    return destination.resolve(strict=True), build


def process_table() -> dict[str, Any]:
    packet = command_packet(["ps", "-axo", "pid=,ppid=,state=,command=", "-ww"])
    if packet["returncode"] != 0:
        raise RuntimeError("process-table query failed")
    return packet


def parsed_processes(packet: dict[str, Any]) -> list[dict[str, Any]]:
    processes = []
    for line in stdout_text(packet).splitlines():
        fields = line.strip().split(None, 3)
        if len(fields) != 4:
            continue
        processes.append(
            {
                "pid": int(fields[0]),
                "ppid": int(fields[1]),
                "state": fields[2],
                "command": fields[3],
            }
        )
    return processes


def process_guard() -> dict[str, Any]:
    packet = process_table()
    processes = parsed_processes(packet)
    protected = next((item for item in processes if item["pid"] == PROTECTED_PID), None)
    if protected is not None:
        raise RuntimeError(
            "protected PID 8770 is active; aborting without signaling it"
        )
    competitors = []
    for item in processes:
        try:
            words = shlex.split(item["command"])
        except ValueError:
            words = item["command"].split()
        executable = Path(words[0]).name if words else ""
        has_model_argument = any(
            word == str(MODEL) or word.endswith(".gguf") or word in {"-m", "--model"}
            for word in words[1:]
        )
        if executable in {"qwen", "qwen-bench"} and has_model_argument:
            competitors.append(item)
    if competitors:
        raise RuntimeError(f"competing qwen model process(es) found: {competitors}")
    return {
        "query": packet,
        "protected_pid": None,
        "competing_qwen_model_processes": [],
    }


def system_snapshot() -> dict[str, Any]:
    commands = {
        "os": ["sw_vers"],
        "uname": ["uname", "-a"],
        "physical_memory": ["sysctl", "-n", "hw.memsize"],
        "ac_power": ["pmset", "-g", "batt"],
        "memory_pressure": ["memory_pressure", "-Q"],
        "swap": ["sysctl", "-n", "vm.swapusage"],
        "thermal": ["pmset", "-g", "therm"],
    }
    packets = {name: command_packet(argv) for name, argv in commands.items()}
    failed = [name for name, packet in packets.items() if packet["returncode"] != 0]
    if failed:
        raise RuntimeError(f"system provenance command(s) failed: {failed}")
    physical = int(stdout_text(packets["physical_memory"]).strip())
    if physical <= 0:
        raise RuntimeError("invalid physical-memory value")
    power_text = stdout_text(packets["ac_power"])
    if "AC Power" not in power_text:
        raise RuntimeError("campaign requires AC power")
    swap_text = stdout_text(packets["swap"])
    match = SWAP_USED.search(swap_text)
    if match is None:
        raise RuntimeError("could not parse swap usage")
    scale = {"K": 2**10, "M": 2**20, "G": 2**30, "T": 2**40, "P": 2**50}[
        match.group(2).upper()
    ]
    return {
        "captured_at": now(),
        "commands": packets,
        "physical_memory_bytes": physical,
        "ac_power": True,
        "memory_pressure_normal": True,
        "swap_used_bytes": int(float(match.group(1)) * scale),
    }


def child_environment() -> tuple[dict[str, str], dict[str, Any]]:
    removed = sorted(name for name in os.environ if FORBIDDEN_ENVIRONMENT.match(name))
    home = os.environ.get("HOME")
    if not home:
        raise RuntimeError("HOME is required for the frozen child environment")
    environment = {
        "HOME": home,
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
        "TMPDIR": os.environ.get("TMPDIR", "/tmp"),
    }
    return environment, {
        "explicit": environment,
        "removed_inherited_names": removed,
        "forbidden_present_after_sanitization": sorted(
            name for name in environment if FORBIDDEN_ENVIRONMENT.match(name)
        ),
    }


def child_argv(
    binary: Path, prefix: int, chunk: int, physical: int, preflight: bool
) -> list[str]:
    argv = [
        str(binary),
        "prefix-cache-vt-ab",
        "--model",
        str(MODEL),
        "--prefix-len",
        str(prefix),
        "--chunk-len",
        str(chunk),
        "--physical-memory-bytes",
        str(physical),
    ]
    if preflight:
        argv.append("--preflight-only")
    return argv


def parse_records(execution: dict[str, Any]) -> list[dict[str, Any]]:
    stdout = stream_bytes(execution["stdout"])
    stderr = stream_bytes(execution["stderr"])
    if MARKER in stderr:
        raise RuntimeError("VT_PRODUCT_JSON appeared on stderr")
    records: list[dict[str, Any]] = []
    for line in stdout.splitlines():
        if MARKER in line and not line.startswith(MARKER):
            raise RuntimeError(
                "VT_PRODUCT_JSON marker is not at the start of a stdout line"
            )
        if not line.startswith(MARKER):
            continue
        try:

            def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
                value: dict[str, Any] = {}
                for key, item in pairs:
                    if key in value:
                        raise ValueError(f"duplicate JSON key {key!r}")
                    value[key] = item
                return value

            record = json.loads(
                line[len(MARKER) :],
                object_pairs_hook=unique_object,
                parse_constant=lambda value: (_ for _ in ()).throw(
                    ValueError(f"non-finite JSON constant {value}")
                ),
            )
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
            raise RuntimeError(f"malformed VT_PRODUCT_JSON record: {error}") from error
        if not isinstance(record, dict) or record.get("schema_version") != 1:
            raise RuntimeError("VT_PRODUCT_JSON record has invalid type or schema")
        records.append(record)
    return records


def require_int(record: dict[str, Any], name: str, expected: int | None = None) -> int:
    value = record.get(name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeError(f"record field {name} is not an integer")
    if expected is not None and value != expected:
        raise RuntimeError(f"record field {name} is {value}, expected {expected}")
    return value


def validate_records(
    records: list[dict[str, Any]], prefix: int, chunk: int, preflight: bool
) -> None:
    expected_kinds = (
        [
            "setup",
            "correctness_arm",
            "correctness_arm",
            "correctness",
            "preflight_complete",
        ]
        if preflight
        else ["setup", "correctness_arm", "correctness_arm", "correctness"]
        + ["warmup"] * 4
        + ["arm"] * 12
        + ["complete"]
    )
    kinds = [record.get("kind") for record in records]
    if kinds != expected_kinds:
        raise RuntimeError(f"VT_PRODUCT_JSON kind/order drift: {kinds}")
    for record in records:
        require_int(record, "prefix", prefix)
        require_int(record, "chunk", chunk)
    setup = records[0]
    if (
        setup.get("model") != str(MODEL)
        or require_int(setup, "model_bytes") != MODEL_BYTES
    ):
        raise RuntimeError("setup model identity drift")
    if (
        setup.get("model_sha256") != MODEL_SHA256
        or setup.get("model_sha256_after_load") != MODEL_SHA256
    ):
        raise RuntimeError("setup model SHA-256 drift")
    if setup.get("preflight_only") is not preflight:
        raise RuntimeError("setup preflight flag drift")
    build = setup.get("build_identity")
    if (
        not isinstance(build, dict)
        or build.get("status") != "match"
        or build.get("problems") != []
    ):
        raise RuntimeError("qwen-bench build identity is not an exact match")
    correctness = records[3]
    if correctness.get("match") is not True:
        raise RuntimeError("correctness record did not report an exact match")
    arm_records = [
        record
        for record in records
        if record.get("kind") in {"correctness_arm", "warmup", "arm"}
    ]
    for record in arm_records:
        role = record.get("role")
        if role not in {"A", "B"} or record.get("compact") is not (role == "B"):
            raise RuntimeError("arm role/treatment mismatch")
        dispatch = record.get("vt_dispatch")
        if not isinstance(dispatch, dict):
            raise RuntimeError("missing V_T dispatch record")
        expected_elements = 16 * 4 * 256 * prefix
        require_int(dispatch, "calls", 16)
        require_int(dispatch, "row_sum", 16 * prefix)
        require_int(dispatch, "element_sum", expected_elements)
        require_int(dispatch, "base_pos_sum", 0)
        require_int(dispatch, "n_pos_sum", 16 * (prefix + chunk))
        require_int(dispatch, "legacy_calls", 16 if role == "A" else 0)
        require_int(dispatch, "compact_calls", 16 if role == "B" else 0)
        require_int(
            dispatch,
            "threadgroup_sum",
            expected_elements if role == "A" else 16 * 4 * prefix,
        )
    if preflight:
        return
    measured = [record for record in records if record.get("kind") == "arm"]
    expected_schedule = [
        (pair, order, role, bank)
        for pair, (order, arms) in enumerate(
            [
                ("AB", [("A", "X"), ("B", "Y")]),
                ("BA", [("B", "X"), ("A", "Y")]),
                ("BA", [("B", "Y"), ("A", "X")]),
                ("AB", [("A", "Y"), ("B", "X")]),
                ("AB", [("A", "X"), ("B", "Y")]),
                ("BA", [("B", "X"), ("A", "Y")]),
            ],
            1,
        )
        for role, bank in arms
    ]
    actual_schedule = [
        (
            record.get("pair"),
            record.get("order"),
            record.get("role"),
            record.get("bank"),
        )
        for record in measured
    ]
    if actual_schedule != expected_schedule:
        raise RuntimeError("measured arm schedule drift")


def process_identity(pid: int) -> dict[str, Any]:
    packet = command_packet(
        ["ps", "-p", str(pid), "-o", "pid=,ppid=,state=,lstart=,command=", "-ww"]
    )
    if packet["returncode"] != 0 or not stdout_text(packet).strip():
        raise RuntimeError(f"could not authenticate running child PID {pid}")
    return {"query": packet, "pid": pid}


def execute_before_chronology(
    binary: Path,
    argv: list[str],
    environment: dict[str, str],
    expected_identity: dict[str, Any],
    untracked: list[dict[str, Any]],
) -> dict[str, Any]:
    process_guard()
    require_identity(source_identity(binary, untracked), expected_identity)
    started_at = now()
    started = time.monotonic()
    process = subprocess.Popen(
        argv, cwd=ROOT, env=environment, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    identity = process_identity(process.pid)
    stdout, stderr = process.communicate()
    execution = {
        "argv": argv,
        "pid": process.pid,
        "process_identity": identity,
        "started_at": started_at,
        "completed_at": now(),
        "elapsed_seconds": time.monotonic() - started,
        "returncode": process.returncode,
        "termination_signal": -process.returncode if process.returncode < 0 else None,
        "stdout": stream_packet(stdout),
        "stderr": stream_packet(stderr),
    }
    if process.returncode != 0:
        raise RuntimeError("repairable preflight child failed")
    require_identity(source_identity(binary, untracked), expected_identity)
    records = parse_records(execution)
    validate_records(records, 8192, 128, True)
    execution["records"] = records
    return execution


def run_recorded_child(
    chronology: dict[str, Any],
    binary: Path,
    expected_identity: dict[str, Any],
    environment: dict[str, str],
    environment_record: dict[str, Any],
    prefix: int,
    chunk: int,
    *,
    preflight: bool = False,
) -> dict[str, Any]:
    process_guard()
    before = source_identity(binary, expected_identity["untracked"])
    require_identity(before, expected_identity)
    argv = child_argv(
        binary,
        prefix,
        chunk,
        chronology["environment_before"]["system"]["physical_memory_bytes"],
        preflight,
    )
    child: dict[str, Any] = {
        "cell": f"P{prefix}/C{chunk}",
        "preflight_only": preflight,
        "state": "prepared",
        "argv": argv,
        "environment": environment_record,
        "identity_before": before,
        "prepared_at": now(),
    }
    chronology["children"].append(child)
    atomic_json(CHRONOLOGY, chronology)
    started = time.monotonic()
    started_at = now()
    process = subprocess.Popen(
        argv, cwd=ROOT, env=environment, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    try:
        child.update(
            {
                "state": "running",
                "pid": process.pid,
                "started_at": started_at,
                "process_identity": process_identity(process.pid),
            }
        )
        atomic_json(CHRONOLOGY, chronology)
    except Exception as error:
        stdout, stderr = process.communicate()
        child.update(
            {
                "state": "persistence_failed_after_spawn",
                "persistence_error": f"{type(error).__name__}: {error}",
                "returncode": process.returncode,
                "stdout": stream_packet(stdout),
                "stderr": stream_packet(stderr),
            }
        )
        raise
    stdout, stderr = process.communicate()
    child.update(
        {
            "state": "completed_unparsed",
            "completed_at": now(),
            "elapsed_seconds": time.monotonic() - started,
            "returncode": process.returncode,
            "termination_signal": -process.returncode
            if process.returncode < 0
            else None,
            "stdout": stream_packet(stdout),
            "stderr": stream_packet(stderr),
        }
    )
    atomic_json(CHRONOLOGY, chronology)
    child["identity_after"] = source_identity(binary, expected_identity["untracked"])
    atomic_json(CHRONOLOGY, chronology)
    require_identity(child["identity_after"], expected_identity)
    if process.returncode != 0:
        child["state"] = "failed"
        atomic_json(CHRONOLOGY, chronology)
        raise RuntimeError(f"child failed: P{prefix}/C{chunk} preflight={preflight}")
    try:
        records = parse_records(child)
        validate_records(records, prefix, chunk, preflight)
    except Exception as error:
        child["state"] = "parse_or_validation_failed"
        child["validation_error"] = f"{type(error).__name__}: {error}"
        atomic_json(CHRONOLOGY, chronology)
        raise
    child["records"] = records
    child["state"] = "validated"
    atomic_json(CHRONOLOGY, chronology)
    return child


def legacy_suffix_walls(child: dict[str, Any]) -> list[float]:
    values = [
        record.get("suffix_wall_ms")
        for record in child["records"]
        if record.get("kind") == "arm" and record.get("role") == "A"
    ]
    if len(values) != 6 or any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
        for value in values
    ):
        raise RuntimeError("cell did not publish six valid legacy suffix-wall samples")
    return [float(value) for value in values]


def record_safety(chronology: dict[str, Any], name: str, value: dict[str, Any]) -> None:
    chronology["safety"][name] = value
    atomic_json(CHRONOLOGY, chronology)


def provenance() -> dict[str, Any]:
    return {
        "captured_at": now(),
        "process_guard": process_guard(),
        "system": system_snapshot(),
    }


def invoke_analyzer() -> None:
    if RESULTS.exists():
        raise RuntimeError("refusing to overwrite existing results")
    analyzer = HERE / "analyze.py"
    if not analyzer.is_file():
        raise RuntimeError(f"missing frozen analyzer: {analyzer}")
    temporary = HERE / f".results.json.tmp-{os.getpid()}"
    if temporary.exists():
        raise RuntimeError(f"refusing stale analyzer output {temporary}")
    try:
        packet = command_packet(
            ["uv", "run", str(analyzer), str(CHRONOLOGY), str(temporary)], cwd=HERE
        )
        if packet["returncode"] != 0 or not temporary.is_file():
            raise RuntimeError("analyze.py failed to produce results")
        with temporary.open("r+b") as handle:
            json.load(handle)
            os.fsync(handle.fileno())
        os.replace(temporary, RESULTS)
        fsync_directory(HERE)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> None:
    if RAW.exists() or RESULTS.exists():
        raise SystemExit("refusing to overwrite existing raw packet or results")
    if sha256_file(PREREGISTRATION) != PREREGISTRATION_SHA256:
        raise SystemExit("preregistration README SHA-256 mismatch")
    untracked = untracked_inventory()
    head = git_text("rev-parse", "HEAD")
    tree = git_text("rev-parse", "HEAD^{tree}")
    model = authenticate_model()
    process_guard()
    environment_before_preflight = system_snapshot()
    binary, build = build_binary()
    expected_identity = source_identity(binary, untracked)
    if expected_identity["head"] != head or expected_identity["tree"] != tree:
        raise SystemExit("source identity drifted during build")
    environment, environment_record = child_environment()
    preflight = execute_before_chronology(
        binary,
        child_argv(
            binary,
            8192,
            128,
            environment_before_preflight["physical_memory_bytes"],
            True,
        ),
        environment,
        expected_identity,
        untracked,
    )
    environment_after_preflight = system_snapshot()
    if (
        not environment_before_preflight["memory_pressure_normal"]
        or not environment_after_preflight["memory_pressure_normal"]
        or environment_after_preflight["swap_used_bytes"]
        > environment_before_preflight["swap_used_bytes"]
    ):
        raise SystemExit("repairable P8192/C128 preflight pressure/swap gate failed")
    chronology: dict[str, Any] = {
        "schema_version": 1,
        "campaign": "qwen-vt-product-ab",
        "started_at": now(),
        "preregistration": {
            "path": str(PREREGISTRATION.relative_to(ROOT)),
            "sha256": PREREGISTRATION_SHA256,
        },
        "source": {
            "head": head,
            "tree": tree,
            "tracked_clean": True,
            "allowed_untracked": untracked,
        },
        "model": model,
        "binary": expected_identity["binary"],
        "expected_identity": expected_identity,
        "build": build,
        "sanitized_child_environment": environment_record,
        "repairable_preflight": {
            "environment_before": environment_before_preflight,
            "execution": preflight,
            "environment_after": environment_after_preflight,
        },
        "environment_before": provenance(),
        "children": [],
        "safety": {},
    }
    initialize_chronology(chronology)
    try:
        p8 = run_recorded_child(
            chronology,
            binary,
            expected_identity,
            environment,
            environment_record,
            8192,
            128,
        )
        p8_walls = legacy_suffix_walls(p8)
        p16_c128_allowed = (
            max(p8_walls) <= 10_000.0 and 2.0 * statistics.median(p8_walls) <= 8_000.0
        )
        record_safety(
            chronology,
            "before_p16384_c128",
            {
                "predicate": "max(A_suffix_wall_ms) <= 10000 and 2 * median(A_suffix_wall_ms) <= 8000",
                "source_cell_valid": True,
                "samples_ms": p8_walls,
                "median_ms": statistics.median(p8_walls),
                "maximum_ms": max(p8_walls),
                "allowed": p16_c128_allowed,
                "recorded_at": now(),
            },
        )
        if not p16_c128_allowed:
            chronology["safety_stop"] = "before P16384/C128"
        else:
            p16 = run_recorded_child(
                chronology,
                binary,
                expected_identity,
                environment,
                environment_record,
                16_384,
                128,
            )
            p16_walls = legacy_suffix_walls(p16)
            safety_before = system_snapshot()
            pressure_and_swap_before = safety_before["memory_pressure_normal"]
            timing_allowed = (
                max(p16_walls) <= 10_000.0
                and 8.0 * statistics.median(p16_walls) <= 40_000.0
            )
            admission_ok = False
            safety_after: dict[str, Any] | None = None
            safety_record: dict[str, Any] = {
                "predicate": "max(A_suffix_wall_ms) <= 10000 and 8 * median(A_suffix_wall_ms) <= 40000, fresh admission succeeds, pressure normal, swap does not increase",
                "source_cell_valid": True,
                "samples_ms": p16_walls,
                "median_ms": statistics.median(p16_walls),
                "maximum_ms": max(p16_walls),
                "timing_allowed": timing_allowed,
                "environment_before_admission": safety_before,
                "environment_after_admission": None,
                "fresh_admission_ok": False,
                "allowed": False,
                "state": "admission_required"
                if timing_allowed and pressure_and_swap_before
                else "stopped",
                "recorded_at": now(),
            }
            record_safety(chronology, "before_p16384_c1024", safety_record)
            if timing_allowed and pressure_and_swap_before:
                run_recorded_child(
                    chronology,
                    binary,
                    expected_identity,
                    environment,
                    environment_record,
                    16_384,
                    1024,
                    preflight=True,
                )
                safety_after = system_snapshot()
                admission_ok = (
                    safety_after["memory_pressure_normal"]
                    and safety_after["swap_used_bytes"]
                    <= safety_before["swap_used_bytes"]
                )
            p16_c1024_allowed = (
                timing_allowed and pressure_and_swap_before and admission_ok
            )
            safety_record.update(
                {
                    "environment_after_admission": safety_after,
                    "fresh_admission_ok": admission_ok,
                    "allowed": p16_c1024_allowed,
                    "state": "applied",
                    "applied_at": now(),
                }
            )
            record_safety(chronology, "before_p16384_c1024", safety_record)
            if p16_c1024_allowed:
                run_recorded_child(
                    chronology,
                    binary,
                    expected_identity,
                    environment,
                    environment_record,
                    16_384,
                    1024,
                )
            else:
                chronology["safety_stop"] = "before P16384/C1024"
        chronology["environment_after"] = provenance()
        chronology["completed_at"] = now()
        chronology["state"] = "completed"
        atomic_json(CHRONOLOGY, chronology)
    except Exception as error:
        chronology["state"] = "invalidated"
        chronology["failure"] = {
            "at": now(),
            "detail": f"{type(error).__name__}: {error}",
        }
        try:
            chronology["environment_after"] = provenance()
        except Exception as postflight_error:
            chronology["postflight_failure"] = (
                f"{type(postflight_error).__name__}: {postflight_error}"
            )
        chronology["completed_at"] = now()
        atomic_json(CHRONOLOGY, chronology)
        raise
    invoke_analyzer()


if __name__ == "__main__":
    main()
