#!/usr/bin/env python3
"""Sealed v0.653 A10B copied versus parallel-pread mechanism floor.

The self-test path is synthetic and must remain free of model and qwen-bench access.
"""

import argparse
import ctypes
import hashlib
import json
import math
import mmap
import os
import re
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNNER = Path(__file__).resolve()
CONTRACT = ROOT / "docs/bench/v0653-a10b-parallel-pread-floor.md"
DESCRIBE = ROOT / "docs/bench/v0653-a10b-parallel-pread-floor.describe.json"
BINARY = "./target/release/qwen-bench"
PACKET = ROOT / "target/profiles/v0653-a10b-parallel-pread-floor-p1"
WORK = ROOT / "target/profiles/v0653-a10b-parallel-pread-floor-work"
MODEL_DIR = Path("/Users/tito/models/unsloth-Qwen3.5-122B-A10B-GGUF/UD-Q4_K_XL")
MODEL_NAMES = (
    "Qwen3.5-122B-A10B-UD-Q4_K_XL-00001-of-00003.gguf",
    "Qwen3.5-122B-A10B-UD-Q4_K_XL-00002-of-00003.gguf",
    "Qwen3.5-122B-A10B-UD-Q4_K_XL-00003-of-00003.gguf",
)
MODELS = tuple(MODEL_DIR / name for name in MODEL_NAMES)
MODEL = MODELS[0]
MODEL_SHA256 = (
    "467c9bd92ea518539cf75bf5a5fbfbd35e9a0b40d766ccaa67bf120e12041df3",
    "ecdbd42d43b0df9fa0ef9a584e09e95a43966ef03a122aba0b87a99d44d9ad98",
    "13300e0f059e6fa21aa0fabde2a554f9deea366c0e54f268045769b214b28c97",
)
MODEL_SIZES = (10_943_552, 49_640_779_424, 27_378_273_056)
MODEL_PAGES = (668, 3_029_833, 1_671_038)
PAGE_SIZE = 16_384
PROFILE = "a10b-q4xl-v1"
DESCRIPTOR = "0x3eb290915bec2041"
INVENTORY = "b331c475123dbee3bc862a495266dee3996c5f3adabcd6fbeaff9bbabd71a4f8"
COPY_BYTES = 77_018_996_736
COUNT = 879
HEADROOM = 85_608_931_328
DESCRIBE_SHA256 = "ce1b3ccfd67a1a5b8cdaf71050dfd9547ec4ca06f29559da1b4a19473a0cdef9"
ORDER = ("A", "B", "B", "A")
COOLDOWN_NS = 120_000_000_000
LAUNCH_LIMIT_NS = 5_000_000_000
MAX_OUTPUT = 4 * 1024 * 1024
READ_SIZE = 8 * 1024 * 1024
PROBE_DEADLINE_S = 300
CHILD_DEADLINE_S = 1_800
IDENTITY_DEADLINE_S = 300
PIPE_JOIN_S = 2.0
ENDPOINT = "host-population/no-GPU-command"
IMPLEMENTATION_SEAL = "gguf-arena-floor-copied-pread-v1"
PROBE_ENDPOINT = "metal-memory-headroom/no-model-open/no-GPU-command"
SAFE_ENV_KEYS = {
    "HOME",
    "PATH",
    "TMPDIR",
    "USER",
    "LOGNAME",
    "SHELL",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "LC_MESSAGES",
}
FORBIDDEN_ENV_PREFIXES = (
    "QWEN_",
    "METAL_",
    "MTL_",
    "DYLD_",
    "Malloc",
    "MALLOC_",
    "OMP_",
    "MKL_",
    "VECLIB_",
    "BLAS_",
    "ACCELERATE_",
    "RAYON_",
    "TOKIO_",
)
COUNTER_ERROR_PREFIXES = (
    "Error: getrusage",
    "Error: proc_pid_rusage v4",
    "Error: getrusage time overflow",
    "Error: user CPU time regressed",
    "Error: system CPU time regressed",
    "Error: total CPU time overflow",
    "Error: minor-fault delta overflow",
    "Error: major-fault delta overflow",
    "Error: block-input delta overflow",
    "Error: swap delta overflow",
    "Error: getrusage counter regressed:",
    "Error: A10B getrusage counter regressed:",
    "Error: proc instructions regressed",
    "Error: proc cycles regressed",
    "Error: proc billed energy regressed",
    "Error: proc serviced energy regressed",
    "Error: duration does not fit u64 microseconds",
    "Error: subinterval duration overflow",
    "Error: subintervals exceed ready wall",
    "Error: partial subinterval timing is invalid",
    "Error: parallel-pread phase duration overflow",
    "Error: parallel-pread phase timing does not reconcile",
    "Error: ready wall is zero",
    "Error: CPU per wall is invalid",
)
RUST_ERROR_FIXTURES = (
    b"Error: getrusage\n\nCaused by:\n    fixture\n",
    b"Error: proc_pid_rusage v4\n\nCaused by:\n    fixture\n",
    b"Error: getrusage time overflow\n",
    b"Error: user CPU time regressed\n",
    b"Error: system CPU time regressed\n",
    b"Error: total CPU time overflow\n",
    b"Error: minor-fault delta overflow\n",
    b"Error: major-fault delta overflow\n",
    b"Error: block-input delta overflow\n",
    b"Error: swap delta overflow\n",
    b"Error: proc instructions regressed\n",
    b"Error: proc cycles regressed\n",
    b"Error: proc billed energy regressed\n",
    b"Error: proc serviced energy regressed\n",
    b"Error: parallel-pread phase duration overflow\n",
    b"Error: parallel-pread phase timing does not reconcile\n",
    b"Error: subinterval duration overflow\n",
    b"Error: subintervals exceed ready wall\n",
    b"Error: partial subinterval timing is invalid\n",
)
OPERATOR_SIGNALS = []
SIGNAL_CONTROLLER = None
AUTHORITY_ENV = None

ARCHITECTURE = {
    "kind": "moe",
    "n_layer": 48,
    "hidden_size": 3072,
    "intermediate_size": 0,
    "vocab_size": 248_320,
    "full_attention_interval": 4,
    "n_q_heads": 32,
    "n_kv_heads": 2,
    "attn_head_dim": 256,
    "rope_theta": 10_000_000.0,
    "partial_rotary_factor": 0.25,
    "gdn_n_v_heads": 64,
    "gdn_n_k_heads": 16,
    "gdn_head_dim": 128,
    "gdn_conv_kernel": 4,
    "expert_count": 256,
    "expert_used_count": 8,
    "expert_feed_forward_length": 1024,
    "expert_shared_feed_forward_length": 1024,
    "mtp_n_hidden_layers": 0,
}
RESOURCE_MODES = {
    "creation_storage": "shared",
    "creation_cpu_cache": "default_cache",
    "creation_hazard_tracking": "default",
    "observed_storage": "shared",
    "observed_cpu_cache": "default_cache",
    "observed_hazard_tracking": "tracked",
}
SCHEDULE = {
    "algorithm": "minimax-contiguous-v1",
    "workers": 4,
    "cuts": [214, 435, 658],
    "task_counts": [214, 221, 223, 221],
    "worker_bytes": [19_474_295_808, 19_228_744_704, 19_231_902_720, 19_084_053_504],
    "max_to_ideal": 1.011402206380462,
    "max_to_min": 1.0204486066819194,
    "partitions": [
        (
            0,
            214,
            19_474_295_808,
            (2, "output.weight", 1, 35_488, 810_516_480),
            (220, "blk.11.ffn_down_exps.weight", 1, 18_920_683_168, 553_648_128),
        ),
        (
            214,
            435,
            19_228_744_704,
            (213, "blk.11.ffn_down_shexp.weight", 1, 19_474_331_296, 3_342_336),
            (437, "blk.23.ffn_gate_exps.weight", 1, 38_250_091_168, 452_984_832),
        ),
        (
            435,
            658,
            19_231_902_720,
            (436, "blk.23.ffn_gate_inp.weight", 1, 38_703_076_000, 3_145_728),
            (657, "blk.35.ffn_up_exps.weight", 2, 7_841_234_720, 452_984_832),
        ),
        (
            658,
            879,
            19_084_053_504,
            (650, "blk.35.ffn_up_shexp.weight", 2, 8_294_219_552, 3_342_336),
            (867, "blk.47.post_attention_norm.weight", 2, 27_378_260_768, 12_288),
        ),
    ],
}

TOP_KEYS = {
    "schema_version",
    "arm",
    "profile",
    "model",
    "architecture",
    "architecture_tuple",
    "tied_embeddings",
    "mtp_present",
    "shard_mapped_lengths",
    "descriptor_layout_digest",
    "inventory_digest",
    "native_quant_embedding",
    "native_quant_embedding_supported",
    "native_quant_embedding_selection",
    "page_size",
    "required_alignment",
    "max_buffer_length",
    "device_name",
    "unified_memory",
    "request_count",
    "resource_count",
    "binding_count",
    "logical_copy_bytes",
    "physical_copy_bytes",
    "resource_modes",
    "parallel_copy_schedule",
    "timing",
    "throughput",
    "rusage",
    "proc_rusage_v4",
    "metal_allocated_bytes",
    "correctness",
    "worker_count",
    "build_identity",
    "embedding_policy",
    "memory_admission",
    "retained_shard_stamps",
    "endpoint",
    "implementation_seal",
}
TIMING_KEYS = {
    "ready_wall_ms",
    "ready_us",
    "allocation_wall_ms",
    "allocation_us",
    "source_resolution_wall_ms",
    "source_us",
    "source_resolution_us",
    "copy_wall_ms",
    "copy_us",
    "binding_wall_ms",
    "binding_us",
    "unattributed_wall_ms",
    "unattributed_us",
    "teardown_wall_ms",
    "teardown_us",
}
RUSAGE_KEYS = {
    "timer_minor_faults",
    "timer_major_faults",
    "timer_block_inputs",
    "timer_swaps",
    "user_cpu_us",
    "system_cpu_us",
    "total_cpu_us",
    "cpu_per_wall",
}
PROC_KEYS = {
    "instructions_delta_raw",
    "cycles_delta_raw",
    "billed_energy_delta_raw",
    "serviced_energy_delta_raw",
}
TIME_LABELS = (
    "maximum resident set size",
    "average shared memory size",
    "average unshared data size",
    "average unshared stack size",
    "page reclaims",
    "page faults",
    "swaps",
    "block input operations",
    "block output operations",
    "messages sent",
    "messages received",
    "signals received",
    "voluntary context switches",
    "involuntary context switches",
    "instructions retired",
    "cycles elapsed",
    "peak memory footprint",
)
ATTEMPT_KEYS = {
    "schema",
    "position",
    "pair",
    "pair_order",
    "arm",
    "stem",
    "command",
    "environment_sha256",
    "pid",
    "pgid",
    "spawn_error",
    "ownership_error",
    "returncode",
    "reaped",
    "interrupted",
    "output_overflow",
    "drain_errors",
    "timed_out",
    "operator_interrupted",
    "cleanup_actions",
    "cleanup_errors",
    "operator_signals",
    "signal_start_sequence",
    "signal_end_sequence",
    "signal_events",
    "group_disposition_initial",
    "group_disposition",
    "launch_attempted_monotonic_ns",
    "launch_acquired_monotonic_ns",
    "completion_monotonic_ns",
    "residency_to_acquired_ns",
    "terminal_activity_monotonic_ns",
    "process_resources",
    "result",
    "parse_error",
    "defect_reasons",
    "inconclusive_reasons",
    "stop_classification",
    "stop_reason",
    "artifacts",
    "bundle_sha256",
}
DECISION_KEYS = {
    "schema",
    "status",
    "authority",
    "completed_attempts",
    "expected_attempts",
    "contract_error",
    "inconclusive_error",
    "analysis",
    "closure",
    "manifest_sha256",
    "headroom_probe_sha256",
    "final_identity",
    "signal_cutoff_sha256",
    "signal_log_sha256",
    "subordinate_evidence",
}


class ContractDefect(RuntimeError):
    pass


class Inconclusive(RuntimeError):
    pass


class HeadroomFailure(RuntimeError):
    pass


class PublicationFailure(RuntimeError):
    pass


def install_operator_signal_handlers():
    def record(signum, _frame):
        OPERATOR_SIGNALS.append(
            {
                "sequence": len(OPERATOR_SIGNALS) + 1,
                "signal": signum,
                "monotonic_ns": time.monotonic_ns(),
            }
        )

    signal.signal(signal.SIGINT, record)
    signal.signal(signal.SIGTERM, record)


class SignalController:
    def __init__(self):
        self.blocked = False

    def install(self):
        install_operator_signal_handlers()

    def block(self):
        if not self.blocked:
            signal.pthread_sigmask(signal.SIG_BLOCK, (signal.SIGINT, signal.SIGTERM))
            self.blocked = True

    def unblock(self):
        if self.blocked:
            signal.pthread_sigmask(signal.SIG_UNBLOCK, (signal.SIGINT, signal.SIGTERM))
            self.blocked = False

    def final_cutoff(self, packet):
        self.block()
        boundary = time.monotonic_ns()
        pending = sorted(
            int(item)
            for item in signal.sigpending()
            if item in (signal.SIGINT, signal.SIGTERM)
        )
        events = [dict(row) for row in OPERATOR_SIGNALS]
        log = {
            "schema": 1,
            "boundary_monotonic_ns": boundary,
            "event_count": len(events),
            "events": events,
            "pending_signals": pending,
            "authority_signal_count": len(events) + len(pending),
        }
        log_path = packet / "signal-log.json"
        write_json(log_path, log)
        cutoff = {
            "schema": 1,
            "event": "final-signal-cutoff",
            "boundary_monotonic_ns": boundary,
            "signal_log_sha256": sha_file(log_path),
            "authority_signal_count": log["authority_signal_count"],
        }
        write_json(packet / "signal-cutoff.json", cutoff)
        fsync_dir(packet)
        return cutoff


def require(condition, message):
    if not condition:
        raise ContractDefect(message)


def uint(value, label, expected=None, positive=False):
    require(
        type(value) is int and value >= (1 if positive else 0),
        f"{label} is not an unsigned integer",
    )
    if expected is not None:
        require(value == expected, f"{label} drifted")
    return value


def finite(value, label, positive=False):
    require(
        type(value) in (int, float) and not isinstance(value, bool),
        f"{label} is not numeric",
    )
    parsed = float(value)
    require(
        math.isfinite(parsed) and (parsed > 0 if positive else parsed >= 0),
        f"{label} is invalid",
    )
    return parsed


def typed_equal(actual, expected, label):
    require(type(actual) is type(expected), f"{label} type drifted")
    if isinstance(expected, dict):
        require(set(actual) == set(expected), f"{label} keys drifted")
        for key in expected:
            typed_equal(actual[key], expected[key], f"{label}.{key}")
    elif isinstance(expected, list):
        require(len(actual) == len(expected), f"{label} length drifted")
        for index, (left, right) in enumerate(zip(actual, expected)):
            typed_equal(left, right, f"{label}[{index}]")
    else:
        require(actual == expected, f"{label} value drifted")


def json_bytes(value, pretty=False):
    kwargs = {"sort_keys": True, "ensure_ascii": True}
    kwargs.update(indent=2) if pretty else kwargs.update(separators=(",", ":"))
    return (json.dumps(value, **kwargs) + "\n").encode("ascii")


def parse_json_bytes(raw, label):
    def pairs(items):
        value = {}
        for key, item in items:
            if key in value:
                raise ValueError(f"duplicate key {key!r}")
            value[key] = item
        return value

    def constant(value):
        raise ValueError(f"non-finite constant {value}")

    def floating(value):
        parsed = float(value)
        if not math.isfinite(parsed):
            raise ValueError(f"non-finite float {value}")
        return parsed

    try:
        text = raw.decode("utf-8")
        decoder = json.JSONDecoder(
            object_pairs_hook=pairs, parse_constant=constant, parse_float=floating
        )
        value, end = decoder.raw_decode(text)
        if text[end:].strip():
            raise ValueError("trailing bytes")
        return value
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ContractDefect(f"{label} malformed JSON: {error}") from error


def sha_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


def sha_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fsync_dir(path):
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_exclusive(path, raw):
    try:
        with path.open("xb") as output:
            output.write(raw)
            output.flush()
            os.fsync(output.fileno())
    except OSError as error:
        raise PublicationFailure(f"durable write failed for {path}: {error}") from error


def write_json(path, value):
    write_exclusive(path, json_bytes(value, True))


def append_jsonl(path, value):
    try:
        with path.open("ab") as output:
            output.write(json_bytes(value))
            output.flush()
            os.fsync(output.fileno())
    except OSError as error:
        raise PublicationFailure(
            f"durable append failed for {path}: {error}"
        ) from error


def normalized_environment(source=None):
    inherited = dict(os.environ if source is None else source)
    removed = sorted(key for key in inherited if key not in SAFE_ENV_KEYS)
    env = {key: inherited[key] for key in sorted(SAFE_ENV_KEYS) if key in inherited}
    forbidden = [
        key
        for key in env
        if key == "RUST_LOG" or key.startswith(FORBIDDEN_ENV_PREFIXES)
    ]
    require(not forbidden, f"forbidden environment survived: {forbidden}")
    for key in ("HOME", "PATH", "TMPDIR"):
        require(key in env and env[key], f"normalized environment lacks {key}")
    require("QWEN_NATIVE_QUANT_EMBED" not in env, "native embedding override survived")
    record = {"environment": env, "removed_keys": removed}
    record["sha256"] = sha_bytes(json_bytes(record))
    return env, record


def child_command(arm):
    label = "copied" if arm == "A" else "parallel-pread"
    return [
        "/usr/bin/time",
        "-l",
        str(BINARY),
        "gguf-arena-floor",
        "--model",
        str(MODEL),
        "--profile",
        PROFILE,
        "--arm",
        label,
        "--embedding-policy",
        "force-native-if-supported",
        "--workers",
        "4",
        "--output",
        "json",
    ]


def probe_command():
    return [
        str(BINARY),
        "gguf-arena-floor",
        "--model",
        str(MODEL),
        "--headroom-probe",
        "--output",
        "json",
    ]


def complete_stamp(value):
    return {
        "device": value.st_dev,
        "inode": value.st_ino,
        "size": value.st_size,
        "mtime_sec": value.st_mtime_ns // 1_000_000_000,
        "mtime_nsec": value.st_mtime_ns % 1_000_000_000,
        "ctime_sec": value.st_ctime_ns // 1_000_000_000,
        "ctime_nsec": value.st_ctime_ns % 1_000_000_000,
    }


def path_stamp(path):
    before = os.lstat(path)
    require(
        stat.S_ISREG(before.st_mode)
        and before.st_nlink == 1
        and not stat.S_ISLNK(before.st_mode),
        f"source is not a regular non-symlink: {path}",
    )
    return complete_stamp(before)


def path_stamps():
    return [{"path": str(path), **path_stamp(path)} for path in MODELS]


def condition_sources():
    require(os.sysconf("SC_PAGE_SIZE") == PAGE_SIZE, "host page size drifted")
    descriptors = []
    records = []
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
    libc.mincore.restype = ctypes.c_int
    libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
    libc.munmap.restype = ctypes.c_int
    libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    try:
        for path, size in zip(MODELS, MODEL_SIZES):
            initial = path_stamp(path)
            require(initial["size"] == size, f"source size drifted: {path}")
            descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
            opened = complete_stamp(os.fstat(descriptor))
            typed_equal(opened, initial, "opened source stamp")
            descriptors.append((path, descriptor, opened))
        # Hash all three stable descriptors before the compact residency proof.
        for (path, descriptor, opened), size, digest_expected in zip(
            descriptors, MODEL_SIZES, MODEL_SHA256
        ):
            hash_started = time.monotonic_ns()
            digest = hashlib.sha256()
            count = 0
            while True:
                chunk = os.read(descriptor, READ_SIZE)
                if not chunk:
                    break
                digest.update(chunk)
                count += len(chunk)
            require(
                count == size and digest.hexdigest() == digest_expected,
                f"source hash drifted: {path}",
            )
            typed_equal(
                complete_stamp(os.fstat(descriptor)),
                opened,
                "post-hash descriptor stamp",
            )
            records.append(
                {
                    "path": str(path),
                    "descriptor_stamp_before": opened,
                    "bytes_hashed": count,
                    "sha256": digest.hexdigest(),
                    "hash_started_monotonic_ns": hash_started,
                    "hash_completed_monotonic_ns": time.monotonic_ns(),
                }
            )
        # Prove every file through those same descriptors immediately before launch.
        for index, ((path, descriptor, opened), size, pages) in enumerate(
            zip(descriptors, MODEL_SIZES, MODEL_PAGES)
        ):
            address = libc.mmap(
                None, size, mmap.PROT_READ, mmap.MAP_SHARED, descriptor, 0
            )
            if address == ctypes.c_void_p(-1).value:
                number = ctypes.get_errno()
                raise OSError(number, os.strerror(number))
            try:
                vector = (ctypes.c_ubyte * pages)()
                if libc.mincore(address, size, vector) != 0:
                    number = ctypes.get_errno()
                    raise OSError(number, os.strerror(number))
                resident = sum(bool(item & 1) for item in vector)
            finally:
                if libc.munmap(address, size) != 0:
                    number = ctypes.get_errno()
                    raise OSError(number, os.strerror(number))
            require(resident == pages, f"source is not fully resident: {path}")
            final = complete_stamp(os.fstat(descriptor))
            typed_equal(final, opened, "post-mincore descriptor stamp")
            typed_equal(path_stamp(path), opened, "post-mincore path stamp")
            records[index].update(
                {
                    "descriptor_stamp_after": final,
                    "page_size": PAGE_SIZE,
                    "total_pages": pages,
                    "resident_pages": resident,
                    "all_pages_resident": True,
                    "residency_checked_monotonic_ns": time.monotonic_ns(),
                }
            )
        proved = time.monotonic_ns()
        return records, proved
    finally:
        for _path, descriptor, _opened in descriptors:
            os.close(descriptor)


def bounded_utility(command, deadline_s=30, env=None):
    if env is None:
        env = AUTHORITY_ENV or {
            key: os.environ[key] for key in SAFE_ENV_KEYS if key in os.environ
        }
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
        env=env,
    )
    pgid = os.getpgid(process.pid)
    if pgid != process.pid:
        os.kill(process.pid, signal.SIGKILL)
        raise RuntimeError("utility process group authentication failed")
    try:
        stdout, stderr = process.communicate(timeout=deadline_s)
    except subprocess.TimeoutExpired:
        os.killpg(pgid, signal.SIGKILL)
        try:
            stdout, stderr = process.communicate(timeout=PIPE_JOIN_S)
        except subprocess.TimeoutExpired as error:
            raise RuntimeError("utility reap timeout") from error
        raise RuntimeError("utility deadline exceeded")
    if process.returncode or stderr or len(stdout) > MAX_OUTPUT:
        raise RuntimeError(f"utility command failed: {command!r}")
    return stdout


def raw_command(command):
    return bounded_utility(command).decode("utf-8")


def vm_state():
    errors = []
    values = {}
    try:
        text = raw_command(["/usr/bin/vm_stat"])
        for key, label in (
            ("pageouts", "Pageouts"),
            ("compressions", "Compressions"),
            ("swapouts", "Swapouts"),
            ("compressor_stored_pages", "Pages stored in compressor"),
            ("compressor_occupied_pages", "Pages occupied by compressor"),
        ):
            match = re.search(rf"^{re.escape(label)}:\s+(\d+)\.$", text, re.M)
            if not match:
                raise RuntimeError(f"cannot parse {label}")
            values[key] = int(match.group(1))
    except Exception as error:
        errors.append(f"vm_stat:{type(error).__name__}:{error}")
    try:
        text = raw_command(["/usr/sbin/sysctl", "-n", "vm.swapusage"])
        match = re.search(r"\bused\s*=\s*([0-9.]+)([BKMGT])", text)
        if not match:
            raise RuntimeError("cannot parse swap occupancy")
        values["swap_used_bytes"] = round(
            float(match.group(1))
            * {"B": 1, "K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4}[
                match.group(2)
            ]
        )
    except Exception as error:
        errors.append(f"swap:{type(error).__name__}:{error}")
    for key in (
        "pageouts",
        "compressions",
        "swapouts",
        "compressor_stored_pages",
        "compressor_occupied_pages",
        "swap_used_bytes",
    ):
        values.setdefault(key, None)
    return {**values, "errors": errors, "captured_monotonic_ns": time.monotonic_ns()}


def vm_interval(label, before, after):
    keys = (
        "pageouts",
        "compressions",
        "swapouts",
        "compressor_stored_pages",
        "compressor_occupied_pages",
        "swap_used_bytes",
    )
    deltas = {
        key: after.get(key) - before.get(key)
        if type(before.get(key)) is int and type(after.get(key)) is int
        else None
        for key in keys
    }
    defects = []
    invalid = []
    if before.get("errors") or after.get("errors"):
        defects.append(f"{label}_capture_failed")
    for key, value in deltas.items():
        if value is None:
            defects.append(f"{label}_{key}_unavailable")
        elif key in ("pageouts", "compressions", "swapouts") and value < 0:
            defects.append(f"{label}_{key}_regressed")
    if type(deltas["swapouts"]) is int and deltas["swapouts"] != 0:
        invalid.append(f"{label}_swapouts_grew")
    for key in (
        "swap_used_bytes",
        "compressor_stored_pages",
        "compressor_occupied_pages",
    ):
        if type(deltas[key]) is int and deltas[key] > 0:
            invalid.append(f"{label}_{key}_grew")
    return {
        "label": label,
        "before": before,
        "after": after,
        "deltas": deltas,
        "defect_reasons": defects,
        "inconclusive_reasons": invalid,
        "advisory": {
            "pageouts_delta": deltas["pageouts"],
            "compressions_delta": deltas["compressions"],
        },
    }


def validate_vm_snapshot(value, label):
    require(
        isinstance(value, dict)
        and set(value)
        == {
            "pageouts",
            "compressions",
            "swapouts",
            "compressor_stored_pages",
            "compressor_occupied_pages",
            "swap_used_bytes",
            "errors",
            "captured_monotonic_ns",
        },
        f"{label} VM schema drifted",
    )
    for key in (
        "pageouts",
        "compressions",
        "swapouts",
        "compressor_stored_pages",
        "compressor_occupied_pages",
        "swap_used_bytes",
    ):
        require(
            value[key] is None or (type(value[key]) is int and value[key] >= 0),
            f"{label} VM {key} type drifted",
        )
    require(
        isinstance(value["errors"], list)
        and all(isinstance(item, str) for item in value["errors"]),
        f"{label} VM errors drifted",
    )
    uint(value["captured_monotonic_ns"], f"{label} VM timestamp", positive=True)


def host_state():
    errors = []
    values = {}
    for key, command in (
        ("thermal", ["/usr/bin/pmset", "-g", "therm"]),
        ("battery", ["/usr/bin/pmset", "-g", "batt"]),
        ("memory_pressure", ["/usr/bin/memory_pressure", "-Q"]),
    ):
        try:
            values[key] = raw_command(command)
        except Exception as error:
            values[key] = None
            errors.append(f"{key}:{type(error).__name__}:{error}")
    match = re.search(
        r"System-wide memory free percentage: (\d+)%",
        values.get("memory_pressure") or "",
    )
    available = int(match.group(1)) if match else None
    competitors = []
    process_snapshot = None
    ignored_process_ids = sorted({os.getpid(), os.getppid()})
    try:
        own = set(ignored_process_ids)
        text = raw_command(["/bin/ps", "-axo", "pid=,command="])
        process_snapshot = text
        executable = re.compile(r"(?:^|/)(?:qwen|qwen-bench|llama[^/ ]*)(?:\s|$)", re.I)
        for line in text.splitlines():
            match = re.match(r"\s*(\d+)\s+(.*)", line)
            if (
                match
                and int(match.group(1)) not in own
                and (
                    executable.search(match.group(2))
                    or any(
                        name.lower() in match.group(2).lower() for name in MODEL_NAMES
                    )
                )
            ):
                competitors.append(
                    {"pid": int(match.group(1)), "command": match.group(2)}
                )
    except Exception as error:
        competitors = None
        errors.append(f"processes:{type(error).__name__}:{error}")
    valid = (
        not errors
        and "AC Power" in (values.get("battery") or "")
        and "No thermal warning level has been recorded"
        in (values.get("thermal") or "")
        and "No performance warning level has been recorded"
        in (values.get("thermal") or "")
        and type(available) is int
        and available >= 50
        and competitors == []
    )
    return {
        **values,
        "memory_available_percent": available,
        "process_snapshot": process_snapshot,
        "ignored_process_ids": ignored_process_ids,
        "competing_processes": competitors,
        "errors": errors,
        "valid": valid,
        "captured_monotonic_ns": time.monotonic_ns(),
    }


def validate_build_identity(value):
    require(isinstance(value, dict), "build identity missing")
    require(
        value.get("status") == "match"
        and value.get("build_dirty") is False
        and value.get("runtime_dirty") is False
        and value.get("build_commit") == value.get("runtime_commit")
        and value.get("build_source_state") == value.get("runtime_source_state"),
        "build/runtime identity is not clean and equal",
    )
    return value


def validate_admission(value, label):
    require(
        isinstance(value, dict)
        and set(value)
        == {
            "admitted",
            "reason",
            "required_bytes",
            "scratch_upper_bytes",
            "reserve_bytes",
            "allow_zero_process_budget",
            "zero_process_budget_semantics",
            "working_set_headroom_bytes",
            "signals",
        },
        f"{label} schema drifted",
    )
    uint(value["required_bytes"], f"{label} required", HEADROOM)
    uint(value["scratch_upper_bytes"], f"{label} scratch", HEADROOM)
    uint(value["reserve_bytes"], f"{label} reserve", 0)
    require(
        value["allow_zero_process_budget"] is True
        and value["zero_process_budget_semantics"] == "omitted-limit-sentinel",
        f"{label} zero sentinel semantics drifted",
    )
    signals = value["signals"]
    require(
        isinstance(signals, dict)
        and set(signals)
        == {
            "recommended_max_working_set_size",
            "current_allocated_size",
            "working_set_headroom_bytes",
            "process_limit_remaining_bytes",
        },
        f"{label} signals schema drifted",
    )
    recommended = uint(
        signals["recommended_max_working_set_size"],
        f"{label} recommended",
        positive=True,
    )
    allocated = uint(signals["current_allocated_size"], f"{label} allocated")
    headroom = uint(signals["working_set_headroom_bytes"], f"{label} headroom")
    require(
        recommended >= allocated
        and headroom == recommended - allocated
        and value["working_set_headroom_bytes"] == headroom,
        f"{label} working-set arithmetic failed",
    )
    process = signals["process_limit_remaining_bytes"]
    uint(process, f"{label} process limit")
    require(
        type(value["admitted"]) is bool
        and isinstance(value["reason"], str)
        and value["reason"],
        f"{label} decision fields drifted",
    )
    working_fits = headroom >= HEADROOM
    process_fits = process == 0 or process >= HEADROOM
    sufficient = working_fits and process_fits
    if sufficient:
        expected_reason = (
            "admitted_process_budget_omitted"
            if process == 0
            else "admitted_with_process_budget"
        )
    elif not working_fits and process > 0 and not process_fits:
        expected_reason = "both_insufficient"
    elif not working_fits:
        expected_reason = "working_set_insufficient"
    else:
        expected_reason = "process_insufficient"
    require(
        value["admitted"] is sufficient and value["reason"] == expected_reason,
        f"{label} admission decision does not match raw signals",
    )
    if not sufficient:
        raise HeadroomFailure(f"{label} failed the frozen headroom requirement")
    return value


def validate_probe(value, build):
    require(
        isinstance(value, dict)
        and set(value)
        == {
            "schema_version",
            "mode",
            "model_label",
            "endpoint",
            "model_opened",
            "payload_allocated",
            "device_name",
            "unified_memory",
            "max_buffer_length",
            "memory_admission",
            "build_identity",
        },
        "headroom probe key set drifted",
    )
    uint(value["schema_version"], "probe schema", 1)
    require(
        value["mode"] == "metal-memory-headroom-probe"
        and value["model_label"] == str(MODEL)
        and value["endpoint"] == PROBE_ENDPOINT
        and value["model_opened"] is False
        and value["payload_allocated"] is False
        and value["device_name"] == "Apple M4 Max"
        and value["unified_memory"] is True
        and value["max_buffer_length"] == 77_309_411_328,
        "headroom probe scope or host drifted",
    )
    typed_equal(value["build_identity"], build, "probe build identity")
    validate_build_identity(value["build_identity"])
    validate_admission(value["memory_admission"], "probe admission")
    return value


def validate_tracked_describe(value):
    require(isinstance(value, dict), "tracked describe is not an object")
    require(
        value.get("schema_version") == 2
        and value.get("mode") == "describe"
        and value.get("model") == str(MODEL)
        and value.get("matched_profile") is None
        and value.get("materialization_supported") is False
        and value.get("embedding_policy") == "force-native-if-supported"
        and value.get("native_quant_embedding_selection")
        == "bench-force-native-if-supported"
        and value.get("architecture") == "qwen35moe"
        and value.get("descriptor_layout_digest") == DESCRIPTOR
        and value.get("inventory_digest") == INVENTORY
        and value.get("request_count") == COUNT
        and value.get("logical_copy_bytes") == COPY_BYTES,
        "tracked describe identity drifted",
    )
    typed_equal(
        value.get("architecture_tuple"), ARCHITECTURE, "tracked describe architecture"
    )
    typed_equal(
        value.get("shard_mapped_lengths"),
        list(MODEL_SIZES),
        "tracked describe mapped lengths",
    )
    typed_equal(
        value.get("computed_schedule"),
        schedule_json(),
        "tracked describe computed schedule",
    )
    typed_equal(
        value.get("parallel_copy_schedule"),
        schedule_json(),
        "tracked describe parallel schedule",
    )
    validate_build_identity(value.get("build_identity"))
    return value


def schedule_json():
    partitions = []
    for start, end, size, first, last in SCHEDULE["partitions"]:

        def endpoint(item):
            index, name, shard, offset, length = item
            return {
                "request_index": index,
                "name": name,
                "shard_idx": shard,
                "source_offset": offset,
                "n_bytes": length,
            }

        partitions.append(
            {
                "start": start,
                "end": end,
                "task_count": end - start,
                "bytes": size,
                "first_shard": first[2],
                "first_source_offset": first[3],
                "last_shard": last[2],
                "last_source_offset": last[3],
                "first": endpoint(first),
                "last": endpoint(last),
            }
        )
    return {
        "algorithm": SCHEDULE["algorithm"],
        "workers": SCHEDULE["workers"],
        "cuts": list(SCHEDULE["cuts"]),
        "task_counts": list(SCHEDULE["task_counts"]),
        "worker_bytes": list(SCHEDULE["worker_bytes"]),
        "max_to_ideal": SCHEDULE["max_to_ideal"],
        "max_to_min": SCHEDULE["max_to_min"],
        "partitions": partitions,
    }


def validate_schedule(value):
    typed_equal(value, schedule_json(), "literal W4 schedule")


def wall_pair(timing, prefix, positive=False):
    wall = finite(timing[prefix + "_wall_ms"], prefix + " wall", positive)
    microseconds = uint(timing[prefix + "_us"], prefix + " us", positive=positive)
    require(abs(wall - microseconds / 1000) <= 0.002, f"{prefix} wall/us drifted")


def validate_stamps(value, baseline):
    require(
        isinstance(value, dict)
        and set(value) == {"before_timing", "after_verification"},
        "child stamp report drifted",
    )
    expected = []
    for index, row in enumerate(baseline):
        expected.append(
            {"shard_idx": index, "path": row["path"], **row["descriptor_stamp_before"]}
        )
    typed_equal(value["before_timing"], expected, "child baseline stamps")
    typed_equal(value["after_verification"], expected, "child final stamps")


def validate_result(value, arm, build, source_records):
    require(
        isinstance(value, dict) and set(value) == TOP_KEYS,
        "A10B top-level key set drifted",
    )
    uint(value["schema_version"], "execution schema", 3)
    expected_arm = "copied" if arm == "A" else "parallel-pread"
    require(
        value["arm"] == expected_arm
        and value["profile"] == PROFILE
        and value["model"] == str(MODEL)
        and value["architecture"] == "qwen35moe"
        and value["tied_embeddings"] is False
        and value["mtp_present"] is False
        and value["descriptor_layout_digest"] == DESCRIPTOR
        and value["inventory_digest"] == INVENTORY
        and value["embedding_policy"] == "force-native-if-supported"
        and value["native_quant_embedding"] is True
        and value["native_quant_embedding_supported"] is True
        and value["native_quant_embedding_selection"]
        == "bench-force-native-if-supported"
        and value["endpoint"] == ENDPOINT,
        "execution identity drifted",
    )
    typed_equal(value["architecture_tuple"], ARCHITECTURE, "architecture tuple")
    typed_equal(value["shard_mapped_lengths"], list(MODEL_SIZES), "mapped lengths")
    uint(value["page_size"], "page size", PAGE_SIZE)
    uint(value["required_alignment"], "alignment", 32)
    uint(value["max_buffer_length"], "maximum buffer", 77_309_411_328)
    require(
        value["device_name"] == "Apple M4 Max" and value["unified_memory"] is True,
        "device geometry drifted",
    )
    for key in ("request_count", "resource_count", "binding_count"):
        uint(value[key], key, COUNT)
    for key in ("logical_copy_bytes", "physical_copy_bytes"):
        uint(value[key], key, COPY_BYTES)
    typed_equal(value["resource_modes"], RESOURCE_MODES, "resource topology")
    validate_schedule(value["parallel_copy_schedule"])
    require("blit_population" not in value, "blit population present")
    timing = value["timing"]
    require(
        isinstance(timing, dict) and set(timing) == TIMING_KEYS,
        "timing key set drifted",
    )
    for prefix in ("ready", "binding", "teardown"):
        wall_pair(timing, prefix, True)
    if arm == "A":
        for key in (
            "allocation_wall_ms",
            "allocation_us",
            "source_resolution_wall_ms",
            "source_us",
            "source_resolution_us",
            "copy_wall_ms",
            "copy_us",
            "unattributed_wall_ms",
            "unattributed_us",
        ):
            require(timing[key] is None, f"copied timing {key} must be null")
    else:
        for prefix in ("allocation", "source_resolution", "copy"):
            wall_pair(timing, prefix, True)
        wall_pair(timing, "unattributed")
    require(
        timing["source_us"] == timing["source_resolution_us"],
        "source timing aliases drifted",
    )
    if arm == "B":
        phase_sum = sum(
            timing[key]
            for key in ("allocation_us", "source_us", "copy_us", "binding_us")
        )
        require(
            abs(timing["ready_us"] - phase_sum - timing["unattributed_us"]) <= 4,
            "ready subintervals do not reconcile",
        )
    throughput = value["throughput"]
    require(
        isinstance(throughput, dict)
        and set(throughput) == {"ready_gbps_decimal", "copy_gbps_decimal"},
        "throughput schema drifted",
    )
    finite(throughput["ready_gbps_decimal"], "ready throughput", True)
    if arm == "A":
        require(
            throughput["copy_gbps_decimal"] is None,
            "copied copy throughput must be null",
        )
    else:
        finite(throughput["copy_gbps_decimal"], "copy throughput", True)
    usage = value["rusage"]
    require(
        isinstance(usage, dict) and set(usage) == RUSAGE_KEYS, "rusage key set drifted"
    )
    user = uint(usage["user_cpu_us"], "user CPU")
    system = uint(usage["system_cpu_us"], "system CPU")
    total = uint(usage["total_cpu_us"], "total CPU", positive=True)
    require(total == user + system, "timer CPU does not reconcile")
    uint(usage["timer_minor_faults"], "minor faults")
    uint(usage["timer_major_faults"], "major faults")
    uint(usage["timer_block_inputs"], "timer block inputs")
    uint(usage["timer_swaps"], "timer swaps")
    cpu_per_wall = finite(usage["cpu_per_wall"], "CPU per wall")
    require(
        math.isclose(
            cpu_per_wall, total / timing["ready_us"], rel_tol=1e-12, abs_tol=1e-12
        ),
        "CPU per wall drifted",
    )
    proc = value["proc_rusage_v4"]
    require(
        isinstance(proc, dict) and set(proc) == PROC_KEYS, "proc rusage key set drifted"
    )
    for key in PROC_KEYS:
        uint(proc[key], f"proc {key}")
    allocation = value["metal_allocated_bytes"]
    require(
        isinstance(allocation, dict)
        and set(allocation) == {"before", "ready", "after_drop", "drop_valid"},
        "Metal allocation key set drifted",
    )
    for key in ("before", "ready", "after_drop"):
        uint(allocation[key], f"Metal {key}")
    require(
        allocation["drop_valid"] is True
        and allocation["after_drop"] <= allocation["before"],
        "Metal drop invariant failed",
    )
    typed_equal(
        value["correctness"],
        {"passed": True, "payload_bytes_checked": COPY_BYTES, "entries_checked": COUNT},
        "correctness",
    )
    uint(value["worker_count"], "worker count", 0 if arm == "A" else 4)
    validate_admission(value["memory_admission"], "child admission")
    validate_stamps(value["retained_shard_stamps"], source_records)
    typed_equal(
        value["implementation_seal"],
        {
            "schema_version": 1,
            "seal": IMPLEMENTATION_SEAL,
            "scope": ["materialize_copied", "materialize_parallel_pread"],
            "no_gpu_command": True,
            "build_identity_bound": True,
        },
        "implementation seal",
    )
    typed_equal(value["build_identity"], build, "child build identity")
    validate_build_identity(value["build_identity"])
    return value


def parse_time(raw):
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ContractDefect(f"time stderr is not UTF-8: {error}") from error
    lines = text.splitlines()
    require(len(lines) == 1 + len(TIME_LABELS), "time stderr line count drifted")
    match = re.fullmatch(
        r"\s*([0-9]+\.[0-9]{2}) real\s+([0-9]+\.[0-9]{2}) user\s+"
        r"([0-9]+\.[0-9]{2}) sys\s*",
        lines[0],
    )
    require(match is not None, "time summary grammar drifted")
    real, user, system = map(float, match.groups())
    values = {}
    for line, label in zip(lines[1:], TIME_LABELS):
        found = re.fullmatch(rf"\s*(\d+)\s+{re.escape(label)}\s*", line)
        require(found is not None, f"time label grammar drifted: {label}")
        values[label.replace(" ", "_")] = int(found.group(1))
    require(
        values["maximum_resident_set_size"] > 0 and values["peak_memory_footprint"] > 0,
        "time RSS/footprint not positive",
    )
    return {
        "real_s": real,
        "user_cpu_s": user,
        "system_cpu_s": system,
        "total_cpu_s": user + system,
        **values,
    }


def process_group_members(pgid, deadline_ns=None):
    remaining = (
        1.0
        if deadline_ns is None
        else max(0.001, (deadline_ns - time.monotonic_ns()) / 1e9)
    )
    if deadline_ns is not None and time.monotonic_ns() >= deadline_ns:
        raise RuntimeError("process-group inspection deadline exceeded")
    text = bounded_utility(
        ["/bin/ps", "-axo", "pid=,pgid="], deadline_s=min(1.0, remaining)
    ).decode("utf-8")
    members = []
    for line in text.splitlines():
        match = re.fullmatch(r"\s*(\d+)\s+(\d+)\s*", line)
        if match is None:
            raise RuntimeError("malformed process-group row")
        if int(match.group(2)) == pgid:
            members.append(int(match.group(1)))
    return sorted(members)


def group_disposition(pgid, deadline_ns=None):
    if deadline_ns is None:
        deadline_ns = time.monotonic_ns() + 2_000_000_000
    samples = []
    while time.monotonic_ns() < deadline_ns:
        try:
            members, error = process_group_members(pgid, deadline_ns), None
        except Exception as failure:
            members, error = None, f"{type(failure).__name__}:{failure}"
        samples.append(
            {
                "captured_monotonic_ns": time.monotonic_ns(),
                "members": members,
                "error": error,
            }
        )
        if members == []:
            break
        time.sleep(min(0.02, max(0, (deadline_ns - time.monotonic_ns()) / 1e9)))
    if not samples:
        samples.append(
            {
                "captured_monotonic_ns": time.monotonic_ns(),
                "members": None,
                "error": "inspection-deadline-exceeded",
            }
        )
    return {
        "pgid": pgid,
        "samples": samples,
        "no_live_group_observed": samples[-1]["members"] == [],
    }


def bounded_child(
    command,
    env,
    on_acquired=None,
    deadline_s=CHILD_DEADLINE_S,
    max_output=MAX_OUTPUT,
    _interrupt_after=None,
    _pgid_getter=os.getpgid,
    _controller=None,
):
    attempted = time.monotonic_ns()
    signal_start = len(OPERATOR_SIGNALS)
    try:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
    except (OSError, subprocess.SubprocessError) as error:
        if _controller is not None:
            _controller.unblock()
        return {
            "pid": None,
            "pgid": None,
            "attempted_monotonic_ns": attempted,
            "acquired_monotonic_ns": None,
            "completed_monotonic_ns": time.monotonic_ns(),
            "spawn_error": f"{type(error).__name__}:{error}",
            "returncode": None,
            "stdout": b"",
            "stderr": b"",
            "output_overflow": False,
            "drain_errors": [],
            "interrupted": False,
            "timed_out": False,
            "operator_interrupted": False,
            "cleanup_actions": [],
            "cleanup_errors": [],
            "operator_signals": [dict(row) for row in OPERATOR_SIGNALS[signal_start:]],
            "reaped": False,
            "group_disposition": None,
            "group_disposition_initial": None,
        }
    acquired = time.monotonic_ns()
    pgid = None
    ownership_error = None
    try:
        observed = _pgid_getter(process.pid)
        if observed != process.pid:
            ownership_error = f"pgid mismatch pid={process.pid} pgid={observed}"
        else:
            pgid = observed
    except OSError as error:
        ownership_error = f"getpgid:{type(error).__name__}:{error}"
    if pgid is not None and on_acquired is not None:
        try:
            on_acquired(process.pid, pgid, acquired)
        except BaseException as error:
            ownership_error = f"acquired-publication:{type(error).__name__}:{error}"
    if _controller is not None:
        _controller.unblock()
    buffers = [bytearray(), bytearray()]
    overflow = [False, False]
    errors = []

    def drain(stream, index):
        try:
            while True:
                chunk = stream.read(65_536)
                if not chunk:
                    break
                room = max_output - len(buffers[index])
                buffers[index].extend(chunk[: max(0, room)])
                if len(chunk) > max(0, room):
                    overflow[index] = True
        except Exception as error:
            errors.append(f"pipe{index}:{type(error).__name__}:{error}")
        finally:
            try:
                stream.close()
            except Exception as error:
                errors.append(f"close{index}:{type(error).__name__}:{error}")

    threads = [
        threading.Thread(target=drain, args=(stream, index), daemon=True)
        for index, stream in enumerate((process.stdout, process.stderr))
    ]
    started_threads = []
    for index, thread in enumerate(threads):
        try:
            thread.start()
            started_threads.append(thread)
        except RuntimeError as error:
            errors.append(f"thread{index}:{type(error).__name__}:{error}")
            try:
                (process.stdout, process.stderr)[index].close()
            except (OSError, ValueError):
                pass
    interrupted = False
    timed_out = False
    cleanup_actions = []
    cleanup_errors = []
    deadline_ns = attempted + int(deadline_s * 1e9)

    def terminate(reason, force=False):
        if cleanup_actions and not force:
            return
        action = {
            "reason": reason,
            "signal": int(signal.SIGKILL),
            "target_pgid": pgid,
            "target_pid": process.pid,
            "attempted_monotonic_ns": time.monotonic_ns(),
            "succeeded": False,
            "error": None,
        }
        try:
            if pgid is not None:
                os.killpg(pgid, signal.SIGKILL)
            else:
                os.kill(process.pid, signal.SIGKILL)
            action["succeeded"] = True
        except OSError as error:
            action["error"] = f"{type(error).__name__}:{error}"
            cleanup_errors.append(action["error"])
        action["completed_monotonic_ns"] = time.monotonic_ns()
        cleanup_actions.append(action)

    if ownership_error is not None:
        terminate("process-group-authentication-failed")
    while process.poll() is None:
        if _interrupt_after is not None and time.monotonic_ns() >= attempted + int(
            _interrupt_after * 1e9
        ):
            interrupted = True
            terminate("operator-interruption")
        if len(OPERATOR_SIGNALS) > signal_start:
            interrupted = True
            terminate("operator-signal")
        if time.monotonic_ns() >= deadline_ns:
            timed_out = True
            terminate("deadline-exceeded")
        if any(overflow):
            terminate("output-overflow")
        if errors:
            terminate("output-drain-error")
        try:
            process.wait(timeout=0.05)
        except subprocess.TimeoutExpired:
            pass
        except KeyboardInterrupt:
            interrupted = True
            terminate("operator-interruption")
        except (OSError, ChildProcessError, subprocess.SubprocessError) as error:
            errors.append(f"wait:{type(error).__name__}:{error}")
            terminate("wait-error")
            break
    for thread in started_threads:
        thread.join(timeout=PIPE_JOIN_S)
    alive = [index for index, thread in enumerate(started_threads) if thread.is_alive()]
    if alive:
        errors.append(f"pipe_threads_alive:{alive}")
        terminate("bounded-pipe-join-failed")
        for stream in (process.stdout, process.stderr):
            try:
                stream.close()
            except (OSError, ValueError):
                pass
        for thread in started_threads:
            thread.join(timeout=0.2)
    if process.poll() is None:
        terminate("leader-not-reaped")
        try:
            process.wait(timeout=PIPE_JOIN_S)
        except subprocess.TimeoutExpired:
            cleanup_errors.append("leader_reap_timeout")
    disposition_deadline = deadline_ns + int(PIPE_JOIN_S * 1e9)
    disposition_value = (
        group_disposition(pgid, disposition_deadline) if pgid is not None else None
    )
    disposition_initial = disposition_value
    if pgid is not None and (
        not disposition_value["no_live_group_observed"]
        or any(sample["error"] is not None for sample in disposition_value["samples"])
    ):
        terminate("surviving-or-unverifiable-process-group", force=True)
        disposition_value = group_disposition(pgid, disposition_deadline)
    completed = time.monotonic_ns()
    result = {
        "pid": process.pid,
        "pgid": pgid,
        "attempted_monotonic_ns": attempted,
        "acquired_monotonic_ns": acquired,
        "completed_monotonic_ns": completed,
        "spawn_error": None,
        "ownership_error": ownership_error,
        "returncode": process.returncode,
        "stdout": bytes(buffers[0]),
        "stderr": bytes(buffers[1]),
        "output_overflow": any(overflow),
        "drain_errors": errors,
        "interrupted": interrupted,
        "operator_interrupted": interrupted,
        "timed_out": timed_out,
        "cleanup_actions": cleanup_actions,
        "cleanup_errors": cleanup_errors,
        "operator_signals": [dict(row) for row in OPERATOR_SIGNALS[signal_start:]],
        "reaped": process.poll() is not None,
        "group_disposition": disposition_value,
        "group_disposition_initial": disposition_initial,
    }
    return result


def cooldown(activity):
    started = time.monotonic_ns()
    eligible = activity + COOLDOWN_NS
    if eligible > started:
        time.sleep((eligible - started) / 1e9)
    completed = time.monotonic_ns()
    require(completed - activity >= COOLDOWN_NS, "cooldown was short")
    return {
        "prior_activity_monotonic_ns": activity,
        "required_interval_ns": COOLDOWN_NS,
        "eligible_monotonic_ns": eligible,
        "started_monotonic_ns": started,
        "completed_monotonic_ns": completed,
        "observed_interval_ns": completed - activity,
    }


def git_output(args, env):
    outcome = bounded_child(
        ["/usr/bin/git", *args], env, deadline_s=IDENTITY_DEADLINE_S
    )
    require(
        outcome.get("spawn_error") is None
        and outcome.get("returncode") == 0
        and outcome.get("stderr") == b""
        and not outcome.get("timed_out")
        and not outcome.get("cleanup_actions")
        and not outcome.get("drain_errors")
        and outcome.get("reaped") is True
        and outcome.get("group_disposition", {}).get("no_live_group_observed") is True,
        f"bounded git command failed: {args!r}",
    )
    return outcome["stdout"].decode("utf-8").strip()


def bounded_json_command(
    command, env, label, deadline_s=IDENTITY_DEADLINE_S, _runner=bounded_child
):
    outcome = _runner(command, env, deadline_s=deadline_s)
    if not (
        outcome.get("spawn_error") is None
        and outcome.get("returncode") == 0
        and outcome.get("stderr") == b""
        and not outcome.get("output_overflow")
        and not outcome.get("drain_errors")
        and not outcome.get("timed_out")
        and not outcome.get("cleanup_actions")
        and not outcome.get("ownership_error")
        and outcome.get("reaped") is True
        and isinstance(outcome.get("group_disposition"), dict)
        and outcome["group_disposition"].get("no_live_group_observed") is True
        and all(
            sample.get("error") is None
            for sample in outcome["group_disposition"].get("samples", [])
        )
    ):
        raise ContractDefect(f"{label} bounded command failed")
    return parse_json_bytes(outcome["stdout"], label), outcome


def source_and_build_identity(env):
    require(Path.cwd().resolve() == ROOT, "runner must execute from repository root")
    require(
        not git_output(["status", "--porcelain=v1", "--untracked-files=all"], env),
        "implementation worktree must be clean",
    )
    source_commit = git_output(["rev-parse", "HEAD"], env)
    build, _ = bounded_json_command(
        [str(BINARY), "build-info", "--output", "json"], env, "build-info"
    )
    validate_build_identity(build)
    require(
        build["build_commit"] == source_commit, "binary does not match source commit"
    )
    return source_commit, build


def stable_file_record(path):
    before = os.lstat(path)
    require(
        stat.S_ISREG(before.st_mode)
        and before.st_nlink == 1
        and not stat.S_ISLNK(before.st_mode),
        f"unsafe artifact member: {path}",
    )
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        opened = os.fstat(fd)
        require(
            (before.st_dev, before.st_ino) == (opened.st_dev, opened.st_ino),
            "artifact descriptor identity drifted",
        )
        digest = hashlib.sha256()
        count = 0
        for chunk in iter(lambda: os.read(fd, 1024 * 1024), b""):
            digest.update(chunk)
            count += len(chunk)
        require(
            complete_stamp(os.fstat(fd)) == complete_stamp(opened)
            and count == opened.st_size,
            "artifact changed while hashing",
        )
    finally:
        os.close(fd)
    require(
        complete_stamp(os.lstat(path)) == complete_stamp(before),
        "artifact path changed while hashing",
    )
    return {
        "path": str(path.relative_to(ROOT)),
        "size_bytes": count,
        "sha256": digest.hexdigest(),
        "descriptor_stamp": complete_stamp(opened),
    }


def stable_read_bytes(path):
    before = os.lstat(path)
    require(
        stat.S_ISREG(before.st_mode)
        and before.st_nlink == 1
        and not stat.S_ISLNK(before.st_mode),
        f"stable read member is unsafe: {path}",
    )
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        opened = os.fstat(descriptor)
        require(
            (before.st_dev, before.st_ino) == (opened.st_dev, opened.st_ino),
            "stable read identity drifted",
        )
        require(opened.st_nlink == 1, "stable read link count drifted")
        chunks = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        require(
            complete_stamp(os.fstat(descriptor)) == complete_stamp(opened),
            "stable read member changed",
        )
    finally:
        os.close(descriptor)
    after = os.lstat(path)
    require(
        complete_stamp(after) == complete_stamp(before) and after.st_nlink == 1,
        "stable read pathname changed",
    )
    return b"".join(chunks)


def manifest_identity(manifest):
    return {
        "source_commit": manifest["source_commit"],
        "build_identity": manifest["build_identity"],
        "binary": manifest["binary"],
        "runner": manifest["runner"],
        "contract": manifest["contract"],
        "tracked_describe": manifest["tracked_describe"],
        "model_path_stamps": manifest["model_path_stamps"],
    }


def capture_live_identity(env):
    source_commit, build = source_and_build_identity(env)
    binary_path = ROOT / str(BINARY).removeprefix("./")
    return {
        "source_commit": source_commit,
        "build_identity": build,
        "binary": stable_file_record(binary_path),
        "runner": stable_file_record(RUNNER),
        "contract": stable_file_record(CONTRACT),
        "tracked_describe": stable_file_record(DESCRIBE),
        "model_path_stamps": path_stamps(),
    }


def revalidate_live_identity(env, manifest, label):
    observed = capture_live_identity(env)
    typed_equal(observed, manifest_identity(manifest), label)
    return {
        "schema": 1,
        "label": label,
        "captured_monotonic_ns": time.monotonic_ns(),
        "identity": observed,
        "matches_manifest": True,
    }


def run_probe(env, build):
    outcome = bounded_child(probe_command(), env, deadline_s=PROBE_DEADLINE_S)
    if not (
        outcome["spawn_error"] is None
        and outcome["returncode"] == 0
        and not outcome["output_overflow"]
        and not outcome["drain_errors"]
        and not outcome["timed_out"]
        and not outcome["cleanup_actions"]
        and not outcome["ownership_error"]
        and outcome["group_disposition"]["no_live_group_observed"]
        and all(
            sample["error"] is None
            for sample in outcome["group_disposition"]["samples"]
        )
        and outcome["stderr"] == b""
    ):
        raise Inconclusive("headroom probe execution failed")
    value = validate_probe(parse_json_bytes(outcome["stdout"], "headroom probe"), build)
    record = {
        "schema": 1,
        "command": probe_command(),
        "stdout": value,
        "raw_stdout_sha256": sha_bytes(outcome["stdout"]),
        "process": {
            key: outcome[key]
            for key in (
                "pid",
                "pgid",
                "attempted_monotonic_ns",
                "acquired_monotonic_ns",
                "completed_monotonic_ns",
                "returncode",
                "reaped",
                "group_disposition",
                "ownership_error",
                "output_overflow",
                "drain_errors",
                "timed_out",
                "operator_interrupted",
                "cleanup_actions",
                "cleanup_errors",
                "operator_signals",
            )
        },
    }
    return record, outcome["stdout"]


def attempt_spec(position):
    arm = ORDER[position - 1]
    return {
        "position": position,
        "pair": 1 if position <= 2 else 2,
        "pair_order": "AB" if position <= 2 else "BA",
        "arm": arm,
        "stem": f"p{position}-{arm.lower()}",
        "command": child_command(arm),
    }


def launch_delay_within_limit(proved, acquired):
    return (
        type(proved) is int
        and type(acquired) is int
        and 0 <= acquired - proved <= LAUNCH_LIMIT_NS
    )


def classify_attempt(outcome, parse_error, defects, invalid):
    if parse_error or defects:
        return "implementation_or_contract_defect", parse_error or ";".join(defects)
    if outcome.get("spawn_error") or outcome.get("returncode") != 0 or invalid:
        reason = (
            outcome.get("spawn_error")
            or ";".join(invalid)
            or (f"child exit {outcome.get('returncode')}")
        )
        return "inconclusive", reason
    return None, None


def structured_counter_defect(raw):
    lines = raw.decode("utf-8", errors="replace").splitlines()
    return any(
        line.startswith(prefix) for line in lines for prefix in COUNTER_ERROR_PREFIXES
    )


def launched_child_outcome(command, env, acquired, controller, _runner=bounded_child):
    try:
        return _runner(
            command,
            env,
            acquired,
            deadline_s=CHILD_DEADLINE_S,
            _controller=controller,
        )
    except BaseException:
        if controller is not None:
            controller.unblock()
        raise


def require_sealable_child_outcome(outcome):
    if outcome.get("pid") is None:
        require(
            isinstance(outcome.get("spawn_error"), str)
            and outcome.get("pgid") is None
            and outcome.get("ownership_error") is None,
            "pidless outcome is not a plain spawn failure",
        )
        return
    require(
        outcome.get("ownership_error") is None
        and outcome.get("pgid") == outcome.get("pid"),
        "child process-group ownership was not authenticated",
    )
    validate_group_disposition(outcome.get("group_disposition"), outcome["pgid"])
    typed_equal(
        outcome.get("group_disposition_initial"),
        outcome.get("group_disposition"),
        "runtime initial/final group disposition",
    )


def run_one(spec, env, manifest, activity):
    signal_start = len(OPERATOR_SIGNALS)
    if OPERATOR_SIGNALS:
        raise Inconclusive("operator signal before attempt")
    cooldown_record = cooldown(activity)
    pre_attempt_identity = revalidate_live_identity(
        env, manifest, f"pre-attempt-{spec['position']}"
    )
    host_before = host_state()
    vm_before = vm_state()
    stamps_before = path_stamps()
    stamps_before_ns = time.monotonic_ns()
    invalid = []
    defects = []
    if host_before["valid"] is not True:
        invalid.append("host_invalid_before_conditioning")
    if vm_before["errors"]:
        defects.append("vm_invalid_before_conditioning")
    if defects:
        raise ContractDefect(";".join(defects))
    if invalid:
        raise Inconclusive(";".join(invalid))
    if OPERATOR_SIGNALS:
        raise Inconclusive("operator signal during conditioning")
    records, proved = condition_sources()
    typed_equal(path_stamps(), stamps_before, "conditioning source path stamps")
    host_launch = host_state()
    vm_launch = vm_state()
    stamps_launch = path_stamps()
    stamps_launch_ns = time.monotonic_ns()
    interval = vm_interval("preconditioning", vm_before, vm_launch)
    invalid.extend(interval["inconclusive_reasons"])
    defects.extend(interval["defect_reasons"])
    if host_launch["valid"] is not True:
        invalid.append("host_invalid_before_spawn")
    typed_equal(stamps_launch, stamps_before, "pre-spawn source path stamps")
    conditioning = {
        "schema": 1,
        "pre_attempt_identity": pre_attempt_identity,
        "cooldown": cooldown_record,
        "host_before_conditioning": host_before,
        "vm_before_conditioning": vm_before,
        "source_path_stamps_before": stamps_before,
        "source_path_stamps_before_monotonic_ns": stamps_before_ns,
        "source_records": records,
        "residency_proved_monotonic_ns": proved,
        "host_before_spawn": host_launch,
        "vm_before_spawn": vm_launch,
        "source_path_stamps_before_spawn": stamps_launch,
        "source_path_stamps_before_spawn_monotonic_ns": stamps_launch_ns,
        "preconditioning_interval": interval,
    }
    conditioning_path = WORK / f"{spec['stem']}.conditioning.json"
    write_json(conditioning_path, conditioning)
    fsync_dir(WORK)
    if defects:
        raise ContractDefect(";".join(defects))
    if invalid:
        raise Inconclusive(";".join(invalid))
    if OPERATOR_SIGNALS:
        raise Inconclusive("operator signal during conditioning")
    launch = {
        "schema": 1,
        "event": "launch",
        **spec,
        "environment_sha256": manifest["environment_record"]["sha256"],
        "conditioning_sha256": sha_file(conditioning_path),
        "residency_proved_monotonic_ns": proved,
        "monotonic_ns": time.monotonic_ns(),
    }
    controller = SIGNAL_CONTROLLER
    if controller is not None:
        controller.block()
    try:
        if len(OPERATOR_SIGNALS) != signal_start:
            raise Inconclusive("operator signal before durable launch")
        append_jsonl(PACKET / "lifecycle.jsonl", launch)
        fsync_dir(PACKET)
    except BaseException:
        if controller is not None:
            controller.unblock()
        raise

    def acquired(pid, pgid, timestamp):
        append_jsonl(
            PACKET / "lifecycle.jsonl",
            {
                "schema": 1,
                "event": "acquired",
                "stem": spec["stem"],
                "pid": pid,
                "pgid": pgid,
                "monotonic_ns": timestamp,
            },
        )

    outcome = launched_child_outcome(spec["command"], env, acquired, controller)
    require_sealable_child_outcome(outcome)
    stdout_path = WORK / f"{spec['stem']}.stdout"
    stderr_path = WORK / f"{spec['stem']}.stderr"
    append_jsonl(
        PACKET / "lifecycle.jsonl",
        {
            "schema": 1,
            "event": "completion",
            "stem": spec["stem"],
            **{
                key: outcome.get(key)
                for key in (
                    "pid",
                    "pgid",
                    "spawn_error",
                    "ownership_error",
                    "returncode",
                    "reaped",
                    "output_overflow",
                    "drain_errors",
                    "interrupted",
                    "operator_interrupted",
                    "timed_out",
                    "cleanup_actions",
                    "cleanup_errors",
                    "operator_signals",
                    "attempted_monotonic_ns",
                    "acquired_monotonic_ns",
                    "completed_monotonic_ns",
                    "group_disposition",
                    "group_disposition_initial",
                )
            },
        },
    )
    write_exclusive(stdout_path, outcome["stdout"])
    write_exclusive(stderr_path, outcome["stderr"])
    post_errors = []
    try:
        host_after = host_state()
    except Exception as error:
        host_after = None
        post_errors.append(f"host:{type(error).__name__}:{error}")
    try:
        vm_after = vm_state()
    except Exception as error:
        vm_after = None
        post_errors.append(f"vm:{type(error).__name__}:{error}")
    try:
        stamps_after = path_stamps()
        stamps_after_ns = time.monotonic_ns()
    except Exception as error:
        stamps_after = None
        stamps_after_ns = time.monotonic_ns()
        post_errors.append(f"stamps:{type(error).__name__}:{error}")
    child_interval = (
        vm_interval("child", vm_launch, vm_after)
        if isinstance(vm_after, dict)
        else None
    )
    post = {
        "schema": 1,
        "host_after_exit": host_after,
        "vm_after_exit": vm_after,
        "source_path_stamps_after_exit": stamps_after,
        "source_path_stamps_after_exit_monotonic_ns": stamps_after_ns,
        "child_interval": child_interval,
        "operation_errors": post_errors,
    }
    post_path = WORK / f"{spec['stem']}.post.json"
    write_json(post_path, post)
    fsync_dir(WORK)
    terminal_activity = time.monotonic_ns()
    if child_interval is None:
        defects.append("child_vm_capture_failed")
    else:
        defects.extend(child_interval["defect_reasons"])
        invalid.extend(child_interval["inconclusive_reasons"])
    if not isinstance(host_after, dict) or host_after["valid"] is not True:
        invalid.append("host_invalid_after_exit")
    if post_errors:
        defects.extend(post_errors)
    if outcome.get("output_overflow"):
        invalid.append("bounded_output_exceeded")
    if outcome.get("drain_errors"):
        invalid.append("output_drain_failed")
    if outcome.get("interrupted"):
        invalid.append("operator_interrupt")
    if outcome.get("timed_out"):
        invalid.append("child_deadline_exceeded")
    if outcome.get("ownership_error"):
        invalid.append("process_group_authentication_failed")
    if outcome.get("cleanup_errors"):
        invalid.append("process_cleanup_failed")
    if outcome.get("cleanup_actions"):
        invalid.append("process_cleanup_invoked")
    if outcome.get("pid") is not None and (
        not outcome.get("reaped")
        or not outcome.get("group_disposition", {}).get("no_live_group_observed")
        or any(
            sample.get("error") is not None
            for sample in outcome.get("group_disposition", {}).get("samples", [])
        )
    ):
        invalid.append("process_lifecycle_invalid")
    acquired_ns = outcome.get("acquired_monotonic_ns")
    if type(acquired_ns) is int:
        if acquired_ns < proved:
            defects.append("launch_clock_regressed")
        elif not launch_delay_within_limit(proved, acquired_ns):
            invalid.append("launch_exceeded_five_seconds")
    try:
        typed_equal(stamps_after, stamps_before, "post-child source path stamps")
    except ContractDefect as error:
        defects.append(str(error))
    result = resources = None
    parse_error = None
    if outcome.get("returncode") == 0:
        try:
            resources = parse_time(outcome["stderr"])
            result = validate_result(
                parse_json_bytes(outcome["stdout"], spec["stem"]),
                spec["arm"],
                manifest["build_identity"],
                records,
            )
            if result["rusage"]["timer_block_inputs"] != 0:
                invalid.append("timer_block_input_nonzero")
            if result["rusage"]["timer_swaps"] != 0:
                invalid.append("timer_swap_nonzero")
        except ContractDefect as error:
            parse_error = f"{type(error).__name__}:{error}"
    elif structured_counter_defect(outcome["stderr"]):
        defects.append("structured_child_counter_regression")
    classification, reason = classify_attempt(outcome, parse_error, defects, invalid)
    attempt = {
        "schema": 1,
        **spec,
        "environment_sha256": manifest["environment_record"]["sha256"],
        "pid": outcome.get("pid"),
        "pgid": outcome.get("pgid"),
        "spawn_error": outcome.get("spawn_error"),
        "ownership_error": outcome.get("ownership_error"),
        "returncode": outcome.get("returncode"),
        "reaped": outcome.get("reaped"),
        "interrupted": outcome.get("interrupted"),
        "output_overflow": outcome.get("output_overflow"),
        "drain_errors": outcome.get("drain_errors"),
        "timed_out": outcome.get("timed_out"),
        "operator_interrupted": outcome.get("operator_interrupted"),
        "cleanup_actions": outcome.get("cleanup_actions"),
        "cleanup_errors": outcome.get("cleanup_errors"),
        "operator_signals": outcome.get("operator_signals"),
        "signal_start_sequence": signal_start,
        "signal_end_sequence": len(OPERATOR_SIGNALS),
        "signal_events": [dict(row) for row in OPERATOR_SIGNALS[signal_start:]],
        "group_disposition": outcome.get("group_disposition"),
        "group_disposition_initial": outcome.get("group_disposition_initial"),
        "launch_attempted_monotonic_ns": outcome["attempted_monotonic_ns"],
        "launch_acquired_monotonic_ns": acquired_ns,
        "completion_monotonic_ns": outcome["completed_monotonic_ns"],
        "residency_to_acquired_ns": acquired_ns - proved
        if type(acquired_ns) is int
        else None,
        "terminal_activity_monotonic_ns": terminal_activity,
        "process_resources": resources,
        "result": result,
        "parse_error": parse_error,
        "defect_reasons": sorted(set(defects)),
        "inconclusive_reasons": sorted(set(invalid)),
        "stop_classification": classification,
        "stop_reason": reason,
        "artifacts": {
            "stdout": stable_file_record(stdout_path),
            "stderr": stable_file_record(stderr_path),
            "conditioning": stable_file_record(conditioning_path),
            "post": stable_file_record(post_path),
        },
    }
    bundle_path = PACKET / f"{spec['stem']}.attempt.json"
    write_json(bundle_path, attempt)
    attempt["bundle_sha256"] = sha_file(bundle_path)
    append_jsonl(PACKET / "attempts.jsonl", attempt)
    fsync_dir(PACKET)
    return attempt, terminal_activity


def score_rows(rows):
    require(
        len(rows) == 4 and [row["arm"] for row in rows] == list(ORDER),
        "ABBA attempt order drifted",
    )
    require(
        all(
            row["stop_classification"] is None and row["result"] is not None
            for row in rows
        ),
        "scoring requires four valid attempts",
    )
    pairs = []
    for pair, (a_index, b_index) in enumerate(((0, 1), (3, 2)), 1):
        a, b = rows[a_index], rows[b_index]
        a_ready = uint(a["result"]["timing"]["ready_us"], "A ready", positive=True)
        b_ready = uint(b["result"]["timing"]["ready_us"], "B ready", positive=True)
        a_cpu = uint(a["result"]["rusage"]["total_cpu_us"], "A CPU", positive=True)
        b_cpu = uint(b["result"]["rusage"]["total_cpu_us"], "B CPU", positive=True)
        a_rss = uint(
            a["process_resources"]["maximum_resident_set_size"], "A RSS", positive=True
        )
        b_rss = uint(
            b["process_resources"]["maximum_resident_set_size"], "B RSS", positive=True
        )
        a_foot = uint(
            a["process_resources"]["peak_memory_footprint"],
            "A footprint",
            positive=True,
        )
        b_foot = uint(
            b["process_resources"]["peak_memory_footprint"],
            "B footprint",
            positive=True,
        )
        d = a_ready - b_ready
        pairs.append(
            {
                "pair": pair,
                "positions": [a_index + 1, b_index + 1],
                "orientation": "A1-B2" if pair == 1 else "A4-B3",
                "d_us": d,
                "q": b_ready / a_ready,
                "cpu_b_over_a": b_cpu / a_cpu,
                "rss_b_over_a": b_rss / a_rss,
                "footprint_b_over_a": b_foot / a_foot,
                "gates": {
                    "saving": d >= 1_500_000,
                    "b_wins": b_ready < a_ready,
                    "cpu": b_cpu / a_cpu <= 1.50,
                    "rss": b_rss / a_rss <= 1.05,
                    "footprint": b_foot / a_foot <= 1.05,
                },
            }
        )
    qualifies = all(all(pair["gates"].values()) for pair in pairs)
    return {
        "schema": 1,
        "pairs": pairs,
        "median_d_us": sum(sorted(pair["d_us"] for pair in pairs)) / 2,
        "median_q": sum(sorted(pair["q"] for pair in pairs)) / 2,
        "qualifies": qualifies,
    }


def disposition(contract=False, invalid=False, complete=False, qualifies=False):
    if contract:
        return "implementation_or_contract_defect"
    if invalid or not complete:
        return "inconclusive"
    return "GO" if qualifies else "KILL"


def artifact_inventory(packet, work):
    members = []
    for root in (packet, work):
        for path in sorted(root.iterdir()):
            if path.name in {"artifact-inventory.json", "packet-complete.json"}:
                continue
            members.append(stable_file_record(path))
    return {
        "schema": 1,
        "logical_records": members,
        "logical_record_count": len(members),
    }


def read_json_file(path, label):
    require(path.is_file() and not path.is_symlink(), f"{label} is not regular")
    return parse_json_bytes(stable_read_bytes(path), label)


def read_jsonl_file(path, label):
    require(path.is_file() and not path.is_symlink(), f"{label} is not regular")
    raw = stable_read_bytes(path)
    require(raw.endswith(b"\n"), f"{label} lacks final newline")
    rows = []
    for index, line in enumerate(raw.splitlines(), 1):
        value = parse_json_bytes(line, f"{label}:{index}")
        require(isinstance(value, dict), f"{label}:{index} is not an object")
        require(
            line + b"\n" == json_bytes(value), f"{label}:{index} is not canonical JSONL"
        )
        rows.append(value)
    return rows


def root_snapshot(root):
    root_stat = os.lstat(root)
    require(
        stat.S_ISDIR(root_stat.st_mode) and not root.is_symlink(),
        f"artifact root is not a real directory: {root}",
    )
    before = sorted(os.listdir(root))
    result = {}
    for name in before:
        path = root / name
        require(
            path.is_file() and not path.is_symlink(),
            f"non-direct-regular packet member: {path}",
        )
        result[name] = stable_file_record(path)
    require(sorted(os.listdir(root)) == before, f"root changed during snapshot: {root}")
    return result


def require_snapshot_unchanged(before, after, label):
    require(set(before) <= set(after), f"{label} lost a preexisting member")
    for name, record in before.items():
        typed_equal(after[name], record, f"{label}.{name}")


def validate_group_disposition_observation(value, pgid):
    require(
        isinstance(value, dict)
        and set(value) == {"pgid", "samples", "no_live_group_observed"}
        and value["pgid"] == pgid,
        "group disposition identity drifted",
    )
    samples = value["samples"]
    require(isinstance(samples, list) and samples, "group disposition samples missing")
    prior_timestamp = None
    for sample in samples:
        require(
            isinstance(sample, dict)
            and set(sample) == {"captured_monotonic_ns", "members", "error"},
            "group disposition sample schema drifted",
        )
        timestamp = uint(
            sample["captured_monotonic_ns"], "disposition timestamp", positive=True
        )
        require(
            prior_timestamp is None or prior_timestamp <= timestamp,
            "group disposition sample time regressed",
        )
        prior_timestamp = timestamp
        if sample["error"] is None:
            require(
                isinstance(sample["members"], list)
                and sample["members"] == sorted(sample["members"])
                and len(set(sample["members"])) == len(sample["members"])
                and all(type(pid) is int and pid > 0 for pid in sample["members"]),
                "group disposition member list drifted",
            )
        else:
            require(
                isinstance(sample["error"], str)
                and sample["error"]
                and sample["members"] is None,
                "group disposition error sample drifted",
            )
    require(
        value["no_live_group_observed"] is (samples[-1]["members"] == []),
        "group disposition summary drifted",
    )
    return value


def validate_group_disposition(value, pgid):
    validate_group_disposition_observation(value, pgid)
    require(
        all(sample["error"] is None for sample in value["samples"])
        and value["no_live_group_observed"] is True,
        "process group survived",
    )
    return value


def validate_host_snapshot(value, label):
    require(
        isinstance(value, dict)
        and set(value)
        == {
            "thermal",
            "battery",
            "memory_pressure",
            "memory_available_percent",
            "process_snapshot",
            "ignored_process_ids",
            "competing_processes",
            "errors",
            "valid",
            "captured_monotonic_ns",
        },
        f"{label} host schema drifted",
    )
    match = re.search(
        r"System-wide memory free percentage: (\d+)%", value["memory_pressure"] or ""
    )
    available = int(match.group(1)) if match else None
    ignored = value["ignored_process_ids"]
    require(
        isinstance(ignored, list)
        and all(type(pid) is int and pid > 0 for pid in ignored),
        f"{label} ignored process IDs drifted",
    )
    recomputed_competitors = []
    executable = re.compile(r"(?:^|/)(?:qwen|qwen-bench|llama[^/ ]*)(?:\s|$)", re.I)
    for line in (value["process_snapshot"] or "").splitlines():
        process = re.match(r"\s*(\d+)\s+(.*)", line)
        if (
            process
            and int(process.group(1)) not in ignored
            and (
                executable.search(process.group(2))
                or any(name.lower() in process.group(2).lower() for name in MODEL_NAMES)
            )
        ):
            recomputed_competitors.append(
                {"pid": int(process.group(1)), "command": process.group(2)}
            )
    recomputed = (
        not value["errors"]
        and isinstance(value["thermal"], str)
        and "No thermal warning level has been recorded" in value["thermal"]
        and "No performance warning level has been recorded" in value["thermal"]
        and isinstance(value["battery"], str)
        and "AC Power" in value["battery"]
        and available is not None
        and available >= 50
        and value["competing_processes"] == recomputed_competitors == []
        and isinstance(value["process_snapshot"], str)
    )
    require(
        value["memory_available_percent"] == available and value["valid"] is recomputed,
        f"{label} host validity drifted",
    )
    uint(value["captured_monotonic_ns"], f"{label} host timestamp", positive=True)


def validate_source_records(records, manifest):
    require(
        isinstance(records, list) and len(records) == 3,
        "conditioned source record count drifted",
    )
    stamp_keys = {
        "device",
        "inode",
        "size",
        "mtime_sec",
        "mtime_nsec",
        "ctime_sec",
        "ctime_nsec",
    }
    keys = {
        "path",
        "descriptor_stamp_before",
        "descriptor_stamp_after",
        "bytes_hashed",
        "sha256",
        "hash_started_monotonic_ns",
        "hash_completed_monotonic_ns",
        "page_size",
        "total_pages",
        "resident_pages",
        "all_pages_resident",
        "residency_checked_monotonic_ns",
    }
    prior_hash = None
    prior_residency = None
    for index, (row, model, baseline) in enumerate(
        zip(records, manifest["models"], manifest["model_path_stamps"])
    ):
        require(
            isinstance(row, dict) and set(row) == keys and row["path"] == model["path"],
            f"source[{index}] schema/path drifted",
        )
        expected_stamp = {key: baseline[key] for key in stamp_keys}
        typed_equal(
            row["descriptor_stamp_before"],
            expected_stamp,
            f"source[{index}] before stamp",
        )
        typed_equal(
            row["descriptor_stamp_after"],
            expected_stamp,
            f"source[{index}] after stamp",
        )
        require(
            row["bytes_hashed"] == model["size"]
            and row["sha256"] == model["sha256"]
            and row["page_size"] == PAGE_SIZE
            and row["total_pages"] == model["pages"]
            and row["resident_pages"] == row["total_pages"]
            and row["all_pages_resident"] is True,
            f"source[{index}] hash/residency drifted",
        )
        for key in (
            "hash_started_monotonic_ns",
            "hash_completed_monotonic_ns",
            "residency_checked_monotonic_ns",
        ):
            uint(row[key], f"source[{index}].{key}", positive=True)
        require(
            row["hash_started_monotonic_ns"]
            <= row["hash_completed_monotonic_ns"]
            <= row["residency_checked_monotonic_ns"],
            f"source[{index}] timeline drifted",
        )
        require(
            prior_hash is None or prior_hash <= row["hash_started_monotonic_ns"],
            f"source[{index}] overlaps prior hash",
        )
        require(
            prior_residency is None
            or prior_residency <= row["residency_checked_monotonic_ns"],
            f"source[{index}] residency order drifted",
        )
        prior_hash = row["hash_completed_monotonic_ns"]
        prior_residency = row["residency_checked_monotonic_ns"]
    require(
        records[0]["residency_checked_monotonic_ns"]
        >= records[-1]["hash_completed_monotonic_ns"],
        "residency proof began before all hashes completed",
    )


def validate_cooldown(value, prior_activity):
    require(
        isinstance(value, dict)
        and set(value)
        == {
            "prior_activity_monotonic_ns",
            "required_interval_ns",
            "eligible_monotonic_ns",
            "started_monotonic_ns",
            "completed_monotonic_ns",
            "observed_interval_ns",
        },
        "cooldown schema drifted",
    )
    require(
        value["prior_activity_monotonic_ns"] == prior_activity
        and value["required_interval_ns"] == COOLDOWN_NS
        and value["eligible_monotonic_ns"] == prior_activity + COOLDOWN_NS
        and value["observed_interval_ns"]
        == value["completed_monotonic_ns"] - prior_activity
        and value["completed_monotonic_ns"] >= value["eligible_monotonic_ns"]
        and value["started_monotonic_ns"] <= value["completed_monotonic_ns"],
        "cooldown arithmetic drifted",
    )


def validate_initialization(path, packet, work, manifest):
    rows = read_jsonl_file(path, "initialization progress")
    require(
        [row.get("stage") for row in rows]
        == [
            "reservation",
            "model-stamps",
            "manifest",
            "probe-artifacts",
            "roots-fsynced",
        ],
        "initialization stage sequence drifted",
    )
    for row in rows:
        require(
            set(row) == {"schema", "stage", "monotonic_ns", "evidence"}
            and row["schema"] == 1,
            "initialization row schema drifted",
        )
    reservation_value = read_json_file(work / "reservation.json", "init reservation")
    require(
        rows[0]["monotonic_ns"] >= reservation_value["activity_monotonic_ns"]
        and all(
            left["monotonic_ns"] <= right["monotonic_ns"]
            for left, right in zip(rows, rows[1:])
        ),
        "initialization timeline drifted",
    )
    require(
        rows[0]["evidence"]
        == {"reservation_sha256": sha_file(work / "reservation.json")},
        "initialization reservation binding drifted",
    )
    typed_equal(
        rows[1]["evidence"]["model_path_stamps"],
        manifest["model_path_stamps"],
        "initialization model stamps",
    )
    require(
        rows[2]["evidence"]["manifest_sha256"] == sha_file(packet / "manifest.json")
        and rows[3]["evidence"]
        == {
            "probe_sha256": sha_file(packet / "headroom-probe.json"),
            "probe_stdout_sha256": sha_file(packet / "headroom-probe.stdout"),
        }
        and rows[4]["evidence"] == {"complete": True},
        "initialization artifact binding drifted",
    )


def validate_manifest(manifest):
    require(
        isinstance(manifest, dict)
        and set(manifest)
        == {
            "schema",
            "protocol",
            "source_commit",
            "build_identity",
            "contract",
            "tracked_describe",
            "runner",
            "binary",
            "models",
            "model_path_stamps",
            "environment_record",
            "commands",
            "probe_command",
            "order",
            "attempt_count",
            "retry_count",
            "deadlines_seconds",
            "runner_process_ids",
        },
        "manifest key set drifted",
    )
    uint(manifest["schema"], "manifest schema", 1)
    require(
        manifest["protocol"] == "v0653-a10b-parallel-pread-floor"
        and manifest["order"] == list(ORDER)
        and manifest["attempt_count"] == 4
        and manifest["retry_count"] == 0,
        "manifest protocol drifted",
    )
    typed_equal(
        manifest["commands"], [child_command(arm) for arm in ORDER], "manifest commands"
    )
    typed_equal(manifest["probe_command"], probe_command(), "manifest probe command")
    typed_equal(
        manifest["deadlines_seconds"],
        {
            "probe": PROBE_DEADLINE_S,
            "child": CHILD_DEADLINE_S,
            "identity": IDENTITY_DEADLINE_S,
        },
        "manifest deadlines",
    )
    record = manifest["environment_record"]
    require(
        isinstance(record, dict)
        and set(record) == {"environment", "removed_keys", "sha256"},
        "environment record drifted",
    )
    require(
        record["sha256"]
        == sha_bytes(
            json_bytes(
                {
                    "environment": record["environment"],
                    "removed_keys": record["removed_keys"],
                }
            )
        ),
        "environment digest drifted",
    )
    env = record["environment"]
    require(
        set(env) <= SAFE_ENV_KEYS
        and all(key not in env for key in record["removed_keys"]),
        "environment allowlist drifted",
    )
    validate_build_identity(manifest["build_identity"])
    require(
        isinstance(manifest["runner_process_ids"], list)
        and len(manifest["runner_process_ids"]) == 2
        and all(type(pid) is int and pid > 0 for pid in manifest["runner_process_ids"]),
        "manifest runner process IDs drifted",
    )
    require(
        manifest["source_commit"] == manifest["build_identity"]["build_commit"],
        "manifest source/build commit drifted",
    )
    expected_identity_paths = {
        "contract": str(CONTRACT.relative_to(ROOT)),
        "tracked_describe": str(DESCRIBE.relative_to(ROOT)),
        "runner": str(RUNNER.relative_to(ROOT)),
        "binary": str((ROOT / str(BINARY).removeprefix("./")).relative_to(ROOT)),
    }
    for key, expected_path in expected_identity_paths.items():
        record_value = manifest[key]
        require(
            isinstance(record_value, dict)
            and set(record_value)
            == {"path", "size_bytes", "sha256", "descriptor_stamp"}
            and record_value["path"] == expected_path
            and re.fullmatch(r"[0-9a-f]{64}", record_value["sha256"]) is not None
            and type(record_value["size_bytes"]) is int
            and record_value["size_bytes"]
            == record_value["descriptor_stamp"].get("size"),
            f"manifest {key} record drifted",
        )
    typed_equal(
        manifest["models"],
        [
            {"path": str(path), "size": size, "pages": pages, "sha256": digest}
            for path, size, pages, digest in zip(
                MODELS, MODEL_SIZES, MODEL_PAGES, MODEL_SHA256
            )
        ],
        "manifest models",
    )
    validate_model_path_stamps(manifest["model_path_stamps"], "manifest model stamps")


def validate_model_path_stamps(value, label):
    require(isinstance(value, list) and len(value) == 3, f"{label} count drifted")
    identities = set()
    expected_keys = {
        "path",
        "device",
        "inode",
        "size",
        "mtime_sec",
        "mtime_nsec",
        "ctime_sec",
        "ctime_nsec",
    }
    for index, (row, path, size) in enumerate(zip(value, MODELS, MODEL_SIZES)):
        require(
            isinstance(row, dict)
            and set(row) == expected_keys
            and row["path"] == str(path),
            f"{label}[{index}] schema/path drifted",
        )
        for key in expected_keys - {"path"}:
            uint(row[key], f"{label}[{index}].{key}")
        require(
            row["size"] == size
            and row["device"] > 0
            and row["inode"] > 0
            and row["mtime_nsec"] < 1_000_000_000
            and row["ctime_nsec"] < 1_000_000_000,
            f"{label}[{index}] geometry drifted",
        )
        identity = (row["device"], row["inode"])
        require(identity not in identities, f"{label} identity repeated")
        identities.add(identity)
    return value


def validate_probe_bundle(packet, manifest):
    raw_path = packet / "headroom-probe.stdout"
    record = read_json_file(packet / "headroom-probe.json", "headroom probe record")
    raw = stable_read_bytes(raw_path)
    require(
        record.get("schema") == 1
        and record.get("command") == probe_command()
        and record.get("raw_stdout_sha256") == sha_bytes(raw),
        "headroom probe raw binding drifted",
    )
    parsed = validate_probe(
        parse_json_bytes(raw, "raw headroom probe"), manifest["build_identity"]
    )
    typed_equal(record.get("stdout"), parsed, "headroom parsed/raw binding")
    process = record.get("process")
    require(
        isinstance(process, dict)
        and set(process)
        == {
            "pid",
            "pgid",
            "attempted_monotonic_ns",
            "acquired_monotonic_ns",
            "completed_monotonic_ns",
            "returncode",
            "reaped",
            "group_disposition",
            "ownership_error",
            "output_overflow",
            "drain_errors",
            "timed_out",
            "operator_interrupted",
            "cleanup_actions",
            "cleanup_errors",
            "operator_signals",
        }
        and process.get("returncode") == 0
        and process.get("reaped") is True
        and type(process.get("pid")) is int
        and process["pid"] > 0
        and process.get("pgid") == process.get("pid")
        and process.get("ownership_error") is None
        and process.get("output_overflow") is False
        and process.get("drain_errors") == []
        and process.get("timed_out") is False
        and process.get("operator_interrupted") is False
        and process.get("cleanup_actions") == []
        and process.get("cleanup_errors") == []
        and process.get("operator_signals") == [],
        "headroom process lifecycle drifted",
    )
    for field in (
        "attempted_monotonic_ns",
        "acquired_monotonic_ns",
        "completed_monotonic_ns",
    ):
        uint(process[field], f"headroom {field}", positive=True)
    validate_group_disposition(process.get("group_disposition"), process["pgid"])
    require(
        process["attempted_monotonic_ns"]
        <= process["acquired_monotonic_ns"]
        <= process["completed_monotonic_ns"],
        "headroom timestamps drifted",
    )
    require(
        all(
            process["attempted_monotonic_ns"]
            <= sample["captured_monotonic_ns"]
            <= process["completed_monotonic_ns"]
            for sample in process["group_disposition"]["samples"]
        ),
        "headroom group observation escaped process interval",
    )
    require(
        process["completed_monotonic_ns"] - process["attempted_monotonic_ns"]
        <= int(PROBE_DEADLINE_S * 1e9)
        and len(raw) <= MAX_OUTPUT,
        "headroom duration/output bound drifted",
    )


def validate_signal_cutoff(packet, decision, attempts):
    log_path, cutoff_path = packet / "signal-log.json", packet / "signal-cutoff.json"
    log = read_json_file(log_path, "signal log")
    cutoff = read_json_file(cutoff_path, "signal cutoff")
    require(
        set(log)
        == {
            "schema",
            "boundary_monotonic_ns",
            "event_count",
            "events",
            "pending_signals",
            "authority_signal_count",
        }
        and log["schema"] == 1
        and log["event_count"] == len(log["events"])
        and log["authority_signal_count"]
        == len(log["events"]) + len(log["pending_signals"]),
        "signal log schema drifted",
    )
    uint(log["schema"], "signal log schema", expected=1)
    uint(log["boundary_monotonic_ns"], "signal boundary", positive=True)
    uint(log["event_count"], "signal event count")
    uint(log["authority_signal_count"], "signal authority count")
    for index, event in enumerate(log["events"], 1):
        require(
            set(event) == {"sequence", "signal", "monotonic_ns"}
            and event["sequence"] == index
            and event["signal"] in (int(signal.SIGINT), int(signal.SIGTERM))
            and type(event["monotonic_ns"]) is int
            and 0 < event["monotonic_ns"] <= log["boundary_monotonic_ns"],
            "signal log event drifted",
        )
    require(
        log["pending_signals"] == sorted(set(log["pending_signals"]))
        and all(
            item in (int(signal.SIGINT), int(signal.SIGTERM))
            for item in log["pending_signals"]
        ),
        "pending signal set drifted",
    )
    require(
        set(cutoff)
        == {
            "schema",
            "event",
            "boundary_monotonic_ns",
            "signal_log_sha256",
            "authority_signal_count",
        }
        and cutoff["schema"] == 1
        and cutoff["event"] == "final-signal-cutoff"
        and cutoff["boundary_monotonic_ns"] == log["boundary_monotonic_ns"]
        and cutoff["signal_log_sha256"] == sha_file(log_path)
        and cutoff["authority_signal_count"] == log["authority_signal_count"]
        and decision["signal_log_sha256"] == sha_file(log_path)
        and decision["signal_cutoff_sha256"] == sha_file(cutoff_path),
        "signal cutoff binding drifted",
    )
    uint(cutoff["boundary_monotonic_ns"], "cutoff boundary", positive=True)
    uint(cutoff["schema"], "cutoff schema", expected=1)
    uint(cutoff["authority_signal_count"], "cutoff authority count")
    signal_cursor = 0
    for attempt in attempts:
        start, end = attempt["signal_start_sequence"], attempt["signal_end_sequence"]
        require(
            start == signal_cursor
            and end >= start
            and end <= len(log["events"])
            and attempt["signal_events"] == log["events"][start:end],
            "attempt signal slice drifted",
        )
        signal_cursor = end
    require(
        signal_cursor == len(log["events"]) and log["pending_signals"] == [],
        "signal exists outside a complete attempt ledger",
    )
    return log, cutoff


def semantic_decision(attempts):
    stopped = next((row for row in attempts if row["stop_classification"]), None)
    contract = (
        stopped is not None
        and stopped["stop_classification"] == "implementation_or_contract_defect"
    )
    invalid = stopped is not None and stopped["stop_classification"] == "inconclusive"
    complete = len(attempts) == 4 and stopped is None
    analysis = score_rows(attempts) if complete else None
    status = disposition(
        contract, invalid, complete, bool(analysis and analysis["qualifies"])
    )
    return status, analysis


def validate_attempt_semantics(
    attempt, spec, manifest, packet, work, lifecycle_rows, prior_activity
):
    require(set(attempt) == ATTEMPT_KEYS, "attempt key set drifted")
    for field in (
        "reaped",
        "interrupted",
        "output_overflow",
        "timed_out",
        "operator_interrupted",
    ):
        require(type(attempt[field]) is bool, f"attempt {field} type drifted")
    for field in (
        "drain_errors",
        "cleanup_actions",
        "cleanup_errors",
        "operator_signals",
        "signal_events",
        "defect_reasons",
        "inconclusive_reasons",
    ):
        require(isinstance(attempt[field], list), f"attempt {field} type drifted")
    require(
        all(isinstance(item, str) for item in attempt["drain_errors"])
        and all(isinstance(item, str) for item in attempt["cleanup_errors"])
        and all(isinstance(item, str) for item in attempt["defect_reasons"])
        and all(isinstance(item, str) for item in attempt["inconclusive_reasons"]),
        "attempt string-list evidence drifted",
    )
    for field in ("pid", "pgid"):
        require(
            attempt[field] is None
            or (type(attempt[field]) is int and attempt[field] > 0),
            f"attempt {field} type drifted",
        )
    require(
        attempt["returncode"] is None or type(attempt["returncode"]) is int,
        "attempt returncode type drifted",
    )
    for field in (
        "signal_start_sequence",
        "signal_end_sequence",
        "launch_attempted_monotonic_ns",
        "completion_monotonic_ns",
        "terminal_activity_monotonic_ns",
    ):
        uint(attempt[field], f"attempt {field}", positive="monotonic" in field)
    require(
        attempt["launch_acquired_monotonic_ns"] is None
        or (
            type(attempt["launch_acquired_monotonic_ns"]) is int
            and attempt["launch_acquired_monotonic_ns"] > 0
        ),
        "attempt acquired timestamp type drifted",
    )
    require(
        attempt["residency_to_acquired_ns"] is None
        or (
            type(attempt["residency_to_acquired_ns"]) is int
            and attempt["residency_to_acquired_ns"] >= 0
        ),
        "attempt residency delta type drifted",
    )
    for field in ("spawn_error", "ownership_error"):
        require(
            attempt[field] is None or isinstance(attempt[field], str),
            f"attempt {field} type drifted",
        )
    require(
        {key: attempt[key] for key in spec} == spec, "attempt ABBA identity drifted"
    )
    bundle_path = packet / f"{spec['stem']}.attempt.json"
    bundle = read_json_file(bundle_path, f"{spec['stem']} bundle")
    ledger_bundle = dict(attempt)
    bundle_digest = ledger_bundle.pop("bundle_sha256", None)
    typed_equal(bundle, ledger_bundle, f"{spec['stem']} ledger/bundle")
    require(bundle_digest == sha_file(bundle_path), "attempt bundle digest drifted")
    paths = {
        name: work / f"{spec['stem']}.{suffix}"
        for name, suffix in (
            ("stdout", "stdout"),
            ("stderr", "stderr"),
            ("conditioning", "conditioning.json"),
            ("post", "post.json"),
        )
    }
    for name, path in paths.items():
        typed_equal(
            attempt["artifacts"][name],
            stable_file_record(path),
            f"{spec['stem']} {name} artifact",
        )
    conditioning = read_json_file(paths["conditioning"], "conditioning")
    post = read_json_file(paths["post"], "post")
    require(
        set(conditioning)
        == {
            "schema",
            "pre_attempt_identity",
            "cooldown",
            "host_before_conditioning",
            "vm_before_conditioning",
            "source_path_stamps_before",
            "source_path_stamps_before_monotonic_ns",
            "source_records",
            "residency_proved_monotonic_ns",
            "host_before_spawn",
            "vm_before_spawn",
            "source_path_stamps_before_spawn",
            "source_path_stamps_before_spawn_monotonic_ns",
            "preconditioning_interval",
        },
        "conditioning key set drifted",
    )
    require(
        set(post)
        == {
            "schema",
            "host_after_exit",
            "vm_after_exit",
            "source_path_stamps_after_exit",
            "source_path_stamps_after_exit_monotonic_ns",
            "child_interval",
            "operation_errors",
        },
        "post key set drifted",
    )
    validate_cooldown(conditioning["cooldown"], prior_activity)
    validate_host_snapshot(conditioning["host_before_conditioning"], "before")
    validate_host_snapshot(conditioning["host_before_spawn"], "spawn")
    validate_vm_snapshot(conditioning["vm_before_conditioning"], "before")
    validate_vm_snapshot(conditioning["vm_before_spawn"], "spawn")
    require(
        isinstance(post["operation_errors"], list)
        and all(isinstance(item, str) for item in post["operation_errors"]),
        "postflight operation errors drifted",
    )
    if post["host_after_exit"] is None:
        require(
            any(item.startswith("host:") for item in post["operation_errors"]),
            "missing post host lacks operation error",
        )
    else:
        validate_host_snapshot(post["host_after_exit"], "postflight")
    if post["vm_after_exit"] is None:
        require(
            any(item.startswith("vm:") for item in post["operation_errors"]),
            "missing post VM lacks operation error",
        )
    else:
        validate_vm_snapshot(post["vm_after_exit"], "postflight")
    if post["source_path_stamps_after_exit"] is None:
        require(
            any(item.startswith("stamps:") for item in post["operation_errors"]),
            "missing post stamps lack operation error",
        )
    for host in (
        conditioning["host_before_conditioning"],
        conditioning["host_before_spawn"],
    ):
        typed_equal(
            host["ignored_process_ids"],
            manifest["runner_process_ids"],
            "host ignored process IDs",
        )
    if post["host_after_exit"] is not None:
        typed_equal(
            post["host_after_exit"]["ignored_process_ids"],
            manifest["runner_process_ids"],
            "post ignored process IDs",
        )
    require(
        conditioning["schema"] == 1
        and post["schema"] == 1
        and attempt["schema"] == 1
        and attempt["environment_sha256"] == manifest["environment_record"]["sha256"],
        "attempt record version/environment drifted",
    )
    for key, label in (
        ("source_path_stamps_before", "before path stamps"),
        ("source_path_stamps_before_spawn", "spawn path stamps"),
    ):
        validate_model_path_stamps(conditioning[key], label)
        typed_equal(conditioning[key], manifest["model_path_stamps"], label)
    if post["source_path_stamps_after_exit"] is not None:
        validate_model_path_stamps(
            post["source_path_stamps_after_exit"], "post path stamps"
        )
        typed_equal(
            post["source_path_stamps_after_exit"],
            manifest["model_path_stamps"],
            "post/manifest path stamps",
        )
    validate_source_records(conditioning["source_records"], manifest)
    typed_equal(
        conditioning["pre_attempt_identity"]["identity"],
        manifest_identity(manifest),
        "pre-attempt identity",
    )
    require(
        set(conditioning["pre_attempt_identity"])
        == {"schema", "label", "captured_monotonic_ns", "identity", "matches_manifest"}
        and conditioning["pre_attempt_identity"]["schema"] == 1
        and conditioning["pre_attempt_identity"]["label"]
        == f"pre-attempt-{spec['position']}"
        and type(conditioning["pre_attempt_identity"]["captured_monotonic_ns"]) is int
        and conditioning["pre_attempt_identity"]["matches_manifest"] is True,
        "pre-attempt identity was not matched",
    )
    typed_equal(
        conditioning["preconditioning_interval"],
        vm_interval(
            "preconditioning",
            conditioning["vm_before_conditioning"],
            conditioning["vm_before_spawn"],
        ),
        "preconditioning VM replay",
    )
    if post["vm_after_exit"] is None:
        require(
            post["child_interval"] is None, "missing post VM retained child interval"
        )
    else:
        typed_equal(
            post["child_interval"],
            vm_interval(
                "child", conditioning["vm_before_spawn"], post["vm_after_exit"]
            ),
            "child VM replay",
        )
    require(
        len(lifecycle_rows) in (2, 3)
        and lifecycle_rows[0]["event"] == "launch"
        and lifecycle_rows[-1]["event"] == "completion",
        "attempt lifecycle drifted",
    )
    launch, completion = lifecycle_rows[0], lifecycle_rows[-1]
    require(
        set(launch)
        == {
            "schema",
            "event",
            *spec.keys(),
            "environment_sha256",
            "conditioning_sha256",
            "residency_proved_monotonic_ns",
            "monotonic_ns",
        },
        "launch key set drifted",
    )
    require(
        set(completion)
        == {
            "schema",
            "event",
            "stem",
            "pid",
            "pgid",
            "spawn_error",
            "ownership_error",
            "returncode",
            "reaped",
            "output_overflow",
            "drain_errors",
            "interrupted",
            "operator_interrupted",
            "timed_out",
            "cleanup_actions",
            "cleanup_errors",
            "operator_signals",
            "attempted_monotonic_ns",
            "acquired_monotonic_ns",
            "completed_monotonic_ns",
            "group_disposition_initial",
            "group_disposition",
        },
        "completion key set drifted",
    )
    require(
        all(launch[key] == value for key, value in spec.items())
        and launch["schema"] == 1
        and launch["conditioning_sha256"] == sha_file(paths["conditioning"])
        and launch["environment_sha256"] == manifest["environment_record"]["sha256"],
        "launch binding drifted",
    )
    require(
        launch["residency_proved_monotonic_ns"]
        == conditioning["residency_proved_monotonic_ns"],
        "launch residency proof binding drifted",
    )
    require(
        completion["schema"] == 1 and completion["stem"] == spec["stem"],
        "completion identity drifted",
    )
    for key in (
        "pid",
        "pgid",
        "spawn_error",
        "ownership_error",
        "returncode",
        "reaped",
        "output_overflow",
        "drain_errors",
        "interrupted",
        "operator_interrupted",
        "timed_out",
        "cleanup_actions",
        "cleanup_errors",
        "operator_signals",
        "group_disposition_initial",
        "group_disposition",
    ):
        typed_equal(completion[key], attempt[key], f"completion.{key}")
    require(
        completion["attempted_monotonic_ns"] == attempt["launch_attempted_monotonic_ns"]
        and completion["acquired_monotonic_ns"]
        == attempt["launch_acquired_monotonic_ns"]
        and completion["completed_monotonic_ns"] == attempt["completion_monotonic_ns"],
        "completion timestamp binding drifted",
    )
    expected_residency_delta = (
        completion["acquired_monotonic_ns"]
        - conditioning["residency_proved_monotonic_ns"]
        if type(completion["acquired_monotonic_ns"]) is int
        else None
    )
    require(
        attempt["residency_to_acquired_ns"] == expected_residency_delta,
        "residency arithmetic drifted",
    )
    require(
        launch["monotonic_ns"]
        <= completion["attempted_monotonic_ns"]
        <= completion["completed_monotonic_ns"],
        "lifecycle timestamps drifted",
    )
    require(
        completion["completed_monotonic_ns"] - completion["attempted_monotonic_ns"]
        <= int(CHILD_DEADLINE_S * 1e9),
        "child duration exceeded deadline",
    )
    require(
        paths["stdout"].stat().st_size <= MAX_OUTPUT
        and paths["stderr"].stat().st_size <= MAX_OUTPUT,
        "raw stream exceeded frozen bound",
    )
    identity_time = conditioning["pre_attempt_identity"]["captured_monotonic_ns"]
    source_records = conditioning["source_records"]
    acquired_point = completion["acquired_monotonic_ns"]
    if acquired_point is None:
        acquired_point = completion["attempted_monotonic_ns"]
    post_host_time = (
        post["host_after_exit"]["captured_monotonic_ns"]
        if post["host_after_exit"] is not None
        else completion["completed_monotonic_ns"]
    )
    post_vm_time = (
        post["vm_after_exit"]["captured_monotonic_ns"]
        if post["vm_after_exit"] is not None
        else post_host_time
    )
    require(
        conditioning["cooldown"]["completed_monotonic_ns"]
        <= identity_time
        <= conditioning["host_before_conditioning"]["captured_monotonic_ns"]
        <= conditioning["vm_before_conditioning"]["captured_monotonic_ns"]
        <= conditioning["source_path_stamps_before_monotonic_ns"]
        <= source_records[0]["hash_started_monotonic_ns"]
        <= source_records[-1]["residency_checked_monotonic_ns"]
        <= conditioning["residency_proved_monotonic_ns"]
        <= conditioning["host_before_spawn"]["captured_monotonic_ns"]
        <= conditioning["vm_before_spawn"]["captured_monotonic_ns"]
        <= conditioning["source_path_stamps_before_spawn_monotonic_ns"]
        <= launch["monotonic_ns"]
        <= completion["attempted_monotonic_ns"]
        <= acquired_point
        <= completion["completed_monotonic_ns"]
        <= post_host_time
        <= post_vm_time
        <= post["source_path_stamps_after_exit_monotonic_ns"]
        <= attempt["terminal_activity_monotonic_ns"],
        "attempt authority timeline drifted",
    )
    require(
        attempt["completion_monotonic_ns"] == completion["completed_monotonic_ns"],
        "attempt completion timestamp drifted",
    )
    if attempt["pgid"] is None:
        require(
            len(lifecycle_rows) == 2
            and attempt["pid"] is None
            and isinstance(attempt["spawn_error"], str)
            and bool(attempt["spawn_error"])
            and attempt["ownership_error"] is None
            and attempt["returncode"] is None
            and attempt["reaped"] is False
            and attempt["interrupted"] is False
            and attempt["output_overflow"] is False
            and attempt["timed_out"] is False
            and attempt["operator_interrupted"] is False
            and attempt["drain_errors"] == []
            and attempt["cleanup_actions"] == []
            and attempt["cleanup_errors"] == []
            and attempt["operator_signals"] == []
            and attempt["signal_events"] == []
            and attempt["group_disposition"] is None
            and attempt["group_disposition_initial"] is None,
            "plain spawn-failure terminal variant drifted",
        )
    else:
        require(
            type(attempt["pid"]) is int
            and attempt["pid"] > 0
            and attempt["pid"] == attempt["pgid"]
            and attempt["spawn_error"] is None
            and attempt["ownership_error"] is None
            and type(attempt["returncode"]) is int
            and attempt["reaped"] is True,
            "spawn-success variant drifted",
        )
        require(
            set(lifecycle_rows[1])
            == {"schema", "event", "stem", "pid", "pgid", "monotonic_ns"},
            "acquired key set drifted",
        )
        require(
            len(lifecycle_rows) == 3
            and lifecycle_rows[1]["schema"] == 1
            and lifecycle_rows[1]["event"] == "acquired"
            and lifecycle_rows[1]["stem"] == spec["stem"]
            and lifecycle_rows[1]["pid"] == attempt["pid"]
            and lifecycle_rows[1]["pgid"] == attempt["pgid"]
            and lifecycle_rows[1]["monotonic_ns"]
            == attempt["launch_acquired_monotonic_ns"],
            "acquired binding drifted",
        )
        validate_group_disposition(attempt["group_disposition"], attempt["pgid"])
        initial_disposition = attempt["group_disposition_initial"]
        validate_group_disposition_observation(initial_disposition, attempt["pgid"])
        typed_equal(
            initial_disposition,
            attempt["group_disposition"],
            "initial/final group disposition",
        )
    require(
        isinstance(attempt["cleanup_actions"], list)
        and isinstance(attempt["cleanup_errors"], list)
        and isinstance(attempt["operator_signals"], list),
        "cleanup evidence types drifted",
    )
    require(
        attempt["interrupted"] is attempt["operator_interrupted"]
        and attempt["signal_end_sequence"]
        == attempt["signal_start_sequence"] + len(attempt["signal_events"])
        and attempt["operator_signals"] == attempt["signal_events"],
        "attempt signal summary drifted",
    )
    for offset, event in enumerate(attempt["signal_events"], 1):
        require(
            isinstance(event, dict)
            and set(event) == {"sequence", "signal", "monotonic_ns"}
            and event["sequence"] == attempt["signal_start_sequence"] + offset
            and event["signal"] in (int(signal.SIGINT), int(signal.SIGTERM))
            and type(event["monotonic_ns"]) is int
            and attempt["launch_attempted_monotonic_ns"]
            <= event["monotonic_ns"]
            <= attempt["completion_monotonic_ns"],
            "attempt signal row drifted",
        )
    for action in attempt["cleanup_actions"]:
        require(
            isinstance(action, dict)
            and set(action)
            == {
                "reason",
                "signal",
                "target_pgid",
                "target_pid",
                "attempted_monotonic_ns",
                "succeeded",
                "error",
                "completed_monotonic_ns",
            }
            and action["target_pid"] == attempt["pid"]
            and action["target_pgid"] == attempt["pgid"]
            and action["signal"] == int(signal.SIGKILL)
            and isinstance(action["reason"], str)
            and action["reason"]
            and type(action["succeeded"]) is bool
            and (action["error"] is None or isinstance(action["error"], str))
            and (action["error"] is None) is action["succeeded"]
            and type(action["attempted_monotonic_ns"]) is int
            and type(action["completed_monotonic_ns"]) is int
            and action["attempted_monotonic_ns"] <= action["completed_monotonic_ns"],
            "cleanup action drifted",
        )
        require(
            completion["attempted_monotonic_ns"]
            <= action["attempted_monotonic_ns"]
            <= action["completed_monotonic_ns"]
            <= completion["completed_monotonic_ns"],
            "cleanup action escaped child interval",
        )
    for disposition_name in ("group_disposition_initial", "group_disposition"):
        disposition_value = attempt[disposition_name]
        if disposition_value is not None:
            require(
                all(
                    completion["attempted_monotonic_ns"]
                    <= sample["captured_monotonic_ns"]
                    <= completion["completed_monotonic_ns"]
                    for sample in disposition_value["samples"]
                ),
                f"{disposition_name} sample escaped child interval",
            )
    raw_out = stable_read_bytes(paths["stdout"])
    raw_err = stable_read_bytes(paths["stderr"])
    if attempt["pid"] is None:
        require(raw_out == b"" and raw_err == b"", "spawn failure emitted output")
    expected_result = expected_resources = expected_parse = None
    if attempt["returncode"] == 0:
        try:
            expected_resources = parse_time(raw_err)
            expected_result = validate_result(
                parse_json_bytes(raw_out, spec["stem"]),
                spec["arm"],
                manifest["build_identity"],
                conditioning["source_records"],
            )
        except ContractDefect as error:
            expected_parse = f"{type(error).__name__}:{error}"
    typed_equal(
        attempt["process_resources"], expected_resources, "replayed process resources"
    )
    typed_equal(attempt["result"], expected_result, "replayed child result")
    require(attempt["parse_error"] == expected_parse, "replayed parse error drifted")
    derived_defects = list(conditioning["preconditioning_interval"]["defect_reasons"])
    derived_invalid = list(
        conditioning["preconditioning_interval"]["inconclusive_reasons"]
    )
    if conditioning["host_before_conditioning"]["valid"] is not True:
        derived_invalid.append("host_invalid_before_conditioning")
    if conditioning["host_before_spawn"]["valid"] is not True:
        derived_invalid.append("host_invalid_before_spawn")
    if post["child_interval"] is None:
        derived_defects.append("child_vm_capture_failed")
    else:
        derived_defects.extend(post["child_interval"]["defect_reasons"])
        derived_invalid.extend(post["child_interval"]["inconclusive_reasons"])
    derived_defects.extend(post["operation_errors"])
    if post["host_after_exit"] is None or post["host_after_exit"]["valid"] is not True:
        derived_invalid.append("host_invalid_after_exit")
    for field, reason in (
        ("output_overflow", "bounded_output_exceeded"),
        ("timed_out", "child_deadline_exceeded"),
        ("operator_interrupted", "operator_interrupt"),
    ):
        if attempt[field]:
            derived_invalid.append(reason)
    if attempt["drain_errors"]:
        derived_invalid.append("output_drain_failed")
    if attempt["ownership_error"]:
        derived_invalid.append("process_group_authentication_failed")
    if attempt["cleanup_errors"]:
        derived_invalid.append("process_cleanup_failed")
    if attempt["cleanup_actions"]:
        derived_invalid.append("process_cleanup_invoked")
    if attempt["pid"] is not None and (
        attempt["reaped"] is not True or attempt["group_disposition"] is None
    ):
        derived_invalid.append("process_lifecycle_invalid")
    acquired_ns = attempt["launch_acquired_monotonic_ns"]
    proved = conditioning["residency_proved_monotonic_ns"]
    if type(acquired_ns) is int:
        if acquired_ns < proved:
            derived_defects.append("launch_clock_regressed")
        elif not launch_delay_within_limit(proved, acquired_ns):
            derived_invalid.append("launch_exceeded_five_seconds")
    try:
        typed_equal(
            post["source_path_stamps_after_exit"],
            conditioning["source_path_stamps_before"],
            "post-child source path stamps",
        )
    except ContractDefect as error:
        derived_defects.append(str(error))
    if expected_result is not None:
        if expected_result["rusage"]["timer_block_inputs"] != 0:
            derived_invalid.append("timer_block_input_nonzero")
        if expected_result["rusage"]["timer_swaps"] != 0:
            derived_invalid.append("timer_swap_nonzero")
    if attempt["returncode"] not in (None, 0) and structured_counter_defect(raw_err):
        derived_defects.append("structured_child_counter_regression")
    typed_equal(
        attempt["defect_reasons"],
        sorted(set(derived_defects)),
        "replayed defect reasons",
    )
    typed_equal(
        attempt["inconclusive_reasons"],
        sorted(set(derived_invalid)),
        "replayed inconclusive reasons",
    )
    expected_stop = classify_attempt(
        attempt,
        attempt["parse_error"],
        attempt["defect_reasons"],
        attempt["inconclusive_reasons"],
    )
    typed_equal(
        [attempt["stop_classification"], attempt["stop_reason"]],
        list(expected_stop),
        "attempt stop classification",
    )


def validate_packet_semantics(proposed_decision, packet=PACKET, work=WORK):
    require(
        isinstance(proposed_decision, dict)
        and set(proposed_decision) == DECISION_KEYS
        and proposed_decision["schema"] == 1,
        "decision key set drifted",
    )
    uint(proposed_decision["schema"], "decision schema", expected=1)
    packet_members = root_snapshot(packet)
    work_members = root_snapshot(work)
    manifest = read_json_file(packet / "manifest.json", "manifest")
    validate_manifest(manifest)
    require(
        proposed_decision["manifest_sha256"] == sha_file(packet / "manifest.json"),
        "decision manifest binding drifted",
    )
    validate_probe_bundle(packet, manifest)
    require(
        proposed_decision["headroom_probe_sha256"]
        == sha_file(packet / "headroom-probe.json"),
        "decision probe binding drifted",
    )
    require(
        "attempts.jsonl" in packet_members and "lifecycle.jsonl" in packet_members,
        "sealed packet lacks a complete attempt ledger",
    )
    attempts = read_jsonl_file(packet / "attempts.jsonl", "attempts")
    lifecycle = read_jsonl_file(packet / "lifecycle.jsonl", "lifecycle")
    require(1 <= len(attempts) <= 4, "sealed attempt count is invalid")
    expected_packet = {
        "manifest.json",
        "headroom-probe.stdout",
        "headroom-probe.json",
        "signal-log.json",
        "signal-cutoff.json",
        "decision.json",
    }
    expected_work = {"reservation.json", "initialization-progress.jsonl"}
    reservation = read_json_file(work / "reservation.json", "reservation")
    require(
        isinstance(reservation, dict)
        and set(reservation)
        == {
            "schema",
            "activity_monotonic_ns",
            "packet",
            "signal_sequence",
            "runner_process_ids",
        }
        and reservation["schema"] == 1
        and type(reservation["activity_monotonic_ns"]) is int
        and reservation["activity_monotonic_ns"] > 0
        and type(reservation["signal_sequence"]) is int
        and reservation["signal_sequence"] == 0
        and reservation["runner_process_ids"] == manifest["runner_process_ids"]
        and reservation["packet"] == str(packet.relative_to(ROOT)),
        "reservation record drifted",
    )
    prior_activity = reservation["activity_monotonic_ns"]
    validate_initialization(
        work / "initialization-progress.jsonl", packet, work, manifest
    )
    cursor = 0
    stopped = False
    for index, attempt in enumerate(attempts):
        spec = attempt_spec(index + 1)
        require(not stopped, "post-stop attempt evidence exists")
        event_count = 3 if attempt.get("pgid") is not None else 2
        rows = lifecycle[cursor : cursor + event_count]
        require(len(rows) == event_count, "lifecycle prefix lacks terminal bundle")
        validate_attempt_semantics(
            attempt, spec, manifest, packet, work, rows, prior_activity
        )
        cursor += event_count
        expected_packet.add(f"{spec['stem']}.attempt.json")
        expected_work.update(
            {
                f"{spec['stem']}.{suffix}"
                for suffix in (
                    "stdout",
                    "stderr",
                    "conditioning.json",
                    "post.json",
                )
            }
        )
        stopped = attempt["stop_classification"] is not None
        prior_activity = attempt["terminal_activity_monotonic_ns"]
    require(cursor == len(lifecycle), "orphan lifecycle event exists")
    expected_packet.update({"attempts.jsonl", "lifecycle.jsonl"})
    complete_form = len(attempts) == 4 and not stopped
    terminal_form = stopped and attempts[-1]["stop_classification"] is not None
    require(
        complete_form or terminal_form,
        "sealed packet is neither complete ABBA nor a terminal attempt prefix",
    )
    require(not (set(work_members) - expected_work), "orphan work evidence exists")
    require(set(packet_members) == expected_packet, "packet member set drifted")
    require(set(work_members) == expected_work, "work member set drifted")
    status, analysis = semantic_decision(attempts)
    final_identity = proposed_decision.get("final_identity")
    require(
        isinstance(final_identity, dict)
        and set(final_identity)
        == {
            "schema",
            "label",
            "captured_monotonic_ns",
            "identity",
            "matches_manifest",
        }
        and final_identity["schema"] == 1
        and final_identity["label"] == "final"
        and final_identity["matches_manifest"] is True,
        "final matched identity seal is missing",
    )
    typed_equal(
        final_identity["identity"],
        manifest_identity(manifest),
        "final identity binding",
    )
    signal_log, signal_cutoff = validate_signal_cutoff(
        packet, proposed_decision, attempts
    )
    require(
        attempts[-1]["terminal_activity_monotonic_ns"]
        <= final_identity["captured_monotonic_ns"]
        <= signal_cutoff["boundary_monotonic_ns"],
        "final identity/cutoff timeline drifted",
    )
    require(
        proposed_decision["status"] == status
        and proposed_decision["completed_attempts"] == len(attempts)
        and proposed_decision["expected_attempts"] == 4,
        "decision disposition drifted",
    )
    typed_equal(proposed_decision["analysis"], analysis, "decision analysis")
    expected_authority = (
        "one-separately-preregistered-force-only-pilot" if status == "GO" else "none"
    )
    require(
        proposed_decision["authority"] == expected_authority
        and proposed_decision["closure"]
        == (status if status in ("GO", "KILL") else "no-authority"),
        "decision authority drifted",
    )
    stop_reason = attempts[-1]["stop_reason"] if terminal_form else None
    expected_contract_error = (
        f"ContractDefect:{stop_reason}"
        if status == "implementation_or_contract_defect"
        else None
    )
    expected_inconclusive_error = (
        f"Inconclusive:{stop_reason}" if status == "inconclusive" else None
    )
    typed_equal(
        proposed_decision["contract_error"],
        expected_contract_error,
        "decision contract error",
    )
    typed_equal(
        proposed_decision["inconclusive_error"],
        expected_inconclusive_error,
        "decision inconclusive error",
    )
    typed_equal(
        proposed_decision["subordinate_evidence"],
        [],
        "decision subordinate evidence",
    )
    require(
        signal_log["authority_signal_count"]
        == sum(len(attempt["signal_events"]) for attempt in attempts),
        "signal authority escaped complete attempt ledgers",
    )
    return attempts, packet_members, work_members


def publish(
    decision,
    packet=PACKET,
    work=WORK,
    _after_decision=None,
    _after_semantic=None,
    _after_inventory=None,
    _before_final_snapshot=None,
):
    initial_packet = root_snapshot(packet)
    initial_work = root_snapshot(work)
    decision_path = packet / "decision.json"
    write_json(decision_path, decision)
    typed_equal(
        parse_json_bytes(stable_read_bytes(decision_path), "decision reread"),
        decision,
        "persisted/proposed decision",
    )
    if _after_decision is not None:
        _after_decision()
    typed_equal(
        parse_json_bytes(stable_read_bytes(decision_path), "decision post-hook reread"),
        decision,
        "post-hook decision",
    )
    validate_packet_semantics(decision, packet, work)
    semantic_packet = root_snapshot(packet)
    semantic_work = root_snapshot(work)
    require_snapshot_unchanged(initial_packet, semantic_packet, "packet pre-seal")
    require_snapshot_unchanged(initial_work, semantic_work, "work pre-seal")
    if _after_semantic is not None:
        _after_semantic()
    inventory = artifact_inventory(packet, work)
    inventory_path = packet / "artifact-inventory.json"
    write_json(inventory_path, inventory)
    typed_equal(
        parse_json_bytes(stable_read_bytes(inventory_path), "inventory reread"),
        inventory,
        "persisted/proposed inventory",
    )
    if _after_inventory is not None:
        _after_inventory()
    typed_equal(
        parse_json_bytes(
            stable_read_bytes(inventory_path), "inventory post-hook reread"
        ),
        inventory,
        "post-hook inventory",
    )
    if _before_final_snapshot is not None:
        _before_final_snapshot()
    final_packet = root_snapshot(packet)
    final_work = root_snapshot(work)
    require_snapshot_unchanged(semantic_packet, final_packet, "packet final")
    require_snapshot_unchanged(semantic_work, final_work, "work final")
    require(
        set(final_packet) == set(semantic_packet) | {"artifact-inventory.json"},
        "final packet membership drifted",
    )
    require(set(final_work) == set(semantic_work), "final work membership drifted")
    typed_equal(
        artifact_inventory(packet, work), inventory, "final recomputed inventory"
    )
    decision_bytes = stable_read_bytes(decision_path)
    inventory_bytes = stable_read_bytes(inventory_path)
    completion = {
        "schema": 1,
        "decision_sha256": sha_bytes(decision_bytes),
        "inventory_sha256": sha_bytes(inventory_bytes),
        "aggregate_sha256": sha_bytes(
            json_bytes(
                {
                    "decision_sha256": sha_bytes(decision_bytes),
                    "inventory_sha256": sha_bytes(inventory_bytes),
                }
            )
        ),
        "publication": "exclusive-fsynced-complete",
    }
    require(
        completion["decision_sha256"] == sha_bytes(decision_bytes)
        and completion["inventory_sha256"] == sha_bytes(inventory_bytes)
        and completion["aggregate_sha256"]
        == sha_bytes(
            json_bytes(
                {
                    "decision_sha256": sha_bytes(decision_bytes),
                    "inventory_sha256": sha_bytes(inventory_bytes),
                }
            )
        ),
        "final completion hashes drifted",
    )
    fsync_dir(packet)
    fsync_dir(work)
    fsync_dir(packet.parent)
    write_json(packet / "packet-complete.json", completion)
    fsync_dir(packet)
    return completion


def execute():
    global AUTHORITY_ENV, SIGNAL_CONTROLLER
    require(not PACKET.exists() and not WORK.exists(), "packet roots already exist")
    SIGNAL_CONTROLLER = SignalController()
    SIGNAL_CONTROLLER.install()
    env, environment_record = normalized_environment()
    AUTHORITY_ENV = env
    source_commit, build = source_and_build_identity(env)
    require(sha_file(CONTRACT) and sha_file(RUNNER), "contract/runner hash unavailable")
    require(sha_file(DESCRIBE) == DESCRIBE_SHA256, "tracked describe digest drifted")
    validate_tracked_describe(
        parse_json_bytes(DESCRIBE.read_bytes(), "tracked describe")
    )
    binary_path = ROOT / str(BINARY).removeprefix("./")
    binary_record = stable_file_record(binary_path)
    try:
        probe, probe_raw = run_probe(env, build)
    except HeadroomFailure as error:
        print(
            json.dumps(
                {
                    "schema": 1,
                    "status": "preflight-headroom-failure",
                    "authority": "none",
                    "attempts_reserved": 0,
                    "closure": "return-to-grammar",
                    "reason": str(error),
                },
                sort_keys=True,
            )
        )
        return
    except Inconclusive as error:
        print(
            json.dumps(
                {
                    "schema": 1,
                    "status": "inconclusive",
                    "authority": "none",
                    "attempts_reserved": 0,
                    "closure": "return-to-grammar",
                    "reason": str(error),
                },
                sort_keys=True,
            )
        )
        return
    if OPERATOR_SIGNALS:
        print(
            json.dumps(
                {
                    "schema": 1,
                    "status": "preflight-operator-signal",
                    "authority": "none",
                    "attempts_reserved": 0,
                    "closure": "return-to-grammar",
                    "reason": "operator signal before packet reservation",
                },
                sort_keys=True,
            )
        )
        return
    # Reservation is deliberately after the sole probe.
    PACKET.parent.mkdir(parents=True, exist_ok=True)
    PACKET.mkdir()
    WORK.mkdir()
    activity = time.monotonic_ns()
    write_json(
        WORK / "reservation.json",
        {
            "schema": 1,
            "activity_monotonic_ns": activity,
            "packet": str(PACKET.relative_to(ROOT)),
            "signal_sequence": len(OPERATOR_SIGNALS),
            "runner_process_ids": sorted({os.getpid(), os.getppid()}),
        },
    )
    fsync_dir(WORK)
    fsync_dir(PACKET.parent)
    initialization_path = WORK / "initialization-progress.jsonl"
    append_jsonl(
        initialization_path,
        {
            "schema": 1,
            "stage": "reservation",
            "monotonic_ns": time.monotonic_ns(),
            "evidence": {"reservation_sha256": sha_file(WORK / "reservation.json")},
        },
    )
    model_path_stamp_baseline = path_stamps()
    append_jsonl(
        initialization_path,
        {
            "schema": 1,
            "stage": "model-stamps",
            "monotonic_ns": time.monotonic_ns(),
            "evidence": {"model_path_stamps": model_path_stamp_baseline},
        },
    )
    manifest = {
        "schema": 1,
        "protocol": "v0653-a10b-parallel-pread-floor",
        "source_commit": source_commit,
        "build_identity": build,
        "contract": stable_file_record(CONTRACT),
        "tracked_describe": stable_file_record(DESCRIBE),
        "runner": stable_file_record(RUNNER),
        "binary": binary_record,
        "models": [
            {"path": str(path), "size": size, "pages": pages, "sha256": digest}
            for path, size, pages, digest in zip(
                MODELS, MODEL_SIZES, MODEL_PAGES, MODEL_SHA256
            )
        ],
        "model_path_stamps": model_path_stamp_baseline,
        "runner_process_ids": sorted({os.getpid(), os.getppid()}),
        "environment_record": environment_record,
        "commands": [child_command(arm) for arm in ORDER],
        "probe_command": probe_command(),
        "order": list(ORDER),
        "attempt_count": 4,
        "retry_count": 0,
        "deadlines_seconds": {
            "probe": PROBE_DEADLINE_S,
            "child": CHILD_DEADLINE_S,
            "identity": IDENTITY_DEADLINE_S,
        },
    }
    write_json(PACKET / "manifest.json", manifest)
    append_jsonl(
        initialization_path,
        {
            "schema": 1,
            "stage": "manifest",
            "monotonic_ns": time.monotonic_ns(),
            "evidence": {"manifest_sha256": sha_file(PACKET / "manifest.json")},
        },
    )
    write_exclusive(PACKET / "headroom-probe.stdout", probe_raw)
    require(
        sha_file(PACKET / "headroom-probe.stdout") == probe["raw_stdout_sha256"],
        "headroom probe raw capture drifted",
    )
    write_json(PACKET / "headroom-probe.json", probe)
    append_jsonl(
        initialization_path,
        {
            "schema": 1,
            "stage": "probe-artifacts",
            "monotonic_ns": time.monotonic_ns(),
            "evidence": {
                "probe_sha256": sha_file(PACKET / "headroom-probe.json"),
                "probe_stdout_sha256": sha_file(PACKET / "headroom-probe.stdout"),
            },
        },
    )
    fsync_dir(PACKET)
    fsync_dir(WORK)
    fsync_dir(PACKET.parent)
    append_jsonl(
        initialization_path,
        {
            "schema": 1,
            "stage": "roots-fsynced",
            "monotonic_ns": time.monotonic_ns(),
            "evidence": {"complete": True},
        },
    )
    fsync_dir(WORK)
    rows = []
    for position in range(1, 5):
        row, activity = run_one(attempt_spec(position), env, manifest, activity)
        rows.append(row)
        if row["stop_classification"] is not None:
            break
    status, analysis = semantic_decision(rows)
    require(
        (len(rows) == 4 and status in ("GO", "KILL"))
        or (
            rows
            and rows[-1]["stop_classification"]
            in ("implementation_or_contract_defect", "inconclusive")
        ),
        "runner stopped without a complete terminal attempt ledger",
    )
    final_identity = revalidate_live_identity(env, manifest, "final")
    cutoff = SIGNAL_CONTROLLER.final_cutoff(PACKET)
    require(
        cutoff["authority_signal_count"]
        == sum(len(row["signal_events"]) for row in rows),
        "operator signal occurred outside a complete attempt ledger",
    )
    stop_reason = rows[-1]["stop_reason"] if rows[-1]["stop_classification"] else None
    decision = {
        "schema": 1,
        "status": status,
        "authority": "one-separately-preregistered-force-only-pilot"
        if status == "GO"
        else "none",
        "completed_attempts": len(rows),
        "expected_attempts": 4,
        "contract_error": f"ContractDefect:{stop_reason}"
        if status == "implementation_or_contract_defect"
        else None,
        "inconclusive_error": f"Inconclusive:{stop_reason}"
        if status == "inconclusive"
        else None,
        "analysis": analysis,
        "closure": status if status in ("GO", "KILL") else "no-authority",
        "manifest_sha256": sha_file(PACKET / "manifest.json"),
        "headroom_probe_sha256": sha_file(PACKET / "headroom-probe.json"),
        "final_identity": final_identity,
        "signal_cutoff_sha256": sha_file(PACKET / "signal-cutoff.json"),
        "signal_log_sha256": sha_file(PACKET / "signal-log.json"),
        "subordinate_evidence": [],
    }
    try:
        publish(decision)
    except Exception as error:
        raise PublicationFailure(f"packet publication failed: {error}") from error
    print(json.dumps(decision, sort_keys=True))


def expect(exception, function, *args):
    try:
        function(*args)
    except exception:
        return
    raise RuntimeError(f"expected {exception.__name__}")


def fixture_build():
    return {
        "schema_version": 2,
        "build_commit": "a" * 40,
        "build_commit_short": "a" * 9,
        "build_dirty": False,
        "build_source_state": "state",
        "overrides": [],
        "problems": [],
        "runtime_commit": "a" * 40,
        "runtime_dirty": False,
        "runtime_source_state": "state",
        "stamp_error": None,
        "stamp_source": "git",
        "status": "match",
    }


def fixture_admission(process=0, headroom=HEADROOM):
    working_fits = headroom >= HEADROOM
    process_fits = process == 0 or process >= HEADROOM
    if working_fits and process_fits:
        reason = (
            "admitted_process_budget_omitted"
            if process == 0
            else "admitted_with_process_budget"
        )
    elif not working_fits and process > 0 and not process_fits:
        reason = "both_insufficient"
    elif not working_fits:
        reason = "working_set_insufficient"
    else:
        reason = "process_insufficient"
    return {
        "admitted": working_fits and process_fits,
        "reason": reason,
        "required_bytes": HEADROOM,
        "scratch_upper_bytes": HEADROOM,
        "reserve_bytes": 0,
        "allow_zero_process_budget": True,
        "zero_process_budget_semantics": "omitted-limit-sentinel",
        "working_set_headroom_bytes": headroom,
        "signals": {
            "recommended_max_working_set_size": headroom + 7,
            "current_allocated_size": 7,
            "working_set_headroom_bytes": headroom,
            "process_limit_remaining_bytes": process,
        },
    }


def fixture_probe():
    return {
        "schema_version": 1,
        "mode": "metal-memory-headroom-probe",
        "model_label": str(MODEL),
        "endpoint": PROBE_ENDPOINT,
        "model_opened": False,
        "payload_allocated": False,
        "device_name": "Apple M4 Max",
        "unified_memory": True,
        "max_buffer_length": 77_309_411_328,
        "memory_admission": fixture_admission(),
        "build_identity": fixture_build(),
    }


def fixture_records():
    rows = []
    for index, (path, size, digest, pages) in enumerate(
        zip(MODELS, MODEL_SIZES, MODEL_SHA256, MODEL_PAGES)
    ):
        stamp_value = {
            "device": 1,
            "inode": index + 2,
            "size": size,
            "mtime_sec": 3,
            "mtime_nsec": 4,
            "ctime_sec": 5,
            "ctime_nsec": 6,
        }
        rows.append(
            {
                "path": str(path),
                "descriptor_stamp_before": stamp_value,
                "descriptor_stamp_after": dict(stamp_value),
                "bytes_hashed": size,
                "sha256": digest,
                "hash_started_monotonic_ns": 100 + index * 10,
                "hash_completed_monotonic_ns": 101 + index * 10,
                "page_size": PAGE_SIZE,
                "total_pages": pages,
                "resident_pages": pages,
                "all_pages_resident": True,
                "residency_checked_monotonic_ns": 102 + index * 10,
            }
        )
    return rows


def fixture_result(arm="A", ready=3_000_000, cpu=1_000_000):
    if arm == "A":
        timing = {
            "ready_wall_ms": ready / 1000,
            "ready_us": ready,
            "allocation_wall_ms": None,
            "allocation_us": None,
            "source_resolution_wall_ms": None,
            "source_us": None,
            "source_resolution_us": None,
            "copy_wall_ms": None,
            "copy_us": None,
            "binding_wall_ms": 100.0,
            "binding_us": 100_000,
            "unattributed_wall_ms": None,
            "unattributed_us": None,
            "teardown_wall_ms": 0.001,
            "teardown_us": 1,
        }
        copy_throughput = None
    else:
        copy_us = ready - 500_000
        require(copy_us > 0, "fixture ready duration is too short")
        timing = {
            "ready_wall_ms": ready / 1000,
            "ready_us": ready,
            "allocation_wall_ms": 200.0,
            "allocation_us": 200_000,
            "source_resolution_wall_ms": 200.0,
            "source_us": 200_000,
            "source_resolution_us": 200_000,
            "copy_wall_ms": copy_us / 1000,
            "copy_us": copy_us,
            "binding_wall_ms": 100.0,
            "binding_us": 100_000,
            "unattributed_wall_ms": 0.0,
            "unattributed_us": 0,
            "teardown_wall_ms": 0.001,
            "teardown_us": 1,
        }
        copy_throughput = 1.0
    value = {key: None for key in TOP_KEYS}
    value.update(
        schema_version=3,
        arm="copied" if arm == "A" else "parallel-pread",
        profile=PROFILE,
        model=str(MODEL),
        architecture="qwen35moe",
        architecture_tuple=dict(ARCHITECTURE),
        tied_embeddings=False,
        mtp_present=False,
        shard_mapped_lengths=list(MODEL_SIZES),
        descriptor_layout_digest=DESCRIPTOR,
        inventory_digest=INVENTORY,
        native_quant_embedding=True,
        native_quant_embedding_supported=True,
        native_quant_embedding_selection="bench-force-native-if-supported",
        page_size=PAGE_SIZE,
        required_alignment=32,
        max_buffer_length=77_309_411_328,
        device_name="Apple M4 Max",
        unified_memory=True,
        request_count=COUNT,
        resource_count=COUNT,
        binding_count=COUNT,
        logical_copy_bytes=COPY_BYTES,
        physical_copy_bytes=COPY_BYTES,
        resource_modes=dict(RESOURCE_MODES),
        parallel_copy_schedule=schedule_json(),
        timing=timing,
        throughput={
            "ready_gbps_decimal": 1.0,
            "copy_gbps_decimal": copy_throughput,
        },
        rusage={
            "timer_minor_faults": 0,
            "timer_major_faults": 0,
            "timer_block_inputs": 0,
            "timer_swaps": 0,
            "user_cpu_us": cpu // 2,
            "system_cpu_us": cpu - cpu // 2,
            "total_cpu_us": cpu,
            "cpu_per_wall": cpu / ready,
        },
        proc_rusage_v4={key: 0 for key in PROC_KEYS},
        metal_allocated_bytes={
            "before": 0,
            "ready": COPY_BYTES,
            "after_drop": 0,
            "drop_valid": True,
        },
        correctness={
            "passed": True,
            "payload_bytes_checked": COPY_BYTES,
            "entries_checked": COUNT,
        },
        worker_count=0 if arm == "A" else 4,
        build_identity=fixture_build(),
        embedding_policy="force-native-if-supported",
        memory_admission=fixture_admission(),
        retained_shard_stamps={"before_timing": [], "after_verification": []},
        endpoint=ENDPOINT,
        implementation_seal={
            "schema_version": 1,
            "seal": IMPLEMENTATION_SEAL,
            "scope": ["materialize_copied", "materialize_parallel_pread"],
            "no_gpu_command": True,
            "build_identity_bound": True,
        },
    )
    stamps = []
    for index, row in enumerate(fixture_records()):
        stamps.append(
            {"shard_idx": index, "path": row["path"], **row["descriptor_stamp_before"]}
        )
    value["retained_shard_stamps"] = {
        "before_timing": stamps,
        "after_verification": [dict(x) for x in stamps],
    }
    return value


def scoring_rows(d=1_500_000, cpu_ratio=1.5, rss_ratio=1.05, foot_ratio=1.05):
    rows = []
    for position, arm in enumerate(ORDER, 1):
        ready = 3_000_000 if arm == "A" else 3_000_000 - d
        cpu = 1_000_000 if arm == "A" else round(1_000_000 * cpu_ratio)
        rows.append(
            {
                **attempt_spec(position),
                "stop_classification": None,
                "result": fixture_result(arm, ready, cpu),
                "process_resources": {
                    "maximum_resident_set_size": 1000
                    if arm == "A"
                    else round(1000 * rss_ratio),
                    "peak_memory_footprint": 1000
                    if arm == "A"
                    else round(1000 * foot_ratio),
                },
            }
        )
    return rows


def fixture_vm(timestamp):
    return {
        "pageouts": 3,
        "compressions": 5,
        "swapouts": 7,
        "compressor_stored_pages": 11,
        "compressor_occupied_pages": 13,
        "swap_used_bytes": 17,
        "errors": [],
        "captured_monotonic_ns": timestamp,
    }


def fixture_host(timestamp):
    return {
        "thermal": "No thermal warning level has been recorded\n"
        "No performance warning level has been recorded\n",
        "battery": "AC Power\n",
        "memory_pressure": "System-wide memory free percentage: 90%\n",
        "memory_available_percent": 90,
        "process_snapshot": "",
        "ignored_process_ids": [1, 2],
        "competing_processes": [],
        "errors": [],
        "valid": True,
        "captured_monotonic_ns": timestamp,
    }


def fixture_time_bytes(rss=1000, footprint=1000):
    values = {label: 0 for label in TIME_LABELS}
    values["maximum resident set size"] = rss
    values["peak memory footprint"] = footprint
    return (
        "  1.00 real  0.20 user  0.30 sys\n"
        + "".join(f"  {values[label]}  {label}\n" for label in TIME_LABELS)
    ).encode()


def make_complete_packet_fixture(root, saving_us=1_500_000):
    packet, work = root / "packet", root / "work"
    packet.mkdir()
    work.mkdir()
    env = {"HOME": "/fixture", "PATH": "/usr/bin:/bin", "TMPDIR": "/tmp"}
    environment_record = {"environment": env, "removed_keys": ["SECRET"]}
    environment_record["sha256"] = sha_bytes(json_bytes(environment_record))
    dummy_record = {
        "path": "fixture",
        "size_bytes": 1,
        "sha256": "0" * 64,
        "descriptor_stamp": {
            "device": 1,
            "inode": 1,
            "size": 1,
            "mtime_sec": 1,
            "mtime_nsec": 0,
            "ctime_sec": 1,
            "ctime_nsec": 0,
        },
    }
    model_stamps = [
        {"path": row["path"], **row["descriptor_stamp_before"]}
        for row in fixture_records()
    ]
    manifest = {
        "schema": 1,
        "protocol": "v0653-a10b-parallel-pread-floor",
        "source_commit": "a" * 40,
        "build_identity": fixture_build(),
        "contract": dict(dummy_record, path=str(CONTRACT.relative_to(ROOT))),
        "tracked_describe": dict(dummy_record, path=str(DESCRIBE.relative_to(ROOT))),
        "runner": dict(dummy_record, path=str(RUNNER.relative_to(ROOT))),
        "binary": dict(
            dummy_record,
            path=str((ROOT / str(BINARY).removeprefix("./")).relative_to(ROOT)),
        ),
        "models": [
            {"path": str(path), "size": size, "pages": pages, "sha256": digest}
            for path, size, pages, digest in zip(
                MODELS, MODEL_SIZES, MODEL_PAGES, MODEL_SHA256
            )
        ],
        "model_path_stamps": model_stamps,
        "runner_process_ids": [1, 2],
        "environment_record": environment_record,
        "commands": [child_command(arm) for arm in ORDER],
        "probe_command": probe_command(),
        "order": list(ORDER),
        "attempt_count": 4,
        "retry_count": 0,
        "deadlines_seconds": {
            "probe": PROBE_DEADLINE_S,
            "child": CHILD_DEADLINE_S,
            "identity": IDENTITY_DEADLINE_S,
        },
    }
    write_json(packet / "manifest.json", manifest)
    probe_raw = json_bytes(fixture_probe(), True)
    write_exclusive(packet / "headroom-probe.stdout", probe_raw)
    disposition_value = {
        "pgid": 4000,
        "samples": [{"captured_monotonic_ns": 3, "members": [], "error": None}],
        "no_live_group_observed": True,
    }
    probe_record = {
        "schema": 1,
        "command": probe_command(),
        "stdout": fixture_probe(),
        "raw_stdout_sha256": sha_bytes(probe_raw),
        "process": {
            "pid": 4000,
            "pgid": 4000,
            "attempted_monotonic_ns": 1,
            "acquired_monotonic_ns": 2,
            "completed_monotonic_ns": 3,
            "returncode": 0,
            "reaped": True,
            "group_disposition": disposition_value,
            "ownership_error": None,
            "output_overflow": False,
            "drain_errors": [],
            "timed_out": False,
            "operator_interrupted": False,
            "cleanup_actions": [],
            "cleanup_errors": [],
            "operator_signals": [],
        },
    }
    write_json(packet / "headroom-probe.json", probe_record)
    prior_activity = 1_000_000_000
    write_json(
        work / "reservation.json",
        {
            "schema": 1,
            "activity_monotonic_ns": prior_activity,
            "packet": str(packet.relative_to(ROOT)),
            "signal_sequence": 0,
            "runner_process_ids": [1, 2],
        },
    )
    init_path = work / "initialization-progress.jsonl"
    init_rows = (
        ("reservation", {"reservation_sha256": sha_file(work / "reservation.json")}),
        ("model-stamps", {"model_path_stamps": model_stamps}),
        ("manifest", {"manifest_sha256": sha_file(packet / "manifest.json")}),
        (
            "probe-artifacts",
            {
                "probe_sha256": sha_file(packet / "headroom-probe.json"),
                "probe_stdout_sha256": sha_file(packet / "headroom-probe.stdout"),
            },
        ),
        ("roots-fsynced", {"complete": True}),
    )
    for index, (stage, evidence) in enumerate(init_rows, 1):
        append_jsonl(
            init_path,
            {
                "schema": 1,
                "stage": stage,
                "monotonic_ns": prior_activity + index,
                "evidence": evidence,
            },
        )
    attempts = []
    for index, spec in enumerate([attempt_spec(i) for i in range(1, 5)], 1):
        base = prior_activity + COOLDOWN_NS
        before_vm, spawn_vm = fixture_vm(base + 3), fixture_vm(base + 35)
        identity = {
            "schema": 1,
            "label": f"pre-attempt-{index}",
            "captured_monotonic_ns": base + 1,
            "identity": manifest_identity(manifest),
            "matches_manifest": True,
        }
        conditioned_records = fixture_records()
        for source_index, source in enumerate(conditioned_records):
            source["hash_started_monotonic_ns"] = base + 20 + source_index * 4
            source["hash_completed_monotonic_ns"] = base + 21 + source_index * 4
            source["residency_checked_monotonic_ns"] = base + 30 + source_index
        conditioning = {
            "schema": 1,
            "pre_attempt_identity": identity,
            "cooldown": {
                "prior_activity_monotonic_ns": prior_activity,
                "required_interval_ns": COOLDOWN_NS,
                "eligible_monotonic_ns": base,
                "started_monotonic_ns": base,
                "completed_monotonic_ns": base,
                "observed_interval_ns": COOLDOWN_NS,
            },
            "host_before_conditioning": fixture_host(base + 2),
            "vm_before_conditioning": before_vm,
            "source_path_stamps_before": model_stamps,
            "source_path_stamps_before_monotonic_ns": base + 4,
            "source_records": conditioned_records,
            "residency_proved_monotonic_ns": base + 32,
            "host_before_spawn": fixture_host(base + 34),
            "vm_before_spawn": spawn_vm,
            "source_path_stamps_before_spawn": model_stamps,
            "source_path_stamps_before_spawn_monotonic_ns": base + 35,
            "preconditioning_interval": vm_interval(
                "preconditioning", before_vm, spawn_vm
            ),
        }
        conditioning_path = work / f"{spec['stem']}.conditioning.json"
        write_json(conditioning_path, conditioning)
        ready = 3_000_000 if spec["arm"] == "A" else 3_000_000 - saving_us
        result = fixture_result(spec["arm"], ready, 1_000_000)
        stdout_path, stderr_path = (
            work / f"{spec['stem']}.stdout",
            work / f"{spec['stem']}.stderr",
        )
        write_exclusive(stdout_path, json_bytes(result, True))
        write_exclusive(stderr_path, fixture_time_bytes())
        after_vm = fixture_vm(base + 45)
        post = {
            "schema": 1,
            "host_after_exit": fixture_host(base + 44),
            "vm_after_exit": after_vm,
            "source_path_stamps_after_exit": model_stamps,
            "source_path_stamps_after_exit_monotonic_ns": base + 47,
            "child_interval": vm_interval("child", spawn_vm, after_vm),
            "operation_errors": [],
        }
        post_path = work / f"{spec['stem']}.post.json"
        write_json(post_path, post)
        pid = 5000 + index
        group = {
            "pgid": pid,
            "samples": [
                {"captured_monotonic_ns": base + 40, "members": [], "error": None}
            ],
            "no_live_group_observed": True,
        }
        launch = {
            "schema": 1,
            "event": "launch",
            **spec,
            "environment_sha256": environment_record["sha256"],
            "conditioning_sha256": sha_file(conditioning_path),
            "residency_proved_monotonic_ns": base + 32,
            "monotonic_ns": base + 36,
        }
        acquired = {
            "schema": 1,
            "event": "acquired",
            "stem": spec["stem"],
            "pid": pid,
            "pgid": pid,
            "monotonic_ns": base + 38,
        }
        completion = {
            "schema": 1,
            "event": "completion",
            "stem": spec["stem"],
            "pid": pid,
            "pgid": pid,
            "spawn_error": None,
            "ownership_error": None,
            "returncode": 0,
            "reaped": True,
            "interrupted": False,
            "output_overflow": False,
            "drain_errors": [],
            "operator_interrupted": False,
            "timed_out": False,
            "cleanup_actions": [],
            "cleanup_errors": [],
            "operator_signals": [],
            "attempted_monotonic_ns": base + 37,
            "acquired_monotonic_ns": base + 38,
            "completed_monotonic_ns": base + 41,
            "group_disposition": group,
            "group_disposition_initial": group,
        }
        for event in (launch, acquired, completion):
            append_jsonl(packet / "lifecycle.jsonl", event)
        attempt = {
            "schema": 1,
            **spec,
            "environment_sha256": environment_record["sha256"],
            "pid": pid,
            "pgid": pid,
            "spawn_error": None,
            "ownership_error": None,
            "returncode": 0,
            "reaped": True,
            "interrupted": False,
            "output_overflow": False,
            "drain_errors": [],
            "timed_out": False,
            "operator_interrupted": False,
            "cleanup_actions": [],
            "cleanup_errors": [],
            "operator_signals": [],
            "signal_start_sequence": 0,
            "signal_end_sequence": 0,
            "signal_events": [],
            "group_disposition": group,
            "group_disposition_initial": group,
            "launch_attempted_monotonic_ns": base + 37,
            "launch_acquired_monotonic_ns": base + 38,
            "completion_monotonic_ns": base + 41,
            "residency_to_acquired_ns": 6,
            "terminal_activity_monotonic_ns": base + 48,
            "process_resources": parse_time(stderr_path.read_bytes()),
            "result": result,
            "parse_error": None,
            "defect_reasons": [],
            "inconclusive_reasons": [],
            "stop_classification": None,
            "stop_reason": None,
            "artifacts": {
                "stdout": stable_file_record(stdout_path),
                "stderr": stable_file_record(stderr_path),
                "conditioning": stable_file_record(conditioning_path),
                "post": stable_file_record(post_path),
            },
        }
        bundle_path = packet / f"{spec['stem']}.attempt.json"
        write_json(bundle_path, attempt)
        attempt["bundle_sha256"] = sha_file(bundle_path)
        append_jsonl(packet / "attempts.jsonl", attempt)
        attempts.append(attempt)
        prior_activity = attempt["terminal_activity_monotonic_ns"]
    analysis = score_rows(attempts)
    status = "GO" if analysis["qualifies"] else "KILL"
    final_identity = {
        "schema": 1,
        "label": "final",
        "captured_monotonic_ns": prior_activity + 1,
        "identity": manifest_identity(manifest),
        "matches_manifest": True,
    }
    signal_log = {
        "schema": 1,
        "boundary_monotonic_ns": prior_activity + 2,
        "event_count": 0,
        "events": [],
        "pending_signals": [],
        "authority_signal_count": 0,
    }
    write_json(packet / "signal-log.json", signal_log)
    signal_cutoff = {
        "schema": 1,
        "event": "final-signal-cutoff",
        "boundary_monotonic_ns": prior_activity + 2,
        "signal_log_sha256": sha_file(packet / "signal-log.json"),
        "authority_signal_count": 0,
    }
    write_json(packet / "signal-cutoff.json", signal_cutoff)
    decision = {
        "schema": 1,
        "status": status,
        "authority": "one-separately-preregistered-force-only-pilot"
        if status == "GO"
        else "none",
        "completed_attempts": 4,
        "expected_attempts": 4,
        "contract_error": None,
        "inconclusive_error": None,
        "analysis": analysis,
        "closure": status,
        "manifest_sha256": sha_file(packet / "manifest.json"),
        "headroom_probe_sha256": sha_file(packet / "headroom-probe.json"),
        "final_identity": final_identity,
        "signal_cutoff_sha256": sha_file(packet / "signal-cutoff.json"),
        "signal_log_sha256": sha_file(packet / "signal-log.json"),
        "subordinate_evidence": [],
    }
    return packet, work, decision


def make_terminal_packet_fixture(root, classification):
    require(
        classification in {"implementation_or_contract_defect", "inconclusive"},
        "terminal fixture classification drifted",
    )
    packet, work, decision = make_complete_packet_fixture(root)
    attempts = read_jsonl_file(packet / "attempts.jsonl", "fixture attempts")[:2]
    terminal = attempts[-1]
    if classification == "inconclusive":
        terminal["result"]["rusage"]["timer_block_inputs"] = 1
        stdout_path = work / "p2-b.stdout"
        stdout_path.unlink()
        write_exclusive(stdout_path, json_bytes(terminal["result"], True))
        terminal["artifacts"]["stdout"] = stable_file_record(stdout_path)
        terminal["inconclusive_reasons"] = ["timer_block_input_nonzero"]
        terminal["stop_reason"] = "timer_block_input_nonzero"
    else:
        post_path = work / "p2-b.post.json"
        post = read_json_file(post_path, "fixture terminal post")
        before = post["child_interval"]["before"]
        post["vm_after_exit"]["swapouts"] = before["swapouts"] - 1
        post["child_interval"] = vm_interval("child", before, post["vm_after_exit"])
        post_path.unlink()
        write_json(post_path, post)
        terminal["artifacts"]["post"] = stable_file_record(post_path)
        terminal["defect_reasons"] = ["child_swapouts_regressed"]
        terminal["inconclusive_reasons"] = ["child_swapouts_grew"]
        terminal["stop_reason"] = "child_swapouts_regressed"
    terminal["stop_classification"] = classification
    bundle_path = packet / "p2-b.attempt.json"
    bundle_path.unlink()
    bundle = dict(terminal)
    bundle.pop("bundle_sha256")
    write_json(bundle_path, bundle)
    terminal["bundle_sha256"] = sha_file(bundle_path)
    for name in ("p3-b.attempt.json", "p4-a.attempt.json"):
        (packet / name).unlink()
    for stem in ("p3-b", "p4-a"):
        for suffix in ("stdout", "stderr", "conditioning.json", "post.json"):
            (work / f"{stem}.{suffix}").unlink()
    attempts_path = packet / "attempts.jsonl"
    attempts_path.unlink()
    for attempt in attempts:
        append_jsonl(attempts_path, attempt)
    lifecycle = read_jsonl_file(packet / "lifecycle.jsonl", "fixture lifecycle")[:6]
    lifecycle_path = packet / "lifecycle.jsonl"
    lifecycle_path.unlink()
    for event in lifecycle:
        append_jsonl(lifecycle_path, event)
    decision.update(
        status=classification,
        authority="none",
        completed_attempts=2,
        contract_error=f"ContractDefect:{terminal['stop_reason']}"
        if classification == "implementation_or_contract_defect"
        else None,
        inconclusive_error=f"Inconclusive:{terminal['stop_reason']}"
        if classification == "inconclusive"
        else None,
        analysis=None,
        closure="no-authority",
        subordinate_evidence=[],
    )
    return packet, work, decision


def make_spawn_failure_packet_fixture(root):
    packet, work, decision = make_complete_packet_fixture(root)
    attempts = read_jsonl_file(packet / "attempts.jsonl", "fixture attempts")[:2]
    terminal = attempts[-1]
    spawn_error = "OSError:fixture spawn failure"
    for suffix in ("stdout", "stderr"):
        path = work / f"p2-b.{suffix}"
        path.unlink()
        write_exclusive(path, b"")
        terminal["artifacts"][suffix] = stable_file_record(path)
    terminal.update(
        pid=None,
        pgid=None,
        spawn_error=spawn_error,
        ownership_error=None,
        returncode=None,
        reaped=False,
        interrupted=False,
        output_overflow=False,
        drain_errors=[],
        timed_out=False,
        operator_interrupted=False,
        cleanup_actions=[],
        cleanup_errors=[],
        operator_signals=[],
        signal_start_sequence=0,
        signal_end_sequence=0,
        signal_events=[],
        group_disposition=None,
        group_disposition_initial=None,
        launch_acquired_monotonic_ns=None,
        residency_to_acquired_ns=None,
        process_resources=None,
        result=None,
        parse_error=None,
        defect_reasons=[],
        inconclusive_reasons=[],
        stop_classification="inconclusive",
        stop_reason=spawn_error,
    )
    bundle_path = packet / "p2-b.attempt.json"
    bundle_path.unlink()
    bundle = dict(terminal)
    bundle.pop("bundle_sha256")
    write_json(bundle_path, bundle)
    terminal["bundle_sha256"] = sha_file(bundle_path)
    for name in ("p3-b.attempt.json", "p4-a.attempt.json"):
        (packet / name).unlink()
    for stem in ("p3-b", "p4-a"):
        for suffix in ("stdout", "stderr", "conditioning.json", "post.json"):
            (work / f"{stem}.{suffix}").unlink()
    attempts_path = packet / "attempts.jsonl"
    attempts_path.unlink()
    for attempt in attempts:
        append_jsonl(attempts_path, attempt)
    lifecycle = read_jsonl_file(packet / "lifecycle.jsonl", "fixture lifecycle")
    spawn_completion = dict(lifecycle[5])
    spawn_completion.update(
        pid=None,
        pgid=None,
        spawn_error=spawn_error,
        ownership_error=None,
        returncode=None,
        reaped=False,
        interrupted=False,
        output_overflow=False,
        drain_errors=[],
        operator_interrupted=False,
        timed_out=False,
        cleanup_actions=[],
        cleanup_errors=[],
        operator_signals=[],
        acquired_monotonic_ns=None,
        group_disposition=None,
        group_disposition_initial=None,
    )
    lifecycle_path = packet / "lifecycle.jsonl"
    lifecycle_path.unlink()
    for event in [*lifecycle[:4], spawn_completion]:
        append_jsonl(lifecycle_path, event)
    decision.update(
        status="inconclusive",
        authority="none",
        completed_attempts=2,
        contract_error=None,
        inconclusive_error=f"Inconclusive:{spawn_error}",
        analysis=None,
        closure="no-authority",
        subordinate_evidence=[],
    )
    return packet, work, decision


def rewrite_fixture_attempt(packet, index, mutate):
    attempts_path = packet / "attempts.jsonl"
    attempts = read_jsonl_file(attempts_path, "fixture attempt rewrite")
    mutate(attempts[index])
    bundle_path = packet / f"{attempts[index]['stem']}.attempt.json"
    bundle_path.unlink()
    bundle = dict(attempts[index])
    bundle.pop("bundle_sha256")
    write_json(bundle_path, bundle)
    attempts[index]["bundle_sha256"] = sha_file(bundle_path)
    attempts_path.unlink()
    for attempt in attempts:
        append_jsonl(attempts_path, attempt)
    return attempts[index]


def run_self_test():
    # Strict JSON, UTF-8, duplicate, nonfinite, and trailing-byte rejection.
    typed_equal(parse_json_bytes(b'{"a":1}\n', "fixture"), {"a": 1}, "JSON")
    for raw in (b'{"a":1,"a":2}', b'{"a":NaN}', b'{"a":1} x', b"\xff"):
        expect(ContractDefect, parse_json_bytes, raw, "fixture")
    require(sha_file(DESCRIBE) == DESCRIBE_SHA256, "tracked describe digest drifted")
    validate_tracked_describe(
        parse_json_bytes(DESCRIBE.read_bytes(), "tracked describe")
    )

    values = {label: 0 for label in TIME_LABELS}
    values["maximum resident set size"] = 1
    values["peak memory footprint"] = 2
    stderr = (
        "  1.00 real  0.20 user  0.30 sys\n"
        + "".join(f"  {values[label]}  {label}\n" for label in TIME_LABELS)
    ).encode()
    require(parse_time(stderr)["total_cpu_s"] == 0.5, "time CPU parse failed")
    expect(ContractDefect, parse_time, stderr + b"extra\n")

    env, record = normalized_environment(
        {
            "HOME": "/home",
            "PATH": "/bin",
            "TMPDIR": "/tmp",
            "QWEN_X": "1",
            "METAL_X": "2",
            "MTL_X": "3",
            "RUST_LOG": "debug",
            "KEEP": "yes",
        }
    )
    typed_equal(
        env, {"HOME": "/home", "PATH": "/bin", "TMPDIR": "/tmp"}, "scrubbed environment"
    )
    require(
        record["removed_keys"] == ["KEEP", "METAL_X", "MTL_X", "QWEN_X", "RUST_LOG"],
        "removed environment record drifted",
    )

    for process in (0, HEADROOM, HEADROOM + 1):
        validate_admission(fixture_admission(process), "fixture admission")
    expect(ContractDefect, validate_admission, fixture_admission(-1), "fixture")
    for malformed in (None, True):
        value = fixture_admission()
        value["signals"]["process_limit_remaining_bytes"] = malformed
        expect(ContractDefect, validate_admission, value, "fixture")
    missing_process = fixture_admission()
    missing_process["signals"].pop("process_limit_remaining_bytes")
    expect(ContractDefect, validate_admission, missing_process, "fixture")
    failed_process = fixture_admission(HEADROOM - 1)
    expect(HeadroomFailure, validate_admission, failed_process, "fixture")
    failed_headroom = fixture_admission(0, HEADROOM - 1)
    expect(
        HeadroomFailure,
        validate_admission,
        failed_headroom,
        "fixture",
    )
    validate_probe(fixture_probe(), fixture_build())
    malformed_probe = fixture_probe()
    malformed_probe["model_opened"] = True
    expect(ContractDefect, validate_probe, malformed_probe, fixture_build())

    build = fixture_build()
    validate_result(fixture_result(), "A", build, fixture_records())
    validate_result(
        fixture_result("B", 1_500_000, 1_500_000), "B", build, fixture_records()
    )
    for mutate in ("schema", "schedule", "topology", "stamp"):
        value = fixture_result()
        if mutate == "schema":
            value.pop("endpoint")
        elif mutate == "schedule":
            value["parallel_copy_schedule"]["cuts"][0] += 1
        elif mutate == "topology":
            value["resource_modes"]["observed_storage"] = "private"
        elif mutate == "stamp":
            value["retained_shard_stamps"]["after_verification"][0]["inode"] += 1
        expect(ContractDefect, validate_result, value, "A", build, fixture_records())
    io_activity = fixture_result()
    io_activity["rusage"]["timer_block_inputs"] = 1
    validate_result(io_activity, "A", build, fixture_records())
    require(
        classify_attempt(
            {"spawn_error": None, "returncode": 0},
            None,
            [],
            ["timer_block_input_nonzero"],
        )[0]
        == "inconclusive",
        "measured block input must be inconclusive",
    )

    base = {
        "pageouts": 3,
        "compressions": 5,
        "swapouts": 7,
        "compressor_stored_pages": 11,
        "compressor_occupied_pages": 13,
        "swap_used_bytes": 17,
        "errors": [],
        "captured_monotonic_ns": 1,
    }
    same = dict(
        base,
        pageouts=4,
        compressions=6,
        compressor_stored_pages=10,
        compressor_occupied_pages=12,
        captured_monotonic_ns=2,
    )
    interval = vm_interval("fixture", base, same)
    require(
        not interval["defect_reasons"] and not interval["inconclusive_reasons"],
        "valid VM interval rejected",
    )
    growth = dict(
        same,
        swapouts=8,
        swap_used_bytes=18,
        compressor_stored_pages=12,
        compressor_occupied_pages=14,
    )
    require(
        len(vm_interval("fixture", base, growth)["inconclusive_reasons"]) == 4,
        "VM growth gates drifted",
    )
    regressed = dict(base, swapouts=6)
    require(
        "fixture_swapouts_regressed"
        in vm_interval("fixture", base, regressed)["defect_reasons"],
        "VM regression classification drifted",
    )
    missing = dict(base, swapouts=None)
    require(
        "fixture_swapouts_unavailable"
        in vm_interval("fixture", base, missing)["defect_reasons"],
        "missing VM counter precedence drifted",
    )
    capture_failed = dict(base, errors=["fixture"])
    require(
        "fixture_capture_failed"
        in vm_interval("fixture", base, capture_failed)["defect_reasons"],
        "VM capture failure precedence drifted",
    )

    records = fixture_records()
    require(
        [row["total_pages"] for row in records] == list(MODEL_PAGES)
        and [row["sha256"] for row in records] == list(MODEL_SHA256)
        and len({row["descriptor_stamp_before"]["inode"] for row in records}) == 3,
        "three-source synthetic evidence drifted",
    )
    require(
        launch_delay_within_limit(10, 10 + LAUNCH_LIMIT_NS)
        and not launch_delay_within_limit(10, 11 + LAUNCH_LIMIT_NS)
        and not launch_delay_within_limit(10, 9),
        "five-second boundary drifted",
    )

    specs = [attempt_spec(index) for index in range(1, 5)]
    require(
        [row["arm"] for row in specs] == ["A", "B", "B", "A"]
        and [row["pair"] for row in specs] == [1, 1, 2, 2],
        "ABBA plan drifted",
    )
    analysis = score_rows(scoring_rows())
    require(
        analysis["qualifies"] and analysis["pairs"][1]["orientation"] == "A4-B3",
        "inclusive threshold or pair orientation drifted",
    )
    require(
        not score_rows(scoring_rows(d=1_499_999))["qualifies"],
        "saving threshold weakened",
    )
    require(
        not score_rows(scoring_rows(cpu_ratio=1.501))["qualifies"]
        and not score_rows(scoring_rows(rss_ratio=1.051))["qualifies"]
        and not score_rows(scoring_rows(foot_ratio=1.051))["qualifies"],
        "ratio thresholds weakened",
    )
    require(
        disposition(True, True, True, True) == "implementation_or_contract_defect"
        and disposition(False, True, True, True) == "inconclusive"
        and disposition(False, False, True, False) == "KILL"
        and disposition(False, False, True, True) == "GO",
        "precedence drifted",
    )
    spawn = {"spawn_error": "fixture", "returncode": None}
    require(
        classify_attempt(spawn, None, [], [])[0] == "inconclusive"
        and classify_attempt(spawn, "parse", [], [])[0]
        == "implementation_or_contract_defect",
        "spawn/parse precedence drifted",
    )
    # A stop consumes its sole slot; plan execution cannot advance it.
    calls = []
    for spec in specs:
        calls.append(spec["position"])
        if spec["position"] == 2:
            break
    require(calls == [1, 2], "stop/no-retry lifecycle drifted")

    helper_env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin")}
    timeout = bounded_child(
        [sys.executable, "-c", "import time; time.sleep(2)"],
        helper_env,
        deadline_s=0.05,
    )
    require(
        timeout["timed_out"]
        and timeout["cleanup_actions"]
        and timeout["group_disposition"]["no_live_group_observed"],
        "deadline containment failed",
    )
    overflow = bounded_child(
        [
            sys.executable,
            "-c",
            "import sys,time; sys.stdout.write('x'*100000);"
            "sys.stdout.flush(); time.sleep(2)",
        ],
        helper_env,
        deadline_s=2,
        max_output=32,
    )
    require(
        overflow["output_overflow"]
        and overflow["cleanup_actions"]
        and overflow["group_disposition"]["no_live_group_observed"],
        "output-overflow containment failed",
    )
    interrupted = bounded_child(
        [sys.executable, "-c", "import time; time.sleep(2)"],
        helper_env,
        deadline_s=2,
        _interrupt_after=0.05,
    )
    require(
        interrupted["operator_interrupted"] and interrupted["cleanup_actions"],
        "operator interruption containment failed",
    )
    ownership = bounded_child(
        [sys.executable, "-c", "import time; time.sleep(2)"],
        helper_env,
        deadline_s=2,
        _pgid_getter=lambda _pid: -1,
    )
    require(
        ownership["ownership_error"]
        and ownership["cleanup_actions"]
        and ownership["reaped"],
        "ownership-loss containment failed",
    )
    expect(ContractDefect, require_sealable_child_outcome, ownership)

    def runner_failure(*_args, **_kwargs):
        raise RuntimeError("fixture runner failure")

    expect(
        RuntimeError,
        lambda: launched_child_outcome([], helper_env, None, None, runner_failure),
    )
    contradictory_group = {
        "pgid": 77,
        "samples": [{"captured_monotonic_ns": 1, "members": [77], "error": None}],
        "no_live_group_observed": True,
    }
    expect(
        ContractDefect,
        validate_group_disposition_observation,
        contradictory_group,
        77,
    )
    for samples in (
        [
            {"captured_monotonic_ns": 2, "members": [77], "error": None},
            {"captured_monotonic_ns": 1, "members": [], "error": None},
        ],
        [
            {
                "captured_monotonic_ns": 1,
                "members": [78, 77],
                "error": None,
            }
        ],
    ):
        expect(
            ContractDefect,
            validate_group_disposition_observation,
            {
                "pgid": 77,
                "samples": samples,
                "no_live_group_observed": samples[-1]["members"] == [],
            },
            77,
        )
    inspection_started = time.monotonic_ns()
    inspection = group_disposition(999_999, deadline_ns=inspection_started)
    require(
        time.monotonic_ns() - inspection_started < 100_000_000
        and inspection["samples"][0]["error"] == "inspection-deadline-exceeded",
        "failed process inspection exceeded shared bound",
    )
    expect(
        ContractDefect,
        bounded_json_command,
        [sys.executable, "-c", "import time; time.sleep(2)"],
        helper_env,
        "hung identity helper",
        0.05,
    )
    for prefix in COUNTER_ERROR_PREFIXES:
        require(
            structured_counter_defect((prefix + " fixture\n").encode()),
            f"structured counter category missed: {prefix}",
        )
    for envelope in RUST_ERROR_FIXTURES:
        require(
            structured_counter_defect(envelope),
            f"literal Rust error envelope missed: {envelope!r}",
        )
    require(
        not structured_counter_defect(b"Error: unrelated failure\n"),
        "arbitrary child failure became a counter defect",
    )
    require(
        classify_attempt(
            {"spawn_error": None, "returncode": -9},
            None,
            [],
            ["process_cleanup_invoked"],
        )[0]
        == "inconclusive",
        "successful forced cleanup was not inconclusive",
    )

    with tempfile.TemporaryDirectory(dir=ROOT / "target") as directory:
        root = Path(directory)
        packet, work = root / "packet", root / "work"
        packet.mkdir()
        work.mkdir()
        write_json(packet / "manifest.json", {"schema": 1, "fixture": True})
        write_json(packet / "headroom-probe.json", {"schema": 1, "fixture": True})
        for index in range(4):
            write_json(
                packet / f"p{index}.attempt.json", {"schema": 1, "position": index + 1}
            )
        decision = {"schema": 1, "status": "GO", "completed_attempts": 4}
        expect(ContractDefect, publish, decision, packet, work)
    with tempfile.TemporaryDirectory(dir=ROOT / "target") as directory:
        packet, work, decision = make_complete_packet_fixture(Path(directory))
        completion = publish(decision, packet, work)
        require(
            completion["publication"] == "exclusive-fsynced-complete",
            "complete synthetic seal failed",
        )
        inventory = read_json_file(packet / "artifact-inventory.json", "inventory")
        require(
            all(
                not row["path"].endswith("artifact-inventory.json")
                for row in inventory["logical_records"]
            ),
            "inventory self-included",
        )
    with tempfile.TemporaryDirectory(dir=ROOT / "target") as directory:
        packet, work, decision = make_complete_packet_fixture(
            Path(directory), saving_us=1_499_999
        )
        require(decision["status"] == "KILL", "synthetic KILL did not miss gate")
        completion = publish(decision, packet, work)
        require(
            completion["publication"] == "exclusive-fsynced-complete",
            "complete synthetic KILL seal failed",
        )
    for classification in ("implementation_or_contract_defect", "inconclusive"):
        with tempfile.TemporaryDirectory(dir=ROOT / "target") as directory:
            packet, work, decision = make_terminal_packet_fixture(
                Path(directory), classification
            )
            completion = publish(decision, packet, work)
            require(
                completion["publication"] == "exclusive-fsynced-complete",
                "terminal synthetic seal failed",
            )
    with tempfile.TemporaryDirectory(dir=ROOT / "target") as directory:
        packet, work, decision = make_spawn_failure_packet_fixture(Path(directory))
        completion = publish(decision, packet, work)
        require(
            completion["publication"] == "exclusive-fsynced-complete",
            "spawn-failure terminal seal failed",
        )
    mutation_cases = (
        "pid",
        "timestamp",
        "hash",
        "orphan",
        "terminal",
        "cooldown",
        "host",
        "source",
        "model_stamp",
        "impossible_timeline",
        "completion_field",
        "launch_schema",
        "acquired_schema",
        "acquired_stem",
        "environment_binding",
        "residency_arithmetic",
        "pid_type",
        "ownership_value",
        "flag_type",
        "group_member_order",
        "group_time_order",
        "group_disposition_difference",
        "cleanup_field",
        "group_field",
        "analysis",
        "authority",
        "decision_schema",
        "cutoff_schema",
        "probe_timeline",
        "post_stop",
        "signal_time",
        "unbound_signal",
        "pending_signal",
        "identity",
    )
    for mutation in mutation_cases:
        with tempfile.TemporaryDirectory(dir=ROOT / "target") as directory:
            packet, work, decision = make_complete_packet_fixture(Path(directory))
            if mutation == "pid":
                rows = read_jsonl_file(packet / "lifecycle.jsonl", "lifecycle")
                rows[1]["pid"] += 1
                (packet / "lifecycle.jsonl").unlink()
                for row in rows:
                    append_jsonl(packet / "lifecycle.jsonl", row)
            elif mutation == "timestamp":
                rows = read_jsonl_file(packet / "lifecycle.jsonl", "lifecycle")
                rows[0]["monotonic_ns"] = rows[2]["completed_monotonic_ns"] + 1
                (packet / "lifecycle.jsonl").unlink()
                for row in rows:
                    append_jsonl(packet / "lifecycle.jsonl", row)
            elif mutation == "hash":
                attempt = read_jsonl_file(packet / "attempts.jsonl", "attempts")[0]
                attempt["artifacts"]["stdout"]["sha256"] = "f" * 64
                rows = read_jsonl_file(packet / "attempts.jsonl", "attempts")
                rows[0] = attempt
                (packet / "attempts.jsonl").unlink()
                for row in rows:
                    append_jsonl(packet / "attempts.jsonl", row)
            elif mutation == "orphan":
                append_jsonl(packet / "lifecycle.jsonl", {"event": "orphan"})
            elif mutation == "terminal":
                rows = read_jsonl_file(packet / "attempts.jsonl", "attempts")[:-1]
                (packet / "attempts.jsonl").unlink()
                for row in rows:
                    append_jsonl(packet / "attempts.jsonl", row)
            elif mutation in {"cooldown", "host", "source", "impossible_timeline"}:
                path = work / "p1-a.conditioning.json"
                value = read_json_file(path, "conditioning mutation")
                if mutation == "cooldown":
                    value["cooldown"]["observed_interval_ns"] -= 1
                elif mutation == "host":
                    value["host_before_conditioning"]["battery"] = "Battery Power\n"
                    value["host_before_conditioning"]["valid"] = False
                elif mutation == "source":
                    value["source_records"][0]["sha256"] = "f" * 64
                else:
                    value["pre_attempt_identity"]["captured_monotonic_ns"] = 1
                path.unlink()
                write_json(path, value)
            elif mutation == "model_stamp":
                path = packet / "manifest.json"
                value = read_json_file(path, "manifest mutation")
                value["model_path_stamps"][1]["inode"] = value["model_path_stamps"][0][
                    "inode"
                ]
                path.unlink()
                write_json(path, value)
            elif mutation in {
                "completion_field",
                "launch_schema",
                "acquired_schema",
                "acquired_stem",
                "environment_binding",
                "group_field",
            }:
                rows = read_jsonl_file(packet / "lifecycle.jsonl", "lifecycle")
                if mutation == "completion_field":
                    rows[2]["timed_out"] = True
                elif mutation == "launch_schema":
                    rows[0]["schema"] = 2
                elif mutation == "acquired_schema":
                    rows[1]["schema"] = 2
                elif mutation == "acquired_stem":
                    rows[1]["stem"] = "forged"
                elif mutation == "environment_binding":
                    rows[0]["environment_sha256"] = "f" * 64
                else:
                    rows[2]["group_disposition"]["samples"][-1]["members"] = [999]
                (packet / "lifecycle.jsonl").unlink()
                for row in rows:
                    append_jsonl(packet / "lifecycle.jsonl", row)
            elif mutation in {"residency_arithmetic", "cleanup_field"}:
                rows = read_jsonl_file(packet / "attempts.jsonl", "attempts")
                if mutation == "residency_arithmetic":
                    rows[0]["residency_to_acquired_ns"] += 1
                else:
                    rows[0]["cleanup_actions"] = [{"forged": True}]
                (packet / "attempts.jsonl").unlink()
                for row in rows:
                    append_jsonl(packet / "attempts.jsonl", row)
            elif mutation in {"pid_type", "ownership_value", "flag_type"}:

                def mutate_attempt(attempt):
                    if mutation == "pid_type":
                        attempt["pid"] = str(attempt["pid"])
                    elif mutation == "ownership_value":
                        attempt["ownership_error"] = ""
                    else:
                        attempt["timed_out"] = 0

                rewrite_fixture_attempt(packet, 0, mutate_attempt)
                if mutation == "ownership_value":
                    rows = read_jsonl_file(packet / "lifecycle.jsonl", "lifecycle")
                    rows[2]["ownership_error"] = ""
                    (packet / "lifecycle.jsonl").unlink()
                    for row in rows:
                        append_jsonl(packet / "lifecycle.jsonl", row)
            elif mutation in {
                "group_member_order",
                "group_time_order",
                "group_disposition_difference",
            }:
                rows = read_jsonl_file(packet / "lifecycle.jsonl", "lifecycle")
                pid = rows[2]["pid"]
                if mutation == "group_member_order":
                    samples = [
                        {
                            "captured_monotonic_ns": rows[2]["attempted_monotonic_ns"],
                            "members": [pid + 1, pid],
                            "error": None,
                        },
                        {
                            "captured_monotonic_ns": rows[2]["completed_monotonic_ns"],
                            "members": [],
                            "error": None,
                        },
                    ]
                elif mutation == "group_time_order":
                    samples = [
                        {
                            "captured_monotonic_ns": rows[2]["completed_monotonic_ns"],
                            "members": [pid],
                            "error": None,
                        },
                        {
                            "captured_monotonic_ns": rows[2]["attempted_monotonic_ns"],
                            "members": [],
                            "error": None,
                        },
                    ]
                else:
                    final_group = rows[2]["group_disposition"]
                    initial_group = {
                        "pgid": pid,
                        "samples": [
                            {
                                "captured_monotonic_ns": rows[2][
                                    "attempted_monotonic_ns"
                                ],
                                "members": [pid],
                                "error": None,
                            },
                            *final_group["samples"],
                        ],
                        "no_live_group_observed": True,
                    }
                group = {
                    "pgid": pid,
                    "samples": samples
                    if mutation != "group_disposition_difference"
                    else final_group["samples"],
                    "no_live_group_observed": True,
                }

                def mutate_group(attempt):
                    attempt["group_disposition"] = group
                    attempt["group_disposition_initial"] = (
                        initial_group
                        if mutation == "group_disposition_difference"
                        else dict(group)
                    )

                rewrite_fixture_attempt(packet, 0, mutate_group)
                rows[2]["group_disposition"] = group
                rows[2]["group_disposition_initial"] = (
                    initial_group
                    if mutation == "group_disposition_difference"
                    else dict(group)
                )
                (packet / "lifecycle.jsonl").unlink()
                for row in rows:
                    append_jsonl(packet / "lifecycle.jsonl", row)
            elif mutation == "analysis":
                decision["analysis"]["qualifies"] = False
            elif mutation == "authority":
                decision["authority"] = "forged"
            elif mutation == "decision_schema":
                decision["schema"] = 2
            elif mutation == "cutoff_schema":
                path = packet / "signal-cutoff.json"
                value = read_json_file(path, "cutoff schema mutation")
                value["schema"] = 2
                path.unlink()
                write_json(path, value)
                decision["signal_cutoff_sha256"] = sha_file(path)
            elif mutation == "probe_timeline":
                path = packet / "headroom-probe.json"
                value = read_json_file(path, "probe timeline mutation")
                value["process"]["group_disposition"]["samples"][0][
                    "captured_monotonic_ns"
                ] = value["process"]["completed_monotonic_ns"] + 1
                path.unlink()
                write_json(path, value)
                decision["headroom_probe_sha256"] = sha_file(path)
            elif mutation == "post_stop":
                rows = read_jsonl_file(packet / "attempts.jsonl", "attempts")
                rows[0]["stop_classification"] = "inconclusive"
                rows[0]["stop_reason"] = "forged"
                (packet / "attempts.jsonl").unlink()
                for row in rows:
                    append_jsonl(packet / "attempts.jsonl", row)
            elif mutation == "signal_time":
                rows = read_jsonl_file(packet / "lifecycle.jsonl", "lifecycle")
                event = {
                    "sequence": 1,
                    "signal": int(signal.SIGINT),
                    "monotonic_ns": rows[2]["attempted_monotonic_ns"] - 1,
                }

                def mutate_signal(attempt):
                    attempt["interrupted"] = True
                    attempt["operator_interrupted"] = True
                    attempt["operator_signals"] = [event]
                    attempt["signal_start_sequence"] = 0
                    attempt["signal_end_sequence"] = 1
                    attempt["signal_events"] = [event]

                rewrite_fixture_attempt(packet, 0, mutate_signal)
                rows[2]["interrupted"] = True
                rows[2]["operator_interrupted"] = True
                rows[2]["operator_signals"] = [event]
                (packet / "lifecycle.jsonl").unlink()
                for row in rows:
                    append_jsonl(packet / "lifecycle.jsonl", row)
                log_path = packet / "signal-log.json"
                cutoff_path = packet / "signal-cutoff.json"
                log = read_json_file(log_path, "signal-time mutation")
                log["events"] = [event]
                log["event_count"] = 1
                log["authority_signal_count"] = 1
                log_path.unlink()
                write_json(log_path, log)
                cutoff = read_json_file(cutoff_path, "signal-time cutoff")
                cutoff["signal_log_sha256"] = sha_file(log_path)
                cutoff["authority_signal_count"] = 1
                cutoff_path.unlink()
                write_json(cutoff_path, cutoff)
                decision["signal_log_sha256"] = sha_file(log_path)
                decision["signal_cutoff_sha256"] = sha_file(cutoff_path)
            elif mutation in {"unbound_signal", "pending_signal"}:
                log_path = packet / "signal-log.json"
                cutoff_path = packet / "signal-cutoff.json"
                log = read_json_file(log_path, "signal mutation")
                if mutation == "unbound_signal":
                    log["events"] = [
                        {
                            "sequence": 1,
                            "signal": int(signal.SIGINT),
                            "monotonic_ns": log["boundary_monotonic_ns"],
                        }
                    ]
                    log["event_count"] = 1
                else:
                    log["pending_signals"] = [int(signal.SIGTERM)]
                log["authority_signal_count"] = 1
                log_path.unlink()
                write_json(log_path, log)
                cutoff = read_json_file(cutoff_path, "cutoff mutation")
                cutoff["signal_log_sha256"] = sha_file(log_path)
                cutoff["authority_signal_count"] = 1
                cutoff_path.unlink()
                write_json(cutoff_path, cutoff)
                decision["signal_log_sha256"] = sha_file(log_path)
                decision["signal_cutoff_sha256"] = sha_file(cutoff_path)
            else:
                decision["final_identity"]["identity"]["source_commit"] = "b" * 40
            expect(ContractDefect, publish, decision, packet, work)
    with tempfile.TemporaryDirectory(dir=ROOT / "target") as directory:
        packet, work, decision = make_complete_packet_fixture(Path(directory))

        def mutate_late():
            with (work / "p1-a.stdout").open("ab") as output:
                output.write(b"late")

        expect(ContractDefect, publish, decision, packet, work, mutate_late)
        require(
            not (packet / "packet-complete.json").exists(),
            "failed semantic publication left an authority marker",
        )
    with tempfile.TemporaryDirectory(dir=ROOT / "target") as directory:
        packet, work, decision = make_complete_packet_fixture(Path(directory))

        def replace_decision():
            path = packet / "decision.json"
            path.unlink()
            write_json(path, {"forged": True})

        expect(
            ContractDefect,
            lambda: publish(decision, packet, work, _after_decision=replace_decision),
        )
        require(
            not (packet / "packet-complete.json").exists(),
            "failed decision publication left an authority marker",
        )
    with tempfile.TemporaryDirectory(dir=ROOT / "target") as directory:
        packet, work, decision = make_complete_packet_fixture(Path(directory))

        def mutate_inventory():
            with (packet / "artifact-inventory.json").open("ab") as output:
                output.write(b"forged")

        expect(
            ContractDefect,
            lambda: publish(decision, packet, work, _after_inventory=mutate_inventory),
        )
        require(
            not (packet / "packet-complete.json").exists(),
            "failed inventory publication left an authority marker",
        )
    for hook_name in ("after-semantic", "before-final-snapshot"):
        with tempfile.TemporaryDirectory(dir=ROOT / "target") as directory:
            packet, work, decision = make_complete_packet_fixture(Path(directory))

            def mutate_hook():
                with (work / "p1-a.stderr").open("ab") as output:
                    output.write(b"late")

            kwargs = (
                {"_after_semantic": mutate_hook}
                if hook_name == "after-semantic"
                else {"_before_final_snapshot": mutate_hook}
            )
            expect(
                ContractDefect,
                lambda: publish(decision, packet, work, **kwargs),
            )
            require(
                not (packet / "packet-complete.json").exists(),
                f"{hook_name} failure left an authority marker",
            )
    print("self-test: ok")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        run_self_test()
    else:
        execute()


if __name__ == "__main__":
    main()
