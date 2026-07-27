#!/usr/bin/env python3
"""Independent v0.643 transient mmap-blit mechanism-floor preregistration."""

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
PACKET = ROOT / "target/profiles/v0643-a3b-transient-mmap-blit-floor-p1"
WORK = ROOT / "target/profiles/v0643-a3b-transient-mmap-blit-floor-work"
PREREG = ROOT / "docs/bench/v0643-a3b-transient-mmap-blit-floor.md"
RUNNER = Path(__file__).resolve()
BINARY = ROOT / "target/release/qwen-bench"
MODEL = Path("/Users/tito/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")
IMPLEMENTATION_PARENT = "d859345118774b934218a7eb55460ef0870da882"
MODEL_SIZE = 22_134_528_992
MODEL_PAGES = 1_350_985
PAGE_SIZE = 16_384
MODEL_SHA256 = "ac0e2c1189e055faa36eff361580e79c5bd6f8e76bffb4ce547f167d53e31a61"
DESCRIPTOR = "0x5ae645df5cf7d568"
INVENTORY = "f57153febec22463c7789b892d4d084041d722483a93191c81c40ab86be7d9e5"
PROFILE = "a3b-q4km-v1"
COUNT = 733
COPY_BYTES = 22_123_538_944
SCHEDULE = {
    "digest": "800bf469d09879187e07a832ab573a84f19714f6d3fce8d47fce87adc5808329",
    "cuts": [155, 359, 539],
    "task_counts": [155, 204, 180, 194],
    "worker_bytes": [5_532_746_240, 5_462_315_776, 5_595_522_304, 5_532_954_624],
}
BLIT_PLAN = "fa2685e223ad8ea6271c6061041fe8d996b4e6cc70e060588b750732577c92af"
PAIR_ORDERS = (("A", "B"), ("B", "A"), ("B", "A"), ("A", "B"), ("A", "B"), ("B", "A"))
RESOURCE_MODES = {
    "creation_storage": "shared",
    "creation_cpu_cache": "default_cache",
    "creation_hazard_tracking": "default",
    "observed_storage": "shared",
    "observed_cpu_cache": "default_cache",
    "observed_hazard_tracking": "tracked",
}
TIME_LABELS = (
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
SAFE_ENV = (
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
COOLDOWN_NS = 30_000_000_000
LAUNCH_LIMIT_NS = 5_000_000_000
MAX_OUTPUT = 4 * 1024 * 1024
BUFFER_SIZE = 8 * 1024 * 1024
EXPECTED_INVENTORY_MEMBERS = 58
RUSAGE_PREFIX = "Error: getrusage counter regressed:"
OPERATOR_SIGNALS = (signal.SIGINT, signal.SIGTERM)
ARCHITECTURE_TUPLE = {
    "kind": "moe",
    "n_layer": 40,
    "hidden_size": 2048,
    "intermediate_size": 0,
    "vocab_size": 248_320,
    "full_attention_interval": 4,
    "n_q_heads": 16,
    "n_kv_heads": 2,
    "attn_head_dim": 256,
    "rope_theta": 10_000_000.0,
    "partial_rotary_factor": 0.25,
    "gdn_n_v_heads": 32,
    "gdn_n_k_heads": 16,
    "gdn_head_dim": 128,
    "gdn_conv_kernel": 4,
    "expert_count": 256,
    "expert_used_count": 8,
    "expert_feed_forward_length": 512,
    "expert_shared_feed_forward_length": 512,
    "mtp_n_hidden_layers": 0,
}
PROC_RUSAGE_KEYS = {
    "instructions_delta_raw",
    "cycles_delta_raw",
    "billed_energy_delta_raw",
    "serviced_energy_delta_raw",
}
DECISION_KEYS = {
    "schema",
    "status",
    "authority",
    "force_authorized",
    "successor_authorization",
    "implementation_parent",
    "source_commit",
    "final_identity",
    "completed_attempts",
    "expected_attempts",
    "contract_error",
    "invalid_error",
    "rusage_self_counter_regression",
    "analysis",
    "closure",
    "scope",
    "signal_cutoff_sha256",
    "signal_log_sha256",
}
ATTEMPT_KEYS = {
    "schema",
    "stem",
    "pair",
    "position",
    "order",
    "arm",
    "command",
    "pid",
    "pgid",
    "spawn_error",
    "returncode",
    "reaped",
    "output_overflow",
    "interrupted",
    "wait_errors",
    "poll_errors",
    "pipe_errors",
    "termination_errors",
    "cleanup_actions",
    "cleanup_signal_sent",
    "group_disposition",
    "signal_start_sequence",
    "residency_proved_ns",
    "launch_attempted_ns",
    "launch_acquired_ns",
    "residency_to_acquired_ns",
    "process_resources",
    "process_page_faults",
    "rusage_self_counter_regression",
    "rusage_regression_stderr_prefix",
    "parse_error",
    "validity_reasons",
    "stop_classification",
    "stop_reason",
    "result",
    "stdout_sha256",
    "stderr_sha256",
    "conditioning_sha256",
    "post_sha256",
    "activity_boundary_monotonic_ns",
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
}
A_TIMING_KEYS = {
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


class ContractDefect(RuntimeError):
    pass


class Inconclusive(RuntimeError):
    pass


class DurabilityError(RuntimeError):
    pass


class OwnershipLost(BaseException):
    pass


def require(value, message):
    if not value:
        raise ContractDefect(message)


def require_typed_equal(actual, expected, label):
    require(type(actual) is type(expected), f"{label} JSON type drifted")
    if isinstance(expected, dict):
        require(set(actual) == set(expected), f"{label} key set drifted")
        for key in expected:
            require_typed_equal(actual[key], expected[key], f"{label}.{key}")
    elif isinstance(expected, list):
        require(len(actual) == len(expected), f"{label} length drifted")
        for index, (observed, wanted) in enumerate(zip(actual, expected)):
            require_typed_equal(observed, wanted, f"{label}[{index}]")
    else:
        require(actual == expected, f"{label} value drifted")


def typed_equal(actual, expected):
    try:
        require_typed_equal(actual, expected, "typed equality")
    except ContractDefect:
        return False
    return True


def lexists(path):
    return os.path.lexists(path)


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_bytes(value, pretty=False):
    options = {"sort_keys": True, "ensure_ascii": True}
    options.update(indent=2) if pretty else options.update(separators=(",", ":"))
    return (json.dumps(value, **options) + "\n").encode("ascii")


def parse_json_bytes(value, label):
    def pairs(items):
        result = {}
        for key, item in items:
            if key in result:
                raise ValueError(f"duplicate object key {key!r}")
            result[key] = item
        return result

    def constant(value):
        raise ValueError(f"non-finite JSON constant {value}")

    def floating(value):
        parsed = float(value)
        if not math.isfinite(parsed):
            raise ValueError(f"non-finite JSON float {value}")
        return parsed

    try:
        return json.loads(
            value.decode("utf-8"),
            object_pairs_hook=pairs,
            parse_constant=constant,
            parse_float=floating,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ContractDefect(f"{label} is malformed JSON: {error}") from error


def write_exclusive(path, value):
    try:
        with path.open("xb") as output:
            output.write(value)
            output.flush()
            os.fsync(output.fileno())
    except OSError as error:
        raise DurabilityError(f"durable write failed for {path}: {error}") from error


def write_json(path, value):
    write_exclusive(path, json_bytes(value, True))


def append_jsonl(path, value):
    try:
        with path.open("ab") as output:
            output.write(json_bytes(value))
            output.flush()
            os.fsync(output.fileno())
    except OSError as error:
        raise DurabilityError(f"durable append failed for {path}: {error}") from error


def fsync_dir(path):
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def command_output(command, env=None):
    result = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    require(
        result.returncode == 0,
        f"command failed: {command!r}: {result.stderr[-4096:]!r}",
    )
    require(
        not result.stderr and len(result.stdout) <= MAX_OUTPUT,
        f"command output drifted: {command!r}",
    )
    return result.stdout.decode("utf-8")


def git_output(args):
    return command_output(["git", *args]).strip()


def normalized_environment():
    env = {key: os.environ[key] for key in SAFE_ENV if key in os.environ}
    for key in ("HOME", "PATH", "TMPDIR"):
        require(key in env, f"missing environment {key}")
    removed = sorted(
        key
        for key in os.environ
        if key not in SAFE_ENV
        or key.upper().startswith(("QWEN", "MTL", "METAL"))
        or key.upper() == "RUST_LOG"
        or "QOS" in key.upper()
        or "CHUNK" in key.upper()
    )
    env["RUST_BACKTRACE"] = "0"
    return env, removed


def complete_stamp(value):
    return {
        "device": value.st_dev,
        "inode": value.st_ino,
        "mode": value.st_mode,
        "nlink": value.st_nlink,
        "uid": value.st_uid,
        "gid": value.st_gid,
        "rdev": value.st_rdev,
        "size_bytes": value.st_size,
        "block_size": getattr(value, "st_blksize", 0),
        "blocks": getattr(value, "st_blocks", 0),
        "mtime_ns": value.st_mtime_ns,
        "ctime_ns": value.st_ctime_ns,
        "birthtime_ns": round(getattr(value, "st_birthtime", 0.0) * 1e9),
        "flags": getattr(value, "st_flags", 0),
    }


def descriptor_hash(path, expected_size=None):
    before_l, before_s = os.lstat(path), os.stat(path)
    require(
        stat.S_ISREG(before_l.st_mode)
        and before_l.st_nlink == 1
        and not before_l.st_mode & 0o022,
        f"unsafe descriptor {path}",
    )
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        opened = os.fstat(descriptor)
        stamp_value = complete_stamp(opened)
        require(
            (before_l.st_dev, before_l.st_ino)
            == (opened.st_dev, opened.st_ino)
            == (before_s.st_dev, before_s.st_ino),
            "descriptor/path identity drifted",
        )
        size = opened.st_size if expected_size is None else expected_size
        require(opened.st_size == size, "descriptor size drifted")
        digest = hashlib.sha256()
        count = 0
        for chunk in iter(lambda: os.read(descriptor, 1024 * 1024), b""):
            digest.update(chunk)
            count += len(chunk)
        require(
            count == size and complete_stamp(os.fstat(descriptor)) == stamp_value,
            "descriptor changed while hashing",
        )
        after_l, after_s = os.lstat(path), os.stat(path)
        require(
            complete_stamp(after_l) == complete_stamp(before_l)
            and complete_stamp(after_s) == complete_stamp(before_s)
            and (after_l.st_dev, after_l.st_ino) == (opened.st_dev, opened.st_ino),
            "pathname changed while hashing",
        )
        return {
            "descriptor_stamp": stamp_value,
            "bytes_hashed": count,
            "sha256": digest.hexdigest(),
        }
    finally:
        os.close(descriptor)


def file_identity(path):
    value = path.stat()
    return {
        "device": value.st_dev,
        "inode": value.st_ino,
        "size_bytes": value.st_size,
        "mtime_ns": value.st_mtime_ns,
    }


def source_identity(env):
    require(Path.cwd().resolve() == ROOT, "runner must execute from repository root")
    require(
        not git_output(["status", "--porcelain=v1", "--untracked-files=all"]),
        "worktree not clean",
    )
    head = git_output(["rev-parse", "HEAD"])
    parents = git_output(["rev-list", "--parents", "-n", "1", "HEAD"]).split()
    require(
        parents == [head, IMPLEMENTATION_PARENT],
        "v0.643 is not clean direct implementation child",
    )
    expected = sorted((str(PREREG.relative_to(ROOT)), str(RUNNER.relative_to(ROOT))))
    require(
        sorted(git_output(["diff", "--name-status", "HEAD^..HEAD"]).splitlines())
        == sorted(f"A\t{path}" for path in expected),
        "v0.643 additions drifted",
    )
    require(
        git_output(["rev-parse", "HEAD:crates"])
        == git_output(["rev-parse", f"{IMPLEMENTATION_PARENT}:crates"]),
        "crates tree drifted",
    )
    build = parse_json_bytes(
        command_output([str(BINARY), "build-info", "--output", "json"], env).encode(),
        "build-info",
    )
    require(
        isinstance(build, dict)
        and build.get("build_commit") == head
        and build.get("runtime_commit") == head
        and build.get("status") == "match"
        and build.get("build_dirty") is False
        and build.get("runtime_dirty") is False
        and build.get("build_source_state") == build.get("runtime_source_state"),
        "source/build/runtime identity mismatch",
    )
    return {
        "head": head,
        "parent": IMPLEMENTATION_PARENT,
        "implementation_parent": IMPLEMENTATION_PARENT,
        "build_identity": build,
    }


def verify_binary_identity(manifest, env):
    expected = manifest["binary_bytes"]
    stamp_value = expected.get("descriptor_stamp")
    require(isinstance(stamp_value, dict), "manifest binary stamp is malformed")
    size = stamp_value.get("size_bytes")
    require(type(size) is int and size > 0, "manifest binary size is malformed")
    require(
        descriptor_hash(BINARY, size) == expected, "qwen-bench byte identity drifted"
    )
    build = parse_json_bytes(
        command_output([str(BINARY), "build-info", "--output", "json"], env).encode(),
        "qwen-bench identity recheck",
    )
    require(build == manifest["build_identity"], "qwen-bench semantic stamp drifted")


def process_group_rows(text, pgid):
    members = []
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        match = re.fullmatch(r"\s*(\d+)\s+(\d+)\s*", line)
        if match is None:
            raise RuntimeError(f"malformed ps row {number}: {line!r}")
        if int(match.group(2)) == pgid:
            members.append(int(match.group(1)))
    return sorted(members)


def group_members(pgid):
    result = subprocess.run(
        ["ps", "-axo", "pid=,pgid="],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode or result.stderr:
        raise RuntimeError("process-group inspection failed")
    return process_group_rows(result.stdout, pgid)


def disposition(pgid):
    samples = []
    for _ in range(50):
        try:
            members, error = group_members(pgid), None
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
        time.sleep(0.02)
    return {
        "pgid": pgid,
        "samples": samples,
        "no_live_group_observed": samples[-1]["members"] == [],
    }


class SignalController:
    def __init__(self, restore=False):
        self.restore_on_exit, self.events, self.previous, self.cutoff = (
            restore,
            [],
            {},
            False,
        )

    def install(self):
        def record(signum, _frame):
            self.events.append(
                {
                    "sequence": len(self.events) + 1,
                    "signal": signum,
                    "monotonic_ns": time.monotonic_ns(),
                }
            )

        for signum in OPERATOR_SIGNALS:
            self.previous[signum] = signal.getsignal(signum)
            signal.signal(signum, record)

    def restore(self):
        for signum, handler in self.previous.items():
            signal.signal(signum, handler)

    def __enter__(self):
        self.install()
        return self

    def __exit__(self, *_):
        if self.restore_on_exit:
            self.restore()

    def sequence(self):
        return len(self.events)

    def since(self, sequence):
        return [dict(event) for event in self.events if event["sequence"] > sequence]

    def quiet(self, sequence, stage):
        if self.since(sequence):
            raise Inconclusive(f"operator signal before {stage}")

    def final_cutoff(self, path):
        signal.pthread_sigmask(signal.SIG_BLOCK, OPERATOR_SIGNALS)
        boundary, sequence = time.monotonic_ns(), self.sequence()
        events = tuple(dict(event) for event in self.events)
        pending = sorted(
            int(value) for value in signal.sigpending() if value in OPERATOR_SIGNALS
        )
        log = {
            "schema": 1,
            "logical_boundary_monotonic_ns": boundary,
            "logical_boundary_sequence": sequence,
            "post_block_snapshot_monotonic_ns": time.monotonic_ns(),
            "post_block_snapshot_sequence": events[-1]["sequence"]
            if events
            else sequence,
            "events": list(events),
            "pending_signals": pending,
            "attribution": "all snapshot events and pending signals are pre-cutoff",
        }
        log_path = path.with_name("packet-signal-log.json")
        write_json(log_path, log)
        record = {
            "schema": 1,
            "event": "packet-signal-cutoff",
            "cutoff_monotonic_ns": boundary,
            **log,
            "authority_event_count": len(events) + len(pending),
            "signals_after_snapshot": "blocked-post-cutoff-outside-authority",
            "signal_log_sha256": sha_file(log_path),
        }
        write_json(path, record)
        self.cutoff = True
        return record


def reap_leader_without_signal(process):
    diagnostics = {
        "signal_sent": False,
        "communicate_errors": [],
        "leader_reaped": False,
        "returncode_observed": None,
    }
    try:
        process.communicate()
    except (OSError, subprocess.SubprocessError, ValueError) as error:
        diagnostics["communicate_errors"].append(f"{type(error).__name__}:{error}")
        while process.returncode is None:
            try:
                process.wait(timeout=0.05)
            except subprocess.TimeoutExpired:
                continue
    diagnostics["leader_reaped"] = process.returncode is not None
    diagnostics["returncode_observed"] = process.returncode
    return diagnostics


def close_pipe_fds_after_ownership_loss(process, streams, threads):
    diagnostics = {
        "pid": process.pid,
        "signal_sent": False,
        "fd_close_errors": [],
        "threads_alive_after_bounded_join": [],
        "returncode_observed": process.returncode,
    }
    for index, stream in enumerate(streams):
        if stream is None:
            continue
        try:
            os.close(stream.fileno())
        except (OSError, ValueError) as error:
            diagnostics["fd_close_errors"].append(
                f"fd_close{index}:{type(error).__name__}:{error}"
            )
    for index, thread in enumerate(threads):
        thread.join(timeout=0.2)
        if thread.is_alive():
            diagnostics["threads_alive_after_bounded_join"].append(index)
    diagnostics["returncode_observed"] = process.returncode
    return diagnostics


def bounded_child(
    command,
    env,
    controller=None,
    start=0,
    on_acquired=None,
    _faults=None,
    _pgid_getter=os.getpgid,
    _cleanup_observer=None,
    _ownership_loss_observer=None,
):
    faults = set() if _faults is None else set(_faults)

    def consume_fault(name):
        if name not in faults:
            return False
        faults.remove(name)
        return True

    attempted = time.monotonic_ns()
    try:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
    except (OSError, subprocess.SubprocessError, KeyboardInterrupt) as error:
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
            "interrupted": isinstance(error, KeyboardInterrupt),
            "errors": [],
            "wait_errors": [],
            "poll_errors": [],
            "pipe_errors": [],
            "termination_errors": [],
            "cleanup_actions": [],
            "cleanup_signal_sent": False,
            "group_disposition": None,
            "reaped": False,
        }
    acquired = time.monotonic_ns()
    try:
        pgid = _pgid_getter(process.pid)
    except OSError as error:
        diagnostics = reap_leader_without_signal(process)
        if _ownership_loss_observer is not None:
            _ownership_loss_observer(dict(diagnostics))
        error.add_note(f"ownership-lost no-signal cleanup: {diagnostics}")
        raise OwnershipLost(f"getpgid failed: {error}") from error
    if pgid != process.pid:
        error = OwnershipLost(f"pgid mismatch pid={process.pid} pgid={pgid}")
        diagnostics = reap_leader_without_signal(process)
        if _ownership_loss_observer is not None:
            _ownership_loss_observer(dict(diagnostics))
        error.add_note(f"ownership-lost no-signal cleanup: {diagnostics}")
        raise error
    if on_acquired:
        try:
            on_acquired(
                {"pid": process.pid, "pgid": pgid, "acquired_monotonic_ns": acquired}
            )
        except BaseException as error:
            try:
                os.killpg(pgid, signal.SIGKILL)
                process.communicate()
                observed = disposition(pgid)
                if _cleanup_observer is not None:
                    _cleanup_observer({"pgid": pgid, "disposition": observed})
                error.add_note(
                    f"owned acquisition cleanup: pgid={pgid} disposition={observed}"
                )
            except BaseException as cleanup_error:
                error.add_note(
                    f"owned acquisition cleanup failed: {type(cleanup_error).__name__}:{
                        cleanup_error
                    }"
                )
            raise
    buffers = [bytearray(), bytearray()]
    overflow = [False, False]
    errors = []

    def drain(stream, index):
        try:
            while True:
                if consume_fault(f"read_{index}"):
                    raise OSError("injected drain read failure")
                chunk = stream.read(65536)
                if not chunk:
                    break
                room = MAX_OUTPUT - len(buffers[index])
                buffers[index].extend(chunk[: max(room, 0)])
                if len(chunk) > room:
                    overflow[index] = True
        except Exception as error:
            errors.append(f"pipe{index}:{type(error).__name__}:{error}")
        finally:
            try:
                if consume_fault(f"close_{index}"):
                    raise OSError("injected close failure")
                stream.close()
            except Exception as error:
                errors.append(f"close{index}:{type(error).__name__}:{error}")

    threads = []
    for index, stream in enumerate((process.stdout, process.stderr)):
        thread = threading.Thread(target=drain, args=(stream, index), daemon=True)
        try:
            if consume_fault(f"thread_start_{index}"):
                raise RuntimeError("injected thread start failure")
            thread.start()
            threads.append(thread)
        except RuntimeError as error:
            errors.append(f"thread{index}:{type(error).__name__}:{error}")
            try:
                stream.close()
            except (OSError, ValueError) as close_error:
                errors.append(
                    f"thread_close{index}:{type(close_error).__name__}:{close_error}"
                )
    actions = []
    interrupted = False
    while True:
        if (
            errors or (controller and controller.since(start)) or interrupted
        ) and not actions:
            action = {
                "reason": "operator-signal-or-pipe-failure",
                "target_pgid": pgid,
                "signal": int(signal.SIGKILL),
                "attempted_monotonic_ns": time.monotonic_ns(),
                "succeeded": False,
            }
            try:
                os.killpg(pgid, signal.SIGKILL)
                action["succeeded"] = True
                action["error"] = None
            except OSError as error:
                action["error"] = f"{type(error).__name__}:{error}"
            action["completed_monotonic_ns"] = time.monotonic_ns()
            actions.append(action)
        try:
            if consume_fault("wait_once"):
                raise OSError("injected wait failure")
            if consume_fault("ownership_lost"):
                raise ChildProcessError("injected exact-wait ownership loss")
            process.wait(timeout=0.05)
            break
        except subprocess.TimeoutExpired:
            pass
        except KeyboardInterrupt:
            interrupted = True
        except ChildProcessError as error:
            ownership_error = OwnershipLost("exact wait ownership lost")
            diagnostics = close_pipe_fds_after_ownership_loss(
                process,
                (process.stdout, process.stderr),
                threads,
            )
            if _ownership_loss_observer is not None:
                _ownership_loss_observer(dict(diagnostics))
            ownership_error.add_note(f"ownership-lost no-signal cleanup: {diagnostics}")
            raise ownership_error from error
        except Exception as error:
            errors.append(f"wait:{type(error).__name__}:{error}")
    for index, thread in enumerate(threads):
        try:
            if consume_fault(f"join_{index}"):
                raise RuntimeError("injected join failure")
            thread.join()
        except RuntimeError as error:
            errors.append(f"join{index}:{type(error).__name__}:{error}")
            thread.join()
    completed = time.monotonic_ns()
    wait_errors = [item for item in errors if item.startswith("wait:")]
    pipe_errors = [item for item in errors if not item.startswith("wait:")]
    termination_errors = [action["error"] for action in actions if action.get("error")]
    return {
        "pid": process.pid,
        "pgid": pgid,
        "attempted_monotonic_ns": attempted,
        "acquired_monotonic_ns": acquired,
        "completed_monotonic_ns": completed,
        "spawn_error": None,
        "returncode": process.returncode,
        "stdout": bytes(buffers[0]),
        "stderr": bytes(buffers[1]),
        "output_overflow": any(overflow),
        "interrupted": interrupted,
        "errors": errors,
        "wait_errors": wait_errors,
        "poll_errors": [],
        "pipe_errors": pipe_errors,
        "termination_errors": termination_errors,
        "cleanup_actions": actions,
        "cleanup_signal_sent": bool(actions),
        "group_disposition": disposition(pgid),
        "reaped": process.returncode is not None,
    }


def finite(value, label, positive=False):
    require(
        type(value) in (int, float) and not isinstance(value, bool),
        f"{label} not numeric",
    )
    parsed = float(value)
    require(
        math.isfinite(parsed) and (parsed > 0 if positive else parsed >= 0),
        f"{label} invalid",
    )
    return parsed


def uint(value, label, expected=None, positive=False):
    require(
        type(value) is int and value >= (1 if positive else 0), f"{label} not unsigned"
    )
    if expected is not None:
        require(value == expected, f"{label} drifted")
    return value


def wall_pair(timing, ms, us, positive=False):
    observed = finite(timing.get(ms), ms, positive)
    microseconds = uint(timing.get(us), us, positive=positive)
    require(abs(observed - microseconds / 1000) <= 0.002, f"{ms}/{us} disagree")


def validate_schedule(value):
    require(isinstance(value, dict), "W4 schedule is not an object")
    require(value.get("algorithm") == "minimax-contiguous-v1", "W4 algorithm drifted")
    uint(value.get("workers"), "W4 workers", 4)
    for key in ("cuts", "task_counts", "worker_bytes"):
        require_typed_equal(value.get(key), SCHEDULE[key], f"W4 {key}")
    partitions = value.get("partitions")
    require(isinstance(partitions, list) and len(partitions) == 4, "partitions drifted")
    cursor = 0
    for index, part in enumerate(partitions):
        require(
            isinstance(part, dict) and part.get("start") == cursor,
            "partition start drifted",
        )
        cursor += SCHEDULE["task_counts"][index]
        require(
            part.get("end") == cursor
            and part.get("task_count") == SCHEDULE["task_counts"][index]
            and part.get("bytes") == SCHEDULE["worker_bytes"][index],
            "partition accounting drifted",
        )
        for endpoint in ("first", "last"):
            require(isinstance(part.get(endpoint), dict), "partition endpoint missing")
    require(
        cursor == COUNT
        and sha_bytes(
            json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii")
        )
        == SCHEDULE["digest"],
        "schedule digest drifted",
    )


def validate_blit(value):
    require(
        isinstance(value, dict)
        and set(value)
        == {"schema_version", "order", "sources", "copies", "command", "release"},
        "blit schema keys drifted",
    )
    uint(value["schema_version"], "blit schema version", 1)
    require_typed_equal(
        value["order"],
        {
            "algorithm": "shard-offset-request-v1",
            "count": 733,
            "first_request_index": 2,
            "last_request_index": 721,
        },
        "blit order",
    )
    require_typed_equal(
        value["sources"],
        {
            "window_count": 1,
            "window_bytes": 22_123_544_576,
            "window_gap_bytes": 13_824,
            "fallback_count": 1,
            "fallback_bytes": 8_192,
            "cpu_staging_copy_bytes": 8_192,
        },
        "blit sources",
    )
    require_typed_equal(
        value["copies"],
        {
            "window_count": 732,
            "window_bytes": 22_123_530_752,
            "total_count": 733,
            "total_bytes": COPY_BYTES,
        },
        "blit copies",
    )
    command = value["command"]
    require(
        isinstance(command, dict)
        and set(command)
        == {
            "buffer_count",
            "encoder_count",
            "commit_count",
            "wait_count",
            "error_count",
            "status",
            "status_code",
            "retained_references",
            "gpu_start_time",
            "gpu_end_time",
            "gpu_wall_ms",
        },
        "blit command keys drifted",
    )
    for key in ("buffer_count", "encoder_count", "commit_count", "wait_count"):
        uint(command[key], f"blit command {key}", 1)
    uint(command["error_count"], "blit command error count", 0)
    uint(command["status_code"], "blit command status code", 4)
    require(
        command["status"] == "completed" and command["retained_references"] is True,
        "blit command drifted",
    )
    triple = [command[key] for key in ("gpu_start_time", "gpu_end_time", "gpu_wall_ms")]
    if triple != [None, None, None]:
        start, end, wall = (finite(item, "GPU timestamp", True) for item in triple)
        require(
            end > start and abs(wall - (end - start) * 1000) <= 0.002,
            "GPU timestamps inconsistent",
        )
    release = value["release"]
    require(
        isinstance(release, dict)
        and set(release)
        == {
            "window_deallocator_calls",
            "window_deallocator_mismatches",
            "source_buffers_alive",
            "allocated_after_destinations",
            "allocated_with_sources",
            "allocated_after_source_release",
        },
        "blit release keys drifted",
    )
    uint(release["window_deallocator_calls"], "window deallocator calls", 1)
    uint(release["window_deallocator_mismatches"], "window deallocator mismatches", 0)
    uint(release["source_buffers_alive"], "source buffers alive", 0)
    for key in (
        "allocated_after_destinations",
        "allocated_with_sources",
        "allocated_after_source_release",
    ):
        uint(release[key], key)


def validate_result(value, arm, build):
    require(isinstance(value, dict), "child JSON not object")
    expected_keys = TOP_KEYS | ({"blit_population"} if arm == "B" else set())
    require(set(value) == expected_keys, "top-level key set drifted")
    label = "parallel-pread" if arm == "A" else "transient-mmap-blit"
    uint(value.get("schema_version"), "result schema version", 2)
    require(
        value.get("profile") == PROFILE
        and value.get("arm") == label
        and value.get("model") == str(MODEL)
        and value.get("descriptor_layout_digest") == DESCRIPTOR
        and value.get("inventory_digest") == INVENTORY,
        "common identity drifted",
    )
    require_typed_equal(value.get("build_identity"), build, "build identity")
    for key in ("request_count", "resource_count", "binding_count"):
        uint(value.get(key), key, COUNT)
    for key in ("logical_copy_bytes", "physical_copy_bytes"):
        uint(value.get(key), key, COPY_BYTES)
    require_typed_equal(value.get("resource_modes"), RESOURCE_MODES, "resource modes")
    require_typed_equal(
        value.get("correctness"),
        {
            "passed": True,
            "payload_bytes_checked": COPY_BYTES,
            "entries_checked": COUNT,
        },
        "correctness",
    )
    require(value.get("architecture") == "qwen35moe", "architecture drifted")
    require_typed_equal(
        value.get("architecture_tuple"), ARCHITECTURE_TUPLE, "architecture tuple"
    )
    require_typed_equal(
        value.get("shard_mapped_lengths"), [MODEL_SIZE], "shard mapped lengths"
    )
    require(
        value.get("tied_embeddings") is False
        and value.get("mtp_present") is False
        and value.get("native_quant_embedding") is True
        and value.get("native_quant_embedding_supported") is True
        and value.get("native_quant_embedding_selection") == "production-auto-promoted"
        and value.get("device_name") == "Apple M4 Max"
        and value.get("unified_memory") is True,
        "common architecture/device contract drifted",
    )
    uint(value.get("page_size"), "page size", PAGE_SIZE)
    uint(value.get("required_alignment"), "required alignment", 32)
    uint(value.get("max_buffer_length"), "maximum buffer length", 77_309_411_328)
    timing = value.get("timing")
    require(isinstance(timing, dict), "timing missing")
    expected_timing = A_TIMING_KEYS | (
        {"source_release_wall_ms", "source_release_us"} if arm == "B" else set()
    )
    require(set(timing) == expected_timing, "arm timing keys drifted")
    phases = (
        ["allocation", "source_resolution", "copy"]
        + (["source_release"] if arm == "B" else [])
        + ["binding"]
    )
    for phase in phases:
        wall_pair(
            timing,
            f"{phase}_wall_ms",
            "source_us" if phase == "source_resolution" else f"{phase}_us",
            True,
        )
    require(
        timing["source_resolution_us"] == timing["source_us"], "source alias drifted"
    )
    wall_pair(timing, "ready_wall_ms", "ready_us", True)
    wall_pair(timing, "unattributed_wall_ms", "unattributed_us")
    wall_pair(timing, "teardown_wall_ms", "teardown_us", True)
    named_phase_sum = sum(
        timing["source_us" if phase == "source_resolution" else phase + "_us"]
        for phase in phases
    )
    require(
        0 <= timing["unattributed_us"] <= 5
        and abs(timing["ready_us"] - named_phase_sum - timing["unattributed_us"]) <= 5,
        "phase reconciliation drifted",
    )
    throughput = value.get("throughput")
    require(
        isinstance(throughput, dict)
        and set(throughput) == {"ready_gbps_decimal", "copy_gbps_decimal"},
        "throughput schema drifted",
    )
    finite(throughput.get("ready_gbps_decimal"), "ready throughput", True)
    finite(throughput.get("copy_gbps_decimal"), "copy throughput", True)
    usage = value.get("rusage")
    require(
        isinstance(usage, dict)
        and set(usage)
        == {
            "timer_minor_faults",
            "timer_major_faults",
            "user_cpu_us",
            "system_cpu_us",
            "total_cpu_us",
            "cpu_per_wall",
        },
        "rusage schema drifted",
    )
    user, system = (
        uint(usage.get("user_cpu_us"), "user CPU"),
        uint(usage.get("system_cpu_us"), "system CPU"),
    )
    require(
        uint(usage.get("total_cpu_us"), "total CPU", positive=True) == user + system,
        "total CPU mismatch",
    )
    uint(usage.get("timer_major_faults"), "timer major faults")
    uint(usage.get("timer_minor_faults"), "minor faults")
    cpu_per_wall = finite(usage.get("cpu_per_wall"), "CPU per wall")
    expected_cpu_per_wall = usage["total_cpu_us"] / timing["ready_us"]
    require(
        math.isclose(cpu_per_wall, expected_cpu_per_wall, rel_tol=1e-12, abs_tol=1e-12),
        "CPU per wall does not reconcile",
    )
    proc_usage = value.get("proc_rusage_v4")
    require(
        isinstance(proc_usage, dict) and set(proc_usage) == PROC_RUSAGE_KEYS,
        "proc_rusage_v4 schema drifted",
    )
    for key in PROC_RUSAGE_KEYS:
        uint(proc_usage[key], f"proc_rusage_v4 {key}")
    allocations = value.get("metal_allocated_bytes")
    require(
        isinstance(allocations, dict)
        and set(allocations) == {"before", "ready", "after_drop"},
        "Metal allocation schema drifted",
    )
    for key in allocations:
        uint(allocations[key], f"Metal allocation {key}")
    if arm == "A":
        uint(value.get("worker_count"), "worker count", 4)
        validate_schedule(value.get("parallel_copy_schedule"))
    else:
        uint(value.get("worker_count"), "worker count", 0)
        require(value.get("parallel_copy_schedule") is None, "B schedule not null")
        validate_blit(value.get("blit_population"))
    return value


def result_validity_reasons(value):
    timer_major_faults = value["rusage"]["timer_major_faults"]
    return [f"timer_major_faults={timer_major_faults}"] if timer_major_faults else []


def is_rusage_self_regression(returncode, stderr_text):
    return returncode != 0 and stderr_text.startswith(RUSAGE_PREFIX)


def classify_attempt_stop(returncode, regression, parse_error, validity):
    if parse_error is not None:
        return "implementation_or_contract_defect", parse_error
    if regression:
        return "inconclusive", "RUSAGE_SELF counter regression"
    if returncode is None:
        return "inconclusive", "child did not produce a return code"
    if returncode < 0:
        return "inconclusive", f"child terminated by signal {-returncode}"
    if validity:
        return "inconclusive", f"validity failed: {sorted(set(validity))}"
    if returncode != 0:
        return "implementation_or_contract_defect", f"quiet nonzero exit {returncode}"
    return None, None


def child_command(arm):
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
        "parallel-pread" if arm == "A" else "transient-mmap-blit",
        "--workers",
        "4",
        "--output",
        "json",
    ]


def parse_time(stderr):
    match = re.findall(
        r"^\s*([0-9]+\.[0-9]{2}) real\s+([0-9]+\.[0-9]{2}) user\s+([0-9]+\.[0-9]{2}) sys$",
        stderr,
        re.M,
    )
    require(len(match) == 1, "time summary drifted")
    real, user, system = map(float, match[0])
    values = {}
    for label in TIME_LABELS:
        found = re.findall(rf"^\s*(\d+)\s+{re.escape(label)}$", stderr, re.M)
        require(len(found) == 1, f"time label {label} drifted")
        values[label.replace(" ", "_")] = int(found[0])
    return {
        "real_s": real,
        "user_cpu_s": user,
        "system_cpu_s": system,
        "total_cpu_s": user + system,
        **values,
    }


def resource_reasons(value):
    reasons = []
    uint(value.get("page_faults"), "process page faults")
    for key in ("block_input_operations", "swaps"):
        number = uint(value.get(key), key)
        if number:
            reasons.append(f"{key}={number}")
    return reasons


def vm_counter(text, label):
    match = re.search(rf"^{re.escape(label)}:\s+(\d+)\.$", text, re.M)
    if not match:
        raise RuntimeError(f"cannot parse {label}")
    return int(match.group(1))


def raw(command):
    return subprocess.run(
        command, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    ).stdout


def vm_state():
    errors = []
    values = {}
    try:
        text = raw(["vm_stat"])
        for key, label in (
            ("pageouts", "Pageouts"),
            ("compressions", "Compressions"),
            ("swapouts", "Swapouts"),
            ("compressor_stored_pages", "Pages stored in compressor"),
            ("compressor_occupied_pages", "Pages occupied by compressor"),
        ):
            values[key] = vm_counter(text, label)
    except Exception as error:
        errors.append(f"vm_stat:{error}")
    try:
        text = raw(["sysctl", "-n", "vm.swapusage"])
        match = re.search(r"\bused\s*=\s*([0-9.]+)([BKMGT])", text)
        if not match:
            raise RuntimeError("cannot parse swap")
        values["swap_used_bytes"] = round(
            float(match.group(1))
            * {"B": 1, "K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4}[
                match.group(2)
            ]
        )
    except Exception as error:
        errors.append(f"swap:{error}")
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
        if type(after.get(key)) is int and type(before.get(key)) is int
        else None
        for key in keys
    }
    reasons = (
        [f"{label}_capture_invalid"]
        if before.get("errors") or after.get("errors")
        else []
    )
    for key in keys:
        if deltas[key] is None:
            reasons.append(f"{label}_{key}_unavailable")
    for key in ("pageouts", "compressions", "swapouts"):
        if type(deltas[key]) is int and deltas[key] < 0:
            reasons.append(f"{label}_{key}_regressed")
    for key in ("compressions", "swapouts", "swap_used_bytes"):
        if type(deltas[key]) is int and deltas[key] != 0:
            reasons.append(f"{label}_{key}_changed")
    return {
        "label": label,
        "before": before,
        "after": after,
        "deltas": deltas,
        "advisory": {
            "pageouts_delta": deltas["pageouts"],
            "compressor_stored_pages_delta": deltas["compressor_stored_pages"],
            "compressor_occupied_pages_delta": deltas["compressor_occupied_pages"],
        },
        "failure_reasons": sorted(set(reasons)),
    }


def host_state():
    errors = []
    try:
        thermal = raw(["pmset", "-g", "therm"])
    except Exception as error:
        thermal = None
        errors.append(f"thermal:{error}")
    try:
        battery = raw(["pmset", "-g", "batt"])
    except Exception as error:
        battery = None
        errors.append(f"battery:{error}")
    try:
        pressure = raw(["memory_pressure", "-Q"])
        match = re.search(r"System-wide memory free percentage: (\d+)%", pressure)
        available = int(match.group(1)) if match else None
    except Exception as error:
        pressure, available = None, None
        errors.append(f"memory:{error}")
    competitors = []
    try:
        own = {os.getpid(), os.getppid()}
        text = raw(["ps", "-axo", "pid=,command="])
        pattern = re.compile(r"(?:^|/)(?:qwen|qwen-bench|llama[^/ ]*)(?:\s|$)")
        for line in text.splitlines():
            match = re.match(r"\s*(\d+)\s+(.*)", line)
            if (
                match
                and int(match.group(1)) not in own
                and (
                    MODEL.name.lower() in match.group(2).lower()
                    or pattern.search(match.group(2).lower())
                )
            ):
                competitors.append(
                    {"pid": int(match.group(1)), "command": match.group(2)}
                )
    except Exception as error:
        competitors = None
        errors.append(f"processes:{error}")
    valid = (
        not errors
        and isinstance(thermal, str)
        and "No thermal warning level has been recorded" in thermal
        and "No performance warning level has been recorded" in thermal
        and isinstance(battery, str)
        and "AC Power" in battery
        and type(available) is int
        and available >= 50
        and competitors == []
    )
    return {
        "thermal": thermal,
        "battery": battery,
        "memory_pressure": pressure,
        "memory_available_percent": available,
        "competing_processes": competitors,
        "errors": errors,
        "valid": valid,
        "captured_monotonic_ns": time.monotonic_ns(),
    }


def validate_host_snapshot(value, label):
    require(
        isinstance(value, dict)
        and set(value)
        == {
            "thermal",
            "battery",
            "memory_pressure",
            "memory_available_percent",
            "competing_processes",
            "errors",
            "valid",
            "captured_monotonic_ns",
        },
        f"{label} host schema drifted",
    )
    require(
        isinstance(value["errors"], list)
        and all(type(item) is str for item in value["errors"]),
        f"{label} host errors drifted",
    )
    valid = (
        not value["errors"]
        and isinstance(value["thermal"], str)
        and "No thermal warning level has been recorded" in value["thermal"]
        and "No performance warning level has been recorded" in value["thermal"]
        and isinstance(value["battery"], str)
        and "AC Power" in value["battery"]
        and type(value["memory_available_percent"]) is int
        and value["memory_available_percent"] >= 50
        and value["competing_processes"] == []
    )
    require(value["valid"] is valid, f"{label} host validity drifted")
    uint(value["captured_monotonic_ns"], f"{label} host timestamp", positive=True)


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
            f"{label} {key} drifted",
        )
    require(
        isinstance(value["errors"], list)
        and all(type(item) is str for item in value["errors"]),
        f"{label} VM errors drifted",
    )
    uint(value["captured_monotonic_ns"], f"{label} VM timestamp", positive=True)


def validate_residency_snapshot(value, label):
    require(
        isinstance(value, dict)
        and set(value)
        == {
            "page_size",
            "total_pages",
            "resident_pages",
            "all_pages_resident",
            "file_identity",
        },
        f"{label} residency schema drifted",
    )
    uint(value["page_size"], f"{label} page size", PAGE_SIZE)
    uint(value["total_pages"], f"{label} total pages", MODEL_PAGES)
    resident = uint(value["resident_pages"], f"{label} resident pages")
    require(
        value["all_pages_resident"] is (resident == MODEL_PAGES),
        f"{label} residency summary drifted",
    )


def validate_cooldown_evidence(value):
    require(isinstance(value, dict), "cooldown evidence missing")
    for key in (
        "prior_activity_monotonic_ns",
        "required_interval_ns",
        "eligible_monotonic_ns",
        "started_monotonic_ns",
        "requested_sleep_ns",
        "completed_monotonic_ns",
        "observed_interval_ns",
    ):
        uint(value.get(key), f"cooldown {key}")
    require(
        value["required_interval_ns"] == COOLDOWN_NS
        and value["eligible_monotonic_ns"]
        == value["prior_activity_monotonic_ns"] + COOLDOWN_NS
        and value["observed_interval_ns"]
        == value["completed_monotonic_ns"] - value["prior_activity_monotonic_ns"]
        and value["observed_interval_ns"] >= COOLDOWN_NS
        and value["completed_monotonic_ns"] >= value["started_monotonic_ns"],
        "cooldown evidence does not reconcile",
    )


def validate_conditioning_record(value, stem, require_launchable):
    require(
        isinstance(value, dict)
        and set(value)
        == {
            "schema",
            "stem",
            "signal_start_sequence",
            "cooldown",
            "host_before_conditioning",
            "vm_before_conditioning",
            "conditioning",
            "residency_before",
            "host_before_launch",
            "vm_before_launch",
            "conditioning_interval",
            "residency_proved_ns",
            "operation_errors",
        },
        "conditioning record schema drifted",
    )
    uint(value["schema"], "conditioning schema", 1)
    require(value["stem"] == stem, "conditioning stem drifted")
    uint(value["signal_start_sequence"], "conditioning signal sequence")
    validate_cooldown_evidence(value["cooldown"])
    validate_host_snapshot(value["host_before_conditioning"], "conditioning-before")
    validate_vm_snapshot(value["vm_before_conditioning"], "conditioning-before")
    operation = value["conditioning"]
    if operation is not None:
        require(
            isinstance(operation, dict)
            and set(operation) == {"bytes_read", "wall_ns", "buffer_bytes"},
            "conditioning operation schema drifted",
        )
        uint(operation["bytes_read"], "conditioning bytes", MODEL_SIZE)
        uint(operation["wall_ns"], "conditioning wall", positive=True)
        uint(operation["buffer_bytes"], "conditioning buffer", BUFFER_SIZE)
    residency_value = value["residency_before"]
    if residency_value is not None:
        validate_residency_snapshot(residency_value, "pre-launch")
    host_launch = value["host_before_launch"]
    vm_launch = value["vm_before_launch"]
    interval = value["conditioning_interval"]
    if host_launch is not None:
        validate_host_snapshot(host_launch, "pre-launch")
    if vm_launch is not None:
        validate_vm_snapshot(vm_launch, "pre-launch")
    if interval is not None:
        require(vm_launch is not None, "conditioning interval lacks launch VM state")
        require_typed_equal(
            interval,
            vm_interval("conditioning", value["vm_before_conditioning"], vm_launch),
            "conditioning interval",
        )
    proved = value["residency_proved_ns"]
    require(
        proved is None or (type(proved) is int and proved > 0),
        "residency proof timestamp drifted",
    )
    if type(proved) is int:
        require(
            proved >= value["cooldown"]["completed_monotonic_ns"],
            "residency proof predates cooldown completion",
        )
    errors = value["operation_errors"]
    require(
        isinstance(errors, list) and all(type(item) is str for item in errors),
        "conditioning operation errors drifted",
    )
    reasons = list(errors)
    if value["host_before_conditioning"]["valid"] is not True:
        reasons.append("host_invalid_before_conditioning")
    if value["vm_before_conditioning"]["errors"]:
        reasons.append("vm_invalid_before_conditioning")
    if (
        not isinstance(residency_value, dict)
        or residency_value["resident_pages"] != MODEL_PAGES
    ):
        reasons.append("incomplete_residency_before_launch")
    if not isinstance(host_launch, dict) or host_launch["valid"] is not True:
        reasons.append("host_invalid_before_launch")
    if isinstance(interval, dict):
        reasons.extend(interval["failure_reasons"])
    if require_launchable:
        require(
            not reasons and type(proved) is int,
            "launched attempt had invalid conditioning",
        )
    return sorted(set(reasons))


def identity_from_stat(value):
    return {
        "device": value.st_dev,
        "inode": value.st_ino,
        "size_bytes": value.st_size,
        "mtime_ns": value.st_mtime_ns,
    }


def sequential_condition_path(path, identity, expected_size):
    require(file_identity(path) == identity, "descriptor drifted before conditioning")
    descriptor = os.open(path, os.O_RDONLY)
    started = time.perf_counter_ns()
    total = 0
    try:
        require(
            identity_from_stat(os.fstat(descriptor)) == identity,
            "open descriptor identity drifted",
        )
        buffer = bytearray(BUFFER_SIZE)
        with os.fdopen(descriptor, "rb", buffering=0, closefd=False) as source:
            while True:
                count = source.readinto(buffer)
                if not count:
                    break
                total += count
        require(total == expected_size, "conditioning byte count drifted")
        require(
            identity_from_stat(os.fstat(descriptor)) == identity
            and file_identity(path) == identity,
            "descriptor drifted during conditioning",
        )
    finally:
        os.close(descriptor)
    return {
        "bytes_read": total,
        "wall_ns": time.perf_counter_ns() - started,
        "buffer_bytes": BUFFER_SIZE,
    }


def sequential_condition(identity):
    return sequential_condition_path(MODEL, identity, MODEL_SIZE)


def mincore_residency(path, identity, expected_size):
    page_size = os.sysconf("SC_PAGE_SIZE")
    pages = (expected_size + page_size - 1) // page_size
    descriptor = os.open(path, os.O_RDONLY)
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
    address = None
    try:
        require(
            file_identity(path) == identity
            and identity_from_stat(os.fstat(descriptor)) == identity,
            "descriptor drifted before mincore",
        )
        address = libc.mmap(
            None,
            expected_size,
            mmap.PROT_READ,
            mmap.MAP_SHARED,
            descriptor,
            0,
        )
        if address == ctypes.c_void_p(-1).value:
            number = ctypes.get_errno()
            raise OSError(number, os.strerror(number))
        vector = (ctypes.c_ubyte * pages)()
        if libc.mincore(address, expected_size, vector) != 0:
            number = ctypes.get_errno()
            raise OSError(number, os.strerror(number))
        resident = sum(bool(item & 1) for item in vector)
        require(
            file_identity(path) == identity
            and identity_from_stat(os.fstat(descriptor)) == identity,
            "descriptor drifted during mincore",
        )
    finally:
        if address is not None and address != ctypes.c_void_p(-1).value:
            if libc.munmap(address, expected_size) != 0:
                number = ctypes.get_errno()
                os.close(descriptor)
                raise OSError(number, os.strerror(number))
        os.close(descriptor)
    return {
        "page_size": page_size,
        "total_pages": pages,
        "resident_pages": resident,
        "all_pages_resident": resident == pages,
        "file_identity": identity,
    }


def residency(identity):
    return mincore_residency(MODEL, identity, MODEL_SIZE)


def validate_post_record(value, stem, conditioning):
    require(
        isinstance(value, dict)
        and set(value)
        == {
            "schema",
            "stem",
            "host_after_exit",
            "vm_after_exit",
            "residency_after",
            "child_interval",
            "operation_errors",
        },
        "post-exit record schema drifted",
    )
    uint(value["schema"], "post-exit schema", 1)
    require(value["stem"] == stem, "post-exit stem drifted")
    validate_host_snapshot(value["host_after_exit"], "post-exit")
    validate_vm_snapshot(value["vm_after_exit"], "post-exit")
    if value["residency_after"] is not None:
        validate_residency_snapshot(value["residency_after"], "post-exit")
    require_typed_equal(
        value["child_interval"],
        vm_interval(
            "child",
            conditioning["vm_before_launch"],
            value["vm_after_exit"],
        ),
        "child VM interval",
    )
    require(
        isinstance(value["operation_errors"], list)
        and all(type(item) is str for item in value["operation_errors"]),
        "post-exit errors drifted",
    )


def validate_cleanup_evidence(outcome, events, acquired_time, completed_time):
    actions = outcome.get("cleanup_actions")
    require(isinstance(actions, list), "cleanup actions are not a list")
    require(
        outcome.get("cleanup_signal_sent") is bool(actions),
        "cleanup signal summary drifted",
    )
    require(len(actions) <= 1, "cleanup action cardinality drifted")
    derived_errors = []
    for action in actions:
        require(
            isinstance(action, dict)
            and set(action)
            == {
                "reason",
                "target_pgid",
                "signal",
                "attempted_monotonic_ns",
                "succeeded",
                "error",
                "completed_monotonic_ns",
            },
            "cleanup action schema drifted",
        )
        require(
            action["reason"] == "operator-signal-or-pipe-failure",
            "cleanup reason drifted",
        )
        uint(action["target_pgid"], "cleanup target pgid", outcome.get("pgid"), True)
        uint(action["signal"], "cleanup signal", int(signal.SIGKILL), True)
        started = uint(
            action["attempted_monotonic_ns"], "cleanup start time", positive=True
        )
        completed = uint(
            action["completed_monotonic_ns"], "cleanup completion time", positive=True
        )
        require(
            type(acquired_time) is int
            and type(completed_time) is int
            and acquired_time <= started <= completed <= completed_time,
            "cleanup timestamps drifted",
        )
        require(type(action["succeeded"]) is bool, "cleanup success type drifted")
        if action["succeeded"]:
            require(action["error"] is None, "successful cleanup retained an error")
        else:
            require(type(action["error"]) is str, "failed cleanup lacks an error")
            derived_errors.append(action["error"])
    require_typed_equal(
        outcome.get("termination_errors"), derived_errors, "cleanup termination errors"
    )
    trigger_present = bool(
        events
        or outcome.get("wait_errors")
        or outcome.get("pipe_errors")
        or outcome.get("interrupted") is True
    )
    require(not actions or trigger_present, "cleanup action has no stored trigger")


def derive_attempt_validity(post, proved, outcome, resources, result, events):
    validity = list(post["child_interval"]["failure_reasons"])
    validity.extend(post["operation_errors"])
    if post["host_after_exit"]["valid"] is not True:
        validity.append("host_invalid_after_exit")
    after = post["residency_after"]
    if not isinstance(after, dict) or after.get("resident_pages") != MODEL_PAGES:
        validity.append("incomplete_residency_after_exit")
    acquired = outcome.get("acquired_monotonic_ns")
    if acquired is None and "launch_acquired_ns" in outcome:
        acquired = outcome.get("launch_acquired_ns")
    if type(acquired) is not int or acquired - proved > LAUNCH_LIMIT_NS:
        validity.append("launch_exceeded_five_seconds")
    if outcome.get("spawn_error") is not None:
        validity.append(f"spawn_error={outcome['spawn_error']}")
    returncode = outcome.get("returncode")
    if type(returncode) is int and returncode < 0:
        validity.append(f"child_signal={-returncode}")
    disposition_value = outcome.get("group_disposition")
    if (
        outcome.get("reaped") is not True
        or not isinstance(disposition_value, dict)
        or disposition_value.get("no_live_group_observed") is not True
    ):
        validity.append("lifecycle_invalid")
    validity.extend(outcome.get("wait_errors", []))
    validity.extend(outcome.get("pipe_errors", []))
    validity.extend(
        f"termination_error={item}" for item in outcome.get("termination_errors", [])
    )
    validity.extend(resource_reasons(resources) if resources else [])
    if result is not None:
        validity.extend(result_validity_reasons(result))
    if outcome.get("output_overflow") is True:
        validity.append("bounded_output_exceeded")
    if outcome.get("interrupted") is True:
        validity.append("operator_interrupt_during_child_wait")
    actions = outcome.get("cleanup_actions", [])
    if outcome.get("cleanup_signal_sent") is not bool(actions) or len(actions) > 1:
        validity.append("cleanup_action_cardinality_invalid")
    if actions and not (
        events
        or outcome.get("wait_errors")
        or outcome.get("pipe_errors")
        or outcome.get("interrupted") is True
    ):
        validity.append("cleanup_without_trigger")
    if isinstance(disposition_value, dict):
        for sample in disposition_value.get("samples", []):
            if sample.get("error") is not None:
                validity.append("process_group_inspection_error")
    if events:
        validity.append("deferred_operator_signal")
    return sorted(set(validity))


def median(values):
    require(bool(values), "empty median")
    ordered = sorted(values)
    middle = len(ordered) // 2
    return (
        float(ordered[middle])
        if len(ordered) % 2
        else (ordered[middle - 1] + ordered[middle]) / 2
    )


def score_rows(rows):
    require(
        len(rows) == 12
        and all(not row.get("validity_reasons") and row.get("result") for row in rows),
        "scoring requires 12 valid rows",
    )
    pairs = []
    for pair in range(1, 7):
        values = {row["arm"]: row for row in rows if row["pair"] == pair}
        require(set(values) == {"A", "B"}, "pair incomplete")
        a, b = values["A"], values["B"]
        ar, br = a["result"], b["result"]
        for label, denominator in (
            ("A ready", ar["timing"]["ready_us"]),
            ("A CPU", ar["rusage"]["total_cpu_us"]),
            ("A RSS", a["process_resources"]["maximum_resident_set_size"]),
            ("A footprint", a["process_resources"]["peak_memory_footprint"]),
        ):
            finite(denominator, label, positive=True)
        finite(b["process_resources"]["maximum_resident_set_size"], "B RSS", True)
        finite(b["process_resources"]["peak_memory_footprint"], "B footprint", True)
        d = ar["timing"]["ready_us"] - br["timing"]["ready_us"]
        pairs.append(
            {
                "pair": pair,
                "order": "".join(PAIR_ORDERS[pair - 1]),
                "d_us": d,
                "q": br["timing"]["ready_us"] / ar["timing"]["ready_us"],
                "c": br["rusage"]["total_cpu_us"] / ar["rusage"]["total_cpu_us"],
                "rss_ratio": b["process_resources"]["maximum_resident_set_size"]
                / a["process_resources"]["maximum_resident_set_size"],
                "footprint_ratio": b["process_resources"]["peak_memory_footprint"]
                / a["process_resources"]["peak_memory_footprint"],
            }
        )
    ab = [item for item in pairs if item["order"] == "AB"]
    ba = [item for item in pairs if item["order"] == "BA"]
    gates = {
        "median_d_at_least_112000": median([x["d_us"] for x in pairs]) >= 112_000,
        "AB_median_d_at_least_112000": median([x["d_us"] for x in ab]) >= 112_000,
        "BA_median_d_at_least_112000": median([x["d_us"] for x in ba]) >= 112_000,
        "wins_at_least_5_of_6": sum(x["d_us"] > 0 for x in pairs) >= 5,
        "AB_wins_at_least_2_of_3": sum(x["d_us"] > 0 for x in ab) >= 2,
        "BA_wins_at_least_2_of_3": sum(x["d_us"] > 0 for x in ba) >= 2,
        "median_cpu_at_most_1_10": median([x["c"] for x in pairs]) <= 1.10,
        "AB_median_cpu_at_most_1_10": median([x["c"] for x in ab]) <= 1.10,
        "BA_median_cpu_at_most_1_10": median([x["c"] for x in ba]) <= 1.10,
        "max_RSS_at_most_1_05": max(x["rss_ratio"] for x in pairs) <= 1.05,
        "max_footprint_at_most_1_05": max(x["footprint_ratio"] for x in pairs) <= 1.05,
    }
    return {
        "pairs": pairs,
        "D_us": median([x["d_us"] for x in pairs]),
        "D_AB_us": median([x["d_us"] for x in ab]),
        "D_BA_us": median([x["d_us"] for x in ba]),
        "Q": median([x["q"] for x in pairs]),
        "C": median([x["c"] for x in pairs]),
        "C_AB": median([x["c"] for x in ab]),
        "C_BA": median([x["c"] for x in ba]),
        "wins": sum(x["d_us"] > 0 for x in pairs),
        "AB_wins": sum(x["d_us"] > 0 for x in ab),
        "BA_wins": sum(x["d_us"] > 0 for x in ba),
        "max_paired_RSS_ratio": max(x["rss_ratio"] for x in pairs),
        "max_paired_footprint_ratio": max(x["footprint_ratio"] for x in pairs),
        "gates": gates,
        "qualifies": all(gates.values()),
        "descriptive": {
            "A_phases": [x["result"]["timing"] for x in rows if x["arm"] == "A"],
            "B_phases": [x["result"]["timing"] for x in rows if x["arm"] == "B"],
            "allocations": [x["result"]["metal_allocated_bytes"] for x in rows],
            "B_blit": [x["result"]["blit_population"] for x in rows if x["arm"] == "B"],
        },
    }


def classify(
    contract=False, invalid=False, regression=False, complete=False, qualifies=False
):
    if contract:
        return "implementation_or_contract_defect"
    if invalid or regression or not complete:
        return "inconclusive"
    return "GO-mechanism-floor" if qualifies else "mechanism-floor-miss"


def parse_cargo_summary(text):
    matches = re.findall(
        r"^test result: ok\. (\d+) passed; (\d+) failed; (\d+) ignored; "
        r"(\d+) measured; (\d+) filtered out; finished in "
        r"[0-9]+(?:\.[0-9]+)?s$",
        text,
        re.M,
    )
    require(len(matches) == 1, "Cargo summary not unique")
    values = tuple(map(int, matches[0]))
    require(values[:4] == (13, 0, 0, 0), "Cargo summary is not exactly 13 passed")
    return dict(
        zip(("passed", "failed", "ignored", "measured", "filtered_out"), values)
    )


def gate(command, env, controller, cargo=False):
    start = controller.sequence()
    controller.quiet(start, "preflight child launch")
    outcome = bounded_child(command, env, controller, start)
    events = controller.since(start)
    require(
        outcome["returncode"] == 0
        and not outcome["wait_errors"]
        and not outcome["poll_errors"]
        and not outcome["pipe_errors"]
        and not outcome["termination_errors"]
        and not outcome["output_overflow"]
        and not outcome["interrupted"]
        and not outcome["cleanup_actions"]
        and outcome["reaped"]
        and outcome["group_disposition"]["no_live_group_observed"]
        and all(
            not sample["error"] for sample in outcome["group_disposition"]["samples"]
        )
        and not events,
        "preflight gate failed",
    )
    stdout, stderr = outcome["stdout"].decode(), outcome["stderr"].decode()
    return {
        "command": command,
        "returncode": 0,
        "stdout": stdout,
        "stderr": stderr,
        "stdout_sha256": sha_bytes(outcome["stdout"]),
        "stderr_sha256": sha_bytes(outcome["stderr"]),
        "cargo_test_summary": parse_cargo_summary(stdout + stderr) if cargo else None,
        "signal_start_sequence": start,
        "signal_events": events,
        "pid": outcome["pid"],
        "pgid": outcome["pgid"],
        "attempted_monotonic_ns": outcome["attempted_monotonic_ns"],
        "acquired_monotonic_ns": outcome["acquired_monotonic_ns"],
        "completed_monotonic_ns": outcome["completed_monotonic_ns"],
        "spawn_error": outcome["spawn_error"],
        "reaped": outcome["reaped"],
        "output_overflow": outcome["output_overflow"],
        "interrupted": outcome["interrupted"],
        "wait_errors": outcome["wait_errors"],
        "poll_errors": outcome["poll_errors"],
        "pipe_errors": outcome["pipe_errors"],
        "termination_errors": outcome["termination_errors"],
        "group_disposition": outcome["group_disposition"],
        "cleanup_actions": outcome["cleanup_actions"],
        "cleanup_signal_sent": outcome["cleanup_signal_sent"],
    }


def self_test_gate(env, controller, optimized):
    command = [sys.executable]
    if optimized:
        command.append("-O")
    command.extend([str(RUNNER), "--self-test-child"])
    evidence = gate(command, env, controller)
    require(
        evidence["stdout"] == "self-test-child: ok\n",
        "optimized self-test output drifted",
    )
    require(evidence["stderr"] == "", "optimized self-test wrote stderr")
    return evidence


def describe(env, build):
    command = [
        str(BINARY),
        "gguf-arena-floor",
        "--model",
        str(MODEL),
        "--describe",
        "--workers",
        "4",
        "--output",
        "json",
    ]
    value = parse_json_bytes(command_output(command, env).encode(), "describe")
    uint(value.get("schema_version"), "describe schema version", 2)
    uint(value.get("request_count"), "describe request count", COUNT)
    uint(value.get("logical_copy_bytes"), "describe logical bytes", COPY_BYTES)
    require(
        value.get("mode") == "describe"
        and value.get("matched_profile") == PROFILE
        and value.get("descriptor_layout_digest") == DESCRIPTOR
        and value.get("inventory_digest") == INVENTORY,
        "describe identity drifted",
    )
    require_typed_equal(value.get("build_identity"), build, "describe build identity")
    validate_schedule(value.get("computed_schedule"))
    require_typed_equal(
        value.get("parallel_copy_schedule"),
        value.get("computed_schedule"),
        "parallel/computed schedule",
    )
    require_typed_equal(
        value.get("frozen_schedule"),
        value.get("computed_schedule"),
        "frozen/computed schedule",
    )
    planner = value.get("retained_planner")
    require(
        isinstance(planner, dict)
        and planner.get("status") == "ok"
        and planner.get("planner_digest") == BLIT_PLAN,
        "blit plan drifted",
    )
    return {"command": command, "output": value}


def preflight():
    require(not lexists(PACKET) and not lexists(WORK), "packet roots already exist")
    env, removed = normalized_environment()
    with SignalController(True) as preflight_controller:
        process_tests = run_process_self_tests(preflight_controller)
        normal = self_test_gate(env, preflight_controller, False)
        optimized = self_test_gate(env, preflight_controller, True)
        gates = [
            gate(
                [
                    "cargo",
                    "build",
                    "--release",
                    "-p",
                    "qwen-cli",
                    "--bin",
                    "qwen-bench",
                ],
                env,
                preflight_controller,
            ),
            gate(
                [
                    "cargo",
                    "test",
                    "--release",
                    "-p",
                    "qwen-cli",
                    "--bin",
                    "qwen-bench",
                    "gguf_arena_floor::tests",
                    "--",
                    "--test-threads=1",
                ],
                env,
                preflight_controller,
                True,
            ),
        ]
        preflight_controller.quiet(0, "preflight child gates")
    source = source_identity(env)
    binary = descriptor_hash(BINARY)
    identity = file_identity(MODEL)
    require(
        identity["size_bytes"] == MODEL_SIZE
        and os.sysconf("SC_PAGE_SIZE") == PAGE_SIZE,
        "asset geometry drifted",
    )
    require(sha_file(MODEL) == MODEL_SHA256, "model SHA drifted")
    controls = describe(env, source["build_identity"])
    manifest = {
        "schema": 1,
        "protocol": "v0643-a3b-transient-mmap-blit-floor",
        "created_unix_ms": time.time_ns() // 1_000_000,
        "source": source,
        "source_commit": source["head"],
        "implementation_parent": IMPLEMENTATION_PARENT,
        "build_identity": source["build_identity"],
        "binary_bytes": binary,
        "model": str(MODEL),
        "model_size_bytes": MODEL_SIZE,
        "model_pages": MODEL_PAGES,
        "model_sha256": MODEL_SHA256,
        "model_file_identity": identity,
        "descriptor_digest": DESCRIPTOR,
        "inventory_digest": INVENTORY,
        "normalized_environment": env,
        "removed_environment": removed,
        "commands": {arm: child_command(arm) for arm in ("A", "B")},
        "pair_orders": [list(row) for row in PAIR_ORDERS],
        "controls": controls,
        "build_and_test_gates": gates,
        "process_self_tests": process_tests,
        "normal_self_test_gate": normal,
        "optimized_self_test_gate": optimized,
        "attempt_count": 12,
        "retry_count": 0,
        "expected_inventory_members": EXPECTED_INVENTORY_MEMBERS,
        "authority": "none",
        "force_authorized": False,
        "successor_authorization": "none",
        "independent_justification": "roadmap-candidate",
        "imported_rows": 0,
        "imported_timings": 0,
        "imported_authority": False,
        "historical_112ms_role": "portfolio-gate-only-not-MDE",
    }
    return manifest, env


def cooldown(activity):
    started = time.monotonic_ns()
    eligible = activity + COOLDOWN_NS
    remaining = eligible - started
    if remaining > 0:
        time.sleep(remaining / 1e9)
    completed = time.monotonic_ns()
    require(completed - activity >= COOLDOWN_NS, "cooldown short")
    return {
        "prior_activity_monotonic_ns": activity,
        "required_interval_ns": COOLDOWN_NS,
        "eligible_monotonic_ns": eligible,
        "started_monotonic_ns": started,
        "requested_sleep_ns": max(remaining, 0),
        "completed_monotonic_ns": completed,
        "observed_interval_ns": completed - activity,
    }


def launch_delay_failure(residency_proved_ns, acquired_ns):
    if type(acquired_ns) is not int:
        return None
    if acquired_ns - residency_proved_ns > LAUNCH_LIMIT_NS:
        return "launch_exceeded_five_seconds"
    return None


def run_one(pair, position, arm, env, manifest, activity, controller):
    stem = f"pair{pair}-p{position}-{arm}"
    controller.quiet(0, f"{stem} attempt baseline")
    attempt_start = controller.sequence()
    controller.quiet(attempt_start, f"{stem} cooldown")
    cooldown_evidence = cooldown(activity)
    controller.quiet(attempt_start, f"{stem} conditioning")
    verify_binary_identity(manifest, env)
    host_before = host_state()
    vm_before = vm_state()
    conditioning = None
    before = None
    proved = None
    host_launch = None
    vm_launch = None
    conditioning_interval = None
    operation_errors = []
    if host_before.get("valid") is True and not vm_before.get("errors"):
        try:
            conditioning = sequential_condition(manifest["model_file_identity"])
        except (OSError, RuntimeError) as error:
            operation_errors.append(f"conditioning:{type(error).__name__}:{error}")
        if conditioning is not None:
            try:
                before = residency(manifest["model_file_identity"])
                proved = time.monotonic_ns()
            except (OSError, RuntimeError) as error:
                operation_errors.append(
                    f"pre_spawn_residency:{type(error).__name__}:{error}"
                )
        if before is not None:
            if before["page_size"] != PAGE_SIZE or before["total_pages"] != MODEL_PAGES:
                operation_errors.append("pre_spawn_residency:page_geometry_drift")
            host_launch = host_state()
            vm_launch = vm_state()
            conditioning_interval = vm_interval("conditioning", vm_before, vm_launch)
    record = {
        "schema": 1,
        "stem": stem,
        "signal_start_sequence": attempt_start,
        "cooldown": cooldown_evidence,
        "host_before_conditioning": host_before,
        "vm_before_conditioning": vm_before,
        "conditioning": conditioning,
        "residency_before": before,
        "host_before_launch": host_launch,
        "vm_before_launch": vm_launch,
        "conditioning_interval": conditioning_interval,
        "residency_proved_ns": proved,
        "operation_errors": operation_errors,
    }
    write_json(WORK / f"{stem}.conditioning.json", record)
    reasons = list(operation_errors)
    if host_before.get("valid") is not True:
        reasons.append("host_invalid_before_conditioning")
    if vm_before.get("errors"):
        reasons.append("vm_invalid_before_conditioning")
    if not isinstance(before, dict) or before.get("resident_pages") != MODEL_PAGES:
        reasons.append("incomplete_residency_before_launch")
    if not isinstance(host_launch, dict) or host_launch.get("valid") is not True:
        reasons.append("host_invalid_before_launch")
    if isinstance(conditioning_interval, dict):
        reasons.extend(conditioning_interval["failure_reasons"])
    if controller.since(attempt_start):
        reasons.append("deferred_operator_signal")
    if reasons:
        raise Inconclusive(f"{stem} conditioning invalid: {sorted(set(reasons))}")
    require(type(proved) is int, "residency proof timestamp is missing")
    require(isinstance(vm_launch, dict), "pre-launch VM state is missing")
    controller.quiet(attempt_start, f"{stem} launch")
    command = child_command(arm)
    append_jsonl(
        PACKET / "lifecycle.jsonl",
        {
            "schema": 1,
            "event": "launch",
            "stem": stem,
            "pair": pair,
            "position": position,
            "arm": arm,
            "command": command,
            "signal_start_sequence": attempt_start,
            "conditioning_sha256": sha_file(WORK / f"{stem}.conditioning.json"),
            "residency_proved_ns": proved,
            "monotonic_ns": time.monotonic_ns(),
        },
    )
    outcome = bounded_child(
        command,
        env,
        controller,
        attempt_start,
        lambda acquired: append_jsonl(
            PACKET / "lifecycle.jsonl",
            {"schema": 1, "event": "acquired", "stem": stem, **acquired},
        ),
    )
    write_exclusive(WORK / f"{stem}.stdout", outcome["stdout"])
    write_exclusive(WORK / f"{stem}.stderr", outcome["stderr"])
    completion = {
        "schema": 1,
        "event": "completion",
        "stem": stem,
        "pid": outcome["pid"],
        "pgid": outcome["pgid"],
        "spawn_error": outcome["spawn_error"],
        "returncode": outcome["returncode"],
        "interrupted": outcome["interrupted"],
        "wait_errors": outcome["wait_errors"],
        "poll_errors": outcome["poll_errors"],
        "pipe_errors": outcome["pipe_errors"],
        "termination_errors": outcome["termination_errors"],
        "output_overflow": outcome["output_overflow"],
        "cleanup_actions": outcome["cleanup_actions"],
        "cleanup_signal_sent": outcome["cleanup_signal_sent"],
        "group_disposition": outcome["group_disposition"],
        "reaped": outcome["reaped"],
        "attempted_monotonic_ns": outcome["attempted_monotonic_ns"],
        "acquired_monotonic_ns": outcome["acquired_monotonic_ns"],
        "completed_monotonic_ns": outcome["completed_monotonic_ns"],
        "residency_to_acquired_ns": (
            outcome["acquired_monotonic_ns"] - proved
            if type(outcome["acquired_monotonic_ns"]) is int
            else None
        ),
    }
    append_jsonl(PACKET / "lifecycle.jsonl", completion)
    host_after = host_state()
    vm_after = vm_state()
    after = None
    post_errors = []
    try:
        after = residency(manifest["model_file_identity"])
    except (OSError, RuntimeError) as error:
        post_errors.append(f"post_exit_residency:{type(error).__name__}:{error}")
    child_interval = vm_interval("child", vm_launch, vm_after)
    post = {
        "schema": 1,
        "stem": stem,
        "host_after_exit": host_after,
        "vm_after_exit": vm_after,
        "residency_after": after,
        "child_interval": child_interval,
        "operation_errors": post_errors,
    }
    write_json(WORK / f"{stem}.post.json", post)
    stderr_text = outcome["stderr"].decode("utf-8", errors="replace")
    regression = is_rusage_self_regression(outcome["returncode"], stderr_text)
    parse_error = None
    result = None
    resources = None
    if outcome["returncode"] == 0:
        try:
            resources = parse_time(stderr_text)
            result = validate_result(
                parse_json_bytes(outcome["stdout"], stem),
                arm,
                manifest["build_identity"],
            )
        except ContractDefect as error:
            parse_error = f"{type(error).__name__}:{error}"
    events = controller.since(attempt_start)
    validity = derive_attempt_validity(post, proved, outcome, resources, result, events)
    stop_classification, stop_reason = classify_attempt_stop(
        outcome["returncode"], regression, parse_error, validity
    )
    activity_boundary = time.monotonic_ns()
    attempt = {
        "schema": 1,
        "stem": stem,
        "pair": pair,
        "position": position,
        "order": "".join(PAIR_ORDERS[pair - 1]),
        "arm": arm,
        "command": command,
        "pid": outcome["pid"],
        "pgid": outcome["pgid"],
        "spawn_error": outcome["spawn_error"],
        "returncode": outcome["returncode"],
        "reaped": outcome["reaped"],
        "output_overflow": outcome["output_overflow"],
        "interrupted": outcome["interrupted"],
        "wait_errors": outcome["wait_errors"],
        "poll_errors": outcome["poll_errors"],
        "pipe_errors": outcome["pipe_errors"],
        "termination_errors": outcome["termination_errors"],
        "cleanup_actions": outcome["cleanup_actions"],
        "cleanup_signal_sent": outcome["cleanup_signal_sent"],
        "group_disposition": outcome["group_disposition"],
        "signal_start_sequence": attempt_start,
        "residency_proved_ns": proved,
        "launch_attempted_ns": outcome["attempted_monotonic_ns"],
        "launch_acquired_ns": outcome["acquired_monotonic_ns"],
        "residency_to_acquired_ns": completion["residency_to_acquired_ns"],
        "process_resources": resources,
        "process_page_faults": resources.get("page_faults") if resources else None,
        "rusage_self_counter_regression": regression,
        "rusage_regression_stderr_prefix": RUSAGE_PREFIX if regression else None,
        "parse_error": parse_error,
        "validity_reasons": sorted(set(validity)),
        "stop_classification": stop_classification,
        "stop_reason": stop_reason,
        "result": result,
        "stdout_sha256": sha_file(WORK / f"{stem}.stdout"),
        "stderr_sha256": sha_file(WORK / f"{stem}.stderr"),
        "conditioning_sha256": sha_file(WORK / f"{stem}.conditioning.json"),
        "post_sha256": sha_file(WORK / f"{stem}.post.json"),
        "activity_boundary_monotonic_ns": activity_boundary,
    }
    append_jsonl(PACKET / "attempts.jsonl", attempt)
    append_jsonl(
        PACKET / "attempt-signal-closures.jsonl",
        {
            "schema": 1,
            "stem": stem,
            "signal_start_sequence": attempt_start,
            "signal_end_sequence": events[-1]["sequence"] if events else attempt_start,
            "events": events,
            "invalid": bool(events),
            "attempt_sha256": sha_bytes(json_bytes(attempt)),
        },
    )
    if stop_classification == "implementation_or_contract_defect":
        raise ContractDefect(f"{stem} {stop_reason}")
    if stop_classification == "inconclusive":
        raise Inconclusive(f"{stem} {stop_reason}")
    return attempt, activity_boundary


def reserve(manifest):
    require(not lexists(PACKET) and not lexists(WORK), "packet roots already exist")
    PACKET.parent.mkdir(parents=True, exist_ok=True)
    PACKET.mkdir()
    WORK.mkdir()
    write_json(PACKET / "manifest.json", manifest)
    write_json(
        PACKET / "order.json", {"schema": 1, "pairs": [list(x) for x in PAIR_ORDERS]}
    )
    activity_boundary = time.monotonic_ns()
    write_json(
        WORK / "reservation.json",
        {
            "schema": 1,
            "packet": str(PACKET.relative_to(ROOT)),
            "activity_boundary_monotonic_ns": activity_boundary,
        },
    )
    fsync_dir(PACKET)
    fsync_dir(WORK)
    fsync_dir(PACKET.parent)
    return activity_boundary


def read_json_file(path, label):
    require(path.is_file() and not path.is_symlink(), f"{label} is not a regular file")
    return parse_json_bytes(path.read_bytes(), label)


def read_jsonl_file(path, label):
    require(path.is_file() and not path.is_symlink(), f"{label} is not a regular file")
    encoded = path.read_bytes()
    require(encoded.endswith(b"\n"), f"{label} lacks its final newline")
    lines = encoded.splitlines()
    require(bool(lines), f"{label} is empty")
    rows = []
    for index, line in enumerate(lines, 1):
        value = parse_json_bytes(line, f"{label} line {index}")
        require(
            line + b"\n" == json_bytes(value),
            f"{label} line {index} is not canonical JSONL",
        )
        rows.append(value)
    return rows


def attempt_specs():
    return [
        {
            "stem": f"pair{pair}-p{position}-{arm}",
            "pair": pair,
            "position": position,
            "arm": arm,
            "order": "".join(order),
            "command": child_command(arm),
        }
        for pair, order in enumerate(PAIR_ORDERS, 1)
        for position, arm in enumerate(order, 1)
    ]


def stable_regular_member(path):
    before = os.lstat(path)
    require(
        stat.S_ISREG(before.st_mode) and before.st_nlink == 1,
        f"nonregular or multiply linked packet member: {path}",
    )
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        opened = os.fstat(descriptor)
        require(
            (opened.st_dev, opened.st_ino) == (before.st_dev, before.st_ino)
            and stat.S_ISREG(opened.st_mode)
            and opened.st_nlink == 1,
            f"packet member identity drifted: {path}",
        )
        digest = hashlib.sha256()
        size = 0
        for chunk in iter(lambda: os.read(descriptor, 1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
        after = os.fstat(descriptor)
        require(
            complete_stamp(after) == complete_stamp(opened) and size == opened.st_size,
            f"packet member changed while hashing: {path}",
        )
    finally:
        os.close(descriptor)
    final = os.lstat(path)
    require(
        complete_stamp(final) == complete_stamp(before),
        f"packet member pathname changed: {path}",
    )
    return {"path": path, "sha256": digest.hexdigest(), "size_bytes": size}


def regular_member_snapshot(root):
    root_stat = os.lstat(root)
    require(
        stat.S_ISDIR(root_stat.st_mode) and not root.is_symlink(),
        f"packet root is not a real directory: {root}",
    )
    names_before = sorted(os.listdir(root))
    members = {name: stable_regular_member(root / name) for name in names_before}
    require(
        sorted(os.listdir(root)) == names_before,
        f"packet population changed: {root}",
    )
    return members


def require_snapshot_unchanged(before, after, label):
    require(set(before) <= set(after), f"{label} lost an inventoried member")
    for name, original in before.items():
        current = after[name]
        require(
            current["path"] == original["path"]
            and current["sha256"] == original["sha256"]
            and current["size_bytes"] == original["size_bytes"],
            f"{label} member changed after inventory: {name}",
        )


def complete_preinventory_packet_names():
    return {
        "attempt-signal-closures.jsonl",
        "attempts.jsonl",
        "decision.json",
        "decision.sha256",
        "lifecycle.jsonl",
        "manifest.json",
        "order.json",
        "packet-signal-cutoff.json",
        "packet-signal-log.json",
    }


def complete_work_names():
    names = {"reservation.json"}
    for pair, order in enumerate(PAIR_ORDERS, 1):
        for position, arm in enumerate(order, 1):
            stem = f"pair{pair}-p{position}-{arm}"
            names.update(
                {
                    f"{stem}.conditioning.json",
                    f"{stem}.post.json",
                    f"{stem}.stdout",
                    f"{stem}.stderr",
                }
            )
    return names


def require_exact_names(observed, expected, label):
    require(observed == expected, f"{label} filename set drifted")


def validate_group_disposition(value, pgid):
    require(isinstance(value, dict), "group disposition is not an object")
    uint(value.get("pgid"), "disposition pgid", pgid, positive=True)
    samples = value.get("samples")
    require(isinstance(samples, list) and samples, "group disposition samples missing")
    for sample in samples:
        require(
            isinstance(sample, dict)
            and type(sample.get("captured_monotonic_ns")) is int
            and isinstance(sample.get("members"), list)
            and sample.get("error") is None,
            "group disposition sample drifted",
        )
        require(
            all(type(member) is int and member > 0 for member in sample["members"]),
            "group disposition member drifted",
        )
    require(
        value.get("no_live_group_observed") is True and samples[-1]["members"] == [],
        "group disposition did not prove absence",
    )


def validate_attempt_bundle(
    spec, attempt, closure, lifecycle_rows, packet, work, manifest
):
    require(set(attempt) == ATTEMPT_KEYS, "attempt key set drifted")
    require(
        isinstance(closure, dict)
        and set(closure)
        == {
            "schema",
            "stem",
            "signal_start_sequence",
            "signal_end_sequence",
            "events",
            "invalid",
            "attempt_sha256",
        },
        "attempt closure key set drifted",
    )
    uint(attempt.get("schema"), "attempt schema", 1)
    uint(closure.get("schema"), "closure schema", 1)
    for key in ("stem", "pair", "position", "arm", "order", "command"):
        require_typed_equal(
            attempt.get(key), spec[key], f"attempt {spec['stem']} {key}"
        )
    conditioning_path = work / f"{spec['stem']}.conditioning.json"
    post_path = work / f"{spec['stem']}.post.json"
    stdout_path = work / f"{spec['stem']}.stdout"
    stderr_path = work / f"{spec['stem']}.stderr"
    conditioning = read_json_file(conditioning_path, f"{spec['stem']} conditioning")
    post = read_json_file(post_path, f"{spec['stem']} post")
    validate_conditioning_record(conditioning, spec["stem"], True)
    validate_post_record(post, spec["stem"], conditioning)
    require_typed_equal(
        conditioning["residency_before"]["file_identity"],
        manifest["model_file_identity"],
        "pre-launch residency model identity",
    )
    residency_after = post["residency_after"]
    residency_errors = [
        item
        for item in post["operation_errors"]
        if item.startswith("post_exit_residency:")
    ]
    if residency_after is None:
        require(residency_errors, "missing post residency has no captured error")
    else:
        require(not residency_errors, "successful post residency retained an error")
        require_typed_equal(
            residency_after["file_identity"],
            manifest["model_file_identity"],
            "post-exit residency model identity",
        )
    artifact_hashes = {
        "stdout_sha256": sha_file(stdout_path),
        "stderr_sha256": sha_file(stderr_path),
        "conditioning_sha256": sha_file(conditioning_path),
        "post_sha256": sha_file(post_path),
    }
    for key, digest in artifact_hashes.items():
        require(attempt.get(key) == digest, f"attempt {key} drifted")
    require(
        closure.get("stem") == spec["stem"]
        and closure.get("attempt_sha256") == sha_bytes(json_bytes(attempt))
        and closure.get("signal_start_sequence")
        == attempt.get("signal_start_sequence"),
        "attempt closure binding drifted",
    )
    events = closure.get("events")
    require(isinstance(events, list), "attempt closure events drifted")
    for index, event in enumerate(events, 1):
        require(isinstance(event, dict), "attempt closure event is not an object")
        uint(
            event.get("sequence"),
            f"closure event {index} sequence",
            attempt["signal_start_sequence"] + index,
            True,
        )
        uint(event.get("signal"), f"closure event {index} signal", positive=True)
        uint(event.get("monotonic_ns"), f"closure event {index} time", positive=True)
    expected_end = (
        events[-1].get("sequence") if events else attempt["signal_start_sequence"]
    )
    require(
        closure.get("signal_end_sequence") == expected_end
        and closure.get("invalid") is bool(events),
        "attempt closure signal summary drifted",
    )
    require(
        lifecycle_rows and lifecycle_rows[0].get("event") == "launch", "launch missing"
    )
    launch = lifecycle_rows[0]
    require(
        set(launch)
        == {
            "schema",
            "event",
            "stem",
            "pair",
            "position",
            "arm",
            "command",
            "signal_start_sequence",
            "conditioning_sha256",
            "residency_proved_ns",
            "monotonic_ns",
        },
        "launch key set drifted",
    )
    uint(launch.get("schema"), "launch schema", 1)
    for key in ("stem", "pair", "position", "arm", "command"):
        require_typed_equal(launch.get(key), spec[key], f"launch {key}")
    require(
        launch.get("conditioning_sha256") == artifact_hashes["conditioning_sha256"]
        and launch.get("signal_start_sequence") == attempt.get("signal_start_sequence")
        and launch.get("residency_proved_ns")
        == conditioning["residency_proved_ns"]
        == attempt.get("residency_proved_ns"),
        "launch binding drifted",
    )
    uint(attempt.get("signal_start_sequence"), "attempt signal sequence")
    uint(attempt.get("residency_proved_ns"), "attempt residency proof", positive=True)
    uint(attempt.get("launch_attempted_ns"), "attempt launch time", positive=True)
    for key in (
        "wait_errors",
        "poll_errors",
        "pipe_errors",
        "termination_errors",
        "cleanup_actions",
    ):
        require(isinstance(attempt.get(key), list), f"attempt {key} type drifted")
    for key in ("reaped", "output_overflow", "interrupted", "cleanup_signal_sent"):
        require(type(attempt.get(key)) is bool, f"attempt {key} type drifted")
    pid = attempt.get("pid")
    if pid is None:
        require(
            [row.get("event") for row in lifecycle_rows] == ["launch", "completion"],
            "spawn-failure lifecycle drifted",
        )
        completion = lifecycle_rows[1]
        require(
            type(attempt.get("spawn_error")) is str
            and attempt.get("pgid") is None
            and attempt.get("returncode") is None
            and attempt.get("group_disposition") is None
            and attempt.get("launch_acquired_ns") is None
            and attempt.get("residency_to_acquired_ns") is None
            and attempt.get("reaped") is False
            and attempt.get("cleanup_actions") == []
            and attempt.get("cleanup_signal_sent") is False,
            "spawn-failure process shape drifted",
        )
    else:
        uint(pid, "attempt pid", positive=True)
        uint(attempt.get("pgid"), "attempt pgid", pid, positive=True)
        require(
            attempt.get("spawn_error") is None
            and type(attempt.get("returncode")) is int
            and attempt.get("reaped") is True,
            "acquired process shape drifted",
        )
        require(
            [row.get("event") for row in lifecycle_rows]
            == ["launch", "acquired", "completion"],
            "acquired lifecycle drifted",
        )
        acquired, completion = lifecycle_rows[1], lifecycle_rows[2]
        require(
            set(acquired)
            == {
                "schema",
                "event",
                "stem",
                "pid",
                "pgid",
                "acquired_monotonic_ns",
            },
            "acquired key set drifted",
        )
        uint(acquired.get("schema"), "acquired schema", 1)
        uint(attempt.get("launch_acquired_ns"), "attempt acquired time", positive=True)
        require(
            acquired.get("stem") == spec["stem"]
            and acquired.get("pid") == pid
            and acquired.get("pgid") == pid
            and acquired.get("acquired_monotonic_ns")
            == attempt.get("launch_acquired_ns"),
            "acquired binding drifted",
        )
        validate_group_disposition(attempt.get("group_disposition"), pid)
        for action in attempt.get("cleanup_actions", []):
            require(action.get("target_pgid") == pid, "cleanup pgid drifted")
    require(
        set(completion)
        == {
            "schema",
            "event",
            "stem",
            "pid",
            "pgid",
            "spawn_error",
            "returncode",
            "interrupted",
            "wait_errors",
            "poll_errors",
            "pipe_errors",
            "termination_errors",
            "output_overflow",
            "cleanup_actions",
            "cleanup_signal_sent",
            "group_disposition",
            "reaped",
            "attempted_monotonic_ns",
            "acquired_monotonic_ns",
            "completed_monotonic_ns",
            "residency_to_acquired_ns",
        },
        "completion key set drifted",
    )
    uint(completion.get("schema"), "completion schema", 1)
    require(completion.get("stem") == spec["stem"], "completion stem drifted")
    for key in (
        "pid",
        "pgid",
        "spawn_error",
        "returncode",
        "interrupted",
        "wait_errors",
        "poll_errors",
        "pipe_errors",
        "termination_errors",
        "output_overflow",
        "cleanup_actions",
        "cleanup_signal_sent",
        "group_disposition",
        "reaped",
    ):
        require_typed_equal(completion.get(key), attempt.get(key), f"completion {key}")
    require(
        completion.get("attempted_monotonic_ns") == attempt.get("launch_attempted_ns")
        and completion.get("acquired_monotonic_ns") == attempt.get("launch_acquired_ns")
        and completion.get("residency_to_acquired_ns")
        == attempt.get("residency_to_acquired_ns"),
        "completion timing binding drifted",
    )
    attempted = completion["attempted_monotonic_ns"]
    acquired_time = completion["acquired_monotonic_ns"]
    completed_time = completion["completed_monotonic_ns"]
    launch_time = uint(
        launch.get("monotonic_ns"), "launch lifecycle time", positive=True
    )
    uint(attempted, "completion attempted time", positive=True)
    uint(completed_time, "completion completed time", positive=True)
    require(
        launch_time <= attempted
        and (acquired_time is None or attempted <= acquired_time <= completed_time),
        "lifecycle timestamps are not ordered",
    )
    if acquired_time is not None:
        uint(acquired_time, "completion acquired time", positive=True)
        require(
            completion["residency_to_acquired_ns"]
            == acquired_time - attempt["residency_proved_ns"],
            "residency-to-acquired interval drifted",
        )
    validate_cleanup_evidence(attempt, events, acquired_time, completed_time)
    stderr_text = stderr_path.read_bytes().decode("utf-8", errors="replace")
    regression = is_rusage_self_regression(attempt.get("returncode"), stderr_text)
    require(
        attempt.get("rusage_self_counter_regression") is regression,
        "RUSAGE regression binding drifted",
    )
    require(
        attempt.get("rusage_regression_stderr_prefix")
        == (RUSAGE_PREFIX if regression else None),
        "RUSAGE prefix binding drifted",
    )
    require(
        attempt.get("parse_error") is None or type(attempt.get("parse_error")) is str,
        "attempt parse-error type drifted",
    )
    require(
        isinstance(attempt.get("validity_reasons"), list)
        and all(type(item) is str for item in attempt["validity_reasons"])
        and attempt["validity_reasons"] == sorted(set(attempt["validity_reasons"])),
        "attempt validity list drifted",
    )
    if attempt.get("returncode") == 0:
        parsed_resources = parsed_result = expected_parse_error = None
        try:
            parsed_resources = parse_time(stderr_text)
            parsed_result = validate_result(
                parse_json_bytes(stdout_path.read_bytes(), spec["stem"]),
                spec["arm"],
                manifest["build_identity"],
            )
        except ContractDefect as error:
            expected_parse_error = f"{type(error).__name__}:{error}"
        require_typed_equal(
            attempt.get("process_resources"),
            parsed_resources,
            "process resources",
        )
        require_typed_equal(
            attempt.get("result"), parsed_result, "stdout/result binding"
        )
        require(
            attempt.get("parse_error") == expected_parse_error,
            "persisted parse outcome drifted",
        )
    else:
        require(
            attempt.get("process_resources") is None
            and attempt.get("result") is None
            and attempt.get("parse_error") is None,
            "nonzero child retained parsed result evidence",
        )
    require(
        attempt.get("process_page_faults")
        == (
            attempt["process_resources"].get("page_faults")
            if isinstance(attempt.get("process_resources"), dict)
            else None
        ),
        "process page-fault binding drifted",
    )
    expected_validity = derive_attempt_validity(
        post,
        attempt["residency_proved_ns"],
        attempt,
        attempt.get("process_resources"),
        attempt.get("result"),
        events,
    )
    require_typed_equal(
        attempt.get("validity_reasons"), expected_validity, "attempt validity"
    )
    stop = classify_attempt_stop(
        attempt.get("returncode"),
        regression,
        attempt.get("parse_error"),
        attempt.get("validity_reasons"),
    )
    require_typed_equal(
        [attempt.get("stop_classification"), attempt.get("stop_reason")],
        list(stop),
        "attempt stop classification",
    )


def validate_signal_binding(decision, packet):
    cutoff_path = packet / "packet-signal-cutoff.json"
    log_path = packet / "packet-signal-log.json"
    cutoff = read_json_file(cutoff_path, "packet signal cutoff")
    log = read_json_file(log_path, "packet signal log")
    require(
        set(log)
        == {
            "schema",
            "logical_boundary_monotonic_ns",
            "logical_boundary_sequence",
            "post_block_snapshot_monotonic_ns",
            "post_block_snapshot_sequence",
            "events",
            "pending_signals",
            "attribution",
        },
        "packet signal log schema drifted",
    )
    require(
        set(cutoff)
        == set(log)
        | {
            "event",
            "cutoff_monotonic_ns",
            "authority_event_count",
            "signals_after_snapshot",
            "signal_log_sha256",
        },
        "packet signal cutoff schema drifted",
    )
    uint(log.get("schema"), "packet signal schema", 1)
    require(
        cutoff.get("event") == "packet-signal-cutoff"
        and cutoff.get("signals_after_snapshot")
        == "blocked-post-cutoff-outside-authority"
        and log.get("attribution")
        == "all snapshot events and pending signals are pre-cutoff",
        "packet signal constants drifted",
    )
    require(
        decision.get("signal_cutoff_sha256") == sha_file(cutoff_path)
        and decision.get("signal_log_sha256") == sha_file(log_path)
        and cutoff.get("signal_log_sha256") == sha_file(log_path),
        "decision signal digest binding drifted",
    )
    for key, value in log.items():
        require_typed_equal(cutoff.get(key), value, f"cutoff/log {key}")
    events, pending = log.get("events"), log.get("pending_signals")
    require(
        isinstance(events, list) and isinstance(pending, list), "signal log malformed"
    )
    for index, event in enumerate(events, 1):
        require(isinstance(event, dict), "packet signal event is not an object")
        uint(event.get("sequence"), f"packet signal sequence {index}", index, True)
        uint(event.get("signal"), f"packet signal {index}", positive=True)
        uint(event.get("monotonic_ns"), f"packet signal time {index}", positive=True)
    require(
        all(type(signum) is int and signum in OPERATOR_SIGNALS for signum in pending),
        "pending signal set drifted",
    )
    require(
        pending == sorted(set(pending)), "pending signals are not sorted and unique"
    )
    uint(
        cutoff.get("authority_event_count"),
        "cutoff authority event count",
        len(events) + len(pending),
    )
    require(
        cutoff.get("cutoff_monotonic_ns") == log.get("logical_boundary_monotonic_ns"),
        "signal cutoff accounting drifted",
    )
    boundary = uint(
        log.get("logical_boundary_monotonic_ns"),
        "signal logical boundary",
        positive=True,
    )
    post_snapshot = uint(
        log.get("post_block_snapshot_monotonic_ns"),
        "signal snapshot time",
        positive=True,
    )
    uint(log.get("logical_boundary_sequence"), "signal boundary sequence", len(events))
    uint(
        log.get("post_block_snapshot_sequence"), "signal snapshot sequence", len(events)
    )
    require(
        post_snapshot >= boundary
        and all(event["monotonic_ns"] <= boundary for event in events),
        "signal cutoff timestamps drifted",
    )
    return events


def validate_closure_signal_slices(closures, packet_events):
    for closure in closures:
        start = uint(closure.get("signal_start_sequence"), "closure signal start")
        end = uint(closure.get("signal_end_sequence"), "closure signal end")
        require(start <= end <= len(packet_events), "closure signal bounds drifted")
        require_typed_equal(
            closure.get("events"),
            packet_events[start:end],
            f"closure signal slice {closure.get('stem')}",
        )


def validate_packet_semantics(proposed_decision, packet=PACKET, work=WORK):
    packet_members = regular_member_snapshot(packet)
    work_members = regular_member_snapshot(work)
    decision = read_json_file(packet / "decision.json", "persisted decision")
    require_typed_equal(decision, proposed_decision, "persisted/proposed decision")
    manifest = read_json_file(packet / "manifest.json", "manifest")
    require_typed_equal(
        read_json_file(packet / "order.json", "order"),
        {"schema": 1, "pairs": [list(item) for item in PAIR_ORDERS]},
        "sealed order",
    )
    reservation_value = read_json_file(work / "reservation.json", "reservation")
    require(
        isinstance(reservation_value, dict)
        and set(reservation_value)
        == {"schema", "packet", "activity_boundary_monotonic_ns"},
        "reservation schema drifted",
    )
    uint(reservation_value["schema"], "reservation schema", 1)
    require(
        reservation_value["packet"] == str(packet.relative_to(ROOT)),
        "reservation packet binding drifted",
    )
    prior_activity = uint(
        reservation_value["activity_boundary_monotonic_ns"],
        "reservation activity boundary",
        positive=True,
    )
    has_attempts = "attempts.jsonl" in packet_members
    attempts = (
        read_jsonl_file(packet / "attempts.jsonl", "attempts") if has_attempts else []
    )
    lifecycle = (
        read_jsonl_file(packet / "lifecycle.jsonl", "lifecycle") if has_attempts else []
    )
    closures = (
        read_jsonl_file(
            packet / "attempt-signal-closures.jsonl", "attempt signal closures"
        )
        if has_attempts
        else []
    )
    require(len(attempts) == len(closures) <= 12, "attempt/closure cardinality drifted")
    specs = attempt_specs()
    expected_packet = {
        "decision.json",
        "decision.sha256",
        "manifest.json",
        "order.json",
        "packet-signal-cutoff.json",
        "packet-signal-log.json",
    }
    if attempts:
        expected_packet.update(
            {"attempts.jsonl", "lifecycle.jsonl", "attempt-signal-closures.jsonl"}
        )
    require_exact_names(set(packet_members), expected_packet, "pre-inventory packet")
    expected_work = {"reservation.json"}
    lifecycle_cursor = 0
    for index, (spec, attempt, closure) in enumerate(zip(specs, attempts, closures)):
        require(attempt.get("stem") == spec["stem"], "attempt prefix order drifted")
        event_count = 2 if attempt.get("pid") is None else 3
        lifecycle_rows = lifecycle[lifecycle_cursor : lifecycle_cursor + event_count]
        require(len(lifecycle_rows) == event_count, "lifecycle prefix is incomplete")
        validate_attempt_bundle(
            spec, attempt, closure, lifecycle_rows, packet, work, manifest
        )
        conditioning_value = read_json_file(
            work / f"{spec['stem']}.conditioning.json",
            f"{spec['stem']} timing conditioning",
        )
        post_value = read_json_file(
            work / f"{spec['stem']}.post.json",
            f"{spec['stem']} timing post",
        )
        require(
            conditioning_value["cooldown"]["prior_activity_monotonic_ns"]
            == prior_activity,
            "cross-attempt cooldown boundary drifted",
        )
        completion = lifecycle_rows[-1]
        activity_boundary = uint(
            attempt.get("activity_boundary_monotonic_ns"),
            "attempt activity boundary",
            positive=True,
        )
        host_times = [
            conditioning_value["host_before_conditioning"]["captured_monotonic_ns"],
            conditioning_value["host_before_launch"]["captured_monotonic_ns"],
            post_value["host_after_exit"]["captured_monotonic_ns"],
        ]
        vm_times = [
            conditioning_value["vm_before_conditioning"]["captured_monotonic_ns"],
            conditioning_value["vm_before_launch"]["captured_monotonic_ns"],
            post_value["vm_after_exit"]["captured_monotonic_ns"],
        ]
        acquired_point = completion["acquired_monotonic_ns"]
        if acquired_point is None:
            acquired_point = completion["attempted_monotonic_ns"]
        require(
            prior_activity
            <= conditioning_value["cooldown"]["started_monotonic_ns"]
            <= conditioning_value["cooldown"]["completed_monotonic_ns"]
            <= min(host_times[0], vm_times[0])
            and max(host_times[0], vm_times[0])
            <= conditioning_value["residency_proved_ns"]
            <= min(host_times[1], vm_times[1])
            and max(host_times[1], vm_times[1])
            <= lifecycle_rows[0]["monotonic_ns"]
            <= completion["attempted_monotonic_ns"]
            <= acquired_point
            <= completion["completed_monotonic_ns"]
            <= min(host_times[2], vm_times[2])
            and max(host_times[2], vm_times[2]) <= activity_boundary,
            "cross-attempt activity timeline drifted",
        )
        prior_activity = activity_boundary
        lifecycle_cursor += event_count
        expected_work.update(
            {
                f"{spec['stem']}.conditioning.json",
                f"{spec['stem']}.post.json",
                f"{spec['stem']}.stdout",
                f"{spec['stem']}.stderr",
            }
        )
    require(lifecycle_cursor == len(lifecycle), "orphan lifecycle evidence")
    extras = set(work_members) - expected_work
    if len(attempts) < len(specs) and extras:
        require(
            not attempts or attempts[-1].get("stop_classification") is None,
            "packet retained next conditioning after a stopping attempt",
        )
        next_conditioning = f"{specs[len(attempts)]['stem']}.conditioning.json"
        require(extras == {next_conditioning}, "incomplete work prefix drifted")
        value = read_json_file(work / next_conditioning, "incomplete conditioning")
        validate_conditioning_record(
            value,
            specs[len(attempts)]["stem"],
            False,
        )
        require(
            value["cooldown"]["prior_activity_monotonic_ns"] == prior_activity,
            "incomplete conditioning activity boundary drifted",
        )
        expected_work.add(next_conditioning)
    require_exact_names(set(work_members), expected_work, "work evidence")
    require(set(decision) == DECISION_KEYS, "decision key set drifted")
    uint(decision.get("schema"), "decision schema", 1)
    uint(
        decision.get("completed_attempts"), "decision completed attempts", len(attempts)
    )
    uint(decision.get("expected_attempts"), "decision expected attempts", 12)
    require(
        decision.get("authority") == "none"
        and decision.get("force_authorized") is False
        and decision.get("successor_authorization") == "none"
        and decision.get("scope") == "exact-host-asset-cache-warm-mechanism-only",
        "decision authority or scope drifted",
    )
    require(
        decision.get("source_commit") == manifest.get("source_commit")
        and decision.get("implementation_parent")
        == manifest.get("implementation_parent"),
        "decision/manifest source binding drifted",
    )
    final_identity = decision.get("final_identity")
    require(
        isinstance(final_identity, dict)
        and set(final_identity)
        == {
            "captured_monotonic_ns",
            "expected",
            "observed",
            "errors",
            "matches_manifest",
        },
        "final identity evidence missing",
    )
    uint(
        final_identity["captured_monotonic_ns"],
        "final identity timestamp",
        positive=True,
    )
    expected_identity = {
        "source": manifest["source"],
        "binary_bytes": manifest["binary_bytes"],
        "model_file_identity": manifest["model_file_identity"],
    }
    require_typed_equal(
        final_identity["expected"], expected_identity, "final expected identity"
    )
    observed_identity = final_identity["observed"]
    require(
        isinstance(observed_identity, dict)
        and set(observed_identity) == set(expected_identity),
        "final observed identity schema drifted",
    )
    errors = final_identity["errors"]
    require(
        isinstance(errors, list) and all(type(item) is str for item in errors),
        "final identity errors drifted",
    )
    recomputed_match = not errors and all(
        typed_equal(observed_identity[key], expected_identity[key])
        for key in expected_identity
    )
    require(
        final_identity["matches_manifest"] is recomputed_match,
        "final identity match summary drifted",
    )
    if not recomputed_match:
        require(
            decision.get("contract_error") is not None,
            "final identity defect was not classified",
        )
    packet_events = validate_signal_binding(decision, packet)
    validate_closure_signal_slices(closures, packet_events)
    require(
        (packet / "decision.sha256").read_bytes()
        == (sha_file(packet / "decision.json") + "\n").encode(),
        "decision digest file drifted",
    )
    scoreable = len(attempts) == 12 and all(
        row.get("result") is not None
        and not row.get("validity_reasons")
        and row.get("stop_classification") is None
        for row in attempts
    )
    expected_analysis = score_rows(attempts) if scoreable else None
    require_typed_equal(
        decision.get("analysis"), expected_analysis, "decision analysis"
    )
    regression = any(
        row.get("rusage_self_counter_regression") is True for row in attempts
    )
    require(
        decision.get("rusage_self_counter_regression") is regression,
        "decision RUSAGE regression drifted",
    )
    for key in ("contract_error", "invalid_error"):
        require(
            decision.get(key) is None or type(decision.get(key)) is str,
            f"decision {key} type drifted",
        )
    if any(
        row.get("stop_classification") == "implementation_or_contract_defect"
        for row in attempts
    ):
        require(decision.get("contract_error") is not None, "attempt defect was lost")
    if any(row.get("stop_classification") == "inconclusive" for row in attempts):
        require(
            decision.get("invalid_error") is not None, "attempt invalidity was lost"
        )
    require(
        all(row.get("stop_classification") is None for row in attempts[:-1]),
        "packet advanced after a stopping attempt",
    )
    cutoff_value = read_json_file(packet / "packet-signal-cutoff.json", "signal cutoff")
    if (
        cutoff_value.get("authority_event_count")
        and decision.get("contract_error") is None
    ):
        require(decision.get("invalid_error") is not None, "signal invalidity was lost")
    complete = scoreable and expected_analysis is not None
    expected_status = classify(
        bool(decision.get("contract_error")),
        bool(decision.get("invalid_error")),
        regression,
        complete,
        bool(expected_analysis and expected_analysis["qualifies"]),
    )
    expected_closure = (
        "GO-mechanism-floor"
        if expected_status == "GO-mechanism-floor"
        else "KILL"
        if expected_status == "mechanism-floor-miss"
        else expected_status
    )
    require(
        decision.get("status") == expected_status
        and decision.get("closure") == expected_closure,
        "decision status or closure drifted",
    )
    if len(attempts) == 12:
        require(
            len(packet_members) + len(work_members) == EXPECTED_INVENTORY_MEMBERS,
            "complete inventory cardinality drifted",
        )
    return attempts, packet_members, work_members


def seal(
    decision,
    packet=PACKET,
    work=WORK,
    _before_final_snapshot=None,
    _after_decision_write=None,
):
    write_json(packet / "decision.json", decision)
    write_exclusive(
        packet / "decision.sha256", (sha_file(packet / "decision.json") + "\n").encode()
    )
    if _after_decision_write is not None:
        _after_decision_write()
    attempts, packet_members, work_members = validate_packet_semantics(
        decision, packet, work
    )
    members = sorted(
        [value for value in packet_members.values()]
        + [value for value in work_members.values()],
        key=lambda value: str(value["path"].relative_to(ROOT)),
    )
    inventory = b"".join(
        f"{value['sha256']}  {value['path'].relative_to(ROOT)}\n".encode()
        for value in members
    )
    write_exclusive(packet / "artifact-inventory.sha256", inventory)
    require(
        (packet / "artifact-inventory.sha256").read_bytes() == inventory,
        "inventory write drifted",
    )
    complete = {
        "schema": 1,
        "decision_sha256": sha_file(packet / "decision.json"),
        "inventory_sha256": sha_file(packet / "artifact-inventory.sha256"),
        "inventory_members": len(members),
        "expected_complete_inventory_members": EXPECTED_INVENTORY_MEMBERS,
        "semantic_completed_attempts": len(attempts),
    }
    write_json(packet / "packet-complete.json", complete)
    write_exclusive(
        packet / "packet-complete.sha256",
        (sha_file(packet / "packet-complete.json") + "\n").encode(),
    )
    if _before_final_snapshot is not None:
        _before_final_snapshot()
    final_packet = regular_member_snapshot(packet)
    final_work = regular_member_snapshot(work)
    require_snapshot_unchanged(packet_members, final_packet, "packet")
    require_snapshot_unchanged(work_members, final_work, "work")
    require_exact_names(set(final_work), set(work_members), "final work")
    require_exact_names(
        set(final_packet),
        set(packet_members)
        | {
            "artifact-inventory.sha256",
            "packet-complete.json",
            "packet-complete.sha256",
        },
        "final packet",
    )
    require_typed_equal(
        read_json_file(packet / "packet-complete.json", "completion"),
        complete,
        "completion binding",
    )
    require(
        (packet / "packet-complete.sha256").read_bytes()
        == (sha_file(packet / "packet-complete.json") + "\n").encode(),
        "completion digest file drifted",
    )
    fsync_dir(packet)
    fsync_dir(work)
    fsync_dir(packet.parent)


def execute_attempt_plan(callback, rows, activity):
    for pair, order in enumerate(PAIR_ORDERS, 1):
        for position, arm in enumerate(order, 1):
            row, activity = callback(pair, position, arm, activity)
            rows.append(row)
    return activity


def capture_final_identity(manifest, env):
    expected = {
        "source": manifest["source"],
        "binary_bytes": manifest["binary_bytes"],
        "model_file_identity": manifest["model_file_identity"],
    }
    observed = {}
    errors = []
    operations = {
        "source": lambda: source_identity(env),
        "binary_bytes": lambda: descriptor_hash(
            BINARY, manifest["binary_bytes"]["descriptor_stamp"]["size_bytes"]
        ),
        "model_file_identity": lambda: file_identity(MODEL),
    }
    for key, operation in operations.items():
        try:
            observed[key] = operation()
        except Exception as error:
            observed[key] = None
            errors.append(f"{key}:{type(error).__name__}:{error}")
    mismatches = [
        key
        for key in expected
        if observed.get(key) is not None and observed.get(key) != expected[key]
    ]
    errors.extend(f"{key}:mismatch" for key in mismatches)
    return {
        "captured_monotonic_ns": time.monotonic_ns(),
        "expected": expected,
        "observed": observed,
        "errors": errors,
        "matches_manifest": not errors,
    }


def execute():
    manifest, env = preflight()
    controller = SignalController()
    controller.install()
    activity = reserve(manifest)
    rows = []
    contract_error = invalid_error = analysis = None
    regression = False
    try:
        activity = execute_attempt_plan(
            lambda pair, position, arm, prior: run_one(
                pair, position, arm, env, manifest, prior, controller
            ),
            rows,
            activity,
        )
        analysis = score_rows(rows)
    except ContractDefect as error:
        contract_error = f"{type(error).__name__}:{error}"
    except (Inconclusive, KeyboardInterrupt) as error:
        invalid_error = f"{type(error).__name__}:{error}"
        regression = "RUSAGE_SELF counter regression" in invalid_error
    final_identity = capture_final_identity(manifest, env)
    if not final_identity["matches_manifest"]:
        contract_error = contract_error or (
            "ContractDefect:final identity drifted: " + repr(final_identity["errors"])
        )
    cutoff = controller.final_cutoff(PACKET / "packet-signal-cutoff.json")
    if cutoff["authority_event_count"] and not contract_error:
        invalid_error = invalid_error or "Inconclusive:operator signal before cutoff"
    attempts = (
        read_jsonl_file(PACKET / "attempts.jsonl", "decision attempts")
        if (PACKET / "attempts.jsonl").is_file()
        else []
    )
    complete = len(rows) == 12 and analysis is not None
    status = classify(
        bool(contract_error),
        bool(invalid_error),
        regression,
        complete,
        bool(analysis and analysis["qualifies"]),
    )
    decision = {
        "schema": 1,
        "status": status,
        "authority": "none",
        "force_authorized": False,
        "successor_authorization": "none",
        "implementation_parent": IMPLEMENTATION_PARENT,
        "source_commit": manifest["source_commit"],
        "final_identity": final_identity,
        "completed_attempts": len(attempts),
        "expected_attempts": 12,
        "contract_error": contract_error,
        "invalid_error": invalid_error,
        "rusage_self_counter_regression": regression,
        "analysis": analysis,
        "closure": "GO-mechanism-floor"
        if status == "GO-mechanism-floor"
        else "KILL"
        if status == "mechanism-floor-miss"
        else status,
        "scope": "exact-host-asset-cache-warm-mechanism-only",
        "signal_cutoff_sha256": sha_file(PACKET / "packet-signal-cutoff.json"),
        "signal_log_sha256": cutoff["signal_log_sha256"],
    }
    seal(decision)
    print(json.dumps(decision, sort_keys=True))


def expect(exception, function, *args):
    try:
        function(*args)
    except exception:
        return
    raise RuntimeError(f"expected {exception.__name__}")


def fixture_result(ready, cpu, blit=False):
    value = {
        "timing": {"ready_us": ready},
        "rusage": {"total_cpu_us": cpu},
        "metal_allocated_bytes": {"before": 0, "ready": 1, "after_drop": 0},
    }
    if blit:
        value["blit_population"] = {"descriptive": True}
    return value


def scoring_fixture(d=112_000, cpu=1.10, rss=1.05, footprint=1.05):
    rows = []
    for pair, order in enumerate(PAIR_ORDERS, 1):
        for position, arm in enumerate(order, 1):
            rows.append(
                {
                    "pair": pair,
                    "position": position,
                    "arm": arm,
                    "validity_reasons": [],
                    "result": fixture_result(
                        1_000_000 if arm == "A" else 1_000_000 - d,
                        1_000_000 if arm == "A" else round(1_000_000 * cpu),
                        arm == "B",
                    ),
                    "process_resources": {
                        "maximum_resident_set_size": 1000
                        if arm == "A"
                        else round(1000 * rss),
                        "peak_memory_footprint": 1000
                        if arm == "A"
                        else round(1000 * footprint),
                    },
                }
            )
    return rows


def validator_fixture(arm):
    timing = {
        "ready_wall_ms": 5.004 if arm == "A" else 6.005,
        "ready_us": 5004 if arm == "A" else 6005,
        "allocation_wall_ms": 1.0,
        "allocation_us": 1000,
        "source_resolution_wall_ms": 1.0,
        "source_us": 1000,
        "source_resolution_us": 1000,
        "copy_wall_ms": 2.0,
        "copy_us": 2000,
        "binding_wall_ms": 1.0,
        "binding_us": 1000,
        "unattributed_wall_ms": 0.004 if arm == "A" else 0.005,
        "unattributed_us": 4 if arm == "A" else 5,
        "teardown_wall_ms": 0.001,
        "teardown_us": 1,
    }
    if arm == "B":
        timing.update(source_release_wall_ms=1.0, source_release_us=1000)
    schedule = None
    if arm == "A":
        schedule = {
            "algorithm": "minimax-contiguous-v1",
            "workers": 4,
            "cuts": list(SCHEDULE["cuts"]),
            "task_counts": list(SCHEDULE["task_counts"]),
            "worker_bytes": list(SCHEDULE["worker_bytes"]),
            "max_to_min": 1.1,
            "max_to_ideal": 1.1,
            "partitions": [],
        }
        cursor = 0
        for count, size in zip(SCHEDULE["task_counts"], SCHEDULE["worker_bytes"]):
            endpoint = {
                "request_index": 0,
                "shard_idx": 0,
                "source_offset": 0,
                "n_bytes": 1,
                "name": "fixture",
            }
            schedule["partitions"].append(
                {
                    "start": cursor,
                    "end": cursor + count,
                    "task_count": count,
                    "bytes": size,
                    "first_shard": 0,
                    "first_source_offset": 0,
                    "last_shard": 0,
                    "last_source_offset": 0,
                    "first": dict(endpoint),
                    "last": dict(endpoint),
                }
            )
            cursor += count
    value = {key: None for key in TOP_KEYS}
    value.update(
        schema_version=2,
        arm="parallel-pread" if arm == "A" else "transient-mmap-blit",
        profile=PROFILE,
        model=str(MODEL),
        descriptor_layout_digest=DESCRIPTOR,
        inventory_digest=INVENTORY,
        request_count=COUNT,
        resource_count=COUNT,
        binding_count=COUNT,
        logical_copy_bytes=COPY_BYTES,
        physical_copy_bytes=COPY_BYTES,
        architecture="qwen35moe",
        architecture_tuple=dict(ARCHITECTURE_TUPLE),
        tied_embeddings=False,
        mtp_present=False,
        shard_mapped_lengths=[MODEL_SIZE],
        native_quant_embedding=True,
        native_quant_embedding_supported=True,
        native_quant_embedding_selection="production-auto-promoted",
        page_size=PAGE_SIZE,
        required_alignment=32,
        max_buffer_length=77_309_411_328,
        device_name="Apple M4 Max",
        unified_memory=True,
        resource_modes=dict(RESOURCE_MODES),
        parallel_copy_schedule=schedule,
        timing=timing,
        throughput={"ready_gbps_decimal": 1.0, "copy_gbps_decimal": 1.0},
        rusage={
            "user_cpu_us": 10,
            "system_cpu_us": 10,
            "total_cpu_us": 20,
            "timer_major_faults": 0,
            "timer_minor_faults": 0,
            "cpu_per_wall": 20 / timing["ready_us"],
        },
        proc_rusage_v4={key: 0 for key in PROC_RUSAGE_KEYS},
        metal_allocated_bytes={"before": 0, "ready": 10, "after_drop": 0},
        correctness={
            "passed": True,
            "payload_bytes_checked": COPY_BYTES,
            "entries_checked": COUNT,
        },
        worker_count=4 if arm == "A" else 0,
        build_identity={"fixture": True},
    )
    if arm == "B":
        value["blit_population"] = {
            "schema_version": 1,
            "order": {
                "algorithm": "shard-offset-request-v1",
                "count": 733,
                "first_request_index": 2,
                "last_request_index": 721,
            },
            "sources": {
                "window_count": 1,
                "window_bytes": 22_123_544_576,
                "window_gap_bytes": 13_824,
                "fallback_count": 1,
                "fallback_bytes": 8_192,
                "cpu_staging_copy_bytes": 8_192,
            },
            "copies": {
                "window_count": 732,
                "window_bytes": 22_123_530_752,
                "total_count": 733,
                "total_bytes": COPY_BYTES,
            },
            "command": {
                "buffer_count": 1,
                "encoder_count": 1,
                "commit_count": 1,
                "wait_count": 1,
                "error_count": 0,
                "status": "completed",
                "status_code": 4,
                "retained_references": True,
                "gpu_start_time": None,
                "gpu_end_time": None,
                "gpu_wall_ms": None,
            },
            "release": {
                "window_deallocator_calls": 1,
                "window_deallocator_mismatches": 0,
                "source_buffers_alive": 0,
                "allocated_after_destinations": 30,
                "allocated_with_sources": 10,
                "allocated_after_source_release": 20,
            },
        }
    return value


def seal_host_fixture(timestamp, valid=True):
    return {
        "thermal": (
            "No thermal warning level has been recorded\n"
            "No performance warning level has been recorded\n"
        ),
        "battery": "Now drawing from 'AC Power'\n",
        "memory_pressure": "System-wide memory free percentage: 90%\n",
        "memory_available_percent": 90,
        "competing_processes": [] if valid else [{"pid": 7, "command": "fixture"}],
        "errors": [],
        "valid": valid,
        "captured_monotonic_ns": timestamp,
    }


def seal_vm_fixture(timestamp):
    return {
        "pageouts": 0,
        "compressions": 0,
        "swapouts": 0,
        "compressor_stored_pages": 0,
        "compressor_occupied_pages": 0,
        "swap_used_bytes": 0,
        "errors": [],
        "captured_monotonic_ns": timestamp,
    }


def seal_residency_fixture():
    return {
        "page_size": PAGE_SIZE,
        "total_pages": MODEL_PAGES,
        "resident_pages": MODEL_PAGES,
        "all_pages_resident": True,
        "file_identity": {"fixture": True},
    }


def seal_conditioning_fixture(stem, launchable=True, prior_activity=1000):
    completed_cooldown = prior_activity + COOLDOWN_NS
    base = completed_cooldown + 100
    host_before = seal_host_fixture(base + 1, launchable)
    vm_before = seal_vm_fixture(base + 2)
    host_launch = seal_host_fixture(base + 5) if launchable else None
    vm_launch = seal_vm_fixture(base + 6) if launchable else None
    return {
        "schema": 1,
        "stem": stem,
        "signal_start_sequence": 0,
        "cooldown": {
            "prior_activity_monotonic_ns": prior_activity,
            "required_interval_ns": COOLDOWN_NS,
            "eligible_monotonic_ns": completed_cooldown,
            "started_monotonic_ns": prior_activity,
            "requested_sleep_ns": COOLDOWN_NS,
            "completed_monotonic_ns": completed_cooldown,
            "observed_interval_ns": COOLDOWN_NS,
        },
        "host_before_conditioning": host_before,
        "vm_before_conditioning": vm_before,
        "conditioning": {
            "bytes_read": MODEL_SIZE,
            "wall_ns": 1,
            "buffer_bytes": BUFFER_SIZE,
        }
        if launchable
        else None,
        "residency_before": seal_residency_fixture() if launchable else None,
        "host_before_launch": host_launch,
        "vm_before_launch": vm_launch,
        "conditioning_interval": vm_interval("conditioning", vm_before, vm_launch)
        if launchable
        else None,
        "residency_proved_ns": base + 4 if launchable else None,
        "operation_errors": [] if launchable else ["conditioning:fixture"],
    }


def seal_post_fixture(stem, conditioning):
    after_vm = seal_vm_fixture(conditioning["residency_proved_ns"] + 10)
    return {
        "schema": 1,
        "stem": stem,
        "host_after_exit": seal_host_fixture(conditioning["residency_proved_ns"] + 11),
        "vm_after_exit": after_vm,
        "residency_after": seal_residency_fixture(),
        "child_interval": vm_interval(
            "child", conditioning["vm_before_launch"], after_vm
        ),
        "operation_errors": [],
    }


def make_seal_fixture(base, with_attempt=True, conditioning_only=False):
    packet, work = base / "packet", base / "work"
    packet.mkdir()
    work.mkdir()
    manifest = {
        "schema": 1,
        "source_commit": "fixture-source",
        "implementation_parent": IMPLEMENTATION_PARENT,
        "source": {"fixture": "source"},
        "build_identity": {"fixture": True},
        "binary_bytes": {"fixture": "binary"},
        "model_file_identity": {"fixture": True},
    }
    write_json(packet / "manifest.json", manifest)
    write_json(
        packet / "order.json",
        {"schema": 1, "pairs": [list(value) for value in PAIR_ORDERS]},
    )
    reservation_activity = 1000
    write_json(
        work / "reservation.json",
        {
            "schema": 1,
            "packet": str(packet.relative_to(ROOT)),
            "activity_boundary_monotonic_ns": reservation_activity,
        },
    )
    completed = 0
    if with_attempt:
        spec = attempt_specs()[0]
        stem = spec["stem"]
        conditioning_value = seal_conditioning_fixture(
            stem, prior_activity=reservation_activity
        )
        proved = conditioning_value["residency_proved_ns"]
        launch_time, attempted_time, acquired_time, completed_time = (
            proved + 3,
            proved + 4,
            proved + 5,
            proved + 6,
        )
        write_json(work / f"{stem}.conditioning.json", conditioning_value)
        post_value = seal_post_fixture(stem, conditioning_value)
        write_json(work / f"{stem}.post.json", post_value)
        activity_boundary = (
            max(
                post_value["host_after_exit"]["captured_monotonic_ns"],
                post_value["vm_after_exit"]["captured_monotonic_ns"],
            )
            + 1
        )
        write_exclusive(work / f"{stem}.stdout", b"")
        write_exclusive(work / f"{stem}.stderr", b"fixture error\n")
        disposition_value = {
            "pgid": 4242,
            "samples": [
                {
                    "captured_monotonic_ns": completed_time + 1,
                    "members": [],
                    "error": None,
                }
            ],
            "no_live_group_observed": True,
        }
        process_fields = {
            "pid": 4242,
            "pgid": 4242,
            "spawn_error": None,
            "returncode": 1,
            "interrupted": False,
            "wait_errors": [],
            "poll_errors": [],
            "pipe_errors": [],
            "termination_errors": [],
            "output_overflow": False,
            "cleanup_actions": [],
            "cleanup_signal_sent": False,
            "group_disposition": disposition_value,
            "reaped": True,
        }
        attempt = {
            "schema": 1,
            **spec,
            **process_fields,
            "signal_start_sequence": 0,
            "residency_proved_ns": proved,
            "launch_attempted_ns": attempted_time,
            "launch_acquired_ns": acquired_time,
            "residency_to_acquired_ns": acquired_time - proved,
            "process_resources": None,
            "process_page_faults": None,
            "rusage_self_counter_regression": False,
            "rusage_regression_stderr_prefix": None,
            "parse_error": None,
            "validity_reasons": [],
            "stop_classification": "implementation_or_contract_defect",
            "stop_reason": "quiet nonzero exit 1",
            "result": None,
            "stdout_sha256": sha_file(work / f"{stem}.stdout"),
            "stderr_sha256": sha_file(work / f"{stem}.stderr"),
            "conditioning_sha256": sha_file(work / f"{stem}.conditioning.json"),
            "post_sha256": sha_file(work / f"{stem}.post.json"),
            "activity_boundary_monotonic_ns": activity_boundary,
        }
        lifecycle = [
            {
                "schema": 1,
                "event": "launch",
                "stem": stem,
                "pair": spec["pair"],
                "position": spec["position"],
                "arm": spec["arm"],
                "command": spec["command"],
                "signal_start_sequence": 0,
                "conditioning_sha256": attempt["conditioning_sha256"],
                "residency_proved_ns": proved,
                "monotonic_ns": launch_time,
            },
            {
                "schema": 1,
                "event": "acquired",
                "stem": stem,
                "pid": 4242,
                "pgid": 4242,
                "acquired_monotonic_ns": acquired_time,
            },
            {
                "schema": 1,
                "event": "completion",
                "stem": stem,
                **process_fields,
                "attempted_monotonic_ns": attempted_time,
                "acquired_monotonic_ns": acquired_time,
                "completed_monotonic_ns": completed_time,
                "residency_to_acquired_ns": acquired_time - proved,
            },
        ]
        for value in lifecycle:
            append_jsonl(packet / "lifecycle.jsonl", value)
        append_jsonl(packet / "attempts.jsonl", attempt)
        append_jsonl(
            packet / "attempt-signal-closures.jsonl",
            {
                "schema": 1,
                "stem": stem,
                "signal_start_sequence": 0,
                "signal_end_sequence": 0,
                "events": [],
                "invalid": False,
                "attempt_sha256": sha_bytes(json_bytes(attempt)),
            },
        )
        completed = 1
    elif conditioning_only:
        stem = attempt_specs()[0]["stem"]
        write_json(
            work / f"{stem}.conditioning.json",
            seal_conditioning_fixture(
                stem,
                False,
                reservation_activity,
            ),
        )
    signal_log = {
        "schema": 1,
        "logical_boundary_monotonic_ns": 10,
        "logical_boundary_sequence": 0,
        "post_block_snapshot_monotonic_ns": 11,
        "post_block_snapshot_sequence": 0,
        "events": [],
        "pending_signals": [],
        "attribution": "all snapshot events and pending signals are pre-cutoff",
    }
    write_json(packet / "packet-signal-log.json", signal_log)
    cutoff = {
        "schema": 1,
        "event": "packet-signal-cutoff",
        "cutoff_monotonic_ns": 10,
        **signal_log,
        "authority_event_count": 0,
        "signals_after_snapshot": "blocked-post-cutoff-outside-authority",
        "signal_log_sha256": sha_file(packet / "packet-signal-log.json"),
    }
    write_json(packet / "packet-signal-cutoff.json", cutoff)
    decision = {
        "schema": 1,
        "status": "implementation_or_contract_defect" if completed else "inconclusive",
        "authority": "none",
        "force_authorized": False,
        "successor_authorization": "none",
        "implementation_parent": IMPLEMENTATION_PARENT,
        "source_commit": manifest["source_commit"],
        "final_identity": {
            "captured_monotonic_ns": 12,
            "expected": {
                "source": manifest["source"],
                "binary_bytes": manifest["binary_bytes"],
                "model_file_identity": manifest["model_file_identity"],
            },
            "observed": {
                "source": manifest["source"],
                "binary_bytes": manifest["binary_bytes"],
                "model_file_identity": manifest["model_file_identity"],
            },
            "errors": [],
            "matches_manifest": True,
        },
        "completed_attempts": completed,
        "expected_attempts": 12,
        "contract_error": "fixture" if completed else None,
        "invalid_error": None if completed else "fixture",
        "rusage_self_counter_regression": False,
        "analysis": None,
        "closure": "implementation_or_contract_defect" if completed else "inconclusive",
        "scope": "exact-host-asset-cache-warm-mechanism-only",
        "signal_cutoff_sha256": sha_file(packet / "packet-signal-cutoff.json"),
        "signal_log_sha256": sha_file(packet / "packet-signal-log.json"),
    }
    return packet, work, decision


def append_valid_seal_attempt(packet, work, spec, index, prior_activity):
    stem = spec["stem"]
    conditioning_value = seal_conditioning_fixture(
        stem,
        prior_activity=prior_activity,
    )
    proved = conditioning_value["residency_proved_ns"]
    launch_time = proved + 3
    attempted_time = launch_time + 1
    acquired_time = attempted_time + 1
    completed_time = acquired_time + 1
    write_json(work / f"{stem}.conditioning.json", conditioning_value)
    post_value = seal_post_fixture(stem, conditioning_value)
    write_json(work / f"{stem}.post.json", post_value)
    activity_boundary = (
        max(
            post_value["host_after_exit"]["captured_monotonic_ns"],
            post_value["vm_after_exit"]["captured_monotonic_ns"],
        )
        + 1
    )
    result = validator_fixture(spec["arm"])
    write_exclusive(work / f"{stem}.stdout", json_bytes(result))
    resource_values = {label: 0 for label in TIME_LABELS}
    resource_values["maximum resident set size"] = 1000
    resource_values["peak memory footprint"] = 1000
    stderr = "  1.00 real  0.20 user  0.30 sys\n" + "".join(
        f"  {resource_values[label]}  {label}\n" for label in TIME_LABELS
    )
    write_exclusive(work / f"{stem}.stderr", stderr.encode())
    resources = parse_time(stderr)
    pid = 5000 + index
    disposition_value = {
        "pgid": pid,
        "samples": [
            {
                "captured_monotonic_ns": completed_time + 1,
                "members": [],
                "error": None,
            }
        ],
        "no_live_group_observed": True,
    }
    process_fields = {
        "pid": pid,
        "pgid": pid,
        "spawn_error": None,
        "returncode": 0,
        "interrupted": False,
        "wait_errors": [],
        "poll_errors": [],
        "pipe_errors": [],
        "termination_errors": [],
        "output_overflow": False,
        "cleanup_actions": [],
        "cleanup_signal_sent": False,
        "group_disposition": disposition_value,
        "reaped": True,
    }
    attempt = {
        "schema": 1,
        **spec,
        **process_fields,
        "signal_start_sequence": 0,
        "residency_proved_ns": proved,
        "launch_attempted_ns": attempted_time,
        "launch_acquired_ns": acquired_time,
        "residency_to_acquired_ns": acquired_time - proved,
        "process_resources": resources,
        "process_page_faults": resources["page_faults"],
        "rusage_self_counter_regression": False,
        "rusage_regression_stderr_prefix": None,
        "parse_error": None,
        "validity_reasons": [],
        "stop_classification": None,
        "stop_reason": None,
        "result": result,
        "stdout_sha256": sha_file(work / f"{stem}.stdout"),
        "stderr_sha256": sha_file(work / f"{stem}.stderr"),
        "conditioning_sha256": sha_file(work / f"{stem}.conditioning.json"),
        "post_sha256": sha_file(work / f"{stem}.post.json"),
        "activity_boundary_monotonic_ns": activity_boundary,
    }
    lifecycle = [
        {
            "schema": 1,
            "event": "launch",
            "stem": stem,
            "pair": spec["pair"],
            "position": spec["position"],
            "arm": spec["arm"],
            "command": spec["command"],
            "signal_start_sequence": 0,
            "conditioning_sha256": attempt["conditioning_sha256"],
            "residency_proved_ns": proved,
            "monotonic_ns": launch_time,
        },
        {
            "schema": 1,
            "event": "acquired",
            "stem": stem,
            "pid": pid,
            "pgid": pid,
            "acquired_monotonic_ns": acquired_time,
        },
        {
            "schema": 1,
            "event": "completion",
            "stem": stem,
            **process_fields,
            "attempted_monotonic_ns": attempted_time,
            "acquired_monotonic_ns": acquired_time,
            "completed_monotonic_ns": completed_time,
            "residency_to_acquired_ns": acquired_time - proved,
        },
    ]
    for value in lifecycle:
        append_jsonl(packet / "lifecycle.jsonl", value)
    append_jsonl(packet / "attempts.jsonl", attempt)
    append_jsonl(
        packet / "attempt-signal-closures.jsonl",
        {
            "schema": 1,
            "stem": stem,
            "signal_start_sequence": 0,
            "signal_end_sequence": 0,
            "events": [],
            "invalid": False,
            "attempt_sha256": sha_bytes(json_bytes(attempt)),
        },
    )
    return attempt


def make_complete_seal_fixture(base):
    packet, work, decision = make_seal_fixture(base, with_attempt=False)
    prior_activity = read_json_file(work / "reservation.json", "fixture reservation")[
        "activity_boundary_monotonic_ns"
    ]
    attempts = []
    for index, spec in enumerate(attempt_specs(), 1):
        attempt = append_valid_seal_attempt(packet, work, spec, index, prior_activity)
        attempts.append(attempt)
        prior_activity = attempt["activity_boundary_monotonic_ns"]
    analysis = score_rows(attempts)
    status = "GO-mechanism-floor" if analysis["qualifies"] else "mechanism-floor-miss"
    decision.update(
        status=status,
        completed_attempts=12,
        contract_error=None,
        invalid_error=None,
        analysis=analysis,
        closure="GO-mechanism-floor" if analysis["qualifies"] else "KILL",
    )
    fixture_digest = sha_bytes(
        json.dumps(
            attempts[0]["result"]["parallel_copy_schedule"],
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    )
    return packet, work, decision, fixture_digest


def run_process_self_tests(controller=None):
    controller_start = controller.sequence() if controller is not None else 0

    def run_child(command, env, **kwargs):
        start = controller.sequence() if controller is not None else 0
        return bounded_child(command, env, controller, start, **kwargs)

    expect(RuntimeError, process_group_rows, "1 2\nmalformed\n", 2)
    spawn = run_child(["/definitely/missing/v0643"], {"PATH": "/usr/bin:/bin"})
    require(spawn["pid"] is None, "spawn lifecycle")
    success = run_child([sys.executable, "-c", "print('ok')"], os.environ.copy())
    require(
        success["returncode"] == 0
        and success["reaped"]
        and success["group_disposition"]["no_live_group_observed"],
        "reap lifecycle",
    )
    wait_fault = run_child(
        [sys.executable, "-c", "print('wait')"],
        os.environ.copy(),
        _faults={"wait_once", "join_0"},
    )
    require(
        any(item.startswith("wait:") for item in wait_fault["errors"])
        and any(item.startswith("join0:") for item in wait_fault["errors"]),
        "wait/join fault persistence",
    )
    drain_fault = run_child(
        [sys.executable, "-c", "print('drain')"],
        os.environ.copy(),
        _faults={"read_0"},
    )
    require(
        drain_fault["cleanup_signal_sent"]
        and len(drain_fault["cleanup_actions"]) == 1
        and drain_fault["group_disposition"]["no_live_group_observed"],
        "drain exact-group cleanup",
    )
    ownership_diagnostics = []

    def lose_exact_wait_ownership():
        run_child(
            [sys.executable, "-c", "import time;time.sleep(0.2)"],
            os.environ.copy(),
            _faults={"ownership_lost"},
            _ownership_loss_observer=ownership_diagnostics.append,
        )

    expect(OwnershipLost, lose_exact_wait_ownership)
    require(
        len(ownership_diagnostics) == 1
        and ownership_diagnostics[0]["signal_sent"] is False
        and not ownership_diagnostics[0]["threads_alive_after_bounded_join"],
        "exact-wait ownership-loss cleanup drifted",
    )
    try:
        os.waitpid(ownership_diagnostics[0]["pid"], 0)
    except ChildProcessError:
        pass
    descendant_code = (
        "import subprocess,sys,time;"
        "subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)']);"
        "print('ready',flush=True);time.sleep(30)"
    )
    with SignalController(True) as signal_controller:
        signal_start = signal_controller.sequence()

        def send_sigterm():
            time.sleep(0.2)
            os.kill(os.getpid(), signal.SIGTERM)

        sender = threading.Thread(target=send_sigterm)
        sender.start()
        terminated_gate = bounded_child(
            [sys.executable, "-c", descendant_code],
            os.environ.copy(),
            signal_controller,
            signal_start,
        )
        sender.join()
        require(
            signal_controller.since(signal_start)[0]["signal"] == signal.SIGTERM
            and terminated_gate["cleanup_signal_sent"]
            and len(terminated_gate["cleanup_actions"]) == 1
            and terminated_gate["group_disposition"]["no_live_group_observed"],
            "preflight SIGTERM did not clean the exact process group",
        )
    close_fault = run_child(
        [sys.executable, "-c", "print('close')"],
        os.environ.copy(),
        _faults={"close_0"},
    )
    require(
        any(item.startswith("close0:") for item in close_fault["errors"]),
        "close fault persistence",
    )
    overflow = run_child(
        [sys.executable, "-c", f"print('x' * {MAX_OUTPUT + 1})"],
        os.environ.copy(),
    )
    require(overflow["output_overflow"], "bounded output overflow")
    authentication_diagnostics = []

    def lose_initial_group_authentication():
        run_child(
            [sys.executable, "-c", "import time;time.sleep(0.2)"],
            os.environ.copy(),
            _pgid_getter=lambda pid: pid + 1,
            _ownership_loss_observer=authentication_diagnostics.append,
        )

    expect(OwnershipLost, lose_initial_group_authentication)
    require(
        len(authentication_diagnostics) == 1
        and authentication_diagnostics[0]["signal_sent"] is False
        and authentication_diagnostics[0]["leader_reaped"] is True,
        "initial process-group authentication loss did not reap the leader",
    )
    cleanup = []

    def reject_acquired(_value):
        raise DurabilityError("injected acquired write failure")

    def lose_acquired_record():
        run_child(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            os.environ.copy(),
            on_acquired=reject_acquired,
            _pgid_getter=os.getpgid,
            _cleanup_observer=cleanup.append,
        )

    expect(DurabilityError, lose_acquired_record)
    require(
        cleanup and cleanup[0]["disposition"]["no_live_group_observed"],
        "acquired durability cleanup",
    )
    if controller is not None:
        controller.quiet(controller_start, "process self-tests")
    return {
        "schema": 1,
        "tests": 10,
        "signal_start_sequence": controller_start,
        "signal_end_sequence": controller.sequence() if controller is not None else 0,
    }


def run_self_tests(include_process_tests=True):
    initial_handlers = {item: signal.getsignal(item) for item in OPERATOR_SIGNALS}
    initial_mask = signal.pthread_sigmask(signal.SIG_BLOCK, [])
    require(
        PAIR_ORDERS
        == (("A", "B"), ("B", "A"), ("B", "A"), ("A", "B"), ("A", "B"), ("B", "A")),
        "order drifted",
    )
    packet_names = complete_preinventory_packet_names()
    work_names = complete_work_names()
    require(
        len(packet_names) == 9
        and len(work_names) == 49
        and len(packet_names) + len(work_names) == EXPECTED_INVENTORY_MEMBERS,
        "exact complete inventory semantics",
    )
    require_exact_names(set(packet_names), packet_names, "inventory fixture")
    expect(
        ContractDefect,
        require_exact_names,
        packet_names | {"unexpected"},
        packet_names,
        "inventory mutation",
    )
    require(
        child_command("A")[-6:]
        == ["--arm", "parallel-pread", "--workers", "4", "--output", "json"]
        and child_command("B")[-6:]
        == ["--arm", "transient-mmap-blit", "--workers", "4", "--output", "json"],
        "commands drifted",
    )
    for malformed in (
        b"{",
        b'{"x":1,"x":2}',
        b'{"x":NaN}',
        b'{"x":1e999}',
        b'{"x":1} {"y":2}',
    ):
        expect(ContractDefect, parse_json_bytes, malformed, "fixture")
    require(
        parse_json_bytes(b'{"x":1.25e2}\n', "fixture") == {"x": 125.0},
        "strict parser acceptance",
    )
    require(
        parse_cargo_summary(
            "test result: ok. 13 passed; 0 failed; 0 ignored; 0 measured; "
            "2 filtered out; finished in 0.01s\n"
        )["passed"]
        == 13,
        "Cargo parser",
    )
    expect(
        ContractDefect,
        parse_cargo_summary,
        "test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; "
        "15 filtered out; finished in 0.01s\n",
    )
    time_fixture = "  1.00 real  0.20 user  0.30 sys\n" + "".join(
        f"  0  {label}\n" for label in TIME_LABELS
    )
    resources = parse_time(time_fixture)
    resources["page_faults"] = 89
    require(resource_reasons(resources) == [], "page faults not advisory")
    for key in ("block_input_operations", "swaps"):
        changed = dict(resources)
        changed[key] = 1
        require(resource_reasons(changed) == [f"{key}=1"], "fatal resource gate")
    a_fixture, b_fixture = validator_fixture("A"), validator_fixture("B")
    frozen_digest = SCHEDULE["digest"]
    SCHEDULE["digest"] = sha_bytes(
        json.dumps(
            a_fixture["parallel_copy_schedule"], sort_keys=True, separators=(",", ":")
        ).encode("ascii")
    )
    try:
        validate_result(a_fixture, "A", {"fixture": True})
        validate_result(b_fixture, "B", {"fixture": True})
        a_mutation = dict(a_fixture)
        a_mutation["blit_population"] = b_fixture["blit_population"]
        expect(ContractDefect, validate_result, a_mutation, "A", {"fixture": True})
        b_mutation = dict(b_fixture)
        b_mutation["worker_count"] = 4
        expect(ContractDefect, validate_result, b_mutation, "B", {"fixture": True})
        timing_mutation = dict(a_fixture)
        timing_mutation["timing"] = dict(a_fixture["timing"])
        timing_mutation["timing"]["ready_us"] += 10
        expect(ContractDefect, validate_result, timing_mutation, "A", {"fixture": True})
        fault_mutation = dict(a_fixture)
        fault_mutation["rusage"] = dict(a_fixture["rusage"])
        fault_mutation["rusage"]["timer_major_faults"] = 1
        validate_result(fault_mutation, "A", {"fixture": True})
        require(
            result_validity_reasons(fault_mutation) == ["timer_major_faults=1"],
            "timer major fault must be inconclusive validity",
        )
        proc_mutation = dict(a_fixture)
        proc_mutation["proc_rusage_v4"] = dict(a_fixture["proc_rusage_v4"])
        proc_mutation["proc_rusage_v4"]["unexpected"] = 0
        expect(ContractDefect, validate_result, proc_mutation, "A", {"fixture": True})
        cpu_mutation = dict(a_fixture)
        cpu_mutation["rusage"] = dict(a_fixture["rusage"])
        cpu_mutation["rusage"]["cpu_per_wall"] += 0.1
        expect(ContractDefect, validate_result, cpu_mutation, "A", {"fixture": True})
        typed_arch = validator_fixture("A")
        typed_arch["architecture_tuple"]["intermediate_size"] = False
        expect(ContractDefect, validate_result, typed_arch, "A", {"fixture": True})
        typed_command = validator_fixture("B")
        typed_command["blit_population"]["command"]["buffer_count"] = True
        expect(ContractDefect, validate_result, typed_command, "B", {"fixture": True})
        release_failure = validator_fixture("B")
        release_failure["blit_population"]["release"]["window_deallocator_calls"] = True
        expect(ContractDefect, validate_result, release_failure, "B", {"fixture": True})
        live_source = validator_fixture("B")
        live_source["blit_population"]["release"]["source_buffers_alive"] = 1
        expect(ContractDefect, validate_result, live_source, "B", {"fixture": True})
        gpu_fixture = validator_fixture("B")
        gpu_fixture["blit_population"]["command"].update(
            gpu_start_time=10.0, gpu_end_time=10.5, gpu_wall_ms=500.0
        )
        validate_result(gpu_fixture, "B", {"fixture": True})
        gpu_fixture["blit_population"]["command"]["gpu_wall_ms"] = 499.0
        expect(ContractDefect, validate_result, gpu_fixture, "B", {"fixture": True})
    finally:
        SCHEDULE["digest"] = frozen_digest
    scored = score_rows(scoring_fixture())
    require(scored["qualifies"], "inclusive boundaries")
    for kwargs in (
        {"d": 111_999},
        {"cpu": 1.101},
        {"rss": 1.051},
        {"footprint": 1.051},
    ):
        require(
            not score_rows(scoring_fixture(**kwargs))["qualifies"],
            "exclusive failing boundary",
        )
    expect(ContractDefect, score_rows, scoring_fixture()[:-1])
    invalid = scoring_fixture()
    invalid[0]["validity_reasons"] = ["fault"]
    expect(ContractDefect, score_rows, invalid)
    require(
        classify(True, True, True, False, True) == "implementation_or_contract_defect"
        and classify(False, True, False, True, True) == "inconclusive"
        and classify(False, False, True, True, True) == "inconclusive"
        and classify(False, False, False, True, True) == "GO-mechanism-floor"
        and classify(False, False, False, True, False) == "mechanism-floor-miss",
        "decision precedence",
    )
    require(
        RUSAGE_PREFIX == "Error: getrusage counter regressed:"
        and is_rusage_self_regression(1, RUSAGE_PREFIX + " fixture")
        and not is_rusage_self_regression(0, RUSAGE_PREFIX + " fixture")
        and not is_rusage_self_regression(1, "prefix " + RUSAGE_PREFIX)
        and classify(regression=True) == "inconclusive",
        "regression classification",
    )
    require(
        classify_attempt_stop(1, False, None, [])
        == ("implementation_or_contract_defect", "quiet nonzero exit 1")
        and classify_attempt_stop(-9, False, None, ["child_signal=9"])[0]
        == "inconclusive"
        and classify_attempt_stop(1, False, None, ["host_invalid"])[0] == "inconclusive"
        and classify_attempt_stop(1, True, None, ["host_invalid"])
        == ("inconclusive", "RUSAGE_SELF counter regression"),
        "attempt stop precedence",
    )
    plan_calls = []
    plan_rows = []

    def stop_on_second(pair, position, arm, activity):
        plan_calls.append((pair, position, arm))
        if len(plan_calls) == 2:
            raise Inconclusive("injected first invalid attempt")
        return {"pair": pair, "position": position, "arm": arm}, activity + 1

    expect(Inconclusive, execute_attempt_plan, stop_on_second, plan_rows, 0)
    require(
        plan_calls == [(1, 1, "A"), (1, 2, "B")] and len(plan_rows) == 1,
        "attempt plan retried or advanced after first invalidity",
    )
    if include_process_tests:
        run_process_self_tests()
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "seal.json"
        write_json(path, {"x": 1})
        require(
            parse_json_bytes(path.read_bytes(), "seal") == {"x": 1}, "durable mechanics"
        )
        expect(DurabilityError, write_json, path, {"x": 2})
        tiny = Path(directory) / "tiny.bin"
        tiny.write_bytes(bytes(range(251)) * 200)
        tiny_identity = file_identity(tiny)
        conditioned = sequential_condition_path(
            tiny, tiny_identity, tiny.stat().st_size
        )
        resident = mincore_residency(tiny, tiny_identity, tiny.stat().st_size)
        require(
            conditioned["bytes_read"] == tiny.stat().st_size
            and resident["all_pages_resident"] is True,
            "small-file conditioning/mincore ABI",
        )
        cutoff_path = Path(directory) / "cutoff.json"
        prior_mask = signal.pthread_sigmask(signal.SIG_BLOCK, [])
        with SignalController(True) as cutoff_controller:
            cutoff = cutoff_controller.final_cutoff(cutoff_path)
            require(cutoff["authority_event_count"] == 0, "quiet final cutoff")
            signal.pthread_sigmask(signal.SIG_SETMASK, prior_mask)
        require(read_json_file(cutoff_path, "cutoff") == cutoff, "cutoff persistence")
    target_dir = ROOT / "target"
    target_dir.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=target_dir) as directory:
        packet, work, decision, fixture_digest = make_complete_seal_fixture(
            Path(directory)
        )
        frozen_digest = SCHEDULE["digest"]
        SCHEDULE["digest"] = fixture_digest
        try:
            seal(decision, packet, work)
        finally:
            SCHEDULE["digest"] = frozen_digest
        completion = read_json_file(
            packet / "packet-complete.json", "complete fixture completion"
        )
        require(
            completion["semantic_completed_attempts"] == 12
            and completion["inventory_members"] == EXPECTED_INVENTORY_MEMBERS,
            "actual 58-member complete fixture did not seal",
        )
    with tempfile.TemporaryDirectory(dir=target_dir) as directory:
        packet, work, decision = make_seal_fixture(Path(directory), with_attempt=True)
        seal(decision, packet, work)
        completion = read_json_file(
            packet / "packet-complete.json", "fixture completion"
        )
        persisted_decision = read_json_file(
            packet / "decision.json", "fixture decision"
        )
        require(
            completion["semantic_completed_attempts"] == 1
            and persisted_decision["final_identity"] == decision["final_identity"],
            "complete cross-linked fixture did not seal",
        )
    with tempfile.TemporaryDirectory(dir=target_dir) as directory:
        packet, work, decision = make_seal_fixture(
            Path(directory), with_attempt=False, conditioning_only=True
        )
        seal(decision, packet, work)
        require(
            read_json_file(packet / "packet-complete.json", "incomplete completion")[
                "semantic_completed_attempts"
            ]
            == 0,
            "incomplete conditioning prefix did not seal",
        )
    with tempfile.TemporaryDirectory(dir=target_dir) as directory:
        packet, work, decision = make_seal_fixture(Path(directory), with_attempt=True)
        attempt = read_jsonl_file(packet / "attempts.jsonl", "stopped attempt")[0]
        next_stem = attempt_specs()[1]["stem"]
        write_json(
            work / f"{next_stem}.conditioning.json",
            seal_conditioning_fixture(
                next_stem,
                False,
                attempt["activity_boundary_monotonic_ns"],
            ),
        )
        expect(ContractDefect, seal, decision, packet, work)
    with tempfile.TemporaryDirectory(dir=target_dir) as directory:
        packet, work, decision = make_seal_fixture(Path(directory), with_attempt=True)
        closures = read_jsonl_file(
            packet / "attempt-signal-closures.jsonl", "closure mutation"
        )
        closures[0]["attempt_sha256"] = "0" * 64
        (packet / "attempt-signal-closures.jsonl").write_bytes(
            b"".join(json_bytes(value) for value in closures)
        )
        expect(ContractDefect, seal, decision, packet, work)
    with tempfile.TemporaryDirectory(dir=target_dir) as directory:
        packet, work, decision = make_seal_fixture(Path(directory), with_attempt=True)
        (work / f"{attempt_specs()[0]['stem']}.stdout").write_bytes(b"drift")
        expect(ContractDefect, seal, decision, packet, work)
    with tempfile.TemporaryDirectory(dir=target_dir) as directory:
        packet, work, decision = make_seal_fixture(Path(directory), with_attempt=True)
        stem = attempt_specs()[0]["stem"]
        post_path = work / f"{stem}.post.json"
        post = read_json_file(post_path, "missing residency post")
        post["residency_after"] = None
        post["operation_errors"] = ["post_exit_residency:OSError:fixture"]
        post_path.write_bytes(json_bytes(post, True))
        attempts = read_jsonl_file(
            packet / "attempts.jsonl", "missing residency attempt"
        )
        attempts[0]["post_sha256"] = sha_file(post_path)
        attempts[0]["validity_reasons"] = [
            "incomplete_residency_after_exit",
            "post_exit_residency:OSError:fixture",
        ]
        attempts[0]["stop_classification"] = "inconclusive"
        attempts[0]["stop_reason"] = (
            "validity failed: ['incomplete_residency_after_exit', "
            "'post_exit_residency:OSError:fixture']"
        )
        (packet / "attempts.jsonl").write_bytes(
            b"".join(json_bytes(value) for value in attempts)
        )
        closures = read_jsonl_file(
            packet / "attempt-signal-closures.jsonl", "missing residency closure"
        )
        closures[0]["attempt_sha256"] = sha_bytes(json_bytes(attempts[0]))
        (packet / "attempt-signal-closures.jsonl").write_bytes(
            b"".join(json_bytes(value) for value in closures)
        )
        decision.update(
            contract_error=None,
            invalid_error="fixture post-residency failure",
            status="inconclusive",
            closure="inconclusive",
        )
        seal(decision, packet, work)
        persisted = read_json_file(packet / "decision.json", "post-residency decision")
        require(
            persisted["status"] == "inconclusive"
            and persisted["closure"] == "inconclusive",
            "post-residency failure was not sealed inconclusive",
        )
    with tempfile.TemporaryDirectory(dir=target_dir) as directory:
        packet, work, decision = make_seal_fixture(Path(directory), with_attempt=True)
        attempts = read_jsonl_file(packet / "attempts.jsonl", "late post attempt")
        stem = attempts[0]["stem"]
        post_path = work / f"{stem}.post.json"
        post = read_json_file(post_path, "late post mutation")
        post["host_after_exit"]["captured_monotonic_ns"] = (
            attempts[0]["activity_boundary_monotonic_ns"] + 1
        )
        post_path.write_bytes(json_bytes(post, True))
        attempts[0]["post_sha256"] = sha_file(post_path)
        (packet / "attempts.jsonl").write_bytes(
            b"".join(json_bytes(value) for value in attempts)
        )
        closures = read_jsonl_file(
            packet / "attempt-signal-closures.jsonl", "late post closure"
        )
        closures[0]["attempt_sha256"] = sha_bytes(json_bytes(attempts[0]))
        (packet / "attempt-signal-closures.jsonl").write_bytes(
            b"".join(json_bytes(value) for value in closures)
        )
        expect(ContractDefect, seal, decision, packet, work)
    with tempfile.TemporaryDirectory(dir=target_dir) as directory:
        packet, work, decision = make_seal_fixture(Path(directory), with_attempt=True)
        stem = attempt_specs()[0]["stem"]
        stdout_path = work / f"{stem}.stdout"
        stderr_path = work / f"{stem}.stderr"
        stdout_path.write_bytes(b"{")
        values = {label: 0 for label in TIME_LABELS}
        values["maximum resident set size"] = 1000
        values["peak memory footprint"] = 1000
        stderr_text = "  1.00 real  0.20 user  0.30 sys\n" + "".join(
            f"  {values[label]}  {label}\n" for label in TIME_LABELS
        )
        stderr_path.write_bytes(stderr_text.encode())
        try:
            parse_json_bytes(stdout_path.read_bytes(), stem)
        except ContractDefect as error:
            parse_error = f"{type(error).__name__}:{error}"
        else:
            raise RuntimeError("malformed stdout fixture unexpectedly parsed")
        resources = parse_time(stderr_text)
        attempts = read_jsonl_file(packet / "attempts.jsonl", "parse-failure attempt")
        attempts[0].update(
            returncode=0,
            process_resources=resources,
            process_page_faults=resources["page_faults"],
            parse_error=parse_error,
            result=None,
            stop_classification="implementation_or_contract_defect",
            stop_reason=parse_error,
            stdout_sha256=sha_file(stdout_path),
            stderr_sha256=sha_file(stderr_path),
        )
        (packet / "attempts.jsonl").write_bytes(
            b"".join(json_bytes(value) for value in attempts)
        )
        lifecycle = read_jsonl_file(packet / "lifecycle.jsonl", "parse lifecycle")
        lifecycle[2]["returncode"] = 0
        (packet / "lifecycle.jsonl").write_bytes(
            b"".join(json_bytes(value) for value in lifecycle)
        )
        closures = read_jsonl_file(
            packet / "attempt-signal-closures.jsonl", "parse closure"
        )
        closures[0]["attempt_sha256"] = sha_bytes(json_bytes(attempts[0]))
        (packet / "attempt-signal-closures.jsonl").write_bytes(
            b"".join(json_bytes(value) for value in closures)
        )
        seal(decision, packet, work)
    with tempfile.TemporaryDirectory(dir=target_dir) as directory:
        packet, work, decision = make_seal_fixture(Path(directory), with_attempt=True)
        lifecycle = read_jsonl_file(packet / "lifecycle.jsonl", "lifecycle mutation")
        lifecycle[1]["pid"] += 1
        (packet / "lifecycle.jsonl").write_bytes(
            b"".join(json_bytes(value) for value in lifecycle)
        )
        expect(ContractDefect, seal, decision, packet, work)
    with tempfile.TemporaryDirectory(dir=target_dir) as directory:
        packet, work, decision = make_seal_fixture(Path(directory), with_attempt=True)
        attempts = read_jsonl_file(packet / "attempts.jsonl", "cleanup mutation")
        action = {
            "reason": "operator-signal-or-pipe-failure",
            "target_pgid": attempts[0]["pgid"],
            "signal": int(signal.SIGKILL),
            "attempted_monotonic_ns": 1,
            "succeeded": True,
            "error": None,
            "completed_monotonic_ns": 2,
        }
        attempts[0]["cleanup_actions"] = [action]
        attempts[0]["cleanup_signal_sent"] = True
        attempts[0]["pipe_errors"] = ["pipe0:fixture"]
        attempts[0]["validity_reasons"] = ["pipe0:fixture"]
        attempts[0]["stop_classification"] = "inconclusive"
        attempts[0]["stop_reason"] = "validity failed: ['pipe0:fixture']"
        lifecycle = read_jsonl_file(packet / "lifecycle.jsonl", "cleanup lifecycle")
        lifecycle[2]["cleanup_actions"] = [action]
        lifecycle[2]["cleanup_signal_sent"] = True
        lifecycle[2]["pipe_errors"] = ["pipe0:fixture"]
        (packet / "attempts.jsonl").write_bytes(
            b"".join(json_bytes(value) for value in attempts)
        )
        (packet / "lifecycle.jsonl").write_bytes(
            b"".join(json_bytes(value) for value in lifecycle)
        )
        closures = read_jsonl_file(
            packet / "attempt-signal-closures.jsonl", "cleanup closure"
        )
        closures[0]["attempt_sha256"] = sha_bytes(json_bytes(attempts[0]))
        (packet / "attempt-signal-closures.jsonl").write_bytes(
            b"".join(json_bytes(value) for value in closures)
        )
        decision["invalid_error"] = "fixture cleanup invalidity"
        expect(ContractDefect, seal, decision, packet, work)
    with tempfile.TemporaryDirectory(dir=target_dir) as directory:
        packet, work, decision = make_seal_fixture(Path(directory), with_attempt=False)
        os.symlink(work / "reservation.json", work / "bad-link")
        expect(ContractDefect, seal, decision, packet, work)
    with tempfile.TemporaryDirectory(dir=target_dir) as directory:
        packet, work, decision = make_seal_fixture(Path(directory), with_attempt=False)
        (work / "bad-directory").mkdir()
        expect(ContractDefect, seal, decision, packet, work)
    with tempfile.TemporaryDirectory(dir=target_dir) as directory:
        packet, work, decision = make_seal_fixture(Path(directory), with_attempt=False)
        decision["signal_log_sha256"] = "0" * 64
        expect(ContractDefect, seal, decision, packet, work)
    with tempfile.TemporaryDirectory(dir=target_dir) as directory:
        packet, work, decision = make_seal_fixture(Path(directory), with_attempt=False)
        decision.update(
            status="GO-mechanism-floor",
            closure="GO-mechanism-floor",
            authority="invented",
            force_authorized=True,
            invalid_error=None,
        )
        expect(ContractDefect, seal, decision, packet, work)
    with tempfile.TemporaryDirectory(dir=target_dir) as directory:
        packet, work, decision = make_seal_fixture(Path(directory), with_attempt=False)

        def mutate_persisted_decision():
            forged = dict(decision)
            forged.update(
                status="GO-mechanism-floor",
                closure="GO-mechanism-floor",
                authority="invented",
                force_authorized=True,
                invalid_error=None,
            )
            (packet / "decision.json").write_bytes(json_bytes(forged, True))
            (packet / "decision.sha256").write_bytes(
                (sha_file(packet / "decision.json") + "\n").encode()
            )

        def seal_with_persisted_mutation():
            seal(
                decision,
                packet,
                work,
                _after_decision_write=mutate_persisted_decision,
            )

        expect(ContractDefect, seal_with_persisted_mutation)
    with tempfile.TemporaryDirectory(dir=target_dir) as directory:
        packet, work, decision = make_seal_fixture(Path(directory), with_attempt=False)
        decision["analysis"] = {"forged": True}
        expect(ContractDefect, seal, decision, packet, work)
    with tempfile.TemporaryDirectory(dir=target_dir) as directory:
        packet, work, decision = make_seal_fixture(Path(directory), with_attempt=False)
        decision["closure"] = "KILL"
        expect(ContractDefect, seal, decision, packet, work)
    with tempfile.TemporaryDirectory(dir=target_dir) as directory:
        packet, work, decision = make_seal_fixture(Path(directory), with_attempt=True)
        attempts = read_jsonl_file(packet / "attempts.jsonl", "signal slice attempt")
        event = {"sequence": 1, "signal": int(signal.SIGTERM), "monotonic_ns": 9}
        attempts[0]["validity_reasons"] = ["deferred_operator_signal"]
        attempts[0]["stop_classification"] = "inconclusive"
        attempts[0]["stop_reason"] = "validity failed: ['deferred_operator_signal']"
        (packet / "attempts.jsonl").write_bytes(
            b"".join(json_bytes(value) for value in attempts)
        )
        closures = read_jsonl_file(
            packet / "attempt-signal-closures.jsonl", "signal slice closure"
        )
        closures[0].update(
            signal_end_sequence=1,
            events=[event],
            invalid=True,
            attempt_sha256=sha_bytes(json_bytes(attempts[0])),
        )
        (packet / "attempt-signal-closures.jsonl").write_bytes(
            b"".join(json_bytes(value) for value in closures)
        )
        decision["invalid_error"] = "fixture signal invalidity"
        expect(ContractDefect, seal, decision, packet, work)
    with tempfile.TemporaryDirectory(dir=target_dir) as directory:
        packet, work, decision = make_seal_fixture(Path(directory), with_attempt=True)
        attempts = read_jsonl_file(packet / "attempts.jsonl", "spawn mutation")
        attempts[0]["pid"] = None
        (packet / "attempts.jsonl").write_bytes(
            b"".join(json_bytes(value) for value in attempts)
        )
        closures = read_jsonl_file(
            packet / "attempt-signal-closures.jsonl", "spawn closure mutation"
        )
        closures[0]["attempt_sha256"] = sha_bytes(json_bytes(attempts[0]))
        (packet / "attempt-signal-closures.jsonl").write_bytes(
            b"".join(json_bytes(value) for value in closures)
        )
        expect(ContractDefect, seal, decision, packet, work)
    with tempfile.TemporaryDirectory(dir=target_dir) as directory:
        packet, work, decision = make_seal_fixture(
            Path(directory), with_attempt=False, conditioning_only=True
        )
        conditioning_path = work / f"{attempt_specs()[0]['stem']}.conditioning.json"
        conditioning = read_json_file(conditioning_path, "conditioning mutation")
        del conditioning["cooldown"]
        conditioning_path.write_bytes(json_bytes(conditioning, True))
        expect(ContractDefect, seal, decision, packet, work)
    with tempfile.TemporaryDirectory(dir=target_dir) as directory:
        packet, work, decision = make_seal_fixture(Path(directory), with_attempt=True)

        def mutate_after_inventory():
            (work / f"{attempt_specs()[0]['stem']}.stderr").write_bytes(b"late drift")

        expect(
            ContractDefect,
            seal,
            decision,
            packet,
            work,
            mutate_after_inventory,
        )
    with tempfile.TemporaryDirectory(dir=target_dir) as directory:
        packet, work, decision = make_seal_fixture(Path(directory), with_attempt=False)
        decision["final_identity"]["observed"]["source"] = {"fixture": "drift"}
        decision["final_identity"]["errors"] = ["source:mismatch"]
        decision["final_identity"]["matches_manifest"] = False
        decision["contract_error"] = "fixture final identity drift"
        decision["status"] = "implementation_or_contract_defect"
        decision["closure"] = "implementation_or_contract_defect"
        seal(decision, packet, work)
        persisted = read_json_file(packet / "decision.json", "identity-defect decision")
        require(
            persisted["final_identity"]["matches_manifest"] is False,
            "final identity defect evidence was not sealed",
        )
    require(
        launch_delay_failure(100, 100 + LAUNCH_LIMIT_NS) is None
        and launch_delay_failure(100, 101 + LAUNCH_LIMIT_NS)
        == "launch_exceeded_five_seconds",
        "inclusive launch boundary",
    )
    with SignalController(True) as controller:
        start = controller.sequence()
        os.kill(os.getpid(), signal.SIGINT)
        require(
            controller.since(start)[0]["signal"] == signal.SIGINT, "signal mechanics"
        )
    require(
        {item: signal.getsignal(item) for item in OPERATOR_SIGNALS} == initial_handlers
        and signal.pthread_sigmask(signal.SIG_BLOCK, []) == initial_mask,
        "signal restoration",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument(
        "--self-test-child", action="store_true", help=argparse.SUPPRESS
    )
    args = parser.parse_args()
    if args.self_test or args.self_test_child:
        run_self_tests(include_process_tests=not args.self_test_child)
        print("self-test-child: ok" if args.self_test_child else "self-test: ok")
        return
    execute()


if __name__ == "__main__":
    main()
