#!/usr/bin/env python3
"""v0.642 repair for process-wide page-fault validity attribution."""

import argparse
import ctypes
import hashlib
import json
import math
import mmap
import os
import re
import signal
import stat as stat_module
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PACKET = ROOT / "target/profiles/v0642-a3b-pread-worker-screen-fault-repair-p1"
WORK = ROOT / "target/profiles/v0642-a3b-pread-worker-screen-fault-repair-work"
PREREG = ROOT / "docs/bench/v0642-a3b-pread-worker-screen-fault-repair.md"
RUNNER = Path(__file__).resolve()
BINARY = ROOT / "target/release/qwen-bench"
MODEL = Path("/Users/tito/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")
IMPLEMENTATION_PARENT = "1b8dd176fe8647f276ed8f50c7714ae89dbf62ef"
V0641_SOURCE = "18fd67aff6bcc990e51b1f45fa488b7cad6c671b"
AUTHORITY_PARENT = "dfde5e04b381a641537d91f616b7b8fba5560f9d"
SEALED_PACKET = ROOT / "target/profiles/v0641-a3b-pread-worker-screen-p1"
SEALED_WORK = ROOT / "target/profiles/v0641-a3b-pread-worker-screen-work"
SEALED_HASHES = {
    "decision.json": "230b661b1192aa132f6b7c1327ed9dd797809edbd8d4f60a025813bd3cc36861",
    "artifact-inventory.sha256": "6d8c8f9ebdee6d57ac7df946579153cd8e13d3f5b40b0efa871a467f8238c2e7",
    "packet-complete.json": "4b130372658bf26b06e7fdb18d62f6b8aa38fc92b8bb5db42d0a3fc2b61f5a63",
    "manifest.json": "e9aa6f8b4faf0d636365d6d4c87d5ec2de109a104b5812a2198d323cd5723124",
    "order.json": "824959ff40ea26c033e5703721c32fb4ac7d11f4ee063956ef137d6c4045ff6e",
    "lifecycle.jsonl": "4772e3b7424f5a60d78cf4d590e3327b9e0805acd5e6ecefe41978fbfed942b9",
    "attempts.jsonl": "e955c6e6d1ccb39040c69d4c52be6d27467977b0f9e661b2b56f2212ffd3ff4f",
    "r1-p1-w1.conditioning.json": (
        "719974351a3c7b132dd4ca0c1d8d31bf8b1da6c8d9f085bb4f5ace3d7f81276d"
    ),
    "r1-p1-w1.post.json": "00ce8f13ccda12e0cf6d59c2877840db3480c2474b0121dd4b343dc21469bbcd",
    "r1-p1-w1.stdout": "a0192f16fc4acba41b9852f26fb4412ea68b1b88b9dbeea71dc5f5d074f8dfdb",
    "r1-p1-w1.stderr": "3143d70c35383b75d836cc2146e6bc87f0132658fe27ed4027ef00a1a7e25230",
}
SEALED_INVENTORY = {
    (
        "target/profiles/v0641-a3b-pread-worker-screen-p1/attempt-signal-closures.jsonl"
    ): "d3998083b3b043a74753bec537e16bc8bc20ebe48a9cbacddc0c6b34fe9babce",
    "target/profiles/v0641-a3b-pread-worker-screen-p1/attempts.jsonl": SEALED_HASHES[
        "attempts.jsonl"
    ],
    "target/profiles/v0641-a3b-pread-worker-screen-p1/decision.json": SEALED_HASHES[
        "decision.json"
    ],
    "target/profiles/v0641-a3b-pread-worker-screen-p1/decision.sha256": (
        "f9b749e0c8333148525839970d51b97b6fb3a7c9e1b1ca2ef5dd4fbfeafb47c8"
    ),
    "target/profiles/v0641-a3b-pread-worker-screen-p1/lifecycle.jsonl": SEALED_HASHES[
        "lifecycle.jsonl"
    ],
    "target/profiles/v0641-a3b-pread-worker-screen-p1/manifest.json": SEALED_HASHES[
        "manifest.json"
    ],
    "target/profiles/v0641-a3b-pread-worker-screen-p1/order.json": SEALED_HASHES[
        "order.json"
    ],
    (
        "target/profiles/v0641-a3b-pread-worker-screen-p1/packet-signal-cutoff.json"
    ): "6e6ec429a5fa0ee38398518c3ea7b4ba0b75be10846bafabe7f6cdcfc2e01a71",
    (
        "target/profiles/v0641-a3b-pread-worker-screen-p1/packet-signal-log.json"
    ): "fa3584640aa93ac65a99837646b57b040c2351be0550ea9ce2af4f4e51d4d2d3",
    "target/profiles/v0641-a3b-pread-worker-screen-work/r1-p1-w1.conditioning.json": SEALED_HASHES[
        "r1-p1-w1.conditioning.json"
    ],
    "target/profiles/v0641-a3b-pread-worker-screen-work/r1-p1-w1.post.json": SEALED_HASHES[
        "r1-p1-w1.post.json"
    ],
    "target/profiles/v0641-a3b-pread-worker-screen-work/r1-p1-w1.stderr": SEALED_HASHES[
        "r1-p1-w1.stderr"
    ],
    "target/profiles/v0641-a3b-pread-worker-screen-work/r1-p1-w1.stdout": SEALED_HASHES[
        "r1-p1-w1.stdout"
    ],
    "target/profiles/v0641-a3b-pread-worker-screen-work/reservation.json": (
        "55b3a3893c4e4a757d14dfa209a0f6198f81e9f8948906613cf2fe9bcfaa04e9"
    ),
}
MODEL_SIZE = 22_134_528_992
MODEL_PAGES = 1_350_985
MODEL_SHA256 = "ac0e2c1189e055faa36eff361580e79c5bd6f8e76bffb4ce547f167d53e31a61"
DESCRIPTOR_DIGEST = "0x5ae645df5cf7d568"
INVENTORY_DIGEST = "f57153febec22463c7789b892d4d084041d722483a93191c81c40ab86be7d9e5"
PROFILE = "a3b-q4km-v1"
ALGORITHM = "minimax-contiguous-v1"
REQUEST_COUNT = 733
COPY_BYTES = 22_123_538_944
PAGE_SIZE = 16_384
BUFFER_SIZE = 8 * 1024 * 1024
COOLDOWN_NS = 30_000_000_000
LAUNCH_LIMIT_NS = 5_000_000_000
MAX_OUTPUT_BYTES = 4 * 1024 * 1024
WORKERS = (1, 2, 4, 6, 8, 12)
ORDERS = (
    (1, 2, 12, 4, 8, 6),
    (2, 4, 1, 6, 12, 8),
    (4, 6, 2, 8, 1, 12),
    (6, 8, 4, 12, 2, 1),
    (8, 12, 6, 1, 4, 2),
    (12, 1, 8, 2, 6, 4),
)
BEFORE_ROWS = {
    1: (1, 5, 6),
    2: (1, 2, 6),
    6: (4, 5, 6),
    8: (4, 5, 6),
    12: (1, 5, 6),
}
SCHEDULES = {
    1: {
        "digest": "c22c86524146da70a3913fb3db7daa1149a8cc6cc36eb7d2d7aa1e1f7492adfb",
        "cuts": [],
        "task_counts": [733],
        "worker_bytes": [22_123_538_944],
    },
    2: {
        "digest": "19165cbe11e4e023881ccfd31de196bc40240020ca68f91a41e83f32496a6b56",
        "cuts": [359],
        "task_counts": [359, 374],
        "worker_bytes": [10_995_062_016, 11_128_476_928],
    },
    4: {
        "digest": "800bf469d09879187e07a832ab573a84f19714f6d3fce8d47fce87adc5808329",
        "cuts": [155, 359, 539],
        "task_counts": [155, 204, 180, 194],
        "worker_bytes": [5_532_746_240, 5_462_315_776, 5_595_522_304, 5_532_954_624],
    },
    6: {
        "digest": "9d8459edc9e4920eecaf528e95037705a3c43ac522a8051410414b10489d1b87",
        "cuts": [86, 220, 343, 470, 609],
        "task_counts": [86, 134, 123, 127, 139, 124],
        "worker_bytes": [
            3_581_055_232,
            3_693_068_800,
            3_681_269_504,
            3_683_499_776,
            3_720_937_984,
            3_763_707_648,
        ],
    },
    8: {
        "digest": "0432332f6a5f4417852bc30b527ae2f3fd26e6d82690b5d18f9d57da3c4bb438",
        "cuts": [68, 155, 251, 357, 445, 542, 635],
        "task_counts": [68, 87, 96, 106, 88, 97, 93, 98],
        "worker_bytes": [
            2_755_989_760,
            2_776_756_480,
            2_788_430_848,
            2_672_769_792,
            2_810_311_936,
            2_789_544_960,
            2_820_862_976,
            2_708_872_192,
        ],
    },
    12: {
        "digest": "b92041f460cb6f65b3132d03a072e540b045620c0650403eb771e1149130411a",
        "cuts": [26, 86, 153, 220, 287, 343, 410, 470, 537, 609, 666],
        "task_counts": [26, 60, 67, 67, 67, 56, 67, 60, 67, 72, 57, 67],
        "worker_bytes": [
            1_697_137_408,
            1_883_917_824,
            1_799_581_952,
            1_893_486_848,
            1_799_581_952,
            1_881_687_552,
            1_799_581_952,
            1_883_917_824,
            1_799_581_952,
            1_921_356_032,
            1_949_904_384,
            1_813_803_264,
        ],
    },
}
SCHEDULE_CONSTANTS_SHA256 = (
    "36554d65abcbe0c06edfd5c203410139e18608126f7494c0f59c8034f7c0d17f"
)
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


class ContractDefect(RuntimeError):
    pass


class Inconclusive(RuntimeError):
    pass


class DurabilityError(RuntimeError):
    pass


class OwnershipLost(BaseException):
    pass


OPERATOR_SIGNALS = (signal.SIGINT, signal.SIGTERM)


class PacketSignalController:
    def __init__(self, *, restore_on_exit: bool) -> None:
        self.restore_on_exit = restore_on_exit
        self.events: list[dict[str, int]] = []
        self.previous: dict[int, object] = {}
        self.installed = False
        self.cutoff = False

    def install(self) -> None:
        require(not self.installed, "signal controller is already installed")

        def record(signum: int, _frame: object) -> None:
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
        self.installed = True

    def restore(self) -> None:
        if not self.installed:
            return
        for signum in OPERATOR_SIGNALS:
            signal.signal(signum, self.previous[signum])
        self.installed = False

    def __enter__(self) -> "PacketSignalController":
        self.install()
        return self

    def __exit__(self, _kind: object, _error: object, _traceback: object) -> None:
        if self.restore_on_exit:
            self.restore()

    def sequence(self) -> int:
        return len(self.events)

    def since(self, sequence: int) -> list[dict[str, int]]:
        return [dict(event) for event in self.events if event["sequence"] > sequence]

    def require_quiet_since(self, sequence: int, stage: str) -> None:
        events = self.since(sequence)
        if events:
            raise Inconclusive(f"operator signal before {stage}: {events}")

    def final_cutoff(
        self, path: Path, *, _after_boundary=None, _after_block=None
    ) -> dict[str, object]:
        require(self.installed and not self.cutoff, "signal cutoff state is invalid")
        boundary_ns = time.monotonic_ns()
        boundary_sequence = self.sequence()
        if _after_boundary is not None:
            _after_boundary()
        signal.pthread_sigmask(signal.SIG_BLOCK, OPERATOR_SIGNALS)
        if _after_block is not None:
            _after_block()
        snapshot_events = tuple(dict(event) for event in self.events)
        pending = sorted(
            int(value) for value in signal.sigpending() if value in OPERATOR_SIGNALS
        )
        snapshot_ns = time.monotonic_ns()
        log = {
            "schema": 1,
            "logical_boundary_monotonic_ns": boundary_ns,
            "logical_boundary_sequence": boundary_sequence,
            "post_block_snapshot_monotonic_ns": snapshot_ns,
            "post_block_snapshot_sequence": (
                snapshot_events[-1]["sequence"]
                if snapshot_events
                else boundary_sequence
            ),
            "events": list(snapshot_events),
            "pending_signals": pending,
            "attribution": "all snapshot events and pending signals are pre-cutoff",
        }
        log_path = path.with_name("packet-signal-log.json")
        write_json(log_path, log)
        record = {
            "schema": 1,
            "event": "packet-signal-cutoff",
            "cutoff_monotonic_ns": boundary_ns,
            **log,
            "authority_event_count": len(snapshot_events) + len(pending),
            "signals_after_snapshot": "blocked-post-cutoff-outside-authority",
            "signal_log_sha256": sha256_file(log_path),
        }
        write_json(path, record)
        self.cutoff = True
        return record


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractDefect(message)


def lexists(path: Path) -> bool:
    return os.path.lexists(path)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_bytes(value: object, *, pretty: bool = False) -> bytes:
    if pretty:
        text = json.dumps(value, sort_keys=True, indent=2, ensure_ascii=True)
    else:
        text = json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        )
    return (text + "\n").encode("ascii")


def parse_json_bytes(value: bytes, label: str) -> object:
    def reject_duplicate(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result = {}
        for key, item in pairs:
            if key in result:
                raise ValueError(f"duplicate object key {key!r}")
            result[key] = item
        return result

    def reject_constant(constant: str) -> object:
        raise ValueError(f"non-finite JSON constant {constant}")

    def finite_float(number: str) -> float:
        parsed = float(number)
        if not math.isfinite(parsed):
            raise ValueError(f"non-finite JSON float {number}")
        return parsed

    try:
        text = value.decode("utf-8")
        parsed = json.loads(
            text,
            object_pairs_hook=reject_duplicate,
            parse_constant=reject_constant,
            parse_float=finite_float,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ContractDefect(f"{label} is malformed JSON: {error}") from error
    return parsed


def write_exclusive(path: Path, value: bytes) -> None:
    try:
        with path.open("xb") as output:
            output.write(value)
            output.flush()
            os.fsync(output.fileno())
    except OSError as error:
        raise DurabilityError(f"durable write failed for {path}: {error}") from error


def write_json(path: Path, value: object) -> None:
    write_exclusive(path, json_bytes(value, pretty=True))


def append_jsonl(path: Path, value: object) -> None:
    try:
        with path.open("ab") as output:
            output.write(json_bytes(value))
            output.flush()
            os.fsync(output.fileno())
    except OSError as error:
        raise DurabilityError(f"durable append failed for {path}: {error}") from error


def fsync_dir(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError as error:
        raise DurabilityError(f"directory fsync failed for {path}: {error}") from error


def command_output(command: list[str], env: dict[str, str] | None = None) -> str:
    result = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        detail = result.stderr[-4096:].decode("utf-8", errors="replace")
        raise ContractDefect(
            f"command failed ({result.returncode}): {command!r}: {detail}"
        )
    if len(result.stdout) > MAX_OUTPUT_BYTES or len(result.stderr) > MAX_OUTPUT_BYTES:
        raise ContractDefect(f"preflight command output exceeded bound: {command!r}")
    if result.stderr:
        raise ContractDefect(f"unexpected command stderr: {command!r}")
    return result.stdout.decode("utf-8")


def git_output(arguments: list[str]) -> str:
    return command_output(["git", *arguments]).strip()


def read_sealed_json(path: Path, label: str) -> object:
    require(
        path.is_file() and not path.is_symlink(),
        f"{label} is not a regular nonsymlink file",
    )
    return parse_json_bytes(path.read_bytes(), label)


def read_sealed_jsonl(path: Path, label: str) -> list[object]:
    require(
        path.is_file() and not path.is_symlink(),
        f"{label} is not a regular nonsymlink file",
    )
    lines = path.read_bytes().splitlines()
    require(bool(lines), f"{label} is empty")
    return [
        parse_json_bytes(line, f"{label} line {index}")
        for index, line in enumerate(lines, 1)
    ]


def authenticate_v0641_forensics() -> dict[str, object]:
    require(
        SEALED_PACKET.is_dir()
        and not SEALED_PACKET.is_symlink()
        and SEALED_WORK.is_dir()
        and not SEALED_WORK.is_symlink(),
        "sealed v0.641 roots are not real directories",
    )
    sealed_paths = {
        "decision.json": SEALED_PACKET / "decision.json",
        "artifact-inventory.sha256": SEALED_PACKET / "artifact-inventory.sha256",
        "packet-complete.json": SEALED_PACKET / "packet-complete.json",
        "manifest.json": SEALED_PACKET / "manifest.json",
        "order.json": SEALED_PACKET / "order.json",
        "lifecycle.jsonl": SEALED_PACKET / "lifecycle.jsonl",
        "attempts.jsonl": SEALED_PACKET / "attempts.jsonl",
        "r1-p1-w1.conditioning.json": SEALED_WORK / "r1-p1-w1.conditioning.json",
        "r1-p1-w1.post.json": SEALED_WORK / "r1-p1-w1.post.json",
        "r1-p1-w1.stdout": SEALED_WORK / "r1-p1-w1.stdout",
        "r1-p1-w1.stderr": SEALED_WORK / "r1-p1-w1.stderr",
    }
    for name, expected in SEALED_HASHES.items():
        path = sealed_paths[name]
        require(
            path.is_file() and not path.is_symlink() and sha256_file(path) == expected,
            f"sealed v0.641 hash drifted: {name}",
        )

    expected_inventory = b"".join(
        f"{digest}  {name}\n".encode("ascii")
        for name, digest in SEALED_INVENTORY.items()
    )
    inventory_path = SEALED_PACKET / "artifact-inventory.sha256"
    require(
        inventory_path.read_bytes() == expected_inventory,
        "sealed v0.641 inventory content drifted",
    )
    for name, expected in SEALED_INVENTORY.items():
        path = ROOT / name
        require(
            path.is_file() and not path.is_symlink() and sha256_file(path) == expected,
            f"sealed v0.641 inventory member drifted: {name}",
        )
    packet_names = {path.name for path in SEALED_PACKET.iterdir()}
    work_names = {path.name for path in SEALED_WORK.iterdir()}
    require(
        packet_names
        == {
            "artifact-inventory.sha256",
            "attempt-signal-closures.jsonl",
            "attempts.jsonl",
            "decision.json",
            "decision.sha256",
            "lifecycle.jsonl",
            "manifest.json",
            "order.json",
            "packet-complete.json",
            "packet-complete.sha256",
            "packet-signal-cutoff.json",
            "packet-signal-log.json",
        },
        "sealed v0.641 packet file set drifted",
    )
    require(
        work_names
        == {
            "r1-p1-w1.conditioning.json",
            "r1-p1-w1.post.json",
            "r1-p1-w1.stderr",
            "r1-p1-w1.stdout",
            "reservation.json",
        },
        "sealed v0.641 work file set drifted",
    )
    require(
        all(
            path.is_file() and not path.is_symlink() for path in SEALED_PACKET.iterdir()
        )
        and all(
            path.is_file() and not path.is_symlink() for path in SEALED_WORK.iterdir()
        ),
        "sealed v0.641 final set is not all regular nonsymlink files",
    )

    complete = read_sealed_json(
        SEALED_PACKET / "packet-complete.json", "v0.641 packet completion"
    )
    require(
        complete
        == {
            "schema": 1,
            "decision_sha256": SEALED_HASHES["decision.json"],
            "inventory_sha256": SEALED_HASHES["artifact-inventory.sha256"],
            "inventory_members": 14,
        },
        "sealed v0.641 completion binding drifted",
    )
    require(
        (SEALED_PACKET / "packet-complete.sha256").read_bytes()
        == (SEALED_HASHES["packet-complete.json"] + "\n").encode("ascii"),
        "sealed v0.641 completion digest file drifted",
    )
    require(
        (SEALED_PACKET / "decision.sha256").read_bytes()
        == (SEALED_HASHES["decision.json"] + "\n").encode("ascii"),
        "sealed v0.641 decision digest file drifted",
    )

    decision = read_sealed_json(SEALED_PACKET / "decision.json", "v0.641 decision")
    required_decision = {
        "schema": 1,
        "status": "inconclusive",
        "authority": "none",
        "force_authorized": False,
        "successor_authorization": "none",
        "source_commit": V0641_SOURCE,
        "implementation_parent": IMPLEMENTATION_PARENT,
        "completed_attempts": 1,
        "expected_attempts": 36,
        "contract_error": None,
        "invalid_error": "Inconclusive:r1-p1-w1 validity failed: ['page_faults=89']",
        "analysis": None,
    }
    require(isinstance(decision, dict), "sealed v0.641 decision is malformed")
    for key, expected in required_decision.items():
        require(decision.get(key) == expected, f"sealed v0.641 decision drifted: {key}")

    manifest = read_sealed_json(SEALED_PACKET / "manifest.json", "v0.641 manifest")
    require(isinstance(manifest, dict), "sealed v0.641 manifest is malformed")
    require(
        manifest.get("protocol") == "v0641-a3b-pread-worker-screen"
        and manifest.get("source_commit") == V0641_SOURCE
        and manifest.get("implementation_parent") == IMPLEMENTATION_PARENT
        and manifest.get("attempt_count") == 36
        and manifest.get("retry_count") == 0
        and manifest.get("authority") == "none",
        "sealed v0.641 manifest identity drifted",
    )
    require(
        manifest.get("orders") == [list(row) for row in ORDERS],
        "sealed v0.641 manifest order drifted",
    )

    attempts = read_sealed_jsonl(SEALED_PACKET / "attempts.jsonl", "v0.641 attempts")
    require(
        len(attempts) == 1 and isinstance(attempts[0], dict),
        "v0.641 attempt count drifted",
    )
    attempt = attempts[0]
    resources = attempt.get("process_resources")
    result = attempt.get("result")
    require(
        attempt.get("stem") == "r1-p1-w1"
        and attempt.get("row_index") == 1
        and attempt.get("position") == 1
        and attempt.get("workers") == 1
        and attempt.get("returncode") == 0
        and attempt.get("validity_reasons") == ["page_faults=89"]
        and attempt.get("parse_error") is None
        and isinstance(resources, dict)
        and resources.get("page_faults") == 89
        and resources.get("block_input_operations") == 0
        and resources.get("swaps") == 0
        and isinstance(result, dict)
        and result.get("rusage", {}).get("timer_major_faults") == 0
        and result.get("correctness")
        == {
            "passed": True,
            "payload_bytes_checked": COPY_BYTES,
            "entries_checked": REQUEST_COUNT,
        },
        "sealed v0.641 sole attempt drifted",
    )
    require(
        attempt.get("conditioning_sha256")
        == SEALED_HASHES["r1-p1-w1.conditioning.json"]
        and attempt.get("post_sha256") == SEALED_HASHES["r1-p1-w1.post.json"]
        and attempt.get("stdout_sha256") == SEALED_HASHES["r1-p1-w1.stdout"]
        and attempt.get("stderr_sha256") == SEALED_HASHES["r1-p1-w1.stderr"],
        "sealed v0.641 sole attempt evidence binding drifted",
    )

    conditioning = read_sealed_json(
        SEALED_WORK / "r1-p1-w1.conditioning.json", "v0.641 conditioning"
    )
    post = read_sealed_json(SEALED_WORK / "r1-p1-w1.post.json", "v0.641 post")
    require(
        isinstance(conditioning, dict)
        and conditioning.get("residency_before", {}).get("resident_pages")
        == MODEL_PAGES
        and conditioning.get("residency_before", {}).get("total_pages") == MODEL_PAGES
        and conditioning.get("residency_before", {}).get("all_pages_resident") is True
        and conditioning.get("conditioning_interval", {}).get("failure_reasons") == []
        and all(
            conditioning.get("conditioning_interval", {}).get("deltas", {}).get(key)
            == 0
            for key in ("compressions", "swapouts", "swap_used_bytes")
        ),
        "sealed v0.641 conditioning evidence drifted",
    )
    require(
        isinstance(post, dict)
        and post.get("residency_after", {}).get("resident_pages") == MODEL_PAGES
        and post.get("residency_after", {}).get("total_pages") == MODEL_PAGES
        and post.get("residency_after", {}).get("all_pages_resident") is True
        and post.get("child_interval", {}).get("failure_reasons") == []
        and all(
            post.get("child_interval", {}).get("deltas", {}).get(key) == 0
            for key in ("compressions", "swapouts", "swap_used_bytes")
        ),
        "sealed v0.641 post-exit evidence drifted",
    )
    lifecycle = read_sealed_jsonl(SEALED_PACKET / "lifecycle.jsonl", "v0.641 lifecycle")
    require(
        len(lifecycle) == 3
        and [event.get("event") for event in lifecycle if isinstance(event, dict)]
        == ["launch", "acquired", "completion"]
        and all(event.get("stem") == "r1-p1-w1" for event in lifecycle),
        "sealed v0.641 lifecycle or later-launch evidence drifted",
    )
    closures = read_sealed_jsonl(
        SEALED_PACKET / "attempt-signal-closures.jsonl", "v0.641 closures"
    )
    require(
        len(closures) == 1
        and isinstance(closures[0], dict)
        and closures[0].get("stem") == "r1-p1-w1"
        and closures[0].get("events") == []
        and closures[0].get("invalid") is False,
        "sealed v0.641 attempt closure drifted",
    )
    return {
        "packet": str(SEALED_PACKET.relative_to(ROOT)),
        "work": str(SEALED_WORK.relative_to(ROOT)),
        "source_commit": V0641_SOURCE,
        "implementation_parent": IMPLEMENTATION_PARENT,
        "forensic_class": "process-page-fault-validity-gate-overbroad",
        "authority_imported": False,
        "observations_imported": 0,
        "gates_imported": 0,
        "timings_imported": 0,
        "rows_imported": 0,
        "results_imported": 0,
        "scoring_inputs_imported": 0,
        "decision_sha256": SEALED_HASHES["decision.json"],
        "inventory_sha256": SEALED_HASHES["artifact-inventory.sha256"],
        "completion_sha256": SEALED_HASHES["packet-complete.json"],
        "inventory_members": len(SEALED_INVENTORY),
    }


def file_identity(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {
        "device": stat.st_dev,
        "inode": stat.st_ino,
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def complete_stat_stamp(stat: os.stat_result) -> dict[str, int]:
    return {
        "device": stat.st_dev,
        "inode": stat.st_ino,
        "mode": stat.st_mode,
        "nlink": stat.st_nlink,
        "uid": stat.st_uid,
        "gid": stat.st_gid,
        "rdev": stat.st_rdev,
        "size_bytes": stat.st_size,
        "block_size": getattr(stat, "st_blksize", 0),
        "blocks": getattr(stat, "st_blocks", 0),
        "mtime_ns": stat.st_mtime_ns,
        "ctime_ns": stat.st_ctime_ns,
        "birthtime_ns": round(getattr(stat, "st_birthtime", 0.0) * 1_000_000_000),
        "flags": getattr(stat, "st_flags", 0),
    }


def descriptor_bound_sha256(
    path: Path,
    expected_size: int | None = None,
    _after_open=None,
) -> dict[str, object]:
    path_lstat_before = os.lstat(path)
    path_stat_before = os.stat(path)
    require(stat_module.S_ISREG(path_lstat_before.st_mode), f"{path} is not regular")
    require(path_lstat_before.st_nlink == 1, f"{path} link count is not one")
    require(path_lstat_before.st_mode & 0o022 == 0, f"{path} is group/world writable")
    if path == BINARY:
        require(path_lstat_before.st_mode & 0o100 != 0, "qwen-bench is not executable")
    require(hasattr(os, "O_NOFOLLOW"), "O_NOFOLLOW is unavailable")
    flags = os.O_RDONLY | os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        descriptor_before = os.fstat(descriptor)
        descriptor_stamp = complete_stat_stamp(descriptor_before)
        require(
            stat_module.S_ISREG(descriptor_before.st_mode), "descriptor is not regular"
        )
        require(
            (path_lstat_before.st_dev, path_lstat_before.st_ino)
            == (descriptor_before.st_dev, descriptor_before.st_ino)
            == (path_stat_before.st_dev, path_stat_before.st_ino),
            "pathname does not name opened descriptor",
        )
        size = descriptor_before.st_size if expected_size is None else expected_size
        require(descriptor_before.st_size == size, f"{path} size drifted")
        if _after_open is not None:
            _after_open()
        digest = hashlib.sha256()
        bytes_read = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            bytes_read += len(chunk)
        descriptor_after = os.fstat(descriptor)
        require(
            complete_stat_stamp(descriptor_after) == descriptor_stamp,
            "descriptor stamp changed while hashing",
        )
        require(bytes_read == size, "descriptor hash byte count drifted")
        path_lstat_after = os.lstat(path)
        path_stat_after = os.stat(path)
        require(
            complete_stat_stamp(path_lstat_after)
            == complete_stat_stamp(path_lstat_before)
            and complete_stat_stamp(path_stat_after)
            == complete_stat_stamp(path_stat_before),
            "pathname stamp changed while hashing",
        )
        require(
            (path_lstat_after.st_dev, path_lstat_after.st_ino)
            == (descriptor_after.st_dev, descriptor_after.st_ino),
            "pathname no longer names opened descriptor",
        )
        return {
            "descriptor_stamp": descriptor_stamp,
            "bytes_hashed": bytes_read,
            "sha256": digest.hexdigest(),
        }
    finally:
        os.close(descriptor)


def normalized_environment() -> tuple[dict[str, str], list[str]]:
    inherited = os.environ
    env = {
        key: inherited[key]
        for key in SAFE_ENVIRONMENT_KEYS
        if inherited.get(key) is not None
    }
    for required in ("HOME", "PATH", "TMPDIR"):
        require(required in env, f"required environment key is absent: {required}")
    removed = []
    for key in sorted(inherited):
        upper = key.upper()
        if (
            upper.startswith(("QWEN", "MTL", "METAL"))
            or upper == "RUST_LOG"
            or "QOS" in upper
            or "CHUNK" in upper
            or key not in SAFE_ENVIRONMENT_KEYS
        ):
            removed.append(key)
    env["RUST_BACKTRACE"] = "0"
    return env, removed


def source_and_build_identity(env: dict[str, str]) -> dict[str, object]:
    require(Path.cwd().resolve() == ROOT, "runner must execute from repository root")
    require(
        not git_output(["status", "--porcelain=v1", "--untracked-files=all"]),
        "worktree is not clean",
    )
    head = git_output(["rev-parse", "HEAD"])
    parents = git_output(["rev-list", "--parents", "-n", "1", "HEAD"]).split()
    require(len(parents) == 2 and parents[0] == head, "R642 must have one parent")
    require(parents[1] == AUTHORITY_PARENT, "R642 authority parent drifted")
    expected = sorted((str(PREREG.relative_to(ROOT)), str(RUNNER.relative_to(ROOT))))
    changed = git_output(["diff", "--name-only", "HEAD^..HEAD"]).splitlines()
    require(sorted(changed) == expected, "R642 path boundary drifted")
    status = git_output(["diff", "--name-status", "HEAD^..HEAD"]).splitlines()
    require(
        sorted(status) == sorted(f"A\t{path}" for path in expected),
        "R642 files are not exact additions",
    )
    require(
        git_output(["rev-parse", f"{AUTHORITY_PARENT}^"]) == V0641_SOURCE
        and git_output(["rev-parse", f"{V0641_SOURCE}^"]) == IMPLEMENTATION_PARENT,
        "v0.641/authority ancestry drifted",
    )
    for commit, parent in (
        (AUTHORITY_PARENT, V0641_SOURCE),
        (V0641_SOURCE, IMPLEMENTATION_PARENT),
    ):
        ancestry = git_output(["rev-list", "--parents", "-n", "1", commit]).split()
        require(
            ancestry == [commit, parent],
            f"authenticated lineage is not a direct non-merge child: {commit}",
        )
    authority_paths = git_output(
        ["diff", "--name-status", f"{V0641_SOURCE}..{AUTHORITY_PARENT}"]
    ).splitlines()
    require(
        sorted(authority_paths) == ["M\tdocs/PERF-LOG.md", "M\tdocs/PERF-ROADMAP.md"],
        "post-v0.641 PERF authority diff drifted",
    )
    v0641_paths = (
        "docs/bench/v0641-a3b-pread-worker-screen.md",
        "scripts/profile/v0641_a3b_pread_worker_screen.py",
    )
    require(
        sorted(
            git_output(
                ["diff", "--name-status", f"{IMPLEMENTATION_PARENT}..{V0641_SOURCE}"]
            ).splitlines()
        )
        == sorted(f"A\t{path}" for path in v0641_paths),
        "v0.641 source diff drifted",
    )
    require(
        git_output(["rev-parse", "HEAD:crates"])
        == git_output(["rev-parse", f"{IMPLEMENTATION_PARENT}:crates"]),
        "crates tree differs from authenticated implementation",
    )
    build = parse_json_bytes(
        command_output([str(BINARY), "build-info", "--output", "json"], env).encode(),
        "build-info",
    )
    require(isinstance(build, dict), "build-info is not an object")
    require(
        build.get("build_commit") == head
        and build.get("runtime_commit") == head
        and build.get("status") == "match"
        and build.get("build_dirty") is False
        and build.get("runtime_dirty") is False
        and build.get("build_source_state") == build.get("runtime_source_state"),
        "source/build/runtime identity mismatch",
    )
    return {
        "head": head,
        "parent": parents[1],
        "authority_parent": AUTHORITY_PARENT,
        "v0641_source": V0641_SOURCE,
        "implementation_parent": IMPLEMENTATION_PARENT,
        "build_identity": build,
    }


def parse_cargo_test_summary(text: str) -> dict[str, int]:
    matches = re.findall(
        r"^test result: ok\. (\d+) passed; (\d+) failed; (\d+) ignored; "
        r"(\d+) measured; (\d+) filtered out; finished in "
        r"[0-9]+(?:\.[0-9]+)?s$",
        text,
        re.MULTILINE,
    )
    require(len(matches) == 1, "Cargo test summary is not unique")
    passed, failed, ignored, measured, filtered = map(int, matches[0])
    require(
        (passed, failed, ignored, measured) == (11, 0, 0, 0),
        "Cargo test summary is not exactly 11 passed",
    )
    return {
        "passed": passed,
        "failed": failed,
        "ignored": ignored,
        "measured": measured,
        "filtered_out": filtered,
    }


def validate_gate_lifecycle(outcome: dict[str, object]) -> dict[str, object]:
    pid = outcome.get("pid")
    pgid = outcome.get("pgid")
    require(type(pid) is int and pid > 0, "gate pid is missing")
    require(type(pgid) is int and pgid == pid, "gate pgid ownership drifted")
    actions = outcome.get("cleanup_actions")
    require(isinstance(actions, list), "gate cleanup actions are malformed")
    require(actions == [], "normal gate unexpectedly required cleanup")
    require(outcome.get("cleanup_signal_sent") is False, "gate cleanup state drifted")
    disposition = outcome.get("group_disposition")
    require(isinstance(disposition, dict), "gate group disposition is missing")
    require(
        disposition.get("no_live_group_observed") is True,
        "gate process group did not disappear",
    )
    require(disposition.get("pgid") == pgid, "gate disposition pgid drifted")
    samples = disposition.get("samples")
    require(
        isinstance(samples, list) and samples, "gate disposition samples are missing"
    )
    for sample in samples:
        require(isinstance(sample, dict), "gate disposition sample is malformed")
        require(sample.get("error") is None, "gate group inspection reported an error")
        require(
            isinstance(sample.get("members"), list), "gate group members are malformed"
        )
    require(samples[-1].get("members") == [], "gate group still has surviving members")
    return {
        "pid": pid,
        "pgid": pgid,
        "cleanup_signal_sent": False,
        "cleanup_action_cardinality": len(actions),
        "cleanup_actions": actions,
        "group_disposition": disposition,
    }


def run_gate(
    command: list[str], env: dict[str, str], *, cargo_test: bool = False
) -> dict[str, object]:
    outcome = bounded_child(command, env)
    require(outcome["spawn_error"] is None, f"gate spawn failed: {command!r}")
    require(outcome["wait_errors"] == [], f"gate wait failed: {command!r}")
    require(outcome["poll_errors"] == [], f"gate poll failed: {command!r}")
    require(outcome["pipe_errors"] == [], f"gate pipe failed: {command!r}")
    require(
        outcome["termination_errors"] == [], f"gate termination failed: {command!r}"
    )
    require(outcome["reaped"] is True, f"gate child was not reaped: {command!r}")
    require(outcome["interrupted"] is False, f"gate was interrupted: {command!r}")
    require(
        outcome["output_overflow"] is False, f"gate output exceeded bound: {command!r}"
    )
    require(outcome["returncode"] == 0, f"preflight gate failed: {command!r}")
    lifecycle = validate_gate_lifecycle(outcome)
    stdout = outcome["stdout"].decode("utf-8")
    stderr = outcome["stderr"].decode("utf-8")
    summary = parse_cargo_test_summary(stdout + stderr) if cargo_test else None
    return {
        "command": command,
        "returncode": outcome["returncode"],
        "wall_ns": outcome["wall_ns"],
        "stdout": stdout,
        "stderr": stderr,
        "stdout_sha256": sha256_bytes(outcome["stdout"]),
        "stderr_sha256": sha256_bytes(outcome["stderr"]),
        "cargo_test_summary": summary,
        **lifecycle,
    }


def fresh_build_and_tests(env: dict[str, str]) -> list[dict[str, object]]:
    build = ["cargo", "build", "--release", "-p", "qwen-cli", "--bin", "qwen-bench"]
    tests = [
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
    ]
    return [run_gate(build, env), run_gate(tests, env, cargo_test=True)]


def canonical_schedule_digest(schedule: object) -> str:
    require(isinstance(schedule, dict), "schedule is not an object")
    encoded = json.dumps(
        schedule, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return sha256_bytes(encoded)


def exact_uint(value: object, expected: int, label: str) -> int:
    require(type(value) is int and value == expected, f"{label} drifted")
    return value


def finite(value: object, label: str, *, positive: bool = False) -> float:
    require(
        type(value) in (int, float) and not isinstance(value, bool),
        f"{label} is not numeric",
    )
    parsed = float(value)
    require(
        math.isfinite(parsed) and (parsed > 0 if positive else parsed >= 0),
        f"{label} is not finite {'positive' if positive else 'nonnegative'}",
    )
    return parsed


def validate_schedule(schedule: object, workers: int) -> dict[str, object]:
    require(isinstance(schedule, dict), "schedule is not an object")
    frozen = SCHEDULES[workers]
    require(schedule.get("algorithm") == ALGORITHM, "schedule algorithm drifted")
    exact_uint(schedule.get("workers"), workers, "schedule worker count")
    require(schedule.get("cuts") == frozen["cuts"], "schedule cuts drifted")
    require(
        schedule.get("task_counts") == frozen["task_counts"],
        "schedule task counts drifted",
    )
    require(
        schedule.get("worker_bytes") == frozen["worker_bytes"],
        "schedule worker bytes drifted",
    )
    require(sum(frozen["task_counts"]) == REQUEST_COUNT, "task-count total drifted")
    require(sum(frozen["worker_bytes"]) == COPY_BYTES, "worker-byte total drifted")
    partitions = schedule.get("partitions")
    require(
        isinstance(partitions, list) and len(partitions) == workers,
        "schedule partitions drifted",
    )
    cursor = 0
    for index, partition in enumerate(partitions):
        require(isinstance(partition, dict), "schedule partition is malformed")
        exact_uint(partition.get("start"), cursor, "partition start")
        cursor += frozen["task_counts"][index]
        exact_uint(partition.get("end"), cursor, "partition end")
        exact_uint(
            partition.get("task_count"),
            frozen["task_counts"][index],
            "partition task count",
        )
        exact_uint(
            partition.get("bytes"), frozen["worker_bytes"][index], "partition bytes"
        )
        for key in (
            "first_shard",
            "first_source_offset",
            "last_shard",
            "last_source_offset",
        ):
            require(
                type(partition.get(key)) is int and partition[key] >= 0,
                f"partition {key} is malformed",
            )
        for key in ("first", "last"):
            endpoint = partition.get(key)
            require(isinstance(endpoint, dict), f"partition {key} is malformed")
            for field in ("request_index", "shard_idx", "source_offset", "n_bytes"):
                require(
                    type(endpoint.get(field)) is int and endpoint[field] >= 0,
                    f"partition endpoint {field} is malformed",
                )
            require(
                type(endpoint.get("name")) is str and endpoint["name"],
                "partition endpoint name is malformed",
            )
    require(cursor == REQUEST_COUNT, "partition union drifted")
    finite(schedule.get("max_to_min"), "schedule max_to_min", positive=True)
    finite(schedule.get("max_to_ideal"), "schedule max_to_ideal", positive=True)
    require(
        canonical_schedule_digest(schedule) == frozen["digest"],
        "canonical schedule digest drifted",
    )
    return schedule


def child_command(workers: int) -> list[str]:
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
        "parallel-pread",
        "--workers",
        str(workers),
        "--output",
        "json",
    ]


def describe_schedules(env: dict[str, str], build: object) -> dict[str, object]:
    controls = {}
    for workers in WORKERS:
        command = [
            str(BINARY),
            "gguf-arena-floor",
            "--model",
            str(MODEL),
            "--describe",
            "--workers",
            str(workers),
            "--output",
            "json",
        ]
        value = parse_json_bytes(
            command_output(command, env).encode(), f"W{workers} describe"
        )
        require(isinstance(value, dict), "describe output is not an object")
        require(
            value.get("schema_version") == 2 and value.get("mode") == "describe",
            "describe schema/mode drifted",
        )
        require(value.get("matched_profile") == PROFILE, "describe profile drifted")
        require(
            value.get("descriptor_layout_digest") == DESCRIPTOR_DIGEST,
            "describe descriptor drifted",
        )
        require(
            value.get("inventory_digest") == INVENTORY_DIGEST,
            "describe inventory drifted",
        )
        require(
            value.get("request_count") == REQUEST_COUNT
            and value.get("logical_copy_bytes") == COPY_BYTES,
            "describe counts drifted",
        )
        require(value.get("build_identity") == build, "describe build identity drifted")
        validate_schedule(value.get("computed_schedule"), workers)
        require(
            value.get("parallel_copy_schedule") == value.get("computed_schedule"),
            "describe schedule alias drifted",
        )
        if workers == 4:
            require(
                value.get("frozen_schedule") == value.get("computed_schedule"),
                "W4 frozen schedule drifted",
            )
        controls[str(workers)] = {"command": command, "output": value}
    return controls


def parse_vm_counter(text: str, label: str) -> int:
    match = re.search(rf"^{re.escape(label)}:\s+(\d+)\.$", text, re.MULTILINE)
    if match is None:
        raise RuntimeError(f"cannot parse vm_stat label {label}")
    return int(match.group(1))


def parse_swap_bytes(text: str) -> int:
    match = re.search(r"\bused\s*=\s*([0-9]+(?:\.[0-9]+)?)([BKMGT])", text)
    if match is None:
        raise RuntimeError("cannot parse swap occupancy")
    scale = {"B": 1, "K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4}
    value = float(match.group(1)) * scale[match.group(2)]
    if not math.isfinite(value) or value < 0:
        raise RuntimeError("invalid swap occupancy")
    return round(value)


def raw_command(command: list[str]) -> str:
    return subprocess.run(
        command, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    ).stdout


def capture_vm_state() -> dict[str, object]:
    capture_started_ns = time.monotonic_ns()
    errors = []
    values: dict[str, object] = {}
    try:
        vm_text = raw_command(["vm_stat"])
        for key, label in (
            ("pageouts", "Pageouts"),
            ("compressions", "Compressions"),
            ("swapouts", "Swapouts"),
            ("compressor_stored_pages", "Pages stored in compressor"),
            ("compressor_occupied_pages", "Pages occupied by compressor"),
        ):
            values[key] = parse_vm_counter(vm_text, label)
    except Exception as error:
        vm_text = None
        errors.append(f"vm_stat:{type(error).__name__}:{error}")
    try:
        swap_text = raw_command(["sysctl", "-n", "vm.swapusage"])
        values["swap_used_bytes"] = parse_swap_bytes(swap_text)
    except Exception as error:
        swap_text = None
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
    return {
        **values,
        "vm_stat": vm_text,
        "swapusage": swap_text,
        "errors": errors,
        "capture_started_monotonic_ns": capture_started_ns,
        "captured_monotonic_ns": time.monotonic_ns(),
    }


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
    deltas = {}
    reasons = []
    if before.get("errors") or after.get("errors"):
        reasons.append(f"{label}_capture_invalid")
    for key in keys:
        left, right = before.get(key), after.get(key)
        if type(left) is int and type(right) is int:
            deltas[key] = right - left
        else:
            deltas[key] = None
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


def ancestor_pids() -> set[int]:
    ancestors = {os.getpid()}
    pid = os.getppid()
    while pid > 1 and pid not in ancestors:
        ancestors.add(pid)
        try:
            line = raw_command(["ps", "-p", str(pid), "-o", "ppid="]).strip()
            pid = int(line)
        except (ValueError, subprocess.SubprocessError):
            break
    return ancestors


def competing_processes() -> list[dict[str, object]]:
    own = ancestor_pids()
    rows = []
    text = raw_command(["ps", "-axo", "pid=,ppid=,command="])
    model_name = MODEL.name.lower()
    executable = re.compile(r"(?:^|/)(?:qwen|qwen-bench|llama[^/ ]*)(?:\s|$)")
    for line in text.splitlines():
        match = re.match(r"\s*(\d+)\s+(\d+)\s+(.*)", line)
        if match is None:
            continue
        pid, ppid, command = int(match.group(1)), int(match.group(2)), match.group(3)
        if pid in own:
            continue
        lower = command.lower()
        gpu_experiment = ("metal" in lower or " gpu" in lower) and any(
            token in lower for token in ("bench", "profile", "experiment")
        )
        if model_name in lower or executable.search(lower) or gpu_experiment:
            rows.append({"pid": pid, "ppid": ppid, "command": command})
    return rows


def capture_host_state() -> dict[str, object]:
    capture_started_ns = time.monotonic_ns()
    errors = []
    try:
        thermal = raw_command(["pmset", "-g", "therm"])
    except Exception as error:
        thermal = None
        errors.append(f"thermal:{type(error).__name__}:{error}")
    try:
        battery = raw_command(["pmset", "-g", "batt"])
    except Exception as error:
        battery = None
        errors.append(f"battery:{type(error).__name__}:{error}")
    try:
        pressure = raw_command(["memory_pressure", "-Q"])
        match = re.search(r"System-wide memory free percentage: (\d+)%", pressure)
        available = int(match.group(1)) if match else None
    except Exception as error:
        pressure, available = None, None
        errors.append(f"memory:{type(error).__name__}:{error}")
    try:
        competitors = competing_processes()
    except Exception as error:
        competitors = None
        errors.append(f"processes:{type(error).__name__}:{error}")
    valid = host_state_valid(thermal, battery, available, competitors, errors)
    return {
        "thermal": thermal,
        "battery": battery,
        "memory_pressure": pressure,
        "memory_available_percent": available,
        "competing_processes": competitors,
        "errors": errors,
        "valid": valid,
        "capture_started_monotonic_ns": capture_started_ns,
        "captured_monotonic_ns": time.monotonic_ns(),
    }


def host_state_valid(
    thermal: object,
    battery: object,
    available: object,
    competitors: object,
    errors: list[str],
) -> bool:
    return (
        not errors
        and type(thermal) is str
        and "No thermal warning level has been recorded" in thermal
        and "No performance warning level has been recorded" in thermal
        and type(battery) is str
        and "AC Power" in battery
        and type(available) is int
        and available >= 50
        and competitors == []
    )


def identity_from_stat(stat: os.stat_result) -> dict[str, int]:
    return {
        "device": stat.st_dev,
        "inode": stat.st_ino,
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def sequential_condition(
    path: Path, expected: dict[str, int], expected_size: int
) -> dict[str, int]:
    require(file_identity(path) == expected, "descriptor drifted before conditioning")
    descriptor = os.open(path, os.O_RDONLY)
    started = time.perf_counter_ns()
    total = 0
    try:
        require(
            identity_from_stat(os.fstat(descriptor)) == expected,
            "open descriptor identity drifted",
        )
        buffer = bytearray(BUFFER_SIZE)
        with os.fdopen(descriptor, "rb", buffering=0, closefd=False) as source:
            while True:
                count = source.readinto(buffer)
                if count == 0:
                    break
                total += count
        require(total == expected_size, "conditioning byte count drifted")
        require(
            identity_from_stat(os.fstat(descriptor)) == expected
            and file_identity(path) == expected,
            "descriptor drifted during conditioning",
        )
    finally:
        os.close(descriptor)
    return {
        "bytes_read": total,
        "wall_ns": time.perf_counter_ns() - started,
        "buffer_bytes": BUFFER_SIZE,
    }


def mincore_residency(
    path: Path, expected: dict[str, int], expected_size: int
) -> dict[str, object]:
    page_size = os.sysconf("SC_PAGE_SIZE")
    page_count = (expected_size + page_size - 1) // page_size
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
            file_identity(path) == expected
            and identity_from_stat(os.fstat(descriptor)) == expected,
            "descriptor drifted before mincore",
        )
        address = libc.mmap(
            None, expected_size, mmap.PROT_READ, mmap.MAP_SHARED, descriptor, 0
        )
        if address == ctypes.c_void_p(-1).value:
            number = ctypes.get_errno()
            raise OSError(number, os.strerror(number))
        vector = (ctypes.c_ubyte * page_count)()
        if libc.mincore(address, expected_size, vector) != 0:
            number = ctypes.get_errno()
            raise OSError(number, os.strerror(number))
        resident = sum(1 for value in vector if value & 1)
        require(
            file_identity(path) == expected
            and identity_from_stat(os.fstat(descriptor)) == expected,
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
        "total_pages": page_count,
        "resident_pages": resident,
        "all_pages_resident": resident == page_count,
        "file_identity": expected,
    }


def parse_time_resource(stderr: str, label: str) -> int:
    values = re.findall(rf"^\s*(\d+)\s+{re.escape(label)}$", stderr, re.MULTILINE)
    if len(values) != 1:
        raise ContractDefect(f"/usr/bin/time label drifted: {label}")
    return int(values[0])


def parse_time_output(stderr: str) -> dict[str, object]:
    summaries = re.findall(
        r"^\s*([0-9]+\.[0-9]{2}) real\s+([0-9]+\.[0-9]{2}) user\s+"
        r"([0-9]+\.[0-9]{2}) sys$",
        stderr,
        re.MULTILINE,
    )
    require(len(summaries) == 1, "/usr/bin/time summary drifted")
    real, user, system = (float(value) for value in summaries[0])
    values = {
        label.replace(" ", "_"): parse_time_resource(stderr, label)
        for label in TIME_LABELS
    }
    return {
        "real_s": real,
        "user_cpu_s": user,
        "system_cpu_s": system,
        "total_cpu_s": user + system,
        **values,
    }


def process_page_faults_advisory(resources: dict[str, object]) -> int:
    page_faults = resources.get("page_faults")
    require(
        type(page_faults) is int and page_faults >= 0,
        "process page faults are not unsigned",
    )
    return page_faults


def process_resource_failure_reasons(resources: dict[str, object]) -> list[str]:
    reasons = []
    for key in ("block_input_operations", "swaps"):
        value = resources.get(key)
        require(type(value) is int and value >= 0, f"{key} is not unsigned")
        if value != 0:
            reasons.append(f"{key}={value}")
    process_page_faults_advisory(resources)
    return reasons


def validate_result(value: object, workers: int, build: object) -> dict[str, object]:
    require(isinstance(value, dict), "child JSON is not an object")
    require(value.get("schema_version") == 2, "child schema drifted")
    require(
        value.get("profile") == PROFILE and value.get("arm") == "parallel-pread",
        "child profile or arm drifted",
    )
    require(value.get("model") == str(MODEL), "child model path drifted")
    require(
        value.get("descriptor_layout_digest") == DESCRIPTOR_DIGEST,
        "child descriptor digest drifted",
    )
    require(
        value.get("inventory_digest") == INVENTORY_DIGEST,
        "child inventory digest drifted",
    )
    require(value.get("build_identity") == build, "child build identity drifted")
    for key in ("request_count", "resource_count", "binding_count"):
        exact_uint(value.get(key), REQUEST_COUNT, key)
    exact_uint(value.get("logical_copy_bytes"), COPY_BYTES, "logical bytes")
    exact_uint(value.get("physical_copy_bytes"), COPY_BYTES, "physical bytes")
    exact_uint(value.get("worker_count"), workers, "worker count")
    require(value.get("resource_modes") == RESOURCE_MODES, "resource topology drifted")
    validate_schedule(value.get("parallel_copy_schedule"), workers)
    require(
        value.get("correctness")
        == {
            "passed": True,
            "payload_bytes_checked": COPY_BYTES,
            "entries_checked": REQUEST_COUNT,
        },
        "correctness contract drifted",
    )
    timing = value.get("timing")
    require(isinstance(timing, dict), "timing is missing")
    ready = exact_positive_uint(timing.get("ready_us"), "ready_us")
    phases = [
        exact_positive_uint(timing.get(key), key)
        for key in ("allocation_us", "source_us", "copy_us", "binding_us")
    ]
    require(timing.get("source_resolution_us") == phases[1], "source alias drifted")
    unattributed = timing.get("unattributed_us")
    require(
        type(unattributed) is int and 0 <= unattributed <= 4,
        "unattributed timing drifted",
    )
    exact_positive_uint(timing.get("teardown_us"), "teardown_us")
    require(abs(ready - sum(phases)) <= 4, "phase timing does not reconcile")
    for ms_key, us_key in (
        ("ready_wall_ms", "ready_us"),
        ("allocation_wall_ms", "allocation_us"),
        ("source_resolution_wall_ms", "source_us"),
        ("copy_wall_ms", "copy_us"),
        ("binding_wall_ms", "binding_us"),
        ("unattributed_wall_ms", "unattributed_us"),
        ("teardown_wall_ms", "teardown_us"),
    ):
        validate_wall_us_pair(
            timing,
            ms_key,
            us_key,
            positive=ms_key != "unattributed_wall_ms",
        )
    throughput = value.get("throughput")
    require(isinstance(throughput, dict), "throughput is missing")
    finite(throughput.get("ready_gbps_decimal"), "ready throughput", positive=True)
    finite(throughput.get("copy_gbps_decimal"), "copy throughput", positive=True)
    usage = value.get("rusage")
    require(isinstance(usage, dict), "rusage is missing")
    user = exact_nonnegative_uint(usage.get("user_cpu_us"), "user_cpu_us")
    system = exact_nonnegative_uint(usage.get("system_cpu_us"), "system_cpu_us")
    total = exact_positive_uint(usage.get("total_cpu_us"), "total_cpu_us")
    require(total == user + system, "total CPU does not reconcile")
    exact_uint(usage.get("timer_major_faults"), 0, "timer major faults")
    exact_nonnegative_uint(usage.get("timer_minor_faults"), "timer minor faults")
    finite(usage.get("cpu_per_wall"), "cpu_per_wall", positive=True)
    return value


def exact_nonnegative_uint(value: object, label: str) -> int:
    require(type(value) is int and value >= 0, f"{label} is not unsigned")
    return value


def exact_positive_uint(value: object, label: str) -> int:
    require(type(value) is int and value > 0, f"{label} is not positive unsigned")
    return value


def validate_wall_us_pair(
    timing: dict[str, object], ms_key: str, us_key: str, *, positive: bool = False
) -> float:
    observed = finite(timing.get(ms_key), ms_key, positive=positive)
    microseconds = exact_nonnegative_uint(timing.get(us_key), us_key)
    require(
        abs(observed - microseconds / 1000.0) <= 0.002,
        f"{ms_key}/{us_key} disagree",
    )
    return observed


def median(values: list[float | int]) -> float:
    require(bool(values), "median input is empty")
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[middle])
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def score_rows(rows: list[dict[str, object]]) -> dict[str, object]:
    require(len(rows) == 36, "scoring requires all 36 rows")
    by_cell = {(int(row["row_index"]), int(row["workers"])): row for row in rows}
    require(len(by_cell) == 36, "scoring cells are not unique")
    candidates = {}
    for workers in (1, 2, 6, 8, 12):
        differences, ratios, cpu_ratios = [], [], []
        before, after = [], []
        for row_index in range(1, 7):
            candidate = by_cell[(row_index, workers)]["result"]
            baseline = by_cell[(row_index, 4)]["result"]
            c_ready = candidate["timing"]["ready_us"]
            b_ready = baseline["timing"]["ready_us"]
            difference = b_ready - c_ready
            differences.append(difference)
            ratios.append(c_ready / b_ready)
            c_cpu = candidate["rusage"]["total_cpu_us"]
            b_cpu = baseline["rusage"]["total_cpu_us"]
            cpu_ratios.append(c_cpu / b_cpu)
            (before if row_index in BEFORE_ROWS[workers] else after).append(difference)
        d_value, d_before, d_after = median(differences), median(before), median(after)
        c_value = median(cpu_ratios)
        wins = sum(value > 0 for value in differences)
        gates = {
            "D_at_least_60000_us": d_value >= 60_000,
            "D_before_at_least_60000_us": d_before >= 60_000,
            "D_after_at_least_60000_us": d_after >= 60_000,
            "wins_at_least_5": wins >= 5,
            "C_at_most_1_10": c_value <= 1.10,
        }
        candidate_ready = [
            by_cell[(index, workers)]["result"]["timing"]["ready_us"]
            for index in range(1, 7)
        ]
        baseline_ready = [
            by_cell[(index, 4)]["result"]["timing"]["ready_us"] for index in range(1, 7)
        ]
        candidates[str(workers)] = {
            "differences_us": differences,
            "ready_ratios": ratios,
            "cpu_ratios": cpu_ratios,
            "D_us": d_value,
            "D_before_us": d_before,
            "D_after_us": d_after,
            "Q": median(ratios),
            "C": c_value,
            "wins": wins,
            "ratio_of_ready_medians": median(candidate_ready) / median(baseline_ready),
            "max_paired_cpu_ratio": max(cpu_ratios),
            "gates": gates,
            "qualifies": all(gates.values()),
        }
    qualified = [int(key) for key, value in candidates.items() if value["qualifies"]]
    winner = None
    near_best = []
    if qualified:
        d_max = max(candidates[str(worker)]["D_us"] for worker in qualified)
        near_best = [
            worker
            for worker in qualified
            if candidates[str(worker)]["D_us"] >= d_max - 10_000
        ]
        winner = min(near_best)
    return {
        "candidates": candidates,
        "qualified_workers": qualified,
        "near_best_workers": near_best,
        "winner": winner,
    }


def classify(
    *, contract_defect: bool, invalid: bool, winner: int | None, complete: bool
) -> str:
    if contract_defect:
        return "implementation_or_contract_defect"
    if invalid or not complete:
        return "inconclusive"
    return "GO-diagnostic" if winner is not None else "warm-worker-screen-miss"


def reserve_roots(manifest: dict[str, object]) -> None:
    require(not lexists(PACKET) and not lexists(WORK), "packet roots already exist")
    PACKET.parent.mkdir(parents=True, exist_ok=True)
    PACKET.mkdir(exist_ok=False)
    try:
        WORK.mkdir(exist_ok=False)
        write_json(PACKET / "manifest.json", manifest)
        write_json(
            PACKET / "order.json",
            {
                "schema": 1,
                "orders": [list(row) for row in ORDERS],
                "before_rows": {
                    str(key): list(value) for key, value in BEFORE_ROWS.items()
                },
            },
        )
        write_json(
            WORK / "reservation.json",
            {
                "schema": 1,
                "packet": str(PACKET.relative_to(ROOT)),
                "created_unix_ms": time.time_ns() // 1_000_000,
            },
        )
        fsync_dir(PACKET)
        fsync_dir(WORK)
        fsync_dir(PACKET.parent)
    except BaseException:
        raise


def parse_process_group_rows(text: str, pgid: int) -> list[int]:
    members = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        match = re.fullmatch(r"\s*(\d+)\s+(\d+)\s*", line)
        if match is None:
            raise RuntimeError(f"malformed ps row {line_number}: {line!r}")
        if int(match.group(2)) == pgid:
            members.append(int(match.group(1)))
    return sorted(members)


def process_group_members(pgid: int) -> list[int]:
    result = subprocess.run(
        ["ps", "-axo", "pid=,pgid="],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0 or result.stderr:
        raise RuntimeError(
            f"ps process-group inspection failed: rc={result.returncode} "
            f"stderr={result.stderr!r}"
        )
    return parse_process_group_rows(result.stdout, pgid)


def observe_group_disposition(pgid: int) -> dict[str, object]:
    samples = []
    for _ in range(50):
        captured_ns = time.monotonic_ns()
        try:
            members = process_group_members(pgid)
            error = None
        except (OSError, subprocess.SubprocessError, RuntimeError) as failure:
            members = None
            error = f"{type(failure).__name__}:{failure}"
        samples.append(
            {"captured_monotonic_ns": captured_ns, "members": members, "error": error}
        )
        if members == []:
            break
        time.sleep(0.02)
    return {
        "pgid": pgid,
        "samples": samples,
        "no_live_group_observed": samples[-1]["members"] == [],
    }


def authenticate_process_group(pid: int, getter=os.getpgid) -> int:
    try:
        pgid = getter(pid)
    except OSError as error:
        raise OwnershipLost(f"getpgid({pid}) failed: {error}") from error
    if pgid != pid:
        raise OwnershipLost(f"process-group ownership mismatch: pid={pid} pgid={pgid}")
    return pgid


def exact_wait_once(process: object, timeout: float) -> int:
    try:
        return process.wait(timeout=timeout)
    except ChildProcessError as error:
        raise OwnershipLost("exact wait ownership was lost") from error


def drain_close_and_reap_leader(process: subprocess.Popen[bytes]) -> list[str]:
    errors = []
    try:
        process.communicate()
    except (OSError, subprocess.SubprocessError, ValueError) as error:
        errors.append(f"communicate:{type(error).__name__}:{error}")
        for index, stream in enumerate((process.stdout, process.stderr)):
            if stream is None:
                continue
            try:
                stream.close()
            except (OSError, ValueError) as close_error:
                errors.append(
                    f"close{index}:{type(close_error).__name__}:{close_error}"
                )
        while process.returncode is None:
            try:
                exact_wait_once(process, 0.05)
            except subprocess.TimeoutExpired:
                continue
    return errors


def kill_once_then_reap(
    process: subprocess.Popen[bytes], pgid: int
) -> dict[str, object]:
    action = {
        "target_pgid": pgid,
        "signal": int(signal.SIGKILL),
        "attempted_monotonic_ns": time.monotonic_ns(),
        "succeeded": False,
        "error": None,
    }
    try:
        os.killpg(pgid, signal.SIGKILL)
        action["succeeded"] = True
    except OSError as error:
        action["error"] = f"{type(error).__name__}:{error}"
    action["completed_monotonic_ns"] = time.monotonic_ns()
    action["reap_errors"] = drain_close_and_reap_leader(process)
    action["leader_reaped"] = process.returncode is not None
    action["group_disposition"] = observe_group_disposition(pgid)
    return action


def bounded_child(
    command: list[str],
    env: dict[str, str],
    *,
    controller: PacketSignalController | None = None,
    event_start_sequence: int = 0,
    _faults: set[str] | None = None,
    _pgid_getter=os.getpgid,
    _on_acquired=None,
    _exception_cleanup_observer=None,
) -> dict[str, object]:
    faults = set() if _faults is None else set(_faults)

    def consume_fault(name: str) -> bool:
        if name not in faults:
            return False
        faults.remove(name)
        return True

    attempted_ns = time.monotonic_ns()
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
        completed_ns = time.monotonic_ns()
        return {
            "pid": None,
            "pgid": None,
            "attempted_monotonic_ns": attempted_ns,
            "acquired_monotonic_ns": None,
            "completed_monotonic_ns": completed_ns,
            "spawn_error": f"{type(error).__name__}:{error}",
            "returncode": None,
            "stdout": b"",
            "stderr": b"",
            "output_overflow": False,
            "interrupted": isinstance(error, KeyboardInterrupt),
            "wait_errors": [],
            "poll_errors": [],
            "pipe_errors": [],
            "termination_errors": [],
            "cleanup_actions": [],
            "cleanup_signal_sent": False,
            "group_disposition": None,
            "reaped": False,
            "wall_ns": completed_ns - attempted_ns,
        }
    acquired_ns = time.monotonic_ns()
    try:
        pgid = authenticate_process_group(process.pid, _pgid_getter)
    except OwnershipLost as error:
        try:
            cleanup_errors = drain_close_and_reap_leader(process)
        except BaseException as cleanup_error:
            error.add_note(
                f"leader cleanup failed: {type(cleanup_error).__name__}:{cleanup_error}"
            )
        else:
            if cleanup_errors:
                error.add_note(f"leader cleanup diagnostics: {cleanup_errors}")
        raise
    if _on_acquired is not None:
        try:
            _on_acquired(
                {
                    "pid": process.pid,
                    "pgid": pgid,
                    "acquired_monotonic_ns": acquired_ns,
                }
            )
        except BaseException as error:
            try:
                action = kill_once_then_reap(process, pgid)
            except BaseException as cleanup_error:
                error.add_note(
                    f"owned acquisition cleanup failed: "
                    f"{type(cleanup_error).__name__}:{cleanup_error}"
                )
            else:
                if _exception_cleanup_observer is not None:
                    try:
                        _exception_cleanup_observer(dict(action))
                    except BaseException as observer_error:
                        error.add_note(
                            f"cleanup observer failed: "
                            f"{type(observer_error).__name__}:{observer_error}"
                        )
                error.add_note(f"owned acquisition cleanup: {action}")
            raise
    buffers = [bytearray(), bytearray()]
    overflow = [False, False]
    pipe_errors: list[str] = []

    def drain(stream: object, index: int) -> None:
        try:
            while True:
                if consume_fault(f"read_{index}"):
                    raise OSError("injected drain read failure")
                chunk = stream.read(64 * 1024)
                if not chunk:
                    return
                room = MAX_OUTPUT_BYTES - len(buffers[index])
                if room > 0:
                    buffers[index].extend(chunk[:room])
                if len(chunk) > room:
                    overflow[index] = True
        except Exception as error:
            pipe_errors.append(f"pipe{index}:{type(error).__name__}:{error}")
        finally:
            try:
                stream.close()
            except (OSError, ValueError) as error:
                pipe_errors.append(f"drain_close{index}:{type(error).__name__}:{error}")

    streams = (process.stdout, process.stderr)
    threads = []
    thread_start_failed = False
    for index, stream in enumerate(streams):
        if stream is None:
            pipe_errors.append(f"pipe{index}:unavailable")
            continue
        thread = threading.Thread(target=drain, args=(stream, index), daemon=True)
        try:
            if consume_fault(f"thread_start_{index}"):
                raise RuntimeError("injected thread start failure")
            thread.start()
            threads.append(thread)
        except RuntimeError as error:
            pipe_errors.append(f"thread{index}:{type(error).__name__}:{error}")
            thread_start_failed = True
    termination_errors = []
    cleanup_actions = []
    cleanup_requested = thread_start_failed
    cleanup_signal_sent = False

    def kill_owned_group(reason: str) -> None:
        nonlocal cleanup_requested, cleanup_signal_sent
        cleanup_requested = False
        require(not cleanup_signal_sent, "duplicate owned-group cleanup signal")
        cleanup_signal_sent = True
        action = {
            "reason": reason,
            "target_pgid": pgid,
            "signal": int(signal.SIGKILL),
            "attempted_monotonic_ns": time.monotonic_ns(),
            "succeeded": False,
            "error": None,
        }
        try:
            os.killpg(pgid, signal.SIGKILL)
            action["succeeded"] = True
        except OSError as error:
            action["error"] = f"{type(error).__name__}:{error}"
            termination_errors.append(action["error"])
        action["completed_monotonic_ns"] = time.monotonic_ns()
        cleanup_actions.append(action)

    interrupted = False
    wait_errors = []
    poll_errors = []
    while True:
        controller_events = (
            controller.since(event_start_sequence) if controller is not None else []
        )
        if controller_events:
            cleanup_requested = True
        if pipe_errors:
            cleanup_requested = True
        if cleanup_requested and not cleanup_signal_sent:
            kill_owned_group(
                "operator-signal" if controller_events else "drain-or-thread-failure"
            )
        try:
            if consume_fault("wait_once"):
                raise OSError("injected wait failure")
            if consume_fault("ownership_lost"):
                raise OwnershipLost("injected ownership-lost wait")
            returncode = exact_wait_once(process, 0.05)
            break
        except subprocess.TimeoutExpired:
            continue
        except OwnershipLost as error:
            close_errors = []
            for index, stream in enumerate(streams):
                if stream is None:
                    continue
                try:
                    stream.close()
                except (OSError, ValueError) as close_error:
                    close_errors.append(
                        f"close{index}:{type(close_error).__name__}:{close_error}"
                    )
            for thread in threads:
                thread.join(timeout=0.2)
            if close_errors:
                error.add_note(f"ownership-lost pipe closure: {close_errors}")
            raise
        except KeyboardInterrupt:
            interrupted = True
            cleanup_requested = True
        except (OSError, subprocess.SubprocessError) as error:
            wait_errors.append(f"{type(error).__name__}:{error}")
        except Exception as error:
            wait_errors.append(f"unexpected:{type(error).__name__}:{error}")
            cleanup_requested = True
    for index, thread in enumerate(threads):
        try:
            if consume_fault(f"join_{index}"):
                raise RuntimeError("injected join failure")
            thread.join()
        except RuntimeError as error:
            pipe_errors.append(f"join{index}:{type(error).__name__}:{error}")
            thread.join()
    for index, stream in enumerate(streams):
        if stream is None:
            continue
        try:
            if consume_fault(f"close_{index}"):
                raise OSError("injected close failure")
            stream.close()
        except (OSError, ValueError) as error:
            pipe_errors.append(f"close{index}:{type(error).__name__}:{error}")
    completed_ns = time.monotonic_ns()
    group_disposition = observe_group_disposition(pgid)
    return {
        "pid": process.pid,
        "pgid": pgid,
        "attempted_monotonic_ns": attempted_ns,
        "acquired_monotonic_ns": acquired_ns,
        "completed_monotonic_ns": completed_ns,
        "spawn_error": None,
        "returncode": returncode,
        "stdout": bytes(buffers[0]),
        "stderr": bytes(buffers[1]),
        "output_overflow": any(overflow),
        "interrupted": interrupted,
        "wait_errors": wait_errors,
        "poll_errors": poll_errors,
        "pipe_errors": pipe_errors,
        "termination_errors": termination_errors,
        "cleanup_actions": cleanup_actions,
        "cleanup_signal_sent": cleanup_signal_sent,
        "group_disposition": group_disposition,
        "reaped": process.returncode is not None,
        "wall_ns": completed_ns - acquired_ns,
    }


def completion_record(
    stem: str, outcome: dict[str, object], residency_proved_ns: int
) -> dict[str, object]:
    acquired = outcome["acquired_monotonic_ns"]
    return {
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
        "cleanup_actions": outcome["cleanup_actions"],
        "cleanup_signal_sent": outcome["cleanup_signal_sent"],
        "group_disposition": outcome["group_disposition"],
        "reaped": outcome["reaped"],
        "attempted_monotonic_ns": outcome["attempted_monotonic_ns"],
        "acquired_monotonic_ns": acquired,
        "completed_monotonic_ns": outcome["completed_monotonic_ns"],
        "residency_to_acquired_ns": (
            acquired - residency_proved_ns if type(acquired) is int else None
        ),
    }


def cooldown(activity_ns: int) -> dict[str, int]:
    started = time.monotonic_ns()
    remaining = activity_ns + COOLDOWN_NS - started
    if remaining > 0:
        time.sleep(remaining / 1e9)
    ended = time.monotonic_ns()
    if ended - activity_ns < COOLDOWN_NS:
        raise Inconclusive("30-second cooldown was short")
    return {
        "prior_activity_ns": activity_ns,
        "wait_started_ns": started,
        "wait_ended_ns": ended,
        "elapsed_since_activity_ns": ended - activity_ns,
    }


def verify_binary_identity(manifest: dict[str, object], env: dict[str, str]) -> None:
    expected = manifest["binary_bytes"]
    require(isinstance(expected, dict), "manifest binary identity is malformed")
    stamp = expected.get("descriptor_stamp")
    require(isinstance(stamp, dict), "manifest binary stamp is malformed")
    size = stamp.get("size_bytes")
    require(type(size) is int and size > 0, "manifest binary size is malformed")
    observed = descriptor_bound_sha256(BINARY, size)
    require(observed == expected, "qwen-bench byte identity drifted")
    build = parse_json_bytes(
        command_output([str(BINARY), "build-info", "--output", "json"], env).encode(),
        "qwen-bench identity recheck",
    )
    require(build == manifest["build_identity"], "qwen-bench semantic stamp drifted")


def conditioning_failure_reasons(record: dict[str, object]) -> list[str]:
    reasons = list(record.get("operation_errors", []))
    host_before = record.get("host_before_conditioning")
    vm_before = record.get("vm_before_conditioning")
    host_launch = record.get("host_before_launch")
    residency = record.get("residency_before")
    interval = record.get("conditioning_interval")
    if not isinstance(host_before, dict) or host_before.get("valid") is not True:
        reasons.append("host_invalid_before_conditioning")
    if not isinstance(vm_before, dict) or vm_before.get("errors"):
        reasons.append("vm_invalid_before_conditioning")
    if (
        not isinstance(residency, dict)
        or residency.get("resident_pages") != MODEL_PAGES
    ):
        reasons.append("incomplete_residency_before_launch")
    if not isinstance(host_launch, dict) or host_launch.get("valid") is not True:
        reasons.append("host_invalid_before_launch")
    if isinstance(interval, dict):
        reasons.extend(interval.get("failure_reasons", []))
    elif record.get("residency_proved_ns") is not None:
        reasons.append("conditioning_interval_unavailable")
    return sorted(set(reasons))


def collect_post_exit_evidence(
    stem: str,
    vm_launch: dict[str, object],
    host_after: dict[str, object],
    vm_after: dict[str, object],
    residency_probe,
) -> dict[str, object]:
    residency_after = None
    operation_errors = []
    try:
        residency_after = residency_probe()
    except (OSError, RuntimeError) as error:
        operation_errors.append(f"post_exit_residency:{type(error).__name__}:{error}")
    return {
        "schema": 1,
        "stem": stem,
        "host_after_exit": host_after,
        "vm_after_exit": vm_after,
        "residency_after": residency_after,
        "child_interval": vm_interval("child", vm_launch, vm_after),
        "operation_errors": operation_errors,
    }


def launch_delay_failure(residency_proved_ns: int, acquired_ns: object) -> str | None:
    if type(acquired_ns) is not int:
        return None
    if acquired_ns - residency_proved_ns > LAUNCH_LIMIT_NS:
        return "launch_exceeded_five_seconds"
    return None


def persist_attempt_signal_closure(
    path: Path,
    stem: str,
    start_sequence: int,
    controller: PacketSignalController,
    attempt_digest: str,
) -> list[dict[str, int]]:
    events = tuple(controller.since(start_sequence))
    end_sequence = events[-1]["sequence"] if events else start_sequence
    append_jsonl(
        path,
        {
            "schema": 1,
            "stem": stem,
            "signal_start_sequence": start_sequence,
            "signal_end_sequence": end_sequence,
            "events": list(events),
            "invalid": bool(events),
            "attempt_sha256": attempt_digest,
        },
    )
    return list(events)


def execute_launched_attempt(
    stem: str,
    row_index: int,
    position: int,
    workers: int,
    env: dict[str, str],
    manifest: dict[str, object],
    residency_proved_ns: int,
    host_launch: dict[str, object],
    vm_launch: dict[str, object],
    controller: PacketSignalController,
    attempt_start_sequence: int,
) -> tuple[dict[str, object], int, str | None, list[str]]:
    controller.require_quiet_since(attempt_start_sequence, f"{stem} launch")
    command = child_command(workers)
    append_jsonl(
        PACKET / "lifecycle.jsonl",
        {
            "schema": 1,
            "event": "launch",
            "stem": stem,
            "row_index": row_index,
            "position": position,
            "workers": workers,
            "command": command,
            "conditioning_sha256": sha256_file(WORK / f"{stem}.conditioning.json"),
            "residency_proved_ns": residency_proved_ns,
            "monotonic_ns": time.monotonic_ns(),
        },
    )
    outcome = bounded_child(
        command,
        env,
        controller=controller,
        event_start_sequence=attempt_start_sequence,
        _on_acquired=lambda acquired: append_jsonl(
            PACKET / "lifecycle.jsonl",
            {
                "schema": 1,
                "event": "acquired",
                "stem": stem,
                **acquired,
            },
        ),
    )
    stdout = outcome["stdout"]
    stderr = outcome["stderr"]
    require(
        type(stdout) is bytes and type(stderr) is bytes, "child output is malformed"
    )
    write_exclusive(WORK / f"{stem}.stdout", stdout)
    write_exclusive(WORK / f"{stem}.stderr", stderr)
    completion = completion_record(stem, outcome, residency_proved_ns)
    append_jsonl(PACKET / "lifecycle.jsonl", completion)
    host_after = capture_host_state()
    vm_after = capture_vm_state()
    post = collect_post_exit_evidence(
        stem,
        vm_launch,
        host_after,
        vm_after,
        lambda: mincore_residency(MODEL, manifest["model_file_identity"], MODEL_SIZE),
    )
    write_json(WORK / f"{stem}.post.json", post)
    parse_error = None
    result = None
    resources = None
    if outcome["pid"] is not None and outcome["returncode"] == 0:
        try:
            resources = parse_time_output(stderr.decode("utf-8"))
            parsed = parse_json_bytes(stdout, stem)
            result = validate_result(parsed, workers, manifest["build_identity"])
        except (UnicodeDecodeError, ContractDefect) as error:
            parse_error = f"{type(error).__name__}:{error}"
    validity = list(post["child_interval"]["failure_reasons"])
    validity.extend(post["operation_errors"])
    if host_after.get("valid") is not True:
        validity.append("host_invalid_after_exit")
    residency_after = post["residency_after"]
    if not isinstance(residency_after, dict) or (
        residency_after.get("resident_pages") != MODEL_PAGES
    ):
        validity.append("incomplete_residency_after_exit")
    if outcome["spawn_error"] is not None:
        validity.append(f"spawn_error={outcome['spawn_error']}")
    if outcome["returncode"] != 0:
        validity.append(f"returncode={outcome['returncode']}")
    if outcome["reaped"] is not (outcome["pid"] is not None):
        validity.append("exact_child_reap_invalid")
    if outcome["interrupted"]:
        validity.append("operator_interrupt_during_child_wait")
    validity.extend(f"wait_error={value}" for value in outcome["wait_errors"])
    validity.extend(f"poll_error={value}" for value in outcome["poll_errors"])
    validity.extend(f"pipe_error={value}" for value in outcome["pipe_errors"])
    validity.extend(
        f"termination_error={value}" for value in outcome["termination_errors"]
    )
    if outcome["cleanup_signal_sent"] is True and len(outcome["cleanup_actions"]) != 1:
        validity.append("cleanup_action_cardinality_invalid")
    if outcome["cleanup_signal_sent"] is False and outcome["cleanup_actions"]:
        validity.append("cleanup_action_without_signal")
    disposition = outcome["group_disposition"]
    if not isinstance(disposition, dict) or not disposition.get(
        "no_live_group_observed"
    ):
        validity.append("process_group_still_observed")
    acquired_ns = outcome["acquired_monotonic_ns"]
    launch_delay_ns = (
        acquired_ns - residency_proved_ns if type(acquired_ns) is int else None
    )
    launch_failure = launch_delay_failure(residency_proved_ns, acquired_ns)
    if launch_failure is not None:
        validity.append(launch_failure)
    if outcome["output_overflow"]:
        validity.append("bounded_output_exceeded")
    if resources is not None:
        validity.extend(process_resource_failure_reasons(resources))
    attempt_events = controller.since(attempt_start_sequence)
    if attempt_events:
        validity.append("deferred_operator_signal")
    attempt = {
        "schema": 1,
        "stem": stem,
        "row_index": row_index,
        "position": position,
        "workers": workers,
        "command": command,
        "pid": outcome["pid"],
        "pgid": outcome["pgid"],
        "spawn_error": outcome["spawn_error"],
        "returncode": outcome["returncode"],
        "reaped": outcome["reaped"],
        "process_wall_ns": outcome["wall_ns"],
        "output_overflow": outcome["output_overflow"],
        "wait_errors": outcome["wait_errors"],
        "poll_errors": outcome["poll_errors"],
        "pipe_errors": outcome["pipe_errors"],
        "termination_errors": outcome["termination_errors"],
        "cleanup_actions": outcome["cleanup_actions"],
        "cleanup_signal_sent": outcome["cleanup_signal_sent"],
        "group_disposition": outcome["group_disposition"],
        "signal_start_sequence": attempt_start_sequence,
        "signal_events_before_attempt_fsync": attempt_events,
        "residency_proved_ns": residency_proved_ns,
        "launch_attempted_ns": outcome["attempted_monotonic_ns"],
        "launch_acquired_ns": acquired_ns,
        "residency_to_acquired_ns": launch_delay_ns,
        "host_before_launch_captured_ns": host_launch["captured_monotonic_ns"],
        "vm_before_launch_captured_ns": vm_launch["captured_monotonic_ns"],
        "stdout_sha256": sha256_file(WORK / f"{stem}.stdout"),
        "stderr_sha256": sha256_file(WORK / f"{stem}.stderr"),
        "conditioning_sha256": sha256_file(WORK / f"{stem}.conditioning.json"),
        "post_sha256": sha256_file(WORK / f"{stem}.post.json"),
        "process_resources": resources,
        "process_page_faults": (
            process_page_faults_advisory(resources) if resources is not None else None
        ),
        "parse_error": parse_error,
        "validity_reasons": sorted(set(validity)),
        "result": result,
    }
    append_jsonl(PACKET / "attempts.jsonl", attempt)
    closure_events = persist_attempt_signal_closure(
        PACKET / "attempt-signal-closures.jsonl",
        stem,
        attempt_start_sequence,
        controller,
        sha256_bytes(json_bytes(attempt)),
    )
    if closure_events and "deferred_operator_signal" not in validity:
        validity.append("deferred_operator_signal")
    return attempt, time.monotonic_ns(), parse_error, validity


def run_one(
    row_index: int,
    position: int,
    workers: int,
    env: dict[str, str],
    manifest: dict[str, object],
    activity_ns: int,
    controller: PacketSignalController,
) -> tuple[dict[str, object], int]:
    stem = f"r{row_index}-p{position}-w{workers}"
    for suffix in ("stdout", "stderr", "conditioning.json", "post.json"):
        require(not (WORK / f"{stem}.{suffix}").exists(), "attempt artifact reuse")
    controller.require_quiet_since(0, f"{stem} cooldown")
    wait = cooldown(activity_ns)
    controller.require_quiet_since(0, f"{stem} conditioning")
    verify_binary_identity(manifest, env)
    host_before = capture_host_state()
    vm_before = capture_vm_state()
    conditioning = None
    residency_before = None
    residency_proved_ns = None
    host_launch = None
    vm_launch = None
    interval_conditioning = None
    operation_errors = []
    if host_before.get("valid") is True and not vm_before.get("errors"):
        try:
            conditioning = sequential_condition(
                MODEL, manifest["model_file_identity"], MODEL_SIZE
            )
        except (OSError, RuntimeError) as error:
            operation_errors.append(f"conditioning:{type(error).__name__}:{error}")
        if conditioning is not None:
            try:
                residency_before = mincore_residency(
                    MODEL, manifest["model_file_identity"], MODEL_SIZE
                )
                residency_proved_ns = time.monotonic_ns()
            except (OSError, RuntimeError) as error:
                operation_errors.append(
                    f"pre_spawn_residency:{type(error).__name__}:{error}"
                )
        if residency_before is not None:
            if (
                residency_before["page_size"] != PAGE_SIZE
                or residency_before["total_pages"] != MODEL_PAGES
            ):
                operation_errors.append("pre_spawn_residency:page_geometry_drift")
            host_launch = capture_host_state()
            vm_launch = capture_vm_state()
            interval_conditioning = vm_interval("conditioning", vm_before, vm_launch)
    conditioning_record = {
        "schema": 1,
        "stem": stem,
        "cooldown": wait,
        "host_before_conditioning": host_before,
        "vm_before_conditioning": vm_before,
        "conditioning": conditioning,
        "residency_before": residency_before,
        "host_before_launch": host_launch,
        "vm_before_launch": vm_launch,
        "conditioning_interval": interval_conditioning,
        "residency_proved_ns": residency_proved_ns,
        "operation_errors": operation_errors,
    }
    write_json(WORK / f"{stem}.conditioning.json", conditioning_record)
    reasons = conditioning_failure_reasons(conditioning_record)
    if reasons:
        raise Inconclusive(f"{stem} conditioning invalid: {reasons}")
    require(type(residency_proved_ns) is int, "residency proof timestamp is absent")
    require(isinstance(vm_launch, dict), "pre-launch VM sample is absent")
    require(isinstance(host_launch, dict), "pre-launch host sample is absent")
    controller.require_quiet_since(0, f"{stem} launch")
    attempt_start_sequence = controller.sequence()
    attempt, activity, parse_error, validity = execute_launched_attempt(
        stem,
        row_index,
        position,
        workers,
        env,
        manifest,
        residency_proved_ns,
        host_launch,
        vm_launch,
        controller,
        attempt_start_sequence,
    )
    if controller.since(attempt_start_sequence) and (
        "deferred_operator_signal" not in validity
    ):
        validity.append("deferred_operator_signal")
    if parse_error is not None:
        raise ContractDefect(f"{stem} result contract failed: {parse_error}")
    if validity:
        raise Inconclusive(f"{stem} validity failed: {validity}")
    require(
        attempt["result"] is not None and attempt["process_resources"] is not None,
        "valid row lacks parsed evidence",
    )
    return attempt, activity


def preflight_time(env: dict[str, str]) -> dict[str, object]:
    result = subprocess.run(
        ["/usr/bin/time", "-l", "/usr/bin/true"],
        cwd=ROOT,
        env=env,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    require(result.returncode == 0 and not result.stdout, "/usr/bin/time probe failed")
    text = result.stderr.decode("utf-8")
    return {"stderr": text, "parsed": parse_time_output(text)}


def preflight() -> tuple[dict[str, object], dict[str, str]]:
    require(not lexists(PACKET) and not lexists(WORK), "packet roots already exist")
    run_self_tests()
    optimized = subprocess.run(
        [sys.executable, "-O", str(RUNNER), "--self-test-child"],
        cwd=ROOT,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    require(
        optimized.returncode == 0 and optimized.stdout == b"self-test-child: ok\n",
        "optimized self-test failed",
    )
    require(len(optimized.stderr) == 0, "optimized self-test wrote stderr")
    env, removed = normalized_environment()
    require(not lexists(PACKET) and not lexists(WORK), "packet roots already exist")
    gates = fresh_build_and_tests(env)
    source = source_and_build_identity(env)
    binary_bytes = descriptor_bound_sha256(BINARY)
    identity = file_identity(MODEL)
    require(identity["size_bytes"] == MODEL_SIZE, "model size drifted")
    require(os.sysconf("SC_PAGE_SIZE") == PAGE_SIZE, "host page size drifted")
    require(sha256_file(MODEL) == MODEL_SHA256, "model SHA-256 drifted")
    schedules = describe_schedules(env, source["build_identity"])
    time_probe = preflight_time(env)
    host = capture_host_state()
    vm = capture_vm_state()
    require(host.get("valid") is True, "preflight host is invalid")
    require(not vm.get("errors"), "preflight VM capture is invalid")
    forensic_v0641 = authenticate_v0641_forensics()
    manifest = {
        "schema": 1,
        "protocol": "v0642-a3b-pread-worker-screen-fault-repair",
        "created_unix_ms": time.time_ns() // 1_000_000,
        "source": source,
        "source_commit": source["head"],
        "implementation_parent": IMPLEMENTATION_PARENT,
        "build_identity": source["build_identity"],
        "binary_bytes": binary_bytes,
        "model": str(MODEL),
        "model_size_bytes": MODEL_SIZE,
        "model_pages": MODEL_PAGES,
        "model_sha256": MODEL_SHA256,
        "model_file_identity": identity,
        "descriptor_digest": DESCRIPTOR_DIGEST,
        "inventory_digest": INVENTORY_DIGEST,
        "normalized_environment": env,
        "removed_environment": removed,
        "commands": {str(worker): child_command(worker) for worker in WORKERS},
        "orders": [list(row) for row in ORDERS],
        "schedules": schedules,
        "build_and_test_gates": gates,
        "time_probe": time_probe,
        "host_preflight": host,
        "vm_preflight": vm,
        "forensic_v0641": forensic_v0641,
        "packet_input_sha256": {
            str(PREREG): sha256_file(PREREG),
            str(RUNNER): sha256_file(RUNNER),
            str(BINARY): binary_bytes["sha256"],
            str(MODEL): MODEL_SHA256,
        },
        "attempt_count": 36,
        "retry_count": 0,
        "authority": "none",
    }
    return manifest, env


def verify_final_identity(manifest: dict[str, object], env: dict[str, str]) -> None:
    final_source = source_and_build_identity(env)
    require(final_source == manifest["source"], "final source/build identity drifted")
    verify_binary_identity(manifest, env)
    require(
        file_identity(MODEL) == manifest["model_file_identity"],
        "final model descriptor drifted",
    )


def seal(decision: dict[str, object]) -> None:
    write_json(PACKET / "decision.json", decision)
    write_exclusive(
        PACKET / "decision.sha256",
        (sha256_file(PACKET / "decision.json") + "\n").encode("ascii"),
    )
    excluded = {
        PACKET / "artifact-inventory.sha256",
        PACKET / "packet-complete.json",
        PACKET / "packet-complete.sha256",
    }
    members = []
    for root in (PACKET, WORK):
        for path in sorted(root.iterdir()):
            if path.is_file() and path not in excluded:
                members.append((path, sha256_file(path)))
    inventory = b"".join(
        f"{digest}  {path.relative_to(ROOT)}\n".encode("ascii")
        for path, digest in members
    )
    write_exclusive(PACKET / "artifact-inventory.sha256", inventory)
    inventory_digest = sha256_file(PACKET / "artifact-inventory.sha256")
    complete = {
        "schema": 1,
        "decision_sha256": sha256_file(PACKET / "decision.json"),
        "inventory_sha256": inventory_digest,
        "inventory_members": len(members),
    }
    write_json(PACKET / "packet-complete.json", complete)
    write_exclusive(
        PACKET / "packet-complete.sha256",
        (sha256_file(PACKET / "packet-complete.json") + "\n").encode("ascii"),
    )
    fsync_dir(PACKET)
    fsync_dir(WORK)
    fsync_dir(PACKET.parent)


def execute() -> None:
    manifest, env = preflight()
    controller = PacketSignalController(restore_on_exit=False)
    controller.install()
    reserve_roots(manifest)
    rows = []
    contract_error = None
    invalid_error = None
    analysis = None
    activity = time.monotonic_ns()
    try:
        controller.require_quiet_since(0, "packet execution")
        for row_index, order in enumerate(ORDERS, 1):
            for position, workers in enumerate(order, 1):
                attempt, activity = run_one(
                    row_index,
                    position,
                    workers,
                    env,
                    manifest,
                    activity,
                    controller,
                )
                rows.append(attempt)
        analysis = score_rows(rows)
    except ContractDefect as error:
        contract_error = f"{type(error).__name__}:{error}"
    except (Inconclusive, KeyboardInterrupt) as error:
        invalid_error = f"{type(error).__name__}:{error}"
    try:
        verify_final_identity(manifest, env)
    except ContractDefect as error:
        contract_error = contract_error or f"{type(error).__name__}:{error}"
    cutoff = controller.final_cutoff(PACKET / "packet-signal-cutoff.json")
    if cutoff["authority_event_count"] and contract_error is None:
        invalid_error = invalid_error or "Inconclusive:operator signal before cutoff"
    winner = analysis.get("winner") if analysis is not None else None
    attempts_path = PACKET / "attempts.jsonl"
    attempted_count = (
        len(attempts_path.read_text(encoding="utf-8").splitlines())
        if attempts_path.is_file()
        else 0
    )
    status = classify(
        contract_defect=contract_error is not None,
        invalid=invalid_error is not None,
        winner=winner,
        complete=len(rows) == 36 and analysis is not None,
    )
    decision = {
        "schema": 1,
        "status": status,
        "authority": "none",
        "force_authorized": False,
        "successor_authorization": "none",
        "implementation_parent": IMPLEMENTATION_PARENT,
        "source_commit": manifest["source_commit"],
        "completed_attempts": attempted_count,
        "expected_attempts": 36,
        "contract_error": contract_error,
        "invalid_error": invalid_error,
        "analysis": analysis,
        "signal_cutoff_sha256": sha256_file(PACKET / "packet-signal-cutoff.json"),
        "signal_log_sha256": cutoff["signal_log_sha256"],
        "closure": "GO-diagnostic"
        if status == "GO-diagnostic"
        else "KILL"
        if status == "warm-worker-screen-miss"
        else status,
        "scope": "exact-asset-host-cache-warm-diagnostic-only",
    }
    seal(decision)
    print(json.dumps(decision, sort_keys=True))


def fixture_result(ready: int, cpu: int) -> dict[str, object]:
    return {"timing": {"ready_us": ready}, "rusage": {"total_cpu_us": cpu}}


def synthetic_rows(
    differences: dict[int, list[int]], cpu_ratios: dict[int, float]
) -> list[dict[str, object]]:
    rows = []
    for row_index, order in enumerate(ORDERS, 1):
        baseline_ready = 1_000_000
        baseline_cpu = 1_000_000
        for position, workers in enumerate(order, 1):
            if workers == 4:
                result = fixture_result(baseline_ready, baseline_cpu)
            else:
                result = fixture_result(
                    baseline_ready - differences[workers][row_index - 1],
                    round(baseline_cpu * cpu_ratios[workers]),
                )
            rows.append(
                {
                    "row_index": row_index,
                    "position": position,
                    "workers": workers,
                    "result": result,
                }
            )
    return rows


def expect_contract(function: object, *args: object, **kwargs: object) -> None:
    try:
        function(*args, **kwargs)
    except ContractDefect:
        return
    raise RuntimeError("expected ContractDefect")


def expect_inconclusive(function: object, *args: object, **kwargs: object) -> None:
    try:
        function(*args, **kwargs)
    except Inconclusive:
        return
    raise RuntimeError("expected Inconclusive")


def expect_ownership_lost(function: object, *args: object, **kwargs: object) -> None:
    try:
        function(*args, **kwargs)
    except OwnershipLost:
        return
    raise RuntimeError("expected OwnershipLost")


def expect_runtime(function: object, *args: object, **kwargs: object) -> None:
    try:
        function(*args, **kwargs)
    except RuntimeError:
        return
    raise RuntimeError("expected RuntimeError")


def run_self_tests() -> None:
    initial_handlers = {signum: signal.getsignal(signum) for signum in OPERATOR_SIGNALS}
    initial_mask = signal.pthread_sigmask(signal.SIG_BLOCK, [])
    require(set(ORDERS) and len(ORDERS) == 6, "order count self-test")
    for workers in WORKERS:
        require(
            sorted(row.index(workers) + 1 for row in ORDERS) == list(range(1, 7)),
            "position balance self-test",
        )
    predecessor_counts = {}
    for order in ORDERS:
        for left, right in zip(order, order[1:]):
            predecessor_counts[(left, right)] = (
                predecessor_counts.get((left, right), 0) + 1
            )
    require(
        len(predecessor_counts) == 30 and set(predecessor_counts.values()) == {1},
        "directed predecessor balance self-test",
    )
    for workers, expected in BEFORE_ROWS.items():
        observed = tuple(
            index
            for index, order in enumerate(ORDERS, 1)
            if order.index(workers) < order.index(4)
        )
        require(
            observed == expected and len(set(range(1, 7)) - set(observed)) == 3,
            "before/after self-test",
        )
    require(
        median([9, 1, 5]) == 5 and median([9, 1, 5, 3, 7, 11]) == 6,
        "median behavior self-test",
    )
    require(median([1, 2]) == 1.5, "even median self-test")
    for workers, frozen in SCHEDULES.items():
        require(
            len(frozen["digest"]) == 64
            and re.fullmatch(r"[0-9a-f]{64}", frozen["digest"]) is not None,
            "schedule digest constant self-test",
        )
        require(
            len(frozen["cuts"]) == workers - 1
            and sum(frozen["task_counts"]) == REQUEST_COUNT
            and sum(frozen["worker_bytes"]) == COPY_BYTES,
            "schedule constants self-test",
        )
    constants_digest = sha256_bytes(
        json.dumps(SCHEDULES, sort_keys=True, separators=(",", ":")).encode("ascii")
    )
    require(
        constants_digest == SCHEDULE_CONSTANTS_SHA256,
        "aggregate schedule constants digest self-test",
    )
    expect_contract(parse_json_bytes, b"{", "malformed fixture")
    expect_contract(parse_json_bytes, b'{"x":1,"x":2}', "duplicate fixture")
    expect_contract(parse_json_bytes, b'{"x":NaN}', "nonfinite fixture")
    expect_contract(parse_json_bytes, b'{"x":1e999}', "positive overflow fixture")
    expect_contract(parse_json_bytes, b'{"x":-1e999}', "negative overflow fixture")
    require(
        parse_json_bytes(b'{"x":1.25e2}', "finite exponent fixture") == {"x": 125.0},
        "finite exponent JSON self-test",
    )
    expect_contract(parse_json_bytes, b'{"x":1} {"y":2}', "multiple fixture")
    require(
        parse_json_bytes(b'{"x":1}\n', "strict JSON fixture") == {"x": 1},
        "strict JSON acceptance self-test",
    )
    expect_contract(validate_schedule, [], 1)
    cargo_pass = (
        "test result: ok. 11 passed; 0 failed; 0 ignored; 0 measured; "
        "3 filtered out; finished in 0.01s\n"
    )
    require(
        parse_cargo_test_summary(cargo_pass)["passed"] == 11,
        "Cargo summary pass self-test",
    )
    expect_contract(
        parse_cargo_test_summary,
        "test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; "
        "14 filtered out; finished in 0.00s\n",
    )
    expect_contract(parse_cargo_test_summary, "cargo succeeded without a summary\n")
    gate_disposition = {
        "pgid": 321,
        "samples": [
            {
                "captured_monotonic_ns": 1,
                "members": [],
                "error": None,
            }
        ],
        "no_live_group_observed": True,
    }
    gate_outcome = {
        "pid": 321,
        "pgid": 321,
        "cleanup_signal_sent": False,
        "cleanup_actions": [],
        "group_disposition": gate_disposition,
    }
    gate_evidence = validate_gate_lifecycle(gate_outcome)
    require(
        gate_evidence["cleanup_action_cardinality"] == 0
        and gate_evidence["group_disposition"] == gate_disposition,
        "normal zero-cleanup gate lifecycle self-test",
    )
    missing_disposition = dict(gate_outcome)
    missing_disposition["group_disposition"] = None
    expect_contract(validate_gate_lifecycle, missing_disposition)
    false_disposition = dict(gate_outcome)
    false_disposition["group_disposition"] = {
        **gate_disposition,
        "no_live_group_observed": False,
    }
    expect_contract(validate_gate_lifecycle, false_disposition)
    error_disposition = dict(gate_outcome)
    error_disposition["group_disposition"] = {
        **gate_disposition,
        "samples": [{"captured_monotonic_ns": 1, "members": None, "error": "ps"}],
    }
    expect_contract(validate_gate_lifecycle, error_disposition)
    time_fixture = "        1.00 real         0.20 user         0.30 sys\n" + "".join(
        f"        0  {label}\n" for label in TIME_LABELS
    )
    parsed_time = parse_time_output(time_fixture)
    require(
        parsed_time["total_cpu_s"] == 0.5 and parsed_time["page_faults"] == 0,
        "time parser self-test",
    )
    advisory_fixture = dict(parsed_time)
    advisory_fixture["page_faults"] = 89
    require(
        process_resource_failure_reasons(advisory_fixture) == []
        and process_page_faults_advisory(advisory_fixture) == 89,
        "positive process page faults must remain valid advisory evidence",
    )
    for key in ("block_input_operations", "swaps"):
        fatal_fixture = dict(advisory_fixture)
        fatal_fixture[key] = 1
        require(
            process_resource_failure_reasons(fatal_fixture) == [f"{key}=1"],
            f"fatal process resource gate self-test: {key}",
        )
    expect_contract(exact_uint, 1, 0, "timer major faults")
    expect_contract(parse_time_output, time_fixture.replace("page faults", "faults"))
    validate_wall_us_pair(
        {"unattributed_wall_ms": 0.001, "unattributed_us": 1},
        "unattributed_wall_ms",
        "unattributed_us",
        positive=False,
    )
    expect_contract(
        validate_wall_us_pair,
        {"unattributed_wall_ms": 0.010, "unattributed_us": 1},
        "unattributed_wall_ms",
        "unattributed_us",
    )
    vm_a = {
        "pageouts": 10,
        "compressions": 20,
        "swapouts": 30,
        "compressor_stored_pages": 40,
        "compressor_occupied_pages": 50,
        "swap_used_bytes": 60,
        "errors": [],
    }
    vm_b = {
        **vm_a,
        "pageouts": 11,
        "compressor_stored_pages": 35,
        "compressor_occupied_pages": 55,
    }
    require(
        vm_interval("x", vm_a, vm_b)["failure_reasons"] == [],
        "VM advisory interval self-test",
    )
    for key in ("compressions", "swapouts", "swap_used_bytes"):
        drift = dict(vm_b)
        drift[key] = int(vm_a[key]) + 1
        require(
            vm_interval("x", vm_a, drift)["failure_reasons"],
            "VM hard interval self-test",
        )
    clean_thermal = (
        "No thermal warning level has been recorded\n"
        "No performance warning level has been recorded\n"
    )
    require(
        host_state_valid(clean_thermal, "AC Power", 50, [], []),
        "host inclusive memory self-test",
    )
    require(
        not host_state_valid(clean_thermal, "Battery Power", 50, [], [])
        and not host_state_valid(clean_thermal, "AC Power", 49, [], [])
        and not host_state_valid(clean_thermal, "AC Power", 50, [{"pid": 1}], []),
        "host invalidity self-test",
    )
    spawn = bounded_child(
        ["/definitely/missing/v0642-self-test-command"], {"PATH": "/usr/bin:/bin"}
    )
    require(
        spawn["pid"] is None
        and spawn["spawn_error"] is not None
        and spawn["returncode"] is None,
        "structured spawn-error self-test",
    )
    spawn_completion = completion_record("fixture", spawn, 123)
    require(
        spawn_completion["event"] == "completion"
        and spawn_completion["pid"] is None
        and spawn_completion["residency_to_acquired_ns"] is None,
        "spawn-error completion evidence self-test",
    )
    child_env = os.environ.copy()
    success = bounded_child([sys.executable, "-c", "print('fixture-child')"], child_env)
    require(
        success["returncode"] == 0
        and success["reaped"] is True
        and success["stdout"] == b"fixture-child\n",
        "successful exact-child reap self-test",
    )
    faulted = bounded_child(
        [sys.executable, "-c", "print('fault-fixture')"],
        child_env,
        _faults={"wait_once", "join_0"},
    )
    require(
        faulted["reaped"] is True
        and faulted["wait_errors"]
        and any("join0" in value for value in faulted["pipe_errors"]),
        "wait/join structured failure self-test",
    )
    drain_fault = bounded_child(
        [sys.executable, "-c", "print('drain-fixture')"],
        child_env,
        _faults={"read_0"},
    )
    require(
        drain_fault["reaped"] is True
        and any("pipe0" in value for value in drain_fault["pipe_errors"]),
        "drain failure closure self-test",
    )
    start_fault = bounded_child(
        [sys.executable, "-c", "import time; time.sleep(10)"],
        child_env,
        _faults={"thread_start_0"},
    )
    require(
        start_fault["reaped"] is True
        and any("thread0" in value for value in start_fault["pipe_errors"]),
        "thread-start exact-group cleanup self-test",
    )
    descendant_code = (
        "import signal,subprocess,sys,time;"
        "p=subprocess.Popen([sys.executable,'-c',"
        "'import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(30)']);"
        "print(p.pid,flush=True);time.sleep(30)"
    )
    with PacketSignalController(restore_on_exit=True) as group_controller:
        group_start = group_controller.sequence()
        group_sender = threading.Thread(
            target=lambda: (time.sleep(0.20), os.kill(os.getpid(), signal.SIGINT))
        )
        group_sender.start()
        group_outcome = bounded_child(
            [sys.executable, "-c", descendant_code],
            child_env,
            controller=group_controller,
            event_start_sequence=group_start,
        )
        group_sender.join()
    descendant_pid = int(group_outcome["stdout"].decode("ascii").strip())
    require(
        group_outcome["pgid"] == group_outcome["pid"]
        and group_outcome["reaped"] is True
        and group_outcome["cleanup_signal_sent"] is True
        and group_outcome["cleanup_actions"]
        and len(group_outcome["cleanup_actions"]) == 1
        and group_outcome["cleanup_actions"][0]["signal"] == signal.SIGKILL
        and group_outcome["cleanup_actions"][0]["target_pgid"] == group_outcome["pgid"]
        and group_outcome["cleanup_actions"][0]["succeeded"] is True
        and group_outcome["group_disposition"]["no_live_group_observed"] is True
        and descendant_pid not in process_group_members(int(group_outcome["pgid"])),
        "owned descendant process-group KILL/reap self-test",
    )
    expect_runtime(parse_process_group_rows, "123 456\nmalformed\n", 456)
    expect_ownership_lost(authenticate_process_group, 123, lambda _pid: 124)

    for failure_kind in ("getpgid-failure", "pgid-mismatch"):
        observed_pids = []

        def failing_getter(pid: int, kind: str = failure_kind) -> int:
            observed_pids.append(pid)
            if kind == "getpgid-failure":
                raise OSError("injected getpgid failure")
            return pid + 1

        expect_ownership_lost(
            bounded_child,
            [sys.executable, "-c", "import time;time.sleep(0.03)"],
            child_env,
            _pgid_getter=failing_getter,
        )
        require(
            len(observed_pids) == 1 and process_group_members(observed_pids[0]) == [],
            "real pgid-authentication failure reap self-test",
        )

    acquired_cleanup = []
    acquired_identity = []

    def reject_acquired(acquired: dict[str, int]) -> None:
        acquired_identity.append(dict(acquired))
        raise DurabilityError("injected acquired-record durability failure")

    try:
        bounded_child(
            [sys.executable, "-c", "import time;time.sleep(30)"],
            child_env,
            _on_acquired=reject_acquired,
            _exception_cleanup_observer=acquired_cleanup.append,
        )
    except DurabilityError:
        pass
    else:
        raise RuntimeError("expected acquired-record DurabilityError")
    require(
        len(acquired_identity) == 1
        and len(acquired_cleanup) == 1
        and acquired_cleanup[0]["signal"] == signal.SIGKILL
        and acquired_cleanup[0]["leader_reaped"] is True
        and acquired_cleanup[0]["group_disposition"]["no_live_group_observed"] is True
        and process_group_members(acquired_identity[0]["pgid"]) == [],
        "acquired-record durability cleanup self-test",
    )

    class LostWait:
        def wait(self, *, timeout: float) -> int:
            raise ChildProcessError(f"lost at {timeout}")

    expect_ownership_lost(exact_wait_once, LostWait(), 0.01)
    with tempfile.TemporaryDirectory() as signal_directory:
        signal_root = Path(signal_directory)
        for signum in OPERATOR_SIGNALS:
            with PacketSignalController(restore_on_exit=True) as signal_controller:
                start_sequence = signal_controller.sequence()
                os.kill(os.getpid(), signum)
                attempt_path = signal_root / f"attempt-{signum}.jsonl"
                closure_path = signal_root / f"closure-{signum}.jsonl"
                attempt = {
                    "event": "attempt",
                    "events": signal_controller.since(start_sequence),
                }
                append_jsonl(attempt_path, attempt)
                late_events = persist_attempt_signal_closure(
                    closure_path,
                    f"signal-{signum}",
                    start_sequence,
                    signal_controller,
                    sha256_bytes(json_bytes(attempt)),
                )
                expect_inconclusive(
                    signal_controller.require_quiet_since,
                    start_sequence,
                    "between-attempt fixture",
                )
            require(
                [event["signal"] for event in late_events] == [signum],
                "late-window operator signal evidence self-test",
            )
            persisted_attempt = parse_json_bytes(
                attempt_path.read_bytes().splitlines()[0], "signal attempt"
            )
            persisted_closure = parse_json_bytes(
                closure_path.read_bytes().splitlines()[0], "signal closure"
            )
            require(
                persisted_attempt["events"][0]["signal"] == signum
                and persisted_closure["invalid"] is True,
                "written late-window invalidation self-test",
            )
        atomic_path = signal_root / "atomic-closure.jsonl"
        with PacketSignalController(restore_on_exit=True) as atomic_controller:
            atomic_start = atomic_controller.sequence()
            os.kill(os.getpid(), signal.SIGINT)
            captured_atomic = persist_attempt_signal_closure(
                atomic_path, "atomic", atomic_start, atomic_controller, "0" * 64
            )
            os.kill(os.getpid(), signal.SIGTERM)
            require(
                atomic_controller.sequence() == 2 and len(captured_atomic) == 1,
                "atomic closure later-event setup self-test",
            )
        atomic_record = parse_json_bytes(
            atomic_path.read_bytes().splitlines()[0], "atomic closure"
        )
        require(
            atomic_record["signal_end_sequence"]
            == atomic_record["events"][-1]["sequence"]
            == 1,
            "immutable attempt signal slice self-test",
        )
        prior_mask = signal.pthread_sigmask(signal.SIG_BLOCK, [])
        with PacketSignalController(restore_on_exit=True) as cutoff_controller:
            cutoff = cutoff_controller.final_cutoff(
                signal_root / "cutoff.json",
                _after_block=lambda: os.kill(os.getpid(), signal.SIGTERM),
            )
            os.kill(os.getpid(), signal.SIGINT)
            require(
                cutoff["authority_event_count"] == 1
                and cutoff["pending_signals"] == [signal.SIGTERM]
                and signal.SIGINT in signal.sigpending(),
                "final signal cutoff self-test",
            )
            signal.pthread_sigmask(signal.SIG_SETMASK, prior_mask)
        require(
            cutoff["authority_event_count"] == 1,
            "post-cutoff signal authority self-test",
        )
    require(
        launch_delay_failure(100, 100 + LAUNCH_LIMIT_NS) is None
        and launch_delay_failure(100, 101 + LAUNCH_LIMIT_NS)
        == "launch_exceeded_five_seconds",
        "inclusive five-second launch boundary self-test",
    )
    invalid_conditioning = {
        "host_before_conditioning": {"valid": False},
        "vm_before_conditioning": {"errors": []},
        "host_before_launch": None,
        "residency_before": None,
        "conditioning_interval": None,
        "residency_proved_ns": None,
        "operation_errors": [],
    }
    require(
        "host_invalid_before_conditioning"
        in conditioning_failure_reasons(invalid_conditioning),
        "conditioning invalidity evidence self-test",
    )
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "tiny.bin"
        path.write_bytes(bytes(range(251)) * 200)
        identity = file_identity(path)
        conditioned = sequential_condition(path, identity, path.stat().st_size)
        residency = mincore_residency(path, identity, path.stat().st_size)
        require(
            conditioned["bytes_read"] == path.stat().st_size
            and residency["all_pages_resident"] is True,
            "tiny conditioning/mincore self-test",
        )
        descriptor = descriptor_bound_sha256(path, path.stat().st_size)
        require(
            descriptor["bytes_hashed"] == path.stat().st_size
            and descriptor["sha256"] == sha256_file(path),
            "descriptor-bound identity self-test",
        )
        lifecycle_path = Path(directory) / "lifecycle.jsonl"
        attempts_path = Path(directory) / "attempts.jsonl"
        append_jsonl(lifecycle_path, {"event": "launch", "stem": "fixture"})
        ordered_outcome = bounded_child(
            [sys.executable, "-c", "print('ordered-fixture')"],
            child_env,
            _on_acquired=lambda acquired: append_jsonl(
                lifecycle_path, {"event": "acquired", **acquired}
            ),
        )
        append_jsonl(lifecycle_path, completion_record("fixture", ordered_outcome, 1))
        append_jsonl(attempts_path, {"event": "attempt", "stem": "fixture"})
        lifecycle_rows = [
            parse_json_bytes(line, "fixture lifecycle")
            for line in lifecycle_path.read_bytes().splitlines()
        ]
        attempt_rows = [
            parse_json_bytes(line, "fixture attempt")
            for line in attempts_path.read_bytes().splitlines()
        ]
        require(
            [row["event"] for row in lifecycle_rows]
            == ["launch", "acquired", "completion"]
            and [row["event"] for row in attempt_rows] == ["attempt"],
            "launch/completion/attempt durable order self-test",
        )

        def failing_residency_probe() -> dict[str, object]:
            raise OSError("injected post-residency failure")

        post_fixture = collect_post_exit_evidence(
            "fixture", vm_a, {"valid": True}, vm_a, failing_residency_probe
        )
        post_path = Path(directory) / "post.json"
        write_json(post_path, post_fixture)
        persisted_post = parse_json_bytes(post_path.read_bytes(), "post fixture")
        require(
            persisted_post["residency_after"] is None
            and persisted_post["operation_errors"],
            "post-residency exception persistence self-test",
        )
        replacement = Path(directory) / "replacement.bin"
        replacement.write_bytes(b"replacement")
        expect_contract(
            descriptor_bound_sha256,
            path,
            path.stat().st_size,
            lambda: os.replace(replacement, path),
        )
    differences = {worker: [60_000] * 6 for worker in (1, 2, 6, 8, 12)}
    cpu_ratios = {worker: 1.10 for worker in differences}
    scored = score_rows(synthetic_rows(differences, cpu_ratios))
    require(
        scored["winner"] == 1 and scored["qualified_workers"] == [1, 2, 6, 8, 12],
        "inclusive gates and W1 eligibility self-test",
    )
    differences[1] = [70_000] * 6
    differences[2] = [80_000] * 6
    scored = score_rows(synthetic_rows(differences, cpu_ratios))
    require(
        scored["winner"] == 1 and scored["near_best_workers"][:2] == [1, 2],
        "10ms indifference self-test",
    )
    differences[1] = [59_999] * 6
    scored = score_rows(synthetic_rows(differences, cpu_ratios))
    require(
        scored["candidates"]["1"]["qualifies"] is False,
        "inclusive threshold failure self-test",
    )
    require(
        classify(contract_defect=True, invalid=True, winner=1, complete=True)
        == "implementation_or_contract_defect",
        "decision precedence defect self-test",
    )
    require(
        classify(contract_defect=False, invalid=True, winner=1, complete=True)
        == "inconclusive",
        "decision precedence validity self-test",
    )
    require(
        classify(contract_defect=False, invalid=False, winner=None, complete=True)
        == "warm-worker-screen-miss",
        "decision miss self-test",
    )
    require(
        classify(contract_defect=False, invalid=False, winner=1, complete=True)
        == "GO-diagnostic",
        "decision winner self-test",
    )
    require(
        {signum: signal.getsignal(signum) for signum in OPERATOR_SIGNALS}
        == initial_handlers
        and signal.pthread_sigmask(signal.SIG_BLOCK, []) == initial_mask,
        "self-test signal handler/mask restoration self-test",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument(
        "--self-test-child", action="store_true", help=argparse.SUPPRESS
    )
    args = parser.parse_args()
    if args.self_test or args.self_test_child:
        run_self_tests()
        print("self-test-child: ok" if args.self_test_child else "self-test: ok")
        return
    execute()


if __name__ == "__main__":
    main()
