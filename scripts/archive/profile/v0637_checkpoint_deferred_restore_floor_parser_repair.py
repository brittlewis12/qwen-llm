#!/usr/bin/env python3
"""v0.637 runner-only repair of the deferred-restore floor parser."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import stat
import subprocess
import tempfile
import time


ROOT = Path(__file__).resolve().parents[2]
RUNNER = Path(__file__).resolve()
PREREG = ROOT / "docs/bench/v0637-checkpoint-deferred-restore-floor-parser-repair.md"
STORE_SOURCE = ROOT / "crates/qwen-llm/src/checkpoint_store.rs"
CLI_SOURCE = ROOT / "crates/qwen-cli/src/main.rs"
ROOT_MANIFEST = ROOT / "Cargo.toml"
ROOT_LOCK = ROOT / "Cargo.lock"
ARTIFACT = ROOT / "target/profiles/v0637-checkpoint-deferred-restore-floor-p1"
WORK_ROOT = ROOT / "target/profiles/v0637-checkpoint-deferred-restore-floor-work"
SEALED_V0636_A_STDOUT = (
    ROOT
    / "target/profiles/v0636-checkpoint-deferred-restore-floor-p1"
    / "floor-p01-ab-r1-a.out"
)
SEALED_V0636_ARTIFACT = SEALED_V0636_A_STDOUT.parent
FIXTURE = (
    ROOT
    / "target/profiles/v0615-long-history/cache-ring0/v1/blobs"
    / "fe9d70c202425683190d1c6a3ef50474940915e8c2e918f3ad50574e12e92b4e"
    / "6500-p0-63362870ab8f272dfc0c5a5f1dbbe474921b1554a4f589e728d5a2cfbb1bbc2a.qcp"
)

BASE_COMMIT = "b6a212ab78049607fae9ba9b0ccc3535e3fa7254"
R636_COMMIT = "2acee4244a6b3f5bd9757c36d0911bcb5114e53a"
H636_COMMIT = "3d59d94cf314dae79b79cd05f74b4aea455cd30c"
R636_PATCH_SHA256 = "8f6e51fb3f90b8ff810a925daaebe3fab6112b92f68a07ba2992fe0e90a0c77c"
H636_PATCH_SHA256 = "30babdc372cccb5dafb5d28b69c7a2db56620cf028c0e691bd5bc12d4ded78bf"
FROZEN_FILE_SHA256 = {
    "Cargo.lock": "2ea5d138ef0b5813ce29ad39c39d4a34d31d5a72fb71431f6038c5343d2e4fbe",
    "Cargo.toml": "c92ed7c228b43de3a2920cc5d2a7535660fd328cbde1444bf315545c36b162e9",
    "crates/qwen-cli/src/main.rs": (
        "1ee1e959fe5c2616fec20fb924b9edef026fc432b635db1f46166a43a1ce0c71"
    ),
    "crates/qwen-llm/src/checkpoint_store.rs": (
        "a5ea6b001249418febb3f876253bf5989c0603ef42cc43bc50fda7d51927f8da"
    ),
}
SEALED_V0636_A_STDOUT_SHA256 = (
    "ea84c12edc1a9e4bc765e90e7b24ec2cd503f8527b0fae3dadf677dd5f90591f"
)
SEALED_V0636_HASHES = {
    "artifact-inventory.sha256": (
        "19955c6eab0848c597a23c3d7c745cc6106b77477d8fa8195fe2bc5bec833de5"
    ),
    "decision.json": (
        "793a81c8c49fc97b1a2092602be9ae3f3ae3c5d7100f0b8fd3ebd53d3b3b4790"
    ),
    "floor-p01-ab-r1-a.err": (
        "2f37294743ffba841d4363980da35636ea4ae6bfa6154b9f09456aa65d635ae5"
    ),
    "floor-p01-ab-r1-a.out": SEALED_V0636_A_STDOUT_SHA256,
    "launch-seal.jsonl": (
        "bd0ff4fb91d20e61b2ebf49dc417602cf95eb9ee09c220bcce7ab486432d5e92"
    ),
    "packet-complete.json": (
        "6cbeb588919610e8419ad82a4a6b0c5fc62c078119f93c5f4f4f19944af1b987"
    ),
}
GGUF_COMMIT = "c7369fd4868a6f613459fff355477f53bf4ee2f1"
LLAMA_COMMIT = "fe4fb533d1ed2855b6ac5492e56c42007d410409"
LLAMA_ALLOWED_UNTRACKED = (".claude/settings.local.json",)
INTEGRITY_ENV = "QWEN_CHECKPOINT_STAGED_INTEGRITY"
FLOOR_ROOT_ENV = "QWEN_CHECKPOINT_FLOOR_ROOT"
FLOOR_FIXTURE_ENV = "QWEN_CHECKPOINT_FLOOR_FIXTURE"
TEST_NAME = "checkpoint_store::tests::checkpoint_deferred_restore_exact_size_floor"
EXPECTED_RECORD_BYTES = 582_854_188
EXPECTED_BLOB_SHA256 = (
    "69c883f5130e5108cc3b948c5cc500c4d92db1fc2f96aa25b75f5f38abda1e65"
)
EXPECTED_ENCODER_BLAKE3 = (
    "2833fd870009a299dbaa389ae7ed379ea69d50fc31ff250f4e349dc3056eeb67"
)
EXPECTED_BLOB_RELATIVE = (
    "v1/blobs/fe9d70c202425683190d1c6a3ef50474940915e8c2e918f3ad50574e12e92b4e/"
    "6500-p0-63362870ab8f272dfc0c5a5f1dbbe474921b1554a4f589e728d5a2cfbb1bbc2a.qcp"
)
EXPECTED_VOCAB_SIZE = 248_320
EXPECTED_MAX_CONTEXT = 6_516
EXPECTED_MAX_RECORD_BYTES = 805_306_368
EXPECTED_STORE_BUDGET_BYTES = 805_306_368
GATE_US = 250_000
PAIR_ORDERS = (("AB", "A", "B"), ("BA", "B", "A"))
OPERATOR_SIGNALS = {signal.SIGINT, signal.SIGTERM}
MARKER_PREFIX = b"[checkpoint-deferred-floor]"
LIBTEST_LEAD = (
    b"test checkpoint_store::tests::checkpoint_deferred_restore_exact_size_floor ... "
)
MARKER = re.compile(
    rb"\[checkpoint-deferred-floor\] schema=1 "
    rb"mode=(decode|deferred-restore) record_bytes=(\d+) "
    rb"encoder_blake3=([0-9a-f]{64}) blob_sha256=([0-9a-f]{64}) "
    rb"staged_integrity_us=(\d+) publish_us=(\d+) full_decode_us=(\d+) "
    rb"outcome=(\S+) evicted=(\d+) managed_bytes_after=(\d+) "
    rb"vocab_size=(\d+) max_context=(\d+) max_record_bytes=(\d+) "
    rb"store_budget_bytes=(\d+) "
    rb"matched=(\d+) restored=(\d+) "
    rb"pending=(true|false) file_mode=([0-7]{4}) nlink=(\d+) "
    rb"temp_files=(\d+) blob_relative=(\S+)"
)


class ContractDefect(RuntimeError):
    pass


class IncompleteRun(RuntimeError):
    pass


def terminate_process_group(process: subprocess.Popen[bytes]) -> None:
    group = process.pid

    def group_exists() -> bool:
        try:
            os.killpg(group, 0)
            return True
        except ProcessLookupError:
            return False

    def wait_for_group_exit(seconds: float) -> bool:
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            process.poll()
            if not group_exists():
                return True
            time.sleep(0.05)
        process.poll()
        return not group_exists()

    try:
        os.killpg(group, signal.SIGTERM)
    except ProcessLookupError:
        pass
    if not wait_for_group_exit(10.0):
        try:
            os.killpg(group, signal.SIGKILL)
        except ProcessLookupError:
            pass
        if not wait_for_group_exit(10.0):
            raise ContractDefect(f"process group {group} survived SIGKILL")
    process.wait()


def run_process(
    argv: list[str],
    env: dict[str, str],
    stdout: object,
    stderr: object,
    label: str,
) -> int:
    process: subprocess.Popen[bytes] | None = None
    try:
        process = subprocess.Popen(
            argv,
            cwd=ROOT,
            env=env,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
        )
        try:
            actual_group = os.getpgid(process.pid)
        except ProcessLookupError:
            actual_group = process.pid
        if actual_group != process.pid:
            terminate_process_group(process)
            raise ContractDefect(
                f"child {process.pid} entered unexpected process group {actual_group}"
            )
        return process.wait()
    except (KeyboardInterrupt, IncompleteRun) as error:
        signal.pthread_sigmask(signal.SIG_BLOCK, OPERATOR_SIGNALS)
        if process is not None:
            terminate_process_group(process)
        raise IncompleteRun(f"interrupted process: {label}") from error
    except OSError as error:
        if process is not None:
            terminate_process_group(process)
        raise IncompleteRun(f"process launch failed: {label}: {error}") from error


def json_bytes(value: object, *, pretty: bool = False) -> bytes:
    if pretty:
        text = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True)
    else:
        text = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
    return (text + "\n").encode()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def json_from_bytes(data: bytes, label: str) -> object:
    try:
        return json.loads(data.decode("utf-8", errors="strict"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ContractDefect(f"invalid sealed v0.636 JSON: {label}") from error


def read_regular_directory(path: Path) -> dict[str, bytes]:
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    try:
        directory = os.open(path, flags)
    except OSError as error:
        raise ContractDefect(
            f"cannot open sealed directory without following: {path}"
        ) from error
    try:
        names = sorted(os.listdir(directory))
        sealed: dict[str, bytes] = {}
        stamps: dict[str, tuple[int, int, int]] = {}
        for name in names:
            try:
                descriptor = os.open(
                    name,
                    os.O_RDONLY | os.O_NOFOLLOW,
                    dir_fd=directory,
                )
            except OSError as error:
                raise ContractDefect(
                    f"cannot open sealed member without following: {name}"
                ) from error
            try:
                before = os.fstat(descriptor)
                if not stat.S_ISREG(before.st_mode):
                    raise ContractDefect(f"sealed member is not regular: {name}")
                chunks = []
                while chunk := os.read(descriptor, 1024 * 1024):
                    chunks.append(chunk)
                after = os.fstat(descriptor)
                stamp = (before.st_dev, before.st_ino, before.st_size)
                if (
                    stamp != (after.st_dev, after.st_ino, after.st_size)
                    or sum(map(len, chunks)) != before.st_size
                ):
                    raise ContractDefect(f"sealed member changed while reading: {name}")
                sealed[name] = b"".join(chunks)
                stamps[name] = stamp
            finally:
                os.close(descriptor)
        if sorted(os.listdir(directory)) != names:
            raise ContractDefect("sealed directory population changed while reading")
        for name in names:
            metadata = os.stat(name, dir_fd=directory, follow_symlinks=False)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or (metadata.st_dev, metadata.st_ino, metadata.st_size) != stamps[name]
            ):
                raise ContractDefect(
                    f"sealed directory member changed after read: {name}"
                )
        return sealed
    except OSError as error:
        raise ContractDefect(f"cannot authenticate sealed directory: {path}") from error
    finally:
        os.close(directory)


def authenticate_v0636_bridge() -> dict[str, object]:
    sealed = read_regular_directory(SEALED_V0636_ARTIFACT)
    if len(sealed) != 16:
        raise ContractDefect("sealed v0.636 final regular-file population drifted")
    actual_seal_hashes = {
        name: sha256_bytes(sealed.get(name, b"")) for name in SEALED_V0636_HASHES
    }
    if actual_seal_hashes != SEALED_V0636_HASHES:
        raise ContractDefect("sealed v0.636 bridge file authentication failed")

    inventory_bytes = sealed["artifact-inventory.sha256"]
    inventory: dict[str, str] = {}
    for line in inventory_bytes.splitlines():
        match = re.fullmatch(rb"([0-9a-f]{64})  ([A-Za-z0-9._-]+)", line)
        if match is None:
            raise ContractDefect("sealed v0.636 inventory grammar drifted")
        digest = match.group(1).decode("ascii")
        name = match.group(2).decode("ascii")
        if name in inventory:
            raise ContractDefect("sealed v0.636 inventory contains a duplicate")
        inventory[name] = digest
    if len(inventory) != 14:
        raise ContractDefect("sealed v0.636 inventory member count drifted")
    if set(sealed) != set(inventory) | {
        "artifact-inventory.sha256",
        "packet-complete.json",
    }:
        raise ContractDefect("sealed v0.636 final file names drifted")
    member_hashes = {name: sha256_bytes(sealed[name]) for name in inventory}
    if member_hashes != inventory:
        raise ContractDefect("sealed v0.636 inventory member hash drifted")

    completion = json_from_bytes(sealed["packet-complete.json"], "completion")
    if not isinstance(completion, dict) or completion != {
        "decision_sha256": SEALED_V0636_HASHES["decision.json"],
        "final_files": 16,
        "inventory_members": 14,
        "inventory_sha256": SEALED_V0636_HASHES["artifact-inventory.sha256"],
        "schema": 1,
    }:
        raise ContractDefect("sealed v0.636 completion bindings drifted")

    decision = json_from_bytes(sealed["decision.json"], "decision")
    if not isinstance(decision, dict) or any(
        (
            decision.get("status") != "implementation_or_contract_defect",
            decision.get("authority") != "no-production-authority",
            decision.get("force_authorized") is not False,
            decision.get("successor_authorization") != "none",
            decision.get("error")
            != "ContractDefect: expected one floor marker, found 0",
            decision.get("rows") != [],
            decision.get("floor") is not None,
            decision.get("source_commit") != H636_COMMIT,
        )
    ):
        raise ContractDefect("sealed v0.636 decision contract drifted")
    gates = decision.get("gates")
    if (
        not isinstance(gates, list)
        or len(gates) != 4
        or any(
            not isinstance(gate, dict) or gate.get("returncode") != 0 for gate in gates
        )
    ):
        raise ContractDefect("sealed v0.636 decision gate facts drifted")

    launch_text = sealed["launch-seal.jsonl"]
    try:
        launch_records = [
            json.loads(line.decode("utf-8", errors="strict"))
            for line in launch_text.splitlines()
        ]
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ContractDefect("sealed v0.636 launch seal is invalid") from error
    if len(launch_records) != 10 or any(
        not isinstance(record, dict) for record in launch_records
    ):
        raise ContractDefect("sealed v0.636 launch record population drifted")
    gate_names = {
        "qwen-release-build",
        "checkpoint-codec",
        "checkpoint-store",
        "qwen-cli",
    }
    gate_records = [
        record for record in launch_records if record.get("stage") == "gate"
    ]
    floor_records = [
        record for record in launch_records if record.get("stage") != "gate"
    ]
    if len(gate_records) != 8:
        raise ContractDefect("sealed v0.636 gate launch population drifted")
    for name in gate_names:
        pair = [record for record in gate_records if record.get("stem") == name]
        if (
            len(pair) != 2
            or [record.get("event") for record in pair] != ["launch", "completion"]
            or pair[1].get("returncode") != 0
            or pair[1].get("error") is not None
        ):
            raise ContractDefect(f"sealed v0.636 gate seal drifted: {name}")
    if any(record.get("arm") == "B" for record in launch_records):
        raise ContractDefect("sealed v0.636 unexpectedly launched B")
    if (
        len(floor_records) != 2
        or [record.get("event") for record in floor_records] != ["launch", "completion"]
        or floor_records[0].get("arm") != "A"
        or floor_records[0].get("mode") != "decode"
        or floor_records[0].get("stem") != "floor-p01-ab-r1-a"
        or floor_records[1].get("stem") != "floor-p01-ab-r1-a"
        or floor_records[1].get("returncode") != 0
        or floor_records[1].get("error") is not None
    ):
        raise ContractDefect("sealed v0.636 A launch/completion facts drifted")
    parse_marker(
        sealed["floor-p01-ab-r1-a.out"],
        sealed["floor-p01-ab-r1-a.err"],
        "decode",
    )
    return {
        "artifact_path": str(SEALED_V0636_ARTIFACT),
        "seal_sha256": dict(sorted(SEALED_V0636_HASHES.items())),
        "inventory_members": 14,
        "final_regular_files": 16,
        "launched_children_forensic": 1,
        "b_observations_forensic": 0,
        "parser_fixture_authenticated": True,
        "timing_observations_imported": 0,
    }


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_new(path: Path, data: bytes) -> None:
    with path.open("xb", buffering=0) as output:
        written = output.write(data)
        if written != len(data):
            raise OSError(f"short write for {path}: {written}/{len(data)}")
        output.flush()
        os.fsync(output.fileno())
    fsync_directory(path.parent)


def write_json(path: Path, value: object) -> None:
    write_new(path, json_bytes(value, pretty=True))


def append_jsonl(path: Path, value: object) -> None:
    data = json_bytes(value)
    with path.open("ab", buffering=0) as output:
        written = output.write(data)
        if written != len(data):
            raise OSError(f"short append for {path}: {written}/{len(data)}")
        output.flush()
        os.fsync(output.fileno())
    fsync_directory(path.parent)


def fsync_file(path: Path) -> None:
    with path.open("rb", buffering=0) as source:
        os.fsync(source.fileno())


def command(
    argv: list[str],
    *,
    cwd: Path = ROOT,
    env: dict[str, str] | None = None,
) -> str:
    return subprocess.run(
        argv,
        cwd=cwd,
        env=env,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    ).stdout


def git(argv: list[str], *, cwd: Path = ROOT) -> str:
    return command(["git", *argv], cwd=cwd).strip()


def parent(commit: str) -> str:
    fields = git(["rev-list", "--parents", "-n", "1", commit]).split()
    if len(fields) != 2 or fields[0] != commit:
        raise ContractDefect(f"commit is not single-parent: {commit}")
    return fields[1]


def changed_paths(left: str, right: str) -> list[str]:
    text = git(["diff", "--name-status", f"{left}..{right}"])
    return text.splitlines() if text else []


def patch_sha256(left: str, right: str, paths: tuple[Path, ...]) -> str:
    result = subprocess.run(
        [
            "git",
            "diff",
            "--binary",
            f"{left}..{right}",
            "--",
            *(str(path.relative_to(ROOT)) for path in paths),
        ],
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
    )
    return sha256_bytes(result.stdout)


def complete_patch_sha256(left: str, right: str) -> str:
    result = subprocess.run(
        ["git", "diff", "--binary", f"{left}..{right}"],
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
    )
    return sha256_bytes(result.stdout)


def verify_sibling(
    path: Path,
    expected_commit: str,
    allowed_untracked: tuple[str, ...] = (),
) -> dict[str, object]:
    if git(["rev-parse", "HEAD"], cwd=path) != expected_commit:
        raise ContractDefect(f"sibling commit drifted: {path}")
    status = git(
        ["status", "--porcelain=v1", "--untracked-files=all"],
        cwd=path,
    )
    lines = status.splitlines() if status else []
    if any(not line.startswith("?? ") for line in lines):
        raise ContractDefect(f"sibling tracked state is dirty: {path}")
    untracked = sorted(line[3:] for line in lines)
    if untracked != sorted(allowed_untracked):
        raise ContractDefect(f"sibling untracked state drifted: {path}: {untracked}")
    return {
        "path": str(path),
        "commit": expected_commit,
        "allowed_untracked": untracked,
    }


def verify_source(v0636_bridge: dict[str, object]) -> dict[str, object]:
    frozen_paths = (ROOT_LOCK, ROOT_MANIFEST, CLI_SOURCE, STORE_SOURCE)
    for path in (RUNNER, PREREG, *frozen_paths):
        git(["ls-files", "--error-unmatch", str(path.relative_to(ROOT))])
    dirty = git(["status", "--porcelain=v1", "--untracked-files=all"])
    if dirty:
        raise ContractDefect(f"worktree is dirty: {dirty!r}")

    execution = git(["rev-parse", "HEAD"])
    if parent(execution) != H636_COMMIT:
        raise ContractDefect("R637 is not a single-parent child of H636")
    if parent(H636_COMMIT) != R636_COMMIT or parent(R636_COMMIT) != BASE_COMMIT:
        raise ContractDefect("frozen v0.636 ancestry drifted")

    r636_expected = sorted(
        [
            "A\tdocs/bench/v0636-checkpoint-deferred-restore-floor.md",
            "A\tscripts/profile/v0636_checkpoint_deferred_restore_floor.py",
        ]
    )
    if sorted(changed_paths(BASE_COMMIT, R636_COMMIT)) != r636_expected:
        raise ContractDefect("R636 patch paths drifted")
    h636_expected = sorted(
        [
            f"M\t{CLI_SOURCE.relative_to(ROOT)}",
            f"M\t{STORE_SOURCE.relative_to(ROOT)}",
        ]
    )
    if sorted(changed_paths(R636_COMMIT, H636_COMMIT)) != h636_expected:
        raise ContractDefect("H636 patch paths drifted")
    r637_expected = sorted(
        [f"A\t{PREREG.relative_to(ROOT)}", f"A\t{RUNNER.relative_to(ROOT)}"]
    )
    if sorted(changed_paths(H636_COMMIT, execution)) != r637_expected:
        raise ContractDefect("R637 must add exactly the preregistration and runner")
    if complete_patch_sha256(BASE_COMMIT, R636_COMMIT) != R636_PATCH_SHA256:
        raise ContractDefect("R636 patch authentication failed")
    if complete_patch_sha256(R636_COMMIT, H636_COMMIT) != H636_PATCH_SHA256:
        raise ContractDefect("H636 patch authentication failed")
    actual_hashes = {
        str(path.relative_to(ROOT)): sha256_file(path) for path in frozen_paths
    }
    if actual_hashes != FROZEN_FILE_SHA256:
        raise ContractDefect("frozen Cargo or implementation source drifted")

    if not FIXTURE.is_file():
        raise ContractDefect(f"frozen v0.615 checkpoint fixture is missing: {FIXTURE}")
    if (
        FIXTURE.stat().st_size != EXPECTED_RECORD_BYTES
        or sha256_file(FIXTURE) != EXPECTED_BLOB_SHA256
    ):
        raise ContractDefect("frozen v0.615 checkpoint fixture drifted")

    paths = (RUNNER, PREREG, *frozen_paths)
    return {
        "base_commit": BASE_COMMIT,
        "v0636_preregistration_commit": R636_COMMIT,
        "implementation_commit": H636_COMMIT,
        "execution_commit": execution,
        "v0636_preregistration_patch_sha256": R636_PATCH_SHA256,
        "implementation_patch_sha256": H636_PATCH_SHA256,
        "execution_patch_sha256": patch_sha256(
            H636_COMMIT, execution, (RUNNER, PREREG)
        ),
        "v0636_forensic_bridge": v0636_bridge,
        "file_sha256": {str(path): sha256_file(path) for path in paths},
        "fixture": {
            "path": str(FIXTURE),
            "bytes": EXPECTED_RECORD_BYTES,
            "sha256": EXPECTED_BLOB_SHA256,
            "encoder_blake3": EXPECTED_ENCODER_BLAKE3,
        },
        "siblings": [
            verify_sibling(ROOT.parent / "gguf", GGUF_COMMIT),
            verify_sibling(
                ROOT.parent / "llama-cpp-rs",
                LLAMA_COMMIT,
                LLAMA_ALLOWED_UNTRACKED,
            ),
        ],
    }


def child_environment() -> tuple[dict[str, str], dict[str, object]]:
    kept_names = {
        "HOME",
        "LANG",
        "LC_ALL",
        "LOGNAME",
        "PATH",
        "SHELL",
        "TERM",
        "TMPDIR",
        "USER",
    }
    retained = {name: value for name, value in os.environ.items() if name in kept_names}
    overrides = {"RUST_BACKTRACE": "0", "RUST_TEST_THREADS": "1"}
    env = {**retained, **overrides}
    removed = sorted(set(os.environ) - set(retained) - set(overrides))
    if any(name in env for name in (INTEGRITY_ENV, FLOOR_ROOT_ENV, FLOOR_FIXTURE_ENV)):
        raise ContractDefect("normalized environment retained v0.637 controls")
    return env, {
        "effective": dict(sorted(env.items())),
        "retained": dict(sorted(retained.items())),
        "removed_names": removed,
        "overridden": {
            name: {
                "original": os.environ.get(name),
                "effective": value,
            }
            for name, value in sorted(overrides.items())
        },
    }


def arm_mode(arm: str) -> str:
    if arm == "A":
        return "decode"
    if arm == "B":
        return "deferred-restore"
    raise ContractDefect(f"unknown arm: {arm}")


def test_command() -> list[str]:
    return [
        "cargo",
        "test",
        "--release",
        "-p",
        "qwen-llm",
        TEST_NAME,
        "--",
        "--ignored",
        "--exact",
        "--nocapture",
        "--test-threads=1",
    ]


def build_manifest(
    source: dict[str, object],
    environment: dict[str, object],
) -> dict[str, object]:
    return {
        "schema": 1,
        "created_unix_ms": time.time_ns() // 1_000_000,
        "source": source,
        "execution_commit": source["execution_commit"],
        "implementation_commit": source["implementation_commit"],
        "v0636_forensic_bridge": source["v0636_forensic_bridge"],
        "v0636_authority_imported": False,
        "v0636_gate_results_imported": False,
        "v0636_scored_rows_imported": 0,
        "v0636_performance_observations_imported": 0,
        "v0636_timing_observations_imported": 0,
        "v0636_launched_children_forensic": 1,
        "v0636_b_observations_forensic": 0,
        "child_environment": environment,
        "test_command": test_command(),
        "integrity_environment": INTEGRITY_ENV,
        "floor_root_environment": FLOOR_ROOT_ENV,
        "floor_fixture_environment": FLOOR_FIXTURE_ENV,
        "record_bytes": EXPECTED_RECORD_BYTES,
        "blob_sha256": EXPECTED_BLOB_SHA256,
        "encoder_blake3": EXPECTED_ENCODER_BLAKE3,
        "blob_relative": EXPECTED_BLOB_RELATIVE,
        "vocab_size": EXPECTED_VOCAB_SIZE,
        "max_context_tokens": EXPECTED_MAX_CONTEXT,
        "max_record_bytes": EXPECTED_MAX_RECORD_BYTES,
        "store_budget_bytes": EXPECTED_STORE_BUDGET_BYTES,
        "gate_us": GATE_US,
        "pair_orders": [list(order) for order in PAIR_ORDERS],
        "retry_count": 0,
        "host": {
            "product_version": command(["sw_vers", "-productVersion"]).strip(),
            "build_version": command(["sw_vers", "-buildVersion"]).strip(),
            "hardware": command(["uname", "-m"]).strip(),
        },
        "tools": {
            "cargo": command(["cargo", "--version"]).strip(),
            "rustc": command(["rustc", "--version", "--verbose"]).strip(),
        },
    }


def parse_marker(stdout: bytes, stderr: bytes, expected_mode: str) -> dict[str, object]:
    try:
        stdout.decode("utf-8", errors="strict")
        stderr.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise ContractDefect("child output is not strict UTF-8") from error
    stdout_markers = stdout.count(MARKER_PREFIX)
    stderr_markers = stderr.count(MARKER_PREFIX)
    if stdout_markers != 1 or stderr_markers != 0:
        raise ContractDefect(
            "expected exactly one stdout marker and zero stderr markers"
        )
    marker_line: bytes | None = None
    for physical_line in stdout.split(b"\n"):
        if physical_line.endswith(b"\r"):
            physical_line = physical_line[:-1]
        if MARKER_PREFIX not in physical_line:
            continue
        marker_line = physical_line
        break
    if marker_line is None:
        raise ContractDefect("stdout marker prefix was split across physical lines")
    if marker_line.startswith(MARKER_PREFIX):
        marker_bytes = marker_line
    elif marker_line.startswith(LIBTEST_LEAD + MARKER_PREFIX):
        marker_bytes = marker_line[len(LIBTEST_LEAD) :]
    else:
        raise ContractDefect("floor marker has an invalid Cargo libtest lead")
    match = MARKER.fullmatch(marker_bytes)
    if match is None:
        raise ContractDefect("floor marker is malformed or has trailing data")
    try:
        (
            mode,
            record_bytes_text,
            encoder_blake3,
            blob_sha256,
            integrity_us_text,
            publish_us_text,
            full_decode_us_text,
            outcome,
            evicted_text,
            managed_bytes_after_text,
            vocab_size_text,
            max_context_text,
            max_record_bytes_text,
            store_budget_bytes_text,
            matched_text,
            restored_text,
            pending,
            file_mode,
            nlink_text,
            temp_files_text,
            blob_relative,
        ) = [field.decode("ascii", errors="strict") for field in match.groups()]
        (
            record_bytes,
            integrity_us,
            publish_us,
            full_decode_us,
            evicted,
            managed_bytes_after,
            vocab_size,
            max_context,
            max_record_bytes,
            store_budget_bytes,
            matched,
            restored,
            nlink,
            temp_files,
        ) = map(
            int,
            (
                record_bytes_text,
                integrity_us_text,
                publish_us_text,
                full_decode_us_text,
                evicted_text,
                managed_bytes_after_text,
                vocab_size_text,
                max_context_text,
                max_record_bytes_text,
                store_budget_bytes_text,
                matched_text,
                restored_text,
                nlink_text,
                temp_files_text,
            ),
        )
    except (UnicodeDecodeError, ValueError) as error:
        raise ContractDefect("floor marker field encoding is invalid") from error
    if (
        mode != expected_mode
        or record_bytes != EXPECTED_RECORD_BYTES
        or encoder_blake3 != EXPECTED_ENCODER_BLAKE3
        or blob_sha256 != EXPECTED_BLOB_SHA256
        or outcome != "published"
        or evicted != 0
        or managed_bytes_after != EXPECTED_RECORD_BYTES
        or vocab_size != EXPECTED_VOCAB_SIZE
        or max_context != EXPECTED_MAX_CONTEXT
        or max_record_bytes != EXPECTED_MAX_RECORD_BYTES
        or store_budget_bytes != EXPECTED_STORE_BUDGET_BYTES
        or matched != 6500
        or restored != 6499
        or pending != "true"
        or file_mode != "0600"
        or nlink != 1
        or temp_files != 0
        or blob_relative != EXPECTED_BLOB_RELATIVE
    ):
        raise ContractDefect("floor marker contract drifted")
    return {
        "mode": mode,
        "record_bytes": record_bytes,
        "encoder_blake3": encoder_blake3,
        "blob_sha256": blob_sha256,
        "staged_integrity_us": integrity_us,
        "publish_us": publish_us,
        "full_decode_us": full_decode_us,
        "outcome": outcome,
        "evicted": evicted,
        "managed_bytes_after": managed_bytes_after,
        "vocab_size": vocab_size,
        "max_context": max_context,
        "max_record_bytes": max_record_bytes,
        "store_budget_bytes": store_budget_bytes,
        "matched": matched,
        "restored": restored,
        "pending": True,
        "file_mode": file_mode,
        "nlink": nlink,
        "temp_files": temp_files,
        "blob_relative": blob_relative,
    }


def inspect_store(root: Path, marker: dict[str, object]) -> dict[str, object]:
    files = sorted(path for path in root.rglob("*") if path.is_file())
    blobs = [path for path in files if path.suffix == ".qcp"]
    temporary = [path for path in files if path.name.startswith(".tmp-")]
    locks = [path for path in files if path.name == "store.lock"]
    if len(blobs) != 1 or len(locks) != 1 or len(files) != 2 or temporary:
        raise ContractDefect("floor store topology drifted")
    blob = blobs[0]
    metadata = blob.stat()
    relative = str(blob.relative_to(root))
    digest = sha256_file(blob)
    if (
        relative != EXPECTED_BLOB_RELATIVE
        or relative != marker["blob_relative"]
        or metadata.st_size != EXPECTED_RECORD_BYTES
        or stat.S_IMODE(metadata.st_mode) != 0o600
        or metadata.st_nlink != 1
        or digest != marker["blob_sha256"]
        or blob.is_symlink()
        or any(path.is_symlink() for path in root.rglob("*"))
    ):
        raise ContractDefect("floor blob identity drifted")
    return {
        "blob_relative": relative,
        "blob_sha256": digest,
        "blob_bytes": metadata.st_size,
        "blob_mode": oct(stat.S_IMODE(metadata.st_mode)),
        "blob_nlink": metadata.st_nlink,
        "file_count": len(files),
        "temporary_count": len(temporary),
    }


def launch_child(
    arm: str,
    pair_index: int,
    order: str,
    position: int,
    base_env: dict[str, str],
    expected_source: dict[str, object],
) -> dict[str, object]:
    bridge = expected_source["v0636_forensic_bridge"]
    if not isinstance(bridge, dict) or verify_source(bridge) != expected_source:
        raise ContractDefect("source drifted immediately before floor child")
    stem = f"floor-p{pair_index:02d}-{order.lower()}-r{position}-{arm.lower()}"
    root = WORK_ROOT / stem
    stdout_path = ARTIFACT / f"{stem}.out"
    stderr_path = ARTIFACT / f"{stem}.err"
    if root.exists():
        raise ContractDefect(f"refusing to reuse floor root: {root}")
    root.parent.mkdir(parents=True, exist_ok=True)
    root.mkdir(mode=0o700)
    if stat.S_IMODE(root.stat().st_mode) != 0o700 or any(root.iterdir()):
        raise ContractDefect("floor root is not empty mode 0700")
    fsync_directory(root.parent)
    env = base_env.copy()
    env[INTEGRITY_ENV] = arm_mode(arm)
    env[FLOOR_ROOT_ENV] = str(root)
    env[FLOOR_FIXTURE_ENV] = str(FIXTURE)
    argv = test_command()
    launch = {
        "schema": 1,
        "event": "launch",
        "unix_ms": time.time_ns() // 1_000_000,
        "stem": stem,
        "arm": arm,
        "mode": arm_mode(arm),
        "pair_index": pair_index,
        "pair_order": order,
        "position": position,
        "command": argv,
    }
    append_jsonl(ARTIFACT / "launch-seal.jsonl", launch)
    started = time.perf_counter()
    caught: BaseException | None = None
    returncode: int | None = None
    try:
        with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
            returncode = run_process(argv, env, stdout, stderr, stem)
    except (KeyboardInterrupt, IncompleteRun) as error:
        caught = IncompleteRun(f"operator interrupted floor child: {stem}: {error}")
    except OSError as error:
        caught = IncompleteRun(f"floor child launch failed: {stem}: {error}")
    except BaseException as error:
        caught = error
    finally:
        prior_mask = signal.pthread_sigmask(signal.SIG_BLOCK, OPERATOR_SIGNALS)
        if stdout_path.exists():
            fsync_file(stdout_path)
        if stderr_path.exists():
            fsync_file(stderr_path)
        fsync_directory(ARTIFACT)
        append_jsonl(
            ARTIFACT / "launch-seal.jsonl",
            {
                "schema": 1,
                "event": "completion",
                "unix_ms": time.time_ns() // 1_000_000,
                "stem": stem,
                "returncode": returncode,
                "error": str(caught) if caught is not None else None,
            },
        )
    wall_ms = (time.perf_counter() - started) * 1e3
    pending = set(signal.sigpending()) & OPERATOR_SIGNALS
    if caught is not None:
        raise caught
    if returncode is None:
        raise IncompleteRun(f"floor child has no return code: {stem}")
    stdout_bytes = stdout_path.read_bytes()
    stderr_bytes = stderr_path.read_bytes()
    if returncode < 0:
        raise IncompleteRun(f"floor child terminated by signal: {stem}: {returncode}")
    if returncode != 0:
        raise ContractDefect(f"floor child failed: {stem}: {returncode}")
    if pending:
        raise IncompleteRun(f"operator signal after floor child: {stem}")
    signal.pthread_sigmask(signal.SIG_SETMASK, prior_mask)
    marker = parse_marker(stdout_bytes, stderr_bytes, arm_mode(arm))
    store = inspect_store(root, marker)
    return {
        "schema": 1,
        "stage": "exact-size-floor",
        "stem": stem,
        "arm": arm,
        "mode": arm_mode(arm),
        "pair_index": pair_index,
        "pair_order": order,
        "position": position,
        "command": argv,
        "process_wall_ms": wall_ms,
        "stdout_sha256": sha256_bytes(stdout_bytes),
        "stderr_sha256": sha256_bytes(stderr_bytes),
        "marker": marker,
        "store": store,
        "valid": True,
    }


def analyze(rows: list[dict[str, object]]) -> dict[str, object]:
    if len(rows) != 4:
        raise ContractDefect("floor child count drifted")
    blob_hashes = {row["store"]["blob_sha256"] for row in rows}
    encoder_hashes = {row["marker"]["encoder_blake3"] for row in rows}
    if len(blob_hashes) != 1 or len(encoder_hashes) != 1:
        raise ContractDefect("floor records differ across arms")
    pairs = []
    for pair_index, (order, first, second) in enumerate(PAIR_ORDERS, 1):
        pair = [row for row in rows if row["pair_index"] == pair_index]
        if len(pair) != 2 or [row["arm"] for row in pair] != [first, second]:
            raise ContractDefect("floor pair membership drifted")
        by_arm = {row["arm"]: row for row in pair}
        baseline = by_arm["A"]["marker"]
        candidate = by_arm["B"]["marker"]
        integrity_saving = (
            baseline["staged_integrity_us"] - candidate["staged_integrity_us"]
        )
        publish_saving = baseline["publish_us"] - candidate["publish_us"]
        pairs.append(
            {
                "pair_index": pair_index,
                "pair_order": order,
                "integrity_saving_us": integrity_saving,
                "publish_saving_us": publish_saving,
                "baseline_integrity_us": baseline["staged_integrity_us"],
                "candidate_integrity_us": candidate["staged_integrity_us"],
                "baseline_publish_us": baseline["publish_us"],
                "candidate_publish_us": candidate["publish_us"],
                "integrity_gate": integrity_saving >= GATE_US,
                "publish_gate": publish_saving >= GATE_US,
            }
        )
    passes = all(pair["integrity_gate"] and pair["publish_gate"] for pair in pairs)
    return {
        "schema": 1,
        "pairs": pairs,
        "passes": passes,
        "blob_sha256": next(iter(blob_hashes)),
        "encoder_blake3": next(iter(encoder_hashes)),
    }


def run_gate(name: str, argv: list[str], env: dict[str, str]) -> dict[str, object]:
    stdout_path = ARTIFACT / f"gate-{name}.out"
    stderr_path = ARTIFACT / f"gate-{name}.err"
    append_jsonl(
        ARTIFACT / "launch-seal.jsonl",
        {
            "schema": 1,
            "event": "launch",
            "stage": "gate",
            "unix_ms": time.time_ns() // 1_000_000,
            "stem": name,
            "command": argv,
        },
    )
    started = time.perf_counter()
    caught: BaseException | None = None
    returncode: int | None = None
    try:
        with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
            returncode = run_process(argv, env, stdout, stderr, f"gate-{name}")
    except (KeyboardInterrupt, IncompleteRun) as error:
        caught = IncompleteRun(f"gate interrupted: {name}: {error}")
    except OSError as error:
        caught = IncompleteRun(f"gate launch failed: {name}: {error}")
    except BaseException as error:
        caught = error
    finally:
        prior_mask = signal.pthread_sigmask(signal.SIG_BLOCK, OPERATOR_SIGNALS)
        if stdout_path.exists():
            fsync_file(stdout_path)
        if stderr_path.exists():
            fsync_file(stderr_path)
        fsync_directory(ARTIFACT)
        append_jsonl(
            ARTIFACT / "launch-seal.jsonl",
            {
                "schema": 1,
                "event": "completion",
                "stage": "gate",
                "unix_ms": time.time_ns() // 1_000_000,
                "stem": name,
                "returncode": returncode,
                "error": str(caught) if caught is not None else None,
            },
        )
    pending = set(signal.sigpending()) & OPERATOR_SIGNALS
    if caught is not None:
        raise caught
    if returncode is None:
        raise IncompleteRun(f"gate has no return code: {name}")
    report = {
        "schema": 1,
        "name": name,
        "command": argv,
        "returncode": returncode,
        "wall_ms": (time.perf_counter() - started) * 1e3,
        "stdout_sha256": sha256_file(stdout_path),
        "stderr_sha256": sha256_file(stderr_path),
    }
    append_jsonl(ARTIFACT / "gates.jsonl", report)
    if returncode < 0:
        raise IncompleteRun(f"gate terminated by signal: {name}: {returncode}")
    if returncode != 0:
        raise ContractDefect(f"gate failed: {name}: {returncode}")
    if pending:
        raise IncompleteRun(f"operator signal after gate: {name}")
    signal.pthread_sigmask(signal.SIG_SETMASK, prior_mask)
    return report


def required_gates() -> tuple[tuple[str, list[str]], ...]:
    return (
        (
            "qwen-release-build",
            ["cargo", "build", "--release", "-p", "qwen-cli", "--bin", "qwen"],
        ),
        (
            "checkpoint-codec",
            [
                "cargo",
                "test",
                "--release",
                "-p",
                "qwen-llm",
                "checkpoint_codec::tests",
                "--",
                "--test-threads=1",
            ],
        ),
        (
            "checkpoint-store",
            [
                "cargo",
                "test",
                "--release",
                "-p",
                "qwen-llm",
                "checkpoint_store::tests",
                "--",
                "--test-threads=1",
            ],
        ),
        (
            "qwen-cli",
            [
                "cargo",
                "test",
                "--release",
                "-p",
                "qwen-cli",
                "--bin",
                "qwen",
                "--",
                "--test-threads=1",
            ],
        ),
    )


def seal_artifact(decision: dict[str, object]) -> None:
    write_json(ARTIFACT / "decision.json", decision)
    for path in ARTIFACT.iterdir():
        if path.is_file():
            fsync_file(path)
    fsync_directory(ARTIFACT)
    members = sorted(
        path
        for path in ARTIFACT.iterdir()
        if path.is_file()
        and path.name not in {"artifact-inventory.sha256", "packet-complete.json"}
    )
    lines = [f"{sha256_file(path)}  {path.name}\n" for path in members]
    write_new(ARTIFACT / "artifact-inventory.sha256", "".join(lines).encode())
    completion = {
        "schema": 1,
        "decision_sha256": sha256_file(ARTIFACT / "decision.json"),
        "inventory_sha256": sha256_file(ARTIFACT / "artifact-inventory.sha256"),
        "inventory_members": len(members),
        "final_files": len(members) + 2,
    }
    write_json(ARTIFACT / "packet-complete.json", completion)


def reserve_artifact(manifest: dict[str, object]) -> None:
    if ARTIFACT.exists() or WORK_ROOT.exists():
        raise ContractDefect("refusing to reuse v0.637 artifact or work root")
    if not ARTIFACT.parent.is_dir():
        raise ContractDefect(f"artifact parent is missing: {ARTIFACT.parent}")
    ARTIFACT.mkdir()
    fsync_directory(ARTIFACT.parent)
    write_json(ARTIFACT / "manifest.json", manifest)


def run(*, preflight_only: bool) -> None:
    v0636_bridge = parser_self_test()
    source = verify_source(v0636_bridge)
    env, environment = child_environment()
    manifest = build_manifest(source, environment)
    if preflight_only:
        command(test_command()[:6] + ["--no-run"], env=env)
        print(json_bytes({"status": "preflight-pass", "source": source}).decode())
        return

    reserve_artifact(manifest)
    rows: list[dict[str, object]] = []
    status = "implementation_or_contract_defect"
    authority = "no-production-authority"
    successor = "none"
    error: str | None = None
    floor: dict[str, object] | None = None
    gates: list[dict[str, object]] = []
    signals_blocked = False
    try:
        for name, argv in required_gates():
            gates.append(run_gate(name, argv, env))
        if verify_source(v0636_bridge) != source:
            raise ContractDefect("source drifted after build/test gates")
        for pair_index, (order, first, second) in enumerate(PAIR_ORDERS, 1):
            for position, arm in enumerate((first, second), 1):
                row = launch_child(
                    arm,
                    pair_index,
                    order,
                    position,
                    env,
                    source,
                )
                append_jsonl(ARTIFACT / "attempts.jsonl", row)
                rows.append(row)
        if verify_source(v0636_bridge) != source:
            raise ContractDefect("source drifted after final floor child")
        floor = analyze(rows)
        status = "go" if floor["passes"] else "kill"
        if status == "go":
            successor = "preregister-one-real-27b-deferred-restore-product-packet"
        signal.pthread_sigmask(signal.SIG_BLOCK, OPERATOR_SIGNALS)
        signals_blocked = True
    except (KeyboardInterrupt, IncompleteRun) as failure:
        signal.pthread_sigmask(signal.SIG_BLOCK, OPERATOR_SIGNALS)
        signals_blocked = True
        status = "inconclusive"
        error = f"{type(failure).__name__}: {failure}"
    except Exception as failure:
        signal.pthread_sigmask(signal.SIG_BLOCK, OPERATOR_SIGNALS)
        signals_blocked = True
        error = f"{type(failure).__name__}: {failure}"
        floor = None
    finally:
        if not signals_blocked:
            signal.pthread_sigmask(signal.SIG_BLOCK, OPERATOR_SIGNALS)
            signals_blocked = True
        if WORK_ROOT.exists():
            try:
                shutil.rmtree(WORK_ROOT)
            except Exception as cleanup_failure:
                status = "implementation_or_contract_defect"
                successor = "none"
                cleanup_text = (
                    f"cleanup failed: {type(cleanup_failure).__name__}: "
                    f"{cleanup_failure}"
                )
                error = f"{error}; {cleanup_text}" if error else cleanup_text
        pending = set(signal.sigpending()) & OPERATOR_SIGNALS
        if pending:
            pending_text = "pending operator signals: " + ",".join(
                str(int(value)) for value in sorted(pending)
            )
            error = f"{error}; {pending_text}" if error else pending_text
            if status != "implementation_or_contract_defect":
                status = "inconclusive"
                successor = "none"
    decision = {
        "schema": 1,
        "status": status,
        "authority": authority,
        "force_authorized": False,
        "successor_authorization": successor,
        "execution_commit": source["execution_commit"],
        "implementation_commit": source["implementation_commit"],
        "v0636_forensic_bridge": source["v0636_forensic_bridge"],
        "v0636_authority_imported": False,
        "v0636_gate_results_imported": False,
        "v0636_scored_rows_imported": 0,
        "v0636_performance_observations_imported": 0,
        "v0636_timing_observations_imported": 0,
        "v0636_launched_children_forensic": 1,
        "v0636_b_observations_forensic": 0,
        "stage": "exact-size-floor",
        "gates": gates,
        "rows": rows,
        "floor": floor,
        "error": error,
    }
    pending = set(signal.sigpending()) & OPERATOR_SIGNALS
    if pending:
        pending_text = "pending operator signals before publication: " + ",".join(
            str(int(value)) for value in sorted(pending)
        )
        decision["error"] = (
            f"{decision['error']}; {pending_text}"
            if decision["error"]
            else pending_text
        )
        if decision["status"] != "implementation_or_contract_defect":
            decision["status"] = "inconclusive"
            decision["successor_authorization"] = "none"
            status = "inconclusive"
    seal_artifact(decision)
    if status == "implementation_or_contract_defect":
        raise ContractDefect(error or "unknown v0.637 contract defect")


def handle_operator_signal(signum: int, _frame: object) -> None:
    raise IncompleteRun(f"received signal {signum}")


def parser_self_test() -> dict[str, object]:
    sample = (
        b"[checkpoint-deferred-floor] schema=1 mode=deferred-restore "
        b"record_bytes=582854188 encoder_blake3="
        + EXPECTED_ENCODER_BLAKE3.encode("ascii")
        + b" blob_sha256="
        + EXPECTED_BLOB_SHA256.encode("ascii")
        + b" staged_integrity_us=100 publish_us=300000 full_decode_us=250000 "
        b"outcome=published evicted=0 managed_bytes_after=582854188 "
        b"vocab_size=248320 max_context=6516 max_record_bytes=805306368 "
        b"store_budget_bytes=805306368 "
        b"matched=6500 restored=6499 pending=true file_mode=0600 nlink=1 "
        b"temp_files=0 blob_relative=" + EXPECTED_BLOB_RELATIVE.encode("ascii")
    )

    def rejected(stdout: bytes, stderr: bytes = b"") -> None:
        try:
            parse_marker(stdout, stderr, "deferred-restore")
        except ContractDefect:
            return
        raise ContractDefect("self-test accepted a malformed marker")

    parsed = parse_marker(sample + b"\n", b"", "deferred-restore")
    if parsed["record_bytes"] != EXPECTED_RECORD_BYTES or parsed["pending"] is not True:
        raise ContractDefect("self-test standalone marker result drifted")
    parse_marker(LIBTEST_LEAD + sample + b"\r\n", b"", "deferred-restore")
    with tempfile.TemporaryDirectory(prefix="v0637-bridge-self-test-") as temporary:
        temporary_path = Path(temporary)
        regular = temporary_path / "regular"
        regular.write_bytes(b"sealed")
        member_link = temporary_path / "member-link"
        member_link.symlink_to(regular)
        try:
            read_regular_directory(temporary_path)
        except ContractDefect:
            pass
        else:
            raise ContractDefect("self-test accepted a symlinked sealed member")
        member_link.unlink()
        directory_link = temporary_path.parent / (temporary_path.name + "-link")
        directory_link.symlink_to(temporary_path, target_is_directory=True)
        try:
            read_regular_directory(directory_link)
        except ContractDefect:
            pass
        else:
            raise ContractDefect("self-test followed a symlinked sealed directory")
        finally:
            directory_link.unlink(missing_ok=True)
    bridge = authenticate_v0636_bridge()
    rejected(sample + b"\n" + sample)
    rejected(sample + b" " + sample)
    rejected(sample, sample)
    rejected(b"", sample)
    rejected(sample.replace(b"schema=1 mode=", b"mode=deferred-restore schema=1 "))
    rejected(sample + b" trailing")
    rejected(sample.replace(MARKER_PREFIX, b"[checkpoint-deferred-\nfloor]"))
    rejected(b"test wrong::test ... " + sample)
    rejected(sample + b"\n\xff")
    rejected(sample.replace(b"outcome=published", b"outcome=pub\xc3\xa9"))
    rejected(sample + b"\r\r\n")
    rows = []
    for pair_index, (order, first, second) in enumerate(PAIR_ORDERS, 1):
        for position, arm in enumerate((first, second), 1):
            marker = dict(parsed)
            marker["mode"] = arm_mode(arm)
            marker["staged_integrity_us"] = 250_000 if arm == "A" else 0
            marker["publish_us"] = 500_000 if arm == "A" else 250_000
            rows.append(
                {
                    "pair_index": pair_index,
                    "pair_order": order,
                    "position": position,
                    "arm": arm,
                    "marker": marker,
                    "store": {"blob_sha256": EXPECTED_BLOB_SHA256},
                }
            )
    if analyze(rows)["passes"] is not True:
        raise ContractDefect("self-test rejected the exact 250000-us boundary")
    rows[0]["marker"]["publish_us"] -= 1
    if analyze(rows)["passes"] is not False:
        raise ContractDefect("self-test accepted a 249999-us boundary")
    return bridge


def self_test() -> None:
    parser_self_test()
    print(
        json_bytes({"optimized_mode_safe": True, "status": "self-test-pass"}).decode()
    )


if __name__ == "__main__":
    signal.signal(signal.SIGINT, handle_operator_signal)
    signal.signal(signal.SIGTERM, handle_operator_signal)
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--self-test", action="store_true")
    group.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        run(preflight_only=args.preflight_only)
