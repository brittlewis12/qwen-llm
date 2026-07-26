#!/usr/bin/env python3
"""v0.639 repaired real-27B deferred-restore product packet runner."""

from __future__ import annotations

import argparse
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import re
import resource
import shutil
import signal
import select
import stat
import statistics
import subprocess
import sys
import tempfile
import threading
import time

import v0602_a3b_parallel_copied_loader as host_protocol


ROOT = Path(__file__).resolve().parents[2]
RUNNER = Path(__file__).resolve()
PREREG = ROOT / "docs/bench/v0639-checkpoint-deferred-restore-product-repair.md"
ARTIFACT = ROOT / "target/profiles/v0639-checkpoint-deferred-restore-product-repair-p1"
WORK_ROOT = (
    ROOT / "target/profiles/v0639-checkpoint-deferred-restore-product-repair-work"
)
V0638_PREREG = ROOT / "docs/bench/v0638-checkpoint-deferred-restore-product.md"
V0638_RUNNER = ROOT / "scripts/profile/v0638_checkpoint_deferred_restore_product.py"
V0638_ARTIFACT = ROOT / "target/profiles/v0638-checkpoint-deferred-restore-product-p1"
V0638_WORK_ROOT = (
    ROOT / "target/profiles/v0638-checkpoint-deferred-restore-product-work"
)
SEALED_V0637 = ROOT / "target/profiles/v0637-checkpoint-deferred-restore-floor-p1"
MODEL = Path("/Users/tito/models/Qwen3.6-27B-Q4_K_M.gguf")
MESSAGES = ROOT / "target/profiles/v0615-long-history/ring0-turn-2.json"
IDENTITY_SEED = (
    ROOT / "target/profiles/v0615-long-history/cache-ring0/v1/identity/"
    "3f6fb8c12c7fbfe881e2c43b7d98742873161ffa51a5e039773252207541734e.mid"
)
PACKET_MESSAGES = ARTIFACT / "input-messages.json"
PACKET_IDENTITY = ARTIFACT / "input-identity.mid"
RESTORE_MESSAGES = ARTIFACT / "restore-messages.json"
CLI = ROOT / "target/release/qwen"
BENCH = ROOT / "target/release/qwen-bench"
CLI_SOURCE = ROOT / "crates/qwen-cli/src/main.rs"
STORE_SOURCE = ROOT / "crates/qwen-llm/src/checkpoint_store.rs"
HOST_HELPER = ROOT / "scripts/profile/v0602_a3b_parallel_copied_loader.py"
COMMON_HELPER = ROOT / "scripts/profile/v0593_demand_paged_no_copy.py"

PARENT = "ca2ee114d3140c1ed3bb577b6b7293ff8ec865ee"
V0638_PARENT = "482fa05341971755ecfcc0e1820a09be65490356"
IMPLEMENTATION = "3d59d94cf314dae79b79cd05f74b4aea455cd30c"
V0637_COMMIT = "61f852e557d42180468a9c7ce77ccf606b68d8e8"
GGUF_COMMIT = "c7369fd4868a6f613459fff355477f53bf4ee2f1"
LLAMA_COMMIT = "fe4fb533d1ed2855b6ac5492e56c42007d410409"
LLAMA_UNTRACKED = (".claude/settings.local.json",)
SOURCE_STATE_PREFIX = "git-source-sha256-v2:"
V0638_SHA256 = {
    V0638_PREREG: "2fdeb1f5980ebbb8a51eaaacfcacd3b6ac25ccd1acb34a5c509e32921e987309",
    V0638_RUNNER: "da99dcc7f8feeaff70e1abe21bcfad3a0f5afd45dba9ce7170bdc8cd9c391efd",
}
FROZEN_SHA256 = {
    ROOT
    / "Cargo.lock": "2ea5d138ef0b5813ce29ad39c39d4a34d31d5a72fb71431f6038c5343d2e4fbe",
    ROOT
    / "Cargo.toml": "c92ed7c228b43de3a2920cc5d2a7535660fd328cbde1444bf315545c36b162e9",
    CLI_SOURCE: "1ee1e959fe5c2616fec20fb924b9edef026fc432b635db1f46166a43a1ce0c71",
    STORE_SOURCE: "a5ea6b001249418febb3f876253bf5989c0603ef42cc43bc50fda7d51927f8da",
    HOST_HELPER: "e5775489802dddc2d88dc9b950940417f325ebcc4ece29194550b7e430f851e3",
    COMMON_HELPER: "4c1b73a2897d8bf882988407f45b59be7b19d236c06a612991491c57af0d7be4",
}
SEALED_HASHES = {
    "decision.json": "29c4ae3c50eac56001a8ae89c226ce490f1d4617c7bd6f01d8b2dd44d304c3a0",
    "artifact-inventory.sha256": "8d218da1a4d20980256d4584e972e8ddb5f6304f5967872084fb96b117666403",
    "packet-complete.json": "6f1ed3481a729b565a05a70996d17a17f2e638819255248113df84fe65cf41f4",
    "attempts.jsonl": "18796d6137b4542c975661765474ab77b7d3fca7412d84f25dcd1160c87a8382",
}
MODEL_SHA256 = "5ed60d0af4650a854b1755bd392f9aef4872643dc25a254bc68043fa638392a0"
MESSAGES_SHA256 = "5c2455738af1d789ce1f86cf1d2064fb8e80a242c80273273f57c25783bdd4a1"
IDENTITY_SHA256 = "17afb5fc1e9c3e56b7ffd2612f501d1714eefbb3c1e81d27e9501e0a18120c69"
BLOB_SHA256 = "69c883f5130e5108cc3b948c5cc500c4d92db1fc2f96aa25b75f5f38abda1e65"
ENCODER_TRAILER = "2833fd870009a299dbaa389ae7ed379ea69d50fc31ff250f4e349dc3056eeb67"
STDOUT_BYTES = b"<think>\n"
STDOUT_SHA256 = "9ebc01769b176bb074a065ea0974c130fc8afd12814360aaf809046160b2a999"
MODEL_BYTES = 16_817_244_384
BLOB_BYTES = 582_854_188
IDENTITY_BYTES = 128
COMPATIBILITY = "fe9d70c202425683190d1c6a3ef50474940915e8c2e918f3ad50574e12e92b4e"
BLOB_NAME = (
    "6500-p0-63362870ab8f272dfc0c5a5f1dbbe474921b1554a4f589e728d5a2cfbb1bbc2a.qcp"
)
BLOB_RELATIVE = Path("v1/blobs") / COMPATIBILITY / BLOB_NAME
PAIR_ORDERS = ("AB", "BA", "BA", "AB", "AB", "BA")
INTEGRITY_ENV = "QWEN_CHECKPOINT_STAGED_INTEGRITY"
GATE_US = 250_000
MAX_STREAM_BYTES = 16 * 1024 * 1024
OPERATOR_SIGNALS = {signal.SIGINT, signal.SIGTERM}
CHILD_WAIT_SIGNALS = OPERATOR_SIGNALS | {signal.SIGCHLD, signal.SIGALRM}


class ContractDefect(RuntimeError):
    pass


class Inconclusive(RuntimeError):
    def __init__(self, stage: str, detail: str):
        super().__init__(detail)
        self.stage = stage


class UnreapedProcess(BaseException):
    pass


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def path_lexists(path: Path) -> bool:
    return os.path.lexists(path)


def json_bytes(value: object, *, pretty: bool = False) -> bytes:
    options = {"sort_keys": True, "ensure_ascii": True, "allow_nan": False}
    if pretty:
        options["indent"] = 2
    else:
        options["separators"] = (",", ":")
    return (json.dumps(value, **options) + "\n").encode("ascii")


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_new(path: Path, data: bytes, mode: int = 0o600) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        mode,
    )
    try:
        view = memoryview(data)
        while view:
            count = os.write(descriptor, view)
            if count <= 0:
                raise OSError(f"short write: {path}")
            view = view[count:]
        os.fsync(descriptor)
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise ContractDefect(f"new artifact is not private regular file: {path}")
    finally:
        os.close(descriptor)
    fsync_directory(path.parent)


def write_json(path: Path, value: object) -> None:
    write_new(path, json_bytes(value, pretty=True))


def append_jsonl(path: Path, value: object) -> None:
    data = json_bytes(value)
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW,
        0o600,
    )
    try:
        view = memoryview(data)
        while view:
            count = os.write(descriptor, view)
            if count <= 0:
                raise OSError(f"short append: {path}")
            view = view[count:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    fsync_directory(path.parent)


def command(
    argv: list[str], *, cwd: Path = ROOT, env: dict[str, str] | None = None
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


def git_bytes(root: Path, *args: str) -> bytes:
    process = subprocess.run(
        ["git", *args],
        cwd=root,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if process.returncode != 0:
        detail = process.stderr.decode("utf-8", errors="replace").strip()
        raise ContractDefect(f"git {' '.join(args)} failed in {root}: {detail}")
    return process.stdout


def git(argv: list[str], *, cwd: Path = ROOT) -> str:
    return command(["git", *argv], cwd=cwd).strip()


def hash_section(digest: hashlib._Hash, label: bytes, value: bytes) -> None:
    digest.update(len(label).to_bytes(8, "big"))
    digest.update(label)
    digest.update(len(value).to_bytes(8, "big"))
    digest.update(value)


def hash_worktree_entry(
    digest: hashlib._Hash, root: Path, scope: bytes, relative: bytes
) -> None:
    hash_section(digest, b"entry-scope", scope)
    hash_section(digest, b"entry-path", relative)
    path = os.path.join(os.fsencode(root), relative)
    try:
        metadata = os.lstat(path)
    except FileNotFoundError:
        hash_section(digest, b"entry-kind", b"missing")
        return
    if stat.S_ISLNK(metadata.st_mode):
        hash_section(digest, b"entry-kind", b"symlink")
        hash_section(digest, b"entry-content", os.readlink(path))
        return
    if not stat.S_ISREG(metadata.st_mode):
        raise ContractDefect(f"unsupported source entry type: {os.fsdecode(path)}")
    hash_section(digest, b"entry-kind", b"file")
    hash_section(
        digest,
        b"entry-executable",
        bytes([int(bool(metadata.st_mode & 0o111))]),
    )
    content = hashlib.sha256()
    size = 0
    with open(path, "rb") as source:
        while chunk := source.read(64 * 1024):
            content.update(chunk)
            size += len(chunk)
    hash_section(digest, b"entry-size", size.to_bytes(8, "big"))
    hash_section(digest, b"entry-content-sha256", content.digest())


def tracked_source_state(root: Path) -> str:
    head = git_bytes(root, "rev-parse", "HEAD")
    index = git_bytes(root, "ls-files", "--stage", "-z")
    index_flags = git_bytes(root, "ls-files", "-v", "-z")
    tracked = git_bytes(root, "ls-files", "-z")
    untracked = git_bytes(root, "ls-files", "--others", "--exclude-standard", "-z")
    digest = hashlib.sha256()
    digest.update(b"qwen-git-source-state-v2\0")
    hash_section(digest, b"head", head)
    hash_section(digest, b"index", index)
    hash_section(digest, b"index-flags", index_flags)
    hash_section(digest, b"tracked-paths", tracked)
    hash_section(digest, b"untracked-paths", untracked)
    for path in tracked.split(b"\0"):
        if path:
            hash_worktree_entry(digest, root, b"tracked", path)
    for path in untracked.split(b"\0"):
        if path:
            hash_worktree_entry(digest, root, b"untracked", path)
    return f"{SOURCE_STATE_PREFIX}{digest.hexdigest()}"


def source_identity(root: Path) -> tuple[str, bool, str]:
    commit_bytes = git_bytes(root, "rev-parse", "HEAD")
    try:
        commit = commit_bytes.decode("ascii").strip().lower()
    except UnicodeDecodeError as error:
        raise ContractDefect("git rev-parse HEAD returned non-ASCII") from error
    if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise ContractDefect(f"git rev-parse HEAD returned invalid commit: {commit!r}")
    flags = git_bytes(root, "ls-files", "-v", "-z")
    hidden = any(
        entry[:1].islower() or entry.startswith(b"S")
        for entry in flags.split(b"\0")
        if entry
    )
    dirty = (
        bool(git_bytes(root, "status", "--porcelain", "--untracked-files=all"))
        or hidden
    )
    return commit, dirty, tracked_source_state(root)


def json_from_sealed(data: bytes, label: str) -> object:
    try:
        return json.loads(data.decode("utf-8", errors="strict"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ContractDefect(f"invalid sealed v0.637 JSON: {label}") from error


def read_regular_directory_once(path: Path) -> dict[str, bytes]:
    try:
        directory = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    except OSError as error:
        raise ContractDefect(
            f"cannot descriptor-open sealed directory: {path}"
        ) from error
    try:
        names = sorted(os.listdir(directory))
        data: dict[str, bytes] = {}
        stamps: dict[str, tuple[int, int, int, int]] = {}
        for name in names:
            try:
                descriptor = os.open(
                    name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory
                )
            except OSError as error:
                raise ContractDefect(
                    f"cannot descriptor-open sealed member: {name}"
                ) from error
            try:
                before = os.fstat(descriptor)
                if not stat.S_ISREG(before.st_mode):
                    raise ContractDefect(f"sealed member is not regular: {name}")
                chunks = []
                while chunk := os.read(descriptor, 1024 * 1024):
                    chunks.append(chunk)
                after = os.fstat(descriptor)
                stamp = (before.st_dev, before.st_ino, before.st_size, before.st_nlink)
                if stamp != (after.st_dev, after.st_ino, after.st_size, after.st_nlink):
                    raise ContractDefect(f"sealed member changed during read: {name}")
                payload = b"".join(chunks)
                if len(payload) != before.st_size:
                    raise ContractDefect(f"sealed member short read: {name}")
                data[name] = payload
                stamps[name] = stamp
            finally:
                os.close(descriptor)
        if sorted(os.listdir(directory)) != names:
            raise ContractDefect("sealed directory population changed")
        for name, expected in stamps.items():
            observed = os.stat(name, dir_fd=directory, follow_symlinks=False)
            stamp = (
                observed.st_dev,
                observed.st_ino,
                observed.st_size,
                observed.st_nlink,
            )
            if not stat.S_ISREG(observed.st_mode) or stamp != expected:
                raise ContractDefect(f"sealed member identity changed: {name}")
        return data
    finally:
        os.close(directory)


def authenticate_v0637() -> dict[str, object]:
    sealed = read_regular_directory_once(SEALED_V0637)
    if len(sealed) != 23:
        raise ContractDefect("sealed v0.637 regular-file count drifted")
    for name, expected in SEALED_HASHES.items():
        if name not in sealed or sha256_bytes(sealed[name]) != expected:
            raise ContractDefect(f"sealed v0.637 hash drifted: {name}")
    inventory: dict[str, str] = {}
    for line in sealed["artifact-inventory.sha256"].splitlines():
        match = re.fullmatch(rb"([0-9a-f]{64})  ([A-Za-z0-9._-]+)", line)
        if match is None:
            raise ContractDefect("sealed v0.637 inventory grammar drifted")
        name = match.group(2).decode("ascii")
        if name in inventory:
            raise ContractDefect("sealed v0.637 inventory duplicate")
        inventory[name] = match.group(1).decode("ascii")
    if len(inventory) != 21 or set(sealed) != set(inventory) | {
        "artifact-inventory.sha256",
        "packet-complete.json",
    }:
        raise ContractDefect("sealed v0.637 inventory population drifted")
    if {name: sha256_bytes(sealed[name]) for name in inventory} != inventory:
        raise ContractDefect("sealed v0.637 inventory member drifted")
    completion = json_from_sealed(sealed["packet-complete.json"], "completion")
    if completion != {
        "decision_sha256": SEALED_HASHES["decision.json"],
        "final_files": 23,
        "inventory_members": 21,
        "inventory_sha256": SEALED_HASHES["artifact-inventory.sha256"],
        "schema": 1,
    }:
        raise ContractDefect("sealed v0.637 completion binding drifted")
    decision = json_from_sealed(sealed["decision.json"], "decision")
    if not isinstance(decision, dict) or any(
        (
            decision.get("status") != "go",
            decision.get("authority") != "no-production-authority",
            decision.get("force_authorized") is not False,
            decision.get("successor_authorization")
            != "preregister-one-real-27b-deferred-restore-product-packet",
            decision.get("execution_commit") != V0637_COMMIT,
            decision.get("implementation_commit") != IMPLEMENTATION,
            decision.get("error") is not None,
        )
    ):
        raise ContractDefect("sealed v0.637 decision authority drifted")
    rows = decision.get("rows")
    floor = decision.get("floor")
    if not isinstance(rows, list) or len(rows) != 4 or not isinstance(floor, dict):
        raise ContractDefect("sealed v0.637 floor population drifted")
    expected = (
        (1, "AB", 1, "A", "decode"),
        (1, "AB", 2, "B", "deferred-restore"),
        (2, "BA", 1, "B", "deferred-restore"),
        (2, "BA", 2, "A", "decode"),
    )
    for row, shape in zip(rows, expected, strict=True):
        pair, order, position, arm, mode = shape
        marker = row.get("marker") if isinstance(row, dict) else None
        store = row.get("store") if isinstance(row, dict) else None
        if any(
            (
                not isinstance(marker, dict),
                not isinstance(store, dict),
                row.get("valid") is not True,
                row.get("pair_index") != pair,
                row.get("pair_order") != order,
                row.get("position") != position,
                row.get("arm") != arm,
                row.get("mode") != mode,
            )
        ):
            raise ContractDefect("sealed v0.637 exact floor row drifted")
        if (
            marker.get("mode") != mode
            or marker.get("record_bytes") != BLOB_BYTES
            or marker.get("blob_sha256") != BLOB_SHA256
            or marker.get("encoder_blake3") != ENCODER_TRAILER
            or marker.get("outcome") != "published"
            or marker.get("evicted") != 0
            or marker.get("matched") != 6500
            or marker.get("restored") != 6499
            or marker.get("pending") is not True
            or marker.get("file_mode") != "0600"
            or marker.get("nlink") != 1
            or marker.get("temp_files") != 0
            or marker.get("blob_relative") != str(BLOB_RELATIVE)
            or store.get("blob_sha256") != BLOB_SHA256
        ):
            raise ContractDefect("sealed v0.637 row evidence drifted")
    try:
        attempts = [json.loads(line) for line in sealed["attempts.jsonl"].splitlines()]
    except json.JSONDecodeError as error:
        raise ContractDefect("sealed v0.637 attempts are invalid") from error
    if attempts != rows:
        raise ContractDefect("sealed v0.637 attempts do not bind exact rows")
    pairs = floor.get("pairs")
    if (
        floor.get("passes") is not True
        or not isinstance(pairs, list)
        or len(pairs) != 2
    ):
        raise ContractDefect("sealed v0.637 floor result drifted")
    for pair in pairs:
        if (
            pair.get("integrity_gate") is not True
            or pair.get("publish_gate") is not True
        ):
            raise ContractDefect("sealed v0.637 exact floor gate drifted")
        if (
            pair.get("integrity_saving_us", -1) < GATE_US
            or pair.get("publish_saving_us", -1) < GATE_US
        ):
            raise ContractDefect("sealed v0.637 floor threshold drifted")
    gates = decision.get("gates")
    if (
        not isinstance(gates, list)
        or len(gates) != 4
        or any(
            not isinstance(gate, dict) or gate.get("returncode") != 0 for gate in gates
        )
    ):
        raise ContractDefect("sealed v0.637 gate descriptor drifted")
    return {
        "path": str(SEALED_V0637),
        "final_regular_files": 23,
        "inventory_members": 21,
        "seal_sha256": dict(SEALED_HASHES),
        "status": "go",
        "authority": "no-production-authority",
        "successor_authorization": decision["successor_authorization"],
        "exact_floor_rows": 4,
    }


def single_parent(commit: str) -> str:
    fields = git(["rev-list", "--parents", "-n", "1", commit]).split()
    if len(fields) != 2 or fields[0] != commit:
        raise ContractDefect(f"commit is not single-parent: {commit}")
    return fields[1]


def authenticate_v0638_commit() -> dict[str, object]:
    if single_parent(PARENT) != V0638_PARENT:
        raise ContractDefect("ca2ee114 is not a direct child of exact 482fa053")
    changes = sorted(
        git(["diff", "--name-status", f"{V0638_PARENT}..{PARENT}"]).splitlines()
    )
    expected = sorted(
        (f"A\t{V0638_PREREG.relative_to(ROOT)}", f"A\t{V0638_RUNNER.relative_to(ROOT)}")
    )
    if changes != expected:
        raise ContractDefect("ca2ee114 does not add exactly the v0.638 doc and runner")
    committed = {}
    for path, expected_digest in V0638_SHA256.items():
        relative = str(path.relative_to(ROOT))
        payload = git_bytes(ROOT, "show", f"{PARENT}:{relative}")
        digest = sha256_bytes(payload)
        if digest != expected_digest or sha256_file(path) != expected_digest:
            raise ContractDefect(f"committed v0.638 bytes drifted: {relative}")
        committed[relative] = digest
    return {
        "commit": PARENT,
        "parent_commit": V0638_PARENT,
        "added_files_sha256": committed,
    }


def verify_sibling(
    path: Path, commit: str, allowed: tuple[str, ...] = ()
) -> dict[str, object]:
    if git(["rev-parse", "HEAD"], cwd=path) != commit:
        raise ContractDefect(f"sibling commit drifted: {path}")
    status_text = git(["status", "--porcelain=v1", "--untracked-files=all"], cwd=path)
    lines = status_text.splitlines() if status_text else []
    if any(not line.startswith("?? ") for line in lines):
        raise ContractDefect(f"sibling tracked state dirty: {path}")
    untracked = sorted(line[3:] for line in lines)
    if untracked != sorted(allowed):
        raise ContractDefect(f"sibling untracked state drifted: {path}: {untracked}")
    return {"path": str(path), "commit": commit, "allowed_untracked": untracked}


def verify_source(bridge: dict[str, object]) -> dict[str, object]:
    tracked = (RUNNER, PREREG, *FROZEN_SHA256)
    for path in tracked:
        git(["ls-files", "--error-unmatch", str(path.relative_to(ROOT))])
    dirty = git(["status", "--porcelain=v1", "--untracked-files=all"])
    if dirty:
        raise ContractDefect(f"execution worktree is dirty: {dirty!r}")
    execution = git(["rev-parse", "HEAD"])
    if single_parent(execution) != PARENT:
        raise ContractDefect("R639 is not a direct single-parent child of exact parent")
    changes = sorted(
        git(["diff", "--name-status", f"{PARENT}..{execution}"]).splitlines()
    )
    expected = sorted(
        (f"A\t{PREREG.relative_to(ROOT)}", f"A\t{RUNNER.relative_to(ROOT)}")
    )
    if changes != expected:
        raise ContractDefect("R639 does not add exactly two authorized files")
    if not git(["merge-base", "--is-ancestor", IMPLEMENTATION, PARENT]) == "":
        raise ContractDefect("implementation is not an ancestor of R639 parent")
    if (
        single_parent(V0637_COMMIT) != IMPLEMENTATION
        or single_parent(V0638_PARENT) != V0637_COMMIT
    ):
        raise ContractDefect("frozen v0.637/v0.638-parent ancestry drifted")
    v0638_commit = authenticate_v0638_commit()
    actual = {path: sha256_file(path) for path in FROZEN_SHA256}
    if actual != FROZEN_SHA256:
        raise ContractDefect("frozen implementation/Cargo/helper hash drifted")
    return {
        "execution_commit": execution,
        "parent_commit": PARENT,
        "implementation_commit": IMPLEMENTATION,
        "v0637_execution_commit": V0637_COMMIT,
        "v0638_preregistration_commit": PARENT,
        "v0638_commit_authentication": v0638_commit,
        "file_sha256": {str(path): sha256_file(path) for path in tracked},
        "v0637_packet": bridge,
        "siblings": [
            verify_sibling(ROOT.parent / "gguf", GGUF_COMMIT),
            verify_sibling(ROOT.parent / "llama-cpp-rs", LLAMA_COMMIT, LLAMA_UNTRACKED),
        ],
    }


def normalized_environment() -> tuple[dict[str, str], dict[str, object]]:
    kept = {
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
    env = {name: value for name, value in os.environ.items() if name in kept}
    removed_controls = sorted(
        name
        for name in os.environ
        if name.startswith(("QWEN", "MTL", "METAL")) or name == "RUST_LOG"
    )
    if any(
        name.startswith(("QWEN", "MTL", "METAL")) or name == "RUST_LOG" for name in env
    ):
        raise ContractDefect("normalized environment retained a controlled variable")
    return env, {
        "effective": dict(sorted(env.items())),
        "removed_control_names": removed_controls,
        "removed_other_names": sorted(
            set(os.environ) - set(env) - set(removed_controls)
        ),
    }


def arm_mode(arm: str) -> str:
    if arm == "A":
        return "decode"
    if arm == "B":
        return "deferred-restore"
    raise ContractDefect(f"invalid arm: {arm}")


def arm_environment(base: dict[str, str], arm: str) -> dict[str, str]:
    env = base.copy()
    env[INTEGRITY_ENV] = arm_mode(arm)
    controls = [
        name
        for name in env
        if name.startswith(("QWEN", "MTL", "METAL")) or name == "RUST_LOG"
    ]
    if controls != [INTEGRITY_ENV]:
        raise ContractDefect(f"arm environment controls drifted: {controls}")
    return env


RAW_BUILD_POLICY = {
    "qwen": {
        "requires_full_commit": True,
        "requires_source_state": True,
    },
    "qwen-bench": {
        "requires_full_commit": False,
        "requires_source_state": True,
    },
}


def validate_raw_build_literals(
    binary_name: str, payload: bytes, execution: str, source_state: str
) -> dict[str, object]:
    policy = RAW_BUILD_POLICY.get(binary_name)
    if policy is None:
        raise ContractDefect(f"no raw build policy for {binary_name}")
    commit_present = execution.encode("ascii") in payload
    state_present = source_state.encode("ascii") in payload
    if policy["requires_full_commit"] and not commit_present:
        raise ContractDefect(f"full execution commit is absent from {binary_name}")
    if policy["requires_source_state"] and not state_present:
        raise ContractDefect(f"source state is absent from {binary_name}")
    return {
        "policy": dict(policy),
        "full_commit_present_diagnostic": commit_present,
        "source_state_present": state_present,
    }


def validate_bench_semantics(
    parsed: object, execution: str, source_state: str
) -> dict[str, object]:
    required = {
        "schema_version",
        "build_commit",
        "build_commit_short",
        "build_dirty",
        "build_source_state",
        "stamp_source",
        "stamp_error",
        "runtime_commit",
        "runtime_dirty",
        "runtime_source_state",
        "status",
        "problems",
        "overrides",
    }
    if not isinstance(parsed, dict) or required - parsed.keys():
        raise ContractDefect("qwen-bench build-info schema-2 fields are incomplete")
    expected = {
        "schema_version": 2,
        "build_commit": execution,
        "build_commit_short": execution[:9],
        "build_dirty": False,
        "build_source_state": source_state,
        "stamp_source": "git",
        "stamp_error": None,
        "runtime_commit": execution,
        "runtime_dirty": False,
        "runtime_source_state": source_state,
        "status": "match",
        "problems": [],
        "overrides": [],
    }
    if any(
        type(parsed.get(name)) is not type(value) or parsed.get(name) != value
        for name, value in expected.items()
    ):
        raise ContractDefect(f"qwen-bench semantic build identity mismatch: {parsed}")
    return parsed


def require_identity_equal(
    before: dict[str, object], after: dict[str, object], label: str
) -> None:
    if before != after:
        raise ContractDefect(f"{label} complete descriptor identity drifted")


def executable_descriptor(path: Path) -> tuple[bytes, dict[str, object]]:
    payload, identity = read_descriptor_regular(path)
    if payload is None:
        raise ContractDefect(f"executable descriptor retained no bytes: {path}")
    stamp = identity["stamp"]
    if stamp["nlink"] != 1 or stamp["mode"] != 0o755:
        raise ContractDefect(f"binary is not private executable regular data: {path}")
    return payload, identity


def bench_semantic_gate(
    execution: str, source_state: str, env: dict[str, str]
) -> tuple[dict[str, object], dict[str, object]]:
    _, before = executable_descriptor(BENCH)
    try:
        parsed = json.loads(
            command([str(BENCH), "build-info", "--output", "json"], env=env)
        )
    except json.JSONDecodeError as error:
        raise ContractDefect("qwen-bench build-info returned invalid JSON") from error
    _, after = executable_descriptor(BENCH)
    require_identity_equal(before, after, "qwen-bench build-info before/after")
    return validate_bench_semantics(parsed, execution, source_state), after


def verify_build(source: dict[str, object], env: dict[str, str]) -> dict[str, object]:
    execution, dirty, source_state = source_identity(ROOT)
    if execution != source["execution_commit"] or dirty:
        raise ContractDefect("independently derived execution source identity drifted")
    embedded = {}
    for binary in (CLI, BENCH):
        payload, identity = executable_descriptor(binary)
        embedded[binary.name] = {
            "descriptor_identity": identity,
            "raw_authentication": validate_raw_build_literals(
                binary.name, payload, execution, source_state
            ),
        }
    bench_info, bench_identity = bench_semantic_gate(execution, source_state, env)
    require_identity_equal(
        embedded["qwen-bench"]["descriptor_identity"],
        bench_identity,
        "qwen-bench raw/semantic gate",
    )
    return {
        "source_identity": {
            "execution_commit": execution,
            "dirty": False,
            "source_state": source_state,
        },
        "raw_policy": RAW_BUILD_POLICY,
        "qwen_bench_build_info": bench_info,
        "binaries": embedded,
    }


def read_descriptor_regular(
    path: Path, *, retain_bytes: bool = True
) -> tuple[bytes | None, dict[str, object]]:
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    except OSError as error:
        raise ContractDefect(
            f"cannot descriptor-open regular identity: {path}"
        ) from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ContractDefect(f"identity path is not regular: {path}")
        chunks = []
        digest = hashlib.sha256()
        total = 0
        while chunk := os.read(descriptor, 1024 * 1024):
            total += len(chunk)
            digest.update(chunk)
            if retain_bytes:
                chunks.append(chunk)
        after = os.fstat(descriptor)
        stamp = {
            "dev": before.st_dev,
            "ino": before.st_ino,
            "bytes": before.st_size,
            "mode": stat.S_IMODE(before.st_mode),
            "nlink": before.st_nlink,
            "mtime_ns": before.st_mtime_ns,
            "ctime_ns": before.st_ctime_ns,
        }
        if (
            stamp
            != {
                "dev": after.st_dev,
                "ino": after.st_ino,
                "bytes": after.st_size,
                "mode": stat.S_IMODE(after.st_mode),
                "nlink": after.st_nlink,
                "mtime_ns": after.st_mtime_ns,
                "ctime_ns": after.st_ctime_ns,
            }
            or total != before.st_size
        ):
            raise ContractDefect(f"identity changed during descriptor hash: {path}")
        return (b"".join(chunks) if retain_bytes else None), {
            "path": str(path),
            "sha256": digest.hexdigest(),
            "stamp": stamp,
        }
    finally:
        os.close(descriptor)


def descriptor_hash_regular(path: Path) -> dict[str, object]:
    _, identity = read_descriptor_regular(path, retain_bytes=False)
    return identity


def expected_binary_identity(
    build_identity: dict[str, object], name: str
) -> dict[str, object]:
    binaries = build_identity.get("binaries")
    binary = binaries.get(name) if isinstance(binaries, dict) else None
    identity = binary.get("descriptor_identity") if isinstance(binary, dict) else None
    if not isinstance(identity, dict):
        raise ContractDefect(
            f"post-gate build identity has no complete {name} identity"
        )
    return identity


def authenticate_qwen_binary(build_identity: dict[str, object]) -> dict[str, object]:
    _, observed = executable_descriptor(CLI)
    require_identity_equal(
        expected_binary_identity(build_identity, "qwen"),
        observed,
        "target/release/qwen post-gate",
    )
    return observed


def authenticate_qwen_bench(
    build_identity: dict[str, object], env: dict[str, str]
) -> dict[str, object]:
    source = build_identity.get("source_identity")
    if not isinstance(source, dict):
        raise ContractDefect("post-gate source identity is absent")
    semantic, observed = bench_semantic_gate(
        str(source.get("execution_commit")), str(source.get("source_state")), env
    )
    require_identity_equal(
        expected_binary_identity(build_identity, "qwen-bench"),
        observed,
        "target/release/qwen-bench post-gate",
    )
    if semantic != build_identity.get("qwen_bench_build_info"):
        raise ContractDefect("qwen-bench semantic result changed after fresh gates")
    return {"descriptor_identity": observed, "build_info": semantic}


def validate_frozen_inputs(*, hash_model: bool) -> dict[str, object]:
    expected = (
        (MESSAGES, MESSAGES_SHA256, None),
        (IDENTITY_SEED, IDENTITY_SHA256, IDENTITY_BYTES),
    )
    result = {}
    for path, digest, size in expected:
        metadata = path.stat()
        if (size is not None and metadata.st_size != size) or sha256_file(
            path
        ) != digest:
            raise ContractDefect(f"frozen input drifted: {path}")
        result[str(path)] = {"bytes": metadata.st_size, "sha256": digest}
    model = MODEL.stat()
    if model.st_size != MODEL_BYTES:
        raise ContractDefect("frozen model size drifted")
    model_digest = sha256_file(MODEL) if hash_model else MODEL_SHA256
    if model_digest != MODEL_SHA256:
        raise ContractDefect("frozen model digest drifted")
    result[str(MODEL)] = {"bytes": model.st_size, "sha256": model_digest}
    return result


def host_boundary(env: dict[str, str]) -> dict[str, object]:
    device = command([str(CLI), "--info"], env=env).strip()
    macos = command(["sw_vers", "-productVersion"], env=env).strip()
    memory = int(command(["sysctl", "-n", "hw.memsize"], env=env).strip())
    if (
        device != host_protocol.EXPECTED_DEVICE
        or not macos.startswith("15.")
        or memory != host_protocol.EXPECTED_HW_MEMSIZE
    ):
        raise ContractDefect("frozen host boundary drifted")
    return {"device": device, "macos": macos, "hw_memsize": memory}


def authority_history() -> dict[str, object]:
    return {
        "authority_origin": V0637_COMMIT,
        "authorized_preregistration": PARENT,
        "sole_successor_authorization_consumed_by": "v0.638",
        "v0638_successor_authorization_imported": True,
        "v0638_successor_authorization_consumed": True,
        "v0639_additional_authorization_imported": False,
        "v0639_additional_authorization_consumed": False,
        "packet_ordinal": 1,
        "v0638_packet_reserved": False,
        "v0638_packet_imported": False,
        "v0638_artifact_imported": False,
        "v0638_work_root_imported": False,
        "product_authority_imported": False,
        "gate_results_imported": 0,
        "scored_rows_imported": 0,
        "timing_observations_imported": 0,
        "performance_observations_imported": 0,
    }


def build_manifest(
    source: dict[str, object],
    environment: dict[str, object],
    env: dict[str, str],
    *,
    require_build: bool,
) -> dict[str, object]:
    messages = MESSAGES.read_bytes()
    restore = create_restore_payload(messages)
    return {
        "schema": 1,
        "created_unix_ms": time.time_ns() // 1_000_000,
        "source": source,
        "execution_commit": source["execution_commit"],
        "implementation_commit": IMPLEMENTATION,
        "build_identity": verify_build(source, env) if require_build else None,
        "post_gate_identity_artifact": "execution-identity.json",
        "host_boundary": host_boundary(env) if require_build else None,
        "inputs": validate_frozen_inputs(hash_model=True),
        "packet_input_expectations": {
            str(PACKET_MESSAGES.relative_to(ROOT)): {
                "bytes": len(messages),
                "sha256": MESSAGES_SHA256,
            },
            str(PACKET_IDENTITY.relative_to(ROOT)): {
                "bytes": IDENTITY_BYTES,
                "sha256": IDENTITY_SHA256,
            },
            str(RESTORE_MESSAGES.relative_to(ROOT)): {
                "bytes": len(restore),
                "sha256": sha256_bytes(restore),
            },
        },
        "environment": environment,
        "pair_orders": list(PAIR_ORDERS),
        "gate_us": GATE_US,
        "retry_count": 0,
        "authority_history": authority_history(),
        "packet_signal_policy": {
            "sigalrm_reserved_blocked_until_process_exit": True,
            "mask_restore_is_not_claimed_atomic": True,
        },
    }


def create_restore_payload(source: bytes) -> bytes:
    try:
        payload = json.loads(source.decode("utf-8", errors="strict"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ContractDefect(
            "messages input is not strict canonical JSON data"
        ) from error
    messages = payload.get("messages") if isinstance(payload, dict) else None
    if not isinstance(messages, list):
        raise ContractDefect("messages input has no message list")
    messages.extend(
        (
            {"role": "assistant", "content": "<think>\n"},
            {"role": "user", "content": "Continue with one short sentence."},
        )
    )
    return json_bytes(payload, pretty=True)


def freeze_inputs() -> dict[str, object]:
    messages = MESSAGES.read_bytes()
    identity = IDENTITY_SEED.read_bytes()
    if (
        sha256_bytes(messages) != MESSAGES_SHA256
        or sha256_bytes(identity) != IDENTITY_SHA256
    ):
        raise ContractDefect("source input changed before packet freeze")
    restore = create_restore_payload(messages)
    rows = []
    for source, target, data in (
        (MESSAGES, PACKET_MESSAGES, messages),
        (IDENTITY_SEED, PACKET_IDENTITY, identity),
        (MESSAGES, RESTORE_MESSAGES, restore),
    ):
        write_new(target, data)
        metadata = target.stat(follow_symlinks=False)
        if stat.S_IMODE(metadata.st_mode) != 0o600 or metadata.st_nlink != 1:
            raise ContractDefect("packet-local input mode/link drifted")
        rows.append(
            {
                "source": str(source),
                "packet_path": str(target.relative_to(ROOT)),
                "bytes": metadata.st_size,
                "sha256": sha256_file(target),
            }
        )
    seal = {"schema": 1, "inputs": rows}
    write_json(ARTIFACT / "input-seal.json", seal)
    return seal


def authenticate_packet_file(
    path: Path, expected_size: int, expected_sha256: str
) -> dict[str, object]:
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    except OSError as error:
        raise ContractDefect(f"cannot descriptor-open packet input: {path}") from error
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or stat.S_IMODE(before.st_mode) != 0o600
            or before.st_nlink != 1
            or before.st_size != expected_size
        ):
            raise ContractDefect(f"packet input metadata drifted: {path}")
        digest = hashlib.sha256()
        total = 0
        while chunk := os.read(descriptor, 1024 * 1024):
            digest.update(chunk)
            total += len(chunk)
        after = os.fstat(descriptor)
        stamp = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mode,
            before.st_nlink,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        if stamp != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mode,
            after.st_nlink,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise ContractDefect(f"packet input changed during read: {path}")
        if total != expected_size or digest.hexdigest() != expected_sha256:
            raise ContractDefect(f"packet input content drifted: {path}")
        return {
            "path": str(path),
            "bytes": total,
            "sha256": expected_sha256,
            "dev": before.st_dev,
            "ino": before.st_ino,
            "mode": oct(stat.S_IMODE(before.st_mode)),
            "nlink": before.st_nlink,
            "stamp": {
                "dev": before.st_dev,
                "ino": before.st_ino,
                "bytes": before.st_size,
                "mode": stat.S_IMODE(before.st_mode),
                "nlink": before.st_nlink,
                "mtime_ns": before.st_mtime_ns,
                "ctime_ns": before.st_ctime_ns,
            },
        }
    finally:
        os.close(descriptor)


def authenticate_packet_inputs(input_seal: dict[str, object]) -> dict[str, object]:
    rows = input_seal.get("inputs")
    if not isinstance(rows, list) or len(rows) != 3:
        raise ContractDefect("retained input seal population drifted")
    expected = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ContractDefect("retained input seal row is invalid")
        expected[str(row.get("packet_path"))] = row
    paths = (PACKET_MESSAGES, PACKET_IDENTITY, RESTORE_MESSAGES)
    if set(expected) != {str(path.relative_to(ROOT)) for path in paths}:
        raise ContractDefect("retained input seal paths drifted")
    authenticated = {}
    for path in paths:
        row = expected[str(path.relative_to(ROOT))]
        size = row.get("bytes")
        digest = row.get("sha256")
        if (
            isinstance(size, bool)
            or not isinstance(size, int)
            or not isinstance(digest, str)
        ):
            raise ContractDefect("retained input seal identity is invalid")
        authenticated[path.name] = authenticate_packet_file(path, size, digest)
    return authenticated


def mkdir_private(path: Path) -> None:
    path.mkdir(mode=0o700)
    if stat.S_IMODE(path.stat(follow_symlinks=False).st_mode) != 0o700:
        raise ContractDefect(f"directory is not mode 0700: {path}")
    fsync_directory(path.parent)


def scan_tree(root: Path) -> tuple[list[Path], list[Path]]:
    directories = [root]
    files = []
    pending = [root]
    while pending:
        current = pending.pop()
        with os.scandir(current) as entries:
            for entry in entries:
                path = Path(entry.path)
                if entry.is_symlink():
                    raise ContractDefect(f"symlink in private store: {path}")
                metadata = entry.stat(follow_symlinks=False)
                if stat.S_ISDIR(metadata.st_mode):
                    directories.append(path)
                    pending.append(path)
                elif stat.S_ISREG(metadata.st_mode):
                    files.append(path)
                else:
                    raise ContractDefect(f"nonregular store entry: {path}")
    return sorted(directories), sorted(files)


def seed_store(
    root: Path,
    seed_source: Path = PACKET_IDENTITY,
    expected_digest: str = IDENTITY_SHA256,
) -> dict[str, object]:
    if root.exists():
        raise ContractDefect(f"refusing to reuse store root: {root}")
    mkdir_private(root)
    v1 = root / "v1"
    mkdir_private(v1)
    identity_dir = v1 / "identity"
    mkdir_private(identity_dir)
    target = identity_dir / IDENTITY_SEED.name
    source_data = seed_source.read_bytes()
    descriptor = os.open(
        target,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        0o600,
    )
    try:
        if os.write(descriptor, source_data) != len(source_data):
            raise OSError("short identity seed write")
        os.fsync(descriptor)
        metadata = os.fstat(descriptor)
        if stat.S_IMODE(metadata.st_mode) != 0o600 or metadata.st_nlink != 1:
            raise ContractDefect("seed descriptor identity drifted")
    finally:
        os.close(descriptor)
    fsync_directory(identity_dir)
    verify_seeded_store(root, expected_digest, len(source_data))
    return {
        "root": str(root),
        "identity_relative": str(target.relative_to(root)),
        "identity_sha256": expected_digest,
        "blob_count": 0,
    }


def verify_seeded_store(
    root: Path, digest: str = IDENTITY_SHA256, size: int = IDENTITY_BYTES
) -> None:
    directories, files = scan_tree(root)
    identity = root / "v1/identity" / IDENTITY_SEED.name
    if files != [identity] or set(directories) != {
        root,
        root / "v1",
        root / "v1/identity",
    }:
        raise ContractDefect("fresh seeded store topology drifted")
    if any(stat.S_IMODE(path.stat().st_mode) != 0o700 for path in directories):
        raise ContractDefect("fresh store directory mode drifted")
    metadata = identity.stat(follow_symlinks=False)
    if (
        stat.S_IMODE(metadata.st_mode) != 0o600
        or metadata.st_nlink != 1
        or metadata.st_size != size
        or sha256_file(identity) != digest
    ):
        raise ContractDefect("fresh store identity seed drifted")


def child_command(store: Path, *, restore: bool = False) -> list[str]:
    return [
        str(CLI),
        "--model",
        str(MODEL),
        "--messages",
        str(RESTORE_MESSAGES if restore else PACKET_MESSAGES),
        "--messages-preserve-thinking",
        "--tokens",
        "1",
        "--temp",
        "0.7",
        "--top-k",
        "200",
        "--top-p",
        "1.0",
        "--min-p",
        "0.05",
        "--seed",
        "42",
        "--prefill-chunk",
        "1024",
        "--max-context-tokens",
        "8192" if restore else "6516",
        "--durable-prefix-cache",
        str(store),
        "--durable-prefix-cache-max-mib",
        "768",
        "--durable-prefix-cache-max-entry-mib",
        "768",
        "--durable-prefix-cache-min-tokens",
        "262144" if restore else "1024",
    ]


def rusage_record(usage: resource.struct_rusage) -> dict[str, object]:
    names = (
        "ru_utime",
        "ru_stime",
        "ru_maxrss",
        "ru_ixrss",
        "ru_idrss",
        "ru_isrss",
        "ru_minflt",
        "ru_majflt",
        "ru_nswap",
        "ru_inblock",
        "ru_oublock",
        "ru_msgsnd",
        "ru_msgrcv",
        "ru_nsignals",
        "ru_nvcsw",
        "ru_nivcsw",
    )
    return {name: getattr(usage, name) for name in names}


class Drain:
    def __init__(self, descriptor: int, expected: bytes | None):
        self.descriptor = descriptor
        self.expected = expected
        self.buffer = bytearray()
        self.overflow = False
        self.error: str | None = None
        self.eof = False
        self.complete_ns: int | None = None
        self.thread = threading.Thread(target=self.run, daemon=True)

    def run(self) -> None:
        try:
            while True:
                chunk = os.read(self.descriptor, 64 * 1024)
                if not chunk:
                    self.eof = True
                    return
                old_length = len(self.buffer)
                if old_length < MAX_STREAM_BYTES:
                    room = MAX_STREAM_BYTES - old_length
                    self.buffer.extend(chunk[:room])
                    if len(chunk) > room:
                        self.overflow = True
                else:
                    self.overflow = True
                if self.expected is not None:
                    if old_length < len(self.expected) <= old_length + len(chunk):
                        self.complete_ns = time.perf_counter_ns()
        except BaseException as error:
            self.error = f"{type(error).__name__}: {error}"
        finally:
            os.close(self.descriptor)


def wait4_exact(
    pid: int, *, nohang: bool = False
) -> tuple[int, resource.struct_rusage] | None:
    options = os.WNOHANG if nohang else 0
    while True:
        try:
            observed, status, usage = os.wait4(pid, options)
        except InterruptedError:
            continue
        if observed == 0:
            return None
        if observed != pid:
            raise Inconclusive(
                "wait4", f"wait4 reaped unexpected PID {observed}, expected {pid}"
            )
        return status, usage


def cloexec_pipe() -> tuple[int, int]:
    read_descriptor, write_descriptor = os.pipe()
    os.set_inheritable(read_descriptor, False)
    os.set_inheritable(write_descriptor, False)
    return read_descriptor, write_descriptor


def consume_pending_operator_signals() -> list[int]:
    consumed = []
    while set(signal.sigpending()) & OPERATOR_SIGNALS:
        value = signal.sigwait(OPERATOR_SIGNALS)
        consumed.append(int(value))
    return consumed


def consume_pending_wait_signals() -> list[int]:
    consumed = []
    while set(signal.sigpending()) & CHILD_WAIT_SIGNALS:
        value = signal.sigwait(CHILD_WAIT_SIGNALS)
        consumed.append(int(value))
    return consumed


def resolve_executable(argv0: str, env: dict[str, str]) -> str:
    if "/" in argv0:
        return argv0
    resolved = shutil.which(argv0, path=env.get("PATH"))
    if resolved is None:
        raise Inconclusive("spawn", f"executable is not resolvable: {argv0}")
    return resolved


def terminate_and_reap_exact(
    pid: int,
) -> tuple[tuple[int, resource.struct_rusage], int]:
    for signum, timeout in ((signal.SIGTERM, 10.0), (signal.SIGKILL, 10.0)):
        try:
            os.killpg(pid, signum)
        except ProcessLookupError:
            pass
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            result = wait4_exact(pid, nohang=True)
            if result is not None:
                return result, time.perf_counter_ns()
            time.sleep(0.05)
    raise UnreapedProcess(f"exact PID/process-group {pid} survived SIGKILL")


def create_signal_queue() -> select.kqueue:
    queue = select.kqueue()
    try:
        changes = [
            select.kevent(
                int(value),
                filter=select.KQ_FILTER_SIGNAL,
                flags=select.KQ_EV_ADD | select.KQ_EV_ENABLE | select.KQ_EV_CLEAR,
            )
            for value in CHILD_WAIT_SIGNALS
        ]
        queue.control(changes, 0, 0)
        return queue
    except BaseException:
        queue.close()
        raise


def wait4_with_signal_control(
    pid: int,
    queue: select.kqueue,
    *,
    term_grace_s: float,
    kill_grace_s: float,
    completion_watchdog_s: float | None,
) -> tuple[tuple[int, resource.struct_rusage], int, list[dict[str, object]]]:
    if (
        not math.isfinite(term_grace_s)
        or term_grace_s <= 0
        or not math.isfinite(kill_grace_s)
        or kill_grace_s <= 0
    ):
        raise ContractDefect("signal-control grace must be finite and positive")
    if completion_watchdog_s is not None and (
        not math.isfinite(completion_watchdog_s) or completion_watchdog_s <= 0
    ):
        raise ContractDefect("completion watchdog must be finite and positive")
    records: list[dict[str, object]] = []
    escalation: str | None = None
    deadline: float | None = None
    watchdog_deadline = (
        None
        if completion_watchdog_s is None
        else time.monotonic() + completion_watchdog_s
    )
    result = wait4_exact(pid, nohang=True)
    if result is not None:
        return result, time.perf_counter_ns(), records
    while True:
        deadlines = [
            value for value in (deadline, watchdog_deadline) if value is not None
        ]
        timeout = None if not deadlines else max(min(deadlines) - time.monotonic(), 0.0)
        events = queue.control(None, 1, timeout)
        if not events:
            result = wait4_exact(pid, nohang=True)
            if result is not None:
                return result, time.perf_counter_ns(), records
            if (
                watchdog_deadline is not None
                and time.monotonic() >= watchdog_deadline
                and escalation is None
            ):
                records.append(
                    {
                        "action": "watchdog",
                        "signal": None,
                        "outcome": "completion-signal-timeout",
                        "monotonic_ns": time.perf_counter_ns(),
                    }
                )
                os.killpg(pid, signal.SIGTERM)
                records.append(
                    {
                        "action": "terminate",
                        "signal": int(signal.SIGTERM),
                        "outcome": "sent",
                        "monotonic_ns": time.perf_counter_ns(),
                    }
                )
                escalation = "term"
                deadline = time.monotonic() + term_grace_s
                watchdog_deadline = None
                continue
            if escalation == "term":
                os.killpg(pid, signal.SIGKILL)
                records.append(
                    {
                        "action": "kill",
                        "signal": int(signal.SIGKILL),
                        "outcome": "sent",
                        "monotonic_ns": time.perf_counter_ns(),
                    }
                )
                escalation = "kill"
                deadline = time.monotonic() + kill_grace_s
                continue
            if escalation == "kill":
                raise UnreapedProcess(f"exact PID {pid} was not reaped after SIGKILL")
            raise ContractDefect("signal-control wait timed out without a deadline")
        received = int(events[0].ident)
        records.append(
            {
                "action": "received",
                "signal": received,
                "monotonic_ns": time.perf_counter_ns(),
            }
        )
        result = wait4_exact(pid, nohang=True)
        if result is not None:
            return result, time.perf_counter_ns(), records
        if received == int(signal.SIGALRM):
            records.append(
                {
                    "action": "contamination",
                    "signal": received,
                    "outcome": "unexpected-sigalrm",
                    "monotonic_ns": time.perf_counter_ns(),
                }
            )
        if (
            received
            in {int(value) for value in OPERATOR_SIGNALS} | {int(signal.SIGALRM)}
            and escalation is None
        ):
            os.killpg(pid, signal.SIGTERM)
            records.append(
                {
                    "action": "terminate",
                    "signal": int(signal.SIGTERM),
                    "outcome": "sent",
                    "monotonic_ns": time.perf_counter_ns(),
                }
            )
            escalation = "term"
            deadline = time.monotonic() + term_grace_s


def direct_process(
    argv: list[str],
    env: dict[str, str],
    *,
    expected_stdout: bytes | None,
    cwd: Path = ROOT,
    term_grace_s: float = 10.0,
    kill_grace_s: float = 10.0,
    completion_watchdog_s: float | None = None,
) -> dict[str, object]:
    if cwd.resolve() != ROOT or Path.cwd().resolve() != ROOT:
        raise ContractDefect("direct posix_spawn requires runner cwd ROOT")
    old_mask = signal.pthread_sigmask(signal.SIG_BLOCK, CHILD_WAIT_SIGNALS)
    out_r = out_w = err_r = err_w = -1
    stdout: Drain | None = None
    stderr: Drain | None = None
    pid: int | None = None
    status_usage: tuple[int, resource.struct_rusage] | None = None
    started_ns = time.perf_counter_ns()
    reaped_ns: int | None = None
    interruption: str | None = None
    pending_signals: list[int] = []
    signal_records: list[dict[str, object]] = []
    control_error: str | None = None
    preserve_alarm_block = False
    signal_queue: select.kqueue | None = None
    try:
        try:
            if signal.getitimer(signal.ITIMER_REAL) != (0.0, 0.0):
                preserve_alarm_block = True
                raise Inconclusive(
                    "signal-control", "ITIMER_REAL is active before process control"
                )
            pending_before_spawn = consume_pending_wait_signals()
            if int(signal.SIGALRM) in pending_before_spawn:
                preserve_alarm_block = True
                raise Inconclusive("signal-control", "pending SIGALRM before spawn")
            pending_signals = [
                value for value in pending_before_spawn if value in OPERATOR_SIGNALS
            ]
            if pending_signals:
                raise Inconclusive(
                    "operator-signal",
                    "pending operator signal before spawn: "
                    + ",".join(str(value) for value in pending_signals),
                )
            out_r, out_w = cloexec_pipe()
            err_r, err_w = cloexec_pipe()
            stdout = Drain(out_r, expected_stdout)
            stderr = Drain(err_r, None)
            stdout.thread.start()
            out_r = -1
            stderr.thread.start()
            err_r = -1
            signal_queue = create_signal_queue()
            pending_after_registration = consume_pending_wait_signals()
            if int(signal.SIGALRM) in pending_after_registration:
                preserve_alarm_block = True
                raise Inconclusive(
                    "signal-control", "pending SIGALRM at pre-spawn registration"
                )
            registration_operator = [
                value
                for value in pending_after_registration
                if value in OPERATOR_SIGNALS
            ]
            if registration_operator:
                raise Inconclusive(
                    "operator-signal",
                    "pending operator signal at pre-spawn registration: "
                    + ",".join(str(value) for value in registration_operator),
                )
            executable = resolve_executable(argv[0], env)
            file_actions = (
                (os.POSIX_SPAWN_DUP2, out_w, 1),
                (os.POSIX_SPAWN_DUP2, err_w, 2),
                (os.POSIX_SPAWN_CLOSE, stdout.descriptor),
                (os.POSIX_SPAWN_CLOSE, stderr.descriptor),
                (os.POSIX_SPAWN_CLOSE, out_w),
                (os.POSIX_SPAWN_CLOSE, err_w),
            )
            pid = os.posix_spawn(
                executable,
                argv,
                env,
                file_actions=file_actions,
                setsid=True,
                setsigmask=(),
                setsigdef=tuple(CHILD_WAIT_SIGNALS),
            )
            try:
                group = os.getpgid(pid)
            except ProcessLookupError:
                group = pid
            if group != pid:
                raise Inconclusive("spawn", f"unexpected child process group {group}")
            os.close(out_w)
            os.close(err_w)
            out_w = err_w = -1
            try:
                status_usage, reaped_ns, signal_records = wait4_with_signal_control(
                    pid,
                    signal_queue,
                    term_grace_s=term_grace_s,
                    kill_grace_s=kill_grace_s,
                    completion_watchdog_s=completion_watchdog_s,
                )
            except BaseException as error:
                interruption = f"wait4={type(error).__name__}: {error}"
                status_usage, reaped_ns = terminate_and_reap_exact(pid)
        except BaseException as error:
            if pid is None:
                if isinstance(error, Inconclusive):
                    raise
                raise Inconclusive(
                    "spawn",
                    f"direct setup/spawn failed: {type(error).__name__}: {error}",
                ) from error
            if status_usage is None:
                interruption = f"{type(error).__name__}: {error}"
                status_usage, reaped_ns = terminate_and_reap_exact(pid)
            else:
                interruption = (
                    interruption or f"post-reap-control={type(error).__name__}: {error}"
                )
    finally:
        for descriptor in (out_w, err_w, out_r, err_r):
            if descriptor >= 0:
                try:
                    os.close(descriptor)
                except OSError:
                    pass
        if signal_queue is not None:
            signal_queue.close()
        for drain in (stdout, stderr):
            if drain is not None and drain.thread.ident is not None:
                drain.thread.join(timeout=30.0)
        if pid is None:
            restored = set(old_mask)
            if preserve_alarm_block:
                restored.add(signal.SIGALRM)
            signal.pthread_sigmask(signal.SIG_SETMASK, restored)
    if pid is None:
        raise Inconclusive("spawn", "direct process has no acquired PID")
    if status_usage is None or reaped_ns is None:
        raise UnreapedProcess(f"exact PID {pid} was not reaped")
    if stdout is None or stderr is None:
        raise RuntimeError("reaped process has incomplete drain ownership")
    if any(
        record.get("signal") in {int(value) for value in OPERATOR_SIGNALS}
        or record.get("action") in {"terminate", "kill"}
        for record in signal_records
    ):
        interruption = interruption or "operator signal received by sole-main control"
    pending_after_reap = consume_pending_wait_signals()
    if int(signal.SIGALRM) in pending_after_reap or any(
        record.get("action") == "contamination" for record in signal_records
    ):
        control_error = "unexpected SIGALRM contamination"
        interruption = interruption or control_error
    pending_signals.extend(
        value for value in pending_after_reap if value in OPERATOR_SIGNALS
    )
    returncode = os.waitstatus_to_exitcode(status_usage[0])
    return {
        "pid": pid,
        "wait_status": status_usage[0],
        "returncode": returncode,
        "started_ns": started_ns,
        "t_reaped_ns": reaped_ns,
        "t_stdout_complete_ns": stdout.complete_ns,
        "stdout": bytes(stdout.buffer),
        "stderr": bytes(stderr.buffer),
        "rusage": rusage_record(status_usage[1]),
        "interruption": interruption,
        "pending_signals": pending_signals,
        "signal_control": {
            "strategy": "sole-main-kqueue-wait4",
            "records": signal_records,
            "error": control_error,
            "thread_alive": False,
            "shutdown_clean": True,
        },
        "previous_signal_mask": [int(value) for value in sorted(old_mask)],
        "operator_signals_blocked": True,
        "drains": {
            "stdout": {
                "eof": stdout.eof,
                "overflow": stdout.overflow,
                "error": stdout.error,
                "thread_alive": stdout.thread.is_alive(),
            },
            "stderr": {
                "eof": stderr.eof,
                "overflow": stderr.overflow,
                "error": stderr.error,
                "thread_alive": stderr.thread.is_alive(),
            },
        },
    }


def release_process_signal_mask(result: dict[str, object]) -> dict[str, object]:
    values = result.get("previous_signal_mask")
    if not isinstance(values, list) or any(
        isinstance(value, bool) or not isinstance(value, int) for value in values
    ):
        raise ContractDefect("process result has invalid previous signal mask")
    snapshot = sorted(
        int(value) for value in (set(signal.sigpending()) & CHILD_WAIT_SIGNALS)
    )
    for value in snapshot:
        signal.sigwait({signal.Signals(value)})
    operator = [value for value in snapshot if value in OPERATOR_SIGNALS]
    contaminated = int(signal.SIGALRM) in snapshot
    signal.pthread_sigmask(
        signal.SIG_SETMASK, {signal.Signals(value) for value in values}
    )
    controlled_after = sorted(
        value for value in values if signal.Signals(value) in CHILD_WAIT_SIGNALS
    )
    return {
        "schema": 1,
        "previous_signal_mask": values,
        "process_operator_signals_blocked_before_release": True,
        "pending_snapshot_before_mask_restore": snapshot,
        "operator_signals_before_mask_restore": operator,
        "sigalrm_contamination_before_mask_restore": contaminated,
        "previous_mask_reapplied": True,
        "controlled_signals_blocked_after_release": controlled_after,
        "sigalrm_reserved_after_release": int(signal.SIGALRM) in values,
        "snapshot_and_mask_restore_are_not_atomic": True,
        "captured_monotonic_ns": time.perf_counter_ns(),
    }


def parse_unsigned(text: str, label: str) -> int:
    if not text or not text.isascii() or not text.isdecimal() or str(int(text)) != text:
        raise ContractDefect(f"{label} is not canonical unsigned decimal")
    return int(text)


def parse_finite_decimal(text: str, label: str) -> float:
    try:
        value = float(text)
    except ValueError as error:
        raise ContractDefect(f"{label} is not a decimal number") from error
    if not math.isfinite(value) or value < 0:
        raise ContractDefect(f"{label} is not finite nonnegative")
    return value


PUBLICATION = re.compile(
    r"durable_prefix_cache: publish=(\S+) capture=(\S+) matched_tokens=(\d+) "
    r"restored_tokens=(\d+) pending=(\S+) stop_reason=(\S+) blob_bytes=(\d+) "
    r"evicted=(\d+) identity=(\S+) staged_integrity=(\S+) "
    r"staged_integrity_us=(\d+) capture_ms=([0-9.]+) publish_us=(\d+) "
    r"post_response_us=(\d+)"
)
STATS = re.compile(
    r"stats: prompt_tokens=(\d+) generated_tokens=(\d+) transitions=(\d+) "
    r"stop_reason=(\S+) load_ms=([0-9.]+) prefill_ms=([0-9.]+) "
    r"ttft_ms=([0-9.]+) decode_tps=([0-9.]+) transition_tps=([0-9.]+) "
    r"cache_entries=(\d+) cache_mib=([0-9.]+)/([0-9.]+)"
)
RESTORE = re.compile(
    r"durable_prefix_cache: identity_cache=hit hashed_bytes=0 "
    r"checkpoint_hit=true matched=6500 restored=6499 exact=false candidates=1 "
    r"corrupt_removed=0 restore_total_ms=([0-9.]+)"
)


def strict_stderr(data: bytes) -> str:
    try:
        return data.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise ContractDefect("child stderr is not strict UTF-8") from error


def parse_scored(stderr_bytes: bytes, arm: str, external_us: int) -> dict[str, object]:
    stderr = strict_stderr(stderr_bytes)
    lines = stderr.splitlines()
    publication_lines = [
        line for line in lines if line.startswith("durable_prefix_cache: publish=")
    ]
    empty_lines = [
        line for line in lines if line.startswith("durable_prefix_cache: store_empty=")
    ]
    stats_lines = [line for line in lines if line.startswith("stats:")]
    if len(publication_lines) != 1 or len(empty_lines) != 1 or len(stats_lines) != 1:
        raise ContractDefect("scored telemetry line count drifted")
    match = PUBLICATION.fullmatch(publication_lines[0])
    if match is None:
        raise ContractDefect("explicit publication telemetry grammar drifted")
    fields = match.groups()
    integers = [
        parse_unsigned(fields[index], f"publication field {index}")
        for index in (2, 3, 6, 7, 10, 12, 13)
    ]
    capture_ms = parse_finite_decimal(fields[11], "capture_ms")
    (
        matched,
        restored,
        blob_bytes,
        evicted,
        integrity_us,
        publish_us,
        post_response_us,
    ) = integers
    if (
        fields[0] != "published"
        or fields[1] != "completed"
        or matched != 6500
        or restored != 6499
        or fields[4] != "true"
        or fields[5] != "token_limit"
        or blob_bytes != BLOB_BYTES
        or evicted != 0
        or fields[8] != "hit"
        or fields[9] != arm_mode(arm)
    ):
        raise ContractDefect("explicit publication contract drifted")
    empty = re.fullmatch(
        r"durable_prefix_cache: store_empty=true restore_total_ms=([0-9.]+)",
        empty_lines[0],
    )
    if empty is None:
        raise ContractDefect("store-empty telemetry drifted")
    store_empty_ms = parse_finite_decimal(empty.group(1), "store_empty_ms")
    stats = STATS.fullmatch(stats_lines[0])
    if stats is None or stats.groups()[:4] != ("6499", "1", "0", "token_limit"):
        raise ContractDefect("current stats telemetry drifted")
    for index, value in enumerate(stats.groups()[4:9] + stats.groups()[10:12]):
        parse_finite_decimal(value, f"stats numeric field {index}")
    forbidden = (
        "warning: durable prefix",
        "checkpoint_hit=",
        "corrupt_removed=",
        "publish=failed",
        "corruption",
        "repaired",
        "repair=",
    )
    if any(token in stderr for token in forbidden):
        raise ContractDefect("forbidden scored checkpoint telemetry appeared")
    return {
        "publish": "published",
        "capture": "completed",
        "matched_tokens": matched,
        "restored_tokens": restored,
        "pending": True,
        "stop_reason": "token_limit",
        "blob_bytes": blob_bytes,
        "evicted": evicted,
        "identity": "hit",
        "staged_integrity": fields[9],
        "staged_integrity_us": integrity_us,
        "capture_ms": capture_ms,
        "publish_us": publish_us,
        "post_response_us": post_response_us,
        "external_us": external_us,
        "store_empty_ms": store_empty_ms,
    }


def require_observation_quality(external_us: int, post_response_us: int) -> None:
    if external_us < post_response_us:
        raise Inconclusive(
            "observation",
            "external_us does not contain the child's post_response_us interval",
        )


def parse_restore(stderr_bytes: bytes) -> dict[str, object]:
    stderr = strict_stderr(stderr_bytes)
    lines = stderr.splitlines()
    restore_lines = [
        line
        for line in lines
        if line.startswith("durable_prefix_cache: identity_cache=")
    ]
    match = RESTORE.fullmatch(restore_lines[0]) if len(restore_lines) == 1 else None
    if match is None:
        raise ContractDefect("restore hit telemetry drifted")
    elapsed = parse_finite_decimal(match.group(1), "restore_total_ms")
    forbidden = (
        "durable_prefix_cache: publish=",
        "warning: durable prefix",
        "corruption",
        "repair=",
    )
    if any(token in stderr for token in forbidden):
        raise ContractDefect(
            "restore emitted publication, warning, corruption, or repair"
        )
    return {
        "identity_cache": "hit",
        "hashed_bytes": 0,
        "checkpoint_hit": True,
        "matched": 6500,
        "restored": 6499,
        "exact": False,
        "candidates": 1,
        "corrupt_removed": 0,
        "restore_total_ms": elapsed,
    }


def inspect_store(root: Path) -> dict[str, object]:
    directories, files = scan_tree(root)
    identity = root / "v1/identity" / IDENTITY_SEED.name
    lock = root / "v1/store.lock"
    blob = root / BLOB_RELATIVE
    if set(files) != {identity, lock, blob}:
        raise ContractDefect(f"published store file topology drifted: {files}")
    expected_dirs = {
        root,
        root / "v1",
        root / "v1/identity",
        root / "v1/blobs",
        root / "v1/blobs" / COMPATIBILITY,
    }
    if set(directories) != expected_dirs:
        raise ContractDefect("published store directory topology drifted")
    if any(stat.S_IMODE(path.stat().st_mode) != 0o700 for path in directories):
        raise ContractDefect("published store directory mode drifted")
    if any(
        stat.S_IMODE(path.stat().st_mode) != 0o600 or path.stat().st_nlink != 1
        for path in files
    ):
        raise ContractDefect("published store file mode/link drifted")
    if (
        identity.stat().st_size != IDENTITY_BYTES
        or sha256_file(identity) != IDENTITY_SHA256
    ):
        raise ContractDefect("published store identity drifted")
    metadata = blob.stat(follow_symlinks=False)
    digest = sha256_file(blob)
    with blob.open("rb", buffering=0) as source:
        source.seek(-32, os.SEEK_END)
        trailer = source.read(32).hex()
    if (
        metadata.st_size != BLOB_BYTES
        or digest != BLOB_SHA256
        or trailer != ENCODER_TRAILER
    ):
        raise ContractDefect("published blob bytes/digest/trailer drifted")
    return {
        "blob_path": str(blob),
        "blob_relative": str(BLOB_RELATIVE),
        "blob_bytes": metadata.st_size,
        "blob_sha256": digest,
        "encoder_trailer": trailer,
        "blob_dev": metadata.st_dev,
        "blob_ino": metadata.st_ino,
        "blob_mode": oct(stat.S_IMODE(metadata.st_mode)),
        "blob_nlink": metadata.st_nlink,
        "files": len(files),
        "directories": len(directories),
    }


def stream_equal(left: Path, right: Path) -> bool:
    if left.stat().st_size != right.stat().st_size:
        return False
    with left.open("rb", buffering=0) as a, right.open("rb", buffering=0) as b:
        while True:
            x = a.read(8 * 1024 * 1024)
            y = b.read(8 * 1024 * 1024)
            if x != y:
                return False
            if not x:
                return True


def model_stamp(metadata: os.stat_result) -> dict[str, int]:
    return {
        "dev": metadata.st_dev,
        "ino": metadata.st_ino,
        "bytes": metadata.st_size,
        "mode": stat.S_IMODE(metadata.st_mode),
        "nlink": metadata.st_nlink,
        "mtime_ns": metadata.st_mtime_ns,
        "ctime_ns": metadata.st_ctime_ns,
    }


def warm_model() -> dict[str, object]:
    started = time.perf_counter_ns()
    digest = hashlib.sha256()
    total = 0
    try:
        descriptor = os.open(MODEL, os.O_RDONLY | os.O_NOFOLLOW)
    except OSError as error:
        raise Inconclusive(
            "conditioning",
            f"model descriptor open failed: {type(error).__name__}: {error}",
        ) from error
    try:
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode):
                raise ContractDefect("frozen model is not a regular file")
            while chunk := os.read(descriptor, 8 * 1024 * 1024):
                total += len(chunk)
                digest.update(chunk)
            after = os.fstat(descriptor)
        except OSError as error:
            raise Inconclusive(
                "conditioning",
                f"model descriptor read/stat failed: {type(error).__name__}: {error}",
            ) from error
    finally:
        os.close(descriptor)
    before_stamp = model_stamp(before)
    after_stamp = model_stamp(after)
    if before_stamp != after_stamp:
        raise ContractDefect("frozen model metadata changed during hash-read")
    result = {
        "bytes": total,
        "sha256": digest.hexdigest(),
        "stamp": before_stamp,
        "wall_ms": (time.perf_counter_ns() - started) / 1e6,
    }
    if total != MODEL_BYTES or result["sha256"] != MODEL_SHA256:
        raise ContractDefect("complete prelaunch model hash-read drifted")
    return result


def reverify_model_stamp(expected: dict[str, object]) -> dict[str, int]:
    try:
        observed = os.stat(MODEL, follow_symlinks=False)
    except OSError as error:
        raise Inconclusive(
            "conditioning",
            f"model nofollow restat failed: {type(error).__name__}: {error}",
        ) from error
    if not stat.S_ISREG(observed.st_mode):
        raise ContractDefect("model pathname became nonregular")
    stamp = model_stamp(observed)
    if stamp != expected:
        raise ContractDefect("model immutable stamp changed after cooldown")
    return stamp


def persist_json_evidence(path: Path, value: dict[str, object]) -> str:
    write_json(path, value)
    return sha256_file(path)


def condition(stem: str, source: dict[str, object]) -> dict[str, object]:
    evidence: dict[str, object] = {
        "schema": 1,
        "stem": stem,
        "host_before_conditioning": host_protocol.capture_host_state(),
        "vm_before_conditioning": host_protocol.capture_vm_state(),
        "host_samples": [],
    }
    try:
        model_read = warm_model()
        evidence["model_hash_read"] = model_read
        time.sleep(30.0)
        samples = []
        for index in range(host_protocol.HOST_SAMPLE_LIMIT):
            if verify_source(source["v0637_packet"]) != source:
                raise ContractDefect("source drifted during host conditioning")
            sample = host_protocol.capture_host_state()
            samples.append(sample)
            evidence["host_samples"] = samples
            if sample.get("valid") is True:
                break
            if index + 1 < host_protocol.HOST_SAMPLE_LIMIT:
                time.sleep(host_protocol.HOST_SAMPLE_INTERVAL_S)
        if not samples or samples[-1].get("valid") is not True:
            raise Inconclusive("host", f"host sampler exhausted before {stem}")
        vm_spawn = host_protocol.capture_vm_state()
        cache_interval = host_protocol.vm_interval(
            "conditioning", evidence["vm_before_conditioning"], vm_spawn
        )
        evidence["host_before_spawn"] = samples[-1]
        evidence["vm_before_spawn"] = vm_spawn
        evidence["conditioning_interval"] = cache_interval
        if cache_interval["failure_reasons"]:
            raise Inconclusive("pressure", ";".join(cache_interval["failure_reasons"]))
        evidence["model_stamp_before_spawn"] = reverify_model_stamp(model_read["stamp"])
        evidence["valid"] = True
        return evidence
    except BaseException as failure:
        evidence["valid"] = False
        evidence["failure"] = f"{type(failure).__name__}: {failure}"
        persist_json_evidence(ARTIFACT / f"{stem}.conditioning.json", evidence)
        raise


def post_exit_evidence(
    conditioning: dict[str, object], rusage: dict[str, object]
) -> dict[str, object]:
    host_after = host_protocol.capture_host_state()
    vm_after = host_protocol.capture_vm_state()
    interval = host_protocol.vm_interval(
        "child", conditioning["vm_before_spawn"], vm_after
    )
    reasons = list(interval["failure_reasons"])
    if host_after.get("valid") is not True:
        reasons.append("post_exit_host_invalid")
    if rusage["ru_nswap"] != 0:
        reasons.append("ru_nswap_nonzero")
    if rusage["ru_inblock"] != 0:
        reasons.append("ru_inblock_nonzero")
    return {
        "schema": 1,
        "host_after_exit": host_after,
        "vm_after_exit": vm_after,
        "child_interval": interval,
        "major_faults_advisory": rusage["ru_majflt"],
        "valid": not reasons,
        "validity_reasons": reasons,
    }


def seal_launch(event: str, stage: str, stem: str, **fields: object) -> None:
    append_jsonl(
        ARTIFACT / "launch-seal.jsonl",
        {
            "schema": 1,
            "event": event,
            "stage": stage,
            "stem": stem,
            "unix_ms": time.time_ns() // 1_000_000,
            **fields,
        },
    )


def persist_process(stem: str, result: dict[str, object]) -> tuple[str, str]:
    stdout = result["stdout"]
    stderr = result["stderr"]
    if not isinstance(stdout, bytes) or not isinstance(stderr, bytes):
        raise ContractDefect("process buffers have invalid types")
    write_new(ARTIFACT / f"{stem}.out", stdout)
    write_new(ARTIFACT / f"{stem}.err", stderr)
    stdout_sha = sha256_bytes(stdout)
    stderr_sha = sha256_bytes(stderr)
    write_json(
        ARTIFACT / f"{stem}.process.json",
        {key: value for key, value in result.items() if key not in {"stdout", "stderr"}}
        | {
            "stdout_bytes": len(stdout),
            "stderr_bytes": len(stderr),
            "stdout_sha256": stdout_sha,
            "stderr_sha256": stderr_sha,
        },
    )
    return stdout_sha, stderr_sha


def process_execution_reasons(result: dict[str, object]) -> list[str]:
    reasons = []
    control = result.get("signal_control")
    if not isinstance(control, dict):
        reasons.append("missing_signal_control_evidence")
    else:
        if control.get("error") is not None:
            reasons.append(f"signal_control_error={control.get('error')}")
        if control.get("thread_alive") is not False:
            reasons.append("signal_control_thread_alive")
        if control.get("shutdown_clean") is not True:
            reasons.append("signal_control_shutdown_invalid")
    drains = result.get("drains")
    if not isinstance(drains, dict):
        return ["missing_drain_evidence"]
    for name in ("stdout", "stderr"):
        row = drains.get(name)
        if not isinstance(row, dict):
            reasons.append(f"{name}_drain_missing")
            continue
        if row.get("eof") is not True:
            reasons.append(f"{name}_drain_no_eof")
        if row.get("thread_alive") is not False:
            reasons.append(f"{name}_drain_thread_alive")
        if row.get("overflow") is not False:
            reasons.append(f"{name}_drain_overflow")
        if row.get("error") is not None:
            reasons.append(f"{name}_drain_error={row.get('error')}")
    return reasons


def operator_control_records(result: dict[str, object]) -> list[dict[str, object]]:
    control = result.get("signal_control")
    records = control.get("records") if isinstance(control, dict) else None
    if not isinstance(records, list):
        return []
    return [
        record
        for record in records
        if record.get("signal") in {int(value) for value in OPERATOR_SIGNALS}
        or record.get("action") in {"terminate", "kill"}
    ]


def persist_post_exit(
    stem: str, conditioning: dict[str, object], result: dict[str, object]
) -> dict[str, object]:
    post = post_exit_evidence(conditioning, result["rusage"])
    persist_json_evidence(ARTIFACT / f"{stem}.post-exit.json", post)
    return post


def launch_scored(
    arm: str,
    pair_index: int,
    order: str,
    position: int,
    base_env: dict[str, str],
    source: dict[str, object],
    input_seal: dict[str, object],
    build_identity: dict[str, object],
) -> dict[str, object]:
    stem = f"product-p{pair_index:02d}-{order.lower()}-r{position}-{arm.lower()}"
    store = WORK_ROOT / stem
    before = seed_store(store)
    conditioning = condition(stem, source)
    conditioning_sha256 = persist_json_evidence(
        ARTIFACT / f"{stem}.conditioning.json", conditioning
    )
    verify_seeded_store(store)
    if verify_source(source["v0637_packet"]) != source:
        raise ContractDefect("source drifted immediately before scored spawn")
    argv = child_command(store)
    env = arm_environment(base_env, arm)
    packet_inputs = authenticate_packet_inputs(input_seal)
    model_stamp_immediate = reverify_model_stamp(
        conditioning["model_hash_read"]["stamp"]
    )
    qwen_identity = authenticate_qwen_binary(build_identity)
    seal_launch(
        "launch",
        "scored",
        stem,
        command=argv,
        arm=arm,
        mode=arm_mode(arm),
        pair_index=pair_index,
        pair_order=order,
        position=position,
        conditioning_sha256=conditioning_sha256,
        packet_inputs=packet_inputs,
        model_stamp_immediate=model_stamp_immediate,
        qwen_identity=qwen_identity,
    )
    result: dict[str, object] | None = None
    error: BaseException | None = None
    try:
        result = direct_process(argv, env, expected_stdout=STDOUT_BYTES)
    except BaseException as failure:
        error = failure
    finally:
        seal_launch(
            "completion",
            "scored",
            stem,
            command=argv,
            returncode=result.get("returncode") if result else None,
            error=f"{type(error).__name__}: {error}" if error else None,
        )
    if error is not None:
        raise error
    if result is None:
        raise Inconclusive("population", f"missing process result: {stem}")
    stdout_sha, stderr_sha = persist_process(stem, result)
    post = persist_post_exit(stem, conditioning, result)
    release = release_process_signal_mask(result)
    write_json(ARTIFACT / f"{stem}.signal-release.json", release)
    if post["valid"] is not True:
        raise Inconclusive("resources", ";".join(post["validity_reasons"]))
    if result["signal_control"]["error"] is not None:
        raise Inconclusive("signal-control", str(result["signal_control"]["error"]))
    if result["interruption"] is not None or result["pending_signals"]:
        raise Inconclusive(
            "operator-signal",
            str(result["interruption"] or result["pending_signals"]),
        )
    if (
        release["operator_signals_before_mask_restore"]
        or release["sigalrm_contamination_before_mask_restore"]
    ):
        raise Inconclusive("signal-control", f"mask-release contamination: {release}")
    if result["returncode"] < 0:
        raise Inconclusive(
            "signal", f"child terminated by signal {-result['returncode']}"
        )
    execution_reasons = process_execution_reasons(result)
    if execution_reasons:
        raise Inconclusive("pipe", ";".join(execution_reasons))
    if result["returncode"] != 0:
        raise ContractDefect(
            f"scored child positive nonzero exit: {stem}: {result['returncode']}"
        )
    if result["stdout"] != STDOUT_BYTES or stdout_sha != STDOUT_SHA256:
        raise ContractDefect("scored response identity drifted")
    complete_ns = result["t_stdout_complete_ns"]
    if not isinstance(complete_ns, int):
        raise ContractDefect("scored stdout completion timestamp missing")
    external_us = (int(result["t_reaped_ns"]) - complete_ns) // 1000
    if external_us < 0:
        raise ContractDefect("external response-to-reap interval is negative")
    telemetry = parse_scored(result["stderr"], arm, external_us)
    require_observation_quality(external_us, telemetry["post_response_us"])
    topology = inspect_store(store)
    return {
        "schema": 1,
        "stage": "scored",
        "stem": stem,
        "arm": arm,
        "mode": arm_mode(arm),
        "pair_index": pair_index,
        "pair_order": order,
        "position": position,
        "command": argv,
        "arm_environment": {INTEGRITY_ENV: arm_mode(arm)},
        "store_before": before,
        "store_after": topology,
        **conditioning,
        **post,
        "returncode": result["returncode"],
        "t_stdout_complete_ns": complete_ns,
        "t_reaped_ns": result["t_reaped_ns"],
        "external_us": external_us,
        "rusage": result["rusage"],
        "stdout_sha256": stdout_sha,
        "stderr_sha256": stderr_sha,
        "telemetry": telemetry,
        "valid": True,
    }


def analyze(rows: list[dict[str, object]]) -> dict[str, object]:
    if len(rows) != 12:
        raise Inconclusive("population", f"expected 12 scored rows, found {len(rows)}")
    pairs = []
    for index, order in enumerate(PAIR_ORDERS, 1):
        pair = [row for row in rows if row["pair_index"] == index]
        if len(pair) != 2 or "".join(str(row["arm"]) for row in pair) != order:
            raise ContractDefect("scored pair order/membership drifted")
        by_arm = {str(row["arm"]): row for row in pair}
        a = by_arm["A"]
        b = by_arm["B"]
        publish = int(a["telemetry"]["publish_us"]) - int(b["telemetry"]["publish_us"])
        external = int(a["external_us"]) - int(b["external_us"])
        pairs.append(
            {
                "pair_index": index,
                "pair_order": order,
                "publish_saving_us": publish,
                "external_saving_us": external,
                "publish_gate": publish >= GATE_US,
                "external_gate": external >= GATE_US,
                "b_wins_publish": publish > 0,
                "b_wins_external": external > 0,
            }
        )
    passes = all(
        pair["publish_gate"]
        and pair["external_gate"]
        and pair["b_wins_publish"]
        and pair["b_wins_external"]
        for pair in pairs
    )
    summaries = {}
    for order in ("AB", "BA"):
        selected = [pair for pair in pairs if pair["pair_order"] == order]
        summaries[order] = {
            "publish_saving_median_us": statistics.median(
                pair["publish_saving_us"] for pair in selected
            ),
            "external_saving_median_us": statistics.median(
                pair["external_saving_us"] for pair in selected
            ),
        }
    return {
        "schema": 1,
        "pairs": pairs,
        "passes": passes,
        "publish_saving_median_us": statistics.median(
            pair["publish_saving_us"] for pair in pairs
        ),
        "external_saving_median_us": statistics.median(
            pair["external_saving_us"] for pair in pairs
        ),
        "order_summaries_descriptive": summaries,
    }


def blob_stamp(path: Path) -> dict[str, object]:
    metadata = path.stat(follow_symlinks=False)
    return {
        "path": str(path),
        "dev": metadata.st_dev,
        "ino": metadata.st_ino,
        "bytes": metadata.st_size,
        "mode": stat.S_IMODE(metadata.st_mode),
        "nlink": metadata.st_nlink,
        "sha256": sha256_file(path),
    }


def run_restore(
    candidate: dict[str, object],
    base_env: dict[str, str],
    source: dict[str, object],
    input_seal: dict[str, object],
    build_identity: dict[str, object],
) -> dict[str, object]:
    stem = "restore-p01-position2-b"
    store = Path(str(candidate["store_before"]["root"]))
    inspect_store(store)
    blob = store / BLOB_RELATIVE
    before = blob_stamp(blob)
    conditioning = condition(stem, source)
    conditioning_sha256 = persist_json_evidence(
        ARTIFACT / f"{stem}.conditioning.json", conditioning
    )
    inspect_store(store)
    if blob_stamp(blob) != before:
        raise ContractDefect("candidate blob changed before restore spawn")
    argv = child_command(store, restore=True)
    env = arm_environment(base_env, "B")
    packet_inputs = authenticate_packet_inputs(input_seal)
    model_stamp_immediate = reverify_model_stamp(
        conditioning["model_hash_read"]["stamp"]
    )
    qwen_identity = authenticate_qwen_binary(build_identity)
    seal_launch(
        "launch",
        "restore",
        stem,
        command=argv,
        arm="B",
        mode=arm_mode("B"),
        conditioning_sha256=conditioning_sha256,
        packet_inputs=packet_inputs,
        model_stamp_immediate=model_stamp_immediate,
        qwen_identity=qwen_identity,
    )
    result: dict[str, object] | None = None
    error: BaseException | None = None
    try:
        result = direct_process(argv, env, expected_stdout=None)
    except BaseException as failure:
        error = failure
    finally:
        seal_launch(
            "completion",
            "restore",
            stem,
            command=argv,
            returncode=result.get("returncode") if result else None,
            error=f"{type(error).__name__}: {error}" if error else None,
        )
    if error is not None:
        raise error
    if result is None:
        raise Inconclusive("population", "restore process result missing")
    stdout_sha, stderr_sha = persist_process(stem, result)
    post = persist_post_exit(stem, conditioning, result)
    release = release_process_signal_mask(result)
    write_json(ARTIFACT / f"{stem}.signal-release.json", release)
    if post["valid"] is not True:
        raise Inconclusive("resources", ";".join(post["validity_reasons"]))
    if result["signal_control"]["error"] is not None:
        raise Inconclusive("signal-control", str(result["signal_control"]["error"]))
    if result["interruption"] is not None or result["pending_signals"]:
        raise Inconclusive(
            "operator-signal",
            str(result["interruption"] or result["pending_signals"]),
        )
    if (
        release["operator_signals_before_mask_restore"]
        or release["sigalrm_contamination_before_mask_restore"]
    ):
        raise Inconclusive("signal-control", f"mask-release contamination: {release}")
    if result["returncode"] < 0:
        raise Inconclusive(
            "signal", f"restore terminated by signal {-result['returncode']}"
        )
    execution_reasons = process_execution_reasons(result)
    if execution_reasons:
        raise Inconclusive("pipe", ";".join(execution_reasons))
    if result["returncode"] != 0:
        raise ContractDefect(f"restore positive nonzero exit: {result['returncode']}")
    if not result["stdout"]:
        raise ContractDefect("restore stdout is empty")
    telemetry = parse_restore(result["stderr"])
    topology = inspect_store(store)
    after = blob_stamp(blob)
    if before != after:
        raise ContractDefect(
            "restore changed candidate path/inode/bytes/mode/link/digest"
        )
    return {
        "schema": 1,
        "stage": "restore",
        "stem": stem,
        "valid": True,
        "command": argv,
        "arm_environment": {INTEGRITY_ENV: arm_mode("B")},
        **conditioning,
        **post,
        "returncode": result["returncode"],
        "rusage": result["rusage"],
        "telemetry": telemetry,
        "stdout_sha256": stdout_sha,
        "stderr_sha256": stderr_sha,
        "stdout_bytes": len(result["stdout"]),
        "blob_before": before,
        "blob_after": after,
        "store_after": topology,
    }


def retain_candidate(candidate: dict[str, object]) -> dict[str, object]:
    source = Path(str(candidate["store_after"]["blob_path"]))
    target = ARTIFACT / "candidate.qcp"
    descriptor = os.open(
        target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600
    )
    try:
        with (
            source.open("rb", buffering=0) as input_file,
            os.fdopen(descriptor, "wb", buffering=0, closefd=False) as output,
        ):
            shutil.copyfileobj(input_file, output, length=8 * 1024 * 1024)
            output.flush()
            os.fsync(output.fileno())
    finally:
        os.close(descriptor)
    fsync_directory(ARTIFACT)
    metadata = target.stat(follow_symlinks=False)
    digest = sha256_file(target)
    with target.open("rb", buffering=0) as retained:
        retained.seek(-32, os.SEEK_END)
        trailer = retained.read(32).hex()
    if (
        metadata.st_size != BLOB_BYTES
        or digest != BLOB_SHA256
        or stat.S_IMODE(metadata.st_mode) != 0o600
        or metadata.st_nlink != 1
        or trailer != ENCODER_TRAILER
    ):
        raise ContractDefect("retained candidate identity drifted")
    return {
        "path": str(target.relative_to(ROOT)),
        "bytes": metadata.st_size,
        "sha256": digest,
        "nlink": metadata.st_nlink,
        "encoder_trailer": trailer,
    }


def required_gates() -> tuple[tuple[str, list[str]], ...]:
    return (
        (
            "qwen-and-bench-release-build",
            [
                "cargo",
                "build",
                "--release",
                "-p",
                "qwen-cli",
                "--bin",
                "qwen",
                "--bin",
                "qwen-bench",
            ],
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


def run_gate(name: str, argv: list[str], env: dict[str, str]) -> dict[str, object]:
    stem = f"gate-{name}"
    seal_launch("launch", "gate", stem, command=argv)
    result: dict[str, object] | None = None
    error: BaseException | None = None
    try:
        result = direct_process(argv, env, expected_stdout=None)
    except BaseException as failure:
        error = failure
    finally:
        seal_launch(
            "completion",
            "gate",
            stem,
            command=argv,
            returncode=result.get("returncode") if result else None,
            error=f"{type(error).__name__}: {error}" if error else None,
        )
    if error is not None:
        raise error
    if result is None:
        raise Inconclusive("population", f"gate result missing: {name}")
    stdout_sha, stderr_sha = persist_process(stem, result)
    release = release_process_signal_mask(result)
    write_json(ARTIFACT / f"{stem}.signal-release.json", release)
    if result["signal_control"]["error"] is not None:
        raise Inconclusive("signal-control", str(result["signal_control"]["error"]))
    if result["interruption"] is not None or result["pending_signals"]:
        raise Inconclusive(
            "operator-signal",
            str(result["interruption"] or result["pending_signals"]),
        )
    if (
        release["operator_signals_before_mask_restore"]
        or release["sigalrm_contamination_before_mask_restore"]
    ):
        raise Inconclusive("signal-control", f"mask-release contamination: {release}")
    if result["returncode"] < 0:
        raise Inconclusive("signal", f"gate terminated by signal: {name}")
    execution_reasons = process_execution_reasons(result)
    if execution_reasons:
        raise Inconclusive("pipe", ";".join(execution_reasons))
    if result["returncode"] != 0:
        raise ContractDefect(f"release gate failed: {name}: {result['returncode']}")
    row = {
        "schema": 1,
        "stage": "gate",
        "name": name,
        "command": argv,
        "returncode": 0,
        "wall_ms": (int(result["t_reaped_ns"]) - int(result["started_ns"])) / 1e6,
        "rusage": result["rusage"],
        "stdout_sha256": stdout_sha,
        "stderr_sha256": stderr_sha,
    }
    append_jsonl(ARTIFACT / "gates.jsonl", row)
    return row


def reserve(manifest: dict[str, object]) -> None:
    if any(
        path_lexists(path)
        for path in (ARTIFACT, WORK_ROOT, V0638_ARTIFACT, V0638_WORK_ROOT)
    ):
        raise ContractDefect("refusing v0.638 import or v0.639 root reuse")
    mkdir_private(ARTIFACT)
    write_json(ARTIFACT / "manifest.json", manifest)


def seal_packet(
    decision: dict[str, object],
    frozen_authority_inputs: dict[str, dict[str, object]] | None,
) -> None:
    write_json(ARTIFACT / "decision.json", decision)
    directory = os.open(ARTIFACT, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        names = sorted(
            name
            for name in os.listdir(directory)
            if name not in {"artifact-inventory.sha256", "packet-complete.json"}
        )
        hashes = []
        stamps = {}
        for name in names:
            descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory)
            try:
                before = os.fstat(descriptor)
                if (
                    not stat.S_ISREG(before.st_mode)
                    or stat.S_IMODE(before.st_mode) != 0o600
                    or before.st_nlink != 1
                ):
                    raise ContractDefect(
                        f"final artifact member is not private regular data: {name}"
                    )
                digest = hashlib.sha256()
                total = 0
                while chunk := os.read(descriptor, 1024 * 1024):
                    digest.update(chunk)
                    total += len(chunk)
                os.fsync(descriptor)
                after = os.fstat(descriptor)
                stamp = (
                    before.st_dev,
                    before.st_ino,
                    before.st_size,
                    before.st_mode,
                    before.st_nlink,
                    before.st_mtime_ns,
                    before.st_ctime_ns,
                )
                if stamp != (
                    after.st_dev,
                    after.st_ino,
                    after.st_size,
                    after.st_mode,
                    after.st_nlink,
                    after.st_mtime_ns,
                    after.st_ctime_ns,
                ):
                    raise ContractDefect(f"final artifact changed during seal: {name}")
                if total != before.st_size:
                    raise ContractDefect(f"final artifact short read: {name}")
                digest_text = digest.hexdigest()
                actual_identity = {
                    "path": str(ARTIFACT / name),
                    "sha256": digest_text,
                    "stamp": {
                        "dev": before.st_dev,
                        "ino": before.st_ino,
                        "bytes": before.st_size,
                        "mode": stat.S_IMODE(before.st_mode),
                        "nlink": before.st_nlink,
                        "mtime_ns": before.st_mtime_ns,
                        "ctime_ns": before.st_ctime_ns,
                    },
                }
                expected_identity = (frozen_authority_inputs or {}).get(name)
                if (
                    expected_identity is not None
                    and actual_identity != expected_identity
                ):
                    raise ContractDefect(
                        f"authority input changed between validation and seal: {name}"
                    )
                hashes.append((name, digest_text))
                stamps[name] = stamp
            finally:
                os.close(descriptor)
        if sorted(os.listdir(directory)) != names:
            raise ContractDefect("final artifact population changed during seal")
        for name, expected_stamp in stamps.items():
            observed = os.stat(name, dir_fd=directory, follow_symlinks=False)
            observed_stamp = (
                observed.st_dev,
                observed.st_ino,
                observed.st_size,
                observed.st_mode,
                observed.st_nlink,
                observed.st_mtime_ns,
                observed.st_ctime_ns,
            )
            if not stat.S_ISREG(observed.st_mode) or observed_stamp != expected_stamp:
                raise ContractDefect(
                    f"final artifact identity changed after authenticated read: {name}"
                )
        if frozen_authority_inputs is not None:
            if not set(frozen_authority_inputs).issubset(names):
                raise ContractDefect(
                    "validated authority input is absent at final seal"
                )
            generated = set(names) - set(frozen_authority_inputs)
            if generated != {"decision-cutoff.json", "decision.json"}:
                raise ContractDefect(
                    f"unexpected post-validation packet members: {sorted(generated)}"
                )
    finally:
        os.close(directory)
    fsync_directory(ARTIFACT)
    inventory = b"".join(
        f"{digest}  {name}\n".encode("ascii") for name, digest in hashes
    )
    write_new(ARTIFACT / "artifact-inventory.sha256", inventory)
    write_json(
        ARTIFACT / "packet-complete.json",
        {
            "schema": 1,
            "decision_sha256": sha256_file(ARTIFACT / "decision.json"),
            "inventory_sha256": sha256_file(ARTIFACT / "artifact-inventory.sha256"),
            "inventory_members": len(hashes),
            "final_files": len(hashes) + 2,
        },
    )
    for path in (
        ARTIFACT / "artifact-inventory.sha256",
        ARTIFACT / "packet-complete.json",
    ):
        identity = descriptor_hash_regular(path)
        if identity["stamp"]["mode"] != 0o600 or identity["stamp"]["nlink"] != 1:
            raise ContractDefect(f"final seal file is not private: {path.name}")
    if sorted(path.name for path in ARTIFACT.iterdir()) != sorted(
        [name for name, _ in hashes]
        + ["artifact-inventory.sha256", "packet-complete.json"]
    ):
        raise ContractDefect("final packet population changed after completion seal")


def authority_fields(status: str) -> dict[str, object]:
    go = status == "go"
    return {
        "authority": "force-only-exact-qwen3.6-27b-q4_k_m-deferred-restore"
        if go
        else "none",
        "force_authorized": go,
        "default_authorized": False,
        "automatic_authorized": False,
        "other_model_authorized": False,
        "other_shape_authorized": False,
        "successor_authorization": "none",
    }


def authenticate_json_artifact(path: Path, expected: object) -> dict[str, object]:
    data, identity = read_descriptor_regular(path)
    if identity["stamp"]["mode"] != 0o600 or identity["stamp"]["nlink"] != 1:
        raise ContractDefect(f"JSON artifact is not private: {path.name}")
    if data != json_bytes(expected, pretty=True):
        raise ContractDefect(f"JSON artifact content drifted: {path.name}")
    return identity


def read_json_artifact(path: Path) -> tuple[object, dict[str, object]]:
    data, identity = read_descriptor_regular(path)
    if (
        data is None
        or identity["stamp"]["mode"] != 0o600
        or identity["stamp"]["nlink"] != 1
    ):
        raise ContractDefect(f"JSON artifact is not private regular data: {path.name}")
    try:
        return json.loads(data.decode("utf-8", errors="strict")), identity
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ContractDefect(f"JSON artifact is invalid: {path.name}") from error


def read_jsonl_artifact(path: Path) -> tuple[list[object], dict[str, object]]:
    data, identity = read_descriptor_regular(path)
    if (
        data is None
        or identity["stamp"]["mode"] != 0o600
        or identity["stamp"]["nlink"] != 1
    ):
        raise ContractDefect(f"JSONL artifact is not private regular data: {path.name}")
    try:
        rows = [
            json.loads(line.decode("utf-8", errors="strict"))
            for line in data.splitlines()
        ]
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ContractDefect(f"JSONL artifact is invalid: {path.name}") from error
    return rows, identity


def authenticate_retained_candidate(retained: dict[str, object]) -> dict[str, object]:
    relative = retained.get("path")
    if not isinstance(relative, str):
        raise ContractDefect("retained candidate path is invalid")
    path = ROOT / relative
    identity = descriptor_hash_regular(path)
    stamp = identity["stamp"]
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        try:
            before = os.fstat(descriptor)
            os.lseek(descriptor, -32, os.SEEK_END)
            trailer = os.read(descriptor, 32).hex()
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
    except OSError as error:
        raise ContractDefect(
            "cannot authenticate retained candidate trailer"
        ) from error
    if (
        identity["sha256"] != BLOB_SHA256
        or stamp["bytes"] != BLOB_BYTES
        or stamp["mode"] != 0o600
        or stamp["nlink"] != 1
        or trailer != ENCODER_TRAILER
        or stamp["dev"] != before.st_dev
        or stamp["ino"] != before.st_ino
        or stamp["bytes"] != before.st_size
        or stamp["mode"] != stat.S_IMODE(before.st_mode)
        or stamp["nlink"] != before.st_nlink
        or stamp["mtime_ns"] != before.st_mtime_ns
        or stamp["ctime_ns"] != before.st_ctime_ns
        or (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mode,
            before.st_nlink,
        )
        != (after.st_dev, after.st_ino, after.st_size, after.st_mode, after.st_nlink)
    ):
        raise ContractDefect("retained candidate descriptor identity drifted")
    return identity | {"encoder_trailer": trailer}


def bind_authority_identity(
    frozen: dict[str, dict[str, object]], path: Path, identity: dict[str, object]
) -> None:
    try:
        relative = path.relative_to(ARTIFACT)
    except ValueError as error:
        raise ContractDefect(f"authority input is outside packet: {path}") from error
    if len(relative.parts) != 1:
        raise ContractDefect(f"authority input is not top-level packet data: {path}")
    normalized = {
        "path": identity.get("path"),
        "sha256": identity.get("sha256"),
        "stamp": identity.get("stamp"),
    }
    prior = frozen.get(relative.name)
    if prior is not None and prior != normalized:
        raise ContractDefect(f"authority input identity changed: {relative.name}")
    frozen[relative.name] = normalized


def validate_process_crosslinks(
    process: dict[str, object], returncode: int, rusage: object
) -> None:
    wait_status = process.get("wait_status")
    if (
        isinstance(wait_status, bool)
        or not isinstance(wait_status, int)
        or os.waitstatus_to_exitcode(wait_status) != returncode
        or process.get("returncode") != returncode
        or process.get("rusage") != rusage
        or process.get("signal_control", {}).get("strategy") != "sole-main-kqueue-wait4"
    ):
        raise ContractDefect("process wait status, rusage, or strategy drifted")


def validate_signal_release_evidence(
    process: dict[str, object],
    release: dict[str, object],
    *,
    require_packet_sigalrm_reservation: bool,
) -> None:
    snapshot = release.get("pending_snapshot_before_mask_restore")
    previous = release.get("previous_signal_mask")
    if (
        not isinstance(snapshot, list)
        or any(
            isinstance(value, bool) or not isinstance(value, int) for value in snapshot
        )
        or snapshot != sorted(set(snapshot))
        or not set(snapshot).issubset({int(value) for value in CHILD_WAIT_SIGNALS})
        or not isinstance(previous, list)
        or any(
            isinstance(value, bool) or not isinstance(value, int) for value in previous
        )
        or previous != sorted(set(previous))
    ):
        raise ContractDefect("signal-release masks are not canonical controlled sets")
    operator = [value for value in snapshot if value in OPERATOR_SIGNALS]
    controlled_after = sorted(
        value for value in previous if signal.Signals(value) in CHILD_WAIT_SIGNALS
    )
    alarm_reserved = int(signal.SIGALRM) in previous
    if (
        release.get("schema") != 1
        or release.get("operator_signals_before_mask_restore") != operator
        or release.get("sigalrm_contamination_before_mask_restore")
        is not (int(signal.SIGALRM) in snapshot)
        or process.get("operator_signals_blocked") is not True
        or process.get("previous_signal_mask") != previous
        or release.get("process_operator_signals_blocked_before_release") is not True
        or release.get("previous_mask_reapplied") is not True
        or release.get("controlled_signals_blocked_after_release") != controlled_after
        or release.get("sigalrm_reserved_after_release") is not alarm_reserved
        or release.get("snapshot_and_mask_restore_are_not_atomic") is not True
        or (require_packet_sigalrm_reservation and not alarm_reserved)
    ):
        raise ContractDefect("signal-release evidence does not independently rederive")


def validate_conditioning_evidence(conditioning: dict[str, object], stem: str) -> None:
    model_read = conditioning.get("model_hash_read")
    if (
        conditioning.get("schema") != 1
        or conditioning.get("stem") != stem
        or conditioning.get("valid") is not True
        or not isinstance(model_read, dict)
        or model_read.get("bytes") != MODEL_BYTES
        or model_read.get("sha256") != MODEL_SHA256
        or conditioning.get("model_stamp_before_spawn") != model_read.get("stamp")
    ):
        raise ContractDefect("conditioning schema, stem, or model identity drifted")
    recomputed = host_protocol.vm_interval(
        "conditioning",
        conditioning.get("vm_before_conditioning"),
        conditioning.get("vm_before_spawn"),
    )
    if (
        recomputed != conditioning.get("conditioning_interval")
        or recomputed.get("failure_reasons") != []
        or conditioning.get("host_before_spawn", {}).get("valid") is not True
    ):
        raise ContractDefect("conditioning host/VM evidence does not rederive")


def validate_post_exit_evidence(
    conditioning: dict[str, object],
    post: dict[str, object],
    rusage: dict[str, object],
) -> None:
    interval = host_protocol.vm_interval(
        "child", conditioning.get("vm_before_spawn"), post.get("vm_after_exit")
    )
    reasons = list(interval["failure_reasons"])
    if post.get("host_after_exit", {}).get("valid") is not True:
        reasons.append("post_exit_host_invalid")
    if rusage.get("ru_nswap") != 0:
        reasons.append("ru_nswap_nonzero")
    if rusage.get("ru_inblock") != 0:
        reasons.append("ru_inblock_nonzero")
    if (
        post.get("schema") != 1
        or post.get("child_interval") != interval
        or post.get("major_faults_advisory") != rusage.get("ru_majflt")
        or post.get("validity_reasons") != reasons
        or post.get("valid") is not (not reasons)
    ):
        raise ContractDefect("post-exit validity does not independently rederive")


def validate_go_conjunction(
    manifest: dict[str, object],
    gates: list[dict[str, object]],
    rows: list[dict[str, object]],
    build_identity: dict[str, object] | None,
    input_seal: dict[str, object] | None,
    performance: dict[str, object] | None,
    restore: dict[str, object] | None,
    retained: dict[str, object] | None,
    execution_host: dict[str, object] | None,
    final_qwen_identity: dict[str, object] | None,
    final_bench_identity: dict[str, object] | None,
    env: dict[str, str],
) -> dict[str, dict[str, object]]:
    frozen: dict[str, dict[str, object]] = {}
    if authority_fields("go") != {
        "authority": "force-only-exact-qwen3.6-27b-q4_k_m-deferred-restore",
        "force_authorized": True,
        "default_authorized": False,
        "automatic_authorized": False,
        "other_model_authorized": False,
        "other_shape_authorized": False,
        "successor_authorization": "none",
    }:
        raise ContractDefect("GO authorized scope drifted")
    bind_authority_identity(
        frozen,
        ARTIFACT / "manifest.json",
        authenticate_json_artifact(ARTIFACT / "manifest.json", manifest),
    )
    if not isinstance(build_identity, dict):
        raise ContractDefect("GO has no post-gate execution build identity")
    expected_history = authority_history()
    if manifest.get("authority_history") != expected_history:
        raise ContractDefect("GO authority history does not rederive exactly")
    fresh_execution, fresh_dirty, fresh_source_state = source_identity(ROOT)
    fresh_index_flags = git_bytes(ROOT, "ls-files", "-v", "-z")
    fresh_hidden_index = any(
        entry[:1].islower() or entry.startswith(b"S")
        for entry in fresh_index_flags.split(b"\0")
        if entry
    )
    recorded_source = build_identity.get("source_identity")
    manifest_source = manifest.get("source")
    fresh_source = {
        "execution_commit": fresh_execution,
        "dirty": False,
        "source_state": fresh_source_state,
    }
    if (
        re.fullmatch(r"[0-9a-f]{40}", fresh_execution) is None
        or fresh_dirty is not False
        or fresh_hidden_index is not False
        or single_parent(fresh_execution) != PARENT
        or manifest.get("execution_commit") != fresh_execution
        or not isinstance(manifest_source, dict)
        or manifest_source.get("execution_commit") != fresh_execution
        or recorded_source != fresh_source
    ):
        raise ContractDefect("GO fresh R639 source identity does not rederive exactly")
    exact_raw_policy = {
        "qwen": {"requires_full_commit": True, "requires_source_state": True},
        "qwen-bench": {
            "requires_full_commit": False,
            "requires_source_state": True,
        },
    }
    if (
        RAW_BUILD_POLICY != exact_raw_policy
        or build_identity.get("raw_policy") != exact_raw_policy
    ):
        raise ContractDefect("GO recorded asymmetric raw policy drifted")
    binaries = build_identity.get("binaries")
    if not isinstance(binaries, dict) or set(binaries) != {"qwen", "qwen-bench"}:
        raise ContractDefect("GO post-gate binary records are incomplete")
    current_identities = {}
    for name, path in (("qwen", CLI), ("qwen-bench", BENCH)):
        payload, current_identity = executable_descriptor(path)
        current_raw = validate_raw_build_literals(
            name, payload, fresh_execution, fresh_source_state
        )
        expected_record = {
            "descriptor_identity": current_identity,
            "raw_authentication": current_raw,
        }
        if binaries.get(name) != expected_record:
            raise ContractDefect(f"GO recorded {name} raw authentication drifted")
        current_identities[name] = current_identity
    current_qwen = current_identities["qwen"]
    if current_qwen != final_qwen_identity:
        raise ContractDefect("GO final qwen identity does not match current binary")
    bench_semantics, bracketed_bench_identity = bench_semantic_gate(
        fresh_execution, fresh_source_state, env
    )
    require_identity_equal(
        current_identities["qwen-bench"],
        bracketed_bench_identity,
        "GO qwen-bench raw/semantic",
    )
    if bench_semantics != build_identity.get("qwen_bench_build_info"):
        raise ContractDefect("GO recorded qwen-bench semantic identity drifted")
    current_bench = {
        "descriptor_identity": bracketed_bench_identity,
        "build_info": bench_semantics,
    }
    if current_bench != final_bench_identity:
        raise ContractDefect("GO final qwen-bench identity/semantics drifted")
    if not isinstance(execution_host, dict):
        raise ContractDefect("GO has no post-gate host identity")
    bind_authority_identity(
        frozen,
        ARTIFACT / "execution-identity.json",
        authenticate_json_artifact(
            ARTIFACT / "execution-identity.json",
            {"build_identity": build_identity, "host_boundary": execution_host},
        ),
    )
    bind_authority_identity(
        frozen,
        ARTIFACT / "final-qwen-bench-identity.json",
        authenticate_json_artifact(
            ARTIFACT / "final-qwen-bench-identity.json", final_bench_identity
        ),
    )
    bind_authority_identity(
        frozen,
        ARTIFACT / "final-execution-identity.json",
        authenticate_json_artifact(
            ARTIFACT / "final-execution-identity.json", final_qwen_identity
        ),
    )
    if not isinstance(input_seal, dict):
        raise ContractDefect("GO has no immutable input seal")
    packet_input_identities = authenticate_packet_inputs(input_seal)
    for identity in packet_input_identities.values():
        bind_authority_identity(frozen, Path(str(identity["path"])), identity)
    bind_authority_identity(
        frozen,
        ARTIFACT / "input-seal.json",
        authenticate_json_artifact(ARTIFACT / "input-seal.json", input_seal),
    )

    expected_gates = required_gates()
    gate_artifact_rows, gate_artifact_identity = read_jsonl_artifact(
        ARTIFACT / "gates.jsonl"
    )
    row_artifact_rows, row_artifact_identity = read_jsonl_artifact(
        ARTIFACT / "rows.jsonl"
    )
    restore_artifact, restore_artifact_identity = read_json_artifact(
        ARTIFACT / "restore.json"
    )
    attempt_artifact_rows, attempt_artifact_identity = read_jsonl_artifact(
        ARTIFACT / "attempts.jsonl"
    )
    for path, identity in (
        (ARTIFACT / "gates.jsonl", gate_artifact_identity),
        (ARTIFACT / "rows.jsonl", row_artifact_identity),
        (ARTIFACT / "restore.json", restore_artifact_identity),
        (ARTIFACT / "attempts.jsonl", attempt_artifact_identity),
    ):
        bind_authority_identity(frozen, path, identity)
    if gate_artifact_rows != gates:
        raise ContractDefect("GO gates.jsonl does not exactly match gate state")
    if row_artifact_rows != rows:
        raise ContractDefect("GO rows.jsonl does not exactly match scored state")
    if restore_artifact != restore:
        raise ContractDefect("GO restore.json does not exactly match restore state")
    if attempt_artifact_rows != [*rows, restore]:
        raise ContractDefect("GO attempts.jsonl is not exactly rows plus restore")
    if len(gates) != len(expected_gates):
        raise ContractDefect("GO gate population drifted")
    for gate, (name, argv) in zip(gates, expected_gates, strict=True):
        if (
            gate.get("name") != name
            or gate.get("schema") != 1
            or gate.get("stage") != "gate"
            or gate.get("command") != argv
            or gate.get("returncode") != 0
        ):
            raise ContractDefect("GO gate name/command/order/result drifted")
        gate_process_path = ARTIFACT / f"gate-{name}.process.json"
        gate_process, gate_process_identity = read_json_artifact(gate_process_path)
        gate_release_path = ARTIFACT / f"gate-{name}.signal-release.json"
        gate_release, gate_release_identity = read_json_artifact(gate_release_path)
        gate_stdout_path = ARTIFACT / f"gate-{name}.out"
        gate_stderr_path = ARTIFACT / f"gate-{name}.err"
        gate_stdout_identity = descriptor_hash_regular(gate_stdout_path)
        gate_stderr_identity = descriptor_hash_regular(gate_stderr_path)
        validate_process_crosslinks(
            gate_process, int(gate["returncode"]), gate.get("rusage")
        )
        validate_signal_release_evidence(
            gate_process,
            gate_release,
            require_packet_sigalrm_reservation=True,
        )
        if (
            gate_process.get("returncode") != 0
            or gate_process.get("interruption") is not None
            or gate_process.get("pending_signals") != []
            or gate_process.get("signal_control", {}).get("error") is not None
            or operator_control_records(gate_process)
            or process_execution_reasons(gate_process)
            or gate_stdout_identity["sha256"] != gate.get("stdout_sha256")
            or gate_stderr_identity["sha256"] != gate.get("stderr_sha256")
            or gate_release.get("operator_signals_before_mask_restore") != []
            or gate_release.get("sigalrm_contamination_before_mask_restore")
            is not False
            or gate.get("wall_ms")
            != (int(gate_process["t_reaped_ns"]) - int(gate_process["started_ns"]))
            / 1e6
        ):
            raise ContractDefect("GO gate process evidence drifted")
        for path, identity in (
            (gate_process_path, gate_process_identity),
            (gate_release_path, gate_release_identity),
            (gate_stdout_path, gate_stdout_identity),
            (gate_stderr_path, gate_stderr_identity),
        ):
            bind_authority_identity(frozen, path, identity)

    if len(rows) != 12:
        raise ContractDefect("GO scored row population drifted")
    current_packet_inputs = authenticate_packet_inputs(input_seal)
    launch_records, launch_identity = read_jsonl_artifact(
        ARTIFACT / "launch-seal.jsonl"
    )
    bind_authority_identity(frozen, ARTIFACT / "launch-seal.jsonl", launch_identity)
    expected_history = [
        {
            "stem": f"gate-{name}",
            "stage": "gate",
            "command": argv,
        }
        for name, argv in expected_gates
    ]
    for pair_index, order in enumerate(PAIR_ORDERS, 1):
        for position, arm in enumerate(order, 1):
            stem = (
                f"product-p{pair_index:02d}-{order.lower()}-r{position}-{arm.lower()}"
            )
            expected_history.append(
                {
                    "stem": stem,
                    "stage": "scored",
                    "command": child_command(WORK_ROOT / stem),
                }
            )
    restore_stem = "restore-p01-position2-b"
    restore_store = WORK_ROOT / "product-p01-ab-r2-b"
    expected_history.append(
        {
            "stem": restore_stem,
            "stage": "restore",
            "command": child_command(restore_store, restore=True),
        }
    )
    if len(launch_records) != 2 * len(expected_history):
        raise ContractDefect("GO launch/completion record count drifted")
    for index, expected in enumerate(expected_history):
        pair = launch_records[index * 2 : index * 2 + 2]
        if (
            any(record.get("schema") != 1 for record in pair)
            or [record.get("event") for record in pair] != ["launch", "completion"]
            or any(record.get("stem") != expected["stem"] for record in pair)
            or any(record.get("stage") != expected["stage"] for record in pair)
            or any(record.get("command") != expected["command"] for record in pair)
            or pair[1].get("returncode") != 0
            or pair[1].get("error") is not None
        ):
            raise ContractDefect("GO launch/completion sequence drifted")
    for offset, row in enumerate(rows):
        pair_index = offset // 2 + 1
        position = offset % 2 + 1
        order = PAIR_ORDERS[pair_index - 1]
        arm = order[position - 1]
        mode = arm_mode(arm)
        expected_stem = (
            f"product-p{pair_index:02d}-{order.lower()}-r{position}-{arm.lower()}"
        )
        store_root = Path(str(row.get("store_before", {}).get("root")))
        if (
            row.get("stem") != expected_stem
            or row.get("schema") != 1
            or row.get("stage") != "scored"
            or store_root != WORK_ROOT / expected_stem
            or row.get("pair_index") != pair_index
            or row.get("pair_order") != order
            or row.get("position") != position
            or row.get("arm") != arm
            or row.get("mode") != mode
            or row.get("command") != child_command(store_root)
            or row.get("arm_environment") != {INTEGRITY_ENV: mode}
            or row.get("returncode") != 0
            or row.get("stdout_sha256") != STDOUT_SHA256
            or row.get("valid") is not True
            or row.get("store_before", {}).get("identity_sha256") != IDENTITY_SHA256
            or row.get("store_before", {}).get("blob_count") != 0
        ):
            raise ContractDefect("GO scored row command/order/identity drifted")
        store = row.get("store_after")
        telemetry = row.get("telemetry")
        if (
            not isinstance(store, dict)
            or store.get("blob_relative") != str(BLOB_RELATIVE)
            or store.get("blob_bytes") != BLOB_BYTES
            or store.get("blob_sha256") != BLOB_SHA256
            or store.get("encoder_trailer") != ENCODER_TRAILER
            or store.get("blob_mode") != "0o600"
            or store.get("blob_nlink") != 1
            or store.get("files") != 3
            or store.get("directories") != 5
            or not isinstance(telemetry, dict)
            or telemetry.get("staged_integrity") != mode
            or telemetry.get("publish") != "published"
            or telemetry.get("capture") != "completed"
            or telemetry.get("matched_tokens") != 6500
            or telemetry.get("restored_tokens") != 6499
            or telemetry.get("pending") is not True
            or telemetry.get("stop_reason") != "token_limit"
            or telemetry.get("blob_bytes") != BLOB_BYTES
            or telemetry.get("evicted") != 0
            or telemetry.get("identity") != "hit"
            or telemetry.get("external_us") != row.get("external_us")
        ):
            raise ContractDefect("GO scored row telemetry/topology drifted")
        for name in ("staged_integrity_us", "publish_us", "post_response_us"):
            if isinstance(telemetry.get(name), bool) or not isinstance(
                telemetry.get(name), int
            ):
                raise ContractDefect("GO scored integer telemetry drifted")
        require_observation_quality(
            int(row["external_us"]), int(telemetry["post_response_us"])
        )
        conditioning_evidence, conditioning_identity = read_json_artifact(
            ARTIFACT / f"{row['stem']}.conditioning.json"
        )
        process_evidence, process_identity = read_json_artifact(
            ARTIFACT / f"{row['stem']}.process.json"
        )
        post, post_identity = read_json_artifact(
            ARTIFACT / f"{row['stem']}.post-exit.json"
        )
        release, release_identity = read_json_artifact(
            ARTIFACT / f"{row['stem']}.signal-release.json"
        )
        if (
            conditioning_evidence.get("valid") is not True
            or post.get("valid") is not True
            or process_evidence.get("returncode") != 0
            or process_evidence.get("interruption") is not None
            or process_evidence.get("pending_signals") != []
            or process_evidence.get("signal_control", {}).get("error") is not None
            or operator_control_records(process_evidence)
            or process_execution_reasons(process_evidence)
            or release.get("operator_signals_before_mask_restore") != []
            or release.get("sigalrm_contamination_before_mask_restore") is not False
        ):
            raise ContractDefect(
                "GO scored conditioning/process/post evidence is invalid"
            )
        validate_process_crosslinks(
            process_evidence, int(row["returncode"]), row.get("rusage")
        )
        validate_signal_release_evidence(
            process_evidence,
            release,
            require_packet_sigalrm_reservation=True,
        )
        validate_conditioning_evidence(conditioning_evidence, expected_stem)
        validate_post_exit_evidence(
            conditioning_evidence, post, process_evidence["rusage"]
        )
        stdout_data, stdout_identity = read_descriptor_regular(
            ARTIFACT / f"{row['stem']}.out"
        )
        stderr_data, stderr_identity = read_descriptor_regular(
            ARTIFACT / f"{row['stem']}.err"
        )
        if stdout_data != STDOUT_BYTES or stdout_identity["sha256"] != STDOUT_SHA256:
            raise ContractDefect("GO scored stdout artifact identity drifted")
        if (
            stderr_data is None
            or stderr_identity["sha256"] != row.get("stderr_sha256")
            or parse_scored(stderr_data, arm, int(row["external_us"])) != telemetry
        ):
            raise ContractDefect("GO scored stderr artifact identity drifted")
        complete_ns = row.get("t_stdout_complete_ns")
        reaped_ns = row.get("t_reaped_ns")
        if (
            isinstance(complete_ns, bool)
            or not isinstance(complete_ns, int)
            or isinstance(reaped_ns, bool)
            or not isinstance(reaped_ns, int)
            or (reaped_ns - complete_ns) // 1000 != row.get("external_us")
            or process_evidence.get("t_stdout_complete_ns") != complete_ns
            or process_evidence.get("t_reaped_ns") != reaped_ns
            or process_evidence.get("stdout_sha256") != row.get("stdout_sha256")
            or process_evidence.get("stderr_sha256") != row.get("stderr_sha256")
        ):
            raise ContractDefect("GO external/process endpoint does not rederive")
        for path, identity in (
            (ARTIFACT / f"{row['stem']}.conditioning.json", conditioning_identity),
            (ARTIFACT / f"{row['stem']}.process.json", process_identity),
            (ARTIFACT / f"{row['stem']}.post-exit.json", post_identity),
            (ARTIFACT / f"{row['stem']}.signal-release.json", release_identity),
            (ARTIFACT / f"{row['stem']}.out", stdout_identity),
            (ARTIFACT / f"{row['stem']}.err", stderr_identity),
        ):
            bind_authority_identity(frozen, path, identity)
        launches = [
            record
            for record in launch_records
            if record.get("event") == "launch" and record.get("stem") == row["stem"]
        ]
        if (
            len(launches) != 1
            or launches[0].get("command") != row["command"]
            or launches[0].get("arm") != arm
            or launches[0].get("mode") != mode
            or launches[0].get("pair_index") != pair_index
            or launches[0].get("pair_order") != order
            or launches[0].get("position") != position
            or launches[0].get("conditioning_sha256") != conditioning_identity["sha256"]
            or launches[0].get("packet_inputs") != current_packet_inputs
            or launches[0].get("model_stamp_immediate")
            != conditioning_evidence.get("model_stamp_before_spawn")
            or launches[0].get("qwen_identity") != current_qwen
        ):
            raise ContractDefect("GO scored launch evidence drifted")

    recomputed = analyze(rows)
    if recomputed != performance or any(
        not pair["publish_gate"] or not pair["external_gate"]
        for pair in recomputed["pairs"]
    ):
        raise ContractDefect("GO performance does not rederive all six pair gates")

    candidate = rows[1]
    if (
        candidate.get("pair_index") != 1
        or candidate.get("position") != 2
        or candidate.get("arm") != "B"
        or not isinstance(restore, dict)
        or restore.get("schema") != 1
        or restore.get("stage") != "restore"
        or restore.get("stem") != "restore-p01-position2-b"
        or restore.get("valid") is not True
        or restore.get("command")
        != child_command(Path(str(candidate["store_before"]["root"])), restore=True)
        or restore.get("arm_environment") != {INTEGRITY_ENV: arm_mode("B")}
        or restore.get("returncode") != 0
        or restore.get("blob_before") != restore.get("blob_after")
        or restore.get("blob_before", {}).get("sha256") != BLOB_SHA256
        or restore.get("blob_before", {}).get("dev")
        != candidate.get("store_after", {}).get("blob_dev")
        or restore.get("blob_before", {}).get("ino")
        != candidate.get("store_after", {}).get("blob_ino")
    ):
        raise ContractDefect("GO restore command/source/blob identity drifted")
    telemetry = restore.get("telemetry")
    exact_restore = {
        "identity_cache": "hit",
        "hashed_bytes": 0,
        "checkpoint_hit": True,
        "matched": 6500,
        "restored": 6499,
        "exact": False,
        "candidates": 1,
        "corrupt_removed": 0,
    }
    if not isinstance(telemetry, dict) or any(
        telemetry.get(key) != value for key, value in exact_restore.items()
    ):
        raise ContractDefect("GO restore telemetry does not rederive")
    restore_conditioning, restore_conditioning_identity = read_json_artifact(
        ARTIFACT / f"{restore['stem']}.conditioning.json"
    )
    restore_process, restore_process_identity = read_json_artifact(
        ARTIFACT / f"{restore['stem']}.process.json"
    )
    restore_post, restore_post_identity = read_json_artifact(
        ARTIFACT / f"{restore['stem']}.post-exit.json"
    )
    restore_release, restore_release_identity = read_json_artifact(
        ARTIFACT / f"{restore['stem']}.signal-release.json"
    )
    if (
        restore_conditioning.get("valid") is not True
        or restore_post.get("valid") is not True
        or restore_process.get("returncode") != 0
        or restore_process.get("interruption") is not None
        or restore_process.get("pending_signals") != []
        or restore_process.get("signal_control", {}).get("error") is not None
        or operator_control_records(restore_process)
        or process_execution_reasons(restore_process)
        or restore_release.get("operator_signals_before_mask_restore") != []
        or restore_release.get("sigalrm_contamination_before_mask_restore") is not False
    ):
        raise ContractDefect("GO restore evidence validity drifted")
    validate_process_crosslinks(
        restore_process, int(restore["returncode"]), restore.get("rusage")
    )
    validate_signal_release_evidence(
        restore_process,
        restore_release,
        require_packet_sigalrm_reservation=True,
    )
    validate_conditioning_evidence(restore_conditioning, "restore-p01-position2-b")
    validate_post_exit_evidence(
        restore_conditioning, restore_post, restore_process["rusage"]
    )
    restore_stdout, restore_stdout_identity = read_descriptor_regular(
        ARTIFACT / f"{restore['stem']}.out"
    )
    restore_stderr, restore_stderr_identity = read_descriptor_regular(
        ARTIFACT / f"{restore['stem']}.err"
    )
    if (
        restore_stdout is None
        or len(restore_stdout) == 0
        or len(restore_stdout) != restore.get("stdout_bytes")
        or restore_stdout_identity["sha256"] != restore.get("stdout_sha256")
        or restore_stderr is None
        or restore_stderr_identity["sha256"] != restore.get("stderr_sha256")
        or restore_process.get("stdout_sha256") != restore.get("stdout_sha256")
        or restore_process.get("stderr_sha256") != restore.get("stderr_sha256")
        or parse_restore(restore_stderr) != telemetry
    ):
        raise ContractDefect("GO restore stdout/stderr semantics or digest drifted")
    restore_store = restore.get("store_after")
    if (
        not isinstance(restore_store, dict)
        or restore_store.get("blob_relative") != str(BLOB_RELATIVE)
        or restore_store.get("blob_sha256") != BLOB_SHA256
        or restore_store.get("encoder_trailer") != ENCODER_TRAILER
        or restore_store.get("blob_bytes") != BLOB_BYTES
        or restore_store.get("blob_mode") != "0o600"
        or restore_store.get("blob_nlink") != 1
        or restore_store.get("files") != 3
        or restore_store.get("directories") != 5
    ):
        raise ContractDefect("GO restore store topology drifted")
    restore_launches = [
        record
        for record in launch_records
        if record.get("event") == "launch" and record.get("stem") == restore["stem"]
    ]
    if (
        len(restore_launches) != 1
        or restore_launches[0].get("command") != restore["command"]
        or restore_launches[0].get("arm") != "B"
        or restore_launches[0].get("mode") != arm_mode("B")
        or restore_launches[0].get("conditioning_sha256")
        != restore_conditioning_identity["sha256"]
        or restore_launches[0].get("packet_inputs") != current_packet_inputs
        or restore_launches[0].get("model_stamp_immediate")
        != restore_conditioning.get("model_stamp_before_spawn")
        or restore_launches[0].get("qwen_identity") != current_qwen
    ):
        raise ContractDefect("GO restore launch identity drifted")
    if not isinstance(retained, dict):
        raise ContractDefect("GO retained candidate record is absent")
    retained_identity = authenticate_retained_candidate(retained)
    if (
        retained.get("sha256") != retained_identity["sha256"]
        or retained.get("bytes") != retained_identity["stamp"]["bytes"]
        or retained.get("nlink") != retained_identity["stamp"]["nlink"]
        or retained.get("encoder_trailer") != retained_identity["encoder_trailer"]
    ):
        raise ContractDefect("GO retained candidate record does not match descriptor")
    for path, identity in (
        (
            ARTIFACT / f"{restore['stem']}.conditioning.json",
            restore_conditioning_identity,
        ),
        (ARTIFACT / f"{restore['stem']}.process.json", restore_process_identity),
        (ARTIFACT / f"{restore['stem']}.post-exit.json", restore_post_identity),
        (
            ARTIFACT / f"{restore['stem']}.signal-release.json",
            restore_release_identity,
        ),
        (ARTIFACT / f"{restore['stem']}.out", restore_stdout_identity),
        (ARTIFACT / f"{restore['stem']}.err", restore_stderr_identity),
        (ROOT / str(retained["path"]), retained_identity),
    ):
        bind_authority_identity(frozen, path, identity)
    if path_lexists(WORK_ROOT):
        raise ContractDefect("GO work root cleanup is incomplete")
    return dict(sorted(frozen.items()))


def run(*, preflight_only: bool) -> None:
    self_test_core()
    bridge = authenticate_v0637()
    source = verify_source(bridge)
    env, environment = normalized_environment()
    if preflight_only:
        manifest = build_manifest(source, environment, env, require_build=True)
        vm = host_protocol.capture_vm_state()
        host = host_protocol.capture_host_state()
        if vm["capture_errors"] or host.get("valid") is not True:
            raise Inconclusive("preflight", "host or VM preflight invalid")
        print(
            json_bytes(
                {
                    "status": "preflight-pass",
                    "execution_commit": source["execution_commit"],
                    "manifest_sha256": sha256_bytes(json_bytes(manifest, pretty=True)),
                },
                pretty=True,
            ).decode(),
            end="",
        )
        return
    signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGALRM})
    manifest = build_manifest(source, environment, env, require_build=False)
    gates: list[dict[str, object]] = []
    rows: list[dict[str, object]] = []
    performance: dict[str, object] | None = None
    restore: dict[str, object] | None = None
    retained: dict[str, object] | None = None
    build_identity: dict[str, object] | None = None
    final_qwen_identity: dict[str, object] | None = None
    final_bench_identity: dict[str, object] | None = None
    execution_host: dict[str, object] | None = None
    input_seal: dict[str, object] | None = None
    frozen_authority_inputs: dict[str, dict[str, object]] | None = None
    status = "implementation_or_contract_defect"
    stopped_after = "reservation"
    error: str | None = None
    reserve(manifest)
    try:
        stopped_after = "inputs"
        input_seal = freeze_inputs()
        authenticate_packet_inputs(input_seal)
        stopped_after = "work-root"
        mkdir_private(WORK_ROOT)
        stopped_after = "gates"
        for name, argv in required_gates():
            gates.append(run_gate(name, argv, env))
        if verify_source(bridge) != source:
            raise ContractDefect("source drifted after fresh build/test gates")
        build_identity = verify_build(source, env)
        execution_host = host_boundary(env)
        write_json(
            ARTIFACT / "execution-identity.json",
            {
                "build_identity": build_identity,
                "host_boundary": execution_host,
            },
        )
        stopped_after = "scored"
        for pair_index, order in enumerate(PAIR_ORDERS, 1):
            pair_rows = []
            for position, arm in enumerate(order, 1):
                row = launch_scored(
                    arm,
                    pair_index,
                    order,
                    position,
                    env,
                    source,
                    input_seal,
                    build_identity,
                )
                append_jsonl(ARTIFACT / "attempts.jsonl", row)
                append_jsonl(ARTIFACT / "rows.jsonl", row)
                rows.append(row)
                pair_rows.append(row)
            left = Path(str(pair_rows[0]["store_after"]["blob_path"]))
            right = Path(str(pair_rows[1]["store_after"]["blob_path"]))
            if not stream_equal(left, right):
                raise ContractDefect(f"pair {pair_index} blobs are not streaming-equal")
        if {row["stdout_sha256"] for row in rows} != {STDOUT_SHA256} or {
            row["store_after"]["blob_sha256"] for row in rows
        } != {BLOB_SHA256}:
            raise ContractDefect("global response/blob identity drifted")
        performance = analyze(rows)
        status = "kill"
        if performance["passes"]:
            stopped_after = "restore"
            candidate = next(
                row
                for row in rows
                if row["pair_index"] == 1 and row["position"] == 2 and row["arm"] == "B"
            )
            restore = run_restore(candidate, env, source, input_seal, build_identity)
            append_jsonl(ARTIFACT / "attempts.jsonl", restore)
            write_json(ARTIFACT / "restore.json", restore)
            retained = retain_candidate(candidate)
            status = "go"
        if verify_source(bridge) != source:
            raise ContractDefect("source drifted after final v0.639 child")
        final_qwen_identity = authenticate_qwen_binary(build_identity)
        write_json(ARTIFACT / "final-execution-identity.json", final_qwen_identity)
        final_bench_identity = authenticate_qwen_bench(build_identity, env)
        write_json(ARTIFACT / "final-qwen-bench-identity.json", final_bench_identity)
    except Inconclusive as failure:
        status = "inconclusive"
        stopped_after = failure.stage
        error = f"{type(failure).__name__}: {failure}"
    except KeyboardInterrupt as failure:
        status = "inconclusive"
        stopped_after = "operator-signal"
        error = f"{type(failure).__name__}: {failure}"
    except Exception as failure:
        status = "implementation_or_contract_defect"
        error = f"{type(failure).__name__}: {failure}"
    finally:
        signal.pthread_sigmask(signal.SIG_BLOCK, OPERATOR_SIGNALS | {signal.SIGALRM})
        if path_lexists(WORK_ROOT):
            try:
                shutil.rmtree(WORK_ROOT)
                fsync_directory(WORK_ROOT.parent)
            except Exception as cleanup_error:
                status = "implementation_or_contract_defect"
                error = (
                    f"{error}; cleanup={type(cleanup_error).__name__}: {cleanup_error}"
                )
        if path_lexists(WORK_ROOT):
            status = "implementation_or_contract_defect"
            error = f"{error}; work root residue remains"
    if status == "go":
        try:
            frozen_authority_inputs = validate_go_conjunction(
                manifest,
                gates,
                rows,
                build_identity,
                input_seal,
                performance,
                restore,
                retained,
                execution_host,
                final_qwen_identity,
                final_bench_identity,
                env,
            )
        except Exception as failure:
            status = "implementation_or_contract_defect"
            stopped_after = "go-validation"
            error = f"{type(failure).__name__}: {failure}"
    cutoff_signal_types = OPERATOR_SIGNALS | {signal.SIGALRM}
    cutoff_snapshot = sorted(
        int(value) for value in (set(signal.sigpending()) & cutoff_signal_types)
    )
    cutoff_monotonic_ns = time.perf_counter_ns()
    cutoff_unix_ns = time.time_ns()
    for value in cutoff_snapshot:
        signal.sigwait({signal.Signals(value)})
    decision_cutoff = {
        "schema": 1,
        "cutoff_unix_ns": cutoff_unix_ns,
        "cutoff_monotonic_ns": cutoff_monotonic_ns,
        "authority_cutoff_operation": "single-sigpending-snapshot",
        "controlled_signal_types_in_cutoff_snapshot": cutoff_snapshot,
        "sigalrm_contamination_in_cutoff_snapshot": int(signal.SIGALRM)
        in cutoff_snapshot,
        "timestamp_is_observational_after_snapshot": True,
        "authority_inputs_frozen_at_cutoff": True,
        "signals_after_cutoff_outside_packet_authority": True,
    }
    if cutoff_snapshot:
        signal_text = "controlled signals in decision cutoff snapshot: " + ",".join(
            str(value) for value in cutoff_snapshot
        )
        error = f"{error}; {signal_text}" if error else signal_text
        if status != "implementation_or_contract_defect":
            status = "inconclusive"
            stopped_after = (
                "sigalrm-contamination-cutoff"
                if int(signal.SIGALRM) in cutoff_snapshot
                else "operator-signal-cutoff"
            )
    write_json(ARTIFACT / "decision-cutoff.json", decision_cutoff)
    decision = {
        "schema": 1,
        "status": status,
        **authority_fields(status),
        "execution_commit": source["execution_commit"],
        "implementation_commit": IMPLEMENTATION,
        "build_identity": build_identity,
        "final_qwen_identity": final_qwen_identity,
        "final_qwen_bench_identity": final_bench_identity,
        "decision_cutoff": decision_cutoff,
        "frozen_authority_inputs": frozen_authority_inputs,
        "input_seal_sha256": sha256_file(ARTIFACT / "input-seal.json")
        if (ARTIFACT / "input-seal.json").is_file()
        else None,
        "stopped_after": stopped_after,
        "error": error,
        "gates": gates,
        "rows": rows,
        "performance": performance,
        "restore": restore,
        "retained_candidate": retained,
        "v0637_packet": bridge,
        "authority_history": authority_history(),
    }
    seal_packet(decision, frozen_authority_inputs)
    if status == "implementation_or_contract_defect":
        raise ContractDefect(error or "unknown v0.639 contract defect")


def expect_rejected(function: object, *args: object) -> None:
    try:
        function(*args)
    except ContractDefect:
        return
    raise ContractDefect("self-test accepted adversarial input")


def expect_inconclusive(function: object, *args: object) -> None:
    try:
        function(*args)
    except Inconclusive:
        return
    raise ContractDefect("self-test did not classify observation as inconclusive")


def self_test_build_authentication() -> None:
    execution = "a" * 40
    source_state = SOURCE_STATE_PREFIX + "b" * 64
    expected_history = {
        "authority_origin": V0637_COMMIT,
        "authorized_preregistration": PARENT,
        "sole_successor_authorization_consumed_by": "v0.638",
        "v0638_successor_authorization_imported": True,
        "v0638_successor_authorization_consumed": True,
        "v0639_additional_authorization_imported": False,
        "v0639_additional_authorization_consumed": False,
        "packet_ordinal": 1,
        "v0638_packet_reserved": False,
        "v0638_packet_imported": False,
        "v0638_artifact_imported": False,
        "v0638_work_root_imported": False,
        "product_authority_imported": False,
        "gate_results_imported": 0,
        "scored_rows_imported": 0,
        "timing_observations_imported": 0,
        "performance_observations_imported": 0,
    }
    if authority_history() != expected_history:
        raise ContractDefect("same-one-packet authority history drifted")
    mutated_history = authority_history()
    mutated_history["packet_ordinal"] = 2
    if authority_history() != expected_history:
        raise ContractDefect("authority history helper did not return fresh state")
    if RAW_BUILD_POLICY != {
        "qwen": {"requires_full_commit": True, "requires_source_state": True},
        "qwen-bench": {
            "requires_full_commit": False,
            "requires_source_state": True,
        },
    }:
        raise ContractDefect("asymmetric raw policy table drifted")
    bench_without_commit = validate_raw_build_literals(
        "qwen-bench", source_state.encode(), execution, source_state
    )
    bench_with_commit = validate_raw_build_literals(
        "qwen-bench", f"{execution}:{source_state}".encode(), execution, source_state
    )
    if (
        bench_without_commit["full_commit_present_diagnostic"] is not False
        or bench_with_commit["full_commit_present_diagnostic"] is not True
    ):
        raise ContractDefect("qwen-bench commit-presence diagnostic drifted")
    validate_raw_build_literals(
        "qwen", f"{execution}:{source_state}".encode(), execution, source_state
    )
    expect_rejected(
        validate_raw_build_literals,
        "qwen",
        source_state.encode(),
        execution,
        source_state,
    )
    expect_rejected(
        validate_raw_build_literals, "qwen", execution.encode(), execution, source_state
    )
    expect_rejected(
        validate_raw_build_literals, "qwen-bench", b"unrelated", execution, source_state
    )

    semantic = {
        "schema_version": 2,
        "build_commit": execution,
        "build_commit_short": execution[:9],
        "build_dirty": False,
        "build_source_state": source_state,
        "stamp_source": "git",
        "stamp_error": None,
        "runtime_commit": execution,
        "runtime_dirty": False,
        "runtime_source_state": source_state,
        "status": "match",
        "problems": [],
        "overrides": [],
    }
    validate_bench_semantics(semantic, execution, source_state)
    mismatches = {
        "schema_version": 1,
        "build_commit": "c" * 40,
        "build_commit_short": "deadbeef0",
        "build_dirty": True,
        "build_source_state": SOURCE_STATE_PREFIX + "c" * 64,
        "stamp_source": "environment-verified",
        "stamp_error": "failure",
        "runtime_commit": "c" * 40,
        "runtime_dirty": True,
        "runtime_source_state": SOURCE_STATE_PREFIX + "c" * 64,
        "status": "dirty",
        "problems": ["dirty"],
        "overrides": ["QWEN_BUILD_COMMIT"],
    }
    for name, mismatch in mismatches.items():
        changed = dict(semantic)
        changed[name] = mismatch
        expect_rejected(validate_bench_semantics, changed, execution, source_state)
        missing = dict(semantic)
        del missing[name]
        expect_rejected(validate_bench_semantics, missing, execution, source_state)

    identity = {
        "path": "/tmp/qwen-bench",
        "sha256": "d" * 64,
        "stamp": {
            "dev": 1,
            "ino": 2,
            "bytes": 3,
            "mode": 0o755,
            "nlink": 1,
            "mtime_ns": 4,
            "ctime_ns": 5,
        },
    }
    require_identity_equal(
        identity, json.loads(json.dumps(identity)), "stable identity"
    )
    digest_drift = json.loads(json.dumps(identity))
    digest_drift["sha256"] = "e" * 64
    expect_rejected(require_identity_equal, identity, digest_drift, "digest drift")
    path_drift = json.loads(json.dumps(identity))
    path_drift["path"] = "/tmp/replaced-qwen-bench"
    expect_rejected(require_identity_equal, identity, path_drift, "path drift")
    for name in ("dev", "ino", "bytes", "mode", "nlink", "mtime_ns", "ctime_ns"):
        changed = json.loads(json.dumps(identity))
        changed["stamp"][name] += 1
        expect_rejected(require_identity_equal, identity, changed, f"{name} drift")
    replaced = json.loads(json.dumps(identity))
    replaced["stamp"]["ino"] = 9
    expect_rejected(
        require_identity_equal, identity, replaced, "same-byte inode replacement"
    )


def self_test_telemetry() -> None:
    publication = (
        "durable_prefix_cache: store_empty=true restore_total_ms=1.0\n"
        "durable_prefix_cache: publish=published capture=completed matched_tokens=6500 "
        "restored_tokens=6499 pending=true stop_reason=token_limit blob_bytes=582854188 "
        "evicted=0 identity=hit staged_integrity=deferred-restore staged_integrity_us=1 "
        "capture_ms=2.5 publish_us=300000 post_response_us=250000\n"
        "stats: prompt_tokens=6499 generated_tokens=1 transitions=0 stop_reason=token_limit "
        "load_ms=1.0 prefill_ms=2.0 ttft_ms=3.0 decode_tps=4.0 transition_tps=0.0 "
        "cache_entries=0 cache_mib=0.0/0.0\n"
    ).encode()
    parsed = parse_scored(publication, "B", 250000)
    if parsed["publish_us"] != 300000 or parsed["external_us"] != 250000:
        raise ContractDefect("self-test telemetry result drifted")
    expect_rejected(
        parse_scored,
        publication.replace(b"publish=published", b"publish=failed"),
        "B",
        250000,
    )
    expect_rejected(parse_scored, publication + publication, "B", 250000)
    observation = parse_scored(
        publication.replace(b"post_response_us=250000", b"post_response_us=250001"),
        "B",
        250000,
    )
    expect_inconclusive(
        require_observation_quality,
        observation["external_us"],
        observation["post_response_us"],
    )
    expect_rejected(
        parse_scored,
        publication.replace(b"publish_us=300000", b"publish_us=0300000"),
        "B",
        250000,
    )
    expect_rejected(parse_scored, publication + b"\xff", "B", 250000)
    restore = (
        b"durable_prefix_cache: identity_cache=hit hashed_bytes=0 checkpoint_hit=true "
        b"matched=6500 restored=6499 exact=false candidates=1 corrupt_removed=0 "
        b"restore_total_ms=1.0\n"
    )
    if parse_restore(restore)["checkpoint_hit"] is not True:
        raise ContractDefect("self-test restore parser result drifted")
    expect_rejected(
        parse_restore, restore + b"durable_prefix_cache: publish=published\n"
    )
    synthetic = []
    for index, order in enumerate(PAIR_ORDERS, 1):
        for position, arm in enumerate(order, 1):
            synthetic.append(
                {
                    "pair_index": index,
                    "pair_order": order,
                    "position": position,
                    "arm": arm,
                    "external_us": 500000 if arm == "A" else 250000,
                    "telemetry": {"publish_us": 500000 if arm == "A" else 250000},
                }
            )
    if analyze(synthetic)["passes"] is not True:
        raise ContractDefect("self-test rejected inclusive 250000 boundary")
    synthetic[0]["external_us"] -= 1
    if analyze(synthetic)["passes"] is not False:
        raise ContractDefect("self-test accepted 249999 external boundary")
    synthetic[0]["external_us"] += 1
    synthetic[0]["telemetry"]["publish_us"] -= 1
    if analyze(synthetic)["passes"] is not False:
        raise ContractDefect("self-test accepted 249999 publish boundary")


def self_test_descriptors_and_store() -> None:
    with tempfile.TemporaryDirectory(prefix="v0639-self-test-") as temporary:
        root = Path(temporary)
        sealed = root / "sealed"
        sealed.mkdir()
        (sealed / "regular").write_bytes(b"sealed")
        regular_identity = descriptor_hash_regular(sealed / "regular")
        if regular_identity["sha256"] != sha256_bytes(b"sealed"):
            raise ContractDefect("descriptor hash self-test drifted")
        if read_regular_directory_once(sealed) != {"regular": b"sealed"}:
            raise ContractDefect("descriptor self-test regular read drifted")
        (sealed / "link").symlink_to(sealed / "regular")
        expect_rejected(read_regular_directory_once, sealed)
        (sealed / "link").unlink()
        directory_link = root / "sealed-link"
        directory_link.symlink_to(sealed, target_is_directory=True)
        expect_rejected(read_regular_directory_once, directory_link)
        broken = root / "broken-root"
        broken.symlink_to(root / "missing")
        if not path_lexists(broken) or broken.exists():
            raise ContractDefect("lexists self-test did not detect a broken symlink")
        executable = root / "executable"
        executable.write_bytes(b"binary")
        executable.chmod(0o755)
        executable_descriptor(executable)
        executable.chmod(0o700)
        expect_rejected(executable_descriptor, executable)
        seed = root / "seed.mid"
        seed.write_bytes(b"x" * IDENTITY_BYTES)
        digest = sha256_file(seed)
        store = root / "store"
        seed_store(store, seed, digest)
        verify_seeded_store(store, digest, IDENTITY_BYTES)
        if stat.S_IMODE(store.stat().st_mode) != 0o700:
            raise ContractDefect("store self-test mode drifted")


def self_test_direct_process() -> None:
    env = {"PATH": os.environ.get("PATH", "")}
    result = direct_process(
        [
            sys.executable,
            "-c",
            "import os; os.write(1, b'<think>\\n'); os.write(2, b'ok\\n')",
        ],
        env,
        expected_stdout=STDOUT_BYTES,
        cwd=ROOT,
        completion_watchdog_s=2.0,
    )
    release_process_signal_mask(result)
    normal_control = result.get("signal_control")
    if (
        result["returncode"] != 0
        or result["stdout"] != STDOUT_BYTES
        or result["stderr"] != b"ok\n"
        or not isinstance(result["t_stdout_complete_ns"], int)
        or int(result["t_reaped_ns"]) < int(result["t_stdout_complete_ns"])
        or not isinstance(result["rusage"].get("ru_nswap"), int)
        or not isinstance(normal_control, dict)
        or normal_control.get("strategy") != "sole-main-kqueue-wait4"
        or normal_control.get("error") is not None
        or normal_control.get("thread_alive") is not False
        or normal_control.get("shutdown_clean") is not True
        or operator_control_records(result)
        or process_execution_reasons(result)
    ):
        raise ContractDefect(f"direct process self-test evidence drifted: {result!r}")
    try:
        os.waitpid(int(result["pid"]), os.WNOHANG)
    except ChildProcessError:
        pass
    else:
        raise ContractDefect("direct process self-test left a waitable zombie")

    for index in range(32):
        immediate = direct_process(
            [sys.executable, "-c", "import os; os._exit(0)"],
            env,
            expected_stdout=None,
            cwd=ROOT,
            completion_watchdog_s=0.5,
        )
        immediate_release = release_process_signal_mask(immediate)
        if (
            immediate["returncode"] != 0
            or immediate["interruption"] is not None
            or immediate["signal_control"]["error"] is not None
            or immediate_release["operator_signals_before_mask_restore"]
            or immediate_release["sigalrm_contamination_before_mask_restore"]
        ):
            raise ContractDefect(
                f"immediate-exit registration stress failed at {index}: {immediate!r}"
            )
        try:
            os.waitpid(int(immediate["pid"]), os.WNOHANG)
        except ChildProcessError:
            pass
        else:
            raise ContractDefect(
                f"immediate-exit registration stress left zombie at {index}"
            )

    interruption_started = time.monotonic()
    interrupted = direct_process(
        [
            sys.executable,
            "-c",
            (
                "import os,signal,time; "
                "os.kill(os.getppid(), signal.SIGINT); time.sleep(30)"
            ),
        ],
        env,
        expected_stdout=None,
        cwd=ROOT,
        completion_watchdog_s=2.0,
    )
    interruption_wall_s = time.monotonic() - interruption_started
    release_process_signal_mask(interrupted)
    control = interrupted.get("signal_control")
    records = control.get("records") if isinstance(control, dict) else None
    if (
        interrupted["interruption"] is None
        or interrupted["returncode"] != -int(signal.SIGTERM)
        or interruption_wall_s >= 5.0
        or not isinstance(control, dict)
        or control.get("error") is not None
        or control.get("thread_alive") is not False
        or control.get("shutdown_clean") is not True
        or not isinstance(records, list)
        or [record.get("action") for record in records[:2]] != ["received", "terminate"]
        or records[0].get("signal") != int(signal.SIGINT)
        or records[1].get("signal") != int(signal.SIGTERM)
        or records[1].get("outcome") != "sent"
    ):
        raise ContractDefect(
            f"interruption self-test control evidence drifted: {interrupted!r}"
        )
    try:
        os.waitpid(int(interrupted["pid"]), os.WNOHANG)
    except ChildProcessError:
        pass
    else:
        raise ContractDefect("interruption self-test left a waitable zombie")

    kill_started = time.monotonic()
    killed = direct_process(
        [
            sys.executable,
            "-c",
            (
                "import os,signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                "os.kill(os.getppid(), signal.SIGINT); time.sleep(30)"
            ),
        ],
        env,
        expected_stdout=None,
        cwd=ROOT,
        term_grace_s=0.1,
        kill_grace_s=0.5,
        completion_watchdog_s=2.0,
    )
    kill_wall_s = time.monotonic() - kill_started
    kill_release = release_process_signal_mask(killed)
    kill_actions = [
        record.get("action") for record in killed["signal_control"]["records"]
    ]
    if (
        killed["returncode"] != -int(signal.SIGKILL)
        or kill_wall_s >= 3.0
        or kill_actions[:3] != ["received", "terminate", "kill"]
        or killed["signal_control"]["error"] is not None
        or kill_release["sigalrm_contamination_before_mask_restore"] is not False
    ):
        raise ContractDefect(f"SIGKILL escalation self-test drifted: {killed!r}")
    try:
        os.waitpid(int(killed["pid"]), os.WNOHANG)
    except ChildProcessError:
        pass
    else:
        raise ContractDefect("SIGKILL escalation self-test left a zombie")

    contaminated = direct_process(
        [
            sys.executable,
            "-c",
            "import os,signal,time; os.kill(os.getppid(), signal.SIGALRM); time.sleep(30)",
        ],
        env,
        expected_stdout=None,
        cwd=ROOT,
        term_grace_s=0.1,
        kill_grace_s=0.5,
        completion_watchdog_s=2.0,
    )
    contamination_release = release_process_signal_mask(contaminated)
    contamination_actions = [
        record.get("action") for record in contaminated["signal_control"]["records"]
    ]
    if (
        contaminated["returncode"] != -int(signal.SIGTERM)
        or contaminated["signal_control"]["error"] != "unexpected SIGALRM contamination"
        or contamination_actions[:3] != ["received", "contamination", "terminate"]
        or contamination_release["sigalrm_contamination_before_mask_restore"]
        is not False
    ):
        raise ContractDefect(
            f"SIGALRM contamination self-test drifted: {contaminated!r}"
        )
    try:
        os.waitpid(int(contaminated["pid"]), os.WNOHANG)
    except ChildProcessError:
        pass
    else:
        raise ContractDefect("SIGALRM contamination self-test left a zombie")


def self_test_core() -> None:
    roots = (ARTIFACT, WORK_ROOT, V0638_ARTIFACT, V0638_WORK_ROOT)
    if any(path_lexists(path) for path in roots):
        raise ContractDefect("self-test requires absent v0.638 and v0.639 packet roots")
    self_test_build_authentication()
    self_test_telemetry()
    self_test_descriptors_and_store()
    self_test_direct_process()
    if any(path_lexists(path) for path in roots):
        raise ContractDefect("self-test created or imported a packet root")


def self_test() -> None:
    self_test_core()
    print(
        json_bytes(
            {
                "status": "self-test-pass",
                "optimized_mode_safe": True,
            }
        ).decode(),
        end="",
    )


def handle_signal(signum: int, _frame: object) -> None:
    raise KeyboardInterrupt(f"received operator signal {signum}")


def handle_sigchld(_signum: int, _frame: object) -> None:
    return


if __name__ == "__main__":
    previous_umask = os.umask(0o077)
    if previous_umask < 0:
        raise RuntimeError("could not establish private umask")
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGCHLD, handle_sigchld)
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--self-test", action="store_true")
    group.add_argument("--preflight-only", action="store_true")
    arguments = parser.parse_args()
    if arguments.self_test:
        self_test()
    else:
        run(preflight_only=arguments.preflight_only)
