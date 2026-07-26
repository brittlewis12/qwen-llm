#!/usr/bin/env python3
"""v0.636 CPU-only deferred-restore checkpoint publication floor."""

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
import time


ROOT = Path(__file__).resolve().parents[2]
RUNNER = Path(__file__).resolve()
PREREG = ROOT / "docs/bench/v0636-checkpoint-deferred-restore-floor.md"
STORE_SOURCE = ROOT / "crates/qwen-llm/src/checkpoint_store.rs"
CLI_SOURCE = ROOT / "crates/qwen-cli/src/main.rs"
ARTIFACT = ROOT / "target/profiles/v0636-checkpoint-deferred-restore-floor-p1"
WORK_ROOT = ROOT / "target/profiles/v0636-checkpoint-deferred-restore-floor-work"
FIXTURE = (
    ROOT
    / "target/profiles/v0615-long-history/cache-ring0/v1/blobs"
    / "fe9d70c202425683190d1c6a3ef50474940915e8c2e918f3ad50574e12e92b4e"
    / "6500-p0-63362870ab8f272dfc0c5a5f1dbbe474921b1554a4f589e728d5a2cfbb1bbc2a.qcp"
)

BASE_COMMIT = "b6a212ab78049607fae9ba9b0ccc3535e3fa7254"
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
MARKER = re.compile(
    r"\[checkpoint-deferred-floor\] schema=1 "
    r"mode=(decode|deferred-restore) record_bytes=(\d+) "
    r"encoder_blake3=([0-9a-f]{64}) blob_sha256=([0-9a-f]{64}) "
    r"staged_integrity_us=(\d+) publish_us=(\d+) full_decode_us=(\d+) "
    r"outcome=(\S+) evicted=(\d+) managed_bytes_after=(\d+) "
    r"vocab_size=(\d+) max_context=(\d+) max_record_bytes=(\d+) "
    r"store_budget_bytes=(\d+) "
    r"matched=(\d+) restored=(\d+) "
    r"pending=(true|false) file_mode=([0-7]{4}) nlink=(\d+) "
    r"temp_files=(\d+) blob_relative=(\S+)"
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


def verify_source() -> dict[str, object]:
    for path in (RUNNER, PREREG, STORE_SOURCE, CLI_SOURCE):
        git(["ls-files", "--error-unmatch", str(path.relative_to(ROOT))])
    dirty = git(["status", "--porcelain=v1", "--untracked-files=all"])
    if dirty:
        raise ContractDefect(f"worktree is dirty: {dirty!r}")

    implementation = git(["rev-parse", "HEAD"])
    preregistration = parent(implementation)
    base = parent(preregistration)
    if base != BASE_COMMIT:
        raise ContractDefect(f"v0.636 base drifted: {base}")

    prereg_expected = sorted(
        [
            f"A\t{PREREG.relative_to(ROOT)}",
            f"A\t{RUNNER.relative_to(ROOT)}",
        ]
    )
    if sorted(changed_paths(base, preregistration)) != prereg_expected:
        raise ContractDefect("R must add exactly the preregistration and runner")
    implementation_expected = sorted(
        [
            f"M\t{CLI_SOURCE.relative_to(ROOT)}",
            f"M\t{STORE_SOURCE.relative_to(ROOT)}",
        ]
    )
    if (
        sorted(changed_paths(preregistration, implementation))
        != implementation_expected
    ):
        raise ContractDefect("H must modify exactly main.rs and checkpoint_store.rs")

    if not FIXTURE.is_file():
        raise ContractDefect(f"frozen v0.615 checkpoint fixture is missing: {FIXTURE}")
    if (
        FIXTURE.stat().st_size != EXPECTED_RECORD_BYTES
        or sha256_file(FIXTURE) != EXPECTED_BLOB_SHA256
    ):
        raise ContractDefect("frozen v0.615 checkpoint fixture drifted")

    paths = (RUNNER, PREREG, STORE_SOURCE, CLI_SOURCE)
    return {
        "base_commit": base,
        "preregistration_commit": preregistration,
        "implementation_commit": implementation,
        "preregistration_patch_sha256": patch_sha256(
            base,
            preregistration,
            (RUNNER, PREREG),
        ),
        "implementation_patch_sha256": patch_sha256(
            preregistration,
            implementation,
            (STORE_SOURCE, CLI_SOURCE),
        ),
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
        raise ContractDefect("normalized environment retained v0.636 controls")
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


def parse_marker(output: str, expected_mode: str) -> dict[str, object]:
    matches = [MARKER.fullmatch(line) for line in output.splitlines()]
    matches = [match for match in matches if match is not None]
    if len(matches) != 1:
        raise ContractDefect(f"expected one floor marker, found {len(matches)}")
    match = matches[0]
    (
        mode,
        record_bytes,
        encoder_blake3,
        blob_sha256,
        integrity_us,
        publish_us,
        full_decode_us,
        outcome,
        evicted,
        managed_bytes_after,
        vocab_size,
        max_context,
        max_record_bytes,
        store_budget_bytes,
        matched,
        restored,
        pending,
        file_mode,
        nlink,
        temp_files,
        blob_relative,
    ) = match.groups()
    timings = [int(integrity_us), int(publish_us), int(full_decode_us)]
    if (
        mode != expected_mode
        or int(record_bytes) != EXPECTED_RECORD_BYTES
        or encoder_blake3 != EXPECTED_ENCODER_BLAKE3
        or blob_sha256 != EXPECTED_BLOB_SHA256
        or outcome != "published"
        or int(evicted) != 0
        or int(managed_bytes_after) != EXPECTED_RECORD_BYTES
        or int(vocab_size) != EXPECTED_VOCAB_SIZE
        or int(max_context) != EXPECTED_MAX_CONTEXT
        or int(max_record_bytes) != EXPECTED_MAX_RECORD_BYTES
        or int(store_budget_bytes) != EXPECTED_STORE_BUDGET_BYTES
        or int(matched) != 6500
        or int(restored) != 6499
        or pending != "true"
        or file_mode != "0600"
        or int(nlink) != 1
        or int(temp_files) != 0
        or blob_relative != EXPECTED_BLOB_RELATIVE
    ):
        raise ContractDefect("floor marker contract drifted")
    return {
        "mode": mode,
        "record_bytes": int(record_bytes),
        "encoder_blake3": encoder_blake3,
        "blob_sha256": blob_sha256,
        "staged_integrity_us": timings[0],
        "publish_us": timings[1],
        "full_decode_us": timings[2],
        "outcome": outcome,
        "evicted": int(evicted),
        "managed_bytes_after": int(managed_bytes_after),
        "vocab_size": int(vocab_size),
        "max_context": int(max_context),
        "max_record_bytes": int(max_record_bytes),
        "store_budget_bytes": int(store_budget_bytes),
        "matched": int(matched),
        "restored": int(restored),
        "pending": True,
        "file_mode": file_mode,
        "nlink": int(nlink),
        "temp_files": int(temp_files),
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
    if verify_source() != expected_source:
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
    combined = (
        stdout_bytes.decode(errors="replace")
        + "\n"
        + stderr_bytes.decode(errors="replace")
    )
    if returncode < 0:
        raise IncompleteRun(f"floor child terminated by signal: {stem}: {returncode}")
    if returncode != 0:
        raise ContractDefect(f"floor child failed: {stem}: {returncode}")
    if pending:
        raise IncompleteRun(f"operator signal after floor child: {stem}")
    signal.pthread_sigmask(signal.SIG_SETMASK, prior_mask)
    marker = parse_marker(combined, arm_mode(arm))
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
        raise ContractDefect("refusing to reuse v0.636 artifact or work root")
    if not ARTIFACT.parent.is_dir():
        raise ContractDefect(f"artifact parent is missing: {ARTIFACT.parent}")
    ARTIFACT.mkdir()
    fsync_directory(ARTIFACT.parent)
    write_json(ARTIFACT / "manifest.json", manifest)


def run(*, preflight_only: bool) -> None:
    source = verify_source()
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
        if verify_source() != source:
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
        if verify_source() != source:
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
        "source_commit": source["implementation_commit"],
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
        raise ContractDefect(error or "unknown v0.636 contract defect")


def handle_operator_signal(signum: int, _frame: object) -> None:
    raise IncompleteRun(f"received signal {signum}")


def self_test() -> None:
    sample = (
        "[checkpoint-deferred-floor] schema=1 mode=deferred-restore "
        "record_bytes=582854188 encoder_blake3="
        + EXPECTED_ENCODER_BLAKE3
        + " blob_sha256="
        + EXPECTED_BLOB_SHA256
        + " staged_integrity_us=100 publish_us=300000 full_decode_us=250000 "
        "outcome=published evicted=0 managed_bytes_after=582854188 "
        "vocab_size=248320 max_context=6516 max_record_bytes=805306368 "
        "store_budget_bytes=805306368 "
        "matched=6500 restored=6499 pending=true file_mode=0600 nlink=1 "
        "temp_files=0 blob_relative=" + EXPECTED_BLOB_RELATIVE
    )
    parsed = parse_marker(sample, "deferred-restore")
    assert parsed["record_bytes"] == EXPECTED_RECORD_BYTES
    assert parsed["pending"] is True
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
    assert analyze(rows)["passes"] is True
    rows[0]["marker"]["publish_us"] -= 1
    assert analyze(rows)["passes"] is False
    print(json_bytes({"status": "self-test-pass"}).decode())


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
