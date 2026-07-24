#!/usr/bin/env python3

"""v0.622 allocation-free checkpoint staged-validation packet."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import stat
import statistics
import subprocess
import time

import v0602_a3b_parallel_copied_loader as protocol


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = ROOT / "target/profiles/v0622-checkpoint-staged-validation-p1"
WORK_ROOT = ROOT / "target/profiles/v0622-checkpoint-staged-validation-work"
PREREG = ROOT / "docs/bench/v0622-checkpoint-staged-validation.md"
BASE_PROTOCOL = ROOT / "scripts/profile/v0602_a3b_parallel_copied_loader.py"
COMMON_RUNNER = ROOT / "scripts/profile/v0593_demand_paged_no_copy.py"
MODEL = Path("/Users/tito/models/Qwen3.6-27B-Q4_K_M.gguf")
MESSAGES = ROOT / "target/profiles/v0615-long-history/ring0-turn-2.json"
IDENTITY_SEED = (
    ROOT / "target/profiles/v0615-long-history/cache-ring0/v1/identity/"
    "3f6fb8c12c7fbfe881e2c43b7d98742873161ffa51a5e039773252207541734e.mid"
)
PACKET_MESSAGES = ARTIFACT / "input-messages.json"
PACKET_IDENTITY_SEED = ARTIFACT / "input-identity.mid"
CLI_BINARY = ROOT / "target/release/qwen"
BENCH_BINARY = ROOT / "target/release/qwen-bench"
PRIMITIVE_COMMIT = "b9362e6c2c6aa3f2530ce565ae100a4f6552c99a"
INTEGRATION_COMMIT = "a0aec06fea54c6be9484f3a802b29775d82166fe"
EXPECTED_MODEL_SHA256 = (
    "5ed60d0af4650a854b1755bd392f9aef4872643dc25a254bc68043fa638392a0"
)
EXPECTED_MESSAGES_SHA256 = (
    "5c2455738af1d789ce1f86cf1d2064fb8e80a242c80273273f57c25783bdd4a1"
)
EXPECTED_IDENTITY_SHA256 = (
    "17afb5fc1e9c3e56b7ffd2612f501d1714eefbb3c1e81d27e9501e0a18120c69"
)
EXPECTED_BLOB_SHA256 = (
    "69c883f5130e5108cc3b948c5cc500c4d92db1fc2f96aa25b75f5f38abda1e65"
)
EXPECTED_STDOUT_SHA256 = (
    "9ebc01769b176bb074a065ea0974c130fc8afd12814360aaf809046160b2a999"
)
EXPECTED_BLOB_BYTES = 582_854_188
COMPATIBILITY_DIR = "fe9d70c202425683190d1c6a3ef50474940915e8c2e918f3ad50574e12e92b4e"
BLOB_NAME = (
    "6500-p0-63362870ab8f272dfc0c5a5f1dbbe474921b1554a4f589e728d5a2cfbb1bbc2a.qcp"
)
EXPECTED_BLOB_RELATIVE = Path("v1/blobs") / COMPATIBILITY_DIR / BLOB_NAME
PAIR_ORDERS = (("AB", "A", "B"), ("BA", "B", "A"))
VALIDATION_ENV = "QWEN_CHECKPOINT_STAGED_VALIDATION"
WALL_GATE_MS = 250.0
FOOTPRINT_GATE_BYTES = 500_000_000


def json_text(value: object, *, pretty: bool = False) -> str:
    return protocol.json_text(value, pretty=pretty)


def required_manifest_paths() -> tuple[Path, ...]:
    return (
        Path(__file__).resolve(),
        PREREG,
        BASE_PROTOCOL,
        COMMON_RUNNER,
        MODEL,
        MESSAGES,
        IDENTITY_SEED,
        CLI_BINARY,
        BENCH_BINARY,
    )


def source_and_build_identity() -> tuple[str, dict[str, object]]:
    tracked = (Path(__file__).resolve(), PREREG, BASE_PROTOCOL, COMMON_RUNNER)
    for path in tracked:
        protocol.command_text(
            ["git", "ls-files", "--error-unmatch", str(path.relative_to(ROOT))]
        )
    commit = protocol.command_text(["git", "rev-parse", "HEAD"]).strip()
    dirty = protocol.command_text(["git", "status", "--porcelain=v1"]).strip()
    if dirty:
        raise RuntimeError(f"source is dirty: {dirty!r}")
    packet_parent = protocol.command_text(["git", "rev-parse", "HEAD^"]).strip()
    integration_parent = protocol.command_text(
        ["git", "rev-parse", f"{INTEGRATION_COMMIT}^"]
    ).strip()
    integration_paths = set(
        protocol.command_text(
            [
                "git",
                "diff-tree",
                "--no-commit-id",
                "--name-only",
                "-r",
                INTEGRATION_COMMIT,
            ]
        ).splitlines()
    )
    if (
        packet_parent != INTEGRATION_COMMIT
        or integration_parent != PRIMITIVE_COMMIT
        or integration_paths
        != {
            "crates/qwen-cli/src/main.rs",
            "crates/qwen-llm/src/checkpoint_store.rs",
        }
    ):
        raise RuntimeError("staged-validation implementation bridge drifted")
    changed = set(
        protocol.command_text(
            ["git", "diff", "--name-only", f"{INTEGRATION_COMMIT}..{commit}"]
        ).splitlines()
    )
    expected_changed = {
        str(PREREG.relative_to(ROOT)),
        str(Path(__file__).resolve().relative_to(ROOT)),
    }
    if changed != expected_changed:
        raise RuntimeError(f"packet source delta drifted: {sorted(changed)}")
    build = protocol.parse_json(
        protocol.command_text([str(BENCH_BINARY), "build-info", "--output", "json"])
    )
    if not isinstance(build, dict) or (
        build.get("build_commit") != commit
        or build.get("runtime_commit") != commit
        or build.get("status") != "match"
        or build.get("build_dirty") is not False
        or build.get("runtime_dirty") is not False
        or build.get("build_source_state") != build.get("runtime_source_state")
    ):
        raise RuntimeError(f"source/build identity mismatch: {build}")
    binary = CLI_BINARY.read_bytes()
    for value in (commit, str(build["build_source_state"])):
        if value.encode() not in binary:
            raise RuntimeError("qwen identity is not embedded")
    return commit, build


def build_manifest(
    removed_environment: list[str], base_env: dict[str, str]
) -> dict[str, object]:
    commit, build = source_and_build_identity()
    paths = required_manifest_paths()
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError(f"missing packet inputs: {missing}")
    hashes = {str(path): protocol.common.sha256_file(path) for path in paths}
    expected = {
        str(MODEL): EXPECTED_MODEL_SHA256,
        str(MESSAGES): EXPECTED_MESSAGES_SHA256,
        str(IDENTITY_SEED): EXPECTED_IDENTITY_SHA256,
    }
    for path, digest in expected.items():
        if hashes[path] != digest:
            raise RuntimeError(f"frozen input digest drifted: {path}")
    if VALIDATION_ENV in base_env:
        raise RuntimeError("normalized child environment retained validation control")
    device = protocol.command_text([str(CLI_BINARY), "--info"], env=base_env).strip()
    macos = protocol.command_text(["sw_vers", "-productVersion"], env=base_env).strip()
    hw_memsize = int(
        protocol.command_text(["sysctl", "-n", "hw.memsize"], env=base_env)
    )
    if (
        device != protocol.EXPECTED_DEVICE
        or not macos.startswith("15.")
        or hw_memsize != protocol.EXPECTED_HW_MEMSIZE
    ):
        raise RuntimeError("host boundary drifted")
    return {
        "schema": 1,
        "created_unix_ms": time.time_ns() // 1_000_000,
        "source_commit": commit,
        "build_identity": build,
        "device": device,
        "macos": macos,
        "hw_memsize": hw_memsize,
        "removed_environment": removed_environment,
        "child_environment": protocol.child_environment_record(base_env),
        "time_resource_probe": protocol.preflight_time_resources(base_env),
        "sha256": hashes,
        "model_size_bytes": MODEL.stat().st_size,
        "messages_size_bytes": MESSAGES.stat().st_size,
        "identity_seed_size_bytes": IDENTITY_SEED.stat().st_size,
        "pair_orders": [list(order) for order in PAIR_ORDERS],
        "expected_blob_relative": str(EXPECTED_BLOB_RELATIVE),
        "expected_blob_bytes": EXPECTED_BLOB_BYTES,
        "expected_blob_sha256": EXPECTED_BLOB_SHA256,
        "expected_stdout_sha256": EXPECTED_STDOUT_SHA256,
        "wall_gate_ms": WALL_GATE_MS,
        "footprint_gate_bytes": FOOTPRINT_GATE_BYTES,
        "cooldown_s": protocol.COOLDOWN_S,
        "host_sample_limit": protocol.HOST_SAMPLE_LIMIT,
        "host_sample_interval_s": protocol.HOST_SAMPLE_INTERVAL_S,
        "child_retry_count": 0,
    }


def freeze_packet_inputs() -> None:
    frozen = (
        (MESSAGES, PACKET_MESSAGES, EXPECTED_MESSAGES_SHA256),
        (IDENTITY_SEED, PACKET_IDENTITY_SEED, EXPECTED_IDENTITY_SHA256),
    )
    rows = []
    for source, target, expected in frozen:
        with (
            source.open("rb", buffering=0) as left,
            target.open("xb", buffering=0) as right,
        ):
            shutil.copyfileobj(left, right, length=1024 * 1024)
            right.flush()
            os.fsync(right.fileno())
        target.chmod(0o600)
        digest = protocol.common.sha256_file(target)
        if digest != expected:
            raise RuntimeError(f"packet-local input drifted: {target.name}")
        rows.append(
            {
                "source": str(source),
                "packet_path": str(target.relative_to(ROOT)),
                "bytes": target.stat().st_size,
                "sha256": digest,
            }
        )
    seal_path = ARTIFACT / "input-seal.json"
    with seal_path.open("x", encoding="utf-8") as output:
        output.write(json_text({"schema": 1, "inputs": rows}, pretty=True) + "\n")
        output.flush()
        os.fsync(output.fileno())
    fsync_directory(ARTIFACT)


def verify_non_model_identity(manifest: dict[str, object]) -> None:
    commit, build = source_and_build_identity()
    if commit != manifest["source_commit"] or build != manifest["build_identity"]:
        raise RuntimeError("packet source/build identity drifted")
    hashes = {
        str(path): protocol.common.sha256_file(path)
        for path in required_manifest_paths()
        if path != MODEL
    }
    expected = {
        path: digest
        for path, digest in manifest["sha256"].items()
        if path != str(MODEL)
    }
    if hashes != expected:
        raise RuntimeError("packet non-model input hashes drifted")
    packet_inputs = {
        PACKET_MESSAGES: EXPECTED_MESSAGES_SHA256,
        PACKET_IDENTITY_SEED: EXPECTED_IDENTITY_SHA256,
    }
    for path, digest in packet_inputs.items():
        if protocol.common.sha256_file(path) != digest:
            raise RuntimeError(f"packet-local input hash drifted: {path.name}")


def arm_mode(arm: str) -> str:
    if arm == "A":
        return "decode"
    if arm == "B":
        return "encoder-digest"
    raise ValueError(f"unknown arm {arm!r}")


def child_command(store_root: Path, messages: Path | None = None) -> list[str]:
    messages = PACKET_MESSAGES if messages is None else messages
    return [
        "/usr/bin/time",
        "-l",
        str(CLI_BINARY),
        "--model",
        str(MODEL),
        "--messages",
        str(messages),
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
        "6516",
        "--durable-prefix-cache",
        str(store_root),
        "--durable-prefix-cache-max-mib",
        "768",
        "--durable-prefix-cache-max-entry-mib",
        "768",
        "--durable-prefix-cache-min-tokens",
        "1024",
    ]


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def prepare_store(root: Path) -> dict[str, object]:
    if root.exists():
        raise RuntimeError(f"refusing to reuse store root {root}")
    identity_dir = root / "v1/identity"
    identity_dir.mkdir(parents=True)
    target = identity_dir / IDENTITY_SEED.name
    with PACKET_IDENTITY_SEED.open("rb") as source, target.open("xb") as output:
        shutil.copyfileobj(source, output)
        output.flush()
        os.fsync(output.fileno())
    target.chmod(0o600)
    fsync_directory(identity_dir)
    fsync_directory(identity_dir.parent)
    files = [path for path in root.rglob("*") if path.is_file()]
    if files != [target] or any(path.is_symlink() for path in root.rglob("*")):
        raise RuntimeError("fresh store contains unexpected entries")
    digest = protocol.common.sha256_file(target)
    if digest != EXPECTED_IDENTITY_SHA256:
        raise RuntimeError("copied identity seed drifted")
    return {
        "root": str(root),
        "identity_relative": str(target.relative_to(root)),
        "identity_sha256": digest,
        "blob_count": 0,
    }


def parse_publication(stderr: str, arm: str) -> dict[str, object]:
    publication_lines = [
        line
        for line in stderr.splitlines()
        if line.startswith("durable_prefix_cache: publish=")
    ]
    store_empty_lines = [
        line
        for line in stderr.splitlines()
        if line.startswith("durable_prefix_cache: store_empty=")
    ]
    stats_lines = [line for line in stderr.splitlines() if line.startswith("stats:")]
    if (
        len(publication_lines) != 1
        or len(store_empty_lines) != 1
        or len(stats_lines) != 1
    ):
        raise RuntimeError("publication/store-empty/stats marker count drifted")
    pattern = re.compile(
        r"durable_prefix_cache: publish=(\S+) capture=(\S+) "
        r"matched_tokens=(\d+) restored_tokens=(\d+) pending=(\S+) "
        r"stop_reason=(\S+) blob_bytes=(\d+) evicted=(\d+) identity=(\S+) "
        r"staged_validation=(\S+) staged_validation_ms=([0-9.]+) "
        r"capture_ms=([0-9.]+) publish_ms=([0-9.]+)"
    )
    match = pattern.fullmatch(publication_lines[0])
    if match is None:
        raise RuntimeError("publication marker shape drifted")
    (
        outcome,
        capture,
        matched,
        restored,
        pending,
        stop_reason,
        blob_bytes,
        evicted,
        identity,
        validation,
        validation_ms,
        capture_ms,
        publish_ms,
    ) = match.groups()
    floats = [float(validation_ms), float(capture_ms), float(publish_ms)]
    if not all(math.isfinite(value) and value >= 0 for value in floats):
        raise RuntimeError("publication timing is invalid")
    if floats[2] <= 0:
        raise RuntimeError("publication total wall is nonpositive")
    expected_mode = arm_mode(arm)
    if (
        outcome != "published"
        or capture != "completed"
        or int(matched) != 6500
        or int(restored) != 6499
        or pending != "true"
        or stop_reason != "token_limit"
        or int(blob_bytes) != EXPECTED_BLOB_BYTES
        or int(evicted) != 0
        or identity != "hit"
        or validation != expected_mode
    ):
        raise RuntimeError("publication contract drifted")
    store_empty = re.fullmatch(
        r"durable_prefix_cache: store_empty=true restore_total_ms=([0-9.]+)",
        store_empty_lines[0],
    )
    if store_empty is None:
        raise RuntimeError("store-empty marker drifted")
    store_empty_ms = float(store_empty.group(1))
    if not math.isfinite(store_empty_ms) or store_empty_ms < 0:
        raise RuntimeError("store-empty timing is invalid")
    if "warning: durable prefix" in stderr or "corrupt_removed=" in stderr:
        raise RuntimeError("unexpected checkpoint warning or repair marker")
    stats = re.fullmatch(
        r"stats: prompt_tokens=(\d+) generated_tokens=(\d+) transitions=(\d+) "
        r"load_ms=([0-9.]+) prefill_ms=([0-9.]+) ttft_ms=([0-9.]+) "
        r"decode_tps=([0-9.]+) transition_tps=([0-9.]+) "
        r"cache_entries=(\d+) cache_mib=([0-9.]+)/([0-9.]+)",
        stats_lines[0],
    )
    if stats is None or stats.groups()[:3] != ("6499", "1", "0"):
        raise RuntimeError("request stats contract drifted")
    numeric_stats = [float(value) for value in stats.groups()[3:8]]
    numeric_stats.extend(float(value) for value in stats.groups()[9:11])
    if not all(math.isfinite(value) and value >= 0 for value in numeric_stats):
        raise RuntimeError("request stats contain invalid values")
    return {
        "outcome": outcome,
        "capture": capture,
        "matched_tokens": int(matched),
        "restored_tokens": int(restored),
        "pending": True,
        "stop_reason": stop_reason,
        "blob_bytes": int(blob_bytes),
        "evicted_entries": int(evicted),
        "identity": identity,
        "staged_validation": validation,
        "staged_validation_ms": floats[0],
        "capture_ms": floats[1],
        "publish_ms": floats[2],
        "store_empty_ms": store_empty_ms,
    }


def inspect_store(root: Path) -> dict[str, object]:
    blob = root / EXPECTED_BLOB_RELATIVE
    files = [path for path in root.rglob("*") if path.is_file()]
    expected_files = {
        root / "v1/identity" / IDENTITY_SEED.name,
        root / "v1/store.lock",
        blob,
    }
    qcp = [path for path in files if path.suffix == ".qcp"]
    temporary = [path for path in files if path.name.startswith(".tmp-")]
    if (
        set(files) != expected_files
        or qcp != [blob]
        or temporary
        or not blob.is_file()
        or blob.is_symlink()
        or any(path.is_symlink() for path in root.rglob("*"))
    ):
        raise RuntimeError("published store topology drifted")
    metadata = blob.stat()
    digest = protocol.common.sha256_file(blob)
    if (
        metadata.st_size != EXPECTED_BLOB_BYTES
        or stat.S_IMODE(metadata.st_mode) != 0o600
        or metadata.st_nlink != 1
        or digest != EXPECTED_BLOB_SHA256
    ):
        raise RuntimeError("published blob identity drifted")
    return {
        "blob_path": str(blob),
        "blob_relative": str(EXPECTED_BLOB_RELATIVE),
        "blob_bytes": metadata.st_size,
        "blob_sha256": digest,
        "blob_mode": oct(stat.S_IMODE(metadata.st_mode)),
        "blob_nlink": metadata.st_nlink,
        "file_count": len(files),
    }


def stream_equal(left: Path, right: Path) -> bool:
    if left.stat().st_size != right.stat().st_size:
        return False
    with left.open("rb", buffering=0) as a, right.open("rb", buffering=0) as b:
        while True:
            left_chunk = a.read(8 * 1024 * 1024)
            right_chunk = b.read(8 * 1024 * 1024)
            if left_chunk != right_chunk:
                return False
            if not left_chunk:
                return True


def launch_child(
    arm: str,
    pair_index: int,
    order: str,
    position: int,
    base_env: dict[str, str],
    manifest: dict[str, object],
) -> dict[str, object]:
    stem = f"publish-p{pair_index:02d}-{order.lower()}-r{position}-{arm.lower()}"
    stdout_path = ARTIFACT / f"{stem}.out"
    stderr_path = ARTIFACT / f"{stem}.err"
    store_root = WORK_ROOT / stem
    store_before = prepare_store(store_root)
    conditioning = protocol.condition_for_child(stem, manifest)
    env = base_env.copy()
    env[VALIDATION_ENV] = arm_mode(arm)
    command = child_command(store_root)
    protocol.record_launch("publish", stem, command, arm, pair_index, position)
    process = None
    wait_errors: list[str] = []
    started = time.perf_counter()
    try:
        with (
            stdout_path.open("xb") as stdout_file,
            stderr_path.open("xb") as stderr_file,
        ):
            prior_sigint = signal.signal(signal.SIGINT, signal.SIG_IGN)
            try:
                process = subprocess.Popen(
                    command,
                    cwd=ROOT,
                    env=env,
                    stdout=stdout_file,
                    stderr=stderr_file,
                )
                returncode, wait_errors = protocol.wait_for_child(process)
                process_wall_ms = (time.perf_counter() - started) * 1e3
            finally:
                signal.signal(signal.SIGINT, prior_sigint)
    except (OSError, KeyboardInterrupt) as error:
        error_text = f"{type(error).__name__}:{error}"
        if process is None:
            protocol.record_completion("publish", stem, None, error_text)
            raise protocol.InconclusivePacket("publish", stem, [error_text]) from error
        returncode, deferred = protocol.wait_for_child(process)
        process_wall_ms = (time.perf_counter() - started) * 1e3
        wait_errors.extend(deferred)
        wait_errors.append(error_text)
    protocol.record_completion(
        "publish",
        stem,
        returncode,
        ";".join(wait_errors) if wait_errors else None,
    )
    post_exit = protocol.capture_post_exit_state(conditioning)
    protocol.record_post_exit_state("publish", stem, returncode, post_exit)
    stdout = stdout_path.read_bytes()
    stderr = stderr_path.read_text(encoding="utf-8")
    validity, reasons = protocol.finish_child_validity(
        stderr, post_exit, gate_major_faults=True
    )
    if wait_errors:
        reasons.append(f"child_wait_interrupted={';'.join(wait_errors)}")
    if returncode != 0:
        reasons.append(f"child_nonzero_exit={returncode}")
    if hashlib.sha256(stdout).hexdigest() != EXPECTED_STDOUT_SHA256:
        reasons.append("generated_stdout_digest_drifted")
    publication = parse_publication(stderr, arm) if returncode == 0 else None
    store_after = inspect_store(store_root) if returncode == 0 else None
    row = {
        "stage": "publish",
        "artifact_stem": stem,
        "arm": arm,
        "arm_mode": arm_mode(arm),
        "pair_index": pair_index,
        "pair_order": order,
        "position": position,
        "command": command,
        "arm_environment": {VALIDATION_ENV: arm_mode(arm)},
        "store_before": store_before,
        "store_after": store_after,
        **conditioning,
        **validity,
        "process_wall_ms": process_wall_ms,
        "stdout_sha256": hashlib.sha256(stdout).hexdigest(),
        "stderr_sha256": hashlib.sha256(stderr.encode()).hexdigest(),
        "publication": publication,
        "valid": not reasons,
        "validity_reasons": reasons,
    }
    protocol.record_post_exit_evidence("publish", stem, returncode, validity, reasons)
    return row


def run_pairs(
    base_env: dict[str, str],
    manifest: dict[str, object],
    attempts_path: Path,
) -> list[dict[str, object]]:
    rows = []
    for pair_index, (order, first, second) in enumerate(PAIR_ORDERS, 1):
        pair = []
        for position, arm in enumerate((first, second), 1):
            row = launch_child(arm, pair_index, order, position, base_env, manifest)
            protocol.append_row(attempts_path, row)
            rows.append(row)
            pair.append(row)
            if not row["valid"]:
                raise protocol.InconclusivePacket(
                    "publish", row["artifact_stem"], list(row["validity_reasons"])
                )
        left = Path(pair[0]["store_after"]["blob_path"])
        right = Path(pair[1]["store_after"]["blob_path"])
        if not stream_equal(left, right):
            raise RuntimeError(f"pair {pair_index} blobs differ")
    if {row["stdout_sha256"] for row in rows} != {EXPECTED_STDOUT_SHA256}:
        raise RuntimeError("global stdout identity drifted")
    if {row["store_after"]["blob_sha256"] for row in rows} != {EXPECTED_BLOB_SHA256}:
        raise RuntimeError("global blob identity drifted")
    return rows


def analyze(rows: list[dict[str, object]]) -> dict[str, object]:
    if len(rows) != 4:
        raise RuntimeError("scored child count drifted")
    pairs = []
    for pair_index, (order, first, second) in enumerate(PAIR_ORDERS, 1):
        pair = [row for row in rows if row["pair_index"] == pair_index]
        if len(pair) != 2 or [row["arm"] for row in pair] != [first, second]:
            raise RuntimeError("pair membership drifted")
        by_arm = {row["arm"]: row for row in pair}
        baseline = by_arm["A"]
        candidate = by_arm["B"]
        wall_saving_ms = baseline["process_wall_ms"] - candidate["process_wall_ms"]
        footprint_saving_bytes = (
            baseline["process_resources"]["peak_memory_footprint"]
            - candidate["process_resources"]["peak_memory_footprint"]
        )
        validation_saving_ms = (
            baseline["publication"]["staged_validation_ms"]
            - candidate["publication"]["staged_validation_ms"]
        )
        publication_saving_ms = (
            baseline["publication"]["publish_ms"]
            - candidate["publication"]["publish_ms"]
        )
        pairs.append(
            {
                "pair_index": pair_index,
                "pair_order": order,
                "wall_saving_ms": wall_saving_ms,
                "footprint_saving_bytes": footprint_saving_bytes,
                "validation_saving_ms": validation_saving_ms,
                "publication_saving_ms": publication_saving_ms,
                "baseline_validation_ms": baseline["publication"][
                    "staged_validation_ms"
                ],
                "candidate_validation_ms": candidate["publication"][
                    "staged_validation_ms"
                ],
                "baseline_publish_ms": baseline["publication"]["publish_ms"],
                "candidate_publish_ms": candidate["publication"]["publish_ms"],
                "wall_win": wall_saving_ms > 0,
                "validation_win": validation_saving_ms > 0,
                "publication_win": publication_saving_ms > 0,
                "footprint_nonregression": footprint_saving_bytes >= 0,
                "wall_gate_passes": wall_saving_ms >= WALL_GATE_MS,
                "footprint_gate_passes": (
                    footprint_saving_bytes >= FOOTPRINT_GATE_BYTES
                ),
            }
        )
    wall_median = statistics.median(row["wall_saving_ms"] for row in pairs)
    footprint_median = statistics.median(row["footprint_saving_bytes"] for row in pairs)
    admissible = all(
        row["wall_win"]
        and row["validation_win"]
        and row["publication_win"]
        and row["footprint_nonregression"]
        for row in pairs
    )
    wall_gate = all(row["wall_gate_passes"] for row in pairs)
    footprint_gate = all(row["footprint_gate_passes"] for row in pairs)
    return {
        "stage": "publication",
        "pairs": pairs,
        "wall_saving_median_ms": wall_median,
        "footprint_saving_median_bytes": footprint_median,
        "wall_gate_passes": wall_gate,
        "footprint_gate_passes": footprint_gate,
        "admissible": admissible,
        "passes": admissible and (wall_gate or footprint_gate),
        "global_stdout_sha256": EXPECTED_STDOUT_SHA256,
        "global_blob_sha256": EXPECTED_BLOB_SHA256,
    }


def write_restore_messages() -> Path:
    payload = json.loads(PACKET_MESSAGES.read_text(encoding="utf-8"))
    messages = payload.get("messages")
    if not isinstance(messages, list):
        raise RuntimeError("wrapped messages fixture drifted")
    messages.extend(
        [
            {"role": "assistant", "content": "<think>\n"},
            {"role": "user", "content": "Continue with one short sentence."},
        ]
    )
    path = ARTIFACT / "restore-messages.json"
    with path.open("x", encoding="utf-8") as output:
        output.write(json_text(payload, pretty=True) + "\n")
        output.flush()
        os.fsync(output.fileno())
    return path


def run_restore_smoke(
    candidate_row: dict[str, object],
    base_env: dict[str, str],
    manifest: dict[str, object],
) -> dict[str, object]:
    stem = "restore-smoke-b"
    store_root = Path(candidate_row["store_before"]["root"])
    blob = Path(candidate_row["store_after"]["blob_path"])
    before_sha = protocol.common.sha256_file(blob)
    messages = write_restore_messages()
    conditioning = protocol.condition_for_child(stem, manifest)
    env = base_env.copy()
    env[VALIDATION_ENV] = "encoder-digest"
    command = child_command(store_root, messages)
    min_index = command.index("--durable-prefix-cache-min-tokens") + 1
    command[min_index] = "262144"
    context_index = command.index("--max-context-tokens") + 1
    command[context_index] = "8192"
    stdout_path = ARTIFACT / f"{stem}.out"
    stderr_path = ARTIFACT / f"{stem}.err"
    protocol.record_launch("restore", stem, command, "B", 0, 1)
    process = None
    wait_errors: list[str] = []
    started = time.perf_counter()
    try:
        with (
            stdout_path.open("xb") as stdout_file,
            stderr_path.open("xb") as stderr_file,
        ):
            prior_sigint = signal.signal(signal.SIGINT, signal.SIG_IGN)
            try:
                process = subprocess.Popen(
                    command,
                    cwd=ROOT,
                    env=env,
                    stdout=stdout_file,
                    stderr=stderr_file,
                )
                returncode, wait_errors = protocol.wait_for_child(process)
                wall_ms = (time.perf_counter() - started) * 1e3
            finally:
                signal.signal(signal.SIGINT, prior_sigint)
    except (OSError, KeyboardInterrupt) as error:
        error_text = f"{type(error).__name__}:{error}"
        if process is None:
            protocol.record_completion("restore", stem, None, error_text)
            raise protocol.InconclusivePacket("restore", stem, [error_text]) from error
        returncode, deferred = protocol.wait_for_child(process)
        wall_ms = (time.perf_counter() - started) * 1e3
        wait_errors.extend(deferred)
        wait_errors.append(error_text)
    protocol.record_completion(
        "restore",
        stem,
        returncode,
        ";".join(wait_errors) if wait_errors else None,
    )
    post_exit = protocol.capture_post_exit_state(conditioning)
    protocol.record_post_exit_state("restore", stem, returncode, post_exit)
    stderr = stderr_path.read_text(encoding="utf-8")
    stdout = stdout_path.read_bytes()
    validity, reasons = protocol.finish_child_validity(
        stderr, post_exit, gate_major_faults=True
    )
    if wait_errors:
        reasons.append(f"child_wait_interrupted={';'.join(wait_errors)}")
    if returncode != 0:
        reasons.append(f"child_nonzero_exit={returncode}")
    if not stdout:
        reasons.append("restore_smoke_generated_empty_stdout")
    restore_lines = [
        line
        for line in stderr.splitlines()
        if line.startswith("durable_prefix_cache: identity_cache=")
    ]
    restore_pattern = re.compile(
        r"durable_prefix_cache: identity_cache=hit hashed_bytes=0 "
        r"checkpoint_hit=true matched=6500 restored=6499 exact=false "
        r"candidates=1 corrupt_removed=0 restore_total_ms=([0-9.]+)$",
    )
    restore = (
        restore_pattern.fullmatch(restore_lines[0]) if len(restore_lines) == 1 else None
    )
    if restore is None or not math.isfinite(float(restore.group(1))):
        reasons.append("restore_marker_drifted")
    if any(
        line.startswith("durable_prefix_cache: publish=")
        for line in stderr.splitlines()
    ):
        reasons.append("restore_smoke_published_new_checkpoint")
    after_sha = protocol.common.sha256_file(blob)
    if before_sha != EXPECTED_BLOB_SHA256 or after_sha != before_sha:
        reasons.append("restore_smoke_changed_candidate_blob")
    try:
        store_after = inspect_store(store_root)
    except Exception as error:
        store_after = None
        reasons.append(f"restore_store_invalid={type(error).__name__}:{error}")
    row = {
        "stage": "restore",
        "artifact_stem": stem,
        "command": command,
        "arm_environment": {VALIDATION_ENV: "encoder-digest"},
        **conditioning,
        **validity,
        "process_wall_ms": wall_ms,
        "restore_total_ms": float(restore.group(1)) if restore is not None else None,
        "blob_sha256_before": before_sha,
        "blob_sha256_after": after_sha,
        "store_after": store_after,
        "stdout_sha256": hashlib.sha256(stdout).hexdigest(),
        "valid": not reasons,
        "validity_reasons": reasons,
    }
    protocol.record_post_exit_evidence("restore", stem, returncode, validity, reasons)
    return row


def retain_candidate_blob(rows: list[dict[str, object]]) -> dict[str, object]:
    candidate = next(row for row in rows if row["arm"] == "B")
    source = Path(candidate["store_after"]["blob_path"])
    target = ARTIFACT / "candidate.qcp"
    with (
        source.open("rb", buffering=0) as left,
        target.open("xb", buffering=0) as right,
    ):
        shutil.copyfileobj(left, right, length=8 * 1024 * 1024)
        right.flush()
        os.fsync(right.fileno())
    digest = protocol.common.sha256_file(target)
    if target.stat().st_size != EXPECTED_BLOB_BYTES or digest != EXPECTED_BLOB_SHA256:
        raise RuntimeError("retained candidate blob drifted")
    return {
        "path": str(target.relative_to(ROOT)),
        "bytes": target.stat().st_size,
        "sha256": digest,
    }


def write_decision(decision: dict[str, object], manifest: dict[str, object]) -> None:
    if decision.get("status") == "go":
        publication = decision.get("stages", {}).get("publication")
        restore = decision.get("stages", {}).get("restore")
        retained = decision.get("retained_candidate")
        if (
            decision.get("authority")
            != "force-only-exact-allocation-free-staged-validation"
            or decision.get("source_commit") != manifest.get("source_commit")
            or not isinstance(publication, dict)
            or publication.get("passes") is not True
            or not isinstance(restore, dict)
            or restore.get("valid") is not True
            or not isinstance(retained, dict)
            or retained.get("sha256") != EXPECTED_BLOB_SHA256
        ):
            raise RuntimeError("v0.622 GO authority conjunction is incomplete")
    elif decision.get("authority") != "none":
        raise RuntimeError("non-GO v0.622 decision carries authority")
    protocol.write_identity_checked_decision(decision, manifest)


def run(*, preflight_only: bool) -> None:
    if ARTIFACT.exists() or WORK_ROOT.exists():
        raise RuntimeError("refusing to reuse packet or work directory")
    base_env, removed_environment = protocol.common.normalized_environment()
    manifest = build_manifest(removed_environment, base_env)
    if preflight_only:
        vm_state = protocol.capture_vm_state()
        if vm_state["capture_errors"]:
            raise RuntimeError(f"VM preflight failed: {vm_state['capture_errors']}")
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
    protocol.reserve_artifact(manifest)
    freeze_packet_inputs()
    WORK_ROOT.mkdir()
    attempts_path = ARTIFACT / "attempts.jsonl"
    stages: dict[str, object] = {}
    stopped_after = "publication"
    try:
        rows = run_pairs(base_env, manifest, attempts_path)
        stages["publication"] = analyze(rows)
        status = "kill"
        authority = "none"
        retained = None
        if stages["publication"]["passes"]:
            stopped_after = "restore"
            candidate_row = next(row for row in rows if row["arm"] == "B")
            restore = run_restore_smoke(candidate_row, base_env, manifest)
            protocol.append_row(attempts_path, restore)
            stages["restore"] = restore
            if not restore["valid"]:
                raise protocol.InconclusivePacket(
                    "restore", restore["artifact_stem"], restore["validity_reasons"]
                )
            retained = retain_candidate_blob(rows)
            status = "go"
            authority = "force-only-exact-allocation-free-staged-validation"
        shutil.rmtree(WORK_ROOT)
        write_decision(
            {
                "schema": 1,
                "status": status,
                "authority": authority,
                "stopped_after": stopped_after,
                "source_commit": manifest["source_commit"],
                "stages": stages,
                "retained_candidate": retained,
            },
            manifest,
        )
    except protocol.InconclusivePacket as error:
        if WORK_ROOT.exists():
            shutil.rmtree(WORK_ROOT)
        write_decision(
            {
                "schema": 1,
                "status": "inconclusive",
                "authority": "none",
                "stopped_after": error.stage,
                "failed_child": error.child,
                "reasons": error.reasons,
                "source_commit": manifest["source_commit"],
                "stages": stages,
            },
            manifest,
        )
    except KeyboardInterrupt:
        if WORK_ROOT.exists():
            shutil.rmtree(WORK_ROOT)
        write_decision(
            {
                "schema": 1,
                "status": "inconclusive",
                "authority": "none",
                "stopped_after": stopped_after,
                "reasons": ["operator_interrupt_after_child_cleanup"],
                "source_commit": manifest["source_commit"],
                "stages": stages,
            },
            manifest,
        )
    except Exception as error:
        if protocol.decision_publication_started():
            raise
        if WORK_ROOT.exists():
            shutil.rmtree(WORK_ROOT)
        write_decision(
            {
                "schema": 1,
                "status": "implementation_or_contract_defect",
                "authority": "none",
                "stopped_after": stopped_after,
                "error_type": type(error).__name__,
                "error": str(error),
                "source_commit": manifest["source_commit"],
                "stages": stages,
            },
            manifest,
        )
        raise


def configure_protocol() -> None:
    protocol.ARTIFACT = ARTIFACT
    protocol.PREREG = PREREG
    protocol.MODEL = MODEL
    protocol.required_manifest_paths = required_manifest_paths
    protocol.source_and_build_identity = source_and_build_identity
    protocol.verify_non_model_identity = verify_non_model_identity


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true")
    arguments = parser.parse_args()
    configure_protocol()
    run(preflight_only=arguments.preflight_only)
