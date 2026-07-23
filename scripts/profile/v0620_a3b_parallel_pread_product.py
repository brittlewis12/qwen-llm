#!/usr/bin/env python3

"""Fresh-only v0.620 product confirmation with a sealed v0.619 bridge."""

import argparse
import hashlib
import json
from pathlib import Path
import time

import v0602_a3b_parallel_copied_loader as protocol
import v0619_a3b_parallel_pread_loader as pread_protocol


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = ROOT / "target/profiles/v0620-a3b-parallel-pread-product-p1"
PREREG = ROOT / "docs/bench/v0620-a3b-parallel-pread-product.md"
BASE_PROTOCOL = ROOT / "scripts/profile/v0602_a3b_parallel_copied_loader.py"
PREAD_PROTOCOL = ROOT / "scripts/profile/v0619_a3b_parallel_pread_loader.py"
COMMON_RUNNER = ROOT / "scripts/profile/v0593_demand_paged_no_copy.py"
PREDECESSOR = ROOT / "target/profiles/v0619-a3b-parallel-pread-loader-p1"
PREDECESSOR_COMMIT = "c7884742c425a6cb5d80b88e0c67ee05ba258892"
STDERR_WRITER_COMMIT = "ea8ec11a7afa2f2647544669cffc9e23cd56a72b"
PREDECESSOR_COMPLETE_SHA256 = (
    "daf5473344fdeec2851942e593d31b3fafd816fdf8624a86d74882ad12b815e1"
)
PREDECESSOR_DECISION_SHA256 = (
    "c5ac80870199ab8e99029a33f2e7fb3510ca70711a006e05480717d68714f6d1"
)
PREDECESSOR_INVENTORY_SHA256 = (
    "c870d3a35c91d0789cde16a177f63a577bea999c1f24248ae5e61d6224360b1b"
)
EXPECTED_STDOUT_SHA256 = (
    "e18fd50a1e2add01cfed4f498bf0052b630653517057cda3620b302d1a68f198"
)
COPIED_MARKER = "[metal-gguf-parallel-copied]"
PREAD_MARKER = "[metal-gguf-parallel-pread]"
TIMING_FIELDS = (
    "allocation_us",
    "source_us",
    "copy_us",
    "binding_us",
    "ready_us",
)
BASE_RUN_FRESH_CHILD = protocol.run_fresh_child


def parse_json_file(path: Path) -> dict[str, object]:
    value = json.loads(
        path.read_text(encoding="utf-8"), parse_constant=protocol.reject_json_constant
    )
    if not isinstance(value, dict):
        raise RuntimeError(f"{path} is not a JSON object")
    return value


def verify_predecessor() -> dict[str, object]:
    complete_path = PREDECESSOR / "packet-complete.json"
    decision_path = PREDECESSOR / "decision.json"
    inventory_path = PREDECESSOR / "artifact-inventory.sha256"
    expected_hashes = {
        complete_path: PREDECESSOR_COMPLETE_SHA256,
        decision_path: PREDECESSOR_DECISION_SHA256,
        inventory_path: PREDECESSOR_INVENTORY_SHA256,
    }
    for path, expected in expected_hashes.items():
        if protocol.common.sha256_file(path) != expected:
            raise RuntimeError(f"predecessor {path.name} digest drifted")
    complete = parse_json_file(complete_path)
    if (
        complete.get("schema") != 1
        or complete.get("decision_sha256") != PREDECESSOR_DECISION_SHA256
        or complete.get("inventory_sha256") != PREDECESSOR_INVENTORY_SHA256
    ):
        raise RuntimeError("predecessor completion seal drifted")

    listed_paths = []
    for line in inventory_path.read_text(encoding="utf-8").splitlines():
        digest, separator, relative = line.partition("  ")
        if separator != "  " or len(digest) != 64:
            raise RuntimeError("predecessor inventory row is malformed")
        path = ROOT / relative
        if path.parent != PREDECESSOR or path.name in {
            "artifact-inventory.sha256",
            "packet-complete.json",
        }:
            raise RuntimeError("predecessor inventory path escapes its packet")
        if protocol.common.sha256_file(path) != digest:
            raise RuntimeError(f"predecessor inventory drifted at {path.name}")
        listed_paths.append(path)
    actual_files = {path for path in PREDECESSOR.iterdir() if path.is_file()}
    expected_files = set(listed_paths) | {inventory_path, complete_path}
    if len(listed_paths) != len(set(listed_paths)) or actual_files != expected_files:
        raise RuntimeError("predecessor packet file inventory drifted")

    manifest = parse_json_file(PREDECESSOR / "manifest.json")
    decision = parse_json_file(decision_path)
    loaded = decision.get("stages", {}).get("loaded")
    if (
        manifest.get("source_commit") != PREDECESSOR_COMMIT
        or decision.get("source_commit") != PREDECESSOR_COMMIT
        or decision.get("status") != "implementation_or_contract_defect"
        or decision.get("authority") != "none"
        or decision.get("stopped_after") != "fresh-128"
        or decision.get("error") != "outer_residual_ms.left is not positive"
        or not isinstance(decision.get("correctness"), dict)
        or decision["correctness"].get("passed") is not True
        or not isinstance(loaded, dict)
        or loaded.get("stable") is not True
        or loaded.get("performance_passes") is not True
        or loaded.get("passes") is not True
    ):
        raise RuntimeError("predecessor decision contract drifted")

    attempts = [
        json.loads(line, parse_constant=protocol.reject_json_constant)
        for line in (PREDECESSOR / "attempts.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    ]
    if len(attempts) != 12 or any(row.get("stage") != "loaded" for row in attempts):
        raise RuntimeError("predecessor loaded attempt set drifted")
    launch_rows = [
        json.loads(line, parse_constant=protocol.reject_json_constant)
        for line in (PREDECESSOR / "launch-seal.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    ]
    if len(launch_rows) != 26:
        raise RuntimeError("predecessor launch seal count drifted")
    fresh_rows = [row for row in launch_rows if row.get("stage") == "fresh-128"]
    if (
        len(fresh_rows) != 2
        or fresh_rows[0].get("event") != "launch"
        or fresh_rows[0].get("arm") != "A"
        or fresh_rows[1].get("event") != "completion"
        or fresh_rows[1].get("returncode") != 0
    ):
        raise RuntimeError("predecessor fresh stop boundary drifted")
    return {
        "packet_complete_sha256": PREDECESSOR_COMPLETE_SHA256,
        "decision_sha256": PREDECESSOR_DECISION_SHA256,
        "inventory_sha256": PREDECESSOR_INVENTORY_SHA256,
        "source_commit": PREDECESSOR_COMMIT,
        "loaded_global_output_sha256": loaded["global_output_sha256"],
        "loaded_pairs": len(loaded["pairs"]),
        "loaded_max_rss_b_over_a": loaded["rss_b_over_a"]["maximum"],
        "loaded_max_footprint_b_over_a": loaded["footprint_b_over_a"]["maximum"],
    }


def required_manifest_paths() -> tuple[Path, ...]:
    return (
        Path(__file__).resolve(),
        PREREG,
        BASE_PROTOCOL,
        PREAD_PROTOCOL,
        COMMON_RUNNER,
        protocol.MODEL,
        protocol.PROMPT,
        protocol.CLI_BINARY,
        protocol.BENCH_BINARY,
        PREDECESSOR / "packet-complete.json",
        PREDECESSOR / "decision.json",
        PREDECESSOR / "artifact-inventory.sha256",
    )


def source_and_build_identity() -> tuple[str, dict[str, object]]:
    tracked = (
        Path(__file__).resolve(),
        PREREG,
        BASE_PROTOCOL,
        PREAD_PROTOCOL,
        COMMON_RUNNER,
        protocol.PROMPT,
    )
    for path in tracked:
        protocol.command_text(
            ["git", "ls-files", "--error-unmatch", str(path.relative_to(ROOT))]
        )
    commit = protocol.command_text(["git", "rev-parse", "HEAD"]).strip()
    dirty = protocol.command_text(["git", "status", "--porcelain=v1"]).strip()
    if dirty:
        raise RuntimeError(f"source is dirty: {dirty!r}")
    writer_parent = protocol.command_text(
        ["git", "rev-parse", f"{STDERR_WRITER_COMMIT}^"]
    ).strip()
    packet_parent = protocol.command_text(["git", "rev-parse", "HEAD^"]).strip()
    if writer_parent != PREDECESSOR_COMMIT or packet_parent != STDERR_WRITER_COMMIT:
        raise RuntimeError("stdout-writer evidence bridge ancestry drifted")
    changed = set(
        protocol.command_text(
            ["git", "diff", "--name-only", f"{STDERR_WRITER_COMMIT}..{commit}"]
        ).splitlines()
    )
    expected_changed = {
        str(PREREG.relative_to(ROOT)),
        str(Path(__file__).resolve().relative_to(ROOT)),
    }
    if changed != expected_changed:
        raise RuntimeError(f"executable-source bridge drifted: {sorted(changed)}")
    build = protocol.parse_json(
        protocol.command_text(
            [str(protocol.BENCH_BINARY), "build-info", "--output", "json"]
        )
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
    binary = protocol.CLI_BINARY.read_bytes()
    embedded = (
        commit.encode(),
        str(build["build_source_state"]).encode(),
    )
    if any(value not in binary for value in embedded):
        raise RuntimeError("product qwen binary identity is not embedded")
    return commit, build


def build_manifest(
    removed_environment: list[str], base_env: dict[str, str]
) -> dict[str, object]:
    predecessor = verify_predecessor()
    commit, build = source_and_build_identity()
    paths = required_manifest_paths()
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError(f"missing packet inputs: {missing}")
    hashes = {str(path): protocol.common.sha256_file(path) for path in paths}
    if hashes[str(protocol.MODEL)] != protocol.EXPECTED_MODEL_SHA256:
        raise RuntimeError("model SHA-256 drifted")
    if hashes[str(protocol.PROMPT)] != protocol.EXPECTED_PROMPT_SHA256:
        raise RuntimeError("prompt SHA-256 drifted")
    device = protocol.command_text(
        [str(protocol.CLI_BINARY), "--info"], env=base_env
    ).strip()
    macos = protocol.command_text(["sw_vers", "-productVersion"], env=base_env).strip()
    hw_memsize = int(
        protocol.command_text(["sysctl", "-n", "hw.memsize"], env=base_env)
    )
    if device != protocol.EXPECTED_DEVICE:
        raise RuntimeError(f"device boundary drifted: {device!r}")
    if not macos.startswith("15."):
        raise RuntimeError(f"macOS boundary drifted: {macos!r}")
    if hw_memsize != protocol.EXPECTED_HW_MEMSIZE:
        raise RuntimeError(f"memory boundary drifted: {hw_memsize}")
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
        "model_size_bytes": protocol.MODEL.stat().st_size,
        "prompt_bytes": protocol.PROMPT.stat().st_size,
        "prompt_tokens": protocol.EXPECTED_PROMPT_TOKENS,
        "output_tokens": protocol.EXPECTED_OUTPUT_TOKENS,
        "transition_count": protocol.EXPECTED_TRANSITIONS,
        "pair_orders": list(protocol.PAIR_ORDERS),
        "cooldown_s": protocol.COOLDOWN_S,
        "host_sample_limit": protocol.HOST_SAMPLE_LIMIT,
        "host_sample_interval_s": protocol.HOST_SAMPLE_INTERVAL_S,
        "child_retry_count": 0,
        "imported_predecessor": predecessor,
        "candidate": "forced-pread-over-forced-mmap-copy",
        "expected_stdout_sha256": EXPECTED_STDOUT_SHA256,
    }


def arm_environment(arm: str) -> dict[str, str | None]:
    if arm not in ("A", "B"):
        raise ValueError(f"unknown arm {arm!r}")
    return {
        "QWEN_GGUF_PARALLEL_COPY": "1" if arm == "A" else "pread",
        "QWEN_GGUF_OWNED_ARENA": "0",
        "QWEN_GGUF_NO_COPY": "0",
        "QWEN_GGUF_NO_COPY_PREFAULT": None,
        "QWEN_NATIVE_QUANT_EMBED": None,
        "QWEN_MOE_ROUTER_F16": None,
    }


def marker_prefix(marker: str) -> str:
    return protocol.MARKER_PREFIX.replace("[metal-gguf-parallel-copied]", marker, 1)


def parse_marker(line: str, expected_marker: str) -> dict[str, int]:
    suffix = line.removeprefix(marker_prefix(expected_marker))
    if suffix == line:
        raise RuntimeError("parallel population marker contract drifted")
    fields = suffix.split(" ")
    if len(fields) != len(TIMING_FIELDS):
        raise RuntimeError("parallel population timing field count drifted")
    values = {}
    for field, name in zip(fields, TIMING_FIELDS, strict=True):
        prefix = f"{name}="
        value = field.removeprefix(prefix)
        if value == field or not value.isascii() or not value.isdecimal():
            raise RuntimeError(f"parallel population {name} is not decimal")
        parsed = int(value)
        if str(parsed) != value or parsed > protocol.U64_MAX:
            raise RuntimeError(f"parallel population {name} is not canonical")
        values[name] = parsed
    if abs(values["ready_us"] - sum(values[name] for name in TIMING_FIELDS[:-1])) > 4:
        raise RuntimeError("parallel population timing does not reconcile")
    return values


def expected_marker(arm: str) -> str:
    return COPIED_MARKER if arm == "A" else PREAD_MARKER


def parse_load_contract(stderr: str, arm: str) -> dict[str, object]:
    marker = expected_marker(arm)
    if stderr.count(marker) != 1 or stderr.count("[metal-gguf-") != 1:
        raise RuntimeError(f"{arm} population marker occurrence count drifted")
    if stderr.count("[metal-load] native quantized token embedding policy:") != 1:
        raise RuntimeError(f"{arm} native-policy occurrence count drifted")
    if stderr.count("[metal-load-ledger]") != 1:
        raise RuntimeError(f"{arm} load-ledger occurrence count drifted")
    prefixes = (
        "[metal-load] native quantized token embedding policy:",
        marker,
        "[metal-load-ledger]",
    )
    recognized = [line for line in stderr.splitlines() if line.startswith(prefixes)]
    if recognized[:1] != [protocol.POLICY_LINE] or recognized[2:] != [
        protocol.LEDGER_LINE
    ]:
        raise RuntimeError(f"{arm} load-line ordering drifted: {recognized!r}")
    timings = parse_marker(recognized[1], marker)
    return {
        "storage": "parallel-copied" if arm == "A" else "parallel-pread",
        "marker": recognized[1],
        "phase_us": timings,
    }


def parse_observed_load_contract(stderr: str, arm: str) -> dict[str, object] | None:
    marker = expected_marker(arm)
    if stderr.count("[metal-gguf-") != stderr.count(marker):
        raise RuntimeError("observed unrequested storage marker")
    tokens = (
        "[metal-load] native quantized token embedding policy:",
        marker,
        "[metal-load-ledger]",
    )
    occurrence_count = sum(stderr.count(token) for token in tokens)
    if occurrence_count == 0:
        return None
    recognized = [line for line in stderr.splitlines() if line.startswith(tokens)]
    if occurrence_count != len(recognized) or len(recognized) > 3:
        raise RuntimeError("observed load contract has malformed or extra lines")
    expected = (protocol.POLICY_LINE, marker, protocol.LEDGER_LINE)
    for observed, expected_prefix in zip(recognized, expected, strict=False):
        if expected_prefix == marker:
            parse_marker(observed, marker)
        elif observed != expected_prefix:
            raise RuntimeError("observed load contract contradicts expected prefix")
    if len(recognized) < 3:
        return {"status": "incomplete-valid-prefix", "recognized_lines": recognized}
    return {"status": "complete", "contract": parse_load_contract(stderr, arm)}


def analyze_fresh(rows: list[dict[str, object]]) -> dict[str, object]:
    analysis = pread_protocol.analyze_fresh(rows)
    if analysis.get("global_output_sha256") != EXPECTED_STDOUT_SHA256:
        raise RuntimeError("fresh generated stdout digest drifted")
    return analysis


def run_fresh_child(*args: object, **kwargs: object) -> dict[str, object]:
    row = BASE_RUN_FRESH_CHILD(*args, **kwargs)
    if row.get("valid") is True and row.get("stdout_sha256") != EXPECTED_STDOUT_SHA256:
        raise RuntimeError("fresh child generated stdout digest drifted")
    return row


def configure_protocol() -> None:
    protocol.ARTIFACT = ARTIFACT
    protocol.PREREG = PREREG
    protocol.required_manifest_paths = required_manifest_paths
    protocol.source_and_build_identity = source_and_build_identity
    protocol.build_manifest = build_manifest
    protocol.arm_environment = arm_environment
    protocol.parse_load_contract = parse_load_contract
    protocol.parse_observed_load_contract = parse_observed_load_contract
    protocol.analyze_fresh = analyze_fresh
    protocol.run_fresh_child = run_fresh_child


def write_decision(decision: dict[str, object], manifest: dict[str, object]) -> None:
    if decision.get("status") == "go":
        fresh = decision.get("stages", {}).get("fresh_128")
        predecessor = verify_predecessor()
        if (
            decision.get("stopped_after") != "fresh-128"
            or decision.get("authority")
            != "force-only-exact-a3b-pread-over-forced-copy"
            or not isinstance(fresh, dict)
            or fresh.get("passes") is not True
            or fresh.get("global_output_sha256") != EXPECTED_STDOUT_SHA256
            or manifest.get("imported_predecessor") != predecessor
            or decision.get("imported_predecessor") != predecessor
        ):
            raise RuntimeError("v0.620 GO authority conjunction is incomplete")
    elif decision.get("authority") != "none":
        raise RuntimeError("non-GO v0.620 decision carries authority")
    protocol.write_identity_checked_decision(decision, manifest)


def run(*, preflight_only: bool) -> None:
    if ARTIFACT.exists():
        raise RuntimeError(f"refusing to reuse packet directory {ARTIFACT}")
    base_env, removed_environment = protocol.common.normalized_environment()
    manifest = build_manifest(removed_environment, base_env)
    if preflight_only:
        vm_state = protocol.capture_vm_state()
        if vm_state["capture_errors"]:
            raise RuntimeError(
                f"VM preflight capture failed: {vm_state['capture_errors']}"
            )
        print(
            protocol.json_text(
                {
                    "status": "preflight-passed",
                    "source_commit": manifest["source_commit"],
                    "manifest_sha256": hashlib.sha256(
                        (protocol.json_text(manifest, pretty=True) + "\n").encode()
                    ).hexdigest(),
                },
                pretty=True,
            )
        )
        return
    protocol.reserve_artifact(manifest)
    attempts_path = ARTIFACT / "attempts.jsonl"
    stages = {}
    try:
        rows = protocol.run_stage("fresh-128", base_env, manifest, attempts_path)
        stages["fresh_128"] = analyze_fresh(rows)
        status = "go" if stages["fresh_128"]["passes"] else "kill"
        authority = (
            "force-only-exact-a3b-pread-over-forced-copy" if status == "go" else "none"
        )
        write_decision(
            {
                "schema": 1,
                "status": status,
                "authority": authority,
                "stopped_after": "fresh-128",
                "source_commit": manifest["source_commit"],
                "imported_predecessor": manifest["imported_predecessor"],
                "stages": stages,
            },
            manifest,
        )
    except protocol.InconclusivePacket as error:
        write_decision(
            {
                "schema": 1,
                "status": "inconclusive",
                "authority": "none",
                "stopped_after": error.stage,
                "failed_child": error.child,
                "reasons": error.reasons,
                "source_commit": manifest["source_commit"],
                "imported_predecessor": manifest["imported_predecessor"],
                "stages": stages,
            },
            manifest,
        )
    except KeyboardInterrupt:
        write_decision(
            {
                "schema": 1,
                "status": "inconclusive",
                "authority": "none",
                "stopped_after": "fresh-128",
                "reasons": ["operator_interrupt_after_child_cleanup"],
                "source_commit": manifest["source_commit"],
                "imported_predecessor": manifest["imported_predecessor"],
                "stages": stages,
            },
            manifest,
        )
    except Exception as error:
        if protocol.decision_publication_started():
            raise
        write_decision(
            {
                "schema": 1,
                "status": "implementation_or_contract_defect",
                "authority": "none",
                "stopped_after": "fresh-128",
                "error_type": type(error).__name__,
                "error": str(error),
                "source_commit": manifest["source_commit"],
                "imported_predecessor": manifest["imported_predecessor"],
                "stages": stages,
            },
            manifest,
        )
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true")
    arguments = parser.parse_args()
    configure_protocol()
    run(preflight_only=arguments.preflight_only)
