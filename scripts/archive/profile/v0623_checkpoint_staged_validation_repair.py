#!/usr/bin/env python3

"""v0.623 contract-only repair of the sealed v0.622 packet."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil

import v0602_a3b_parallel_copied_loader as protocol
import v0622_checkpoint_staged_validation as base


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = ROOT / "target/profiles/v0623-checkpoint-staged-validation-p1"
WORK_ROOT = ROOT / "target/profiles/v0623-checkpoint-staged-validation-work"
PREREG = ROOT / "docs/bench/v0623-checkpoint-staged-validation-repair.md"
PACKET_MESSAGES = ARTIFACT / "input-messages.json"
PACKET_IDENTITY_SEED = ARTIFACT / "input-identity.mid"
V0622_COMMIT = "b72844376905ca624f7443014fa1318751a086e9"
V0622_ROOT = ROOT / "target/profiles/v0622-checkpoint-staged-validation-p1"
V0622_PREREG = base.PREREG
V0622_COMPLETE_SHA256 = (
    "5a0ccbfa1fa3f3322a17bce18241ba794c995a00f55ea3c182310539c9821657"
)
V0622_DECISION_SHA256 = (
    "5a1043125c6cb42cf1ab37376f6937b1bd79ccb594dbce5304e99894fe4f4f8f"
)
V0622_INVENTORY_SHA256 = (
    "da62de99ec511910f24216427b9de1ef9ff9c46ab3fcac0b399cd8908e8e1855"
)

BASE_BUILD_MANIFEST = base.build_manifest
BASE_FREEZE_PACKET_INPUTS = base.freeze_packet_inputs
BASE_PARSE_PUBLICATION = base.parse_publication
BASE_FINISH_CHILD_VALIDITY = protocol.finish_child_validity
BASE_WRITE_DECISION = base.write_decision


def parse_json(path: Path) -> dict[str, object]:
    value = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=protocol.reject_json_constant,
    )
    if not isinstance(value, dict):
        raise RuntimeError(f"{path} is not a JSON object")
    return value


def bridge_members() -> tuple[Path, ...]:
    inventory = V0622_ROOT / "artifact-inventory.sha256"
    members = []
    for line in inventory.read_text(encoding="utf-8").splitlines():
        digest, separator, relative = line.partition("  ")
        path = ROOT / relative
        if (
            separator != "  "
            or len(digest) != 64
            or path.parent != V0622_ROOT
            or path.name in {"artifact-inventory.sha256", "packet-complete.json"}
        ):
            raise RuntimeError("v0.622 inventory row is malformed")
        if protocol.common.sha256_file(path) != digest:
            raise RuntimeError(f"v0.622 inventory drifted at {path.name}")
        members.append(path)
    if len(members) != len(set(members)):
        raise RuntimeError("v0.622 inventory contains duplicate members")
    return tuple(members)


def verify_v0622_bridge() -> dict[str, object]:
    complete_path = V0622_ROOT / "packet-complete.json"
    decision_path = V0622_ROOT / "decision.json"
    inventory_path = V0622_ROOT / "artifact-inventory.sha256"
    expected = {
        complete_path: V0622_COMPLETE_SHA256,
        decision_path: V0622_DECISION_SHA256,
        inventory_path: V0622_INVENTORY_SHA256,
    }
    for path, digest in expected.items():
        if protocol.common.sha256_file(path) != digest:
            raise RuntimeError(f"v0.622 {path.name} seal drifted")
    complete = parse_json(complete_path)
    if complete != {
        "schema": 1,
        "decision_sha256": V0622_DECISION_SHA256,
        "inventory_sha256": V0622_INVENTORY_SHA256,
    }:
        raise RuntimeError("v0.622 completion contract drifted")
    members = bridge_members()
    actual = set(V0622_ROOT.iterdir())
    if actual != set(members) | {inventory_path, complete_path}:
        raise RuntimeError("v0.622 packet membership drifted")
    decision = parse_json(decision_path)
    if decision != {
        "schema": 1,
        "status": "implementation_or_contract_defect",
        "authority": "none",
        "stopped_after": "publication",
        "source_commit": V0622_COMMIT,
        "stages": {},
        "error_type": "RuntimeError",
        "error": "request stats contract drifted",
    }:
        raise RuntimeError("v0.622 decision drifted")
    launch_rows = [
        json.loads(line)
        for line in (V0622_ROOT / "launch-seal.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    if (
        len(launch_rows) != 2
        or launch_rows[0].get("event") != "launch"
        or launch_rows[0].get("arm") != "A"
        or launch_rows[0].get("artifact_stem") != "publish-p01-ab-r1-a"
        or launch_rows[1].get("event") != "completion"
        or launch_rows[1].get("artifact_stem") != "publish-p01-ab-r1-a"
        or launch_rows[1].get("returncode") != 0
    ):
        raise RuntimeError("v0.622 launch boundary drifted")
    stderr = (V0622_ROOT / "publish-p01-ab-r1-a.err").read_text(encoding="utf-8")
    stats = [line for line in stderr.splitlines() if line.startswith("stats:")]
    publication = [
        line
        for line in stderr.splitlines()
        if line.startswith("durable_prefix_cache: publish=")
    ]
    if (
        len(stats) != 1
        or "transitions=0 stop_reason=token_limit load_ms=" not in stats[0]
        or len(publication) != 1
        or "staged_validation=decode staged_validation_ms=284.2" not in publication[0]
        or "blob_bytes=582854188" not in publication[0]
        or "identity=hit" not in publication[0]
    ):
        raise RuntimeError("v0.622 observed repair evidence drifted")
    if protocol.common.sha256_file(V0622_ROOT / "publish-p01-ab-r1-a.out") != (
        base.EXPECTED_STDOUT_SHA256
    ):
        raise RuntimeError("v0.622 stdout drifted")
    post = parse_json(V0622_ROOT / "publish-p01-ab-r1-a.post-exit-state.json")
    post_exit = post.get("post_exit")
    if (
        not isinstance(post_exit, dict)
        or post_exit.get("host_after_exit", {}).get("valid") is not True
        or post_exit.get("child_interval", {}).get("failure_reasons") != []
    ):
        raise RuntimeError("v0.622 post-exit evidence drifted")
    return {
        "packet_complete_sha256": V0622_COMPLETE_SHA256,
        "decision_sha256": V0622_DECISION_SHA256,
        "inventory_sha256": V0622_INVENTORY_SHA256,
        "source_commit": V0622_COMMIT,
        "members": len(members),
        "a_children_launched": 1,
        "b_children_launched": 0,
        "authority": "none",
        "repair": "current stats stop_reason field and advisory major faults",
    }


def required_manifest_paths() -> tuple[Path, ...]:
    bridge = bridge_members() + (
        V0622_ROOT / "artifact-inventory.sha256",
        V0622_ROOT / "packet-complete.json",
    )
    return (
        Path(__file__).resolve(),
        PREREG,
        Path(base.__file__).resolve(),
        V0622_PREREG,
        base.BASE_PROTOCOL,
        base.COMMON_RUNNER,
        base.MODEL,
        base.MESSAGES,
        base.IDENTITY_SEED,
        base.CLI_BINARY,
        base.BENCH_BINARY,
        *bridge,
    )


def source_and_build_identity() -> tuple[str, dict[str, object]]:
    tracked = (
        Path(__file__).resolve(),
        PREREG,
        Path(base.__file__).resolve(),
        V0622_PREREG,
        base.BASE_PROTOCOL,
        base.COMMON_RUNNER,
    )
    for path in tracked:
        protocol.command_text(
            ["git", "ls-files", "--error-unmatch", str(path.relative_to(ROOT))]
        )
    commit = protocol.command_text(["git", "rev-parse", "HEAD"]).strip()
    dirty = protocol.command_text(["git", "status", "--porcelain=v1"]).strip()
    if dirty:
        raise RuntimeError(f"source is dirty: {dirty!r}")
    parent = protocol.command_text(["git", "rev-parse", "HEAD^"]).strip()
    if parent != V0622_COMMIT:
        raise RuntimeError("v0.623 is not the direct child of v0.622")
    changed = set(
        protocol.command_text(
            ["git", "diff", "--name-only", f"{V0622_COMMIT}..{commit}"]
        ).splitlines()
    )
    expected_changed = {
        str(PREREG.relative_to(ROOT)),
        str(Path(__file__).resolve().relative_to(ROOT)),
    }
    if changed != expected_changed:
        raise RuntimeError(f"v0.623 repair scope drifted: {sorted(changed)}")
    build = protocol.parse_json(
        protocol.command_text(
            [str(base.BENCH_BINARY), "build-info", "--output", "json"]
        )
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
    binary = base.CLI_BINARY.read_bytes()
    for value in (commit, str(build["build_source_state"])):
        if value.encode() not in binary:
            raise RuntimeError("qwen identity is not embedded")
    return commit, build


def build_manifest(
    removed_environment: list[str], child_env: dict[str, str]
) -> dict[str, object]:
    manifest = BASE_BUILD_MANIFEST(removed_environment, child_env)
    manifest["imported_v0622"] = verify_v0622_bridge()
    manifest["stats_grammar_repair"] = "stop_reason=token_limit after transitions"
    manifest["major_faults"] = "recorded-advisory"
    return manifest


def copy_bridge_evidence() -> None:
    for source in bridge_members() + (
        V0622_ROOT / "artifact-inventory.sha256",
        V0622_ROOT / "packet-complete.json",
    ):
        target = ARTIFACT / f"v0622-{source.name}"
        with (
            source.open("rb", buffering=0) as left,
            target.open("xb", buffering=0) as right,
        ):
            shutil.copyfileobj(left, right, length=1024 * 1024)
            right.flush()
            os.fsync(right.fileno())
        if protocol.common.sha256_file(target) != protocol.common.sha256_file(source):
            raise RuntimeError(f"copied v0.622 evidence drifted: {source.name}")


def freeze_packet_inputs() -> None:
    BASE_FREEZE_PACKET_INPUTS()
    copy_bridge_evidence()
    base.fsync_directory(ARTIFACT)


def verify_non_model_identity(manifest: dict[str, object]) -> None:
    commit, build = source_and_build_identity()
    if commit != manifest["source_commit"] or build != manifest["build_identity"]:
        raise RuntimeError("v0.623 source/build identity drifted")
    actual = {
        str(path): protocol.common.sha256_file(path)
        for path in required_manifest_paths()
        if path != base.MODEL
    }
    expected = {
        path: digest
        for path, digest in manifest["sha256"].items()
        if path != str(base.MODEL)
    }
    if actual != expected:
        raise RuntimeError("v0.623 non-model inputs drifted")
    packet_inputs = {
        PACKET_MESSAGES: base.EXPECTED_MESSAGES_SHA256,
        PACKET_IDENTITY_SEED: base.EXPECTED_IDENTITY_SHA256,
    }
    for path, digest in packet_inputs.items():
        if protocol.common.sha256_file(path) != digest:
            raise RuntimeError(f"v0.623 packet input drifted: {path.name}")
    for source in bridge_members() + (
        V0622_ROOT / "artifact-inventory.sha256",
        V0622_ROOT / "packet-complete.json",
    ):
        target = ARTIFACT / f"v0622-{source.name}"
        if protocol.common.sha256_file(target) != protocol.common.sha256_file(source):
            raise RuntimeError(f"v0.623 bridge copy drifted: {source.name}")


def parse_publication(stderr: str, arm: str) -> dict[str, object]:
    stats = [line for line in stderr.splitlines() if line.startswith("stats:")]
    if len(stats) != 1:
        raise RuntimeError("current stats marker count drifted")
    current = re.fullmatch(
        r"stats: prompt_tokens=6499 generated_tokens=1 transitions=0 "
        r"stop_reason=token_limit load_ms=([0-9.]+) prefill_ms=([0-9.]+) "
        r"ttft_ms=([0-9.]+) decode_tps=([0-9.]+) transition_tps=([0-9.]+) "
        r"cache_entries=(\d+) cache_mib=([0-9.]+)/([0-9.]+)",
        stats[0],
    )
    if current is None:
        raise RuntimeError("current stats grammar drifted")
    values = [float(value) for value in current.groups()[:5]]
    values.extend(float(value) for value in current.groups()[6:])
    if not all(math.isfinite(value) and value >= 0 for value in values):
        raise RuntimeError("current stats values are invalid")
    old = stats[0].replace(" transitions=0 stop_reason=token_limit ", " transitions=0 ")
    normalized = stderr.replace(stats[0], old, 1)
    parsed = BASE_PARSE_PUBLICATION(normalized, arm)
    parsed["stats_stop_reason"] = "token_limit"
    return parsed


def finish_child_validity(
    stderr: str,
    post_exit: dict[str, object],
    *,
    gate_major_faults: bool,
) -> tuple[dict[str, object], list[str]]:
    del gate_major_faults
    return BASE_FINISH_CHILD_VALIDITY(stderr, post_exit, gate_major_faults=False)


def write_decision(decision: dict[str, object], manifest: dict[str, object]) -> None:
    imported = verify_v0622_bridge()
    if manifest.get("imported_v0622") != imported:
        raise RuntimeError("v0.623 manifest bridge drifted")
    decision = {
        **decision,
        "imported_v0622": imported,
        "repair_scope": {
            "stats": "current stop_reason field",
            "major_faults": "recorded-advisory",
        },
    }
    BASE_WRITE_DECISION(decision, manifest)


def configure() -> None:
    base.ARTIFACT = ARTIFACT
    base.WORK_ROOT = WORK_ROOT
    base.PREREG = PREREG
    base.PACKET_MESSAGES = PACKET_MESSAGES
    base.PACKET_IDENTITY_SEED = PACKET_IDENTITY_SEED
    base.required_manifest_paths = required_manifest_paths
    base.source_and_build_identity = source_and_build_identity
    base.build_manifest = build_manifest
    base.freeze_packet_inputs = freeze_packet_inputs
    base.verify_non_model_identity = verify_non_model_identity
    base.parse_publication = parse_publication
    base.write_decision = write_decision
    protocol.ARTIFACT = ARTIFACT
    protocol.PREREG = PREREG
    protocol.MODEL = base.MODEL
    protocol.required_manifest_paths = required_manifest_paths
    protocol.source_and_build_identity = source_and_build_identity
    protocol.verify_non_model_identity = verify_non_model_identity
    protocol.finish_child_validity = finish_child_validity


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true")
    arguments = parser.parse_args()
    configure()
    base.run(preflight_only=arguments.preflight_only)
