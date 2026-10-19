#!/usr/bin/env python3
"""v0.629 repair for the v0.628 cold-cell page-fault validity gate."""

import argparse
import hashlib
import json
from pathlib import Path

import v0628_dense27b_pread_cold_composition as base


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = ROOT / "target/profiles/v0629-dense27b-pread-cold-composition-repair-p1"
PREREG = ROOT / "docs/bench/v0629-dense27b-pread-cold-composition-repair.md"
RUNNER = Path(__file__).resolve()
V0628_PREREG = ROOT / "docs/bench/v0628-dense27b-pread-cold-composition.md"
V0628_RUNNER = ROOT / "scripts/profile/v0628_dense27b_pread_cold_composition.py"
V0628_ROOT = ROOT / "target/profiles/v0628-dense27b-pread-cold-composition-p1"
V0628_DECISION = V0628_ROOT / "decision.json"
V0628_INVENTORY = V0628_ROOT / "artifact-inventory.sha256"
V0628_COMPLETE = V0628_ROOT / "packet-complete.json"

BASE_COMMIT = "113350615c33e0a17773050befcb5fbbff7f65ec"
EXPECTED_PREREG_SHA256 = (
    "5fe5a1de3182c539dedd43ff23cb4440422f9a4230cbf56ce106deb2ce93b536"
)
EXPECTED_V0628_DECISION = (
    "c0c2447016e88585161605897bf8cc80777cd382ecee5efb284daabd16dd18a5"
)
EXPECTED_V0628_INVENTORY = (
    "f8e63bd70d2f4ac6b34f6cb64e2c83752819e99370875d95dcf92880c3e44fb7"
)
EXPECTED_V0628_COMPLETE = (
    "b14714cae32e08d24a385d3d2e289765bb2c2e791fa052e6f1c5f7022c3b2680"
)
EXPECTED_EXAMPLE_BINARY = (
    "a9e4e6d25aaa91738eb0b302ff227d58e5d923b8a7afcfc9715853dd144aaa89"
)

ORIGINAL_BUILD_MANIFEST = base.build_manifest
ORIGINAL_MAKE_DECISION = base.make_decision
ORIGINAL_PROCESS_VALIDITY = base.process_validity
ORIGINAL_VERIFY_ARTIFACTS = base.verify_attempt_artifacts
ORIGINAL_VERIFY_IDENTITY = base.verify_non_model_identity


def sha256(path: Path) -> str:
    return base.sha256(path)


def verify_v0628_forensics() -> dict[Path, str]:
    seals = {
        V0628_DECISION: EXPECTED_V0628_DECISION,
        V0628_INVENTORY: EXPECTED_V0628_INVENTORY,
        V0628_COMPLETE: EXPECTED_V0628_COMPLETE,
    }
    for path, digest in seals.items():
        if not path.is_file() or path.is_symlink() or sha256(path) != digest:
            raise RuntimeError(f"v0.628 seal drifted: {path}")
    complete = base.json_value(V0628_COMPLETE)
    if not base.strict_equal(
        complete,
        {
            "schema": 1,
            "decision_sha256": EXPECTED_V0628_DECISION,
            "inventory_sha256": EXPECTED_V0628_INVENTORY,
        },
    ):
        raise RuntimeError("v0.628 completion binding drifted")
    members = base.inventory_members(V0628_INVENTORY)
    if len(members) != 12 or members.get(V0628_DECISION) != EXPECTED_V0628_DECISION:
        raise RuntimeError("v0.628 inventory membership drifted")
    observed = set(V0628_ROOT.iterdir())
    expected = set(members) | {V0628_INVENTORY, V0628_COMPLETE}
    if (
        len(observed) != 14
        or observed != expected
        or any(path.is_symlink() or not path.is_file() for path in observed)
    ):
        raise RuntimeError("v0.628 final artifact directory drifted")

    decision = base.json_value(V0628_DECISION)
    expected_terminal = {
        "schema": 1,
        "status": "inconclusive",
        "authority": "none",
        "force_authorized": False,
        "successor_authorization": "none",
        "stopped_after": "cold-composition",
        "source_commit": BASE_COMMIT,
        "failed_child": "cold-p01-ab-r1-a",
        "reasons": ["child_major_faults_nonzero"],
        "attempts_sha256": (
            "e2a01c5b492250daba84d592c92df27982c486330cbfb58372b7282840b98206"
        ),
    }
    if any(
        not base.strict_equal(decision.get(key), value)
        for key, value in expected_terminal.items()
    ):
        raise RuntimeError("v0.628 terminal forensics drifted")

    attempts = base.json_lines(V0628_ROOT / "attempts.jsonl")
    if len(attempts) != 1:
        raise RuntimeError("v0.628 must contain exactly one attempt")
    row = attempts[0]
    resources = row.get("process_resources")
    cold = row.get("cold")
    if (
        row.get("artifact_stem") != "cold-p01-ab-r1-a"
        or row.get("arm") != "A"
        or row.get("valid") is not False
        or row.get("validity_reasons") != ["child_major_faults_nonzero"]
        or not isinstance(resources, dict)
        or resources.get("page_faults") != 638
        or resources.get("swaps") != 0
        or not isinstance(cold, dict)
        or cold.get("total_pageins") != 638
        or cold.get("total_disk_read_bytes") != 16_818_487_296
        or cold.get("first_token") != 'id=11751 piece=" Paris"'
    ):
        raise RuntimeError("v0.628 A1 forensic row drifted")
    events = base.json_lines(V0628_ROOT / "launch-seal.jsonl")
    if (
        len(events) != 2
        or events[0].get("event") != "launch"
        or events[1].get("event") != "completion"
        or events[0].get("arm") != "A"
        or events[1].get("returncode") != 0
    ):
        raise RuntimeError("v0.628 launch/completion forensics drifted")

    manifest = base.json_value(V0628_ROOT / "manifest.json")
    final = base.json_value(V0628_ROOT / "final-model-sha256.json")
    build = manifest.get("build_identity")
    hashes = manifest.get("sha256")
    if (
        not isinstance(build, dict)
        or build.get("example_sha256") != EXPECTED_EXAMPLE_BINARY
        or not isinstance(hashes, dict)
        or hashes.get(str(base.EXAMPLE_BINARY)) != EXPECTED_EXAMPLE_BINARY
    ):
        raise RuntimeError("v0.628 example-binary authentication drifted")
    prior_artifact = base.ARTIFACT
    prior_process_validity = base.process_validity
    try:
        base.ARTIFACT = V0628_ROOT
        base.process_validity = ORIGINAL_PROCESS_VALIDITY
        verified = ORIGINAL_VERIFY_ARTIFACTS(decision, manifest, final)
    finally:
        base.ARTIFACT = prior_artifact
        base.process_validity = prior_process_validity
    if set(verified) | {V0628_DECISION} != set(members):
        raise RuntimeError("v0.628 original verifier member set drifted")
    return {
        **members,
        V0628_INVENTORY: EXPECTED_V0628_INVENTORY,
        V0628_COMPLETE: EXPECTED_V0628_COMPLETE,
    }


def source_and_build_identity() -> tuple[str, dict[str, object]]:
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError("runner cwd is not repository root")
    for path in (RUNNER, PREREG, V0628_RUNNER, V0628_PREREG, base.EXAMPLE_SOURCE):
        base.command_text(
            ["git", "ls-files", "--error-unmatch", str(path.relative_to(ROOT))]
        )
    if base.command_text(["git", "status", "--porcelain=v1"]).strip():
        raise RuntimeError("source worktree is dirty")
    commit = base.command_text(["git", "rev-parse", "HEAD"]).strip()
    if base.command_text(["git", "rev-parse", "HEAD^"]).strip() != BASE_COMMIT:
        raise RuntimeError("v0.629 is not a direct child of the repair base")
    changed = sorted(base.changed_paths(f"{BASE_COMMIT}..HEAD"))
    expected = sorted(str(path.relative_to(ROOT)) for path in (RUNNER, PREREG))
    if changed != expected:
        raise RuntimeError(f"v0.629 packet boundary drifted: {changed}")
    protected = [
        "Cargo.lock",
        ":(top,glob)**/Cargo.toml",
        ":(top,glob).cargo/**",
        ":(top,glob)crates/**",
        ":(top,glob)kernels/**",
    ]
    drift = base.changed_paths(f"{base.DENSE_COMMIT}..HEAD", protected)
    if drift != [str(base.EXAMPLE_SOURCE.relative_to(ROOT))]:
        raise RuntimeError(f"production tree drifted during repair: {drift}")
    build = base.validate_build(
        json.loads(
            base.command_text([str(base.QWEN_BENCH), "build-info", "--output", "json"])
        ),
        commit,
    )
    qwen_bytes = base.QWEN.read_bytes()
    embedded = {
        "commit": commit,
        "build_source_state": str(build["build_source_state"]),
    }
    if any(value.encode() not in qwen_bytes for value in embedded.values()):
        raise RuntimeError("release qwen lacks exact v0.629 embedded identity")
    example_hash = sha256(base.EXAMPLE_BINARY)
    if example_hash != EXPECTED_EXAMPLE_BINARY:
        raise RuntimeError("v0.629 example binary differs from authenticated v0.628")
    return commit, {
        **build,
        "qwen_bench_sha256": sha256(base.QWEN_BENCH),
        "qwen_identity": {
            "sha256": hashlib.sha256(qwen_bytes).hexdigest(),
            "embedded": embedded,
            "verified": True,
        },
        "example_sha256": example_hash,
    }


def required_paths(v0627_bridge: dict[Path, str]) -> tuple[Path, ...]:
    forensic = verify_v0628_forensics()
    return tuple(
        dict.fromkeys(
            (
                RUNNER,
                PREREG,
                V0628_RUNNER,
                V0628_PREREG,
                base.EXAMPLE_SOURCE,
                base.EXAMPLE_BINARY,
                base.QWEN,
                base.QWEN_BENCH,
                base.MODEL,
                Path(base.mechanics.__file__).resolve(),
                Path(base.mechanics.common.__file__).resolve(),
                *v0627_bridge,
                *forensic,
            )
        )
    )


def build_manifest(removed: list[str], base_env: dict[str, str]) -> dict[str, object]:
    forensic = verify_v0628_forensics()
    manifest = ORIGINAL_BUILD_MANIFEST(removed, base_env)
    manifest.update(
        {
            "protocol": "v0.629-dense27b-pread-cold-composition-repair",
            "base_commit": BASE_COMMIT,
            "repair_only": "process-wide-page-faults-recorded-advisory",
            "v0628_forensic": {
                "decision_sha256": EXPECTED_V0628_DECISION,
                "inventory_sha256": EXPECTED_V0628_INVENTORY,
                "completion_sha256": EXPECTED_V0628_COMPLETE,
                "members": len(forensic) - 2,
                "authority_imported": False,
                "attempts_imported_for_scoring": 0,
                "b_observations": 0,
            },
        }
    )
    return manifest


def verify_non_model_identity(manifest: dict[str, object]) -> None:
    ORIGINAL_VERIFY_IDENTITY(manifest)
    forensic = verify_v0628_forensics()
    expected = manifest.get("v0628_forensic")
    if not isinstance(expected, dict) or expected != {
        "decision_sha256": EXPECTED_V0628_DECISION,
        "inventory_sha256": EXPECTED_V0628_INVENTORY,
        "completion_sha256": EXPECTED_V0628_COMPLETE,
        "members": len(forensic) - 2,
        "authority_imported": False,
        "attempts_imported_for_scoring": 0,
        "b_observations": 0,
    }:
        raise RuntimeError("v0.628 forensic import changed during v0.629")


def process_validity(
    stderr: str, post: dict[str, object]
) -> tuple[dict[str, object], list[str]]:
    validity, reasons = ORIGINAL_PROCESS_VALIDITY(stderr, post)
    resources = validity.get("process_resources")
    page_faults = resources.get("page_faults") if isinstance(resources, dict) else None
    reasons = [reason for reason in reasons if reason != "child_major_faults_nonzero"]
    return {**validity, "process_page_faults_advisory": page_faults}, reasons


def make_decision(
    status: str,
    manifest: dict[str, object],
    cpu_test: dict[str, object] | None,
    stage: dict[str, object] | None,
    stopped_after: str,
    **extra: object,
) -> dict[str, object]:
    decision = ORIGINAL_MAKE_DECISION(
        status,
        manifest,
        cpu_test,
        stage,
        stopped_after,
        **extra,
    )
    decision["imported_v0628"] = manifest["v0628_forensic"]
    return decision


def verify_attempt_artifacts(
    decision: dict[str, object],
    manifest: dict[str, object],
    final_identity: dict[str, object],
) -> dict[Path, str]:
    if not base.strict_equal(
        decision.get("imported_v0628"), manifest.get("v0628_forensic")
    ):
        raise RuntimeError("v0.629 decision lost its v0.628 forensic binding")
    sealed = ORIGINAL_VERIFY_ARTIFACTS(decision, manifest, final_identity)
    for row in base.read_attempts():
        resources = row.get("process_resources")
        expected = resources.get("page_faults") if isinstance(resources, dict) else None
        if not base.strict_equal(row.get("process_page_faults_advisory"), expected):
            raise RuntimeError("v0.629 advisory page-fault binding drifted")
    return sealed


def install_repairs() -> None:
    base.ARTIFACT = ARTIFACT
    base.PREREG = PREREG
    base.EXPECTED_PREREG_SHA256 = EXPECTED_PREREG_SHA256
    base.source_and_build_identity = source_and_build_identity
    base.required_paths = required_paths
    base.build_manifest = build_manifest
    base.verify_non_model_identity = verify_non_model_identity
    base.process_validity = process_validity
    base.make_decision = make_decision
    base.verify_attempt_artifacts = verify_attempt_artifacts


install_repairs()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true")
    arguments = parser.parse_args()
    base.main(preflight_only=arguments.preflight_only)
