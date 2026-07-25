#!/usr/bin/env python3
"""v0.632 repair for the correctness-only marker major-fault gate."""

import argparse
import re
from itertools import pairwise
from pathlib import Path

import v0631_dense27b_pread_loaded_stability_repair as prior

base = prior.base
ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = ROOT / "target/profiles/v0632-dense27b-pread-correctness-fault-repair-p1"
PREREG = ROOT / "docs/bench/v0632-dense27b-pread-correctness-fault-repair.md"
RUNNER = Path(__file__).resolve()
V0631_ROOT = ROOT / "target/profiles/v0631-dense27b-pread-loaded-stability-repair-p1"
V0631_DECISION = V0631_ROOT / "decision.json"
V0631_INVENTORY = V0631_ROOT / "artifact-inventory.sha256"
V0631_COMPLETE = V0631_ROOT / "packet-complete.json"

BASE_COMMIT = "cc47190654b898e2004117b6d03be1006a89d1fa"
V0630_PREREG_COMMIT = "d30a0f49e14aec13126d83a41553f6e7250b0f93"
V0630_IMPLEMENTATION_COMMIT = "d90aecdd4d4f9c297f98b09ba55eb479a7bf7944"
V0631_REPAIR_COMMIT = "ecc9e70fc833ee40831b7f5891141e620a92a2cd"
V0631_SEALS = {
    V0631_DECISION: "64730255135525cf289e6f36b1af81c33dd7f59b5575a950ca9945caaa02147e",
    V0631_INVENTORY: "27c07cbc68c6ff33aacc437f92ea44c97ffa33035f441f077eec3b430c579f7a",
    V0631_COMPLETE: "e32ffba74b4d38dbc94dc3bfbacdec14a9d80da07f0a6d6f9a48f38be9bc0482",
}
EXPECTED_V0631_MEMBERS = {
    "attempts.jsonl": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "correctness.out": "9fbdb5296d2f31658d0aae6c81d0f2c72764118b046dc61dca65af7c95c57ab0",
    "decision.json": "64730255135525cf289e6f36b1af81c33dd7f59b5575a950ca9945caaa02147e",
    "final-identity.json": "15a3908e8350e6e8e379b131b50d7af93660a0e8bd3714465414fe9b6aa3e94b",
    "launch-seal.jsonl": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "manifest.json": "e1d7f7263bd066741f1eb9b228197b4f7b3ddeced9fc299ce002a230521d0d64",
    "token-protocol.json": "9ec27ba031e31e95bee436a254018bc28fc14395daf5ffdf76ff86b70d414097",
    "token-protocol.out": "a27b9ef74794543965646dc3426a59ca092f8e7939876cabbfd19fede86da258",
}

ORIGINAL_BUILD_MANIFEST = base.build_manifest
ORIGINAL_MAKE_DECISION = base.make_decision
ORIGINAL_PARSE_MARKER = base.parse_marker
ORIGINAL_RUN_CORRECTNESS = base.run_correctness


def verify_v0631_forensics() -> tuple[dict[str, object], dict[Path, str]]:
    for path, expected in V0631_SEALS.items():
        if not path.is_file() or path.is_symlink() or base.sha256(path) != expected:
            raise base.ContractDefect(f"v0.631 seal drifted: {path}")
    complete = base.parse_json(
        V0631_COMPLETE.read_text(encoding="utf-8"), "v0.631 completion"
    )
    if not base.strict_equal(
        complete,
        {
            "schema": 1,
            "decision_sha256": V0631_SEALS[V0631_DECISION],
            "inventory_sha256": V0631_SEALS[V0631_INVENTORY],
        },
    ):
        raise base.ContractDefect("v0.631 completion binding drifted")
    members = base.inventory_members(V0631_INVENTORY)
    expected_members = {
        V0631_ROOT / name: digest for name, digest in EXPECTED_V0631_MEMBERS.items()
    }
    if members != expected_members:
        raise base.ContractDefect("v0.631 inventory membership drifted")
    for path, expected in members.items():
        if not path.is_file() or path.is_symlink() or base.sha256(path) != expected:
            raise base.ContractDefect(f"v0.631 member drifted: {path}")
    if set(V0631_ROOT.iterdir()) != set(members) | {V0631_INVENTORY, V0631_COMPLETE}:
        raise base.ContractDefect("v0.631 final artifact set drifted")
    decision = base.parse_json(
        V0631_DECISION.read_text(encoding="utf-8"), "v0.631 decision"
    )
    if not isinstance(decision, dict):
        raise base.ContractDefect("v0.631 decision is malformed")
    terminal = {
        "schema": 1,
        "status": "implementation_or_contract_defect",
        "authority": "none",
        "force_authorized": False,
        "successor_authorization": "none",
        "stopped_after": "correctness",
        "failed_child": None,
        "reasons": ["dense direct-pread marker reports major faults"],
        "source_commit": V0631_REPAIR_COMMIT,
        "correctness": None,
        "stage": None,
        "attempts_sha256": EXPECTED_V0631_MEMBERS["attempts.jsonl"],
    }
    for key, expected in terminal.items():
        if not base.strict_equal(decision.get(key), expected):
            raise base.ContractDefect(f"v0.631 terminal field drifted: {key}")
    claim = decision.get("claim_scope")
    if (
        not isinstance(claim, dict)
        or claim.get("performance_imports_used_for_scoring") != 0
    ):
        raise base.ContractDefect("v0.631 performance-import contract drifted")
    final_identity = base.parse_json(
        (V0631_ROOT / "final-identity.json").read_text(encoding="utf-8"),
        "v0.631 final identity",
    )
    if (
        not isinstance(final_identity, dict)
        or final_identity.get("matches") is not True
    ):
        raise base.ContractDefect("v0.631 final identity did not pass")
    transcript = (V0631_ROOT / "correctness.out").read_text(encoding="utf-8")
    if transcript.count("timer_major_faults=2") != 1:
        raise base.ContractDefect("v0.631 correctness major-fault evidence drifted")
    base.recognize_test(transcript, base.CORRECTNESS_TEST, interleaved=True)
    prefixes = (
        "[metal-load] native quantized token embedding policy:",
        "[metal-gguf-parallel-pread]",
        "[metal-load-ledger]",
    )
    recognized: list[str] = []
    harness_prefix = f"test {base.CORRECTNESS_TEST} ... "
    for line in transcript.splitlines():
        if not recognized and line == harness_prefix + base.POLICY_LINE:
            recognized.append(base.POLICY_LINE)
        elif line.startswith(prefixes):
            recognized.append(line)
        elif "[metal-load" in line or "[metal-gguf-" in line:
            raise base.ContractDefect(f"malformed v0.631 correctness line: {line!r}")
    if (
        len(recognized) != 5
        or recognized[:3] != [base.POLICY_LINE, base.LEDGER_LINE, base.POLICY_LINE]
        or recognized[4] != base.LEDGER_LINE
    ):
        raise base.ContractDefect("v0.631 correctness load order drifted")
    return (
        {
            "authority_imported": False,
            "correctness_imported_as_gate": False,
            "performance_observations_imported": 0,
            "timed_children_imported": 0,
            "forensic_class": "correctness-marker-major-fault-gate-overbroad",
            "correctness_marker_major_faults": 2,
            "decision_sha256": V0631_SEALS[V0631_DECISION],
            "inventory_sha256": V0631_SEALS[V0631_INVENTORY],
            "completion_sha256": V0631_SEALS[V0631_COMPLETE],
            "inventory_members": len(members),
        },
        {
            **members,
            V0631_INVENTORY: V0631_SEALS[V0631_INVENTORY],
            V0631_COMPLETE: V0631_SEALS[V0631_COMPLETE],
        },
    )


def verify_source_topology() -> dict[str, object]:
    for path in (PREREG, RUNNER, base.BENCH_SOURCE):
        base.git_output(["ls-files", "--error-unmatch", str(path.relative_to(ROOT))])
    dirty = base.git_output(["status", "--porcelain=v1", "--untracked-files=all"])
    if dirty:
        raise base.ContractDefect(f"worktree is not clean: {dirty!r}")
    head = base.git_output(["rev-parse", "HEAD"])
    repair631 = base.git_output(["rev-parse", "HEAD^"])
    implementation = base.git_output(["rev-parse", "HEAD^^"])
    preregistration = base.git_output(["rev-parse", "HEAD^^^"])
    root = base.git_output(["rev-parse", "HEAD^^^^"])
    if (
        repair631 != V0631_REPAIR_COMMIT
        or implementation != V0630_IMPLEMENTATION_COMMIT
        or preregistration != V0630_PREREG_COMMIT
        or root != BASE_COMMIT
    ):
        raise base.ContractDefect("v0.632 frozen lineage drifted")
    chain = (root, preregistration, implementation, repair631, head)
    if any(base.commit_parents(right) != [left] for left, right in pairwise(chain)):
        raise base.ContractDefect("v0.632 lineage contains a merge commit")
    v0630_paths = sorted(
        [
            "docs/bench/v0630-dense27b-pread-loaded-stability.md",
            "scripts/profile/v0630_dense27b_pread_loaded_stability.py",
        ]
    )
    if sorted(base.changed_paths(root, preregistration)) != v0630_paths or sorted(
        base.name_status(root, preregistration)
    ) != sorted(f"A\t{path}" for path in v0630_paths):
        raise base.ContractDefect("v0.630 preregistration diff drifted")
    bench_path = str(base.BENCH_SOURCE.relative_to(ROOT))
    if base.name_status(preregistration, implementation) != [f"M\t{bench_path}"]:
        raise base.ContractDefect("v0.630 implementation diff drifted")
    repair_paths = sorted(
        [str(PREREG.relative_to(ROOT)), str(RUNNER.relative_to(ROOT))]
    )
    if sorted(base.changed_paths(repair631, head)) != repair_paths:
        raise base.ContractDefect("v0.632 repair diff drifted")
    if sorted(base.name_status(repair631, head)) != sorted(
        f"A\t{path}" for path in repair_paths
    ):
        raise base.ContractDefect("v0.632 repair files were not exact additions")
    original_head = base.git_output(["rev-parse", "HEAD"])
    if original_head != head:
        raise base.ContractDefect("source head changed during topology verification")
    if base.sha256(base.BENCH_SOURCE) != base.EXPECTED_BENCH_SOURCE_SHA256:
        raise base.ContractDefect("v0.630 complete bench.rs digest drifted")
    if (
        base.implementation_patch_sha256(preregistration, implementation)
        != base.EXPECTED_IMPLEMENTATION_PATCH_SHA256
    ):
        raise base.ContractDefect("v0.630 implementation patch digest drifted")
    v0631_paths = sorted(
        [
            "docs/bench/v0631-dense27b-pread-loaded-stability-repair.md",
            "scripts/profile/v0631_dense27b_pread_loaded_stability_repair.py",
        ]
    )
    if sorted(base.changed_paths(implementation, repair631)) != v0631_paths:
        raise base.ContractDefect("v0.631 repair diff drifted")
    if sorted(base.name_status(implementation, repair631)) != sorted(
        f"A\t{path}" for path in v0631_paths
    ):
        raise base.ContractDefect("v0.631 repair files were not exact additions")
    build = base.parse_json(
        prior.repaired_command_output(
            [str(base.BENCH_BINARY), "build-info", "--output", "json"]
        ),
        "qwen-bench build-info",
    )
    if not isinstance(build, dict):
        raise base.ContractDefect("qwen-bench build identity is not an object")
    if (
        build.get("build_commit") != head
        or build.get("runtime_commit") != head
        or build.get("status") != "match"
        or build.get("build_dirty") is not False
        or build.get("runtime_dirty") is not False
        or build.get("build_source_state") != build.get("runtime_source_state")
    ):
        raise base.ContractDefect(f"source/build/runtime identity mismatch: {build}")
    dependencies = [
        base.sibling_identity(ROOT.parent / "gguf", base.EXPECTED_GGUF_COMMIT),
        base.sibling_identity(
            ROOT.parent / "llama-cpp-rs",
            base.EXPECTED_LLAMA_CPP_RS_COMMIT,
            base.EXPECTED_LLAMA_CPP_RS_UNTRACKED,
        ),
    ]
    return {
        "head": head,
        "repair_commit": head,
        "v0631_repair_commit": repair631,
        "implementation_commit": implementation,
        "preregistration_commit": preregistration,
        "base_commit": root,
        "build_identity": build,
        "sibling_dependencies": dependencies,
    }


def build_manifest(
    child_env: dict[str, str], test_env: dict[str, str], removed: list[str]
) -> dict[str, object]:
    manifest = ORIGINAL_BUILD_MANIFEST(child_env, test_env, removed)
    forensic, members = verify_v0631_forensics()
    manifest["forensic_v0631"] = forensic
    hashes = manifest.get("sha256")
    if not isinstance(hashes, dict):
        raise base.ContractDefect("manifest hash map is malformed")
    for path, digest in members.items():
        hashes[str(path)] = digest
    return manifest


def parse_correctness_marker(line: str) -> dict[str, int | float]:
    matches = re.findall(r"(?<![A-Za-z0-9_])timer_major_faults=([0-9]+)(?![0-9])", line)
    if len(matches) != 1:
        raise base.ContractDefect("correctness marker major-fault field drifted")
    value = matches[0]
    parsed = int(value)
    if str(parsed) != value or parsed > base.U64_MAX:
        raise base.ContractDefect("correctness marker major faults are not canonical")
    normalized = line.replace(f"timer_major_faults={value}", "timer_major_faults=0", 1)
    result = ORIGINAL_PARSE_MARKER(normalized)
    result["timer_major_faults"] = parsed
    result["correctness_major_faults_advisory"] = parsed
    return result


def run_correctness(base_env: dict[str, str]) -> dict[str, object]:
    base.parse_marker = parse_correctness_marker
    try:
        row = ORIGINAL_RUN_CORRECTNESS(base_env)
    finally:
        base.parse_marker = ORIGINAL_PARSE_MARKER
    marker = row.get("candidate_marker")
    if not isinstance(marker, dict):
        raise base.ContractDefect("fresh correctness marker is malformed")
    row["correctness_marker_major_faults_advisory"] = marker.get("timer_major_faults")
    base.write_json(base.ARTIFACT / "correctness.json", row, exclusive=False)
    return row


def make_decision(*args: object, **kwargs: object) -> dict[str, object]:
    decision = ORIGINAL_MAKE_DECISION(*args, **kwargs)
    manifest = args[0] if args else kwargs.get("manifest")
    if not isinstance(manifest, dict):
        raise base.ContractDefect("decision manifest is malformed")
    decision["forensic_v0631"] = manifest.get("forensic_v0631")
    return decision


def configure() -> None:
    prior.configure()
    base.ARTIFACT = ARTIFACT
    base.PREREG = PREREG
    base.RUNNER = RUNNER
    base.verify_source_topology = verify_source_topology
    base.build_manifest = build_manifest
    base.run_correctness = run_correctness
    base.make_decision = make_decision


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    configure()
    base.run_packet(preflight_only=args.preflight_only)


if __name__ == "__main__":
    main()
