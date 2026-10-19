#!/usr/bin/env python3
"""v0.631 repair for the v0.630 device-description preflight command."""

import argparse
from pathlib import Path

import v0630_dense27b_pread_loaded_stability as base

ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = ROOT / "target/profiles/v0631-dense27b-pread-loaded-stability-repair-p1"
PREREG = ROOT / "docs/bench/v0631-dense27b-pread-loaded-stability-repair.md"
RUNNER = Path(__file__).resolve()

BASE_COMMIT = "cc47190654b898e2004117b6d03be1006a89d1fa"
V0630_PREREG_COMMIT = "d30a0f49e14aec13126d83a41553f6e7250b0f93"
V0630_IMPLEMENTATION_COMMIT = "d90aecdd4d4f9c297f98b09ba55eb479a7bf7944"
EXPECTED_DEVICE_LINE = (
    "device\tApple M4 Max | unified_memory=true | max_threadgroup_memory=32768 bytes"
)

ORIGINAL_COMMAND_OUTPUT = base.command_output


def repaired_command_output(
    command: list[str], *, env: dict[str, str] | None = None
) -> str:
    if command == [str(base.BENCH_BINARY), "metal-info"]:
        output = ORIGINAL_COMMAND_OUTPUT(
            [str(base.BENCH_BINARY), "metal-counters"], env=env
        )
        lines = output.splitlines()
        if (
            len(lines) != 5
            or lines[0] != EXPECTED_DEVICE_LINE
            or lines[1] != "sampling\tstage=true\tdispatch=false\tblit=false"
            or lines[2] != "counter_sets\t1"
            or lines[3] != "set\ttimestamp\tcounters=1\tsample_buffer=ok"
            or lines[4] != "counter\ttimestamp\tGPUTimestamp"
        ):
            raise base.ContractDefect(
                f"v0.631 Metal-counter device protocol drifted: {lines!r}"
            )
        return base.EXPECTED_DEVICE + "\n"
    return ORIGINAL_COMMAND_OUTPUT(command, env=env)


def verify_source_topology() -> dict[str, object]:
    for path in (PREREG, RUNNER, base.BENCH_SOURCE):
        relative = path.relative_to(ROOT)
        base.git_output(["ls-files", "--error-unmatch", str(relative)])
    dirty = base.git_output(["status", "--porcelain=v1", "--untracked-files=all"])
    if dirty:
        raise base.ContractDefect(f"worktree is not clean: {dirty!r}")
    head = base.git_output(["rev-parse", "HEAD"])
    implementation = base.git_output(["rev-parse", "HEAD^"])
    preregistration = base.git_output(["rev-parse", "HEAD^^"])
    root = base.git_output(["rev-parse", "HEAD^^^"])
    if (
        implementation != V0630_IMPLEMENTATION_COMMIT
        or preregistration != V0630_PREREG_COMMIT
        or root != BASE_COMMIT
    ):
        raise base.ContractDefect("v0.631 frozen commit lineage drifted")
    if (
        base.commit_parents(preregistration) != [root]
        or base.commit_parents(implementation) != [preregistration]
        or base.commit_parents(head) != [implementation]
    ):
        raise base.ContractDefect("v0.631 lineage contains a merge commit")
    v0630_paths = sorted(
        [
            "docs/bench/v0630-dense27b-pread-loaded-stability.md",
            "scripts/profile/v0630_dense27b_pread_loaded_stability.py",
        ]
    )
    if sorted(base.changed_paths(root, preregistration)) != v0630_paths:
        raise base.ContractDefect("v0.630 preregistration diff drifted")
    if sorted(base.name_status(root, preregistration)) != sorted(
        f"A\t{path}" for path in v0630_paths
    ):
        raise base.ContractDefect("v0.630 preregistration files were not additions")
    bench_path = str(base.BENCH_SOURCE.relative_to(ROOT))
    if base.name_status(preregistration, implementation) != [f"M\t{bench_path}"]:
        raise base.ContractDefect("v0.630 implementation diff drifted")
    repair_paths = sorted(
        [str(PREREG.relative_to(ROOT)), str(RUNNER.relative_to(ROOT))]
    )
    if sorted(base.changed_paths(implementation, head)) != repair_paths:
        raise base.ContractDefect("v0.631 repair diff drifted")
    if sorted(base.name_status(implementation, head)) != sorted(
        f"A\t{path}" for path in repair_paths
    ):
        raise base.ContractDefect("v0.631 repair files were not exact additions")
    if base.sha256(base.BENCH_SOURCE) != base.EXPECTED_BENCH_SOURCE_SHA256:
        raise base.ContractDefect("v0.630 complete bench.rs digest drifted")
    if (
        base.implementation_patch_sha256(preregistration, implementation)
        != base.EXPECTED_IMPLEMENTATION_PATCH_SHA256
    ):
        raise base.ContractDefect("v0.630 implementation patch digest drifted")
    build = base.parse_json(
        repaired_command_output(
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
        "implementation_commit": implementation,
        "preregistration_commit": preregistration,
        "base_commit": root,
        "build_identity": build,
        "sibling_dependencies": dependencies,
    }


def configure() -> None:
    base.ARTIFACT = ARTIFACT
    base.PREREG = PREREG
    base.RUNNER = RUNNER
    base.command_output = repaired_command_output
    base.verify_source_topology = verify_source_topology


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    configure()
    base.run_packet(preflight_only=args.preflight_only)


if __name__ == "__main__":
    main()
