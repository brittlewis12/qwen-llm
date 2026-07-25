#!/usr/bin/env python3
"""v0.627 repair for the v0.626 correctness-harness recognizer."""

import argparse
import hashlib
import os
from pathlib import Path
import re
import signal
import subprocess
import time

import v0626_dense27b_parallel_pread_fresh as base


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = ROOT / "target/profiles/v0627-dense27b-parallel-pread-fresh-repair-p1"
PREREG = ROOT / "docs/bench/v0627-dense27b-parallel-pread-fresh-repair.md"
RUNNER = Path(__file__).resolve()
V0626_PREREG = ROOT / "docs/bench/v0626-dense27b-parallel-pread-fresh.md"
V0626_RUNNER = ROOT / "scripts/profile/v0626_dense27b_parallel_pread_fresh.py"
V0626_ROOT = ROOT / "target/profiles/v0626-dense27b-parallel-pread-fresh-p1"
V0626_DECISION = V0626_ROOT / "decision.json"
V0626_INVENTORY = V0626_ROOT / "artifact-inventory.sha256"
V0626_COMPLETE = V0626_ROOT / "packet-complete.json"

PRODUCTION_COMMIT = "7fb0488a5c075d8c62c8f9802352f717ec95b085"
EXPECTED_PREREG_SHA256 = (
    "5fcd787dcda1544de062f82d7a2ef42ea460e683870443ff089a909b7467de1c"
)
EXPECTED_V0626_DECISION_SHA256 = (
    "d2fe54d9b7a7c81adae9d85611f9a95e12a51547539d3e576993184f64471f5a"
)
EXPECTED_V0626_INVENTORY_SHA256 = (
    "c596fa38affa3bcca3717b2878b7b2a121b41cf5b04b9e1d779c59ca2a689b87"
)
EXPECTED_V0626_COMPLETE_SHA256 = (
    "8563e7929c07925ffc1fa2846776dd6d78cc3c7645dfc4558c9ee99cda780d03"
)

common = base.common
mechanics = base.mechanics
BASE_BUILD_MANIFEST = base.build_manifest
BASE_REQUIRED_MANIFEST_PATHS = base.required_manifest_paths
BASE_VERIFY_NON_MODEL_IDENTITY = base.verify_non_model_identity


def sha256(path: Path) -> str:
    return common.sha256_file(path)


def anchored_matches(pattern: str, text: str) -> list[re.Match[str]]:
    return list(re.finditer(pattern, text, re.MULTILINE))


def recognize_test_output(text: str, test_name: str, *, correctness: bool) -> None:
    escaped = re.escape(test_name)
    starts = anchored_matches(rf"^test {escaped} \.\.\. ", text)
    summaries = anchored_matches(
        r"^test result: ok\. 1 passed; 0 failed; 0 ignored; 0 measured; "
        r"\d+ filtered out; finished in [0-9]+(?:\.[0-9]+)?s\r?$",
        text,
    )
    contiguous = anchored_matches(rf"^test {escaped} \.\.\. ok\r?$", text)
    if (
        len(starts) != 1
        or len(summaries) != 1
        or starts[0].start() >= summaries[0].start()
    ):
        raise RuntimeError(f"test harness identity is not unique: {test_name}")
    if not correctness:
        if len(contiguous) != 1:
            raise RuntimeError(
                f"CPU selector result is not strict contiguous: {test_name}"
            )
        return
    standalone = anchored_matches(r"^ok\r?$", text)
    contiguous_form = len(contiguous) == 1 and not standalone
    interleaved_form = (
        not contiguous
        and len(standalone) == 1
        and starts[0].end() <= standalone[0].start() < summaries[0].start()
    )
    if not (contiguous_form or interleaved_form):
        raise RuntimeError(
            f"correctness result form is not uniquely recognized: {test_name}"
        )


def run_test_command(
    command: list[str], output_path: Path, base_env: dict[str, str], test_name: str
) -> dict[str, object]:
    deferred_sigint: list[int] = []

    def defer_sigint(signum: int, _frame: object) -> None:
        deferred_sigint.append(signum)

    prior = signal.signal(signal.SIGINT, defer_sigint)
    started = time.perf_counter()
    result = None
    execution_error = None
    durability_error = None
    try:
        try:
            with output_path.open("xb") as output:
                try:
                    result = subprocess.run(
                        command,
                        cwd=ROOT,
                        env=base_env,
                        stdout=output,
                        stderr=subprocess.STDOUT,
                        check=False,
                    )
                except BaseException as error:
                    execution_error = error
                try:
                    output.flush()
                    os.fsync(output.fileno())
                except OSError as error:
                    durability_error = error
        except OSError as error:
            durability_error = error
    finally:
        signal.signal(signal.SIGINT, prior)
    if durability_error is not None:
        raise mechanics.UnsealedPacket(
            f"test output durability failed: {durability_error}"
        )
    stage = "correctness" if "--ignored" in command else "cpu-selectors"
    if deferred_sigint:
        raise mechanics.InconclusivePacket(
            stage, test_name, [f"operator_sigint_deferred={len(deferred_sigint)}"]
        )
    if execution_error is not None or result is None:
        raise mechanics.UnsealedPacket(
            f"test execution did not return: {execution_error}"
        )
    text = output_path.read_text(encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(f"exact release test returned nonzero: {test_name}")
    recognize_test_output(text, test_name, correctness=stage == "correctness")
    return {
        "test": test_name,
        "command": command,
        "wall_ms": (time.perf_counter() - started) * 1e3,
        "output_path": str(output_path),
        "output_sha256": sha256(output_path),
        "passed": True,
    }


def verify_v0626_forensic_seal() -> dict[Path, str]:
    expected_seals = {
        V0626_DECISION: EXPECTED_V0626_DECISION_SHA256,
        V0626_INVENTORY: EXPECTED_V0626_INVENTORY_SHA256,
        V0626_COMPLETE: EXPECTED_V0626_COMPLETE_SHA256,
    }
    for path, expected in expected_seals.items():
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"v0.626 sealed evidence drifted: {path}")
    complete = base.parse_json(V0626_COMPLETE.read_text(encoding="utf-8"))
    if complete != {
        "schema": 1,
        "decision_sha256": EXPECTED_V0626_DECISION_SHA256,
        "inventory_sha256": EXPECTED_V0626_INVENTORY_SHA256,
    }:
        raise RuntimeError("v0.626 completion binding drifted")
    members = base.inventory_members(V0626_INVENTORY, "v0.626")
    if len(members) != 9:
        raise RuntimeError("v0.626 inventory must contain exactly nine members")
    if members.get(V0626_DECISION) != EXPECTED_V0626_DECISION_SHA256:
        raise RuntimeError("v0.626 inventory does not bind its decision")
    exact_paths = set(members) | {V0626_INVENTORY, V0626_COMPLETE}
    observed_paths = set(V0626_ROOT.iterdir())
    if observed_paths != exact_paths:
        raise RuntimeError(
            "v0.626 exact artifact set or fresh-artifact absence drifted"
        )

    decision = base.parse_json(V0626_DECISION.read_text(encoding="utf-8"))
    if not isinstance(decision, dict):
        raise RuntimeError("v0.626 decision is malformed")
    terminal = {
        "schema": 1,
        "status": "implementation_or_contract_defect",
        "authority": "none",
        "force_authorized": False,
        "successor_authorization": "none",
        "stopped_after": "correctness",
        "source_commit": PRODUCTION_COMMIT,
        "attempts_sha256": None,
        "correctness": None,
        "stages": {},
        "error_type": "RuntimeError",
        "error": (
            "exact release test failed or was not uniquely selected: "
            + base.CORRECTNESS_TEST
        ),
    }
    if any(decision.get(key) != value for key, value in terminal.items()):
        raise RuntimeError("v0.626 terminal fields drifted")
    verify_v0626_selectors(decision, members)
    verify_v0626_correctness(members)
    verify_v0626_manifest(members)
    return {
        **members,
        V0626_INVENTORY: sha256(V0626_INVENTORY),
        V0626_COMPLETE: sha256(V0626_COMPLETE),
    }


def verify_v0626_selectors(
    decision: dict[str, object], members: dict[Path, str]
) -> None:
    metadata_path = V0626_ROOT / "cpu-selectors.json"
    metadata = base.parse_json(metadata_path.read_text(encoding="utf-8"))
    selectors = decision.get("cpu_selectors")
    if selectors != metadata or not isinstance(selectors, dict):
        raise RuntimeError("v0.626 selector metadata binding drifted")
    rows = selectors.get("tests")
    if selectors.get("passed") is not True or not isinstance(rows, list):
        raise RuntimeError("v0.626 selectors are not passed")
    if len(rows) != len(base.CPU_SELECTOR_TESTS):
        raise RuntimeError("v0.626 selector count drifted")
    for index, (row, name) in enumerate(
        zip(rows, base.CPU_SELECTOR_TESTS, strict=True), 1
    ):
        output = V0626_ROOT / f"cpu-selector-{index:02d}.out"
        if (
            not isinstance(row, dict)
            or row.get("test") != name
            or row.get("passed") is not True
            or row.get("command") != base.cargo_test_command(name, ignored=False)
            or row.get("output_sha256") != members.get(output)
        ):
            raise RuntimeError(f"v0.626 selector identity drifted: {name}")
        recognize_test_output(
            output.read_text(encoding="utf-8"), name, correctness=False
        )


def verify_v0626_correctness(members: dict[Path, str]) -> None:
    output = V0626_ROOT / "correctness.out"
    if output not in members:
        raise RuntimeError("v0.626 correctness transcript is not inventory-bound")
    text = output.read_text(encoding="utf-8")
    recognize_test_output(text, base.CORRECTNESS_TEST, correctness=True)
    if text.count("[metal-gguf-") != 1:
        raise RuntimeError("v0.626 correctness storage-marker count drifted")
    if text.count("[metal-gguf-parallel-pread]") != 1:
        raise RuntimeError("v0.626 correctness direct-pread marker count drifted")
    lines = base.extract_correctness_load_lines(text)
    expected = [base.POLICY_LINE, base.LEDGER_LINE, base.POLICY_LINE]
    if len(lines) != 5 or lines[:3] != expected or lines[4] != base.LEDGER_LINE:
        raise RuntimeError("v0.626 correctness five-line load order drifted")
    base.parse_marker(lines[3])


def verify_v0626_manifest(members: dict[Path, str]) -> None:
    path = V0626_ROOT / "manifest.json"
    if path not in members:
        raise RuntimeError("v0.626 manifest is not inventory-bound")
    manifest = base.parse_json(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise RuntimeError("v0.626 manifest is malformed")
    build = manifest.get("build_identity")
    if (
        manifest.get("source_commit") != PRODUCTION_COMMIT
        or manifest.get("authority") != "none"
        or not isinstance(build, dict)
        or build.get("build_commit") != PRODUCTION_COMMIT
        or build.get("runtime_commit") != PRODUCTION_COMMIT
        or build.get("build_dirty") is not False
        or build.get("runtime_dirty") is not False
        or build.get("build_source_state") != build.get("runtime_source_state")
    ):
        raise RuntimeError("v0.626 manifest source identity drifted")
    qwen = build.get("qwen_identity")
    embedded = qwen.get("embedded") if isinstance(qwen, dict) else None
    if not isinstance(embedded, dict) or embedded != {
        "commit": PRODUCTION_COMMIT,
        "build_source_state": build.get("build_source_state"),
    }:
        raise RuntimeError("v0.626 embedded qwen identity drifted")


def git_changed_paths(revision: str, paths: list[str] | None = None) -> list[str]:
    command = ["git", "diff", "--name-only", revision]
    if paths:
        command.extend(["--", *paths])
    return [line for line in base.command_text(command).splitlines() if line]


def source_and_build_identity() -> tuple[str, dict[str, object]]:
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError("runner working directory is not repository root")
    tracked = (
        RUNNER,
        PREREG,
        V0626_RUNNER,
        V0626_PREREG,
        base.MECHANICS,
        base.COMMON_RUNNER,
        base.PROMPT,
    )
    for path in tracked:
        base.command_text(
            ["git", "ls-files", "--error-unmatch", str(path.relative_to(ROOT))]
        )
    if base.command_text(["git", "status", "--porcelain=v1"]).strip():
        raise RuntimeError("source worktree is dirty")
    commit = base.command_text(["git", "rev-parse", "HEAD"]).strip()
    parent = base.command_text(["git", "rev-parse", "HEAD^"]).strip()
    if parent != PRODUCTION_COMMIT:
        raise RuntimeError("v0.627 HEAD is not a direct child of production")
    packet_paths = sorted(git_changed_paths(f"{PRODUCTION_COMMIT}..HEAD"))
    expected_paths = sorted(str(path.relative_to(ROOT)) for path in (PREREG, RUNNER))
    if packet_paths != expected_paths:
        raise RuntimeError(f"v0.627 packet path boundary drifted: {packet_paths}")
    production_paths = [
        "Cargo.lock",
        ":(top,glob)**/Cargo.toml",
        ":(top,glob).cargo/**",
        ":(top,glob)crates/**",
        ":(top,glob)kernels/**",
    ]
    if git_changed_paths(f"{PRODUCTION_COMMIT}..HEAD", production_paths):
        raise RuntimeError("production Cargo/.cargo/crates/kernels tree drifted")
    bench = base.validate_build_identity(
        base.parse_json(
            base.command_text(
                [str(base.BENCH_BINARY), "build-info", "--output", "json"]
            )
        ),
        commit,
        "qwen-bench",
    )
    qwen_bytes = base.CLI_BINARY.read_bytes()
    embedded = {
        "commit": commit,
        "build_source_state": str(bench["build_source_state"]),
    }
    if any(value.encode() not in qwen_bytes for value in embedded.values()):
        raise RuntimeError("release qwen does not embed v0.627 source identity")
    return commit, {
        **bench,
        "qwen_identity": {
            "sha256": hashlib.sha256(qwen_bytes).hexdigest(),
            "embedded": embedded,
            "verified": True,
        },
        "qwen_bench_sha256": sha256(base.BENCH_BINARY),
    }


def required_manifest_paths(
    v0603_members: dict[Path, str], v0593_members: dict[Path, str]
) -> tuple[Path, ...]:
    inherited = BASE_REQUIRED_MANIFEST_PATHS(v0603_members, v0593_members)
    forensic = verify_v0626_forensic_seal()
    return tuple(
        dict.fromkeys(
            (RUNNER, PREREG, V0626_RUNNER, V0626_PREREG, *inherited, *forensic)
        )
    )


def build_manifest(
    removed_environment: list[str], base_env: dict[str, str]
) -> dict[str, object]:
    forensic = verify_v0626_forensic_seal()
    manifest = BASE_BUILD_MANIFEST(removed_environment, base_env)
    manifest.update(
        {
            "protocol": "v0.627-dense27b-parallel-pread-fresh-repair",
            "base_commit": PRODUCTION_COMMIT,
            "repair_only": "correctness-harness-interleaved-ok-recognition",
            "v0626_forensic_members": {
                str(path): digest for path, digest in forensic.items()
            },
            "v0626_authority_imported": False,
            "v0626_fresh_children": 0,
        }
    )
    return manifest


def verify_non_model_identity(manifest: dict[str, object]) -> None:
    BASE_VERIFY_NON_MODEL_IDENTITY(manifest)
    forensic = verify_v0626_forensic_seal()
    expected = manifest.get("v0626_forensic_members")
    if not isinstance(expected, dict) or expected != {
        str(path): digest for path, digest in forensic.items()
    }:
        raise RuntimeError("v0.626 forensic evidence changed during v0.627")


def install_repairs() -> None:
    base.ARTIFACT = ARTIFACT
    base.PREREG = PREREG
    base.BASE_COMMIT = PRODUCTION_COMMIT
    base.EXPECTED_PREREG_SHA256 = EXPECTED_PREREG_SHA256
    base.run_test_command = run_test_command
    base.source_and_build_identity = source_and_build_identity
    base.required_manifest_paths = required_manifest_paths
    base.build_manifest = build_manifest
    base.verify_non_model_identity = verify_non_model_identity
    for name, value in {
        "ARTIFACT": ARTIFACT,
        "PREREG": PREREG,
        "source_and_build_identity": source_and_build_identity,
        "verify_non_model_identity": verify_non_model_identity,
        "verify_packet_identity": base.verify_packet_identity,
    }.items():
        setattr(mechanics, name, value)


install_repairs()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    base.main(preflight_only=args.preflight_only)
