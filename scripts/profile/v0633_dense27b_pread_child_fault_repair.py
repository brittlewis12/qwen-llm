#!/usr/bin/env python3
"""v0.633 repair for process-wide timed-child page-fault attribution."""

import argparse
import os
import signal
import subprocess
import time
from itertools import pairwise
from pathlib import Path

import v0632_dense27b_pread_correctness_fault_repair as prior

base = prior.base
ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = ROOT / "target/profiles/v0633-dense27b-pread-child-fault-repair-p1"
PREREG = ROOT / "docs/bench/v0633-dense27b-pread-child-fault-repair.md"
RUNNER = Path(__file__).resolve()
V0632_ROOT = ROOT / "target/profiles/v0632-dense27b-pread-correctness-fault-repair-p1"
V0632_DECISION = V0632_ROOT / "decision.json"
V0632_INVENTORY = V0632_ROOT / "artifact-inventory.sha256"
V0632_COMPLETE = V0632_ROOT / "packet-complete.json"

BASE_COMMIT = "cc47190654b898e2004117b6d03be1006a89d1fa"
V0630_PREREG_COMMIT = "d30a0f49e14aec13126d83a41553f6e7250b0f93"
V0630_IMPLEMENTATION_COMMIT = "d90aecdd4d4f9c297f98b09ba55eb479a7bf7944"
V0631_REPAIR_COMMIT = "ecc9e70fc833ee40831b7f5891141e620a92a2cd"
V0632_REPAIR_COMMIT = "2e71e1a8caba14ddc0fcc5ea3232354dbab86504"
V0632_SEALS = {
    V0632_DECISION: "561399219b2ccb066d584932b5ca1a72ca42f56757fc7fa9d6f68b2a8ae5d353",
    V0632_INVENTORY: "e3ce48c108578401ebf0e1950b683aa14a7273394160a4183da8f0c2623e01eb",
    V0632_COMPLETE: "535f82c9c2d032b29e944cee4b133b1058d3741b71bdbb8b659bf7f7ae178366",
}
EXPECTED_V0632_MEMBERS = {
    "attempts.jsonl": "bb5d2d607e16df57978b2bf019ea0adbaa2b4ae8fbc8cdf0c16762d714406375",
    "correctness.json": "117315601f95f7fa46005e793bd726cdb4e561a722541b6900c70422bb71438c",
    "correctness.out": "6f72e539b4789f3d016c29665c6a16525a87235d48d228d2b4f15642af30a982",
    "decision.json": "561399219b2ccb066d584932b5ca1a72ca42f56757fc7fa9d6f68b2a8ae5d353",
    "final-identity.json": "0a50784c6d9560fe3cadf3123a547765d08390e78cf07940f69f76d6376f35d6",
    "launch-seal.jsonl": "7fdc0b5fc3851d4af4b400adae7e9bd3856d43961d4e856f2c7025bbe4131c3b",
    "loaded-q01-abba-r1-a.conditioning.json": "03feedabc8caaa9aeaafd2ed30c03c32e0cab6423c6e991680fea03c1583547c",
    "loaded-q01-abba-r1-a.err": "45dcf886246493b700027511cd832de253713fc3e1c1750278e10ef0a5db83f2",
    "loaded-q01-abba-r1-a.out": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "loaded-q01-abba-r1-a.post-exit.json": "23aee701b8ea16c89cd7d904a24c6dc75fd5ef33cbc65e70f924cc8f390a5779",
    "manifest.json": "6b6b010469d9a6540b65292d68a29fb3f9d636ef4333f92c4cb4af57adc808d6",
    "token-protocol.json": "b6ead36b92c771a1e640774a7a831bccbaa752191c987b76126d9d590f497e54",
    "token-protocol.out": "a27b9ef74794543965646dc3426a59ca092f8e7939876cabbfd19fede86da258",
}

ORIGINAL_MAKE_DECISION = base.make_decision


def verify_v0632_forensics() -> tuple[dict[str, object], dict[Path, str]]:
    for path, expected in V0632_SEALS.items():
        if not path.is_file() or path.is_symlink() or base.sha256(path) != expected:
            raise base.ContractDefect(f"v0.632 seal drifted: {path}")
    complete = base.parse_json(
        V0632_COMPLETE.read_text(encoding="utf-8"), "v0.632 completion"
    )
    if not base.strict_equal(
        complete,
        {
            "schema": 1,
            "decision_sha256": V0632_SEALS[V0632_DECISION],
            "inventory_sha256": V0632_SEALS[V0632_INVENTORY],
        },
    ):
        raise base.ContractDefect("v0.632 completion binding drifted")
    members = base.inventory_members(V0632_INVENTORY)
    expected_members = {
        V0632_ROOT / name: digest for name, digest in EXPECTED_V0632_MEMBERS.items()
    }
    if members != expected_members:
        raise base.ContractDefect("v0.632 inventory membership drifted")
    for path, expected in members.items():
        if not path.is_file() or path.is_symlink() or base.sha256(path) != expected:
            raise base.ContractDefect(f"v0.632 member drifted: {path}")
    if set(V0632_ROOT.iterdir()) != set(members) | {V0632_INVENTORY, V0632_COMPLETE}:
        raise base.ContractDefect("v0.632 final artifact set drifted")
    decision = base.parse_json(
        V0632_DECISION.read_text(encoding="utf-8"), "v0.632 decision"
    )
    if not isinstance(decision, dict):
        raise base.ContractDefect("v0.632 decision is malformed")
    terminal = {
        "schema": 1,
        "status": "inconclusive",
        "authority": "none",
        "force_authorized": False,
        "successor_authorization": "none",
        "stopped_after": "loaded",
        "failed_child": "loaded-q01-abba-r1-a",
        "reasons": ["child_major_faults"],
        "source_commit": V0632_REPAIR_COMMIT,
        "stage": None,
        "attempts_sha256": EXPECTED_V0632_MEMBERS["attempts.jsonl"],
    }
    for key, expected in terminal.items():
        if not base.strict_equal(decision.get(key), expected):
            raise base.ContractDefect(f"v0.632 terminal field drifted: {key}")
    correctness = decision.get("correctness")
    if not isinstance(correctness, dict) or correctness.get("passed") is not True:
        raise base.ContractDefect("v0.632 correctness did not freshly pass")
    claim = decision.get("claim_scope")
    if (
        not isinstance(claim, dict)
        or claim.get("performance_imports_used_for_scoring") != 0
    ):
        raise base.ContractDefect("v0.632 performance-import contract drifted")
    attempts = base.read_jsonl(V0632_ROOT / "attempts.jsonl")
    if len(attempts) != 1:
        raise base.ContractDefect("v0.632 attempt count drifted")
    attempt = attempts[0]
    resources = attempt.get("process_resources")
    bench = attempt.get("bench")
    if (
        attempt.get("artifact_stem") != "loaded-q01-abba-r1-a"
        or attempt.get("arm") != "A"
        or attempt.get("returncode") != 0
        or attempt.get("valid") is not False
        or attempt.get("validity_reasons") != ["child_major_faults"]
        or not isinstance(resources, dict)
        or resources.get("page_faults") != 92
        or resources.get("block_input_operations") != 0
        or resources.get("swaps") != 0
        or not isinstance(bench, dict)
        or len(bench.get("token_trace", [])) != 32
    ):
        raise base.ContractDefect("v0.632 sole A attempt drifted")
    conditioning = base.parse_json(
        (V0632_ROOT / "loaded-q01-abba-r1-a.conditioning.json").read_text(
            encoding="utf-8"
        ),
        "v0.632 conditioning",
    )
    post = base.parse_json(
        (V0632_ROOT / "loaded-q01-abba-r1-a.post-exit.json").read_text(
            encoding="utf-8"
        ),
        "v0.632 post-exit",
    )
    if (
        not isinstance(conditioning, dict)
        or conditioning.get("residency", {}).get("all_pages_resident") is not True
        or conditioning.get("cache", {}).get("bytes_read") != base.EXPECTED_MODEL_SIZE
        or not isinstance(post, dict)
        or post.get("interval", {}).get("deltas", {}).get("pageouts") != 130
        or post.get("interval", {}).get("deltas", {}).get("swapouts") != 0
        or post.get("interval", {}).get("deltas", {}).get("swap_used_bytes") != 0
        or post.get("interval", {}).get("deltas", {}).get("compressions") != 0
    ):
        raise base.ContractDefect("v0.632 conditioning or pressure evidence drifted")
    events = base.read_jsonl(V0632_ROOT / "launch-seal.jsonl")
    if (
        len(events) != 2
        or events[0].get("event") != "launch"
        or events[1].get("event") != "completion"
        or any(
            event.get("artifact_stem") != attempt["artifact_stem"] for event in events
        )
    ):
        raise base.ContractDefect("v0.632 launch/completion evidence drifted")
    final_identity = base.parse_json(
        (V0632_ROOT / "final-identity.json").read_text(encoding="utf-8"),
        "v0.632 final identity",
    )
    if (
        not isinstance(final_identity, dict)
        or final_identity.get("matches") is not True
    ):
        raise base.ContractDefect("v0.632 final identity did not pass")
    return (
        {
            "authority_imported": False,
            "correctness_imported_as_gate": False,
            "performance_observations_imported": 0,
            "timed_children_imported": 0,
            "observed_a_children": 1,
            "observed_b_children": 0,
            "forensic_class": "timed-child-process-wide-page-fault-gate-overbroad",
            "observed_a_page_faults": 92,
            "decision_sha256": V0632_SEALS[V0632_DECISION],
            "inventory_sha256": V0632_SEALS[V0632_INVENTORY],
            "completion_sha256": V0632_SEALS[V0632_COMPLETE],
            "inventory_members": len(members),
        },
        {
            **members,
            V0632_INVENTORY: V0632_SEALS[V0632_INVENTORY],
            V0632_COMPLETE: V0632_SEALS[V0632_COMPLETE],
        },
    )


def verify_source_topology() -> dict[str, object]:
    for path in (PREREG, RUNNER, base.BENCH_SOURCE):
        base.git_output(["ls-files", "--error-unmatch", str(path.relative_to(ROOT))])
    dirty = base.git_output(["status", "--porcelain=v1", "--untracked-files=all"])
    if dirty:
        raise base.ContractDefect(f"worktree is not clean: {dirty!r}")
    head = base.git_output(["rev-parse", "HEAD"])
    repair632 = base.git_output(["rev-parse", "HEAD^"])
    repair631 = base.git_output(["rev-parse", "HEAD^^"])
    implementation = base.git_output(["rev-parse", "HEAD^^^"])
    preregistration = base.git_output(["rev-parse", "HEAD^^^^"])
    root = base.git_output(["rev-parse", "HEAD^^^^^"])
    expected = (
        BASE_COMMIT,
        V0630_PREREG_COMMIT,
        V0630_IMPLEMENTATION_COMMIT,
        V0631_REPAIR_COMMIT,
        V0632_REPAIR_COMMIT,
    )
    if (root, preregistration, implementation, repair631, repair632) != expected:
        raise base.ContractDefect("v0.633 frozen lineage drifted")
    chain = (*expected, head)
    if any(base.commit_parents(right) != [left] for left, right in pairwise(chain)):
        raise base.ContractDefect("v0.633 lineage contains a merge commit")
    repair_paths = sorted(
        [str(PREREG.relative_to(ROOT)), str(RUNNER.relative_to(ROOT))]
    )
    if sorted(base.changed_paths(repair632, head)) != repair_paths or sorted(
        base.name_status(repair632, head)
    ) != sorted(f"A\t{path}" for path in repair_paths):
        raise base.ContractDefect("v0.633 repair diff drifted")
    historical_repairs = {
        (implementation, repair631): (
            "docs/bench/v0631-dense27b-pread-loaded-stability-repair.md",
            "scripts/profile/v0631_dense27b_pread_loaded_stability_repair.py",
        ),
        (repair631, repair632): (
            "docs/bench/v0632-dense27b-pread-correctness-fault-repair.md",
            "scripts/profile/v0632_dense27b_pread_correctness_fault_repair.py",
        ),
    }
    for (left, right), paths in historical_repairs.items():
        if sorted(base.name_status(left, right)) != sorted(
            f"A\t{path}" for path in paths
        ):
            raise base.ContractDefect(f"historical repair diff drifted: {right}")
    v0630_paths = (
        "docs/bench/v0630-dense27b-pread-loaded-stability.md",
        "scripts/profile/v0630_dense27b_pread_loaded_stability.py",
    )
    if sorted(base.name_status(root, preregistration)) != sorted(
        f"A\t{path}" for path in v0630_paths
    ):
        raise base.ContractDefect("v0.630 preregistration diff drifted")
    bench_path = str(base.BENCH_SOURCE.relative_to(ROOT))
    if base.name_status(preregistration, implementation) != [f"M\t{bench_path}"]:
        raise base.ContractDefect("v0.630 implementation diff drifted")
    if base.sha256(base.BENCH_SOURCE) != base.EXPECTED_BENCH_SOURCE_SHA256:
        raise base.ContractDefect("v0.630 complete bench.rs digest drifted")
    if (
        base.implementation_patch_sha256(preregistration, implementation)
        != base.EXPECTED_IMPLEMENTATION_PATCH_SHA256
    ):
        raise base.ContractDefect("v0.630 implementation patch digest drifted")
    build = base.parse_json(
        prior.prior.repaired_command_output(
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
        "v0632_repair_commit": repair632,
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
    manifest = prior.build_manifest(child_env, test_env, removed)
    forensic, members = verify_v0632_forensics()
    manifest["forensic_v0632"] = forensic
    hashes = manifest.get("sha256")
    if not isinstance(hashes, dict):
        raise base.ContractDefect("manifest hash map is malformed")
    for path, digest in members.items():
        hashes[str(path)] = digest
    return manifest


def run_child(
    base_env: dict[str, str],
    manifest: dict[str, object],
    prompt: str,
    quartet_index: int,
    order: str,
    position: int,
    arm: str,
) -> dict[str, object]:
    stem = f"loaded-q{quartet_index:02d}-{order.lower()}-r{position}-{arm.lower()}"
    stdout_path = base.ARTIFACT / f"{stem}.out"
    stderr_path = base.ARTIFACT / f"{stem}.err"
    conditioning = base.condition_child(stem, manifest)
    env, delta = base.arm_environment(base_env, arm)
    command = base.child_command(prompt)
    started_wall = time.time_ns()
    started_mono = time.monotonic_ns()
    process: subprocess.Popen[bytes] | None = None
    returncode: int | None = None
    spawn_error: Exception | None = None
    spawn_started_mono: int | None = None
    spawn_completed_mono: int | None = None
    wait_errors: list[str] = []
    lifecycle_reaped = True
    durability_error: OSError | None = None
    launch_recorded = False
    deferred_signals: list[int] = []

    def defer_sigint(signum: int, _frame: object) -> None:
        deferred_signals.append(signum)

    prior_handler = signal.signal(signal.SIGINT, defer_sigint)
    try:
        try:
            with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
                base.record_launch(
                    stem,
                    arm,
                    quartet_index,
                    order,
                    position,
                    command,
                    delta,
                    manifest,
                )
                launch_recorded = True
                try:
                    spawn_started_mono = time.monotonic_ns()
                    process = subprocess.Popen(
                        command,
                        cwd=ROOT,
                        env=env,
                        stdout=stdout,
                        stderr=stderr,
                    )
                except Exception as error:  # noqa: BLE001 - seal spawn evidence
                    spawn_error = error
                finally:
                    spawn_completed_mono = time.monotonic_ns()
                if process is not None:
                    returncode, wait_errors, lifecycle_reaped = (
                        base.wait_for_child_process(process)
                    )
                try:
                    stdout.flush()
                    os.fsync(stdout.fileno())
                    stderr.flush()
                    os.fsync(stderr.fileno())
                except OSError as error:
                    durability_error = error
        except OSError as error:
            durability_error = error
    finally:
        signal.signal(signal.SIGINT, prior_handler)
        if launch_recorded:
            errors = []
            if spawn_error is not None:
                errors.append(f"{type(spawn_error).__name__}:{spawn_error}")
            errors.extend(wait_errors)
            if not lifecycle_reaped:
                errors.append("child_lifecycle_not_reaped")
            if durability_error is not None:
                errors.append(f"{type(durability_error).__name__}:{durability_error}")
            base.record_completion(
                stem,
                returncode,
                ";".join(errors) if errors else None,
            )
    ended_mono = time.monotonic_ns()
    ended_wall = time.time_ns()
    if durability_error is not None:
        raise base.UnsealedPacket(
            f"child raw durability failed: {stem}: {durability_error}"
        )
    if not lifecycle_reaped:
        raise base.UnsealedPacket(
            f"child lifecycle could not be reaped exactly: {stem}"
        )
    if spawn_started_mono is None or spawn_completed_mono is None:
        raise base.UnsealedPacket(f"child spawn bracket was not reached: {stem}")
    host_after = base.capture_host_state()
    vm_after = base.capture_vm_state()
    interval = base.vm_interval("child", conditioning["interval"]["after"], vm_after)
    base_row = {
        "schema": 1,
        "artifact_stem": stem,
        "arm": arm,
        "quartet_index": quartet_index,
        "quartet_order": order,
        "position": position,
        "started_unix_ns": started_wall,
        "ended_unix_ns": ended_wall,
        "conditioning_monotonic_ns": conditioning["monotonic_ns"],
        "spawn_started_monotonic_ns": spawn_started_mono,
        "spawn_completed_monotonic_ns": spawn_completed_mono,
        "started_monotonic_ns": started_mono,
        "ended_monotonic_ns": ended_mono,
        "process_wall_ns": ended_mono - started_mono,
        "spawn_to_exit_ns": ended_mono - spawn_completed_mono,
        "command": command,
        "environment_delta": delta,
        "returncode": returncode,
        "stdout_sha256": base.sha256(stdout_path),
        "stderr_sha256": base.sha256(stderr_path),
        "conditioning_sha256": base.sha256(base.ARTIFACT / f"{stem}.conditioning.json"),
    }
    if deferred_signals or spawn_error is not None or wait_errors or returncode != 0:
        reasons = list(interval["failure_reasons"])
        if deferred_signals:
            reasons.append(f"operator_sigint_deferred={len(deferred_signals)}")
        if spawn_error is not None:
            reasons.append(
                f"child_spawn_failed={type(spawn_error).__name__}:{spawn_error}"
            )
        if wait_errors:
            reasons.extend(f"child_{error}" for error in wait_errors)
        if spawn_error is None and returncode != 0:
            reasons.append(f"child_returncode={returncode}")
        post = {
            "schema": 1,
            "artifact_stem": stem,
            "host_after_exit": host_after,
            "vm_after_exit": vm_after,
            "interval": interval,
            "process_resources": None,
            "page_faults_advisory": None,
            "validity_reasons": reasons,
        }
        post_path = base.ARTIFACT / f"{stem}.post-exit.json"
        base.write_json(post_path, post)
        row = {
            **base_row,
            "post_exit_sha256": base.sha256(post_path),
            "load_contract": None,
            "bench": None,
            "process_resources": None,
            "page_faults_advisory": None,
            "valid": False,
            "validity_reasons": reasons,
            "failure_class": "invalid-child",
        }
        base.append_jsonl(base.ARTIFACT / "attempts.jsonl", row)
        raise base.InconclusivePacket("loaded", stem, reasons)

    parse_error: base.ContractDefect | None = None
    load: dict[str, object] | None = None
    bench: dict[str, object] | None = None
    resources: dict[str, int] | None = None
    try:
        stdout = stdout_path.read_bytes()
        stderr = stderr_path.read_text(encoding="utf-8")
        if stdout:
            raise base.ContractDefect(f"qwen-bench emitted unexpected stdout: {stem}")
        load = base.parse_load_contract(stderr, arm)
        bench = base.parse_bench(stderr, prompt)
        resources = base.process_resources(stderr)
    except base.ContractDefect as error:
        parse_error = error
    except Exception as error:  # noqa: BLE001 - normalize child parser failures
        parse_error = base.ContractDefect(
            f"child parser failed: {type(error).__name__}:{error}"
        )
    if parse_error is not None:
        post = {
            "schema": 1,
            "artifact_stem": stem,
            "host_after_exit": host_after,
            "vm_after_exit": vm_after,
            "interval": interval,
            "process_resources": resources,
            "page_faults_advisory": (
                resources.get("page_faults") if resources is not None else None
            ),
            "validity_reasons": [],
            "contract_error": str(parse_error),
        }
        post_path = base.ARTIFACT / f"{stem}.post-exit.json"
        base.write_json(post_path, post)
        row = {
            **base_row,
            "post_exit_sha256": base.sha256(post_path),
            "load_contract": load,
            "bench": bench,
            "process_resources": resources,
            "page_faults_advisory": post["page_faults_advisory"],
            "valid": False,
            "validity_reasons": [],
            "failure_class": "implementation_or_contract_defect",
            "contract_error": str(parse_error),
        }
        base.append_jsonl(base.ARTIFACT / "attempts.jsonl", row)
        raise base.ChildContractDefect(stem, str(parse_error))

    assert load is not None and bench is not None and resources is not None
    reasons = list(interval["failure_reasons"])
    if host_after.get("valid") is not True:
        reasons.append("host_invalid_after_child")
    if resources["block_input_operations"] != 0:
        reasons.append("child_block_input")
    if resources["swaps"] != 0:
        reasons.append("child_swaps")
    page_faults_advisory = resources["page_faults"]
    post = {
        "schema": 1,
        "artifact_stem": stem,
        "host_after_exit": host_after,
        "vm_after_exit": vm_after,
        "interval": interval,
        "process_resources": resources,
        "page_faults_advisory": page_faults_advisory,
        "validity_reasons": reasons,
    }
    post_path = base.ARTIFACT / f"{stem}.post-exit.json"
    base.write_json(post_path, post)
    row = {
        **base_row,
        "post_exit_sha256": base.sha256(post_path),
        "load_contract": load,
        "bench": bench,
        "process_resources": resources,
        "page_faults_advisory": page_faults_advisory,
        "valid": not reasons,
        "validity_reasons": reasons,
        "failure_class": None,
    }
    base.append_jsonl(base.ARTIFACT / "attempts.jsonl", row)
    if reasons:
        raise base.InconclusivePacket("loaded", stem, reasons)
    return row


def make_decision(*args: object, **kwargs: object) -> dict[str, object]:
    decision = prior.make_decision(*args, **kwargs)
    manifest = args[0] if args else kwargs.get("manifest")
    if not isinstance(manifest, dict):
        raise base.ContractDefect("decision manifest is malformed")
    decision["forensic_v0632"] = manifest.get("forensic_v0632")
    return decision


def configure() -> None:
    prior.configure()
    base.ARTIFACT = ARTIFACT
    base.PREREG = PREREG
    base.RUNNER = RUNNER
    base.verify_source_topology = verify_source_topology
    base.build_manifest = build_manifest
    base.run_child = run_child
    base.make_decision = make_decision


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    configure()
    base.run_packet(preflight_only=args.preflight_only)


if __name__ == "__main__":
    main()
