#!/usr/bin/env python3
"""v0.634 terminal ramp-controlled dense-27B pread stability packet."""

import argparse
import os
import signal
import subprocess
import time
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

import v0633_dense27b_pread_child_fault_repair as prior

base = prior.base
ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = ROOT / "target/profiles/v0634-dense27b-pread-ramp-controlled-p1"
PREREG = ROOT / "docs/bench/v0634-dense27b-pread-ramp-controlled-stability.md"
RUNNER = Path(__file__).resolve()
V0633_ROOT = ROOT / "target/profiles/v0633-dense27b-pread-child-fault-repair-p1"
V0633_DECISION = V0633_ROOT / "decision.json"
V0633_INVENTORY = V0633_ROOT / "artifact-inventory.sha256"
V0633_COMPLETE = V0633_ROOT / "packet-complete.json"

BASE_COMMIT = "cc47190654b898e2004117b6d03be1006a89d1fa"
V0630_PREREG_COMMIT = "d30a0f49e14aec13126d83a41553f6e7250b0f93"
V0630_IMPLEMENTATION_COMMIT = "d90aecdd4d4f9c297f98b09ba55eb479a7bf7944"
V0631_REPAIR_COMMIT = "ecc9e70fc833ee40831b7f5891141e620a92a2cd"
V0632_REPAIR_COMMIT = "2e71e1a8caba14ddc0fcc5ea3232354dbab86504"
V0633_REPAIR_COMMIT = "e237917ddeaec5635fb599fd19ae1ac58e3d0c51"
V0633_CERTIFICATION_COMMIT = "e307f273ee8d1c2cc6c8d67985e88e32876e0654"
V0633_SEALS = {
    V0633_DECISION: "3cf6427e32056d30356176e03e18e1b37c3205aa1cf591e60c54cac04be34d8a",
    V0633_INVENTORY: "1faac9baebd9f453ea1dfb23149502cdc34b25697ffd6cfd4e439cf4291509f5",
    V0633_COMPLETE: "1892674cd1e0c5df303ecd60c24beb2f5e76590b1c55334043b010d003d186ec",
}
RAMP_ORDERS = ("ABBA", "BAAB")
SCORED_ORDERS = base.QUARTET_ORDERS


@dataclass(frozen=True)
class ChildSpec:
    population: str
    used_for_scoring: bool
    execution_index: int
    quartet_index: int
    quartet_order: str
    position: int
    arm: str
    artifact_stem: str

    def identity(self) -> dict[str, object]:
        return {
            "population": self.population,
            "used_for_scoring": self.used_for_scoring,
            "execution_index": self.execution_index,
            "quartet_index": self.quartet_index,
            "quartet_order": self.quartet_order,
            "position": self.position,
            "arm": self.arm,
            "artifact_stem": self.artifact_stem,
        }


def build_schedule() -> tuple[ChildSpec, ...]:
    specs: list[ChildSpec] = []
    execution_index = 0
    for population, used_for_scoring, orders, prefix in (
        ("ramp", False, RAMP_ORDERS, "ramp"),
        ("scored", True, SCORED_ORDERS, "loaded"),
    ):
        for quartet_index, order in enumerate(orders, 1):
            for position, arm in enumerate(order, 1):
                execution_index += 1
                stem = (
                    f"{prefix}-q{quartet_index:02d}-{order.lower()}-"
                    f"r{position}-{arm.lower()}"
                )
                specs.append(
                    ChildSpec(
                        population,
                        used_for_scoring,
                        execution_index,
                        quartet_index,
                        order,
                        position,
                        arm,
                        stem,
                    )
                )
    return tuple(specs)


EXECUTION_SCHEDULE = build_schedule()
RAMP_SPECS = tuple(spec for spec in EXECUTION_SCHEDULE if not spec.used_for_scoring)
SCORED_SPECS = tuple(spec for spec in EXECUTION_SCHEDULE if spec.used_for_scoring)


def verify_v0633_forensics() -> tuple[dict[str, object], dict[Path, str]]:
    for path, expected in V0633_SEALS.items():
        if not path.is_file() or path.is_symlink() or base.sha256(path) != expected:
            raise base.ContractDefect(f"v0.633 seal drifted: {path}")
    complete = base.parse_json(
        V0633_COMPLETE.read_text(encoding="utf-8"), "v0.633 completion"
    )
    if not base.strict_equal(
        complete,
        {
            "schema": 1,
            "decision_sha256": V0633_SEALS[V0633_DECISION],
            "inventory_sha256": V0633_SEALS[V0633_INVENTORY],
        },
    ):
        raise base.ContractDefect("v0.633 completion binding drifted")
    members = base.inventory_members(V0633_INVENTORY)
    if (
        len(members) != 137
        or members.get(V0633_DECISION) != V0633_SEALS[V0633_DECISION]
    ):
        raise base.ContractDefect("v0.633 inventory membership drifted")
    for path, expected in members.items():
        if not path.is_file() or path.is_symlink() or base.sha256(path) != expected:
            raise base.ContractDefect(f"v0.633 inventory member drifted: {path}")
    if set(V0633_ROOT.iterdir()) != set(members) | {
        V0633_INVENTORY,
        V0633_COMPLETE,
    }:
        raise base.ContractDefect("v0.633 final artifact set drifted")

    decision = base.parse_json(
        V0633_DECISION.read_text(encoding="utf-8"), "v0.633 decision"
    )
    if not isinstance(decision, dict):
        raise base.ContractDefect("v0.633 decision is malformed")
    terminal = {
        "schema": 1,
        "status": "inconclusive",
        "authority": "none",
        "force_authorized": False,
        "successor_authorization": "none",
        "stopped_after": "loaded-stability",
        "failed_child": None,
        "reasons": ["inconclusive-instability"],
        "source_commit": V0633_REPAIR_COMMIT,
        "attempts_sha256": (
            "619dcfe6bc0f3d4987e302e85d4511fee25a8a2578c3e94e3f1d05b3c1ca0954"
        ),
    }
    for key, expected in terminal.items():
        if not base.strict_equal(decision.get(key), expected):
            raise base.ContractDefect(f"v0.633 terminal field drifted: {key}")
    correctness = decision.get("correctness")
    if not isinstance(correctness, dict) or correctness.get("passed") is not True:
        raise base.ContractDefect("v0.633 correctness did not pass")
    claim = decision.get("claim_scope")
    if (
        not isinstance(claim, dict)
        or claim.get("performance_imports_used_for_scoring") != 0
    ):
        raise base.ContractDefect("v0.633 performance-import contract drifted")
    stage = decision.get("stage")
    if (
        not isinstance(stage, dict)
        or stage.get("classification") != "inconclusive-instability"
        or stage.get("stability", {}).get("passes") is not False
        or stage.get("performance", {}).get("passes") is not False
    ):
        raise base.ContractDefect("v0.633 stage classification drifted")
    expected_medians = {
        "all": 1.0130895866677476,
        "ABBA": 1.0134791837986845,
        "BAAB": 1.0124417930371252,
    }
    if not base.strict_equal(
        stage.get("medians", {}).get("prefill_b_over_a"), expected_medians
    ):
        raise base.ContractDefect("v0.633 prefill medians drifted")

    attempts = base.read_jsonl(V0633_ROOT / "attempts.jsonl")
    expected = base.expected_children()
    if len(attempts) != 32 or len(expected) != 32:
        raise base.ContractDefect("v0.633 attempt count drifted")
    golden: list[int] | None = None
    for row, (quartet, order, position, arm, stem) in zip(
        attempts, expected, strict=True
    ):
        resources = row.get("process_resources")
        bench = row.get("bench")
        load = row.get("load_contract")
        if (
            row.get("artifact_stem") != stem
            or row.get("quartet_index") != quartet
            or row.get("quartet_order") != order
            or row.get("position") != position
            or row.get("arm") != arm
            or row.get("returncode") != 0
            or row.get("valid") is not True
            or row.get("validity_reasons") != []
            or not isinstance(resources, dict)
            or resources.get("page_faults") != 92
            or resources.get("block_input_operations") != 0
            or resources.get("swaps") != 0
            or not isinstance(bench, dict)
            or len(bench.get("token_trace", [])) != 32
            or not isinstance(load, dict)
        ):
            raise base.ContractDefect(f"v0.633 attempt drifted: {stem}")
        if arm == "B" and load.get("marker", {}).get("timer_major_faults") != 0:
            raise base.ContractDefect(f"v0.633 B marker drifted: {stem}")
        if golden is None:
            golden = bench["token_trace"]
        elif not base.strict_equal(bench["token_trace"], golden):
            raise base.ContractDefect("v0.633 token identity drifted")
        for suffix, field in (
            ("out", "stdout_sha256"),
            ("err", "stderr_sha256"),
            ("conditioning.json", "conditioning_sha256"),
            ("post-exit.json", "post_exit_sha256"),
        ):
            path = V0633_ROOT / f"{stem}.{suffix}"
            if not path.is_file() or base.sha256(path) != row.get(field):
                raise base.ContractDefect(f"v0.633 child binding drifted: {path}")

    events = base.read_jsonl(V0633_ROOT / "launch-seal.jsonl")
    if len(events) != 65:
        raise base.ContractDefect("v0.633 launch ledger count drifted")
    cursor = 0
    for index, row in enumerate(attempts):
        launch, completion = events[cursor : cursor + 2]
        cursor += 2
        if (
            launch.get("event") != "launch"
            or completion.get("event") != "completion"
            or launch.get("artifact_stem") != row["artifact_stem"]
            or completion.get("artifact_stem") != row["artifact_stem"]
            or completion.get("returncode") != 0
        ):
            raise base.ContractDefect("v0.633 launch/completion order drifted")
        if index == 0:
            golden_event = events[cursor]
            cursor += 1
            if (
                golden_event.get("event") != "golden-token-trace"
                or golden_event.get("source_artifact_stem") != row["artifact_stem"]
                or not base.strict_equal(
                    golden_event.get("token_trace"), row["bench"]["token_trace"]
                )
            ):
                raise base.ContractDefect("v0.633 golden event drifted")
    if cursor != len(events):
        raise base.ContractDefect("v0.633 launch ledger has extra events")
    final_identity = base.parse_json(
        (V0633_ROOT / "final-identity.json").read_text(encoding="utf-8"),
        "v0.633 final identity",
    )
    if (
        not isinstance(final_identity, dict)
        or final_identity.get("matches") is not True
    ):
        raise base.ContractDefect("v0.633 final identity did not pass")
    return (
        {
            "authority_imported": False,
            "correctness_imported_as_gate": False,
            "performance_observations_imported": 0,
            "timed_children_imported": 0,
            "observed_valid_children": 32,
            "forensic_class": "inconclusive-instability",
            "decision_sha256": V0633_SEALS[V0633_DECISION],
            "inventory_sha256": V0633_SEALS[V0633_INVENTORY],
            "completion_sha256": V0633_SEALS[V0633_COMPLETE],
            "inventory_members": len(members),
        },
        {
            **members,
            V0633_INVENTORY: V0633_SEALS[V0633_INVENTORY],
            V0633_COMPLETE: V0633_SEALS[V0633_COMPLETE],
        },
    )


def verify_source_topology() -> dict[str, object]:
    for path in (PREREG, RUNNER, base.BENCH_SOURCE):
        base.git_output(["ls-files", "--error-unmatch", str(path.relative_to(ROOT))])
    dirty = base.git_output(["status", "--porcelain=v1", "--untracked-files=all"])
    if dirty:
        raise base.ContractDefect(f"worktree is not clean: {dirty!r}")
    head = base.git_output(["rev-parse", "HEAD"])
    certification = base.git_output(["rev-parse", "HEAD^"])
    if certification != V0633_CERTIFICATION_COMMIT:
        raise base.ContractDefect("v0.634 certification parent drifted")
    chain = (
        BASE_COMMIT,
        V0630_PREREG_COMMIT,
        V0630_IMPLEMENTATION_COMMIT,
        V0631_REPAIR_COMMIT,
        V0632_REPAIR_COMMIT,
        V0633_REPAIR_COMMIT,
        V0633_CERTIFICATION_COMMIT,
        head,
    )
    if any(base.commit_parents(right) != [left] for left, right in pairwise(chain)):
        raise base.ContractDefect("v0.634 lineage contains a merge or drift")
    current_paths = sorted(
        [str(PREREG.relative_to(ROOT)), str(RUNNER.relative_to(ROOT))]
    )
    if sorted(base.changed_paths(certification, head)) != current_paths or sorted(
        base.name_status(certification, head)
    ) != sorted(f"A\t{path}" for path in current_paths):
        raise base.ContractDefect("v0.634 preregistration diff drifted")
    certification_paths = (
        "docs/PERF-LOG.md",
        "docs/PERF-ROADMAP.md",
    )
    if sorted(base.name_status(V0633_REPAIR_COMMIT, certification)) != sorted(
        f"M\t{path}" for path in certification_paths
    ):
        raise base.ContractDefect("v0.633 certification diff drifted")
    repair_specs = (
        (
            V0630_IMPLEMENTATION_COMMIT,
            V0631_REPAIR_COMMIT,
            "docs/bench/v0631-dense27b-pread-loaded-stability-repair.md",
            "scripts/profile/v0631_dense27b_pread_loaded_stability_repair.py",
        ),
        (
            V0631_REPAIR_COMMIT,
            V0632_REPAIR_COMMIT,
            "docs/bench/v0632-dense27b-pread-correctness-fault-repair.md",
            "scripts/profile/v0632_dense27b_pread_correctness_fault_repair.py",
        ),
        (
            V0632_REPAIR_COMMIT,
            V0633_REPAIR_COMMIT,
            "docs/bench/v0633-dense27b-pread-child-fault-repair.md",
            "scripts/profile/v0633_dense27b_pread_child_fault_repair.py",
        ),
    )
    for left, right, document, runner in repair_specs:
        if sorted(base.name_status(left, right)) != sorted(
            (f"A\t{document}", f"A\t{runner}")
        ):
            raise base.ContractDefect(f"historical repair diff drifted: {right}")
    v0630_paths = (
        "docs/bench/v0630-dense27b-pread-loaded-stability.md",
        "scripts/profile/v0630_dense27b_pread_loaded_stability.py",
    )
    if sorted(base.name_status(BASE_COMMIT, V0630_PREREG_COMMIT)) != sorted(
        f"A\t{path}" for path in v0630_paths
    ):
        raise base.ContractDefect("v0.630 preregistration diff drifted")
    bench_path = str(base.BENCH_SOURCE.relative_to(ROOT))
    if base.name_status(V0630_PREREG_COMMIT, V0630_IMPLEMENTATION_COMMIT) != [
        f"M\t{bench_path}"
    ]:
        raise base.ContractDefect("v0.630 implementation diff drifted")
    if base.sha256(base.BENCH_SOURCE) != base.EXPECTED_BENCH_SOURCE_SHA256:
        raise base.ContractDefect("v0.630 complete bench.rs digest drifted")
    if (
        base.implementation_patch_sha256(
            V0630_PREREG_COMMIT, V0630_IMPLEMENTATION_COMMIT
        )
        != base.EXPECTED_IMPLEMENTATION_PATCH_SHA256
    ):
        raise base.ContractDefect("v0.630 implementation patch drifted")
    build = base.parse_json(
        base.command_output([str(base.BENCH_BINARY), "build-info", "--output", "json"]),
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
        "preregistration_commit": head,
        "certification_commit": certification,
        "v0633_repair_commit": V0633_REPAIR_COMMIT,
        "base_commit": BASE_COMMIT,
        "build_identity": build,
        "sibling_dependencies": dependencies,
    }


def build_manifest(
    child_env: dict[str, str], test_env: dict[str, str], removed: list[str]
) -> dict[str, object]:
    manifest = prior.build_manifest(child_env, test_env, removed)
    forensic, members = verify_v0633_forensics()
    manifest["schema"] = 2
    manifest["forensic_v0633"] = forensic
    hashes = manifest.get("sha256")
    if not isinstance(hashes, dict):
        raise base.ContractDefect("manifest hash map is malformed")
    for path, digest in members.items():
        hashes[str(path)] = digest
    manifest["cell"] = {
        "ramp": {
            "quartet_orders": list(RAMP_ORDERS),
            "children": len(RAMP_SPECS),
            "used_for_scoring": False,
            "used_for_stability": False,
            "used_for_medians": False,
        },
        "scored": {
            "quartet_orders": list(SCORED_ORDERS),
            "children": len(SCORED_SPECS),
            "used_for_scoring": True,
        },
        "total_children": len(EXECUTION_SCHEDULE),
        "prompt_bytes": base.EXPECTED_PROMPT_BYTES,
        "prompt_tokens": base.EXPECTED_PROMPT_TOKENS,
        "transitions": base.EXPECTED_TRANSITIONS,
        "runs": 1,
        "prefill_chunk": 1024,
        "kv_capacity": 1024,
        "full_logits_decode": True,
        "generated_token_trace": True,
    }
    return manifest


def record_launch(
    spec: ChildSpec,
    command: list[str],
    environment_delta: dict[str, str],
    manifest: dict[str, object],
) -> None:
    base.append_jsonl(
        base.ARTIFACT / "launch-seal.jsonl",
        {
            "schema": 2,
            "event": "launch",
            "unix_ms": time.time_ns() // 1_000_000,
            "monotonic_ns": time.monotonic_ns(),
            **spec.identity(),
            "command": command,
            "environment_delta": environment_delta,
            "source_commit": manifest["source_commit"],
            "conditioning_sha256": base.sha256(
                base.ARTIFACT / f"{spec.artifact_stem}.conditioning.json"
            ),
        },
    )


def record_completion(
    spec: ChildSpec, returncode: int | None, error: str | None
) -> None:
    base.append_jsonl(
        base.ARTIFACT / "launch-seal.jsonl",
        {
            "schema": 2,
            "event": "completion",
            "unix_ms": time.time_ns() // 1_000_000,
            "monotonic_ns": time.monotonic_ns(),
            **spec.identity(),
            "returncode": returncode,
            "error": error,
        },
    )


def record_golden(row: dict[str, object]) -> None:
    base.append_jsonl(
        base.ARTIFACT / "launch-seal.jsonl",
        {
            "schema": 2,
            "event": "golden-token-trace",
            "unix_ms": time.time_ns() // 1_000_000,
            "monotonic_ns": time.monotonic_ns(),
            "population": "ramp",
            "used_for_scoring": False,
            "source_artifact_stem": row["artifact_stem"],
            "token_trace": row["bench"]["token_trace"],
            "token_trace_sha256": row["bench"]["token_trace_sha256"],
            "generated_debug_sha256": row["bench"]["generated_debug_sha256"],
        },
    )


def run_child(
    base_env: dict[str, str],
    manifest: dict[str, object],
    prompt: str,
    spec: ChildSpec,
    packet_signals: list[int],
) -> dict[str, object]:
    stem = spec.artifact_stem
    stdout_path = base.ARTIFACT / f"{stem}.out"
    stderr_path = base.ARTIFACT / f"{stem}.err"
    conditioning = base.condition_child(stem, manifest)
    if packet_signals:
        raise base.InconclusivePacket(
            spec.population,
            stem,
            [f"operator_sigint_deferred={len(packet_signals)}"],
        )
    env, delta = base.arm_environment(base_env, spec.arm)
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
                record_launch(spec, command, delta, manifest)
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
            record_completion(
                spec,
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
        "schema": 2,
        **spec.identity(),
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
            "schema": 2,
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
        raise base.InconclusivePacket(spec.population, stem, reasons)

    parse_error: base.ContractDefect | None = None
    load: dict[str, object] | None = None
    bench: dict[str, object] | None = None
    resources: dict[str, int] | None = None
    try:
        stdout = stdout_path.read_bytes()
        stderr = stderr_path.read_text(encoding="utf-8")
        if stdout:
            raise base.ContractDefect(f"qwen-bench emitted unexpected stdout: {stem}")
        load = base.parse_load_contract(stderr, spec.arm)
        bench = base.parse_bench(stderr, prompt)
        resources = base.process_resources(stderr)
    except base.ContractDefect as error:
        parse_error = error
    except Exception as error:  # noqa: BLE001 - normalize parser failures
        parse_error = base.ContractDefect(
            f"child parser failed: {type(error).__name__}:{error}"
        )
    if parse_error is not None:
        post = {
            "schema": 2,
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
        "schema": 2,
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
        raise base.InconclusivePacket(spec.population, stem, reasons)
    return row


def rows_match_specs(
    rows: list[dict[str, object]], specs: tuple[ChildSpec, ...]
) -> None:
    if len(rows) != len(specs):
        raise base.ContractDefect("population does not contain its frozen children")
    for row, spec in zip(rows, specs, strict=True):
        for key, expected in spec.identity().items():
            if not base.strict_equal(row.get(key), expected):
                raise base.ContractDefect(
                    f"population identity drifted: {spec.artifact_stem}.{key}"
                )


def validate_ramp(
    ramp_rows: list[dict[str, object]], scored_rows: list[dict[str, object]]
) -> dict[str, object]:
    rows_match_specs(ramp_rows, RAMP_SPECS)
    rows_match_specs(scored_rows, SCORED_SPECS)
    token_arrays = [row["bench"]["token_trace"] for row in ramp_rows]
    if any(not base.strict_equal(value, token_arrays[0]) for value in token_arrays[1:]):
        raise base.ContractDefect("ramp token identity differs")
    conditioning_to_start = [
        row["spawn_completed_monotonic_ns"] - row["conditioning_monotonic_ns"]
        for row in ramp_rows
    ]
    consecutive_start = [
        right["spawn_completed_monotonic_ns"] - left["spawn_completed_monotonic_ns"]
        for left, right in pairwise(ramp_rows)
    ]
    quartet_spans = [
        group[-1]["ended_monotonic_ns"] - group[0]["spawn_completed_monotonic_ns"]
        for group in (ramp_rows[:4], ramp_rows[4:])
    ]
    reversal_span = (
        ramp_rows[-1]["ended_monotonic_ns"]
        - ramp_rows[0]["spawn_completed_monotonic_ns"]
    )
    ramp_to_scored_start = (
        scored_rows[0]["spawn_completed_monotonic_ns"]
        - ramp_rows[-1]["spawn_completed_monotonic_ns"]
    )
    all_rows = [*ramp_rows, *scored_rows]
    no_overlap = all(
        left["ended_monotonic_ns"] <= right["spawn_completed_monotonic_ns"]
        for left, right in pairwise(all_rows)
    )
    gates = {
        "conditioning_to_start": min(conditioning_to_start) >= 0
        and max(conditioning_to_start) <= base.MAX_CONDITIONING_TO_LAUNCH_NS,
        "consecutive_start": min(consecutive_start) > 0
        and max(consecutive_start) <= base.MAX_CONSECUTIVE_START_NS,
        "quartet_span": min(quartet_spans) > 0
        and max(quartet_spans) <= base.MAX_QUARTET_SPAN_NS,
        "reversal_span": 0 < reversal_span <= base.MAX_REVERSAL_SPAN_NS,
        "ramp_to_scored_start": 0
        < ramp_to_scored_start
        <= base.MAX_CONSECUTIVE_START_NS,
        "no_overlap": no_overlap,
    }
    return {
        "schema": 1,
        "children": len(ramp_rows),
        "quartet_orders": list(RAMP_ORDERS),
        "used_for_scoring": False,
        "used_for_stability": False,
        "used_for_medians": False,
        "ramp_observations_used_for_scoring": 0,
        "ramp_observations_used_for_stability": 0,
        "ramp_observations_used_for_medians": 0,
        "gates": gates,
        "passes": all(gates.values()),
        "timing": {
            "conditioning_to_start_s": [value / 1e9 for value in conditioning_to_start],
            "consecutive_start_s": [value / 1e9 for value in consecutive_start],
            "quartet_span_s": [value / 1e9 for value in quartet_spans],
            "reversal_span_s": reversal_span / 1e9,
            "ramp_to_scored_start_s": ramp_to_scored_start / 1e9,
            "first_child": ramp_rows[0]["artifact_stem"],
            "last_child": ramp_rows[-1]["artifact_stem"],
        },
    }


def analyze_packet(
    ramp_rows: list[dict[str, object]], scored_rows: list[dict[str, object]]
) -> dict[str, object]:
    ramp_validation = validate_ramp(ramp_rows, scored_rows)
    if ramp_validation["passes"] is True:
        scored = base.analyze(scored_rows)
        classification = scored["classification"]
    else:
        scored = None
        classification = "inconclusive-ramp-invalid"
    return {
        "schema": 2,
        "stage": "ramp-controlled-loaded-stability",
        "ramp_validation": ramp_validation,
        "scored": scored,
        "classification": classification,
        "claim": "terminal narrow engineering admission",
    }


def verify_attempt_artifacts(rows: list[dict[str, object]]) -> None:
    if len(rows) > len(EXECUTION_SCHEDULE):
        raise base.ContractDefect("attempt ledger exceeds frozen child list")
    specs = EXECUTION_SCHEDULE[: len(rows)]
    for row, spec in zip(rows, specs, strict=True):
        for key, expected in spec.identity().items():
            if not base.strict_equal(row.get(key), expected):
                raise base.ContractDefect("attempt order or identity drifted")
        if row.get("schema") != 2:
            raise base.ContractDefect("attempt schema drifted")
        for suffix, field in (
            ("out", "stdout_sha256"),
            ("err", "stderr_sha256"),
            ("conditioning.json", "conditioning_sha256"),
            ("post-exit.json", "post_exit_sha256"),
        ):
            path = base.ARTIFACT / f"{spec.artifact_stem}.{suffix}"
            if not path.is_file() or base.sha256(path) != row.get(field):
                raise base.ContractDefect(
                    f"attempt raw artifact binding drifted: {path.name}"
                )
    events = base.read_jsonl(base.ARTIFACT / "launch-seal.jsonl")
    has_golden = bool(
        rows and rows[0].get("valid") is True and isinstance(rows[0].get("bench"), dict)
    )
    if len(events) != 2 * len(rows) + int(has_golden):
        raise base.ContractDefect("launch/completion/golden event count drifted")
    cursor = 0
    for index, (row, spec) in enumerate(zip(rows, specs, strict=True)):
        launch, completion = events[cursor : cursor + 2]
        cursor += 2
        for event, event_name in ((launch, "launch"), (completion, "completion")):
            if event.get("event") != event_name or event.get("schema") != 2:
                raise base.ContractDefect(f"{event_name} event order drifted")
            for key, expected in spec.identity().items():
                if not base.strict_equal(event.get(key), expected):
                    raise base.ContractDefect(f"{event_name} identity drifted")
        if completion.get("returncode") != row.get("returncode"):
            raise base.ContractDefect("completion return code drifted")
        if index == 0 and has_golden:
            golden = events[cursor]
            cursor += 1
            if (
                golden.get("event") != "golden-token-trace"
                or golden.get("population") != "ramp"
                or golden.get("used_for_scoring") is not False
                or golden.get("source_artifact_stem") != spec.artifact_stem
                or not base.strict_equal(
                    golden.get("token_trace"), row["bench"]["token_trace"]
                )
            ):
                raise base.ContractDefect("golden token event drifted")
    if cursor != len(events):
        raise base.ContractDefect("unconsumed launch-seal events")


def make_decision(
    manifest: dict[str, object],
    correctness: dict[str, object] | None,
    token_protocol: dict[str, object] | None,
    stage: dict[str, object] | None,
    status: str,
    stopped_after: str,
    failed_child: str | None,
    reasons: list[str],
) -> dict[str, object]:
    decision = prior.make_decision(
        manifest,
        correctness,
        token_protocol,
        stage,
        status,
        stopped_after,
        failed_child,
        reasons,
    )
    attempts = base.read_jsonl(base.ARTIFACT / "attempts.jsonl")
    ramp_attempts = [row for row in attempts if row.get("population") == "ramp"]
    scored_attempts = [row for row in attempts if row.get("population") == "scored"]
    go = status == "go"
    decision.update(
        {
            "schema": 2,
            "successor_authorization": "none",
            "forensic_v0633": manifest.get("forensic_v0633"),
            "ramp_validation": (
                stage.get("ramp_validation") if isinstance(stage, dict) else None
            ),
            "execution_population": {
                "ramp_planned_children": len(RAMP_SPECS),
                "ramp_attempted_children": len(ramp_attempts),
                "ramp_valid_children": sum(
                    row.get("valid") is True for row in ramp_attempts
                ),
                "scored_planned_children": len(SCORED_SPECS),
                "scored_attempted_children": len(scored_attempts),
                "scored_valid_children": sum(
                    row.get("valid") is True for row in scored_attempts
                ),
                "ramp_observations_used_for_scoring": 0,
                "ramp_observations_used_for_stability": 0,
                "ramp_observations_used_for_medians": 0,
            },
            "profile_disposition": (
                "admitted-explicit-force-only-dense27b-direct-pread"
                if go
                else "closed-unable-to-certify-frozen-loaded-noninferiority"
            ),
            "terminal_packet": True,
        }
    )
    claim = decision.get("claim_scope")
    if not isinstance(claim, dict):
        raise base.ContractDefect("decision claim scope is malformed")
    claim.update(
        {
            "ramp_observations_used_for_scoring": 0,
            "ramp_observations_used_for_stability": 0,
            "ramp_observations_used_for_medians": 0,
            "terminal_packet": True,
        }
    )
    return decision


def make_preflight_failure(
    status: str, stage: str, failed_child: str | None, reasons: list[str]
) -> dict[str, object]:
    failure = {
        "schema": 2,
        "status": status,
        "stage": stage,
        "failed_child": failed_child,
        "reasons": reasons,
    }
    base.write_json(base.ARTIFACT / "preflight-failure.json", failure)
    return {
        "schema": 2,
        "status": status,
        "authority": "none",
        "force_authorized": False,
        "successor_authorization": "none",
        "stopped_after": stage,
        "failed_child": failed_child,
        "reasons": reasons,
        "source_commit": None,
        "correctness": None,
        "token_protocol": None,
        "stage": None,
        "attempts_sha256": base.sha256(base.ARTIFACT / "attempts.jsonl"),
        "claim_scope": {
            "profile": "dense27b-q4km-v1",
            "selection": "explicit-QWEN_GGUF_PARALLEL_COPY=pread",
            "auto_default_authorized": False,
            "performance_imports_used_for_scoring": 0,
            "ramp_observations_used_for_scoring": 0,
            "ramp_observations_used_for_stability": 0,
            "ramp_observations_used_for_medians": 0,
            "terminal_packet": True,
        },
        "forensic_v0633": None,
        "ramp_validation": None,
        "execution_population": {
            "ramp_planned_children": len(RAMP_SPECS),
            "ramp_attempted_children": 0,
            "ramp_valid_children": 0,
            "scored_planned_children": len(SCORED_SPECS),
            "scored_attempted_children": 0,
            "scored_valid_children": 0,
            "ramp_observations_used_for_scoring": 0,
            "ramp_observations_used_for_stability": 0,
            "ramp_observations_used_for_medians": 0,
        },
        "profile_disposition": (
            "closed-unable-to-certify-frozen-loaded-noninferiority"
        ),
        "terminal_packet": True,
    }


def interruption_reason(packet_signals: list[int]) -> str:
    return f"operator_sigint_deferred={len(packet_signals)}"


def check_packet_interrupt(
    packet_signals: list[int], stage: str, child: str | None = None
) -> None:
    if packet_signals:
        raise base.InconclusivePacket(
            stage,
            child,
            [interruption_reason(packet_signals)],
        )


def execute_reserved_packet(packet_signals: list[int]) -> dict[str, object]:
    try:
        check_packet_interrupt(packet_signals, "reservation")
        child_env, test_env, removed = base.safe_environments()
        manifest = build_manifest(child_env, test_env, removed)
        prompt = base.PROMPT.read_text(encoding="utf-8")
        if len(prompt.encode("utf-8")) != base.EXPECTED_PROMPT_BYTES:
            raise base.ContractDefect("prompt text decode changed byte count")
        check_packet_interrupt(packet_signals, "preflight")
    except base.InconclusivePacket as error:
        return make_preflight_failure(
            "inconclusive", error.stage, error.child, error.reasons
        )
    except base.ContractDefect as error:
        return make_preflight_failure(
            "implementation_or_contract_defect",
            "preflight",
            None,
            [str(error)],
        )
    except base.UnsealedPacket:
        raise
    except KeyboardInterrupt:
        return make_preflight_failure(
            "inconclusive",
            "preflight",
            None,
            ["operator_keyboard_interrupt"],
        )
    except Exception as error:  # noqa: BLE001 - seal preflight failures
        return make_preflight_failure(
            "implementation_or_contract_defect",
            "preflight",
            None,
            [f"{type(error).__name__}:{error}"],
        )
    base.write_json(base.ARTIFACT / "manifest.json", manifest)
    base.fsync_directory(base.ARTIFACT)
    correctness: dict[str, object] | None = None
    token_protocol: dict[str, object] | None = None
    stage: dict[str, object] | None = None
    status = "implementation_or_contract_defect"
    stopped_after = "preflight"
    failed_child: str | None = None
    reasons: list[str] = []
    rows: list[dict[str, object]] = []
    ramp_rows: list[dict[str, object]] = []
    scored_rows: list[dict[str, object]] = []
    active_spec: ChildSpec | None = None
    try:
        stopped_after = "token-protocol"
        token_protocol = base.run_token_protocol_check(child_env)
        check_packet_interrupt(packet_signals, stopped_after)
        stopped_after = "correctness"
        correctness = base.run_correctness(test_env)
        check_packet_interrupt(packet_signals, stopped_after)
        golden: list[int] | None = None
        for spec in EXECUTION_SCHEDULE:
            active_spec = spec
            stopped_after = spec.population
            row = run_child(child_env, manifest, prompt, spec, packet_signals)
            rows.append(row)
            (scored_rows if spec.used_for_scoring else ramp_rows).append(row)
            if golden is None:
                if spec != RAMP_SPECS[0]:
                    raise base.ChildContractDefect(
                        spec.artifact_stem,
                        "first valid child is not frozen ramp A golden",
                    )
                golden = row["bench"]["token_trace"]
                record_golden(row)
            elif not base.strict_equal(row["bench"]["token_trace"], golden):
                raise base.ChildContractDefect(
                    spec.artifact_stem,
                    f"timed token identity differs: {spec.artifact_stem}",
                )
            check_packet_interrupt(packet_signals, spec.population, spec.artifact_stem)
        active_spec = None
        stopped_after = "ramp-controlled-loaded-stability"
        stage = analyze_packet(ramp_rows, scored_rows)
        check_packet_interrupt(packet_signals, stopped_after)
        classification = stage["classification"]
        if classification == "go":
            status = "go"
        elif classification == "kill":
            status = "kill"
        else:
            status = "inconclusive"
            reasons = [classification]
    except base.InconclusivePacket as error:
        status = "inconclusive"
        stopped_after = error.stage
        failed_child = error.child
        reasons = error.reasons
    except base.ChildContractDefect as error:
        status = "implementation_or_contract_defect"
        failed_child = error.child
        reasons = [str(error)]
    except base.ContractDefect as error:
        status = "implementation_or_contract_defect"
        failed_child = (
            active_spec.artifact_stem
            if active_spec is not None
            else (rows[-1]["artifact_stem"] if rows else None)
        )
        reasons = [str(error)]
    except base.UnsealedPacket:
        raise
    except KeyboardInterrupt:
        status = "inconclusive"
        failed_child = active_spec.artifact_stem if active_spec is not None else None
        reasons = ["operator_keyboard_interrupt"]
    except Exception as error:  # noqa: BLE001 - seal packet failures
        status = "implementation_or_contract_defect"
        failed_child = (
            active_spec.artifact_stem
            if active_spec is not None
            else (rows[-1]["artifact_stem"] if rows else None)
        )
        reasons = [f"{type(error).__name__}:{error}"]
    try:
        base.final_identity(manifest)
    except base.ContractDefect as error:
        status = "implementation_or_contract_defect"
        stopped_after = "final-identity"
        failed_child = None
        reasons = [str(error)]
    except base.UnsealedPacket:
        raise
    except KeyboardInterrupt:
        status = "inconclusive"
        stopped_after = "final-identity"
        failed_child = None
        reasons = ["operator_keyboard_interrupt"]
    except Exception as error:  # noqa: BLE001 - seal identity failures
        status = "implementation_or_contract_defect"
        stopped_after = "final-identity"
        failed_child = None
        reasons = [f"{type(error).__name__}:{error}"]
    if packet_signals and status != "implementation_or_contract_defect":
        status = "inconclusive"
        reasons = [interruption_reason(packet_signals)]
    return make_decision(
        manifest,
        correctness,
        token_protocol,
        stage,
        status,
        stopped_after,
        failed_child,
        reasons,
    )


def apply_late_interrupt(
    decision: dict[str, object], packet_signals: list[int]
) -> None:
    if (
        not packet_signals
        or decision.get("status") == "implementation_or_contract_defect"
    ):
        return
    decision.update(
        {
            "status": "inconclusive",
            "authority": "none",
            "force_authorized": False,
            "successor_authorization": "none",
            "reasons": [interruption_reason(packet_signals)],
            "profile_disposition": (
                "closed-unable-to-certify-frozen-loaded-noninferiority"
            ),
        }
    )


def run_packet(*, preflight_only: bool) -> None:
    if base.ARTIFACT.exists():
        raise RuntimeError(f"refusing to reuse artifact directory: {base.ARTIFACT}")
    if preflight_only:
        child_env, test_env, removed = base.safe_environments()
        manifest = build_manifest(child_env, test_env, removed)
        prompt = base.PROMPT.read_text(encoding="utf-8")
        if len(prompt.encode("utf-8")) != base.EXPECTED_PROMPT_BYTES:
            raise base.ContractDefect("prompt text decode changed byte count")
        print(
            base.json_text(
                {
                    "status": "preflight_pass",
                    "source_commit": manifest["source_commit"],
                    "ramp_quartets": len(RAMP_ORDERS),
                    "ramp_children": len(RAMP_SPECS),
                    "scored_quartets": len(SCORED_ORDERS),
                    "scored_children": len(SCORED_SPECS),
                    "total_children": len(EXECUTION_SCHEDULE),
                    "model_sha256": manifest["sha256"][str(base.MODEL)],
                },
                pretty=True,
            )
        )
        return
    packet_signals: list[int] = []

    def defer_packet_sigint(signum: int, _frame: object) -> None:
        packet_signals.append(signum)

    previous_handler = signal.signal(signal.SIGINT, defer_packet_sigint)
    handler_installed = True
    try:
        base.ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
        base.ARTIFACT.mkdir()
        base.fsync_directory(base.ARTIFACT.parent)
        base.write_fsynced(base.ARTIFACT / "attempts.jsonl", b"")
        base.write_fsynced(base.ARTIFACT / "launch-seal.jsonl", b"")
        base.fsync_directory(base.ARTIFACT)
        decision = execute_reserved_packet(packet_signals)
        old_mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT})
        try:
            if signal.SIGINT in signal.sigpending():
                packet_signals.append(signal.SIGINT)
            apply_late_interrupt(decision, packet_signals)
            base.publish(decision)
            print(base.json_text(decision, pretty=True))
        finally:
            signal.signal(signal.SIGINT, previous_handler)
            handler_installed = False
            signal.pthread_sigmask(signal.SIG_SETMASK, old_mask)
    finally:
        if handler_installed:
            signal.signal(signal.SIGINT, previous_handler)


def synthetic_rows(
    *, ramp_variant: int
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    token_trace = list(range(base.EXPECTED_TRACE_TOKENS))
    origin = 1_000_000_000_000
    for spec in EXECUTION_SCHEDULE:
        start = origin + (spec.execution_index - 1) * 7_000_000_000
        if spec.population == "ramp":
            direction = 1.0 if spec.arm == "A" else 11.0
            seed = float(spec.execution_index + 3 * spec.position)
            if ramp_variant == 1:
                prefill = direction * seed
                decode = (30.0 - direction) * (seed + 1.0)
                request = (direction + spec.position) * (seed + 2.0)
                cpu = (2.0 * direction + spec.position) * (seed + 3.0)
                rss = 100 + spec.execution_index * (17 if spec.arm == "A" else 31)
                footprint = 200 + spec.position * (41 if spec.arm == "A" else 13)
            else:
                prefill = (40.0 - direction) * (seed + 7.0)
                decode = (direction + 5.0) * (seed + 11.0)
                request = (17.0 / direction) * (seed + 13.0)
                cpu = (direction + 19.0) * (seed + 17.0)
                rss = 10_000 + spec.position * (97 if spec.arm == "A" else 7)
                footprint = 20_000 + spec.execution_index * (
                    5 if spec.arm == "A" else 101
                )
        else:
            prefill = 1000.0
            decode = 1000.0
            request = 2000.0
            cpu = 1000.0
            rss = 1000
            footprint = 1000
        row = {
            "schema": 2,
            **spec.identity(),
            "conditioning_monotonic_ns": start - 1_000_000,
            "spawn_completed_monotonic_ns": start,
            "ended_monotonic_ns": start + 5_000_000_000,
            "spawn_to_exit_ns": 5_000_000_000,
            "valid": True,
            "bench": {
                "prefill_ms": prefill,
                "decode_ms": decode,
                "request_ms": request,
                "token_trace": token_trace,
                "token_trace_sha256": "trace",
                "generated_debug_sha256": "debug",
            },
            "process_resources": {
                "total_cpu_ms": cpu,
                "maximum_resident_set_size": rss,
                "peak_memory_footprint": footprint,
            },
        }
        rows.append(row)
    return rows[: len(RAMP_SPECS)], rows[len(RAMP_SPECS) :]


def run_self_test() -> None:
    if (
        len(EXECUTION_SCHEDULE) != 40
        or len(RAMP_SPECS) != 8
        or len(SCORED_SPECS) != 32
        or len({spec.artifact_stem for spec in EXECUTION_SCHEDULE}) != 40
        or "".join(spec.arm for spec in RAMP_SPECS) != "ABBABAAB"
        or tuple(spec.quartet_order for spec in SCORED_SPECS[::4]) != SCORED_ORDERS
    ):
        raise AssertionError("frozen execution schedule is malformed")
    ramp_a, scored_a = synthetic_rows(ramp_variant=1)
    ramp_b, scored_b = synthetic_rows(ramp_variant=2)
    result_a = analyze_packet(ramp_a, scored_a)
    result_b = analyze_packet(ramp_b, scored_b)
    if not base.strict_equal(result_a, result_b):
        raise AssertionError("ramp performance contaminated scored analysis")
    scored_text = base.json_text(result_a["scored"])
    if "ramp-" in scored_text or result_a["classification"] != "go":
        raise AssertionError("scored analysis contains ramp evidence")
    invalid_ramp, valid_scored = synthetic_rows(ramp_variant=1)
    invalid_ramp[1]["spawn_completed_monotonic_ns"] += 20_000_000_000
    original_analyze = base.analyze
    scorer_calls = 0

    def forbidden_analyze(_rows: list[dict[str, object]]) -> dict[str, object]:
        nonlocal scorer_calls
        scorer_calls += 1
        raise AssertionError("invalid ramp invoked the scored analyzer")

    base.analyze = forbidden_analyze
    try:
        invalid_result = analyze_packet(invalid_ramp, valid_scored)
    finally:
        base.analyze = original_analyze
    if (
        scorer_calls != 0
        or invalid_result["classification"] != "inconclusive-ramp-invalid"
        or invalid_result["scored"] is not None
    ):
        raise AssertionError("invalid ramp evidence did not suppress scoring")
    interrupted = {
        "status": "go",
        "authority": "explicit-force-only-dense27b-direct-pread",
        "force_authorized": True,
        "successor_authorization": "none",
        "reasons": [],
        "profile_disposition": ("admitted-explicit-force-only-dense27b-direct-pread"),
    }
    apply_late_interrupt(interrupted, [signal.SIGINT])
    if (
        interrupted["status"] != "inconclusive"
        or interrupted["authority"] != "none"
        or interrupted["force_authorized"] is not False
    ):
        raise AssertionError("late interruption retained performance authority")
    print(
        base.json_text(
            {
                "status": "self_test_pass",
                "ramp_children": len(RAMP_SPECS),
                "scored_children": len(SCORED_SPECS),
                "total_children": len(EXECUTION_SCHEDULE),
                "ramp_performance_invariant": True,
            },
            pretty=True,
        )
    )


def configure() -> None:
    prior.configure()
    base.ARTIFACT = ARTIFACT
    base.PREREG = PREREG
    base.RUNNER = RUNNER
    base.verify_source_topology = verify_source_topology
    base.build_manifest = build_manifest
    base.verify_attempt_artifacts = verify_attempt_artifacts
    base.make_decision = make_decision


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--preflight-only", action="store_true")
    mode.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        run_self_test()
        return
    configure()
    run_packet(preflight_only=args.preflight_only)


if __name__ == "__main__":
    main()
