#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Secondary pilot guard, not core admission or permission to overcommit.

Run one pilot, then authorize_repetitions(timed_native_samples_ns) before any
further commands. Failures latch scheduling closed. No retries or Metal lease.
Output and JSONL events use caller-selected new files. Use one scheduling thread
and retain the Session until its owned child exits; never reap it elsewhere.
"""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
from typing import Sequence


@dataclass(frozen=True)
class Policy:
    max_spread: float = 1.5
    min_available_bytes: int = 512 * 1024**2
    max_nonreclaimable_fraction: float = 0.90
    interrupt_grace_s: float = 3.0
    terminate_grace_s: float = 2.0
    poll_s: float = 0.05
    health_interval_s: float = 2.0

    def __post_init__(self):
        if not (1 <= self.max_spread <= 1.5):
            raise ValueError("max_spread must be between 1 and 1.5")
        if (
            type(self.min_available_bytes) is not int
            or self.min_available_bytes <= 0
            or not (0 < self.max_nonreclaimable_fraction < 1)
        ):
            raise ValueError("positive memory headroom and fraction below 1 required")
        bounds = (self.interrupt_grace_s, self.terminate_grace_s, self.poll_s)
        if not all(math.isfinite(v) and v >= 0 for v in bounds) or self.poll_s == 0:
            raise ValueError("finite cancellation bounds and positive poll_s required")
        if not math.isfinite(self.health_interval_s) or self.health_interval_s <= 0:
            raise ValueError("finite positive health_interval_s required")


@dataclass(frozen=True)
class Snapshot:
    free_bytes: int
    available_bytes: int
    wired_bytes: int
    compressed_bytes: int
    physical_bytes: int

    def __post_init__(self):
        if any(type(v) is not int or v < 0 for v in asdict(self).values()):
            raise ValueError("snapshot values must be nonnegative byte counts")
        accounted = self.available_bytes + self.nonreclaimable_bytes
        valid = 0 < self.physical_bytes >= accounted
        if not valid or self.free_bytes > self.available_bytes:
            raise ValueError("inconsistent snapshot accounting")

    @property
    def nonreclaimable_bytes(self):
        return self.wired_bytes + self.compressed_bytes


def _stop_owned(child, policy, emit=lambda *args, **kwargs: None):
    for sig, grace in (
        (signal.SIGINT, policy.interrupt_grace_s),
        (signal.SIGTERM, policy.terminate_grace_s),
    ):
        if child.poll() is not None:
            return
        # Never reap this handle elsewhere or signal a group.
        try:
            child.send_signal(sig)
        except ProcessLookupError:
            pass
        try:
            emit("signal", pid=child.pid, signal=sig.name)
        except OSError:
            pass
        try:
            child.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            pass
    if child.poll() is None:
        emit("child_still_running", pid=child.pid)


def _host_output(argv):
    child = subprocess.Popen(
        argv, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True
    )
    try:
        text, _ = child.communicate(timeout=2)
        if child.returncode:
            raise OSError(f"host telemetry failed: {argv}")
        return text
    finally:
        _stop_owned(child, Policy(interrupt_grace_s=0.2, terminate_grace_s=0.2))
        child.stdout.close()


def sample_host_health() -> Snapshot:
    """Conservative macOS health: available=free; nonreclaimable=wired+compressor.
    Never sum overlapping speculative/file-backed/active/purgeable counters or
    use logical compressed-page counts as resident bytes.
    """
    if sys.platform != "darwin":
        raise OSError("macOS telemetry unavailable; inject a host sampler")
    command = ["/usr/sbin/sysctl", "-n", "hw.memsize"]
    physical = int(_host_output(command))
    text = _host_output(["/usr/bin/vm_stat"])
    page_size = int(re.search(r"page size of (\d+) bytes", text).group(1))
    counts = dict(re.findall(r"^([^:\n]+):\s+(\d+)\.", text, re.MULTILINE))
    free = int(counts["Pages free"]) * page_size
    wired = int(counts["Pages wired down"]) * page_size
    compressed = int(counts["Pages occupied by compressor"]) * page_size
    return Snapshot(free, free, wired, compressed, physical)


@dataclass(frozen=True)
class Assessment:
    passed: bool
    reasons: tuple[str, ...]
    samples_ns: tuple[float, ...]
    spread: float | None
    before: Snapshot | None
    after: Snapshot | None


def _health_reasons(snapshot, label, policy):
    if not isinstance(snapshot, Snapshot):
        return [f"{label}: required host telemetry unavailable"]
    reasons = []
    floor = max(policy.min_available_bytes, (snapshot.physical_bytes + 63) // 64)
    if snapshot.available_bytes < floor:
        reasons.append(f"{label}: available bytes below {floor}")
    fraction = snapshot.nonreclaimable_bytes / snapshot.physical_bytes
    if fraction > policy.max_nonreclaimable_fraction:
        reasons.append(f"{label}: wired + compressed fraction {fraction:.4g} too high")
    return reasons


def assess_pilot(samples_ns, before=None, after=None, *, policy=Policy()) -> Assessment:
    """Pure assessment. Missing before/after evidence rejects, never opts out."""
    samples, reasons = [], []
    for index, value in enumerate(samples_ns):
        try:
            if isinstance(value, (bool, str, bytes)):
                raise ValueError
            number = float(value)
            if not math.isfinite(number) or number <= 0:
                raise ValueError
            samples.append(number)
        except (TypeError, ValueError, OverflowError):
            reasons.append(f"sample {index}: invalid finite positive ns: {value!r}")
    if len(samples) < 2:
        reasons.append("at least two finite positive pilot samples required")
    spread = max(samples) / min(samples) if len(samples) >= 2 else None
    if spread is not None and (not math.isfinite(spread) or spread > policy.max_spread):
        reasons.append(f"pilot max/min spread {spread:.6g} exceeds {policy.max_spread}")
    reasons += _health_reasons(before, "before", policy)
    reasons += _health_reasons(after, "after", policy)
    if spread is not None and not math.isfinite(spread):
        spread = None
    return Assessment(
        not reasons, tuple(reasons), tuple(samples), spread, before, after
    )


class GuardStopped(RuntimeError):
    pass


@dataclass(frozen=True)
class CommandResult:
    pid: int
    returncode: int | None
    reason: str


class Session:
    def __init__(self, policy: Policy = Policy(), *, sampler=sample_host_health):
        self.policy, self.sampler = policy, sampler
        self.child: subprocess.Popen | None = None
        self.assessment: Assessment | None = None
        self._started = self._authorized = self._stopped = False
        self._before = self._after = self._events_path = None

    def _event(self, event, *, durable=True, **fields):
        record = dict(event=event, time_ns=time.time_ns(), **fields)
        record["monotonic_ns"] = time.monotonic_ns()
        with self._events_path.open("a", encoding="ascii") as stream:
            stream.write(json.dumps(record, ensure_ascii=True, allow_nan=False) + "\n")
            stream.flush()
            if durable:
                os.fsync(stream.fileno())

    def _sample(self):
        try:
            snapshot = self.sampler()
            if not isinstance(snapshot, Snapshot):
                raise ValueError("sampler did not return Snapshot")
            return snapshot
        except Exception as error:
            self._event("telemetry_unavailable", error=str(error))
            return None

    def authorize_repetitions(self, samples_ns: Sequence[float]) -> Assessment:
        if self._stopped or not self._started or self._authorized:
            raise GuardStopped("authorization requires one completed, successful pilot")
        self._stopped = True
        self.assessment = assess_pilot(
            samples_ns, self._before, self._after, policy=self.policy
        )
        self._event("pilot_assessment", assessment=asdict(self.assessment))
        self._authorized = self.assessment.passed
        self._stopped = not self._authorized
        return self.assessment

    def _cancel(self):
        _stop_owned(self.child, self.policy, self._event)

    def run_command(
        self,
        argv: Sequence[str],
        *,
        stdout_path: Path,
        stderr_path: Path,
        events_path: Path,
        timeout_s: float,
        stop_file: Path | None = None,
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
    ) -> CommandResult:
        """Stream to exclusive files. After bounded INT/TERM waits, return even
        if self.child remains alive; keep it for observation, never SIGKILL.
        """
        if self._stopped or (self._started and not self._authorized):
            raise GuardStopped("session stopped or pilot not authorized; no launch")
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("finite positive timeout_s required")
        if isinstance(argv, (str, bytes)) or not argv:
            raise ValueError("argv must be a nonempty argument sequence")
        self._stopped = True
        pilot = not self._started
        self._started = True
        self._events_path = Path(events_path)
        with ExitStack() as stack:
            stack.enter_context(self._events_path.open("x", encoding="ascii"))
            out = stack.enter_context(Path(stdout_path).open("xb"))
            err = stack.enter_context(Path(stderr_path).open("xb"))
            before = self._sample()
            self._event("preflight", snapshot=asdict(before) if before else None)
            reasons = _health_reasons(before, "preflight", self.policy)
            if stop_file is not None and Path(stop_file).exists():
                reasons.append("stop file present")
            if reasons:
                self._event("blocked", reasons=reasons)
                raise GuardStopped("; ".join(reasons))
            if pilot:
                self._before = before
            self._event("launch", argv=list(argv), pilot=pilot)
            started = time.monotonic()
            try:
                self.child = subprocess.Popen(
                    argv,
                    stdin=subprocess.DEVNULL,
                    stdout=out,
                    stderr=err,
                    cwd=cwd,
                    env=env,
                    shell=False,
                )
            except OSError as error:
                self._event("spawn_failed", error=str(error))
                raise
            reason = "completed"
            health = []
            next_health = started + self.policy.health_interval_s
            try:
                self._event("started", pid=self.child.pid)
                while self.child.poll() is None:
                    if stop_file is not None and Path(stop_file).exists():
                        reason = "stop_file"
                        break
                    now = time.monotonic()
                    if now - started >= timeout_s:
                        reason = "timeout"
                        break
                    if now >= next_health:
                        snapshot = self._sample()
                        health = _health_reasons(snapshot, "during", self.policy)
                        self._event(
                            "health",
                            pid=self.child.pid,
                            snapshot=asdict(snapshot) if snapshot else None,
                            health_reasons=health,
                            durable=bool(health),
                        )
                        if health:
                            reason = "unhealthy"
                            break
                        next_health = time.monotonic() + self.policy.health_interval_s
                    time.sleep(self.policy.poll_s)
            except KeyboardInterrupt:
                reason = "interrupted"
            finally:
                if self.child.poll() is None:
                    self._cancel()
            if stop_file is not None and Path(stop_file).exists():
                reason = "stop_file"
            if reason == "completed" and self.child.returncode != 0:
                reason = "exit_nonzero"
            after = self._sample()
            if pilot:
                self._after = after
            health += _health_reasons(after, "after", self.policy)
            if reason == "completed" and health:
                reason = "unhealthy"
            result = CommandResult(self.child.pid, self.child.returncode, reason)
            self._event(
                "finished",
                **asdict(result),
                snapshot=asdict(after) if after else None,
                health_reasons=health,
            )
            self._stopped = reason != "completed"
            return result
