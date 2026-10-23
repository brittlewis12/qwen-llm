"""CPU only; parent runs: uv run --no-project python -B -m unittest discover -s scripts/bench -p test_pilot.py"""

from dataclasses import FrozenInstanceError, asdict, replace
import json
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pilot


GIB = 1024**3
HEALTHY = pilot.Snapshot(4 * GIB, 4 * GIB, 2 * GIB, GIB, 16 * GIB)
FAST = pilot.Policy(interrupt_grace_s=0.15, terminate_grace_s=0.15, poll_s=0.01)


class AssessmentTests(unittest.TestCase):
    def assess(self, samples, before=HEALTHY, after=HEALTHY):
        return pilot.assess_pilot(samples, before, after)

    def test_cdf_catastrophic_spread_rejected_and_persistable(self):
        assessment = self.assess([18.52e9, 34.16e9])
        self.assertFalse(assessment.passed)
        self.assertAlmostEqual(assessment.spread, 1.8444924406047516)
        self.assertIn("spread", assessment.reasons[0])
        json.dumps(asdict(assessment), allow_nan=False)

    def test_two_samples_and_inclusive_spread_boundary(self):
        self.assertTrue(self.assess([10, 15]).passed)
        self.assertFalse(self.assess([10, 15.00001]).passed)
        for samples in (
            [],
            [10],
            [1, 1, float("nan")],
            [0, 1],
            [-1, 1],
            [float("inf"), 1],
            [True, 1],
            ["1", 1],
            [None, 1],
        ):
            with self.subTest(samples=samples):
                assessment = self.assess(samples)
                self.assertFalse(assessment.passed)
                json.dumps(asdict(assessment), allow_nan=False)

    def test_missing_or_unhealthy_telemetry_fails_closed(self):
        low_free = replace(HEALTHY, free_bytes=1, available_bytes=1)
        high_wired = pilot.Snapshot(GIB // 2, GIB // 2, 14 * GIB, GIB, 16 * GIB)
        for snapshot in (None, low_free, high_wired):
            self.assertFalse(self.assess([10, 10], before=snapshot).passed)
            self.assertFalse(self.assess([10, 10], after=snapshot).passed)
        self.assertFalse(pilot.assess_pilot([10, 10]).passed)

    def test_available_floor_scales_with_physical_ram(self):
        large = replace(HEALTHY, physical_bytes=256 * GIB)
        self.assertTrue(self.assess([10, 10], large, large).passed)
        low = replace(large, free_bytes=2 * GIB, available_bytes=2 * GIB)
        assessment = self.assess([10, 10], large, low)
        self.assertFalse(assessment.passed)
        self.assertIn(str(4 * GIB), assessment.reasons[0])
        small = replace(HEALTHY, free_bytes=GIB // 2, available_bytes=GIB // 2)
        self.assertTrue(self.assess([10, 10], small, small).passed)

    def test_health_interval_is_positive_and_finite(self):
        self.assertEqual(pilot.Policy().health_interval_s, 2)
        for value in (0, -1, float("inf"), float("nan")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                pilot.Policy(health_interval_s=value)

    def test_policy_and_snapshots_are_immutable_and_accounting_is_validated(self):
        with self.assertRaises(FrozenInstanceError):
            FAST.max_spread = 2
        with self.assertRaises(FrozenInstanceError):
            HEALTHY.free_bytes = 0
        with self.assertRaises(ValueError):
            pilot.Policy(max_spread=2)
        with self.assertRaises(ValueError):
            replace(HEALTHY, available_bytes=16 * GIB)
        self.assertEqual(HEALTHY.nonreclaimable_bytes, 3 * GIB)

    def test_macos_sampler_does_not_add_overlapping_or_logical_counters(self):
        vm_stat = """Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free: 100.
Pages active: 900.
Pages inactive: 800.
Pages speculative: 50.
Pages wired down: 200.
Pages purgeable: 300.
File-backed pages: 1000.
Anonymous pages: 600.
Pages stored in compressor: 2000.
Pages occupied by compressor: 150.
"""
        with (
            patch.object(pilot.sys, "platform", "darwin"),
            patch.object(
                pilot, "_host_output", side_effect=[str(16 * GIB), vm_stat]
            ) as read,
        ):
            snapshot = pilot.sample_host_health()
        self.assertEqual(snapshot.free_bytes, 100 * 16384)
        self.assertEqual(snapshot.available_bytes, 300 * 16384)
        self.assertEqual(snapshot.nonreclaimable_bytes, 350 * 16384)
        self.assertEqual(read.call_count, 2)

    def test_idle_file_cache_is_not_a_free_page_exhaustion(self):
        pages_per_gib = GIB // 16384
        snapshot = pilot.snapshot_from_vm_stat(
            256 * GIB,
            f"""Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free: {3 * pages_per_gib}.
Pages inactive: {220 * pages_per_gib}.
Anonymous pages: {4 * pages_per_gib}.
Pages wired down: {8 * pages_per_gib}.
Pages occupied by compressor: {pages_per_gib}.
""",
        )
        self.assertLess(snapshot.free_bytes, 4 * GIB)
        self.assertGreater(snapshot.available_bytes, 200 * GIB)
        self.assertTrue(self.assess([10, 10], snapshot, snapshot).passed)

    def test_anonymous_inactive_pages_never_receive_file_cache_credit(self):
        snapshot = pilot.snapshot_from_vm_stat(
            16 * GIB,
            """Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free: 100.
Pages inactive: 800.
Anonymous pages: 900.
Pages wired down: 200.
Pages occupied by compressor: 150.
""",
        )
        self.assertEqual(snapshot.available_bytes, snapshot.free_bytes)
        self.assertFalse(self.assess([10, 10], snapshot, snapshot).passed)

    def test_sampler_refuses_missing_or_invalid_availability_categories(self):
        text = """Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free: 100.
Pages inactive: 800.
Anonymous pages: 600.
Pages wired down: 200.
Pages occupied by compressor: 150.
"""
        for bad in (
            text.replace("Anonymous pages: 600.\n", ""),
            text.replace("Pages inactive: 800.\n", ""),
            text.replace("page size of 16384", "page size of 3"),
            text.replace("Pages inactive: 800.", f"Pages inactive: {16 * GIB}."),
        ):
            with self.subTest(bad=bad), self.assertRaises((KeyError, ValueError)):
                pilot.snapshot_from_vm_stat(16 * GIB, bad)

    def test_telemetry_timeout_also_never_forces_a_kill(self):
        child = Mock(pid=123, returncode=None)
        child.poll.return_value = None
        child.communicate.side_effect = subprocess.TimeoutExpired("telemetry", 2)
        child.wait.side_effect = subprocess.TimeoutExpired("telemetry", 0.2)
        with patch.object(pilot.subprocess, "Popen", return_value=child):
            with self.assertRaises(subprocess.TimeoutExpired):
                pilot._host_output(["inert-telemetry-fixture"])
        self.assertEqual(
            [call.args[0] for call in child.send_signal.call_args_list],
            [signal.SIGINT, signal.SIGTERM],
        )
        child.kill.assert_not_called()
        child.stdout.close.assert_called_once()


class SessionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.session = pilot.Session(FAST, sampler=lambda: HEALTHY)
        self.index = 0

    def run_child(self, code="print('toy output')", **kwargs):
        self.index += 1
        self.events = self.root / f"{self.index}.jsonl"
        return self.session.run_command(
            [sys.executable, "-B", "-c", code],
            stdout_path=self.root / f"{self.index}.out",
            stderr_path=self.root / f"{self.index}.err",
            events_path=self.events,
            timeout_s=kwargs.pop("timeout_s", 2),
            **kwargs,
        )

    def timeline(self):
        return [json.loads(line) for line in self.events.read_text().splitlines()]

    def test_success_requires_explicit_authorization_and_streams_to_files(self):
        with self.assertRaises(pilot.GuardStopped):
            self.session.authorize_repetitions([10, 10])
        result = self.run_child(
            "import sys; print('out'); print('err', file=sys.stderr)"
        )
        self.assertEqual((result.returncode, result.reason), (0, "completed"))
        self.assertEqual((self.root / "1.out").read_text(), "out\n")
        self.assertEqual((self.root / "1.err").read_text(), "err\n")
        with patch.object(pilot.subprocess, "Popen") as spawn:
            with self.assertRaises(pilot.GuardStopped):
                self.run_child()
            spawn.assert_not_called()
        self.assertTrue(self.session.authorize_repetitions([10, 11]).passed)
        with self.assertRaises(pilot.GuardStopped):
            self.session.authorize_repetitions([10, 11])
        self.assertEqual(self.run_child().reason, "completed")

    def test_unstable_pilot_persists_rejection_and_never_cold_loads_again(self):
        self.run_child()
        assessment = self.session.authorize_repetitions([18.52e9, 34.16e9])
        self.assertFalse(assessment.passed)
        record = self.timeline()[-1]
        self.assertEqual(record["event"], "pilot_assessment")
        self.assertFalse(record["assessment"]["passed"])
        with patch.object(pilot.subprocess, "Popen") as spawn:
            with self.assertRaises(pilot.GuardStopped):
                self.run_child()
            with self.assertRaises(pilot.GuardStopped):
                self.session.authorize_repetitions([10, 10])
            spawn.assert_not_called()

    def test_preexisting_stop_file_and_missing_telemetry_do_not_spawn(self):
        stop = self.root / "stop"
        stop.touch()
        for sampler, stop_file in (
            (lambda: HEALTHY, stop),
            (lambda: None, None),
            (Mock(side_effect=OSError("unavailable")), None),
        ):
            self.session = pilot.Session(FAST, sampler=sampler)
            with patch.object(pilot.subprocess, "Popen") as spawn:
                with self.assertRaises(pilot.GuardStopped):
                    self.run_child(stop_file=stop_file)
                spawn.assert_not_called()
            self.assertEqual(self.timeline()[-1]["event"], "blocked")

    def test_nonzero_exit_and_postflight_telemetry_loss_latch_closed(self):
        for code, sampler in (
            ("raise SystemExit(7)", lambda: HEALTHY),
            ("pass", Mock(side_effect=[HEALTHY, None])),
        ):
            self.session = pilot.Session(FAST, sampler=sampler)
            self.assertNotEqual(self.run_child(code).reason, "completed")
            with self.assertRaises(pilot.GuardStopped):
                self.session.authorize_repetitions([10, 10])
            with patch.object(pilot.subprocess, "Popen") as spawn:
                with self.assertRaises(pilot.GuardStopped):
                    self.run_child()
                spawn.assert_not_called()

    def test_failed_spawn_is_persisted_and_not_retried(self):
        with patch.object(
            pilot.subprocess, "Popen", side_effect=OSError("toy failure")
        ):
            with self.assertRaises(OSError):
                self.run_child()
        self.assertEqual(self.timeline()[-1]["event"], "spawn_failed")
        with patch.object(pilot.subprocess, "Popen") as spawn:
            with self.assertRaises(pilot.GuardStopped):
                self.run_child()
            spawn.assert_not_called()

    @unittest.skipUnless(sys.platform != "win32", "POSIX signals")
    def test_stop_file_sends_int_then_term_to_only_owned_child(self):
        stop = self.root / "stop"
        code = (
            "import signal, time, pathlib\n"
            "signal.signal(signal.SIGINT, signal.SIG_IGN)\n"
            "signal.signal(signal.SIGTERM, lambda *_: exit(0))\n"
            f"pathlib.Path({str(stop)!r}).touch()\n"
            "time.sleep(2)\n"
        )
        result = self.run_child(code, stop_file=stop)
        self.assertEqual(result.reason, "stop_file")
        self.assertEqual(result.returncode, 0)
        signals = [row for row in self.timeline() if row["event"] == "signal"]
        self.assertEqual([row["signal"] for row in signals], ["SIGINT", "SIGTERM"])
        self.assertEqual({row["pid"] for row in signals}, {result.pid})
        with self.assertRaises(pilot.GuardStopped):
            self.run_child()

    @unittest.skipUnless(sys.platform != "win32", "POSIX signals")
    def test_timeout_interrupts_disposable_child_without_retry(self):
        result = self.run_child("import time; time.sleep(2)", timeout_s=0.2)
        self.assertEqual(result.reason, "timeout")
        self.assertIsNotNone(result.returncode)
        self.assertEqual(
            [row["signal"] for row in self.timeline() if row["event"] == "signal"],
            ["SIGINT"],
        )

    @unittest.skipUnless(sys.platform != "win32", "POSIX signals")
    def test_midflight_unhealthy_or_missing_telemetry_stops_owned_child(self):
        low = replace(HEALTHY, free_bytes=1, available_bytes=1)
        for failure in (low, None, OSError("telemetry disappeared")):
            with self.subTest(failure=failure):
                sampler = Mock(side_effect=[HEALTHY, HEALTHY, failure, HEALTHY])
                policy = replace(FAST, health_interval_s=0.05)
                self.session = pilot.Session(policy, sampler=sampler)
                started = time.monotonic()
                result = self.run_child("import time; time.sleep(5)", timeout_s=3)
                elapsed = time.monotonic() - started
                self.assertEqual(result.reason, "unhealthy")
                self.assertEqual(result.pid, self.session.child.pid)
                self.assertIsNotNone(self.session.child.poll())
                self.assertEqual(result.returncode, self.session.child.returncode)
                self.assertLess(elapsed, 2)
                rows = self.timeline()
                health = [row for row in rows if row["event"] == "health"]
                self.assertEqual(len(health), 2)
                self.assertEqual(health[0]["snapshot"], asdict(HEALTHY))
                self.assertFalse(health[0]["health_reasons"])
                self.assertTrue(health[1]["health_reasons"])
                self.assertTrue(rows[-1]["health_reasons"])
                signals = [row for row in rows if row["event"] == "signal"]
                self.assertEqual({row["pid"] for row in signals}, {result.pid})
                self.assertTrue(
                    all(row["signal"] in ("SIGINT", "SIGTERM") for row in signals)
                )
                cancellation_s = (
                    rows[-1]["monotonic_ns"] - health[1]["monotonic_ns"]
                ) / 1e9
                self.assertLess(
                    cancellation_s,
                    policy.interrupt_grace_s + policy.terminate_grace_s + 0.5,
                )
                with self.assertRaises(pilot.GuardStopped):
                    self.session.authorize_repetitions([10, 10])
                with patch.object(pilot.subprocess, "Popen") as spawn:
                    with self.assertRaises(pilot.GuardStopped):
                        self.run_child()
                    spawn.assert_not_called()

    def test_healthy_sampling_is_interval_limited_and_not_fsynced(self):
        sampler = Mock(return_value=HEALTHY)
        self.session = pilot.Session(
            replace(FAST, health_interval_s=0.05), sampler=sampler
        )
        with (
            patch.object(self.session, "_event", wraps=self.session._event) as emit,
            patch.object(pilot.os, "fsync", wraps=pilot.os.fsync) as fsync,
        ):
            result = self.run_child("import time; time.sleep(0.18)")
        self.assertEqual(result.reason, "completed")
        health = [row for row in self.timeline() if row["event"] == "health"]
        self.assertGreaterEqual(len(health), 2)
        self.assertEqual(sampler.call_count, len(health) + 2)
        self.assertEqual(fsync.call_count, len(self.timeline()) - len(health))
        for earlier, later in zip(health, health[1:]):
            self.assertGreaterEqual(
                (later["monotonic_ns"] - earlier["monotonic_ns"]) / 1e9, 0.05
            )
        health_calls = [
            call for call in emit.call_args_list if call.args[0] == "health"
        ]
        self.assertTrue(all(call.kwargs["durable"] is False for call in health_calls))

    def test_unresponsive_handle_returns_bounded_and_never_sigkills(self):
        child = Mock(pid=123, returncode=None)
        child.poll.return_value = None
        child.wait.side_effect = subprocess.TimeoutExpired("toy", 0)
        with patch.object(pilot.subprocess, "Popen", return_value=child):
            result = self.run_child(timeout_s=0.01)
        self.assertEqual(result.reason, "timeout")
        self.assertIsNone(result.returncode)
        self.assertIs(self.session.child, child)
        self.assertEqual(
            [call.args[0] for call in child.send_signal.call_args_list],
            [signal.SIGINT, signal.SIGTERM],
        )
        self.assertEqual(
            [call.kwargs["timeout"] for call in child.wait.call_args_list],
            [FAST.interrupt_grace_s, FAST.terminate_grace_s],
        )
        child.kill.assert_not_called()
        self.assertIn("child_still_running", [row["event"] for row in self.timeline()])
        with self.assertRaises(pilot.GuardStopped):
            self.run_child()

    def test_exited_handle_is_never_signaled(self):
        child = Mock(pid=123, returncode=0)
        child.poll.return_value = 0
        self.session.child = child
        self.session._cancel()
        child.send_signal.assert_not_called()
        child.wait.assert_not_called()

    def test_exit_between_poll_and_signal_uses_owned_handle_only(self):
        child = Mock(pid=123, returncode=0)
        child.poll.side_effect = [None, 0]
        child.send_signal.side_effect = ProcessLookupError
        child.wait.return_value = 0
        self.session.child = child
        with patch.object(self.session, "_event"):
            self.session._cancel()
        child.send_signal.assert_called_once_with(signal.SIGINT)
        child.kill.assert_not_called()

    def test_log_failure_does_not_skip_terminate_grace(self):
        child = Mock(pid=123, returncode=None)
        child.poll.return_value = None
        child.wait.side_effect = subprocess.TimeoutExpired("toy", 0)
        self.session.child = child
        with patch.object(self.session, "_event", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                self.session._cancel()
        self.assertEqual(child.send_signal.call_count, 2)
        self.assertEqual(child.wait.call_count, 2)


if __name__ == "__main__":
    unittest.main()
