"""CPU only; parent runs: uv run --no-project python -B -m unittest discover -s scripts/bench -p test_with_metal_lease.py"""

import fcntl
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import textwrap
import unittest

WRAPPER = Path(__file__).resolve().parent / "with-metal-lease"

# A campaign driven from Python. Each arm re-requests LOCK_EX on the inherited
# descriptor (a no-op on the holding description, as qwen's adoption does).
# Between arms an outsider with its own open file description must not get the
# lease. A wrapper nested in the campaign (a driver wrapping llama-bench) joins
# it; a launcher that drops the descriptor gets a refusal, not a deadlock.
CAMPAIGN = textwrap.dedent(
    """
    import fcntl, os, subprocess, sys
    wrapper, lease_dir, lock = sys.argv[1:4]
    fds = (int(os.environ["QWEN_METAL_LEASE_FD"]),)
    arm = '''
    import fcntl, os, sys
    fd = int(os.environ["QWEN_METAL_LEASE_FD"])
    opened, linked = os.fstat(fd), os.stat(sys.argv[1])
    assert (opened.st_dev, opened.st_ino) == (linked.st_dev, linked.st_ino)
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    print("arm", sys.argv[2], "adopted", flush=True)
    '''
    subprocess.run([sys.executable, "-c", arm, lock, "A"], check=True, pass_fds=fds)
    outsider = os.open(lock, os.O_RDWR)
    try:
        fcntl.flock(outsider, fcntl.LOCK_EX | fcntl.LOCK_NB)
        print("outsider got the lease between arms", flush=True)
    except BlockingIOError:
        print("outsider blocked between arms", flush=True)
    nested = [sys.executable, wrapper, "--lease-dir", lease_dir, "--task", "inner", "--"]
    subprocess.run(nested + [sys.executable, "-c", arm, lock, "nested"], check=True, pass_fds=fds)
    dropped = subprocess.run(nested + ["true"], capture_output=True, text=True, timeout=20)
    print("dropped descriptor exit", dropped.returncode, "pass_fds" in dropped.stderr, flush=True)
    subprocess.run([sys.executable, "-c", arm, lock, "B"], check=True, pass_fds=fds)
    print("record", open(lock).read().strip(), flush=True)
    """
)


class WithMetalLeaseTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.lease_dir = os.path.join(self.directory.name, "lease")
        self.lock = os.path.join(self.lease_dir, "metal.lock")

    def tearDown(self) -> None:
        self.directory.cleanup()

    def wrapper(self, *args: str, wait: bool = False) -> subprocess.CompletedProcess:
        env = dict(os.environ)
        env.pop("QWEN_METAL_LEASE_FD", None)
        env.pop("OPENCODE_CALLER_SESSION_ID", None)
        env["QWEN_METAL_LEASE_WAIT"] = "1" if wait else "0"
        return subprocess.run(
            [sys.executable, str(WRAPPER), "--lease-dir", self.lease_dir, *args],
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )

    def test_campaign_holds_the_lease_across_arms(self) -> None:
        result = self.wrapper(
            "--task",
            "V41-TEST",
            "--for",
            "40m",
            "--session",
            "ses_remote",
            "--",
            sys.executable,
            "-c",
            CAMPAIGN,
            str(WRAPPER),
            self.lease_dir,
            self.lock,
            wait=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        lines = result.stdout.splitlines()
        self.assertEqual(
            lines[:5],
            [
                "arm A adopted",
                "outsider blocked between arms",
                "arm nested adopted",
                "dropped descriptor exit 1 True",
                "arm B adopted",
            ],
        )
        record = lines[5]
        self.assertRegex(record, r"^record pid=\d+ executable=python[\w.]* ")
        self.assertIn("opencode_session=ses_remote", record)
        self.assertRegex(record, r"started=\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ")
        self.assertIn("task=V41-TEST", record)
        self.assertIn("expected=40m", record)

        fd = os.open(self.lock, os.O_RDWR)
        try:
            fcntl.flock(
                fd, fcntl.LOCK_EX | fcntl.LOCK_NB
            )  # free once the campaign exits
        finally:
            os.close(fd)

    def test_busy_lease_refuses_with_record_and_hold_time(self) -> None:
        os.makedirs(self.lease_dir, mode=0o700)
        holder = os.open(self.lock, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(holder, fcntl.LOCK_EX)
            os.write(holder, b"pid=1 executable=long-run opencode_session=none\n")
            result = self.wrapper("--", sys.executable, "-c", "pass")
        finally:
            os.close(holder)
        self.assertEqual(result.returncode, 75, result.stderr)
        self.assertIn("executable=long-run", result.stderr)
        self.assertTrue(re.search(r"; held \d+s\)", result.stderr), result.stderr)

    def test_session_falls_back_to_none_without_forwarding(self) -> None:
        result = self.wrapper(
            "--",
            sys.executable,
            "-c",
            "import sys; print(open(sys.argv[1]).read())",
            self.lock,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("opencode_session=none", result.stdout)
        self.assertNotIn("task=", result.stdout)

    def test_option_without_value_is_a_usage_error(self) -> None:
        result = self.wrapper("--task")
        self.assertEqual(result.returncode, 2)


if __name__ == "__main__":
    unittest.main()
