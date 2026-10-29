# /// script
# requires-python = ">=3.11"
# ///
"""CPU-only controls for the aggregate contract runner's lease boundary."""

import os
from pathlib import Path
import re
import subprocess
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "check-contracts.sh"
REFUSAL = "CPU-only contracts refuse an inherited GPU lease"


class UnleasedContracts(unittest.TestCase):
    def invoke(self, root, arguments, marker=None, wait=False):
        tools = root / "bin"
        tools.mkdir(exist_ok=True)
        reached = root / "git-reached"
        stub = tools / "git"
        stub.write_text(
            '#!/bin/sh\nprintf "git reached\\n" > "$CONTROL_REACHED"\nexit 97\n'
        )
        stub.chmod(0o700)
        env = {
            k: v
            for k, v in os.environ.items()
            if k not in ("QWEN_METAL_LEASE_FD", "QWEN_METAL_LEASE_WAIT")
        }
        env.update(PATH=str(tools), CONTROL_REACHED=str(reached))
        if marker is not None:
            env["QWEN_METAL_LEASE_FD"] = marker
        if wait:
            env["QWEN_METAL_LEASE_WAIT"] = "1"
        result = subprocess.run(
            ["/bin/bash", str(SCRIPT), *arguments],
            cwd=root,
            env=env,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result, reached

    def test_lease_marker_refuses_before_git_or_any_build(self):
        for marker in ("", "-1", "not-a-descriptor"):
            for arguments in ([], ["--full"], ["--invalid"]):
                with self.subTest(marker=marker, arguments=arguments):
                    with tempfile.TemporaryDirectory() as directory:
                        result, reached = self.invoke(
                            Path(directory), arguments, marker
                        )
                        self.assertEqual(result.returncode, 2)
                        self.assertIn(REFUSAL, result.stderr)
                        self.assertFalse(reached.exists())

    def test_wait_marker_alone_does_not_mean_a_held_lease(self):
        for wait in (False, True):
            with self.subTest(wait=wait):
                with tempfile.TemporaryDirectory() as directory:
                    result, reached = self.invoke(Path(directory), [], wait=wait)
                    self.assertNotIn(REFUSAL, result.stderr)
                    self.assertEqual(reached.read_text(), "git reached\n")
                    self.assertNotEqual(result.returncode, 0)

    def test_all_cargo_execution_is_unleased_and_offline_locked(self):
        source = SCRIPT.read_text()
        self.assertNotIn("cpu_leased", source)
        self.assertNotIn("with-metal-lease", source)
        execution = re.findall(
            r"(?m)^\s*(?:output=\$\()?cargo (?:test|build)\b[^\n]*", source
        )
        self.assertEqual(len(execution), 4)
        for line in execution:
            self.assertIn("--offline", line)
            self.assertIn("--locked", line)


if __name__ == "__main__":
    unittest.main()
