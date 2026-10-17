import pathlib
import subprocess
import sys
import unittest


SCRIPT = pathlib.Path(__file__).with_name("deadline.py")


class DeadlineTests(unittest.TestCase):
    def invoke(self, *args):
        return subprocess.run(
            [sys.executable, "-B", str(SCRIPT), *args],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )

    def test_success(self):
        result = self.invoke(sys.executable, "-c", "print('completed')")
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "completed\n")

    def test_failure_status_is_preserved(self):
        result = self.invoke(sys.executable, "-c", "raise SystemExit(7)")
        self.assertEqual(result.returncode, 7)

    def test_timeout(self):
        result = self.invoke(
            "--seconds", "0.1", sys.executable, "-c", "import time; time.sleep(30)"
        )
        self.assertEqual(result.returncode, 124)
        self.assertIn("exceeded", result.stderr)

    def test_missing_command(self):
        self.assertEqual(self.invoke().returncode, 2)

    def test_nonpositive_timeout(self):
        self.assertEqual(self.invoke("--seconds", "0", "unused").returncode, 2)


if __name__ == "__main__":
    unittest.main()
