"""Run the required offline source-contract suite, rejecting empty selection."""

import unittest


def require_tests(suite):
    if suite.countTestCases() == 0:
        raise ValueError("V4.1 source-contract suite selected no tests")
    return suite


if __name__ == "__main__":
    suite = unittest.TestSuite(
        require_tests(unittest.defaultTestLoader.loadTestsFromName(name))
        for name in ("test_sources", "test_tensor_headers", "test_engram_oracle")
    )
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)
