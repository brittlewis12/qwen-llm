"""Offline stdlib-only tests; optional NumPy is never imported here."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import engram_oracle as oracle


class EngramOracleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = oracle.verify()
        cls.params = cls.data["parameters"]

    def test_regeneration_preserves_pinned_values(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.json"
            oracle.write_fixture(path, self.data)
            oracle.write_fixture(path, self.data)
            changed = dict(self.data, synthetic_pad_compressed_id=18)
            with self.assertRaisesRegex(ValueError, "differs from the pinned"):
                oracle.write_fixture(path, changed)
            self.assertEqual(json.loads(path.read_text()), self.data)

    def test_empty_capture_does_not_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.json"
            path.write_text(json.dumps(dict(self.data, cases=[])))
            with patch.object(oracle, "FIXTURE", path):
                with self.assertRaisesRegex(ValueError, "four cases and 21 positions"):
                    oracle.verify()

    def test_prime_substitute_against_independent_sieve(self):
        # Sieve is independent of the generator's trial division, including
        # the complete interval containing every table modulus.
        limit = max(
            p for layer in self.params["primes"] for group in layer for p in group
        )
        sieve = bytearray(b"\x01") * (limit + 1)
        sieve[:2] = b"\x00\x00"
        for n in range(2, limit + 1):
            if n * n > limit:
                break
            if sieve[n]:
                sieve[n * n :: n] = b"\x00" * ((limit - n * n) // n + 1)
        for n in list(range(1000)) + list(range(16000000, limit + 1)):
            self.assertEqual(oracle.isprime(n), bool(sieve[n]), n)
        actual = [
            p for layer in self.params["primes"] for group in layer for p in group
        ]
        self.assertEqual(actual, [n for n in range(16000000, limit + 1) if sieve[n]])

    def test_rng_capture_and_integer_bounds(self):
        self.assertEqual(
            self.params["multipliers"],
            [
                [76632096046245, 4839876093313, 35959672319349, 73987337458391],
                [67716810739261, 51510806800915, 30921347202721, 82619226485591],
            ],
        )
        for row in self.params["multipliers"]:
            for m in row:
                self.assertEqual(m % 2, 1)
                self.assertLess(99091 * m, 2**63)

    def test_all_48_addresses_and_offsets(self):
        for layer, groups in enumerate(self.params["primes"]):
            offset = 0
            flat = [p for group in groups for p in group]
            for column, prime in enumerate(flat):
                self.assertEqual(self.params["offsets"][layer][column], offset)
                for case in self.data["cases"]:
                    for rows in case["expected_rows"]:
                        self.assertEqual(len(rows), 2)
                        self.assertEqual(len(rows[layer]), 24)
                        self.assertTrue(offset <= rows[layer][column] < offset + prime)
                offset += prime
            self.assertEqual(offset, self.params["table_totals"][layer])

    def test_prefill_incremental_chunks_and_restoration(self):
        for case in self.data["cases"]:
            ids = case["compressed_ids"]
            for chunks in [case["chunks"], [1] * len(ids), [len(ids)]]:
                history, rows, consumed = case["history"], [], 0
                for size in chunks:
                    result, history = oracle.reference(
                        ids[consumed : consumed + size], self.params, history
                    )
                    rows.extend(result)
                    consumed += size
                    self.assertLessEqual(len(history), 3)
                self.assertEqual(consumed, len(ids))
                self.assertEqual(rows, case["expected_rows"])
                self.assertEqual(history, case["final_history"])
            # A full prefix and its restored last-three history must agree.
            prefix = [45, 67, 89] + case["history"]
            full, _ = oracle.reference(prefix + ids, self.params)
            restored, _ = oracle.reference(ids, self.params, prefix[-3:])
            self.assertEqual(full[len(prefix) :], restored)

    def test_dead_stops_lookback_including_current(self):
        dead, _ = oracle.reference([-1], self.params, [99091, 0, 42])
        padded, _ = oracle.reference([oracle.PAD], self.params, [oracle.PAD] * 3)
        self.assertEqual(dead, padded)
        after, _ = oracle.reference([7, 8, 9], self.params, [99091, 42, -1])
        fresh, _ = oracle.reference([7, 8, 9], self.params)
        self.assertEqual(after, fresh)
        self.assertNotEqual(oracle.reference([0], self.params)[0], dead)

    def test_invalid_ids_and_history(self):
        for invalid in [-2, 99092, 129264, 2**63, True, 1.0, "1", None]:
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    oracle.reference([invalid], self.params)
                with self.assertRaises(ValueError):
                    oracle.reference([0], self.params, [invalid])
        with self.assertRaises(ValueError):
            oracle.reference([0], self.params, [0] * 4)
        with self.assertRaises(ValueError):
            oracle.reference([0], self.params, pad_id=-1)
        self.assertEqual(oracle.reference([], self.params, [1, -1]), ([], [1, -1]))


if __name__ == "__main__":
    unittest.main()
