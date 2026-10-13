"""Small CPU fixtures only.

uv run --with 'numpy>=2,<3' python -B -m unittest discover \
    -s scripts/reference/glm53 -p test_replay_precision.py
"""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import replay_precision as study
import numpy as np


class ReplayPrecisionTests(unittest.TestCase):
    def test_kda_orientation_and_true_f64_recurrence(self):
        stream = study.Stream(study.PRESETS[1], heads=2, dim=4)
        state = stream.initial.astype(np.float64)
        rounded = state.copy()
        for _ in range(33):
            raw, q = stream.next()
            r = raw.cast(np.float64)
            expected = np.empty_like(state)
            for h in range(2):
                for v in range(4):
                    decayed = [state[h, v, k] * float(raw.d[h, k]) for k in range(4)]
                    correction = (float(raw.v[h, v]) - sum(
                        decayed[k] * float(raw.k[h, k]) for k in range(4))) * float(raw.beta[h])
                    for k in range(4):
                        expected[h, v, k] = decayed[k] + correction * float(raw.k[h, k])
            state = study.update(state, r)
            np.testing.assert_allclose(state, expected, rtol=1e-14, atol=1e-16)
            self.assertEqual(state.dtype, np.float64)
            rounded = study.update(rounded, r).astype(np.float32).astype(np.float64)
        self.assertFalse(np.array_equal(state, rounded))
        # Verify the runner actually keeps that reference, not just update().
        rows = []
        study.run_case(study.PRESETS[1], 33, rows.append, heads=2, dim=4)
        self.assertEqual(rows[-1]["reference_state_dtype"], "float64")
        self.assertEqual(rows[-1]["reference_state_f64_sha256"],
                         hashlib.sha256(state.astype("<f8").tobytes()).hexdigest())

    def test_lp_outputs_cannot_feed_authoritative_anchor(self):
        stream = study.Stream(study.PRESETS[0], heads=2, dim=16)
        low = study.Replay(stream.initial, 8, np.float32, np.float16)
        high = study.Replay(stream.initial, 8, np.float32, np.float32)
        ordinary = study.Ordinary(stream.initial, np.float32)
        different_reads = 0
        for t in range(1, 33):
            raw, q = stream.next()
            lo, flush, _ = low.step(raw, q)
            hi, high_flush, _ = high.step(raw, q)
            expected, _, _ = ordinary.step(raw, q)
            self.assertEqual(flush, high_flush)
            if flush:
                self.assertEqual(low.anchor.tobytes(), high.anchor.tobytes())
                self.assertEqual(low.anchor.tobytes(), ordinary.anchor.tobytes())
                np.testing.assert_array_equal(lo, expected)
                np.testing.assert_array_equal(hi, expected)
                np.testing.assert_array_equal(low.read, low.anchor.astype(np.float16))
                self.assertFalse(low.raw or low.corrections)
            else:
                different_reads += not np.array_equal(lo, hi)
        self.assertGreater(different_reads, 0)

    def test_bad_cached_lp_corrections_contaminate_flush(self):
        stream = study.Stream(study.PRESETS[0], heads=2, dim=16)
        good = study.Replay(stream.initial, 8, np.float32, np.float16)
        for _ in range(7):
            raw, q = stream.next()
            good.step(raw, q)
        wrong = good.anchor.copy()
        clean = good.anchor.copy()
        for record, lp_u in zip(good.raw, good.corrections):
            wrong = wrong * record.d[:, None, :] + lp_u[:, :, None] * record.k[:, None, :]
            clean = study.update(clean, record)
        raw, q = stream.next()
        wrong, clean = study.update(wrong, raw), study.update(clean, raw)
        _, flush, _ = good.step(raw, q)
        self.assertTrue(flush)
        np.testing.assert_array_equal(good.anchor, clean)
        self.assertGreater(float(np.max(np.abs(wrong - good.anchor))), 0)

    def test_L1_controls_and_shared_f16_storage(self):
        stream = study.Stream(study.PRESETS[0], heads=2, dim=8)
        candidates = study.arms(stream.initial)
        for _ in range(17):
            raw, q = stream.next()
            outputs = {n: a.step(raw, q)[0] for n, a in candidates.items()}
            for replay, ordinary in (("shared_f16", "ordinary_f16_feedback"),
                                     ("separate_f16_read_f32_anchor", "ordinary_f32"),
                                     ("f32", "ordinary_f32")):
                name = f"replay_{replay}_L1"
                np.testing.assert_array_equal(outputs[name], outputs[ordinary])
                np.testing.assert_array_equal(candidates[name].anchor, candidates[ordinary].anchor)
        shared = candidates["replay_shared_f16_L8"]
        self.assertEqual(shared.anchor.dtype, np.float16)
        self.assertIs(shared.read, shared.anchor)

    def test_output_only_per_key_products_and_extremes(self):
        # L8 nonflush output matches explicit working-state recurrence within
        # F32 regrouping error, even when products underflow; no inverse needed.
        stream = study.Stream(study.PRESETS[0], heads=2, dim=4)
        replay = study.Replay(stream.initial, 8, np.float32, np.float32)
        state = stream.initial.copy()
        for _ in range(16):
            raw, q = stream.next()
            d = np.array([[np.finfo(np.float32).tiny, 0.5, 1, 1-2**-24],
                          [1, np.finfo(np.float32).tiny, 0.01, 1]], dtype=np.float32)
            raw = study.Raw(raw.k, d, raw.v, raw.beta)
            state = study.update(state, raw)
            actual, _, finite = replay.step(raw, q)
            self.assertTrue(finite)
            np.testing.assert_allclose(actual, study.mv(state, q), rtol=3e-6, atol=2e-8)

    def test_checkpoints_include_powers_and_nonflush_neighbors(self):
        points = study.checkpoints(65536)
        self.assertEqual(points, sorted(set(points)))
        self.assertEqual(points[-2:], [65535, 65536])
        for power in (2**i for i in range(17)):
            self.assertIn(power, points)
            for neighbor in (power - 1, power + 1):
                if 1 <= neighbor <= 65536:
                    self.assertIn(neighbor, points)
                    if power >= 8:
                        self.assertNotEqual(neighbor % 8, 0)
        for steps in (1, 2, 3, 7, 8, 9, 33):
            selected = study.checkpoints(steps)
            self.assertEqual(selected[-1], steps)
            self.assertTrue(all(1 <= step <= steps for step in selected))

    def test_streams_metrics_bytes_and_partial_last_window(self):
        for preset in study.PRESETS:
            a, b = study.Stream(preset, dim=8), study.Stream(preset, dim=8)
            self.assertFalse(np.array_equal(a.initial, a.initial.transpose(0, 2, 1)))
            for _ in range(3):
                ra, qa = a.next()
                rb, qb = b.next()
                for x, y in zip((ra.k, ra.d, ra.v, ra.beta, qa), (rb.k, rb.d, rb.v, rb.beta, qb)):
                    self.assertEqual(x.dtype, np.float32)
                    np.testing.assert_array_equal(x, y)
                self.assertTrue(np.all((ra.d > 0) & (ra.d <= 1)))
            rows = []
            study.run_case(preset, 9, rows.append, dim=8)
            json.dumps(rows, allow_nan=False)
            summary = rows[-1]["arms"]["replay_separate_f16_read_f32_anchor_L8"]
            self.assertEqual(summary["last_flush_step"], 8)
            self.assertEqual(summary["aggregates"]["output_all"]["samples"], 9)
            self.assertEqual(summary["aggregates"]["flush_state"]["samples"], 1)
            self.assertIsNone(rows[-2]["arms"]["replay_f32_L8"]["flush_state"])
            observed = {row["step"]: row["arms"] for row in rows if row["event"] == "checkpoint"}
            for step in (7, 8, 9):
                for name, observation in observed[step].items():
                    flush = step % 8 == 0 if name.endswith("_L8") else True
                    self.assertEqual(observation["flush"], flush)
                    self.assertEqual(observation["phase"], "flush" if flush else "nonflush")
                    self.assertEqual(observation["flush_state"] is not None, flush)
        formats = study.Replay(np.zeros((2, 128, 128), np.float32), 8, np.float32, np.float16).formats()
        self.assertEqual(formats["anchor_bytes"], 2*128*128*4)
        self.assertEqual(formats["extra_read_bytes"], 2*128*128*2)
        self.assertEqual(formats["raw_cache_capacity_bytes"], 8*2*(3*128+1)*4)
        self.assertEqual(formats["correction_cache_capacity_bytes"], 7*2*128*4)
        stats = study.Errors()
        stats.add(np.array([np.nan, np.inf, 2]), np.array([0., 0., 1.]))
        self.assertEqual(stats.report()["nonfinite_candidate"], 2)
        self.assertEqual(stats.report()["rmse_finite_pairs"], 1)

    def test_cli_exclusive_output_and_complete_small_packet(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "small.jsonl"
            cmd = [sys.executable, "-B", study.__file__, "--steps", "8", "--out", str(out)]
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            saved = out.read_bytes()
            rows = [json.loads(line) for line in saved.splitlines()]
            self.assertEqual(rows[0]["event"], "header")
            self.assertEqual(rows[-1]["event"], "complete")
            self.assertEqual(sum(row["event"] == "case_complete" for row in rows), 3)
            result = subprocess.run(cmd, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(out.read_bytes(), saved)


if __name__ == "__main__":
    unittest.main()
