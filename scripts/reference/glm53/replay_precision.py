# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2,<3"]
# ///
"""CPU-only KDA precision study; fixed synthetic streams, no model-quality gate.

uv run scripts/reference/glm53/replay_precision.py --out /tmp/replay-precision.jsonl

ReplaySSM https://arxiv.org/html/2610.13778v1, Section 6 / Appendix B.4:
flush from raw inputs, never from corrections computed using an LP read copy.
Adapted here from scalar-decay GDN to KDA's per-key decay (right multiplication).
S is [head,value,key]: A=S*d; u=beta*(v-A@k); S'=A+outer(u,k); o=S'@q.
Inputs are already prepared F32 q/k/v/decay/beta, not raw convolution inputs.
Outputs exclude RMS norm, output gating and the surrounding model.

All candidate arithmetic accumulates in F32. NumPy separates multiply/add and
uses its own einsum reduction order, NOT Metal's FMA/dot/simd_sum order. The F64
reference retains F64 state at EVERY step, starting from the same F32 inputs.
Ordinary F16 feedback stores after updating; output uses the unrounded working
state, just as decode reads its updated registers. Replay flush outputs likewise
use the recomputed working state before independently casting the two anchors.
No latency, DRAM-traffic, KL, universal precision-win or noninferiority claims.
"""

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess

# No BLAS calls in the recurrence; also bound any library initialization threads.
THREAD_KEYS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
INHERITED_THREADS = {key: os.environ.get(key) for key in THREAD_KEYS}
for _key in THREAD_KEYS:
    os.environ[_key] = "1"

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
PRESETS = (
    {
        "name": "normal_correlated",
        "seed": 1103,
        "log_decay_range": [-0.08, -0.002],
        "beta_range": [0.1, 0.6],
        "value_scale": 0.25,
    },
    {
        "name": "slow_small_updates",
        "seed": 1109,
        "log_decay_range": [-2e-5, -1e-7],
        "beta_range": [1e-4, 2e-3],
        "value_scale": 0.01,
    },
    {
        "name": "extreme_channels",
        "seed": 1117,
        "log_decay_range": None,
        "beta_range": [0.01, 0.9],
        "value_scale": 0.1,
    },
)


@dataclass(frozen=True)
class Raw:
    k: np.ndarray
    d: np.ndarray
    v: np.ndarray
    beta: np.ndarray

    def cast(self, dtype):
        return Raw(
            *(x.astype(dtype, copy=False) for x in (self.k, self.d, self.v, self.beta))
        )


def mv(state, vector):
    return np.einsum("hvk,hk->hv", state, vector, optimize=False)


def inner(a, b):
    return np.einsum("hk,hk->h", a, b, optimize=False)


def update(state, raw):
    """No storage cast: all operands must already have the accumulation dtype."""
    decayed = state * raw.d[:, None, :]
    u = (raw.v - mv(decayed, raw.k)) * raw.beta[:, None]
    return decayed + u[:, :, None] * raw.k[:, None, :]


class Ordinary:
    def __init__(self, initial, storage):
        self.storage = np.dtype(storage)
        self.anchor = initial.astype(self.storage)

    def step(self, raw, q):
        work = update(self.anchor.astype(np.float32), raw)
        output = mv(work, q)
        self.anchor = work.astype(self.storage)
        return output, True, bool(np.isfinite(work).all())

    def formats(self):
        return {
            "anchor": self.storage.name,
            "read": "same as anchor",
            "window": 1,
            "anchor_bytes": self.anchor.nbytes,
            "extra_read_bytes": 0,
            "raw_cache_capacity_bytes": 0,
            "correction_cache_capacity_bytes": 0,
            "total_persistent_capacity_bytes": self.anchor.nbytes,
        }


class Replay:
    def __init__(self, initial, window, anchor_dtype, read_dtype):
        if window not in (1, 8):
            raise ValueError("this study has only L1 and L8")
        self.window = window
        self.anchor_dtype, self.read_dtype = (
            np.dtype(anchor_dtype),
            np.dtype(read_dtype),
        )
        self.raw, self.corrections = [], []
        self.publish(initial.astype(np.float32))

    def publish(self, work):
        # Both casts originate in the recomputed accumulator, independently.
        self.anchor = work.astype(self.anchor_dtype)
        self.read = (
            self.anchor
            if self.anchor_dtype == self.read_dtype
            else work.astype(self.read_dtype)
        )

    def step(self, raw, q):
        self.raw.append(raw)  # Cache pre-correction v, k, d, beta, all F32.
        if len(self.raw) == self.window:
            work = self.anchor.astype(np.float32)
            finite = True
            for record in self.raw:
                work = update(work, record)  # Deliberately ignores LP corrections.
                finite &= bool(np.isfinite(work).all())
            output = mv(work, q)
            self.publish(work)
            self.raw.clear()
            self.corrections.clear()
            return output, True, finite

        # Direct interval products: never divide cumulative decays. This remains
        # defined when a channel/product underflows to zero or has decay one.
        product = raw.d.copy()
        weighted_keys = []
        for record in reversed(self.raw[:-1]):
            weighted_keys.append(record.k * product)
            product = product * record.d
        read = self.read.astype(np.float32, copy=False)
        prediction = mv(read, product * raw.k)
        output = mv(read, product * q)
        for key, u in zip(reversed(weighted_keys), self.corrections):
            prediction = prediction + u * inner(key, raw.k)[:, None]
            output = output + u * inner(key, q)[:, None]
        u = (raw.v - prediction) * raw.beta[:, None]
        output = output + u * inner(raw.k, q)[:, None]
        self.corrections.append(u)
        return output, False, bool(np.isfinite(u).all())

    def formats(self):
        h, v, k = self.anchor.shape
        sizes = {
            "anchor_bytes": self.anchor.nbytes,
            "extra_read_bytes": 0 if self.read is self.anchor else self.read.nbytes,
            "raw_cache_capacity_bytes": self.window * h * (2 * k + v + 1) * 4,
            "correction_cache_capacity_bytes": (self.window - 1) * h * v * 4,
        }
        return {
            "anchor": self.anchor_dtype.name,
            "read": self.read_dtype.name,
            "window": self.window,
            **sizes,
            "total_persistent_capacity_bytes": sum(sizes.values()),
        }


def arms(initial):
    result = {
        "ordinary_f32": Ordinary(initial, np.float32),
        "ordinary_f16_feedback": Ordinary(initial, np.float16),
    }
    for window in (1, 8):
        for name, anchor, read in (
            ("shared_f16", np.float16, np.float16),
            ("separate_f16_read_f32_anchor", np.float32, np.float16),
            ("f32", np.float32, np.float32),
        ):
            result[f"replay_{name}_L{window}"] = Replay(initial, window, anchor, read)
    return result


def unit(x):
    return x / np.sqrt(np.sum(x * x, axis=-1, keepdims=True) + 1e-6)


class Stream:
    """Fixed PCG64 draw order, independent of arms and requested run length."""

    def __init__(self, preset, heads=2, dim=128):
        self.preset, self.shape = preset, (heads, dim)
        self.rng = np.random.Generator(np.random.PCG64(preset["seed"]))
        self.initial = self.rng.normal(0, 0.125, (heads, dim, dim)).astype(np.float32)
        self.direction = unit(self.rng.normal(size=self.shape))
        self.value_direction = self.rng.normal(size=self.shape)
        self.index = 0

    def next(self):
        rng, p = self.rng, self.preset
        k = unit(0.98 * self.direction + 0.02 * unit(rng.normal(size=self.shape)))
        q = unit(rng.normal(size=self.shape)) / math.sqrt(self.shape[1])
        v = p["value_scale"] * (
            0.5 * self.value_direction + rng.normal(size=self.shape)
        )
        beta = rng.uniform(*p["beta_range"], size=self.shape[0])
        if p["log_decay_range"] is None:
            levels = np.array(
                [
                    math.exp(-5),
                    math.exp(-2),
                    0.5,
                    np.nextafter(np.float32(1), np.float32(0)),
                    1.0,
                ]
            )
            d = levels[
                (np.arange(np.prod(self.shape)).reshape(self.shape) + self.index) % 5
            ]
        else:
            d = np.exp(rng.uniform(*p["log_decay_range"], size=self.shape))
        self.index += 1
        return Raw(*(x.astype(np.float32) for x in (k, d, v, beta))), q.astype(
            np.float32
        )


class Errors:
    """No epsilon floor or benefit threshold; retain nonfinite counts explicitly."""

    def __init__(self):
        self.samples = self.elements = self.pairs = self.bad_candidate = (
            self.bad_reference
        ) = 0
        self.squared = self.reference_squared = self.absolute = self.maximum = 0.0

    def add(self, actual, reference):
        a, b = actual.astype(np.float64), reference.astype(np.float64)
        valid_a, valid_b = np.isfinite(a), np.isfinite(b)
        valid = valid_a & valid_b
        diff = np.abs(a[valid] - b[valid])
        self.samples += 1
        self.elements += a.size
        self.pairs += int(valid.sum())
        self.bad_candidate += int((~valid_a).sum())
        self.bad_reference += int((~valid_b).sum())
        self.squared += float(np.sum(diff * diff))
        self.reference_squared += float(np.sum(b[valid] * b[valid]))
        self.absolute += float(diff.sum())
        self.maximum = max(self.maximum, float(diff.max(initial=0)))

    def report(self):
        relative = (
            math.sqrt(self.squared / self.reference_squared)
            if self.reference_squared
            else (0.0 if self.squared == 0 and self.pairs else None)
        )
        return {
            "samples": self.samples,
            "elements": self.elements,
            "finite_pairs": self.pairs,
            "nonfinite_candidate": self.bad_candidate,
            "nonfinite_reference": self.bad_reference,
            "rmse_finite_pairs": math.sqrt(self.squared / self.pairs)
            if self.pairs
            else None,
            "mae_finite_pairs": self.absolute / self.pairs if self.pairs else None,
            "max_abs_finite_pairs": self.maximum if self.pairs else None,
            "relative_l2_finite_pairs": relative,
        }


def error(actual, reference):
    stats = Errors()
    stats.add(actual, reference)
    return stats.report()


def checkpoints(steps):
    # Powers >= 8 are all L8 flushes. Sample their neighbors too so trajectory
    # observations expose low-precision readout error between flushes.
    return sorted(
        {steps}
        | {
            2**i + offset
            for i in range(steps.bit_length())
            for offset in (-1, 0, 1)
            if 1 <= 2**i + offset <= steps
        }
    )


def run_case(preset, steps, emit, heads=2, dim=128):
    stream = Stream(preset, heads, dim)
    reference = stream.initial.astype(np.float64)  # NEVER rounded back to F32.
    candidates = arms(stream.initial)
    stats = {
        name: {
            key: Errors()
            for key in ("output_all", "output_flush", "output_nonflush", "flush_state")
        }
        for name in candidates
    }
    bad_steps = {name: 0 for name in candidates}
    first_bad = {name: None for name in candidates}
    checkpoints_set = set(checkpoints(steps))
    digest = hashlib.sha256(stream.initial.astype("<f4").tobytes())
    emit(
        {
            "event": "case",
            "preset": preset,
            "heads": heads,
            "dim": dim,
            "initial_f32_sha256": digest.hexdigest(),
            "formats": {n: a.formats() for n, a in candidates.items()},
        }
    )
    for t in range(1, steps + 1):
        raw, q = stream.next()
        for x in (raw.k, raw.d, raw.v, raw.beta, q):
            digest.update(x.astype("<f4", copy=False).tobytes())
        reference = update(reference, raw.cast(np.float64))
        ref_output = mv(reference, q.astype(np.float64))
        at_checkpoint = t in checkpoints_set
        observations = {}
        for name, candidate in candidates.items():
            output, flush, work_finite = candidate.step(raw, q)
            finite = (
                work_finite
                and bool(np.isfinite(output).all())
                and bool(np.isfinite(candidate.anchor).all())
                and (
                    not isinstance(candidate, Replay)
                    or bool(np.isfinite(candidate.read).all())
                )
            )
            if not finite:
                bad_steps[name] += 1
                if first_bad[name] is None:
                    first_bad[name] = t
            s = stats[name]
            s["output_all"].add(output, ref_output)
            s["output_flush" if flush else "output_nonflush"].add(output, ref_output)
            if flush:
                s["flush_state"].add(candidate.anchor, reference)
            if at_checkpoint:
                observations[name] = {
                    "flush": flush,
                    "phase": "flush" if flush else "nonflush",
                    "output": error(output, ref_output),
                    "flush_state": error(candidate.anchor, reference)
                    if flush
                    else None,
                }
        if at_checkpoint:
            emit(
                {
                    "event": "checkpoint",
                    "case": preset["name"],
                    "step": t,
                    "input_prefix_sha256": digest.hexdigest(),
                    "reference_finite": bool(
                        np.isfinite(reference).all() and np.isfinite(ref_output).all()
                    ),
                    "arms": observations,
                }
            )
    emit(
        {
            "event": "case_complete",
            "case": preset["name"],
            "steps": steps,
            "input_sha256": digest.hexdigest(),
            "reference_state_dtype": reference.dtype.name,
            "reference_state_f64_sha256": hashlib.sha256(
                reference.astype("<f8").tobytes()
            ).hexdigest(),
            "arms": {
                name: {
                    "aggregates": {key: value.report() for key, value in s.items()},
                    "nonfinite_steps": bad_steps[name],
                    "first_nonfinite_step": first_bad[name],
                    "last_flush_step": steps
                    // candidates[name].formats()["window"]
                    * candidates[name].formats()["window"],
                }
                for name, s in stats.items()
            },
        }
    )


def provenance():
    paths = [
        Path(__file__).resolve(),
        Path(__file__).with_name("test_replay_precision.py"),
        ROOT / "kernels/glm53_kda.metal",
    ]
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "git_head": subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
        ).strip(),
        "source_sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in paths
        },
        "inherited_thread_env": INHERITED_THREADS,
        "effective_thread_env": {k: os.environ[k] for k in THREAD_KEYS},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, required=True, help="new JSONL file; never overwritten"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=65536,
        help="default 64K; same fixed streams, no sweep",
    )
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    with args.out.open("x") as out:

        def emit(record):
            out.write(json.dumps(record, allow_nan=False) + "\n")
            out.flush()

        emit(
            {
                "event": "header",
                "schema": "glm53.replay_precision.v1",
                "steps": args.steps,
                "checkpoints": checkpoints(args.steps),
                "presets": PRESETS,
                "provenance": provenance(),
                "reference": "persistent F64 state; exactly the same prepared F32 inputs and initial state",
                "rounding": "NumPy F16 casts use round-to-nearest, ties-to-even; no stochastic rounding or FP8/NVFP4 emulation",
                "flush": "every L steps, including current input; recompute raw k/d/v/beta from authoritative anchor; output before casts",
                "cache": "F32 raw k/d/pre-correction v/beta and F32 LP-derived corrections; no cached queries or inverse products",
                "memory": "logical per-arm format bytes, excludes transient accumulators, current query, Python overhead and study reference; not measured RSS or DRAM traffic",
                "metrics": "output aggregates cover every step, also split by flush/nonflush; state aggregates sample ordinary after every store and replay only after flush; checkpoint state is null off-flush; no forced partial-window flush",
                "input_hash_order": "initial state then each step k,d,v,beta,q; contiguous little-endian F32",
                "generator": "PCG64; nonzero N(0,.125) nonsymmetric initial state; normalized .98 fixed direction + .02 normalized noise keys; independent normalized q/sqrt(D); v=scale*(.5 fixed direction + noise)",
                "limits": __doc__,
            }
        )
        try:
            with np.errstate(
                over="ignore", invalid="ignore", under="ignore", divide="ignore"
            ):
                for preset in PRESETS:
                    run_case(preset, args.steps, emit)
            emit(
                {
                    "event": "complete",
                    "cases": len(PRESETS),
                    "steps_per_case": args.steps,
                    "interpretation": "descriptive numerical emulation only; inspect all cases and nonfinite counts, no automatic benefit gate",
                }
            )
        except Exception as exc:
            emit({"event": "error", "type": type(exc).__name__, "message": str(exc)})
            raise
    print(args.out)


if __name__ == "__main__":
    main()
