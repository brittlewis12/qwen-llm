# /// script
# requires-python = ">=3.11"
# dependencies = ["mlx==0.32.4", "numpy"]
# ///
"""External anchor: what does MLX achieve on this GPU for the operation
classes an LLM engine spends time in? Not a qwen-llm measurement.

Prints JSON lines: {probe, dtype, M, K, N, ms_median, tflops | gbps}.
"""

import json
import statistics
import sys
import time

import mlx.core as mx

REPS = 12
WARM = 4


def timed(fn):
    for _ in range(WARM):
        mx.eval(fn())
    xs = []
    for _ in range(REPS):
        t = time.perf_counter()
        mx.eval(fn())
        xs.append(time.perf_counter() - t)
    return statistics.median(xs) * 1e3, statistics.pstdev(xs) * 1e3


def emit(**kw):
    print(json.dumps(kw), flush=True)


info = mx.device_info()
emit(
    probe="device",
    name=info["device_name"],
    arch=info["architecture"],
    mlx=mx.__version__,
)

# 1. Streaming bandwidth: large elementwise copy (read + write).
for gib in (1, 4):
    n = gib * (1 << 30) // 4
    a = mx.random.normal((n,), dtype=mx.float32)
    mx.eval(a)
    ms, sd = timed(lambda: a * 1.0000001)
    emit(
        probe="stream_scale",
        GiB=gib,
        ms=round(ms, 3),
        sd=round(sd, 3),
        gbps=round(2 * n * 4 / (ms / 1e3) / 1e9, 1),
    )
    del a

# 2. Dense GEMM by dtype at LLM prefill shapes: (M rows) x K -> N.
SHAPES = [
    ("expert_up 5120->2304", 5120, 2304),
    ("expert_down 2304->5120", 2304, 5120),
    ("q_b 1280->32768", 1280, 32768),
    ("o_b 8192->5120", 8192, 5120),
    ("square 4096", 4096, 4096),
    ("square 8192", 8192, 8192),
]
for dt_name, dt in (
    ("float32", mx.float32),
    ("float16", mx.float16),
    ("bfloat16", mx.bfloat16),
):
    for label, K, N in SHAPES:
        for M in (256, 1024, 2048):
            x = mx.random.normal((M, K)).astype(dt)
            w = mx.random.normal((K, N)).astype(dt)
            mx.eval(x, w)
            ms, sd = timed(lambda: x @ w)
            emit(
                probe="gemm",
                dtype=dt_name,
                shape=label,
                M=M,
                K=K,
                N=N,
                ms=round(ms, 4),
                sd=round(sd, 4),
                tflops=round(2 * M * K * N / (ms / 1e3) / 1e12, 2),
            )
            del x, w

# 3. Quantized matmul: MXFP4 (E2M1 + UE8M0/32) and affine 4-bit, prefill and decode.
QSHAPES = [
    ("expert_up 5120->2304", 5120, 2304),
    ("o_b 8192->5120", 8192, 5120),
    ("square 8192", 8192, 8192),
]
for mode, kw in (
    ("mxfp4", dict(group_size=32, bits=4, mode="mxfp4")),
    ("affine4_g64", dict(group_size=64, bits=4, mode="affine")),
    ("affine8_g64", dict(group_size=64, bits=8, mode="affine")),
):
    for label, K, N in QSHAPES:
        w = mx.random.normal((N, K)).astype(mx.float16)
        try:
            qw = mx.quantize(w, **kw)
        except Exception as e:  # mode not supported in this build
            emit(probe="qmm", mode=mode, error=str(e)[:200])
            break
        mx.eval(qw)
        wq, scales, *rest = qw
        biases = rest[0] if rest else None
        wbytes = (
            wq.nbytes + scales.nbytes + (biases.nbytes if biases is not None else 0)
        )
        for M in (1, 256, 1024):
            x = mx.random.normal((M, K)).astype(mx.float16)
            mx.eval(x)
            f = lambda: mx.quantized_matmul(x, wq, scales, biases, transpose=True, **kw)
            try:
                ms, sd = timed(f)
            except Exception as e:
                emit(probe="qmm", mode=mode, M=M, error=str(e)[:200])
                continue
            rec = dict(
                probe="qmm",
                mode=mode,
                shape=label,
                M=M,
                K=K,
                N=N,
                ms=round(ms, 4),
                sd=round(sd, 4),
                tflops=round(2 * M * K * N / (ms / 1e3) / 1e12, 2),
            )
            if M == 1:
                rec["weight_gbps"] = round(wbytes / (ms / 1e3) / 1e9, 1)
            emit(**rec)
            del x
        del w, qw

# 4. Decode GEMV, half weights: bandwidth-bound reference.
for label, K, N in (
    ("o_b 8192->5120", 8192, 5120),
    ("square 8192", 8192, 8192),
    ("vocab 5120->129280", 5120, 129280),
):
    x = mx.random.normal((1, K)).astype(mx.float16)
    w = mx.random.normal((K, N)).astype(mx.float16)
    mx.eval(x, w)
    ms, sd = timed(lambda: x @ w)
    emit(
        probe="gemv_f16",
        shape=label,
        K=K,
        N=N,
        ms=round(ms, 4),
        sd=round(sd, 4),
        weight_gbps=round(K * N * 2 / (ms / 1e3) / 1e9, 1),
    )
    del x, w

# 5. Fused attention (MLX sdpa) at prefill: 64 heads, head_dim 128, causal.
for L in (1024, 4096):
    for dt_name, dt in (("float16", mx.float16), ("float32", mx.float32)):
        q = mx.random.normal((1, 32, L, 128)).astype(dt)
        k = mx.random.normal((1, 32, L, 128)).astype(dt)
        v = mx.random.normal((1, 32, L, 128)).astype(dt)
        mx.eval(q, k, v)
        f = lambda: mx.fast.scaled_dot_product_attention(
            q, k, v, scale=128**-0.5, mask="causal"
        )
        try:
            ms, sd = timed(f)
            flops = 4 * 32 * L * L * 128 / 2
            emit(
                probe="sdpa_causal",
                dtype=dt_name,
                L=L,
                heads=32,
                head_dim=128,
                ms=round(ms, 4),
                tflops=round(flops / (ms / 1e3) / 1e12, 2),
            )
        except Exception as e:
            emit(probe="sdpa_causal", dtype=dt_name, L=L, error=str(e)[:200])
        del q, k, v

# 6. Precision of MLX's matmul paths against a float64 reference.
#    float32 goes through NAX as TF32-like when MLX_ENABLE_TF32=1 (default).
import numpy as np  # noqa: E402

rng = np.random.default_rng(7)
for M, K, N in ((256, 5120, 2304), (256, 1280, 4096)):
    a64 = rng.standard_normal((M, K))
    b64 = rng.standard_normal((K, N)) / np.sqrt(K)
    ref = a64 @ b64
    for dt_name, dt in (
        ("float32", mx.float32),
        ("float16", mx.float16),
        ("bfloat16", mx.bfloat16),
    ):
        a = mx.array(a64.astype(np.float32)).astype(dt)
        b = mx.array(b64.astype(np.float32)).astype(dt)
        c = (a @ b).astype(mx.float32)
        mx.eval(c)
        got = np.array(c, dtype=np.float64)
        err = got - ref
        emit(
            probe="precision",
            dtype=dt_name,
            M=M,
            K=K,
            N=N,
            rel_rms=float(np.sqrt((err**2).mean()) / np.sqrt((ref**2).mean())),
            max_abs=float(np.abs(err).max()),
        )

print("done", file=sys.stderr)
