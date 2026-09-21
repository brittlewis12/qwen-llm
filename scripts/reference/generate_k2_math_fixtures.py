# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy==2.2.6"]
# ///
"""Independent, tiny equation fixtures; no checkpoint, model download, or GPU.

Uses batch causal attention rather than the Rust oracle's incremental algorithm.
F32 activations; F64 dot/reduction/RoPE/softmax intermediates; F16 cache rounding
occurs before all attention reads. This is not a HF/BF16 checkpoint oracle.
Run from any directory with: uv run scripts/reference/generate_k2_math_fixtures.py
"""

import json
from pathlib import Path

import numpy as np


def norm(x, gamma, g, fault):
    groups = 1 if fault == "whole_rms" else g["norm_groups"]
    grouped = x.astype(np.float64).reshape(len(x), groups, -1)
    normalized = grouped / np.sqrt(
        np.mean(grouped**2, axis=-1, keepdims=True) + g["epsilon"]
    )
    gamma = gamma.astype(np.float64) + (1 if fault == "gamma_plus_one" else 0)
    return (normalized.reshape(x.shape) * gamma).astype(np.float32)


def linear(x, weight):
    return (x.astype(np.float64) @ weight.astype(np.float64).T).astype(np.float32)


def rotate(x, positions, g, fault):
    shape = x.shape
    heads = x.astype(np.float64).reshape(len(x), -1, g["head_dim"])
    half = g["head_dim"] // 2
    angles = positions[:, None] * g["theta"] ** (-2 * np.arange(half) / g["head_dim"])
    cosine, sine = np.cos(angles)[:, None, :], np.sin(angles)[:, None, :]
    result = np.empty_like(heads)
    a_slice, b_slice = (
        (slice(None, None, 2), slice(1, None, 2))
        if fault == "interleaved_rope"
        else (slice(0, half), slice(half, None))
    )
    a, b = heads[..., a_slice], heads[..., b_slice]
    result[..., a_slice] = a * cosine - b * sine
    result[..., b_slice] = b * cosine + a * sine
    return result.reshape(shape).astype(np.float32)


def forward(g, weights, tokens, base, storage, fault=None):
    count = len(tokens)
    positions = np.arange(base, base + count, dtype=np.float64)
    x = weights["embedding"][tokens].copy()
    traces = [{"position": int(p), "layers": []} for p in positions]
    for layer in weights["layers"]:
        u = norm(x, layer["attention_norm"], g, fault)
        q = rotate(linear(u, layer["query"]), positions, g, fault)
        k = rotate(linear(u, layer["key"]), positions, g, fault)
        v = linear(u, layer["value"])
        if storage == "F16" and fault != "unrounded_cache":
            k, v = (
                k.astype(np.float16).astype(np.float32),
                v.astype(np.float16).astype(np.float32),
            )
        keys = k.reshape(count, g["kv_heads"], g["head_dim"])
        values = v.reshape(count, g["kv_heads"], g["head_dim"])
        group = g["query_heads"] // g["kv_heads"]
        indices = np.arange(g["query_heads"])
        indices = indices % g["kv_heads"] if fault == "modulo_gqa" else indices // group
        query = (
            q.reshape(count, g["query_heads"], g["head_dim"])
            .transpose(1, 0, 2)
            .astype(np.float64)
        )
        key = keys[:, indices].transpose(1, 0, 2).astype(np.float64)
        value = values[:, indices].transpose(1, 0, 2).astype(np.float64)
        scores = query @ key.transpose(0, 2, 1)
        scores *= 1.0 if fault == "missing_scale" else g["head_dim"] ** -0.5
        scores = np.where(np.tri(count, dtype=bool)[None], scores, -np.inf)
        probabilities = np.exp(scores - scores.max(axis=-1, keepdims=True))
        probabilities /= probabilities.sum(axis=-1, keepdims=True)
        attention = (
            (probabilities @ value)
            .transpose(1, 0, 2)
            .reshape(count, -1)
            .astype(np.float32)
        )
        post = (x + linear(attention, layer["attention_output"])).astype(np.float32)
        ff = norm(post, layer["feed_forward_norm"], g, fault)
        gate, up = linear(ff, layer["gate"]), linear(ff, layer["up"])
        gate64 = gate.astype(np.float64)
        silu = (gate64 / (1 + np.exp(-gate64))).astype(np.float32)
        gated = (silu * up).astype(np.float32)
        x = (post + linear(gated, layer["down"])).astype(np.float32)
        captures = dict(
            attention_norm=u,
            query_rotated=q,
            key_stored=k,
            value_stored=v,
            attention=attention,
            post_attention=post,
            feed_forward_norm=ff,
            gated=gated,
            residual=x,
        )
        for i, trace in enumerate(traces):
            trace["layers"].append(
                {name: value[i].tolist() for name, value in captures.items()}
            )
    final = norm(x, weights["output_norm"], g, fault)
    output = weights["output"]
    if fault == "head_transpose":
        output = output.ravel().reshape(g["hidden"], g["vocab"]).T
    logits = linear(final, output)
    for i, trace in enumerate(traces):
        trace.update(output_norm=final[i].tolist(), logits=logits[i].tolist())
    return traces


def main():
    g = dict(
        hidden=16,
        intermediate=24,
        query_heads=8,
        kv_heads=2,
        head_dim=4,
        norm_groups=4,
        vocab=23,
        theta=500000.0,
        epsilon=1e-6,
    )
    rng = np.random.Generator(np.random.PCG64(20260918))

    def matrix(out, width):
        return (rng.standard_normal((out, width)) * 0.23).astype(np.float32)

    def gamma():
        return rng.uniform(0.3, 1.7, g["hidden"]).astype(np.float32)

    h, f = g["hidden"], g["intermediate"]
    q, k = g["query_heads"] * g["head_dim"], g["kv_heads"] * g["head_dim"]
    weights = dict(
        embedding=matrix(g["vocab"], h),
        layers=[],
        output_norm=gamma(),
        output=matrix(g["vocab"], h),
    )
    for _ in range(2):
        weights["layers"].append(
            dict(
                attention_norm=gamma(),
                query=matrix(q, h),
                key=matrix(k, h),
                value=matrix(k, h),
                attention_output=matrix(h, q),
                feed_forward_norm=gamma(),
                gate=matrix(f, h),
                up=matrix(f, h),
                down=matrix(h, f),
            )
        )
    tokens = [2, 7, 3, 11]
    cases = []
    for theta, base in [(500000.0, 0), (1000000.0, 7919), (10000000.0, 131069)]:
        for storage in ["F32", "F16"]:
            geometry = dict(g, theta=theta)
            traces = forward(geometry, weights, tokens, base, storage)
            cases.append(
                dict(
                    theta=theta,
                    base=base,
                    storage=storage,
                    tokens=tokens,
                    traces=traces,
                )
            )
    faults = [
        "whole_rms",
        "gamma_plus_one",
        "interleaved_rope",
        "modulo_gqa",
        "missing_scale",
        "head_transpose",
        "unrounded_cache",
    ]
    control = cases[-1]
    geometry = dict(g, theta=control["theta"])
    expected = np.array([t["logits"] for t in control["traces"]])
    corruption = []
    for fault in faults:
        wrong = forward(geometry, weights, tokens, control["base"], "F16", fault)
        wrong = np.array([t["logits"] for t in wrong])
        delta = float(np.max(np.abs(expected - wrong)))
        assert delta > 5e-5, (fault, delta)
        corruption.append(
            dict(fault=fault, max_logit_delta=delta, logits=wrong.tolist())
        )

    def flatten(item):
        if isinstance(item, np.ndarray):
            return item.ravel().tolist()
        if isinstance(item, list):
            return [flatten(value) for value in item]
        if isinstance(item, dict):
            return {key: flatten(value) for key, value in item.items()}
        return item

    fixture = dict(
        schema=1,
        provenance="Synthetic equations, NumPy 2.2.6; not checkpoint parity",
        geometry=g,
        weights=flatten(weights),
        cases=cases,
        corruption=corruption,
    )
    destination = (
        Path(__file__).resolve().parents[2]
        / "crates/qwen-llm/tests/fixtures/k2_math_numpy.json"
    )
    destination.write_text(json.dumps(fixture, indent=2, allow_nan=False) + "\n")
    print(
        f"Wrote {destination}: {len(cases)} batch-causal cases; {len(corruption)} negative controls"
    )


if __name__ == "__main__":
    main()
