"""Compressed-ID addressing only; offline verification needs only the stdlib.

Regenerate (optional, isolated dependency):
  uv run --no-project --python 3.12 --with numpy==2.2.6 python -B scripts/reference/deepseek_v41/engram_oracle.py --generate

No tokenizer mapping, weights, or floating-point Engram operator is exercised.
"""

import argparse
import ast
from dataclasses import dataclass
import hashlib
import json
from math import isqrt
from pathlib import Path
from types import SimpleNamespace

import capture

FIXTURE = Path(__file__).parent / "fixtures/engram-addresses.json"
NUMPY_VERSION = "2.2.6"
VOCAB = 99092
PAD = 17  # Synthetic compressed control, not the mapping of raw token 2.


def isprime(n):
    return n >= 2 and all(n % d for d in range(2, isqrt(n) + 1))


def extracted(contents, torch=None, np=None):
    tree = ast.parse(contents["inference/engram.py"])
    names = {"find_next_prime", "EngramLayout"}
    if torch is not None:
        names.update({"compute_hash_multipliers", "NgramHashState"})
    tree.body = [node for node in tree.body if getattr(node, "name", None) in names]
    for node in tree.body:
        if getattr(node, "name", None) == "NgramHashState":
            node.body = [n for n in node.body if getattr(n, "name", None) != "__init__"]
    env = dict(
        dataclass=dataclass,
        isprime=isprime,
        torch=torch,
        np=np,
        nn=SimpleNamespace(Module=object),
        __name__=__name__,
    )
    exec(compile(tree, "verified-inference/engram.py", "exec"), env)
    return env


def layout_from_source(contents, env):
    args = SimpleNamespace(**json.loads(contents["config.json"])["text_config"])
    layout = env["EngramLayout"].from_args(args)
    totals = tuple(sum(sum(group) for group in layer) for layer in layout.primes)
    assert totals == layout.num_embeddings == (384006168, 384016682)
    return layout


def validate_ids(ids):
    if any(type(t) is not int or not -1 <= t < VOCAB for t in ids):
        raise ValueError("expected compressed IDs -1 (dead) or 0..99091")


def reference(ids, params, history=(), pad_id=PAD):
    """Independent scalar integer oracle, carrying only the last three IDs."""
    validate_ids(ids)
    validate_ids(history)
    validate_ids([pad_id])
    if len(history) > 3 or pad_id < 0:
        raise ValueError("history must be bounded to three IDs; pad must be live")
    history = list(history)
    rows = []
    for token in ids:
        window = history + [token]
        tokens, blocked = [], False
        for shift in range(4):
            source = window[-1 - shift] if shift < len(window) else -1
            blocked = blocked or source == -1
            tokens.append(pad_id if blocked else source)
        layers = []
        for primes, multipliers in zip(params["primes"], params["multipliers"]):
            rolling = tokens[0] * multipliers[0]
            offset, layer = 0, []
            for i, group in enumerate(primes, 1):
                rolling ^= tokens[i] * multipliers[i]
                for prime in group:
                    layer.append(offset + rolling % prime)
                    offset += prime
            layers.append(layer)
        rows.append(layers)
        history = window[-3:]
    return rows, history


def numpy_facade(np):
    """Only the int64/bool CPU operations used by extracted source.

    All products are nonnegative and bounded below INT64_MAX; NumPy remainder,
    XOR, broadcasting and take_along_axis thus match these torch operations.
    No general torch compatibility or floating-point equivalence is claimed.
    """

    class Tensor(np.ndarray):
        device = "cpu"

        def expand(self, *shape):
            return wrap(np.broadcast_to(self, shape))

        def clamp_min(self, minimum):
            return wrap(np.maximum(self, minimum))

        def gather(self, dim, indices):
            return wrap(np.take_along_axis(self, indices, axis=dim))

        def unsqueeze(self, dim):
            return wrap(np.expand_dims(self, dim))

    def wrap(value):
        return np.asarray(value).view(Tensor)

    return SimpleNamespace(
        Tensor=Tensor,
        bool=np.bool_,
        int64=np.int64,
        tensor=lambda x: wrap(np.asarray(x, dtype=np.int64)),
        stack=lambda xs, dim=0: wrap(np.stack(xs, axis=dim)),
        cat=lambda xs, dim=0: wrap(np.concatenate(xs, axis=dim)),
        where=lambda condition, a, b: wrap(np.where(condition, a, b)),
        arange=lambda a, b, device=None: wrap(np.arange(a, b, dtype=np.int64)),
        zeros_like=lambda x, dtype: wrap(np.zeros_like(x, dtype=dtype)),
        bitwise_xor=lambda a, b: wrap(np.bitwise_xor(a, b)),
        inference_mode=lambda: lambda function: function,
    )


def source_rows(env, torch, np, layout, params, ids, chunks, history=()):
    validate_ids(ids)
    validate_ids(history)
    assert sum(chunks) == len(ids) and all(n > 0 for n in chunks)
    state = env["NgramHashState"]()
    state.layout, state.pad_id = layout, PAD
    state.primes = torch.tensor(params["primes"])
    state.multipliers = torch.tensor(params["multipliers"])
    state.offsets = torch.tensor(params["offsets"])
    # Identity mapping is deliberate: this fixture's inputs are already compressed.
    state.token_map = torch.tensor(range(VOCAB))
    state.cache = torch.tensor(np.zeros((1, len(history) + len(ids)), dtype=np.int64))
    state.cache[0, : len(history)] = history
    rows, consumed = [], 0
    for size in chunks:
        part = ids[consumed : consumed + size]
        inputs = torch.tensor([[max(0, t) for t in part]])
        mask = np.array([[t != -1 for t in part]], dtype=np.bool_)
        rows.extend(state.forward(inputs, len(history) + consumed, mask).tolist()[0])
        consumed += size
    return rows


def generate():
    import numpy as np

    if np.__version__ != NUMPY_VERSION:
        raise ValueError(f"generation requires numpy=={NUMPY_VERSION}")
    contents = capture.verify()
    torch = numpy_facade(np)
    env = extracted(contents, torch, np)
    layout = layout_from_source(contents, env)
    primes = [[list(group) for group in layer] for layer in layout.primes]
    offsets = []
    for layer in primes:
        total, row = 0, []
        for group in layer:
            for prime in group:
                row.append(total)
                total += prime
        offsets.append(row)
    params = dict(
        layer_ids=list(layout.layer_ids),
        primes=primes,
        offsets=offsets,
        table_totals=list(layout.num_embeddings),
        multipliers=env["compute_hash_multipliers"](
            layout.layer_ids, 4, VOCAB
        ).tolist(),
    )
    cases = []
    specifications = [
        ("start_max_and_dead_boundary", [], [0, 99091, 42, -1, 7, 8, 9, 10], [2, 3, 3]),
        ("leading_and_consecutive_dead", [], [-1, -1, 99091, PAD, 0], [1, 2, 2]),
        ("restored_live_history", [99091, 23, 41], [5, 6, 7, 8], [2, 2]),
        ("restored_dead_history", [12, -1, 13], [99091, 0, 1, 2], [1, 3]),
    ]
    for name, history, ids, chunks in specifications:
        expected = source_rows(env, torch, np, layout, params, ids, [len(ids)], history)
        assert expected == source_rows(
            env, torch, np, layout, params, ids, chunks, history
        )
        assert expected == source_rows(
            env, torch, np, layout, params, ids, [1] * len(ids), history
        )
        scalar, final_history = reference(ids, params, history)
        assert expected == scalar
        cases.append(
            dict(
                name=name,
                history=history,
                compressed_ids=ids,
                chunks=chunks,
                expected_rows=expected,
                final_history=final_history,
            )
        )
    return dict(
        schema_version=1,
        source_revision=capture.REVISION,
        source_sha256=hashlib.sha256(contents["inference/engram.py"]).hexdigest(),
        generation=dict(
            numpy=NUMPY_VERSION,
            rng="numpy.random.default_rng/PCG64",
            seed="10007 * layer_id",
            execution="extracted upstream AST with NumPy int64 CPU facade",
        ),
        scope="Compressed IDs only; identity token_map; no raw tokenizer mapping or floating Engram numerics",
        compressed_vocab_size=VOCAB,
        synthetic_pad_compressed_id=PAD,
        dead_id=-1,
        row_axes="position, layer [1,14], column [(2gram heads 0..7),(3gram heads 0..7),(4gram heads 0..7)]",
        parameters=params,
        cases=cases,
    )


def verify():
    contents = capture.verify()
    data = json.loads(FIXTURE.read_text())
    assert data["source_revision"] == capture.REVISION
    assert (
        data["source_sha256"]
        == hashlib.sha256(contents["inference/engram.py"]).hexdigest()
    )
    layout = layout_from_source(contents, extracted(contents))
    params = data["parameters"]
    assert params["primes"] == [[list(g) for g in layer] for layer in layout.primes]
    assert params["table_totals"] == list(layout.num_embeddings)
    assert params["layer_ids"] == list(layout.layer_ids)
    assert data["generation"]["numpy"] == NUMPY_VERSION
    assert data["compressed_vocab_size"] == VOCAB
    assert data["synthetic_pad_compressed_id"] == PAD and data["dead_id"] == -1
    if (
        len(data["cases"]) != 4
        or sum(len(case["compressed_ids"]) for case in data["cases"]) != 21
    ):
        raise ValueError("Engram fixture must cover four cases and 21 positions")
    for case in data["cases"]:
        rows, history = reference(case["compressed_ids"], params, case["history"])
        assert rows == case["expected_rows"]
        assert history == case["final_history"]
    return data


def write_fixture(path, data):
    if path.exists() and json.loads(path.read_text()) != data:
        raise ValueError(
            "Engram generation differs from the pinned fixture; review first"
        )
    path.write_text(json.dumps(data, indent=None) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generate", action="store_true")
    if parser.parse_args().generate:
        # Compact JSON keeps the complete address vectors small.
        write_fixture(FIXTURE, generate())
    verify()
    print(
        "Verified compressed-ID Engram addressing fixture (no model/tokenizer numerics)"
    )
