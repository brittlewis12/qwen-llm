# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "torch", "einops"]
# ///
"""GLM-5.3 KDA decode fixture from FLA's naive recurrence.

Three tokens, two heads of width 128, nonzero nonsymmetric initial state and
nonuniform per-channel decay. Inputs are integer formulas evaluated identically
in Rust (`glm5_next::oracle` tests); conv, L2 norm, decay, beta and the output
norm/gate follow llama.cpp's glm5-next build_kda_layer; the recurrence itself is
`fla.ops.kda.naive.naive_recurrent_kda` (an independent implementation).

    uv run --offline scripts/reference/generate_glm53_kda_fixture.py [--check]
"""

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

FLA = Path(os.environ.get("FLA_DIR", Path.home() / "code/flash-linear-attention"))
# Load naive.py by path: the `fla` package imports triton, unavailable on macOS.
_spec = importlib.util.spec_from_file_location(
    "fla_kda_naive", FLA / "fla/ops/kda/naive.py"
)
_naive = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_naive)
naive_recurrent_kda = _naive.naive_recurrent_kda

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "crates/qwen-llm/tests/fixtures/glm53_kda_fla_v1"
H, D, T = 2, 128, 3
W = H * D
LOWER_BOUND, NORM_EPS = -5.0, 1e-5


def val(i, salt, scale):
    """Shared formula: ((i * 7919 + salt * 104729) % 2003) / 2003 - 0.5, scaled, as f32."""
    i = np.asarray(i, dtype=np.int64)
    return (
        ((i * 7919 + salt * 104729) % 2003).astype(np.float64) / 2003.0 - 0.5
    ) * scale


def f32(x):
    return np.asarray(x, dtype=np.float64).astype(np.float32)


def inputs():
    tw = np.arange(T * W)
    th = np.arange(T * H)
    return {
        "q": f32(val(tw, 1, 2.0)).reshape(T, W),
        "k": f32(val(tw, 2, 2.0)).reshape(T, W),
        "v": f32(val(tw, 3, 2.0)).reshape(T, W),
        "raw_gate": f32(val(tw, 4, 6.0)).reshape(T, W),
        "raw_beta": f32(val(th, 5, 4.0)).reshape(T, H),
        "output_gate": f32(val(tw, 6, 4.0)).reshape(T, W),
        "q_conv": f32(val(np.arange(W * 4), 7, 1.0)),
        "k_conv": f32(val(np.arange(W * 4), 8, 1.0)),
        "v_conv": f32(val(np.arange(W * 4), 9, 1.0)),
        "neg_exp_a_log": f32(-(0.75 + val(np.arange(H), 10, 1.0))),
        "dt_bias": f32(val(np.arange(W), 11, 2.0)),
        "output_norm": f32(1.0 + val(np.arange(D), 12, 0.5)),
        "conv_state": f32(val(np.arange(3 * 3 * W), 13, 1.0)),
        # [head][value][key]
        "state": f32(val(np.arange(H * D * D), 14, 0.2)),
    }


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def main():
    x = inputs()
    conv = x["conv_state"].astype(np.float64).reshape(3, 3, W).copy()
    taps = [
        x[n].astype(np.float64).reshape(W, 4) for n in ("q_conv", "k_conv", "v_conv")
    ]
    q_all, k_all, v_all, g_all, beta_all = [], [], [], [], []
    for t in range(T):
        conved = []
        for which, name in enumerate(("q", "k", "v")):
            new = x[name][t].astype(np.float64)
            acc = new * taps[which][:, 3] + (conv[which] * taps[which][:, :3].T).sum(
                axis=0
            )
            conv[which] = np.concatenate([conv[which][1:], new[None, :]], axis=0)
            conv[which] = conv[which].astype(np.float32).astype(np.float64)
            conved.append(acc * sigmoid(acc))
        q, k, v = (c.reshape(H, D) for c in conved)
        q = q / np.sqrt((q * q).sum(-1, keepdims=True) + 1e-6)
        k = k / np.sqrt((k * k).sum(-1, keepdims=True) + 1e-6)
        gate = (x["raw_gate"][t] + x["dt_bias"]).astype(np.float64).reshape(H, D)
        a = -x["neg_exp_a_log"].astype(np.float64)[:, None]
        g = LOWER_BOUND * sigmoid(a * gate)  # log-decay per key channel
        q_all.append(q), k_all.append(k), v_all.append(v), g_all.append(g)
        beta_all.append(sigmoid(x["raw_beta"][t].astype(np.float64)))

    tensor = lambda a: torch.tensor(np.stack(a)[None], dtype=torch.float64)  # noqa: E731
    s0 = torch.tensor(
        x["state"].reshape(H, D, D).transpose(0, 2, 1).copy()[None], dtype=torch.float64
    )
    o, s = naive_recurrent_kda(
        tensor(q_all),
        tensor(k_all),
        tensor(v_all),
        tensor(g_all),
        tensor(beta_all),
        scale=D**-0.5,
        initial_state=s0,
        output_final_state=True,
    )
    o = o[0].double().numpy()  # [T, H, D]
    rms = 1.0 / np.sqrt((o * o).mean(-1, keepdims=True) + NORM_EPS)
    og = x["output_gate"].astype(np.float64).reshape(T, H, D)
    out = o * rms * x["output_norm"].astype(np.float64) * sigmoid(og)
    final_state = s[0].double().numpy().transpose(0, 2, 1)  # back to [head][value][key]

    blob = np.concatenate(
        [
            out.reshape(-1).astype(np.float32),
            final_state.reshape(-1).astype(np.float32),
            conv.reshape(-1).astype(np.float32),
        ]
    )
    meta = {
        "generator": "scripts/reference/generate_glm53_kda_fixture.py",
        "fla_revision": subprocess.run(
            ["git", "-C", str(FLA), "rev-parse", "HEAD"], capture_output=True, text=True
        ).stdout.strip(),
        "fla_precision": "naive_recurrent_kda casts to float32 internally",
        "torch": torch.__version__,
        "heads": H,
        "head_dim": D,
        "tokens": T,
        "lower_bound": LOWER_BOUND,
        "norm_eps": NORM_EPS,
        "layout": {
            "out": [0, T * W],
            "state": [T * W, T * W + H * D * D],
            "conv_state": [T * W + H * D * D, T * W + H * D * D + 9 * W],
        },
        "f32_sha256": hashlib.sha256(blob.tobytes()).hexdigest(),
    }
    if "--check" in sys.argv:
        old = json.loads(OUT.with_suffix(".json").read_text())
        if old["f32_sha256"] != meta["f32_sha256"]:
            raise SystemExit("fixture drift")
        print("no drift")
        return
    OUT.with_suffix(".f32").write_bytes(blob.tobytes())
    OUT.with_suffix(".json").write_text(json.dumps(meta, indent=1) + "\n")
    print(f"wrote {OUT}.{{json,f32}} ({blob.size} floats)")


if __name__ == "__main__":
    main()
