# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy"]
# ///
"""Map #12: llama.cpp's own batched-vs-serial divergence on the frozen
natural-trajectory tokens (`reuse-natural-v1.json`), beside the native Fast
lineage's divergence from Exact (`reuse_natural_evaluate` report).

Inputs: per case, `glm53_oracle` logits (`GLMREF01`) from a serial run and a
`--batch` run over turn 2 plus the continuation. Only the final positions
(prompt end and each continuation position) are read. Native Exact matches
llama.cpp serial (P2/P4 gates), so llama.cpp batched vs serial is the
reference engine's envelope for the same prompt kernels question. The
oracle's --batch runs the whole sequence in one ubatch (continuation
positions included), while the native arms prefill the prompt and decode the
continuation serially; the prompt-end position is the closest analog.

  uv run scripts/reference/glm53/reuse_natural_envelope.py --oracle-dir DIR \\
    --report target/profiles/glm53/natural/report.json --out envelope.json
"""

import argparse
import json
import struct
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def read_tail(path, positions):
    """The last `positions` logit rows of a GLMREF01 file: (tokens, logits)."""
    with open(path, "rb") as f:
        magic = f.read(8)
        assert magic == b"GLMREF01", path
        vocab, count = struct.unpack("<II", f.read(8))
        row = 8 + 4 * vocab
        f.seek(16 + (count - positions) * row)
        tokens, rows = [], []
        for _ in range(positions):
            _, token = struct.unpack("<II", f.read(8))
            tokens.append(token)
            rows.append(
                np.frombuffer(f.read(4 * vocab), dtype="<f4").astype(np.float64)
            )
        return count, tokens, np.stack(rows)


def log_softmax(x):
    m = x.max(axis=-1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(axis=-1, keepdims=True))


def compare(reference, other):
    p, q = log_softmax(reference), log_softmax(other)
    kl = (np.exp(p) * (p - q)).sum(axis=-1)
    flips = []
    for i, (r, o) in enumerate(zip(reference, other)):
        a, b = int(r.argmax()), int(o.argmax())
        if a != b:
            flips.append([i, float(r[a] - r[b]), float(o[b] - o[a])])
    return kl, flips


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle-dir", required=True)
    parser.add_argument("--fixture", default=str(HERE / "reuse-natural-v1.json"))
    parser.add_argument("--report")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    fixture = json.loads(Path(args.fixture).read_text())
    report = json.loads(Path(args.report).read_text()) if args.report else None
    native = {c["id"]: c for c in report["cases"]} if report else {}
    rows = []
    for case in fixture["cases"]:
        cid = case["id"]
        cont = case["continuation"].split()
        positions = len(cont) + 1
        expected = (
            len(case["turn1"].split())
            + case["consumed"]
            + len(case["suffix"].split())
            + len(cont)
        )
        files = {
            m: Path(args.oracle_dir) / f"{cid}.{m}.bin" for m in ("serial", "batch")
        }
        if not all(f.exists() for f in files.values()):
            print(f"{cid}: oracle logits missing")
            continue
        (n_s, tok_s, serial), (n_b, tok_b, batched) = (
            read_tail(files[m], positions) for m in ("serial", "batch")
        )
        assert n_s == n_b == expected, (cid, n_s, n_b, expected)
        assert tok_s == tok_b
        kl, flips = compare(serial, batched)
        row = {
            "id": cid,
            "positions": positions,
            "tokens": expected,
            "llama_batched_vs_serial": {
                "prompt_end_kl": float(kl[0]),
                "worst_kl": float(kl.max()),
                "worst_at": int(kl.argmax()),
                "mean_kl": float(kl.mean()),
                "flips": flips,
            },
        }
        if cid in native:
            fast = native[cid]["fast"]
            row["native_fast_vs_exact"] = {
                rows_: {
                    arm: {
                        k: fast[rows_][arm][k] for k in ("worst_kl", "mean_kl", "flips")
                    }
                    | {"prompt_end_kl": fast[rows_][arm]["kl"][0]}
                    for arm in ("cold", "warm")
                }
                for rows_ in fast
            }
        rows.append(row)
        print(
            f"{cid}: llama.cpp batched vs serial: prompt end {kl[0]:.3e}, worst {kl.max():.3e}, "
            f"mean {kl.mean():.3e}, flips {len(flips)}"
            + (
                f" | native Fast cold 512 vs Exact: prompt end {native[cid]['fast']['512']['cold']['kl'][0]:.3e}, "
                f"worst {native[cid]['fast']['512']['cold']['worst_kl']:.3e}, "
                f"flips {len(native[cid]['fast']['512']['cold']['flips'])}"
                if cid in native
                else ""
            )
        )
    Path(args.out).write_text(
        json.dumps(
            {"schema": "glm53.reuse_natural_envelope.v1", "cases": rows}, indent=2
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
