#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path


TENSOR_RE = re.compile(
    r"^\|\s*\d+\s*\|\s*(?P<name>[^|]+?)\s*\|\s*(?P<dtype>[A-Za-z0-9_]+)\s*\|\s*(?P<shape>[^|]+?)\s*\|"
)
BLK_RE = re.compile(r"^blk\.(?P<layer>\d+)\.(?P<suffix>.+)$")
SHARD_RE = re.compile(r"-\d{5}-of-(?P<total>\d{5})\.gguf$")

# These sets mirror the current prefill routing gates in metal_dflash.rs. They
# are intentionally stricter than generic mat-mat dispatch when the prefill path
# has a narrower fast-path predicate.
DENSE_FFN_PREFILL = {
    "F32",
    "F16",
    "BF16",
    "Q2_K",
    "Q3_K",
    "IQ2_S",
    "IQ3_XXS",
    "IQ3_S",
    "Q4_0",
    "Q4_1",
    "Q4_K",
    "Q5_K",
    "Q6_K",
    "Q8_0",
    "IQ4_NL",
    "IQ4_XS",
}
LM_PREFILL = DENSE_FFN_PREFILL
MATRIX_PREFILL = DENSE_FFN_PREFILL
GDN_SKINNY_PREFILL = MATRIX_PREFILL | {"F32"}
MOE_GATE_UP_FAST = {
    ("Q4_K", "Q4_K"),
    ("Q5_K", "Q5_K"),
    ("Q6_K", "Q6_K"),
    ("Q8_0", "Q8_0"),
    ("IQ3_XXS", "IQ3_XXS"),
    ("IQ3_S", "IQ3_S"),
    ("BF16", "BF16"),
}
MOE_DOWN_FAST = {"Q5_K", "Q6_K", "Q8_0", "IQ4_XS", "BF16"}
BLOCK_ALIGNMENT = {
    "Q2_K": 256,
    "Q3_K": 256,
    "IQ2_S": 256,
    "IQ3_XXS": 256,
    "IQ3_S": 256,
    "Q4_0": 32,
    "Q4_1": 32,
    "Q4_K": 256,
    "Q5_K": 256,
    "Q6_K": 256,
    "Q8_0": 32,
    "IQ4_NL": 32,
    "IQ4_XS": 256,
}


def parse_shape(text: str) -> list[int]:
    vals = []
    for part in text.strip().split(","):
        part = part.strip()
        if not part:
            continue
        try:
            vals.append(int(part))
        except ValueError:
            break
    return vals


def expand_models(items: list[str]) -> list[Path]:
    out: list[Path] = []
    for item in items:
        expanded = os.path.expanduser(item)
        matches = [Path(p) for p in glob.glob(expanded)]
        out.extend(matches if matches else [Path(expanded)])
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in out:
        path = path.resolve()
        if path not in seen:
            seen.add(path)
            unique.append(path)
    return unique


def gguf_bin_default() -> str:
    env = os.environ.get("GGUF_BIN")
    if env:
        return env
    local = Path("/Users/tito/code/gguf/target/release/gguf")
    return str(local) if local.exists() else "gguf"


def load_tensors(gguf_bin: str, model: Path) -> dict[str, dict[str, object]]:
    proc = subprocess.run(
        [gguf_bin, "--tensors", str(model)],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    tensors: dict[str, dict[str, object]] = {}
    for line in proc.stdout.splitlines():
        match = TENSOR_RE.match(line)
        if not match:
            continue
        name = match.group("name").strip()
        dtype = match.group("dtype").strip()
        if not (
            name.startswith("blk.") or name in {"token_embd.weight", "output.weight"}
        ):
            continue
        tensors[name] = {"dtype": dtype, "shape": parse_shape(match.group("shape"))}
    return tensors


def tensor_sources(model: Path) -> list[Path]:
    match = SHARD_RE.search(model.name)
    if not match:
        return [model]
    base = model.with_name(SHARD_RE.sub(".gguf", model.name))
    if base.exists():
        return [base]
    pattern = SHARD_RE.sub("-*-of-*.gguf", model.name)
    shards = sorted(model.parent.glob(pattern))
    return shards if shards else [model]


def load_model_tensors(
    gguf_bin: str, model: Path
) -> tuple[dict[str, dict[str, object]], list[Path]]:
    tensors: dict[str, dict[str, object]] = {}
    sources = tensor_sources(model)
    for source in sources:
        tensors.update(load_tensors(gguf_bin, source))
    return tensors, sources


def count_ok(total: int, bad: list[str]) -> str:
    return f"{total - len(bad)}/{total}" if total else "n/a"


def dtype_summary(items: list[tuple[str, ...]]) -> str:
    counts = Counter(items)
    parts = []
    for key, count in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        parts.append("/".join(key) + f":{count}")
    return ",".join(parts) if parts else "-"


def matrix_fast(info: dict[str, object], allowed: set[str]) -> bool:
    dtype = str(info["dtype"])
    if dtype not in allowed:
        return False
    align = BLOCK_ALIGNMENT.get(dtype)
    if align is None:
        return True
    shape = info.get("shape", [])
    return bool(shape) and int(shape[0]) % align == 0


def moe_q5_gateup_auto_ok(gate_shape: list[int]) -> bool:
    # Current auto gate is h == 3072 && f_exp == 1024 && n_expert == 256.
    # GGUF expert gate/up shape is [h, f_exp, n_expert, 1] for the observed A10B.
    return len(gate_shape) >= 3 and gate_shape[:3] == [3072, 1024, 256]


def moe_q6_gateup_auto_ok(gate_shape: list[int]) -> bool:
    # Current auto gate is h == 2048 && f_exp == 512 && n_expert == 256.
    # GGUF expert gate/up shape is [h, f_exp, n_expert, 1] for the observed A3B.
    return len(gate_shape) >= 3 and gate_shape[:3] == [2048, 512, 256]


def moe_q8_gateup_auto_ok(gate_shape: list[int]) -> bool:
    # Current auto gate is h == 2048 && f_exp == 512 && n_expert == 256.
    # GGUF expert gate/up shape is [h, f_exp, n_expert, 1] for the observed A3B.
    return len(gate_shape) >= 3 and gate_shape[:3] == [2048, 512, 256]


def moe_iq3_gateup_auto_ok(gate_shape: list[int]) -> bool:
    # Current auto gate is h == 2048 && f_exp == 512 && n_expert == 256.
    # GGUF expert gate/up shape is [h, f_exp, n_expert, 1] for the observed A3B.
    return len(gate_shape) >= 3 and gate_shape[:3] == [2048, 512, 256]


def moe_bf16_gateup_auto_ok(gate_shape: list[int]) -> bool:
    # Current auto gate is h == 2048 && f_exp == 512 && n_expert == 256.
    # GGUF expert gate/up shape is [h, f_exp, n_expert, 1] for local A3B BF16.
    return len(gate_shape) >= 3 and gate_shape[:3] == [2048, 512, 256]


def audit_model(
    model: Path, tensors: dict[str, dict[str, object]], sources: list[Path]
) -> dict[str, object]:
    layers: dict[int, dict[str, dict[str, object]]] = defaultdict(dict)
    for name, info in tensors.items():
        match = BLK_RE.match(name)
        if match:
            layers[int(match.group("layer"))][match.group("suffix")] = info

    dense_ffn_bad: list[str] = []
    dense_ffn_dtypes: list[tuple[str, ...]] = []
    gdn_bad: list[str] = []
    gdn_dtypes: list[tuple[str, ...]] = []
    attn_bad: list[str] = []
    attn_dtypes: list[tuple[str, ...]] = []
    moe_bad: list[str] = []
    moe_dtypes: list[tuple[str, ...]] = []

    dense_ffn_total = 0
    gdn_total = 0
    attn_total = 0
    moe_total = 0

    for layer, block in sorted(layers.items()):
        if "ffn_gate_exps.weight" in block:
            moe_total += 1
            missing = [
                name
                for name in (
                    "ffn_gate_exps.weight",
                    "ffn_up_exps.weight",
                    "ffn_down_exps.weight",
                )
                if name not in block
            ]
            if missing:
                moe_bad.append(f"{layer}:missing({','.join(missing)})")
                moe_dtypes.append(
                    tuple(
                        str(block[name]["dtype"]) if name in block else "missing"
                        for name in (
                            "ffn_gate_exps.weight",
                            "ffn_up_exps.weight",
                            "ffn_down_exps.weight",
                        )
                    )
                )
                continue
            gate = str(block["ffn_gate_exps.weight"]["dtype"])
            up = str(block["ffn_up_exps.weight"]["dtype"])
            down = str(block["ffn_down_exps.weight"]["dtype"])
            moe_dtypes.append((gate, up, down))
            gate_shape = block["ffn_gate_exps.weight"].get("shape", [])
            gateup_ok = (gate, up) in MOE_GATE_UP_FAST
            if (gate, up) == ("Q5_K", "Q5_K"):
                gateup_ok = gateup_ok and moe_q5_gateup_auto_ok(
                    gate_shape
                )  # scoped auto path
            if (gate, up) == ("Q6_K", "Q6_K"):
                gateup_ok = gateup_ok and moe_q6_gateup_auto_ok(
                    gate_shape
                )  # scoped auto path
            if (gate, up) == ("Q8_0", "Q8_0"):
                gateup_ok = gateup_ok and moe_q8_gateup_auto_ok(
                    gate_shape
                )  # scoped auto path
            if (gate, up) in (("IQ3_XXS", "IQ3_XXS"), ("IQ3_S", "IQ3_S")):
                gateup_ok = gateup_ok and moe_iq3_gateup_auto_ok(
                    gate_shape
                )  # scoped auto path
            if (gate, up) == ("BF16", "BF16"):
                gateup_ok = gateup_ok and moe_bf16_gateup_auto_ok(
                    gate_shape
                )  # scoped auto path
            if not gateup_ok or down not in MOE_DOWN_FAST:
                moe_bad.append(f"{layer}:{gate}/{up}/{down}")
            continue

        if all(
            name in block
            for name in ("ffn_gate.weight", "ffn_up.weight", "ffn_down.weight")
        ):
            dense_ffn_total += 1
            gate = str(block["ffn_gate.weight"]["dtype"])
            up = str(block["ffn_up.weight"]["dtype"])
            down = str(block["ffn_down.weight"]["dtype"])
            dense_ffn_dtypes.append((gate, up, down))
            if (
                not matrix_fast(block["ffn_gate.weight"], DENSE_FFN_PREFILL)
                or not matrix_fast(block["ffn_up.weight"], DENSE_FFN_PREFILL)
                or not matrix_fast(block["ffn_down.weight"], DENSE_FFN_PREFILL)
            ):
                dense_ffn_bad.append(f"{layer}:{gate}/{up}/{down}")

        if "ssm_out.weight" in block or "attn_qkv.weight" in block:
            wanted = [
                "attn_gate.weight",
                "attn_qkv.weight",
                "ssm_alpha.weight",
                "ssm_beta.weight",
                "ssm_out.weight",
            ]
            present = [name for name in wanted if name in block]
            if present:
                gdn_total += 1
                dtypes = tuple(str(block[name]["dtype"]) for name in present)
                gdn_dtypes.append(dtypes)
                ok = True
                for name in ("attn_gate.weight", "attn_qkv.weight", "ssm_out.weight"):
                    if name in block and not matrix_fast(block[name], MATRIX_PREFILL):
                        ok = False
                for name in ("ssm_alpha.weight", "ssm_beta.weight"):
                    if name in block and not matrix_fast(
                        block[name], GDN_SKINNY_PREFILL
                    ):
                        ok = False
                if not ok:
                    gdn_bad.append(f"{layer}:" + "/".join(dtypes))

        if all(
            name in block
            for name in (
                "attn_q.weight",
                "attn_k.weight",
                "attn_v.weight",
                "attn_output.weight",
            )
        ):
            attn_total += 1
            dtypes = tuple(
                str(block[name]["dtype"])
                for name in (
                    "attn_q.weight",
                    "attn_k.weight",
                    "attn_v.weight",
                    "attn_output.weight",
                )
            )
            attn_dtypes.append(dtypes)
            if any(
                not matrix_fast(block[name], MATRIX_PREFILL)
                for name in (
                    "attn_q.weight",
                    "attn_k.weight",
                    "attn_v.weight",
                    "attn_output.weight",
                )
            ):
                attn_bad.append(f"{layer}:" + "/".join(dtypes))

    lm_info = tensors.get("output.weight") or tensors.get("token_embd.weight")
    lm_dtype = str(lm_info["dtype"]) if lm_info else "missing"
    lm_fast = bool(lm_info) and matrix_fast(lm_info, LM_PREFILL)

    gaps = []
    if dense_ffn_bad:
        gaps.append(
            "dense_ffn="
            + ";".join(dense_ffn_bad[:8])
            + (";..." if len(dense_ffn_bad) > 8 else "")
        )
    if gdn_bad:
        gaps.append(
            "gdn=" + ";".join(gdn_bad[:8]) + (";..." if len(gdn_bad) > 8 else "")
        )
    if attn_bad:
        gaps.append(
            "attn=" + ";".join(attn_bad[:8]) + (";..." if len(attn_bad) > 8 else "")
        )
    if moe_bad:
        gaps.append(
            "moe=" + ";".join(moe_bad[:8]) + (";..." if len(moe_bad) > 8 else "")
        )
    if lm_info and not lm_fast:
        gaps.append(f"lm={lm_dtype}")

    return {
        "model": str(model),
        "name": model.name,
        "tensor_sources": ",".join(source.name for source in sources),
        "layers": len(layers),
        "dense_ffn_fast": count_ok(dense_ffn_total, dense_ffn_bad),
        "gdn_matrix_fast": count_ok(gdn_total, gdn_bad),
        "attn_matrix_fast": count_ok(attn_total, attn_bad),
        "moe_grouped_fast": count_ok(moe_total, moe_bad),
        "lm_fast": "yes" if lm_fast else f"no:{lm_dtype}",
        "dense_ffn_dtypes": dtype_summary(dense_ffn_dtypes),
        "gdn_dtypes": dtype_summary(gdn_dtypes),
        "attn_dtypes": dtype_summary(attn_dtypes),
        "moe_dtypes": dtype_summary(moe_dtypes),
        "gaps": " | ".join(gaps) if gaps else "-",
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit GGUF tensor dtypes against current qwen prefill fast-path gates."
    )
    parser.add_argument("models", nargs="+", help="GGUF files or glob patterns.")
    parser.add_argument("--gguf-bin", default=gguf_bin_default())
    parser.add_argument(
        "--json", action="store_true", help="Emit compact JSON instead of TSV."
    )
    args = parser.parse_args()

    rows = []
    for model in expand_models(args.models):
        if not model.exists():
            print(f"missing model: {model}", file=sys.stderr)
            return 2
        tensors, sources = load_model_tensors(args.gguf_bin, model)
        rows.append(audit_model(model, tensors, sources))

    if args.json:
        print(json.dumps({"models": rows}, separators=(",", ":")))
        return 0

    fields = [
        "name",
        "tensor_sources",
        "layers",
        "dense_ffn_fast",
        "gdn_matrix_fast",
        "attn_matrix_fast",
        "moe_grouped_fast",
        "lm_fast",
        "dense_ffn_dtypes",
        "gdn_dtypes",
        "attn_dtypes",
        "moe_dtypes",
        "gaps",
    ]
    print("\t".join(fields))
    for row in rows:
        print("\t".join(str(row[field]) for field in fields))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
