# /// script
# requires-python = ">=3.11"
# dependencies = ["torch>=2.10", "numpy"]
# ///
"""Compare CPU .pt traces with runtime little-endian F32 traces, one tensor at a time."""

import argparse
import json
import math
from pathlib import Path
import re
import sys

import numpy as np
import torch


class TraceError(ValueError):
    pass


def read_json(path):
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError) as error:
        raise TraceError(f"{path}: {error}") from error
    if not isinstance(value, dict):
        raise TraceError(f"{path}: expected JSON object")
    return value


def inventory(directory, reference):
    result = {}
    layer_pattern = (
        r"pos-(\d+)-layer-(\d+)\.pt"
        if reference
        else r"pos-(\d+)-layer-(\d+)\.(hidden|pre)\.f32"
    )
    logits_pattern = r"pos-(\d+)-logits\." + ("pt" if reference else "f32")
    for path in directory.iterdir():
        layer = re.fullmatch(layer_pattern, path.name)
        logits = re.fullmatch(logits_pattern, path.name)
        if layer:
            pos, index = int(layer[1]), int(layer[2])
            kinds = ("hidden", "pre") if reference else (layer[3],)
        elif logits:
            pos, index, kinds = int(logits[1]), None, ("logits",)
        elif path.name.startswith("pos-") and path.suffix in (".pt", ".f32"):
            raise TraceError(f"Unrecognized trace filename: {path}")
        else:
            continue
        for kind in kinds:
            key = (pos, index, kind)
            if key in result:
                raise TraceError(f"Duplicate trace: {path} and {result[key]}")
            result[key] = path
    if not result:
        raise TraceError(f"No trace tensors in {directory}")
    return result


def order(key):
    pos, layer, kind = key
    return (
        pos,
        math.inf if layer is None else layer,
        {"hidden": 0, "pre": 1, "logits": 2}[kind],
    )


def check_inventory(reference, runtime):
    for label, missing in (
        ("runtime", reference.keys() - runtime.keys()),
        ("reference", runtime.keys() - reference.keys()),
    ):
        if missing:
            key = min(missing, key=order)
            raise TraceError(
                f"Missing {label} tensor at position={key[0]}, layer={key[1]}, tensor={key[2]}"
            )
    positions = sorted({key[0] for key in reference})
    layers = sorted({key[1] for key in reference if key[1] is not None})
    if (
        positions != list(range(len(positions)))
        or not layers
        or layers != list(range(len(layers)))
    ):
        raise TraceError("Positions and layers must be contiguous and start at zero")
    for pos in positions:
        for layer, kind in [
            (layer, kind) for layer in layers for kind in ("hidden", "pre")
        ] + [(None, "logits")]:
            if (pos, layer, kind) not in reference:
                raise TraceError(
                    f"Both traces missing tensor at position={pos}, layer={layer}, tensor={kind}"
                )
    return len(positions), len(layers)


def token_prefix(directory, metadata, positions):
    candidates = []
    sources = [("run.json", metadata)]
    if (directory / "tokens.json").exists():
        sources.append(("tokens.json", read_json(directory / "tokens.json")))

    def ids(value, label):
        if not isinstance(value, list) or any(
            type(x) is not int or x < 0 for x in value
        ):
            raise TraceError(
                f"{directory}/{label}: expected nonnegative integer token IDs"
            )
        return value

    for source, obj in sources:
        if "positions" in obj:
            entries = obj["positions"]
            if not isinstance(entries, list) or len(entries) != positions:
                raise TraceError(
                    f"{directory}/{source}: manifest positions do not cover trace"
                )
            sequence = []
            for pos, entry in enumerate(entries):
                if not isinstance(entry, dict) or entry.get("position") != pos:
                    raise TraceError(
                        f"{directory}/{source}: manifest positions must be contiguous from zero"
                    )
                sequence.append(entry.get("token_id"))
            candidates.append(ids(sequence, source + ":positions.token_id"))
        completed = obj.get("tokens_completed")
        if completed is not None and not isinstance(completed, list):
            if type(completed) is not int or completed != positions:
                raise TraceError(
                    f"{directory}/{source}: tokens_completed={completed} but trace has {positions} positions"
                )
        for key in ("consumed_token_ids", "input_token_ids", "token_ids", "tokens"):
            if key in obj:
                sequence = ids(obj[key], source + ":" + key)
                if key == "consumed_token_ids" and len(sequence) != positions:
                    raise TraceError(
                        f"{directory}/{source}: consumed_token_ids length differs from trace"
                    )
                candidates.append(sequence[:positions])
        if isinstance(completed, list):
            sequence = ids(completed, source + ":tokens_completed")
            if len(sequence) != positions:
                raise TraceError(
                    f"{directory}/{source}: tokens_completed list length differs from trace"
                )
            candidates.append(sequence)
        prompt_key = next(
            (
                k
                for k in ("prompt", "prompt_ids", "prompt_token_ids")
                if isinstance(obj.get(k), list)
            ),
            None,
        )
        if prompt_key:
            sequence = ids(obj[prompt_key], source + ":" + prompt_key)
            generated_key = next(
                (
                    k
                    for k in ("generated", "generated_ids", "generated_token_ids")
                    if k in obj
                ),
                None,
            )
            if generated_key:
                generated = ids(obj[generated_key], source + ":" + generated_key)
                sequence = sequence + generated
                if generated and len(sequence) - 1 != positions:
                    raise TraceError(
                        f"{directory}/{source}: consumed prompt/generated token count differs from trace"
                    )
                if len(sequence) < positions:
                    raise TraceError(
                        f"{directory}/{source}: prompt plus generated IDs do not cover trace"
                    )
            # The last sampled token generally has no forward pass; compare consumed IDs only.
            candidates.append(sequence[:positions])
    for a in candidates:
        for b in candidates:
            if a[: len(b)] != b[: len(a)]:
                raise TraceError(f"Conflicting token metadata in {directory}")
    return max(candidates, key=len, default=[])


def reference_tensor(path, kind, loaded=None):
    try:
        value = (
            torch.load(path, map_location="cpu", weights_only=True)
            if loaded is None
            else loaded
        )
    except Exception as error:
        raise TraceError(f"Cannot load reference {path}: {error}") from error
    tensor = (
        value.get("pre_mix" if kind == "pre" else kind)
        if isinstance(value, dict)
        else value
    )
    if not isinstance(tensor, torch.Tensor) or tensor.dtype not in (
        torch.float32,
        torch.bfloat16,
        torch.float16,
    ):
        raise TraceError(f"{path}: {kind} must be an F32/BF16/F16 tensor")
    rank = {"hidden": 2, "pre": 1, "logits": 1}[kind]
    while tensor.ndim > rank and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)
    if tensor.ndim != rank or not tensor.numel():
        raise TraceError(
            f"{path}: invalid {kind} shape {tuple(tensor.shape)}; expected rank {rank} after singleton batch/token axes"
        )
    if not torch.isfinite(tensor).all():
        raise TraceError(f"Nonfinite reference {kind}: {path}")
    return tensor.double(), value


def runtime_tensor(path, shape):
    expected = math.prod(shape) * 4
    size = path.stat().st_size
    if size != expected:
        raise TraceError(
            f"{path}: size/shape mismatch: {size} bytes, expected {expected} for {tuple(shape)} little-endian F32 (truncated or extra data)"
        )
    data = path.read_bytes()
    if len(data) != expected:
        raise TraceError(f"{path}: trace changed while reading")
    value = torch.from_numpy(
        np.frombuffer(data, dtype="<f4").astype(np.float64)
    ).reshape(shape)
    if not torch.isfinite(value).all():
        raise TraceError(f"Nonfinite runtime tensor: {path}")
    return value


def metrics(reference, runtime, atol, rtol):
    delta = runtime - reference
    absolute = delta.abs()
    norm = torch.linalg.vector_norm(reference).item()
    difference_norm = torch.linalg.vector_norm(delta).item()
    relative = (
        difference_norm / norm if norm else (0.0 if difference_norm == 0 else "inf")
    )
    return {
        "maxabs": absolute.max().item(),
        "rmse": delta.square().mean().sqrt().item(),
        "normrelative": relative,
    }, bool((absolute > atol + rtol * reference.abs()).any())


def compare(reference_dir, runtime_dir, atol=None, rtol=None):
    characterization = atol is None and rtol is None
    atol, rtol = 0.0 if atol is None else atol, 0.0 if rtol is None else rtol
    if not all(math.isfinite(x) and x >= 0 for x in (atol, rtol)):
        raise TraceError("Tolerances must be finite and nonnegative")
    reference_dir, runtime_dir = Path(reference_dir), Path(runtime_dir)
    reference_meta, runtime_meta = (
        read_json(reference_dir / "run.json"),
        read_json(runtime_dir / "run.json"),
    )
    modes = (reference_meta.get("mode"), runtime_meta.get("mode"))
    if modes[0] not in ("NativeF32", "ReferenceFp8") or modes[0] != modes[1]:
        raise TraceError(
            f"Mode mismatch or missing/unknown mode: reference={modes[0]}, runtime={modes[1]}"
        )
    for directory, metadata in (
        (reference_dir, reference_meta),
        (runtime_dir, runtime_meta),
    ):
        if "status" in metadata and metadata["status"] != "success":
            raise TraceError(
                f"{directory}: trace status={metadata['status']}; only completed successful traces can be compared"
            )
    for key, expected in (("dtype", "float32"), ("byte_order", "little")):
        if key in runtime_meta and runtime_meta[key] != expected:
            raise TraceError(
                f"{runtime_dir}: declared {key}={runtime_meta[key]}, expected {expected}"
            )
    reference, runtime = inventory(reference_dir, True), inventory(runtime_dir, False)
    positions, layers = check_inventory(reference, runtime)
    for directory, metadata in (
        (reference_dir, reference_meta),
        (runtime_dir, runtime_meta),
    ):
        for field in ("n_layers", "num_hidden_layers", "layer_count"):
            if field in metadata and metadata[field] != layers:
                raise TraceError(
                    f"{directory}: {field}={metadata[field]} but trace has {layers} layers"
                )
    ref_ids = token_prefix(reference_dir, reference_meta, positions)
    run_ids = token_prefix(runtime_dir, runtime_meta, positions)
    compared_ids = min(len(ref_ids), len(run_ids))
    for pos in range(compared_ids):
        if ref_ids[pos] != run_ids[pos]:
            raise TraceError(
                f"Token mismatch at position {pos}: reference={ref_ids[pos]}, runtime={run_ids[pos]}"
            )
    report = {
        "mode": modes[0],
        "comparison": "characterization_only"
        if characterization
        else "user_supplied_thresholds_not_certified",
        "atol": atol,
        "rtol": rtol,
        "threshold_rule": "abs(runtime-reference) > atol + rtol*abs(reference)",
        "positions": positions,
        "layers": layers,
        "token_positions_verified": compared_ids,
        "token_check": "complete"
        if compared_ids == positions
        else "partial"
        if compared_ids
        else "unavailable",
        "tensors_compared": 0,
        "first_divergence": None,
        "worst_maxabs": None,
        "logits": [],
    }
    shapes = {}
    previous_path, loaded = None, None
    for key in sorted(reference, key=order):
        pos, layer, kind = key
        path = reference[key]
        ref, loaded = reference_tensor(
            path, kind, loaded if path == previous_path else None
        )
        previous_path = path
        shape = tuple(ref.shape)
        if kind in shapes and shape != shapes[kind]:
            raise TraceError(
                f"{path}: {kind} shape changed from {shapes[kind]} to {shape}"
            )
        shapes[kind] = shape
        if kind == "pre" and shapes["hidden"][0] != shape[0]:
            raise TraceError(
                f"{path}: pre length {shape[0]} disagrees with hidden HC count {shapes['hidden'][0]}"
            )
        for directory, metadata in (
            (reference_dir, reference_meta),
            (runtime_dir, runtime_meta),
        ):
            declared = metadata.get(kind + "_shape")
            shape_map = metadata.get("shapes", {})
            if not isinstance(shape_map, dict):
                raise TraceError(f"{directory}: shapes must be an object")
            nested = shape_map.get("next_pre_mix" if kind == "pre" else kind)
            if nested is not None and nested != list(shape):
                raise TraceError(
                    f"{directory}: declared shapes.{kind}={nested}, reference implies {shape}"
                )
            if declared is not None and declared != list(shape):
                raise TraceError(
                    f"{directory}: declared {kind}_shape={declared}, reference implies {shape}"
                )
            sizes = metadata.get("bytes_per_file", {})
            if not isinstance(sizes, dict) or (
                kind in sizes and sizes[kind] != math.prod(shape) * 4
            ):
                raise TraceError(
                    f"{directory}: bytes_per_file disagrees with {kind} shape {shape}"
                )
            dimensions = {
                "hidden": {
                    "hc_mult": shape[0],
                    "dim": shape[-1],
                    "hidden_dim": shape[-1],
                },
                "pre": {"hc_mult": shape[0]},
                "logits": {"vocab_size": shape[0]},
            }[kind]
            for name, value in dimensions.items():
                if name in metadata and metadata[name] != value:
                    raise TraceError(
                        f"{directory}: {name}={metadata[name]} disagrees with {kind} shape {shape}"
                    )
        run = runtime_tensor(runtime[key], shape)
        stats, divergent = metrics(ref, run, atol, rtol)
        record = {"position": pos, "layer": layer, "tensor": kind, **stats}
        report["tensors_compared"] += 1
        if divergent and report["first_divergence"] is None:
            report["first_divergence"] = record
        if (
            report["worst_maxabs"] is None
            or stats["maxabs"] > report["worst_maxabs"]["maxabs"]
        ):
            report["worst_maxabs"] = record
        if kind == "logits":
            logp, logq = ref.log_softmax(-1), run.log_softmax(-1)
            kl = (logp.exp() * (logp - logq)).sum().item()
            ref_top, run_top = ref.argmax().item(), run.argmax().item()
            report["logits"].append(
                {
                    "position": pos,
                    **stats,
                    "kl_reference_to_runtime": max(0.0, kl),
                    "reference_top1": ref_top,
                    "runtime_top1": run_top,
                    "top1_match": ref_top == run_top,
                }
            )
    report["shapes"] = {key: list(value) for key, value in shapes.items()}
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument(
        "--atol",
        type=float,
        help="absolute characterization threshold; omitted means zero",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        help="relative characterization threshold; omitted means zero",
    )
    args = parser.parse_args(argv)
    try:
        report = compare(args.reference_dir, args.runtime_dir, args.atol, args.rtol)
    except (TraceError, OSError) as error:
        print(json.dumps({"error": str(error)}), file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
