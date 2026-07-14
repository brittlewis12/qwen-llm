#!/usr/bin/env python3

import json
from pathlib import Path
import statistics
import subprocess


ROOT = Path(__file__).resolve().parents[2]
MODELS = Path("/Users/tito/models")
BINARY = ROOT / "target/release/qwen-bench"
ARTIFACT = ROOT / "target/profiles/v0594-retained-storage-coverage"
EXTRA_MODELS = (
    MODELS / "BF16" / "Qwen3.5-35B-A3B-BF16-00001-of-00002.gguf",
    MODELS
    / "unsloth-Qwen3.5-122B-A10B-GGUF/UD-Q4_K_XL"
    / "Qwen3.5-122B-A10B-UD-Q4_K_XL-00001-of-00003.gguf",
    MODELS / "unsloth-Qwen3.6-35B-A3B-MTP-GGUF" / "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
)


def command_text(command: list[str]) -> str:
    return subprocess.run(
        command,
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    ).stdout


def source_and_build_identity() -> tuple[str, dict[str, object]]:
    relative = Path(__file__).resolve().relative_to(ROOT)
    command_text(["git", "ls-files", "--error-unmatch", str(relative)])
    commit = command_text(["git", "rev-parse", "HEAD"]).strip()
    dirty = command_text(
        ["git", "status", "--porcelain=v1", "--untracked-files=no"]
    ).strip()
    if dirty:
        raise RuntimeError(f"tracked source is dirty: {dirty!r}")
    build = json.loads(command_text([str(BINARY), "build-info", "--output", "json"]))
    if (
        build.get("build_commit") != commit
        or build.get("runtime_commit") != commit
        or build.get("status") != "match"
        or build.get("build_dirty") is not False
        or build.get("runtime_dirty") is not False
    ):
        raise RuntimeError(f"source/build identity mismatch: {build}")
    return commit, build


def discover_models() -> list[Path]:
    models = sorted(MODELS.glob("Qwen3.[56]-*.gguf"))
    models.extend(EXTRA_MODELS)
    missing = [path for path in models if not path.is_file()]
    if missing:
        raise RuntimeError(f"missing model assets: {missing}")
    if len(models) != len(set(models)):
        raise RuntimeError("model discovery produced duplicate paths")
    return models


def run_model(path: Path, build: dict[str, object]) -> dict[str, object]:
    output = command_text(
        [
            str(BINARY),
            "gguf-storage-plan",
            "--model",
            str(path),
            "--embedding-policy",
            "force-native-if-supported",
            "--output",
            "json",
        ]
    )
    row = json.loads(output)
    if row.get("model") != str(path):
        raise RuntimeError(f"model identity drift for {path}")
    if row.get("schema_version") != 1:
        raise RuntimeError(f"storage-plan schema drift for {path}")
    if row.get("build_identity") != build:
        raise RuntimeError(f"build identity drift for {path}")
    if row.get("embedding_policy") != "force-native-if-supported":
        raise RuntimeError(f"embedding policy drift for {path}")
    if row.get("router_f16") is not False or row.get("required_alignment") != 32:
        raise RuntimeError(f"materialization policy drift for {path}")
    if row.get("planned_max_buffer_length") != row.get("device_max_buffer_length"):
        raise RuntimeError(f"device-limit policy drift for {path}")
    if not isinstance(row.get("descriptor_layout_digest"), str):
        raise RuntimeError(f"descriptor digest missing for {path}")
    if (
        row["planned_base_weight_private_bytes_under_policy"]
        > row["copied_base_weight_private_bytes_under_policy"]
    ):
        raise RuntimeError(f"planned residency exceeds current residency for {path}")
    removed = (
        row["copied_base_weight_private_bytes_under_policy"]
        - row["planned_base_weight_private_bytes_under_policy"]
    )
    if row["estimated_base_weight_private_bytes_removed_under_policy"] != removed:
        raise RuntimeError(f"private-byte accounting drift for {path}")
    bad_fallbacks = [
        fallback
        for fallback in row["fallbacks"]
        if fallback["reason"] != "FinalPartialPage"
    ]
    if bad_fallbacks:
        raise RuntimeError(f"non-tail fallback for {path}: {bad_fallbacks}")
    if row["unbound_descriptor_count"] or row["unbound_descriptor_bytes"]:
        raise RuntimeError(f"truly unbound descriptors for {path}")
    if bool(row["mtp_descriptor_count"]) != row["mtp_present"]:
        raise RuntimeError(f"MTP descriptor accounting drift for {path}")
    return row


def main() -> None:
    if not BINARY.is_file():
        raise SystemExit(f"missing release binary {BINARY}")
    commit, build = source_and_build_identity()
    models = discover_models()
    if ARTIFACT.exists():
        raise SystemExit(f"refusing to reuse artifact directory {ARTIFACT}")
    if not ARTIFACT.parent.is_dir():
        raise SystemExit(f"missing artifact parent {ARTIFACT.parent}")
    ARTIFACT.mkdir()

    rows = []
    rows_path = ARTIFACT / "rows.jsonl"
    for index, model in enumerate(models, 1):
        row = run_model(model, build)
        rows.append(row)
        with rows_path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(row, sort_keys=True) + "\n")
        ratio = (
            row["estimated_base_weight_private_bytes_removed_under_policy"]
            / row["copied_base_weight_private_bytes_under_policy"]
        )
        removed_gb = (
            row["estimated_base_weight_private_bytes_removed_under_policy"] / 1e9
        )
        print(
            f"{index:02d}/{len(models):02d} {model.name}: "
            f"shards={len(row['shard_mapped_lengths'])} windows={len(row['windows'])} "
            f"removed={removed_gb:.3f} GB "
            f"coverage={ratio:.4f}"
        )

    coverage = [
        row["estimated_base_weight_private_bytes_removed_under_policy"]
        / row["copied_base_weight_private_bytes_under_policy"]
        for row in rows
    ]
    architecture_counts = {}
    for row in rows:
        architecture = row["architecture"]
        architecture_counts[architecture] = architecture_counts.get(architecture, 0) + 1
    summary = {
        "schema_version": 1,
        "source_commit": commit,
        "build_identity": build,
        "embedding_policy": "force-native-if-supported",
        "router_f16": False,
        "models": len(rows),
        "architecture_counts": architecture_counts,
        "multi_shard_models": sum(len(row["shard_mapped_lengths"]) > 1 for row in rows),
        "mtp_models": sum(row["mtp_present"] for row in rows),
        "models_with_only_tail_fallbacks": sum(
            all(
                fallback["reason"] == "FinalPartialPage"
                for fallback in row["fallbacks"]
            )
            for row in rows
        ),
        "minimum_base_weight_private_removal_ratio_under_policy": min(coverage),
        "median_base_weight_private_removal_ratio_under_policy": statistics.median(
            coverage
        ),
        "maximum_base_weight_private_removal_ratio_under_policy": max(coverage),
        "minimum_base_weight_private_bytes_removed_under_policy": min(
            row["estimated_base_weight_private_bytes_removed_under_policy"]
            for row in rows
        ),
        "maximum_base_weight_private_bytes_removed_under_policy": max(
            row["estimated_base_weight_private_bytes_removed_under_policy"]
            for row in rows
        ),
    }
    (ARTIFACT / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
