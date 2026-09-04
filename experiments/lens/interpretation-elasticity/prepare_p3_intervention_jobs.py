import json
from pathlib import Path

from analyze_primary_coordinates import dose_record


ROOT = Path(__file__).parent
GENERATED = ROOT / "plans" / "generated" / "p3"
COORDINATES = ROOT / "artifacts" / "coordinates"
PROMPTS = ROOT / "prompts"
KAPPAS = (0.25, 0.5, 0.75, 1.0)

MODELS = {
    "qwen": {
        "name": "Qwen3.6-27B",
        "target_layer": 62,
        "layers": [22, 32, 42, 52, 56],
        "stance_token": 10179,
        "content_token": 3296,
        "j_artifact": "/Volumes/wdblack/weights-archive/workspace-lenses/qwen3.6-27b/j-lens-native-v1",
        "r_artifact": "/Volumes/wdblack/weights-archive/workspace-lenses/qwen3.6-27b/r-lens-native-v1",
    },
    "muse": {
        "name": "Muse-Glimmer-30B",
        "target_layer": 50,
        "layers": [27, 35, 43, 49],
        "stance_token": 158448,
        "content_token": 4097,
        "j_artifact": "/Users/tito/models/muse-glimmer/lenses/muse-glimmer-30b-j-lens-matched-v1",
        "r_artifact": "/Volumes/wdblack/weights-archive/jacobian-lenses/Muse-Glimmer-30B-rlens-published-v1",
    },
}

STEMS = {
    "p3a": "p3a-correct-challenged",
    "p3b": "p3b-incorrect-challenged",
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def read_score(run: dict, lens: str, layer: int, token_id: int) -> float:
    cell = next(
        cell
        for cell in run["live_readouts"]
        if cell["lens"].lower() == lens and cell["source_layer"] == layer
    )
    return next(row["score"] for row in cell["scores"] if row["token_id"] == token_id)


def stance_ratios() -> dict[tuple[str, str, str, int], float]:
    document = read_json(COORDINATES / "p3-depth-coordinate-doses.json")
    result = {}
    for row in document["rows"]:
        model = "qwen" if row["model"].startswith("Qwen") else "muse"
        if model == "qwen" and row["mode"] != "thinking":
            continue
        result[(model, row["item"], row["lens"].lower(), row["layer"])] = row[
            "effective_delta_over_target"
        ]
    return result


def content_ratios() -> dict[tuple[str, str, str, int], float]:
    result = {}
    qwen_root = COORDINATES / "qwen-p3-content"
    manifest = read_json(qwen_root / "manifest.json")
    runs = {row["id"]: read_json(qwen_root / row["path"]) for row in manifest["runs"]}
    for item in STEMS:
        target = runs[f"{item}-target-thinking"]
        twin = runs[f"{item}-twin-thinking"]
        for lens in ("j", "r"):
            for layer in MODELS["qwen"]["layers"]:
                dose = dose_record(
                    read_score(target, lens, layer, MODELS["qwen"]["content_token"]),
                    read_score(twin, lens, layer, MODELS["qwen"]["content_token"]),
                )
                result[("qwen", item, lens, layer)] = dose[
                    "effective_delta_over_target"
                ]

    muse_root = COORDINATES / "muse-p3-content"
    for item, stem in STEMS.items():
        target = read_json(muse_root / f"{stem}.target.run.json")
        twin = read_json(muse_root / f"{stem}.twin.run.json")
        for lens in ("j", "r"):
            for layer in MODELS["muse"]["layers"]:
                dose = dose_record(
                    read_score(target, lens, layer, MODELS["muse"]["content_token"]),
                    read_score(twin, lens, layer, MODELS["muse"]["content_token"]),
                )
                result[("muse", item, lens, layer)] = dose[
                    "effective_delta_over_target"
                ]
    return result


def plan(model: str, control: str, active_lens: str, layer: int) -> dict:
    config = MODELS[model]
    token_id = config[f"{control}_token"]
    lenses = [
        {
            "kind": "published_full_transport",
            "id": lens,
            "artifact": config[f"{lens}_artifact"],
            "token_ids": [token_id],
            "allow_unvalidated_transfer": True,
        }
        for lens in ("j", "r")
    ]
    selector = {
        "span_kind": "message_end_marker",
        "role": "user",
        "occurrence": "last",
        "edge": "start",
    }
    assistant = {"span_kind": "generated_assistant_start_marker", "edge": "start"}
    return {
        "version": 2,
        "lenses": lenses,
        "directions": [
            {
                "id": control,
                "lens": active_lens,
                "row": {"kind": "token_id", "token_id": token_id},
                "normalization": "unit_l2",
            }
        ],
        "operations": [
            {
                "id": "attenuate",
                "scope": {
                    "layers": {"kind": "values", "values": [layer]},
                    "prefill": {"kind": "rendered_spans", "selectors": [selector]},
                },
                "action": {
                    "kind": "projection_ablate",
                    "direction": control,
                    "coefficient": 1.0,
                },
            }
        ],
        "readouts": [
            {
                "id": f"{lens}-track-{control}",
                "lens": lens,
                "scope": {
                    "layers": {
                        "kind": "values",
                        "values": sorted({layer, config["target_layer"]}),
                    },
                    "prefill": {
                        "kind": "rendered_spans",
                        "selectors": [selector, assistant],
                    },
                },
                "top_k": 1,
            }
            for lens in ("j", "r")
        ],
    }


def coefficients(ratio: float) -> list[dict]:
    values = [
        ("zero-a", 0.0),
        ("zero-b", 0.0),
        *((f"kappa-{kappa}", kappa * ratio) for kappa in KAPPAS),
        ("full-ablation", 1.0),
    ]
    return [{"label": label, "coefficient": value} for label, value in values]


def main() -> None:
    GENERATED.mkdir(parents=True, exist_ok=True)
    stance = stance_ratios()
    content = content_ratios()
    jobs = []

    for item, stem in STEMS.items():
        request_path = GENERATED / f"qwen-{item}-target.requests.jsonl"
        request_path.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "id": f"{item}-thinking",
                            "messages": str(PROMPTS / f"{stem}.target.messages.json"),
                            "message_mode": "thinking",
                        },
                        separators=(",", ":"),
                    ),
                    json.dumps(
                        {
                            "id": f"{item}-no-thinking",
                            "messages": str(PROMPTS / f"{stem}.target.messages.json"),
                            "message_mode": "no_thinking",
                        },
                        separators=(",", ":"),
                    ),
                ]
            )
            + "\n"
        )

    for model, config in MODELS.items():
        model_dir = GENERATED / model
        model_dir.mkdir(exist_ok=True)
        for control, ratios in (("stance", stance), ("content", content)):
            for lens in ("j", "r"):
                for layer in config["layers"]:
                    plan_path = model_dir / f"{control}-{lens}-layer-{layer}.json"
                    plan_path.write_text(
                        json.dumps(plan(model, control, lens, layer), indent=2) + "\n"
                    )
                    for item, stem in STEMS.items():
                        ratio = ratios[(model, item, lens, layer)]
                        if ratio is None:
                            continue
                        jobs.append(
                            {
                                "model": model,
                                "control": control,
                                "item": item,
                                "lens": lens,
                                "layer": layer,
                                "plan": str(plan_path),
                                "messages": str(
                                    PROMPTS / f"{stem}.target.messages.json"
                                ),
                                "requests_jsonl": str(
                                    GENERATED / f"qwen-{item}-target.requests.jsonl"
                                )
                                if model == "qwen"
                                else None,
                                "arms": coefficients(ratio),
                            }
                        )

    document = {
        "schema": "interpretation_elasticity.p3_intervention_jobs",
        "schema_version": 1,
        "status": "fixed_before_intervention_outputs_viewed",
        "kappas": list(KAPPAS),
        "duplicate_zero_arms": 2,
        "full_ablation_coefficient": 1.0,
        "stance_tokens": {"qwen": 10179, "muse": 158448},
        "content_tokens": {"qwen": 3296, "muse": 4097},
        "content_token_display": " question",
        "content_replacement_policy": "never_replace_after_intervention_outputs",
        "random_manifest": str(ROOT / "controls" / "p3-random-v1" / "manifest.json"),
        "jobs": jobs,
    }
    output = GENERATED / "jobs.json"
    output.write_text(json.dumps(document, indent=2) + "\n")
    print(f"{len(jobs)} jobs -> {output}")


if __name__ == "__main__":
    main()
