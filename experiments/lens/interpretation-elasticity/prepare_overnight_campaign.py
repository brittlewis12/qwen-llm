import json
from pathlib import Path

from analyze_primary_coordinates import dose_record


ROOT = Path(__file__).parent
GENERATED = ROOT / "plans" / "generated" / "overnight-2026-09-04"
COORDINATES = ROOT / "artifacts" / "coordinates"
KAPPAS = (0.25, 0.5, 0.75, 1.0)
TARGET_LAYER = 62
J_ARTIFACT = (
    "/Volumes/wdblack/weights-archive/workspace-lenses/qwen3.6-27b/j-lens-native-v1"
)
R_ARTIFACT = (
    "/Volumes/wdblack/weights-archive/workspace-lenses/qwen3.6-27b/r-lens-native-v1"
)

ITEMS = {
    "p2": {
        "batch": 1,
        "stem": "p2-deadpan-review",
        "layer": 43,
        "target_position": 35,
        "twin_position": 36,
        "readout_positions": [35, 41],
        "site": "generated_assistant_start_marker",
        "stance_token": 65105,
        "stance_display": " mediocre",
        "content_token": 10413,
        "content_display": " restaurant",
    },
    "p4": {
        "batch": 1,
        "stem": "p4-wrong-test",
        "layer": 38,
        "target_position": 119,
        "twin_position": 119,
        "readout_positions": [119, 121, 127],
        "site": "message_end_marker_user",
        "stance_token": 1412,
        "stance_display": " error",
        "content_token": 9584,
        "content_display": " bug",
    },
    "p5": {
        "batch": 2,
        "stem": "p5-riemann",
        "layer": 43,
        "target_position": 22,
        "twin_position": 21,
        "readout_positions": [22],
        "site": "final_prefill_after_no_thinking_marker",
        "stance_token": 11656,
        "stance_display": " impossible",
        "content_token": 52652,
        "content_display": " computational",
    },
    "p6": {
        "batch": 2,
        "stem": "p6-grad-school",
        "layer": 46,
        "target_position": 34,
        "twin_position": 29,
        "readout_positions": [34],
        "site": "final_prefill_after_no_thinking_marker",
        "stance_token": 63901,
        "stance_display": " skepticism",
        "content_token": 4087,
        "content_display": " answer",
    },
    "p7": {
        "batch": 2,
        "stem": "p7-quitting",
        "layer": 49,
        "target_position": 34,
        "twin_position": 35,
        "readout_positions": [34],
        "site": "final_prefill_after_no_thinking_marker",
        "stance_token": 15191,
        "stance_display": " feelings",
        "content_token": 27332,
        "content_display": " entrepreneur",
    },
    "p9": {
        "batch": 2,
        "stem": "p9-relationship",
        "layer": 33,
        "target_position": 30,
        "twin_position": 28,
        "readout_positions": [30, 34],
        "site": "assistant_separator_before_no_thinking_marker",
        "stance_token": 13861,
        "stance_display": " emotional",
        "content_token": 10125,
        "content_display": " conversation",
    },
}


def load_runs(batch: int) -> dict[str, dict]:
    directory = COORDINATES / f"qwen-overnight-batch-{batch:02}"
    manifest = json.loads((directory / "manifest.json").read_text())
    return {
        row["id"]: json.loads((directory / row["path"]).read_text())
        for row in manifest["runs"]
    }


def score(run: dict, lens: str, layer: int, position: int, token_id: int) -> float:
    cells = [
        cell
        for cell in run["live_readouts"]
        if cell["lens"].lower() == lens
        and cell["source_layer"] == layer
        and cell["phase"] == "prefill"
        and cell["index"] == position
    ]
    if len(cells) != 1:
        raise ValueError(f"expected one {lens} L{layer} p{position} cell")
    rows = [row for row in cells[0]["scores"] if row["token_id"] == token_id]
    if len(rows) != 1:
        raise ValueError(f"expected one selected score for token {token_id}")
    return rows[0]["score"]


def make_plan(item: dict, control: str, active_lens: str) -> dict:
    token_ids = [item["stance_token"], item["content_token"]]
    token_id = item[f"{control}_token"]
    lenses = [
        {
            "kind": "published_full_transport",
            "id": lens,
            "artifact": J_ARTIFACT if lens == "j" else R_ARTIFACT,
            "token_ids": token_ids,
            "allow_unvalidated_transfer": True,
        }
        for lens in ("j", "r")
    ]
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
                    "layers": {"kind": "values", "values": [item["layer"]]},
                    "prefill": {
                        "kind": "values",
                        "values": [item["target_position"]],
                    },
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
                        "values": sorted({item["layer"], TARGET_LAYER}),
                    },
                    "prefill": {
                        "kind": "values",
                        "values": item["readout_positions"],
                    },
                },
                "top_k": len(token_ids),
            }
            for lens in ("j", "r")
        ],
    }


def arms(ratio: float) -> list[dict]:
    values = [
        ("zero-a", 0.0),
        ("zero-b", 0.0),
        *((f"kappa-{kappa}", kappa * ratio) for kappa in KAPPAS),
        ("full-ablation", 1.0),
    ]
    return [{"label": label, "coefficient": value} for label, value in values]


def main() -> None:
    GENERATED.mkdir(parents=True, exist_ok=True)
    runs = {batch: load_runs(batch) for batch in (1, 2)}
    measurements = []
    jobs = []

    for item_id, item in ITEMS.items():
        batch_runs = runs[item["batch"]]
        target = batch_runs[f"{item_id}-target-no-thinking"]
        twin = batch_runs[f"{item_id}-twin-no-thinking"]
        item_dir = GENERATED / item_id
        item_dir.mkdir(exist_ok=True)
        for control in ("stance", "content"):
            token_id = item[f"{control}_token"]
            for lens in ("j", "r"):
                dose = dose_record(
                    score(
                        target,
                        lens,
                        item["layer"],
                        item["target_position"],
                        token_id,
                    ),
                    score(
                        twin,
                        lens,
                        item["layer"],
                        item["twin_position"],
                        token_id,
                    ),
                )
                measurement = {
                    "item": item_id,
                    "control": control,
                    "lens": lens,
                    "token_id": token_id,
                    "token": item[f"{control}_display"],
                    "layer": item["layer"],
                    "target_position": item["target_position"],
                    "twin_position": item["twin_position"],
                    "site": item["site"],
                    **dose,
                }
                measurements.append(measurement)
                if dose["effective_delta_over_target"] is None:
                    continue
                plan_path = item_dir / f"{control}-{lens}.json"
                plan_path.write_text(
                    json.dumps(make_plan(item, control, lens), indent=2) + "\n"
                )
                jobs.append(
                    {
                        "item": item_id,
                        "control": control,
                        "lens": lens,
                        "plan": str(plan_path),
                        "messages": str(
                            ROOT / "prompts" / f"{item['stem']}.target.messages.json"
                        ),
                        "mode": "no_thinking",
                        "max_new_tokens": 1024,
                        "arms": arms(dose["effective_delta_over_target"]),
                    }
                )

    output = {
        "schema": "interpretation_elasticity.overnight_campaign",
        "schema_version": 1,
        "created_date": "2026-09-04",
        "selection_status": "fixed_from_passive_and_exact_coordinates_before_intervention_outputs",
        "decode": "greedy",
        "kappas": list(KAPPAS),
        "duplicate_zero_arms": 2,
        "full_ablation_coefficient": 1.0,
        "random_control_status": "unavailable_in_current_runtime",
        "deferred": [
            {
                "item": "p11",
                "reason": "refusal-boundary probe held behind main construal items",
            }
        ],
        "excluded": [
            {
                "item": "p8",
                "token": " ambition",
                "reason": "exact J and R target coordinates were below twin",
            },
            {
                "item": "p10",
                "token": " procrast",
                "reason": "exact J and R target coordinates were below twin",
            },
        ],
        "measurements": measurements,
        "jobs": jobs,
    }
    output_path = GENERATED / "campaign.json"
    output_path.write_text(json.dumps(output, indent=2) + "\n")
    print(f"{len(jobs)} admissible jobs -> {output_path}")
    for row in measurements:
        ratio = row["effective_delta_over_target"]
        print(
            f"{row['item']} {row['control']:<7} {row['lens'].upper()} "
            f"{row['geometry']:<20} "
            f"ratio={'n/a' if ratio is None else f'{ratio:.6f}'}"
        )


if __name__ == "__main__":
    main()
