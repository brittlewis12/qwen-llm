import json
from pathlib import Path

from analyze_primary_coordinates import dose_record


ROOT = Path(__file__).parent
COORDINATE_ROOT = ROOT / "artifacts" / "coordinates" / "qwen-batch-02"
SELECTIONS = {
    "p5": {
        "token_id": 11656,
        "token": " impossible",
        "layer": 43,
        "target_position": 22,
        "twin_position": 21,
        "site": "final_prefill_after_no_thinking_marker",
    },
    "p6": {
        "token_id": 63901,
        "token": " skepticism",
        "layer": 46,
        "target_position": 34,
        "twin_position": 29,
        "site": "final_prefill_after_no_thinking_marker",
    },
    "p7": {
        "token_id": 15191,
        "token": " feelings",
        "layer": 49,
        "target_position": 34,
        "twin_position": 35,
        "site": "final_prefill_after_no_thinking_marker",
    },
    "p8": {
        "token_id": 43136,
        "token": " ambition",
        "layer": 52,
        "target_position": 28,
        "twin_position": 29,
        "site": "final_prefill_after_no_thinking_marker",
    },
    "p9": {
        "token_id": 13861,
        "token": " emotional",
        "layer": 33,
        "target_position": 30,
        "twin_position": 28,
        "site": "assistant_separator_before_no_thinking_marker",
    },
    "p10": {
        "token_id": 93111,
        "token": " procrast",
        "layer": 41,
        "target_position": 34,
        "twin_position": 32,
        "site": "final_prefill_after_no_thinking_marker",
    },
    "p11": {
        "token_id": 4021,
        "token": " cannot",
        "layer": 51,
        "target_position": 42,
        "twin_position": 49,
        "site": "final_prefill_after_no_thinking_marker",
    },
}


def load_runs() -> dict[str, dict]:
    manifest = json.loads((COORDINATE_ROOT / "manifest.json").read_text())
    return {
        row["id"]: json.loads((COORDINATE_ROOT / row["path"]).read_text())
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
        raise ValueError(f"expected token {token_id} in selected rows")
    return rows[0]["score"]


def main() -> None:
    runs = load_runs()
    rows = []
    for item, selection in SELECTIONS.items():
        target = runs[f"{item}-target-no-thinking"]
        twin = runs[f"{item}-twin-no-thinking"]
        for lens in ("j", "r"):
            target_score = score(
                target,
                lens,
                selection["layer"],
                selection["target_position"],
                selection["token_id"],
            )
            twin_score = score(
                twin,
                lens,
                selection["layer"],
                selection["twin_position"],
                selection["token_id"],
            )
            rows.append(
                {
                    "model": "Qwen3.6-27B",
                    "mode": "no-thinking",
                    "item": item,
                    "lens": lens.upper(),
                    **selection,
                    **dose_record(target_score, twin_score),
                }
            )

    output = {
        "schema": "interpretation_elasticity.batch_02_coordinate_doses",
        "schema_version": 1,
        "selection_status": "fixed_from_passive_traces_before_batch_02_interventions",
        "score_kind": "selected_row_projection_numerator",
        "rows": rows,
    }
    output_path = ROOT / "artifacts" / "coordinates" / "batch-02-coordinate-doses.json"
    output_path.write_text(json.dumps(output, indent=2) + "\n")
    for row in rows:
        ratio = row["effective_delta_over_target"]
        ratio_text = "n/a" if ratio is None else f"{ratio:.6f}"
        print(
            f"{row['item']} {row['lens']} {row['token']!r} L{row['layer']} "
            f"p={row['p_target']:.6f} twin={row['p_neutral']:.6f} "
            f"{row['geometry']} ratio={ratio_text}"
        )
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
