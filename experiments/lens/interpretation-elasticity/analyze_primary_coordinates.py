import json
from pathlib import Path


ROOT = Path(__file__).parent
COORDINATES = ROOT / "artifacts" / "coordinates"
KAPPAS = (0.25, 0.5, 0.75, 1.0)

QWEN_SELECTIONS = {
    "p2": {
        "token_id": 65105,
        "token": " mediocre",
        "layer": 43,
        "site": "generated_assistant_start_marker",
    },
    "p3a": {
        "token_id": 10179,
        "token": " doubt",
        "layer": 32,
        "site": "message_end_marker",
    },
    "p3b": {
        "token_id": 10179,
        "token": " doubt",
        "layer": 32,
        "site": "message_end_marker",
    },
    "p4": {
        "token_id": 1412,
        "token": " error",
        "layer": 38,
        "site": "message_end_marker",
    },
}

MUSE_SELECTIONS = {
    "p2": {
        "stem": "p2-deadpan-review",
        "token_id": 190536,
        "token": " mediocre",
        "layer": 29,
        "site": "generated_assistant_start_marker",
    },
    "p3a": {
        "stem": "p3a-correct-challenged",
        "token_id": 158448,
        "token": " skeptical",
        "layer": 35,
        "site": "message_end_marker",
    },
    "p3b": {
        "stem": "p3b-incorrect-challenged",
        "token_id": 158448,
        "token": " skeptical",
        "layer": 35,
        "site": "message_end_marker",
    },
    "p4": {
        "stem": "p4-wrong-test",
        "token_id": 170202,
        "token": " contradictory",
        "layer": 32,
        "site": "message_end_marker",
    },
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def site_index(run: dict, lens: str, span_kind: str) -> int:
    owner = f"{lens}-primary-coordinates"
    matches = [
        binding["resolved_index"]
        for binding in run["position_bindings"]
        if binding["owner_id"] == owner
        and binding["selector"]["span_kind"] == span_kind
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one {owner} {span_kind} binding, got {matches}")
    return matches[0]


def coordinate(run: dict, lens: str, layer: int, site: str, token_id: int) -> float:
    index = site_index(run, lens, site)
    cells = [
        cell
        for cell in run["live_readouts"]
        if cell["lens"].lower() == lens
        and cell["source_layer"] == layer
        and cell["phase"] == "prefill"
        and cell["index"] == index
    ]
    if len(cells) != 1:
        raise ValueError(f"expected one {lens} L{layer} position {index} readout")
    scores = [row["score"] for row in cells[0]["scores"] if row["token_id"] == token_id]
    if len(scores) != 1:
        raise ValueError(f"expected one score for token {token_id}")
    return scores[0]


def dose_record(target: float, neutral: float) -> dict:
    delta = target - neutral
    raw_ratio = delta / target
    if target <= 0 or delta <= 0:
        geometry = "inadmissible"
        effective_ratio = None
    elif neutral <= 0:
        geometry = "zero_floor"
        effective_ratio = 1.0
    else:
        geometry = "target_minus_neutral"
        effective_ratio = raw_ratio
    return {
        "p_target": target,
        "p_neutral": neutral,
        "delta_p": delta,
        "raw_delta_over_target": raw_ratio,
        "geometry": geometry,
        "effective_delta_over_target": effective_ratio,
        "lambda_by_kappa": None
        if effective_ratio is None
        else {str(kappa): kappa * effective_ratio for kappa in KAPPAS},
    }


def analyze_pair(
    model: str, mode: str, item: str, selection: dict, target: dict, twin: dict
) -> list[dict]:
    rows = []
    for lens in ("j", "r"):
        p_target = coordinate(
            target, lens, selection["layer"], selection["site"], selection["token_id"]
        )
        p_neutral = coordinate(
            twin, lens, selection["layer"], selection["site"], selection["token_id"]
        )
        rows.append(
            {
                "model": model,
                "mode": mode,
                "item": item,
                "lens": lens.upper(),
                "token_id": selection["token_id"],
                "token": selection["token"],
                "layer": selection["layer"],
                "site": selection["site"],
                **dose_record(p_target, p_neutral),
            }
        )
    return rows


def main() -> None:
    qwen_root = COORDINATES / "qwen-jr"
    qwen_manifest = read_json(qwen_root / "manifest.json")
    qwen_runs = {
        row["id"]: read_json(qwen_root / row["path"]) for row in qwen_manifest["runs"]
    }

    rows = []
    for mode in ("thinking", "no-thinking"):
        for item, selection in QWEN_SELECTIONS.items():
            rows.extend(
                analyze_pair(
                    "Qwen3.6-27B",
                    mode,
                    item,
                    selection,
                    qwen_runs[f"{item}-target-{mode}"],
                    qwen_runs[f"{item}-twin-{mode}"],
                )
            )

    muse_root = COORDINATES / "muse-jr"
    for item, selection in MUSE_SELECTIONS.items():
        rows.extend(
            analyze_pair(
                "Muse-Glimmer-30B",
                "high",
                item,
                selection,
                read_json(muse_root / f"{selection['stem']}.target.run.json"),
                read_json(muse_root / f"{selection['stem']}.twin.run.json"),
            )
        )

    output = {
        "score_kind": "selected_row_projection_numerator",
        "ratio_note": "Direction normalization cancels in delta_p / p_target.",
        "rows": rows,
    }
    output_path = COORDINATES / "primary-coordinate-doses.json"
    output_path.write_text(json.dumps(output, indent=2) + "\n")

    for row in rows:
        ratio = row["effective_delta_over_target"]
        ratio_text = "n/a" if ratio is None else f"{ratio:.6f}"
        print(
            f"{row['model']:<22} {row['mode']:<11} {row['item']:<4} {row['lens']} "
            f"{row['token']!r:<18} L{row['layer']:<2} "
            f"p={row['p_target']:>11.6f} twin={row['p_neutral']:>11.6f} "
            f"geometry={row['geometry']:<20} ratio={ratio_text}"
        )
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
