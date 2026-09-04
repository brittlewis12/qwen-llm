import json
from pathlib import Path

from analyze_primary_coordinates import dose_record


ROOT = Path(__file__).parent
COORDINATES = ROOT / "artifacts" / "coordinates"
TOKEN_IDS = {"Qwen3.6-27B": 10179, "Muse-Glimmer-30B": 158448}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def score(run: dict, lens: str, layer: int, token_id: int) -> float:
    cells = [
        cell
        for cell in run["live_readouts"]
        if cell["lens"].lower() == lens.lower() and cell["source_layer"] == layer
    ]
    if len(cells) != 1:
        raise ValueError(f"expected one {lens} L{layer} readout")
    scores = [row["score"] for row in cells[0]["scores"] if row["token_id"] == token_id]
    if len(scores) != 1:
        raise ValueError(f"expected one score for token {token_id}")
    return scores[0]


def append_pair(
    rows: list[dict],
    model: str,
    mode: str,
    item: str,
    layers: list[int],
    target: dict,
    twin: dict,
) -> None:
    token_id = TOKEN_IDS[model]
    for layer in layers:
        for lens in ("j", "r"):
            rows.append(
                {
                    "model": model,
                    "mode": mode,
                    "item": item,
                    "lens": lens.upper(),
                    "token_id": token_id,
                    "layer": layer,
                    **dose_record(
                        score(target, lens, layer, token_id),
                        score(twin, lens, layer, token_id),
                    ),
                }
            )


def main() -> None:
    rows = []

    qwen_root = COORDINATES / "qwen-p3-depth"
    manifest = read_json(qwen_root / "manifest.json")
    qwen = {row["id"]: read_json(qwen_root / row["path"]) for row in manifest["runs"]}
    for mode in ("thinking", "no-thinking"):
        for item in ("p3a", "p3b"):
            append_pair(
                rows,
                "Qwen3.6-27B",
                mode,
                item,
                [22, 32, 42, 52, 56],
                qwen[f"{item}-target-{mode}"],
                qwen[f"{item}-twin-{mode}"],
            )

    muse_root = COORDINATES / "muse-p3-depth"
    stems = {"p3a": "p3a-correct-challenged", "p3b": "p3b-incorrect-challenged"}
    for item, stem in stems.items():
        append_pair(
            rows,
            "Muse-Glimmer-30B",
            "high",
            item,
            [27, 35, 43, 49],
            read_json(muse_root / f"{stem}.target.run.json"),
            read_json(muse_root / f"{stem}.twin.run.json"),
        )

    output_path = COORDINATES / "p3-depth-coordinate-doses.json"
    output_path.write_text(json.dumps({"rows": rows}, indent=2) + "\n")
    for row in rows:
        ratio = row["effective_delta_over_target"]
        ratio_text = "n/a" if ratio is None else f"{ratio:.6f}"
        print(
            f"{row['model']:<22} {row['mode']:<11} {row['item']} {row['lens']} "
            f"L{row['layer']:<2} p={row['p_target']:>10.5f} twin={row['p_neutral']:>10.5f} "
            f"{row['geometry']:<20} lambda1={ratio_text}"
        )
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
