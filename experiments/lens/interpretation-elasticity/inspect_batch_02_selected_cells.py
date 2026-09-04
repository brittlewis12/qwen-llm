import json
from pathlib import Path

from analyze_batch_02_coordinates import ROOT, SELECTIONS


PASSIVE = ROOT / "artifacts" / "passive"


def load_cohort(name: str) -> dict[str, dict]:
    directory = PASSIVE / name
    manifest = json.loads((directory / "manifest.json").read_text())
    return {
        row["request_id"]: json.loads((directory / row["path"]).read_text())
        for row in manifest["artifacts"]
    }


def cell(trace: dict, layer: int, position: int) -> dict:
    return next(
        row
        for row in trace["cells"]
        if row["source_layer"] == layer and row["source_position"] == position
    )


def index(row: dict) -> dict[int, dict]:
    return {score["token_id"]: score for score in row["top_k"]}


def main() -> None:
    j = load_cohort("qwen-j-batch-02")
    r = load_cohort("qwen-r-batch-02")
    for item, selection in SELECTIONS.items():
        print(
            f"\n## {item} L{selection['layer']} "
            f"target p{selection['target_position']} twin p{selection['twin_position']}"
        )
        traces = []
        for cohort in (j, r):
            traces.append(
                index(
                    cell(
                        cohort[f"{item}-target-no-thinking"],
                        selection["layer"],
                        selection["target_position"],
                    )
                )
            )
            traces.append(
                index(
                    cell(
                        cohort[f"{item}-twin-no-thinking"],
                        selection["layer"],
                        selection["twin_position"],
                    )
                )
            )
        target_j, twin_j, target_r, twin_r = traces
        shared = set(target_j) & set(target_r)
        rows = []
        for token_id in shared:
            tj = target_j[token_id]
            tr = target_r[token_id]
            nj = twin_j.get(token_id)
            nr = twin_r.get(token_id)
            rows.append(
                (
                    max(tj["rank"], tr["rank"]),
                    token_id,
                    tj["token_display_lossy"],
                    tj["rank"],
                    None if nj is None else nj["rank"],
                    tr["rank"],
                    None if nr is None else nr["rank"],
                )
            )
        for _, token_id, display, tj, nj, tr, nr in sorted(rows)[:25]:
            print(f"{token_id:>6} {display!r:<20} J={tj}/{nj} R={tr}/{nr}")


if __name__ == "__main__":
    main()
