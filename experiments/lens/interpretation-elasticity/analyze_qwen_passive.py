import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).parent
ARTIFACT_ROOT = ROOT / "artifacts" / "passive"

PAIRS = [
    ("p1", "p1-target", "p1-twin"),
    ("p2", "p2-target", "p2-twin"),
    ("p3a", "p3a-target", "p3a-twin"),
    ("p3b", "p3b-target", "p3b-twin"),
    ("p4", "p4-target", "p4-twin"),
]


def load_cohort(name: str) -> dict[str, dict]:
    directory = ARTIFACT_ROOT / name
    manifest = json.loads((directory / "manifest.json").read_text())
    return {
        row["request_id"]: json.loads((directory / row["path"]).read_text())
        for row in manifest["artifacts"]
    }


def anchors(trace: dict) -> dict[str, int]:
    spans = trace["rendering"]["spans"]
    result = {"prefill_last": len(trace["input_token_ids"]) - 1}

    user_content = [
        span
        for span in spans
        if span["kind"] == "message_content" and span.get("role") == "user"
    ]
    user_end = [
        span
        for span in spans
        if span["kind"] == "message_end_marker" and span.get("role") == "user"
    ]
    if user_content:
        result["user_content_last"] = user_content[-1]["token_end"] - 1
    if user_end:
        result["user_end"] = user_end[-1]["token_start"]

    structural = {
        "generated_assistant_start_marker": "assistant_start",
        "generated_assistant_role": "assistant_role",
        "thinking_channel_start_marker": "thinking_start",
        "thinking_channel_end_marker": "thinking_end",
    }
    for span in spans:
        label = structural.get(span["kind"])
        if label is not None:
            result[label] = span["token_start"]
    return result


def cell_index(trace: dict) -> dict[tuple[int, int], dict]:
    return {
        (cell["source_layer"], cell["source_position"]): cell for cell in trace["cells"]
    }


def score_index(cell: dict) -> dict[int, dict]:
    return {score["token_id"]: score for score in cell["top_k"]}


def reciprocal_rank(score: dict | None) -> float:
    return 0.0 if score is None else 1.0 / (score["rank"] + 1)


def longest_run(layers: list[int]) -> int:
    best = current = 0
    previous = None
    for layer in sorted(set(layers)):
        current = current + 1 if previous is not None and layer == previous + 1 else 1
        best = max(best, current)
        previous = layer
    return best


def ascii_word(display: str) -> bool:
    word = display.strip().replace("-", "")
    return len(word) >= 3 and word.isascii() and word.isalpha()


def analyze_pair(
    target_j: dict,
    twin_j: dict,
    target_r: dict,
    twin_r: dict,
) -> dict[str, list[dict]]:
    traces = [target_j, twin_j, target_r, twin_r]
    trace_anchors = [anchors(trace) for trace in traces]
    common_anchors = sorted(set.intersection(*(set(value) for value in trace_anchors)))
    indices = [cell_index(trace) for trace in traces]
    layers = target_j["selected_layers"]
    result = {}

    for anchor in common_anchors:
        positions = [value[anchor] for value in trace_anchors]
        by_token = defaultdict(list)
        for layer in layers:
            cells = [
                index[(layer, position)] for index, position in zip(indices, positions)
            ]
            target_j_scores, twin_j_scores, target_r_scores, twin_r_scores = [
                score_index(cell) for cell in cells
            ]
            for token_id in set(target_j_scores) & set(target_r_scores):
                tj = target_j_scores[token_id]
                tr = target_r_scores[token_id]
                nj = twin_j_scores.get(token_id)
                nr = twin_r_scores.get(token_id)
                lift_j = reciprocal_rank(tj) - reciprocal_rank(nj)
                lift_r = reciprocal_rank(tr) - reciprocal_rank(nr)
                if lift_j <= 0 or lift_r <= 0:
                    continue
                by_token[token_id].append(
                    {
                        "layer": layer,
                        "display": tj["token_display_lossy"],
                        "rank_j": tj["rank"],
                        "rank_r": tr["rank"],
                        "twin_rank_j": None if nj is None else nj["rank"],
                        "twin_rank_r": None if nr is None else nr["rank"],
                        "lift_j": lift_j,
                        "lift_r": lift_r,
                        "logit_j": tj["logit"],
                        "logit_r": tr["logit"],
                    }
                )

        summaries = []
        for token_id, rows in by_token.items():
            best = max(
                rows,
                key=lambda row: (
                    min(row["lift_j"], row["lift_r"]),
                    -max(row["rank_j"], row["rank_r"]),
                    -row["layer"],
                ),
            )
            summaries.append(
                {
                    "token_id": token_id,
                    "display": best["display"],
                    "positive_lift_layers": len(rows),
                    "longest_run": longest_run([row["layer"] for row in rows]),
                    "best": best,
                    "rows": rows,
                }
            )
        summaries.sort(
            key=lambda row: (
                row["longest_run"],
                row["positive_lift_layers"],
                min(row["best"]["lift_j"], row["best"]["lift_r"]),
                -max(row["best"]["rank_j"], row["best"]["rank_r"]),
            ),
            reverse=True,
        )
        result[anchor] = summaries
    return result


def main() -> None:
    j = load_cohort("qwen-j")
    r = load_cohort("qwen-r")
    output = {}
    for mode in ["thinking", "no-thinking"]:
        suffix = "thinking" if mode == "thinking" else "no-thinking"
        for item, target, twin in PAIRS:
            key = f"{item}-{mode}"
            output[key] = analyze_pair(
                j[f"{target}-{suffix}"],
                j[f"{twin}-{suffix}"],
                r[f"{target}-{suffix}"],
                r[f"{twin}-{suffix}"],
            )

    output_path = ARTIFACT_ROOT / "qwen-passive-candidates.json"
    output_path.write_text(json.dumps(output, indent=2) + "\n")

    for key, anchor_rows in output.items():
        print(f"\n## {key}")
        for anchor in [
            "user_content_last",
            "user_end",
            "assistant_start",
            "assistant_role",
            "thinking_start",
            "thinking_end",
            "prefill_last",
        ]:
            rows = [
                row for row in anchor_rows.get(anchor, []) if ascii_word(row["display"])
            ]
            if not rows:
                continue
            print(f"### {anchor}")
            for row in rows[:8]:
                best = row["best"]
                print(
                    f"{row['token_id']:>6} {row['display']!r:<20} "
                    f"run={row['longest_run']:>2} layers={row['positive_lift_layers']:>2} "
                    f"best=L{best['layer']} J{best['rank_j']} R{best['rank_r']} "
                    f"twin=J{best['twin_rank_j']} R{best['twin_rank_r']}"
                )
    print(f"\nWrote {output_path}")


if __name__ == "__main__":
    main()
