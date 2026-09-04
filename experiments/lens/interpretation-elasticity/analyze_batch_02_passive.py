import json
from pathlib import Path

from analyze_qwen_passive import analyze_pair, ascii_word


ROOT = Path(__file__).parent
ARTIFACT_ROOT = ROOT / "artifacts" / "passive"
PAIRS = [f"p{index}" for index in range(5, 12)]


def load_cohort(name: str) -> dict[str, dict]:
    directory = ARTIFACT_ROOT / name
    manifest = json.loads((directory / "manifest.json").read_text())
    return {
        row["request_id"]: json.loads((directory / row["path"]).read_text())
        for row in manifest["artifacts"]
    }


def main() -> None:
    j = load_cohort("qwen-j-batch-02")
    r = load_cohort("qwen-r-batch-02")
    output = {}
    for mode in ["thinking", "no-thinking"]:
        for item in PAIRS:
            key = f"{item}-{mode}"
            suffix = "thinking" if mode == "thinking" else "no-thinking"
            output[key] = analyze_pair(
                j[f"{item}-target-{suffix}"],
                j[f"{item}-twin-{suffix}"],
                r[f"{item}-target-{suffix}"],
                r[f"{item}-twin-{suffix}"],
            )

    output_path = ARTIFACT_ROOT / "qwen-batch-02-passive-candidates.json"
    output_path.write_text(json.dumps(output, indent=2) + "\n")

    anchor_order = [
        "user_content_last",
        "user_end",
        "assistant_start",
        "assistant_role",
        "thinking_start",
        "thinking_end",
        "prefill_last",
    ]
    for key, anchor_rows in output.items():
        print(f"\n## {key}")
        for anchor in anchor_order:
            rows = [
                row for row in anchor_rows.get(anchor, []) if ascii_word(row["display"])
            ]
            if not rows:
                continue
            print(f"### {anchor}")
            for row in rows[:10]:
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
