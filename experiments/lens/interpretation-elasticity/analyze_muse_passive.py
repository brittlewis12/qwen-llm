import json

from analyze_qwen_passive import ARTIFACT_ROOT, analyze_pair, ascii_word


PAIRS = [
    ("p1", "p1-client-delay.target", "p1-client-delay.twin"),
    ("p2", "p2-deadpan-review.target", "p2-deadpan-review.twin"),
    ("p3a", "p3a-correct-challenged.target", "p3a-correct-challenged.twin"),
    ("p3b", "p3b-incorrect-challenged.target", "p3b-incorrect-challenged.twin"),
    ("p4", "p4-wrong-test.target", "p4-wrong-test.twin"),
]


def load(method: str, stem: str) -> dict:
    path = ARTIFACT_ROOT / f"muse-{method}" / f"{stem}.trace.json"
    return json.loads(path.read_text())


def main() -> None:
    output = {}
    for item, target, twin in PAIRS:
        output[item] = analyze_pair(
            load("j", target),
            load("j", twin),
            load("r", target),
            load("r", twin),
        )

    output_path = ARTIFACT_ROOT / "muse-passive-candidates.json"
    output_path.write_text(json.dumps(output, indent=2) + "\n")

    for item, anchor_rows in output.items():
        print(f"\n## {item}")
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
