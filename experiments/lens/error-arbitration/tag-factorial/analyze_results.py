# /// script
# requires-python = ">=3.12"
# ///

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


FAULT_RE = re.compile(
    r"\b(typo|error|mistake|mismatch|malformed|incorrect|wrong)\b", re.I
)
TAG_RE = re.compile(r"\b(tag|xml|markup|opening|closing)\b", re.I)


def load_json(path: Path) -> object:
    return json.loads(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cohort",
        type=Path,
        default=Path("target/qwen36-q8-error-tag-factorial-v1"),
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    panel = load_json(root / "panel.json")
    cohort = load_json(args.cohort / "manifest.json")
    items = {item["id"]: item for item in panel["items"]}
    rows = []
    for child in cohort["sweeps"]:
        item = items[child["id"]]
        sweep = args.cohort / child["path"]
        child_manifest = load_json(sweep / "manifest.json")
        if child_manifest["coefficients"] != [0.0, 0.45, 0.0]:
            raise RuntimeError(f"unexpected coefficients for {child['id']}")
        arm_paths = [sweep / arm["artifact"] for arm in child_manifest["arms"]]
        arm_bytes = [path.read_bytes() for path in arm_paths]
        if arm_bytes[0] != arm_bytes[2]:
            raise RuntimeError(f"duplicate zero mismatch for {child['id']}")
        runs = [json.loads(content) for content in arm_bytes]
        baseline = runs[0]["decoded_text"]
        active = runs[1]["decoded_text"]
        prompt = load_json(Path(child["messages_path"]))[0]["content"]
        lower = active.lower()
        rows.append(
            {
                **item,
                "request_index": child["index"],
                "prompt": prompt,
                "baseline": baseline,
                "active": active,
                "baseline_equals_active": (
                    runs[0]["generated_token_ids"] == runs[1]["generated_token_ids"]
                ),
                "duplicate_zero_byte_identical": True,
                "active_mentions_fault": bool(FAULT_RE.search(active)),
                "active_mentions_tag": bool(TAG_RE.search(active)),
                "active_contains_canonical_stem": item["stem"].lower() in lower,
                "active_contains_corrupted_stem": item["corrupted_stem"].lower()
                in lower,
                "active_operation_applications": len(runs[1]["operation_applications"]),
                "primary_code": None,
                "coding_note": None,
            }
        )

    output = {
        "schema": "qwen.lens.error_tag_factorial.responses",
        "schema_version": 1,
        "cohort_manifest": str((args.cohort / "manifest.json").resolve()),
        "producer": cohort["producer"],
        "row_count": len(rows),
        "rows": rows,
    }
    (root / "response-data.json").write_text(json.dumps(output, indent=2) + "\n")

    summaries = defaultdict(Counter)
    for row in rows:
        for key in (
            "baseline_equals_active",
            "active_mentions_fault",
            "active_mentions_tag",
            "active_contains_canonical_stem",
            "active_contains_corrupted_stem",
        ):
            summaries[row["condition"]][key] += bool(row[key])

    lines = [
        "# Unblinded Response Ledger",
        "",
        "Generated mechanically from the immutable cohort. Primary semantic codes are",
        "intentionally blank until responses are reviewed.",
        "",
        "## Mechanical Summary",
        "",
        "| Condition | N | Same output | Fault words | Tag words | Canonical stem | Corrupted stem |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for condition in panel["coding"]["strict_false_assertion_denominators"] + [
        "closing_mismatch",
        "opening_mismatch",
    ]:
        summary = summaries[condition]
        count = sum(row["condition"] == condition for row in rows)
        lines.append(
            f"| {condition} | {count} | {summary['baseline_equals_active']} | "
            f"{summary['active_mentions_fault']} | {summary['active_mentions_tag']} | "
            f"{summary['active_contains_canonical_stem']} | "
            f"{summary['active_contains_corrupted_stem']} |"
        )
    for row in rows:
        lines.extend(
            [
                "",
                f"## {row['request_index']:02d} - {row['id']}",
                "",
                "```text",
                row["prompt"],
                "```",
                "",
                "Baseline:",
                "",
                "```text",
                row["baseline"],
                "```",
                "",
                "Error +0.45:",
                "",
                "```text",
                row["active"],
                "```",
            ]
        )
    (root / "RESPONSES.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
