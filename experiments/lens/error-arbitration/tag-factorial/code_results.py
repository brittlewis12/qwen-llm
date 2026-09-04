# /// script
# requires-python = ">=3.12"
# ///

import json
from collections import Counter, defaultdict
from pathlib import Path


# Unblinded single-rater coding. The objective flags distinguish lexical stem
# normalization from an explicit claim that the opening and closing tags differ.
CODES = {
    "coffee_break__matched_correct": (
        "social_continuation",
        "social_continuation",
        0,
        0,
        0,
        0,
        "",
    ),
    "coffee_break__matched_corrupted": (
        "social_continuation",
        "social_continuation",
        0,
        0,
        0,
        0,
        "",
    ),
    "coffee_break__closing_mismatch": (
        "social_continuation",
        "social_continuation",
        0,
        0,
        0,
        0,
        "",
    ),
    "coffee_break__opening_mismatch": (
        "social_continuation",
        "social_continuation",
        0,
        0,
        0,
        0,
        "",
    ),
    "weekend_plans__matched_correct": (
        "social_continuation",
        "social_continuation",
        0,
        0,
        0,
        0,
        "",
    ),
    "weekend_plans__matched_corrupted": (
        "lexical_normalization_without_pair_claim",
        "lexical_normalization_without_pair_claim",
        1,
        1,
        0,
        0,
        "Both arms normalize palns to plans.",
    ),
    "weekend_plans__closing_mismatch": (
        "social_continuation",
        "social_continuation",
        0,
        0,
        0,
        0,
        "",
    ),
    "weekend_plans__opening_mismatch": (
        "lexical_normalization_without_pair_claim",
        "lexical_normalization_without_pair_claim",
        1,
        1,
        0,
        0,
        "Both arms normalize the opening lexeme without asserting pair inequality.",
    ),
    "project_update__matched_correct": (
        "social_continuation",
        "social_continuation",
        0,
        0,
        0,
        0,
        "",
    ),
    "project_update__matched_corrupted": (
        "social_continuation",
        "social_continuation",
        0,
        0,
        0,
        0,
        "",
    ),
    "project_update__closing_mismatch": (
        "social_continuation",
        "correct_pair_repair",
        0,
        1,
        0,
        1,
        "Active arm explicitly identifies and repairs the closing tag.",
    ),
    "project_update__opening_mismatch": (
        "social_continuation",
        "lexical_normalization_without_pair_claim",
        0,
        1,
        0,
        0,
        "Active arm normalizes the opening lexeme but does not state that the pair differs.",
    ),
    "music_share__matched_correct": (
        "social_continuation",
        "social_continuation",
        0,
        0,
        0,
        0,
        "",
    ),
    "music_share__matched_corrupted": (
        "social_continuation",
        "lexical_normalization_without_pair_claim",
        0,
        1,
        0,
        0,
        "Active arm proposes music-share without claiming unequal tags.",
    ),
    "music_share__closing_mismatch": (
        "vigilance_without_fault_assertion",
        "lexical_normalization_without_pair_claim",
        0,
        1,
        0,
        0,
        "Baseline notices the corrupted lexeme; active arm supplies the canonical lexeme without naming pair inequality.",
    ),
    "music_share__opening_mismatch": (
        "social_continuation",
        "lexical_normalization_without_pair_claim",
        0,
        1,
        0,
        0,
        "Active arm supplies the canonical lexeme without naming pair inequality.",
    ),
    "garden_note__matched_correct": (
        "social_continuation",
        "social_continuation",
        0,
        0,
        0,
        0,
        "",
    ),
    "garden_note__matched_corrupted": (
        "social_continuation",
        "lexical_normalization_without_pair_claim",
        0,
        1,
        0,
        0,
        "Active arm proposes garden note without claiming unequal tags.",
    ),
    "garden_note__closing_mismatch": (
        "social_continuation",
        "correct_pair_repair",
        0,
        1,
        0,
        1,
        "Active arm explicitly identifies and repairs the closing tag.",
    ),
    "garden_note__opening_mismatch": (
        "social_continuation",
        "lexical_normalization_without_pair_claim",
        0,
        1,
        0,
        0,
        "Active arm normalizes the opening lexeme without naming pair inequality.",
    ),
    "travel_checkin__matched_correct": (
        "social_continuation",
        "vigilance_without_fault_assertion",
        0,
        0,
        0,
        0,
        "Active arm introduces generic error-checking language without asserting a concrete fault.",
    ),
    "travel_checkin__matched_corrupted": (
        "social_continuation",
        "vigilance_without_fault_assertion",
        0,
        0,
        0,
        0,
        "Active arm introduces generic mistake language without asserting a concrete fault.",
    ),
    "travel_checkin__closing_mismatch": (
        "social_continuation",
        "vigilance_without_fault_assertion",
        0,
        0,
        0,
        0,
        "Active arm reframes the message as a travel error but does not identify the tag anomaly.",
    ),
    "travel_checkin__opening_mismatch": (
        "social_continuation",
        "vigilance_without_fault_assertion",
        0,
        0,
        0,
        0,
        "Active arm reframes the message as a travel error but does not identify the tag anomaly.",
    ),
    "lunch_message__matched_correct": (
        "social_continuation",
        "social_continuation",
        0,
        0,
        0,
        0,
        "",
    ),
    "lunch_message__matched_corrupted": (
        "social_continuation",
        "social_continuation",
        0,
        0,
        0,
        0,
        "",
    ),
    "lunch_message__closing_mismatch": (
        "social_continuation",
        "lexical_normalization_without_pair_claim",
        0,
        1,
        0,
        0,
        "Active arm normalizes the corrupted stem but renders both examples as opening tags.",
    ),
    "lunch_message__opening_mismatch": (
        "social_continuation",
        "social_continuation",
        0,
        0,
        0,
        0,
        "",
    ),
    "weather_report__matched_correct": (
        "social_continuation",
        "other",
        0,
        0,
        0,
        0,
        "Active arm asserts a content/tag semantic mismatch, not a tag-pair mismatch.",
    ),
    "weather_report__matched_corrupted": (
        "other",
        "lexical_normalization_without_pair_claim",
        0,
        1,
        0,
        0,
        "Baseline asserts a content/tag mismatch; active arm explicitly normalizes reprot to report.",
    ),
    "weather_report__closing_mismatch": (
        "correct_pair_repair",
        "correct_pair_repair",
        1,
        1,
        1,
        1,
        "Both arms explicitly identify the malformed closing tag.",
    ),
    "weather_report__opening_mismatch": (
        "other",
        "lexical_normalization_without_pair_claim",
        0,
        1,
        0,
        0,
        "Baseline asserts a content/tag mismatch; active arm normalizes the opening lexeme.",
    ),
}


def main() -> None:
    root = Path(__file__).resolve().parent
    responses = json.loads((root / "response-data.json").read_text())
    rows = responses["rows"]
    if set(CODES) != {row["id"] for row in rows}:
        raise RuntimeError("coding IDs do not exactly match response IDs")
    coded = []
    for row in rows:
        (
            baseline_primary,
            active_primary,
            baseline_normalizes,
            active_normalizes,
            baseline_pair_repair,
            active_pair_repair,
            note,
        ) = CODES[row["id"]]
        coded.append(
            {
                "id": row["id"],
                "stem": row["stem"],
                "condition": row["condition"],
                "baseline_primary": baseline_primary,
                "active_primary": active_primary,
                "baseline_normalizes_corrupted_stem": bool(baseline_normalizes),
                "active_normalizes_corrupted_stem": bool(active_normalizes),
                "baseline_explicit_pair_repair": bool(baseline_pair_repair),
                "active_explicit_pair_repair": bool(active_pair_repair),
                "note": note,
            }
        )

    summary = {}
    for condition in (
        "matched_correct",
        "matched_corrupted",
        "closing_mismatch",
        "opening_mismatch",
    ):
        condition_rows = [row for row in coded if row["condition"] == condition]
        baseline_codes = Counter(row["baseline_primary"] for row in condition_rows)
        active_codes = Counter(row["active_primary"] for row in condition_rows)
        summary[condition] = {
            "n": len(condition_rows),
            "baseline_primary": dict(sorted(baseline_codes.items())),
            "active_primary": dict(sorted(active_codes.items())),
            "baseline_normalizes_corrupted_stem": sum(
                row["baseline_normalizes_corrupted_stem"] for row in condition_rows
            ),
            "active_normalizes_corrupted_stem": sum(
                row["active_normalizes_corrupted_stem"] for row in condition_rows
            ),
            "baseline_explicit_pair_repair": sum(
                row["baseline_explicit_pair_repair"] for row in condition_rows
            ),
            "active_explicit_pair_repair": sum(
                row["active_explicit_pair_repair"] for row in condition_rows
            ),
        }
    output = {
        "schema": "qwen.lens.error_tag_factorial.coding",
        "schema_version": 1,
        "coder": "unblinded_single_rater",
        "coding_status": "complete_provisional_pending_independent_review",
        "definitions": {
            "normalizes_corrupted_stem": (
                "Explicitly supplies or proposes the intended canonical stem; "
                "mere topic interpretation is insufficient."
            ),
            "explicit_pair_repair": (
                "States or unambiguously renders that the opening and closing tags "
                "differ and repairs the actual unequal pair."
            ),
        },
        "summary": summary,
        "rows": coded,
    }
    (root / "CODING.json").write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
