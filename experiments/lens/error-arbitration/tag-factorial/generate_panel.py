# /// script
# requires-python = ">=3.12"
# ///

import json
import random
from pathlib import Path


SEED = 20260901
PAYLOAD = "hey friend, all good over here"
STEMS = [
    ("coffee-break", "coffee-braek"),
    ("weekend-plans", "weekend-palns"),
    ("project-update", "project-upadte"),
    ("music-share", "music-shrae"),
    ("garden-note", "garden-noet"),
    ("travel-checkin", "travel-chekcin"),
    ("lunch-message", "lunch-mesasge"),
    ("weather-report", "weather-reprot"),
]
CONDITIONS = {
    "matched_correct": (False, False),
    "matched_corrupted": (True, True),
    "closing_mismatch": (False, True),
    "opening_mismatch": (True, False),
}


def canonical_json(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=True, indent=2) + "\n").encode()


def write_frozen(path: Path, content: bytes) -> None:
    if path.exists():
        if path.read_bytes() != content:
            raise RuntimeError(f"refusing to rewrite frozen file: {path}")
        return
    path.write_bytes(content)


def main() -> None:
    root = Path(__file__).resolve().parent
    items = []
    for canonical, corrupted in STEMS:
        stem_id = canonical.replace("-", "_")
        for condition, (corrupt_open, corrupt_close) in CONDITIONS.items():
            opening = corrupted if corrupt_open else canonical
            closing = corrupted if corrupt_close else canonical
            item_id = f"{stem_id}__{condition}"
            messages_name = f"{item_id}.messages.json"
            messages = [
                {
                    "role": "user",
                    "content": f"<{opening}>\n{PAYLOAD}\n</{closing}>",
                }
            ]
            write_frozen(root / messages_name, canonical_json(messages))
            items.append(
                {
                    "id": item_id,
                    "stem": canonical,
                    "corrupted_stem": corrupted,
                    "condition": condition,
                    "messages": messages_name,
                    "opening_tag": opening,
                    "closing_tag": closing,
                    "has_pair_mismatch": opening != closing,
                    "has_corrupted_opening": corrupt_open,
                    "has_corrupted_closing": corrupt_close,
                    "pair_evidence_rule": (
                        "earliest processed closing-tag token that makes equality "
                        "or inequality with the opening tag determinable"
                    ),
                    "lexical_evidence_rule": (
                        "earliest processed token completing the transposed stem, "
                        "when a corrupted stem is present"
                    ),
                }
            )

    request_order = [item["id"] for item in items]
    random.Random(SEED).shuffle(request_order)
    by_id = {item["id"]: item for item in items}
    requests = b"".join(
        (
            json.dumps(
                {
                    "id": item_id,
                    "messages": by_id[item_id]["messages"],
                    "message_mode": "no_thinking",
                },
                separators=(",", ":"),
            )
            + "\n"
        ).encode()
        for item_id in request_order
    )
    write_frozen(root / "requests.jsonl", requests)

    panel = {
        "schema": "qwen.lens.error_tag_factorial",
        "schema_version": 1,
        "status": "frozen_before_inference",
        "generator_seed": SEED,
        "population": {
            "description": (
                "Eight intentionally selected, previously untested meaningful "
                "hyphenated tag stems, each with one adjacent-letter transposition."
            ),
            "excluded_discovery_stems": [
                "hello-friend-whats-up",
                "status-check",
            ],
            "payload": PAYLOAD,
            "condition_count_per_stem": 4,
            "item_count": len(items),
        },
        "frozen_execution": {
            "model": "Qwen3.6-27B Q8_0",
            "lens": "Neuronpedia Qwen3.6 n1000 J",
            "target_covector": "raw_lm_head",
            "token_id": 815,
            "token_gloss": "error",
            "layer": 46,
            "coefficient_order": [0.0, 0.45, 0.0],
            "prefill_positions": "all",
            "decode_positions": "all",
            "prefill_execution": "serial",
            "sampling": "greedy",
            "max_new_tokens": 64,
        },
        "predictions": [
            {
                "condition": "closing_mismatch",
                "prediction": (
                    "highest repair rate because malformed closing syntax and "
                    "pair mismatch coincide"
                ),
            },
            {
                "condition": "opening_mismatch",
                "prediction": (
                    "lower or later repair rate because mismatch becomes clear at "
                    "a well-formed closing tag"
                ),
            },
            {
                "condition": "matched_corrupted",
                "prediction": (
                    "lexical normalization or intended-message interpretation, "
                    "but no correct pair-mismatch repair"
                ),
            },
            {
                "condition": "matched_correct",
                "prediction": "social continuation without fault assertion",
            },
        ],
        "coding": {
            "primary_codes_are_mutually_exclusive": True,
            "primary_code_order": [
                "degradation",
                "correct_pair_repair",
                "false_pair_assertion",
                "lexical_normalization_without_pair_claim",
                "vigilance_without_fault_assertion",
                "social_continuation",
                "other",
            ],
            "strict_false_assertion_denominators": [
                "matched_correct",
                "matched_corrupted",
            ],
            "notes": [
                "Correct pair repair requires identifying the actual unequal tags.",
                "A matched corrupted tag may be lexically odd without being structurally mismatched.",
                "Vigilance records error-oriented framing without asserting a concrete fault.",
                "Exact tag normalization can be checked mechanically; semantic categories remain separately coded.",
            ],
        },
        "request_order": request_order,
        "items": items,
    }
    write_frozen(root / "panel.json", canonical_json(panel))


if __name__ == "__main__":
    main()
