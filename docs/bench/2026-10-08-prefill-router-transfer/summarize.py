# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Summarize router-tail evidence without equating execution with promotion."""

import json
import sys
from pathlib import Path

for name in sys.argv[1:]:
    events = [json.loads(line) for line in Path(name).read_text().splitlines()]
    comparisons = [e for e in events if e["event"] == "comparison"]
    witnesses = [e for e in events if e["event"] == "dispatch_witness"]
    endpoints = {e["label"]: e["state"] for e in events if e["event"] == "endpoint"}
    state_diffs = []
    for event in comparisons:
        if event["comparison"]["state_digests_equal"]:
            continue
        suffix = f"/continuation{event['step'] - 1}" if event["step"] else ""
        a = endpoints[event["reference"] + suffix]
        b = endpoints[event["candidate"] + suffix]
        state_diffs.append(
            {
                "reference": event["reference"],
                "candidate": event["candidate"],
                "step": event["step"],
                "changed_keys": [k for k in a if a[k] != b[k]],
                "changed_tensors": [
                    x["index"] for x, y in zip(a["tensors"], b["tensors"]) if x != y
                ],
            }
        )
    print(
        json.dumps(
            {
                "file": name,
                "abba": [e for e in events if e["event"] == "abba"],
                "qualification": [
                    e for e in events if e["event"] == "qualification_summary"
                ],
                "complete": [e for e in events if e["event"] == "complete"],
                "comparison_count": len(comparisons),
                "all_finite": all(e["comparison"]["finite"] for e in comparisons),
                "all_metadata_equal": all(e["metadata_equal"] for e in comparisons),
                "all_choices_equal": all(e["greedy_choice_equal"] for e in comparisons),
                "all_logits_equal": all(
                    e["comparison"]["logits_bits_equal"] for e in comparisons
                ),
                "all_state_equal": all(
                    e["comparison"]["state_digests_equal"] for e in comparisons
                ),
                "max_kl": max(
                    # Successful execution alone is not a numerical qualification.
                    (
                        e["logit_kl_reference_candidate"]
                        for e in comparisons
                        if e["logit_kl_reference_candidate"] is not None
                    ),
                    default=None,
                ),
                "witness_count": len(witnesses),
                "state_difference_count": len(state_diffs),
                "state_difference_examples": state_diffs[:8],
                "all_witnesses_valid": bool(witnesses)
                and all(e["valid"] for e in witnesses),
            },
            indent=2,
        )
    )
