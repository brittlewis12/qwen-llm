#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Blind one post-hoc four-case condition against retained Maya; never rescore the pilot."""

import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import secrets

spec = importlib.util.spec_from_file_location(
    "_pilot_blind", Path(__file__).with_name("blind-v41-pilot.py")
)
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)
FROZEN = pilot.FROZEN
require = pilot.require
LABELS = pilot.LABELS
SELECTED_IDS = (
    "stream-lookahead",
    "shared-load-cancellation",
    "generation-fenced-cache",
    "incremental-utf8-lines",
)
PROMPT = pilot.PROMPT.replace("exactly 16 records", "exactly 8 records").replace(
    "eight cases, maximum 32", "four cases, maximum 16"
)


def validate_rows(summary, sha, expected):
    require(
        isinstance(summary, dict) and summary.get("status") == "completed",
        "run not completed",
    )
    require(summary.get("cases_sha256") == sha, "frozen cases hash mismatch")
    rows = summary.get("results")
    require(isinstance(rows, list) and len(rows) == len(expected), "wrong result count")
    for row, (ordinal, case_id) in zip(rows, expected):
        require(
            isinstance(row, dict)
            and row.get("id") == case_id
            and type(row.get("ordinal")) is int
            and row["ordinal"] == ordinal,
            "case order/ordinal mismatch",
        )
        require(
            row.get("status") in ("completed", "incomplete", "model_output_error")
            and isinstance(row.get("answer"), str),
            "invalid answer/status",
        )
    return rows


def build_packet(case_bytes, summaries, *, seed):
    """Accept [one diagnostic summary, full Maya summary], without mutating either."""
    require(type(seed) is int, "seed must be an integer")
    require(
        isinstance(summaries, list) and len(summaries) == 2,
        "expected one diagnostic and one Maya summary",
    )
    cases = json.loads(case_bytes)
    ids = pilot.frozen.case_ids(cases)
    selected = [(i, cid) for i, cid in enumerate(ids, 1) if cid in SELECTED_IDS]
    require(
        tuple(cid for _, cid in selected) == SELECTED_IDS, "invalid frozen selection"
    )
    sha = hashlib.sha256(case_bytes).hexdigest()
    diagnostic, maya = summaries
    diagnostic_rows = validate_rows(diagnostic, sha, selected)
    maya_rows = validate_rows(maya, sha, list(enumerate(ids, 1)))
    condition = diagnostic.get("condition")
    require(
        isinstance(condition, str) and condition in ("none-frozen", "low-double"),
        "expected one supported condition",
    )
    require(
        all(row.get("condition", condition) == condition for row in diagnostic_rows),
        "mixed diagnostic conditions",
    )
    rng = random.Random(seed)
    packet = {"cases": []}
    key = {
        "seed": seed,
        "cases_sha256": sha,
        "condition": condition,
        "scope": "post-hoc diagnostic only; not a rescore of the original eight cases",
        "sources": {"v41": diagnostic.get("arm", "v41"), "maya": maya.get("arm")},
        "cases": {},
    }
    for row, (ordinal, case_id) in zip(diagnostic_rows, selected):
        case = cases[ordinal - 1]
        require(
            isinstance(case.get("system"), str)
            and isinstance(case.get("user"), str)
            and case["user"].strip(),
            "invalid case text",
        )
        require(
            isinstance(case.get("rubric"), dict)
            and isinstance(case.get("reference"), dict)
            and isinstance(case["reference"].get("facts"), dict),
            "invalid case rubric/reference",
        )
        sources = ["v41", "maya"]
        rng.shuffle(sources)
        rows = {"v41": row, "maya": maya_rows[ordinal - 1]}
        mapping = dict(zip(LABELS, sources))
        candidates = [
            {
                "candidate": label,
                "answer": rows[source]["answer"],
                "status": rows[source]["status"],
            }
            for label, source in mapping.items()
        ]
        rng.shuffle(candidates)
        key["cases"][case_id] = mapping
        packet["cases"].append(
            {
                "id": case_id,
                "task": case["user"],
                "system": case["system"],
                "rubric": copy.deepcopy(case["rubric"]),
                "reference_facts": copy.deepcopy(case["reference"]["facts"]),
                "candidates": candidates,
            }
        )
    return packet, key


def validate_grades(cases, packet, grades):
    """Validate eight grades; cases is the original frozen eight-case list."""
    ids = pilot.frozen.case_ids(cases)
    require(
        tuple(cid for cid in ids if cid in SELECTED_IDS) == SELECTED_IDS,
        "invalid frozen selection",
    )
    require(
        isinstance(packet, dict)
        and isinstance(packet.get("cases"), list)
        and len(packet["cases"]) == 4,
        "invalid packet",
    )
    for case, cid in zip(packet["cases"], SELECTED_IDS):
        require(isinstance(case, dict) and case.get("id") == cid, "invalid packet case")
        candidates = case.get("candidates")
        require(
            isinstance(candidates, list)
            and len(candidates) == 2
            and all(isinstance(c, dict) for c in candidates)
            and sorted(c.get("candidate", "") for c in candidates) == list(LABELS),
            "invalid packet labels",
        )
    require(
        isinstance(grades, dict)
        and set(grades) == {"grades"}
        and isinstance(grades["grades"], list),
        "invalid grades",
    )
    expected = {(cid, label) for cid in SELECTED_IDS for label in LABELS}
    seen = set()
    for grade in grades["grades"]:
        require(
            isinstance(grade, dict)
            and set(grade)
            == {
                "case_id",
                "candidate",
                "score",
                "definite_bugs",
                "uncertain_concerns",
                "rationale",
            },
            "invalid grade fields",
        )
        require(
            isinstance(grade["case_id"], str) and isinstance(grade["candidate"], str),
            "invalid grade identifiers",
        )
        pair = (grade["case_id"], grade["candidate"])
        require(pair in expected and pair not in seen, "unknown or duplicate grade")
        seen.add(pair)
        require(
            type(grade["score"]) is int and 0 <= grade["score"] <= 4, "invalid score"
        )
        require(
            all(
                isinstance(grade[k], list) and all(isinstance(s, str) for s in grade[k])
                for k in ("definite_bugs", "uncertain_concerns")
            ),
            "invalid concerns",
        )
        require(
            isinstance(grade["rationale"], str) and grade["rationale"].strip(),
            "missing rationale",
        )
    require(seen == expected, "missing grades")
    return grades


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic-summary", type=Path, required=True)
    parser.add_argument(
        "--maya-summary", type=Path, default=FROZEN / "results/maya.json"
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args(argv)
    require(
        args.diagnostic_summary.resolve() != args.maya_summary.resolve(),
        "must supply two distinct source summaries",
    )
    packet, key = build_packet(
        (FROZEN / "cases.json").read_bytes(),
        [
            json.loads(p.read_bytes())
            for p in (args.diagnostic_summary, args.maya_summary)
        ],
        seed=secrets.randbits(128) if args.seed is None else args.seed,
    )
    args.out.mkdir(mode=0o700, exist_ok=False)
    public, private = args.out / "blind", args.out / "private"
    public.mkdir(mode=0o700)
    private.mkdir(mode=0o700)
    reverse = copy.deepcopy(packet)
    for case in reverse["cases"]:
        case["candidates"].reverse()
    for path, value in (
        (public / "packet.json", packet),
        (public / "packet-b.json", reverse),
        (private / "map.json", key),
    ):
        pilot.frozen.write_private(
            path, json.dumps(value, ensure_ascii=True, indent=2) + "\n"
        )
    for label, name in (("a", "packet.json"), ("b", "packet-b.json")):
        pilot.frozen.write_private(
            public / f"judge-{label}.txt", f"Read only {name}.\n\n" + PROMPT
        )
    pilot.frozen.write_private(public / "README.md", pilot.frozen.HANDOFF)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
