#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Blind two real eight-case summaries; create packets only, never evaluate models."""

import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import secrets

FROZEN = (
    Path(__file__).resolve().parents[1] / "docs/bench/2026-10-16-glm-quality-comparison"
)
spec = importlib.util.spec_from_file_location("_frozen_blind", FROZEN / "blind.py")
frozen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(frozen)
require = frozen.require
LABELS = ("A", "B")
PROMPT = (
    frozen.PROMPT.replace("exactly 24 records", "exactly 16 records").replace(
        '"candidate":"C1"', '"candidate":"A"'
    )
    + """
Use candidate labels A and B only. Assignment is randomized separately per case;
neither label nor presentation order identifies a source. Grade each answer on
the unchanged 0-4 scale (eight cases, maximum 32 per source); do not total scores.
Use only the supplied tasks, rubrics, reference facts and answers. Do not consult
model identities, performance measurements, source files or the private key.
"""
)


def build_packet(case_bytes, summaries, *, seed):
    cases = json.loads(case_bytes)
    ids = frozen.case_ids(cases)
    sha = hashlib.sha256(case_bytes).hexdigest()
    require(type(seed) is int, "seed must be an integer")
    require(
        isinstance(summaries, list) and len(summaries) == 2,
        "expected two genuine summaries",
    )
    arms = []
    for summary in summaries:
        require(
            isinstance(summary, dict) and summary.get("status") == "completed",
            "run not completed",
        )
        require(summary.get("cases_sha256") == sha, "frozen cases hash mismatch")
        arm = summary.get("arm")
        require(isinstance(arm, str) and arm.strip(), "invalid arm")
        arms.append(arm)
        rows = summary.get("results")
        require(isinstance(rows, list) and len(rows) == 8, "expected eight results")
        for ordinal, (row, case_id) in enumerate(zip(rows, ids), 1):
            require(
                isinstance(row, dict)
                and row.get("id") == case_id
                and type(row.get("ordinal")) is int
                and row["ordinal"] == ordinal,
                "case order/ordinal mismatch",
            )
            require(
                row.get("status") in {"completed", "incomplete", "model_output_error"}
                and isinstance(row.get("answer"), str),
                "invalid answer/status",
            )
    require(len(set(arms)) == 2, "duplicate arms")
    rng = random.Random(seed)
    packet = {"cases": []}
    key = {
        "seed": seed,
        "cases_sha256": sha,
        "sources": {"v41": arms[0], "maya": arms[1]},
        "cases": {},
    }
    for index, case in enumerate(cases):
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
        assignment = [0, 1]
        rng.shuffle(assignment)
        candidates = []
        key["cases"][case["id"]] = {}
        for label, source in zip(LABELS, assignment):
            row = summaries[source]["results"][index]
            candidates.append(
                {"candidate": label, "answer": row["answer"], "status": row["status"]}
            )
            key["cases"][case["id"]][label] = arms[source]
        rng.shuffle(candidates)
        packet["cases"].append(
            {
                "id": case["id"],
                "task": case["user"],
                "system": case["system"],
                "rubric": copy.deepcopy(case["rubric"]),
                "reference_facts": copy.deepcopy(case["reference"]["facts"]),
                "candidates": candidates,
            }
        )
    return packet, key


def validate_grades(cases, packet, grades):
    """Validate the adapted 16-record shape; no scoring or aggregation."""
    ids = frozen.case_ids(cases)
    require(
        isinstance(packet, dict)
        and isinstance(packet.get("cases"), list)
        and len(packet["cases"]) == 8,
        "invalid packet",
    )
    for case, expected in zip(packet["cases"], ids):
        require(
            case.get("id") == expected and len(case.get("candidates", [])) == 2,
            "invalid packet case",
        )
        require(
            {c.get("candidate") for c in case["candidates"]} == set(LABELS),
            "invalid packet labels",
        )
    require(
        isinstance(grades, dict)
        and set(grades) == {"grades"}
        and isinstance(grades["grades"], list),
        "invalid grades",
    )
    expected = {(case_id, label) for case_id in ids for label in LABELS}
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
        key = (grade["case_id"], grade["candidate"])
        require(key in expected and key not in seen, "unknown or duplicate grade")
        seen.add(key)
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
    parser.add_argument("--v41-summary", type=Path, required=True)
    parser.add_argument(
        "--maya-summary", type=Path, default=FROZEN / "results/maya.json"
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--seed",
        type=int,
        help="reproducible private seed; default generates and privately records 128 bits",
    )
    args = parser.parse_args(argv)
    require(
        args.v41_summary.resolve() != args.maya_summary.resolve(),
        "must supply two distinct source summaries",
    )
    summaries = [
        json.loads(path.read_bytes()) for path in (args.v41_summary, args.maya_summary)
    ]
    packet, key = build_packet(
        (FROZEN / "cases.json").read_bytes(),
        summaries,
        seed=secrets.randbits(128) if args.seed is None else args.seed,
    )
    args.out.mkdir(mode=0o700, exist_ok=False)
    public, private = args.out / "blind", args.out / "private"
    public.mkdir(mode=0o700)
    private.mkdir(mode=0o700)
    reverse = copy.deepcopy(packet)
    for case in reverse["cases"]:
        case["candidates"].reverse()
    for path, value in [
        (public / "packet.json", packet),
        (public / "packet-b.json", reverse),
        (private / "map.json", key),
    ]:
        frozen.write_private(
            path, json.dumps(value, ensure_ascii=True, indent=2) + "\n"
        )
    frozen.write_private(public / "judge-a.txt", "Read only packet.json.\n\n" + PROMPT)
    frozen.write_private(
        public / "judge-b.txt", "Read only packet-b.json.\n\n" + PROMPT
    )
    frozen.write_private(public / "README.md", frozen.HANDOFF)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
