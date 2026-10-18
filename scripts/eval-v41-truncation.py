#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Four-case truncation diagnostic only; never an eight-case comparison."""

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "_truncation_pilot", Path(__file__).with_name("eval-v41-pilot.py")
)
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)
require = pilot.require
frozen = pilot.frozen


def select(baseline, cases, sha):
    require(isinstance(baseline, dict), "baseline must be an object")
    require(baseline.get("status") == "completed", "baseline not completed")
    require(baseline.get("cases_sha256") == sha, "baseline frozen cases SHA mismatch")
    rows = baseline.get("results")
    require(isinstance(rows, list) and len(rows) == 8, "expected eight baseline rows")
    require(all(isinstance(r, dict) for r in rows), "invalid baseline row")
    require(
        [r.get("id") for r in rows] == [c["id"] for c in cases],
        "baseline order mismatch",
    )
    for ordinal, row in enumerate(rows, 1):
        require(
            type(row.get("ordinal")) is int and row["ordinal"] == ordinal,
            "baseline ordinal mismatch",
        )
        require(
            row.get("status") in {"completed", "incomplete"},
            "baseline technical failure",
        )
        require(
            row.get("native_stop_reason") in {"eos", "token_limit"},
            "invalid baseline stop reason",
        )
        require(
            (row["status"] == "incomplete")
            == (row["native_stop_reason"] == "token_limit"),
            "inconsistent baseline stop",
        )
    selected = [
        (i, c)
        for i, (c, r) in enumerate(zip(cases, rows), 1)
        if r["status"] == "incomplete" and r["native_stop_reason"] == "token_limit"
    ]
    require(len(selected) == 4, "expected exactly four token-limited baseline cases")
    return selected


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument(
        "--condition", choices=["none-frozen", "low-double"], required=True
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--timeout-seconds", "--timeout", type=float, default=900)
    args = parser.parse_args(argv)
    require(args.model_id.strip(), "empty model ID")
    require(
        math.isfinite(args.timeout_seconds) and args.timeout_seconds > 0,
        "invalid timeout",
    )
    baseline = pilot.read_metadata(args.baseline)
    baseline_sha = frozen.digest(args.baseline)
    case_bytes = (pilot.FROZEN / "cases.json").read_bytes()
    cases = json.loads(case_bytes)
    require(len(cases) == 8, "expected eight frozen cases")
    sha = hashlib.sha256(case_bytes).hexdigest()
    selected = select(baseline, cases, sha)
    bodies = []
    for _, case in selected:
        body = pilot.request_body(args.model_id, case)
        require(body["reasoning"]["effort"] == "low", "frozen effort changed")
        if args.condition == "none-frozen":
            body["reasoning"]["effort"] = "none"
        else:
            body["max_output_tokens"] *= 2
        bodies.append(body)
    probe = pilot.connection(args.endpoint, args.timeout_seconds)
    probe.close()  # Validate URL without making an HTTP request.
    args.out.mkdir(mode=0o700, parents=False, exist_ok=False)
    summary = {
        "scope": "diagnostic4 only; no combined scores or eight-case comparison",
        "status": "running",
        "condition": args.condition,
        "results": [],
        "cases_sha256": sha,
        "selected_case_ordinals": [i for i, _ in selected],
        "baseline": {
            "path": str(args.baseline.resolve()),
            "sha256": baseline_sha,
            "metadata": {k: v for k, v in baseline.items() if k != "results"},
        },
        "runner_sha256": frozen.digest(Path(__file__)),
        "endpoint": args.endpoint,
        "model_id": args.model_id,
        "request_deadline_seconds": args.timeout_seconds,
        "raw_artifacts_private": True,
    }
    try:
        pilot.atomic_json(args.out / "summary.json", summary)
        for (ordinal, case), body in zip(selected, bodies):
            directory = args.out / f"case-{ordinal:02d}"
            directory.mkdir(mode=0o700)
            row = {
                "ordinal": ordinal,
                "id": case["id"],
                "original_case_ordinal": ordinal,
                "original_case_id": case["id"],
                "original_cases_sha256": sha,
                "original_case_sha256": hashlib.sha256(
                    json.dumps(case, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest(),
                "case_hash_encoding": "UTF-8 canonical JSON: sort_keys=True, separators=(',', ':')",
                "condition": args.condition,
                "effort": body["reasoning"]["effort"],
                "effective_cap": body["max_output_tokens"],
                "status": "running",
            }
            summary["results"].append(row)
            pilot.atomic_json(args.out / "summary.json", summary)
            row.update(
                pilot.evaluate(args.endpoint, body, directory, args.timeout_seconds)
            )
            row["checks"] = frozen.check_answer(case, row["answer"])
            pilot.atomic_json(directory / "result.json", row)
            pilot.atomic_json(args.out / "summary.json", summary)
        summary["diagnostic_rates"] = pilot.rates(summary)
        summary["status"] = "completed"
    except Exception as error:
        summary.update(status="aborted", error_type=type(error).__name__)
        if isinstance(error, frozen.InvalidRun):
            summary["error"] = str(error)
        if summary["results"] and summary["results"][-1]["status"] == "running":
            summary["results"][-1]["status"] = "error"
    except BaseException:
        summary["status"] = "interrupted"
        raise
    finally:
        pilot.atomic_json(args.out / "summary.json", summary)
    return 0 if summary["status"] == "completed" else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(
            f"Diagnostic refused input ({type(error).__name__}); no server started or stopped."
        )
        raise SystemExit(1)
