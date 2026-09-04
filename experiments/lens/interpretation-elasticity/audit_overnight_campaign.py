import json
from pathlib import Path


ROOT = Path(__file__).parent
ARTIFACT_ROOT = ROOT / "artifacts" / "interventions" / "overnight-2026-09-04"

CATEGORIES = {
    "qwen_single_site_stance": [
        (ARTIFACT_ROOT / "qwen" / "stance" / "p2").glob("*/arms/*/run.json"),
        (ARTIFACT_ROOT / "qwen-4096" / "stance").glob("*/*/arms/*/run.json"),
    ],
    "qwen_content": [
        (ARTIFACT_ROOT / "qwen-4096" / "content").glob("*/*/arms/*/run.json")
    ],
    "qwen_p7_distributed": [
        (ARTIFACT_ROOT / "qwen-4096" / "distributed" / "p7").glob("*/*.run.json")
    ],
    "muse_p7": [(ARTIFACT_ROOT / "muse-4096" / "p7").glob("*/*/*.run.json")],
    "qwen_p5_thinking": [
        (ARTIFACT_ROOT / "qwen-8192" / "thinking" / "p5").glob("*/arms/*/run.json")
    ],
    "qwen_p11_boundary": [
        (ARTIFACT_ROOT / "qwen-4096" / "boundary" / "p11").glob("*/arms/*/run.json")
    ],
}
EXPECTED_COUNTS = {
    "qwen_single_site_stance": 84,
    "qwen_content": 49,
    "qwen_p7_distributed": 14,
    "muse_p7": 28,
    "qwen_p5_thinking": 14,
    "qwen_p11_boundary": 14,
}


def paths(iterators: list) -> list[Path]:
    return sorted({path for iterator in iterators for path in iterator})


def duplicate_zero_checks(run_paths: list[Path]) -> list[dict]:
    by_parent = {}
    for path in run_paths:
        if path.name == "run.json":
            group = path.parent.parent.parent
            arm = path.parent.name
        else:
            group = path.parent
            arm = path.name
        by_parent.setdefault(group, {})[arm] = path

    checks = []
    for group, arms in sorted(by_parent.items()):
        zero_names = sorted(
            name for name in arms if "000000" in name or "00-zero-a" in name
        )
        twin_names = sorted(
            name for name in arms if "000001" in name or "01-zero-b" in name
        )
        if len(zero_names) != 1 or len(twin_names) != 1:
            continue
        first = json.loads(arms[zero_names[0]].read_text())
        second = json.loads(arms[twin_names[0]].read_text())
        checks.append(
            {
                "group": str(group),
                "generated_tokens_identical": first["generated_token_ids"]
                == second["generated_token_ids"],
                "decoded_text_identical": first["decoded_text"]
                == second["decoded_text"],
                "zero_applications": len(first["operation_applications"]) == 0
                and len(second["operation_applications"]) == 0,
            }
        )
    return checks


def main() -> None:
    categories = {}
    all_runs = []
    for name, iterators in CATEGORIES.items():
        run_paths = paths(iterators)
        all_runs.extend(run_paths)
        records = [json.loads(path.read_text()) for path in run_paths]
        categories[name] = {
            "count": len(records),
            "expected_count": EXPECTED_COUNTS[name],
            "all_stop_token": all(
                run["stop_reason"] == "stop_token" for run in records
            ),
        }

    duplicate_checks = duplicate_zero_checks(all_runs)
    audit = {
        "schema": "interpretation_elasticity.overnight_audit",
        "schema_version": 1,
        "categories": categories,
        "total_completed_runs": len(all_runs),
        "expected_total_completed_runs": sum(EXPECTED_COUNTS.values()),
        "duplicate_zero_groups": len(duplicate_checks),
        "all_duplicate_zeros_valid": all(
            all(value for key, value in check.items() if key != "group")
            for check in duplicate_checks
        ),
        "duplicate_zero_checks": duplicate_checks,
        "excluded_pilot": str(ARTIFACT_ROOT / "qwen" / "stance" / "p7"),
    }
    output = ARTIFACT_ROOT / "audit.json"
    output.write_text(json.dumps(audit, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: value
                for key, value in audit.items()
                if key != "duplicate_zero_checks"
            },
            indent=2,
        )
    )
    if (
        audit["total_completed_runs"] != audit["expected_total_completed_runs"]
        or not audit["all_duplicate_zeros_valid"]
        or any(
            row["count"] != row["expected_count"] or not row["all_stop_token"]
            for row in categories.values()
        )
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
