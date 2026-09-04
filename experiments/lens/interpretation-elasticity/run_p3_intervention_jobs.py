import argparse
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).parent
JOBS_PATH = ROOT / "plans" / "generated" / "p3" / "jobs.json"
OUTPUT_ROOT = ROOT / "artifacts" / "interventions" / "p3"
BINARY = Path("/Users/tito/code/qwen-llm-lens-integration/target/release/qwen-lens")
QWEN_MODEL = Path(
    "/Volumes/wdblack/weights-archive/qwen3.6-27b-q8/qwen3.6-27b-q8_0.gguf"
)
MUSE_MODEL = Path("/Users/tito/models/muse-glimmer/Muse-Glimmer-30B-Q8_0.gguf")
IDENTITY_CACHE = Path(
    "/Volumes/wdblack/weights-archive/jacobian-lenses/.identity-cache"
)


def run(command: list[str]) -> None:
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL)


def run_qwen(job: dict) -> None:
    output = (
        OUTPUT_ROOT
        / "qwen"
        / job["control"]
        / job["item"]
        / job["lens"]
        / f"layer-{job['layer']}"
    )
    if output.exists():
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    coefficients = ",".join(format(arm["coefficient"], ".9g") for arm in job["arms"])
    run(
        [
            str(BINARY),
            "sweep",
            "--model",
            str(QWEN_MODEL),
            "--plan",
            job["plan"],
            "--operation",
            "attenuate",
            "--coefficients",
            coefficients,
            "--requests-jsonl",
            job["requests_jsonl"],
            "--max-new-tokens",
            "512",
            "--prefill-execution",
            "auto",
            "--temperature",
            "0",
            "--output",
            str(output),
        ]
    )


def effective_plan(source: dict, coefficient: float) -> dict:
    plan = json.loads(json.dumps(source))
    plan["operations"][0]["action"]["coefficient"] = coefficient
    return plan


def run_muse(job: dict) -> None:
    output = (
        OUTPUT_ROOT
        / "muse"
        / job["control"]
        / job["item"]
        / job["lens"]
        / f"layer-{job['layer']}"
    )
    output.mkdir(parents=True, exist_ok=True)
    source_plan = json.loads(Path(job["plan"]).read_text())
    unique_plans = {}
    records = []
    for index, arm in enumerate(job["arms"]):
        coefficient = arm["coefficient"]
        key = format(coefficient, ".9g")
        if key not in unique_plans:
            plan_path = (
                output
                / f"coefficient-{key.replace('-', 'neg').replace('.', '_')}.plan.json"
            )
            plan_path.write_text(
                json.dumps(effective_plan(source_plan, coefficient), indent=2) + "\n"
            )
            unique_plans[key] = plan_path
        run_path = output / f"arm-{index:02}-{arm['label']}.run.json"
        if not run_path.exists():
            run(
                [
                    str(BINARY),
                    "run",
                    "--model",
                    str(MUSE_MODEL),
                    "--plan",
                    str(unique_plans[key]),
                    "--messages",
                    job["messages"],
                    "--message-mode",
                    "high",
                    "--identity-cache",
                    str(IDENTITY_CACHE),
                    "--max-new-tokens",
                    "512",
                    "--prefill-execution",
                    "auto",
                    "--temperature",
                    "0",
                    "--output",
                    str(run_path),
                    "--format",
                    "summary",
                ]
            )
        records.append({"index": index, **arm, "run": run_path.name})
    manifest = {
        "schema": "interpretation_elasticity.muse_coefficient_sweep",
        "schema_version": 1,
        "job": job,
        "arms": records,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("qwen", "muse"), required=True)
    parser.add_argument("--control", choices=("stance", "content"), required=True)
    args = parser.parse_args()
    document = json.loads(JOBS_PATH.read_text())
    jobs = [
        job
        for job in document["jobs"]
        if job["model"] == args.model and job["control"] == args.control
    ]
    for index, job in enumerate(jobs, 1):
        print(
            f"[{index}/{len(jobs)}] {job['model']} {job['control']} {job['item']} "
            f"{job['lens'].upper()} L{job['layer']}",
            flush=True,
        )
        if args.model == "qwen":
            run_qwen(job)
        else:
            run_muse(job)


if __name__ == "__main__":
    main()
