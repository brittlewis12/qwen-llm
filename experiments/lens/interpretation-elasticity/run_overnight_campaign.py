import argparse
import json
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).parent
CAMPAIGN = ROOT / "plans" / "generated" / "overnight-2026-09-04" / "campaign.json"
ARTIFACT_ROOT = ROOT / "artifacts" / "interventions" / "overnight-2026-09-04"
BINARY = Path("/Users/tito/code/qwen-llm-lens-integration/target/release/qwen-lens")
MODEL = Path("/Volumes/wdblack/weights-archive/qwen3.6-27b-q8/qwen3.6-27b-q8_0.gguf")


def coefficients(job: dict) -> str:
    return ",".join(format(arm["coefficient"], ".9g") for arm in job["arms"])


def sweep_output(job: dict, bound: int) -> Path:
    return ARTIFACT_ROOT / f"qwen-{bound}" / job["control"] / job["item"] / job["lens"]


def run_sweep(job: dict, bound: int) -> Path:
    output = sweep_output(job, bound)
    if output.exists():
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(
        {
            "QWEN_GGUF_NO_COPY": "1",
            "QWEN_GGUF_NO_COPY_PREFAULT": "0",
            "QWEN_METAL_LEASE_WAIT": "1",
        }
    )
    subprocess.run(
        [
            str(BINARY),
            "sweep",
            "--model",
            str(MODEL),
            "--plan",
            job["plan"],
            "--operation",
            "attenuate",
            "--coefficients",
            coefficients(job),
            "--messages",
            job["messages"],
            "--message-mode",
            job["mode"].replace("_", "-"),
            "--max-new-tokens",
            str(bound),
            "--prefill-execution",
            "auto",
            "--temperature",
            "0",
            "--output",
            str(output),
        ],
        check=True,
        env=env,
        stdout=subprocess.DEVNULL,
    )
    return output


def inspect(job: dict, output: Path, bound: int) -> None:
    result = subprocess.run(
        [str(BINARY), "inspect-sweep", str(output), "--reference-arm", "0"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    inspection = (
        ARTIFACT_ROOT
        / f"inspections-{bound}"
        / job["control"]
        / f"{job['item']}-{job['lens']}.txt"
    )
    inspection.parent.mkdir(parents=True, exist_ok=True)
    inspection.write_text(result.stdout)


def completion(output: Path) -> dict:
    runs = sorted((output / "arms").glob("*/run.json"))
    records = []
    for path in runs:
        run = json.loads(path.read_text())
        records.append(
            {
                "arm": path.parent.name,
                "stop_reason": run["stop_reason"],
                "generated_tokens": len(run["generated_token_ids"]),
            }
        )
    return {
        "complete": bool(records)
        and all(record["stop_reason"] == "stop_token" for record in records),
        "arms": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control", choices=("stance", "content"), required=True)
    parser.add_argument(
        "--items",
        help="Optional comma-separated item IDs; defaults to every admissible job",
    )
    args = parser.parse_args()
    allowed = None if args.items is None else set(args.items.split(","))
    campaign = json.loads(CAMPAIGN.read_text())
    jobs = [
        job
        for job in campaign["jobs"]
        if job["control"] == args.control
        and (allowed is None or job["item"] in allowed)
    ]
    state = []
    for index, job in enumerate(jobs, 1):
        label = f"{job['control']} {job['item']} {job['lens'].upper()}"
        print(f"[{index}/{len(jobs)}] {label} at 4096", flush=True)
        output = run_sweep(job, 4096)
        inspect(job, output, 4096)
        result = completion(output)
        record = {"job": label, "bound": 4096, "output": str(output), **result}
        if not result["complete"]:
            print(f"  escalating complete cell: {label} to 8192", flush=True)
            output = run_sweep(job, 8192)
            inspect(job, output, 8192)
            result = completion(output)
            record["escalated"] = {
                "bound": 8192,
                "output": str(output),
                **result,
            }
        state.append(record)
        state_path = ARTIFACT_ROOT / f"state-{args.control}.json"
        state_path.write_text(json.dumps(state, indent=2) + "\n")


if __name__ == "__main__":
    main()
