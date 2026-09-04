import argparse
import json
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).parent
MANIFEST = ROOT / "plans" / "generated" / "muse-p7-2026-09-04" / "manifest.json"
OUTPUT = (
    ROOT / "artifacts" / "interventions" / "overnight-2026-09-04" / "muse-4096" / "p7"
)
BINARY = Path("/Users/tito/code/qwen-llm-lens-integration/target/release/qwen-lens")
MODEL = Path("/Users/tito/models/muse-glimmer/Muse-Glimmer-30B-Q8_0.gguf")
IDENTITY_CACHE = Path(
    "/Volumes/wdblack/weights-archive/jacobian-lenses/.identity-cache"
)
MESSAGES = ROOT / "prompts" / "p7-quitting.target.messages.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--probe", choices=("empathy-final", "burnout-distributed"), required=True
    )
    args = parser.parse_args()
    document = json.loads(MANIFEST.read_text())
    jobs = [job for job in document["jobs"] if job["probe"] == args.probe]
    env = os.environ.copy()
    env.update(
        {
            "QWEN_GGUF_NO_COPY": "1",
            "QWEN_GGUF_NO_COPY_PREFAULT": "0",
            "QWEN_METAL_LEASE_WAIT": "1",
        }
    )
    state = []
    for index, job in enumerate(jobs, 1):
        output_dir = OUTPUT / job["probe"] / job["lens"]
        output_dir.mkdir(parents=True, exist_ok=True)
        output = output_dir / f"arm-{job['index']:02}-{job['label']}.run.json"
        print(
            f"[{index}/{len(jobs)}] {job['probe']} {job['lens'].upper()} {job['label']}",
            flush=True,
        )
        if not output.exists():
            subprocess.run(
                [
                    str(BINARY),
                    "run",
                    "--model",
                    str(MODEL),
                    "--plan",
                    job["plan"],
                    "--messages",
                    str(MESSAGES),
                    "--message-mode",
                    "high",
                    "--identity-cache",
                    str(IDENTITY_CACHE),
                    "--max-new-tokens",
                    "4096",
                    "--prefill-execution",
                    "auto",
                    "--temperature",
                    "0",
                    "--output",
                    str(output),
                    "--format",
                    "summary",
                ],
                check=True,
                env=env,
                stdout=subprocess.DEVNULL,
            )
        run = json.loads(output.read_text())
        state.append(
            {
                **job,
                "output": str(output),
                "stop_reason": run["stop_reason"],
                "generated_tokens": len(run["generated_token_ids"]),
            }
        )
        (OUTPUT / f"state-{args.probe}.json").write_text(
            json.dumps(state, indent=2) + "\n"
        )


if __name__ == "__main__":
    main()
