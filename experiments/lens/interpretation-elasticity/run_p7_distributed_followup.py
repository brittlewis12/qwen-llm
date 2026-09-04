import json
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).parent
MANIFEST = ROOT / "plans" / "generated" / "p7-distributed-2026-09-04" / "manifest.json"
OUTPUT = (
    ROOT
    / "artifacts"
    / "interventions"
    / "overnight-2026-09-04"
    / "qwen-4096"
    / "distributed"
    / "p7"
)
BINARY = Path("/Users/tito/code/qwen-llm-lens-integration/target/release/qwen-lens")
MODEL = Path("/Volumes/wdblack/weights-archive/qwen3.6-27b-q8/qwen3.6-27b-q8_0.gguf")
MESSAGES = ROOT / "prompts" / "p7-quitting.target.messages.json"


def main() -> None:
    document = json.loads(MANIFEST.read_text())
    env = os.environ.copy()
    env.update(
        {
            "QWEN_GGUF_NO_COPY": "1",
            "QWEN_GGUF_NO_COPY_PREFAULT": "0",
            "QWEN_METAL_LEASE_WAIT": "1",
        }
    )
    state = []
    for index, job in enumerate(document["jobs"], 1):
        lens_dir = OUTPUT / job["lens"]
        lens_dir.mkdir(parents=True, exist_ok=True)
        output = lens_dir / f"arm-{job['index']:02}-{job['label']}.run.json"
        print(
            f"[{index}/{len(document['jobs'])}] {job['lens'].upper()} {job['label']}",
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
                    "no-thinking",
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
        (OUTPUT / "state.json").write_text(json.dumps(state, indent=2) + "\n")


if __name__ == "__main__":
    main()
