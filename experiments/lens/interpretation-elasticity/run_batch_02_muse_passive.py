import argparse
import subprocess
from pathlib import Path


ROOT = Path(__file__).parent
BINARY = Path("/Users/tito/code/qwen-llm-lens-integration/target/release/qwen-lens")
MODEL = Path("/Users/tito/models/muse-glimmer/Muse-Glimmer-30B-Q8_0.gguf")
IDENTITY_CACHE = Path(
    "/Volumes/wdblack/weights-archive/jacobian-lenses/.identity-cache"
)
LENSES = {
    "j": Path(
        "/Users/tito/models/muse-glimmer/lenses/muse-glimmer-30b-j-lens-matched-v1"
    ),
    "r": Path(
        "/Volumes/wdblack/weights-archive/jacobian-lenses/"
        "Muse-Glimmer-30B-rlens-published-v1"
    ),
}
STEMS = [
    "p5-riemann",
    "p6-grad-school",
    "p7-quitting",
    "p8-promotion",
    "p9-relationship",
    "p10-concert",
    "p11-dimethylmercury",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lens", choices=("j", "r"), required=True)
    args = parser.parse_args()
    output_dir = ROOT / "artifacts" / "passive" / f"muse-{args.lens}-batch-02"
    output_dir.mkdir(parents=True, exist_ok=True)

    for stem in STEMS:
        for variant in ("target", "twin"):
            output = output_dir / f"{stem}.{variant}.trace.json"
            if output.exists():
                continue
            print(f"{args.lens.upper()} {stem} {variant}", flush=True)
            subprocess.run(
                [
                    str(BINARY),
                    "trace-full",
                    "--model",
                    str(MODEL),
                    "--full-lens",
                    str(LENSES[args.lens]),
                    "--messages",
                    str(ROOT / "prompts" / f"{stem}.{variant}.messages.json"),
                    "--message-mode",
                    "high",
                    "--identity-cache",
                    str(IDENTITY_CACHE),
                    "--allow-unvalidated-transfer",
                    "--top-k",
                    "25",
                    "--output",
                    str(output),
                ],
                check=True,
                stdout=subprocess.DEVNULL,
            )


if __name__ == "__main__":
    main()
