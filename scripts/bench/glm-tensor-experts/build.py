"""Build the standalone Metal controls. Both arms use product Metal flags."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parents[3]
out = Path(sys.argv[1]).resolve()
out.mkdir(parents=True, exist_ok=True)


def run(*cmd):
    print(json.dumps({"command": cmd}), flush=True)
    subprocess.run(cmd, check=True, cwd=root)


run(
    sys.executable,
    str(Path(__file__).with_name("generate.py")),
    str(out / "probe.metal"),
)
run(
    "xcrun",
    "-sdk",
    "macosx",
    "metal",
    "-std=metal4.0",
    "-mmacosx-version-min=26.0",
    "-O3",
    "-ffast-math",
    "-c",
    str(out / "probe.metal"),
    "-o",
    str(out / "probe.air"),
)
run(
    "xcrun",
    "-sdk",
    "macosx",
    "metallib",
    str(out / "probe.air"),
    "-o",
    str(out / "probe.metallib"),
)

# Decode-only library for the M4 (Metal 3.1) bit-exactness screen. It holds
# only the shared dequantizers and their unpack drivers, so it needs no
# tensor-ops language feature.
formats = [
    ("iq2_s", 82),
    ("iq2_xs", 74),
    ("iq2_xxs", 66),
    ("iq3_s", 110),
    ("iq3_xxs", 98),
]
decode = (
    '#include <metal_stdlib>\nusing namespace metal;\n#include "'
    + str(root / "kernels/quant_tiles.h")
    + '"\n#include "'
    + str(root / "scripts/bench/glm-tensor-experts/iq_fast_decoders.metalh")
    + '"\n'
)
for dtype, _ in formats:
    for suffix in ["", "_fast", "_vec"] if dtype.startswith("iq2_") else ["", "_fast"]:
        decode += f"""\nkernel void unpack_{dtype}{suffix}(device const uchar *w [[buffer(0)]], device half *h [[buffer(1)]], uint i [[thread_position_in_grid]]) {{
 half4x4 v; qt_dequantize_{dtype}{suffix}(w + ulong(i / 16) * QT_{dtype.upper()}_BYTES, short(i%16), v);
 for (short j=0;j<4;++j) *((device half4 *)(h+ulong(i)*16+4*j))=v[j];
}}
"""
(out / "decode.metal").write_text(decode)
run(
    "xcrun",
    "-sdk",
    "macosx",
    "metal",
    "-std=metal3.1",
    "-mmacosx-version-min=15.0",
    "-O3",
    "-ffast-math",
    "-c",
    str(out / "decode.metal"),
    "-o",
    str(out / "decode.air"),
)
run(
    "xcrun",
    "-sdk",
    "macosx",
    "metallib",
    str(out / "decode.air"),
    "-o",
    str(out / "decode.metallib"),
)

run(
    "xcrun",
    "clang++",
    "-std=c++20",
    "-O3",
    "-fobjc-arc",
    "-framework",
    "Foundation",
    "-framework",
    "Metal",
    str(Path(__file__).with_name("probe.mm")),
    "-o",
    str(out / "probe"),
)
(out / "identity.json").write_text(
    json.dumps(
        {
            "commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=root, text=True
            ).strip(),
            "sha256": {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in out.iterdir()
                if p.is_file() and p.name != "identity.json"
            },
        },
        indent=2,
    )
    + "\n"
)
