# /// script
# requires-python = ">=3.11"
# ///
"""Summarize existing native logs without rerunning a model."""

import json
import re
import sys
from pathlib import Path

text = Path(sys.argv[1]).read_text()
result = {"profiles": [], "maxima": {}}
for line in text.splitlines():
    if "native_split coarse_profile" in line:
        stages = re.findall(
            r"stage: (LayersZeroOne|Tail|PostPle \{ layer: \d+, mixer: (?:GatedDeltaNet|QwenSparseAttention) \}), "
            r"duration_ticks: \d+, gpu_ms: ([0-9.eE+-]+),",
            line,
        )
        assert len(stages) == 48, len(stages)
        buckets = {}
        counts = {}
        for stage, ms in stages:
            bucket = (
                "GDN_blocks"
                if "GatedDeltaNet" in stage
                else "QSA_blocks"
                if "QwenSparseAttention" in stage
                else stage
            )
            buckets[bucket] = buckets.get(bucket, 0.0) + float(ms)
            counts[bucket] = counts.get(bucket, 0) + 1
        result["profiles"].append(
            {"split": "split=true" in line, "gpu_ms": buckets, "counts": counts}
        )
    match = re.search(
        r"state (hyper\d+|tensor\d+/(F32|F16)) rms=([\deE+.-]+) max_abs=([\deE+.-]+)",
        line,
    )
    if match:
        label = match[2] or "hyper"
        maxima = result["maxima"].setdefault(
            label, {"relative_rms": 0.0, "absolute": 0.0}
        )
        maxima["relative_rms"] = max(maxima["relative_rms"], float(match[3]))
        maxima["absolute"] = max(maxima["absolute"], float(match[4]))
assert "test result: ok. 1 passed" in text
assert len(result["profiles"]) == 2
print(json.dumps(result, indent=2))
