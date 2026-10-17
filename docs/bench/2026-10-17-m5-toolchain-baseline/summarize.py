"""Summarize retained private baseline evidence without publishing machine paths."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

SOURCE = "d47eeced1d77b2d780c14315363c27bd8623089a"
RESULT = re.compile(
    r"^test result: (ok|FAILED)\. (\d+) passed; (\d+) failed; (\d+) ignored;"
)


def workspace_totals(text):
    targets = []
    nested = 0
    current = None
    for line in text.splitlines():
        match = re.match(r"\s*Running .*\((?:.*/)?([^/()]+)-[0-9a-f]+\)$", line)
        doc = re.match(r"\s*Doc-tests (\S+)$", line)
        if match or doc:
            current = {"target": match[1] if match else "doc:" + doc[1]}
            targets.append(current)
        running = re.match(r"^running (\d+) tests?$", line)
        if running and current is not None and "declared_tests" not in current:
            current["declared_tests"] = int(running[1])
        result = RESULT.match(line)
        if result:
            if current is None or "declared_tests" not in current:
                raise ValueError("result without a declared Cargo test count")
            if sum(int(result[i]) for i in (2, 3, 4)) != current["declared_tests"]:
                nested += 1
                continue
            if "passed" in current:
                raise ValueError("multiple outer results for one Cargo target")
            current.update(
                passed=int(result[2]), failed=int(result[3]), ignored=int(result[4])
            )
    if not targets or any("passed" not in target for target in targets):
        raise ValueError("missing top-level test results")
    return {
        "targets": targets,
        "passed": sum(t["passed"] for t in targets),
        "failed": sum(t["failed"] for t in targets),
        "ignored": sum(t["ignored"] for t in targets),
        "nested_child_summaries_excluded": nested,
    }


def summarize(root):
    read = lambda name: (root / name).read_text()
    load = lambda name: json.loads(read(name))
    qualification = load("qualification.json")
    workspace = load("workspace.json")
    if qualification["git_commit"] != SOURCE or workspace["source"] != SOURCE:
        raise ValueError("unexpected source revision")
    totals = workspace_totals(read("workspace.log"))
    totals.update(
        exit_code=int(read("workspace.exit")),
        wall_seconds=workspace["finish_epoch"] - workspace["start_epoch"],
        metal_required=workspace["required_metal"],
        metal_validation=workspace["metal_validation"],
        model_search_unset=workspace["model_search_path"] is None,
        gpu_test_count=None,
    )
    facets = Counter()
    records = []
    for record in qualification["records"]:
        witnesses = []
        for witness in record["witnesses"]:
            facets[witness["status"]] += 1
            entry = {
                key: witness[key] for key in ("facet", "kind", "command", "status")
            }
            entry["exit_code"] = witness.get("exit_code")
            if witness["status"] == "failed":
                tail = witness.get("stderr_tail", "")
                entry["failure_class"] = (
                    "missing model fixture"
                    if "fixture " in tail and " is required" in tail
                    else "performance assertion"
                    if "assertion failed" in tail
                    else "unclassified failure"
                )
                entry["assertions"] = [
                    line for line in tail.splitlines() if "assertion failed" in line
                ]
            witnesses.append(entry)
        records.append(
            {
                "path": record["path"],
                "holds_for_device": record["holds_for_device"],
                "witnesses": witnesses,
            }
        )
    model_text = read("models.log")
    model_counts = re.findall(
        r"^(qwen-llm|qwen-cli): selected=(\d+) passed=(\d+) failed=(\d+)$",
        model_text,
        re.M,
    )
    if len(model_counts) != 2:
        raise ValueError("missing model witness package accounting")
    model_tests = re.findall(r"^test (\S+) \.\.\. ok$", model_text, re.M)
    if len(model_tests) != sum(int(row[2]) for row in model_counts):
        raise ValueError("model test names disagree with pass counts")
    replays = load("validated-replays.json")
    if not replays:
        raise ValueError("empty validated replay suite")
    glm = read("glm-runtime.log")
    prefill = re.search(r"fast vs exact after (\d+) tokens: kl ([\deE+.-]+)", glm)
    decode = re.findall(r"decode step \d+: kl ([\deE+.-]+)", glm)
    if (
        prefill is None
        or len(decode) != 3
        or "test result: ok. 1 passed; 0 failed; 0 ignored" not in glm
    ):
        raise ValueError("incomplete GLM runtime evidence")
    environment = read("environment.txt")
    post = read("post-run.txt")

    def field(pattern, text=environment):
        match = re.search(pattern, text, re.M)
        if match is None:
            raise ValueError(f"missing environment field: {pattern}")
        return match[1]

    metal = field(r"^Apple metal version (.+)$")
    if field(r"^Apple metal version (.+)$", post) != metal:
        raise ValueError("Metal version changed between baseline endpoints")

    def number(text, label):
        return int(re.search(r"^" + re.escape(label) + r"\s*(\d+)", text, re.M)[1])

    facts = qualification["device_facts"]
    inputs = [
        "environment.txt",
        "post-run.txt",
        "metal-install.log",
        "build.log",
        "workspace.log",
        "workspace.json",
        "qualification.json",
        "models.log",
        "models.json",
        "validated-replays.json",
        "glm-runtime.log",
        "glm-runtime.json",
        "source-contracts.log",
    ]
    return {
        "schema_version": 1,
        "source_commit": SOURCE,
        "source_tree": "1bfc7d248796507468cf78ca335cd57df081227f",
        "hardware": {
            key: facts[key]
            for key in (
                "name",
                "architecture",
                "gpu_families",
                "physical_memory_bytes",
                "recommended_max_working_set_bytes",
                "host_page_size_bytes",
            )
        },
        "toolchain": {
            "xcode": field(r"^Xcode (.+)$"),
            "xcode_build": field(r"^Build version (.+)$"),
            "macos_sdk": field(r"^Build version [^\n]+\n([0-9.]+)\nApple metal"),
            "metal": metal,
            "metal_asset": field(
                r"Done downloading: Metal Toolchain (\S+)\.", read("metal-install.log")
            ),
            "rust": field(r"^rustc (.+)$"),
            "cargo": field(r"^cargo (.+)$"),
            "macos": field(r"^ProductVersion:\s*(\S+)"),
            "macos_build": field(r"^BuildVersion:\s*(\S+)"),
        },
        "workspace": totals,
        "qualification": {
            "exit_code": int(read("qualification.exit")),
            "facet_status_counts": dict(facets),
            "records": records,
        },
        "validated_correctness_replays": replays,
        "model_witnesses": {
            "exit_code": int(read("models.exit")),
            "packages": [
                {"package": p, "selected": int(s), "passed": int(n), "failed": int(f)}
                for p, s, n, f in model_counts
            ],
            "tests": model_tests,
        },
        "glm_runtime": {
            "exit_code": int(read("glm-runtime.exit")),
            "metal_validation": True,
            "prompt_tokens": int(prefill[1]),
            "prefill_kl": float(prefill[2]),
            "continuation_kl": [float(value) for value in decode],
            "oracle": "native Exact lineage; not independent upstream full-model parity",
        },
        "offline_source_contracts_exit_code": int(read("source-contracts.exit")),
        "memory_endpoints": {
            "wired_limit_mb": [
                number(text, "iogpu.wired_limit_mb:") for text in (environment, post)
            ],
            "compressions": [
                number(text, "Compressions:") for text in (environment, post)
            ],
            "compressor_pages": [
                number(text, "Pages occupied by compressor:")
                for text in (environment, post)
            ],
            "swap_used_zero_both": all(
                "used = 0.00M" in text for text in (environment, post)
            ),
            "peak_memory_measured": False,
        },
        "raw_evidence_sha256": {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in inputs
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.write_text(json.dumps(summarize(args.evidence), indent=2) + "\n")
