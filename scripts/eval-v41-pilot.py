#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Eight frozen cases against an already-loaded, owned V4.1 endpoint.

Run only after parent approval of full-model correctness. This client does not
start/stop servers, download, inspect weights, or enforce performance gates.
Outputs are private run.py-compatible summaries. Use blind-v41-pilot.py for the
two-arm packet against retained Maya; the original frozen blinder is unchanged.
"""

import argparse
import hashlib
import http.client
import importlib.util
import json
import math
from pathlib import Path
import sys
import time
from urllib.parse import urlsplit

FROZEN = (
    Path(__file__).resolve().parents[1] / "docs/bench/2026-10-16-glm-quality-comparison"
)


def load_frozen(name):
    spec = importlib.util.spec_from_file_location(
        "_v41_frozen_" + name, FROZEN / (name + ".py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# run.py uses an absolute `from cases import check_answer`; bind only that import
# to the frozen checker without changing the caller's module/path environment.
_previous_cases = sys.modules.get("cases")
sys.modules["cases"] = load_frozen("cases")
try:
    frozen = load_frozen("run")
finally:
    if _previous_cases is None:
        sys.modules.pop("cases", None)
    else:
        sys.modules["cases"] = _previous_cases

require = frozen.require
atomic_json = frozen.atomic_json


def connection(endpoint, timeout):
    url = urlsplit(endpoint)
    require(
        url.scheme in {"http", "https"} and url.hostname, "endpoint must be HTTP(S)"
    )
    require(
        url.username is None
        and url.password is None
        and not url.query
        and not url.fragment,
        "endpoint must not contain credentials/query/fragment",
    )
    require(
        url.path.rstrip("/") in {"", "/v1"}, "endpoint must be the server root or /v1"
    )
    cls = (
        http.client.HTTPSConnection
        if url.scheme == "https"
        else http.client.HTTPConnection
    )
    return cls(url.hostname, url.port, timeout=timeout)


def request_body(model, case):
    body = frozen.request_body(model, case)
    del body["x_qwen"]["prefill_lineage"]  # GLM-only; every other input stays frozen.
    return body


def native_stats(result, maximum):
    echo = result["x_qwen"]
    p = echo.get("phases", {})
    require(
        echo.get("version") == "serve_stats_v1" and echo.get("restore_ms") == 0,
        "missing native V4.1 stats; rebuild/restart the owned server",
    )
    require(
        p.get("family") == "deepseek_v41"
        and p.get("prefill_mode") == "single_token_forward"
        and p.get("prefill_rows") == 1,
        "not V4.1 singleton prefill",
    )
    require(
        p.get("ced") is False
        and type(p.get("reused_tokens")) is int
        and p["reused_tokens"] == 0,
        "unexpected CED or reuse",
    )
    for key in ("prefill_ms", "decode_ms", "transition_ms"):
        require(
            type(p.get(key)) in (int, float) and math.isfinite(p[key]) and p[key] >= 0,
            "invalid native phase time",
        )
    for key in (
        "prefill_tokens",
        "generated_tokens",
        "transitions",
        "required_forwards",
        "capacity",
    ):
        require(type(p.get(key)) is int and p[key] >= 0, "invalid native phase count")
    usage = result["usage"]
    require(
        p["prefill_tokens"] == usage["input_tokens"]
        and p["generated_tokens"] == usage["output_tokens"],
        "phase/usage mismatch",
    )
    require(
        0 < p["generated_tokens"] <= maximum
        and p["transitions"] == p["generated_tokens"] - 1,
        "decode transition mismatch",
    )
    require(
        p["required_forwards"] == p["prefill_tokens"] + maximum - 1
        and p["required_forwards"] <= p["capacity"],
        "forward budget mismatch",
    )
    require(
        p["transition_ms"] <= p["decode_ms"] + 1e-6, "transition time exceeds decode"
    )
    return {
        "prefill_ms": p["prefill_ms"],
        "decode_ms": p["decode_ms"],
        "transitions": p["transitions"],
        "transition_ms": p["transition_ms"],
        "phases": {
            k: str(v).lower() if isinstance(v, bool) else str(v) for k, v in p.items()
        },
        "stats": {
            "version": echo["version"],
            "prompt_tokens": str(usage["input_tokens"]),
            "generated_tokens": str(usage["output_tokens"]),
            "matched_tokens": "0",
            "restore_ms": "0.0",
            "stop_reason": result["native_stop_reason"],
            "decode_tps": f"{usage['output_tokens'] / (p['decode_ms'] / 1000):.2f}"
            if p["decode_ms"]
            else "0.00",
        },
        "source": "native x_qwen.stats echo; log-shaped fields derived from the same values, not a stderr capture",
    }


def evaluate(endpoint, body, directory, timeout):
    atomic_json(directory / "request.json", body)
    conn = connection(endpoint, timeout)
    start = time.monotonic()
    try:
        with (
            frozen.deadline(timeout),
            (directory / "response.sse").open("wb") as raw,
            (directory / "events.jsonl").open("w") as events,
        ):
            conn.request(
                "POST",
                "/v1/responses",
                json.dumps(body),
                {"Content-Type": "application/json"},
            )
            response = conn.getresponse()
            atomic_json(
                directory / "http.json",
                {"status": response.status, "headers": response.getheaders()},
            )
            if response.status != 200:
                raw.write(response.read())
                raise frozen.InvalidRun("HTTP request failed")
            require(
                response.getheader("Content-Type", "").split(";")[0]
                == "text/event-stream",
                "not SSE",
            )
            terminal, timing = frozen.parse_sse(response, raw, events, start)
            atomic_json(directory / "response.json", terminal)
            require(
                terminal.get("status") in {"completed", "incomplete"},
                "V4.1 failed response; retained raw output, no automatic server restart",
            )
            result = {**frozen.validate_response(terminal), **timing}
            result["native"] = native_stats(result, body["max_output_tokens"])
            return result
    finally:
        conn.close()


def read_metadata(path):
    with path.open("rb") as source:
        data = source.read((8 << 20) + 1)
    require(
        len(data) <= 8 << 20, "metadata file exceeds 8 MiB; do not supply model weights"
    )
    return json.loads(data)


def rates(summary):
    rows = summary["results"]
    inputs = sum(r["usage"]["input_tokens"] for r in rows)
    outputs = sum(r["usage"]["output_tokens"] for r in rows)
    prefill = sum(r["native"]["prefill_ms"] for r in rows) / 1000
    decode = sum(r["native"]["decode_ms"] for r in rows) / 1000
    return {
        "input_tokens": inputs,
        "output_tokens": outputs,
        "weighted_prefill_tps": inputs / prefill if prefill else None,
        "weighted_generation_tps": outputs / decode if decode else None,
        "request_seconds_total": sum(r["request_elapsed_s"] for r in rows),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument(
        "--out", type=Path, required=True, help="fresh private output directory"
    )
    parser.add_argument("--arm", default="v41-old-reap272-nativef32")
    parser.add_argument(
        "--download-manifest",
        type=Path,
        help="owned local metadata only; recorded as expected provenance, never rehashes weights",
    )
    parser.add_argument(
        "--maya-summary", type=Path, default=FROZEN / "results/maya.json"
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=900,
        help="per-request wall deadline, configurable for singleton prefill; not a scoring gate",
    )
    args = parser.parse_args(argv)
    require(args.model_id.strip() and args.arm.strip(), "empty model ID or arm")
    require(
        math.isfinite(args.timeout_seconds) and args.timeout_seconds > 0,
        "invalid deadline",
    )
    probe = connection(args.endpoint, min(args.timeout_seconds, 30))
    probe.close()  # URL validation only; no startup, polling, or reload.
    case_bytes = (FROZEN / "cases.json").read_bytes()
    cases = json.loads(case_bytes)
    require(len(cases) == 8, "expected the eight frozen cases")
    sha = hashlib.sha256(case_bytes).hexdigest()
    maya = read_metadata(args.maya_summary)
    require(
        maya.get("status") == "completed" and maya.get("cases_sha256") == sha,
        "Maya summary must use the same frozen cases",
    )
    require(
        [r["id"] for r in maya["results"]] == [c["id"] for c in cases],
        "Maya case order mismatch",
    )
    baseline_rates = rates(maya)
    manifest = read_metadata(args.download_manifest) if args.download_manifest else None
    args.out.mkdir(mode=0o700, parents=False, exist_ok=False)
    summary = {
        "arm": args.arm,
        "status": "running",
        "results": [],
        "cases_sha256": sha,
        "runner_sha256": frozen.digest(Path(__file__)),
        "checker_sha256": frozen.digest(FROZEN / "cases.py"),
        "frozen_runner_sha256": frozen.digest(FROZEN / "run.py"),
        "raw_artifacts_private": True,
        "request_deadline_seconds": args.timeout_seconds,
        "checkpoint": {
            "profile": "old-reap-272e-not-1017",
            "mode": "NativeF32",
            "expected_reap_revision": "f792019bcb369768220f01088f0f7af230786982",
            "expected_engram_base_revision": "314113f8a7dadb388215396ba5fbee2c610a04d6",
            "owned_download_manifest": manifest,
            "artifact_hashes_verified": False,
            "verification_scope": "metadata declarations only; no weights read and no authentication of resident process/build",
        },
        "servers": [
            {
                "server_number": 1,
                "status": "attached",
                "startup_elapsed_s": None,
                "model_id": args.model_id,
                "endpoint": args.endpoint,
            }
        ],
        "reasoning": {
            "effort": "low",
            "protocol": "official V4.1 tier",
            "same_internal_budget_as_glm": False,
        },
        "comparison_scope": "descriptive whole configurations, not causal quantization or matched-token work; tokenizers, thinking policy, builds and possibly hardware differ; Maya timings are rounded logs, V4.1 timings are native phase echoes",
    }
    try:
        atomic_json(args.out / "summary.json", summary)
        (args.out / "cases.json").write_bytes(case_bytes)
        for ordinal, case in enumerate(cases, 1):
            directory = args.out / f"case-{ordinal:02d}"
            directory.mkdir()
            result = {
                "ordinal": ordinal,
                "id": case["id"],
                "status": "running",
                "server_number": 1,
                "server_request_ordinal": None,
                "pilot_request_ordinal": ordinal,
            }
            summary["results"].append(result)
            atomic_json(args.out / "summary.json", summary)
            result.update(
                evaluate(
                    args.endpoint,
                    request_body(args.model_id, case),
                    directory,
                    args.timeout_seconds,
                )
            )
            result["checks"] = frozen.check_answer(case, result["answer"])
            result["prefill_plan"] = None
            result["selected_prefill_rows"] = 1
            result["selected_prefill_rows_observability"] = (
                "native per-response single_token_forward descriptor, not GLM grouped prefill"
            )
            atomic_json(directory / "result.json", result)
            atomic_json(args.out / "summary.json", summary)
        current, baseline = rates(summary), baseline_rates
        summary["comparison_to_maya"] = {
            "v41": current,
            "maya": baseline,
            "ratios_v41_over_maya": {
                key: current[key] / baseline[key]
                if current[key] is not None and baseline[key]
                else None
                for key in (
                    "weighted_prefill_tps",
                    "weighted_generation_tps",
                    "request_seconds_total",
                )
            },
        }
        summary["status"] = "completed"
    except Exception as error:
        summary["status"] = "aborted"
        summary["error_type"] = type(error).__name__
        if isinstance(error, frozen.InvalidRun):
            summary["error"] = str(error)
        if summary["results"] and summary["results"][-1]["status"] == "running":
            summary["results"][-1]["status"] = "error"
    except BaseException:
        summary["status"] = "interrupted"
        raise
    finally:
        atomic_json(args.out / "summary.json", summary)
    return 0 if summary["status"] == "completed" else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(
            f"Pilot refused input ({type(error).__name__}); no server was started or stopped."
        )
        raise SystemExit(1)
