#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Private, fresh-process native DFlash replay; no lease wrapper or tuning knobs.

Run with uv run --script scripts/bench/dflash_native_replay.py --help.
Only messages are copied from the source; native qwen run owns rendering and
sampling defaults other than temperature=1 and seed=42. Build/template checks
use completed request-stats v2, not qwen-bench's different identity envelope.
The source may be a messages array or an object containing messages. The first
14 messages (configurable) must end in user. Serial/spec warmups have cap 16;
measured runs are serial/spec/spec/serial with cap 256 (configurable).
SIGINT/SIGTERM stop the owned child and persist failure evidence. SIGKILL/power
loss can only leave the last durable 'running' record, never a success claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone


FAMILIES = {
    "flash-next": ("qwen4exp", "qwen38"),
    "glm-flash": ("glm5_next", "glm53_flash_chat_v2"),
}
ENV_ALLOWLIST = {"HOME", "PATH", "TMPDIR", "USER", "LOGNAME", "LANG", "LC_ALL"}
FIXED_ENV = {
    "MTL_DEBUG_LAYER": "1",
    "QWEN_METAL_LEASE_WAIT": "1",
    "RUST_LOG": "warn,qwen_diag=info",
}
COUNTERS = {
    "flash-next": set(
        "emitted_tokens packets draft_calls proposed_tokens verified_inputs "
        "accepted_drafts retained_inputs replay_inputs restores "
        "full_retention_packets target_draws proposal_draws".split()
    ),
    "glm-flash": set(
        "prefill_tokens attempts packets draft_calls drafted_tokens verified_rows "
        "serial_verifications anchor_only_packets accepted_drafts retained_inputs "
        "replayed_inputs full_commits output_tokens callback_tokens target_draws "
        "proposal_draws cancellations failures".split()
    ),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def strict_json(text):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def constant(value):
        raise ValueError(f"non-finite JSON constant: {value}")

    return json.loads(text, object_pairs_hook=pairs, parse_constant=constant)


def json_bytes(value):
    return (
        json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False) + "\n"
    ).encode()


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def private_write(path, data):
    with path.open("xb") as stream:
        os.fchmod(stream.fileno(), 0o600)
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def save_manifest(path, value):
    fd, name = tempfile.mkstemp(prefix=".manifest-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(json_bytes(value))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def freeze_messages(source_bytes, count):
    require(count > 0, "messages-count must be positive")
    source = strict_json(source_bytes)
    messages = source.get("messages") if isinstance(source, dict) else source
    require(
        isinstance(messages, list) and len(messages) >= count,
        f"source must contain at least {count} messages",
    )
    selected = messages[:count]
    require(
        all(
            isinstance(item, dict) and isinstance(item.get("role"), str)
            for item in selected
        ),
        "messages must be objects with roles",
    )
    require(
        selected[-1]["role"] == "user", "frozen prefix must end with a user message"
    )
    return json_bytes({"messages": selected})


def child_environment(ambient):
    env = {key: value for key, value in ambient.items() if key in ENV_ALLOWLIST}
    env.update(FIXED_ENV)
    ignored = sorted(set(ambient) - ENV_ALLOWLIST - FIXED_ENV.keys())
    return env, ignored


def extract_counters(stderr, family):
    # These are CLI stderr summaries, not a stable JSON diagnostics schema:
    # qwen/qwen4exp/dflash.rs::emit_stats and qwen/glm5_next/dflash.rs::generate.
    label = FAMILIES[family][0]
    lines = [
        line for line in stderr.splitlines() if line.startswith(f"{label} dflash: ")
    ]
    require(len(lines) <= 1, "ambiguous duplicate DFlash summaries")
    if not lines:
        return {}
    text = lines[0].split(" dflash: ", 1)[1]
    if family == "glm-flash":
        match = re.search(r"counters=Glm5NextDFlashCounters \{ (.*) \}$", text)
        require(match is not None, "unrecognized GLM DFlash counters format")
        entries = re.findall(r"(?:^|, )([a-z_]+): ([0-9]+)(?=, |$)", match[1])
    else:
        entries = re.findall(r"(?:^| )([a-z_]+)=([0-9]+)(?= |$)", text)
    result = {}
    for key, value in entries:
        if key in COUNTERS[family]:
            require(key not in result, f"duplicate counter: {key}")
            result[key] = int(value)
    return result


def number(value, label, positive=False):
    require(
        type(value) in (int, float)
        and math.isfinite(value)
        and (value > 0 if positive else value >= 0),
        f"invalid {label}: {value!r}",
    )
    return value


def validate_stats(path, args, cap):
    rows = [
        strict_json(line) for line in path.read_bytes().splitlines() if line.strip()
    ]
    require(
        len(rows) == 1 and isinstance(rows[0], dict),
        "expected exactly one stats record",
    )
    row = rows[0]
    for key in (
        "build",
        "model",
        "input",
        "usage",
        "finish",
        "timing_ms",
        "throughput_tps",
    ):
        require(
            isinstance(row.get(key), dict), f"missing or invalid stats.{key} object"
        )
    require(
        row.get("schema") == "qwen-llm.request-stats"
        and type(row.get("schema_version")) is int
        and row.get("schema_version") == 2
        and row.get("record_type") == "request_stats"
        and row.get("status") == "ok",
        "expected successful request-stats v2",
    )
    build = row.get("build", {})
    require(
        build.get("commit") == args.expected_commit and build.get("dirty") is False,
        "build commit mismatch or build is not explicitly clean",
    )
    label, template = FAMILIES[args.family]
    require(row.get("model", {}).get("family") == label, "stats model family mismatch")
    require(
        row.get("input", {}).get("kind") == "messages"
        and row["input"].get("template") == template,
        "native template mismatch",
    )
    usage = row.get("usage", {})
    for key in ("input_tokens", "output_tokens"):
        require(type(usage.get(key)) is int and usage[key] > 0, f"invalid usage.{key}")
    require(usage["output_tokens"] <= cap, "output exceeds requested cap")
    reason = row.get("finish", {}).get("reason")
    require(reason in ("eos", "token_limit"), "unknown finish reason")
    require(
        reason != "token_limit" or usage["output_tokens"] == cap,
        "token_limit output does not match requested cap",
    )
    metrics = dict(usage)
    for key in ("total", "tokenization", "prefill", "decode"):
        metrics[f"{key}_ms"] = number(
            row.get("timing_ms", {}).get(key), f"timing_ms.{key}"
        )
    for key in ("prefill", "decode"):
        metrics[f"{key}_tps"] = number(
            row.get("throughput_tps", {}).get(key),
            f"throughput_tps.{key}",
            positive=True,
        )
    return row, metrics


def command(args, frozen, stats, mode, cap):
    cmd = [
        str(args.binary),
        "run",
        "--model",
        str(args.model),
        "--messages",
        str(frozen),
        "--max-tokens",
        str(cap),
        "--temp",
        "1",
        "--seed",
        "42",
        "--request-stats-jsonl",
        str(stats),
    ]
    if mode == "spec":
        cmd += ["--drafter", str(args.drafter)]
    return cmd


def stop_child(child):
    if child.poll() is None:
        child.terminate()
        # Native shutdown waits for GPU completion; forced exit can strand wired memory.
        child.wait()


def execute_run(record, manifest, args, output, env, check_inputs):
    path = output / "manifest.json"
    child = None
    started = time.monotonic()
    try:
        record.update(status="running", started_at=utc_now())
        manifest["runs"].append(record)
        save_manifest(path, manifest)
        check_inputs()
        for name in ("stdout", "stderr", "stats"):
            private_write(output / record[name], b"")
        with (
            (output / record["stdout"]).open("ab") as stdout,
            (output / record["stderr"]).open("ab") as stderr,
        ):
            process_started = time.monotonic()
            try:
                child = subprocess.Popen(
                    record["command"],
                    stdout=stdout,
                    stderr=stderr,
                    stdin=subprocess.DEVNULL,
                    env=env,
                    cwd=output,
                    start_new_session=True,
                )
                record["pid"] = child.pid
                save_manifest(path, manifest)
                record["returncode"] = child.wait()
            finally:
                if child is not None:
                    stop_child(child)
                    record["returncode"] = child.returncode
                record["process_elapsed_seconds"] = time.monotonic() - process_started
                for stream in (stdout, stderr):
                    stream.flush()
                    os.fsync(stream.fileno())
        require(record["returncode"] == 0, f"child exited {record['returncode']}")
        check_inputs()
        row, metrics = validate_stats(
            output / record["stats"], args, record["output_cap"]
        )
        counters = extract_counters(
            (output / record["stderr"]).read_text(errors="replace"), args.family
        )
        record["counters"] = counters
        require(
            record["mode"] != "spec" or counters.get("draft_calls", 0) > 0,
            "spec run has no positive native draft_calls counter",
        )
        require(
            record["mode"] != "serial" or not counters,
            "serial run reported DFlash counters",
        )
        record.update(
            status="succeeded",
            metrics=metrics,
            finish=row["finish"],
            build=row["build"],
            input=row["input"],
            output_fingerprint=row.get("output_fingerprint"),
        )
    except BaseException as error:
        record.update(
            status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            error=f"{type(error).__name__}: {error}",
        )
        raise
    finally:
        record.update(elapsed_seconds=time.monotonic() - started, finished_at=utc_now())
        record["artifact_sha256"] = {}
        for name in ("stdout", "stderr", "stats"):
            artifact = output / record[name]
            if artifact.is_file():
                with artifact.open("rb") as stream:
                    os.fsync(stream.fileno())
                record["artifact_sha256"][name] = sha256(artifact)
        save_manifest(path, manifest)


def aggregate(runs):
    means = {}
    for mode in ("serial", "spec"):
        rows = [
            run for run in runs if run["phase"] == "measured" and run["mode"] == mode
        ]
        require(
            len(rows) == 2 and all(row["status"] == "succeeded" for row in rows),
            "aggregation requires all four successful measured runs",
        )
        means[mode] = {
            key: statistics.mean(row["metrics"][key] for row in rows)
            for key in rows[0]["metrics"]
        }
        means[mode]["process_elapsed_seconds"] = statistics.mean(
            row["process_elapsed_seconds"] for row in rows
        )
    ratios = {}
    for key in means["serial"]:
        denominator = means["serial"][key]
        ratios[key] = means["spec"][key] / denominator if denominator else None
    return {
        "arithmetic_means": means,
        "spec_over_serial": ratios,
        "notes": [
            "Warmups excluded; ratios divide means, not paired observations.",
            "TPS ratios above 1 favor spec; time ratios below 1 indicate lower observed latency.",
            "Differing continuations or lengths prevent an equal-work speedup interpretation.",
            "EOS may yield unequal output lengths; tokens are not forced to the cap.",
            "Request timing excludes model load; process elapsed includes startup and lease wait.",
            "Native sampled packed verification need not reproduce serial output tokens.",
        ],
    }


def run(args):
    require(
        re.fullmatch(r"[0-9a-f]{40}", args.expected_commit) is not None,
        "expected-commit must be a full lowercase 40-character Git commit",
    )
    require(args.output_cap > 0, "output-cap must be positive")
    output = args.output_dir.absolute()
    require(
        not os.path.lexists(output), "output-dir must not exist (including symlinks)"
    )
    output = output.resolve()
    for name in ("binary", "source", "model", "drafter"):
        path = getattr(args, name).absolute()
        resolved = path.resolve(strict=True)
        require(path.is_file(), f"{name} must be a regular file")
        require(
            resolved != output and output not in resolved.parents,
            f"{name}/output collision",
        )
        # Split GGUF discovery depends on the supplied shard name, not a symlink's backing name.
        setattr(args, name, path)
    require(os.access(args.binary, os.X_OK), "binary must be executable")
    source_bytes = args.source.read_bytes()
    frozen_bytes = freeze_messages(source_bytes, args.messages_count)
    env, ignored = child_environment(os.environ)
    frozen = output / "messages.json"
    hashes = {
        args.source: hashlib.sha256(source_bytes).hexdigest(),
        args.binary: sha256(args.binary),
        frozen: hashlib.sha256(frozen_bytes).hexdigest(),
    }
    assets = {
        name: {
            "path": str(getattr(args, name)),
            "bytes": getattr(args, name).stat().st_size,
            "mtime_ns": getattr(args, name).stat().st_mtime_ns,
        }
        for name in ("model", "drafter")
    }

    def check_inputs():
        for file, digest in hashes.items():
            require(sha256(file) == digest, f"input changed: {file}")
        for asset in assets.values():
            stat = Path(asset["path"]).stat()
            require(
                (stat.st_size, stat.st_mtime_ns) == (asset["bytes"], asset["mtime_ns"]),
                f"model/drafter metadata changed: {asset['path']}",
            )

    manifest = {
        "schema": "dflash-native-replay-v1",
        "runner_sha256": sha256(Path(__file__)),
        "python_version": sys.version,
        "platform": sys.platform,
        "status": "running",
        "created_at": utc_now(),
        "family": args.family,
        "expected_commit": args.expected_commit,
        "expected_template": FAMILIES[args.family][1],
        "messages_count": args.messages_count,
        "output_cap": args.output_cap,
        "temperature": 1,
        "seed": 42,
        "other_options": "native defaults",
        "assets": assets,
        "input_sha256": {str(path): digest for path, digest in hashes.items()},
        "environment_policy": "allowlist plus fixed overrides; all other ambient values dropped",
        "ignored_ambient_names": ignored,
        "environment": env,
        "caveats": [
            "Build identity is checked after each completed request, including warmups.",
            "Model/drafter have path/size/mtime identity only, not content hashes.",
            "DFlash counters are best-effort known integer stderr fields, not JSON stats.",
            "SIGKILL/power loss may leave running records and partial artifacts.",
        ],
        "runs": [],
    }
    old_umask = os.umask(0o077)
    try:
        output.mkdir(mode=0o700)
        try:
            save_manifest(output / "manifest.json", manifest)
            private_write(frozen, frozen_bytes)
            schedule = [("warmup", "serial", 16), ("warmup", "spec", 16)]
            schedule += [
                ("measured", mode, args.output_cap)
                for mode in ("serial", "spec", "spec", "serial")
            ]
            for index, (phase, mode, cap) in enumerate(schedule, 1):
                stem = f"{index:02d}-{phase}-{mode}"
                record = {
                    "id": stem,
                    "phase": phase,
                    "mode": mode,
                    "output_cap": cap,
                    "stdout": f"{stem}.stdout",
                    "stderr": f"{stem}.stderr",
                    "stats": f"{stem}.stats.jsonl",
                    "environment": env,
                    "cwd": str(output),
                }
                record["command"] = command(
                    args, frozen, output / record["stats"], mode, cap
                )
                execute_run(record, manifest, args, output, env, check_inputs)
            manifest["aggregate"] = aggregate(manifest["runs"])
            manifest["status"] = "succeeded"
        except BaseException as error:
            manifest.update(
                status="interrupted"
                if isinstance(error, KeyboardInterrupt)
                else "failed",
                error=f"{type(error).__name__}: {error}",
            )
            raise
        finally:
            manifest["finished_at"] = utc_now()
            save_manifest(output / "manifest.json", manifest)
    finally:
        os.umask(old_umask)
    return manifest


def parser():
    result = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    for name in ("binary", "source", "output-dir", "model", "drafter"):
        result.add_argument(f"--{name}", required=True, type=Path)
    result.add_argument(
        "--expected-commit",
        required=True,
        help="full lowercase Git commit (40 hex characters)",
    )
    result.add_argument("--family", required=True, choices=FAMILIES)
    result.add_argument(
        "--messages-count",
        type=int,
        default=14,
        help="exact prefix length; must end in user",
    )
    result.add_argument(
        "--output-cap",
        type=int,
        default=256,
        help="measured output token cap; warmups always use 16",
    )
    return result


def main(argv=None):
    args = parser().parse_args(argv)

    def interrupt(signum, _frame):
        # A second signal must not interrupt child reaping or manifest persistence.
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, signal.SIG_IGN)
        raise KeyboardInterrupt(f"signal {signum}")

    previous = {
        sig: signal.signal(sig, interrupt) for sig in (signal.SIGINT, signal.SIGTERM)
    }
    try:
        result = run(args)
        print(json.dumps(result["aggregate"], indent=2))
        return 0
    except KeyboardInterrupt as error:
        print(
            f"Interrupted: {error}; inspect output-dir/manifest.json", file=sys.stderr
        )
        return 130
    except Exception as error:
        print(
            f"Failed: {error}; inspect output-dir/manifest.json if created",
            file=sys.stderr,
        )
        return 1
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    raise SystemExit(main())
