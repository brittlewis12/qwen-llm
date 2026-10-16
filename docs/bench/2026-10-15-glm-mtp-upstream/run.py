# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Parent must hold scripts/bench/with-metal-lease for this entire invocation.

No admission, downloads, builds, or extra warm requests here.
--prompt is a JSON array of exactly 32768 nonnegative integer token IDs.
--artifact-manifest requires vocab_size, header_bytes (1..16777216), and
modelshards: [{path (absolute), size, mtime_ns, header_sha256}].
RSS is sampled, not an exact peak. Memory admission/pressure policy belongs to parent.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
import uuid


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def capture(argv):
    p = subprocess.run(argv, capture_output=True, text=True, timeout=15)
    if p.returncode:
        raise RuntimeError(f"{argv}: {p.stderr}")
    return p.stdout


def snapshot(server):
    """Resolve non-system Mach-O dependencies; refuse unresolved loader paths."""
    pending, seen = [(server, ())], {}
    while pending:
        path, inherited = pending.pop()
        path = path.resolve(strict=True)
        if str(path) in seen:
            continue
        seen[str(path)] = sha(path)
        load = capture(["otool", "-l", str(path)]).splitlines()
        rpaths = list(inherited)
        for i, line in enumerate(load):
            if line.strip() == "cmd LC_RPATH":
                rpaths.append(load[i + 2].strip().split(" (offset")[0][5:])

        def expand(s):
            return s.replace("@loader_path", str(path.parent)).replace(
                "@executable_path", str(server.parent)
            )

        rpaths = [expand(r) for r in rpaths]
        for line in capture(["otool", "-L", str(path)]).splitlines()[1:]:
            dep = line.strip().split(" (compatibility")[0]
            if dep.startswith(("/usr/lib/", "/System/Library/")):
                continue
            candidates = (
                [Path(r) / dep[7:] for r in rpaths]
                if dep.startswith("@rpath/")
                else [Path(expand(dep))]
            )
            resolved = next((p for p in candidates if p.is_file()), None)
            if resolved is None:
                raise RuntimeError(f"Unresolved library {dep} from {path}")
            pending.append((resolved, tuple(rpaths)))
    return seen


def environment():
    # Refuse rather than persist potentially secret override values.
    names = sorted(
        k for k in os.environ if k.startswith(("LLAMA_ARG_", "GGML_", "MTL_", "DYLD_"))
    )
    if names:
        raise RuntimeError(
            "Refusing inherited overrides (names only): " + ", ".join(names)
        )
    return dict(os.environ)


def token_ids(tokens, vocab_size, count):
    return (
        isinstance(tokens, list)
        and len(tokens) == count
        and all(type(t) is int and 0 <= t < vocab_size for t in tokens)
    )


def shard_snapshot(manifest, model):
    vocab, header = manifest.get("vocab_size"), manifest.get("header_bytes")
    shards = manifest.get("modelshards")
    if (
        type(vocab) is not int
        or vocab <= 0
        or type(header) is not int
        or not 0 < header <= 16777216
        or not isinstance(shards, list)
        or not shards
    ):
        raise ValueError("Invalid artifact manifest vocabulary/header/modelshards")
    observed = []
    for shard in shards:
        path = Path(shard["path"])
        if not path.is_absolute() or not path.is_file():
            raise ValueError("Shard paths must be absolute existing files")
        stat = path.stat()
        header_size = shard.get("header_bytes", header)
        if type(header_size) is not int or not 0 < header_size <= 16777216:
            raise ValueError("Invalid per-shard header size")
        with path.open("rb") as stream:
            digest = hashlib.sha256(stream.read(header_size)).hexdigest()
        observed.append(
            {
                "path": str(path),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "header_sha256": digest,
            }
        )
    if str(model) not in [s["path"] for s in observed]:
        raise ValueError("--model is not a shard in the artifact manifest")
    return observed


def shards_match(manifest, observed):
    return observed == [
        {k: s[k] for k in ("path", "size", "mtime_ns", "header_sha256")}
        for s in manifest["modelshards"]
    ]


def memory():
    return {
        "vm_stat": capture(["vm_stat"]),
        "swapusage": capture(["sysctl", "vm.swapusage"]),
    }


def stop(proc):
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=20)
    else:
        proc.wait()


def monitor(proc, done, samples):
    while not done.is_set() and proc.poll() is None:
        try:
            rss = capture(["ps", "-p", str(proc.pid), "-o", "rss="])
            samples.append({"monotonic": time.monotonic(), "rss_kib": int(rss)})
        except (ValueError, RuntimeError, subprocess.TimeoutExpired):
            pass
        if len(samples) >= 600:
            return
        done.wait(5)


def http(port, route, timeout, payload=None):
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/{route}",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    # Never route localhost through an inherited proxy.
    with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(
        request, timeout=timeout
    ) as response:
        return response.read()


def command(args, arm, port, alias):
    return [
        str(args.server),
        "--model",
        str(args.model),
        "--alias",
        alias,
        "--spec-type",
        "none" if arm.startswith("A") else "draft-mtp",
        "--spec-draft-n-max",
        "1",
        "-ngl",
        "99",
        "--fit",
        "off",
        "-c",
        "33024",
        "-np",
        "1",
        "-b",
        "512",
        "-ub",
        "512",
        "-ctk",
        "f16",
        "-ctv",
        "f16",
        "--no-mmproj",
        "--offline",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--cache-ram",
        "0",
        "--ctx-checkpoints",
        "0",
        "--no-cache-prompt",
        "--no-webui",
        "--warmup",
        "-fa",
        "on",
        "--no-context-shift",
        "--spec-draft-type-k",
        "f16",
        "--spec-draft-type-v",
        "f16",
        "--spec-draft-n-ubatch",
        "512",
        "--spec-draft-sampling",
        "greedy",
    ]


def assess(response, arm, vocab_size):
    tokens, text = response.get("tokens"), response.get("content")
    valid = token_ids(tokens, vocab_size, 128) and isinstance(text, str)
    timings = response.get("timings") or {}
    draft, accepted = timings.get("draft_n"), timings.get("draft_n_accepted")
    # server_slot_stats::to_json omits both counters when n_draft_tokens == 0.
    omitted_zero = arm.startswith("A") and not any(
        key in timings for key in ("draft_n", "draft_n_accepted")
    )
    if omitted_zero:
        draft, accepted = 0, 0
    timing_valid = all(
        type(timings.get(k)) in (int, float)
        and math.isfinite(timings[k])
        and timings[k] > 0
        for k in ("prompt_ms", "predicted_ms")
    )
    draft_valid = (
        type(draft) is int
        and type(accepted) is int
        and 0 <= accepted <= draft
        and (draft == 0 if arm.startswith("A") else draft > 0)
    )
    result = {
        "fixed_work": bool(
            valid
            and len(tokens) == 128
            and response.get("tokens_predicted") == 128
            and response.get("tokens_evaluated") == 32768
            and not response.get("truncated", False)
            and not response.get("error")
        ),
        "short_output": not valid or len(tokens) < 128,
        "text_sha256": hashlib.sha256(text.encode()).hexdigest()
        if isinstance(text, str)
        else None,
        "timings": response.get("timings"),
        "stop_type": response.get("stop_type"),
        "draft_counters_omitted_zero": omitted_zero,
        "effective_draft_n": draft,
        "effective_draft_n_accepted": accepted,
    }
    result["measurement_valid"] = bool(
        result["fixed_work"]
        and timing_valid
        and draft_valid
        and type(timings.get("cache_n")) is int
        and timings["cache_n"] == 0
        and type(timings.get("prompt_n")) is int
        and timings["prompt_n"] == 32768
        and type(timings.get("predicted_n")) is int
        and timings["predicted_n"] == 128
    )
    return result


def assess_probe(response, vocab_size):
    cache = (response.get("timings") or {}).get("cache_n")
    return {
        "valid": bool(
            token_ids(response.get("tokens"), vocab_size, 3)
            and isinstance(response.get("content"), str)
            and response.get("tokens_predicted") == 3
            and not response.get("truncated", False)
            and not response.get("error")
            and type(cache) is int
            and 32768 <= cache <= 32896
        ),
        "cache_n": cache,
        "tokens": response.get("tokens"),
        "content": response.get("content"),
    }


def completion(proc, port, payload, directory, prefix, result, timeout):
    if proc.poll() is not None:
        raise RuntimeError("Owned server exited before completion")
    (directory / f"{prefix}request.json").write_bytes(payload)
    begin = time.monotonic()
    watchdog = threading.Timer(timeout, stop, args=(proc,))
    watchdog.daemon = True
    watchdog.start()
    try:
        try:
            raw = http(port, "completion", timeout, payload)
        except urllib.error.HTTPError as error:
            (directory / f"{prefix}response.json").write_bytes(error.read())
            raise
        (directory / f"{prefix}response.json").write_bytes(raw)
        if proc.poll() is not None:
            raise RuntimeError("Owned server exited during completion")
        return json.loads(raw)
    finally:
        elapsed = time.monotonic() - begin
        watchdog.cancel()
        watchdog.join()
        result[f"{prefix}request_seconds"] = elapsed
        save(directory / f"{prefix}time.json", {"request_seconds": elapsed})


def run_arm(args, arm, payload, env):
    directory = args.out / arm
    directory.mkdir()
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    alias = "glm-bench-" + uuid.uuid4().hex
    argv = command(args, arm, port, alias)
    save(directory / "command.json", argv)
    (directory / "request.json").write_bytes(payload)
    save(directory / "memory-before.json", memory())
    result, samples, done = {"arm": arm}, [], threading.Event()
    proc = thread = None
    started = time.monotonic()
    try:
        with (directory / "server.log").open("wb") as log:
            proc = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, env=env)
            thread = threading.Thread(
                target=monitor, args=(proc, done, samples), daemon=True
            )
            thread.start()
            deadline = started + 600
            while True:
                if proc.poll() is not None:
                    raise RuntimeError(f"Owned server exited: {proc.returncode}")
                if time.monotonic() >= deadline:
                    raise TimeoutError("Startup exceeded 600 seconds")
                try:
                    raw = http(port, "props", min(2, deadline - time.monotonic()))
                except (urllib.error.URLError, TimeoutError, OSError):
                    time.sleep(0.25)
                    continue
                props = json.loads(raw)
                if props.get("model_alias") != alias:
                    raise RuntimeError(
                        "Port ownership nonce mismatch; refusing completion"
                    )
                if props.get("model_path") != str(args.model):
                    raise RuntimeError("Model identity mismatch")
                health = http(port, "health", 2)
                if json.loads(health).get("status") != "ok":
                    time.sleep(0.25)
                    continue
                (directory / "props.json").write_bytes(raw)
                (directory / "health.json").write_bytes(health)
                break
            result["startup_seconds"] = time.monotonic() - started
            response = completion(proc, port, payload, directory, "", result, 1800)
            result.update(assess(response, arm, args.vocab_size))
            if result["measurement_valid"]:
                probe = json.loads(payload)
                probe.update(
                    prompt=probe["prompt"] + response["tokens"],
                    cache_prompt=True,
                    id_slot=0,
                    n_predict=3,
                )
                response = completion(
                    proc,
                    port,
                    json.dumps(probe).encode(),
                    directory,
                    "probe-",
                    result,
                    180,
                )
                result["probe"] = assess_probe(response, args.vocab_size)
    except BaseException as error:
        result["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        done.set()
        if proc is not None:
            stop(proc)
        if thread is not None:
            thread.join(timeout=16)
        result["rss_samples"] = samples
        result["sampled_peak_rss_kib"] = max(
            (s["rss_kib"] for s in samples), default=None
        )
        save(directory / "result.json", result)
        save(directory / "memory-after.json", memory())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("server", "model", "prompt", "out", "artifact-manifest"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--arms", choices=("ABBA", "B"), default="ABBA")
    args = parser.parse_args()
    for path in (args.server, args.model):
        if not path.is_absolute() or not path.is_file():
            parser.error(
                "--server and --model must be absolute existing files (model shard 1)"
            )
    raw_prompt = args.prompt.read_bytes()
    tokens = json.loads(raw_prompt)
    artifacts = json.loads(args.artifact_manifest.read_bytes())
    observed = shard_snapshot(artifacts, args.model)
    if not shards_match(artifacts, observed):
        parser.error("Model shard identity differs from parent manifest")
    args.vocab_size = artifacts["vocab_size"]
    if not token_ids(tokens, args.vocab_size, 32768):
        parser.error("--prompt must contain exactly 32768 IDs in manifest vocabulary")
    env = environment()
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "prompt.json").write_bytes(raw_prompt)
    save(
        args.out / "manifest.json",
        {
            "prompt_sha256": hashlib.sha256(raw_prompt).hexdigest(),
            "script_sha256": sha(Path(__file__)),
            "artifact_manifest": artifacts,
            "config_by_arm": {
                arm: command(args, arm, "dynamic", "nonce")
                for arm in ("A1", "B1", "B2", "A2")
            },
            "override_policy": "refuse LLAMA_ARG_ / GGML_ / MTL_ / DYLD_; record names only",
            "warmup": "native --warmup only; included in startup, not request",
            "lease": "external parent responsibility",
            "promotion": False,
        },
    )
    baseline = snapshot(args.server)
    save(args.out / "artifacts-before.json", baseline)
    payload = json.dumps(
        {
            "prompt": tokens,
            "temperature": 0,
            "seed": 1234,
            "n_predict": 128,
            "cache_prompt": False,
            "return_tokens": True,
            "stream": False,
            "id_slot": 0,
        }
    ).encode()
    results = []
    try:
        for arm in ["A1", "B1", "B2", "A2"] if args.arms == "ABBA" else ["B1"]:
            before = snapshot(args.server)
            save(args.out / f"{arm}-artifacts-before.json", before)
            shards = shard_snapshot(artifacts, args.model)
            save(args.out / f"{arm}-shards-before.json", shards)
            if not shards_match(artifacts, shards):
                raise RuntimeError("Model shard mutation before arm")
            if before != baseline:
                raise RuntimeError("Binary/library mutation before arm")
            try:
                results.append(run_arm(args, arm, payload, env))
                if not results[-1]["measurement_valid"]:
                    raise RuntimeError(
                        "Invalid fixed work/timing/draft evidence; stopping schedule"
                    )
            finally:
                shards = shard_snapshot(artifacts, args.model)
                save(args.out / f"{arm}-shards-after.json", shards)
                after = snapshot(args.server)
                save(args.out / f"{arm}-artifacts-after.json", after)
                if after != baseline:
                    raise RuntimeError("Binary/library mutation during arm")
                if not shards_match(artifacts, shards):
                    raise RuntimeError("Model shard mutation during arm")
    finally:
        responses = [
            json.loads((args.out / r["arm"] / "response.json").read_bytes())
            for r in results
        ]
        pairs = [
            {
                "arms": [results[i]["arm"], results[j]["arm"]],
                "tokens_equal": a.get("tokens") == b.get("tokens"),
                "text_equal": a.get("content") == b.get("content"),
            }
            for i, a in enumerate(responses)
            for j, b in enumerate(responses)
            if i < j
        ]
        for pair in pairs:
            left, right = [
                next(r for r in results if r["arm"] == arm) for arm in pair["arms"]
            ]
            a, b = left.get("probe", {}), right.get("probe", {})
            eligible = bool(
                pair["tokens_equal"]
                and pair["text_equal"]
                and a.get("valid")
                and b.get("valid")
            )
            pair["continuation_comparable"] = eligible
            pair["continuation_equal"] = (
                (a["tokens"] == b["tokens"] and a["content"] == b["content"])
                if eligible
                else None
            )
        save(
            args.out / "summary.json",
            {
                "results": results,
                "comparisons": pairs,
                "all_fixed_work": bool(results)
                and all(r["fixed_work"] for r in results),
                "complete_schedule": len(results) == (4 if args.arms == "ABBA" else 1),
                "all_measurements_valid": bool(results)
                and all(r["measurement_valid"] for r in results),
                "promotion": False,
                "correctness": "Only eligible pairs compare 3-token live-cache continuations; no general certification",
            },
        )


if __name__ == "__main__":

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    main()
