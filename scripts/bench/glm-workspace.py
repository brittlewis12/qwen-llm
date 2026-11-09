#!/usr/bin/env python3
"""One real-product GLM workspace arm; build first, run under with-metal-lease.

Evidence is the committed tree plus this command. Temporary responses, logs
and prompts are removed at exit; stdout retains compact measurements only.
Run ABBA as separate short leases, cooling between arms. No build lock is
taken by timing. Model discovery uses QWEN_MODEL_SEARCH_PATH.
"""

import argparse
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from model_search import resolve_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workload", choices=["run", "serve"])
    parser.add_argument("--release", choices=["on", "off"], required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--engine-commit", required=True,
                        help="commit used to build the existing executable; driver identity is separate")
    args = parser.parse_args()
    fd = int(os.environ["QWEN_METAL_LEASE_FD"])
    os.fstat(fd)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if not re.fullmatch(r"[0-9a-f]{7,64}", args.engine_commit):
        raise ValueError("engine commit must be a hex object id")
    engine_commit = subprocess.check_output(
        ["git", "rev-parse", "--verify", args.engine_commit + "^{commit}"], cwd=ROOT, text=True).strip()
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT):
        raise RuntimeError("commit the measured tree first")
    model = resolve_model("GLM-5.3-Flash-Maya-M-IQ2_S-00001-of-00003.gguf")
    binary = ROOT / "target/release/qwen"
    tok = ROOT / "target/release/qwen-tok"
    env = dict(os.environ, MTL_DEBUG_LAYER="0")
    env.pop("QWEN_GLM53_RELEASE_PREFILL", None)
    if args.release == "off":
        env["QWEN_GLM53_RELEASE_PREFILL"] = "0"

    def cut(text, count):
        encoded = subprocess.check_output([str(tok), "-m", str(model), "--file", "-", "--ids"],
                                          input=text, text=True, env=env, pass_fds=(fd,))
        ids = encoded.splitlines()[1:]
        if len(ids) < count:
            raise RuntimeError("corpus too short")
        return subprocess.check_output([str(tok), "-m", str(model), "--decode-ids", "-"],
                                       input="\n".join(ids[:count]), text=True, env=env, pass_fds=(fd,))

    corpus = "Review these Rust inference paths for memory lifetime bugs.\n" + "\n".join(
        (ROOT / name).read_text() for name in [
            "crates/qwen-cli/src/serve/backend_glm5_next.rs",
            "crates/qwen-llm/src/glm5_next_metal.rs"])
    prompt = cut(corpus, 8192)
    append = cut("Explain the memory admission and snapshot behavior of this code.\n" +
                 (ROOT / "crates/qwen-llm/src/glm5_next_metal/snapshot.rs").read_text(), 512)
    thermals = []
    compiler_samples = []
    decode_ends = []
    seen_ends = set()
    log = None
    sample_lock = threading.Lock()
    stop = threading.Event()

    def thermal():
        value = subprocess.check_output(
            ["notifyutil", "-g", "com.apple.system.thermalpressurelevel"], text=True).strip()
        thermals.append(int(value.split()[-1]))
        return thermals[-1]

    def sample():
        with sample_lock:
            thermal()
            rows = subprocess.check_output(["ps", "-axo", "rss=,comm="], text=True)
            rss = sum(int(parts[0]) * 1024 for line in rows.splitlines()
                      if len(parts := line.split(maxsplit=1)) == 2
                      and Path(parts[1]).name.strip("()") == "rustc")
            now = time.time()
            compiler_samples.append((now, rss))
            if log is not None and log.exists():
                for line in log.read_text().splitlines():
                    if ("glm5_next stats:" in line or "serve phases:" in line) and line not in seen_ends:
                        seen_ends.add(line)
                        duration = re.search(r"(?:generation|decode)_ms=([0-9.]+)", line)
                        if duration:
                            decode_ends.append((now, float(duration[1])))

    def monitor():
        while not stop.wait(1):
            sample()

    if thermal() != 0:
        raise RuntimeError("nonzero thermal pressure before arm")
    watcher = threading.Thread(target=monitor)
    watcher.start()
    child = None
    result = {"commit": engine_commit, "harness_commit": commit,
              "command": [str(Path(__file__).relative_to(ROOT)), *sys.argv[1:]],
              "model": model.name, "workload": args.workload, "release": args.release,
              "label": args.label}
    try:
        with tempfile.TemporaryDirectory(prefix="glm-workspace-") as temporary:
            tmp = Path(temporary)
            log = tmp / "diagnostics.txt"
            if args.workload == "run":
                stats = tmp / "stats.jsonl"
                with log.open("w") as diagnostics:
                    child = subprocess.Popen([str(binary), "run", "-m", str(model), "--user", "-",
                        "--max-tokens", "128", "--temp", "0", "--seed", "1",
                        "--max-context-tokens", "10000", "--request-stats-jsonl", str(stats)],
                        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=diagnostics,
                        text=True, env=env, pass_fds=(fd,))
                    child.communicate(prompt, timeout=600)
                if child.returncode:
                    raise RuntimeError(log.read_text()[-8000:])
                result["stats"] = json.loads(stats.read_text())
                if result["stats"]["build"]["commit"] != engine_commit:
                    raise RuntimeError("CLI build identity differs from --engine-commit")
            else:
                with socket.socket() as reserve:
                    reserve.bind(("127.0.0.1", 0))
                    port = reserve.getsockname()[1]
                address = f"http://127.0.0.1:{port}"
                with log.open("w") as diagnostics:
                    child = subprocess.Popen([str(binary), "serve", "-m", str(model), "--addr",
                        f"127.0.0.1:{port}", "--max-tokens", "128", "--max-context-tokens", "10000",
                        "--durable-snapshot-dir", "off"], stdout=subprocess.DEVNULL,
                        stderr=diagnostics, env=env, pass_fds=(fd,))
                    deadline = time.monotonic() + 300
                    while True:
                        if child.poll() is not None:
                            raise RuntimeError(log.read_text()[-8000:])
                        try:
                            with urllib.request.urlopen(address + "/v1/models", timeout=1) as reply:
                                model_id = json.load(reply)["data"][0]["id"]
                            break
                        except (OSError, ValueError):
                            if time.monotonic() > deadline:
                                raise TimeoutError("server load")
                            time.sleep(.25)

                    def request(items):
                        body = {"model": model_id, "input": items, "store": False,
                                "max_output_tokens": 128, "temperature": 0,
                                "x_qwen": {"stats": True, "seed": 1}}
                        req = urllib.request.Request(address + "/v1/responses",
                            json.dumps(body).encode(), {"Content-Type": "application/json"})
                        with urllib.request.urlopen(req, timeout=600) as reply:
                            return json.load(reply)

                    history = [{"role": "user", "content": prompt}]
                    first = request(history)
                    history += first["output"] + [{"role": "user", "content": append}]
                    second = request(history)
                    # A third turn exercises re-admission after a captured
                    # append; responses are replayed through the public API.
                    history += second["output"] + [{"role": "user", "content": "Which allocation should be freed first?"}]
                    third = request(history)
                    result["turns"] = [{"usage": row["usage"], "x_qwen": row.get("x_qwen"),
                                        "status": row["status"]} for row in [first, second, third]]
                child.terminate()
                child.wait(timeout=20)
            stop.set()
            watcher.join()
            sample()
            text = re.sub(r"\x1b\[[0-9;]*m", "", log.read_text())
            result["diagnostics"] = [line for line in text.splitlines()
                if any(key in line for key in ["glm workspace:", "glm5_next stats:",
                                                "glm native:", "serve native:",
                                                "serve phases:", "serve stats:"])]
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        stop.set()
        watcher.join()
        thermal()
    result["thermal_samples"] = thermals
    result["thermal_valid"] = all(value == 0 for value in thermals)
    # End observations lag their diagnostic writes by at most one poll. A
    # two-second guard conservatively includes both ends of each decode.
    # Setup builds are allowed; timing never acquires the build lock.
    result["compiler_poll_seconds"] = 1
    result["compiler_window_guard_seconds"] = 2
    result["decode_windows"] = []
    for end, duration in decode_ends:
        observed = [rss for at, rss in compiler_samples
                    if end - duration / 1000 - 2 <= at <= end + 2]
        result["decode_windows"].append({
            "decode_ms": duration, "compiler_samples": len(observed),
            "compiler_peak_rss_bytes": max(observed, default=0),
            "no_compiler_observed": bool(observed) and not any(observed)})
    print(json.dumps(result), flush=True)
    if not result["thermal_valid"]:
        raise RuntimeError("thermal pressure changed; arm excluded")


if __name__ == "__main__":
    main()
