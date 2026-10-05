# /// script
# requires-python = ">=3.11"
# ///
"""Bounded kill check of ordinary command-buffer wiring (PERF-ROADMAP
2026-10-05 #1 protocol). Runs `qwen-bench residency-kill-child` cases over a
disposable file-backed no-copy buffer, SIGKILLs only that child, and records
system wired memory recovery. Run it detached (own session) so a tool
timeout cannot kill the observer; results go to --out.

Cases: normal (control: pulses, exits), stopped (pulses, stops, idles, then
killed), pulse (killed between keep-alive pulses), execute (killed with a
long command confirmed in flight). Aborts on host pressure, child failure or
unrecovered wiring beyond --tolerance-mib at 15 s.
"""

import argparse
import json
import os
import signal
import subprocess
import time
from pathlib import Path

GIB = 1024**3


def wired_bytes():
    out = subprocess.run(["vm_stat"], capture_output=True, text=True, check=True).stdout
    page = int(out.split("page size of ")[1].split(" ")[0])
    stats = {}
    for line in out.splitlines()[1:]:
        if ":" in line:
            key, value = line.split(":", 1)
            stats[key.strip().strip('"')] = int(value.strip().rstrip("."))
    return stats["Pages wired down"] * page, stats.get("Pageouts", 0)


def pressure_level():
    out = subprocess.run(
        ["sysctl", "-n", "kern.memorystatus_vm_pressure_level"],
        capture_output=True,
        text=True,
    )
    return int(out.stdout.strip()) if out.returncode == 0 else None


def averaged_wired(samples=3, gap=0.5):
    values = []
    for i in range(samples):
        values.append(wired_bytes()[0])
        if i + 1 < samples:
            time.sleep(gap)
    return sum(values) / len(values), max(values) - min(values)


def engine_processes():
    out = subprocess.run(
        [
            "pgrep",
            "-fl",
            "llama-server|llama-bench|qwen-bench|glm53_oracle|qwen run|qwen serve|qwen-lens",
        ],
        capture_output=True,
        text=True,
    ).stdout
    return [line for line in out.splitlines() if "residency_kill_check" not in line]


def read_status(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def wait_for(path, predicate, timeout):
    deadline = time.time() + timeout
    while time.time() < deadline:
        records = read_status(path)
        if any(predicate(r) for r in records):
            return records
        time.sleep(0.05)
    raise TimeoutError(f"status {path} never matched within {timeout}s")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench", required=True)
    parser.add_argument("--file", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--cases", default="normal,stopped,pulse,execute")
    parser.add_argument("--tolerance-mib", type=int, default=384)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tolerance = args.tolerance_mib * 1024**2
    report = {
        "schema": "qwen.residency_kill_check.v1",
        "pid": os.getpid(),
        "sid": os.getsid(0),
        "pgid": os.getpgid(0),
        "file_bytes": Path(args.file).stat().st_size,
        "tolerance_bytes": tolerance,
        "cases": [],
    }

    def save():
        (out / "report.json").write_text(json.dumps(report, indent=1) + "\n")

    others = engine_processes()
    level = pressure_level()
    baseline, spread = averaged_wired(samples=5, gap=1.0)
    report["preconditions"] = {
        "engine_processes": others,
        "pressure_level": level,
        "baseline_wired": baseline,
        "baseline_spread": spread,
    }
    if others or level != 1 or spread > 0.3 * GIB:
        report["verdict"] = "aborted: host not quiet"
        save()
        return
    save()

    for case in args.cases.split(","):
        status = out / f"{case}.status.jsonl"
        log = (out / f"{case}.log").open("w")
        record = {"case": case}
        report["cases"].append(record)
        before, _ = averaged_wired()
        record["wired_before"] = before
        env = dict(os.environ, QWEN_METAL_LEASE_WAIT="1")
        child = subprocess.Popen(
            [
                args.bench,
                "residency-kill-child",
                "--file",
                args.file,
                "--mode",
                case,
                "--status",
                str(status),
                "--seconds",
                "8",
            ],
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=log,
            env=env,
            close_fds=True,
        )
        record["child_pid"] = child.pid
        try:
            if case == "normal":
                wait_for(status, lambda r: r.get("phase") == "pulsing", 120)
                time.sleep(3)
                record["wired_loaded"], _ = averaged_wired()
                record["exit_code"] = child.wait(timeout=60)
                record["killed"] = False
            else:
                phase = {
                    "pulse": "pulsing",
                    "stopped": "stopped",
                    "execute": "executing",
                }[case]
                wait_for(status, lambda r: r.get("phase") == phase, 120)
                if case == "pulse":
                    time.sleep(3)
                    record["wired_loaded"], _ = averaged_wired()
                    time.sleep(0.25)  # mid-interval between 500 ms pulses
                elif case == "stopped":
                    record["wired_loaded"], _ = averaged_wired()
                    time.sleep(4)  # past the ~2 s unwire delay
                    record["wired_at_kill_unpulsed"], _ = averaged_wired()
                else:
                    record["wired_loaded"], _ = averaged_wired(samples=2, gap=0.2)
                    # Kill 0.3 s after a fresh commit, before its completion.
                    seen = len(read_status(status))
                    deadline = time.time() + 60
                    while time.time() < deadline:
                        records = read_status(status)
                        if (
                            len(records) > seen
                            and records[-1].get("phase") == "executing"
                        ):
                            break
                        time.sleep(0.01)
                    time.sleep(0.3)
                    records = read_status(status)
                    record["in_flight_at_kill"] = (
                        records[-1].get("phase") == "executing"
                    )
                    record["last_status_before_kill"] = records[-1]
                os.kill(child.pid, signal.SIGKILL)
                record["killed"] = True
                record["kill_t"] = time.time()
                record["exit_code"] = child.wait(timeout=60)
        except Exception as error:  # noqa: BLE001
            record["error"] = repr(error)
            if child.poll() is None:
                os.kill(child.pid, signal.SIGKILL)
                child.wait(timeout=60)
        finally:
            log.close()
        record["status"] = read_status(status)
        recovery = []
        for at in [1, 5, 15, 30, 60]:
            target = record.get("kill_t", time.time()) + at
            time.sleep(max(0.0, target - time.time()))
            wired, spread = averaged_wired()
            recovery.append(
                {
                    "t": at,
                    "wired": wired,
                    "excess": wired - baseline,
                    "spread": spread,
                    "pressure_level": pressure_level(),
                    "pageouts": wired_bytes()[1],
                }
            )
            if at == 15 and wired - baseline > tolerance:
                record["recovery"] = recovery
                report["verdict"] = (
                    f"aborted: {case} excess {wired - baseline:.0f} B at 15 s"
                )
                save()
                return
        record["recovery"] = recovery
        record["increase"] = record.get("wired_loaded", before) - before
        save()
        if "error" in record:
            report["verdict"] = f"aborted: {case} error"
            save()
            return
    final = [c["recovery"][-1]["excess"] for c in report["cases"]]
    report["verdict"] = "recovered" if all(e <= tolerance for e in final) else "excess"
    save()


if __name__ == "__main__":
    main()
