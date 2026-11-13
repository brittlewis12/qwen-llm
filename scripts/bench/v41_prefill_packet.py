#!/usr/bin/env python3
"""Bounded census packet, called inside with-metal-lease (<=15 min).

Records shared load, requires 60 continuous seconds at pressure <=1 before
each load, permits pressure 2 during work and stops its exact child at 3.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def stop(signum, _frame):
    raise SystemExit(128 + signum)


signal.signal(signal.SIGTERM, stop)
signal.signal(signal.SIGINT, stop)


def nice_command(command):
    current=os.getpriority(os.PRIO_PROCESS,0)
    if current>10:raise RuntimeError(f"runner niceness {current} exceeds required 10")
    return ["nice","-n",str(10-current),*command] if current<10 else command


def thermal():
    value = subprocess.check_output(
        ["notifyutil", "-g", "com.apple.system.thermalpressurelevel"],
        text=True, timeout=5,
    )
    return int(value.split()[-1])


def resident_models(processes, exclude_pid=None):
    """Keep known Ghost background work in the record without calling it a model."""
    found=[]
    ghost_prefix=str(Path.home()/"ghost/host-cache")+"/"
    ghost_suffix="/headless/Headless.app/Contents/MacOS/shadps4"
    for line in processes.splitlines()[1:]:
        fields=line.split(None,2)
        if len(fields)!=3:continue
        pid,rss,command=fields
        if int(pid)==exclude_pid:continue
        if command.startswith(ghost_prefix) and command.endswith(ghost_suffix):continue
        if int(rss)>20_000_000 or int(rss)>1_000_000 and any(x in command for x in ("qwen","mlx","deepseek")):
            found.append(line)
    return found


def run(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    fd = int(os.environ["QWEN_METAL_LEASE_FD"])
    record = out / (args.tag + "-monitor.jsonl")
    with record.open("w") as monitor:
        def log(**fields):
            monitor.write(json.dumps(dict(time=time.time(), **fields)) + "\n")
            monitor.flush()
        log(kind="load", uptime=subprocess.check_output(["uptime"], text=True, timeout=5),
            processes=subprocess.check_output(
                ["ps", "-axo", "pid,ppid,%cpu,rss,comm"], text=True, timeout=5))
        deadline = time.monotonic() + 680
        for depth, rows, mode in args.cells:
            cool = None
            while True:
                level = thermal()
                log(kind="settle", depth=depth, rows=rows, mode=mode, thermal=level)
                if level <= 1:
                    cool = cool or time.monotonic()
                    if time.monotonic() - cool >= 60:
                        break
                else:
                    cool = None
                if time.monotonic() + 90 >= deadline:
                    raise RuntimeError("packet budget exhausted during settle")
                time.sleep(5)
            processes = subprocess.check_output(["ps", "-axo", "pid,rss,comm"], text=True, timeout=5)
            if found := resident_models(processes, os.getpid()):
                raise RuntimeError("another model process is resident: " + found[0])
            name = f"{args.tag}-{depth}-{rows}-{mode}"
            env = dict(os.environ)
            if mode == "dispatch":
                # Resolve after all CED work has completed. Native keeps the pipeline.
                env["QWEN_V41_CED_ENCODER_PIPELINE"] = "0"
            cmd = nice_command([args.binary, args.model, args.corpus,
                   str(depth), str(rows), mode, str(args.reps), args.experiment])
            log(kind="start", name=name, cmd=cmd, processes=processes)
            with (out / (name + ".jsonl")).open("w") as stdout, (out / (name + ".log")).open("w") as stderr:
                child = subprocess.Popen(cmd, stdout=stdout, stderr=stderr, env=env,
                                         pass_fds=(fd,))
                try:
                    log(kind="pid", pid=child.pid, niceness=os.getpriority(os.PRIO_PROCESS,child.pid))
                    while child.poll() is None:
                        level = thermal()
                        log(kind="running", pid=child.pid, thermal=level)
                        if level >= 3 or time.monotonic() >= deadline:
                            raise RuntimeError("thermal/budget abort")
                        time.sleep(2)
                finally:
                    if child.poll() is None:
                        child.send_signal(signal.SIGTERM)
                        try:
                            child.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            child.kill()
                            child.wait(timeout=5)
                log(kind="exit", pid=child.pid, code=child.returncode)
                if child.returncode:
                    raise RuntimeError(f"cell failed: {name} ({child.returncode})")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--binary", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--corpus", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--tag", default="census")
    p.add_argument("--reps", type=int, default=1)
    p.add_argument("--experiment", default="baseline")
    p.add_argument("--cell", action="append", required=True, help="depth,rows,mode")
    a = p.parse_args()
    a.cells = [(int(d), int(n), mode) for d, n, mode in (c.split(",") for c in a.cell)]
    run(a)
