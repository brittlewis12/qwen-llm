#!/usr/bin/env python3
"""Run one interleaved realistic three-agent pair inside a <=14-minute lease."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

import v41_prefill_packet as packet
from glm_residency import usage


def resident_check():
    text=subprocess.check_output(["ps","-axo","pid,rss,comm"],text=True,timeout=5)
    if found := packet.resident_models(text):
        raise RuntimeError("another model process is resident: "+found[0])
    return text


def run_harness(command,log_path,binary,deadline,fd):
    model_pids=set();model_niceness={};io=[]
    samples=log_path.with_suffix(".process-io.jsonl")
    with log_path.open("w") as log:
        child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,pass_fds=(fd,))
        try:
            while child.poll() is None:
                if time.monotonic()>=deadline:raise RuntimeError("serve pair budget exhausted")
                if not model_pids:
                    ps=subprocess.check_output(["ps","-axo","pid,ppid,comm"],text=True,timeout=5)
                    for line in ps.splitlines()[1:]:
                        f=line.split(None,2)
                        if len(f)==3 and int(f[1])==child.pid and f[2]==str(binary.resolve()):
                            model_pids.add(int(f[0]))
                            model_niceness[int(f[0])]=int(subprocess.check_output(["ps","-o","nice=","-p",f[0]],text=True,timeout=5).strip())
                            log_path.with_suffix(".model-pids.json").write_text(json.dumps(dict(time=time.time(),harness_pid=child.pid,model_pids=sorted(model_pids),model_niceness=model_niceness)))
                            if model_niceness[int(f[0])]!=0:
                                raise RuntimeError("model niceness is not 0")
                for pid in model_pids:
                    if value:=usage(pid):io.append(dict(time=time.time(),pid=pid,**value))
                samples.write_text("".join(json.dumps(r)+"\n" for r in io))
                time.sleep(.5)
        finally:
            if child.poll() is None:
                child.send_signal(signal.SIGINT)
                try:child.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    child.terminate()
                    try:child.wait(timeout=10)
                    except subprocess.TimeoutExpired:child.kill();child.wait(timeout=5)
            # Exact qwen PIDs observed as this harness's children only.
            for pid in model_pids:
                check=subprocess.run(["ps","-p",str(pid),"-o","comm="],capture_output=True,text=True,timeout=5)
                if check.stdout.strip()==str(binary.resolve()):
                    os.kill(pid,signal.SIGINT)
                    for _ in range(20):
                        if subprocess.run(["kill","-0",str(pid)],capture_output=True).returncode:break
                        time.sleep(1)
                    else:os.kill(pid,signal.SIGKILL)
        samples.write_text("".join(json.dumps(r)+"\n" for r in io))
        if child.returncode:raise RuntimeError(f"serve phase failed: {log_path.name} ({child.returncode})")
    return model_pids,model_niceness


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("binary","tokenizer","model","corpus","output"):
        p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--flag",default="QWEN_V41_MASKED_REINDEX_SHORT")
    p.add_argument("--engram",type=Path,required=True)
    p.add_argument("--tag",required=True);p.add_argument("--order",choices=("ab","ba"),required=True)
    p.add_argument("--port",type=int,default=19973);p.add_argument("--seconds",type=int,default=120)
    p.add_argument("--warm-reply-tokens",type=int,default=32)
    p.add_argument("--sampling",choices=("agent","greedy"),default="greedy")
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    fd=int(os.environ["QWEN_METAL_LEASE_FD"])
    deadline=time.monotonic()+800
    plan=out/f"{a.tag}-plan.json"
    subprocess.run(["python3","scripts/bench/serve_realistic.py","plan","--family","v41",
        "--seconds",str(a.seconds),"--seed","20261112","--out",str(plan)],check=True,timeout=10)
    sha=hashlib.sha256(a.binary.read_bytes()).hexdigest()
    result=dict(binary_sha256=sha,flag=a.flag,order=a.order,sampling=a.sampling,arms=[])
    for arm in a.order:
        processes=resident_check()
        (out/f"{a.tag}-{arm}-processes.txt").write_text(processes)
        purge=out/f"{a.tag}-{arm}-purge.jsonl"
        subprocess.run(["python3","scripts/bench/invalidate_file_cache.py",str(a.engram),"--out",str(purge)],check=True,timeout=60)
        reset=json.loads(purge.read_text().splitlines()[-1])
        if reset["msync_rc"] or reset["resident_pages_after"]!=0:raise RuntimeError("Engram purge did not reach zero resident pages")
        dest=out/f"{a.tag}-{arm}"
        command=packet.nice_command(["python3","scripts/bench/serve_realistic.py","run",
            "--binary",str(a.binary.resolve()),"--tokenizer",str(a.tokenizer.resolve()),
            "--model",str(a.model),"--corpus",str(a.corpus),"--plan",str(plan),"--out",str(dest),
            "--agents","3","--arm","default","--warm-tokens","49152","--context","196608",
            "--warm-reply-tokens",str(a.warm_reply_tokens),
            "--sampling",a.sampling,"--cool","0","--ramp","15","--timeout","650","--port",str(a.port),
            "--max-thermal-pressure","2","--warmup-max-thermal-pressure","2",
            "--settle-thermal-seconds","60","--settle-max-thermal-pressure","1",
            "--env",a.flag+"="+("0" if arm=="a" else "1"),
            "--env","QWEN_SERVE_V41_PREFILL_TIMING=1",
            "--env","QWEN_SERVE_V41_MAX_ACTIVE=4",
            "--env","QWEN_SERVE_V41_SAMPLE_WORKERS=3",
            "--env","QWEN_SERVE_V41_PREFILL_FAIR=1"])
        model_pids,model_niceness=run_harness(command,out/f"{a.tag}-{arm}.log",a.binary,deadline,fd)
        events=[json.loads(line) for line in (out/f"{a.tag}-{arm}.log").read_text().splitlines() if line.startswith("{")]
        roots=[e for e in events if e.get("event")=="turn" and e.get("label","").endswith("-warm")]
        io=[json.loads(line) for line in (out/f"{a.tag}-{arm}.process-io.jsonl").read_text().splitlines()]
        disk=max((r["disk_read"] for r in io),default=None)
        if disk is None:raise RuntimeError("no process disk counter captured")
        result["arms"].append(dict(arm=arm,engram_purge=reset,process_disk_read_bytes=disk,process_disk_counter_note="maximum live counter sampled every 0.5s; native per-call I/O retained",root_turns=roots,model_pids=sorted(model_pids),model_niceness=model_niceness,
            summary=json.loads((dest/"summary.json").read_text())))
        (out/f"{a.tag}-summary.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result),flush=True)


if __name__=="__main__":main()
