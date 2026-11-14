#!/usr/bin/env python3
"""Run a three-agent pair or one arm inside a <=15-minute lease."""
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


def attention_warmup(binary,log_path,fd,names=None):
    """Dispatch both candidate and rollback kernels identically before each arm."""
    names=names or ("deepseek_v41::forward::attention_gpu::tests::radix_select::masked_mma_index_preserves_allowed_scores_and_selection",
                   "deepseek_v41::forward::attention_gpu::tests::f32_pv_batch::tensor64_sink_tile_matches_tensor32_sink")
    record=[]
    for i,name in enumerate(names):
        dest=log_path.with_name(log_path.name+f"-{i}.log")
        env=dict(os.environ,QWEN_REQUIRE_METAL_TESTS="1")
        env.pop("MTL_DEBUG_LAYER",None)
        command=packet.nice_command([str(binary.resolve()),name,"--exact","--nocapture"])
        started=time.time(); thermal_trace=[packet.thermal()]
        if thermal_trace[-1]>=3:raise RuntimeError("warmup thermal pressure>=3")
        with dest.open("w") as log:
            child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env,pass_fds=(fd,))
            try:
                nice=int(subprocess.check_output(["ps","-o","nice=","-p",str(child.pid)],text=True,timeout=5).strip())
                if nice!=0:raise RuntimeError("warmup niceness is not0")
                while child.poll() is None:
                    thermal_trace.append(packet.thermal())
                    if thermal_trace[-1]>=3 or time.time()-started>=60:raise RuntimeError("warmup thermal/budget abort")
                    time.sleep(.1)
            finally:
                if child.poll() is None:
                    child.terminate()
                    try:child.wait(timeout=10)
                    except subprocess.TimeoutExpired:child.kill();child.wait(timeout=5)
        text=dest.read_text()
        if child.returncode or "1 passed" not in text:raise RuntimeError("attention warmup failed: "+str(dest))
        if name.endswith("tensor64_sink_tile_matches_tensor32_sink") and text.count("raw_and_rope_bits_exact=true")!=15:raise RuntimeError("wide warmup did not exercise all15 cases")
        if name.endswith("coalesced_radix_matches_incumbent_bits_and_status") and "RADIX_COALESCE_EXACT cases=13" not in text:raise RuntimeError("radix warmup witness missing")
        if name.endswith("compact_masked_mma_preserves_scores_and_selected_ids_bitwise") and "COMPACT_INDEX_EXACT allowed_score_bits=true" not in text:raise RuntimeError("compact warmup witness missing")
        record.append(dict(test=name,pid=child.pid,niceness=nice,seconds=time.time()-started,thermal_max=max(thermal_trace),log=dest.name))
    return record


def model_weight_warmup(binary,model,corpus,log_path,deadline,fd,experiment="paired-attention-bundle"):
    """Warm GGUF pages with both policies, then discard all KV and Engram pages."""
    cool=None
    while True:
        level=packet.thermal()
        if level>=3:raise RuntimeError("weight warmup thermal pressure>=3")
        if level<=1:
            cool=cool or time.monotonic()
            if time.monotonic()-cool>=60:break
        else:cool=None
        if time.monotonic()+90>=deadline:raise RuntimeError("weight warmup settle budget exhausted")
        time.sleep(2)
    resident_check()
    command=packet.nice_command([str(binary.resolve()),str(model),str(corpus),"0","4096","native","1",experiment])
    started=time.time();samples=[];levels=[]
    with log_path.open("w") as log:
        child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,pass_fds=(fd,))
        try:
            nice=int(subprocess.check_output(["ps","-o","nice=","-p",str(child.pid)],text=True,timeout=5).strip())
            log_path.with_suffix(".model-pids.json").write_text(json.dumps(dict(time=time.time(),model_pids=[child.pid],model_niceness={child.pid:nice},binary=str(binary.resolve()))))
            if nice!=0:raise RuntimeError("weight warmup model niceness is not0")
            while child.poll() is None:
                level=packet.thermal();levels.append(level)
                if level>=3 or time.monotonic()>=deadline:raise RuntimeError("weight warmup thermal/budget abort")
                if value:=usage(child.pid):samples.append(dict(time=time.time(),pid=child.pid,**value))
                log_path.with_suffix(".process-io.jsonl").write_text("".join(json.dumps(r)+"\n" for r in samples))
                time.sleep(.5)
        finally:
            if child.poll() is None:
                child.send_signal(signal.SIGINT)
                try:child.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    child.terminate()
                    try:child.wait(timeout=10)
                    except subprocess.TimeoutExpired:child.kill();child.wait(timeout=5)
    log_path.with_suffix(".process-io.jsonl").write_text("".join(json.dumps(r)+"\n" for r in samples))
    if child.returncode:raise RuntimeError("weight warmup failed: "+str(log_path))
    events=[json.loads(l) for l in log_path.read_text().splitlines() if l.startswith("{")]
    exact=[r for r in events if r.get("kind")=="exact"]
    if len(exact)!=2 or not all(r["exact"] for r in exact):raise RuntimeError("paired weight warmup bitwise witness missing")
    return dict(pid=child.pid,niceness=nice,thermal_max=max(levels),seconds=time.time()-started,
                binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                process_disk_read_bytes=max((r["disk_read"] for r in samples),default=None),
                exact_signatures=2,log=log_path.name)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("binary","tokenizer","model","corpus","output"):
        p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--flag",action="append",required=True)
    p.add_argument("--attention-warmup-binary",type=Path)
    p.add_argument("--model-warmup-binary",type=Path)
    p.add_argument("--model-warmup-experiment",default="paired-attention-bundle")
    p.add_argument("--warmup-test",action="append",help="exact GPU test name; identically run before both arms")
    p.add_argument("--engram",type=Path,required=True)
    p.add_argument("--tag",required=True);p.add_argument("--order",choices=("ab","ba"),required=True)
    p.add_argument("--only-arm",choices=("a","b"),help="run the next arm of this ordered pair; refuse any overwrite")
    p.add_argument("--port",type=int,default=19973);p.add_argument("--seconds",type=int,default=120)
    p.add_argument("--warm-reply-tokens",type=int,default=32)
    p.add_argument("--sampling",choices=("agent","greedy"),default="greedy")
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    fd=int(os.environ["QWEN_METAL_LEASE_FD"])
    deadline=time.monotonic()+860
    plan=out/f"{a.tag}-plan.json"
    subprocess.run(["python3","scripts/bench/serve_realistic.py","plan","--family","v41",
        "--seconds",str(a.seconds),"--seed","20261112","--out",str(plan)],check=True,timeout=10)
    sha=hashlib.sha256(a.binary.read_bytes()).hexdigest()
    result=dict(binary_sha256=sha,flags=a.flag,order=a.order,sampling=a.sampling,seconds=a.seconds,
                warm_reply_tokens=a.warm_reply_tokens,plan_sha256=hashlib.sha256(plan.read_bytes()).hexdigest(),arms=[])
    summary=out/f"{a.tag}-summary.json"
    if summary.exists():
        if not a.only_arm:raise RuntimeError("pair summary already exists; refusing overwrite")
        previous=json.loads(summary.read_text())
        if any(previous[k]!=result[k] for k in ("binary_sha256","flags","order","sampling","seconds","warm_reply_tokens","plan_sha256")):
            raise RuntimeError("one-arm pair metadata differs")
        result=previous
    if a.only_arm:
        done=[r["arm"] for r in result["arms"]]
        if done!=list(a.order[:len(done)]) or len(done)>=2 or a.order[len(done)]!=a.only_arm:
            raise RuntimeError("one-arm call is not the next incomplete arm")
    for arm in (a.only_arm if a.only_arm else a.order):
        if (out/f"{a.tag}-{arm}.log").exists() or (out/f"{a.tag}-{arm}").exists():
            raise RuntimeError("arm artifacts already exist; refusing overwrite")
        processes=resident_check()
        (out/f"{a.tag}-{arm}-processes.txt").write_text(processes)
        weight_binary=a.model_warmup_binary or a.binary.resolve().parents[1]/"iter/examples/v41_prefill_census"
        weight_warmup=model_weight_warmup(weight_binary,a.model,a.corpus,out/f"{a.tag}-{arm}-weight-warmup.log",deadline,fd,a.model_warmup_experiment) if a.attention_warmup_binary else None
        warmup=attention_warmup(a.attention_warmup_binary,out/f"{a.tag}-{arm}-attention-warmup",fd,a.warmup_test) if a.attention_warmup_binary else []
        resident_check()
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
            *[item for flag in a.flag for item in ("--env",flag+"="+("0" if arm=="a" else "1"))],
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
        result["arms"].append(dict(arm=arm,model_weight_warmup=weight_warmup,attention_warmup=warmup,engram_purge=reset,process_disk_read_bytes=disk,process_disk_counter_note="maximum live counter sampled every 0.5s; native per-call I/O retained",root_turns=roots,model_pids=sorted(model_pids),model_niceness=model_niceness,
            summary=json.loads((dest/"summary.json").read_text())))
        (out/f"{a.tag}-summary.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result),flush=True)


if __name__=="__main__":main()
