#!/usr/bin/env python3
"""Seeded warm-agent concurrency workload (stdlib only).

Three warm agents draw continuously from recorded per-family OpenCode calls.
The default is a fixed shared measurement window, excluding ramp and cancellation.
Rates permit natural EOS and unequal completed work. Timing needs a Metal lease,
not the build lock. Build first; retain commit + command and compact summaries.
"""

import argparse
import concurrent.futures
import copy
import csv
import hashlib
import http.client
import json
import math
import os
from pathlib import Path
import random
import re
import signal
import statistics
import subprocess
import sys
import threading
import time


DEFAULT_CALLS = Path(__file__).resolve().parents[2] / "docs/bench/2026-11-08-real-opencode-agent-calls.csv"
DELTA_TYPES = {"response.output_text.delta", "response.reasoning_text.delta",
               "response.function_call_arguments.delta"}
INSTRUCTIONS = (
    "You are reviewing a Rust/Metal inference engine. The source corpus is authentic. "
    "Produce a detailed line-by-line engineering review, including invariants, failure "
    "paths and proposed tests for every function in the supplied source. This is a "
    "continuing report: keep writing substantive observations until interrupted by "
    "the output limit. Do not conclude, summarize, or call tools. Later excerpts are "
    "tool-style source results; continue the same report after each one."
)


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def arm_environment(source, family, arm):
    env = dict(source, RUST_LOG='warn,qwen_diag=info', NO_COLOR='1')
    knobs = dict(v41='QWEN_SERVE_V41_MAX_ACTIVE', glm='QWEN_SERVE_GLM_MAX_ACTIVE',
                 fn='QWEN_SERVE_FN_MAX_ACTIVE')
    for knob in knobs.values():
        env.pop(knob, None)
    if arm == 'serial':
        env[knobs[family]] = '1'
    return env


def percentile(values, fraction):
    if not values:
        return None
    values = sorted(values)
    return values[min(len(values) - 1, math.ceil(fraction * len(values)) - 1)]


def make_plan(rows, family, seed, seconds):
    rows = [r for r in rows if r["model"] == family and r["purpose"] == "primary"
            and int(r["reused"]) > 0 and r["reuse_source"] in ("live_session", "live")]
    if not rows or seconds <= 0:
        raise ValueError("positive window and primary live-prefix observations required")
    population = [dict(source_call=r["call"], append_tokens=int(r["new"]),
                       reply_tokens=int(r["generated"]),
                       tool_gap_s=float(r["tool_union_ms"]) / 1000) for r in rows]
    return dict(version=2, mode="steady", family=family, seed=seed,
                window_seconds=seconds, population=population)


def draws(plan, agent):
    rng = random.Random(f"{plan['seed']}:{plan['family']}:{agent}")
    while True:
        yield dict(rng.choice(plan["population"]))


def token_text(binary, model, text=None, ids=None):
    if ids is None:
        cmd = [str(binary), "-m", str(model), "--file", "-", "--ids"]
        result = subprocess.check_output(cmd, input=text.encode()).decode().splitlines()
        return list(map(int, result[1:]))
    cmd = [str(binary), "-m", str(model), "--decode-ids", "-"]
    return subprocess.check_output(cmd, input=" ".join(map(str, ids)).encode()).decode()


def history_items(items):
    out = []
    has_assistant = False
    for item in items:
        kind = item.get("type")
        if kind == "reasoning":
            out.append({"type": "reasoning", "content": "".join(p.get("text", "") for p in item.get("content", []))})
        elif kind == "message":
            out.append({"role": "assistant", "content": "".join(p.get("text", "") for p in item.get("content", []))})
            has_assistant = True
        elif kind == "function_call":
            out.append({k: item[k] for k in ("type", "call_id", "name", "arguments")})
            out.append({"type": "function_call_output", "call_id": item["call_id"],
                        "output": "Source is supplied in the next excerpt."})
            has_assistant = True
    if not has_assistant:
        out.append({"role": "assistant", "content": ""})
    return out


def pressure():
    return int(subprocess.check_output(
        ["notifyutil", "-g", "com.apple.system.thermalpressurelevel"], text=True).split()[-1])


def thermal_exceeds_limit(level, maximum=1):
    return level < 0 or level > maximum


def parse_compilers(output):
    # Conservative build/compiler presence, including waiting cargo processes.
    names = {'cargo','rustc','clang','clang++','metal','metalfe','metallib','air-lld','ld','ld64.lld'}
    found=[]
    for line in output.splitlines():
        fields=line.strip().split(None,2)
        if len(fields)!=3:continue
        pid,cpu,command=fields
        if Path(command).name in names:
            found.append(dict(pid=int(pid),executable=Path(command).name,cpu_percent=float(cpu)))
    return found


def compiler_snapshot():
    return parse_compilers(subprocess.check_output(['ps','-ww','-axo','pid=,pcpu=,comm='],text=True))


def timing_flags(samples, begin, end):
    window=[s for s in samples if begin <= s['time'] < end]
    return dict(thermal_flagged=any(s['thermal']==1 for s in samples),
                timing_thermal_max=max((s['thermal'] for s in window),default=0),
                compile_observed=any(s.get('compilers') for s in samples),
                compile_overlapped=any(s.get('compilers') for s in window))


def kv(line):
    return {a: b.strip('"') for a, b in re.findall(r'(\w+)=("[^"\n]*"|\[[^\]]*\]|[^ \n]+)', line)}


def interval_buckets(rows, begin, end):
    """Exclusive *observed stream* overlap, not GPU kernel/batch service time.

    Pre-first-token spans include queueing. Decode spans include interleaved
    appends. Native batch counters are reported separately to avoid equating
    the number of open streams with the width of a native forward.
    """
    edges = [(end, 0, 0)]
    for r in rows:
        edges.extend((t, d, w) for t, d, w in
                     [(r["sent"], 0, 1), (r["first"], 1, -1), (r["end"], -1, 0)]
                     if begin < t < end)
    decode = sum(r["first"] <= begin < r["end"] for r in rows)
    waiting = sum(r["sent"] <= begin < r["first"] for r in rows)
    result = {"decode_stream_width_1_s": 0.0, "decode_stream_width_2_s": 0.0,
              "decode_stream_width_3_s": 0.0, "predecode_only_s": 0.0, "idle_s": 0.0}
    last = begin
    for t, d, w in sorted(edges):
        if t > last:
            key = f"decode_stream_width_{decode}_s" if decode else ("predecode_only_s" if waiting else "idle_s")
            result[key] += t - last
        decode += d
        waiting += w
        last = t
    return result


def native_widths(batch):
    steps = batch.get("stepped_transitions", 0)
    if not steps:
        return [0, 0, 0]
    shared = batch["batched_transitions"]
    one = steps - shared
    three = round(steps * batch["width_mean"]) - one - 2 * shared
    two = shared - three
    if min(one, two, three) < 0 or batch["width_max"] > 3:
        raise ValueError("width census requires at most three agents")
    return [one, two, three]


class Arm:
    def __init__(self, args):
        self.args = args
        self.out = args.out.resolve()
        self.out.mkdir(parents=True, exist_ok=False)
        self.plan = json.loads(args.plan.read_text())
        if self.plan.get('version') != 2:
            raise ValueError('regenerate a version-2 steady plan')
        self.rows, self.samples, self.lines = [], [], []
        self.server = None
        self.abort = threading.Event()
        self.finish = threading.Event()
        self.monitor_done = threading.Event()
        self.lock = threading.Lock()
        self.window_begin = self.window_end = None
        self.deadline = time.monotonic() + args.timeout

    def emit(self, **row):
        with self.lock:
            print(json.dumps(row), flush=True)

    def check(self):
        if self.abort.is_set() or time.monotonic() > self.deadline:
            raise RuntimeError('thermal/deadline/transport stop')
        if self.server is not None and self.server.poll() is not None and not self.finish.is_set():
            raise RuntimeError(f'server exited {self.server.returncode}')

    def watch(self):
        try:
            while not self.monitor_done.is_set():
                p = pressure()
                self.samples.append(dict(time=time.time(), thermal=p, compilers=compiler_snapshot()))
                if thermal_exceeds_limit(p,self.args.max_thermal_pressure) or time.monotonic() > self.deadline:
                    self.abort.set()
                    self.finish.set()
                    self.emit(event='abort', thermal=p)
                    if self.server.poll() is None:
                        self.server.send_signal(signal.SIGINT)
                    return
                self.monitor_done.wait(1)
        except Exception as error:
            self.abort.set()
            self.finish.set()
            self.emit(event='monitor_error', error=str(error))
            if self.server.poll() is None:
                self.server.send_signal(signal.SIGINT)

    def readlog(self):
        for line in self.server.stdout:
            self.lines.append(dict(time=time.time(), line=line.rstrip()))

    def prepare(self):
        text = self.args.corpus.read_text().replace('<|', '‹|').replace('<｜', '‹｜')
        ids = token_text(self.args.tokenizer, self.args.model, text=text)
        n = self.args.warm_tokens
        if len(ids) < n * 3:
            raise ValueError('corpus needs three disjoint warm roots')
        warm = [token_text(self.args.tokenizer, self.args.model, ids=ids[a*n:(a+1)*n]) for a in range(3)]
        sizes = sorted({t['append_tokens'] for t in self.plan['population']})
        suffix = {(a, size): token_text(self.args.tokenizer, self.args.model,
                                      ids=ids[(a*977) % (len(ids)-size):(a*977) % (len(ids)-size)+size])
                  for a in range(3) for size in sizes}
        return warm, suffix

    def body(self, history, count):
        body = dict(model=self.model_id, input=copy.deepcopy(history), instructions=INSTRUCTIONS,
                    stream=True, temperature=0, top_p=1, max_output_tokens=count,
                    x_qwen=dict(stats=True, seed=self.plan['seed']),
                    tools=[dict(type='function', name='read_file', description='Read source',
                                parameters=dict(type='object', properties=dict(path=dict(type='string')),
                                                required=['path'], additionalProperties=False))])
        if self.plan['family'] in ('glm', 'fn'):
            body['reasoning'] = dict(effort='low')
        return body

    def request(self, body, label, spec=None, warm=False):
        self.check()
        payload = json.dumps(body, ensure_ascii=False).encode()
        sent = time.time()
        headers = first = final = None
        events = []
        conn = http.client.HTTPConnection('127.0.0.1', self.args.port, timeout=self.args.timeout)
        try:
            conn.request('POST', '/v1/responses', body=payload, headers={'Content-Type':'application/json'})
            response = conn.getresponse()
            headers = time.time()
            if response.status != 200 and not self.finish.is_set():
                raise RuntimeError(f'HTTP {response.status}: {response.read().decode()[:1000]}')
            while True:
                line = response.readline()
                if not line:
                    break
                if not line.startswith(b'data:'):
                    continue
                raw = line[5:].strip()
                if raw == b'[DONE]':
                    break
                event = json.loads(raw)
                now = time.time()
                kind = event.get('type')
                if kind in DELTA_TYPES and event.get('delta'):
                    events.append((now, kind, event['delta']))
                    if first is None:
                        first = now
                if kind in ('response.completed', 'response.incomplete', 'response.failed'):
                    final = event['response']
        except (OSError, http.client.HTTPException):
            if not self.finish.is_set():
                raise
        finally:
            conn.close()
        ended = time.time()
        if final and final.get('error') and not self.finish.is_set():
            raise RuntimeError(str(final['error']))
        usage = final.get('usage', {}) if final else {}
        valid = bool(usage and not final.get('error')) if final else False
        row = dict(label=label, agent=int(label[1]), warm=warm, sent=sent, headers=headers,
                   first=first, end=ended, completed=valid,
                   generated=usage.get('output_tokens') if valid else None,
                   prompt=usage.get('input_tokens') if valid else None,
                   reused=usage.get('input_tokens_details',{}).get('cached_tokens') if valid else None,
                   target=body['max_output_tokens'], spec=spec,
                   x_qwen=final.get('x_qwen',{}) if valid else {}, events=events)
        self.rows.append(row)
        self.emit(event='turn', label=label, completed=valid, generated=row['generated'],
                  ttft_s=first-sent if first else None)
        if self.args.keep_bulk:
            dump(self.out/f'{label}.request.json', body)
            dump(self.out/f'{label}.events.json', events)
            dump(self.out/f'{label}.response.json', final)
        return final if valid else None

    def run(self):
        args = self.args
        fd = int(os.environ['QWEN_METAL_LEASE_FD'])
        os.fstat(fd)
        warm, suffix = self.prepare()
        dump(self.out/'plan.json', self.plan)
        time.sleep(args.cool)
        for _ in range(3):
            if thermal_exceeds_limit(pressure(),args.max_thermal_pressure):
                raise RuntimeError('thermal pressure exceeds arm limit before arm')
            time.sleep(3)
        occupied = subprocess.run(['lsof','-nP',f'-iTCP:{args.port}','-sTCP:LISTEN'],stdout=subprocess.DEVNULL)
        if occupied.returncode != 1:
            raise RuntimeError('port occupied or lsof failed')
        env = arm_environment(os.environ, self.plan['family'], args.arm)
        maximum = max(t['reply_tokens'] for t in self.plan['population'])
        command = [str(args.binary.resolve()),'serve','-m',str(args.model.resolve()),
                   '--addr',f'127.0.0.1:{args.port}','--max-context-tokens',str(args.context),
                   '--max-tokens',str(maximum)]
        dump(self.out/'launch.json', dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
             command=command, harness_command=sys.argv, arm=args.arm,
             harness_commit=args.harness_commit,
             cool_seconds=args.cool, ramp_seconds=args.ramp, context=args.context,
             max_thermal_pressure=args.max_thermal_pressure, thermal_policy='retain flagged level1; abort above limit',
             runtime_env={k:v for k,v in env.items() if k.startswith('QWEN_')},
             corpus_sha256=digest(args.corpus), protocol='steady-own-history-v2'))
        started = time.time()
        self.server = subprocess.Popen(command,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                                       text=True,bufsize=1,pass_fds=(fd,))
        reader = threading.Thread(target=self.readlog)
        watcher = threading.Thread(target=self.watch)
        reader.start();watcher.start()
        error = None
        try:
            while time.time()-started < 180:
                self.check()
                try:
                    c=http.client.HTTPConnection('127.0.0.1',args.port,timeout=1)
                    c.request('GET','/v1/models');response=c.getresponse();data=json.loads(response.read());c.close()
                    if response.status==200:
                        self.model_id=data['data'][0]['id'];break
                except (OSError,ValueError,KeyError):
                    pass
                time.sleep(.25)
            else:
                raise TimeoutError('startup')
            ready=time.time();histories=[]
            for agent in range(3):
                body=self.body([dict(role='user',content=f'Independent agent {agent}. Source corpus follows.\n'+warm[agent])],8)
                final=self.request(body,f'a{agent}-warm',warm=True)
                if final is None:raise RuntimeError('warm-up failed')
                histories.append(body['input']+history_items(final.get('output',[])))
            warmed=time.time()
            self.emit(event='warmed',startup_s=ready-started,warmup_s=warmed-ready)
            if thermal_exceeds_limit(pressure(),args.max_thermal_pressure):raise RuntimeError('thermal pressure exceeds arm limit after warm-up')
            barrier=threading.Barrier(4)
            self.window_begin=time.time()+args.ramp
            self.window_end=self.window_begin+self.plan['window_seconds']
            def agent_loop(agent):
                history=histories[agent]
                barrier.wait()
                for turn,spec in enumerate(draws(self.plan,agent)):
                    if self.finish.is_set() or time.time()>=self.window_end:break
                    self.check()
                    call=f'agent{agent}_read{turn}'
                    history.extend([dict(type='function_call',call_id=call,name='read_file',
                                         arguments=json.dumps({'path':f'src/agent{agent}/excerpt{turn}.rs'})),
                                    dict(type='function_call_output',call_id=call,
                                         output=suffix[(agent,spec['append_tokens'])])])
                    final=self.request(self.body(history,spec['reply_tokens']),f'a{agent}-t{turn}',spec)
                    if final is None:break
                    history.extend(history_items(final.get('output',[])))
                    if self.finish.wait(spec['tool_gap_s']):break
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
                futures=[pool.submit(agent_loop,a) for a in range(3)]
                barrier.wait()
                try:
                    while time.time()<self.window_end and not self.finish.is_set():
                        self.check()
                        for f in futures:
                            if f.done() and time.time()<self.window_end and not self.finish.is_set():
                                f.result()
                                raise RuntimeError('agent exited before common cutoff')
                        self.finish.wait(.1)
                finally:
                    self.finish.set()
                    if self.server.poll() is None:self.server.send_signal(signal.SIGINT)
                try:
                    for f in concurrent.futures.as_completed(futures,timeout=45):f.result()
                except concurrent.futures.TimeoutError:
                    self.server.kill()
                    self.server.wait(timeout=10)
                    raise RuntimeError('server failed to cancel within 45 seconds')
            self.emit(event='window_done',seconds=self.plan['window_seconds'])
        except BaseException as exc:
            error=str(exc)
            raise
        finally:
            self.finish.set()
            if self.server.poll() is None:self.server.send_signal(signal.SIGINT)
            try:self.server.wait(timeout=40)
            except subprocess.TimeoutExpired:self.server.terminate();self.server.wait(timeout=10)
            self.monitor_done.set();watcher.join(timeout=5);reader.join(timeout=5)
            dump(self.out/'thermal.json',self.samples)
            if self.args.keep_bulk:
                dump(self.out/'native-lines.json',self.lines)
            if error or self.abort.is_set():
                dump(self.out/'failed.json',dict(error=error,thermal_abort=self.abort.is_set()))
            self.emit(event='server_stopped',code=self.server.returncode)
        if self.abort.is_set():
            raise RuntimeError('arm aborted; no accepted timing summary')
        result=summarize(self.rows,self.window_begin,self.window_end,
                         lambda text:len(token_text(args.tokenizer,args.model,text=text)),self.lines)
        result.update(startup_s=ready-started,warmup_s=warmed-ready,arm=args.arm,
                      family=self.plan['family'],seed=self.plan['seed'],thermal_max=max(x['thermal'] for x in self.samples),
                      metric='delivered text+reasoning tokens retokenized; excludes EOS/format control')
        result.update(timing_flags(self.samples,self.window_begin,self.window_end))
        result['max_thermal_pressure']=args.max_thermal_pressure
        detailed=result.pop('rows')
        dump(self.out/'summary.json',result)
        with (self.out/'calls.jsonl').open('w') as f:
            for row in detailed:f.write(json.dumps(row)+'\n')
        if not args.keep_bulk:
            for p in self.out.glob('*.request.json'):p.unlink()
            for p in self.out.glob('*.response.json'):p.unlink()
            for p in self.out.glob('*.events.json'):p.unlink()
        self.emit(event='summary',**result)


def window_tokens(events, begin, end, count):
    """Count delivered model-token text, including reasoning, across boundaries.

    Native usage arrives only at completion; using only complete calls would
    discard long requests. Retokenize the channel prefixes bracketing the window
    offline. EOS/template controls are excluded; boundary merges can move ~1 token.
    """
    total=0
    for kind in DELTA_TYPES:
        before=''.join(text for t,k,text in events if k==kind and t<begin)
        through=''.join(text for t,k,text in events if k==kind and t<end)
        if through:
            total+=count(through)-(count(before) if before else 0)
    return total


def summarize(rows, begin, end, count, native_lines=()):
    if begin is None or end<=begin:raise ValueError('invalid shared window')
    active=[r for r in rows if not r['warm'] and r['sent']<end and r['end']>begin]
    if {r['agent'] for r in active}!={0,1,2}:raise ValueError('three active agent loops required')
    cleaned=[]
    for r in active:
        native=r['x_qwen'].get('phases',{})
        # Match legacy serial native metadata by unique prompt/generated counts
        # and nearest completion; partial/cancelled calls have no final usage.
        if r['completed'] and not native:
            phase={};candidates=[]
            for entry in native_lines:
                line=entry['line']
                if 'serve phases:' in line:phase=kv(line)
                if 'serve stats:' in line:
                    stats=kv(line)
                    if int(stats['prompt_tokens'])==r['prompt'] and int(stats['generated_tokens'])==r['generated']:
                        candidates.append((abs(entry['time']-r['end']),dict(phase)))
                    phase={}
            if candidates:native=min(candidates,key=lambda x:x[0])[1]
        first=r['first']
        pending=r['end']>=end and (first is None or first>=end)
        wait_end=first if first is not None else min(r['end'],end)
        cleaned.append(dict(label=r['label'],agent=r['agent'],sent=r['sent'],first=first,end=r['end'],
             offered_in_window=begin<=r['sent']<end,completed=r['completed'],
             ttft_s=first-r['sent'] if first is not None and first<end and r['sent']>=begin else None,
             ttft_censored_s=end-r['sent'] if pending and r['sent']>=begin else None,
             carry_in=r['sent']<begin,
             pending_at_cutoff=pending, empty_output=r['completed'] and first is None,
             queue_exposure_s=max(0,min(wait_end,end)-max(r['sent'],begin)),
             window_tokens=window_tokens(r['events'],begin,end,count),
             target=r['target'],generated=r['generated'],prompt=r['prompt'],reused=r['reused'],
             new=r['prompt']-r['reused'] if r['prompt'] is not None and r['reused'] is not None else None,
             phases=native,spec=r['spec']))
    gaps=[b-a for r in active for a,b in zip([t for t,_,_ in r['events']],[t for t,_,_ in r['events']][1:])
          if begin<=a<b<end]
    wall_rows=[dict(sent=r['sent'],first=r['first'] if r['first'] is not None else r['end'],end=r['end']) for r in active]
    trace=[]
    for offset in range(0,math.ceil(end-begin),20):
        lo=begin+offset;hi=min(end,lo+20)
        gs=[b-a for r in active for a,b in zip([t for t,_,_ in r['events']],[t for t,_,_ in r['events']][1:]) if lo<=a<b<hi]
        trace.append(dict(start_s=offset,duration_s=hi-lo,gap_p50_s=percentile(gs,.5),
                          gap_p95_s=percentile(gs,.95),gap_max_s=max(gs,default=None),
                          **interval_buckets(wall_rows,lo,hi)))
    agents=[]
    for agent in range(3):
        rs=[r for r in cleaned if r['agent']==agent];tt=[r['ttft_s'] for r in rs if r['ttft_s'] is not None]
        agents.append(dict(agent=agent,window_tokens=sum(r['window_tokens'] for r in rs),
                           turns_offered=sum(r['offered_in_window'] for r in rs),
                           ttft_p50_s=percentile(tt,.5),ttft_p95_s=percentile(tt,.95),
                           carry_in_requests=sum(r['carry_in'] for r in rs),
                           pending_at_cutoff=sum(r['pending_at_cutoff'] for r in rs),
                           queue_exposure_s=sum(r['queue_exposure_s'] for r in rs),
                           censored_ttft=sum(r['ttft_censored_s'] is not None for r in rs)))
    return dict(window_begin=begin,window_end=end,window_seconds=end-begin,
                window_tokens=sum(r['window_tokens'] for r in cleaned),
                aggregate_tps=sum(r['window_tokens'] for r in cleaned)/(end-begin),
                gap_p50_s=percentile(gaps,.5),gap_p95_s=percentile(gaps,.95),gap_max_s=max(gaps,default=None),
                agents=agents,trace_20s=trace,rows=cleaned)


def compare(reference,candidate):
    a=json.loads((reference/'summary.json').read_text());b=json.loads((candidate/'summary.json').read_text())
    if (a['family'],a['seed'],a['window_seconds'],a['metric'])!=(b['family'],b['seed'],b['window_seconds'],b['metric']):
        raise ValueError('different rate protocols')
    return dict(reference_tps=a['aggregate_tps'],candidate_tps=b['aggregate_tps'],
                throughput_ratio=b['aggregate_tps']/a['aggregate_tps'],
                reference_window_tokens=a['window_tokens'],candidate_window_tokens=b['window_tokens'],
                reference_thermal_max=a.get('thermal_max',0),candidate_thermal_max=b.get('thermal_max',0),
                thermal_flagged=bool(a.get('thermal_flagged') or b.get('thermal_flagged')),
                compile_overlapped=bool(a.get('compile_overlapped') or b.get('compile_overlapped')))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('plan');p.add_argument('--family',choices=('v41','glm','fn'),required=True)
    p.add_argument('--calls-csv',type=Path,default=DEFAULT_CALLS);p.add_argument('--seed',type=int,default=20261109)
    p.add_argument('--seconds',type=float,default=180);p.add_argument('--out',type=Path,required=True)
    p=sub.add_parser('run')
    for name in ('binary','tokenizer','model','corpus','plan','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--arm',choices=('serial','default'),required=True);p.add_argument('--ramp',type=float,default=15)
    p.add_argument('--warm-tokens',type=int,default=48000);p.add_argument('--context',type=int,default=196608)
    p.add_argument('--port',type=int,default=18737);p.add_argument('--cool',type=float,default=25)
    p.add_argument('--timeout',type=float,default=900);p.add_argument('--keep-bulk',action='store_true')
    p.add_argument('--harness-commit',default='working-tree')
    p.add_argument('--max-thermal-pressure',type=int,choices=(0,1),default=1,
                   help='retain level1 arms with flags by default; 0 restores strict pressure0')
    p=sub.add_parser('analyze');p.add_argument('directory',type=Path)
    p=sub.add_parser('compare');p.add_argument('reference',type=Path);p.add_argument('candidate',type=Path)
    args=parser.parse_args()
    if args.command=='plan':
        with args.calls_csv.open() as f:plan=make_plan(list(csv.DictReader(f)),args.family,args.seed,args.seconds)
        dump(args.out,plan);print(json.dumps({k:v for k,v in plan.items() if k!='population'}))
    elif args.command=='run':
        arm=Arm(args)
        def stop(sig,frame):
            arm.abort.set();arm.finish.set()
            if arm.server is not None and arm.server.poll() is None:arm.server.send_signal(signal.SIGINT)
            raise KeyboardInterrupt
        for sig in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP):signal.signal(sig,stop)
        arm.run()
    elif args.command=='analyze':print((args.directory/'summary.json').read_text())
    else:print(json.dumps(compare(args.reference,args.candidate)))


if __name__=='__main__':main()
