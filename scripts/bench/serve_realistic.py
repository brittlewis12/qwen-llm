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


def arm_environment(source, family, arm, overrides=()):
    env = dict(source, RUST_LOG='warn,qwen_diag=info', NO_COLOR='1')
    knobs = dict(v41='QWEN_SERVE_V41_MAX_ACTIVE', glm='QWEN_SERVE_GLM_MAX_ACTIVE',
                 fn='QWEN_SERVE_FN_MAX_ACTIVE')
    for knob in knobs.values():
        env.pop(knob, None)
    if arm == 'serial':
        env[knobs[family]] = '1'
    for item in overrides:
        key, sep, value = item.partition('=')
        if not sep or not key:
            raise ValueError(f'--env wants KEY=VALUE, got {item!r}')
        env[key] = value
    return env


def sampling_fields(sampling, temperature=None, top_p=None, top_k=None, min_p=None):
    """Request sampling fields, defaulting to what agents actually send.

    Live agents omit every sampling field, so the model's released defaults
    apply (GLM: temperature 1.0, top-p 0.95). Setting fields here would replace
    those defaults and understate host sampling cost.
    """
    if sampling == 'agent':
        return {}
    if sampling == 'greedy':
        return dict(temperature=0, top_p=1)
    if sampling != 'custom':
        raise ValueError(f'unknown sampling mode {sampling!r}')
    fields = {}
    for key, value in (('temperature', temperature), ('top_p', top_p),
                       ('top_k', top_k), ('min_p', min_p)):
        if value is not None:
            fields[key] = value
    if not fields:
        raise ValueError('--sampling custom needs at least one sampling value')
    return fields


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


def thermal_phase_limit(maximum, warmup_maximum, timing_started):
    # The ramp starts when window_begin is assigned. A warm-up exception can
    # never relax the measured-window (or ramp) limit.
    return maximum if timing_started or warmup_maximum is None else warmup_maximum


class ThermalSettle:
    def __init__(self, seconds, maximum):
        self.seconds, self.maximum, self.since = seconds, maximum, None

    def ready(self, level, now):
        if thermal_exceeds_limit(level, self.maximum):
            self.since = None
            return False
        if self.since is None:
            self.since = now
        return now-self.since >= self.seconds


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
        self.sampling = sampling_fields(args.sampling, args.temperature, args.top_p,
                                        args.top_k, args.min_p)
        self.rows, self.samples, self.lines = [], [], []
        self.server = None
        self.abort = threading.Event()
        self.finish = threading.Event()
        self.monitor_done = threading.Event()
        self.lock = threading.Lock()
        self.window_begin = self.window_end = None
        self.warm_settle_s = 0.0
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
                limit = thermal_phase_limit(self.args.max_thermal_pressure,
                    getattr(self.args, 'warmup_max_thermal_pressure', None), self.window_begin is not None)
                if thermal_exceeds_limit(p,limit) or time.monotonic() > self.deadline:
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

    def settle_after_warm(self):
        seconds = getattr(self.args, 'settle_thermal_seconds', 0)
        if seconds <= 0:
            return
        started = time.monotonic()
        settle = ThermalSettle(seconds, self.args.max_thermal_pressure)
        self.emit(event='thermal_settle_begin', seconds=seconds,
                  maximum=self.args.max_thermal_pressure)
        while True:
            self.check()
            if settle.ready(pressure(), time.monotonic()):
                break
            self.finish.wait(1)
        self.warm_settle_s = time.monotonic()-started
        self.emit(event='thermal_settle_done', waited_s=self.warm_settle_s, thermal=pressure())

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
                    stream=True, max_output_tokens=count, **self.sampling,
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
        preflight=[]
        for _ in range(3):
            level=pressure();preflight.append(dict(time=time.time(),thermal=level))
            if thermal_exceeds_limit(level,args.max_thermal_pressure):
                raise RuntimeError('thermal pressure exceeds arm limit before arm')
            time.sleep(3)
        occupied = subprocess.run(['lsof','-nP',f'-iTCP:{args.port}','-sTCP:LISTEN'],stdout=subprocess.DEVNULL)
        if occupied.returncode != 1:
            raise RuntimeError('port occupied or lsof failed')
        env = arm_environment(os.environ, self.plan['family'], args.arm, args.env)
        maximum = max(t['reply_tokens'] for t in self.plan['population'])
        command = [str(args.binary.resolve()),'serve','-m',str(args.model.resolve()),
                   '--addr',f'127.0.0.1:{args.port}','--max-context-tokens',str(args.context),
                   '--max-tokens',str(maximum)]
        dump(self.out/'launch.json', dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
             command=command, harness_command=sys.argv, arm=args.arm,
             harness_commit=args.harness_commit,
             sampling=dict(mode=args.sampling, fields=self.sampling),
             env_overrides=list(args.env),
             cool_seconds=args.cool, warm_cool_seconds=args.warm_cool,
             ramp_seconds=args.ramp, context=args.context,
             preflight_thermal=preflight,max_thermal_pressure=args.max_thermal_pressure,
             warmup_max_thermal_pressure=args.warmup_max_thermal_pressure,
             settle_thermal_seconds=args.settle_thermal_seconds,
             thermal_policy='ramp/window limit unchanged; optional warm-up limit and pre-ramp settle',
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
                if agent<2 and args.warm_cool:
                    if self.finish.wait(args.warm_cool):
                        raise RuntimeError('aborted during warm-up cooling')
            warmed=time.time()
            self.emit(event='warmed',startup_s=ready-started,warmup_s=warmed-ready)
            self.settle_after_warm()
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
            # Server phase lines are part of the evidence: they carry per-round
            # transition and host time for every completed request.
            dump(self.out/'native-lines.json',self.lines)
            if error or self.abort.is_set():
                dump(self.out/'failed.json',dict(error=error,thermal_abort=self.abort.is_set()))
            self.emit(event='server_stopped',code=self.server.returncode)
        if self.abort.is_set():
            raise RuntimeError('arm aborted; no accepted timing summary')
        result=summarize(self.rows,self.window_begin,self.window_end,
                         lambda text:len(token_text(args.tokenizer,args.model,text=text)),self.lines)
        result.update(startup_s=ready-started,warmup_s=warmed-ready,warm_settle_s=self.warm_settle_s,arm=args.arm,
                      family=self.plan['family'],seed=self.plan['seed'],thermal_max=max(x['thermal'] for x in self.samples),
                      sampling=dict(mode=args.sampling, fields=self.sampling),
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


def native_records(native_lines):
    """Parse `serve native:` and `serve phases:` lines into per-call records.

    A GLM `serve native:` line carries transition_ms and precedes its phases
    line; V4.1's phases line already carries transition_ms. Times are kept so a
    record can be matched to the call that finished at that moment.
    """
    records=[];pending=None
    for entry in native_lines:
        line=entry['line']
        if 'serve native:' in line:
            pending=kv(line);pending['_time']=entry['time']
        elif 'serve phases:' in line:
            rec=kv(line);rec['_time']=entry['time']
            if pending is not None and pending.get('family')==rec.get('family'):
                rec['_transition_ms']=pending.get('transition_ms')
            records.append(rec);pending=None
    return records


def match_native(records,row):
    "Nearest phases line with the same prompt length (completed call)."
    if row.get('prompt') is None:return None
    best=None
    for rec in records:
        try:prompt=int(float(rec.get('prefill_tokens',0)))+int(float(rec.get('reused_tokens',0)))
        except (TypeError,ValueError):continue
        if prompt!=row['prompt']:continue
        delta=abs(rec['_time']-row['end'])
        if delta>3:continue
        if best is None or delta<best[0]:best=(delta,rec)
    return best[1] if best else None


def prefill_records(native_lines):
    """Parse `serve prefill:` lines emitted at prefill completion or cancel.

    A completed turn also carries the same `prefill_ms` in its phases line; a
    cancelled turn emits no phases line, so this is the only record of the
    prefill it paid. Each record times the end of the prefill span.
    """
    out=[]
    for entry in native_lines:
        line=entry['line']
        if 'serve prefill:' in line:
            rec=kv(line);rec['_time']=entry['time'];out.append(rec)
    return out


def round_budget(cleaned,native_lines,begin,end,gap_p50_s=None):
    """Owner-time budget from the server's own per-call phase lines.

    Each completed call contributes its average native round wall
    (`transition_ms / steps`) and its estimated prefill/capture span. The
    cohort host per round is the median content gap minus that round wall:
    stable and positive, and it leaves the long append stalls in the gap tail
    rather than smearing them into every round. Prefill/capture are counted
    only where their estimated span overlaps the window. Partial calls have no
    phases line, so their prefill comes from `serve prefill: complete=false`
    records and is reported separately as `partial_prefill_s` (included in
    `prefill_s`); `coverage` still states how many delivered tokens were
    observed by completed phases lines.
    """
    records=native_records(native_lines)
    spans=[]
    for r in cleaned:
        ph=r.get('phases') or {}
        batch=ph.get('batch') or {}
        st=float(batch.get('stepped_transitions') or 0)
        wm=float(batch.get('width_mean') or 0)
        first=r['first']
        if st<=0 or wm<=0 or first is None:continue
        source=ph
        trans=source.get('transition_ms')
        pre=source.get('prefill_ms')
        if trans is None or pre is None:
            rec=match_native(records,r)
            if rec is not None:source=rec
        if trans is None:trans=source.get('_transition_ms',source.get('transition_ms'))
        if pre is None:pre=source.get('prefill_ms')
        trans=float(trans or 0);pre=float(pre or 0)
        cap=sum(float(source.get(k) or 0) for k in
                ('restore_ms','boundary_ms','snapshot_capture_ms',
                 'ladder_tip_capture_ms','ladder_capture_ms','retention_ms'))
        lead=(pre+cap)/1000
        pre_start=first-lead
        spans.append(dict(row=r,first=first,end=r['end'],steps=st,width=wm,
                          transition=trans,prefill=pre,capture=cap,
                          pre_start=pre_start))
    def overlap(a_lo,a_hi,b_lo,b_hi):
        return max(0.0,min(a_hi,b_hi)-max(a_lo,b_lo))
    transition_ms=steps=wsum=0.0
    prefill_s=capture_s=0.0
    for s in spans:
        decode_overlap=overlap(s['first'],s['end'],begin,end)
        if decode_overlap<=0:continue
        lead=s['first']-s['pre_start']
        lead_overlap=overlap(s['pre_start'],s['first'],begin,end)
        if lead>0:
            prefill_s+=s['prefill']/1000*(lead_overlap/lead)
            capture_s+=s['capture']/1000*(lead_overlap/lead)
        decode_ms=s['row']['end']-s['first']
        frac=decode_overlap/decode_ms if decode_ms>0 else 0.0
        steps_in=frac*s['steps']
        transition_ms+=frac*s['transition']
        steps+=steps_in
        wsum+=steps_in*s['width']
    # Prefill records whose turn never completed (cancelled mid-prefill or
    # during decode). A completed turn's record is already in the spans above
    # and is recognised by its prompt length and its near-first-token time.
    finished=[(r.get('prompt'),r['first']) for r in cleaned
              if r.get('completed') and r['first'] is not None]
    partial_prefill_s=0.0
    partial_prefill_calls=0
    for rec in prefill_records(native_lines):
        try:pre=float(rec.get('prefill_ms') or 0)/1000
        except (TypeError,ValueError):continue
        if pre<=0:continue
        try:prompt=int(float(rec.get('prefill_tokens',0)))+int(float(rec.get('reused_tokens',0)))
        except (TypeError,ValueError):prompt=None
        if prompt is not None and any(p==prompt and abs(rec['_time']-f)<=3
                                      for p,f in finished):
            continue
        lo=rec['_time']-pre
        span=rec['_time']-lo
        overlap_s=overlap(lo,rec['_time'],begin,end)
        if overlap_s>0 and span>0:
            partial_prefill_s+=pre*(overlap_s/span)
            partial_prefill_calls+=1
    prefill_s+=partial_prefill_s
    if steps<=0:
        return dict(rounds_est=0.0,avg_width=0.0,coverage=0.0,transition_s=0.0,
                    prefill_s=prefill_s,partial_prefill_s=partial_prefill_s,
                    partial_prefill_calls=partial_prefill_calls,
                    capture_s=capture_s,host_s=0.0,
                    transition_ms_per_round=None,host_ms_per_round=None,
                    host_metric='median content gap minus native round wall')
    avg_width=wsum/steps
    rounds=steps/avg_width
    transition_per_round=transition_ms/steps
    host_per_round=None
    if gap_p50_s is not None:
        host_per_round=max(0.0,gap_p50_s*1000-transition_per_round)
    total_tokens=sum(r['window_tokens'] for r in cleaned)
    covered=sum((s['row']['window_tokens']) for s in spans)
    return dict(rounds_est=rounds,avg_width=avg_width,
                coverage=(covered/total_tokens) if total_tokens else 0.0,
                transition_s=transition_per_round*rounds/1000,
                prefill_s=prefill_s,partial_prefill_s=partial_prefill_s,
                partial_prefill_calls=partial_prefill_calls,capture_s=capture_s,
                host_s=(host_per_round*rounds/1000) if host_per_round is not None else 0.0,
                transition_ms_per_round=transition_per_round,
                host_ms_per_round=host_per_round,
                host_metric='median content gap minus native round wall')

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
             phases=native,spec=r['spec'],events=r['events']))
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
    budget=round_budget(cleaned,native_lines,begin,end,percentile(gaps,.5))
    return dict(window_begin=begin,window_end=end,window_seconds=end-begin,
                window_tokens=sum(r['window_tokens'] for r in cleaned),
                aggregate_tps=sum(r['window_tokens'] for r in cleaned)/(end-begin),
                gap_p50_s=percentile(gaps,.5),gap_p95_s=percentile(gaps,.95),
                gap_p99_s=percentile(gaps,.99),gap_max_s=max(gaps,default=None),
                round_budget=budget,
                agents=agents,trace_20s=trace,rows=cleaned)


def compare(reference,candidate):
    a=json.loads((reference/'summary.json').read_text());b=json.loads((candidate/'summary.json').read_text())
    if ((a['family'],a['seed'],a['window_seconds'],a['metric'],a.get('sampling'))
            !=(b['family'],b['seed'],b['window_seconds'],b['metric'],b.get('sampling'))):
        raise ValueError('different rate protocols')
    ttft=[]
    for x,y in zip(a.get('agents',[]),b.get('agents',[])):
        if x['agent']!=y['agent']:raise ValueError('different agent populations')
        ttft.append(dict(agent=x['agent'],reference_p50_s=x['ttft_p50_s'],candidate_p50_s=y['ttft_p50_s'],
                         reference_p95_s=x['ttft_p95_s'],candidate_p95_s=y['ttft_p95_s'],
                         reference_censored=x.get('censored_ttft',0),candidate_censored=y.get('censored_ttft',0)))
    return dict(ttft=ttft,reference_tps=a['aggregate_tps'],candidate_tps=b['aggregate_tps'],
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
    p.add_argument('--warm-cool',type=float,default=0,
                   help='seconds to idle between warm roots, to shed heat before a deep window')
    p.add_argument('--port',type=int,default=18737);p.add_argument('--cool',type=float,default=25)
    p.add_argument('--sampling',choices=('agent','greedy','custom'),default='agent',
                   help='agent (default) omits sampling fields as live agents do; greedy sets temperature 0/top-p 1; custom uses supplied values')
    p.add_argument('--temperature',type=float);p.add_argument('--top-p',type=float,dest='top_p')
    p.add_argument('--top-k',type=int,dest='top_k');p.add_argument('--min-p',type=float,dest='min_p')
    p.add_argument('--env',action='append',default=[],metavar='KEY=VALUE',
                   help='extra environment for this arm (repeatable), e.g. a rollback knob')
    p.add_argument('--timeout',type=float,default=900);p.add_argument('--keep-bulk',action='store_true')
    p.add_argument('--harness-commit',default='working-tree')
    p.add_argument('--max-thermal-pressure',type=int,choices=(0,1),default=1,
                   help='retain level1 arms with flags by default; 0 restores strict pressure0')
    p.add_argument('--warmup-max-thermal-pressure',type=int,choices=(0,1,2),default=None,
                   help='optional warm-up-only limit; ramp/window retain --max-thermal-pressure')
    p.add_argument('--settle-thermal-seconds',type=float,default=0,
                   help='require this many continuous seconds at the measurement thermal limit after warming')
    p=sub.add_parser('analyze');p.add_argument('directory',type=Path)
    p=sub.add_parser('compare');p.add_argument('reference',type=Path);p.add_argument('candidate',type=Path)
    args=parser.parse_args()
    if args.command=='plan':
        with args.calls_csv.open() as f:plan=make_plan(list(csv.DictReader(f)),args.family,args.seed,args.seconds)
        dump(args.out,plan);print(json.dumps({k:v for k,v in plan.items() if k!='population'}))
    elif args.command=='run':
        if not math.isfinite(args.settle_thermal_seconds) or args.settle_thermal_seconds < 0:
            parser.error('--settle-thermal-seconds must be finite and nonnegative')
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
