#!/usr/bin/env python3
"""Join opt-in shared lifecycle traces to served client calls; emit ms distributions."""
import argparse,collections,json,math,statistics
from pathlib import Path

def percentile(xs,p):
    xs=sorted(x for x in xs if x is not None)
    return xs[min(len(xs)-1,math.ceil(len(xs)*p)-1)] if xs else None

def intersect(a,b):return max(0,min(a[1],b[1])-max(a[0],b[0]))
def union(intervals):
    merged=[]
    for lo,hi in sorted(intervals):
        if hi<=lo:continue
        if merged and lo<=merged[-1][1]:merged[-1][1]=max(hi,merged[-1][1])
        else:merged.append([lo,hi])
    return merged

def length(intervals):return sum(b-a for a,b in union(intervals))
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('--out',type=Path,required=True);p.add_argument('--warm-only',action='store_true');a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    native=json.loads((a.run/'native-lines.json').read_text())
    traces={};parse_errors=[];prefills=[]
    for entry in native:
        line=entry['line']
        if 'serve ttft timing: ' in line:
            try:t=json.JSONDecoder().raw_decode(line.split('serve ttft timing: ',1)[1])[0]
            except Exception as e:parse_errors.append(str(e));continue
            if t['event']=='finished' or t['response_id'] not in traces:traces[t['response_id']]=t
        if 'serve prefill:' in line:
            import re
            f=dict(re.findall(r'(\w+)=([^\s]+)',line));f['_time']=entry['time'];prefills.append(f)
    calls_source=a.run/('warm-calls.jsonl' if a.warm_only else 'calls.jsonl')
    raw_fallback=not calls_source.exists()
    if raw_fallback:calls_source=a.run/'raw-calls.jsonl'
    calls=[] if a.warm_only else [json.loads(l) for l in calls_source.read_text().splitlines()]
    window=json.loads((a.run/'window.json').read_text()) if (a.run/'window.json').exists() else None
    if raw_fallback:
        assert window and window['begin'] is not None, 'no recorded window for interrupted calls'
        calls=[c for c in calls if not c['warm']]
        for c in calls:
            c['offered_in_window']=window['begin']<=c['sent']<window['end']
            c['ttft_s']=c['first']-c['sent'] if c['first'] is not None and c['first']<window['end'] and c['offered_in_window'] else None
    calls += [json.loads(l) for l in (a.run/'warm-calls.jsonl').read_text().splitlines()]
    labels={traces[c['response_id']]['request_id']:c['label'] for c in calls if c.get('response_id') in traces}
    result=[];missing=[];censored=[]
    for c in calls:
        warm=c.get('warm',False) or c['label'].endswith('-warm')
        if not warm and not c.get('offered_in_window'):continue
        if c.get('first') is None or (not warm and c.get('ttft_s') is None):censored.append(c['label']);continue
        t=traces.get(c.get('response_id'))
        if not t:missing.append(dict(label=c['label'],response_id=c.get('response_id')));continue
        s=t['stages_ns'];sp=t['spans'];epoch=int(t['epoch_unix_ns']);rid=t['request_id']
        new=t['new_rows'];kind='root' if warm else 'short' if 0<new<512 else '1k' if 512<=new<=2000 else 'other'
        sent=round(c['sent']*1e9)-epoch;first=round(c['first']*1e9)-epoch
        v={};intervals={}
        def put(name,lo,hi):
            if lo is not None and hi is not None:v[name]=(hi-lo)/1e6;intervals[name]=(lo,hi)
        def stages(name,lo,hi):put(name,s.get(lo),s.get(hi))
        put('client_ttft',sent,first);put('send_to_accept',sent,s.get('accepted'));stages('accept_to_http_worker','accepted','http_receive')
        stages('receive_body','http_receive','body_received');stages('body_to_parsed','body_received','parsed');stages('parsed_to_enqueued','parsed','enqueued')
        stages('work_mailbox_wait','enqueued','dequeued');stages('dequeue_to_submit','dequeued','submitted');stages('slot_queue_wait','submitted','admission_start');stages('admission','admission_start','admitted')
        stages('first_sample_to_raw','first_sampled','first_raw_piece');stages('raw_to_visible_flush','first_raw_piece','first_visible_flushed');put('visible_flush_to_client',s.get('first_visible_flushed'),first)
        put('server_to_visible',s.get('accepted'),s.get('first_visible_flushed'))
        for name in sorted(set(x['name'] for x in sp)):
            xs=[(x['start_ns'],x['end_ns']) for x in sp if x['name']==name and x['start_ns']<=s['first_visible_flushed']]
            # Inclusive parent spans are explicitly named; do not add them to their children.
            v['span_'+name]=length(xs)/1e6
        slices=sorted([x for x in sp if x['name']=='prefill_slice'],key=lambda x:x['start_ns'])
        if slices:
            put('admitted_to_first_slice',s.get('admitted'),slices[0]['start_ns'])
            v['interslice_wait']=sum(max(0,y['start_ns']-x['end_ns']) for x,y in zip(slices,slices[1:]))/1e6
            fin=[x for x in sp if x['name']=='finish_prefill']
            fend=max((x['end_ns'] for x in fin),default=slices[-1]['end_ns'])
            put('last_slice_to_finish',slices[-1]['end_ns'],fend);put('finish_to_first_sample',fend,s.get('first_sampled'))
        phases=c.get('phases') or c.get('x_qwen',{}).get('phases',{})
        native_ms=phases.get('prefill_ms')
        if native_ms is None:
            candidates=[r for r in prefills if int(r.get('prefill_tokens',-1))+int(r.get('reused_tokens',0))==t['prompt_tokens']]
            target=(epoch+s.get('prefill_complete',s['first_visible_flushed']))/1e9
            if candidates:
                near=min(candidates,key=lambda r:abs(r['_time']-target))
                if abs(near['_time']-target)<3:native_ms=float(near['prefill_ms'])
        if native_ms is not None:
            v['native_prefill']=float(native_ms);v['outside_native']=v['client_ttft']-v['native_prefill']
        buckets=('send_to_accept','accept_to_http_worker','receive_body','body_to_parsed','parsed_to_enqueued','work_mailbox_wait','dequeue_to_submit','slot_queue_wait','admission','admitted_to_first_slice','span_prefill_slice','interslice_wait','last_slice_to_finish','finish_to_first_sample','first_sample_to_raw','raw_to_visible_flush','visible_flush_to_client')
        if all(k in v for k in buckets):
            v['additive_balance_error']=v['client_ttft']-sum(v[k] for k in buckets)
            assert abs(v['additive_balance_error'])<.001, (c['label'],v['additive_balance_error'])
        # Wait intervals when this request is ready for the next host phase, excluding its own admission work.
        waits=[intervals[n] for n in ('accept_to_http_worker','work_mailbox_wait','dequeue_to_submit','slot_queue_wait','admitted_to_first_slice') if n in intervals]
        waits += [(x['end_ns'],y['start_ns']) for x,y in zip(slices,slices[1:])]
        waits=union(waits)
        peers=[]
        for r in t['rounds']:
            if rid in r['request_ids']:continue
            covered=[]
            for lo,hi in waits:
                l=max(lo,r['start_ns']);h=min(hi,r['end_ns'])
                if h>l:covered.append((l,h))
            overlap=length(covered)
            if overlap:peers.append(dict(round_id=r['round_id'],kind=r['kind'],request_ids=r['request_ids'],labels=[labels.get(i,'request-id:'+str(i)) for i in r['request_ids']],wait_overlap_ms=overlap/1e6,start_ns=r['start_ns'],end_ns=r['end_ns']))
        for k in ('prefill','decode'):
            # Raw overlaps can nest: peer decode inside another peer's outer prefill slice.
            rr=[r for r in t['rounds'] if rid not in r['request_ids'] and r['kind']==k]
            v['wait_covered_peer_'+k]=length([(max(lo,r['start_ns']),min(hi,r['end_ns'])) for lo,hi in waits for r in rr])/1e6
        v['ready_wait_total']=length(waits)/1e6
        allpeer=[r for r in t['rounds'] if rid not in r['request_ids']]
        v['wait_covered_any_peer']=length([(max(lo,r['start_ns']),min(hi,r['end_ns'])) for lo,hi in waits for r in allpeer])/1e6
        if 'outside_native' in v:v['outside_native_without_logged_wait']=v['outside_native']-v['ready_wait_total']
        v['tokenization_total']=sum(v.get('span_'+k,0) for k in ('tokenize_full','tokenize_full_backend','tokenize_boundaries'))
        v['cache_total_inclusive']=sum(v.get('span_'+k,0) for k in ('prefix_prepare','prefix_take','prefix_checkpoint'))
        row=dict(label=c['label'],response_id=c.get('response_id'),request_id=rid,kind=kind,prompt=t['prompt_tokens'],reused=t['reused_tokens'],new=new,completed=c['completed'],metrics_ms=v,peer_rounds=peers,slices=slices,spans_dropped=t['spans_dropped'],trace_event=t['event'])
        result.append(row)
    distributions={}
    for kind in ('short','1k','root','other'):
        rows=[r for r in result if r['kind']==kind]
        if not rows:continue
        keys=sorted(set(k for r in rows for k in r['metrics_ms']))
        distributions[kind]=dict(n=len(rows),completed=sum(r['completed'] for r in rows),metrics={k:dict(n=sum(k in r['metrics_ms'] for r in rows),p50=percentile([r['metrics_ms'].get(k) for r in rows],.5),p95=percentile([r['metrics_ms'].get(k) for r in rows],.95),max=max(r['metrics_ms'].get(k,-float('inf')) for r in rows)) for k in keys})
    out=dict(calls_source=calls_source.name,interrupted=(a.run/'failed.json').exists(),planned_window=window,distributions=distributions,missing=missing,censored=censored,parse_errors=parse_errors,traces=len(traces),joined=len(result),trace_spans_dropped=sum(r['spans_dropped'] for r in result))
    (a.out/'summary.json').write_text(json.dumps(out,indent=2)+'\n');(a.out/'calls.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in result))
    print(json.dumps(out,indent=2))
if __name__=='__main__':main()
