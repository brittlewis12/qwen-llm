import argparse,collections,json,math,re,statistics,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('runs',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
root=Path.cwd();analyzer=root/'docs/bench/2026-11-14-serve-ttft-instrumentation/analyze.py'
def pct(xs,q):
    xs=sorted(xs);return xs[min(len(xs)-1,math.ceil(len(xs)*q)-1)] if xs else None
def gm(xs):return math.exp(sum(math.log(x) for x in xs)/len(xs)) if xs and all(x is not None and x>0 for x in xs) else None
def lines(path):return [json.loads(l) for l in path.read_text().splitlines()]
arms={};calls={};metadata={}
for tag in ('A1','B1','B2','A2','A3','B3'):
    src=a.runs/tag
    if not (src/'result.json').exists():continue
    result=json.loads((src/'result.json').read_text());serve=src/'serve';window=json.loads((serve/'window.json').read_text())
    subprocess.run([sys.executable,str(analyzer),str(serve),'--out',str(a.out/tag)],check=True,stdout=subprocess.DEVNULL)
    timing=json.loads((a.out/tag/'summary.json').read_text())
    # Inclusive prefill spans include a parent's peer service; native work
    # deliberately excludes it. Keep the residual explicitly labelled rather
    # than claiming every microsecond is GPU or queue time.
    ledger=lines(a.out/tag/'calls.jsonl')
    for kind,distribution in timing['distributions'].items():
        values=[r['metrics_ms']['span_prefill_slice']-r['metrics_ms']['native_prefill'] for r in ledger if r['kind']==kind and 'span_prefill_slice' in r['metrics_ms'] and 'native_prefill' in r['metrics_ms']]
        if values:distribution['metrics']['inside_prefill_residual']=dict(n=len(values),p50=pct(values,.5),p95=pct(values,.95),max=max(values))
    native=json.loads((serve/'native-lines.json').read_text());inputs={};traces={};rounds={}
    for event in native:
        for prefix,out in [('serve input witness: ',inputs),('serve ttft timing: ',traces)]:
            if prefix in event['line']:
                r=json.JSONDecoder().raw_decode(event['line'].split(prefix,1)[1])[0]
                if r.get('response_id'):out[r['response_id']]=r
    for event in native:
        if 'serve round timing: ' in event['line']:
            r=json.JSONDecoder().raw_decode(event['line'].split('serve round timing: ',1)[1])[0];rounds[r['round_id']]=r
    raw=lines(serve/'raw-calls.jsonl')
    gaps=[b[0]-x[0] for r in raw for x,b in zip(r['events'],r['events'][1:]) if window['begin']<=x[0]<b[0]<window['end']]
    gap_tail=dict(n=len(gaps),over_100ms=sum(g>.1 for g in gaps),over_250ms=sum(g>.25 for g in gaps),max_s=max(gaps) if gaps else None)
    prefills=[r for r in rounds.values() if r['kind']=='prefill'];nested=[]
    epoch=int(next(iter(traces.values()))['epoch_unix_ns']);begin_ns=int(window['begin']*1e9)-epoch;end_ns=int(window['end']*1e9)-epoch
    for child in prefills:
        candidates=[r for r in prefills if r['round_id']!=child['round_id'] and r['start_ns']<=child['start_ns']<child['end_ns']<=r['end_ns'] and r['request_ids']!=child['request_ids']]
        if candidates:
            parent=min(candidates,key=lambda r:r['end_ns']-r['start_ns']);nested.append(dict(parent_round_id=parent['round_id'],child_round_id=child['round_id'],parent_requests=parent['request_ids'],child_requests=child['request_ids'],child_ms=(child['end_ns']-child['start_ns'])/1e6,in_window=begin_ns<=child['start_ns']<end_ns))
    yield_events=[];priority_events=[]
    for event in native:
        for marker,out in [('serve prefill yield: ',yield_events),('serve prefill priority: ',priority_events)]:
            if marker not in event['line']:continue
            fields=dict(re.findall(r'(\w+)=([^\s]+)',event['line'].split(marker,1)[1]))
            request=int(fields.get('child_request_id',fields.get('request_id')))
            child_rounds=[r for r in prefills if request in r['request_ids']]
            fields['in_window']=any(begin_ns<=r['start_ns']<end_ns for r in child_rounds)
            out.append(fields)
    poll_rows=[r.get('x_qwen',{}).get('phases',{}) for r in raw]
    conditional_counts=dict(full_arm_yields=len(yield_events),window_yields=sum(r['in_window'] for r in yield_events),full_arm_priority_dispatches=len(priority_events),window_priority_dispatches=sum(r['in_window'] for r in priority_events),completed_response_polls=sum(r.get('yield_polls',0) for r in poll_rows),completed_warm_root_polls=sum(r.get('x_qwen',{}).get('phases',{}).get('yield_polls',0) for r in raw if r['warm']),completed_window_polls=sum(r.get('x_qwen',{}).get('phases',{}).get('yield_polls',0) for r in raw if not r['warm'] and window['begin']<=r['sent']<window['end']),yield_events=yield_events,priority_events=priority_events)
    decode_widths=collections.Counter(len(r['request_ids']) for r in rounds.values() if r['kind']=='decode' and begin_ns<=r['start_ns']<end_ns)
    calls[tag]={r['label']:r for r in lines(serve/'raw-calls.jsonl') if not r['warm'] and window['begin']<=r['sent']<window['end']}
    metadata[tag]=(inputs,traces)
    roots=lines(serve/'warm-calls.jsonl');loads=lines(src/'host-load.jsonl');root_rows=[]
    for r in roots:
        ls=[v['load'][0] for v in loads if r['sent']<=v['time']<=(r['first'] or r['end'])]
        root_rows.append(dict(label=r['label'],selected_rows=r['x_qwen']['phases'].get('prefill_selected_rows'),prefill_mode=r['x_qwen']['phases'].get('prefill_mode'),native_ms=r['x_qwen']['phases']['prefill_ms'],client_ms=(r['first']-r['sent'])*1000,load1_min=min(ls) if ls else None,load1_p50=pct(ls,.5),load1_max=max(ls) if ls else None))
    io=lines(src/'arm.process-io.jsonl');wi=[r for r in io if window['begin']<=r['time']<=window['end']]
    arms[tag]=dict(conditional_counts=conditional_counts,event_gap_tail=gap_tail,nested_peer_prefills=nested,decode_round_widths=dict(decode_widths),tps=result['summary']['aggregate_tps'],gap_p99_s=result['summary']['gap_p99_s'],startup_s=result['summary']['startup_s'],warmup_s=result['summary']['warmup_s'],model_niceness=result['model_niceness'],roots=root_rows,process_disk_read_bytes=result['process_disk_read_bytes'],sampled_window_disk_read_bytes=wi[-1]['disk_read']-wi[0]['disk_read'] if len(wi)>1 else None,thermal_max=result['summary']['thermal_max'],compile_overlapped=result['summary']['compile_overlapped'],binary_sha256=result['binary_sha256'],components=timing['distributions'],censored=timing['censored'],missing=timing['missing'])
pairs=[];pooled_a=[];pooled_b=[]
for number,aa,bb,order in [(1,'A1','B1','ab'),(2,'A2','B2','ba'),(3,'A3','B3','ab')]:
    if aa not in arms or bb not in arms:continue
    assert arms[aa]['binary_sha256']==arms[bb]['binary_sha256']
    ca,cb=calls[aa],calls[bb];ia,ta=metadata[aa];ib,tb=metadata[bb];matched=[];excluded=[]
    for label in sorted(ca.keys()|cb.keys()):
        if label not in ca or label not in cb:excluded.append(dict(label=label,reason='not_offered_in_both_windows'));continue
        x,y=ca[label],cb[label];px,py=ia.get(x.get('response_id')),ib.get(y.get('response_id'));tx,ty=ta.get(x.get('response_id')),tb.get(y.get('response_id'))
        if not all((px,py,tx,ty)):excluded.append(dict(label=label,reason='missing_input_or_trace'));continue
        if x['spec']!=y['spec'] or x['target']!=y['target'] or any(px[k]!=py[k] for k in ('prompt_sha256','token_sha256','token_count')) or tx['new_rows']!=ty['new_rows']:
            excluded.append(dict(label=label,reason='input_spec_or_rows_differ'));continue
        rows=tx['new_rows'];kind='short' if 0<rows<512 else '1k' if 512<=rows<=2000 else 'other'
        r=dict(label=label,rows=rows,kind=kind,ttft_a=None,ttft_b=None,native_a=None,native_b=None,completion_a=None,completion_b=None,generated_a=x['generated'],generated_b=y['generated'])
        if x['first'] is not None and y['first'] is not None:r.update(ttft_a=(x['first']-x['sent'])*1000,ttft_b=(y['first']-y['sent'])*1000)
        if x['completed'] and y['completed']:
            r['native_a']=x['x_qwen'].get('phases',{}).get('prefill_ms');r['native_b']=y['x_qwen'].get('phases',{}).get('prefill_ms')
            if x['generated']==y['generated'] and x['prompt']==y['prompt']:
                r.update(completion_a=(x['end']-x['sent'])*1000,completion_b=(y['end']-y['sent'])*1000)
                pooled_a.append(r['completion_a']);pooled_b.append(r['completion_b'])
        matched.append(r)
    short=[r for r in matched if r['kind']=='short' and r['ttft_a'] is not None]
    short_a=pct([r['ttft_a'] for r in short],.95);short_b=pct([r['ttft_b'] for r in short],.95)
    ratios=dict(short_ttft_p95_ba=short_b/short_a if short_a and short_b else None,tps_ba=arms[bb]['tps']/arms[aa]['tps'],root_native_ab=sum(r['native_ms'] for r in arms[aa]['roots'])/sum(r['native_ms'] for r in arms[bb]['roots']),root_client_ab=sum(r['client_ms'] for r in arms[aa]['roots'])/sum(r['client_ms'] for r in arms[bb]['roots']),gap_p99_ba=arms[bb]['gap_p99_s']/arms[aa]['gap_p99_s'])
    for kind in ('short','1k'):
        group=[r for r in matched if r['kind']==kind and r['native_a'] is not None and r['native_b'] is not None]
        ratios[kind+'_native_ab']=sum(r['native_a'] for r in group)/sum(r['native_b'] for r in group) if group else None
    pairs.append(dict(pair=number,order=order,baseline=aa,candidate=bb,ratios=ratios,short_ttft_p95_ms=dict(a=short_a,b=short_b,n=len(short)),matched=matched,excluded=excluded))
thresholds=dict(short_ttft_p95_ba=('lt',1),tps_ba=('ge',.98),root_native_ab=('ge',.98),root_client_ab=('ge',.98),short_native_ab=('ge',.98),**{'1k_native_ab':('ge',.98)},gap_p99_ba=('le',1.25))
gate={}
if len(pairs)==3:
    for k,(op,t) in thresholds.items():
        vals=[r['ratios'][k] for r in pairs];g=gm(vals);ok=g is not None and (g<t if op=='lt' else g>=t if op=='ge' else g<=t)
        gate[k]=dict(pairs=vals,geomean=g,operation=op,threshold=t,pass_=ok)
    median_a,median_b=pct(pooled_a,.5),pct(pooled_b,.5);ratio=median_b/median_a if median_a and median_b else None
    gate['pooled_completion_p50_ba']=dict(ratio=ratio,baseline_ms=median_a,candidate_ms=median_b,matched_n=len(pooled_a),operation='le',threshold=1,pass_=ratio is not None and ratio<=1)
summary=dict(arms=arms,pairs=pairs,complete=len(pairs)==3,gate=gate,performance_pass=all(r['pass_'] for r in gate.values()) if gate else None)
(a.out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(dict(complete=summary['complete'],performance_pass=summary['performance_pass'],gate=gate),indent=2))
