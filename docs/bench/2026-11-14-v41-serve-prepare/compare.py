#!/usr/bin/env python3
"""Reproduce component, cache and model-output comparisons from retained arms."""
import argparse,collections,hashlib,json,math,subprocess,sys
from pathlib import Path

def load_rows(p):return {r['label']:r for r in map(json.loads,(p/'raw-calls.jsonl').read_text().splitlines())}
def channels(r):
    d=collections.defaultdict(str)
    for _,k,t in r['events']:d[k]+=t
    return dict(d)
def percentile(xs,p):
    xs=sorted(xs);return xs[min(len(xs)-1,math.ceil(len(xs)*p)-1)] if xs else None
def geomean(xs):return math.exp(sum(map(math.log,xs))/len(xs)) if xs else None

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('reference',type=Path);p.add_argument('candidate',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    analyzer=Path(__file__).resolve().parent.parent/'2026-11-14-serve-ttft-instrumentation/analyze.py'
    for name,path in [('reference',a.reference),('candidate',a.candidate)]:
        subprocess.run([sys.executable,str(analyzer),str(path),'--out',str(a.out/name)],check=True,stdout=subprocess.DEVNULL)
    distributions={name:json.loads((a.out/name/'summary.json').read_text()) for name in ('reference','candidate')}
    ledgers={name:{r['label']:r for r in map(json.loads,(a.out/name/'calls.jsonl').read_text().splitlines())} for name in ('reference','candidate')}
    raw={name:load_rows(path) for name,path in [('reference',a.reference),('candidate',a.candidate)]}
    output=[]
    for label in sorted(raw['reference'].keys()&raw['candidate'].keys()):
        x,y=raw['reference'][label],raw['candidate'][label];xc,yc=channels(x),channels(y);complete=x['completed'] and y['completed']
        exact=xc==yc if complete else all(xc.get(k,'').startswith(yc.get(k,'')) or yc.get(k,'').startswith(xc.get(k,'')) for k in xc.keys()|yc.keys())
        if not yc:status='candidate_no_visible_output'
        elif not xc:status='reference_no_visible_output'
        elif complete:status='complete_exact' if exact else 'complete_different'
        else:status='partial_prefix_equal' if exact else 'partial_prefix_different'
        first_difference={}
        if not exact:
            for k in xc.keys()|yc.keys():
                bx,by=xc.get(k,'').encode(),yc.get(k,'').encode()
                if bx!=by:first_difference[k]=next((i for i,(u,v) in enumerate(zip(bx,by)) if u!=v),min(len(bx),len(by)))
        output.append(dict(label=label,status=status,complete_pair=complete,output_exact=exact,first_difference_byte=first_difference,
            reference_generated=x['generated'],candidate_generated=y['generated'],reference_prompt=x['prompt'],candidate_prompt=y['prompt'],reference_reused=x['reused'],candidate_reused=y['reused'],
            reference_sha256={k:hashlib.sha256(v.encode()).hexdigest() for k,v in xc.items()},candidate_sha256={k:hashlib.sha256(v.encode()).hexdigest() for k,v in yc.items()}))
    quality=dict(common_calls=len(output),status_counts=dict(collections.Counter(r['status'] for r in output)),all_completed_output_exact=all(r['output_exact'] for r in output if r['complete_pair']),all_observed_output_prefixes_exact=all(r['output_exact'] for r in output if r['status'] not in ('candidate_no_visible_output','reference_no_visible_output')),rows=output)
    (a.out/'output-comparison.json').write_text(json.dumps(quality,indent=2)+'\n')
    traces={}
    for e in json.loads((a.candidate/'native-lines.json').read_text()):
        if 'serve ttft timing: ' in e['line']:
            t=json.loads(e['line'].split('serve ttft timing: ',1)[1]);traces[t['response_id']]=t
    for r in ledgers['candidate'].values():
        cs=[s for s in traces[r['response_id']]['spans'] if s['name']=='tokenize_reused_prefix']
        r['token_cache']=dict(reused_tokens=sum(c['position'] for c in cs),encoded_suffix_tokens=sum(c['rows'] for c in cs),records=len(cs))
    (a.out/'candidate-calls-with-cache.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in ledgers['candidate'].values()))
    result={}
    for kind in ('short','1k','root','other'):
        ra=distributions['reference']['distributions'].get(kind);rb=distributions['candidate']['distributions'].get(kind)
        if not ra or not rb:continue
        paired=[(ledgers['reference'][k],r) for k,r in ledgers['candidate'].items() if k in ledgers['reference'] and r['kind']==kind]
        metrics={k:dict(reference=ra['metrics'][k],candidate=rb['metrics'][k]) for k in ra['metrics'].keys()&rb['metrics'].keys()}
        gm={k:geomean([x['metrics_ms'][k]/y['metrics_ms'][k] for x,y in paired if x['metrics_ms'].get(k,0)>0 and y['metrics_ms'].get(k,0)>0]) for k in ('client_ttft','native_prefill','outside_native')}
        cs=[r['token_cache'] for r in ledgers['candidate'].values() if r['kind']==kind]
        cache=dict(hits=sum(c['reused_tokens']>0 for c in cs),n=len(cs),reused_percent_p50=percentile([100*c['reused_tokens']/(c['reused_tokens']+c['encoded_suffix_tokens']) for c in cs],.5),encoded_suffix_tokens_p50=percentile([c['encoded_suffix_tokens'] for c in cs],.5),encoded_suffix_tokens_p95=percentile([c['encoded_suffix_tokens'] for c in cs],.95))
        result[kind]=dict(reference_n=ra['n'],candidate_n=rb['n'],matched_labels=len(paired),matched_label_geomeans_reference_over_candidate=gm,cache=cache,metrics=metrics)
    ar=json.loads((a.reference/'summary.json').read_text());br=json.loads((a.candidate/'summary.json').read_text())
    (a.out/'comparison.json').write_text(json.dumps(dict(classes=result,reference_tps=ar['aggregate_tps'],candidate_tps=br['aggregate_tps'],tps_ratio=br['aggregate_tps']/ar['aggregate_tps'],output_status_counts=quality['status_counts'],all_observed_output_prefixes_exact=quality['all_observed_output_prefixes_exact']),indent=2,sort_keys=True)+'\n')
    print(json.dumps(dict(classes={k:dict(reference_n=v['reference_n'],candidate_n=v['candidate_n'],matched_label_geomeans=v['matched_label_geomeans_reference_over_candidate'],cache=v['cache']) for k,v in result.items()},output=quality['status_counts']),indent=2))
if __name__=='__main__':main()
