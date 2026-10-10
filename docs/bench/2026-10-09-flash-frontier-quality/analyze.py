#!/usr/bin/env python3
"""Validate each artifact separately. Document bootstrap is descriptive, not gated."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import statistics

HERE=Path(__file__).resolve().parent
LIMIT=math.log(1.01)
FIXTURE_SHA='d592ccda9273cdc03b26bca284e8bb48e4aa8db404a3b140c2a51b37f009e73e'
class ObjectPairs(list): pass

def answer(raw):
    try:
        text=bytes(raw).decode('utf-8')
        tail=text.split('FINAL_JSON:',1)[1].lstrip(' \r\n\t')
        def invalid_constant(value): raise ValueError('invalid JSON constant '+value)
        value,_=json.JSONDecoder(object_pairs_hook=ObjectPairs,parse_constant=invalid_constant).raw_decode(tail)
        if not isinstance(value,ObjectPairs): return None
        result={}
        for key,v in value:
            if key in ('record','code'):
                if key in result or not isinstance(v,str): return None
                result[key]=v
        return result if set(result)=={'record','code'} else None
    except (ValueError,UnicodeError,IndexError): return None

def interval(deltas):
    assert len(deltas)==8
    rng=random.Random(20261009)
    means=sorted(statistics.mean(rng.choices(deltas,k=8)) for _ in range(20000))
    def q(p):
        x=(len(means)-1)*p; i=int(x); t=x-i
        return means[i]*(1-t)+means[min(i+1,len(means)-1)]*t
    return [q(.025),q(.975)]

def close(a,b):
    assert math.isfinite(a) and math.isfinite(b) and math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-12),(a,b)

def analyze(path,f):
    def invalid_constant(value): raise ValueError('nonfinite JSON literal '+value)
    rows=[json.loads(line,parse_constant=invalid_constant) for line in path.read_text().splitlines()]
    def events(name): return [r for r in rows if r.get('event')==name]
    def one(name):
        found=events(name); assert len(found)==1,(name,len(found)); return found[0]
    header=one('header'); assert header['schema']=='flash.frontier_quality.v1'
    artifact=header['details']['artifact']; assert artifact in ('ud','gsq')
    assert header['details']['fixture_sha256']==FIXTURE_SHA
    complete=one('complete')
    if not complete['execution_complete']:
        errors=events('acquisition_error')
        return {'artifact':artifact,'file':str(path),'verdict':errors[-1]['disposition'] if errors else 'INVALID_ACQUISITION',
                'error':complete['error'],'partial_acquisition':True}
    assert not events('acquisition_error') and not events('error')
    assert one('fixture_verified')['all_token_arrays_match_current_native_tokenizer']
    assert one('artifact_revalidation')['unchanged']
    expected_order=[]
    for group in ('natural','retrieval'):
        for i,d in enumerate(f[group]):
            expected_order.extend((d['id'],a) for a in (('A','B') if i%2==0 else ('B','A')))
    assert [(r['id'],r['arm']) for r in events('attempt_begin')]==expected_order
    prefill=events('prefill_complete'); assert len(prefill)==20
    for r,(id_,arm) in zip(prefill,expected_order):
        assert (r['id'],r['arm'])==(id_,arm)
        expected=[[0,2048],[2048,2051],[2051,4096]] if arm=='A' else [[0,2048],[2048,4096]]
        assert r['ranges']==expected and r['command_count']==len(expected) and r['state']['position']==4096
        assert r['performance_claim'] is False
    target_rows=events('target'); assert len(target_rows)==8*2*64
    natural_rows=events('natural_complete'); assert len(natural_rows)==16
    assert len(events('document_pair'))==8 and len(events('retrieval_pair'))==2
    def keyed(name,id_,arm):
        found=[r for r in events(name) if (r['id'],r['arm'])==(id_,arm)]
        assert len(found)==1,(name,id_,arm); return found[0]
    documents=[]; hit_totals={'A':0,'B':0}
    for d in f['natural']:
        arms={}; states={}
        for arm in ('A','B'):
            rr=[r for r in target_rows if (r['id'],r['arm'])==(d['id'],arm)]
            assert [r['step'] for r in rr]==list(range(64))
            for i,r in enumerate(rr):
                assert r['target_id']==d['token_ids'][4096+i]
                assert r['state_before_feed']['position']==4096+i
                assert r['logit_count']==248320 and r['all_finite']
                assert isinstance(r['top1'],int) and 0<=r['top1']<248320
                assert r['correct']==(r['top1']==r['target_id'])
                assert math.isfinite(r['nll']) and r['nll']>=0
            end=keyed('natural_complete',d['id'],arm)
            assert end['state']['position']==4160 and end['scored_tokens']==64 and not end['terminal_row_scored']
            assert end['nll']==[r['nll'] for r in rr]
            assert end['hits']==sum(r['correct'] for r in rr)
            close(end['mean_nll'],statistics.mean(end['nll']))
            arms[arm]=end; hit_totals[arm]+=end['hits']
            states[arm]=[r['state_before_feed'] for r in rr]+[end['state']]
        assert states['A']==states['B'],'teacher-forced causal metadata differs'
        documents.append({'id':d['id'],'nll_A':arms['A']['mean_nll'],'nll_B':arms['B']['mean_nll'],
                          'delta_B_minus_A':arms['B']['mean_nll']-arms['A']['mean_nll'],
                          'accuracy_A':arms['A']['hits']/64,'accuracy_B':arms['B']['hits']/64})
    known=events('retrieval_complete'); assert len(known)==4
    retrieval=[]
    for d in f['retrieval']:
        arms={}
        for arm in ('A','B'):
            r=keyed('retrieval_complete',d['id'],arm)
            ids=r['generated_token_ids']; assert len(ids)<=64 and all(0<=t<248320 and t!=248046 for t in ids)
            assert r['all_finite'] and r['state']['position']==4096+len(ids)
            assert r['stopped_on_eos'] or len(ids)==64
            assert r['fact_token_range']==d['fact_token_range'] and r['expected']==d['expected']
            parsed=answer(r['generated_bytes']); assert parsed==r['parsed_answer']
            correct=parsed==d['expected']; assert correct==r['correct']
            arms[arm]={'parsed':parsed is not None,'correct':correct}
        verdict='INCONCLUSIVE_INVALID_FIXTURE' if not arms['A']['parsed'] else ('PASS' if arms['B']['correct'] else 'FAIL')
        retrieval.append({'id':d['id'],**arms,'verdict':verdict})
    for d in documents:
        pair=[r for r in events('document_pair') if r['id']==d['id']]
        assert len(pair)==1 and pair[0]['artifact']==artifact
        close(pair[0]['delta_B_minus_A'],d['delta_B_minus_A'])
    for d in retrieval:
        pair=[r for r in events('retrieval_pair') if r['id']==d['id']]
        assert len(pair)==1 and pair[0]['artifact']==artifact and pair[0]['verdict']==d['verdict']
    deltas=[d['delta_B_minus_A'] for d in documents]; mean=statistics.mean(deltas)
    fail=mean>LIMIT or any(r['verdict']=='FAIL' for r in retrieval)
    inconclusive=any(r['verdict']=='INCONCLUSIVE_INVALID_FIXTURE' for r in retrieval)
    verdict='FAIL' if fail else ('INCONCLUSIVE' if inconclusive else 'PASS')
    summary=one('quality_summary'); assert summary['artifact']==artifact and summary['verdict']==verdict
    close(summary['pooled_delta_nll'],mean); close(summary['limit'],LIMIT)
    assert summary['scored_tokens_per_arm']==512
    assert summary['hits_A']==hit_totals['A'] and summary['hits_B']==hit_totals['B']
    return {'artifact':artifact,'file':str(path),'file_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'fixture_sha256':FIXTURE_SHA,'documents':documents,'pooled_delta_nll':mean,'limit':LIMIT,
            'paired_document_bootstrap_central95':interval(deltas),'resamples':20000,'seed':20261009,
            'uncertainty_gate':False,'document_resampling_units':8,
            'accuracy_A':hit_totals['A']/512,'accuracy_B':hit_totals['B']/512,
            'retrieval':retrieval,'verdict':verdict,'automatic_promotion':False,
            'claim':'passes/fails this frozen technical-text and controlled no-thinking retrieval screen; not broad chat quality or statistical non-inferiority'}

def self_test():
    assert answer(b'lead FINAL_JSON: {"record":"R","code":"C"} trailing')=={'record':'R','code':'C'}
    for x in [b'FINAL_JSON: []',b'FINAL_JSON: ["C","R"]',b'FINAL_JSON: {"record":"R","code":"C","x":NaN}',b'FINAL_JSON: {"record":1,"code":"C"}',b'FINAL_JSON: {"record":"R","code":"C","code":"D"}',
              b'FINAL_JSON: bad FINAL_JSON: {"record":"R","code":"C"}']:
        assert answer(x) is None
    assert interval([0.0]*8)==[0.0,0.0]
    assert interval([.001]*8)==[.001,.001]
    assert interval(list(range(8)))==interval(list(range(8)))
    print('CPU scorer and document-bootstrap checks passed')

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('packets',nargs='*',type=Path)
    ap.add_argument('--out',type=Path); ap.add_argument('--self-test',action='store_true'); args=ap.parse_args()
    if args.self_test: self_test(); return
    assert args.packets,'supply one JSONL per artifact'
    data=(HERE/'fixtures.json').read_bytes(); assert hashlib.sha256(data).hexdigest()==FIXTURE_SHA
    f=json.loads(data); results=[analyze(p,f) for p in args.packets]
    assert len({r['artifact'] for r in results})==len(results),'one acquisition per artifact; do not choose among repeats'
    result={'schema':'flash.frontier_quality.analysis.v1','artifact_pooling':False,'results':results}
    text=json.dumps(result,indent=2,allow_nan=False)+'\n'
    if args.out:
        with args.out.open('x') as out: out.write(text)
    else: print(text,end='')
if __name__=='__main__': main()
