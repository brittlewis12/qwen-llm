#!/usr/bin/env python3
import argparse,collections,hashlib,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('left',type=Path);p.add_argument('right',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
def load(folder):
    raw={r['label']:r for r in map(json.loads,(folder/'serve/raw-calls.jsonl').read_text().splitlines())};events=collections.defaultdict(list);traces={}
    for e in json.loads((folder/'serve/native-lines.json').read_text()):
        for prefix,key in [('serve ttft timing: ','trace'),('serve round timing: ','round'),('serve input witness: ','input'),('serve cut witness: ','cut'),('serve logits witness: ','logits')]:
            if prefix in e['line']:
                d=json.JSONDecoder().raw_decode(e['line'].split(prefix,1)[1])[0];events[key].append(d)
                if key=='trace':traces[d['response_id']]=d
                break
    mapping={traces[r['response_id']]['request_id']:label for label,r in raw.items() if r.get('response_id') in traces}
    inp={d['request_id']:d for d in events['input']};cuts={d['request_id']:d for d in events['cut']};logits={d['request_id']:d for d in events['logits']}
    bylabel={}
    for label,r in raw.items():
        t=traces.get(r.get('response_id'));rid=t['request_id'] if t else None
        rounds=[d for d in events['round'] if d['kind']=='decode' and rid in d['request_ids']]
        sig=[tuple(mapping.get(i,'unknown:'+str(i)) for i in d['request_ids']) for d in rounds]
        bylabel[label]=dict(raw=r,trace=t,input=inp.get(rid),cut=cuts.get(rid),logits=logits.get(rid),rounds=rounds,signature=sig,width_counts=dict(collections.Counter(map(len,sig))))
    return bylabel,dict(trace_records=len(traces),round_records=len(events['round']),input_records=len(events['input']),cut_records=len(events['cut']),logits_records=len(events['logits']))
def channels(r):
    out=collections.defaultdict(str)
    for _,k,t in r['events']:out[k]+=t
    return dict(out)
def fields(v,keys):return {k:v.get(k) for k in keys} if v else None
left,lc=load(a.left);right,rc=load(a.right);rows=[]
for label in sorted(left.keys()&right.keys()):
    x,y=left[label],right[label];u,v=channels(x['raw']),channels(y['raw']);complete=x['raw']['completed'] and y['raw']['completed'];observed=bool(u and v)
    exact=u==v if complete else all(u.get(k,'').startswith(v.get(k,'')) or v.get(k,'').startswith(u.get(k,'')) for k in u.keys()|v.keys())
    status='unobserved' if not observed else ('complete_exact' if exact else 'complete_different') if complete else ('partial_prefix_equal' if exact else 'partial_prefix_different')
    input_equal=fields(x['input'],('prompt_sha256','token_sha256','token_count','shared_prefix_byte','generation_header_byte'))==fields(y['input'],('prompt_sha256','token_sha256','token_count','shared_prefix_byte','generation_header_byte')) if x['input'] and y['input'] else None
    cut_equal=fields(x['cut'],('prompt_tokens','shared_cut','generation_cut','checkpoint_cuts'))==fields(y['cut'],('prompt_tokens','shared_cut','generation_cut','checkpoint_cuts')) if x['cut'] and y['cut'] else None
    sigx,sigy=x['signature'],y['signature'];sig_equal=sigx==sigy
    first_round_diff=next((i for i,(xx,yy) in enumerate(zip(sigx,sigy)) if xx!=yy),min(len(sigx),len(sigy))) if not sig_equal else None
    diffs={}
    if observed and not exact:
        for k in u.keys()|v.keys():
            bx,by=u.get(k,'').encode(),v.get(k,'').encode()
            if bx!=by:diffs[k]=next((i for i,(xx,yy) in enumerate(zip(bx,by)) if xx!=yy),min(len(bx),len(by)))
    prompt_equal=x['input']['prompt_sha256']==y['input']['prompt_sha256'] if x['input'] and y['input'] else None
    tokens_equal=x['input']['token_sha256']==y['input']['token_sha256'] if x['input'] and y['input'] else None
    r=dict(prompt_equal=prompt_equal,tokens_equal=tokens_equal,label=label,status=status,observed=observed,complete_pair=complete,output_exact=exact if observed else None,input_equal=input_equal,cut_equal=cut_equal,
        initial_logits_equal=(x['logits']['logits_sha256']==y['logits']['logits_sha256']) if x['logits'] and y['logits'] else None,
        first_difference_byte=diffs,decode_cohorts_equal=sig_equal,first_decode_cohort_difference=first_round_diff,
        left_width_counts=x['width_counts'],right_width_counts=y['width_counts'],left_decode_rounds=len(sigx),right_decode_rounds=len(sigy),
        left_generated=x['raw']['generated'],right_generated=y['raw']['generated'])
    if first_round_diff is not None:
        i=first_round_diff;r['cohort_difference_example']=dict(left=x['rounds'][i] if i<len(x['rounds']) else None,right=y['rounds'][i] if i<len(y['rounds']) else None,left_labels=sigx[i] if i<len(sigx) else None,right_labels=sigy[i] if i<len(sigy) else None)
    if diffs:r['left_input']=x['input'];r['right_input']=y['input'];r['left_cut']=x['cut'];r['right_cut']=y['cut']
    rows.append(r)
observed=[r for r in rows if r['observed']];bad=[r for r in observed if not r['output_exact']];complete=[r for r in rows if r['complete_pair'] and r['observed']]
def chronology(r):
    label=r['label'];return (int(label[1]),-1 if label.endswith('-warm') else int(label.split('-t')[1]))
first={}
for r in sorted(bad,key=chronology):first.setdefault(r['label'].split('-')[0],r['label'])
result=dict(left=str(a.left),right=str(a.right),records=dict(left=lc,right=rc),common_calls=len(rows),observed_common=len(observed),mismatches=len(bad),divergence_rate=len(bad)/len(observed) if observed else None,completed_common=len(complete),completed_mismatches=sum(not r['output_exact'] for r in complete),status_counts=dict(collections.Counter(r['status'] for r in rows)),first_divergence_per_agent=first,
    same_prompt_token_mismatch_labels=[r['label'] for r in rows if r['prompt_equal'] is True and r['tokens_equal'] is False],same_prompt_cut_mismatch_labels=[r['label'] for r in rows if r['prompt_equal'] is True and r['cut_equal'] is False],same_input_and_cut_mismatches=sum(r['input_equal'] is True and r['cut_equal'] is True for r in bad),rows=rows)
a.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))
