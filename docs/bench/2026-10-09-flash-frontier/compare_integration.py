# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Compare retained whole trajectories across two builds; never execute a model.

Pass explicit PRIOR CURRENT pairs. Raw endpoint hashes, not full logits, are
retained. Shard comparisons concern recorded stat stamps, not a new GGUF hash.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct

from summarizer import summarize


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read(path):
    raw = path.read_bytes()
    return raw, [json.loads(line) for line in raw.splitlines() if line.strip()]


def compare(prior, current):
    payloads, rows = zip(read(prior), read(current))
    # zip above separates the two byte payloads from their parsed event lists.
    audits = [summarize(path) for path in (prior, current)]
    problems = [f'{path.name}: {p}' for path, audit in zip((prior,current),audits)
                for p in audit['problems']]

    def check(ok, message):
        if not ok:
            problems.append(message)
        return ok

    def events(index, name):
        return [r for r in rows[index] if r['event'] == name]

    bindings = {}
    for event in ('artifact','tokenizer_binding','ssh_source','loaded','moe_cohorts','device'):
        a,b = events(0,event),events(1,event)
        bindings[event] = dict(equal=check(a==b and bool(b), f'{event} binding differs'),
                               prior=a,current=b)
    prompts = [{r['id']:r for r in events(i,'prompt')} for i in range(2)]
    check(set(prompts[0])==set(prompts[1])=={'prose','ssh_repeated'}, 'prompt set differs')
    prompt_results = []
    for corpus,b in sorted(prompts[1].items()):
        a = prompts[0].get(corpus,{})
        verified = []
        for index,prompt in enumerate((a,b)):
            tokens = prompt['used_token_ids']
            packed = b''.join(struct.pack('<I', t) for t in tokens)
            ok = (len(tokens)==prompt['used_extent']==4100
                  and sha(packed)==prompt['used_token_ids_sha256_u32_le']
                  and sha(packed[:2048*4])==prompt['prefix2048_sha256']
                  and sha(packed[:4096*4])==prompt['prompt4096_sha256']
                  and sha(prompt['full_text'].encode())==prompt['full_text_sha256'])
            verified.append(check(ok, f'{corpus} prompt hash binding failed in input {index}'))
        prompt_results.append(dict(corpus=corpus,all_prompt_fields_equal=check(a==b,f'{corpus} prompt differs'),
            token_ids_equal=a.get('used_token_ids')==b['used_token_ids'],tokens=len(b['used_token_ids']),
            hashes_recomputed=verified,full_text_sha256=b['full_text_sha256'],
            ids_sha256_u32_le=b['used_token_ids_sha256_u32_le']))
    endpoint_sets = [{r['label']:r for r in events(i,'endpoint') if '/whole4096/' in r['label']}
                     for i in range(2)]
    check(set(endpoint_sets[0])==set(endpoint_sets[1]), 'whole endpoint coverage differs')
    check(len(endpoint_sets[1])==len(events(1,'endpoint'))==44, 'expected 44 current whole endpoints')
    endpoints = []
    for label,b in sorted(endpoint_sets[1].items()):
        a = endpoint_sets[0].get(label,{})
        sa,sb = a.get('state',{}),b['state']
        ta,tb = {r['index']:r for r in sa.get('tensors',[])},{r['index']:r for r in sb['tensors']}
        indices = sorted(set(ta)|set(tb))
        changed = [i for i in indices if ta.get(i)!=tb.get(i)]
        result = dict(label=label,whole_record_equal=a==b,
                      logits_hash_equal=a.get('logits_sha256_f32_le')==b['logits_sha256_f32_le'],
                      logit_count_equal=a.get('logit_count')==b['logit_count'],
                      finite=a.get('nonfinite_logits')==b['nonfinite_logits']==0,
                      hyper_hash_equal=sa.get('hyper_sha256_f32_le')==sb['hyper_sha256_f32_le'],
                      causal_metadata_equal=all(sa.get(k)==sb[k] for k in ('position','qsa_lengths','ple_prior_tokens')),
                      state_record_equal=sa==sb,persistent_tensors_checked=len(indices),
                      changed_persistent_indices=changed)
        check(result['whole_record_equal'] and result['finite'] and not changed,
              f'endpoint mismatch: {label}')
        endpoints.append(result)
    continued = [{(r['label'],r['step']):r for r in events(i,'continuation')
                  if '/whole4096/' in r['label']} for i in range(2)]
    check(continued[0]==continued[1] and len(continued[1])==32,
          'continued teacher-token inputs/metadata differ')
    headers = [events(i,'header')[0] for i in range(2)]
    old_source,new_source = [h['source_binding'] for h in headers]
    unchanged_execution = ('runtime_rs','session_rs','checkpoint_rs','qsa_rs','moe_rs',
                           'gdn_rs','dispatch_rs','profile_rs','metallib')
    check(all(old_source[k]==new_source[k] for k in unchanged_execution),
          'runtime/kernel/metallib source binding changed')
    check(headers[1]['details']['candidate_mode']=='production'
          and headers[1]['details']['B_schedule_override'] is None
          and headers[1]['details']['A_schedule_override'] is False
          and headers[1]['details']['stage']=='whole', 'not a whole actual-default confirmation')
    build = headers[1]['details']['metallib_build']
    check(build['language']=='metal3.2' and build['product_deployment_target']=='15.0'
          and build['research_deployment_target']=='15.0','unexpected compiled target metadata')
    for i in (0,1):
        check(events(i,'artifact_revalidation')[0]['unchanged'], f'artifact revalidation input {i}')
    return dict(prior=dict(path=str(prior),bytes=len(payloads[0]),sha256=sha(payloads[0])),
                current=dict(path=str(current),bytes=len(payloads[1]),sha256=sha(payloads[1])),
                problems=problems,model_bindings=bindings,prompts=prompt_results,
                whole_endpoint_records=len(endpoints),matching_endpoint_records=sum(r['whole_record_equal'] for r in endpoints),
                persistent_tensor_records_checked=sum(r['persistent_tensors_checked'] for r in endpoints),
                endpoints=endpoints,continuation_input_records_equal=continued[0]==continued[1],
                continuation_input_records=len(continued[1]),
                source_binding_prior=old_source,source_binding_current=new_source,
                changed_source_binding_fields=[k for k in sorted(set(old_source)|set(new_source)) if old_source.get(k)!=new_source.get(k)],
                metallib_build=build,executable_prior=events(0,'executable_binding'),
                executable_current=events(1,'executable_binding'),
                current_timing=audits[1]['timing'],
                limits='Whole natural prose/SSH trajectories only; equality of retained digests, not new full-logit readbacks. No new GGUF content hashing, quality acquisition, OS compatibility test or performance threshold.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('packets',type=Path,nargs='+',help='Explicit PRIOR CURRENT pairs')
    args = parser.parse_args()
    if len(args.packets)%2:
        parser.error('require complete PRIOR CURRENT pairs')
    reports = [compare(a,b) for a,b in zip(args.packets[::2],args.packets[1::2])]
    print(json.dumps(reports,indent=2,allow_nan=False))
    return any(r['problems'] for r in reports)


if __name__=='__main__':
    raise SystemExit(main())
