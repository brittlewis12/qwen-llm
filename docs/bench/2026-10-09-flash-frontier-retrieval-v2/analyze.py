#!/usr/bin/env python3
"""Independent v2 raw-JSON scorer; never rescoring v1 or pooling artifacts."""
import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
class ObjectPairs(list): pass
def invalid_constant(value): raise ValueError('nonfinite JSON literal '+value)
def sha(data): return hashlib.sha256(data).hexdigest()

def answer(raw):
    try:
        text = bytes(raw).decode('utf-8').lstrip(' \t\r\n')
        if not text.startswith('{'): return None
        value, end = json.JSONDecoder(object_pairs_hook=ObjectPairs,
            parse_constant=invalid_constant).raw_decode(text)
        if text[end:].strip(' \t\r\n') or not isinstance(value, ObjectPairs): return None
        result = {}
        for key, item in value:
            if key in ('code', 'record'):
                if key in result or not isinstance(item, str): return None
                result[key] = item
        return result if set(result) == {'code', 'record'} else None
    except (ValueError, UnicodeError): return None

def verdict(a, b):
    if not a['parsed']: return 'INCONCLUSIVE_INVALID_FIXTURE'
    return 'PASS' if b['correct'] else 'FAIL'

def overall(cases):
    if 'FAIL' in cases: return 'FAIL'
    if 'INCONCLUSIVE_INVALID_FIXTURE' in cases: return 'INCONCLUSIVE'
    return 'PASS'

def analyze(path, f, fixture_sha):
    rows = [json.loads(line, parse_constant=invalid_constant) for line in path.read_text().splitlines()]
    def events(name): return [r for r in rows if r.get('event') == name]
    def one(name):
        found = events(name); assert len(found) == 1, (name, len(found)); return found[0]
    h = one('header'); assert h['schema'] == 'flash.frontier_retrieval.v2'
    artifact = h['details']['artifact']; assert artifact in ('ud', 'gsq')
    assert h['details']['fixture_sha256'] == fixture_sha
    assert h['details']['protocol_sha256'] == sha((HERE/'PROTOCOL.md').read_bytes())
    assert h['details']['v1_reclassified'] is False and h['details']['nll_rerun'] is False
    assert h['details']['performance_claim'] is False
    base = {'artifact':artifact, 'file':str(path), 'file_sha256':sha(path.read_bytes()),
            'fixture_sha256':fixture_sha, 'v1_reclassified':False, 'automatic_promotion':False}
    complete = one('complete')
    if not complete['execution_complete']:
        errors = events('acquisition_error')
        return {**base, 'verdict':errors[-1]['disposition'] if errors else 'INVALID_ACQUISITION',
                'partial_acquisition':True, 'error':complete['error']}
    assert not events('acquisition_error') and not events('error') and not events('target')
    assert one('fixture_verified')['all_token_arrays_match_current_native_tokenizer']
    assert one('fixture_verified')['manifest_sha256'] == fixture_sha
    assert one('artifact_revalidation')['unchanged']
    order = [('v2_before','A'),('v2_before','B'),('v2_after','B'),('v2_after','A')]
    assert [(r['id'],r['arm']) for r in events('attempt_begin')] == order
    assert [(r['id'],r['arm']) for r in events('prefill_complete')] == order
    assert [(r['id'],r['arm']) for r in events('retrieval_complete')] == order
    assert len(events('retrieval_pair')) == 2
    for r in events('prefill_complete'):
        expected = [[0,2048],[2048,2051],[2051,4096]] if r['arm']=='A' else [[0,2048],[2048,4096]]
        assert r['ranges'] == expected and r['command_count'] == len(expected)
        assert r['state']['position'] == 4096 and r['performance_claim'] is False
    cases = []
    for d in f['retrieval']:
        arms = {}
        for arm in ('A','B'):
            rr = [r for r in events('retrieval_complete') if (r['id'],r['arm']) == (d['id'],arm)]
            assert len(rr) == 1; r = rr[0]
            ids = r['generated_token_ids']
            assert len(ids) <= 64 and all(type(t) is int and 0 <= t < 248320 and t != 248046 for t in ids)
            assert r['max_generated_tokens'] == 64 and (r['stopped_on_eos'] or len(ids) == 64)
            assert r['all_finite'] and r['state']['position'] == 4096+len(ids)
            assert r['fact_token_range'] == d['fact_token_range'] and r['expected'] == d['expected']
            assert bytes(r['generated_bytes']).decode('utf-8', errors='replace') == r['generated_text']
            parsed = answer(r['generated_bytes']); correct = parsed == d['expected']
            assert parsed == r['parsed_answer'] and correct == r['correct']
            arms[arm] = {'parsed':parsed is not None, 'correct':correct, 'answer':parsed,
                         'generated_text':r['generated_text'], 'generated_bytes':r['generated_bytes'],
                         'generated_token_count':len(ids), 'stopped_on_eos':r['stopped_on_eos'],
                         'state':r['state']}
        result = verdict(arms['A'], arms['B'])
        pair = [r for r in events('retrieval_pair') if r['id'] == d['id']]
        assert len(pair) == 1 and pair[0]['artifact'] == artifact and pair[0]['verdict'] == result
        for arm in ('A','B'):
            assert pair[0][arm+'_parsed'] == arms[arm]['parsed'] and pair[0][arm+'_correct'] == arms[arm]['correct']
        cases.append({'id':d['id'], 'expected':d['expected'], **arms, 'verdict':result})
    for r in events('prefill_complete')+events('retrieval_complete'):
        s = r['state']; assert len(s['qsa_lengths']) == 12
        assert all(n == s['position'] for _,n in s['qsa_lengths'])
    result = overall([c['verdict'] for c in cases])
    summary = one('retrieval_v2_summary')
    assert summary['artifact'] == artifact and summary['verdict'] == result
    assert summary['retrieval_failed'] == any(c['verdict']=='FAIL' for c in cases)
    assert summary['retrieval_inconclusive'] == any(c['verdict']=='INCONCLUSIVE_INVALID_FIXTURE' for c in cases)
    assert summary['v1_reclassified'] is False and summary['nll_rerun'] is False and summary['automatic_promotion'] is False
    return {**base, 'verdict':result, 'cases':cases, 'scope':'two fresh raw-JSON retrieval cases; format-motivated follow-up, not independent replication'}

def self_test():
    expected = {'code':'C','record':'R'}
    for good in [b'{"code":"C","record":"R"}', b' \t\r\n{"record":"R","code":"C","extra":1}\t\n']:
        assert answer(good) == expected
    for bad in [b'FINAL_JSON: {"code":"C","record":"R"}', b'["C","R"]',
                b'{"code":"C","record":"R"} tail', b'{"code":"C","record":"R"}{}',
                b'{"code":"C","record":"R","code":"C"}', b'{"code":"C","record":"R","record":"R"}',
                b'{"code":1,"record":"R"}', b'{"code":"C"}', b'{"code":"C","record":"R","x":NaN}',
                b'{"code":"C","record":"R"}\x0b', b'\xff', b'{"code":', b'\xc2\xa0{"code":"C","record":"R"}']:
        assert answer(bad) is None, bad
    invalid = {'parsed':False,'correct':False}; wrong = {'parsed':True,'correct':False}; correct = {'parsed':True,'correct':True}
    assert verdict(invalid,correct) == 'INCONCLUSIVE_INVALID_FIXTURE'
    assert verdict(wrong,wrong) == 'FAIL' and verdict(wrong,correct) == 'PASS'
    assert verdict(correct,invalid) == 'FAIL'
    assert overall(['FAIL','INCONCLUSIVE_INVALID_FIXTURE']) == 'FAIL'
    assert overall(['PASS','INCONCLUSIVE_INVALID_FIXTURE']) == 'INCONCLUSIVE'
    assert overall(['PASS','PASS']) == 'PASS'
    print('PASS: strict raw-JSON scorer and frozen decision rules')

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('packets', nargs='*', type=Path)
    ap.add_argument('--out', type=Path); ap.add_argument('--self-test', action='store_true'); args = ap.parse_args()
    if args.self_test: self_test(); return
    assert args.packets, 'supply one new acquisition per artifact'
    data = (HERE/'fixtures.json').read_bytes(); f = json.loads(data)
    assert f['schema'] == 'flash.frontier_retrieval.fixtures.v2'
    for b in f['files']:
        contents = (HERE/b['file']).read_bytes(); assert sha(contents) == b['sha256'] and len(contents) == b['bytes']
    results = [analyze(p,f,sha(data)) for p in args.packets]
    assert len({r['artifact'] for r in results}) == len(results), 'do not select among repeats'
    text = json.dumps({'schema':'flash.frontier_retrieval.analysis.v2', 'artifact_pooling':False,
                       'v1_reclassified':False, 'results':results},indent=2,allow_nan=False)+'\n'
    if args.out:
        with args.out.open('x') as out: out.write(text)
    else: print(text,end='')
if __name__ == '__main__': main()
