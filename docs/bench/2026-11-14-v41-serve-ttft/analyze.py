#!/usr/bin/env python3
"""Offline TTFT accounting; original files are identified by input-manifest.json.

No model or GPU access. Times on native lines are harness receipt times, so
interval reconstruction is approximate; overlap is exposure, not causal wait.
Quantiles use linear interpolation, as serve_realistic.py does.
"""
import argparse
import bisect
import collections
import hashlib
import json
import math
from pathlib import Path
import re
import statistics


def kv(s):
    return dict(re.findall(r'(\w+)=([^\s]+)', s))


def q(v, p):
    v = sorted(x for x in v if x is not None and math.isfinite(x))
    if not v:
        return None
    i = (len(v) - 1) * p
    lo = int(i)
    return v[lo] + (v[min(lo + 1, len(v) - 1)] - v[lo]) * (i - lo)


def dist(v):
    v = [x for x in v if x is not None and math.isfinite(x)]
    return dict(n=len(v), p50=q(v, .5), p95=q(v, .95), p99=q(v, .99), mean=statistics.mean(v) if v else None, maximum=max(v) if v else None)


def io_total(ph, kind):
    return ((ph.get(kind) or {}).get('process_disk_read_bytes') or {}).get('total')


def classify(n, root=False):
    if root:
        return 'root'
    if n is None:
        return 'unknown'
    if 0 < n < 512:
        return 'short'
    if 512 <= n <= 2000:
        return '1k'
    return 'other'


def union_ms(intervals):
    total = 0.; hi = -math.inf
    for a, b in sorted(intervals):
        if b > max(a, hi):
            total += b - max(a, hi)
        hi = max(hi, b)
    return total * 1000


def overlap(a, b, c, d):
    return max(0., min(b, d) - max(a, c)) * 1000


def parse_arm(folder, campaign, tag):
    calls = [json.loads(x) for x in (folder / 'calls.jsonl').read_text().splitlines()]
    lines = json.loads((folder / 'native-lines.json').read_text())
    launch = json.loads((folder / 'launch.json').read_text())
    summary = json.loads((folder / 'summary.json').read_text())
    profiles, completed, finals, steps = [], [], [], []
    last_phase = None
    for i, e in enumerate(lines):
        s, t = e['line'], e['time']
        if s.startswith('serve native prefill timing: '):
            p = json.loads(s.split('timing: ', 1)[1])
            if not p.get('success'):
                continue
            report = p['profile']; ops = report.get('operations', [])
            gpu = [o['gpu_ms'] for o in ops if o['operation'] == 'prefill_chunk_rows' and o.get('gpu_ms') is not None]
            p.update(id=len(profiles), end=t, start=t-p['wall_ms']/1000, line=i,
                     gpu_ms=sum(gpu) if gpu else None,
                     engram_prepare_ms=sum(o['wall_ms'] for o in ops if o['operation'] == 'prefill_engram_prepare'),
                     engram_read_ms=report.get('engram_read_wall_ms'), engram_advise_ms=report.get('engram_advise_wall_ms'),
                     disk_bytes=report.get('process_disk_read_bytes'), call_ids=[])
            p.pop('profile'); profiles.append(p)
        elif s.startswith('serve prefill: '):
            k = kv(s)
            if k.get('complete') == 'true':
                k.update(time=t, line=i); completed.append(k)
        elif s.startswith('serve phases: '):
            last_phase = kv(s); last_phase.update(time=t, line=i)
            finals.append(last_phase)
        elif s.startswith('serve batch: ') and last_phase is not None:
            last_phase['batch'] = json.loads(s[s.index('{'):])
        elif s.startswith('serve forward io: '):
            k = kv(s)
            if k.get('scope', '').startswith('whole_process_native_step') and k.get('success') == 'true':
                steps.append(dict(time=t, position=int(k['position']), rows=int(k['rows']), scope=k['scope'], disk_bytes=int(k['process_disk_read_bytes'])))
    roots = []
    log = folder / 'harness.log'
    if log.exists():
        for line in log.read_text().splitlines():
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get('event') == 'turn' and r.get('label', '').endswith('-warm'):
                roots.append(r)
    root_finals = [r for r in finals if int(r.get('prefill_tokens', 0)) >= 48000]
    root_rows = []
    for i, r in enumerate(root_finals):
        total = roots[i].get('ttft_s')*1000 if i < len(roots) and roots[i].get('ttft_s') is not None else None
        pre = float(r['prefill_ms']); batch = r.get('batch', {})
        root_rows.append(dict(campaign=campaign, arm=tag, label='a%d-warm' % i, kind='root', sampling=launch['sampling']['mode'],
                              client_ms=total, native_ms=pre, queue_ms=batch.get('queue_ms'), native_fraction=pre/total if total else None,
                              outside_native_ms=total-pre if total is not None else None,
                              transitions=int(r.get('transitions', 0)), transition_ms=float(r.get('transition_ms', 0)),
                              client_absolute_timestamps_available=False, temporal_join=False))
    out = []
    for r in calls:
        ph = r.get('phases') or {}; n = r.get('new'); batch = ph.get('batch') or {}
        total = (r['first']-r['sent'])*1000 if r.get('first') is not None else None
        native = float(ph['prefill_ms']) if ph.get('prefill_ms') is not None else None
        x = dict(campaign=campaign, arm=tag, id=campaign+'/'+tag+'/'+r['label'], label=r['label'], agent=r['agent'], kind=classify(n),
                 sampling=launch['sampling']['mode'], sent=r['sent'], first=r.get('first'), end=r['end'], completed=r['completed'],
                 offered=r['offered_in_window'], carry_in=r['carry_in'], observed_in_window=r.get('ttft_s') is not None,
                 censored=r.get('ttft_censored_s'), new=n, prompt=r.get('prompt'), reused=r.get('reused'),
                 client_ms=total, native_ms=native, queue_ms=batch.get('queue_ms'), restore_ms=ph.get('restore_ms'),
                 outside_native_ms=total-native if total is not None and native is not None else None,
                 profile_ids=[], temporal_join=False, prefill_calls=ph.get('prefill_calls'), layer_rounds=ph.get('layer_rounds'),
                 transitions=ph.get('transitions'), transition_ms=ph.get('transition_ms'),
                 prefill_io_bytes=io_total(ph, 'process_prefill_io'),
                 token_io_bytes=io_total(ph, 'process_token_io'),
                 batch_width_mean=batch.get('width_mean'), max_active=batch.get('max_active'), prefill_fair=batch.get('prefill_fair'),
                 decode_percent=batch.get('decode_percent'), parked_ms=batch.get('parked_ms'), terminal_token_limit=r.get('target'), requested_append_tokens=(r.get('spec') or {}).get('append_tokens'),
                 first_delta_kind=r['events'][0][1] if r.get('events') else None)
        if ph and n is not None and r.get('reused') is not None and r.get('first') is not None:
            cands = [z for z in completed if int(z['reused_tokens']) == r['reused'] and int(z['prefill_tokens']) == n and r['sent']-.05 <= z['time'] <= r['first']+.05]
            x['completion_candidates'] = len(cands)
            if len(cands) == 1:
                z = cands[0]; x['prefill_complete_time'] = z['time']
                ranges = ph.get('prefill_chunk_ranges') or []
                chosen = [p for p in profiles if p['start'] >= r['sent']-.05 and p['end'] <= z['time']+.01 and
                          p['position'] >= r['reused'] and p['position']+p['rows'] <= r['prompt'] and
                          [p['position'], p['position']+p['rows']] in ranges]
                x['profile_ids'] = [p['id'] for p in chosen]
                for p in chosen:
                    p['call_ids'].append(x['id'])
                if chosen and len(chosen) == ph.get('prefill_calls') and not ph.get('ced'):
                    s0 = min(p['start'] for p in chosen); e0 = max(p['end'] for p in chosen)
                    pm = sum(p['wall_ms'] for p in chosen)
                    x.update(temporal_join=True, first_native_start=s0, last_native_end=e0,
                             before_native_ms=(s0-r['sent'])*1000, profile_native_ms=pm,
                             between_native_ms=sum(max(0,p2['start']-p1['end'])*1000 for p1,p2 in zip(chosen,chosen[1:])),
                             finish_after_native_ms=(z['time']-e0)*1000,
                             first_visible_after_complete_ms=(r['first']-z['time'])*1000,
                             after_native_ms=(r['first']-e0)*1000,
                             profile_vs_phase_ms=native-pm,
                             gpu_ms=sum(p['gpu_ms'] or 0 for p in chosen),
                             engram_prepare_ms=sum(p['engram_prepare_ms'] for p in chosen),
                             engram_read_ms=sum(p['engram_read_ms'] or 0 for p in chosen),
                             engram_advise_ms=sum(p['engram_advise_ms'] or 0 for p in chosen))
                    identity=x['before_native_ms']+pm+((e0-s0)*1000-pm)+x['after_native_ms']
                    assert abs(identity-total) < .001, (x['id'], identity, total)
        out.append(x)
    byid = {r['id']:r for r in out}
    step_times = [s['time'] for s in steps]
    rounds=[]
    for st in steps:
        if not rounds or st['time']-rounds[-1][-1]['time']>.002:rounds.append([st])
        else:rounds[-1].append(st)
    round_times=[g[-1]['time'] for g in rounds]
    step_scopes = collections.Counter(s['scope'] for s in steps)
    for x in out:
        if not x['temporal_join']:
            continue
        pre=(x['sent'],x['first_native_start']); post=(x['last_native_end'],x['first'])
        peers=[]; pre_int=[]; post_int=[]
        for p in profiles:
            if p['id'] in x['profile_ids']:
                continue
            a=overlap(*pre,p['start'],p['end']);b=overlap(*post,p['start'],p['end'])
            if a+b>0:
                call_ids=p['call_ids']; peers.append(dict(profile_id=p['id'],rows=p['rows'],wall_ms=p['wall_ms'],gpu_ms=p['gpu_ms'],
                    engram_prepare_ms=p['engram_prepare_ms'],disk_bytes=p['disk_bytes'],call_ids=call_ids,pre_exposure_ms=a,post_exposure_ms=b))
                if a:pre_int.append((max(pre[0],p['start']),min(pre[1],p['end'])))
                if b:post_int.append((max(post[0],p['start']),min(post[1],p['end'])))
        x.update(peer_prefill=peers,peer_prefill_before_ms=union_ms(pre_int),peer_prefill_after_ms=union_ms(post_int))
        x['peer_prefill_exposure_ms']=x['peer_prefill_before_ms']+x['peer_prefill_after_ms']
        x['outside_native_not_peer_ms']=x['outside_native_ms']-x['peer_prefill_exposure_ms']
        x['decode_completion_lines_before']=bisect.bisect_left(step_times,pre[1])-bisect.bisect_left(step_times,pre[0])
        x['decode_round_markers_before']=bisect.bisect_left(round_times,pre[1])-bisect.bisect_left(round_times,pre[0])
        x['before_not_peer_ms']=x['before_native_ms']-x['peer_prefill_before_ms']
        x['decode_completion_lines_after']=bisect.bisect_left(step_times,post[1])-bisect.bisect_left(step_times,post[0])
        # Attribute a completion only when time AND position make exactly one
        # eligible request; ambiguous member/round IDs are left unassigned.
        own_before_visible=[]; ambiguous=0
        for s in steps:
            if not (x['last_native_end'] <= s['time'] < x['first']):
                continue
            candidates=[y for y in out if y['temporal_join'] and y.get('prompt') is not None and y['last_native_end'] <= s['time'] <= y['end'] and
                        y['prompt'] <= s['position'] < y['prompt']+(y.get('transitions') or 0)]
            if x in candidates:
                if len(candidates)==1:own_before_visible.append(s['position'])
                else:ambiguous+=1
        x['unique_own_forward_completions_before_visible']=len(own_before_visible)
        x['ambiguous_forward_completions_before_visible']=ambiguous
        x['native_fraction']=x['native_ms']/x['client_ms']
        x['io_price_ms']=x['engram_read_ms']+x['engram_advise_ms']
        x['external_gpu_headroom_ms']=max(0,x['native_ms']-x['gpu_ms'])
        x['finish_excluding_profile_overhead_ms']=(x['finish_after_native_ms']-x['profile_vs_phase_ms']) if len(x['profile_ids'])==1 else None
        x['decode_completion_lines_between']=sum(1 for st in steps if any(p1['end'] <= st['time'] < p2['start'] for p1,p2 in zip([profiles[i] for i in x['profile_ids']],[profiles[i] for i in x['profile_ids']][1:])))
        x['round_ms_mean']=x['transition_ms']/x['transitions'] if x.get('transitions') else None
        # Optimistic opportunity prices, NOT runnable scheduler simulations.
        x['cf_remove_peer_exposure_ms']=x['client_ms']-x['peer_prefill_exposure_ms']
        x['cf_remove_post_peer_ms']=x['client_ms']-x['peer_prefill_after_ms']
        x['cf_remove_all_before_ms']=x['client_ms']-max(0,x['before_native_ms'])
        for cap in (200,400,500):x['cf_peer_cap_'+str(cap)+'_ms']=x['client_ms']-max(0,x['peer_prefill_exposure_ms']-cap)
    meta=dict(campaign=campaign,arm=tag,sampling=launch['sampling'],commit=launch.get('commit'),harness_commit=launch.get('harness_commit'),
              protocol=launch.get('protocol'),env=launch.get('env_overrides'),decode_round_cluster_widths=dict(collections.Counter(len(g) for g in rounds)),
              window_seconds=summary['window_seconds'],seed=summary['seed'],
              plan_sha256=hashlib.sha256((folder/'plan.json').read_bytes()).hexdigest(),
              completed_prefill_lines=len(completed),root_count=len(root_rows),root_client_observed=sum(r['client_ms'] is not None for r in root_rows),
              profile_count=len(profiles),decode_io_lines=len(steps),step_scopes=dict(step_scopes),
              calls=len(calls),primary_completed=sum(r['completed'] and r['offered'] and r['observed_in_window'] for r in out),
              censored=sum(r['censored'] is not None for r in out),partial=sum(not r['completed'] for r in out))
    return out+root_rows,profiles,meta


METRICS=('client_ms','queue_ms','restore_ms','native_ms','outside_native_ms','before_native_ms','profile_native_ms','between_native_ms',
         'finish_after_native_ms','first_visible_after_complete_ms','after_native_ms','profile_vs_phase_ms','gpu_ms','engram_prepare_ms',
         'engram_read_ms','engram_advise_ms','prefill_io_bytes','token_io_bytes','peer_prefill_before_ms','peer_prefill_after_ms',
         'peer_prefill_exposure_ms','outside_native_not_peer_ms','unique_own_forward_completions_before_visible',
         'ambiguous_forward_completions_before_visible','decode_completion_lines_before','decode_completion_lines_after','round_ms_mean',
         'cf_remove_peer_exposure_ms','cf_remove_post_peer_ms','cf_remove_all_before_ms','native_fraction','io_price_ms','external_gpu_headroom_ms',
         'finish_excluding_profile_overhead_ms','decode_round_markers_before','before_not_peer_ms','decode_completion_lines_between','cf_peer_cap_200_ms','cf_peer_cap_400_ms','cf_peer_cap_500_ms')


def summarize(rows):
    return dict(n=len(rows),temporal_join=sum(r.get('temporal_join',False) for r in rows),
                metrics={k:dist(r.get(k) for r in rows) for k in METRICS},
                any_peer_exposure=sum(r.get('peer_prefill_exposure_ms',0)>1 for r in rows),
                any_post_peer_exposure=sum(r.get('peer_prefill_after_ms',0)>1 for r in rows),
                negative_temporal_rows=[r['id'] for r in rows if r.get('temporal_join') and min(r['before_native_ms'],r['after_native_ms']) < -1])


def main():
    ap=argparse.ArgumentParser();ap.add_argument('input',type=Path);ap.add_argument('output',type=Path);ap.add_argument('--manifest',type=Path);a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    input_files=0
    if a.manifest:
        for arm in json.loads(a.manifest.read_text()):
            for f in arm['files']:
                path=a.input/arm['campaign']/arm['tag']/f['name']
                assert path.stat().st_size==f['bytes'] and hashlib.sha256(path.read_bytes()).hexdigest()==f['sha256'],path
                input_files+=1
    rows=[];profiles=[];arms=[]
    for family in sorted(p for p in a.input.iterdir() if p.is_dir()):
        for arm in sorted(p for p in family.iterdir() if p.is_dir()):
            rr,pp,mm=parse_arm(arm,family.name,arm.name);rows.extend(rr);profiles.append(dict(campaign=family.name,arm=arm.name,profiles=pp));arms.append(mm)
    primary=[r for r in rows if r['kind']=='root' or (r['completed'] and r['offered'] and r['observed_in_window'])]
    grouped={}
    for campaign in sorted(set(r['campaign'] for r in rows)):
        grouped[campaign]={k:summarize([r for r in primary if r['campaign']==campaign and r['kind']==k]) for k in ('short','1k','other','root')}
    arm_groups={m['campaign']+'/'+m['arm']:{k:summarize([r for r in primary if r['campaign']==m['campaign'] and r['arm']==m['arm'] and r['kind']==k]) for k in ('short','1k','root')} for m in arms}
    sensitivity={}
    for campaign in grouped:
        scope=[r for r in primary if r['campaign']==campaign and r['kind']=='short']
        sensitivity[campaign]={name:summarize([r for r in scope if predicate(r)]) for name,predicate in (('new_128_511',lambda r:r['new']>=128),('new_150_511',lambda r:r['new']>=150),('prior_gate_population',lambda r:r['new']>=128 and 150<=(r.get('requested_append_tokens') or 0)<=2000))}
    result=dict(sensitivity=sensitivity,quantiles='linear interpolation over calls; percentiles of components do not add',classes='short: 1..511 actual new rows; 1k: 512..2000; roots: cold >=48000; other retained separately',
                primary='completed, offered in window, first visible in window; all roots separate; carry-in/partial/censored preserved in ledger',
                timestamp_limit='native line timestamps are harness receipt wall times, not server operation timestamps; overlap is exposure not causal attribution',
                arms=arms,campaigns=grouped,by_arm=arm_groups,total_rows=len(rows),primary_rows=len(primary))
    (a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    with (a.output/'calls.jsonl').open('w') as f:
        for r in rows:f.write(json.dumps(r)+'\n')
    (a.output/'native-profiles.json').write_text(json.dumps(profiles,indent=2)+'\n')
    opportunities={}
    for campaign in ('attention','radix','engram'):
        if campaign not in grouped:continue
        opportunities[campaign]={}
        for kind in ('short','1k'):
            mm=grouped[campaign][kind]['metrics'];base=mm['client_ms'];costs={}
            for key in ('cf_remove_peer_exposure_ms','cf_peer_cap_200_ms','cf_peer_cap_400_ms','cf_peer_cap_500_ms'):
                z=mm[key];costs[key]=dict(p50_client_ms=z['p50'],p95_client_ms=z['p95'],p50_reduction_ms=base['p50']-z['p50'],p95_reduction_ms=base['p95']-z['p95'])
            opportunities[campaign][kind]=costs
    (a.output/'opportunity.json').write_text(json.dumps(dict(interpretation='Optimistic exposure prices, not policy simulations; deferred peers and chunk inefficiency are unpriced.',campaigns=opportunities),indent=2)+'\n')
    selected_fields=('id','label','agent','new','sent','first_native_start','last_native_end','client_ms','native_ms','queue_ms','before_native_ms','after_native_ms','peer_prefill','prefill_io_bytes')
    swaps=[{key:r.get(key) for key in selected_fields} for r in rows if r['campaign']=='radix' and r['arm'] in ('pair1-a','pair1-b') and r['label'] in ('a0-t11','a1-t6','a0-t18','a1-t8')]
    (a.output/'order-swaps.json').write_text(json.dumps(swaps,indent=2)+'\n')
    cohort=[r for r in primary if r['campaign']!='engram_diag' and r['kind'] in ('short','1k')]
    assert all(r['temporal_join'] and r['completion_candidates']==1 for r in cohort)
    assert all(-.01 <= r['profile_vs_phase_ms'] < 10 for r in cohort)
    assert all(r['peer_prefill_after_ms']==0 and r['unique_own_forward_completions_before_visible']==0 for r in cohort)
    validation=dict(input_files_verified=input_files,arms=len(arms),primary_short_1k=len(cohort),uniquely_joined=len(cohort),max_phase_profile_difference_ms=max(r['profile_vs_phase_ms'] for r in cohort),negative_temporal_rows=[r['id'] for r in cohort if min(r['before_native_ms'],r['after_native_ms']) < -1],root_client_greedy=sum(r['kind']=='root' and r['campaign'] in ('attention','radix') and r['client_ms'] is not None for r in primary),root_client_engram=sum(r['kind']=='root' and r['campaign']=='engram' and r['client_ms'] is not None for r in primary),post_native_peer_exposures=0,additional_own_forwards_before_first_visible=0)
    (a.output/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
    for campaign,kinds in grouped.items():
        for k,g in kinds.items():
            m=g['metrics'];print(campaign,k,'n',g['n'],'joined',g['temporal_join'],{key:(round(m[key]['p50'],3),round(m[key]['p95'],3)) for key in ('client_ms','native_ms','outside_native_ms','before_native_ms','after_native_ms','peer_prefill_exposure_ms') if m[key]['n']})


if __name__=='__main__':main()
