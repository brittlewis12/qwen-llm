#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.13"
# dependencies = ["ortools==9.15.6755"]
# ///
"""Price captured causal DAGs. Inputs are diagnostic causal_graph nodes JSONL.

Durations are isolated stage-boundary GPU samples. Partitions retain current
causal scopes; a separate global span-DAG bound removes artificial scope joins.
This optimistic price does not model bandwidth contention or host submission
overhead and is not a measured model speedup.
"""
import argparse
import json
import statistics
from pathlib import Path


def conflict(a, b):
    return (a['buffer'] == b['buffer'] and (a['write'] or b['write'])
            and a['lo'] < b['hi'] and b['lo'] < a['hi'])


def validate_plan(plan):
    nodes = plan['nodes']
    assert sorted(i for w in plan['waves'] for i in w) == list(range(len(nodes)))
    wave_of = {i: w for w, indices in enumerate(plan['waves']) for i in indices}
    for i, node in enumerate(nodes):
        assert node['id'] == i
        expected = [j for j in range(i) if any(conflict(a, b)
                    for a in nodes[j]['spans'] for b in node['spans'])]
        assert node['predecessors'] == expected, (plan['scope'], i, expected)
        assert all(wave_of[j] < wave_of[i] for j in expected)


def transitive_reduction(nodes):
    ancestors = []
    reduced = []
    for i, node in enumerate(nodes):
        pred = node['predecessors']
        redundant = 0
        mask = 0
        for p in pred:
            redundant |= ancestors[p]
            mask |= ancestors[p] | (1 << p)
        reduced.append([p for p in pred if not (redundant >> p) & 1])
        ancestors.append(mask)
    return reduced, ancestors


def critical_path(nodes, weights, assignment=None, sync_ns=0):
    finish = []
    for i, node in enumerate(nodes):
        q = assignment[i] if assignment is not None else 0
        start = max((finish[p] + (sync_ns if assignment is not None and assignment[p] != q else 0)
                     for p in node['predecessors']), default=sync_ns if q else 0)
        finish.append(start + weights[i])
    successors = {p for node in nodes for p in node['predecessors']}
    return max((finish[i] + (sync_ns if assignment is not None and assignment[i] else 0)
                for i in range(len(nodes)) if i not in successors), default=0)


def global_span_critical(plans, times, assignments=None, sync_ns=0):
    """Reconstruct cross-scope edges from spans; scope boundaries are not edges.

    Each call is one forward/token round, so future teacher-forced tokens do
    not create an artificial cross-token pipeline. A covering write subsumes
    earlier accesses since its own finish already depends on all of them.
    """
    writes, reads = {}, {}
    finish = []
    for plan in plans:
        scope = plan['scope']
        for i,node in enumerate(plan['nodes']):
            q = assignments[scope][i] if assignments is not None else 0
            start = sync_ns if q else 0
            for a in node['spans']:
                previous = list(writes.get(a['buffer'], {}).items())
                if a['write']: previous += list(reads.get(a['buffer'], {}).items())
                for bounds,(end,source_q) in previous:
                    lo,hi = bounds[:2]
                    if a['lo'] < hi and lo < a['hi']:
                        start = max(start,end+(sync_ns if source_q!=q else 0))
            end = start + times[scope,i]['gpu_ns']
            finish.append(end+(sync_ns if q else 0))
            for a in node['spans']:
                target = writes if a['write'] else reads
                if a['write']:
                    for table in (writes,reads):
                        old = table.setdefault(a['buffer'],{})
                        for bounds in list(old):
                            lo,hi = bounds[:2]
                            if a['lo'] <= lo and hi <= a['hi']: del old[bounds]
                ranges = target.setdefault(a['buffer'],{})
                key = (a['lo'],a['hi'],q)
                # Identical reads on different queues need both completion
                # constraints; retain them separately when pricing events.
                ranges[key] = (max(end,ranges.get(key,(0,q))[0]),q)
    return max(finish,default=0)


def partition(nodes, weights, waves, sync_ns, seconds):
    """Two queues of concurrent batches, with exact whole-batch barriers.

    A batch is a concurrent encoder. Its end waits for its slowest member.
    Events can only signal/wait between encoders, so a cross-queue dependency
    waits for the producing batch's end plus the measured handoff price.
    Queue zero supplies scope entry and consumes the join. Original waves on
    queue zero are always a feasible fallback. This admits arbitrary batching
    and independent-node reordering, rather than only two serial machines.
    """
    from ortools.sat.python import cp_model
    n = len(nodes)
    if not n:
        return {'ns':0, 'lower_bound_ns':0, 'optimal':True, 'assignment':[], 'batches':[]}
    model = cp_model.CpModel()
    horizon = sum(weights) + 2*n*sync_ns
    queue = [model.new_bool_var(f'q{i}') for i in range(n)]
    batch = [model.new_int_var(0, n-1, f'b{i}') for i in range(n)]
    start = [model.new_int_var(0, horizon, f's{i}') for i in range(n)]
    end = [model.new_int_var(0, horizon, f'e{i}') for i in range(n)]
    finish = model.new_int_var(0, horizon, 'finish')
    for i in range(n):
        model.add(end[i] >= start[i] + weights[i])
        model.add(start[i] >= sync_ns*queue[i])
        model.add(finish >= end[i] + sync_ns*queue[i])
    same_queue = {}
    for i in range(n):
        for j in range(i+1,n):
            same = model.new_bool_var(f'same{i}_{j}')
            model.add(queue[i] == queue[j]).only_enforce_if(same)
            model.add(queue[i] != queue[j]).only_enforce_if(same.Not())
            same_queue[i,j] = same
            before = model.new_bool_var(f'before{i}_{j}')
            equal = model.new_bool_var(f'equal{i}_{j}')
            after = model.new_bool_var(f'after{i}_{j}')
            model.add_exactly_one(before, equal, after)
            model.add(batch[i] < batch[j]).only_enforce_if(before)
            model.add(batch[i] == batch[j]).only_enforce_if(equal)
            model.add(batch[i] > batch[j]).only_enforce_if(after)
            model.add(start[j] >= end[i]).only_enforce_if([same,before])
            model.add(start[i] >= end[j]).only_enforce_if([same,after])
            model.add(start[i] == start[j]).only_enforce_if([same,equal])
            model.add(end[i] == end[j]).only_enforce_if([same,equal])
    reduced, _ = transitive_reduction(nodes)
    for j, pred in enumerate(reduced):
        for i in pred:
            same = same_queue[min(i,j),max(i,j)]
            model.add(batch[j] > batch[i]).only_enforce_if(same)
            model.add(start[j] >= end[i] + sync_ns).only_enforce_if(same.Not())
    baseline = 0
    for b, wave in enumerate(waves):
        duration = max(weights[i] for i in wave)
        for i in wave:
            model.add_hint(queue[i], 0)
            model.add_hint(batch[i], b)
            model.add_hint(start[i], baseline)
            model.add_hint(end[i], baseline + duration)
        baseline += duration
    model.add(finish <= baseline)
    model.add(finish >= critical_path(nodes, weights))
    model.add_hint(finish, baseline)
    model.minimize(finish)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = seconds
    solver.parameters.num_search_workers = 4
    solver.parameters.random_seed = 311
    status = solver.solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        # A solver timeout may not process the valid original-wave hint yet.
        return {'ns':baseline, 'lower_bound_ns':critical_path(nodes,weights),
                'optimal':False, 'status':solver.status_name(status),
                'assignment':[0]*n, 'batches':waves}
    assignment = [solver.value(q) for q in queue]
    grouped = {}
    for i in range(n):
        grouped.setdefault((assignment[i],solver.value(batch[i])),[]).append(i)
    batches = [{'queue':q,'nodes':ids,'start_ns':solver.value(start[ids[0]]),
                'end_ns':solver.value(end[ids[0]])} for (q,_),ids in grouped.items()]
    # Independently validate the returned schedule, including all unreduced
    # edges and scope fork/join, not just the solver's reduced graph.
    ends = [solver.value(e) for e in end]
    starts = [solver.value(s) for s in start]
    cost = solver.value(finish)
    for i,node in enumerate(nodes):
        assert ends[i]-starts[i] >= weights[i]
        assert starts[i] >= sync_ns*assignment[i]
        assert cost >= ends[i]+sync_ns*assignment[i]
        for p in node['predecessors']:
            assert starts[i] >= ends[p]+(sync_ns if assignment[p]!=assignment[i] else 0)
    for q in (0,1):
        ordered = sorted((b for b in batches if b['queue']==q),key=lambda b:b['start_ns'])
        assert all(a['end_ns'] <= b['start_ns'] for a,b in zip(ordered,ordered[1:]))
    return {'ns':cost, 'lower_bound_ns':round(solver.best_objective_bound),
            'optimal':status==cp_model.OPTIMAL, 'assignment':assignment,
            'batches':batches, 'critical_with_sync_ns':critical_path(nodes,weights,assignment,sync_ns)}


def summarize(path, sync_us=None, solve_seconds=2):
    records = [json.loads(line) for line in Path(path).read_text().splitlines() if line.startswith('{')]
    capture = next(r for r in reversed(records) if r.get('kind') == 'causal_graph')
    assert capture['mode'] == 'nodes'
    assert capture.get('node_isolation') == 'same-queue MTLEvent signal/wait between sampled encoders', 'overlapping-pass capture is provisional; recapture with node isolation'
    assert capture['truncated_waves'] == 0
    assert capture['bitwise']
    for a,b in zip(capture['rows'],capture['rows'][1:]):
        assert b['gpu_start'] >= a['gpu_end'], ('node sample windows overlap', a, b)
    times = {(r['scope'], r['node']): r for r in capture['rows']}
    results = []
    for plan in capture['plans']:
        validate_plan(plan)
        weights = [times[plan['scope'], i]['gpu_ns'] for i in range(len(plan['nodes']))]
        assert all(w > 0 for w in weights)
        serial = sum(weights)
        waves = sum(max(weights[i] for i in wave) for wave in plan['waves'])
        critical = critical_path(plan['nodes'], weights)
        reduced, ancestors = transitive_reduction(plan['nodes'])
        results.append({'scope':plan['scope'], 'nodes':len(weights), 'serial_ns':serial,
                        'wave_ns':waves, 'critical_ns':critical, 'gap_ns':waves-critical,
                        'max_width':max(map(len, plan['waves']), default=0),
                        'reduced_edges':sum(map(len,reduced)),
                        'kernels':[n['kernel'] for n in plan['nodes']],
                        'weights_ns':weights, 'predecessors':reduced, 'waves':plan['waves']})
        if sync_us is not None:
            results[-1]['two_queue'] = partition(plan['nodes'], weights, plan['waves'], round(sync_us*1000), solve_seconds)
    rounds = capture['rounds']
    assert len(capture['plans']) % rounds == 0, 'round scope boundaries need explicit capture'
    scopes_per_round = len(capture['plans'])//rounds
    global_cp = sum(global_span_critical(capture['plans'][r*scopes_per_round:(r+1)*scopes_per_round],times)
                    for r in range(rounds))/rounds/1e6
    walls = capture.get('reference_wall_ms_token', capture.get('reference_wall_ms_round'))
    wall = statistics.mean(walls)
    sums = {name:sum(r[name] for r in results)/rounds/1e6
            for name in ['serial_ns','wave_ns','critical_ns','gap_ns']}
    result = {'file':str(path), 'rounds':rounds, 'width':capture['width'], 'wall_ms_round':wall,
            'serial_ms':sums['serial_ns'], 'wave_ms':sums['wave_ns'], 'critical_ms':sums['critical_ns'],
            'zero_sync_ceiling_ms':sums['gap_ns'], 'zero_sync_ceiling_percent_wall':100*sums['gap_ns']/wall,
            'scopes':results}
    result['critical_global_ms'] = global_cp
    result['global_zero_sync_ceiling_ms'] = result['wave_ms']-global_cp
    result['global_zero_sync_ceiling_percent_wall'] = 100*(result['wave_ms']-global_cp)/wall
    if sync_us is not None:
        result['sync_us_per_handoff'] = sync_us
        result['two_queue_ms'] = sum(r['two_queue']['ns'] for r in results)/rounds/1e6
        result['critical_with_sync_ms'] = sum(r['two_queue'].get('critical_with_sync_ns', r['critical_ns']) for r in results)/rounds/1e6
        result['two_queue_lower_bound_ms'] = sum(r['two_queue']['lower_bound_ns'] for r in results)/rounds/1e6
        result['optimal_scopes'] = sum(r['two_queue']['optimal'] for r in results)
        result['two_queue_ceiling_ms'] = result['wave_ms']-result['two_queue_ms']
        result['two_queue_ceiling_percent_wall'] = 100*result['two_queue_ceiling_ms']/wall
        result['two_queue_saving_upper_bound_ms'] = result['wave_ms']-result['two_queue_lower_bound_ms']
        assignment = {s['scope']:s['two_queue']['assignment'] for s in results}
        result['critical_global_with_sync_ms'] = sum(global_span_critical(
            capture['plans'][r*scopes_per_round:(r+1)*scopes_per_round],times,assignment,round(sync_us*1000))
            for r in range(rounds))/rounds/1e6
    return result


def emit_plan(summary, destination):
    plans = []
    for scope in summary['scopes']:
        partition = scope['two_queue']
        if not any(partition['assignment']) or scope['wave_ns']-partition['ns'] < 5000:
            continue
        queues = []
        for q in (0,1):
            ordered = sorted((b for b in partition['batches'] if b['queue']==q),key=lambda b:b['start_ns'])
            queues.append([b['nodes'] for b in ordered])
        kernels = scope['kernels']
        ranges = []
        for join in [i for i,k in enumerate(kernels) if 'glm53_kda_decode' in k or 'online_selected_attention_f16_split_partial' in k]:
            pre = max((i for i,k in enumerate(kernels[:join]) if 'mhc4_pre_finish' in k),default=-1)
            if pre >= 0: ranges.append([pre+1,join])
        posts = [i for i,k in enumerate(kernels) if 'ffn_post_controls' in k]
        isolated = None
        if posts:
            pre = max((i for i,k in enumerate(kernels[:posts[-1]]) if 'hc_pre_wide' in k),default=-1)
            if pre >= 0: isolated = [pre+1,posts[-1]]
        elif any('moe_down' in k for k in kernels):
            joins = [i for i,k in enumerate(kernels) if k == 'kernel_add_f32']
            if joins:
                pre = max((i for i,k in enumerate(kernels[:joins[-1]]) if 'mhc4_pre_finish' in k),default=-1)
                if pre >= 0: isolated = [pre+1,joins[-1]]
        plans.append({'scope':scope['scope'],'kernels':kernels,'queues':queues,
                      'wave_ns':scope['wave_ns'],'partition_ns':partition['ns'],
                      'isolated_range':isolated,
                      'isolated_ranges':ranges+([isolated] if isolated is not None else [])})
    Path(destination).write_text(json.dumps({'schema':1,'source':summary['file'],
        'sync_us':summary['sync_us_per_handoff'],'plans':plans},indent=2)+'\n')


def self_test():
    # Long branch and 20-us short chain: ASAP waves pay 100+10; the DAG pays 100.
    nodes = [{'predecessors':[]},{'predecessors':[]},{'predecessors':[1]}, {'predecessors':[0,2]}]
    assert critical_path(nodes, [100,10,10,1]) == 101
    assert critical_path(nodes, [100,10,10,1], [0,1,1,0], 4) == 101
    assert critical_path(nodes, [10,100,100,1], [0,1,1,0], 4) == 209
    reduced, _ = transitive_reduction([{'predecessors':[]},{'predecessors':[0]}, {'predecessors':[0,1]}])
    assert reduced == [[],[0],[1]]
    a = {'buffer':1,'lo':0,'hi':4,'write':True}
    assert conflict(a, dict(a, write=False))
    assert not conflict(a, dict(a, lo=4, hi=8))
    assert not conflict(dict(a, write=False), dict(a, write=False))
    schedule = partition(nodes, [100,10,10,1], [[0,1],[2],[3]], 4, 5)
    assert schedule['optimal'] and schedule['ns'] == 101, schedule
    schedule = partition(nodes, [10,100,100,1], [[0,1],[2],[3]], 4, 5)
    assert schedule['optimal'] and schedule['ns'] == 201, schedule
    plans = [{'scope':1,'nodes':[{'spans':[a]}]},
             {'scope':2,'nodes':[{'spans':[dict(a,buffer=2)]}]}]
    times = {(1,0):{'gpu_ns':100},(2,0):{'gpu_ns':20}}
    assert global_span_critical(plans,times)==100
    plans[1]['nodes'][0]['spans']=[dict(a,write=False)]
    assert global_span_critical(plans,times)==120
    assert global_span_critical(plans,times,{1:[0],2:[1]},4)==128
    print('two_queue_ceiling self-tests: passed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('captures', nargs='*')
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--sync-us', type=float)
    parser.add_argument('--solve-seconds', type=float, default=2)
    parser.add_argument('--emit-plan', type=Path)
    args = parser.parse_args()
    if args.self_test:
        self_test()
    for path in args.captures:
        summary = summarize(path, args.sync_us, args.solve_seconds)
        if args.emit_plan:
            assert len(args.captures)==1 and args.sync_us is not None
            emit_plan(summary,args.emit_plan)
        print(json.dumps(summary), flush=True)
