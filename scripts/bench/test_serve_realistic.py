"""CPU-only steady-window accounting and rate comparison checks."""
import itertools
import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parent))
import serve_realistic as bench
import serve_realistic_pair as pair

class SteadyTests(unittest.TestCase):
    def test_serial_arm_is_serial_for_each_family_and_default_clears_inherited_cap(self):
        knobs = dict(fn='QWEN_SERVE_FN_MAX_ACTIVE', glm='QWEN_SERVE_GLM_MAX_ACTIVE',
                     v41='QWEN_SERVE_V41_MAX_ACTIVE')
        inherited = {k: '2' for k in knobs.values()}
        inherited['QWEN_FN_BATCH_GDN'] = '0'
        for family, knob in knobs.items():
            serial = bench.arm_environment(inherited, family, 'serial')
            self.assertEqual({k:v for k,v in serial.items() if k in knobs.values()}, {knob:'1'})
            default = bench.arm_environment(inherited, family, 'default')
            self.assertTrue(all(k not in default for k in knobs.values()))
            self.assertEqual(default['QWEN_FN_BATCH_GDN'], '0')

    def test_sampling_defaults_match_agents_and_env_overrides_reach_the_arm(self):
        self.assertEqual(bench.sampling_fields('agent'), {})
        self.assertEqual(bench.sampling_fields('greedy'), dict(temperature=0, top_p=1))
        self.assertEqual(bench.sampling_fields('custom', temperature=1.0, top_p=0.95),
                         dict(temperature=1.0, top_p=0.95))
        with self.assertRaises(ValueError):
            bench.sampling_fields('custom')
        env = bench.arm_environment({}, 'glm', 'default',
                                    ('QWEN_SERVE_GLM_SAMPLE_WORKERS=0',))
        self.assertEqual(env['QWEN_SERVE_GLM_SAMPLE_WORKERS'], '0')
        with self.assertRaises(ValueError):
            bench.arm_environment({}, 'glm', 'default', ('broken',))

    def test_compare_rejects_different_sampling(self):
        with tempfile.TemporaryDirectory() as d:
            a,b=Path(d)/'a',Path(d)/'b';a.mkdir();b.mkdir()
            common=dict(family='glm',seed=42,window_seconds=90,metric='delivered',
                        aggregate_tps=50,window_tokens=4500)
            bench.dump(a/'summary.json',dict(common,sampling=dict(mode='agent',fields={})))
            bench.dump(b/'summary.json',dict(common,sampling=dict(mode='greedy',
                         fields=dict(temperature=0,top_p=1))))
            with self.assertRaises(ValueError):bench.compare(a,b)

    def test_pressure1_is_flagged_and_pressure2_aborts_by_default(self):
        self.assertFalse(bench.thermal_exceeds_limit(0))
        self.assertFalse(bench.thermal_exceeds_limit(1))
        self.assertTrue(bench.thermal_exceeds_limit(2))
        self.assertTrue(bench.thermal_exceeds_limit(1,0))
        self.assertTrue(bench.thermal_exceeds_limit(-1))

    def test_warmup_exception_cannot_relax_ramp_or_window_limit(self):
        self.assertEqual(bench.thermal_phase_limit(1, None, False),1)
        self.assertEqual(bench.thermal_phase_limit(1, 2, False),2)
        self.assertEqual(bench.thermal_phase_limit(1, 2, True),1)
        self.assertTrue(bench.thermal_exceeds_limit(2,bench.thermal_phase_limit(1,2,True)))
        self.assertEqual(bench.thermal_phase_limit(0, 2, True),0)

    def test_thermal_settling_requires_continuous_low_pressure_and_resets(self):
        settle=bench.ThermalSettle(15,1)
        self.assertFalse(settle.ready(2,0))
        self.assertFalse(settle.ready(1,1))
        self.assertFalse(settle.ready(0,15))
        self.assertFalse(settle.ready(2,16))
        self.assertFalse(settle.ready(0,17))
        self.assertFalse(settle.ready(-1,30))
        self.assertFalse(settle.ready(1,31))
        self.assertTrue(settle.ready(1,46))

    def test_compile_and_pressure_flags_distinguish_warmup_from_window(self):
        rows=[dict(time=1,thermal=1,compilers=[dict(pid=7)]),
              dict(time=11,thermal=0,compilers=[]),dict(time=21,thermal=0,compilers=[])]
        flags=bench.timing_flags(rows,10,20)
        self.assertTrue(flags['thermal_flagged'])
        self.assertTrue(flags['compile_observed'])
        self.assertFalse(flags['compile_overlapped'])
        self.assertEqual(flags['timing_thermal_max'],0)
        rows[1].update(thermal=1,compilers=[dict(pid=8)])
        self.assertTrue(bench.timing_flags(rows,10,20)['compile_overlapped'])
        self.assertEqual(bench.timing_flags(rows,10,20)['timing_thermal_max'],1)
        self.assertEqual(bench.parse_compilers('17 42.0 /tool/rustc\n18 0.0 /tool/cargo\n19 3.0 /tool/qwen'),
                         [dict(pid=17,executable='rustc',cpu_percent=42.),dict(pid=18,executable='cargo',cpu_percent=0.)])

    def test_pair_keeps_both_orders_and_retains_flagged_results(self):
        self.assertEqual([v for _,v in pair.schedule()],[False,True,True,False])
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);common=dict(family='glm',seed=42,window_seconds=180,metric='delivered',
                                    thermal_max=1,thermal_flagged=True,compile_overlapped=True)
            for label,candidate in pair.schedule():
                x=root/label;x.mkdir();bench.dump(x/'summary.json',dict(common,aggregate_tps=60 if candidate else 50,window_tokens=10800 if candidate else 9000))
            r=pair.paired_report(root)
            self.assertTrue(r['complete']);self.assertFalse(r['opposite_throughput_directions'])
            self.assertEqual(r['order_effect_spread_percent'],0)
            self.assertTrue(all(p['thermal_flagged'] and p['compile_overlapped'] for p in r['pairs']))

    def test_sustained_serve_settles_to_one_then_allows_two(self):
        from types import SimpleNamespace
        args = SimpleNamespace(max_thermal_pressure=2,settle_max_thermal_pressure=1)
        self.assertEqual(bench.settle_limit(args),1)
        settle=bench.ThermalSettle(60,bench.settle_limit(args))
        self.assertFalse(settle.ready(2,0))
        self.assertFalse(settle.ready(1,1))
        self.assertTrue(settle.ready(1,61))
        self.assertFalse(bench.thermal_exceeds_limit(2,args.max_thermal_pressure))
        self.assertTrue(bench.thermal_exceeds_limit(3,args.max_thermal_pressure))
        self.assertEqual(bench.settle_limit(SimpleNamespace(max_thermal_pressure=1)),1)

    def test_thermal_seconds_clips_boundaries_and_accounts_for_unknown_time(self):
        samples=[dict(time=8,thermal=0),dict(time=12,thermal=1),dict(time=15,thermal=2),dict(time=24,thermal=0)]
        self.assertEqual(bench.thermal_seconds(samples,10,20),{'0':2.,'1':3.,'2':5.,'unknown':0.})
        self.assertEqual(bench.thermal_seconds(samples[1:],10,20),{'0':0.,'1':3.,'2':5.,'unknown':2.})
        self.assertEqual(sum(bench.thermal_seconds(samples,10,20).values()),10.)
        self.assertTrue(bench.timing_flags(samples,10,20)['thermal_flagged'])
        self.assertEqual(bench.timing_flags(samples,10,20)['timing_thermal_seconds']['2'],5.)

    def test_three_pair_ci_uses_arm_pairs_and_does_not_force_noisy_verdict(self):
        exact=pair.throughput_ci([(100,101),(100,101),(100,101)])
        self.assertEqual(exact['n'],3)
        self.assertAlmostEqual(exact['percent']['mean'],1)
        self.assertEqual(exact['percent']['half_width'],0)
        self.assertTrue(exact['excludes_loss_above_two_percent'])
        noisy=pair.throughput_ci([(100,97),(100,103),(100,100)])
        self.assertAlmostEqual(noisy['percent']['mean'],0)
        self.assertGreater(noisy['percent']['half_width'],2)
        self.assertFalse(noisy['noise_within_two_pp'])
        with self.assertRaises(ValueError):pair.throughput_ci([(100,100)])
        with self.assertRaises(ValueError):pair.throughput_ci([(0,1)]*3)

    def population(self):
        return [dict(model='glm',purpose='primary',reused='48000',reuse_source='live',
                     call=f'glm/{i}',new=str(5+i),generated=str(20+i),tool_union_ms=str(i*10))
                for i in range(6)]

    def test_continuous_draws_are_seeded_independently_per_agent_and_keep_pairs(self):
        rows=self.population();bad=[dict(rows[0],reused='0'),dict(rows[0],reuse_source='snapshot'),
                                   dict(rows[0],purpose='auxiliary')]
        plan=bench.make_plan(rows+bad,'glm',42,180)
        self.assertEqual(plan,bench.make_plan(rows,'glm',42,180))
        a=list(itertools.islice(bench.draws(plan,0),100))
        self.assertEqual(a,list(itertools.islice(bench.draws(plan,0),100)))
        self.assertNotEqual(a,list(itertools.islice(bench.draws(plan,1),100)))
        allowed={(int(r['new']),int(r['generated']),float(r['tool_union_ms'])/1000) for r in rows}
        self.assertTrue(all((x['append_tokens'],x['reply_tokens'],x['tool_gap_s']) in allowed for x in a))

    def test_token_delivery_clips_ramp_and_drain_and_includes_spanning_call(self):
        reason='response.reasoning_text.delta';text='response.output_text.delta'
        events=[(1,reason,'aa'),(11,reason,'bbb'),(15,text,'cccc'),(22,text,'dd')]
        self.assertEqual(bench.window_tokens(events,10,20,len),7)
        self.assertEqual(bench.window_tokens(events,0,30,len),11)

    def test_window_conservation_and_overlap_not_finite_drain(self):
        rows=[dict(sent=0,first=1,end=5),dict(sent=1,first=3,end=6),dict(sent=2,first=4,end=7)]
        x=bench.interval_buckets(rows,3,6)
        self.assertEqual(x['decode_stream_width_2_s'],2)
        self.assertEqual(x['decode_stream_width_3_s'],1)
        self.assertEqual(sum(x.values()),3)

    def row(self,agent,sent,first,end,events):
        return dict(label=f'a{agent}-t0',agent=agent,warm=False,sent=sent,first=first,end=end,
                    events=events,completed=False,prompt=None,reused=None,generated=None,
                    target=9062,spec={},x_qwen={})

    def test_partial_calls_count_and_pending_ttft_is_right_censored(self):
        k='response.output_text.delta'
        rows=[self.row(0,5,6,22,[(6,k,'aa'),(12,k,'bbbb')]),
              self.row(1,11,None,22,[]),self.row(2,9,13,22,[(13,k,'ccc')])]
        x=bench.summarize(rows,10,20,len)
        self.assertEqual(x['window_tokens'],7)
        self.assertEqual(x['aggregate_tps'],.7)
        self.assertEqual(x['agents'][1]['censored_ttft'],1)
        self.assertEqual(x['agents'][1]['queue_exposure_s'],9)
        self.assertEqual(x['agents'][0]['carry_in_requests'],1)
        self.assertAlmostEqual(sum(x['trace_20s'][0][f'decode_stream_width_{i}_s'] for i in (1,2,3))+
                               x['trace_20s'][0]['predecode_only_s']+x['trace_20s'][0]['idle_s'],10)
        with self.assertRaises(ValueError):bench.summarize(rows[:2],10,20,len)

    def test_rates_allow_unequal_token_counts_and_turn_counts(self):
        with tempfile.TemporaryDirectory() as d:
            a,b=Path(d)/'a',Path(d)/'b';a.mkdir();b.mkdir()
            common=dict(family='glm',seed=42,window_seconds=180,metric='delivered')
            bench.dump(a/'summary.json',dict(common,aggregate_tps=50,window_tokens=9000))
            bench.dump(b/'summary.json',dict(common,aggregate_tps=75,window_tokens=13500))
            self.assertEqual(bench.compare(a,b)['throughput_ratio'],1.5)
            bench.dump(b/'summary.json',dict(common,window_seconds=90,aggregate_tps=75,window_tokens=6750))
            with self.assertRaises(ValueError):bench.compare(a,b)

    def test_round_budget_matches_native_transition_and_reports_host(self):
        row=dict(first=0.,end=10.,window_tokens=100,prompt=42,
                 phases=dict(batch=dict(stepped_transitions=30,width_mean=2.0),
                             prefill_ms=1000.0,decode_ms=3000.0))
        lines=[dict(time=10.0,line='serve native: family=glm5_next transitions=30 transition_ms=200'),
               dict(time=10.0,line='serve phases: family=glm5_next prefill_ms=1000.0 '
                    'prefill_tokens=42 reused_tokens=0 boundary_ms=0.0 restore_ms=0.0 '
                    'snapshot_capture_ms=0.0 ladder_tip_capture_ms=0.0')]
        x=bench.round_budget([row],lines,0.,10.,0.1)
        self.assertAlmostEqual(x['transition_s'],0.1)
        self.assertAlmostEqual(x['prefill_s'],0.0)
        self.assertEqual(x['rounds_est'],15)
        self.assertAlmostEqual(x['transition_ms_per_round'],100/15)
        self.assertAlmostEqual(x['host_ms_per_round'],100-100/15)
        self.assertAlmostEqual(x['host_s'],(100-100/15)*15/1000)
        self.assertEqual(x['coverage'],1.0)
        self.assertEqual(bench.round_budget([],lines,0.,10.)['host_s'],0.0)

    def test_partial_prefill_record_is_counted_without_double_counting(self):
        completed=dict(first=5.,end=10.,window_tokens=100,prompt=42,completed=True,
                       phases=dict(batch=dict(stepped_transitions=30,width_mean=2.0),
                                   prefill_ms=1000.0,snapshot_capture_ms=0.0))
        partial=dict(first=None,end=8.,window_tokens=20,prompt=None,completed=False,phases={})
        lines=[dict(time=4.9,line='serve prefill: family=glm5_next complete=true cancelled=false '
                    'prefill_ms=1000.0 reused_tokens=0 prefill_tokens=42'),
               dict(time=8.0,line='serve prefill: family=glm5_next complete=false cancelled=true '
                    'prefill_ms=4000.0 reused_tokens=0 prefill_tokens=20')]
        x=bench.round_budget([completed,partial],lines,0.,10.,0.1)
        self.assertAlmostEqual(x['partial_prefill_s'],4.0)
        self.assertEqual(x['partial_prefill_calls'],1)
        # The completed turn's record is already in its phases span.
        self.assertAlmostEqual(x['prefill_s'],5.0)

    def test_completed_empty_eos_is_not_a_request_pending_at_cutoff(self):
        k='response.output_text.delta'
        empty=self.row(0,11,None,12,[])
        empty.update(completed=True,generated=1)
        rows=[empty,self.row(1,11,13,22,[(13,k,'ab')]),self.row(2,11,14,22,[(14,k,'cd')])]
        x=bench.summarize(rows,10,20,len)
        self.assertTrue(x['rows'][0]['empty_output'])
        self.assertFalse(x['rows'][0]['pending_at_cutoff'])
        self.assertEqual(x['agents'][0]['censored_ttft'],0)
        self.assertEqual(x['agents'][0]['queue_exposure_s'],1)

    def test_partial_reasoning_replays_with_an_assistant_turn(self):
        self.assertEqual(bench.history_items([dict(type='reasoning',content=[dict(text='λ')])]),
                         [dict(type='reasoning',content='λ'),dict(role='assistant',content='')])

if __name__=='__main__':unittest.main()
