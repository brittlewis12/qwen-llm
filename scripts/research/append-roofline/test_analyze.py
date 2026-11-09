import importlib.util
from pathlib import Path
import unittest

spec=importlib.util.spec_from_file_location('append_analysis',Path(__file__).with_name('analyze.py'))
analysis=importlib.util.module_from_spec(spec);spec.loader.exec_module(analysis)

class Floors(unittest.TestCase):
    def fixture(self,rows):
        banks=[dict(name='blk.0.ffn_'+name+'_exps.weight',shape=[2,4,16],bytes=512,
                    role='ExpertDown' if name=='down' else 'ExpertGateUp') for name in ('gate','up','down')]
        inv=banks+[dict(name='output.weight',shape=[2,8],bytes=64,role='Head'),
                   dict(name='token_embd.weight',shape=[2,1000000],bytes=8000000,role='Embedding')]
        routes=[dict(layer=0,counts=[rows]*8+[0]*8,bank_bytes=[512]*3)]
        return inv,routes

    def test_weight_union_is_not_active_bytes_times_rows(self):
        a=analysis.weight_floor(*self.fixture(1),1,'glm')
        b=analysis.weight_floor(*self.fixture(400),400,'glm')
        self.assertEqual(a['unique_weight_bytes'],768+64)
        self.assertEqual(a['unique_weight_bytes'],b['unique_weight_bytes'])
        self.assertEqual(b['matrix_flops']-2*16,(a['matrix_flops']-2*16)*400)

    def test_bandwidth_can_bind_without_more_useful_flops(self):
        inv,routes=self.fixture(400)
        a=analysis.weight_floor(inv,routes,400,'glm')
        routes[0]['bank_bytes']=[1e12]*3
        for t in inv[:3]:t['bytes']=1e12
        b=analysis.weight_floor(inv,routes,400,'glm')
        self.assertEqual(a['useful_flops'],b['useful_flops'])
        self.assertGreater(b['useful_bytes']/1175e9,b['useful_flops']/analysis.PEAK)

    def test_routing_conservation_is_required(self):
        inv,routes=self.fixture(400);routes[0]['counts'][0]-=1
        with self.assertRaises(AssertionError):analysis.weight_floor(inv,routes,400,'glm')

    def test_attention_is_not_also_a_learned_matrix(self):
        self.assertEqual(analysis.category('kernel_qwen4exp_qsa_selected_value_vec8_f32'),'context')
        self.assertEqual(analysis.category('kernel_mat_mat_q6_K_f32'),'learned')
        self.assertEqual(analysis.category('kernel_moe_route_bucket_slots_f32'),'non_gemm')

    def test_overlapping_counter_starts_and_gaps_partition_the_timeline(self):
        rows=[dict(kernel='a',gpu_start=1,gpu_end=101,gpu_ns=100),
              dict(kernel='b',gpu_start=51,gpu_end=201,gpu_ns=150),
              dict(kernel='c',gpu_start=251,gpu_end=301,gpu_ns=50)]
        p=analysis.interval_parts(rows)
        self.assertAlmostEqual(sum(x['core_ms'] for x in p.values()),250/1e6)
        self.assertAlmostEqual(sum(x['gap_ms'] for x in p.values()),50/1e6)
        self.assertAlmostEqual(sum(x['increment_ms'] for x in p.values()),300/1e6)

    def test_profile_inflation_and_empty_launch_are_not_added_as_native_stalls(self):
        residual,total=analysis.control_allocation(547,2822,2336,14)
        self.assertEqual((residual,total),(61,61))
        self.assertEqual(analysis.control_allocation(1,550,576,9),(1,9))

if __name__=='__main__':unittest.main()
