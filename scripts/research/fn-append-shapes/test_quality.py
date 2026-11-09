import importlib.util
from pathlib import Path
import unittest
spec=importlib.util.spec_from_file_location('quality',Path(__file__).with_name('analyze_quality.py'))
q=importlib.util.module_from_spec(spec);spec.loader.exec_module(q)

class Decisions(unittest.TestCase):
 def record(self,doc,delta,noise,kl=0):
  return dict(kind='quality',depth=65536,rows=400,document=doc,repeat_bitwise=True,
   targets=[dict(candidate=dict(dnll=delta,kl=kl),noise_control=dict(dnll=noise)) for _ in range(4)])
 def test_signed_improvement_is_accepted_at_zero_repeat_and_arithmetic_noise(self):
  x=q.summarize([self.record(str(i),-.02,0) for i in range(4)])[0]
  self.assertTrue(x['passes']);self.assertLess(x['upper95']['32'],0)
 def test_positive_loss_above_noise_or_kl_above_limit_is_rejected(self):
  self.assertFalse(q.summarize([self.record(str(i),.02,.001) for i in range(4)])[0]['passes'])
  self.assertFalse(q.summarize([self.record(str(i),-.01,.001,.02) for i in range(4)])[0]['passes'])
 def test_repeat_noise_and_duplicate_documents_are_explicit_failures(self):
  row=self.record('x',0,0);row['repeat_bitwise']=False
  with self.assertRaises(AssertionError):q.summarize([row])
  row['repeat_bitwise']=True
  with self.assertRaises(AssertionError):q.summarize([row,row])
 def test_seeded_block_bootstrap_is_reproducible(self):
  d=[[.1,-.1,.2,-.2],[.3,-.2,.1,0]]
  self.assertEqual(q.upper_bootstrap(d,2,reps=100),q.upper_bootstrap(d,2,reps=100))

if __name__=='__main__':unittest.main()
