#!/usr/bin/env python3
"""Signed document/block bootstrap against the declared arithmetic control."""
import argparse,collections,json,math,random,statistics as st
from pathlib import Path

def quantile(x,q):
 x=sorted(x);p=q*(len(x)-1);lo=math.floor(p);hi=math.ceil(p)
 return x[lo]*(hi-p)+x[hi]*(p-lo) if hi!=lo else x[lo]

def upper_bootstrap(docs,block,seed=20261109,reps=10000):
 rng=random.Random(seed);est=[]
 for _ in range(reps):
  means=[]
  for _ in docs:
   d=rng.choice(docs);n=len(d);draw=[]
   for _ in range(math.ceil(n/block)):
    start=rng.randrange(n);draw.extend(d[(start+j)%n] for j in range(block))
   means.append(sum(draw[:n])/n)
  est.append(sum(means)/len(means))
 return quantile(est,.95)

def summarize(records):
 groups=collections.defaultdict(list)
 for r in records:
  if r.get('kind')=='quality':groups[(r['depth'],r['rows'])].append(r)
 result=[]
 for (depth,rows),rs in sorted(groups.items()):
  assert len({r['document'] for r in rs})==len(rs)
  assert all(r['repeat_bitwise'] for r in rs)
  dnll=[[t['candidate']['dnll'] for t in r['targets']] for r in rs]
  noise=[st.mean(t['noise_control']['dnll'] for t in r['targets']) for r in rs]
  envelope=quantile([abs(x) for x in noise],.95)
  upper={str(b):upper_bootstrap(dnll,b) for b in (16,32,64)}
  kl=st.mean(st.mean(t['candidate']['kl'] for t in r['targets']) for r in rs)
  documents=[dict(id=r['document'],dnll=st.mean(d),noise_dnll=n,kl=st.mean(t['candidate']['kl'] for t in r['targets'])) for r,d,n in zip(rs,dnll,noise)]
  result.append(dict(depth=depth,rows=rows,documents=documents,targets_per_doc=len(dnll[0]),signed_dnll=st.mean(map(st.mean,dnll)),mean_kl=kl,noise_envelope=envelope,upper95=upper,
   passes=upper['32']<=envelope and kl<=.01,sensitivity_passes=all(v<=envelope for v in upper.values())))
 return result

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('files',nargs='+',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 data=[json.loads(l) for f in a.files for l in f.read_text().splitlines() if l.startswith('{')]
 result=summarize(data);a.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
