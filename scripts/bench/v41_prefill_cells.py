#!/usr/bin/env python3
"""Summarize native paired cells; exclude the predeclared first warmup pair."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


def analyze(path, warmup_pairs):
    events=[json.loads(line) for line in path.read_text().splitlines()]
    cells=[r for r in events if r["kind"]=="cell" and r["rep"]>=2*warmup_pairs]
    exact=[r for r in events if r["kind"]=="exact"]
    if len(cells)<6 or len(cells)%2 or len(exact)<len(cells)+2*warmup_pairs:
        raise ValueError("incomplete paired packet")
    if not all(r["exact"] for r in exact):raise ValueError("numerical gate failed")
    baseline=[r for r in cells if r["policy"]=="baseline"]
    candidate=[r for r in cells if r["policy"]!="baseline"]
    pairs=[]
    for a,b in zip(cells[::2],cells[1::2]):
        a,b=(a,b) if a["policy"]=="baseline" else (b,a)
        pairs.append(dict(pair=a["rep"]//2,baseline_ms=a["wall_ms"],candidate_ms=b["wall_ms"],ratio=a["wall_ms"]/b["wall_ms"]))
    def stage(r):
        p=r["profile"]["operations"]
        return sum(o["wall_ms"] for o in p if o["operation"]=="ced_encoder_engram_prepare")+sum(o["encode_ms"] for o in p if o["completed_submits"])
    a,b=(statistics.mean(r["wall_ms"] for r in arm) for arm in (baseline,candidate))
    sa,sb=(statistics.mean(stage(r) for r in arm) for arm in (baseline,candidate))
    return dict(source=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        depth=cells[0]["depth"],rows=cells[0]["rows"],candidate=candidate[0]["policy"],
        warmup_pairs=warmup_pairs,retained_pairs=len(pairs),pairs=pairs,
        baseline_mean_ms=a,candidate_mean_ms=b,mean_ratio=a/b,
        geometric_pair_ratio=math.exp(statistics.mean(math.log(p["ratio"]) for p in pairs)),
        positive_pairs=sum(p["ratio"]>1 for p in pairs),all_exact=True,
        thermal_max=max(r["thermal"] for r in cells),
        exposed_engram_prepare_and_encode_baseline_ms=sa,
        exposed_engram_prepare_and_encode_candidate_ms=sb,
        exposed_engram_prepare_and_encode_ratio=sa/sb if sb else None)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("sources",nargs="+",type=Path);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--warmup-pairs",type=int,default=1)
    a=p.parse_args();result=[analyze(s,a.warmup_pairs) for s in a.sources]
    a.output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result),flush=True)
