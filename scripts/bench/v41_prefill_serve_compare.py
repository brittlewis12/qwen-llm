#!/usr/bin/env python3
"""Reduce served roots and equal-sized short appends without mixing request counts."""
import argparse
import json
from pathlib import Path
import statistics

import serve_realistic as bench


def reduce(folder,tag):
    pair=json.loads((folder/f"{tag}-summary.json").read_text())
    arms={a["arm"]:a for a in pair["arms"]}
    roots={}; calls={}
    def artifact(arm,name):
        flat=folder/f"{tag}-{arm}-{name}"
        return flat if flat.exists() else folder/f"{tag}-{arm}"/name
    for arm in "ab":
        rows=json.loads(artifact(arm,"native-lines.json").read_text())
        found=[]
        for r in rows:
            if "serve prefill:" not in r["line"]:continue
            d=bench.kv(r["line"])
            if d.get("complete")=="true" and int(d.get("prefill_tokens",0))>=48000:
                found.append(dict(prefill_ms=float(d["prefill_ms"]),tokens=int(d["prefill_tokens"]),reused=int(d["reused_tokens"])))
        if len(found)<3:raise ValueError("missing served root phase")
        roots[arm]=found[:3]
        calls[arm]={r["label"]:r for r in (json.loads(s) for s in
            artifact(arm,"calls.jsonl").read_text().splitlines())}
    matched=[]
    for label in sorted(calls["a"].keys() & calls["b"].keys()):
        a,b=(calls[arm][label] for arm in "ab")
        if not all(r["offered_in_window"] and r["completed"] and r.get("spec") and
                   150<=r["spec"]["append_tokens"]<=2000 and r.get("phases") and
                   not r["phases"].get("ced") for r in (a,b)):continue
        if a["new"]!=b["new"]:continue
        matched.append(dict(label=label,new=a["new"],
            baseline_depth=a["reused"],candidate_depth=b["reused"],
            baseline_prefill_rows=a["phases"].get("prefill_rows"),
            candidate_prefill_rows=b["phases"].get("prefill_rows"),
            short_index_eligible=all(128<=r["phases"].get("prefill_rows",0)<=511 and
                r["reused"]>=49152 for r in (a,b)),
            mid_index_eligible=all(512<=r["phases"].get("prefill_rows",0)<=2048 and
                49152<=r["reused"]<131072 for r in (a,b)),baseline_prefill_ms=a["phases"]["prefill_ms"],
            candidate_prefill_ms=b["phases"]["prefill_ms"],baseline_ttft_s=a["ttft_s"],candidate_ttft_s=b["ttft_s"]))
    eligible=[r for r in matched if r["mid_index_eligible"]]
    short_eligible=[r for r in matched if r["short_index_eligible"]]
    summary=dict(tag=tag,order=pair["order"],binary_sha256=pair["binary_sha256"],roots=roots,
        root_prefill_ratio=statistics.mean(r["prefill_ms"] for r in roots["a"])/statistics.mean(r["prefill_ms"] for r in roots["b"]),
        root_client_ttft_s={arm:[r["ttft_s"] for r in arms[arm]["root_turns"]] for arm in "ab"},
        matched_short_appends=matched,
        short_append_median_pair_ratio=statistics.median(r["baseline_prefill_ms"]/r["candidate_prefill_ms"] for r in matched) if matched else None,
        short_append_mean_ratio=sum(r["baseline_prefill_ms"] for r in matched)/sum(r["candidate_prefill_ms"] for r in matched) if matched else None,
        short_append_positive_pairs=sum(r["baseline_prefill_ms"]>r["candidate_prefill_ms"] for r in matched),
        short_append_client_ttft_mean_ratio=sum(r["baseline_ttft_s"] for r in matched)/sum(r["candidate_ttft_s"] for r in matched) if matched else None,
        short_append_client_ttft_median_pair_ratio=statistics.median(r["baseline_ttft_s"]/r["candidate_ttft_s"] for r in matched) if matched else None,
        short_append_client_ttft_positive_pairs=sum(r["baseline_ttft_s"]>r["candidate_ttft_s"] for r in matched),
        mid_index_eligible_matches=len(eligible),
        mid_index_eligible_mean_ratio=sum(r["baseline_prefill_ms"] for r in eligible)/sum(r["candidate_prefill_ms"] for r in eligible) if eligible else None,
        mid_index_eligible_median_pair_ratio=statistics.median(r["baseline_prefill_ms"]/r["candidate_prefill_ms"] for r in eligible) if eligible else None,
        short_index_eligible_matches=len(short_eligible),
        short_index_eligible_mean_ratio=sum(r["baseline_prefill_ms"] for r in short_eligible)/sum(r["candidate_prefill_ms"] for r in short_eligible) if short_eligible else None,
        short_index_eligible_median_pair_ratio=statistics.median(r["baseline_prefill_ms"]/r["candidate_prefill_ms"] for r in short_eligible) if short_eligible else None,
        short_index_eligible_positive_pairs=sum(r["baseline_prefill_ms"]>r["candidate_prefill_ms"] for r in short_eligible),
        short_index_eligible_client_ttft_mean_ratio=sum(r["baseline_ttft_s"] for r in short_eligible)/sum(r["candidate_ttft_s"] for r in short_eligible) if short_eligible else None,
        short_index_eligible_client_ttft_median_pair_ratio=statistics.median(r["baseline_ttft_s"]/r["candidate_ttft_s"] for r in short_eligible) if short_eligible else None,
        short_index_eligible_client_ttft_positive_pairs=sum(r["baseline_ttft_s"]>r["candidate_ttft_s"] for r in short_eligible),
        aggregate_tps={arm:arms[arm]["summary"]["aggregate_tps"] for arm in "ab"},
        gap_p99_s={arm:arms[arm]["summary"]["gap_p99_s"] for arm in "ab"},
        thermal_max={arm:arms[arm]["summary"]["thermal_max"] for arm in "ab"},
        note="Root phases include all three roots; absent visible warm deltas remain null. Short appends match label and exact new-row count, exclude ramp/carry-in and incomplete calls, and retain every matched observation.")
    return summary


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("folder",type=Path);p.add_argument("tag")
    p.add_argument("--output",type=Path,required=True);a=p.parse_args();r=reduce(a.folder,a.tag)
    a.output.write_text(json.dumps(r,indent=2)+"\n");print(json.dumps(r),flush=True)
