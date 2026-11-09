#!/usr/bin/env python3
"""Summarize fn_qsa_select paired whole-token arms: RUN_DIR with DEPTH.jsonl files.

Pairs each candidate arm with the incumbent arm of the same repetition (even
repetitions run incumbent first, odd candidate first). Pressure 2 arms are
rejected, pressure 1 and compiler overlap are flagged. Accuracy rows compare
teacher-forced full-vocabulary heads with the incumbent; the incumbent repeat
gives the noise floor.
"""

import argparse
import json
from pathlib import Path
from statistics import median


def level(conditions):
    return max(int(c["thermal"].split()[-1]) for c in conditions)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    out = []
    for path in sorted(args.directory.glob("*.jsonl"), key=lambda p: int(p.stem)):
        rows = [
            json.loads(line)
            for line in path.read_text().splitlines()
            if line.startswith("{")
        ]
        timing = [r for r in rows if r["kind"] == "timing"]
        by = {}
        for r in timing:
            by.setdefault((r["candidate"], r["rep"]), {})[r["mode"]] = r
        pairs = []
        for (candidate, rep), arms in sorted(by.items()):
            base, cand = arms[0], arms[candidate]
            conditions = base["conditions"] + cand["conditions"]
            pairs.append(
                {
                    "rep": rep,
                    "order": "incumbent first" if rep % 2 == 0 else "candidate first",
                    "incumbent_tps": base["whole_tps"],
                    "candidate_tps": cand["whole_tps"],
                    "gain_pct": 100 * (cand["whole_tps"] / base["whole_tps"] - 1),
                    "gpu_saved_ms_per_token": (base["gpu_ms"] - cand["gpu_ms"])
                    / base["tokens"],
                    "pressure": level(conditions),
                    "compile_overlap": any(c["rustc"] for c in conditions),
                }
            )
        kept = [p for p in pairs if p["pressure"] < 2]
        accuracy = {}
        for r in rows:
            if r["kind"] in ("noise", "accuracy"):
                accuracy[r["kind"]] = {
                    "signed_dnll_mean": sum(x["dnll"] for x in r["rows"])
                    / len(r["rows"]),
                    "kl_mean": sum(x["kl"] for x in r["rows"]) / len(r["rows"]),
                    "kl_max": max(x["kl"] for x in r["rows"]),
                    "max_abs_logit": max(x["max_abs"] for x in r["rows"]),
                    "rows": len(r["rows"]),
                }
        item = {
            "depth": int(path.stem),
            "pairs": pairs,
            "median_incumbent_tps": median(p["incumbent_tps"] for p in kept),
            "median_candidate_tps": median(p["candidate_tps"] for p in kept),
            "median_gain_pct": median(p["gain_pct"] for p in kept),
            "median_gpu_saved_ms": median(p["gpu_saved_ms_per_token"] for p in kept),
            "rejected_pressure2": len(pairs) - len(kept),
            "flagged": [
                p["rep"] for p in kept if p["pressure"] == 1 or p["compile_overlap"]
            ],
            "accuracy": accuracy,
        }
        out.append(item)
        print(
            f"depth {item['depth']}: incumbent {item['median_incumbent_tps']:.2f} -> "
            f"candidate {item['median_candidate_tps']:.2f} tok/s, gain {item['median_gain_pct']:+.2f}% "
            f"(GPU {item['median_gpu_saved_ms']:+.3f} ms/token), rejected {item['rejected_pressure2']}, "
            f"flagged reps {item['flagged']}"
        )
        for p in pairs:
            print(
                f"  rep {p['rep']} {p['order']:>15}: {p['incumbent_tps']:.2f} / {p['candidate_tps']:.2f} "
                f"{p['gain_pct']:+.2f}% gpu {p['gpu_saved_ms_per_token']:+.3f} ms "
                f"pressure {p['pressure']} compile {p['compile_overlap']}"
            )
        for kind, a in accuracy.items():
            print(
                f"  {kind}: dNLL {a['signed_dnll_mean']:.3e} KL mean {a['kl_mean']:.3e} "
                f"max {a['kl_max']:.3e} max|dlogit| {a['max_abs_logit']:.3e} ({a['rows']} rows)"
            )
    (args.directory / "summary.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
