# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy"]
# ///
"""Descriptive paired Half scalar/MMA subset analysis; no quality verdict.

Selection fixed before outputs: first prose/code documents at 2048, first long
document at 4352, 64 real continuation tokens; T03/T10/T11/T12 tools.
uv run scripts/reference/glm53/latent_quality_analysis.py PACKET.jsonl --out SUMMARY.json
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

from quality_analysis import tool_correct

HERE = Path(__file__).resolve().parent
ARMS = ("half_scalar", "half_mma")
TASKS = ("T03", "T10", "T11", "T12")


def selected_documents(fixture):
    return [
        (next(d for d in fixture["documents"] if d["stratum"] == stratum), prefix)
        for stratum, prefix in (("prose", 2048), ("code", 2048), ("long", 4352))
    ]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def analyze(raw, fixture_bytes):
    def invalid_constant(value):
        raise ValueError(f"nonfinite JSON constant: {value}")
    rows = [json.loads(line, parse_constant=invalid_constant) for line in raw.splitlines() if line.strip()]
    fixture = json.loads(fixture_bytes)
    require((fixture["name"], fixture["continuation"], fixture["tool_cap"]) == ("glm53-quality-v1", 64, 256), "fixture contract")
    require(len(rows) >= 2 and rows[0]["event"] == "header" and rows[-1]["event"] == "complete", "incomplete packet")
    header, complete = rows[0], rows[-1]
    require(sum(r["event"] == "header" for r in rows) == sum(r["event"] == "complete" for r in rows) == 1, "duplicate boundaries")
    require(header["schema"] == "glm53.latent_quality.v1", "schema")
    require(header["fixture_sha256"] == hashlib.sha256(fixture_bytes).hexdigest(), "fixture hash")
    require(header["arms"] == list(ARMS) and header["tasks"] == list(TASKS), "arm/task selection")
    require((header["continuation"], header["tool_cap"], header["seeds"], header["workspace_rows"], header["lineage"], header["precision"]) == (64, 256, [11], 2048, "Fast", "Half"), "run policy")
    selected = selected_documents(fixture)
    require(all(n in d["prefix_lengths"] and len(d["tokens"].split()) >= n + 64 for d, n in selected), "fixture continuations")
    require(header["documents"] == [{"path": d["path"], "stratum": d["stratum"], "prefix": n} for d, n in selected], "document selection")
    require(complete["documents_per_arm"] == 3 and complete["task_generations_per_arm"] == 8, "completion counts")
    documents = {(d["path"], n): d for d, n in selected}
    tasks = {t["id"]: t for t in fixture["tool_tasks"] if t["id"] in TASKS}
    results, tokens = {}, {}
    for row in rows[1:-1]:
        event = row["event"]
        require(event in {"session_begin", "warm_census", "document_begin", "document_token", "document", "task_begin", "task_token", "task"}, f"unknown event {event}")
        if event not in {"document", "document_token", "task", "task_token"}:
            continue
        require(row["arm"] in ARMS, "unexpected arm")
        if event.startswith("document"):
            require((row["path"], row["prefix"]) in documents, "unexpected document")
            key = ("document", row["path"], row["prefix"], row["arm"])
        else:
            g = row["generation"]
            require(row["id"] in tasks and (g["mode"], g["seed"]) in {("greedy", 0), ("sampled", 11)} and g["cap"] == 256, "task/generation")
            require(g["sampler"] == "qwen_llm::sampling v1" and g["top_k"] == g["min_p"] == 0 and g["temperature"] == (0 if g["mode"] == "greedy" else 1) and math.isclose(g["top_p"], 1 if g["mode"] == "greedy" else .95, abs_tol=1e-7), "sampling preset")
            key = ("task", row["id"], g["mode"], row["arm"])
        if event.endswith("_token"):
            stream = tokens.setdefault(key, [])
            require(type(row["index"]) is int and row["index"] == len(stream), "token order/duplicate")
            stream.append(row)
        else:
            require(key not in results, "duplicate result")
            results[key] = row
    expected = {("document", p, n, a) for p, n in documents for a in ARMS} | {("task", t, m, a) for t in TASKS for m in ("greedy", "sampled") for a in ARMS}
    require(set(results) == set(tokens) == expected, "missing/extra arm or item")
    text, tools = [], []
    for (path, prefix), document in documents.items():
        target = list(map(int, document["tokens"].split()))[prefix:prefix + 64]
        pair = [results["document", path, prefix, arm] for arm in ARMS]
        for arm, row in zip(ARMS, pair):
            stream = tokens["document", path, prefix, arm]
            require(row["stratum"] == document["stratum"] and row["continuation_tokens"] == target, "continuation binding")
            require(len(row["nll"]) == len(row["hits"]) == len(stream) == 64, "continuation length")
            require(all(type(x) in (int, float) and math.isfinite(x) for x in row["nll"]) and all(type(x) is bool for x in row["hits"]), "NLL/hit values")
            require(all(type(r["nll"]) in (int, float) and math.isfinite(r["nll"]) and type(r["hit"]) is bool for r in stream), "incremental NLL/hit values")
            require([(r["token"], r["nll"], r["hit"]) for r in stream] == list(zip(target, row["nll"], row["hits"])), "incremental document mismatch")
        delta = [b - a for a, b in zip(pair[0]["nll"], pair[1]["nll"])]
        text.append({"path": path, "prefix": prefix, "stratum": document["stratum"],
                     "arms": {a: {"mean_nll": sum(r["nll"]) / 64, "hits": sum(r["hits"])} for a, r in zip(ARMS, pair)},
                     "mean_nll_delta": sum(delta) / 64, "per_token_nll_delta": delta,
                     "hit_delta": sum(pair[1]["hits"]) - sum(pair[0]["hits"])})
    for task in TASKS:
        for mode in ("greedy", "sampled"):
            pair = [results["task", task, mode, a] for a in ARMS]
            require(pair[0]["generation"] == pair[1]["generation"], "paired generation settings")
            arms = {}
            for arm, row in zip(ARMS, pair):
                emitted, stop = row["emitted_ids"], row["stop"]
                require(type(row["emitted"]) is int and row["emitted"] == len(emitted), "emitted count")
                require(isinstance(row["text"], str) and row["expected"] == tasks[task]["expected"], "task binding")
                require(len(emitted) <= 256 and all(type(t) is int and t >= 0 and t not in (154820, 154827, 154829) for t in emitted), "emitted IDs")
                require((stop is None and len(emitted) == 256) or (type(stop) is int and stop in (154820, 154827, 154829) and len(emitted) < 256), "stop/truncation")
                stream = tokens["task", task, mode, arm]
                require([r["token"] for r in stream] == emitted + ([] if stop is None else [stop]) and all(r["generation"] == row["generation"] for r in stream), "incremental task mismatch")
                require(isinstance(row["calls"], list) and all(isinstance(c, dict) and isinstance(c.get("arguments"), dict) and isinstance(c.get("name"), str) for c in row["calls"]), "call structure")
                arms[arm] = {k: row[k] for k in ("text", "emitted_ids", "calls", "stop", "parse_error")}
                arms[arm].update(correct=bool(tool_correct(row, tasks[task]["expected"], task)), truncated=stop is None)
            tools.append({"id": task, "mode": mode, "arms": arms, "text_differs": pair[0]["text"] != pair[1]["text"],
                          "tokens_differ": pair[0]["emitted_ids"] != pair[1]["emitted_ids"], "correct_delta": int(arms[ARMS[1]]["correct"]) - int(arms[ARMS[0]]["correct"])})
    return {"schema": "glm53.latent_quality_analysis.v1", "scope": "Descriptive fixed subset; no margins, quality gate, or noninferiority claim. Deltas are half_mma minus half_scalar.",
            "report_sha256": hashlib.sha256(raw).hexdigest(), "fixture_sha256": header["fixture_sha256"], "packet_header": header,
            "analysis_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "scorer_sha256": hashlib.sha256((HERE / "quality_analysis.py").read_bytes()).hexdigest(),
            "documents": text, "overall": {"mean_nll_delta": sum(r["mean_nll_delta"] for r in text) / 3,
            "hit_delta": sum(r["hit_delta"] for r in text), "hit_rate_delta": sum(r["hit_delta"] for r in text) / 192},
            "tools": tools, "tool_counts": {m: {a: sum(r["arms"][a]["correct"] for r in tools if r["mode"] == m) for a in ARMS} for m in ("greedy", "sampled")}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--fixture", type=Path, default=HERE / "quality-v1.json")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    try:
        result = analyze(args.report.read_bytes(), args.fixture.read_bytes())
    except (ValueError, KeyError, TypeError, StopIteration) as error:
        parser.error(f"invalid packet: {error}")
    output = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.out:
        with args.out.open("x") as out:
            out.write(output)
    else:
        print(output, end="")


if __name__ == "__main__":
    main()
