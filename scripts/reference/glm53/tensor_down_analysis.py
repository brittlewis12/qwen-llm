# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy==2.4.4"]
# ///
"""Validate and describe tensor-down JSONL reports, without qualification limits.

uv run scripts/reference/glm53/tensor_down_analysis.py --archive ARCHIVE --out-dir OUT
Add --extended ARCHIVE/extended-v2r.jsonl only after that report is complete.
Only CPU analysis is performed. Outputs use allowlisted fields and LF newlines.
"""

import argparse
import csv
import hashlib
import itertools
import json
import math
import re
import struct
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from quality_analysis import CONTINUATION, RESAMPLES, SEED, bootstrap, tool_correct

HERE = Path(__file__).resolve().parent
ARMS = ("original", "dense_only", "dense_and_down")
BASELINES = ARMS[:2]
FLAGS = {
    "original": {"dense": False, "down": False},
    "dense_only": {"dense": True, "down": False},
    "dense_and_down": {"dense": True, "down": True},
}
SCHEMAS = {
    "quality": "glm53.tensor_down_quality.v1",
    "extended": "glm53.tensor_down_extended_quality.v1",
    "plain": "glm53.tensor_down_plain_abba.v1",
    "profile": "glm53.tensor_down_stock_profile.v1",
}
HASH_FIELDS = (
    "binary_sha256",
    "fixture_sha256",
    "packet_sha256",
    "candidate_source_sha256",
    "candidate_sha256",
    "packed_source_sha256",
    "dense_source_sha256",
    "down_source_sha256",
    "down_adapter_source_sha256",
)
STOP_IDS = {154820, 154827, 154829}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, label):
    require(
        math.isfinite(actual)
        and math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-12),
        f"{label}: {actual!r} != {expected!r}",
    )


def token_ids(text):
    return [int(x) for x in text.split()]


def token_hash(ids):
    return hashlib.sha256(struct.pack(f"<{len(ids)}I", *ids)).hexdigest()


def events(rows, event):
    return [r for r in rows if r["event"] == event]


def unique(rows, key, label):
    result = {}
    for row in rows:
        k = key(row)
        require(k not in result, f"duplicate {label}: {k}")
        result[k] = row
    return result


def read_report(path, kind, fixture, fixture_sha256):
    raw = path.read_bytes()
    require(raw.endswith(b"\n"), f"{kind}: truncated final line")
    rows = [json.loads(line) for line in raw.splitlines()]
    validate_common(rows, kind, fixture, fixture_sha256)
    return rows, hashlib.sha256(raw).hexdigest()


def selected_documents(fixture, extended):
    docs = fixture["documents"]
    if not extended:
        docs = [
            next(d for d in docs if d["stratum"] == s)
            for s in ("prose", "code", "long")
        ]
    return {
        d["path"]: {
            "path": d["path"],
            "stratum": d["stratum"],
            "prefix": max(d["prefix_lengths"]),
            "tokens_sha256": token_hash(token_ids(d["tokens"])),
        }
        for d in docs
    }


def validate_common(rows, kind, fixture, fixture_sha256):
    require(rows and rows[0]["event"] == "header", f"{kind}: missing header")
    require(len(events(rows, "header")) == 1, f"{kind}: multiple headers")
    require(rows[-1]["event"] == "complete", f"{kind}: not COMPLETE")
    require(len(events(rows, "complete")) == 1, f"{kind}: multiple completion records")
    h, end = rows[0], rows[-1]
    require(h["schema"] == SCHEMAS[kind], f"{kind}: schema")
    require(h["fixture_sha256"] == fixture_sha256, f"{kind}: fixture hash")
    require(
        h["artifact_layout"] == fixture["artifact_layout"], f"{kind}: artifact layout"
    )
    require(h["lineage"] == "Fast" and h["precision"] == "Half", f"{kind}: arithmetic")
    require(h["mtl_debug_layer"] == "1", f"{kind}: Metal validation")
    for field in HASH_FIELDS:
        if field in h:
            require(re.fullmatch(r"[0-9a-f]{64}", h[field]), f"{kind}: {field}")
    required_hashes = (
        ("candidate_sha256", "dense_source_sha256", "packed_source_sha256")
        if kind == "profile"
        else tuple(k for k in HASH_FIELDS if k != "candidate_sha256")
    )
    require(all(k in h for k in required_hashes), f"{kind}: missing provenance hash")
    if kind == "profile":
        require(
            h["dense_tensor"] is True and h["down_tensor"] is False, "profile flags"
        )
        doc = next(d for d in fixture["documents"] if d["path"] == h["document"])
        require(h["rows"] == 2048 and h["capacity"] == 2048, "profile geometry")
        require(
            token_hash(token_ids(doc["tokens"])[:2048]) == h["tokens_sha256"],
            "profile token hash",
        )
        return
    arms = ARMS[1:] if kind == "plain" else ARMS
    require(h["arms"] == list(arms), f"{kind}: arms")
    require(
        h["arm_flags"] == [{"arm": a, **FLAGS[a]} for a in arms], f"{kind}: arm flags"
    )
    require(h["continuation"] == CONTINUATION == 64, f"{kind}: continuation")
    require(
        h["workspace_rows"] == 2048 and h["q6_k_n64_min_n"] == 1024, f"{kind}: geometry"
    )
    require(
        h["artifact_layout_matches_frozen_fixture"] is True, f"{kind}: layout check"
    )
    require(
        h["quality_references"]["retained"] == list(BASELINES), f"{kind}: references"
    )
    require(h["quality_references"]["rows_each"] == 64, f"{kind}: reference rows")
    require(h["quality_references"]["full_vocab"] == 154880, f"{kind}: vocab")
    require(
        h["down_policy"]["dtype"] == "IQ3_S"
        and h["down_policy"]["rows"] == [1024, 2048],
        f"{kind}: down policy",
    )
    expected = selected_documents(fixture, kind == "extended")
    require(
        h["document_selection"]["documents"] == len(expected),
        f"{kind}: selection count",
    )
    require(
        h["document_selection"]["strata"]
        == dict(Counter(d["stratum"] for d in expected.values())),
        f"{kind}: selection strata",
    )
    require(
        unique(h["documents"], lambda r: r["path"], "header document") == expected,
        f"{kind}: frozen documents/hash",
    )
    require(
        end["execution_complete"] is True and end["arms"] == len(arms),
        f"{kind}: execution",
    )
    quality = kind != "plain"
    require(
        end["documents_per_arm"] == (len(expected) if quality else 0),
        f"{kind}: document count",
    )
    require(
        end["quality_rows_per_document_arm"] == (64 if quality else 0),
        f"{kind}: row count",
    )
    require(end["plain_samples"] == (0 if quality else 16), f"{kind}: plain count")
    for row in rows:
        if "arm" in row:
            require(row["arm"] in arms, f"{kind}: unknown arm")
        for field, expected_value in (
            ("lineage", "Fast"),
            ("precision", "Half"),
            ("workspace_rows", 2048),
        ):
            if field in row:
                require(row[field] == expected_value, f"{kind}: row {field}")


def chunks(prefix):
    return [min(2048, prefix - start) for start in range(0, prefix, 2048)]


def validate_quality(rows, fixture, extended):
    end = rows[-1]
    docs = selected_documents(fixture, extended)
    count = 22 if extended else 3
    require(len(docs) == count, "document cohort size")
    summaries = unique(
        events(rows, "document_arm"), lambda r: (r["arm"], r["path"]), "document arm"
    )
    begins = unique(
        events(rows, "document_begin"),
        lambda r: (r["arm"], r["path"]),
        "document begin",
    )
    want = set(itertools.product(ARMS, docs))
    require(set(summaries) == set(begins) == want, "document arm matching")
    tokens = unique(
        events(rows, "document_token"),
        lambda r: (r["arm"], r["path"], r["index"]),
        "document token",
    )
    require(
        set(tokens) == {(a, d, i) for a, d in want for i in range(64)},
        "64 unique tokens per document/arm",
    )
    fixture_docs = {d["path"]: d for d in fixture["documents"]}
    for (arm, doc), summary in summaries.items():
        prefix = docs[doc]["prefix"]
        targets = token_ids(fixture_docs[doc]["tokens"])[prefix : prefix + 64]
        require(
            len(targets) == 64 and summary["continuation_tokens"] == targets,
            "frozen continuation",
        )
        for row in (summary, begins[arm, doc]):
            require(
                row["prefix"] == prefix and row["stratum"] == docs[doc]["stratum"],
                "document metadata",
            )
        require(summary["prefill_chunks"] == chunks(prefix), "document chunks")
        require(
            summary["down_eligible_chunks"]
            == sum(n in (1024, 2048) for n in chunks(prefix)),
            "document eligibility",
        )
        require(summary["actual_prefill_rows"] == 2048, "document workspace")
        require(
            summary["logit_rows"] == 64 and summary["vocab"] == 154880,
            "document logits",
        )
        require(
            summary["final_position"] == prefix + 63 and summary["decode_calls"] == 63,
            "document positions",
        )
        ts = [tokens[arm, doc, i] for i in range(64)]
        for i, t in enumerate(ts):
            require(
                t["prefix"] == prefix and t["stratum"] == docs[doc]["stratum"],
                "token metadata",
            )
            require(
                t["target_position"] == prefix + i and t["target_token"] == targets[i],
                "token alignment",
            )
            require(
                t["logits_finite"] is True and math.isfinite(t["nll"]),
                "finite logits/NLL",
            )
            require(t["top1_hit"] == (t["argmax"] == targets[i]), "top1 hit")
            require(set(t["references"]) == set(BASELINES), "both token references")
            for base in BASELINES:
                ref, b = t["references"][base], tokens[base, doc, i]
                close(ref["nll"], b["nll"], "reference NLL")
                close(ref["nll_delta"], t["nll"] - b["nll"], "paired NLL")
                require(
                    ref["argmax"] == b["argmax"] and ref["top1_hit"] == b["top1_hit"],
                    "reference top1",
                )
                require(
                    ref["top1_agrees"] == (t["argmax"] == b["argmax"]), "top1 agreement"
                )
                require(ref["reference_self"] == (arm == base), "self reference")
                require(math.isfinite(ref["kl_reference_to_arm"]), "finite KL")
                if arm == base:
                    close(ref["kl_reference_to_arm"], 0, "self KL")
        close(summary["mean_nll"], np.mean([t["nll"] for t in ts]), "summary NLL")
        require(summary["top1_hits"] == sum(t["top1_hit"] for t in ts), "summary top1")
        require(set(summary["references"]) == set(BASELINES), "both summary references")
        for base in BASELINES:
            ref = summary["references"][base]
            rs = [t["references"][base] for t in ts]
            close(
                ref["mean_nll"],
                summaries[base, doc]["mean_nll"],
                "summary reference NLL",
            )
            require(
                ref["top1_hits"] == summaries[base, doc]["top1_hits"],
                "summary reference hits",
            )
            require(
                ref["top1_agreements"] == sum(r["top1_agrees"] for r in rs),
                "summary agreements",
            )
            for field, values, statistic in (
                ("mean_nll_delta", [r["nll_delta"] for r in rs], np.mean),
                (
                    "mean_kl_reference_to_arm",
                    [r["kl_reference_to_arm"] for r in rs],
                    np.mean,
                ),
                (
                    "max_kl_reference_to_arm",
                    [r["kl_reference_to_arm"] for r in rs],
                    max,
                ),
            ):
                close(ref[field], statistic(values), field)
    require(
        end["tool_tasks_per_arm"] == 4
        and end["tool_generations_per_task_arm"] == 2
        and end["tool_generations"] == 24,
        "tool completion counts",
    )
    tools = validate_tools(rows, fixture)
    return summaries, tools


def task_key(row):
    g = row["generation"]
    return row["arm"], row["id"], g["mode"], g["seed"]


def validate_tools(rows, fixture):
    h = rows[0]
    tasks = {
        t["id"]: t
        for t in fixture["tool_tasks"]
        if t["id"] in ("T09", "T10", "T11", "T12")
    }
    header_tasks = unique(h["tool_tasks"], lambda r: r["id"], "header task")
    require(set(header_tasks) == set(tasks), "header tool tasks")
    require(h["tool_definitions"] == fixture["tools"], "frozen tool definitions")
    generations = unique(
        h["tool_generations"], lambda r: (r["mode"], r["seed"]), "generation mode"
    )
    require(set(generations) == {("greedy", 0), ("sampled", 11)}, "tool modes/seeds")
    for mode, seed in generations:
        g = generations[mode, seed]
        require(
            g["cap"] == fixture["tool_cap"] == 256
            and g["top_k"] == 0
            and g["min_p"] == 0,
            "tool sampling policy",
        )
        require(
            g["temperature"] == (0 if mode == "greedy" else 1)
            and g["top_p"] == (1 if mode == "greedy" else 0.95),
            "tool sampling parameters",
        )
    want = {
        (a, t, mode, seed) for a in ARMS for t in tasks for mode, seed in generations
    }
    groups = {
        e: unique(events(rows, e), task_key, e)
        for e in ("task_begin", "task_prefill", "task_generation")
    }
    require(
        all(set(g) == want for g in groups.values()),
        "24 unique tool generations/begins/prefills",
    )
    token_rows = defaultdict(list)
    for row in events(rows, "task_token"):
        require(task_key(row) in want, "unexpected tool token")
        token_rows[task_key(row)].append(row)
    for tid, task in tasks.items():
        recorded = header_tasks[tid]
        require(
            recorded["expected"] == task["expected"]
            and recorded["context_level"] == task["context_level"],
            "fixture task expectation",
        )
        require(
            recorded["prompt_tokens"]
            == len(token_ids(task["prompt"]))
            == task["prompt_tokens"],
            "tool prompt length",
        )
        require(
            recorded["prompt_sha256"] == token_hash(token_ids(task["prompt"])),
            "tool frozen prompt hash",
        )
    for key in sorted(want):
        arm, tid, mode, seed = key
        task, recorded = tasks[tid], header_tasks[tid]
        row, begin, pre = (
            groups[e][key] for e in ("task_generation", "task_begin", "task_prefill")
        )
        for r in (row, begin):
            require(
                r["expected"] == task["expected"]
                and r["context_level"] == task["context_level"],
                "tool expectation",
            )
            require(
                r["prompt_sha256"] == recorded["prompt_sha256"]
                and r["prompt_tokens"] == recorded["prompt_tokens"],
                "tool prompt matching",
            )
            require(
                r["token_hash_encoding"] == "u32 little-endian", "tool hash encoding"
            )
        require(
            all(r["generation"] == generations[mode, seed] for r in (row, begin, pre)),
            "generation matching",
        )
        committed = chunks(task["prompt_tokens"])
        eligible = sum(n in (1024, 2048) for n in committed)
        require(
            pre["prefill_chunks"] == committed
            and pre["position"] == task["prompt_tokens"],
            "tool chunks",
        )
        require(
            pre["actual_prefill_rows"] == 2048
            and pre["down_eligible_chunks"] == eligible,
            "tool actual rows/eligibility",
        )
        require(
            pre["expected_down_iq3_s_tensor_calls"]
            == (39 * eligible if FLAGS[arm]["down"] else 0),
            "tool expected dispatches",
        )
        ts = sorted(token_rows[key], key=lambda t: t["sample_index"])
        sampled, emitted = row["sampled_ids"], row["emitted_ids"]
        require(0 < len(sampled) <= 256, "sample cap")
        require(
            [t["sample_index"] for t in ts] == list(range(len(sampled))),
            "unique tool token indexes",
        )
        require([t["token"] for t in ts] == sampled, "tool token IDs")
        require(
            all(
                t["logits_finite"] is True
                and t["position"] == task["prompt_tokens"] + i
                for i, t in enumerate(ts)
            ),
            "tool token positions/finite",
        )
        require(
            all(t["is_stop"] == (t["token"] in STOP_IDS) for t in ts),
            "tool stop markers",
        )
        require(not any(t["is_stop"] for t in ts[:-1]), "early tool stop")
        stop = sampled[-1] if ts[-1]["is_stop"] else None
        require(row["stop"] == stop, "recorded stop")
        require(
            emitted == (sampled[:-1] if stop is not None else sampled),
            "emitted excludes stop",
        )
        require(
            row["emitted_ids_exclude_stop"] is True
            and row["sampled_ids_include_stop"] is True,
            "token conventions",
        )
        for prefix, ids in (("sampled", sampled), ("emitted", emitted)):
            require(row[f"{prefix}_tokens"] == len(ids), "tool token count")
            require(
                row[f"{prefix}_sha256"] == token_hash(ids), "tool output token hash"
            )
        require(
            row["text_sha256"] == hashlib.sha256(row["text"].encode()).hexdigest(),
            "decoded text hash",
        )
        require(
            row["decode_calls"] == len(emitted)
            and row["final_position"] == task["prompt_tokens"] + len(emitted),
            "tool final position",
        )
    return groups["task_generation"]


def metadata(rows, kind, report_hash):
    h = rows[0]
    out = {"report": kind, "schema": h["schema"], "report_sha256": report_hash}
    out.update({k: h[k] for k in HASH_FIELDS if k in h})
    out.update({k: h[k] for k in ("lineage", "precision", "mtl_debug_layer")})
    out["workspace_rows"] = h.get("workspace_rows", h.get("rows"))
    out["tensor_table_sha256"] = h["artifact_layout"]["tensor_table_sha256"]
    for field in ("q6_k_n64_min_n", "tensor_tier", "metal_lease_wait"):
        if field in h:
            out[field] = h[field]
    return out


def quality_analysis(rows, fixture, extended):
    summaries, tools = validate_quality(rows, fixture, extended)
    strata = {d["path"]: d["stratum"] for d in rows[0]["documents"]}
    document_rows = []
    for (arm, doc), row in summaries.items():
        for base in BASELINES:
            ref = row["references"][base]
            document_rows.append(
                {
                    "arm": arm,
                    "dense": FLAGS[arm]["dense"],
                    "down": FLAGS[arm]["down"],
                    "baseline": base,
                    "document": doc,
                    "stratum": row["stratum"],
                    "prefix": row["prefix"],
                    "tokens": 64,
                    "mean_nll": row["mean_nll"],
                    "top1_hits": row["top1_hits"],
                    "top1_accuracy": row["top1_hits"] / 64,
                    "nll_delta": ref["mean_nll_delta"],
                    "top1_accuracy_delta": (row["top1_hits"] - ref["top1_hits"]) / 64,
                    "top1_agreement": ref["top1_agreements"] / 64,
                    "mean_kl_reference_to_arm": ref["mean_kl_reference_to_arm"],
                    "max_kl_reference_to_arm": ref["max_kl_reference_to_arm"],
                    "actual_prefill_rows": row["actual_prefill_rows"],
                    "prefill_chunks": row["prefill_chunks"],
                    "down_eligible_chunks": row["down_eligible_chunks"],
                    "prefill_ms": row["prefill_ms"],
                    "decode_ms": row["decode_ms"],
                    "timing_method": row["timing_scope"],
                }
            )
    comparisons = []
    metrics = (
        "mean_nll",
        "top1_accuracy",
        "nll_delta",
        "top1_accuracy_delta",
        "top1_agreement",
        "mean_kl_reference_to_arm",
    )
    for arm, base, stratum in itertools.product(
        ARMS, BASELINES, ("overall", "prose", "code", "long")
    ):
        sub = [
            r
            for r in document_rows
            if r["arm"] == arm
            and r["baseline"] == base
            and (stratum == "overall" or r["stratum"] == stratum)
        ]
        result = {
            "arm": arm,
            "baseline": base,
            "stratum": stratum,
            "documents": len(sub),
            "tokens": len(sub) * 64,
        }
        for metric in metrics:
            estimate = {"mean": float(np.mean([r[metric] for r in sub]))}
            if extended:
                by_doc = {r["document"]: [r[metric]] for r in sub}
                samples = bootstrap(by_doc, strata, True, np.mean)
                estimate["central90"] = [
                    float(x) for x in np.percentile(samples, [5, 95])
                ]
            result[metric] = estimate
        result["max_kl_reference_to_arm"] = max(
            r["max_kl_reference_to_arm"] for r in sub
        )
        comparisons.append(result)
    task_fixture = {t["id"]: t for t in fixture["tool_tasks"]}
    tool_rows = []
    pre = {task_key(r): r for r in events(rows, "task_prefill")}
    for key, row in tools.items():
        arm, tid, mode, seed = key
        tool_rows.append(
            {
                "arm": arm,
                **FLAGS[arm],
                "task": tid,
                "mode": mode,
                "seed": seed,
                "group": "down_unaffected_control"
                if tid in ("T09", "T10")
                else "down_eligible",
                "prompt_tokens": row["prompt_tokens"],
                "prompt_sha256": row["prompt_sha256"],
                "prefill_chunks": pre[key]["prefill_chunks"],
                "actual_prefill_rows": pre[key]["actual_prefill_rows"],
                "down_eligible_chunks": pre[key]["down_eligible_chunks"],
                "expected_down_iq3_s_tensor_calls": pre[key][
                    "expected_down_iq3_s_tensor_calls"
                ],
                "strict_correct": tool_correct(row, task_fixture[tid]["expected"], tid),
                "stop": row["stop"],
                "parse_error_present": bool(row["parse_error"]),
                "call_count": len(row["calls"] or []),
                "sampled_tokens": row["sampled_tokens"],
                "emitted_tokens": row["emitted_tokens"],
                "sampled_sha256": row["sampled_sha256"],
                "emitted_sha256": row["emitted_sha256"],
            }
        )
    identities = []
    for a, b in itertools.combinations(ARMS, 2):
        for tid, mode, seed in itertools.product(
            ("T09", "T10", "T11", "T12"), ("greedy", "sampled"), (0, 11)
        ):
            if (mode == "greedy") != (seed == 0):
                continue
            ra, rb = tools[a, tid, mode, seed], tools[b, tid, mode, seed]
            identities.append(
                {
                    "arm": a,
                    "other_arm": b,
                    "task": tid,
                    "mode": mode,
                    "seed": seed,
                    "sampled_ids_identical": ra["sampled_ids"] == rb["sampled_ids"],
                    "emitted_ids_identical": ra["emitted_ids"] == rb["emitted_ids"],
                    "stop_identical": ra["stop"] == rb["stop"],
                }
            )
    tool_summary = []
    for arm, mode, group in itertools.product(
        ARMS, ("greedy", "sampled"), ("all", "down_unaffected_control", "down_eligible")
    ):
        sub = [
            r
            for r in tool_rows
            if r["arm"] == arm
            and r["mode"] == mode
            and (group == "all" or r["group"] == group)
        ]
        tool_summary.append(
            {
                "arm": arm,
                "mode": mode,
                "group": group,
                "correct": sum(r["strict_correct"] for r in sub),
                "total": len(sub),
            }
        )
    return (
        {
            "documents": len(strata),
            "quality_token_rows": len(strata) * 3 * 64,
            "intervals": "central90 stratified document bootstrap"
            if extended
            else "none: one document per stratum",
            "comparisons": comparisons,
            "tool_summary": tool_summary,
            "tool_generations": len(tool_rows),
            "identical_token_pairs": sum(
                r["sampled_ids_identical"] for r in identities
            ),
            "token_pairs": len(identities),
        },
        document_rows,
        tool_rows,
        identities,
    )


def plain_analysis(rows):
    sample_rows = events(rows, "plain_sample")
    samples = unique(
        sample_rows, lambda r: (r["prompt_tokens"], r["sample"]), "plain sample"
    )
    require(
        set(samples) == set(itertools.product((2048, 4096), range(8))),
        "16 plain samples",
    )
    widths = unique(
        events(rows, "plain_width"), lambda r: r["prompt_tokens"], "plain width"
    )
    require(set(widths) == {2048, 4096}, "two plain widths")
    order = [ARMS[i] for i in (1, 2, 2, 1, 2, 1, 1, 2)]
    summary, curated = [], []
    for width in (2048, 4096):
        rs = [samples[width, i] for i in range(8)]
        require(
            [r["arm"] for r in rs] == order == widths[width]["order"], "ABBA/BAAB order"
        )
        require(widths[width]["samples_per_arm"] == 4, "plain samples per arm")
        for i, r in enumerate(rs):
            require(r["block"] == ("ABBA" if i < 4 else "BAAB"), "plain block")
            require(
                r["logits_finite"] is True and r["final_position"] == width,
                "plain finite/position",
            )
            require(
                r["actual_prefill_rows"] == 2048
                and r["expected_prefill_chunks"] == chunks(width),
                "plain geometry",
            )
            require(r["wall_ms"] > 0, "plain wall time")
            close(r["wall_ms"], widths[width]["wall_ms"][i], "plain width timing")
            curated.append(
                {
                    k: r[k]
                    for k in (
                        "arm",
                        "prompt_tokens",
                        "sample",
                        "block",
                        "wall_ms",
                        "actual_prefill_rows",
                        "expected_prefill_chunks",
                    )
                }
                | FLAGS[r["arm"]]
                | {"timing_method": r["scope"]}
            )
        means = {
            a: float(np.mean([r["wall_ms"] for r in rs if r["arm"] == a]))
            for a in ARMS[1:]
        }
        summary.append(
            {
                "prompt_tokens": width,
                "samples_per_arm": 4,
                "mean_wall_ms": means,
                "candidate_change_percent": 100 * (means[ARMS[2]] / means[ARMS[1]] - 1),
            }
        )
    return summary, curated


def profile_analysis(rows):
    profiles = events(rows, "profile")
    require(len(profiles) == 1, "one stage profile")
    p = profiles[0]
    runs = unique(events(rows, "run"), lambda r: r["label"], "profile run")
    require(len(runs) == 4, "profile warm/two plain/profiled runs")
    require(
        all(
            r["bits_equal_warm"] is True
            and r["logits_finite"] is True
            and r["actual_prefill_rows"] == 2048
            and r["position"] == 2048
            for r in runs.values()
        ),
        "profile run integrity",
    )
    plain = [r["wall_ms"] for r in runs.values() if r["plain_timing"]]
    require(plain == p["plain_wall_ms"] and len(plain) == 2, "profile plain timings")
    close(
        p["profiled_over_plain_mean"],
        p["profiled_wall_ms"] / np.mean(plain),
        "profiling overhead",
    )
    close(sum(p["stages_ms"].values()), p["scaled_span_sum_ms"], "profile stage sum")
    require(set(p["stages_ms"]) == set(p["stage_shares"]), "profile stages/shares")
    for stage in p["stages_ms"]:
        close(
            p["stage_shares"][stage],
            p["stages_ms"][stage] / p["scaled_span_sum_ms"],
            "profile share",
        )
    result = {
        k: p[k]
        for k in (
            "plain_wall_ms",
            "profiled_wall_ms",
            "profiled_over_plain_mean",
            "command_gpu_ms",
            "scaled_span_sum_ms",
            "stage_shares",
            "warning",
        )
    }
    result["profiling_overhead_percent"] = 100 * (p["profiled_over_plain_mean"] - 1)
    return result, [
        {
            "stage": s,
            "profiled_scaled_span_ms": ms,
            "share": p["stage_shares"][s],
            "dense": True,
            "down": False,
            "timing_method": rows[0]["timing"],
        }
        for s, ms in p["stages_ms"].items()
    ]


def census_rows(rows):
    keys = (
        "arm",
        "prompt_tokens",
        "prefill_chunks",
        "down_eligible_chunks",
        "iq3_s_generic_calls",
        "iq3_s_tensor_calls",
        "iq4_xs_incumbent_calls",
        "fused_gate_up_dispatches",
        "dense_tensor_calls",
        "timed",
    )
    return [
        {k: r[k] for k in keys} | FLAGS[r["arm"]] | {"scope": r["scope"]}
        for r in events(rows, "warm_down_attribution")
    ]


def validate_census(rows, kind):
    arms = ARMS[1:] if kind == "plain" else ARMS
    widths = (2048, 4096) if kind == "plain" else (2048,)
    census = unique(
        events(rows, "warm_down_attribution"),
        lambda r: (r["arm"], r["prompt_tokens"]),
        "down census",
    )
    require(set(census) == set(itertools.product(arms, widths)), "down census coverage")
    warm = unique(
        events(rows, "warm_census"),
        lambda r: (r["arm"], r["prompt_tokens"]),
        "warm census",
    )
    require(set(warm) == set(census), "matching warm/down census")
    for (arm, width), row in census.items():
        n = width // 2048
        require(
            row["prefill_chunks"] == chunks(width) and row["down_eligible_chunks"] == n,
            "census geometry",
        )
        require(
            row["iq3_s_generic_calls"] == (0 if FLAGS[arm]["down"] else 39 * n),
            "generic down calls",
        )
        require(
            row["iq3_s_tensor_calls"] == (39 * n if FLAGS[arm]["down"] else 0),
            "tensor down calls",
        )
        require(
            row["iq4_xs_incumbent_calls"] == 3 * n
            and row["fused_gate_up_dispatches"] == 84 * n,
            "unchanged down/gate-up counts",
        )
        require(
            row["dense_tensor_calls"]
            == warm[arm, width]["dense_tensor_calls"]
            == sum(v for k, v in warm[arm, width]["dense_tensor_kernel_counts"])
            and (row["dense_tensor_calls"] > 0) == FLAGS[arm]["dense"]
            and row["timed"] is False,
            "dense census/timing",
        )
    for width in widths:
        require(
            census["dense_only", width]["unchanged_kernel_counts"]
            == census["dense_and_down", width]["unchanged_kernel_counts"],
            "unchanged kernel counts across down arms",
        )


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    k: json.dumps(v, separators=(",", ":"))
                    if isinstance(v, (dict, list))
                    else v
                    for k, v in row.items()
                }
            )


def compare_dense_checkpoint(rows, checkpoint):
    """Compare exported observations, not unseen logits or quality tolerances."""
    names = (
        "extended-tokens.csv",
        "extended-documents.csv",
        "extended-summary.json",
        "evidence.json",
    )
    digests = {
        name: hashlib.sha256((checkpoint / name).read_bytes()).hexdigest()
        for name in names
    }
    previous = json.loads((checkpoint / "extended-summary.json").read_bytes())
    evidence = json.loads((checkpoint / "evidence.json").read_bytes())
    require(
        previous["fixture_sha256"]
        == evidence["fixture_sha256"]
        == rows[0]["fixture_sha256"],
        "dense checkpoint fixture",
    )
    require(
        previous["report_sha256"] == evidence["reports"]["extended-v1.jsonl"],
        "dense checkpoint report hash",
    )
    mapping = {"original": "old", "dense_only": "tensor"}
    with (checkpoint / "extended-tokens.csv").open(newline="") as stream:
        prior_tokens = unique(
            csv.DictReader(stream),
            lambda r: (r["arm"], r["path"], int(r["index"])),
            "checkpoint token",
        )
    tokens = {
        (mapping[r["arm"]], r["path"], r["index"]): r
        for r in events(rows, "document_token")
        if r["arm"] in mapping
    }
    require(
        set(tokens) == set(prior_tokens) and len(tokens) == 2816,
        "dense checkpoint token coverage",
    )
    for key, row in tokens.items():
        prior = prior_tokens[key]
        numeric = (
            "prefix",
            "index",
            "target_position",
            "target_token",
            "argmax",
            "baseline_argmax",
        )
        require(
            all(row[k] == int(prior[k]) for k in numeric)
            and row["stratum"] == prior["stratum"],
            "dense checkpoint token geometry/argmax",
        )
        require(
            row["nll"] == float(prior["nll"])
            and row["references"]["original"]["kl_reference_to_arm"]
            == float(prior["kl_baseline_to_arm"]),
            "dense checkpoint token NLL/KL",
        )
        require(
            all(
                row[k] == (prior[k] == "True")
                for k in ("top1_hit", "top1_agrees_baseline")
            ),
            "dense checkpoint token top1",
        )
    with (checkpoint / "extended-documents.csv").open(newline="") as stream:
        prior_docs = unique(
            csv.DictReader(stream), lambda r: r["path"], "checkpoint document"
        )
    summaries = [r for r in events(rows, "document_arm") if r["arm"] in mapping]
    require(
        len(summaries) == 44 and len(prior_docs) == 22,
        "dense checkpoint document coverage",
    )
    for row in summaries:
        prior, arm = prior_docs[row["path"]], mapping[row["arm"]]
        require(
            row["mean_nll"] == float(prior[f"{arm}_nll"])
            and row["top1_hits"] == int(prior[f"{arm}_hits"]),
            "dense checkpoint document NLL/top1",
        )
        if arm == "tensor":
            ref = row["references"]["original"]
            require(
                ref["mean_nll_delta"] == float(prior["nll_delta"])
                and ref["top1_agreements"] == int(prior["argmax_agreements"]),
                "dense checkpoint paired document metrics",
            )
            require(
                ref["mean_kl_reference_to_arm"] == float(prior["mean_kl"])
                and ref["max_kl_reference_to_arm"] == float(prior["max_kl"]),
                "dense checkpoint document KL",
            )
    return {
        "checkpoint": "2026-10-22-glm-tensor-prefill",
        "arm_mapping": mapping,
        "exact_recorded_metrics_match": True,
        "checked_document_arms": 44,
        "checked_token_rows": 2816,
        "checkpoint_file_sha256": digests,
        "checkpoint_report_sha256": previous["report_sha256"],
        "checkpoint_binary_sha256": evidence["extended_binary_sha256"],
        "scope": "exact recorded token NLL/top1/argmax/alignment and original-reference KL plus document summaries; not raw-logit or model-wide quality identity",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--extended", type=Path)
    parser.add_argument("--fixture", type=Path, default=HERE / "quality-v1.json")
    parser.add_argument(
        "--dense-checkpoint",
        type=Path,
        default=HERE.parents[2] / "docs/bench/2026-10-22-glm-tensor-prefill",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    fixture_bytes = args.fixture.read_bytes()
    fixture, fixture_hash = (
        json.loads(fixture_bytes),
        hashlib.sha256(fixture_bytes).hexdigest(),
    )
    sources = [
        (k, args.archive / n)
        for k, n in (
            ("plain", "plain-v2r.jsonl"),
            ("profile", "stock-profile-v2r.jsonl"),
            ("quality", "quality-v2r.jsonl"),
        )
    ]
    if args.extended:
        sources.append(("extended", args.extended))
    result = {
        "schema": "glm53.tensor_down_descriptive_analysis.v1",
        "descriptive_only": True,
        "arm_flags": FLAGS,
        "fixture_sha256": fixture_hash,
        "analyzer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "quality_helpers_sha256": hashlib.sha256(
            (HERE / "quality_analysis.py").read_bytes()
        ).hexdigest(),
        "numpy_version": np.__version__,
        "bootstrap": {
            "resamples": RESAMPLES,
            "seed": SEED,
            "unit": "document",
            "stratified": True,
            "percentiles": [5, 95],
            "applies_to": "extended only",
            "weights": "equal documents (one maximum-prefix item each); overall strata 8/22,8/22,6/22",
        },
        "extended_status": "included"
        if args.extended
        else "pending: not supplied; no partial report analyzed",
        "reports": {},
    }
    output = {}
    all_census = []
    for kind, path in sources:
        rows, report_hash = read_report(path, kind, fixture, fixture_hash)
        if kind != "profile":
            validate_census(rows, kind)
        meta = metadata(rows, kind, report_hash)
        attach = lambda rs, meta=meta: [meta | r for r in rs]
        if kind in ("quality", "extended"):
            analysis, documents, tools, identities = quality_analysis(
                rows, fixture, kind == "extended"
            )
            output[f"{kind}-documents.csv"] = attach(documents)
            output[f"{kind}-tools.csv"] = attach(tools)
            output[f"{kind}-token-identity.csv"] = attach(identities)
            if kind == "extended":
                analysis["dense_checkpoint_reproduction"] = compare_dense_checkpoint(
                    rows, args.dense_checkpoint
                )
        elif kind == "plain":
            analysis, samples = plain_analysis(rows)
            output["plain-samples.csv"] = attach(samples)
        else:
            analysis, stages = profile_analysis(rows)
            output["profile-stages.csv"] = attach(stages)
        result["reports"][kind] = {
            "metadata": meta,
            "artifact_layout": rows[0]["artifact_layout"],
            "tensor_shape_counts": rows[0].get("tensor_shape_counts"),
            "analysis": analysis,
        }
        all_census.extend(attach(census_rows(rows)))
    output["dispatch-census.csv"] = all_census
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, values in output.items():
        write_csv(args.out_dir / name, values)
    (args.out_dir / "analysis.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        json.dumps(
            {
                "validated_reports": list(result["reports"]),
                "extended_status": result["extended_status"],
                "plain": result["reports"]["plain"]["analysis"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
