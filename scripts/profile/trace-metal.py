#!/usr/bin/env python3
import argparse
import json
import math
import os
import re
import statistics as st
import subprocess
import sys
import xml.etree.ElementTree as ET


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize Metal System Trace for qwen-llm")
    p.add_argument("trace", help="Path to .trace bundle")
    p.add_argument("--process-prefix", default="qwen-bench", help="Process name prefix")
    p.add_argument("--json", action="store_true", help="Emit JSON instead of text")
    p.add_argument(
        "--include-intervals",
        action="store_true",
        help="Include per-command-buffer compute intervals in JSON output",
    )
    return p.parse_args()


def export_table(trace: str, schema: str) -> ET.Element:
    xml = subprocess.check_output(
        [
            "xcrun",
            "xctrace",
            "export",
            "--input",
            trace,
            "--xpath",
            f'/trace-toc/run[@number="1"]/data/table[@schema="{schema}"]',
        ],
        stderr=subprocess.DEVNULL,
    )
    return ET.fromstring(xml)


def export_table_optional(trace: str, schema: str) -> tuple[ET.Element | None, str]:
    result = subprocess.run(
        [
            "xcrun",
            "xctrace",
            "export",
            "--input",
            trace,
            "--xpath",
            f'/trace-toc/run[@number="1"]/data/table[@schema="{schema}"]',
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        return None, detail or f"xctrace exited {result.returncode}"
    try:
        return ET.fromstring(result.stdout), "available"
    except ET.ParseError as error:
        raise RuntimeError(f"malformed {schema} XML: {error}") from error


def build_refs(root: ET.Element) -> dict[str, str]:
    refs: dict[str, str] = {}
    for el in root.iter():
        if "id" in el.attrib:
            refs[el.attrib["id"]] = el.attrib.get("fmt") or (
                el.text.strip() if el.text else ""
            )
    return refs


def parse_ms(raw: str, fmt: str) -> float | None:
    txt = (raw or fmt or "").strip()
    if not txt:
        return None
    if txt.isdigit():
        return int(txt) / 1e6
    m = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*(ms|µs|s)?", txt)
    if not m:
        return None
    val = float(m.group(1))
    unit = m.group(2) or "ms"
    if unit == "µs":
        return val / 1000.0
    if unit == "s":
        return val * 1000.0
    return val


def rows(root: ET.Element) -> list[dict[str, tuple[str, str]]]:
    refs = build_refs(root)
    schema = root.find(".//schema")
    if schema is None:
        raise ValueError("exported table has no schema")
    columns = [col.findtext("mnemonic") or "" for col in schema.findall("col")]
    if not columns or any(not column for column in columns):
        raise ValueError("exported table has empty schema mnemonics")
    if len(columns) != len(set(columns)):
        raise ValueError("exported table has duplicate schema mnemonics")
    out: list[dict[str, tuple[str, str]]] = []
    for row in root.iter("row"):
        if len(row) != len(columns):
            raise ValueError(
                f"exported row width {len(row)} does not match schema width {len(columns)}"
            )
        item: dict[str, tuple[str, str]] = {}
        for index, child in enumerate(row):
            ref = child.attrib.get("ref") if "ref" in child.attrib else None
            fmt = refs.get(ref, "") if ref is not None else ""
            if not fmt:
                fmt = child.attrib.get("fmt") or (
                    child.text.strip() if child.text else ""
                )
            raw = child.text.strip() if child.text else ""
            key = columns[index] if index < len(columns) else child.tag
            item[key] = (fmt, raw)
        out.append(item)
    return out


def get_fmt(item: dict[str, tuple[str, str]], key: str) -> str:
    return item.get(key, ("", ""))[0]


def get_raw(item: dict[str, tuple[str, str]], key: str) -> str:
    return item.get(key, ("", ""))[1]


def pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    if not 0.0 <= q <= 1.0:
        raise ValueError(f"percentile must be in [0, 1], got {q}")
    ordered = sorted(values)
    rank = max(1, math.ceil(q * len(ordered)))
    return ordered[rank - 1]


def merge_ranges(ranges: list[tuple[float, float]]) -> list[tuple[float, float]]:
    merged: list[tuple[float, float]] = []
    for start, end in sorted(ranges):
        if end < start:
            raise ValueError(f"invalid interval {start}..{end}")
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            previous_start, previous_end = merged[-1]
            merged[-1] = (previous_start, max(previous_end, end))
    return merged


def range_gaps(ranges: list[tuple[float, float]]) -> list[float]:
    return [ranges[index][0] - ranges[index - 1][1] for index in range(1, len(ranges))]


def summarize(trace: str, process_prefix: str, include_intervals: bool = False) -> dict:
    sub_root = export_table(trace, "metal-application-command-buffer-submissions")
    sub_rows = rows(sub_root)
    submissions = []
    submission_cb_ids = set()
    for item in sub_rows:
        proc = get_fmt(item, "process")
        event = get_fmt(item, "event-type")
        if proc.startswith(process_prefix) and event == "CommandBufferSubmission":
            start_ms = parse_ms(get_raw(item, "start"), get_fmt(item, "start"))
            dur_ms = parse_ms(get_raw(item, "duration"), get_fmt(item, "duration"))
            cb = get_fmt(item, "cmdbuffer-id")
            if start_ms is not None and dur_ms is not None and cb:
                submissions.append((start_ms, dur_ms, cb))
                submission_cb_ids.add(cb)
    submissions.sort()

    enc_root = export_table(trace, "metal-application-encoders-list")
    enc_rows = rows(enc_root)
    encoder_count = 0
    encoder_cb_ids = set()
    encoder_labels: dict[str, dict[str, float | int]] = {}
    qwen_prefill_phase_counts: dict[str, int] = {}
    for item in enc_rows:
        proc = get_fmt(item, "process")
        event = get_fmt(item, "event-type")
        if proc.startswith(process_prefix) and event == "Encoding":
            encoder_count += 1
            cb = get_fmt(item, "cmdbuffer-id")
            if cb:
                encoder_cb_ids.add(cb)
            label = get_fmt(item, "encoder-label-indexed") or "<unlabeled>"
            dur_ms = (
                parse_ms(get_raw(item, "duration"), get_fmt(item, "duration")) or 0.0
            )
            stats = encoder_labels.setdefault(label, {"count": 0, "encode_ms": 0.0})
            stats["count"] = int(stats["count"]) + 1
            stats["encode_ms"] = float(stats["encode_ms"]) + dur_ms
            for match in re.finditer(r"qwen-prefill-l(\d+)-([A-Za-z0-9-]+)", label):
                phase = match.group(2)
                qwen_prefill_phase_counts[phase] = (
                    qwen_prefill_phase_counts.get(phase, 0) + 1
                )

    comp_root, completion_table_status = export_table_optional(
        trace, "metal-command-buffer-completed"
    )
    comp_rows = rows(comp_root) if comp_root is not None else []
    completed: dict[str, float] = {}
    for item in comp_rows:
        cb = get_fmt(item, "cmdbuffer-id")
        if cb in submission_cb_ids:
            t_ms = parse_ms(get_raw(item, "timestamp"), get_fmt(item, "timestamp"))
            if t_ms is not None:
                completed[cb] = t_ms

    gpu_root = export_table(trace, "metal-gpu-intervals")
    gpu_rows = rows(gpu_root)
    intervals = []
    process_intervals = []
    for item in gpu_rows:
        proc = get_fmt(item, "process")
        channel = get_fmt(item, "channel-name")
        cb = get_fmt(item, "cmdbuffer-id")
        if proc.startswith(process_prefix) and channel == "Compute":
            start_ms = parse_ms(get_raw(item, "start"), get_fmt(item, "start"))
            dur_ms = parse_ms(get_raw(item, "duration"), get_fmt(item, "duration"))
            if start_ms is not None and dur_ms is not None:
                process_intervals.append((start_ms, dur_ms))
                if cb in encoder_cb_ids:
                    intervals.append((start_ms, dur_ms, cb))
    intervals.sort()
    process_intervals.sort()

    per_cb_compute = []
    by_cb: dict[str, list[tuple[float, float]]] = {}
    for start_ms, dur_ms, cb in intervals:
        by_cb.setdefault(cb, []).append((start_ms, dur_ms))
    for cb, spans in by_cb.items():
        start = min(s for s, _ in spans)
        end = max(s + d for s, d in spans)
        per_cb_compute.append((start, end - start, cb, len(spans)))
    per_cb_compute.sort()

    encode_ms = [dur for _, dur, _ in submissions]
    submit_start_ms = [start for start, _, _ in submissions]
    submit_gaps_ms = [
        submit_start_ms[i] - submit_start_ms[i - 1]
        for i in range(1, len(submit_start_ms))
    ]

    completion_to_next_submit_ms = []
    for i in range(1, len(submissions)):
        prev_cb = submissions[i - 1][2]
        next_submit = submissions[i][0]
        prev_complete = completed.get(prev_cb)
        if prev_complete is not None:
            completion_to_next_submit_ms.append(max(0.0, next_submit - prev_complete))

    compute_spans_ms = [dur for _, dur, _, _ in per_cb_compute]
    compute_counts = [count for _, _, _, count in per_cb_compute]
    compute_busy_ranges = merge_ranges(
        [(start, start + duration) for start, duration, _, _ in per_cb_compute]
    )
    process_busy_ranges = merge_ranges(
        [(start, start + duration) for start, duration in process_intervals]
    )
    compute_gap_ms = range_gaps(compute_busy_ranges)
    process_gap_ms = range_gaps(process_busy_ranges)
    process_compute_work_ms = sum(duration for _, duration in process_intervals)
    process_compute_busy_ms = sum(end - start for start, end in process_busy_ranges)
    short_process_gap_ms = [g for g in process_gap_ms if g <= 10.0]
    long_process_gap_ms = [g for g in process_gap_ms if g > 10.0]
    interval_histogram: dict[int, int] = {}
    for count in compute_counts:
        interval_histogram[count] = interval_histogram.get(count, 0) + 1

    shader_root, shader_table_status = export_table_optional(
        trace, "metal-shader-profiler-intervals"
    )
    shader_rows = rows(shader_root) if shader_root is not None else []
    shader_samples: dict[str, list[float]] = {}
    for item in shader_rows:
        proc = get_fmt(item, "process")
        if not proc.startswith(process_prefix):
            continue
        name = get_fmt(item, "name")
        duration_ms = parse_ms(get_raw(item, "duration"), get_fmt(item, "duration"))
        if not name or duration_ms is None:
            continue
        family = re.sub(r" \(\d+\)$", "", name)
        shader_samples.setdefault(family, []).append(duration_ms)
    shader_profile = [
        {
            "family": family,
            "sample_count": len(samples),
            "sample_duration_ms": sum(samples),
            "sample_median_ms": st.median(samples),
            "sample_p95_ms": pct(samples, 0.95),
            "sample_min_ms": min(samples),
            "sample_max_ms": max(samples),
        }
        for family, samples in sorted(
            shader_samples.items(),
            key=lambda item: sum(item[1]),
            reverse=True,
        )
    ]

    result = {
        "trace": os.path.abspath(trace),
        "process_prefix": process_prefix,
        "encoders": encoder_count,
        "command_buffers": len(submission_cb_ids),
        "matched_completions": len(completed),
        "completion_table_status": completion_table_status,
        "encode_median_ms": st.median(encode_ms) if encode_ms else None,
        "encode_p95_ms": pct(encode_ms, 0.95),
        "submission_gap_median_ms": st.median(submit_gaps_ms)
        if submit_gaps_ms
        else None,
        "submission_gap_p95_ms": pct(submit_gaps_ms, 0.95),
        "completion_to_next_submit_median_ms": st.median(completion_to_next_submit_ms)
        if completion_to_next_submit_ms
        else None,
        "completion_to_next_submit_p95_ms": pct(completion_to_next_submit_ms, 0.95),
        "compute_intervals": len(intervals),
        "compute_cb_count": len(per_cb_compute),
        "compute_cb_span_median_ms": st.median(compute_spans_ms)
        if compute_spans_ms
        else None,
        "compute_cb_span_p95_ms": pct(compute_spans_ms, 0.95),
        "compute_intervals_per_cb_median": st.median(compute_counts)
        if compute_counts
        else None,
        "compute_intervals_per_cb_max": max(compute_counts) if compute_counts else None,
        "compute_cb_gap_median_ms": st.median(compute_gap_ms)
        if compute_gap_ms
        else None,
        "compute_cb_gap_p95_ms": pct(compute_gap_ms, 0.95),
        "process_compute_intervals": len(process_intervals),
        "process_compute_total_ms": process_compute_work_ms,
        "process_compute_busy_ms": process_compute_busy_ms,
        "process_compute_overlap_ms": max(
            0.0, process_compute_work_ms - process_compute_busy_ms
        ),
        "process_compute_gap_median_ms": st.median(process_gap_ms)
        if process_gap_ms
        else None,
        "process_compute_gap_p95_ms": pct(process_gap_ms, 0.95),
        "process_compute_gap_total_ms": sum(process_gap_ms),
        "process_compute_gap_short_count": len(short_process_gap_ms),
        "process_compute_gap_short_total_ms": sum(short_process_gap_ms),
        "process_compute_gap_long_count": len(long_process_gap_ms),
        "process_compute_gap_long_total_ms": sum(long_process_gap_ms),
        "compute_intervals_per_cb_histogram": interval_histogram,
        "encoder_labels": encoder_labels,
        "qwen_prefill_phase_counts": qwen_prefill_phase_counts,
        "shader_profile": shader_profile,
        "shader_profile_sample_duration_ms": sum(
            float(item["sample_duration_ms"]) for item in shader_profile
        ),
        "shader_profile_scope": "sampled_intervals_only_not_whole_graph",
        "shader_profile_table_status": shader_table_status,
    }
    if include_intervals:
        first_start_ms = per_cb_compute[0][0] if per_cb_compute else 0.0
        interval_details = []
        previous_end_ms = None
        for start_ms, duration_ms, cb, count in per_cb_compute:
            gap_ms = (
                max(0.0, start_ms - previous_end_ms)
                if previous_end_ms is not None
                else None
            )
            interval_details.append(
                {
                    "command_buffer": cb,
                    "start_ms": start_ms - first_start_ms,
                    "duration_ms": duration_ms,
                    "gap_ms": gap_ms,
                    "interval_count": count,
                }
            )
            end_ms = start_ms + duration_ms
            previous_end_ms = (
                max(previous_end_ms, end_ms) if previous_end_ms is not None else end_ms
            )
        result["compute_interval_details"] = interval_details
    return result


def main() -> int:
    args = parse_args()
    data = summarize(args.trace, args.process_prefix, args.include_intervals)
    if args.json:
        json.dump(data, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
        return 0

    print(f"trace: {data['trace']}")
    print(f"process_prefix: {data['process_prefix']}")
    print(
        f"encoders: {data['encoders']}  command_buffers: {data['command_buffers']}  matched_completions: {data['matched_completions']}"
    )
    print(
        f"encode: median {data['encode_median_ms']:.3f} ms  p95 {data['encode_p95_ms']:.3f} ms"
        if data["encode_median_ms"] is not None and data["encode_p95_ms"] is not None
        else "encode: n/a"
    )
    print(
        f"submission cadence: median {data['submission_gap_median_ms']:.3f} ms  p95 {data['submission_gap_p95_ms']:.3f} ms"
        if data["submission_gap_median_ms"] is not None
        and data["submission_gap_p95_ms"] is not None
        else "submission cadence: n/a"
    )
    print(
        f"complete->next submit: median {data['completion_to_next_submit_median_ms']:.3f} ms  p95 {data['completion_to_next_submit_p95_ms']:.3f} ms"
        if data["completion_to_next_submit_median_ms"] is not None
        and data["completion_to_next_submit_p95_ms"] is not None
        else "complete->next submit: n/a"
    )
    print(
        f"compute intervals: {data['compute_intervals']}  cb_spans: {data['compute_cb_count']}  median span {data['compute_cb_span_median_ms']:.3f} ms  p95 span {data['compute_cb_span_p95_ms']:.3f} ms"
        if data["compute_cb_span_median_ms"] is not None
        and data["compute_cb_span_p95_ms"] is not None
        else "compute intervals: n/a"
    )
    print(
        f"compute gaps: median {data['compute_cb_gap_median_ms']:.3f} ms  p95 {data['compute_cb_gap_p95_ms']:.3f} ms"
        if data["compute_cb_gap_median_ms"] is not None
        and data["compute_cb_gap_p95_ms"] is not None
        else "compute gaps: n/a"
    )
    print(
        f"process compute intervals: {data['process_compute_intervals']}  median gap {data['process_compute_gap_median_ms']:.3f} ms  p95 gap {data['process_compute_gap_p95_ms']:.3f} ms"
        if data["process_compute_gap_median_ms"] is not None
        and data["process_compute_gap_p95_ms"] is not None
        else "process compute intervals: n/a"
    )
    if data["process_compute_intervals"]:
        print(
            f"process compute work: {data['process_compute_total_ms']:.3f} ms  busy union: {data['process_compute_busy_ms']:.3f} ms  overlap: {data['process_compute_overlap_ms']:.3f} ms  gap total: {data['process_compute_gap_total_ms']:.3f} ms"
        )
        print(
            f"process gap split: <=10ms count {data['process_compute_gap_short_count']} total {data['process_compute_gap_short_total_ms']:.3f} ms | >10ms count {data['process_compute_gap_long_count']} total {data['process_compute_gap_long_total_ms']:.3f} ms"
        )
    if data["compute_intervals_per_cb_histogram"]:
        hist = ", ".join(
            f"{k}:{v}"
            for k, v in sorted(data["compute_intervals_per_cb_histogram"].items())
        )
        print(f"compute intervals per cb histogram: {hist}")
    if data["encoder_labels"]:
        print("encoder labels by CPU encode time:")
        for label, stats in sorted(
            data["encoder_labels"].items(),
            key=lambda kv: float(kv[1]["encode_ms"]),
            reverse=True,
        )[:20]:
            print(
                f"  {label}: count {int(stats['count'])} encode_ms {float(stats['encode_ms']):.3f}"
            )
    if data["qwen_prefill_phase_counts"]:
        counts = ", ".join(
            f"{phase}:{count}"
            for phase, count in sorted(data["qwen_prefill_phase_counts"].items())
        )
        print(f"qwen prefill phase label counts: {counts}")
    if data["shader_profile"]:
        print("shader profile (sampled intervals; top 20):")
        for item in data["shader_profile"][:20]:
            print(
                f"  {item['family']}: {item['sample_duration_ms']:.3f} ms "
                f"across {item['sample_count']} samples; median "
                f"{item['sample_median_ms']:.3f} ms p95 {item['sample_p95_ms']:.3f} ms"
            )
        print("  sampled duration sum is profiler coverage, not whole-graph GPU time")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
