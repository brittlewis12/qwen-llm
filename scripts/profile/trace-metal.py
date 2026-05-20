#!/usr/bin/env python3
import argparse
import json
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
    out: list[dict[str, tuple[str, str]]] = []
    for row in root.iter("row"):
        item: dict[str, tuple[str, str]] = {}
        for child in row:
            ref = child.attrib.get("ref") if "ref" in child.attrib else None
            fmt = refs.get(ref, "") if ref is not None else ""
            if not fmt:
                fmt = child.attrib.get("fmt") or (
                    child.text.strip() if child.text else ""
                )
            raw = child.text.strip() if child.text else ""
            item[child.tag] = (fmt, raw)
        out.append(item)
    return out


def get_fmt(item: dict[str, tuple[str, str]], key: str) -> str:
    return item.get(key, ("", ""))[0]


def get_raw(item: dict[str, tuple[str, str]], key: str) -> str:
    return item.get(key, ("", ""))[1]


def pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    n = max(2, int(round(1.0 / (1.0 - q))))
    idx = min(len(st.quantiles(values, n=n)) - 1, n - 2)
    return st.quantiles(values, n=n)[idx]


def summarize(trace: str, process_prefix: str) -> dict:
    sub_root = export_table(trace, "metal-application-command-buffer-submissions")
    sub_rows = rows(sub_root)
    submissions = []
    submission_cb_ids = set()
    for item in sub_rows:
        proc = get_fmt(item, "process")
        event = get_fmt(item, "metal-event-name")
        if proc.startswith(process_prefix) and event == "CommandBufferSubmission":
            start_ms = parse_ms(
                get_raw(item, "start-time"), get_fmt(item, "start-time")
            )
            dur_ms = parse_ms(get_raw(item, "duration"), get_fmt(item, "duration"))
            cb = get_fmt(item, "metal-command-buffer-id")
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
        event = get_fmt(item, "metal-event-name")
        if proc.startswith(process_prefix) and event == "Encoding":
            encoder_count += 1
            cb = get_fmt(item, "metal-command-buffer-id")
            if cb:
                encoder_cb_ids.add(cb)
            label = get_fmt(item, "metal-object-label") or "<unlabeled>"
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

    comp_root = export_table(trace, "metal-command-buffer-completed")
    comp_rows = rows(comp_root)
    completed: dict[str, float] = {}
    for item in comp_rows:
        cb = get_fmt(item, "metal-command-buffer-id")
        if cb in submission_cb_ids:
            t_ms = parse_ms(get_raw(item, "start-time"), get_fmt(item, "start-time"))
            if t_ms is not None:
                completed[cb] = t_ms

    gpu_root = export_table(trace, "metal-gpu-intervals")
    gpu_rows = rows(gpu_root)
    intervals = []
    process_intervals = []
    for item in gpu_rows:
        proc = get_fmt(item, "process")
        channel = get_fmt(item, "gpu-channel-name")
        cb = get_fmt(item, "metal-command-buffer-id")
        if proc.startswith(process_prefix) and channel == "Compute":
            start_ms = parse_ms(
                get_raw(item, "start-time"), get_fmt(item, "start-time")
            )
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
    compute_gap_ms = [
        max(
            0.0,
            per_cb_compute[i][0]
            - (per_cb_compute[i - 1][0] + per_cb_compute[i - 1][1]),
        )
        for i in range(1, len(per_cb_compute))
    ]
    process_gap_ms = [
        max(
            0.0,
            process_intervals[i][0]
            - (process_intervals[i - 1][0] + process_intervals[i - 1][1]),
        )
        for i in range(1, len(process_intervals))
    ]
    short_process_gap_ms = [g for g in process_gap_ms if g <= 10.0]
    long_process_gap_ms = [g for g in process_gap_ms if g > 10.0]
    interval_histogram: dict[int, int] = {}
    for count in compute_counts:
        interval_histogram[count] = interval_histogram.get(count, 0) + 1

    return {
        "trace": os.path.abspath(trace),
        "process_prefix": process_prefix,
        "encoders": encoder_count,
        "command_buffers": len(submission_cb_ids),
        "matched_completions": len(completed),
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
        "process_compute_total_ms": sum(d for _, d in process_intervals),
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
    }


def main() -> int:
    args = parse_args()
    data = summarize(args.trace, args.process_prefix)
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
            f"process compute total: {data['process_compute_total_ms']:.3f} ms  gap total: {data['process_compute_gap_total_ms']:.3f} ms"
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
