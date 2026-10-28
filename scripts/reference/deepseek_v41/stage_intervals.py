"""CPU-only, exact-nanosecond diagnostics for nonexclusive counter intervals.

No global timestamp-index monotonicity is required. Well-formed records do not
establish exclusive attribution, shader-active time, correctness, or performance
acceptance. This analysis never repairs a previously FAILED packet or latch.
Counter-boundary semantics are uncalibrated: dependency anomalies are diagnostic,
not proof of a race or correctness. No fractions or model-gain extrapolation.
"""

from collections.abc import Mapping


U64_MAX = (1 << 64) - 1
CROSS_API_ALLOWANCE_NS = 5_000
FFN_STAGES = ("hc", "router", "shared", "route", "gate_up", "mid", "down", "sum")
FFN_EDGES = (
    ("hc", "end", "router", "start"),
    ("router", "end", "shared", "start"),
    ("router", "end", "route", "start"),
    ("route", "end", "gate_up", "start"),
    ("gate_up", "end", "mid", "start"),
    ("shared", "end", "mid", "end"),
    ("mid", "end", "down", "start"),
    ("down", "end", "sum", "start"),
    ("shared", "end", "sum", "start"),
)


def _sequence(value, label):
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{label} must be a list or tuple")
    return value


def _timestamp(value, label):
    if type(value) is not int or not 0 < value < U64_MAX:
        raise ValueError(f"{label} must be a plain positive u64, excluding sentinels")
    return value


def _pair(value, label):
    if len(_sequence(value, label)) != 2:
        raise ValueError(f"{label} must contain exactly two values")
    return value


def _uncovered(intervals, start, end):
    """Exact set difference [start, end) minus the interval union."""
    result = []
    cursor = start
    for left, right in intervals:
        if right <= cursor:
            continue
        if left >= end:
            break
        if left > cursor:
            result.append([cursor, left])
        cursor = max(cursor, right)
    if cursor < end:
        result.append([cursor, end])
    return result


def _length(intervals):
    return sum(end - start for start, end in intervals)


def analyze_intervals(
    stage_names, timestamps_ns, command_envelope_ns, *, source_indices=None
):
    """Analyze named pairs without reordering or coercing the supplied records.

    timestamps_ns is a flat buffer with exactly 2 * len(stage_names) samples.
    By default each name selects consecutive (start, end) samples. Optional
    source_indices maps EVERY supplied name to its (start_index, end_index);
    indices must be a permutation of 0..2n-1. This explicit association is used
    verbatim, with no stage identity inferred beyond exact canonical FFN names.

    The command envelope is an absolute (start_ns, end_ns) pair. A fixed 5000 ns
    cross-API counter/driver allowance applies ONLY to envelope containment.
    Samples remain unchanged, including allowed excursions outside the envelope.
    Raw adjacent-encoded gaps always mean buffer[2i+2] - buffer[2i+1], not idle
    time or dependency gaps (especially with overlap or an explicit mapping).
    All intervals are half-open; all arithmetic uses exact Python integers.
    overlap_excess_ns is stage_sum_ns - union_ns, not covered_ge2_ns: triple
    coverage contributes twice to excess but only once to time covered >= 2.
    Uncovered command time is exact set difference, not envelope minus stage sum.
    Invalid input raises ValueError; a returned report is diagnostic only.
    """
    names = _sequence(stage_names, "stage_names")
    if not names or any(type(name) is not str or not name for name in names):
        raise ValueError("stage_names must contain nonempty strings")
    if len(set(names)) != len(names):
        raise ValueError("stage_names must be unique")
    samples = _sequence(timestamps_ns, "timestamps_ns")
    if len(samples) != 2 * len(names):
        raise ValueError("timestamps_ns must contain exactly 2 samples per stage")
    for index, sample in enumerate(samples):
        _timestamp(sample, f"timestamps_ns[{index}]")
    command_start, command_end = _pair(command_envelope_ns, "command_envelope_ns")
    _timestamp(command_start, "command start")
    _timestamp(command_end, "command end")
    if command_end <= command_start:
        raise ValueError("command end must be greater than command start")

    if source_indices is None:
        indices = {name: (2 * i, 2 * i + 1) for i, name in enumerate(names)}
    else:
        if not isinstance(source_indices, Mapping) or set(source_indices) != set(names):
            raise ValueError("source_indices must map exactly the supplied stage names")
        indices = {}
        used = set()
        for name in names:
            pair = _pair(source_indices[name], f"source_indices[{name}]")
            for index in pair:
                if type(index) is not int or not 0 <= index < len(samples):
                    raise ValueError("source indices must be plain integers in 0..2n-1")
                if index in used:
                    raise ValueError("source indices must be unique")
                used.add(index)
            indices[name] = tuple(pair)

    stages = []
    for name in names:
        start_index, end_index = indices[name]
        start, end = samples[start_index], samples[end_index]
        if end <= start:
            raise ValueError(f"{name}: end must be greater than start")
        if (
            start < command_start - CROSS_API_ALLOWANCE_NS
            or end > command_end + CROSS_API_ALLOWANCE_NS
        ):
            raise ValueError(f"{name}: outside absolute command envelope plus 5000 ns")
        stages.append(
            {
                "name": name,
                "source_indices": [start_index, end_index],
                "start_ns": start,
                "end_ns": end,
                "duration_ns": end - start,
                "command_start_margin_ns": start - command_start,
                "command_end_margin_ns": command_end - end,
            }
        )

    events = {}
    for stage in stages:
        start, end = stage["start_ns"], stage["end_ns"]
        events[start] = events.get(start, 0) + 1
        events[end] = events.get(end, 0) - 1
    points = sorted(events)
    union = []
    active = covered_ge2 = 0
    for left, right in zip(points, points[1:]):
        active += events[left]
        if active:
            if union and union[-1][1] == left:
                union[-1][1] = right
            else:
                union.append([left, right])
        if active >= 2:
            covered_ge2 += right - left
    hull_start, hull_end = points[0], points[-1]
    hull_uncovered = _uncovered(union, hull_start, hull_end)
    command_uncovered = _uncovered(union, command_start, command_end)
    stage_sum = sum(stage["duration_ns"] for stage in stages)
    union_ns = _length(union)

    by_name = {stage["name"]: stage for stage in stages}
    edges, omitted = [], []
    for before, before_boundary, after, after_boundary in FFN_EDGES:
        edge = {
            "from_stage": before,
            "from_boundary": before_boundary,
            "to_stage": after,
            "to_boundary": after_boundary,
        }
        if before not in by_name or after not in by_name:
            omitted.append(edge)
            continue
        gap = (
            by_name[after][f"{after_boundary}_ns"]
            - by_name[before][f"{before_boundary}_ns"]
        )
        edge.update(gap_ns=gap, anomaly=gap < 0)
        if (before, after) == ("shared", "mid"):
            edge["constraint"] = (
                "shared half read before overwrite: SharedEnd <= MidEnd"
            )
        edges.append(edge)

    return {
        "units": "ns",
        "records_well_formed": True,
        "attribution": "nonexclusive_attribution",
        "acceptance": {
            "exclusive_attribution": False,
            "shader_active_time": False,
            "correctness": False,
            "performance": False,
        },
        "limits": [
            "Diagnostics only; no existing FAILED packet or latch is changed.",
            "Counter-boundary semantics are uncalibrated; anomalies alone do not prove a race or correctness.",
            "Raw adjacent-encoded gaps are not real gaps or idle time when intervals overlap.",
            "No Amdahl fractions, model-gain extrapolation, or performance acceptance.",
        ],
        "command_envelope_ns": [command_start, command_end],
        "command_envelope_duration_ns": command_end - command_start,
        "cross_api_envelope_allowance_ns": CROSS_API_ALLOWANCE_NS,
        "stages": stages,
        "stage_sum_ns": stage_sum,
        "union_intervals_ns": union,
        "union_ns": union_ns,
        "covered_ge2_ns": covered_ge2,
        "overlap_excess_ns": stage_sum - union_ns,
        "counter_hull_ns": [hull_start, hull_end],
        "counter_hull_duration_ns": hull_end - hull_start,
        "uncovered_hull_intervals_ns": hull_uncovered,
        "uncovered_hull_ns": _length(hull_uncovered),
        "largest_uncovered_hull_ns": max((b - a for a, b in hull_uncovered), default=0),
        "uncovered_command_envelope_intervals_ns": command_uncovered,
        "uncovered_command_envelope_ns": _length(command_uncovered),
        "raw_adjacent_encoded_gaps_ns": [
            samples[i + 1] - samples[i] for i in range(1, len(samples) - 1, 2)
        ],
        "dependency_diagnostics": {
            "basis": "source-derived FFN edges; exact canonical identities only, no name inference",
            "cross_api_allowance_applied": False,
            "edges": edges,
            "omitted_edges": omitted,
            "anomaly_count": sum(edge["anomaly"] for edge in edges),
        },
    }
