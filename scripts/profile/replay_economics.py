#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# ///

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ReplayRow:
    path: Path
    start_block: int
    blocks: int
    tokens: int
    context: int | None
    baseline_ms_per_tok: float
    replay_ms_per_tok: float
    gross_save_pct: float
    validated_ms_per_tok: float | None
    fallback_slots_avg: float | None
    net_save_pct: float | None


@dataclass(frozen=True)
class FallbackRow:
    threshold: str
    fallback_pct: float


@dataclass(frozen=True)
class RequestRow:
    request_id: str
    arrival_ms: float
    tokens: int


def format_float(value: float) -> str:
    return f"{value:.4g}"


def parse_replay(path: Path) -> list[ReplayRow]:
    start_block: int | None = None
    blocks: int | None = None
    header: list[str] | None = None
    baseline_by_tokens: dict[int, float] = {}
    rows: list[ReplayRow] = []
    for line in path.read_text().splitlines():
        if line.startswith("[decode-block-slice-replay]"):
            start_match = re.search(r"\bstart_block=(\d+)", line)
            blocks_match = re.search(r"\bblocks=(\d+)", line)
            if start_match is None or blocks_match is None:
                raise SystemExit(f"missing start_block/blocks in {path}")
            start_block = int(start_match.group(1))
            blocks = int(blocks_match.group(1))
            continue
        if not line or line.startswith("check\t"):
            continue
        parts = line.split("\t")
        if parts[0] == "mode":
            header = parts
            continue
        if header is None or len(parts) != len(header):
            continue
        row = dict(zip(header, parts, strict=True))
        tokens = int(row["tokens"])
        ms_per_tok = float(row["avg_gpu_ms_per_tok"])
        if row["mode"] == "baseline_seq":
            baseline_by_tokens[tokens] = ms_per_tok
        elif row["mode"] == "replay_gdn_batched":
            if start_block is None or blocks is None:
                raise SystemExit(f"missing banner before rows in {path}")
            baseline = baseline_by_tokens.get(tokens)
            if baseline is None:
                raise SystemExit(f"missing baseline tokens={tokens} in {path}")
            rows.append(
                ReplayRow(
                    path=path,
                    start_block=start_block,
                    blocks=blocks,
                    tokens=tokens,
                    context=None,
                    baseline_ms_per_tok=baseline,
                    replay_ms_per_tok=ms_per_tok,
                    gross_save_pct=float(row["saving_pct"]),
                    validated_ms_per_tok=None,
                    fallback_slots_avg=None,
                    net_save_pct=None,
                )
            )
    if not rows:
        raise SystemExit(f"no replay rows found in {path}")
    return rows


def parse_real_margin(path: Path) -> list[ReplayRow]:
    header: list[str] | None = None
    rows: list[ReplayRow] = []
    for line in path.read_text().splitlines():
        if not line or line.startswith("["):
            continue
        parts = line.split("\t")
        if parts[0] == "start_block":
            header = parts
            continue
        if header is None or len(parts) != len(header):
            continue
        row = dict(zip(header, parts, strict=True))
        baseline = row.get("baseline_wall_ms_per_tok", "")
        if baseline == "":
            continue
        start_block = int(row["start_block"])
        end_block = int(row["end_block"])
        rows.append(
            ReplayRow(
                path=path,
                start_block=start_block,
                blocks=end_block - start_block,
                tokens=int(row["slots"]),
                context=int(row["context"]),
                baseline_ms_per_tok=float(baseline),
                replay_ms_per_tok=float(row["replay_wall_ms_per_tok"]),
                gross_save_pct=float(row["gross_wall_save_pct"]),
                validated_ms_per_tok=float(row["validated_wall_ms_per_tok"]),
                fallback_slots_avg=float(row["fallback_slots_avg"]),
                net_save_pct=float(row["net_wall_save_pct"]),
            )
        )
    if not rows:
        raise SystemExit(f"no timed real-margin rows found in {path}")
    return rows


def parse_margin_summary(path: Path) -> list[FallbackRow]:
    rows: list[FallbackRow] = []
    header: list[str] | None = None
    for line in path.read_text().splitlines():
        if not line:
            continue
        parts = line.split("\t")
        if parts[0] == "threshold":
            header = parts
            continue
        if header == ["threshold", "fallback_rows", "fallback_pct"]:
            if len(parts) != len(header):
                continue
            row = dict(zip(header, parts, strict=True))
            rows.append(
                FallbackRow(
                    threshold=row["threshold"],
                    fallback_pct=float(row["fallback_pct"]),
                )
            )
    if not rows:
        raise SystemExit(f"no fallback threshold rows found in {path}")
    return rows


def parse_occupancy(raw: str) -> dict[int, float]:
    out: dict[int, float] = {}
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        slot, weight = item.split("=", 1)
        out[int(slot)] = float(weight)
    total = sum(out.values())
    if total <= 0.0:
        raise SystemExit("--occupancy weights must sum above zero")
    return {slot: weight / total for slot, weight in out.items()}


def parse_occupancy_trace(path: Path) -> dict[int, float]:
    counts: dict[int, float] = {}
    lines = path.read_text().splitlines()
    header: list[str] | None = None
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = re.split(r"[\t, ]+", line)
        if header is None and not parts[0].lstrip("+-").isdigit():
            header = parts
            continue
        if header is None:
            active_slots = int(parts[0])
            weight = float(parts[1]) if len(parts) > 1 else 1.0
        else:
            row = dict(zip(header, parts, strict=False))
            active_slots = int(row["active_slots"])
            weight = float(row.get("steps", row.get("weight", "1")))
        if active_slots > 0:
            counts[active_slots] = counts.get(active_slots, 0.0) + weight
    total = sum(counts.values())
    if total <= 0.0:
        raise SystemExit(f"no active occupancy rows found in {path}")
    return {slot: weight / total for slot, weight in sorted(counts.items())}


def read_text_or_stdin(path: str) -> str:
    if path == "-":
        return sys.stdin.read()
    return Path(path).read_text()


def parse_request_trace(path: str) -> list[RequestRow]:
    rows: list[RequestRow] = []
    header: list[str] | None = None
    for line_i, line in enumerate(read_text_or_stdin(path).splitlines()):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = re.split(r"[\t, ]+", line)
        if header is None and not re.match(r"^[0-9.+-]+$", parts[0]):
            header = parts
            continue
        if header is None:
            arrival_ms = float(parts[0])
            tokens = int(parts[1])
            request_id = parts[2] if len(parts) > 2 else f"req{line_i}"
        else:
            row = dict(zip(header, parts, strict=False))
            arrival_ms = float(row["arrival_ms"])
            tokens = int(row["tokens"])
            request_id = row.get("id", f"req{line_i}")
        if tokens > 0:
            rows.append(RequestRow(request_id, arrival_ms, tokens))
    if not rows:
        raise SystemExit(f"no request rows found in {path}")
    rows = sorted(rows, key=lambda row: (row.arrival_ms, row.request_id))
    first_arrival = rows[0].arrival_ms
    return [
        RequestRow(row.request_id, row.arrival_ms - first_arrival, row.tokens)
        for row in rows
    ]


def observed_save(row: ReplayRow) -> float:
    return row.net_save_pct if row.net_save_pct is not None else row.gross_save_pct


def adjusted_save(row: ReplayRow, fallback_pct: float) -> float:
    return observed_save(row) - fallback_pct


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def print_rows(rows: list[ReplayRow], fallback_rows: list[FallbackRow]) -> None:
    print(
        "path\tcontext\tstart_block\tblocks\ttokens\tbaseline_ms_per_tok"
        "\treplay_ms_per_tok\tgross_save_pct\tvalidated_ms_per_tok"
        "\tfallback_slots_avg\tnet_save_pct\tthreshold\tfallback_pct"
        "\tadjusted_net_save_pct"
    )
    scenarios = fallback_rows or [FallbackRow(threshold="observed", fallback_pct=0.0)]
    for row in sorted(rows, key=lambda r: (r.tokens, r.context or -1, r.start_block)):
        for fallback in scenarios:
            print(
                f"{row.path}\t{row.context if row.context is not None else ''}"
                f"\t{row.start_block}\t{row.blocks}\t{row.tokens}"
                f"\t{format_float(row.baseline_ms_per_tok)}"
                f"\t{format_float(row.replay_ms_per_tok)}"
                f"\t{row.gross_save_pct:.2f}"
                f"\t{format_float(row.validated_ms_per_tok or 0.0)}"
                f"\t{format_float(row.fallback_slots_avg or 0.0)}"
                f"\t{observed_save(row):.2f}\t{fallback.threshold}"
                f"\t{fallback.fallback_pct:.2f}"
                f"\t{adjusted_save(row, fallback.fallback_pct):.2f}"
            )


def print_policy(
    rows: list[ReplayRow], fallback_rows: list[FallbackRow], occupancy: str
) -> None:
    weights = parse_occupancy(occupancy)
    print_policy_for_weights(rows, fallback_rows, weights)


def print_policy_for_weights(
    rows: list[ReplayRow], fallback_rows: list[FallbackRow], weights: dict[int, float]
) -> None:
    by_tokens: dict[int, list[ReplayRow]] = {}
    for row in rows:
        by_tokens.setdefault(row.tokens, []).append(row)

    max_slot = max(weights)
    min_slot_candidates = sorted(set(weights) | set(by_tokens))
    scenarios = fallback_rows or [FallbackRow(threshold="observed", fallback_pct=0.0)]
    print(
        "policy_min_slots\tthreshold\tworkload_weight_replayed"
        "\tbaseline_ms_per_tok\tcharged_ms_per_tok\tblended_save_pct"
    )
    for min_slots in min_slot_candidates:
        if min_slots > max_slot:
            continue
        for fallback in scenarios:
            baseline_total = 0.0
            charged_total = 0.0
            replayed_weight = 0.0
            for slots, weight in weights.items():
                slot_rows = by_tokens.get(slots)
                if not slot_rows:
                    continue
                baseline = mean([row.baseline_ms_per_tok for row in slot_rows])
                save = 0.0
                if slots >= min_slots:
                    replayed_weight += weight
                    save = mean(
                        [adjusted_save(row, fallback.fallback_pct) for row in slot_rows]
                    )
                baseline_total += weight * baseline
                charged_total += weight * baseline * (1.0 - save / 100.0)
            if baseline_total <= 0.0:
                continue
            blended = (baseline_total - charged_total) / baseline_total * 100.0
            print(
                f"{min_slots}\t{fallback.threshold}\t{replayed_weight:.4f}"
                f"\t{baseline_total:.4f}\t{charged_total:.4f}\t{blended:.2f}"
            )


def percentile(xs: list[float], q: float) -> float:
    if not xs:
        return 0.0
    ys = sorted(xs)
    idx = min(len(ys) - 1, max(0, int(round((len(ys) - 1) * q))))
    return ys[idx]


def slot_cost(
    values: dict[int, float], slots: int, allow_interpolate: bool, label: str
) -> float:
    if slots in values:
        return values[slots]
    if not allow_interpolate:
        available = ",".join(str(k) for k in sorted(values))
        raise SystemExit(
            f"missing {label} cost for active_slots={slots}; available={available}; "
            "rerun rows for that slot count or pass --interpolate-slots"
        )
    keys = sorted(values)
    lower = max((k for k in keys if k < slots), default=keys[0])
    upper = min((k for k in keys if k > slots), default=keys[-1])
    if lower == upper:
        return values[lower]
    t = (slots - lower) / (upper - lower)
    return values[lower] * (1.0 - t) + values[upper] * t


def mean_costs(
    rows: list[ReplayRow], fallback_pct: float
) -> tuple[dict[int, float], dict[int, float]]:
    by_tokens: dict[int, list[ReplayRow]] = {}
    for row in rows:
        by_tokens.setdefault(row.tokens, []).append(row)
    baseline: dict[int, float] = {}
    charged: dict[int, float] = {}
    for tokens, token_rows in by_tokens.items():
        base = mean([row.baseline_ms_per_tok for row in token_rows])
        save = mean([adjusted_save(row, fallback_pct) for row in token_rows])
        baseline[tokens] = base
        charged[tokens] = base * (1.0 - save / 100.0)
    return baseline, charged


def simulate_requests(
    requests: list[RequestRow],
    baseline_costs: dict[int, float],
    charged_costs: dict[int, float],
    capacity: int,
    policy_min_slots: int,
    allow_interpolate: bool,
) -> tuple[float, float, float, float, float, dict[int, int]]:
    pending = list(requests)
    waiting: list[dict[str, float | int | str]] = []
    active: list[dict[str, float | int | str]] = []
    done_latencies: list[float] = []
    occupancy_counts: dict[int, int] = {}
    time_ms = 0.0

    while pending or waiting or active:
        while pending and pending[0].arrival_ms <= time_ms:
            req = pending.pop(0)
            waiting.append(
                {
                    "id": req.request_id,
                    "arrival_ms": req.arrival_ms,
                    "remaining": req.tokens,
                }
            )
        while waiting and len(active) < capacity:
            active.append(waiting.pop(0))
        if not active:
            time_ms = max(time_ms, pending[0].arrival_ms)
            continue

        slots = len(active)
        occupancy_counts[slots] = occupancy_counts.get(slots, 0) + 1
        baseline_per_tok = slot_cost(
            baseline_costs, slots, allow_interpolate, "baseline"
        )
        charged_per_tok = (
            slot_cost(charged_costs, slots, allow_interpolate, "charged")
            if slots >= policy_min_slots
            else baseline_per_tok
        )
        time_ms += charged_per_tok * slots
        next_active: list[dict[str, float | int | str]] = []
        for req in active:
            remaining = int(req["remaining"]) - 1
            if remaining <= 0:
                done_latencies.append(time_ms - float(req["arrival_ms"]))
            else:
                req["remaining"] = remaining
                next_active.append(req)
        active = next_active

    token_count = sum(req.tokens for req in requests)
    throughput = token_count / (time_ms / 1000.0) if time_ms > 0.0 else 0.0
    return (
        time_ms,
        throughput,
        percentile(done_latencies, 0.50),
        percentile(done_latencies, 0.95),
        max(done_latencies, default=0.0),
        occupancy_counts,
    )


def occupancy_replay_shares(
    occupancy: dict[int, int], policy_min_slots: int
) -> tuple[float, float]:
    total_steps = sum(occupancy.values())
    total_token_steps = sum(slots * steps for slots, steps in occupancy.items())
    replay_steps = sum(
        steps for slots, steps in occupancy.items() if slots >= policy_min_slots
    )
    replay_token_steps = sum(
        slots * steps for slots, steps in occupancy.items() if slots >= policy_min_slots
    )
    step_share = replay_steps / total_steps * 100.0 if total_steps else 0.0
    token_share = (
        replay_token_steps / total_token_steps * 100.0 if total_token_steps else 0.0
    )
    return step_share, token_share


def print_request_sim(
    rows: list[ReplayRow],
    fallback_rows: list[FallbackRow],
    request_trace: str,
    capacity: int,
    policy_min_slots: int,
    allow_interpolate: bool,
) -> None:
    requests = parse_request_trace(request_trace)
    scenarios = fallback_rows or [FallbackRow(threshold="observed", fallback_pct=0.0)]
    print(
        "request_trace\tthreshold\trequests\ttokens\tcapacity\tpolicy_min_slots"
        "\tbaseline_wall_ms\tpolicy_wall_ms\tblended_save_pct"
        "\tbaseline_tps\tpolicy_tps\tbaseline_p95_ms\tpolicy_p95_ms"
        "\tp95_delta_pct\treplayed_step_pct\treplayed_token_pct\tpolicy_occupancy"
    )
    for fallback in scenarios:
        baseline_costs, charged_costs = mean_costs(rows, fallback.fallback_pct)
        base_wall, base_tps, _, base_p95, _, _ = simulate_requests(
            requests,
            baseline_costs,
            baseline_costs,
            capacity,
            capacity + 1,
            allow_interpolate,
        )
        policy_wall, policy_tps, _, policy_p95, _, occupancy = simulate_requests(
            requests,
            baseline_costs,
            charged_costs,
            capacity,
            policy_min_slots,
            allow_interpolate,
        )
        save = (base_wall - policy_wall) / base_wall * 100.0 if base_wall > 0 else 0.0
        p95_delta = (policy_p95 - base_p95) / base_p95 * 100.0 if base_p95 > 0 else 0.0
        replayed_step_pct, replayed_token_pct = occupancy_replay_shares(
            occupancy, policy_min_slots
        )
        occupancy_str = ",".join(f"{k}:{v}" for k, v in sorted(occupancy.items()))
        print(
            f"{request_trace}\t{fallback.threshold}\t{len(requests)}"
            f"\t{sum(req.tokens for req in requests)}\t{capacity}\t{policy_min_slots}"
            f"\t{base_wall:.4f}\t{policy_wall:.4f}\t{save:.2f}"
            f"\t{base_tps:.2f}\t{policy_tps:.2f}\t{base_p95:.4f}"
            f"\t{policy_p95:.4f}\t{p95_delta:.2f}\t{replayed_step_pct:.2f}"
            f"\t{replayed_token_pct:.2f}\t{occupancy_str}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Model block-slice replay net savings after validation and fallback."
    )
    parser.add_argument("--margin-summary", type=Path)
    parser.add_argument("--replay", type=Path, action="append", default=[])
    parser.add_argument("--real-margin", type=Path, action="append", default=[])
    parser.add_argument(
        "--occupancy",
        help="optional token-step occupancy mix, for example '4=0.2,6=0.3,8=0.5'",
    )
    parser.add_argument(
        "--occupancy-trace",
        type=Path,
        help="optional trace with active_slots and optional weight/steps columns",
    )
    parser.add_argument(
        "--request-trace",
        help="optional request trace: arrival_ms,tokens[,id], or '-' for stdin",
    )
    parser.add_argument("--capacity", type=int, default=8)
    parser.add_argument("--policy-min-slots", type=int, default=8)
    parser.add_argument(
        "--interpolate-slots",
        action="store_true",
        help="linearly interpolate missing active-slot costs in request simulation",
    )
    args = parser.parse_args()

    fallback_rows = (
        parse_margin_summary(args.margin_summary) if args.margin_summary else []
    )
    replay_rows: list[ReplayRow] = []
    for path in args.replay:
        replay_rows.extend(parse_replay(path))
    for path in args.real_margin:
        replay_rows.extend(parse_real_margin(path))
    if not replay_rows:
        raise SystemExit("pass at least one --replay or --real-margin path")

    print_rows(replay_rows, fallback_rows)
    if args.occupancy:
        print_policy(replay_rows, fallback_rows, args.occupancy)
    if args.occupancy_trace:
        print_policy_for_weights(
            replay_rows, fallback_rows, parse_occupancy_trace(args.occupancy_trace)
        )
    if args.request_trace:
        print_request_sim(
            replay_rows,
            fallback_rows,
            args.request_trace,
            args.capacity,
            args.policy_min_slots,
            args.interpolate_slots,
        )


if __name__ == "__main__":
    main()
