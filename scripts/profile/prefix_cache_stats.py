#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# ///

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CacheRow:
    path: Path
    request_id: str
    prompt_tokens: int
    generated_tokens: int
    cache_prefix_tokens: int | None
    cache_hit: bool
    matched_prefix_tokens: int
    exact_cache_hit: bool
    restore_ms: float
    prefix_inserted_bytes: int
    prefix_insert_ms: float
    prefill_ms: float
    decode_ms: float
    model_ttft_ms: float
    first_decode_ms: float
    total_ms: float
    cache_entries: int
    cache_bytes: int
    cache_max_bytes: int


def as_int(row: dict[str, Any], key: str, default: int = 0) -> int:
    value = row.get(key, default)
    if value is None:
        return default
    return int(value)


def as_float(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    if value is None:
        return default
    return float(value)


def parse_stats(path: Path) -> list[CacheRow]:
    rows: list[CacheRow] = []
    for line_no, line in enumerate(path.read_text().splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        raw = json.loads(stripped)
        ttft = raw.get("model_ttft_ms", raw.get("ttft_ms"))
        if ttft is None:
            raise SystemExit(f"{path}:{line_no}: missing model_ttft_ms")
        prefix = raw.get("cache_prefix_tokens")
        rows.append(
            CacheRow(
                path=path,
                request_id=str(raw.get("id", f"line-{line_no}")),
                prompt_tokens=as_int(raw, "prompt_tokens"),
                generated_tokens=as_int(raw, "generated_tokens"),
                cache_prefix_tokens=None if prefix is None else int(prefix),
                cache_hit=bool(raw.get("cache_hit", False)),
                matched_prefix_tokens=as_int(raw, "matched_prefix_tokens"),
                exact_cache_hit=bool(raw.get("exact_cache_hit", False)),
                restore_ms=as_float(raw, "restore_ms"),
                prefix_inserted_bytes=as_int(raw, "prefix_inserted_bytes"),
                prefix_insert_ms=as_float(raw, "prefix_insert_ms"),
                prefill_ms=as_float(raw, "prefill_ms"),
                decode_ms=as_float(raw, "decode_ms"),
                model_ttft_ms=float(ttft),
                first_decode_ms=as_float(raw, "first_decode_ms"),
                total_ms=as_float(raw, "total_ms"),
                cache_entries=as_int(raw, "cache_entries"),
                cache_bytes=as_int(raw, "cache_bytes"),
                cache_max_bytes=as_int(raw, "cache_max_bytes"),
            )
        )
    if not rows:
        raise SystemExit(f"no prefix-cache stats rows found in {path}")
    return rows


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = pct * (len(ordered) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    frac = rank - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def fmt(value: float | int) -> str:
    if isinstance(value, int):
        return str(value)
    return f"{value:.4g}"


def emit_metric(name: str, value: float | int | str) -> None:
    print(f"{name}\t{value if isinstance(value, str) else fmt(value)}")


def emit_group(prefix: str, rows: list[CacheRow]) -> None:
    ttft = [r.model_ttft_ms for r in rows]
    total = [r.total_ms for r in rows]
    restore = [r.restore_ms for r in rows]
    matched = [float(r.matched_prefix_tokens) for r in rows]
    emit_metric(f"{prefix}.requests", len(rows))
    emit_metric(f"{prefix}.model_ttft_p50_ms", percentile(ttft, 0.50))
    emit_metric(f"{prefix}.model_ttft_p95_ms", percentile(ttft, 0.95))
    emit_metric(f"{prefix}.total_p50_ms", percentile(total, 0.50))
    emit_metric(f"{prefix}.restore_p95_ms", percentile(restore, 0.95))
    emit_metric(f"{prefix}.matched_prefix_p50", percentile(matched, 0.50))
    emit_metric(f"{prefix}.matched_prefix_p95", percentile(matched, 0.95))


def summarize(rows: list[CacheRow]) -> None:
    hits = [r for r in rows if r.cache_hit]
    misses = [r for r in rows if not r.cache_hit]
    exact_hits = [r for r in rows if r.exact_cache_hit]
    inserts = [r for r in rows if r.prefix_inserted_bytes > 0]

    emit_metric("requests", len(rows))
    emit_metric("hits", len(hits))
    emit_metric("misses", len(misses))
    emit_metric("hit_rate_pct", 100.0 * len(hits) / len(rows))
    emit_metric("exact_hits", len(exact_hits))
    emit_metric("generated_tokens", sum(r.generated_tokens for r in rows))
    emit_metric("cache_bytes_max", max(r.cache_bytes for r in rows))
    emit_metric("cache_max_bytes", max(r.cache_max_bytes for r in rows))
    emit_metric(
        "snapshot_insert_bytes_max",
        max((r.prefix_inserted_bytes for r in rows), default=0),
    )
    emit_metric(
        "prefix_insert_p95_ms",
        percentile([r.prefix_insert_ms for r in inserts], 0.95),
    )
    emit_group("all", rows)
    if hits:
        emit_group("hit", hits)
    if misses:
        emit_group("miss", misses)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="summarize qwen --requests-jsonl prefix-cache stats JSONL"
    )
    parser.add_argument("stats", nargs="+", type=Path, help="stats JSONL files")
    args = parser.parse_args()

    rows: list[CacheRow] = []
    for path in args.stats:
        rows.extend(parse_stats(path))
    summarize(rows)


if __name__ == "__main__":
    main()
