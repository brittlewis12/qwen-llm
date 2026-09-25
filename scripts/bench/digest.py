#!/usr/bin/env -S uv run --quiet
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Render a `family.sh` result directory as a Markdown scoreboard.

Reads manifest.json + lcpp-*.json + qwen-*.json from a directory, emits
the digest on stdout. Pairs rows by (model, test_shape) only — missing
pairs are dropped and surfaced as a sanity flag.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

# ----- helpers -----

PEAK_GB_S = 474.0  # M4 Max measured stream anchor. See docs/PERF-ROADMAP.md.


def manifest_file_size_gib(model_entry: dict) -> float:
    """On-disk footprint in GiB. Older manifests used `size_bytes`; current
    ones use `file_size_bytes` (renamed when `model_size` in the row
    schema was retargeted to weight tensor bytes only). Read either."""
    bytes_ = model_entry.get("file_size_bytes") or model_entry.get("size_bytes") or 0
    return bytes_ / (1024**3)


def load_dir(root: Path) -> dict[str, Any]:
    """Read everything we need from one family.sh result directory.

    Returns a dict shaped:
        {
          "manifest": ...,
          "lcpp": { tag -> [row, ...] },
          "qwen": { tag -> [row, ...] },
        }
    where each row is one llama-bench-style record dict.
    """
    manifest = json.loads((root / "manifest.json").read_text())

    lcpp: dict[str, list[dict]] = defaultdict(list)
    qwen: dict[str, list[dict]] = defaultdict(list)

    for path in sorted(root.glob("lcpp-*.json")):
        tag = path.stem.removeprefix("lcpp-")
        with path.open() as f:
            for row in json.load(f):
                # llama-bench encodes the shape as `n_prompt` + `n_gen`
                # rather than a single `test` string. Synthesize the same
                # shape vocabulary we emit from qwen-bench so the cross-
                # engine lookup is trivial (`pp512`, `tg128`, etc.).
                if row.get("test") is None:
                    n_prompt = row.get("n_prompt", 0) or 0
                    n_gen = row.get("n_gen", 0) or 0
                    depth = row.get("n_depth", 0) or 0
                    suffix = f"@d{depth}" if depth else ""
                    if n_prompt > 0 and n_gen == 0:
                        row["test"] = f"pp{n_prompt}{suffix}"
                    elif n_gen > 0 and n_prompt == 0:
                        row["test"] = f"tg{n_gen}{suffix}"
                    else:
                        # combined pp+tg "depth" row; we skip these for
                        # the scoreboard since qwen-bench doesn't emit
                        # them and the comparison would be apples to
                        # oranges.
                        row["test"] = f"pp{n_prompt}+tg{n_gen}"
                lcpp[tag].append(row)

    # llama.cpp at non-comparator micro-batch sizes (e.g. its default 512),
    # written by family.py as lcpp_ub<N>-<tag>.json.
    lcpp_variants: dict[int, dict[str, list[dict]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for path in sorted(root.glob("lcpp_ub*-*.json")):
        ubatch_text, tag = path.stem.removeprefix("lcpp_ub").split("-", 1)
        with path.open() as f:
            lcpp_variants[int(ubatch_text)][tag].extend(json.load(f))

    for path in sorted(root.glob("qwen-*-*.json")):
        # qwen-pp512-27B.json / qwen-tg128-27B.json — tag is the suffix after
        # the second dash. The shape prefix (pp512 / tg128) is recoverable
        # from the row's `test` field, so we don't need to parse the
        # filename twice.
        stem = path.stem
        # qwen-pp512-27B -> tag = "27B"; qwen-tg128-122B-A10B -> tag = "122B-A10B"
        rest = stem.split("-", 2)[-1]
        tag = rest
        with path.open() as f:
            for row in json.load(f):
                qwen[tag].append(row)

    return {
        "manifest": manifest,
        "lcpp": dict(lcpp),
        "lcpp_variants": {k: dict(v) for k, v in lcpp_variants.items()},
        "qwen": dict(qwen),
    }


def lcpp_supported(model: dict) -> bool:
    """False when upstream llama.cpp has no implementation (not measured)."""
    return model.get("lcpp", True)


def find_row(rows: list[dict], test: str) -> dict | None:
    """Return the first row with `row['test'] == test`, or None."""
    for r in rows:
        if r.get("test") == test:
            return r
    return None


def fmt(v: float | None, places: int = 0) -> str:
    if v is None:
        return "—"
    if places == 0:
        return f"{v:.0f}"
    return f"{v:.{places}f}"


def fmt_ratio(num: float | None, den: float | None) -> str:
    """Display the qwen/lcpp ratio with a directional emoji.

    Convention: >1.0 means qwen-llm is faster (green), <0.9 is yellow,
    <0.5 is red, 0.95..1.05 is yellow tie. Emoji are useful when scanning
    a long table; they're not load-bearing for any downstream tooling.
    """
    if num is None or den is None or den <= 0:
        return "—"
    r = num / den
    if r >= 1.05:
        marker = "✅"
    elif r >= 0.95:
        marker = "🟡"
    elif r >= 0.5:
        marker = "⚠️"
    else:
        marker = "❌"
    return f"{marker} {r:.2f}x"


def model_size_gib(row: dict) -> float | None:
    s = row.get("model_size") or 0
    return s / (1024**3) if s > 0 else None


def decode_bw_gb_s(row: dict) -> float | None:
    """Effective decode bandwidth = model_size_gib * t/s * 1.073 (GiB→GB)."""
    sz = model_size_gib(row)
    ts = row.get("avg_ts")
    if not sz or not ts:
        return None
    return sz * ts * 1.073


# ----- table builders -----


def pp_table(state: dict, shapes: list[int]) -> str:
    """Build the prompt-processing comparison table."""
    manifest = state["manifest"]
    out = []
    header_shapes = " | ".join(f"pp{p} lcpp | pp{p} qwen | Δ" for p in shapes)
    out.append("| model | size | " + header_shapes + " |")
    out.append("| --- | ---: | " + " | ".join(["---:"] * (3 * len(shapes))) + " |")

    for m in manifest["models"]:
        tag = m["tag"]
        lcpp_rows = state["lcpp"].get(tag, [])
        qwen_rows = state["qwen"].get(tag, [])
        size = manifest_file_size_gib(m)
        cells: list[str] = [f"{m['display']} {m['kind']}", f"{size:.1f} GiB"]
        for p in shapes:
            test = f"pp{p}"
            lc = find_row(lcpp_rows, test)
            qw = find_row(qwen_rows, test)
            lc_ts = lc.get("avg_ts") if lc else None
            qw_ts = qw.get("avg_ts") if qw else None
            cells.append(fmt(lc_ts))
            cells.append(fmt(qw_ts))
            cells.append(fmt_ratio(qw_ts, lc_ts))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def tg_table(state: dict, shapes: list[int]) -> str:
    manifest = state["manifest"]
    out = []
    header_shapes = " | ".join(f"tg{n} lcpp | tg{n} qwen | Δ" for n in shapes)
    out.append("| model | size | " + header_shapes + " |")
    out.append("| --- | ---: | " + " | ".join(["---:"] * (3 * len(shapes))) + " |")

    for m in manifest["models"]:
        tag = m["tag"]
        lcpp_rows = state["lcpp"].get(tag, [])
        qwen_rows = state["qwen"].get(tag, [])
        size = manifest_file_size_gib(m)
        cells: list[str] = [f"{m['display']} {m['kind']}", f"{size:.1f} GiB"]
        for n in shapes:
            test = f"tg{n}"
            lc = find_row(lcpp_rows, test)
            qw = find_row(qwen_rows, test)
            lc_ts = lc.get("avg_ts") if lc else None
            qw_ts = qw.get("avg_ts") if qw else None
            cells.append(fmt(lc_ts))
            cells.append(fmt(qw_ts))
            cells.append(fmt_ratio(qw_ts, lc_ts))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def depth_table(state: dict, shapes: list[int]) -> str | None:
    """tg at each model's nonzero depths (llama-bench -d); rows match on the
    full `tg<N>@d<depth>` label, never on the shape alone."""
    manifest = state["manifest"]
    depths = sorted(
        {d for m in manifest["models"] for d in m.get("tg_depths", []) if d > 0}
    )
    if not depths:
        return None
    labels = [f"tg{n}@d{d}" for n in shapes for d in depths]
    out = [
        "| model | " + " | ".join(f"{t} lcpp | {t} qwen | Δ" for t in labels) + " |",
        "| --- | " + " | ".join(["---:"] * (3 * len(labels))) + " |",
    ]
    for m in manifest["models"]:
        tag = m["tag"]
        cells = [f"{m['display']} {m['kind']}"]
        for test in labels:
            lc = find_row(state["lcpp"].get(tag, []), test)
            qw = find_row(state["qwen"].get(tag, []), test)
            lc_ts = lc.get("avg_ts") if lc else None
            qw_ts = qw.get("avg_ts") if qw else None
            cells += [fmt(lc_ts), fmt(qw_ts), fmt_ratio(qw_ts, lc_ts)]
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def lcpp_default_pp_table(state: dict, shapes: list[int]) -> str | None:
    """qwen-llm against llama.cpp at its other micro-batch sizes (its
    shipped default is 512), next to the tuned comparator."""
    variants = state.get("lcpp_variants") or {}
    if not variants:
        return None
    manifest = state["manifest"]
    out = []
    cols = [(u, p) for u in sorted(variants) for p in shapes]
    out.append(
        "| model | " + " | ".join(f"pp{p} lcpp ub{u} | Δ qwen" for u, p in cols) + " |"
    )
    out.append("| --- | " + " | ".join(["---:"] * (2 * len(cols))) + " |")
    for m in manifest["models"]:
        tag = m["tag"]
        cells = [f"{m['display']} {m['kind']}"]
        for ubatch, p in cols:
            lc = find_row(variants[ubatch].get(tag, []), f"pp{p}")
            qw = find_row(state["qwen"].get(tag, []), f"pp{p}")
            lc_ts = lc.get("avg_ts") if lc else None
            qw_ts = qw.get("avg_ts") if qw else None
            cells += [fmt(lc_ts), fmt_ratio(qw_ts, lc_ts)]
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def bandwidth_table(state: dict, shape: int) -> str:
    """Decode bandwidth utilization (% of peak) at the given tg shape."""
    manifest = state["manifest"]
    out = [
        f"Effective decode bandwidth at tg{shape} "
        f"(`model_size × tg t/s × 1.073`). Peak: {PEAK_GB_S} GB/s.\n",
        "| model | qwen GB/s | qwen % peak | lcpp GB/s | lcpp % peak |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for m in manifest["models"]:
        if m["kind"] == "moe":
            # Skip MoE — active param count is the right denominator and
            # we don't expose it through general.parameter_count. Better
            # to leave the line out than emit a misleading number.
            continue
        tag = m["tag"]
        lcpp_rows = state["lcpp"].get(tag, [])
        qwen_rows = state["qwen"].get(tag, [])
        lc = find_row(lcpp_rows, f"tg{shape}")
        qw = find_row(qwen_rows, f"tg{shape}")
        qw_gb = decode_bw_gb_s(qw) if qw else None
        lc_gb = decode_bw_gb_s(lc) if lc else None
        out.append(
            "| {disp} | {qg} | {qp} | {lg} | {lp} |".format(
                disp=m["display"],
                qg=fmt(qw_gb),
                qp=fmt((qw_gb / PEAK_GB_S * 100) if qw_gb else None) + "%",
                lg=fmt(lc_gb),
                lp=fmt((lc_gb / PEAK_GB_S * 100) if lc_gb else None) + "%",
            )
        )
    return "\n".join(out)


def family_summary_table(state: dict) -> str:
    """Per-variant qualitative status row, mirroring the digest layout we
    used in docs/bench/2026-05-17-family-baseline/README.md."""
    manifest = state["manifest"]
    out = [
        "| variant | decode @ tg128 | prefill @ pp512 |",
        "| --- | --- | --- |",
    ]
    for m in manifest["models"]:
        tag = m["tag"]
        lc = find_row(state["lcpp"].get(tag, []), "tg128")
        qw = find_row(state["qwen"].get(tag, []), "tg128")
        lc_pp = find_row(state["lcpp"].get(tag, []), "pp512")
        qw_pp = find_row(state["qwen"].get(tag, []), "pp512")
        supported = lcpp_supported(m)

        def verdict(qw_v, lc_v):
            if not supported:
                return "no llama.cpp implementation"
            if qw_v is None or lc_v is None:
                return "—"
            r = (qw_v.get("avg_ts") or 0) / (lc_v.get("avg_ts") or 1e-9)
            if r >= 1.05:
                return f"✅ won ({r:.2f}x)"
            if r >= 0.95:
                return f"🟡 parity ({r:.2f}x)"
            if r >= 0.5:
                return f"⚠️ {int((1 - r) * 100)}% behind"
            return f"❌ {1 / r:.1f}x behind"

        out.append(
            f"| {m['display']} {m['kind']} | {verdict(qw, lc)} | {verdict(qw_pp, lc_pp)} |"
        )
    return "\n".join(out)


def sanity_flags(state: dict, pp_shapes: list[int], tg_shapes: list[int]) -> list[str]:
    """Surface anomalies that should make a reader pause."""
    flags: list[str] = []
    if state["manifest"]["engines"]["qwen_llm"].get("build_dirty"):
        flags.append(
            "qwen-llm build was dirty at sweep start — uncommitted changes "
            "in the worktree. Results are reproducible only if you also "
            "stash the same diff."
        )
    # Missing (model, shape) coverage: enumerate every cell the sweep
    # claims to cover and report any that didn't materialize in the JSON
    # set. Without this check a half-complete sweep prints a verdict table
    # full of "—" cells and reads as if the run succeeded.
    expected: list[tuple[str, str]] = []
    supported = {m["tag"]: lcpp_supported(m) for m in state["manifest"]["models"]}
    for m in state["manifest"]["models"]:
        if not supported[m["tag"]]:
            flags.append(
                f"{m['display']}: upstream llama.cpp has no implementation of "
                "this architecture; its cells are qwen-llm only."
            )
        for p in pp_shapes:
            expected.append((m["tag"], f"pp{p}"))
        for n in tg_shapes:
            for d in m.get("tg_depths", [0]):
                expected.append((m["tag"], f"tg{n}@d{d}" if d else f"tg{n}"))
    for tag, test in expected:
        qw = find_row(state["qwen"].get(tag, []), test)
        lc = find_row(state["lcpp"].get(tag, []), test)
        if qw is None and lc is None:
            flags.append(f"{tag} {test}: neither engine ran (missing from JSON set).")
        elif qw is None:
            flags.append(f"{tag} {test}: qwen-llm row missing.")
        elif lc is None and supported[tag]:
            flags.append(f"{tag} {test}: llama.cpp row missing.")
    for engine in ("qwen", "lcpp"):
        for tag, rows in state[engine].items():
            for row in rows:
                if row.get("heterogeneous_fields"):
                    flags.append(
                        f"{tag} {row.get('test')} ({engine}): blocks executed "
                        f"differently: {row['heterogeneous_fields']}"
                    )
    # pp throughput non-monotonic (pp1024 < pp512) — historically a
    # prefill-chunk-tuning artifact worth surfacing.
    for m in state["manifest"]["models"]:
        tag = m["tag"]
        qw = state["qwen"].get(tag, [])
        if len(pp_shapes) >= 2:
            small = find_row(qw, f"pp{pp_shapes[-2]}")
            large = find_row(qw, f"pp{pp_shapes[-1]}")
            if (
                small
                and large
                and (large.get("avg_ts") or 0) < (small.get("avg_ts") or 0) * 0.97
            ):
                flags.append(
                    f"{m['display']}: pp{pp_shapes[-1]} < pp{pp_shapes[-2]} "
                    f"({large['avg_ts']:.0f} vs {small['avg_ts']:.0f} t/s). "
                    "Prefill-chunk default may be suboptimal at the larger prompt."
                )
    return flags


# ----- main -----


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: digest.py <result-dir>", file=sys.stderr)
        return 2
    root = Path(argv[1]).resolve()
    state = load_dir(root)
    manifest = state["manifest"]

    pp_shapes = manifest["sweep"]["pp_shapes"]
    tg_shapes = manifest["sweep"]["tg_shapes"]

    qwen_id = manifest["engines"]["qwen_llm"]
    lcpp_id = manifest["engines"]["llama_cpp"]

    print(f"# Family Baseline — {manifest['stamp']}")
    print()
    print(
        "qwen-llm vs llama.cpp scoreboard across model families. Same "
        "physical box, sequential runs, no concurrent benches. This file "
        "is auto-generated by `scripts/bench/digest.py` from the JSON "
        "results in this directory; edit the inputs, not this README."
    )
    print()
    print("## Identity")
    print()
    print(f"- Host: `{manifest['host']['hostname']}` — {manifest['host']['os']}")
    print(f"- Loadavg at start: `{manifest['host']['loadavg']}`")
    print(
        f"- `qwen-llm`: `{qwen_id['build_commit']}`"
        + (" (**dirty worktree**)" if qwen_id["build_dirty"] else "")
    )
    if lcpp_id.get("status") == "not_probed":
        print(
            "- `llama.cpp`: not probed (no model in this sweep has an upstream implementation)"
        )
    else:
        print(
            f"- `llama.cpp`: `{lcpp_id['build_commit']}` "
            f"(build {lcpp_id['build_number']}, backends `{lcpp_id['backends']}`)"
        )
        print(f"- GPU: `{lcpp_id['gpu_info']}`")
    qe = manifest.get("qwen_env_at_start") or {}
    if qe:
        kv = ", ".join(f"`{k}={v}`" for k, v in qe.items())
        print(f"- QWEN_* env: {kv}")
    else:
        print("- QWEN_* env: (none set)")
    print()
    print("## Prompt processing (tokens/sec)")
    print()
    print(pp_table(state, pp_shapes))
    print()
    print("## Token generation (tokens/sec)")
    print()
    print(tg_table(state, tg_shapes))
    print()
    depth = depth_table(state, tg_shapes)
    if depth:
        print("## Token generation at depth (tokens/sec)")
        print()
        print(depth)
        print()
    defaults = lcpp_default_pp_table(state, pp_shapes)
    if defaults:
        print("## Prompt processing vs llama.cpp micro-batch variants")
        print()
        print(
            "The tables above compare against llama.cpp at its tuned "
            "micro-batch; its shipped default is `-ub 512`."
        )
        print()
        print(defaults)
        print()
    if 128 in tg_shapes:
        print("## Decode bandwidth utilization @ tg128")
        print()
        print(bandwidth_table(state, 128))
        print()
    print("## Family verdict")
    print()
    print(family_summary_table(state))
    print()
    print("## Method")
    print()
    runs = manifest["sweep"]["runs"]
    order = manifest["sweep"].get("engine_order", "per_model_lcpp_then_qwen")
    if order == "per_model_lcpp_then_qwen":
        print(
            f"- Per model: `llama-bench` first (`-r {runs}`), then "
            "`qwen-bench` on the same model. Interleaved by model, never "
            "in parallel."
        )
    elif order == "abba_per_model_alternating":
        blocks = manifest["sweep"].get("blocks")
        print(
            f"- Per model: {blocks} paired blocks with alternating engine order "
            f"(ABBA, first engine also alternating by model), `-r {runs}` per "
            "block; samples merge across blocks (`block_avg_ts` keeps each "
            "block's mean). Never in parallel."
        )
        print(
            f"- llama.cpp pp runs at `-ub {manifest['sweep'].get('lcpp_ubatch')}`; the "
            "main tables compare against the "
            f"`ub{manifest['sweep'].get('lcpp_tuned_ubatch')}` rows (not a per-cell "
            "best; the variant table shows the others)."
        )
        print(
            "- Depth: neither engine times its fill. llama-bench fills once and "
            "restores the cached depth state for later reps; qwen-bench refills "
            "per rep. llama-bench's tg warm-up is one token; qwen-bench runs a "
            "model warm-up plus one full untimed rep per row. At `-b 2048`, "
            "llama-bench pp4096 computes last-token logits at both batch "
            "endpoints; qwen-llm's production prefill computes the final one."
        )
    else:
        print(f"- engine_order = `{order}` (see `manifest.json`).")
    print(
        "- JSON outputs are persisted alongside this digest. To re-derive "
        "this README, run `scripts/bench/digest.py <dir>`."
    )
    print()
    flags = sanity_flags(state, pp_shapes, tg_shapes)
    if flags:
        print("## Sanity flags")
        print()
        for f in flags:
            print(f"- {f}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
