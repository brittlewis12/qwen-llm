# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Independently audit F32LE layer0 sidecars; optional lossless compression trial.

No model execution. Floating metrics use Python binary64 with math.fsum.
Numerical differences are reported without a quality/promotion threshold.
"""

import argparse
from array import array
from collections import Counter
import gzip
import hashlib
import io
import json
import lzma
import math
from pathlib import Path
import sys
import tarfile


def sha(data):
    return hashlib.sha256(data).hexdigest()


def metrics(a, b, ua, ub):
    error = math.fsum((x - y) ** 2 for x, y in zip(a, b))
    norm = math.fsum(x * x for x in a)
    return {"elements": len(a), "different_bits": sum(x != y for x, y in zip(ua, ub)),
            "max_abs": max((abs(x - y) for x, y in zip(a, b)), default=0.0),
            "rms_error": math.sqrt(error / max(len(a), 1)),
            "relative_l2": math.sqrt(error / norm) if norm > 0 else None,
            "relative_l2_defined": norm > 0}


def audit(path, compression_dir):
    payload = path.read_bytes()
    rows = [json.loads(line) for line in payload.splitlines() if line.strip()]
    events = Counter(r["event"] for r in rows)
    problems = []

    def check(ok, message):
        if not ok:
            problems.append(message)

    def event(name):
        return [r for r in rows if r["event"] == name]

    header, = event("header")
    check(header["schema"] == "flash.frontier_layer0.v1", "unknown schema")
    check(events["complete"] == 1 and rows[-1]["event"] == "complete"
          and rows[-1].get("execution_complete") and not rows[-1].get("error"),
          "missing/failed final completion")
    check(not any(r.get("error") or r["event"].endswith("_error") for r in rows), "packet error")
    for name, key in [("production_lease_acquired", "wired_gate_passed"),
                      ("diagnostic_memory_gate", "admitted"),
                      ("layer0_capture_admission", "admitted"),
                      ("artifact_revalidation", "unchanged")]:
        check(len(event(name)) == 1 and event(name)[0][key] is True, f"missing/failed {name}")
    alloc, = event("layer0_capture_allocation")
    check(alloc["observed_bytes"] <= alloc["priced_upper_bytes"], "capture price exceeded")
    concordance = event("layer0_observer_concordance")
    bf16_witnesses = event("layer0_bf16_dispatch_witness")
    router_witnesses = {r["label"]: r for r in event("warm_dispatch_witness")}
    check(set(router_witnesses) == {"A/off", "A/on", "B/off", "B/on"}
          and events["warm_dispatch_witness"] == 4, "missing/duplicate router witnesses")
    for label, r in router_witnesses.items():
        check(r["valid"] and r["strict_router_calls"] == 48
              and r["all_kernels"]["kernel_counts"].get("kernel_mat_mat_f32_f32_router_e8p32_strict") == 48,
              f"production router witness mismatch: {label}")
    if header["details"].get("bf16_activation_mode") == "f32":
        witnesses = {r["label"]: r for r in bf16_witnesses}
        check(len(bf16_witnesses) == 5 and set(witnesses) ==
              {"prefix2048/production", "A/off", "A/on", "B/off", "B/on"},
              "missing/duplicate BF16 intervention witnesses")
        prefix = witnesses.get("prefix2048/production", {})
        check(prefix.get("mode") == "production" and prefix.get("bfloat_activation_kernel_count", 0) > 0,
              "production-prefix BF16 path not witnessed")
        for label in ("A/off", "A/on", "B/off", "B/on"):
            witness = witnesses.get(label, {})
            check(witness.get("mode") == "f32" and witness.get("uniform_f32_valid") is True
                  and witness.get("bfloat_activation_kernel_count") == 0
                  and witness.get("f32_activation_kernel_count", 0) > 0,
                  f"invalid uniform BF16-F32 suffix witness: {label}")
            kernels = router_witnesses.get(label, {}).get("all_kernels", {}).get("kernel_counts", {})
            check(kernels.get("kernel_mat_mat_bf16_bfloat_act_f32", 0) == witness.get("bfloat_activation_kernel_count")
                  and kernels.get("kernel_mat_mat_bf16_f32", 0) == witness.get("f32_activation_kernel_count"),
                  f"BF16 witness disagrees with retained census: {label}")
    check({r["schedule"] for r in concordance} == {"A", "B"} and len(concordance) == 2,
          "missing/duplicate observer concordance")
    for r in concordance:
        check(all(r[k] for k in ["endpoint_bits_equal", "persistent_and_hyper_state_equal",
                                 "original_dispatch_sequence_equal", "capture_copies_valid"]),
              f"observer changed arm {r['schedule']}")
    endpoints = {r["label"]: r for r in event("endpoint")}
    for arm in ("A", "B"):
        off, on = endpoints[f"{arm}/off"], endpoints[f"{arm}/on"]
        check(off["state"] == on["state"] and off["logits_sha256_f32_le"] == on["logits_sha256_f32_le"],
              f"retained endpoint hashes disagree within {arm}")
    check(all(r["nonfinite_logits"] == 0 for r in endpoints.values()), "nonfinite endpoint logits")
    binaries = event("layer0_binary_complete")
    check(len(binaries) == 2 and {r["schedule"] for r in binaries} == {"A", "B"},
          "missing/duplicate binary completion")
    data, sections, files = {}, {}, []
    for record in binaries:
        arm = record["schedule"]
        # Resolve retained evidence beside the JSONL; no dependency on producer checkout path.
        binary_path = path.parent / Path(record["path"]).name
        raw = binary_path.read_bytes()
        check(len(raw) == record["bytes"] and sha(raw) == record["sha256"], f"binary binding failed: {arm}")
        data[arm] = raw
        files.append({"schedule": arm, "path": str(binary_path), "bytes": len(raw), "sha256": sha(raw)})
        descriptors = [r for r in event("layer0_capture_tensor") if r["schedule"] == arm]
        check(len(descriptors) == 17, f"expected 17 sections: {arm}")
        offset = 0
        sections[arm] = {}
        for d in descriptors:
            name, size = d["name"], d["bytes"]
            check(d["byte_offset"] == offset and size == 4 * math.prod(d["shape"]), f"bad extent: {arm}/{name}")
            raw_section = raw[offset:offset + size]
            check(len(raw_section) == size and sha(raw_section) == d["sha256"], f"section binding failed: {arm}/{name}")
            floats, bits = array("f"), array("I")
            floats.frombytes(raw_section)
            bits.frombytes(raw_section)
            if sys.byteorder != "little":
                floats.byteswap()
                bits.byteswap()
            finite = all(math.isfinite(x) for x in floats)
            check(finite and d["all_finite"], f"nonfinite section: {arm}/{name}")
            check(name not in sections[arm], f"duplicate section: {arm}/{name}")
            sections[arm][name] = (d, floats, bits)
            offset += size
        check(offset == len(raw), f"unindexed bytes: {arm}")
    check(sections["A"].keys() == sections["B"].keys(), "section names differ")
    recorded = {r["name"]: r for r in event("layer0_cross_schedule")}
    check(len(recorded) == 17 and events["layer0_cross_schedule"] == 17, "missing/duplicate comparisons")
    recomputed = []
    scalar_comparisons = 0
    largest_rounding_relative_difference = 0.0

    def compare_metrics(actual, expected, label):
        nonlocal scalar_comparisons, largest_rounding_relative_difference
        for k, value in actual.items():
            other = expected[k]
            scalar_comparisons += 1
            if isinstance(value, float) and other is not None:
                scale = max(abs(value), abs(other))
                if scale:
                    largest_rounding_relative_difference = max(largest_rounding_relative_difference, abs(value - other) / scale)
                # Audit summation-order rounding only; this is not a model-quality gate.
                check(math.isclose(value, other, rel_tol=1e-10, abs_tol=1e-15), f"metric differs: {label}/{k}")
            else:
                check(value == other, f"metric differs: {label}/{k}")

    for name, (desc, a, ua) in sections["A"].items():
        db, b, ub = sections["B"][name]
        check(desc["shape"] == db["shape"], f"shape differs: {name}")
        n_slices = 3 if name == "state_checkpoints" else 1 if name.startswith("initial_") else 8
        width = len(a) // n_slices
        aggregate = metrics(a, b, ua, ub)
        compare_metrics(aggregate, recorded[name]["aggregate"], name)
        slices = []
        check(len(recorded[name]["slices"]) == n_slices, f"slice count differs: {name}")
        for i in range(n_slices):
            lo, hi = i * width, (i + 1) * width
            m = metrics(a[lo:hi], b[lo:hi], ua[lo:hi], ub[lo:hi])
            compare_metrics(m, recorded[name]["slices"][i]["metrics"], f"{name}/{i}")
            slices.append({"slice": i, "absolute_position": None if name.startswith("initial_") else 2048 + i,
                           "metrics": m})
        recomputed.append({"name": name, "shape": desc["shape"], "aggregate": aggregate, "slices": slices,
                           "different_positions": [r["absolute_position"] for r in slices if r["metrics"]["different_bits"]]})
    compression = []
    if compression_dir is not None:
        compression_dir.mkdir(parents=True, exist_ok=True)
        tar_stream = io.BytesIO()
        with tarfile.open(fileobj=tar_stream, mode="w", format=tarfile.USTAR_FORMAT) as archive:
            for item in files:
                member = tarfile.TarInfo(Path(item["path"]).name)
                member.size, member.mode, member.mtime = item["bytes"], 0o644, 0
                archive.addfile(member, io.BytesIO(data[item["schedule"]]))
        tar_bytes = tar_stream.getvalue()
        candidates = [
            ("tar.gz", gzip.compress(tar_bytes, compresslevel=9, mtime=0), "gzip level9, deterministic tar"),
            ("tar.xz", lzma.compress(tar_bytes, format=lzma.FORMAT_XZ,
                                    filters=[{"id": lzma.FILTER_LZMA2, "preset": 6, "dict_size": 32 << 20}]),
             "XZ/LZMA2 preset6 with32MiB dictionary, deterministic tar"),
        ]
        for extension, encoded, method in candidates:
            destination = compression_dir / f"{path.stem}-sidecars.{extension}"
            with destination.open("xb") as out:
                out.write(encoded)
            decoded = gzip.decompress(encoded) if extension == "tar.gz" else lzma.decompress(encoded)
            check(decoded == tar_bytes, f"archive byte roundtrip failed: {extension}")
            with tarfile.open(fileobj=io.BytesIO(decoded), mode="r:") as archive:
                members = archive.getmembers()
                check([m.name for m in members] == [Path(f["path"]).name for f in files], "archive member list differs")
                for member, item in zip(members, files):
                    content = archive.extractfile(member).read()
                    check(sha(content) == item["sha256"], f"archive member hash failed: {member.name}")
            compression.append({"path": str(destination), "method": method, "bytes": len(encoded),
                                "sha256": sha(encoded), "original_bytes": sum(f["bytes"] for f in files),
                                "retained_fraction": len(encoded) / sum(f["bytes"] for f in files),
                                "roundtrip_verified": decoded == tar_bytes,
                                "original_members": files})
    return {"packet": str(path), "bytes": len(payload), "sha256": sha(payload), "events": events,
            "problems": problems, "files": files, "all_34_sections_finite": not any("nonfinite section" in p for p in problems),
            "recomputed_metrics": recomputed, "recorded_metric_fields_checked": scalar_comparisons,
            "max_relative_summation_rounding_difference": largest_rounding_relative_difference,
            "observer_concordance": concordance, "weights": event("layer0_weights"),
            "bf16_dispatch_witnesses": bf16_witnesses,
            "layer0_dispatches": event("layer0_dispatches"), "source_binding": header,
            "executable_binding": event("executable_binding"),
            "endpoint_comparison_recorded_not_recomputed": event("layer0_endpoint_cross_schedule"),
            "capture_admission": event("layer0_capture_admission"), "compression_trials": compression}


def compare_packets(baseline, intervention):
    """Same-arm baseline→intervention comparisons from independently audited bytes."""
    paths = [baseline, intervention]
    rows = [[json.loads(line) for line in path.read_bytes().splitlines() if line.strip()]
            for path in paths]
    problems = []
    prompts = [[r["used_token_ids"] for r in records if r["event"] == "prompt"] for records in rows]
    if prompts[0] != prompts[1]:
        problems.append("cross-packet token streams differ")
    payloads = []
    for path, records in zip(paths, rows):
        payloads.append({r["schedule"]: (path.parent / Path(r["path"]).name).read_bytes()
                         for r in records if r["event"] == "layer0_binary_complete"})
    tensors = [{(r["schedule"], r["name"]): r for r in records if r["event"] == "layer0_capture_tensor"}
               for records in rows]
    comparisons = []
    for (arm, name), desc in tensors[0].items():
        other = tensors[1][arm, name]
        if desc["shape"] != other["shape"]:
            problems.append(f"cross-packet shape mismatch: {arm}/{name}")
            continue
        vectors, bits = [], []
        for payload, d in zip(payloads, (desc, other)):
            raw = payload[arm][d["byte_offset"]:d["byte_offset"] + d["bytes"]]
            a, u = array("f"), array("I")
            a.frombytes(raw)
            u.frombytes(raw)
            if sys.byteorder != "little":
                a.byteswap()
                u.byteswap()
            vectors.append(a)
            bits.append(u)
        a, b = vectors
        ua, ub = bits
        count = 3 if name == "state_checkpoints" else 1 if name.startswith("initial_") else 8
        width = len(a) // count
        slices = []
        for i in range(count):
            lo, hi = i * width, (i + 1) * width
            slices.append({"absolute_position": None if name.startswith("initial_") else 2048 + i,
                           "metrics": metrics(a[lo:hi], b[lo:hi], ua[lo:hi], ub[lo:hi])})
        comparisons.append({"schedule": arm, "name": name, "aggregate": metrics(a, b, ua, ub),
                            "slices": slices})
    endpoint_differences = []
    for records, label in zip(rows, ("baseline", "intervention")):
        endpoints = {r["label"]: r for r in records if r["event"] == "endpoint"}
        a, b = endpoints["A/on"], endpoints["B/on"]
        endpoint_differences.append({"packet": label,
            "causal_metadata_equal": all(a["state"][k] == b["state"][k]
                                          for k in ("position", "qsa_lengths", "ple_prior_tokens")),
            "changed_persistent_indices": [x["index"] for x, y in zip(a["state"]["tensors"], b["state"]["tensors"])
                                           if x["sha256"] != y["sha256"]],
            "hyper_hash_equal": a["state"]["hyper_sha256_f32_le"] == b["state"]["hyper_sha256_f32_le"]})
    census = [{r["schedule"]: r["rows"] for r in records if r["event"] == "layer0_dispatches"} for records in rows]
    return {"baseline": str(baseline), "intervention": str(intervention), "problems": problems,
            "same_arm_comparisons": comparisons, "endpoint_cross_schedule_hashes": endpoint_differences,
            "tagged_gdn_dispatches_equal_to_baseline": {arm: census[0][arm] == census[1][arm] for arm in ("A", "B")},
            "limits": "No raw endpoint logits; cross-packet endpoint numerical errors cannot be recomputed."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("packet", type=Path)
    parser.add_argument("--baseline", type=Path, help="audit baseline and recompute each arm's change from it")
    parser.add_argument("--compression-dir", type=Path, help="optional new archives; originals never modified/deleted")
    args = parser.parse_args()
    result = audit(args.packet, args.compression_dir)
    if args.baseline is not None:
        baseline = audit(args.baseline, None)
        result["baseline_audit"] = {k: baseline[k] for k in ("packet", "sha256", "problems", "files")}
        result["cross_packet"] = compare_packets(args.baseline, args.packet)
        result["problems"].extend(baseline["problems"] + result["cross_packet"]["problems"])
    print(json.dumps(result, indent=2, allow_nan=False))
    raise SystemExit(bool(result["problems"]))


if __name__ == "__main__":
    main()
