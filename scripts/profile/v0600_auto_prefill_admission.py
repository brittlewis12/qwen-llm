#!/usr/bin/env python3

import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import time

import v0593_demand_paged_no_copy as common


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = ROOT / "target/profiles/v0600-auto-prefill-admission-p1"
PREREG = ROOT / "docs/bench/v0600-auto-prefill-admission.md"
PERF_TOOLS = ROOT / "docs/PERF-TOOLS.md"
PROMPT = ROOT / "docs/bench/tokenizer-prompts/current-mei-medium-qwen36-strip.txt"
BINARY = ROOT / "target/release/qwen"
BENCH_BINARY = ROOT / "target/release/qwen-bench"
COMMON_RUNNER = ROOT / "scripts/profile/v0593_demand_paged_no_copy.py"

A3B_MODEL = Path("/Users/tito/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")
A10B_DIR = Path("/Users/tito/models/unsloth-Qwen3.5-122B-A10B-GGUF/UD-Q4_K_XL")
A10B_SHARDS = tuple(
    A10B_DIR / f"Qwen3.5-122B-A10B-UD-Q4_K_XL-{index:05d}-of-00003.gguf"
    for index in range(1, 4)
)

EXPECTED_PROMPT_SHA256 = (
    "ca924d7a3613ef8a6aa02fdcbfc56f2a7bffb5ec7aa2ddbd11e7efa3be7ad3f6"
)
EXPECTED_DEVICE = (
    "device: Apple M4 Max | unified_memory=true | max_threadgroup_memory=32768 bytes"
)
EXPECTED_HW_MEMSIZE = 137_438_953_472
EXPECTED_PROMPT_BYTES = 51_876
EXPECTED_PROMPT_TOKENS = 11_287
EXPECTED_CAPACITY = 11_304
TRANSIENT_RESERVE_BYTES = 512 * 1024 * 1024
U64_MAX = 2**64 - 1
PAIR_ORDERS = ("AB", "BA", "BA", "AB")

PROFILES = {
    "a3b": {
        "model": A3B_MODEL,
        "shards": (A3B_MODEL,),
        "sha256": ("ac0e2c1189e055faa36eff361580e79c5bd6f8e76bffb4ce547f167d53e31a61",),
        "profile": "qwen3.6-35b-a3b-filetype15",
        "outer_chunk": 2048,
        "query_rows": 1024,
        "hidden_size": 2048,
        "dense_ffn_size": 0,
        "query_heads": 16,
        "kv_heads": 2,
        "head_dim": 256,
        "attention_layers": 10,
        "expert_count": 256,
        "experts_used": 8,
        "expert_ffn_size": 512,
        "shared_ffn_size": 512,
        "layer_calls": 60,
        "query_tile_calls": 120,
        "overlay": {
            "backing_bytes": 393_052_160,
            "attention_bytes": 393_052_160,
            "gdn_bytes": 235_405_312,
            "saved_bytes": 235_405_312,
        },
        "cooldown_s": 30.0,
        "median_gate": 1.04,
        "stratum_gate": 1.03,
    },
    "a10b": {
        "model": A10B_SHARDS[0],
        "shards": A10B_SHARDS,
        "sha256": (
            "467c9bd92ea518539cf75bf5a5fbfbd35e9a0b40d766ccaa67bf120e12041df3",
            "ecdbd42d43b0df9fa0ef9a584e09e95a43966ef03a122aba0b87a99d44d9ad98",
            "13300e0f059e6fa21aa0fabde2a554f9deea366c0e54f268045769b214b28c97",
        ),
        "profile": "qwen3.5-122b-a10b-filetype15",
        "outer_chunk": 4096,
        "query_rows": 1024,
        "hidden_size": 3072,
        "dense_ffn_size": 0,
        "query_heads": 32,
        "kv_heads": 2,
        "head_dim": 256,
        "attention_layers": 12,
        "expert_count": 256,
        "experts_used": 8,
        "expert_ffn_size": 1024,
        "shared_ffn_size": 1024,
        "layer_calls": 36,
        "query_tile_calls": 144,
        "overlay": {
            "backing_bytes": 807_403_520,
            "attention_bytes": 786_104_320,
            "gdn_bytes": 807_403_520,
            "saved_bytes": 786_104_320,
        },
        "cooldown_s": 120.0,
        "median_gate": 1.15,
        "stratum_gate": 1.10,
    },
}


class InconclusivePacket(RuntimeError):
    def __init__(self, profile: str, pair_index: int, reason: str) -> None:
        super().__init__(reason)
        self.profile = profile
        self.pair_index = pair_index


def command_text(command: list[str], env: dict[str, str] | None = None) -> str:
    return subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    ).stdout


def parse_json(text: str) -> object:
    return json.loads(text, parse_constant=reject_json_constant)


def reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value!r}")


def json_text(value: object, *, pretty: bool = False) -> str:
    return json.dumps(
        value,
        indent=2 if pretty else None,
        sort_keys=True,
        allow_nan=False,
    )


def finite_number(
    value: object,
    label: str,
    *,
    positive: bool = False,
    nonnegative: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"{label} is not numeric")
    number = float(value)
    if not math.isfinite(number):
        raise RuntimeError(f"{label} is not finite")
    if positive and number <= 0:
        raise RuntimeError(f"{label} is not positive")
    if nonnegative and number < 0:
        raise RuntimeError(f"{label} is negative")
    return number


def u64(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeError(f"{label} is not an integer")
    if value < 0 or value > U64_MAX:
        raise RuntimeError(f"{label} is outside u64")
    return value


def checked_add(left: int, right: int, label: str) -> int:
    return u64(u64(left, label) + u64(right, label), label)


def checked_mul(*values: int, label: str) -> int:
    result = 1
    for value in values:
        result = u64(result * u64(value, label), label)
    return result


def required_manifest_paths() -> tuple[Path, ...]:
    model_paths = tuple(
        shard for profile in PROFILES.values() for shard in profile["shards"]
    )
    return (
        Path(__file__).resolve(),
        PREREG,
        PERF_TOOLS,
        PROMPT,
        BINARY,
        BENCH_BINARY,
        COMMON_RUNNER,
        *model_paths,
    )


def source_and_build_identity() -> tuple[str, dict[str, object]]:
    tracked = (
        Path(__file__).resolve(),
        PREREG,
        PERF_TOOLS,
        PROMPT,
        COMMON_RUNNER,
    )
    for path in tracked:
        command_text(
            ["git", "ls-files", "--error-unmatch", str(path.relative_to(ROOT))]
        )
    commit = command_text(["git", "rev-parse", "HEAD"]).strip()
    dirty = command_text(["git", "status", "--porcelain=v1"]).strip()
    if dirty:
        raise RuntimeError(f"tracked source is dirty: {dirty!r}")
    build = parse_json(
        command_text([str(BENCH_BINARY), "build-info", "--output", "json"])
    )
    if not isinstance(build, dict):
        raise RuntimeError("build identity is not an object")
    if (
        build.get("build_commit") != commit
        or build.get("runtime_commit") != commit
        or build.get("status") != "match"
        or build.get("build_dirty") is not False
        or build.get("runtime_dirty") is not False
        or build.get("build_source_state") != build.get("runtime_source_state")
    ):
        raise RuntimeError(f"source/build identity mismatch: {build}")
    return commit, build


def child_environment_record(env: dict[str, str]) -> dict[str, object]:
    digest = hashlib.sha256()
    for key, value in sorted(env.items()):
        key_bytes = key.encode("utf-8")
        value_bytes = value.encode("utf-8")
        digest.update(len(key_bytes).to_bytes(8, "little"))
        digest.update(key_bytes)
        digest.update(len(value_bytes).to_bytes(8, "little"))
        digest.update(value_bytes)
    performance_controls = {
        key: value
        for key, value in sorted(env.items())
        if key.startswith(("QWEN_", "METAL_", "MTL_")) or key == "RUST_LOG"
    }
    if performance_controls:
        raise RuntimeError(
            f"normalized child environment retains controls: {performance_controls}"
        )
    return {
        "schema": 1,
        "complete_sha256": digest.hexdigest(),
        "keys": sorted(env),
        "performance_controls": performance_controls,
    }


def parse_swap_bytes(text: str) -> int:
    match = re.search(r"\bused\s*=\s*([0-9.]+)([KMG])", text)
    if match is None:
        raise RuntimeError(f"could not parse swap usage: {text!r}")
    scale = {"K": 1024, "M": 1024**2, "G": 1024**3}[match.group(2)]
    return round(float(match.group(1)) * scale)


def capture_vm_state() -> dict[str, object]:
    vm_stat = command_text(["vm_stat"])
    swap = command_text(["sysctl", "-n", "vm.swapusage"])
    pageouts = re.search(r"^Pageouts:\s+(\d+)\.$", vm_stat, re.MULTILINE)
    if pageouts is None:
        raise RuntimeError("could not parse vm_stat pageouts")
    return {
        "pageouts": int(pageouts.group(1)),
        "swap_used_bytes": parse_swap_bytes(swap),
        "vm_stat": vm_stat,
        "swapusage": swap,
    }


def build_manifest(
    removed_environment: list[str],
    base_env: dict[str, str],
) -> dict[str, object]:
    commit, build = source_and_build_identity()
    paths = required_manifest_paths()
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError(f"missing packet inputs: {missing}")
    hashes = {str(path): common.sha256_file(path) for path in paths}
    if hashes[str(PROMPT)] != EXPECTED_PROMPT_SHA256:
        raise RuntimeError("prompt SHA-256 drifted")
    for profile in PROFILES.values():
        actual = tuple(hashes[str(path)] for path in profile["shards"])
        if actual != profile["sha256"]:
            raise RuntimeError(f"model SHA-256 drifted for {profile['profile']}")

    device = command_text([str(BINARY), "--info"], env=base_env).strip()
    macos = command_text(["sw_vers", "-productVersion"], env=base_env).strip()
    hw_memsize = int(command_text(["sysctl", "-n", "hw.memsize"], env=base_env))
    if device != EXPECTED_DEVICE:
        raise RuntimeError(f"device boundary drifted: {device!r}")
    if not macos.startswith("15."):
        raise RuntimeError(f"macOS boundary drifted: {macos!r}")
    if hw_memsize != EXPECTED_HW_MEMSIZE:
        raise RuntimeError(f"memory boundary drifted: {hw_memsize}")
    return {
        "schema": 1,
        "created_unix_ms": time.time_ns() // 1_000_000,
        "source_commit": commit,
        "build_identity": build,
        "device": device,
        "macos": macos,
        "hw_memsize": hw_memsize,
        "removed_environment": removed_environment,
        "child_environment": child_environment_record(base_env),
        "sha256": hashes,
        "profile_order": list(PROFILES),
        "pair_orders": list(PAIR_ORDERS),
        "cooldown_s": {
            name: profile["cooldown_s"] for name, profile in PROFILES.items()
        },
        "prompt_bytes": PROMPT.stat().st_size,
        "prompt_tokens": EXPECTED_PROMPT_TOKENS,
        "capacity": EXPECTED_CAPACITY,
    }


def verify_packet_identity(manifest: dict[str, object]) -> None:
    commit, build = source_and_build_identity()
    if commit != manifest["source_commit"] or build != manifest["build_identity"]:
        raise RuntimeError("packet completion source/build identity drifted")
    hashes = {str(path): common.sha256_file(path) for path in required_manifest_paths()}
    if hashes != manifest["sha256"]:
        raise RuntimeError("packet completion file hashes drifted")


def completion_identity_error(manifest: dict[str, object]) -> str | None:
    try:
        verify_packet_identity(manifest)
    except Exception as error:
        return f"{type(error).__name__}: {error}"
    return None


def warm_files(paths: tuple[Path, ...]) -> tuple[float, int]:
    started = time.perf_counter()
    total = 0
    buffer = bytearray(8 * 1024 * 1024)
    for path in paths:
        with path.open("rb", buffering=0) as handle:
            while True:
                count = handle.readinto(buffer)
                if count == 0:
                    break
                total += count
    return (time.perf_counter() - started) * 1e3, total


def validate_common_row(
    row: dict[str, object],
    profile: dict[str, object],
    manifest: dict[str, object],
) -> None:
    build = manifest["build_identity"]
    if not isinstance(build, dict):
        raise RuntimeError("manifest build identity is malformed")
    expected = {
        "request_epoch": "first_post_model_load",
        "request_index": 0,
        "tokenizer_reused": False,
        "pair_requested": False,
        "pair_id": None,
        "pair_request_equal": None,
        "pair_generated_tokens_equal": None,
        "prefix_cache_used": False,
        "build_commit": manifest["source_commit"],
        "build_dirty": "0",
        "build_source_state": build["build_source_state"],
        "model": str(profile["model"]),
        "stdout_sink": "redirected",
        "ttft_endpoint": "stdout_flush_complete",
        "prompt_source": "file",
        "prompt_bytes": EXPECTED_PROMPT_BYTES,
        "prompt_tokens": EXPECTED_PROMPT_TOKENS,
        "requested_tokens": 1,
        "generated_tokens": 1,
        "stop_reason": "token_limit",
        "decode_policy": "greedy_argmax",
        "terminal_token_target_transition_consumed": False,
        "no_special_tokens": True,
        "max_context_tokens": EXPECTED_CAPACITY,
        "transition_count": 0,
        "transition_ms": 0.0,
        "transition_tps": 0.0,
        "runtime_identity_kind": "metadata_compatibility_v1",
    }
    for key, value in expected.items():
        if key not in row:
            raise RuntimeError(f"timing row {key} is missing")
        actual = row[key]
        if isinstance(value, bool):
            matches = actual is value
        else:
            matches = not isinstance(actual, bool) and actual == value
        if not matches:
            raise RuntimeError(f"timing row {key} drifted: {row.get(key)!r}")
    if not isinstance(row.get("runtime_model_id"), str) or not row["runtime_model_id"]:
        raise RuntimeError("runtime model identity is missing")
    if (
        not isinstance(row.get("runtime_tokenizer_id"), str)
        or not row["runtime_tokenizer_id"]
    ):
        raise RuntimeError("runtime tokenizer identity is missing")
    for field in (
        "runtime_and_model_load_ms",
        "prefill_ms",
        "first_token_ready_ms",
        "ttft_ms",
        "inference_complete_ms",
        "total_request_ms",
    ):
        finite_number(row.get(field), field, positive=True)
    if not (
        float(row["first_token_ready_ms"])
        <= float(row["ttft_ms"])
        <= float(row["inference_complete_ms"])
        <= float(row["total_request_ms"])
    ):
        raise RuntimeError("timing milestones are out of order")
    validate_metal_allocation_samples(row)


def expected_plan_allocations(
    profile: dict[str, object],
) -> dict[str, tuple[bool, int]]:
    n = u64(profile["outer_chunk"], "outer chunk")
    hidden = u64(profile["hidden_size"], "hidden size")
    dense_ffn = u64(profile["dense_ffn_size"], "dense FFN size")
    query_heads = u64(profile["query_heads"], "query heads")
    kv_heads = u64(profile["kv_heads"], "KV heads")
    head_dim = u64(profile["head_dim"], "head dimension")
    attention_layers = u64(profile["attention_layers"], "attention layers")
    expert_count = u64(profile["expert_count"], "expert count")
    experts_used = u64(profile["experts_used"], "experts used")
    expert_ffn = u64(profile["expert_ffn_size"], "expert FFN size")
    shared_ffn = u64(profile["shared_ffn_size"], "shared FFN size")
    if query_heads % kv_heads != 0:
        raise RuntimeError("query heads are not divisible by KV heads")
    attention_group = query_heads // kv_heads

    def f32(*shape: int) -> int:
        return checked_mul(*shape, 4, label="expected F32 allocation")

    def f16(*shape: int) -> int:
        return checked_mul(*shape, 2, label="expected F16 allocation")

    query_dim = checked_mul(query_heads, head_dim, label="query dimension")
    kv_dim = checked_mul(kv_heads, head_dim, label="KV dimension")
    slots = checked_mul(n, experts_used, label="MoE slots")
    allocations = {
        "attn_gdn_overlay_backing": (
            False,
            u64(profile["overlay"]["backing_bytes"], "overlay"),
        ),
        "x_pack": (False, f32(n, hidden)),
        "h_pack": (False, f32(n, hidden)),
        "mixer_out_pack": (False, f32(n, hidden)),
        "attn_qkv_fused_pack": (False, f32(1)),
        "attn_q_full_pack": (False, f32(n, 2, query_dim)),
        "attn_q_pack": (False, f32(1)),
        "attn_gate_pack": (False, f32(1)),
        "attn_q_normed_pack": (False, f32(n, query_dim)),
        "attn_k_now_pack": (False, f32(n, kv_dim)),
        "attn_v_now_pack": (False, f32(n, kv_dim)),
        "attn_k_normed_pack": (False, f32(n, kv_dim)),
        "attn_o_pack": (False, f32(n, query_dim)),
        "attn_prefill_v4_o_partial_pack": (
            False,
            f32(8, kv_heads, 1024, attention_group, head_dim),
        ),
        "attn_prefill_v4_ml_partial_pack": (
            False,
            f32(8, kv_heads, 1024, 2, attention_group),
        ),
        "attn_matrix_scores_pack": (False, f32(1)),
        "attn_matrix_vt_pack": (
            False,
            f16(
                attention_layers,
                kv_heads,
                head_dim,
                EXPECTED_PROMPT_TOKENS,
            ),
        ),
        "ffn_gate_pack": (False, f32(n, dense_ffn)),
        "ffn_up_pack": (False, f32(n, dense_ffn)),
        "ffn_inner_pack": (False, f32(n, dense_ffn)),
        "ffn_out_pack": (False, f32(n, hidden)),
        "moe_topk_idx_pack": (False, f32(slots)),
        "moe_router_probs_pack": (False, f32(n, expert_count)),
        "moe_topk_weight_pack": (False, f32(slots)),
        "moe_shared_gate_pack": (False, f32(n)),
        "moe_inner_pack": (False, f32(1)),
        "moe_expert_out_pack": (False, f32(1)),
        "moe_group_slot_idx_pack": (False, f32(slots)),
        "moe_group_count_pack": (False, f32(expert_count)),
        "moe_group_ids_pack": (False, f32(expert_count, n)),
        "moe_group_token_idx_pack": (False, f32(slots)),
        "moe_group_weight_pack": (False, f32(slots)),
        "moe_group_inner_pack": (False, f32(slots, expert_ffn)),
        "moe_group_out_pack": (False, f32(slots, hidden)),
        "moe_shared_ffn_gate_pack": (False, f32(n, shared_ffn)),
        "moe_shared_ffn_up_pack": (False, f32(n, shared_ffn)),
        "moe_shared_ffn_inner_pack": (False, f32(n, shared_ffn)),
        "moe_shared_ffn_out_pack": (False, f32(n, hidden)),
        "final_logits_pack": (False, f32(1)),
        "moe_inner_pack_fallback_growth": (True, f32(slots, expert_ffn)),
        "moe_expert_out_pack_fallback_growth": (True, f32(slots, hidden)),
    }
    return allocations


def validate_plan(
    decision: dict[str, object],
    profile: dict[str, object],
) -> None:
    plan = decision.get("plan")
    if not isinstance(plan, dict):
        raise RuntimeError("admitted candidate plan is missing")
    expected = {
        "block_size": profile["outer_chunk"],
        "matrix_max_pos": EXPECTED_PROMPT_TOKENS,
        "matrix_query_rows": profile["query_rows"],
        "overlay": profile["overlay"],
    }
    for key, value in expected.items():
        if plan.get(key) != value:
            raise RuntimeError(f"candidate plan {key} drifted")
    allocations = plan.get("allocations")
    if not isinstance(allocations, list) or not allocations:
        raise RuntimeError("candidate plan allocations are missing")
    if any(not isinstance(allocation, dict) for allocation in allocations):
        raise RuntimeError("candidate plan allocation is malformed")
    expected_allocations = expected_plan_allocations(profile)
    if [allocation.get("name") for allocation in allocations] != list(
        expected_allocations
    ):
        raise RuntimeError("candidate allocation inventory or order drifted")
    names = []
    eager_count = 0
    deferred_count = 0
    eager_logical = 0
    deferred_logical = 0
    priced_total = 0
    for allocation in allocations:
        name = allocation.get("name")
        if not isinstance(name, str) or not name:
            raise RuntimeError("candidate allocation name is invalid")
        logical = u64(allocation.get("logical_bytes"), f"{name} logical bytes")
        priced = u64(allocation.get("priced_bytes"), f"{name} priced bytes")
        alignment = u64(allocation.get("alignment"), f"{name} alignment")
        deferred = allocation.get("deferred")
        if not isinstance(deferred, bool):
            raise RuntimeError("candidate allocation deferred flag is invalid")
        if (deferred, logical) != expected_allocations[name]:
            raise RuntimeError(f"candidate allocation {name} logical geometry drifted")
        if priced <= 0 or priced < logical:
            raise RuntimeError(f"candidate allocation {name} price is invalid")
        if alignment <= 0 or alignment & (alignment - 1):
            raise RuntimeError(f"candidate allocation {name} alignment is invalid")
        names.append(name)
        priced_total = checked_add(priced_total, priced, "priced allocation total")
        if deferred:
            deferred_count += 1
            deferred_logical = checked_add(
                deferred_logical,
                logical,
                "deferred logical total",
            )
        else:
            eager_count += 1
            eager_logical = checked_add(eager_logical, logical, "eager logical total")
    if len(names) != len(set(names)):
        raise RuntimeError("candidate allocation names are not unique")
    reconciled = {
        "eager_allocation_count": eager_count,
        "deferred_allocation_count": deferred_count,
        "eager_logical_bytes": eager_logical,
        "deferred_logical_bytes": deferred_logical,
        "maximum_logical_bytes": checked_add(
            eager_logical,
            deferred_logical,
            "maximum logical total",
        ),
        "priced_upper_bytes": priced_total,
    }
    for key, value in reconciled.items():
        if plan.get(key) != value:
            raise RuntimeError(f"candidate plan {key} does not reconcile")


def validate_metal_allocation_samples(row: dict[str, object]) -> dict[str, object]:
    allocated = row.get("metal_allocated")
    if not isinstance(allocated, dict):
        raise RuntimeError("Metal allocation samples are missing")
    sample_names = (
        "process_model_ready",
        "request_start",
        "after_scratch",
        "after_sequence",
        "after_prefill",
        "after_first_stdout_flush",
        "request_end_before_state_drop",
        "after_request_state_drop",
    )
    samples = {}
    for name in sample_names:
        sample = allocated.get(name)
        if not isinstance(sample, dict):
            raise RuntimeError(f"Metal allocation sample {name} is missing")
        current = u64(sample.get("current_bytes"), f"{name} current bytes")
        samples[name] = (sample, current)
    model_ready = samples["process_model_ready"][1]
    request_start = samples["request_start"][1]
    for name, (sample, current) in samples.items():
        expected_model_delta = current - model_ready
        expected_request_delta = current - request_start
        for key, expected in (
            ("delta_from_model_ready_bytes", expected_model_delta),
            ("delta_from_request_start_bytes", expected_request_delta),
        ):
            actual = sample.get(key)
            if isinstance(actual, bool) or not isinstance(actual, int):
                raise RuntimeError(f"Metal allocation sample {name}/{key} is invalid")
            if actual != expected:
                raise RuntimeError(f"Metal allocation sample {name}/{key} drifted")
    sampled_max = u64(
        allocated.get("current_allocated_sampled_max_bytes"),
        "sampled maximum Metal allocation",
    )
    if sampled_max != max(current for _, current in samples.values()):
        raise RuntimeError("sampled maximum Metal allocation does not reconcile")
    return allocated


def validate_admission(
    row: dict[str, object],
    decision: dict[str, object],
) -> None:
    plan = decision["plan"]
    admission = decision.get("admission")
    if not isinstance(admission, dict):
        raise RuntimeError("candidate admission is missing")
    before = admission.get("current_allocated_before_sequence")
    after = admission.get("current_allocated_after_sequence")
    delta = admission.get("sequence_allocation_delta_bytes")
    transient = admission.get("transient_reserve_bytes")
    reserve = admission.get("reserve_bytes")
    required = admission.get("required_bytes")
    before = u64(before, "before sequence")
    after = u64(after, "after sequence")
    delta = u64(delta, "sequence delta")
    transient = u64(transient, "transient reserve")
    reserve = u64(reserve, "reserve")
    required = u64(required, "required bytes")
    if after <= before or delta != after - before:
        raise RuntimeError("sequence allocation delta does not reconcile")
    if transient != TRANSIENT_RESERVE_BYTES or reserve != checked_add(
        delta,
        transient,
        "candidate reserve",
    ):
        raise RuntimeError("candidate reserve does not reconcile")
    if required != checked_add(
        u64(plan["priced_upper_bytes"], "priced upper bytes"),
        reserve,
        "candidate required bytes",
    ):
        raise RuntimeError("candidate required bytes do not reconcile")
    signals = admission.get("signals")
    if not isinstance(signals, dict):
        raise RuntimeError("candidate memory signals are missing")
    recommended = signals.get("recommended_max_bytes")
    current = signals.get("current_allocated_bytes")
    process = signals.get("process_limit_remaining_bytes")
    headroom = signals.get("working_set_headroom_bytes")
    recommended = u64(recommended, "recommended maximum bytes")
    current = u64(current, "current allocated bytes")
    if process is not None:
        process = u64(process, "process limit remaining bytes")
    headroom = u64(headroom, "working-set headroom bytes")
    if recommended <= 0 or current != after or headroom != recommended - current:
        raise RuntimeError("candidate working-set headroom does not reconcile")
    if headroom <= 0 or required > headroom:
        raise RuntimeError("candidate memory admission did not fit")
    if process == 0:
        expected_reason = "admitted_process_budget_omitted"
    elif process is not None and required <= process:
        expected_reason = "admitted_with_process_budget"
    else:
        raise RuntimeError("candidate process budget did not fit")
    if admission.get("evaluator_reason") != expected_reason:
        raise RuntimeError("candidate admission reason drifted")
    if admission.get("admitted") is not True:
        raise RuntimeError("candidate admission verdict drifted")
    allocated = validate_metal_allocation_samples(row)
    after_sequence = allocated.get("after_sequence")
    if (
        not isinstance(after_sequence, dict)
        or after_sequence.get("current_bytes") != after
    ):
        raise RuntimeError("sequence admission and timing samples disagree")


def validate_row(
    row: dict[str, object],
    profile: dict[str, object],
    arm: str,
    manifest: dict[str, object],
) -> None:
    validate_common_row(row, profile, manifest)
    if arm == "A":
        if row.get("schema_version") != 3:
            raise RuntimeError("numeric baseline schema drifted")
        if row.get("prefill_chunk_requested") != 1024:
            raise RuntimeError("numeric baseline request width drifted")
        if row.get("prefill_chunk_effective") != 1024:
            raise RuntimeError("numeric baseline effective width drifted")
        for key in (
            "prefill_chunk_decision",
            "prefill_attention_query",
            "prefill_scratch_overlay",
        ):
            if key in row:
                raise RuntimeError(f"numeric baseline unexpectedly reports {key}")
        return

    if row.get("schema_version") != 5:
        raise RuntimeError("automatic candidate schema drifted")
    if row.get("prefill_chunk_requested") != "auto":
        raise RuntimeError("automatic request label drifted")
    if row.get("prefill_chunk_effective") != profile["outer_chunk"]:
        raise RuntimeError("automatic effective width drifted")
    decision = row.get("prefill_chunk_decision")
    if not isinstance(decision, dict):
        raise RuntimeError("automatic decision is missing")
    expected_decision = {
        "policy": "moe_allowlist_v2",
        "profile": profile["profile"],
        "classification": "candidate",
        "reason": "admitted",
        "candidate": profile["outer_chunk"],
        "selected": profile["outer_chunk"],
        "validated_prompt_range": [8192, 16384],
        "evidence_baseline_chunk": 1024,
        "baseline": "legacy_outer_1024_matrix_max_pos_v1",
    }
    for key, value in expected_decision.items():
        if decision.get(key) != value:
            raise RuntimeError(f"automatic decision {key} drifted")
    validate_plan(decision, profile)
    validate_admission(row, decision)
    if row.get("prefill_scratch_overlay") != profile["overlay"]:
        raise RuntimeError("observed scratch overlay drifted")
    query = row.get("prefill_attention_query")
    expected_query = {
        "outer_chunk_rows": profile["outer_chunk"],
        "query_rows": profile["query_rows"],
        "tiled_layer_calls": profile["layer_calls"],
        "query_tile_calls": profile["query_tile_calls"],
    }
    if query != expected_query:
        raise RuntimeError("observed query topology drifted")


def run_one(
    profile_name: str,
    pair_index: int,
    run_index: int,
    order: str,
    arm: str,
    base_env: dict[str, str],
    manifest: dict[str, object],
) -> dict[str, object]:
    profile = PROFILES[profile_name]
    environment = child_environment_record(base_env)
    if environment != manifest["child_environment"]:
        raise RuntimeError("child environment drifted")
    stem = f"{profile_name}-p{pair_index:02d}-r{run_index}-{arm.lower()}"
    stdout_path = ARTIFACT / f"{stem}.out"
    stderr_path = ARTIFACT / f"{stem}.err"
    timing_path = ARTIFACT / f"{stem}.jsonl"
    for path in (stdout_path, stderr_path, timing_path):
        if path.exists():
            raise RuntimeError(f"refusing to reuse child artifact {path}")

    host_before_cache = common.wait_for_valid_host(f"{stem} cache read")
    vm_before_cache = capture_vm_state()
    cache_ms, cache_bytes = warm_files(profile["shards"])
    expected_cache_bytes = sum(path.stat().st_size for path in profile["shards"])
    if cache_bytes != expected_cache_bytes:
        raise RuntimeError("cache precondition did not read every model byte")
    cooldown_s = finite_number(profile["cooldown_s"], "cooldown", nonnegative=True)
    cooldown_started_unix_ms = time.time_ns() // 1_000_000
    cooldown_started = time.perf_counter()
    time.sleep(cooldown_s)
    cooldown_elapsed_ms = (time.perf_counter() - cooldown_started) * 1e3
    if cooldown_elapsed_ms < cooldown_s * 1e3:
        raise RuntimeError("cooldown elapsed for less than its frozen duration")
    host_before_spawn = common.wait_for_valid_host(f"{stem} process spawn")
    vm_before_spawn = capture_vm_state()
    cache_pageout_delta = vm_before_spawn["pageouts"] - vm_before_cache["pageouts"]
    cache_swap_delta = (
        vm_before_spawn["swap_used_bytes"] - vm_before_cache["swap_used_bytes"]
    )
    if cache_pageout_delta != 0 or cache_swap_delta > 0:
        raise InconclusivePacket(
            profile_name,
            pair_index,
            f"{stem} cache interval changed VM pressure",
        )

    command = [
        "/usr/bin/time",
        "-l",
        str(BINARY),
        "-m",
        str(profile["model"]),
        "--prompt-file",
        str(PROMPT),
        "--tokens",
        "1",
        "--no-special-tokens",
        "--prefix-cache-max-mib",
        "16384",
        "--request-timings",
        str(timing_path),
        "--prefill-chunk",
        "1024" if arm == "A" else "auto",
    ]
    started = time.perf_counter()
    with stdout_path.open("xb") as stdout_file, stderr_path.open("xb") as stderr_file:
        process = subprocess.run(
            command,
            cwd=ROOT,
            env=base_env,
            stdout=stdout_file,
            stderr=stderr_file,
            check=False,
        )
    process_wall_ms = (time.perf_counter() - started) * 1e3
    host_after_exit = common.capture_host_state()
    vm_after_exit = capture_vm_state()
    process_pageout_delta = vm_after_exit["pageouts"] - vm_before_spawn["pageouts"]
    process_swap_delta = (
        vm_after_exit["swap_used_bytes"] - vm_before_spawn["swap_used_bytes"]
    )
    stderr = stderr_path.read_text(encoding="utf-8", errors="replace")
    output_bytes = stdout_path.read_bytes()
    validity_reasons = []
    if process.returncode != 0:
        validity_reasons.append(f"child_exit={process.returncode}")
    resources = {}
    for key, label in (
        ("block_input_operations", "block input operations"),
        ("page_faults", "page faults"),
        ("page_reclaims", "page reclaims"),
        ("maximum_resident_set_size", "maximum resident set size"),
        ("peak_memory_footprint", "peak memory footprint"),
    ):
        try:
            resources[key] = common.parse_resource(stderr, label)
        except RuntimeError as error:
            resources[key] = None
            validity_reasons.append(f"resource_parse={error}")

    row = None
    if not timing_path.is_file():
        validity_reasons.append("timing_row=missing")
    else:
        try:
            timing_lines = timing_path.read_text(encoding="utf-8").splitlines()
            if len(timing_lines) != 1:
                raise RuntimeError(f"produced {len(timing_lines)} timing rows")
            parsed = parse_json(timing_lines[0])
            if not isinstance(parsed, dict):
                raise RuntimeError("timing row is not an object")
            row = parsed
            validate_row(row, profile, arm, manifest)
        except (RuntimeError, ValueError, json.JSONDecodeError) as error:
            validity_reasons.append(f"timing_row={error}")
    block_inputs = resources["block_input_operations"]
    if block_inputs != 0:
        validity_reasons.append(f"block_input_operations={block_inputs}")
    if process_pageout_delta != 0:
        validity_reasons.append(f"process_pageout_delta={process_pageout_delta}")
    if process_swap_delta > 0:
        validity_reasons.append(f"process_swap_growth_bytes={process_swap_delta}")
    if not host_after_exit["valid"]:
        validity_reasons.append("post_exit_host_invalid")
    finite_number(process_wall_ms, "process wall", positive=True)
    finite_number(cache_ms, "cache precondition wall", positive=True)
    for key in ("page_faults", "page_reclaims"):
        if resources[key] is not None:
            try:
                finite_number(resources[key], key, nonnegative=True)
            except RuntimeError as error:
                validity_reasons.append(f"resource_value={error}")
    for key in ("maximum_resident_set_size", "peak_memory_footprint"):
        if resources[key] is not None:
            try:
                finite_number(resources[key], key, positive=True)
            except RuntimeError as error:
                validity_reasons.append(f"resource_value={error}")
    return {
        "artifact_stem": stem,
        "profile": profile_name,
        "pair_index": pair_index,
        "pair_order": order,
        "run_index": run_index,
        "arm": arm,
        "command": command,
        "child_environment": environment,
        "cache_precondition_ms": cache_ms,
        "cache_precondition_bytes": cache_bytes,
        "cooldown_started_unix_ms": cooldown_started_unix_ms,
        "cooldown_requested_ms": cooldown_s * 1e3,
        "cooldown_elapsed_ms": cooldown_elapsed_ms,
        "host_before_cache": host_before_cache,
        "host_before_spawn": host_before_spawn,
        "host_after_exit": host_after_exit,
        "vm_before_cache": vm_before_cache,
        "vm_before_spawn": vm_before_spawn,
        "vm_after_exit": vm_after_exit,
        "cache_pageout_delta": cache_pageout_delta,
        "cache_swap_delta_bytes": cache_swap_delta,
        "process_pageout_delta": process_pageout_delta,
        "process_swap_delta_bytes": process_swap_delta,
        "child_returncode": process.returncode,
        "process_wall_ms": process_wall_ms,
        **resources,
        "output_bytes": len(output_bytes),
        "output_sha256": hashlib.sha256(output_bytes).hexdigest(),
        "valid": not validity_reasons,
        "validity_reasons": validity_reasons,
        "result": row,
    }


def append_row(path: Path, row: dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as output:
        output.write(json_text(row) + "\n")


def run_pair(
    profile_name: str,
    pair_index: int,
    order: str,
    base_env: dict[str, str],
    manifest: dict[str, object],
    attempts_path: Path,
    pairs_path: Path,
) -> list[dict[str, object]]:
    rows = []
    for run_index, arm in enumerate(order, 1):
        row = run_one(
            profile_name,
            pair_index,
            run_index,
            order,
            arm,
            base_env,
            manifest,
        )
        append_row(attempts_path, row)
        rows.append(row)
        if not row["valid"]:
            append_row(
                pairs_path,
                {
                    "profile": profile_name,
                    "pair_index": pair_index,
                    "pair_order": order,
                    "accepted": False,
                    "complete": False,
                    "artifact_stems": [item["artifact_stem"] for item in rows],
                    "failed_arm": arm,
                    "reasons": row["validity_reasons"],
                },
            )
            raise InconclusivePacket(
                profile_name,
                pair_index,
                f"invalid arm {profile_name}/{pair_index}/{arm}",
            )
    output_equal = (
        rows[0]["output_sha256"] == rows[1]["output_sha256"]
        and rows[0]["output_bytes"] == rows[1]["output_bytes"]
    )
    runtime_equal = all(
        rows[0]["result"].get(key) == rows[1]["result"].get(key)
        for key in ("runtime_identity_kind", "runtime_model_id", "runtime_tokenizer_id")
    )
    accepted = output_equal and runtime_equal
    pair_row = {
        "profile": profile_name,
        "pair_index": pair_index,
        "pair_order": order,
        "accepted": accepted,
        "complete": True,
        "artifact_stems": [row["artifact_stem"] for row in rows],
        "output_equal": output_equal,
        "runtime_identity_equal": runtime_equal,
        "invalid_rows": [
            {"arm": row["arm"], "reasons": row["validity_reasons"]}
            for row in rows
            if not row["valid"]
        ],
    }
    append_row(pairs_path, pair_row)
    if not accepted:
        raise InconclusivePacket(
            profile_name,
            pair_index,
            f"invalid fixed pair {profile_name}/{pair_index}",
        )
    return rows


def analyze_profile(
    profile_name: str,
    rows: list[dict[str, object]],
) -> dict[str, object]:
    profile = PROFILES[profile_name]
    pairs = []
    output_identities = set()
    runtime_identities = set()
    for pair_index, order in enumerate(PAIR_ORDERS, 1):
        selected = [
            row
            for row in rows
            if row["profile"] == profile_name
            and row["pair_index"] == pair_index
            and row["pair_order"] == order
        ]
        if len(selected) != 2 or {row["arm"] for row in selected} != {"A", "B"}:
            raise RuntimeError(
                f"accepted pair {profile_name}/{pair_index} is incomplete"
            )
        pair = {row["arm"]: row for row in selected}
        pairs.append(pair)
        for row in selected:
            output_identities.add((row["output_bytes"], row["output_sha256"]))
            result = row["result"]
            runtime_identities.add(
                (
                    result["runtime_identity_kind"],
                    result["runtime_model_id"],
                    result["runtime_tokenizer_id"],
                )
            )
    if len(output_identities) != 1 or len(runtime_identities) != 1:
        raise RuntimeError(f"cross-pair identity drift for {profile_name}")

    speedups = []
    wins = 0
    strata = {"AB": [], "BA": []}
    ttft = {"A": [], "B": []}
    for pair, order in zip(pairs, PAIR_ORDERS, strict=True):
        baseline = finite_number(
            pair["A"]["result"]["ttft_ms"], "A TTFT", positive=True
        )
        candidate = finite_number(
            pair["B"]["result"]["ttft_ms"], "B TTFT", positive=True
        )
        speedup = baseline / candidate
        finite_number(speedup, "paired TTFT speedup", positive=True)
        speedups.append(speedup)
        strata[order].append(speedup)
        wins += candidate < baseline
        ttft["A"].append(baseline)
        ttft["B"].append(candidate)
    median_speedup = statistics.median(speedups)
    stratum_medians = {
        order: statistics.median(values) for order, values in strata.items()
    }
    gates = {
        "median_speedup": median_speedup >= profile["median_gate"],
        "ab_speedup": stratum_medians["AB"] >= profile["stratum_gate"],
        "ba_speedup": stratum_medians["BA"] >= profile["stratum_gate"],
        "wins_4_of_4": wins == 4,
    }
    return {
        "status": "go" if all(gates.values()) else "kill",
        "qualifies": all(gates.values()),
        "speedups_by_pair": speedups,
        "median_speedup": median_speedup,
        "stratum_speedups": strata,
        "stratum_median_speedup": stratum_medians,
        "wins": wins,
        "median_ttft_ms": {
            arm: statistics.median(values) for arm, values in ttft.items()
        },
        "output_identity": list(output_identities)[0],
        "runtime_identity": list(runtime_identities)[0],
        "gates": gates,
    }


def analyze(
    rows: list[dict[str, object]],
    manifest: dict[str, object],
) -> dict[str, object]:
    profiles = {
        profile_name: analyze_profile(profile_name, rows) for profile_name in PROFILES
    }
    qualified = [name for name, result in profiles.items() if result["qualifies"]]
    if len(qualified) == len(PROFILES):
        status = "go"
    elif qualified:
        status = "mixed"
    else:
        status = "kill"
    return {
        "schema": 1,
        "status": status,
        "authority": {
            name: "separate-profile-default-decision" if name in qualified else "none"
            for name in PROFILES
        },
        "source_commit": manifest["source_commit"],
        "accepted_pairs": len(rows) // 2,
        "profiles": profiles,
    }


def main() -> None:
    if ARTIFACT.exists():
        raise RuntimeError(f"refusing to reuse packet directory {ARTIFACT}")
    base_env, removed_environment = common.normalized_environment()
    manifest = build_manifest(removed_environment, base_env)
    ARTIFACT.mkdir(parents=True)
    manifest_path = ARTIFACT / "manifest.json"
    attempts_path = ARTIFACT / "attempts.jsonl"
    pairs_path = ARTIFACT / "pairs.jsonl"
    decision_path = ARTIFACT / "decision.json"
    manifest_path.write_text(json_text(manifest, pretty=True) + "\n", encoding="utf-8")
    accepted_rows = []
    try:
        for profile_name in PROFILES:
            for pair_index, order in enumerate(PAIR_ORDERS, 1):
                accepted_rows.extend(
                    run_pair(
                        profile_name,
                        pair_index,
                        order,
                        base_env,
                        manifest,
                        attempts_path,
                        pairs_path,
                    )
                )
    except InconclusivePacket as error:
        identity_error = completion_identity_error(manifest)
        decision = {
            "schema": 1,
            "status": "inconclusive",
            "authority": "none",
            "source_commit": manifest["source_commit"],
            "failed_profile": error.profile,
            "failed_pair_index": error.pair_index,
            "reason": str(error),
            "completion_identity_verified": identity_error is None,
            "completion_identity_error": identity_error,
        }
        decision_path.write_text(
            json_text(decision, pretty=True) + "\n",
            encoding="utf-8",
        )
        print(json_text(decision, pretty=True))
        raise SystemExit(2)
    except Exception as error:
        identity_error = completion_identity_error(manifest)
        decision = {
            "schema": 1,
            "status": "infrastructure_error",
            "authority": "none",
            "source_commit": manifest["source_commit"],
            "reason": f"{type(error).__name__}: {error}",
            "completion_identity_verified": identity_error is None,
            "completion_identity_error": identity_error,
        }
        decision_path.write_text(
            json_text(decision, pretty=True) + "\n",
            encoding="utf-8",
        )
        print(json_text(decision, pretty=True))
        raise SystemExit(2)
    identity_error = completion_identity_error(manifest)
    if identity_error is not None:
        decision = {
            "schema": 1,
            "status": "identity_error",
            "authority": "none",
            "source_commit": manifest["source_commit"],
            "reason": identity_error,
            "completion_identity_verified": False,
        }
        decision_path.write_text(
            json_text(decision, pretty=True) + "\n",
            encoding="utf-8",
        )
        print(json_text(decision, pretty=True))
        raise SystemExit(2)
    try:
        decision = analyze(accepted_rows, manifest)
    except Exception as error:
        decision = {
            "schema": 1,
            "status": "analysis_error",
            "authority": "none",
            "source_commit": manifest["source_commit"],
            "reason": f"{type(error).__name__}: {error}",
            "completion_identity_verified": True,
        }
        decision_path.write_text(
            json_text(decision, pretty=True) + "\n",
            encoding="utf-8",
        )
        print(json_text(decision, pretty=True))
        raise SystemExit(2)
    decision["completion_identity_verified"] = True
    decision_path.write_text(json_text(decision, pretty=True) + "\n", encoding="utf-8")
    print(json_text(decision, pretty=True))


if __name__ == "__main__":
    main()
