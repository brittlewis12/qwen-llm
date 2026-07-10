#!/usr/bin/env python3
"""Causally simulate prompt-lookup proposals from frozen target token IDs."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


SCHEMA = "proposal-economics/v1"


@dataclass(frozen=True)
class Candidate:
    source: str
    source_start: int
    source_end: int
    absolute_source_end: int
    match_len: int
    proposal: tuple[int, ...]


@dataclass(frozen=True)
class Policy:
    source_mode: str
    selector: str
    match_tokens: int
    proposal_tokens: int

    @property
    def policy_id(self) -> str:
        return (
            f"pld-{self.source_mode}-{self.selector}-"
            f"l{self.match_tokens}-d{self.proposal_tokens}"
        )


@dataclass
class Simulation:
    policy: Policy
    events: list[dict]
    summary: dict


def load_fixture(path: Path) -> tuple[dict, list[int], list[int]]:
    row = json.loads(path.read_text())
    fixture = row.get("token_fixture")
    if not isinstance(fixture, dict) or fixture.get("schema_version") != 1:
        raise SystemExit(f"{path}: missing token_fixture schema 1")
    prompt = [int(token) for token in fixture.get("prompt_token_ids", [])]
    target = [int(token) for token in fixture.get("target_generated_token_ids", [])]
    if len(prompt) != int(row.get("prompt_tokens", -1)):
        raise SystemExit(f"{path}: prompt token count mismatch")
    if len(target) != int(row.get("reference", {}).get("emitted", -1)):
        raise SystemExit(f"{path}: target token count mismatch")
    if not target:
        raise SystemExit(f"{path}: target token stream is empty")
    if row.get("semantics", {}).get("sampler") != "greedy_argmax":
        raise SystemExit(f"{path}: only greedy_argmax fixtures are supported")
    return row, prompt, target


def build_prompt_index(
    prompt: list[int], match_tokens: int, proposal_tokens: int
) -> dict[tuple[int, ...], list[int]]:
    index: dict[tuple[int, ...], list[int]] = defaultdict(list)
    last_end = len(prompt) - proposal_tokens
    for end in range(match_tokens, last_end + 1):
        index[tuple(prompt[end - match_tokens : end])].append(end)
    return index


def backward_match_len(visible: list[int], source: list[int], source_end: int) -> int:
    visible_index = len(visible) - 1
    source_index = source_end - 1
    matched = 0
    while (
        visible_index >= 0
        and source_index >= 0
        and visible[visible_index] == source[source_index]
    ):
        matched += 1
        visible_index -= 1
        source_index -= 1
    return matched


def z_values(values: list[object]) -> list[int]:
    result = [0] * len(values)
    left = 0
    right = 0
    for index in range(1, len(values)):
        if index < right:
            result[index] = min(right - index, result[index - left])
        while (
            index + result[index] < len(values)
            and values[result[index]] == values[index + result[index]]
        ):
            result[index] += 1
        if index + result[index] > right:
            left = index
            right = index + result[index]
    return result


def backward_match_lengths(
    visible: list[int], source: list[int], source_ends: list[int]
) -> dict[int, int]:
    sentinel = object()
    reversed_visible: list[object] = list(reversed(visible))
    combined = reversed_visible + [sentinel] + list(reversed(source))
    matches = z_values(combined)
    source_offset = len(reversed_visible) + 1
    return {
        end: min(matches[source_offset + len(source) - end], len(visible), end)
        for end in source_ends
    }


def best_source_candidate(
    source_name: str,
    source: list[int],
    visible: list[int],
    source_ends: list[int],
    proposal_tokens: int,
    absolute_offset: int,
    selector: str,
) -> Candidate | None:
    if not source_ends:
        return None
    if selector == "recent":
        end = source_ends[-1]
        match_len = backward_match_len(visible, source, end)
    elif selector == "longest-recent":
        match_lengths = backward_match_lengths(visible, source, source_ends)
        end = max(source_ends, key=lambda item: (match_lengths[item], item))
        match_len = match_lengths[end]
    else:
        raise ValueError(f"unknown selector: {selector}")
    return Candidate(
        source=source_name,
        source_start=end - match_len,
        source_end=end,
        absolute_source_end=absolute_offset + end,
        match_len=match_len,
        proposal=tuple(source[end : end + proposal_tokens]),
    )


def best_prompt_candidate(
    prompt: list[int],
    committed_output: list[int],
    match_tokens: int,
    proposal_tokens: int,
    prompt_index: dict[tuple[int, ...], list[int]],
    selector: str,
) -> Candidate | None:
    visible = prompt + committed_output
    if len(visible) < match_tokens:
        return None
    key = tuple(visible[-match_tokens:])
    return best_source_candidate(
        "prompt",
        prompt,
        visible,
        prompt_index.get(key, []),
        proposal_tokens,
        0,
        selector,
    )


def best_self_candidate(
    prompt_len: int,
    committed_output: list[int],
    match_tokens: int,
    proposal_tokens: int,
    selector: str,
) -> Candidate | None:
    if len(committed_output) < match_tokens + proposal_tokens:
        return None
    key = tuple(committed_output[-match_tokens:])
    last_end = len(committed_output) - proposal_tokens
    source_ends = [
        end
        for end in range(match_tokens, last_end + 1)
        if tuple(committed_output[end - match_tokens : end]) == key
    ]
    return best_source_candidate(
        "self",
        committed_output,
        committed_output,
        source_ends,
        proposal_tokens,
        prompt_len,
        selector,
    )


def select_candidate(candidates: list[Candidate], selector: str) -> Candidate | None:
    if not candidates:
        return None
    if selector == "recent":
        return max(candidates, key=lambda item: item.absolute_source_end)
    if selector == "longest-recent":
        return max(
            candidates,
            key=lambda item: (item.match_len, item.absolute_source_end),
        )
    raise ValueError(f"unknown selector: {selector}")


def accepted_prefix(proposal: Iterable[int], target: Iterable[int]) -> int:
    accepted = 0
    for proposed, expected in zip(proposal, target):
        if proposed != expected:
            break
        accepted += 1
    return accepted


def find_candidate(
    prompt: list[int],
    target: list[int],
    carry_index: int,
    policy: Policy,
    prompt_index: dict[tuple[int, ...], list[int]],
) -> Candidate | None:
    committed_output = target[: carry_index + 1]
    candidates: list[Candidate] = []
    if policy.source_mode in {"prompt", "both"}:
        prompt_candidate = best_prompt_candidate(
            prompt,
            committed_output,
            policy.match_tokens,
            policy.proposal_tokens,
            prompt_index,
            policy.selector,
        )
        if prompt_candidate is not None:
            candidates.append(prompt_candidate)
    if policy.source_mode in {"self", "both"}:
        self_candidate = best_self_candidate(
            len(prompt),
            committed_output,
            policy.match_tokens,
            policy.proposal_tokens,
            policy.selector,
        )
        if self_candidate is not None:
            candidates.append(self_candidate)
    return select_candidate(candidates, policy.selector)


def simulate(
    fixture_id: str,
    split: str,
    prompt: list[int],
    target: list[int],
    policy: Policy,
    serial_transition_ms: float,
    verify_packet_ms: float,
) -> Simulation:
    prompt_index = build_prompt_index(
        prompt, policy.match_tokens, policy.proposal_tokens
    )
    events: list[dict] = []
    carry_index = 0
    event_index = 0
    lookup_ns = 0
    while carry_index < len(target):
        if carry_index == len(target) - 1:
            events.append(
                {
                    "schema": SCHEMA,
                    "row_type": "event",
                    "fixture_id": fixture_id,
                    "split": split,
                    "policy_id": policy.policy_id,
                    "event": event_index,
                    "carry_index": carry_index,
                    "action": "terminal_emit",
                    "attempted": False,
                    "source": None,
                    "match_len": None,
                    "proposal_token_ids": [],
                    "target_token_ids": [],
                    "accepted_prefix_len": 0,
                    "progress": 1,
                    "cost_ms": 0.0,
                    "cost_class": "terminal_emit",
                }
            )
            carry_index += 1
            break

        started = time.perf_counter_ns()
        candidate = find_candidate(prompt, target, carry_index, policy, prompt_index)
        lookup_ns += time.perf_counter_ns() - started
        if candidate is None:
            events.append(
                {
                    "schema": SCHEMA,
                    "row_type": "event",
                    "fixture_id": fixture_id,
                    "split": split,
                    "policy_id": policy.policy_id,
                    "event": event_index,
                    "carry_index": carry_index,
                    "action": "abstain",
                    "attempted": False,
                    "source": None,
                    "match_len": None,
                    "proposal_token_ids": [],
                    "target_token_ids": [target[carry_index + 1]],
                    "accepted_prefix_len": 0,
                    "progress": 1,
                    "cost_ms": serial_transition_ms,
                    "cost_class": "serial_transition",
                }
            )
            carry_index += 1
            event_index += 1
            continue

        target_tokens = target[
            carry_index + 1 : carry_index + 1 + policy.proposal_tokens
        ]
        scored_proposal = candidate.proposal[: len(target_tokens)]
        terminal_window = carry_index + 1 + len(target_tokens) == len(target)
        accepted = accepted_prefix(scored_proposal, target_tokens)
        progress = 1 + accepted
        events.append(
            {
                "schema": SCHEMA,
                "row_type": "event",
                "fixture_id": fixture_id,
                "split": split,
                "policy_id": policy.policy_id,
                "event": event_index,
                "carry_index": carry_index,
                "action": "attempt",
                "attempted": True,
                "source": candidate.source,
                "source_start": candidate.source_start,
                "source_end": candidate.source_end,
                "match_len": candidate.match_len,
                "proposal_token_ids": list(candidate.proposal),
                "eligible_draft_count": len(scored_proposal),
                "scored_proposal_token_ids": list(scored_proposal),
                "output_limited": len(scored_proposal) < len(candidate.proposal),
                "terminal_window": terminal_window,
                "target_token_ids": target_tokens,
                "accepted_prefix_len": accepted,
                "progress": progress,
                "effective_verify_n": (
                    len(target_tokens)
                    if terminal_window
                    else policy.proposal_tokens + 1
                ),
                "cost_ms": verify_packet_ms,
                "cost_class": (f"fixed_n{policy.proposal_tokens + 1}_optimistic"),
            }
        )
        carry_index += progress
        event_index += 1

    attempts = [event for event in events if event["attempted"]]
    abstentions = [event for event in events if event["action"] == "abstain"]
    accepted = sum(int(event["accepted_prefix_len"]) for event in attempts)
    eligible = sum(int(event["eligible_draft_count"]) for event in attempts)
    proposer_drafts = sum(len(event["proposal_token_ids"]) for event in attempts)
    source_counts = Counter(str(event["source"]) for event in attempts)
    baseline_ms = max(0, len(target) - 1) * serial_transition_ms
    predicted_ms = sum(float(event["cost_ms"]) for event in events)
    speedup = baseline_ms / predicted_ms if predicted_ms > 0 else None
    summary = {
        "schema": SCHEMA,
        "row_type": "policy_summary",
        "fixture_id": fixture_id,
        "split": split,
        "policy_id": policy.policy_id,
        "policy": asdict(policy),
        "target_tokens": len(target),
        "attempts": len(attempts),
        "abstentions": len(abstentions),
        "attempt_rate": len(attempts) / max(1, len(attempts) + len(abstentions)),
        "accepted_drafts": accepted,
        "eligible_drafts": eligible,
        "proposer_drafts": proposer_drafts,
        "acceptance_rate": accepted / eligible if eligible else None,
        "accepted_run_histogram": dict(
            sorted(Counter(event["accepted_prefix_len"] for event in attempts).items())
        ),
        "mean_progress_per_attempt": (
            sum(int(event["progress"]) for event in attempts) / len(attempts)
            if attempts
            else None
        ),
        "source_attempts": dict(sorted(source_counts.items())),
        "baseline_decode_ms": baseline_ms,
        "optimistic_predicted_decode_ms": predicted_ms,
        "optimistic_decode_speedup": speedup,
        "cost_model": (
            f"fixed_all_accepted_n{policy.proposal_tokens + 1}_without_"
            "restore_or_proposer_cost"
        ),
        "lookup_cpu_ms_python": lookup_ns / 1_000_000.0,
    }
    return Simulation(policy=policy, events=events, summary=summary)


def policies(args: argparse.Namespace) -> Iterable[Policy]:
    for source_mode in args.sources:
        for selector in args.selectors:
            for match_tokens in args.match_tokens:
                yield Policy(
                    source_mode=source_mode,
                    selector=selector,
                    match_tokens=match_tokens,
                    proposal_tokens=args.proposal_tokens,
                )


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("fixture", type=Path)
    parser.add_argument("--fixture-id")
    parser.add_argument(
        "--split", choices=["development", "heldout", "guardrail"], required=True
    )
    parser.add_argument("--serial-transition-ms", type=float, required=True)
    parser.add_argument("--verify-packet-ms", type=float, required=True)
    parser.add_argument("--proposal-tokens", type=int, default=7)
    parser.add_argument("--match-tokens", type=int, nargs="+", default=[8, 16, 32])
    parser.add_argument(
        "--sources", nargs="+", choices=["prompt", "self", "both"], default=["both"]
    )
    parser.add_argument(
        "--selectors",
        nargs="+",
        choices=["recent", "longest-recent"],
        default=["longest-recent"],
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    if args.serial_transition_ms <= 0 or args.verify_packet_ms <= 0:
        parser.error("costs must be positive")
    if args.proposal_tokens <= 0:
        parser.error("--proposal-tokens must be positive")
    if any(value <= 0 for value in args.match_tokens):
        parser.error("--match-tokens values must be positive")
    return args


def main() -> None:
    args = parse_args()
    fixture_row, prompt, target = load_fixture(args.fixture)
    fixture_id = args.fixture_id or args.fixture.stem
    simulations = [
        simulate(
            fixture_id,
            args.split,
            prompt,
            target,
            policy,
            args.serial_transition_ms,
            args.verify_packet_ms,
        )
        for policy in policies(args)
    ]
    fixture_header = {
        "schema": SCHEMA,
        "row_type": "fixture",
        "fixture_id": fixture_id,
        "split": args.split,
        "source_path": str(args.fixture),
        "model": fixture_row.get("model"),
        "prompt_tokens": len(prompt),
        "target_tokens": len(target),
        "sampler": fixture_row.get("semantics", {}).get("sampler"),
        "stop_tokens": fixture_row.get("stop_tokens"),
        "serial_transition_ms": args.serial_transition_ms,
        "verify_packet_ms": args.verify_packet_ms,
    }
    jsonl_rows = [fixture_header]
    for simulation in simulations:
        jsonl_rows.append(simulation.summary)
        jsonl_rows.extend(simulation.events)
    write_jsonl(args.output, jsonl_rows)
    summary = {
        "schema": SCHEMA,
        "fixture": fixture_header,
        "policies": [simulation.summary for simulation in simulations],
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
