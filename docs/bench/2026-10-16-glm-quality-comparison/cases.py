"""Frozen public engineering cases; exact checks are only a partial quality signal.

No generated code is executed. Long-context sizes are characters/words, not
token counts: measure with the actual tokenizer before reporting token lengths.
"""

import json


SYSTEM = """Solve the engineering task using only the supplied specification.
Return one JSON object, without markdown or text outside the object, with keys:
"facts": an object using exactly the requested fact names and JSON types;
"code": a string containing the implementation, or an empty string for analysis;
"explanation": a string explaining correctness, edge cases, and limitations.
Do not replace code with pseudocode when implementation is requested.
The output budget is bounded; prioritize the complete final JSON answer.
"""


def _case(identifier, category, task, facts, rubric, *, code=False, budget=4096):
    names = json.dumps({k: type(v).__name__ for k, v in facts.items()}, sort_keys=True)
    return {
        "id": identifier,
        "category": category,
        "system": SYSTEM,
        "user": task + "\n\nRequired facts and types (str=JSON string, int=integer, "
        "bool=boolean, list=array): " + names + "\nReturn the specified JSON format.",
        "max_tokens": budget,
        "reference": {"facts": facts, "code_required": code},
        "rubric": {
            "objective_scope": "Partial exact-fact checks only; not a full quality grade. "
            "Presence of code is not evidence that code is correct.",
            "review": rubric,
        },
    }


def _documents(kind, start=0, stop=48):
    """Fixed synthetic archive, roughly 8K-16K tokens equivalent, not measured."""
    records = []
    for i in range(start, stop):
        if kind == "recovery":
            detail = (
                f"Draft {i:02d} suggests visible_seq={240 + i}, replay_from={241 + i}, "
                f"and action={'truncate' if i % 2 else 'replay'}. These numbers are "
                "hypotheses copied from a simulator, not measurements of the target log. "
                "A green dashboard means the observer process was reachable, not that "
                "the writer acknowledged durability. A segment may contain a complete "
                "record whose transaction is not committed. Retaining bytes does not "
                "mean exposing them to readers. The scratch replay tool used a different "
                "checkpoint and cannot establish a safe truncation boundary. The author "
                "asked for another export but did not attach one. No amendment to the "
                "authority rule was approved in this note."
            )
        else:
            detail = (
                f"Draft {i:02d} proposes generation={70 + i}, "
                f"west={'enabled' if i % 2 else 'disabled'}, and an unconditional cache put. "
                "The author tested a local fake loader that never overlaps invalidation. "
                "Its screenshot cannot establish whether a production request observed "
                "an old generation. A timed-out waiter is not proof that the shared "
                "load stopped. The dashboard counts responses, not loader executions. "
                "A restart may reset process memory but does not revise the durable "
                "control record. The experiment used a private key namespace and was "
                "never promoted. The owner requested a fenced publish test and an "
                "independent waiter cancellation test before adopting any draft. No "
                "amendment to the authority rule was approved in this note."
            )
        records.append(
            f"\nARCHIVE NOTE {i:02d}; status=DRAFT; display_time=2032-04-{1 + i % 28:02d}.\n"
            + detail
            + "\n"
            "Review checklist: retain original evidence, distinguish observed events "
            "from proposed repairs, and record the scope of every claim. This note "
            "is contextual evidence only. Repeated agreement among drafts is not "
            "an approval, and sorting by display time cannot resolve authority.\n"
        )
    return "".join(records)


def build_cases():
    """Return eight fresh dicts; no randomness, filesystem, clock, or network."""
    cases = [
        _case(
            "stream-lookahead",
            "coding/iterator",
            """Implement Python class Peekable(source) for a single-pass iterator.
peek() returns the next item without consuming it; __next__ consumes it;
__iter__ returns self. None is a valid item. Cache exhaustion so subsequent
peek/next calls raise StopIteration without touching source. Other exceptions
must propagate and must not be cached. Do not prefetch in __init__.
The test source's successive __next__ calls return None, raise ValueError,
return 7, then raise StopIteration. Run these operations independently catching
exceptions: peek, peek, next, peek, peek, next, next, peek.
facts.trace is an array using actual values or exception class-name strings;
facts.source_calls is total source calls; facts.prefetch_in_init is boolean.
Include the complete class in code; describe a unique sentinel or equivalent.""",
            {
                "trace": [
                    None,
                    None,
                    None,
                    "ValueError",
                    7,
                    7,
                    "StopIteration",
                    "StopIteration",
                ],
                "source_calls": 4,
                "prefetch_in_init": False,
            },
            [
                "Cached None and exhaustion are distinct states.",
                "Only StopIteration is cached; transient errors leave the buffer empty.",
                "Implementation remains lazy and obeys Python iteration protocol.",
            ],
            code=True,
        ),
        _case(
            "shared-load-cancellation",
            "coding/concurrency",
            """Implement Python 3.11 asyncio SingleFlight with async get(key) and
async close(), taking an async loader(key) in its constructor. No value cache:
overlapping callers share a task; a later call after completion starts a new one.
One waiter's cancellation must not cancel the task or other waiters. When all
waiters cancel, keep the task running until completion; retrieve its exception
even if nobody waits. Remove completed entries only if they still refer to that
task. close rejects new gets, cancels outstanding loader tasks and awaits them.
Assume one event loop, cooperative loaders, and close itself is not cancelled.
Timeline: A and B await key k's first blocked load; A is cancelled; loader returns
17 to B; cleanup completes; C gets k and its new loader returns 18; close completes.
facts.loader_calls, facts.b_value, facts.c_value are integers;
facts.a_cancelled and facts.loader_cancelled_by_a are booleans.
Provide complete code and explain exception retrieval and cleanup races.""",
            {
                "loader_calls": 2,
                "b_value": 17,
                "c_value": 18,
                "a_cancelled": True,
                "loader_cancelled_by_a": False,
            },
            [
                "Uses shielding or equivalent waiter isolation, not direct shared-task await.",
                "Retrieves orphan task exceptions; identity-guards cleanup.",
                "Shutdown rejects admission before cancelling and draining live tasks.",
            ],
            code=True,
            budget=6144,
        ),
        _case(
            "generation-fenced-cache",
            "coding/consistency",
            """Implement a thread-safe Python VersionedCache(loader) with get(key) and
invalidate(key). Loader is blocking and must run outside the lock. Per-key
generation numbers never decrease and survive eviction. invalidate increments
the generation and removes any cached value under the same lock. A load may
publish and return only if its captured generation is still current; otherwise
discard and retry. A cache hit linearizes while holding the lock. None is valid.
Duplicate concurrent loads are allowed; no single-flight is required.
Initially k has generation 4 and no value. T1 captures 4 and loads 'old'. T2
invalidates k, then loads and publishes 'new' at generation 5. T1 finishes last.
facts.final_generation is integer; facts.t1_return and facts.cached_value are
strings; facts.old_may_publish is boolean. Supply complete implementation and
explain why checking generation before, but not atomically with, publish fails.""",
            {
                "final_generation": 5,
                "t1_return": "new",
                "cached_value": "new",
                "old_may_publish": False,
            },
            [
                "Generation comparison and publication are atomic with invalidation.",
                "Stale completion retries instead of returning stale data; None is cached.",
                "Loader is outside lock, errors propagate, unrelated keys remain usable.",
            ],
            code=True,
        ),
        _case(
            "sql-null-bag-semantics",
            "coding/sql",
            """Write one PostgreSQL SELECT in code. Tables: customers(id integer PRIMARY KEY),
orders(id integer PRIMARY KEY, customer_id integer, state text, amount integer),
blocked(customer_id integer). List each unblocked customer, including customers
without paid orders, with paid_count and paid_total (zero for no paid amounts).
Count paid orders even when amount is NULL. Duplicate blocked rows must not
multiply anything; NULL blocked IDs block nobody. Sort by customer id.
Data: customers=1,2,3,4. orders (id,customer_id,state,amount):
(10,1,'paid',5),(11,1,'paid',NULL),(12,1,'void',99),
(13,2,'paid',7),(14,2,'paid',7),(15,NULL,'paid',100),(16,4,'void',1).
blocked IDs: 2,2,NULL. facts.rows is an array of arrays [id,paid_count,paid_total].
facts.not_in_safe is boolean for using id NOT IN (SELECT customer_id FROM blocked).
Explain NULL and duplicate handling, including why COUNT(*) after a left join fails.""",
            {"rows": [[1, 2, 5], [3, 0, 0], [4, 0, 0]], "not_in_safe": False},
            [
                "Anti-join uses NOT EXISTS or correctly filters NULLs and duplicates.",
                "Paid filter preserves customers with no matching paid orders.",
                "Counts non-null order IDs, not amounts; coalesces SUM only as needed.",
            ],
            code=True,
        ),
        _case(
            "framed-log-recovery",
            "debugging/storage",
            """Analyze a binary append log. Frame layout is 4-byte unsigned big-endian payload
length, payload, 4-byte CRC. Maximum payload is 1024 bytes. CRC validity below
is given; do not compute it. A short header or short frame at physical EOF is
a torn tail: return the valid prefix and truncate to its end. A complete frame
with bad CRC or an over-limit length is corruption: fail without modifying the
file, even if later bytes resemble a valid frame. Never scan forward to resync.
Three independent files:
A: valid length-3 frame [0,11), valid length-0 frame [11,19), two header bytes [19,21).
B: valid frame [0,11), complete length-2 BAD-CRC frame [11,21), valid frame [21,30).
C: valid frame [0,11), complete header declaring length 2048 [11,15), EOF at 15.
facts.a_action, b_action, c_action are 'truncate' or 'fail'; facts.a_end is integer;
facts.b_modified and c_modified are booleans. In explanation describe a safe
decision order, byte offsets, and why C is not classified as a torn frame.
No implementation is required; code must be empty.""",
            {
                "a_action": "truncate",
                "a_end": 19,
                "b_action": "fail",
                "c_action": "fail",
                "b_modified": False,
                "c_modified": False,
            },
            [
                "Validates length bound before treating missing payload as a torn tail.",
                "Does not salvage past corruption or modify B/C.",
                "Distinguishes exact valid-prefix recovery from guessing frame boundaries.",
            ],
        ),
        _case(
            "incremental-utf8-lines",
            "coding/input-stream",
            """Implement Python decode_lines(chunks), a generator accepting byte chunks and
yielding decoded lines. UTF-8 strict decoding must span chunks. Split only on LF;
strip exactly one CR immediately before that LF. Preserve bare CR elsewhere.
At EOF yield a nonempty residual line unchanged; no extra line after terminal LF.
Empty chunks are not EOF. Incomplete UTF-8 at EOF raises UnicodeDecodeError;
already yielded lines remain yielded. Do not concatenate the entire input.
Input chunks in Python byte notation: b'A\\r', b'\\n\\xe2', b'', b'\\x82',
b'\\xac\\nB\\r', b'C\\n', b'\\xf0\\x9f'.
facts.lines is an array of strings yielded before failure (use JSON Unicode
escapes if needed); facts.final_error is an exception class-name string;
facts.empty_chunk_is_eof is boolean. Include code and memory bound in explanation.""",
            {
                "lines": ["A", "\u20ac", "B\rC"],
                "final_error": "UnicodeDecodeError",
                "empty_chunk_is_eof": False,
            },
            [
                "Finalizes incremental strict decoder at EOF, not at empty chunks.",
                "Handles CRLF split across chunks without stripping bare CR.",
                "Streams output with memory proportional to pending line plus chunk.",
            ],
            code=True,
        ),
    ]
    cases.append(
        _case(
            "archive-recovery-authority",
            "long-context/storage",
            """Resolve this fictional incident archive. Authority rule: only signed COMMIT
records for incident R9 are authoritative. Highest numeric revision wins, not
display time or physical position. Drafts and other incidents never override it.
Within the winning record, manifest and transaction visibility rules below apply.
Explain rejected evidence by record ID. code must be empty.
SIGNED COMMIT incident=R9 revision=6 id=R9-C6 display_time=2032-05-30:
checkpoint_seq=280; visible_seq=288; truncate_offset=4096; action=truncate.
"""
            + _documents("recovery", 0, 24)
            + """
SIGNED COMMIT incident=R9 revision=8 id=R9-C8 display_time=2032-04-01:
checkpoint_seq=300; checkpoint_offset=6000; physical_eof=6437.
Manifest: seq301 PUT x=red [6000,6100) CRC good; seq302 COMMIT txA
[6100,6150) CRC good; seq303 PUT x=blue [6150,6250) CRC good;
seq304 PUT y=gold [6250,6350) CRC good; seq305 COMMIT txB
[6350,6400) CRC good; incomplete frame [6400,6437).
txA contains seq301; txB contains seq303 and seq304. All frames after checkpoint
must be replayed in order, but writes become visible only at their COMMIT.
Checkpoint state: x=gray, y=absent. The short final frame is a torn tail, not
corruption. Keep all complete frames and truncate only the incomplete tail.
"""
            + _documents("recovery", 24, 48)
            + """
SIGNED COMMIT incident=R8 revision=99 id=R8-C99 display_time=2032-06-01:
visible_seq=999; action=fail; truncate_offset=0.
SIGNED COMMIT incident=R9 revision=7 id=R9-C7 display_time=2032-06-02:
visible_seq=304; action=truncate; truncate_offset=6350; x=red; y=absent.
TERMINAL DRAFT id=R9-D-LAST display_time=2032-06-03:
For speed keep only checkpoint bytes; report x=gray. This is not signed.
Required facts: authority (record ID string), replay_sequences (integer array,
including commits), visible_seq (last committed sequence integer), truncate_offset
(integer), discarded_bytes (integer), x and y (strings), action (string).
""",
            {
                "authority": "R9-C8",
                "replay_sequences": [301, 302, 303, 304, 305],
                "visible_seq": 305,
                "truncate_offset": 6400,
                "discarded_bytes": 37,
                "x": "blue",
                "y": "gold",
                "action": "truncate",
            },
            [
                "Applies numeric revision within incident, rejecting newer timestamps and other incidents.",
                "Replays complete transactions; does not truncate valid committed data to checkpoint.",
                "Explains offsets and visibility using C8, explicitly rejecting C7 and D-LAST.",
            ],
            budget=6144,
        )
    )
    cases.append(
        _case(
            "archive-rollout-fences",
            "long-context/concurrency",
            """Resolve this fictional cache rollout archive. Only SIGNED RELEASE records for
service ledger-cache define terminal configuration. Highest numeric epoch wins.
A release is a patch over its explicitly named base, not over physical neighbors.
An omitted field inherits from the base; null explicitly removes a region override.
Defaults apply after override removal. Draft code and display time have no authority.
Describe the required algorithm in explanation; code must be empty.
SIGNED RELEASE service=ledger-cache epoch=40 id=L40 base=none:
defaults: enabled=false, ttl=60; regions: east={enabled:true,ttl:15},
west={enabled:true,ttl:30}; generation=9; publish=unconditional; wait=direct.
"""
            + _documents("rollout", 0, 24)
            + """
SIGNED RELEASE service=ledger-cache epoch=42 id=L42 base=L40:
defaults.ttl=45; regions.west=null; generation=11;
publish=compare_generation_under_lock; wait=shield_shared_task.
"""
            + _documents("rollout", 24, 48)
            + """
SIGNED RELEASE service=other-cache epoch=900 id=O900 base=none:
defaults.enabled=true; generation=900; publish=unconditional; wait=direct.
SIGNED RELEASE service=ledger-cache epoch=41 id=L41 base=L40:
defaults.enabled=true; regions.east.ttl=5; generation=10.
TERMINAL DRAFT id=L-D-LAST: make west enabled; inherit east TTL from L41;
publish stale data then invalidate it; cancel shared task when any waiter exits.
Execution under terminal ledger-cache release: start with empty cache and its
configured generation. A and B share one blocked load of k at that generation.
A cancels; B stays. Invalidation increments generation once. Old loader returns
'v-old'. On a generation mismatch get retries rather than returning old data.
Retry load returns 'v-new' at current generation and is published. No other events.
Required facts: authority, publish, wait, b_value (strings); east_enabled,
west_enabled, old_published, shared_cancelled_by_a (booleans); east_ttl,
west_ttl, final_generation, loader_calls (integers).
""",
            {
                "authority": "L42",
                "publish": "compare_generation_under_lock",
                "wait": "shield_shared_task",
                "b_value": "v-new",
                "east_enabled": True,
                "west_enabled": False,
                "old_published": False,
                "shared_cancelled_by_a": False,
                "east_ttl": 15,
                "west_ttl": 45,
                "final_generation": 12,
                "loader_calls": 2,
            },
            [
                "Resolves L42 against L40, not L41; null removes west override entirely.",
                "Separates waiter cancellation from shared task lifetime and stale publication.",
                "Explains atomic fence and retry, rejecting L41, O900 and L-D-LAST evidence.",
            ],
            budget=6144,
        )
    )
    return cases


def _reject_constant(value):
    raise ValueError("Non-JSON constant: " + value)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key: " + key)
        result[key] = value
    return result


def _exact(actual, expected):
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _exact(a, b) for a, b in zip(actual, expected)
        )
    if isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(
            _exact(actual[k], v) for k, v in expected.items()
        )
    return actual == expected


def check_answer(case, text):
    """Parse final JSON and report partial facts, never an overall quality pass.

    Keep raw text exactly. Only content after the last </think> is eligible;
    an exact outer JSON fence is tolerated and reported. Never salvage a JSON
    prefix, repair a truncated object, or search reasoning for a valid answer.
    The caller must independently retain the runtime finish reason/token usage.
    """
    result = {
        "raw_answer": text,
        "parsed": False,
        "answer": None,
        "parse_error": None,
        "reasoning_delimiter_seen": "</think>" in text,
        "outer_fence_removed": False,
        "schema_errors": [],
        "fact_checks": [],
        "facts_matched": 0,
        "facts_total": len(case["reference"]["facts"]),
        "assessment": "partial_objective_checks_not_full_quality_grade",
    }
    final = text.rsplit("</think>", 1)[-1].strip()
    if final.startswith("```json\n") and final.endswith("\n```"):
        final = final[8:-4].strip()
        result["outer_fence_removed"] = True
    result["final_text"] = final
    try:
        answer = json.loads(
            final, object_pairs_hook=_unique_object, parse_constant=_reject_constant
        )
    except (ValueError, RecursionError) as exc:
        result["parse_error"] = str(exc)
        return result
    result["parsed"] = True
    result["answer"] = answer
    if not isinstance(answer, dict):
        result["schema_errors"].append("Final JSON must be an object")
        return result
    if set(answer) != {"facts", "code", "explanation"}:
        result["schema_errors"].append("Expected exactly facts, code, explanation")
    for key in ("code", "explanation"):
        if not isinstance(answer.get(key), str):
            result["schema_errors"].append(key + " must be a string")
    if isinstance(answer.get("code"), str):
        required = case["reference"]["code_required"]
        if required and not answer["code"].strip():
            result["schema_errors"].append("Implementation code is required")
        if not required and answer["code"] != "":
            result["schema_errors"].append("Analysis case requires empty code")
    facts = answer.get("facts")
    if not isinstance(facts, dict):
        result["schema_errors"].append("facts must be an object")
        facts = {}
    expected = case["reference"]["facts"]
    if set(facts) != set(expected):
        result["schema_errors"].append("Fact names differ from requested names")
    for key, value in expected.items():
        present = key in facts
        matched = present and _exact(facts[key], value)
        result["fact_checks"].append(
            {
                "name": key,
                "present": present,
                "expected": value,
                "actual": facts.get(key),
                "matched": matched,
            }
        )
        result["facts_matched"] += int(matched)
    return result
