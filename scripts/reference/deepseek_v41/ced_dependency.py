# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Conditional CED dependency witness, using asymmetric exact-integer arithmetic.

Hold encoder-derived globals FIXED when omitting decoder history. For the
default schedule, before physical block 20, D=20/W=128 needs 2541 rows; after
COMPLETE block 20, D=19 needs 2414, including its per-row candidate control.
Publication alone does not complete that block; W is not retained-tail length.
Induction: (residual, incoming mix, candidate control) at depth d is exact for
p >= r+d*(W-1). Raw KV is row-local; the inclusive window makes post-block
outputs exact for p >= r+(d+1)*(W-1). Selection/reindex controls are same-row,
and global values are fixed, so neither adds lateral dependencies. At the
deepest layer, the earliest of the last W raw rows needs the same boundary
as the final output: D*(W-1)+1 rows. Warmup outside these bands is NOT exact.
Two streams, lagged mix, shared query-specific candidate blocks, reindexing,
and pointwise FFN model dependencies, not actual mHC. Prefix sums give O(D*N)
bulk operations, not N*W scans (integer bit costs grow with depth).
NOT model-wide exactness, real GGUF quality/performance, whole-coded-graph
validation, or GPU/production-arithmetic bitwise parity.
"""

from dataclasses import dataclass


def cone_rows(depth, window):
    assert depth >= 0 and window >= 1
    return depth * (window - 1) + 1


def require_fixed_global_profile(split, kv_owners):
    if split not in kv_owners or any(owner > split for owner in kv_owners):
        raise ValueError("decoder globals must come only from the split boundary")


@dataclass(frozen=True)
class Row:
    e: tuple[int, int]
    mix: int
    pool: int = 0


def boundary(p):
    return Row((1 + 2 * p, 3 + (7 * p) % 11), 2 + (5 * p) % 13)


def raw_kv(row, layer):
    a, b = row.e
    return 2 * a + 3 * b + 5 * row.mix + layer + 1


def global_read(row, layer, p, globals_, causal=True):
    visible = p + 1 if causal else len(globals_)
    assert 0 < visible <= len(globals_)
    query = raw_kv(row, layer)
    # Source selection is reused vertically, never from another query row.
    pool = (query + row.mix) % ((visible + 1) // 2) if layer == 0 else row.pool
    assert 0 <= 2 * pool < visible
    index = 2 * pool + (query + layer) % min(2, visible - 2 * pool)
    return globals_[index], pool


def block(row, layer, p, local_sum, globals_, causal=True):
    g, pool = global_read(row, layer, p, globals_, causal)
    a, b = row.e
    attn_pre = 3 * a + b + row.mix + layer
    s = local_sum + g
    half_a, half_b = a + s, 2 * b + 3 * s
    next_mix = half_a + 2 * half_b + attn_pre
    ffn = 2 * half_a + half_b + attn_pre
    return Row((half_a + ffn, half_b + 2 * ffn), next_mix, pool)


class Ring:
    def __init__(self, window):
        self.slots = [0] * window
        self.total = 0

    def push(self, value, p):
        slot = p % len(self.slots)
        self.total += value - self.slots[slot]
        self.slots[slot] = value
        return self.total


@dataclass
class Run:
    states: list[list[Row]]
    raw: list[list[int]]
    rings: list[Ring]


def evaluate(rows, depth, window, globals_, start=0, first_layer=0, causal=True):
    states, raw, rings = [list(rows)], [], []
    for layer in range(first_layer, first_layer + depth):
        values = [raw_kv(row, layer) for row in states[-1]]
        prefix = [0]
        for value in values:
            prefix.append(prefix[-1] + value)
        outputs = []
        for i, row in enumerate(states[-1]):
            local_sum = prefix[i + 1] - prefix[max(0, i + 1 - window)]
            outputs.append(block(row, layer, start + i, local_sum, globals_, causal))
        ring = Ring(window)
        for i in range(max(0, len(values) - window), len(values)):
            ring.push(values[i], start + i)
        states.append(outputs)
        raw.append(values)
        rings.append(ring)
    return Run(states, raw, rings)


def assert_endpoint(full, replay, skip=0):
    assert full.states[-1][-1] == replay.states[-1][-1]
    assert len(full.rings[skip:]) == len(replay.rings)
    for expected, actual in zip(full.rings[skip:], replay.rings):
        assert expected.slots == actual.slots  # EVERY layer, physical ring order
        assert expected.total == actual.total


def shrinking_ranges(n, depth, window):
    start = max(0, n - cone_rows(depth, window))
    for layer in range(depth):
        output = max(0, n - 1 - (depth - 1 - layer) * (window - 1))
        yield start, output, n
        start = output


def evaluate_shrinking(
    rows, depth, window, globals_, omit_seed=False, misalign_pool=False
):
    plan = list(shrinking_ranges(len(rows), depth, window))
    states = [list(rows[plan[0][0] :])] if plan else [list(rows[-1:])]
    raw, rings = [], []
    for layer, (start, output, end) in enumerate(plan):
        inputs = states[-1]
        assert len(inputs) == end - start
        if misalign_pool and layer == 1:
            row = inputs[-1]
            other = next(peer.pool for peer in inputs[:-1] if peer.pool != row.pool)
            inputs = [*inputs[:-1], Row(row.e, row.mix, other)]
        values = [raw_kv(row, layer) for row in inputs]
        ring = Ring(window)
        seed_count = output - start
        for i in range(seed_count):
            if not (omit_seed and layer == 0 and i == 0):
                ring.push(values[i], start + i)
        outputs = []
        for i in range(seed_count, len(inputs)):
            p = start + i
            outputs.append(
                block(inputs[i], layer, p, ring.push(values[i], p), globals_)
            )
        states.append(outputs)
        raw.append(values)
        rings.append(ring)
    return Run(states, raw, rings)


def advance(row, p, rings, globals_):
    for layer, ring in enumerate(rings):
        row = block(row, layer, p, ring.push(raw_kv(row, layer), p), globals_)
    return row


def bounds_and_eligibility():
    assert cone_rows(20, 128) == 2541
    assert cone_rows(19, 128) == 2414
    assert cone_rows(20, 1) == 1
    require_fixed_global_profile(20, (2, 8, 14, 20))
    for owners in ((2, 8, 14), (2, 8, 14, 20, 24)):
        try:
            require_fixed_global_profile(20, owners)
        except ValueError:
            pass
        else:
            raise AssertionError("unsafe ownership accepted")


def small_profiles():
    for depth in (1, 2, 4):
        for window in (1, 2, 4, 7):
            budget = cone_rows(depth, window)
            lengths = {1, window, window + 1, budget, budget + 3}
            lengths.update((max(1, window - 1), max(1, budget - 1)))
            for n in sorted(lengths):
                rows = [boundary(p) for p in range(n)]
                globals_ = tuple(11 + 3 * p for p in range(n + 4))
                full = evaluate(rows, depth, window, globals_)
                r = max(0, n - budget)
                assert_endpoint(full, evaluate(rows[r:], depth, window, globals_, r))
                # Scalar streaming with splits before/at/after the SWA boundary.
                splits = {0, r}
                splits.update(min(n, window + offset) for offset in (-1, 0, 1))
                for split in splits:
                    rings = [Ring(window) for _ in range(depth)]
                    for lo, hi in ((0, split), (split, n)):
                        for p in range(lo, hi):
                            actual = advance(rows[p], p, rings, globals_)
                            assert actual == full.states[-1][p]
                    assert [x.slots for x in rings] == [x.slots for x in full.rings]


def validity_bands():
    depth, window, n = 4, 4, 20
    rows, globals_ = [boundary(p) for p in range(n)], (0,) * n
    full = evaluate(rows, depth, window, globals_)
    r = n - cone_rows(depth, window)
    replay = evaluate(rows[r:], depth, window, globals_, r)
    for d in range(depth + 1):
        first = d * (window - 1)
        assert replay.states[d][first:] == full.states[d][r + first :]
        if first:
            assert replay.states[d][first - 1] != full.states[d][r + first - 1]
        if d < depth:
            assert replay.raw[d][first:] == full.raw[d][r + first :]
    assert_endpoint(full, replay)


def one_row_short_is_wrong():
    # Zero globals make all remaining paths positive linear maps: the omitted
    # oldest contributing row cannot be canceled by a selection discontinuity.
    for depth, window in ((1, 2), (2, 3), (4, 4), (19, 128), (20, 128)):
        n = cone_rows(depth, window) + 5
        rows, globals_ = [boundary(p) for p in range(n)], (0,) * n
        full = evaluate(rows, depth, window, globals_)
        r = n - cone_rows(depth, window) + 1
        short = evaluate(rows[r:], depth, window, globals_, r)
        assert short.states[-1][-1] != full.states[-1][-1]
        assert short.rings[-1].slots != full.rings[-1].slots


def bounded_128_and_wrong_mix():
    depth, window, n = 3, 128, 389
    rows, globals_ = [boundary(p) for p in range(n)], (0,) * n
    full = evaluate(rows, depth, window, globals_)
    short = evaluate(rows[-128:], depth, window, globals_, n - 128)
    assert short.states[-1][-1] != full.states[-1][-1]
    assert short.rings[1].slots != full.rings[1].slots
    r = n - cone_rows(depth, window)
    wrong = [Row(row.e, 0, row.pool) for row in rows[r:]]
    assert [row.e for row in wrong] == [row.e for row in rows[r:]]
    wrong_run = evaluate(wrong, depth, window, globals_, r)
    assert wrong_run.states[-1][-1] != full.states[-1][-1]


def future_visibility():
    rows = [boundary(p) for p in range(4)]
    globals_ = tuple(range(16))  # All physical rows already published.
    poisoned = globals_[:4] + (10**12,) * 12
    assert_endpoint(evaluate(rows, 3, 2, globals_), evaluate(rows, 3, 2, poisoned))
    query = Row((1, 2), 3)
    assert global_read(query, 0, 3, poisoned)[0] < 4
    assert global_read(query, 0, 3, poisoned, causal=False)[0] == 10**12
    safe = evaluate([query], 2, 2, poisoned, start=3)
    leaky = evaluate([query], 2, 2, poisoned, start=3, causal=False)
    assert safe.states[-1][-1] != leaky.states[-1][-1]


def release_and_late_boundary():
    depth, window, n = 20, 128, 2548
    rows = [boundary(p) for p in range(n)]
    globals_ = tuple(7 + p % 17 for p in range(n + 4))
    full = evaluate(rows, depth, window, globals_)
    r = n - cone_rows(depth, window)
    replay = evaluate(rows[r:], depth, window, globals_, r)
    assert_endpoint(full, replay)
    late_r = n - cone_rows(depth - 1, window)
    # Complete block 20's output includes its next mix AND candidate control.
    late = evaluate(full.states[1][late_r:], depth - 1, window, globals_, late_r, 1)
    assert_endpoint(full, late, skip=1)  # Block 20's own ring must be kept separately.
    for p in range(n, n + 4):
        assert advance(boundary(p), p, full.rings, globals_) == advance(
            boundary(p), p, replay.rings, globals_
        )
        assert [x.slots for x in full.rings] == [x.slots for x in replay.rings]


def shrinking_schedule():
    for depth, window, n in ((4, 4, 27), (20, 128, 512), (20, 128, 2548)):
        rows = [boundary(p) for p in range(n)]
        globals_ = tuple(7 + p % 17 for p in range(n))
        full = evaluate(rows, depth, window, globals_)
        shrunk = evaluate_shrinking(rows, depth, window, globals_)
        assert_endpoint(full, shrunk)
        for d, (_, start, _) in enumerate(shrinking_ranges(n, depth, window)):
            assert shrunk.states[d + 1] == full.states[d + 1][start:]
    n, depth, window = 27, 4, 4
    rows = [boundary(p) for p in range(n)]
    full = evaluate(rows, depth, window, (0,) * n)
    broken = evaluate_shrinking(rows, depth, window, (0,) * n, omit_seed=True)
    assert broken.states[-1][-1] != full.states[-1][-1]
    assert broken.rings[-1].slots != full.rings[-1].slots
    globals_ = tuple(range(n))
    full = evaluate(rows, 2, window, globals_)
    wrong_pool = evaluate_shrinking(rows, 2, window, globals_, misalign_pool=True)
    assert wrong_pool.states[-1][-1] != full.states[-1][-1]
    assert [r.slots for r in wrong_pool.rings] == [r.slots for r in full.rings]
    for n, expected in ((512, 8955), (2048, 23433), (8192, 24150)):
        plan = list(shrinking_ranges(n, 20, 128))
        assert sum(end - output for _, output, end in plan) == expected
        assert sum(output - start for start, output, _ in plan) == min(n - 1, 2540)


if __name__ == "__main__":
    checks = (
        bounds_and_eligibility,
        small_profiles,
        validity_bands,
        one_row_short_is_wrong,
        bounded_128_and_wrong_mix,
        future_visibility,
        release_and_late_boundary,
        shrinking_schedule,
    )
    for check in checks:
        check()
        print(f"PASS {check.__name__}")
    print(f"{len(checks)} CPU check groups passed; conditional toy witness only")
