# F04 — Q4_K n64 gate wrong for FFN prefill shapes (unverified against clean tree)

Date: 2026-08-19. HEAD `307007d` + working-tree modifications by another
agent (`crates/qwen-cli/src/bench.rs`, `crates/qwen-llm/src/{loader,metal,metal_dflash}.rs`
all dirty). Build succeeds; runs via `--allow-dirty`.

## Signal

Same binary, same tree, single-cell A/B at pp2048 (5 reps each):

| Config | t/s |
| --- | ---: |
| Production default (n64 Auto) | 173.21 ± 8.10 |
| `QWEN_MATMAT_Q4_K_N64=0` (n64 forced OFF, generic 32-col tile fallback) | 180.36 ± 5.64 |
| Delta | **+4.13%** for n64 OFF |

Single-cell corroboration at pp512 (auto only, n64-off cell was clobbered by
mid-run tree drift): auto = 197.36 t/s.

## Cross-HEAD sanity note

pp2048 on the earlier CLEAN `1ecf208` HEAD was **222.13 t/s** (`baseline.md`).
Current 307007d+dirty auto is 173.21 t/s — a **−22% regression** vs the clean
baseline. Whether this is due to the other agent's WIP mods (very likely — 4
tracked files modified in the prefill/kernel path) or a landed change since
1ecf208 is unknown until the tree stabilizes.

Consequence: **the +4.13% n64-off signal on this dirty tree may not transfer
to a clean tree.** The other agent's mods may be pessimizing n64 specifically,
inverting the gate polarity artificially. Or they may have already patched the
gate condition on the general path. Not confirmable until the tree is clean.

## Why n64 might be wrong for FFN at chunk_p=512

Gate at `metal.rs:7100`:
```rust
Auto => !(n_query <= 512 && (n_in <= 2048 || n_out <= 2048))
```

For FFN gate (M=17408, N=512, K=5120): `n_query=512, n_in=5120, n_out=17408`.
`n_query <= 512` is TRUE (equality); `n_in <= 2048 || n_out <= 2048` is FALSE
(both large). Result: `!(true && false) = TRUE` → n64 fires.

The `n_query <= 512` condition uses `≤`, so N=512 exactly is treated as
"small" for the inner-dim gate. Nothing in the gate distinguishes N=512
(medium prompt) from N=64 (small verify-adjacent). The likely-intended cutover
is that n64 wins for N>=some-threshold where amortization dominates, but the
gate as written classifies N=512 as small-N regardless of inner dims.

## Proposed action, gated on clean-tree re-verification

1. **When tree stabilizes**, re-run this A/B on a clean HEAD.
   - If n64=off still wins by ≥3%: patch the gate. Concrete proposal: change
     Auto to `n_query >= NNN` where NNN is calibrated (probably ≥1024 or
     depends on M×K product). Requires a proper N-sweep at production
     shapes to find the crossover.
   - If n64=off no longer wins: the +4.13% was an artifact of the other
     agent's mods, close F04, revisit after they land whatever they're
     building.

2. **The −22% cross-HEAD regression** (222 → 197 t/s at pp512, similar at
   pp2048) is a separate concern. Either land in the perflog by whoever
   made the change, or investigate once the tree is clean.

## Not recording as a confirmed lever

Publishing F04 as a signal, not a landed finding. The dirty tree contaminates
attribution; the +4.13% might survive a clean run or might not. Fixing the
gate is a small edit (one arithmetic condition) but it needs to be tested
against a controllable baseline.

## Artifacts

- Raw JSON: `/tmp/pp2048-{auto,off}.json`, `/tmp/pp512-auto.json`
- Command: `QWEN_MATMAT_Q4_K_N64=0 qwen-bench pp --allow-dirty ...`
