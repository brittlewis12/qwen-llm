# v0.654 Grammar Forced-Run And Admissible-Row Oracle

Status: complete CPU-only oracle. The exact 36-string response-shape language
has maximum grammar-forced suffix length `1`, below the roadmap's four-token
local floor. Forced-run fast-forward is therefore killed for this exact
language and token-piece policy. The same cell has at most `17` admissible
vocabulary rows and supplies a positive restricted-lm-head topology signal,
not implementation, timing, product, or speedup authority.

## Question And Boundary

For one real constrained-output shape, determine:

1. whether any productive byte-prefix state begins a locally useful run of
   uniquely admissible token IDs; and
2. how many vocabulary rows remain at grammar branches.

The decoder contract is exact concatenated token-piece bytes at a clean
assistant-output boundary, with no token healing, no extra logit masks folded
into grammar forcedness, and a separate terminal action. Every ordinary
nonempty token ID remains distinct even when pieces collide. The language is
the ordered minified object:

```json
{"response_mode":"...","expressed_confidence":"...","length":"..."}
```

Its fields have `4 x 3 x 3 = 36` combinations. This is an exact finite-language
and Qwen3.6 token-piece-policy claim. It does not generalize to other schemas,
tokenizers, healing policies, or combined masks.

## Inputs

- Target tokenizer: `/Users/tito/models/Qwen3.6-27B-Q4_K_M.gguf`.
- Grammar SHA-256:
  `f3a053733738aa8e05c10c2f078ca5a36c26a3bc5746487903463cdcf31b2634`.
- Frozen trace SHA-256:
  `eb702b6b93fe58255cc16783363150f886dbfd0d6f7c032c1798bae1c41b74b1`.
- Result SHA-256:
  `85f0d719411fd20a0a66011cc07dd3c20cd4a3bbb2f98b8f54644c54577dcea7`.
- Trace source SHA-256:
  `4a6de539fbad9cfe89f8210fa3aff542137b4b6bb15ab95dacadafb891e8afcc`.
- Prompt-builder SHA-256:
  `1c7e3595a938aeb13676e85f4e7ff02431e667b3db466c8ced766f9c371a4740`.
- Illustrative dense-27B costs: serial transition `38.619 ms`; physical N8
  packet `117.633 ms`.

The 20 real records are ten `Q-self-pred` and ten `Q-other-pred` generations
from `Qwen/Qwen3-4B`. Their bytes and original 19-token streams are preserved
exactly. They weight four observed enum combinations, but the target tokenizer
retokenizes each visible object to 18 tokens. They are not target-model sampled
token traces or broad workload archetypes.

## Method

`qwen-grammar-oracle` opens only GGUF tokenizer metadata and issues no model
forward or GPU command. It:

1. exposes exact raw token-piece bytes and applies a conservative Qwen content
   policy that excludes control, user-defined, unknown, unused, padded, and
   empty pieces;
2. enumerates all finite-language byte prefixes;
3. admits token ID `t` at prefix `p` exactly when `p + piece(t)` remains a
   productive language prefix;
4. preserves duplicate token IDs rather than deduplicating equal pieces;
5. cross-checks the indexed admissible set against a full eligible-vocabulary
   scan at every state;
6. computes token reachability and the singleton-edge recurrence
   `forced(p) = 1 + forced(p + piece(t))` only when exactly one ID is
   admissible; and
7. separately replays all 36 canonical target-tokenizer paths and the frozen
   real-output weights.

The command was:

```sh
cargo run --release -p qwen-cli --bin qwen-grammar-oracle -- \
  --model /Users/tito/models/Qwen3.6-27B-Q4_K_M.gguf \
  --grammar docs/bench/v0654-grammar-response-shape.json \
  --traces docs/bench/v0654-grammar-response-shape-traces.json \
  --serial-transition-ms 38.619 \
  --fixed-n8-packet-ms 117.633 \
  --output docs/bench/v0654-grammar-response-shape-result.json
```

## Result

| Quantity | Result |
|---|---:|
| Vocabulary rows | 248,320 |
| Ordinary nonempty content rows | 248,044 |
| Productive / token-reachable states | 616 / 616 |
| Nonterminal / terminal states | 580 / 36 |
| Singleton / branch states | 129 / 451 |
| Maximum forced suffix | **1 token** |
| Canonical paths with a forced position | **0 / 36** |
| Canonical token count | 18 for all 36 paths |
| Branch admissible rows p50 / p90 / p95 / max | 3 / 5 / 6 / 17 |
| Maximum admissible fraction | 0.006846% |
| Minimum topological row pruning | 99.993154% |
| State-token / branch incidences | 1,597 / 1,468 |
| Unique admissible / branch token rows | 223 / 222 |

All 616 indexed sets match the naive full-vocabulary scan. The grammar-global
state digest is `afa64a9a...be40`; all 36 canonical paths have no forced run and
digest to `b7221329...3401`. The 20-record weighting has the same empty forced-
run histogram and row p50/p90/p95/max `4/12/17/17` in both source branches.

At the observed maximum run `r=1`, all cursor cases lose against the current
fixed N8 packet: aligned nonterminal and leading-pending terminal are
`-79.014 ms`, aligned direct terminal is `-117.633 ms`, and leading-pending
nonterminal is `-40.395 ms`. No missing `Cpack(r)` measurement can rescue this
frozen current-N8, four-token lane because no qualifying run exists. A
specialized one-token state-transition packet is a changed premise and remains
unpriced.

## Interpretation

The failure is structural rather than merely workload-frequency dependent. At
every canonical boundary, the grammar admits 2-17 token IDs. The oracle does not
attribute those alternatives solely to same-target segmentation rather than
language branching. There are 129 singleton byte-prefix states, but each
immediately reaches a multi-ID state or termination, so none compounds into a
packable run.

The row result points to a different work-removal mechanism. The authenticated
dense descriptor has a `1,042,944,000`-byte Q6_K `output.weight` over 248,320
rows, or exactly `4,200` bytes per row. The maximum 17-row state therefore names
only `71,400` logical source bytes. A unique-row bank is `223 * 4,200 = 936,600`
bytes. A state-major bank duplicating all 1,468 branch incidences has an
unpadded Q6_K payload of `6,165,600` bytes (`5.88 MiB`) and could permit
contiguous tiny mat-vec calls without a runtime gather. Singleton states require
no head at all.

Those are exact logical payload arithmetic, not a physical bank or reader
result. Padding, alignment, offset tables, token-ID maps, bank build, state
lookup, dispatch occupancy, quantized dequantization, selected-row sampling, and
process-cold setup remain unmeasured. The next authorized step is one bench-only
charged state-major Q6_K lm-head floor. It must preserve selected logits and
constrained greedy/seeded-sampling semantics, remove at least 70% of head wall,
project at least 5% whole-token gain in its named structured cell, and charge all
bank construction and metadata against TTFT. Do not build a general grammar
engine before that floor clears.

## Validation

- Nine pure oracle tests cover duplicate pieces, noncanonical convergence,
  four-token recurrence, unreachable productive prefixes, canonical run
  boundaries, empty pieces, quantiles, and all four cursor cases.
- Exact tokenizer tests cover invalid UTF-8 byte pieces, UTF-8 boundary
  concatenation, invalid IDs, and Qwen control exclusion.
- Targeted release tests pass. Targeted clippy exits zero apart from two
  pre-existing library warnings. Edited-file `rustfmt --check` and
  `git diff --check` pass.
- Independent design and implementation/result review: `cx` session
  `019fb654-ce35-7d90-9987-326e405cc5a2`.
