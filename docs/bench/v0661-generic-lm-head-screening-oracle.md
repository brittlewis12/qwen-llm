# v0.661 Generic Lm-Head Screening Oracle

Status: preregistration. No v0.661 capture seam, analyzer, command, packet root,
model observation, screening result, or authority exists.

## Question And Authority

For six exact A3B hidden states, can outward-certified Q6_K block norms prove
that the production greedy winner is also the unique ideal-output-projection
winner while pruning at least 80% of competitors and charging at most 30% of the
full-head denominator?

This is a prophetic fixture oracle. It receives the production winner even
though a deployable screener must discover an incumbent cheaply. It also omits
the production Metal dequantization and F32 reduction error envelope.

A GO may authorize only one separately preregistered dense-27B guard using the
same mechanism and gates. It grants no production-error derivation, adaptive
screen, kernel, product path, full-logit equivalence, sampled decoding, broader
fixture claim, default policy, or performance claim.

There is no adaptive rescue in v0.661. If the simple block-bound candidate
fails, close this screening lane under the current premise.

## Exact A3B Profile

- Path `/Users/tito/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`.
- File size `22,134,528,992` and SHA-256
  `ac0e2c1189e055faa36eff361580e79c5bd6f8e76bffb4ce547f167d53e31a61`.
- Architecture `qwen35moe`, base name `Qwen3.6 35B A3B`, file type 15, MoE,
  40 layers, hidden 2,048, vocabulary 248,320, untied, zero MTP layers, and no
  `output.bias`.
- `output.weight` Q6_K `[2048,248320]`, eight 210-byte blocks per row,
  1,680 bytes per row, and `417,177,600` bytes total.

Authenticate the complete model before model work. Require the full frozen
architecture tuple and bind `output.weight` by canonical descriptor, dtype,
shape, GGUF data offset, extent, and complete raw-tensor SHA-256. The tensor
digest is an observed derivative, not a replacement model identity.

v0.661 must not discover, stat, hash, parse, bind, allocate, or execute any
dense model. Those operations are permitted only for the frozen A3B model.

## Frozen Request And Production Path

```text
prompt: docs/bench/tokenizer-prompts/current-reva-n8-interactive-qwen36.txt
prompt bytes: 1,891
prompt SHA-256: e265de9742d1b22e566fc108ae26331ccf46166c6f071e73f211e0a1a7e8b474
prompt tokens: 419
prompt-token SHA-256: fb4bbb4dc66ca7d219099e2974e787ef976f80789cde3e48b8a905dceece1f9f
generation: greedy, temperature 0, 128 tokens, token-limit termination
stop IDs: [248046]
prefill chunk/max context: 1,024/1,024
capture selection calls: 0,1,7,31,63,127
```

Use packed prompt prefill through `prefill_tokens_with_multi_hidden`, prompt
selection through `Sampler` at temperature zero, each transition through
`MetalForward::single_token_greedy` with `GreedyTotal`, and logical advancement
through `Sequence::advance_by(1)`. Advance the sequence by all 419 prompt tokens
immediately after prefill, exactly as the production path does. Preserve the
ordinary no-op callback and production control order:

```text
select -> append -> stop check -> callback -> token-limit check -> transition
```

The observational capture is not a callback or model command. On a non-stop
call, insert it after the token-limit check and before the resulting branch or
transition. Producer EOS short-circuits the callback and token-limit check; if
that call is named, capture after the stop check and before taking the stop
branch. Record which checks completed rather than claiming the non-stop suffix
on an EOS call. This insertion must add no command-status query or forward.

Call zero selects `gen[0]` from prompt token index 418 at sequence position 419.
For `j > 0`, call `j` sees the state after consuming `gen[j-1]` at input
position `419+j-1`, has sequence position `419+j`, and selects `gen[j]`. Call
127 selects pending, unconsumed `gen[127]`. There are exactly 127 transitions.

Forbid bench `argmax_i32`, speculative-lowest argmax helpers, MTP hidden helpers,
a substituted or second forward, an added command-status query, or any schedule
change. Require `session.h` and `session.logits` to be F32, exact-sized, Shared,
in bounds, non-null, and stable through copying after the ordinary API wait.

Use one fresh sequence and one request. There is no warmup, prefix cache,
durable cache, prompt lookup, speculation, nonzero-temperature sampling,
follow-up, extra sequence, or extra forward.

## Captures And Prophecy Firewall

At each named selection call persist:

- raw little-endian post-output-norm hidden F32 bits;
- raw little-endian full-vocabulary logits F32 bits;
- sequence position, consumed input token, selected output token, and winner
  logit bits;
- prompt and generated-token stream digests; and
- exact type, shape, byte length, and SHA-256 for every raw artifact.

Reconstruct the winner independently from full logits. The production order is
Rust `f32::total_cmp`; equal bit patterns choose the highest token ID, and `+0`
orders above `-0`. Any NaN or infinity in hidden or logits is `INVALID`. Require
the recorded winner, all six calls, exactly 128 generated tokens, 127
transitions, and token-limit termination. A valid ordinary stream ending at EOS
before call 127 is `KILL_CAPTURE_COVERAGE`; a missing or malformed reachable
capture is `INVALID`.

Persist full logits only as sealed audit evidence. The persisted analyzer-input
JSON contains only capture identity and winner ID. The acquisition controller
then drops the complete `Sequence`/`MetalSession`, closes every logits handle,
and releases all Metal model state before constructing the in-memory analyzer
input. Its positive capability set is exactly: profile geometry, capture
identity, winner ID, owned hidden F32 bits, owned authenticated block-norm F32
metadata, a read-only authenticated `output.weight` byte slice, and frozen ledger
constants. It receives no filesystem path, packet root, logits, winner-logit
bits, margin, rank, runner-up, or production score.

One prompt and six positions support only this optimistic kill/continue decision.
They are one correlated stream and support no frequency, confidence,
workload-distribution, or product-policy inference.

## Exact Ideal Arithmetic

For Q6_K element `i` in one 256-value block:

```text
c_i = exact_i8(scale[i/16]) * (unpack_q6(ql,qh) - 32)
w_i = exact_binary16(d) * c_i
z_r = sum_i w[r,i] * exact_binary32(h_i)
```

Binary16 `d` has at most 11 significant bits, integer `c_i` at most 13, and
their finite product at most 24. An F32 hidden has at most 24. Therefore each
weight-hidden product and each individual square needs at most 48 significant
bits and is exact in binary64. The finite F16/F32 exponent range cannot overflow
or underflow binary64 for `H=2048`. Gradual underflow is mandatory; unsupported
host semantics or any nonfinite/overflowed intermediate is `INVALID`.

An arbitrary-width exact-dyadic implementation is authoritative for:

- the complete production-winner ideal dot;
- every Q6_K row/block sum of squares;
- every hidden whole/block sum of squares; and
- every complete survivor ideal dot.

Convert the exact winner dot to an outward F64 interval `[L_w,U_w]` and validate
containment exactly. The scored bulk screen uses only `L_w`; exact dyadic
winner/competitor comparison is reserved for final survivor adjudication.

The bulk norm path may use scalar binary64 RN without fast-math, contraction, or
reassociation. For a positive RN accumulation of `n` exact terms, compute an
outward `gamma_up` for exact
`n*2^-53/(1-n*2^-53)`, then:

```text
denom_down = next_down(1 - gamma_up)
sum_up = next_up(sum_hat / denom_down)
```

Require `0 < denom_down <= 1`. A binary64 square root requires correctly rounded
IEEE-754 RN `sqrt`, followed by `next_up`. Convert a finite F64 norm upper bound
to F32 with the host RN cast, compare the result back in F64, and apply
`next_up_f32` iff the cast is below the source.

Before any bound may prune, exact dyadic comparison must validate every emitted
model block-norm upper bound and every hidden block-norm upper bound by squaring
the F32 bound and comparing it with the exact sum of squares. This is exhaustive,
not sampled. Every norm must also be finite and numerically nonnegative. Require
finite winner endpoints with `L_w <= U_w`.

## Decisive Block-Bound Candidate

For winner `w`, define competitors
`C={0..248320}\{w}`, with `|C|=248319`. The hard pruning floor is
`ceil(0.80*|C|)=198656` competitors. The winner is never pruned and is excluded
from pruning and survivor counts.

Compute the exact ideal winner score and its validated interval `[L_w,U_w]`.
For every competitor, use outward F32 block norms and an outward positive
binary64 sum:

```text
UB_r = up(sum_b(weight_norm[r,b] * hidden_norm[b]))
```

Prune only under strict `UB_r < L_w`. Equality survives.
The number eliminated here is `bound_pruned_count`; later survivor evaluation
grants no pruning credit.

Fully read and exactly evaluate every survivor. Any exact competitor score
greater than or equal to `z_w` is `KILL_IDEAL_MISMATCH`. Thus a GO proves that
the production-selected token is also the unique ideal-Q6_K winner for all six
captures. It still says nothing about the production Metal rounding envelope.

## Frozen Byte Ledger

The full-head denominator is exactly `417,177,600` bytes. The 30% limit is the
integer `125,153,280` bytes.

Only persistent construction of weight block-norm metadata is excluded. Report
its complete head read/hash, metadata population, bytes, and wall separately.
Run exact-dyadic block-norm containment as a second complete head pass and
report its traffic and wall separately from metadata population. Full-logit
capture reports hidden/logit copy traffic and wall separately.

Stable Rust exposes no scoped allocator high-water without replacing the
process-global allocator. For excluded fixture work, report exact owned `Vec`
requested-capacity high-water plus BigInt significant-bit and rounded-limb
payload high-water proxies. Label them as payload proxies that exclude allocator
metadata, spare BigInt capacity, and allocator overhead. This correction grants
no production memory or wall-time premise.

The owned-`Vec` proxy covers explicit raw-capture and analyzer work-unit vectors
through construction of the analyzer result. It excludes `String`, Serde/JSON,
hash, manifest, semantic-reread, and sealing codec allocations, whose capacities
are library implementation details rather than candidate working state. Raw F32
capture and block-norm persistence must borrow the existing little-endian slices
so this exclusion cannot hide another full payload staging copy.

Every conceptual resource has base offset zero and its own 128-byte-aligned
allocation:

```text
block_norm        f32[248320,8]   row-major
hidden            f32[2048]
hidden_block_norm f32[8]
upper             f64[248320]
active            u8[248320]
survivor_ids       u32[248320]
survivor_cmp       u8[248320]
winner_interval    f64[2]
winner_exact       u64[8]          two's-complement dyadic coefficient
exact_accumulator  u64[8]          reusable two's-complement scratch
reduction_record   u64[2]
```

The `winner_exact` and `exact_accumulator` dot resources use a common `2^-173`
unit. Finite F16, i8/Q6, and F32 extrema over 2,048 terms require fewer than 384
signed bits; the frozen 512-bit representation leaves at least 128 guard bits.
Norm-square validation retains its own arbitrary-width dyadic exponent, including
the `2^-298` F32-square floor. Charge the dot accumulator's 64-byte high-water
once in the survivor epoch, while reporting exact-arithmetic operation traffic
and wall as excluded oracle validation work.

Freeze logical byte zero of `output.weight` as the Q6_K span origin. The scored
candidate has these named epochs and accesses:

| Epoch | Reads | Writes | Index set / sharing |
| --- | --- | --- | --- |
| `hidden-norm` | complete `hidden` | complete `hidden_block_norm` | one capture |
| `winner-threshold` | complete `hidden`, complete winner Q6 row | `winner_interval`, `winner_exact` | winner only |
| `block-bound` | complete `block_norm`, complete `hidden_block_norm` | complete `upper`, complete `active` | all competitors; hidden norms perfectly shared |
| `compact` | complete `active` | compact prefix of `survivor_ids` | competitors only |
| `survivor` | complete `hidden`, compact `survivor_ids`, complete Q6 rows for each survivor, `winner_exact` | compact `survivor_cmp`, `exact_accumulator` high-water | hidden read once with optimistic perfect sharing |
| `reduce` | compact `survivor_ids`, compact `survivor_cmp`, `winner_interval` | `reduction_record` | survivors only |

Reads and writes are separate charged streams; repeated access in another epoch
is charged again. For each resource, direction, and epoch, round every accessed
interval `[a,b)` to
`[floor(a/128)*128, ceil(b/128)*128)` and merge overlaps only within that same
resource, direction, and epoch. Charge winner work, all per-capture hidden norms,
bounds, masks, compaction, survivor rows/output, and reduction scratch.

Report logical operand bytes, charged 128-byte bytes, and unique Q6_K footprint.
Also report 64- and 256-byte sensitivity, but gate only the 128-byte total. A
capture passes bytes only when the sum over all listed charged resources is
`<=125153280`.

`survivor_ids` must be strictly increasing, unique, in range, winner-free, and
exactly equal to competitors with `UB_r >= L_w`. `survivor_cmp` has the same
compact length and records the authoritative exact comparison for every
survivor. The reduction record reconciles competitor, bound-pruned, survivor,
less-than-winner, tie, and greater-than-winner counts exactly.

This is a frozen optimistic transaction surrogate, not measured cache, DRAM, or
GPU traffic. A GO authorizes no wall-time inference.

## Gates And Exhaustive Decision

All six captures must independently satisfy:

- `bound_pruned_count >= 198656`;
- `charged_128_bytes <= 125153280`; and
- exact unique ideal-winner validation.

Do not average captures. Report call zero and transition calls separately.
Define `any_pruning_failure` and `any_byte_failure` over all six.

Decision precedence is:

1. `INVALID` for identity, build, host, artifact, nonfinite, arithmetic,
   containment, capture-code, prophecy-firewall, or ledger defect;
2. `KILL_CAPTURE_COVERAGE` for a valid unchanged production stream ending at
   producer EOS before the required calls or at EOS instead of the frozen
   token-limit reason;
3. `KILL_IDEAL_MISMATCH` for any exact competitor `>=` its exact winner;
4. `KILL_PRUNING_AND_BYTES` when both aggregate failure flags are true;
5. `KILL_PRUNING` when only pruning fails;
6. `KILL_BYTES` when only bytes fail; and
7. `GO_OPTIMISTIC_A3B_FIXTURE` otherwise.

A valid `KILL_IDEAL_MISMATCH`, `KILL_PRUNING_AND_BYTES`, `KILL_PRUNING`, or
`KILL_BYTES` closes generic norm screening under this exact A3B block-Cauchy
premise. `INVALID` and `KILL_CAPTURE_COVERAGE` make no mechanism inference and
cannot be rerun under v0.661. A GO authorizes only the separately preregistered
dense guard named above; it does not authorize adaptive rescue or production
work.

## Implementation And Acquisition

Implementation may add one isolated `qwen-bench` subcommand and one module while
reusing validated model binding and the exact production path above. It must not
change a production kernel, sampler, command schedule, model representation, or
product CLI path.

The sole acquisition command is:

```sh
target/release/qwen-bench lm-head-screening-oracle \
  --model /Users/tito/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  --prompt-file docs/bench/tokenizer-prompts/current-reva-n8-interactive-qwen36.txt \
  --tokens 128 --capture-calls 0,1,7,31,63,127 \
  --packet-dir target/profiles/v0661-generic-lm-head-screening-oracle-a3b-p1 \
  --attest-no-other-user-gpu-workload
```

Require an initially absent root, clean matching release build/runtime/source
identity, no inherited `QWEN_*` controls, AC power, no thermal/performance
warning, no competing qwen/llama/Metal process, and explicit operator GPU
attestation. Timings are descriptive only.

Before opening the model, the acquisition process runs and archives in-process
arithmetic, Q6_K decode, exact-dyadic, norm-containment, directed-cast, tie,
survivor, 64/128/256-ledger, malformed-artifact, and prophecy-firewall self-tests.
External `cargo test` remains necessary implementation validation but is not
packet runtime authority.

The initially reserved root contains these deterministic, fixed-schema files:

```text
self-tests.json
capture-manifest.json
captures/call-NNN/hidden.f32le
captures/call-NNN/logits.f32le
winner-evidence/call-NNN.json
analyzer-input/call-NNN.json
metadata/block-norm.f32le
metadata-manifest.json
exact-validation.json
screening-result.json
byte-ledger.json
artifact-manifest.json
decision.json
```

JSON uses fixed deny-unknown schemas, deterministic field order, JSON integers,
and fixed-width hexadecimal strings for binary floating values. Duplicate keys
and nonfinite JSON numbers reject. Raw artifacts are lexicographically
inventoried. Inventory paths are normalized packet-relative paths; every entry
must be a nofollow regular file. Symlinks, hard-link aliases, directories in file
positions, foreign files, and path traversal reject.

`artifact-manifest.json` inventories every packet file except itself and the
terminal `decision.json`. The decision embeds the exact artifact-manifest
SHA-256. Rehash and restat every inventoried file after manifest construction and
immediately before terminal publication. A crash or unsealed root consumes
v0.661; any successor needs a new preregistration and root.

A complete mechanism decision requires the full file set above. A producer-EOS
coverage decision instead inventories only complete named capture groups reached
before the stop plus typed skipped-analysis JSON; it must not fabricate missing
raw captures or block-norm metadata. A recoverable `INVALID` may inventory only
successfully committed evidence. Any write, seal, or publication failure leaves
the root consumed and unsealed rather than publishing a manifest-less decision.

Independent design review: `cx` session
`019fbdd2-77fc-7db2-a960-93efffd0ad14`.
