# Completed-boundary work elision: mechanism confirmed, serving HOLD

## Disposition

The Qwen3.6 Q4 compact-code/prose cache miss is reproduced through a fresh
engine-generated checkpoint. Reusing its consumed state eliminates 127 repeated
forwards, but serial execution is slower than the existing 153-row packed path.
Only combined consumed reuse + bounded single-VT packed execution survives the
phase screen. No production lookup, renderer, admission or serving policy changes.
This does not reopen the failed broad Q4 restored-tail selector.

## Provenance and boundary witness

- Runtime base `922eee5c`; CPU diagnostic `120fde63`, engine witness `667ed282`,
  three-arm/state oracle `f71ed463`. Tests only, release Metal runs strictly serial.
- Inputs are the prior single-chunk-VT `q4-A1` / `q8-A1` recordings. Re-rendering
  adjacent requests 3/4 uses the real template binding and tokenizer. Prompt usage
  counts agree. Canonical re-tokenization alone cannot prove actual emitted IDs.
- A fresh loaded model has an empty cache. Replay requests 0..3 through
  `EngineBackend::generate`, with greedy settings, no drafter and no direct cache
  insertion. The intended consumed boundary is absent before request 3. Usage and
  cached-token counts agree with the recorded requests.
- Strict lookup and physical restore of the reconstructed completed key then
  succeeds exactly: matched 8988, restored 8987, no exact final logits. The strict
  token comparison witnesses canonical consumed IDs and pending ID in this newly
  generated checkpoint; it does not recover historical emitted-ID logs.
- Q4 pending ID 198 is a newline. Next prompt length 9013 shares exactly 8987 tokens;
  rendering replaces the unconsumed newline with ID248046 (`<|im_end|>`). Ordinary
  lookup falls back to consumed 8860, requiring 153 forwards. The completed consumed
  state would require 26. Q8 pending 957 remains matched: next 9014, LCP 8988,
  restored 8987, 27 forwards; no analogous work to remove on this fixture.

## Fixed-order phase screen

Frozen raw protocol precedes the first run. One process; old packed, consumed
serial, consumed packed. The two candidate arms explicitly restore the witnessed
completed key and continue its matching consumed state in the test; they do not
exercise a production consumed-alias lookup. Construction, restore and prefill
are timed separately; decode runs up to 128 greedy output tokens.

| Arm | Forwards | Allocation ms | Restore ms | Prefill ms | Decode ms | Request Metal bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Old packed | 153 | 0.641 | 34.298 | 917.115 | 5516.247 | 2162622464 |
| Consumed serial | 26 | 1.011 | 39.239 | 1066.581 | 5563.468 | 783663104 |
| Consumed packed | 26 | 2.184 | 41.433 | 286.325 | 5455.888 | 829865984 |

Allocation+restore+prefill is 952.054 / 1106.831 / 329.942 ms respectively. Only
combined reuse+packed clears the prospective 25% phase screen. Serial regression
is now measured, not the earlier arithmetic estimate. Single-VT actual scratch
46,071,808 bytes, priced 46,502,992, both below 128 MiB. Request deltas include the
sequence; neither they nor scratch constitute peak-RSS or end-to-end authority.

All three arms produce 128 identical greedy token IDs, SHA256 over i32le IDs:
`abf4649eae66e3290d0eb88f068425d1110fb0f9d9240aadd9963bbf6b4213b3`.
Prompt logits cosine against old packed is 0.9999992832 / 0.9999997800 for serial /
packed consumed reuse. The frozen threshold is 0.999.

## Persistent-state follow-up

A separate run adds state comparisons after prefill, outside the measured phase.
All KV before 8860 agrees bitwise. Per-layer remaining K/V and every GDN state/conv
layer have minimum cosine 0.9999992966 / 0.9999990571 for serial / packed reuse.
Identities, positions and arena lengths agree. Prompt logits and 128 greedy IDs
repeat exactly. This is numerical state/greedy evidence, not global bitwise or
distributional equivalence.

All follow-up timing rows are retained, not pooled into the first screen:

| Arm | Allocation ms | Restore ms | Prefill ms | Decode ms |
| --- | ---: | ---: | ---: | ---: |
| Old packed | 0.642 | 34.658 | 901.666 | 5384.295 |
| Consumed serial | 0.885 | 40.880 | 1040.687 | 5440.842 |
| Consumed packed | 1.311 | 38.990 | 273.233 | 5552.517 |

## Limits and validation

Independent read-only review accepts checkpoint provenance and supports one
bounded combined-plan experiment, not promotion. Any such implementation must
preserve default logical pending-token matching and make alias lookup plus packed
admission/allocation atomic. Fallback must re-admit the original larger workspace,
not retain a smaller candidate's admission. Alias cached-token counts must exclude
the unmatched pending token. No alias API or fallback implementation is tested here.

No balanced HTTP experiment, first-SSE measurement, cold-load result, speculative
result, long-context generalization or sampled quality claim. One reused history
does not establish overall deployment value. CLI regression suite: 374 passed,
11 ignored; formatting and diff checks pass. CPU replay diagnostics pass on both
targets; release checkpoint witnesses pass on both; Q4 phase and state runs pass.

Raw artifacts: `target/profiles/consumed-prefix-reuse/`, including
`PILOT-PROTOCOL.md`, both `*-render-indexed.log`, both `*-checkpoint-witness.log`,
`q4-tail-screen.log`, `q4-tail-state-oracle.log` and earlier CPU attempts.
Reproduce the ignored witness using absolute recording/model paths and
`QWEN_REPLAY_REQUEST_INDEX=3`; add `QWEN_REPLAY_COMPARE_TAILS=1` for the Q4 screen:

```sh
cargo test --release -p qwen-cli --bin qwen recorded_completed_checkpoint_witness -- --ignored --nocapture --test-threads=1
```
