# Fresh short serving: packed phase screen PASS

## Scope and decision

Test-only `77182e1a` compares serving's existing serial <=48 path against existing
single-chunk packed prefill from position zero. No cache, drafter, new kernel or
production selector change. Qwen3.6-27B Q4_K_M and Qwen3.8-27B Q8_0 each pass
the prospective allocation+prefill, memory, numerical-state and greedy screen
at 19, 32 and 48 input tokens.

This qualifies a bounded fresh-serving endpoint experiment, not a shipped speedup.
Ordinary CLI already uses packed fresh prefill, so no CLI or cold-load improvement
is inferred. The removed consumed-alias candidate stays removed.

## Protocol and workspace

CPU architecture screen resolves actual production modes, then prices logical
allocations using synthetic 16 KiB rounding. Actual Metal pricing and full-request
admission are checked separately in each GPU arm before allocation. Cap: 128 MiB.
Same architecture produces the same scratch on both model artifacts:

| Input | Logical bytes | Synthetic rounded | Driver-priced upper | Actual scratch |
| --- | ---: | ---: | ---: | ---: |
| 19 | 11430440 | 12386304 | 11819636 | 11698176 |
| 32 | 19271208 | 20332544 | 19926696 | 19382272 |
| 48 | 28943656 | 30326784 | 29926888 | 29048832 |

Serial baseline uses zero packed scratch. This is a bounded memory-for-latency
trade, not a memory reduction. Admission includes a fresh sequence of input+64
capacity; scratch counts are not whole-process or peak memory.

Valid rendered no-thinking chat requests use color, code and prose instructions
respectively, with neutral `Hello ` prefixes to reach exact 19/32/48 token counts.
The 19-token color prompt is the retained fresh-short endpoint guard. For each
width, serial runs before packed in a single model-loaded test process. The
prospective screen requires >=25% allocation+prefill savings on every width/model,
logits and each KV/GDN state/conv layer cosine >=0.999, and equal greedy tokens
through EOS or 64 tokens. No same-cell endpoint gate is changed.

## All phase rows

Times in ms. These are fixed-order one-shot diagnostic observations, not balanced
latency authority. Initialization/PSO conditioning is not isolated by this screen.

| Model | Input | Serial alloc | Serial prefill | Packed alloc | Packed prefill | Combined savings |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Q4 | 19 | 0.513 | 763.451 | 0.608 | 220.011 | 71.122% |
| Q4 | 32 | 1.071 | 1166.133 | 0.610 | 222.204 | 80.910% |
| Q4 | 48 | 1.531 | 1769.774 | 0.606 | 313.514 | 82.266% |
| Q8 | 19 | 0.539 | 1221.861 | 0.617 | 206.334 | 83.070% |
| Q8 | 32 | 1.275 | 1775.581 | 0.788 | 209.248 | 88.179% |
| Q8 | 48 | 1.546 | 2661.611 | 0.656 | 293.009 | 88.973% |

## Correctness and limits

| Model | Input | Logits cosine | Minimum per-layer state cosine | Equal greedy tokens |
| --- | ---: | ---: | ---: | ---: |
| Q4 | 19 | 0.9999985596 | 0.9999985540 | 3, EOS |
| Q4 | 32 | 0.9999994981 | 0.9999990600 | 64 |
| Q4 | 48 | 0.9999990957 | 0.9999994312 | 64 |
| Q8 | 19 | 0.9999994096 | 0.9999984511 | 3, EOS |
| Q8 | 32 | 0.9999996553 | 0.9999989849 | 64 |
| Q8 | 48 | 0.9999993626 | 0.9999975749 | 64 |

State is compared immediately after prefill. Snapshot identities, positions,
arena lengths and per-layer slices agree; values are numerical, not bitwise.
The color witness really stops after three tokens; it is not called a 64-token
continuation. No distributional, sampled, speculative, cache-hit, HTTP, cold-load
or untested-context claim. Model weights differ, so this is not a Q4/Q8 quality
or speed comparison. Independent review finds no correctness blocker and supports
only a scoped, balanced fresh-serving endpoint experiment next.

## Attempts and validation

- Initial CPU observer reused a MoE test helper that enabled unrelated G8/G16
  packed workspace on G6 and failed the memory assertion. Source inspection
  corrected it to the production mode resolver before GPU execution; no gate
  or candidate topology changed.
- Initial Q4 GPU test loads the model but fails request parsing because its
  synthetic request omitted required `model`. It records no prefill timings.
  The schema-only fix adds the field; original log remains.
- Corrected Q4 and Q8 release tests pass all three widths. CPU architecture
  screen passes. CLI suite: 374 passed, 12 ignored; format and diff checks pass.
- Raw: `target/profiles/fresh-short-packed/{PROTOCOL.md,q4.log,q4-model-field-fixed.log,q8.log}`.
  Reproduce with `QWEN_FRESH_PILOT_MODEL` set to an absolute model path:

```sh
cargo test --release -p qwen-cli --bin qwen fresh_short_packed_matches_serial -- --ignored --nocapture --test-threads=1
```
