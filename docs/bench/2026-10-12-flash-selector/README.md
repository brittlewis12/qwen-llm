# Ordinary selected-prefill integer scan

Evaluate retained candidate `3b21b899`, brought onto current main in `da3d16bf`,
without the retired parent observer or encoder-splitting campaign. Production
selection remains unchanged while measuring.

One native UD artifact, context32768, common prefix8192, suffix2048, actual
packed workspace2048 and compressor capacity8192. Visible blocks range from
2048 to2560; the actual selector dispatch has at most32 queries, not2048.
Restore the same checkpoint outside each warm A/B and measured A1/B1/B2/A2.
Change only the integer prefix-scan selector. Record ordinary call wall and
existing complete GPU time, not cold TTFT. Warm census verifies unchanged
surrounding dispatches; endpoint/state hashes and three common continuation
steps are checked outside timing.

The independent selector oracle passes CPU checks and leased Metal validation:
exact IDs/counts/status, ties, nonfinite values, offsets and immutability at
capacities8192/16384 and query counts1/17. No new persistent GPU memory.

## Result: Do Not Promote

Both ordinary packets complete, with exact full endpoint logits and persistent
state hashes through three common continuation tokens. Warm census confirms
768 selector substitutions (12 QSA layers x64 bands), each with32 queries,
and identical surrounding dispatch/order/geometry. No planner/router override
or observer splits occur during measured calls.

| Packet | Wall A1/B1/B2/A2, ms | Mean A/B | GPU mean A/B |
| --- | --- | ---: | ---: |
| `ud-ordinary-abba.jsonl` | 6425.29 / 5874.90 / 7349.09 / 6331.56 | 0.9647 | 0.9484 |
| `ud-ordinary-repeat.jsonl` | 6654.25 / 6892.79 / 6572.82 / 5789.64 | 0.9241 | 0.9034 |

The first packet has opposite-signed pairs and25% B repeat spread; the second
has both wall/GPU pairs favoring the incumbent. Baseline drift is also visible
in the repeat. No sample is discarded and no threshold is relaxed. This does
not establish a precise regression estimate, but it does not justify replacing
the ordinary selector. Larger-query microbenchmark savings did not establish
benefit for the actual32-query serving launches.

Stop this candidate rather than widen bands or revive the parent-observer
campaign. Production selector and decode remain unchanged. The retained source
checkpoint and complete raw evidence suffice for reproducibility; remove the
candidate and one-off model packet from the active build after recording them.

Reproduction at the experimental revision:

```sh
env -u MTL_DEBUG_LAYER QWEN_METAL_LEASE_WAIT=1 \
  FLASH_PREFILL_MODEL=/path/to/Flash-Next-UD.gguf \
  FLASH_PREFILL_OUT=/path/to/new.jsonl /path/to/release/test-binary \
  qwen4exp_runtime::tests::prefill_map::selected_scan_ab::ordinary_selected_scan_ab \
  --ignored --exact --nocapture --test-threads=1
```

Three selector CPU tests and the ignored independent Metal oracle pass; both
model packets pass execution and exact-output/state checks. Timing acceptance
is a separate judgment, not implied by those passing checks.
