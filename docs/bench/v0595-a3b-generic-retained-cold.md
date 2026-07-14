# v0.595 A3B Generic Retained-Storage Cold Packet

Status: preregistered before timed work. This tracked contract and its tracked
runner are part of the frozen source commit.

## Intent

Price the first production-value model on the generic retained-storage path:
non-MTP Qwen3.6 A3B Q4. The candidate retains planner-selected GGUF windows as
read-only Metal resources and performs no CPU prefault. GPU demand mapping,
first-prefill work, and any transition penalty stay inside external
spawn-to-first-byte and spawn-to-exit endpoints.

This is a warm-filesystem-cache, fresh disposable process experiment. It is not
storage-cold, persistent, server, MTP, arbitrary-prompt, or default-policy
authority.

## Frozen Contract

- Model: `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`, exact SHA-256 in the runner.
- Prompt: the frozen 419-token Qwen3.6 Reva fixture used by v0.593.
- Chunk 1024, effective chunk 419, context 1024, prefix cache disabled.
- Production native-embedding policy must report `auto-promoted`; neither arm
  sets `QWEN_NATIVE_QUANT_EMBED`.
- A sets `QWEN_GGUF_NO_COPY=0` and has no prefault override.
- B sets `QWEN_GGUF_NO_COPY=1` and `QWEN_GGUF_NO_COPY_PREFAULT=0`.
- Output lengths run in order 1 then 128.
- Pair order at each length is prospectively frozen as
  `BA, AB | AB, BA, BA, AB`.
- The first two pairs are a kill screen. The remaining four run only after the
  screen clears. All six pairs decide promotion.
- Processes never overlap. A complete-file cache read and at least 30 seconds of
  cooldown precede every process.

Before every cache read, require AC power, no thermal/performance warning, at
least 50% system memory availability, and capture pageout/swap state. Capture it
again after the read/cooldown and after child exit. The complete file read is
outside the timed endpoint and defines the explicit cache-warm condition. Child
hard faults and block input must be zero. Any pageout or swap growth during
cache conditioning terminates the packet before child launch. Growth during
child execution invalidates the complete pair and may use the post-run retry
policy. These checks do not observe every device-initiated I/O event, so no
storage-cold claim follows.

The runner hashes source, binaries, model, prompt, this contract, and itself. It
requires current clean HEAD and matching clean build identity before creating a
fresh packet directory. An interruption is terminal: retain the directory and
preregister a new attempt rather than deleting or resuming individual rows.

## Correctness And Accounting

The pre-timing gates are already green at the frozen parent history:

- copied versus retained A3B packed-prefill logits, complete KV/GDN prefill
  state, one forced transition, continuation logits, and continuation state are
  bitwise;
- the generic contract is one window, 733 direct requests, 732 unique views over
  22,123,530,752 bytes, zero aliases, and one 8,192-byte final-page fallback;
- no-prefault is explicit and reports zero prefault pages, bytes, wall, and
  checksum;
- the full active library suite is 200 passed and 128 ignored;
- tied aliases, converted embeddings, overlapping resources, split resources,
  and the isolated exact-27B path have independent passing guards.

Every timed row must reproduce the exact load contract, timing schema, model and
tokenizer identities, command, effective arm environment, prompt, token count,
chunk/context, terminal semantics, PSO accounting, and allocation checkpoints.
Every row must generate the requested token count and stop at the token limit.
Early EOS is fixture failure. Stdout must match within every pair and across all
accepted processes at a length.

Post-run host, pageout, swap, hard-fault, or block-input failures retain
attempt-specific artifacts and may rerun the complete pair up to three times.
Pre-run host exhaustion, short cache reads, early EOS, any A/B or cross-pair
output mismatch, parser, schema, ledger, source/build identity, or
command/environment failure terminates the packet.

## Gates

The two-pair screen and six-pair decision use the same practical thresholds.
Two pairs are only an early kill mechanism; they cannot promote the candidate.

At output 1 require:

- B wins first-byte and exit wall in every accepted pair;
- overall, BA-only, and AB-only median first-byte speedup is at least `1.20x`
  with at least 750 ms saved;
- overall, BA-only, and AB-only median exit speedup is at least `1.20x` with at
  least 750 ms saved;
- median internal model-load saving is positive;
- median peak-footprint reduction is at least 19,911,177,677 bytes, 90% of the
  planned retained view bytes.

Only a six-pair output-1 pass authorizes output 128. At output 128 require:

- B wins first-byte and exit wall in all six pairs;
- overall and order-stratified first-byte movement is at least `1.20x` and
  750 ms;
- overall and order-stratified exit movement is at least `1.10x` and 500 ms;
- positive median internal model-load saving and the same footprint gate.

Report per-pair retained/baseline transition TPS, baseline and retained
milliseconds per transition, and their delta over the 127 transitions. Use the
median paired TPS ratio for the 5% regression warning; marginal arm medians are
secondary. Output-128 median first-byte saving differing from output 1 by more
than 200 ms also yields `needs_review`. Record the unattributed outer residual
`spawn_to_first_byte - model_load - ttft`; any pairwise shift above 200 ms or
absolute BA-only or AB-only median shift above 150 ms yields `needs_review`.
The BA-minus-AB interaction remains an additional diagnostic. Compare PSO miss
counts as prefill/generation/total tuples; any A/B difference yields
`needs_review`.

Length-local review warnings are evaluated after each full six-pair row. They are
nonauthorizing: an output-1 warning stops before output 128, and every stopped
decision retains all warnings observed so far.

Do not fit a crossover from two output lengths. Report measured rows directly.

## Authority

Failure of the two-pair output-1 screen stops immediately. Failure of the full
output-1 row stops output 128. Directional loss of either external endpoint, or
nonpositive median model-load saving, closes generic retained storage as an A3B
latency lane while preserving its explicit memory mode. A positive but
sub-threshold A3B result may authorize only a separately sized A10B screen if
conservative byte-scaled arithmetic clears that packet's own bars.

If output 1 passes but output 128 fails, authority ends at output 1. If both pass,
authority is limited to this exact A3B fingerprint, cache-warm fresh disposable
processes, caller-owned opt-in, the frozen prompt/chunk/context regime, and at most
128 outputs. It authorizes an A10B full-state gate before A10B timing. It does not
authorize generic default-on selection, storage-cold use, persistent/server use,
MTP assets, arbitrary prompts, converted-embedding families, or longer outputs.
