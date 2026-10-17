# V4.1 Adapter For The Frozen Eight-Case Pilot

Run after the full-model correctness comparison. The adapter
attaches to an already-loaded, owned endpoint; it never launches/reloads/stops
a server, downloads artifacts, hashes weights, or changes its environment.

With a server rebuilt/restarted to include native phase stats:

```sh
uv run scripts/eval-v41-pilot.py \
  --endpoint http://127.0.0.1:8737 --model-id deepseek-v41 \
  --download-manifest /path/to/owned-download-manifest.json \
  --out /path/to/new-private-v41-pilot
```

The output directory must not exist (its parent must exist). The manifest flag
is optional: when supplied, metadata is copied as expected provenance, not proof
of artifact hashes or resident binary identity. No model paths inside it are
opened. This is old REAP-272E NativeF32, **not** the 1017 profile. The existing
server must have enough context for the unchanged long prompts and output caps;
the adapter neither truncates inputs nor reduces budgets to fit a small test
server. `--timeout-seconds` adjusts the operation deadline (default 900 seconds,
as in the frozen harness), not a performance/quality pass criterion.

## Reuse And Compatibility

- Reads the exact frozen `docs/bench/2026-10-16-glm-quality-comparison/cases.json`
  bytes; does not modify cases, rubrics, references, old results, or harness files.
- Calls the frozen `request_body`, removing only GLM's `x_qwen.prefill_lineage`.
  System/user text, eight-case order, output caps, greedy sampler settings, seed,
  and requested `low` effort are unchanged. V4.1 low is its official tier, **not**
  a claim of the same internal reasoning budget as GLM.
- Uses the frozen `parse_sse`, `validate_response`, and `check_answer`. Raw SSE,
  event timestamps, terminal responses, per-case results and `summary.json` stay
  private. Failed/malformed responses abort with artifacts retained; no restart
  of the borrowed server or invented V4.1 error-scoring whitelist occurs.
- Native phase fields come from the terminal `x_qwen` echo, not ordinal matching
  against logs from a server that may already have served other requests. The
  original log-shaped `native.phases/stats` result fields are derived from that
  echo and explicitly labeled as such. `server_request_ordinal` is unknown;
  `pilot_request_ordinal` is local to this run. Startup/load time is not measured.
- Weighted rates use the frozen calculation: total input/output tokens divided
  by summed native prefill/decode seconds. Counts include EOS. The summary adds
  descriptive ratios to frozen Maya (`--maya-summary` can select another genuine
  matching eight-case run), with no absolute performance gates. Tokenizers,
  builds, hardware, internal thinking policy and actual output lengths can differ;
  this is not matched-token or causal quantization evidence. Maya's old log times
  are rounded; V4.1's phase echo retains native precision.

## Native Timing Contract

The backend logs `serve phases: family=deepseek_v41 ...` and calls the shared
`finish_generation` path for `serve stats: version=serve_stats_v1 ...`.
`x_qwen.stats: true` adds the same phase facts under terminal `x_qwen.phases`;
other families' echoes are unchanged.

`prefill_ms` starts after render/tokenization, initial admission and state creation,
and covers prompt-token forwards plus their bounded admission/cancellation checks.
`prefill_mode=single_token_forward`, `prefill_rows=1`: these are full single-token
forwards, including their heads, not GLM grouped matrix prefills. `decode_ms` covers
sampling, output decoding/final flush, sink waits, and generated-token transitions.
`transition_ms` is the sum of performed transition calls and their admission/checks.
Native phases exclude final response serialization and state teardown; streamed
delta serialization/backpressure contributes to decode sink waits. Client elapsed
time remains separate. `transitions = generated_tokens - 1` for both EOS and token-limit
termination, so max-one and first-token EOS have zero decode forwards. Reuse/matched
counts are zero and CED is false. No phase rate is a pure GPU-kernel measurement.

## Blinding

Use the minimal two-arm adapter against retained Maya. It reuses the frozen case
format, 0-4 rubric (maximum 32 per source), private writer and review handoff;
only the candidate count/labels and grading-record count are adapted. Original
frozen files are unchanged. A/B assignment is seeded independently per case.

```sh
uv run scripts/blind-v41-pilot.py \
  --v41-summary /path/to/new-private-v41-pilot/summary.json \
  --maya-summary docs/bench/2026-10-16-glm-quality-comparison/results/maya.json \
  --out /path/to/new-private-blind-packet
```

No third arm is needed or fabricated. `--seed` optionally fixes the seed; otherwise
a random seed is generated and recorded only in `private/map.json`, alongside the
invertible per-case source mapping. Public packets contain only case/rubric/facts
and two answers/statuses, with anonymous A/B labels. Timings, source names, manifests
and other summary metadata are omitted. The reverse-order packet and adapted
16-record judge prompts keep reviewers independent; do not expose the private
key, seed, source paths or performance data to judges. Answers remain verbatim,
so self-identification inside answer text cannot be made reliably anonymous.
`validate_grades` in the adapter validates shape only; it performs no evaluation.

CPU-only adapter tests (all endpoint operations mocked):

```sh
uv run --no-project python -m unittest discover -s scripts -p 'test_*v41_pilot.py' -v
```
