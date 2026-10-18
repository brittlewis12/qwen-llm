# Old REAP-272E CLI lane

Header-only inspection (no weight payload reads or Metal context):

```sh
qwen deepseek-v41 --model-dir /path/to/old-reap --inspect --report-memory
```

Load once, measure, release and exit:

```sh
qwen deepseek-v41 --model-dir /path/to/old-reap --load-only \
  --context 4096 --reserve-mib 4096 --report-memory --check-engram-map
```

Generate through the actual mixed CPU/Metal forward (greedy by default):

```sh
qwen deepseek-v41 --model-dir /path/to/old-reap --prompt 'Say hello.' \
  --max-tokens 16 --context 4096 --report-memory
qwen deepseek-v41 --model-dir /path/to/old-reap --token-ids 0,42,128 \
  --max-tokens 1 --context 4096
```

`--prompt` is user chat, rendered by the official V4.1 encoder; `--system` and
`--reasoning-effort none|low|high|xhigh|max` apply to that form. `--token-ids` is an
exact raw-ID prompt without inserted BOS or chat delimiters. The two input forms
conflict with each other and inspect/load-only. Tokenization never truncates.
`--temperature`, `--top-k`, `--top-p`, `--min-p` and `--seed` use the existing
sampler. EOS is taken from config and checked against the official tokenizer.
Text streams to stdout; generation-mode load/memory records and token IDs go to
stderr. SIGINT/SIGTERM cancellation reaches the forward's per-layer checks.

For first-token oracle comparisons, add `--trace-dir /path/to/fresh-trace` to
either generation form. It is refused for inspect/load-only/serve, and the
directory must be new or existing-empty, not a symlink (parent must exist).
There is no overwrite/resume mode and no automatic memory-budget change.

Each actual forward, including prompt prefill, writes all 40 layers directly
from the callback slices, with a fixed 4 KiB serialization buffer:

```text
pos-00000-layer-00.hidden.f32
pos-00000-layer-00.pre.f32
... through layer-39 ...
pos-00000-logits.f32
run.json
```

Files contain headerless little-endian F32. Hidden is stream-major `[4, 5120]`
(81,920 bytes per layer), pre is `[4]` (16 bytes), logits are `[129280]`
(517,120 bytes). Shapes, layer order and exact file lengths are checked.
Positions are zero-based consumed raw-token positions, not sampled-token indices;
the final sampled token (EOS or token-limit) is not forwarded just for tracing.

`run.json` starts with `status: "incomplete"`; only after generation and all
trace writes finish is it replaced with `status: "success"`. Binary files from
failed/cancelled steps remain provisional. The manifest records `mode: "NativeF32"`,
`byte_order`, `shapes`, `bytes_per_file`, `layer_count`, full root `config`,
`prompt_token_ids`, `generated_token_ids`, `stop_token_id`, and `positions` entries
of `{position, token_id}` for each completed forward. `config_profile` states
the expected old REAP repository/revision with `provenance_verified: false`;
it does not claim checkpoint hashes or authenticity. No per-layer atomic staging
or background tracing service is used.

## Targeted Long-Context Capture

Use the full GPU step and a fresh capture directory. `ids-10000.csv` must
contain 10,000 raw prompt IDs, comma-delimited, without chat rendering:

```sh
qwen deepseek-v41 --model-dir /path/to/old-reap \
  --engram-dir /path/to/original-base --gpu-step \
  --token-ids "$(cat /path/to/ids-10000.csv)" --max-tokens 1 --context 10000 \
  --attention-capture-dir /path/to/fresh-capture
uv run scripts/reference/deepseek_v41/cpu/long_context.py \
  --model /path/to/old-reap --capture /path/to/fresh-capture
```

Default zero-based checkpoints are `2046,2047,9998,9999`. Override with
`--attention-capture-positions 2046,2047,9998,9999,16383,16384` and at least
16,385 consumed tokens to include the first candidate-block discard. Requested
checkpoints must fit the planned forward count; an early EOS that prevents
capture completion fails rather than claiming a complete comparison.

The writer emits eight contiguous little-endian F32 NPY input histories for
layers 2,8,14,20,24,28,32,36, after HC collapse/RMS and before attention
projection, plus eight post-`wo_b` vectors per checkpoint. Device snapshots are
encoded before scratch reuse and read only after command completion. Exact
compressed IDs, KV owner, index producer, and layer-20 candidate block IDs
accompany each checkpoint in `manifest.json` (`dsv41-long-context-v1`). Capture
stops after the largest checkpoint. No full-layer trace flag is needed.

The manifest records consumed IDs, model paths/config, expected source revision,
and native build commit/dirty/source-state identity; model payload identity is
not authenticated. It starts incomplete. Normal completion finalizes NPY shapes
and marks success; failure/cancellation leaves an incomplete manifest and partial
histories. Abrupt termination may leave the initial manifest and provisional
headers. Never use an incomplete capture as agreement evidence. Existing files
and symlink directories are refused; there is no overwrite/resume mode.

These captures enable the real 2K/10K sparse-selection comparison; toy tests do
not replace that gate or certify a pilot. At 10K there is real index selection
discard but no candidate-block discard. Disk usage for eight 10K histories is
approximately 1.64 GB, independent of checkpoint count.

Serve the existing loopback Responses endpoint:

```sh
qwen deepseek-v41 --model-dir /path/to/old-reap --serve \
  --addr 127.0.0.1:8737 --context 128 --max-tokens 16 \
  --reserve-mib 1024 --report-memory
curl http://127.0.0.1:8737/v1/models
curl http://127.0.0.1:8737/v1/responses \
  -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-v41","input":"Say hello.","max_output_tokens":16,"stream":true}'
```

`--model-id` changes the served name. Startup sampling flags are defaults;
requests can override `temperature`, `top_p`, and `x_qwen.top_k/min_p/seed`.
The same official adapter handles reasoning, tools, and replayed multi-turn
history. Requests receive fresh forward state and report zero cached tokens.
No CED, snapshot/durable cache, prefix reuse or native diagnostics are enabled;
cache extension requests are refused, not silently interpreted as reuse.
Only loopback addresses bind. Context/weight ownership stays on the serial
owner stack, and the borrowed backend never moves to an HTTP worker. The
worker's cancellation flag reaches forward layer checks, including while the
owner is inside a synchronous step; all terminal paths drop request state.

The 221.33 GiB resident / 222.7 GiB recommended case does NOT fit a
4 GiB reserve. The explicit `--reserve-mib 1024` above matches the default
1 GiB weight headroom reserve for initial small-context testing, subject to actual
Metal allocation/alignment and process-memory admission. It is not a guarantee
that KV/scratch or arbitrary output limits fit. `--weight-budget-bytes` can only
lower the resulting ceiling, not bypass the reserve or device/process checks.
No command here changes iogpu/sysctl settings.

Use `--engram-dir /path/to/base` if original-base shards 47/48 are not colocated.
The main root must contain the old config and safetensors index. Downloaded files
must remain immutable during use. Geometry checks do not authenticate provenance.
Missing indexed shards are printed before exact planning fails; partial/truncated
present shards are invalid, not treated as completed downloads.

The command delegates old-checkpoint validation and exact byte planning to
`ResidentWeights::plan`; it does not substitute the incompatible 1017 config.
Packed experts are 204,550,963,200 bytes (190.503 GiB); dense F32 is computed from
the actual headers, not estimated. Original-base Engram tables stay disk-backed.
The loader may repeat header planning for admission, but payloads load only once.

The default weight ceiling is Metal's recommended working set minus current
allocations and 1 GiB reserve. `--weight-budget-bytes` can lower it. Device/process
admission still applies; nothing changes sysctl or iogpu limits. `--context` records
the intended future capacity and checks the config limit: it does not allocate KV
or prove that the reserve suffices for that context in inspect/load-only mode.
Generation creates one fresh forward state, consuming the prompt once and only
feeding sampled tokens when another output token is requested. No weight copies
or self-referential model owner are introduced.

JSON lines distinguish plan, load-start, loaded and released. Payload resident
bytes, actual Metal allocation/delta, load wall time, and optional OS footprint
are separate. Peak payload staging is the loader's conservative bound, not a
sampled peak; headers, allocator overhead, KV and future scratch are excluded.
Generation uses resident weights and real forward logits; synthetic logits occur
only in CPU unit tests. Full checkpoint quality/performance is not established by
those tests. HTTP generation calls the same resident forward API, with request
context and output-memory checks before creating state and forwarding tokens.

Both CLI and HTTP admit the forward's CPU KV/transient and Metal scratch estimate
for `prompt_tokens + max_output_tokens - 1`, not the configured capacity. Output,
sampler/retained-logit and transport buffers are additional CPU allowances. Current
Metal allocation (including weights) and process headroom are checked before state
creation and every prefill/decode forward. Rechecks deliberately charge the full
estimate as additional memory: no upper-bound cache estimate is treated as measured
ownership credit. This conservative policy can refuse earlier near the limit, but
does not hide unrelated process growth. HTTP pressure uses the existing typed
503 `memory_admission_denied`; unavailable/invalid telemetry still fails closed.

The forward constructor can consume
`qwen_cli::deepseek_v41_tokenizer::CompressedTokenMap::for_config`:
`lookup: Vec<u32>`, `vocab_size: u32`, `pad_id: u32`. It uses the upstream
tokenizers normalization sequence and checks raw/compressed sizes against the
root config and pinned hash geometry. The pad ID is mapped, never assumed.

Regenerate tiny Python fixtures (optional argument checks a full local tokenizer):

```sh
uv run crates/qwen-cli/tests/fixtures/v41_engram_token_map.py /path/to/tokenizer.json
QWEN_V41_TOKENIZER=/path/to/tokenizer.json cargo test -p qwen-cli --lib \
  v41_full_tokenizer -- --ignored
```

The generator extracts the exact hash-verified upstream Python function. The full
fixture stores only hashes/counts, not a tokenizer or large compression table.
