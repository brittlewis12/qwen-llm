# Lens Run

`qwen-lens run` performs one fresh Qwen run with live workspace-lens readouts
and optional ordered post-block interventions.

```sh
cargo run -q --release -p qwen-cli --bin qwen-lens -- run \
  --model /path/to/model.gguf \
  --plan /path/to/plan.json \
  --messages /path/to/messages.json \
  --max-new-tokens 32
```

Use exactly one of `--prompt`, `--token-ids`, `--user`, `--messages`, or
`--open-responses`. Raw prompts use the tokenizer's configured special-token
insertion unless `--no-special-tokens` is set, and literal token IDs are passed
unchanged. Ordinary Qwen `--user` and `--messages` inputs use the same strict
system/user/assistant renderer as normal model runs. Muse accepts the shared
structured ATEM request described below.

`--open-responses FILE|-` (alias `--responses-input`) is ordinary-Qwen-only. It
uses the exact parser, model capability gates, and prompt renderer shared with
`qwen serve`, including `instructions`, head system/developer messages,
server-normalized reasoning history, tool definitions, function calls, and
function outputs. Qwen3.8 history follows its server contract and may become a
preclosed empty thinking block rather than preserving supplied reasoning. The
current server deliberately compiles system and developer messages to the same
model-visible `system` record; span metadata retains the source label so this
equivalence remains auditable. Request generation/sampling fields and narrowed
`allowed_tools` are rejected because Lens CLI flags own execution. The request
`model`, streaming, and response-envelope fields do not alter prompt bytes;
`--model` selects the actual local model.

Sampling defaults to greedy. `--temperature`, `--top-k`, `--top-p`, `--min-p`,
and `--seed` expose the existing deterministic native sampler.

`--prefill-execution auto` is the default. For ordinary dense Qwen, it packs
maximal non-final prompt spans of at least 65 tokens that contain no operation
or readout event. Packed blocks are capped at 1,024 tokens; active events, the
final prompt token, and every decode transition stay on the serial intervention
path. Packed reductions use a different topology from serial matvec, so the
artifact classifies the run as numerically approximate rather than bit-exact.
`--prefill-execution serial` is the explicit scientific control.
Ordinary MoE, Muse, and Flash-Next currently resolve `auto` to serial and record
the stable reason in the run artifact and summary.

Without `--output`, stdout defaults to the complete `qwen.lens.run` JSON
document, preserving the original pipe-friendly behavior. With `--output`,
stdout defaults to a compact summary while the document is persisted. An
explicit `--format json` always prints JSON; explicit `--format summary` is also
allowed without an output file when discarding the full artifact is intentional.
`--output PATH` atomically replaces a regular run file in an existing parent
directory; symlink leaves are rejected. Run schema v5 records the runtime and
model path, canonical plan path, exact authored plan and canonical-JSON BLAKE3,
numeric resolved execution plan, semantic position bindings, input source,
exact token IDs, resolved renderer/mode, authored byte/token spans, sampler
settings, decoded text and stop reason, operation applications, requested and
emitted live readouts, native captures, and the requested/effective prefill
topology. Packed spans are half-open and exclude active and final prompt events;
the offline validator re-derives them from the resolved plan. It also recomputes
the resolved plan and every binding from the authored plan plus rendering metadata.
Published Muse runs additionally bind the model content identity and exact
selected lens matrices. Run schemas v1 through v4 remain readable.

## Coefficient Sweep

`qwen-lens sweep` runs an ordered coefficient sweep without reloading the model,
tokenizer, lenses, uploaded directions, or readout buffers:

```sh
cargo run -q --release -p qwen-cli --bin qwen-lens -- sweep \
  --model /path/to/Qwen.gguf \
  --plan /path/to/plan.json \
  --operation prefill-add \
  --coefficients 0,0.025,0.05,0.1,0 \
  --prompt "The capital of France is" \
  --max-new-tokens 8 \
  --seed 17 \
  --output /path/to/new-sweep
```

The source plan must be an ordinary dense or MoE Qwen plan with finite authored
coefficients. Its selected `--operation` must be nonzero so every enabled arm
shares the conservative source-plan prefill topology. Muse Glimmer and
Flash-Next are rejected. The command changes only the exact `--operation`
coefficient, preserves coefficient order and duplicates, and accepts at most 64
values that pass the selected action's finite-scale validation
(`coordinate_swap` also validates `2 * coefficient`).
Positive, negative, and signed-zero values are retained in the artifacts. A
zero-valued selected operation is disabled: no intervention kernel runs and no
operation application is recorded. Its scalar event still uses the same serial
kernel topology selected by the nonzero source plan, avoiding a zero-control
topology confound.

Single sweeps and resident sweep cohorts share a 512 MiB serialized bundle
budget. Cohort `auto` prefill currently resolves to the documented serial
effective policy; callers do not need to restate that implementation choice.
Cohort request count has no independent prompt-by-arm ceiling. The bounded
JSONL input, model context, one-million-transition work budget, and exact
serialized bundle budget admit the requested campaign instead. A conservative
outer-manifest reservation is checked before model residency and held back from
child output throughout execution.

Every arm gets a fresh sequence and a fresh sampler initialized with the same
requested seed. Arms execute sequentially; no KV state, sampler state, or generated
tokens cross arm boundaries. Automatic packed prefill uses one schedule derived
from the nonzero source plan and one reusable scratch allocation for every arm.
Each child is an ordinary `qwen.lens.run` v5 artifact containing its exact
effective authored and resolved plans. The command
writes all children to a private sibling staging directory and exclusively
publishes a new output directory only after every arm and the manifest are
synced:

```text
new-sweep/
  manifest.json
  arms/000000/run.json
  arms/000001/run.json
```

The `qwen.lens.coefficient_sweep` v3 manifest records producer build identity,
the canonical source-plan path, embedded authored source plan and its
canonical-JSON BLAKE3, selected operation, ordered coefficients, and each child
path, byte length, and BLAKE3 digest. Existing output paths are never replaced.

`inspect-sweep` verifies and summarizes the complete bundle without loading a
model:

```sh
qwen-lens inspect-sweep new-sweep --reference-arm 0 --limit 25
qwen-lens inspect-sweep new-sweep --format json
```

It requires the exact manifest/arms directory topology with no extra entries or
symlinks, checks every declared child length and BLAKE3, parses ordinary
`qwen.lens.run` children, and rejects cross-arm runtime, model path, prompt,
sampler, generation-bound, source-plan-path, or effective-plan drift. Only the
selected operation coefficient may differ. The inspector also re-derives the
common passive-span schedule from the embedded source plan. Coefficient
matching is bit-exact, so `0` and `-0` remain distinct; zero arms must not record
the disabled operation. Inspection is bounded to 512 MiB of child JSON and
1,024 retained exact detail records across the complete report.

The report defaults to the first numeric-zero arm and requires
`--reference-arm` when the sweep has no zero control. It groups exact duplicate
coefficients and exact generated outputs, and includes bounded exact readout
comparisons against the reference. Manifest v3 verifies that every effective
authored plan is exactly the embedded source plan with only the selected
coefficient changed, then independently recomputes every child's resolved
semantic position bindings and execution schedule. Manifest v2 remains readable
for v4 children.
Legacy manifest v1 remains readable and is honestly marked
`unverifiable_manifest_v1` because it did not hash or embed the source plan.

Individual children remain compatible with the normal offline comparator:

```sh
qwen-lens compare new-sweep/arms/000000/run.json \
  new-sweep/arms/000004/run.json
qwen-lens compare new-sweep/arms/000000/run.json \
  new-sweep/arms/000003/run.json
```

## Full Readout

`read-full` dispatches imported Qwen, assembled model-bound Muse, and imported
published Muse transports by manifest schema. It captures one selected prompt
position and returns full-vocabulary lens logits for caller-ordered source
layers, using the deployed model's output norm and head after transport. Muse
reads use an exact cached or fresh Hugging Face-declared GGUF identity and verify
each selected F16 matrix. Muse fails closed if that identity is unavailable; it
never falls back to hashing model weights.

Import the exact pinned eyes-ml J asset without interpreting or executing its
pickle metadata:

```sh
cargo run -q --release -p qwen-cli --bin qwen-lens -- import-muse-full \
  --source /path/to/Muse-Glimmer-30B_jacobian_lens.pt \
  --output /path/to/Muse-Glimmer-30B-jlens-published-v1
```

The importer pins the source SHA-256, exact ZIP inventory, opaque `data.pkl`,
whole extracted payload, and all 51 matrix digests. Published Muse reads require
`--allow-unvalidated-transfer`; model-bound locally assembled Muse assets do not.

The active published profiles are the pinned eyes-ml J asset above and the Muse
R asset described next. Import is profile-driven rather than shape-driven: an
arbitrary `.pt` with the same dimensions is rejected. The pickle is never
interpreted. Profiles may declare either 51 separate F16 matrix storages or one
contiguous rank-3 F16 storage; both normalize to matrix-major
`transport.f16le` with orientation
`[source_layer, target_output_coordinate, source_coordinate]`.

### Muse R asset

The active CUDA-produced Muse R profile pins
`brittlewis12/muse-glimmer-30b-r-lens-checkpoints` at immutable revision
`b406c8465c9a49657e30af07753cd08ae7f96f56`. Its logical shape is
`[51,6656,6656]`: post-block residuals 0 through 49 map into target block 50,
with an exact F16 identity at source/target block 50; block 51 is not present.
Its recipe uses the first 25 unfiltered, unshuffled documents from
`NeelNanda/pile-10k`, `max_seq_len=128`, and `skip_first=4`.

Import it with the same command surface:

```sh
cargo run -q --release -p qwen-cli --bin qwen-lens -- import-muse-full \
  --source /path/to/muse-glimmer-30b-r-lens.pt \
  --output /path/to/Muse-Glimmer-30B-rlens-published-v1
```

The profile pins the immutable source and opaque `data.pkl` SHA-256, Torch ZIP
inventory, serialization ID, whole payload and every matrix digest, fitted model
and tokenizer revisions, exact method/estimator/arithmetic contracts, corpus
revision and selection, and raw-text/token-ID digests. Import verifies every F16
value is finite and matrix 50 is bit-exact positive-zero identity without
executing pickle. The companion JSON report is provenance input, not an
integrity manifest. BF16-checkpoint-to-GGUF and image-token transfer remain
explicitly unvalidated and require `--allow-unvalidated-transfer`.

The R fit targets post-block 50 and its fitter decodes that target residual by
applying the final output norm and LM head directly. The local reader does the
same: block 51 is intentionally not executed after transport. These are
target-50 R-lens logits, not claims about the model's unperturbed continuation
logits after its remaining block.

`assemble-muse-full` remains the separate model-bound local J/R path: it accepts
authenticated row-shard artifacts and emits `muse_glimmer.full_transport` v3.
Those local full artifacts currently support `read-full`; published full
transports additionally support `trace-full` and plan projection.

```sh
cargo run -q --release -p qwen-cli --bin qwen-lens -- read-full \
  --model /path/to/model.gguf \
  --full-lens /path/to/full-lens \
  --identity-cache /path/to/private-cache \
  --allow-unvalidated-transfer \
  --prompt "What does this mean?" \
  --layers 25,50 \
  --top-k 10 \
  --include-vector
```

`--include-vector` adds the selected transported target-coordinate residual
before output RMSNorm. Its JSON includes operation, stage, dtype, coordinate,
hidden size, shape, and values. Omit it for compact top-k output.

## Full-Transport Trace

`trace-full` reads an imported Qwen3.6 27B J/R, Qwen3.8 27B J, or published Muse
Glimmer J/R transport across every requested prompt position and source layer:

```sh
cargo run -q --release -p qwen-cli --bin qwen-lens -- trace-full \
  --model /path/to/Qwen3.6-27B.gguf \
  --full-lens /path/to/qwen3.6-27b/r-lens-native-v1 \
  --messages /path/to/messages.json \
  --message-mode thinking \
  --layers 0,31,62 \
  --vectors 31:5,62:5 \
  --top-k 8 \
  --output trace.json
```

Use exactly one of `--prompt`, `--token-ids`, `--user`, `--messages`, or
`--open-responses`. `--message-mode` is accepted only with user/messages. For
Qwen3.6 J/R, omission or `auto` preserves
the existing Auto transition, while `thinking` and `no-thinking` select those
exact transitions; effort tiers are rejected. For Qwen3.8 J, omission,
`xhigh`, or `thinking` selects the normal xhigh transition, `low` and `medium`
select their exact effort tiers, and `no-thinking` selects the exact closed
thinking transition; `auto` is rejected. Open Responses takes its rendering
mode from `reasoning.effort` / `x_qwen.no_thinking`. `rendering.generation_mode` records
the resolved mode (`auto`, `thinking`, `no_thinking`, or a `thinking_*` tier).
Muse accepts `thinking`, `low`, `medium`, `high`, or `xhigh`; `thinking` is an
alias for its released `high` default. A command-line mode overrides an omitted
document value and conflicts fail closed.
Inputs are never truncated and are bounded at 128 tokens. Omit `--layers` to
trace all 63 published source layers.

Qwen performs one packed prompt forward, streams only the selected F16 transport
matrices, and keeps full logits on Metal. Published Muse tracing currently
accepts raw, literal-token, and exact annotated ATEM system/user/assistant
inputs; Open Responses remains Qwen-only. Muse additionally
requires `--identity-cache` and `--allow-unvalidated-transfer`. It captures the
scalar prefill once, uploads each selected 88.6 MB matrix once, and reuses it
across positions; `execution_mode` records this rather than claiming packed
prefill. Muse lens logits apply the deployed output RMSNorm, native head, output
multiplier, and final softcap, with no softmax or post-target blocks.

Without `--output`, stdout
defaults to the complete JSON document. With `--output`, stdout defaults to a
compact summary; explicit `--format json` always prints JSON, while explicit
`--format summary` without an output intentionally discards the full document.
`--output` is resolved and validated before model loading, then atomically
replaces a normal result file after successful execution. The version-3 JSON
contains exact input token pieces, compact layer-position top-k cells, explicit
logit/no-softmax semantics, lightweight model/tokenizer locator identities,
timings, and token occurrence counts globally and per layer. One occurrence
means one token ID in one returned top-k list. Qwen tracing checks artifact
geometry and byte length without rescanning the 3.3 GiB payload. Muse verifies
each selected matrix and resolves a declared/cached GGUF identity without
hashing model weights.

For structured messages and Open Responses, the renderer authors byte spans
while constructing the exact prompt. Exact token boundaries are recorded when the full tokenization has them;
nonstructural role, content, separator, and reasoning spans use null token bounds
when an authored boundary falls inside a BPE token. Structural selector markers
must always have a nonempty exact token range or trace creation fails. This
enables selectors such as
`role:user:end`, `role:assistant:start`, and `channel:thinking:start` without
searching decoded delimiter-shaped text.

`--vectors LAYER:POSITION,...` optionally includes up to 32 selected transported
target-space vectors inline in the same JSON. It may be repeated; cells must be
unique, must use selected layers, and use zero-based tokenized-input positions.
Each F32 vector is `transport_layer * post_block_residual` in target coordinates
before the deployed output RMSNorm and LM head. It is not the source activation,
logits, or an observed target-layer activation. The JSON reports the J/R method,
source revision, payload digest, shape, coordinate semantics, and deterministic
cell order alongside the values.

## Offline Inspection

`inspect` accepts trace schema versions 2 and 3 and never loads a model:

```sh
qwen-lens inspect trace.json summary
qwen-lens inspect trace.json positions
qwen-lens inspect trace.json aggregate --limit 25 --layers 18..31,62 \
  --position message:0:end --position prefill:last
qwen-lens inspect trace.json position message:1:start --layers 31,62 --top-k 8
qwen-lens inspect trace.json token --id 18659 --position role:assistant:end --layers 18..62
```

`--format json` returns a typed result for any view. Version 3 requires producer,
model, tokenizer, score-semantics, execution-mode, and rendering metadata;
version 2 remains readable without those fields. Aggregate rows preserve
exact token IDs and show total top-k occurrences, top-1 occurrences, best rank,
raw per-layer counts, and a one-character-per-selected-layer ASCII stripe. A
token missing at a cell is reported as `outside captured top-k`; it is never
treated as a known zero score or full-vocabulary absence. Token IDs at or above a
declared model vocabulary size are rejected. Semantic selectors require
version-3 renderer-authored exact structural spans; numeric positions and
`prefill:last` also work with version 2. `positions` emits one compact line per
input token and a typed JSON equivalent, including aligned structural labels
and a separate exact anchor inventory. Its anchors include `prefill:last`,
explicit `message:N:start|end`, the generated assistant start, channel edges,
and role aliases. Role aliases resolve to the last matching authored message
boundary and report that message index. Generated channel markers take
precedence over history; an open generated channel makes its `:end` selector
fail rather than falling back to a historical close. Historical markers remain
visible on their token lines with message metadata. Unaligned content spans do
not acquire invented token ranges.

`aggregate`, `position`, and `token` accept `--layers` as comma-separated IDs
or inclusive ranges. Ranges select captured layers in the artifact's captured
order; unknown explicit IDs, empty selections, and duplicates are errors.
Repeatable aggregate `--position` selectors recompute ranking, counts, and the
layer timeline over exactly the resolved positions. JSON results record the
effective layers and positions and retain explicit top-k censor fields.

## Exact Artifact Comparison

`compare` performs one bounded, offline comparison of two artifacts of the same
supported schema:

```sh
qwen-lens compare left.json right.json --format text --limit 25
qwen-lens compare left.json right.json --format json --limit 25
```

It accepts only validated, same-version `qwen.lens.trace` v2 or v3 pairs, or
`qwen.lens.run` v1 through v5 pairs. Each input must be a regular non-symlink file no
larger than 256 MiB. Mixed schemas, unknown versions, incompatible trace
geometry, score semantics, input rendering provenance, and runs with different prompt IDs, runtime, model
path, stable execution identities, or sampler settings are rejected. Cache-state
outcomes remain recorded but do not make identical content identities incomparable.

Trace cells align only by their exact captured layer/position coordinates and
candidates by token ID. Captured-top-k entry/exit is explicit; absent ranks and
logits remain null. The result includes top-1 changes, bounded cell and aggregate
token-ID/display differences, and vector metrics only when target layer,
coordinate metadata, dtype, stage, hidden size, cell coordinates, and dimensions
are compatible. Run readouts align only by the complete documented readout key,
then by `(token_id,row_id,word_id,label)`. Incompatible score kinds or candidate
universes remain unmatched; one-sided returned candidates are reported as
entering or exiting the readout top-k. Generated-token divergence, stop reasons,
operation applications, execution-topology equality, and native-capture counts are reported without executing
or loading a model. `--limit` bounds deterministic detail lists while the typed
JSON retains total counts.

## Plan

Paths are resolved relative to the plan file. This example uses a completed
`fit-tokens` J or R artifact and a published template lens:

```json
{
  "version": 1,
  "lenses": [
    {
      "kind": "native_selected",
      "id": "j",
      "artifact": "artifacts/j-token-fit"
    },
    {
      "kind": "workspace_template",
      "id": "phrases",
      "weights": "template/templates+phrases_v3.safetensors",
      "labels": "template/template_words+phrases_v3.txt"
    }
  ],
  "directions": [
    {
      "id": "token-direction",
      "lens": "j",
      "row": { "kind": "token_id", "token_id": 1234 },
      "normalization": "unit_l2"
    },
    {
      "id": "phrase-direction",
      "lens": "phrases",
      "row": { "kind": "label", "label": "ice cream" },
      "normalization": "unit_l2"
    }
  ],
  "operations": [
    {
      "id": "prefill-steer",
      "scope": {
        "layers": { "kind": "values", "values": [18] },
        "prefill": { "kind": "range", "start": 0, "end": 3 }
      },
      "action": {
        "kind": "residual_l2_fraction",
        "direction": "phrase-direction",
        "coefficient": 0.02
      }
    },
    {
      "id": "decode-ablation",
      "scope": {
        "layers": { "kind": "values", "values": [18] },
        "decode": { "kind": "all" }
      },
      "action": {
        "kind": "projection_ablate",
        "direction": "token-direction",
        "coefficient": 1.0
      }
    }
  ],
  "readouts": [
    {
      "id": "j-live",
      "lens": "j",
      "scope": {
        "layers": { "kind": "values", "values": [18] },
        "prefill": { "kind": "all" },
        "decode": { "kind": "all" }
      },
      "top_k": 10
    },
    {
      "id": "phrase-live",
      "lens": "phrases",
      "scope": {
        "layers": { "kind": "values", "values": [18] },
        "decode": { "kind": "all" }
      },
      "top_k": 10
    }
  ]
}
```

Plan v1 keeps the original numeric-only scope grammar. Plan v2 additionally
allows exact renderer-authored prefill selectors while retaining the same lens,
direction, action, and numeric decode contracts:

```json
{
  "version": 2,
  "lenses": [
    {"kind":"native_selected","id":"j","artifact":"artifacts/j-token-fit"}
  ],
  "directions": [],
  "operations": [],
  "readouts": [{
    "id": "boundary",
    "lens": "j",
    "scope": {
      "layers": {"kind":"range","start":24,"end":50},
      "prefill": {
        "kind": "rendered_spans",
        "selectors": [
          {
            "span_kind": "message_content",
            "role": "user",
            "occurrence": "last",
            "edge": "end"
          },
          {
            "span_kind": "generated_assistant_start_marker",
            "edge": "start"
          }
        ]
      }
    },
    "top_k": 8
  }]
}
```

Each rendered selector requires `span_kind` and `edge`; it may additionally
constrain `message_index`, `tool_call_index`, `role`, `channel`, and `label`.
`occurrence` defaults to `unique`, which rejects ambiguous matches; `first` or
`last` must be authored explicitly when several spans are expected. `start`
binds `token_start`, while `end` binds `token_end - 1`. A selected span without
an exact nonempty token range fails rather than assigning a boundary-crossing
BPE token heuristically. Multiple selectors in one scope may not collapse to
the same numeric position.

Useful exact targets include:

- final user content token: `message_content`, `role=user`, `last`, `end`;
- assistant generation marker: `generated_assistant_start_marker`, `start`;
- final tool-result content token: `tool_result_content`,
  `channel=tool_result`, `last`, `end`;
- tool-result terminator: `message_end_marker`, `channel=tool_result`, `last`,
  `start`;
- compiled developer content: `message_content`, `role=system`,
  `label=developer`.

Channel constraints are portable across exact renderers: Open Responses tool
results compile with `role=user`, while Muse ATEM tool results retain
`role=tool`. Filtering on `span_kind` plus `channel=tool_result` preserves that
actual role in the binding artifact. Semantic selectors work with exact Qwen
messages, Open Responses, Muse ATEM, and structured Flash-Next input. They fail
for raw text and literal IDs, which intentionally carry no authored semantic
spans; those paths retain numeric selectors for malformed or forged inputs.
Decode selectors remain numeric because the last reached decode transition can
depend on an early stop token.

An imported published Qwen3.6 J/R or Qwen3.8 J transport can also supply
selected token directions directly. `run` reads and projects only layers
referenced by the plan; no new fit or intermediate artifact is required:

```json
{
  "version": 1,
  "lenses": [{
    "kind": "published_full_transport",
    "id": "published-j",
    "artifact": "/path/to/Qwen3.8-27B-jlens-native-v1",
    "token_ids": [31367],
    "allow_unvalidated_transfer": true
  }],
  "directions": [{
    "id": "lightning",
    "lens": "published-j",
    "row": {"kind": "token_id", "token_id": 31367},
    "normalization": "unit_l2"
  }],
  "operations": [{
    "id": "prefill-add",
    "scope": {
      "layers": {"kind": "range", "start": 24, "end": 58},
      "prefill": {"kind": "all"}
    },
    "action": {
      "kind": "residual_l2_fraction",
      "direction": "lightning",
      "coefficient": 0.1
    }
  }],
  "readouts": []
}
```

The selected token list is bounded to 32 unique model-vocabulary IDs. Its
directions are `transport_layer^T * (LM-head row * output-RMSNorm gamma)` for
the deployed GGUF. The acknowledgement is required because the transport was
fitted on the published BF16 checkpoint and is being transferred to a GGUF
runtime. Legacy `published_full_j` plans remain accepted as an alias.

Native and published selected J/R artifacts report selected-row projection
numerator scores over only the artifact's selected token rows. Muse selected
rows use the same semantics; published Muse covectors include its positive
output multiplier, while the residual-dependent RMS denominator and nonlinear
softcap remain omitted. These are neither probabilities nor full-vocabulary
logits. `workspace_template` artifacts use the camilablank/workspace-lenses BF16
`[layer, row, hidden]` safetensors plus authoritative row-label TSV and report
cosine similarity over the template rows. Every emitted readout carries its
`score_kind` and `candidate_universe`.

Direction rows may select a native `token_id`, a template `template_row_id`, or
an exact unique template `label`. `unit_l2` is required for residual-relative
addition, projection, source-to-target displacement, and coordinate swap. Fixed
addition also accepts `as_stored`.

```json
{"kind":"coordinate_swap","source":"concept-a","target":"concept-b","coefficient":1.0}
```

### Muse Glimmer

Muse uses the same plan actions and scope semantics with model-bound
`native_selected` artifacts or imported `published_full_transport` artifacts. It
accepts token-ID directions, passive readouts, or operations without readouts;
`--identity-cache` is required. Published plans read and project only referenced
source layers and require `allow_unvalidated_transfer: true`. Template lenses,
native-hyper directions, and Open Responses remain unsupported for Muse. Raw
text, literal token IDs, and exact ATEM-rendered `--user`/`--messages` inputs are
supported.

Muse `--messages` accepts either a bare message array or a wrapper containing
`messages`, `tools`, `tool_namespace_descriptions`, `reasoning_strength`, and
`current_date`. Assistant history can preserve `reasoning_content`, `recipient`,
and `end_turn`; structured calls use `tool_calls`, whose routing and turn
boundary derive from their declared function names. Tool results use role
`tool` plus their exact function `name`. Calls and results must match, names
must be safe ATEM identifiers, and the history must await an assistant
continuation. `current_date` customizes the synthesized system message; with an
explicit system message, put the date in that content instead. The rendered
prompt retains exact role, channel, tool-call, and tool-result spans for
semantic Lens selectors and is tokenized without adding special tokens again.

```json
{
  "messages": [
    {"role": "user", "content": "Check Paris weather"},
    {
      "role": "assistant",
      "content": "",
      "reasoning_content": "I should query the forecast.",
      "tool_calls": [
        {"name": "weather.lookup", "arguments": {"city": "Paris"}}
      ]
    },
    {"role": "tool", "name": "weather.lookup", "content": "Sunny"}
  ],
  "tools": [
    {
      "name": "weather.lookup",
      "description": "Read a city forecast",
      "parameters": {"type": "object"}
    }
  ],
  "reasoning_strength": "high"
}
```

```json
{
  "version": 1,
  "lenses": [
    {"kind": "native_selected", "id": "j", "artifact": "muse-j"}
  ],
  "directions": [
    {
      "id": "token",
      "lens": "j",
      "row": {"kind": "token_id", "token_id": 24},
      "normalization": "unit_l2"
    }
  ],
  "operations": [
    {
      "id": "prefill-add",
      "scope": {
        "layers": {"kind": "values", "values": [50]},
        "prefill": {"kind": "values", "values": [1]}
      },
      "action": {
        "kind": "residual_l2_fraction",
        "direction": "token",
        "coefficient": 0.01
      }
    }
  ],
  "readouts": [
    {
      "id": "live",
      "lens": "j",
      "scope": {
        "layers": {"kind": "values", "values": [50]},
        "prefill": {"kind": "all"}
      },
      "top_k": 8
    }
  ]
}
```

### Flash-Next Native Hyper Direction

Flash-Next uses an explicit raw direction source rather than pretending an
ordinary J/R or template row has compatible coordinates:

```json
{
  "version": 1,
  "lenses": [],
  "directions": [{
    "id": "hyper",
    "source": {
      "kind": "native_hyper_f32",
      "path": "direction.f32le",
      "layer": 23
    }
  }],
  "operations": [{
    "id": "add",
    "scope": {
      "layers": {"kind": "values", "values": [23]},
      "prefill": {"kind": "values", "values": [0]}
    },
    "action": {"kind": "fixed_add", "direction": "hyper", "coefficient": 0.25}
  }],
  "readouts": []
}
```

The file is exactly 10,240 little-endian F32 values in native
`[branch, hidden] = [4, 2560]` flattened order. Values must be finite with
nonzero norm; they are applied as stored with no normalization, lifting,
padding, or branch replication. The direction is bound to one layer, and the
operation must select exactly that layer.

## Coordinates

- Layers are zero-based block indices at the post-block residual.
- Prefill indices are zero-based positions in the resolved prompt token IDs.
- Decode step 0 feeds the first sampled token back to produce second-token
  logits. Target the final prefill position to affect the first sampled token.
- Layer and decode selectors are `all`, sorted unique `values`, or an inclusive
  `range`. Plan-v2 prefill scopes may also use `rendered_spans`; artifacts retain
  the authored selectors and exact resolved numeric positions.
- Operations matching one site execute in JSON array order.
- Live readouts observe the residual after all matching operations at the site.

Action formulas are:

```text
fixed_add:             x <- x + coefficient * v
residual_l2_fraction:  x <- x + coefficient * ||x||_2 * v
projection_ablate:     x <- x - coefficient * dot(x, v) * v
source_to_target:      x <- x + coefficient * dot(x, source) * (target - source)
coordinate_swap:       u <- unit(source - target)
                       x <- x - 2 * coefficient * dot(x, u) * u
```

A finite signed-zero coefficient is a valid disabled control. It emits no
intervention kernel or operation-application record; its authored scope remains
available in provenance.

For unit source and target directions, coefficient-1 `coordinate_swap` is
mathematically identical to `x + V(swap(V^dagger x) - V^dagger x)` for
`V = [source, target]`: it exchanges both coordinates and leaves their
orthogonal complement unchanged. Linearly dependent pairs are rejected.
`source_to_target` remains the older directed one-coordinate displacement.

The output is one JSON object containing prompt and generated token IDs,
decoded text, stop reason, reached operation sites, and requested live scores.
Flash runs also include bounded `native_hyper_captures`; each record identifies
the native coordinate, `after_fixed_add` capture stage, shape, flattening,
direction normalization, coefficient, token position, and 10,240 values. A
sampled stop token is reported but never fed back through a decode step.

## Current Runtime

`run` uses fresh sequence state. Ordinary dense auto mode packs only qualified
passive prompt spans; serial intervention execution resumes at every operation,
readout, final-prompt, and decode event. Ordinary MoE, Muse, and Flash-Next stay
fully serial, and `--prefill-execution serial` keeps dense runs fully serial as
well. Ordinary dense and MoE runs consume completed native
selected-token J/R rows and workspace-template rows. Dense Qwen3.6 can project
selected directions from its released matched J/R pair; dense Qwen3.8 can do so
from its published J transport. Flash-Next runs use only explicit native hyper
directions. Muse runs consume model-bound selected-token J/R rows for readout
and all five post-block action kinds. Concurrent or speculative decode and
prefix caching are not selected silently.

## Flash-Next Capability Boundary

Flash-Next `run` supports one native fixed add and post-add capture per serial
token event at completed decoder layers 1 through 47. Execution order is fixed:
add, capture, then let the next layer consume the modified state. Plans fail if
operations overlap at one token event or can emit more than 32 captures. Layer
zero is excluded because PLE occurs inside the fused layers-zero-one transition.

This is raw native-coordinate control, not Flash J/R/template-lens support.
Ordinary 5,120-wide directions are incompatible; readouts, residual-relative
addition, projection, source-to-target displacement, operation stacking, and
packed prefill instrumentation remain unavailable for Flash. No residual metric
or branch-lifting policy is implied.
