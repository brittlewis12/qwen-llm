# Lens Run

`qwen-lens run` performs one fresh, serial Qwen run with live workspace-lens
readouts and optional ordered post-block interventions.

```sh
cargo run -q -p qwen-cli --bin qwen-lens -- run \
  --model /path/to/model.gguf \
  --plan /path/to/plan.json \
  --messages /path/to/messages.json \
  --max-new-tokens 32
```

Use exactly one of `--prompt`, `--token-ids`, or `--messages`. Raw prompts use
the tokenizer's configured special-token insertion unless
`--no-special-tokens` is set. Message input is a strict system/user/assistant
array (or `{"messages": [...]}`) ending in a user turn. Literal token IDs are
passed unchanged.

Sampling defaults to greedy. `--temperature`, `--top-k`, `--top-p`, `--min-p`,
and `--seed` expose the existing deterministic native sampler.

## Packed Full-J Trace

`trace-full` reads the imported Qwen3.8 27B full J-lens across every requested
prompt position and source layer:

```sh
cargo run -q --release -p qwen-cli --bin qwen-lens -- trace-full \
  --model /path/to/Qwen3.8-27B.gguf \
  --full-lens /path/to/Qwen3.8-27B-jlens-native-v1 \
  --messages /path/to/messages.json \
  --layers 0,31,62 \
  --vectors 31:5,62:5 \
  --top-k 8
```

Use exactly one of `--prompt`, `--token-ids`, or `--messages`. Message input uses
the exact Qwen3.8 generation template with medium thinking and no injected
effort instruction. Inputs are never truncated and are bounded at 128 tokens.
Omit `--layers` to trace all 63 published source layers.

The command performs one packed prompt forward, streams only the selected F16
transport matrices, and keeps full logits on Metal. JSON output contains exact
input token pieces, compact layer-position top-k cells, timings, and token
occurrence counts globally and per layer. One occurrence means one token ID in
one returned top-k list. Runtime tracing checks artifact geometry and byte
length, but does not hash the model or rescan the 3.3 GiB payload.

`--vectors LAYER:POSITION,...` optionally includes up to 32 selected transported
J-space rows inline in the same JSON. It may be repeated; cells must be unique,
must use selected layers, and use zero-based tokenized-input positions. Each
F32 vector is `J_layer * post_block_residual` in target coordinates before the
deployed output RMSNorm and LM head. It is not the source activation, logits,
or an observed target-layer activation. The JSON reports shape, coordinate
semantics, and deterministic cell order alongside the values.

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

Native selected artifacts may be J or R fits. Their live scores are F64 dot
products over the artifact's selected token rows. `workspace_template`
artifacts use the camilablank/workspace-lenses BF16 `[layer, row, hidden]`
safetensors plus authoritative row-label TSV and are scored by F64 cosine over
the raw post-block residual.

Direction rows may select a native `token_id`, a template `template_row_id`, or
an exact unique template `label`. `unit_l2` is required for residual-relative
addition, projection, and source-to-target displacement. Fixed addition also
accepts `as_stored`.

## Coordinates

- Layers are zero-based block indices at the post-block residual.
- Prefill indices are zero-based positions in the resolved prompt token IDs.
- Decode step 0 feeds the first sampled token back to produce second-token
  logits. Target the final prefill position to affect the first sampled token.
- Selectors are `all`, sorted unique `values`, or an inclusive `range`.
- Operations matching one site execute in JSON array order.
- Live readouts observe the residual after all matching operations at the site.

Action formulas are:

```text
fixed_add:             x <- x + coefficient * v
residual_l2_fraction:  x <- x + coefficient * ||x||_2 * v
projection_ablate:     x <- x - coefficient * dot(x, v) * v
source_to_target:      x <- x + coefficient * dot(x, source) * (target - source)
```

The output is one JSON object containing prompt and generated token IDs,
decoded text, stop reason, reached operation sites, and requested live scores.
A sampled stop token is reported but never fed back through a decode step.

## Current Runtime

`run` intentionally uses fresh serial token-major execution for ordinary Qwen
dense and MoE runtimes so intervention schedules remain exact. Concurrent or
speculative decode, prefix caching, and Flash-Next/qwen4exp are not selected
silently. `trace-full` separately uses packed prefill for passive full-J prompt
traces; live `run` consumes completed native selected-token J/R rows and
workspace-template rows.

## Flash-Next Library Seam

Flash-Next/qwen4exp is not yet selected by `qwen-lens run`. Its serial runtime
can now capture the persistent native hyper state after completed decoder layers
1 through 47 and optionally apply one fixed F32 addition at that site. The
direction is exactly `branch_count * hidden_size` values (10,240 for the current
model) in the same flattened order as the capture. Execution order is fixed:
add, capture, then let the next layer consume the modified state. Layer zero is
excluded because PLE occurs inside the fused layers-zero-one transition.

This is an architecture-specific foundation, not J/R-lens support: ordinary
5,120-wide Qwen directions are incompatible, packed prefill is not instrumented,
and no residual-relative metric or branch-lifting policy is implied.
