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

## Full Readout

`read-full` dispatches imported Qwen full-J and assembled Muse full J/R assets
by manifest schema. It captures one selected prompt position and returns
deployed full-vocabulary logits for caller-ordered source layers. Muse reads are
bound to an exact cached or fresh Hugging Face-declared GGUF identity and verify
each selected F16 matrix. Muse fails closed if that identity is unavailable; it
never falls back to hashing model weights.

```sh
cargo run -q --release -p qwen-cli --bin qwen-lens -- read-full \
  --model /path/to/model.gguf \
  --full-lens /path/to/full-lens \
  --identity-cache /path/to/private-cache \
  --prompt "What does this mean?" \
  --layers 25,50 \
  --top-k 10 \
  --include-vector
```

`--include-vector` adds the selected transported target-coordinate residual
before output RMSNorm. Its JSON includes operation, stage, dtype, coordinate,
hidden size, shape, and values. Omit it for compact top-k output.

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

### Muse Glimmer

Muse uses the same plan actions and scope semantics with model-bound
`native_selected` artifacts. It accepts token-ID directions, passive readouts,
or operations without readouts; `--identity-cache` is required. Template lenses,
native-hyper directions, and `--messages` remain unsupported for Muse.

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
Flash runs also include bounded `native_hyper_captures`; each record identifies
the native coordinate, `after_fixed_add` capture stage, shape, flattening,
direction normalization, coefficient, token position, and 10,240 values. A
sampled stop token is reported but never fed back through a decode step.

## Current Runtime

`run` intentionally uses fresh serial token-major execution so intervention
schedules remain exact. Ordinary dense and MoE runs consume completed native
selected-token J/R rows and workspace-template rows. Flash-Next runs use only
explicit native hyper directions. Muse runs consume model-bound selected-token
J/R rows for readout and all four post-block action kinds. Concurrent or
speculative decode and prefix caching are not selected silently. `trace-full`
separately uses packed prefill for passive full-J prompt traces.

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
