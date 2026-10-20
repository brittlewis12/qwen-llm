# Qwen XML Preserve Chat Contract

`qwen_xml_preserve` is a rendering contract, not a model or release identity.
It supports text chat, canonical assistant reasoning, and XML function tools
using the existing Rust Qwen renderer. A renamed posttrain with the supported
template inherits the same behavior; no model-name allowlist is involved.

## Recognition and Boundaries

For `qwen35`/`qwen35moe` headers with the existing Qwen tokenizer gate (`gpt2`,
`qwen35`, 248320 tokens), the resolver compares the **complete tokenized template
program** against the supported reference. It ignores only whitespace between
expression tokens. Output literals, string spelling, branches, all other
syntax, and whitespace-control delimiters must match. Input is bounded at
64 KiB / 16384 tokens. Extra Jinja, comments, altered output, and old
strip-history templates are not this contract. This is not a Jinja interpreter
or a template digest used as model identity.

An explicit match takes precedence over the name-derived rendering fallback.
Otherwise existing official release behavior remains unchanged, including
conversion variants. Unknown releases remain unknown in `capabilities.template`
and sampling identity; `rendering_contract` independently reports recognition.
An unknown template still uses the existing unverified generic fallback, not a
claim of verified chat support. Rendering recognition is not engine admission,
tokenizer byte equality verification, or fitted Lens artifact qualification.
Native Lens remains unavailable for this contract. No Qwen3.8 effort levels are
granted, and no official Qwen3.6 identity is inferred.

## Behavior

- `run --user`, `run --messages`, serve, and info bind the same contract.
- Generation defaults to open `<think>\n`; explicit no-thinking uses the
  canonical empty thinking block. Raw prompts remain verbatim.
- Both house and upstream styles preserve **all** assistant reasoning,
  including turns before later user messages. Missing reasoning becomes the
  canonical empty block. Generation mode does not change replayed history.
- Explicit `history_thinking: "strip"` remains a deliberate override using the
  existing last-query rule, not the upstream template's default behavior.
- Tool arguments use Python scalar spelling (`True`, `False`, `None`, unquoted
  strings) and JSON for arrays/objects, as in the Qwen3.6 renderer.
- Existing text-only input validation still applies; recognizing a template
  containing vision branches does not enable multimodal input.

Unknown Qwen releases overlay valid `general.sampling.temp`, `top_k`, `top_p`,
and `min_p` onto the shared fallback. Invalid or absent fields individually keep
the fallback. Floating values must be finite and within sampler ranges; top-k
must be a nonnegative integer fitting `usize`. Official release defaults remain
release-keyed. Run, serve and info use the same resolver, explicit request values
win, omitted seeds remain fresh, and prompt lookup with omitted temperature
remains greedy. There is no Ornith-specific sampling preset.

## Oracle Provenance

Reference: `ornith-ai/Ornith-1.5-122B`, revision
`dfa07bab90e90746ffbfc0b0cba3126eb9265de2`, **standalone** `chat_template.jinja`:

https://huggingface.co/ornith-ai/Ornith-1.5-122B/resolve/dfa07bab90e90746ffbfc0b0cba3126eb9265de2/chat_template.jinja

SHA-256: `9dd2fbd270feaa1fbef2d4f634d7887c9c506e3bde140f8e7351c8944e8fd235`.
This digest records fixture provenance only. The release's tokenizer config
contains an older strip-history template and is not the rendering oracle.
The standalone template was independently verified against the actual GGUF's
`tokenizer.chat_template`; tests here use synthetic headers, not model weights.

Regenerate/check with Jinja only in the offline-capable CPU fixture tool:

```sh
uv run scripts/reference/render_qwen_chat_template.py \
  crates/qwen-cli/tests/fixtures/templates/qwen_xml_preserve_chat_template.jinja \
  crates/qwen-cli/tests/fixtures/qwen_xml_preserve_chat_template_oracle_cases.json \
  --check crates/qwen-cli/tests/fixtures/qwen_xml_preserve_chat_template_oracle_v1.json
```

Runtime rendering remains Rust with no Jinja dependency. CPU tests cover both
styles, thinking/no-thinking, distinct earlier-turn reasoning, missing reasoning,
tool replay with scalar/compound values, explicit strip, renamed headers,
negative program mutations, release fallback, and GGUF sampling validation.
No GPU inference or actual-weight qualification is implied by these tests.
