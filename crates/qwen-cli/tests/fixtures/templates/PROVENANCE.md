# Chat template oracles

Extracted verbatim from `tokenizer.chat_template` GGUF metadata on
2026-08-18. These are reference oracles for the hand-frozen cases in
`../serve_render_fixtures_v1.json` (the unpinned/legacy ChatML contract).
`../qwen36_chat_template_oracle_v1.json`, `../qwen35_chat_template_oracle_v1.json`,
and `../qwen38_chat_template_oracle_v1.json` are rendered *from* the Qwen3.6,
Qwen3.5, and Qwen3.8 files
by `scripts/reference/render_qwen_chat_template.py` (jinja2, the engine
Transformers uses) and pins the released bytes for the digest-verified
Qwen3.6 template; regenerate with `--check` to detect drift.

| File | Bytes | SHA-256 (first 16) | Source model file |
| --- | ---: | --- | --- |
| `qwen36_a3b_chat_template.jinja` | 8,057 | `55d4931433fe502b` | `Qwen3.6-35B-A3B-UD-Q4_K_S.gguf` |
| `qwen38_27b_chat_template.jinja` | 8,945 | `701ba13a085c0c1b` | `Qwen3.8-27B-Q4_K_M.gguf` |
| `ds4_flash_chat_template.jinja` | 13,772 | `e643c31fcec17f34` | `DeepSeek-V4-Flash-0731-UD-IQ3_XXS-00001-of-00004.gguf` |
| `qwen38_27b_unsloth_chat_template.jinja` | 9,993 | `12827f24b742ea4e` | `Qwen3.8-27B-Q8_0.gguf` (Unsloth-patched; also Flash-Next's template; renders identically to the canonical file on `qwen38_chat_template_oracle_cases.json`, verified 2026-09-06) |
| `qwen35_chat_template.jinja` | 7,816 | `7f0e529032c25183` | `Qwen3.5-0.8B-Q8_0.gguf` (extracted 2026-09-06; pinned as `QWEN35_CHAT_TEMPLATE_SHA256`) |

Key facts pinned from these oracles (see SERVE.md S2 correction):

- Qwen3.6 and Qwen3.8 tool calls use the XML-parameter function form
  (`<tool_call>\n<function=NAME>\n<parameter=KEY>\nVALUE\n</parameter>…`),
  not hermes JSON.
- Consecutive tool results coalesce into one `<|im_start|>user` block,
  each wrapped `\n<tool_response>\n{content}\n</tool_response>`, with a
  single `<|im_end|>\n` closing the run.
- DS4 uses a DSML invoke dialect (`<{dsml}tool_calls>` /
  `<{dsml}invoke name=…>`); its fixture cases land with DS4 serve
  support in S3.
