# Tokenizer Prompt Corpus

Rendered Qwen chat-template prompts derived from the generic message fixtures in `../tokenizer-messages`.

Rendering shape:

```text
<|im_start|>{role}
{content}<|im_end|>
...
<|im_start|>assistant
```

| file | source messages | thinking mode | bytes | chars | lines |
| --- | --- | --- | ---: | ---: | ---: |
| `current-reva-short-qwen36-strip.txt` | `current-reva-short-qwen36.json` | `strip` | 34539 | 34350 | 180 |
| `current-reva-short-qwen36-preserve.txt` | `current-reva-short-qwen36.json` | `preserve` | 37128 | 36938 | 221 |
| `current-mei-medium-qwen36-strip.txt` | `current-mei-medium-qwen36.json` | `strip` | 51876 | 51520 | 278 |
| `current-mei-medium-qwen36-preserve.txt` | `current-mei-medium-qwen36.json` | `preserve` | 100979 | 100435 | 735 |
| `current-marcus-long-strip.txt` | `current-marcus-long.json` | `strip` | 106495 | 106360 | 1254 |
