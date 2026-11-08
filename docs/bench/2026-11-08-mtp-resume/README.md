# A3B MTP terminal resume audit — INF-KHTK

The reported failure is **batched-versus-serial numerics**, compounded by an
incorrect prompt reference in the audit. The normal decoder's retained state is
bitwise identical to a target-only packed replay with no draft head and no
rejected target tokens. No decoder rollback or position fix is warranted.
This establishes numerical continuation equivalence under the budgets below;
it does not establish bitwise state or exact greedy continuation equivalence.

Measured on Apple M4 Max, 128 GiB, macOS 15.7.9. All model execution used
`QWEN_METAL_LEASE_WAIT=1`; builds used the shared build lock and niceness 10.
The artifact was the MTP-aware `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`, 22,663,387,424
bytes. The target-only artifact has the same basename: select the MTP-aware
variant explicitly. No throughput claim is made from these diagnostic runs.

The frozen [prompt](../tokenizer-prompts/current-reva-n8-interactive-qwen36.txt)
has 419 tokens, including its trailing newlines. Its SHA-256 is
`e265de9742d1b22e566fc108ae26331ccf46166c6f071e73f211e0a1a7e8b474`.
Every reported 128-token comparison retains target token hash
`0a8957324c9e0a2634a60f3c4e5d56d7509434efcc857853af297dcdaf3efc60`.
The terminal state has 546 consumed tokens (positions 0–545); the last emitted
token is pending at position 546.

## Localization and controls

Layer and position numbers are zero-based. [Trace receipts](trace.json) retain
all layers at prefill, the first packet, the first amplified packet, terminal
return, and the target-only replay comparison.

| Comparison | First difference / localization |
| --- | --- |
| Serial versus packed **prompt**, before any MTP packet | Layer 0 QKV projection: the existing same-input GEMM/GEMV oracle measures max difference 0.001807 in the first 16-row chunk, worst row 0. GDN recurrence/conv differ in layer 0; the first attention layer (3) has K/V differences from position 0. |
| Product packed prompt versus MTP packed prompt | Every persistent GDN, conv and live K/V buffer is bitwise equal. |
| Serial versus packed **decode**, with identical packed prompt | First post-block residual difference: layer 0, position 419, max 2.235174179e-8. After packet 0 (positions 419–425), layer-0 GDN/conv maxima are 2.861022949e-6 / 3.814697266e-6; K/V differences start at position 419. |
| First decode amplification | Position 523, layer-20 post-block residual, max 0.01091080904. The next layer's GDN state/conv first jump in packet 41 (521–523): 0.04688799381 / 0.1632905006. Later attention K/V maxima are around positions 523–524. |
| Fixed N8 target-only replay versus real MTP return | Every GDN state, conv history and live K/V byte matches exactly at all 40 layers. Only target tokens are supplied; packets have full acceptance except terminal padding. There is no draft model and no rejected draft suffix. |
| Changed rejected suffix, same physical N8 | All eight keep lengths (1–8), every persistent state buffer and three subsequent serial logits vectors match bitwise. Unreachable KV suffix bytes are deliberately excluded. |

The Q8 prefill matrix kernel rounds tiles to half before MMA
(`kernels/mat_mat_q8_0.metal`, `kernel_mat_mat_q8_0_f32_n16`); its GEMV oracle
uses a different arithmetic path. Decode also changes projection, normalization,
GDN and MoE scheduling. Small differences amplify after the layer-20 MoE block;
these measurements do not claim to isolate a particular expert tie. Disabling
batched mixer, batched route or concurrent FFN separately does not restore
strict equivalence. D1/N2 and D7/N8 produce identical resume metrics here.

## Resume quality and audit correction

The old audit replayed the entire prompt serially, although both normal MTP and
the ordinary product reference use packed prefill. Reusing the actual product
baseline session removes that confound. A prefill-only control (one emitted
pending token, no verifier packets) then has **bitwise equal state and logits**
through 128 resume steps, KL = NLL delta = 0.

Continuation comparisons consume the pending terminal token first. Both sessions
then consume each token chosen by the serial reference, so candidate argmax
changes cannot alter the teacher-forced history. KL uses the entire vocabulary
at temperature one, with f64 log-sum-exp. NLL delta is candidate minus reference,
in nats/token. These are serial-reference greedy continuations, not an
independent corpus or a general model-quality certification.

| Reference / continuation | KV min cosine | Logit min cosine | Mean KL | Max KL | Mean NLL delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| Old serial prompt, 16 steps | 0.9994419856 | 0.9988788770 | 0.000474694 | — | +0.00343634 |
| Common packed prompt, 16 steps | 0.9999987573 | 0.9997098164 | 0.000268023 | 0.00347519 | +0.000579680 |
| Common packed prompt, 128 steps, native banks | 0.9999924768 | 0.9965949825 | 0.00109148 | 0.0504773 | +0.000817200 |
| Common packed prompt, 128 steps, F32 banks | identical | identical | identical | identical | identical |
| Common packed prompt, D1/N2, 128 steps | identical | identical | identical | identical | identical |

All 128 generated tokens match. The 16 resumed argmaxes match; the 128-step chain
includes argmax differences. The original strict numeric gate consequently
remains false and is explicitly reported as `strict_numeric_pass`.

Normal MoE resume uses `moe_distribution_v1`: matching valid positions, finite
state/logits, KV cosine >= 0.9999, mean KL <= 0.01, absolute mean NLL delta <=
0.005, and maximum per-step KL <= 0.1. The mean budgets reuse the existing
[turn-reuse quality budgets](../../PERF-LOG.md) rather than fitting a cosine or
absolute GDN bound to this model's buffer scales. The NLL budget limits the
relative perplexity change to approximately 0.5%; the additional per-step ceiling
allows rare routing-boundary amplification while preventing a long continuation
from averaging away a large outlier. The measured mean KL is about nine times
below its budget, NLL delta six times below, and maximum KL about half its cap.
These are explicit numerical-quality budgets, not a proof of exact decoding.

Dense and oracle audit gates retain their prior strict bounds. Token-stream
mismatch remains a failure for normal MTP. JSON retains raw state/logit metrics,
strict pass/fail, mean/max KL and NLL delta; `--mtp-resume-steps` selects 1–4096
teacher-forced steps (default 16). Trace work is diagnostic-only and invalidates
timing. [Results](results.json) include canonical build identities and controls.

## Reproduction and regression checks

Set `QWEN_MODEL_SEARCH_PATH` to the local model roots and set
`QWEN_A3B_MOE_MODEL` to the **MTP-aware** artifact. Preserve the prompt's trailing
newlines; shell command substitution strips them and produces a different
418-token control.

```sh
lockf -k ~/.cache/qwen-llm/locks/build.lock nice -n 10 \
  cargo build --release -p qwen-cli --bin qwen-bench
QWEN_METAL_LEASE_WAIT=1 python3 - <<'PY'
import os, pathlib, subprocess
subprocess.run([
    'nice', '-n', '10', 'target/release/qwen-bench', 'mtp',
    '-m', os.environ['QWEN_A3B_MOE_MODEL'],
    '--prompt', pathlib.Path('docs/bench/tokenizer-prompts/current-reva-n8-interactive-qwen36.txt').read_text(),
    '--spec-tokens', '7', '--mtp-physical-n', '8', '--tokens', '128',
    '--mtp-resume-steps', '128', '--mtp-state-trace',
    '--include-token-ids', '--output', '/tmp/mtp-resume.json',
], check=True)
PY
```

Repeat with `QWEN_MTP_MOE_NATIVE_BANKS=0` for F32 banks. Omit the trace flag for
an ordinary audit. The end-to-end regression is
`mtp::tests::normal_mtp_terminal_resume_from_product_prompt` in `qwen-bench`;
the rollback witness is
`metal_dflash::tests::mtp_layer_major_rejected_suffix_is_unobservable` in the
library. Both are explicitly selected ignored model witnesses. CPU tests cover
full-vocabulary normalization and invalid inputs, gate rejection controls and
sticky invalid-state aggregation.

Validation: full `cargo test --release -p qwen-llm --lib`: **2464 passed, 745
ignored, 0 failed**. The rejected-suffix witness passes; the four MTP bench tests,
including the end-to-end model witness, pass. `scripts/check-contracts.sh` passes
on the final documentation/code tree. No decoder arithmetic or default draft
policy changed, and no performance claim is attached to the audit correction.
