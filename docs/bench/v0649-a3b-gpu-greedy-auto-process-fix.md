# v0.649 A3B automatic GPU-greedy process-predicate repair

Status: preregistration. No v0.649 observation exists. This successor changes
only the quiet-host process classifier and failed-preflight evidence order from
the frozen v0.648 packet.

## Predecessor disposition

v0.648 sealed `INVALID` with `authority=none` after five complete pairs. A host
preflight classified this command as inference because an arbitrary query
argument had basename `qwen`:

```text
opencode recall search qwen --from 2026-05-01 --to 2026-05-16
```

The command's executable was `opencode`; `qwen` occupied neither executable nor
module position. The failure is a classifier false positive, not evidence of
local model or GPU work. The exact rendered line remains in v0.648's sealed
decision and failure, but its failing raw `ps` snapshot was not written because
classification raised before archival.

No v0.648 conformance or timing observation may enter v0.649's estimates,
sample count, gates, or interpretation. Do not mutate its sealed artifact root.

## Changed predicate

The successor may classify a process as competing inference only through a
structurally meaningful command position:

1. exact `comm` basename or `argv[0]` basename for a known inference executable;
2. an MLX console entry point in executable position, matching `mlx_lm` or the
   exact dotted boundary `mlx_lm.*`;
3. for a Python interpreter, the operand immediately following `-m`, again
   matching `mlx_lm` or `mlx_lm.*`.

Known executable names remain `qwen`, `qwen-bench`, `llama-cli`,
`llama-server`, `llama-bench`, `ollama`, and `mlx_lm`. Arbitrary arguments,
query text, prompts, filenames, and shell text must never be compared with that
set.

Every post-creation host check must collect and write raw `ps`, memory-pressure,
and thermal output before evaluating validity. A detected competitor still
invalidates the one-shot packet immediately; the change preserves the evidence
rather than relaxing the quiet-host requirement. Pre-creation failure remains
an unconsumed abort.

If this OpenCode installation is independently shown to perform local Metal
inference in-process, stop before acquisition and add an exact executable plus
subcommand/configuration rule under a new preregistration. Latent capability or
ordinary OpenCode CPU/disk/network activity is not inference evidence.

## Frozen inherited contract

All other semantics from the v0.648 preregistration frozen at
`18ac0eb6560a07d768c6b7ec77c0c52ae2d6c022` remain byte-for-byte controlling:

- candidate implementation `43af9d0f2effd0f86062391a6fddac97bbf46d4a`;
- exact A3B model/prompt hashes and metadata identity pair;
- A=`QWEN_GREEDY_GPU_ARGMAX=0`, B=absent, with the four hybrid flags pinned;
- exact A3B state gate and fresh 16-token product conformance;
- exactly eight new pairs ordered `AB, BA` four times, one first-use plus four
  steady requests per fresh arm process, and five-second cooldowns;
- complete token digest/text, stop, terminal, identity, policy, cache, source,
  build, binary, model, and environment contracts;
- steady `decode_ms/generated_token` primary, request-zero first-use guard,
  paired log-ratio statistics with `n=8`, and diagnostic transition time;
- prefill `[0.97,1.03]`, first-use/TTFT lower-bound `>=0.97`, and primary
  `S>=1.01` plus lower bound `>1.0` gates;
- identical decision precedence and metadata-scoped A3B-only authority.

The committed delta after the candidate implementation must contain only the
sealed v0.648 runner, this preregistration, and the v0.649 runner. Any core,
Cargo, kernel, prompt, base-preregistration, or unrelated path change is a hard
stop.

Artifact root: `target/profiles/v0649-a3b-gpu-greedy-auto-p1/`.
