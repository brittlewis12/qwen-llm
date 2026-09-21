# Guarded-256 candidate policy, version 1

This is an empirical research hypothesis, not an independently derived numerical
tolerance or approval to change the public 32-forward ceiling. The values were
chosen after the repeated calibration corpus and F16/F32 backend control. Freeze
this file and `holdout-256-v1.json` in a reviewed commit before any holdout model
execution. Tokenization/count checks alone are allowed before execution.

## Frozen experiment

- Native production materialized attention, F16 KV, default Q8 matvec; pinned final
  Q8 artifact and IFM F16/flash-off reference. F32 reference is not the target.
- Four newly authored, nonrepeating sources: observational prose,
  transactional code, multilingual editing, and genuinely ambiguous instructions.
  Use native BOS and the first 256 tokens of each fixture, with no repeated filler.
  These are assistant-authored synthetic coverage probes selected after calibration
  but without holdout forward feedback, not a random or independently audited corpus.
  Bind the selected artifact SHA256, tokenizer metadata ID, full token counts,
  and SHA256 of each exact 256-ID prefix before any GPU execution.
- Execute each source at absolute bases 0, 37, and 8191. These are 256-row histories,
  not 8K retained-history tests. Evaluate all rows, not only boundary averages.
- Enforce the JSON's exact top-1, raw error, cosine, KL(reference || native), TV,
  and centered-RMSE gates on each row. Keep the original 42-row strict regression.
  Preserve the failed repeated-corpus results and their original strict bounds.
- Compare native singleton/split/whole append results bitwise at the declared
  boundaries and reject capacity+1 without poisoning the session.
- On one positioned case per source (the continuation bases), compare reference
  and native post-block residuals at layers 0/11/23/35 at visible length 256.
  Enforce the JSON's cosine and relative L2 gates per site, and check traced versus
  untraced reference logits bitwise. This does not validate any fitted transport.
- On those four cases, take a 241-token fixture prefix and follow 15 reference
  argmax transitions to reach 256 rows. Tie-break toward the highest ID. This is
  a fixed mathematical trajectory, including EOS if produced; existing serving
  tests own EOS-stop semantics. Require native token agreement along that path
  and the same per-row numerical gates. Do not exceed the candidate capacity.

## Failure and promotion rules

Record every failing row/site and retain the original artifacts. Do not tune
thresholds, replace difficult fixtures, or rerun with alternate kernels to turn
this frozen experiment green. Any policy revision needs a separately versioned
policy and a newly selected holdout; it cannot erase this result.

Passing this experiment would support only a guarded, explicitly requested final-
Q8 research capacity of 256. It would not qualify all checkpoints, longer history,
chat/tools, compact KV, performance, or arbitrary intervention sensitivity. Public
surface changes still require a separate adversarial checkpoint, exact-boundary
run/bench/lens/serve checks, and clear artifact/precision scope. No additional user
decision is needed to run this experiment; no promotion is automatic.
