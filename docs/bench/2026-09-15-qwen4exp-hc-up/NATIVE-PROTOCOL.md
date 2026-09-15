# Native HC Qualification, Frozen Before Execution

Protocol-2 component screen passes numerically and clears complete-HC floors:
38.2602% GPU saved over twelve full HC reads, controls2.0899%. Up+mean60.0479%
is diagnostic; extrapolated2.5573ms/97 reads is NOT native attribution.

Research-only thread-local scoped override at singleton HC up+mean. Same shader
as the component screen, no production route/default/option. Scope counts both
arms, rejects nesting, unwinds safely. Require exact4/2560/320/Q8 shape and
97calls/token. Raw gate scratch remains written; no new GPU allocation. Split
QSA product option is ON in BOTH arms to test incremental leverage after the
already delivered attention option, not remeasure attention's gain.

Use existing UD-Q3_K_XL, one current packed2179 SSH prefix plus its32 continuation
tokens; no second reference prefill or new weights. Hold production lease and
real wired-memory check before Metal; preflight candidate pipeline before work.
Checkpoint restores persistent state, hyper/logits, QSA/session lengths and PLE
history. Baseline/restored-baseline bitwise32-row/state replay precedes candidate.
Save both32x248320 raw logit matrices before candidate census/numerical gates.

Unchanged existing native gates: full rows same argmax, cosine>0.99999999,
RMS<1e-4, maxabs<1e-3. Hyper/F32 state RMS<=3e-4/maxabs<=.01; newly appended F16
cache rows RMS<=1e-3/maxabs<=.03125, separately from old prefixes. Old prefixes
remain immutable, unused suffixes agree. All finite. Candidate census3104 HC
calls, no old singleton mean,384 split-QSA calls. No model-quality/default claim.

Only after numeric PASS: fixed warm four-forward ABBA, then measured four-forward
ABBA from the same checkpoint. Readbacks/restores outside clocks; own-arm full
rows/hyper remain bitwise during replay. Mean AND both pairwise GPU savings>=5%,
executor-wall>=3%, A spread<=5% each axis. Instability INCONCLUSIVE, budget miss
HOLD. These are executor times, not request throughput. No threshold adjustment
or control-fishing run. Rechart next toward complete-MoE observation; don't spend
another sweep on HC body or attention without parent-budget evidence.
