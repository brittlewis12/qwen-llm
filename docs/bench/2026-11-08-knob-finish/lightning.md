# Retire authoritative F16 Lightning query staging, 2026-11-08

Delete `QWEN_DSV4_LIGHTNING_F16_MATRIX`, its DS4 singleton selector, 16 KiB
query staging allocation, wrapper, qualification record and exclusive tests.
Keep the family-neutral F16 matrix scorer and its GLM callers/tests.

The decisive evidence already exists in **PERF-LOG, 2026-08-08, “DeepSeek V4
Real-History F16 Lightning KILL”**. A real 97,040-token snapshot and 65 causal
transitions compare 1,365 CSA decisions. F32 controls reproduce exactly; F16
changes 935 selected sets and 1,265 routes, then generated tokens, logits and
state. First same-input mismatch: position 97,042, layer 38, F32 cutoff margin
2.992e-5 versus F16 2.043e-4. The candidate's larger margin is not a certificate.
The earlier knob audit missed this entry; a four-token screen cannot overturn
that longer failing replay. Historical PERF-LOG entries are unchanged.

## Additional current M4 evidence (partial probe)

Source `1656c1fab9e791d815f18096ae2d61ca33aec3d2`, based on main `7d90e34a`.
[Probe addition](lightning-probe.patch), [driver](run-lightning.py),
[raw stderr](lightning-partial.log), [identity and monitoring](lightning-partial-manifest.json).
The probe patch applies to the pre-deletion DS4 source (the addition itself
was committed immediately after `927aa880`). It is archival, not a live test.

The mounted released IQ3_XXS checkpoint passed the current census check.
Native tokenization produced 78,236 tokens of real Rust engine source; the
first 65,536 were causally executed through today's default 4096-row packed
prefill. **No zero-filled or tiled synthetic history** was installed.
Prefill took 519.954 s; the prefix digest is
`3a69efba6c6ae0b8f173dc606064d473777bd1bdafc7cabafd26d269ca97d1459`.
Prompt IDs SHA256:
`6d6e1bada31c6a8ec0ba57a4c07e45aad5888218d0007db65322e5285cbaadf3`.

From the same snapshot, four teacher-forced tokens were executed for warm
control, control, matrix, control. Every F32 arm ends at
`158d12abd3493003b8ef20d87ca57e8f5769fa32e4e061de61e1cee04485bbb6`;
the F16 arm ends at
`fc904b1c2eaa565f418c1bbc0960ca776be8071cac2035305b5a60783fa7f5be`.
This corroborates changed behavior on current real history.

**This is not a passed qualification.** The later per-token cutoff collection
failed with `Diagnostics(DuplicateCapture)`: the throwaway probe used the
single-use `take_decision_transcript` rather than its reset variant. Thus no
new cutoff, logit-agreement or speed claim is made from the partial run. The
already complete historical replay settles deletion, and current state already
differs; rerunning an expansive promotion packet is unnecessary. The failed
probe is removed from the live test suite along with the rejected path.

The campaign held the build lock and canonical Metal lease, nice 10. All
monitoring observations show thermal pressure 0 and no rustc. The final
shipping-tree checks and greedy smoke exercise the retained F32 path.
