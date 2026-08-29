# Flash-Next GDN/QSA Named-Stage INVALID

Decision: **INVALID_SCREEN / NO_DECISION**. The preregistered natural N=512
bracket completed, but both split arms produced non-monotonic timestamp streams.
No GDN or QSA child timing is admissible, so no mechanism is kept or killed.

## Candidate

The temporary sampled-only path retained ordinary kernels and dispatch order
while splitting representative layer-5 GDN into seven named encoders and dense
layer-7 QSA into nine. The complete profile plan moved from 128 samples and 68
spans to 156 and 84. The ordinary six-stage path remained in a separately
pinned executable.

Pre-acquisition source review found matching N=512 dispatches, validation,
command reservation, failure poisoning, stage arithmetic, and parent/child
containment. The two binaries shared the same metallib hash.

## Protocol

- Preregistered source and protocol commit:
  `16205e20c6d797fa73574ce7179d4b7464d21603`.
- Protocol blob: `29cef1419b25d291656b50a9ae093010467802fb`.
- Temporary diff SHA-256:
  `6fe43a2698ad487254b254b68aac2a58331d74747f3887e5fabfe72e46b99068`.
- Device: Apple M4 Max. Model: internal-SSD UD-Q3_K_XL from
  `unsloth/Qwen3.8-Flash-Next-GGUF` revision
  `8bdc666649440e9bdc97e16f3f75782c98478ff5`.
- Workload: tracked natural 512-token prompt, no special tokens, one generated
  token, exact 512-forward capacity, one packed command, and no scalar tail.
- Fixed order: separate-process A1-B1-B2-A2 with five seconds between arms. A
  used the pinned six-stage executable; B used the pinned named-stage split.

## Outcome

| Arm | Warm GPU (ms) | Profiled GPU (ms) | Profile resolution |
|:---|---:|---:|:---|
| A1 six-stage | 968.103083 | 968.946500 | accepted |
| B1 named split | 966.756667 | 968.239584 | non-monotonic |
| B2 named split | 968.118625 | 968.821667 | non-monotonic |
| A2 six-stage | 968.613500 | 969.846333 | accepted |

All whole-command observers were accepted and all four generated outputs shared
SHA-256 `e530152ea80c3012dbfdb19a69e554de54aed4511184ae225fcec380233a46e9`.
Six-stage control drift passed every preregistered limit: warm command 0.053%,
profiled command 0.093%, GDN mixer 0.473%, and QSA mixer 0.648%.

Both B arms then failed identically with `packed profile resolved timestamps are
not globally monotonic`. The encode-time sample/span plan checks had passed and
the command completed, but the process did not persist the 156 raw timestamp
values after resolver failure. Offline child recovery is therefore impossible.
Whole-command success and output equivalence do not repair the missing named
timings.

## Disposition

The committed protocol classifies any layout or acquisition failure as invalid
and forbids an optimization decision. No adaptive retry, reordered bracket, or
relaxed resolver was run. A future attempt requires a new preregistered observer
design that persists raw timestamp availability and first-inversion evidence.
It cannot retroactively authorize work from this bracket.

The sampled siblings, constants, and composition changes were removed. Both
temporary executables were deleted and the committed six-stage release was
rebuilt. Raw process logs and the reviewed temporary patch remain under
`target/profiles/qwen4exp-gdn-qsa-stage-screen-20260829/`.

Adversarial protocol, source, and evidence review:
`01a04f90-c9dd-76f3-b063-8d1d01e266ae` (PASS).
