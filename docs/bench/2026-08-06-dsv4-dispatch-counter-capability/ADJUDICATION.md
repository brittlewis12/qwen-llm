# Adjudication

Decision: `STOP - CAPABILITY_FALSIFIED`.

On the frozen Apple M4 Max and macOS 15.6.1 stack, creation of an
`AtDispatchBoundary` timestamp buffer fails before any command encoder,
dispatch, model, or GGUF access:

```text
Counter("device does not support dispatch-boundary counter sampling")
```

The repository correctly checks
`supportsCounterSampling(AtDispatchBoundary)` before allocating the buffer.
`AtStageBoundary` supports compute-pass start/end attachment only; it cannot be
repurposed for legal mid-pass samples through `sampleCountersInBuffer` when
dispatch-boundary support is false.

The earlier green development probe silently returned on this error and is not
evidence. The target-aware frozen test records the unsupported capability
explicitly and fails as intended.

Do not run the rejected same-encoder current-asset packet. Do not replace it
with rotating two-encoder boundaries: each retains the exact unowned transition
that invalidated the prior packet, while recovering internal stages would
require cross-run subtraction of independently perturbed cumulative prefixes.

Remove the rejected observer and target probe after archiving. Close the
`BeforeAttentionBody` measurement branch on this device/API contract. Reopen
only after relevant device, OS, or Metal capability drift, or a genuinely new
legal intra-pass timer.

No GGUF was opened and no ignored/current-asset test ran. Pivot to an
independently qualified lane rather than weakening ownership or evidence gates.
CX session `019fd685-acb3-7890-83c9-192cbea48c6e` returns final `STOP`.

Capability log SHA-256:
`86f9e907c14d72935f0b0f01e9a5bc8efe99b778b034cc76edfdf8908e16e568`.
