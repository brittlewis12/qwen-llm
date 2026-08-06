# Adjudication

Decision: `INCONCLUSIVE_HOLD`. No retry and no static census authority.

## Valid Evidence

The separately corrected packet completed the full warm pair and timed
`A/B/A/B/A` campaign. It preserved exact model identity, packed logits, final
hidden output, continuation state, committed tokens, routes, dispatch
count/order/full geometry, and encoder topology. Every stage interval was
nonempty.

The following gates passed:

- ordinary GPU drift: 0.5135%;
- sampled GPU drift: 0.4220%;
- sampled perturbation: 1.3478% / 1.6715%;
- aggregate coverage error: 0.0000115%;
- aggregate transition ambiguity: 1.2136%;
- every stage and CSA/HCA stage repeat gate;
- aggregate parent reproduction: 0.9352 / 1.3377 points;
- parent repeat: 1.0395 points;
- CSA parent reproduction and repeat; and
- HCA parent repeat.

Wall measurements are provenance only and carry no decision authority.

## Failed Gates

Three frozen validity gates failed in sampled arm one:

1. HCA layer 21 single-transition ambiguity was 36.0204%, above 5%.
2. HCA layer 21 combined transition ambiguity was 36.0225%, above 10%.
3. HCA parent reproduction differed by 3.4530 points, above three points.

The first transition between `RawSetupAndHyperPre` and
`AttentionAndCompressorPrepare` contained a 12.872167 ms physical gap. It was
about 2.72% of HCA cohort command time and directionally explains most of the
parent miss, but the frozen protocol cannot reassign, filter, or excuse it.
The classifier therefore correctly emitted `packet_valid=false`, no selected
stage, and no authorization.

## Direction Only

`AttentionAndCompressorPrepare` measured a 350.752 ms normalized median and a
conservative 333.089 ms / 31.4875% lower bound. Its mean CSA/HCA shares were
32.6560% / 32.8999%. It would clear the economic screen in valid evidence.

Those values authorize no census, candidate, savings claim, decomposition, or
product change.

## Static Observer Audit

The retained source proves that no product dispatch lies between attention
hyper-pre and `attention.encode_prepare`; the sampled observer ends one compute
encoder and begins another at that boundary. The command is committed only
after all four encoders are encoded. The gap is therefore GPU-timeline
transition ambiguity, not attributable product work. Static inspection cannot
distinguish encoder-transition overhead from scheduling or preemption.

The accepted parent used one encoder. The split reconstructs that parent by
summing three encoder intervals and excluding their gaps, so a material
inter-encoder gap can invalidate reproduction even when the enclosed kernels
are stable.

The repository already exposes dispatch-boundary timestamp buffers and
`KernelEncoder::sample_counters`. A materially new observer can keep one
pre-expert compute encoder and place five adjacent dispatch-boundary samples
around the same four logical stages. That removes inter-encoder gaps without
filtering, retrying, or relaxing any evidence gate.

This four-encoder packet remains immutable HOLD. Remove its live one-shot
instrumentation after archiving. Reopen measurement only through a separately
reviewed same-encoder protocol with all identities, campaign order,
reproduction gates, and economic thresholds unchanged.

Retained log SHA-256:
`6186cf7fb3e6b444dc9b28db292d3cc6ef8cb521f08c7eef89a5b830bea57952`.

CX review: `019fd685-acb3-7890-83c9-192cbea48c6e`.
