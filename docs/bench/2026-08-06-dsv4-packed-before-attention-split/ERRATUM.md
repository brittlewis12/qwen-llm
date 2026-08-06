# Provenance Erratum

`ADJUDICATION.md` incorrectly says the predecessor failed before its sampled
warm arm. The frozen source and log show that both untimed warm arms completed,
including exactness and topology checks. The packet then failed during the
post-warm-pair canonical identity assertion on the warm control, before timed
`A/B/A/B/A`.

No warm timing values, timed samples, or performance direction were emitted.
The decision remains `INCONCLUSIVE_HOLD`. This append-only correction does not
change any frozen source, log, or pre-run artifact.
