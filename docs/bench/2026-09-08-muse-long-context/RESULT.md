# Muse long-context qualification

## Capacity reservation decoupled PASS

`a23c3b11` removes the optimized-prefill capacity restriction without widening its
arithmetic range. Matrix projections and packed online attention require each
chunk's absolute end<=7168; crossing/beyond chunks use the original graph. Full
capacity admission remains. CLI reports optimized row counts and mixed fallback.

The Current Marcus transcript fixture has adjacent same-role messages and ends
with a completed assistant message. Its adapter joins adjacent content with two
newlines and renders native ATEM without a generation prompt; embedded Qwen
thinking remains ordinary content. Result:46899 tokens, SHA-256
`bdd33a93003545cc7135d431c0bcfae0b36872f72341b3c0187944a826088bb3`.
These are fixed prefixes of an adapted checked-in transcript, not the user's Mara
recording or complete standalone requests at every prefix length.

`2b94a341` qualification passes in370.15s: original versus optimized6884 prefill,
then391 teacher-forced transcript transitions through position7274, capacity7275.
Endpoint cosine0.999999956227/RMS0.000296832/maxdelta0.040961742 pass inherited
>0.99999/<0.002/<0.1 gates. All391 continuation logits pass >0.99999/<0.006/<0.3;
worst cosine0.999997695293/RMS0.002338264/maxdelta0.23886967. All391 top-1 decisions
agree, including the frozen17-position boundary cohort7160..7176. This is not a
claim about391 autoregressively generated tokens. The original prefix remains
bitwise immutable. Same-candidate rewinds at7160 and7168 followed by16-row prefill
match the original graph bitwise in logits/all active KV, proving straddling and
beyond-boundary fallback. Candidate session allocation431996928B.

Actual CLI native Current6229/high/temp0/seed42/17outputs with capacity32768 passes:
same emitted bytes/fingerprint/token_limit,16 transitions,6224 optimized packed
rows. Prefill34758.6ms/179.21tok/s; process36.310089s. Session1789444096B and
aggregate required31657869312B account for the larger reservation. Clean embedded
source2b94a341. This is delivery/selection evidence, not new paired timing or
32K-prompt qualification. Split decode's position range is unchanged.

Retained attempts: initial capacity-01 fails in0.08s before Metal initialization
because strict ATEM rejects adjacent assistant history. The adapter repair and
CPU fingerprint check precede capacity-02; no timed result was rescued. Review
also caught missing ignored-test arguments and completed-history generation-prompt
handling before the initial build. Raw records remain in
`target/profiles/muse-live-prefix/{capacity-01*,capacity-02*,cli-prefill-R-01*}`.
