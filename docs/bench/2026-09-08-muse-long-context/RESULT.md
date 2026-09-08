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

## Long attention and complete fresh prefill PASS

`46c7f729` moves the host's materialized-score limit into its materialized branch
and separates matrix/online end limits. Production initially remains bounded7168;
test-only attention expansion changes no projections above that boundary.
Seven primitive cases cover crossing7168, full/sliding8K and32K, the32768 endpoint,
offset views, poisoned candidate outputs and an independent CPU sliding-window
average. All numerical/guard gates pass. Separate warm then measured ABBA8chains:

| Base / rows / window | GPU ABBA ms | Mean A -> B ms | Saved | A spread |
| --- | --- | --- | ---: | ---: |
| 8064 /128 /full | 210.359063 /7.720562 /7.610578 /210.098422 | 210.228742 ->7.665570 | 96.354% | 0.124% |
| 32640 /128 /full | 1614.124375 /36.478156 /36.332953 /1614.527734 | 1614.326055 ->36.405555 | 97.745% | 0.025% |
| 32640 /128 /2048 | 19.440760 /1.795651 /1.769771 /19.407880 | 19.424320 ->1.782711 | 90.822% | 0.169% |

All50% mean/both-pair GPU and5% A control gates pass. Full-attention reference
above7168 is the actual per-row decoder fallback, not an invalid materialized call.
An authentic bounded8064-prefix/128-row model chunk then passes: A7295.628ms,
B3515.024ms, identical original projections in both arms. Endpoint cosine
0.999999999350/RMS0.000038973/abs0.006640911; all-row residual RMS0.000231543,
written-KV RMS0.000200276 and preceding-prefix immutability pass (97.44s).
Residual/KV tolerances aggregate the compared rows, not per-row elementwise bounds.

`0cd3c30e` subsequently qualifies complete fresh8K, then32K only after8K passes.
A is the delivered bounded hybrid: matrix/online through7168, original kernels
afterward. B extends BOTH matrix projections and packed online attention to the
cell end. These results do not provide transitive all-original32K tolerance bounds.

Before allocating weights/sessions, aggregate admission prices both full sessions
and a64MiB CPU oracle allowance. Fresh B resets to zero with NaN-poisoned KV and
recomputes the entire prefix. Its unmodified whole-prefill wall is retained, along
with endpoint and all-active-KV hash. The final128-row live attention-only probe
uses original projections in both arms on B's authentic prefix; candidate suffix
KV is poisoned before probe B. The full extended final chunk is then replayed to
restore B, requiring bitwise original endpoint/full active KV and exact frontier.
Only after this cheap admission/restoration passes is complete fresh A executed.

| Tokens | Fresh B ms / tokens per second | Fresh A ms / tokens per second | Active KV cosine / RMS / reported max delta | Test wall |
| --- | --- | --- | --- | --- |
| 8192 | 46937.875083 /174.529 | 91309.284333 /89.717 | 0.999999721522 /0.000746357 /0.8515625 | 156.49s |
| 32768 | 231585.924542 /141.494 | 3136435.917459 /10.448 | 0.999997781869 /0.002107048 /3.6953125 | 3414.30s |

All active KV is finite and passes cosine>0.9999/RMS<0.01 before continuations.
Max delta is reported, NOT an elementwise gate. Both independently generate the
same17 greedy IDs and all endpoint/16-continuation logit gates pass unchanged:
endpoint cosine>0.99999/RMS<0.002/abs<0.1, continuation >0.99999/<0.006/<0.3.
At32K endpoint cosine0.999999988880/RMS0.000162606/abs0.022789955; worst continuation
absolute delta0.024003386. Both original prefixes remain bitwise immutable.
At32K each session observes1790296064B; aggregate required33516208128B includes
the oracle allowance. KV comparisons stream shared storage rather than copying it.

The candidate-primed32K live chunk passes with A25290.831/B3828.577ms and original
projections: residual RMS0.000145710, written-KV RMS0.000164349, endpoint delta
0.001897812. Probe31486.498ms and restoration1107.292ms are separate from fresh B;
they are neither subtracted from nor added to its recorded interval.

All whole-fresh times are **single B-first/A-later diagnostics**, including first
use/progress logging, not ABBA-controlled speedups or cold/HTTP timing. The complete
slow32K reference actually ran; no kernel/chunk projection substitutes for it.
One attempt per fresh cell; no rescue. Independent source/result review passes.

## Bounded 32K rollout and native CLI PASS

`b50ffc19` (rebased from cf059154) raises both production packed-prefill limits to
32768, retaining Q8/unified M4 Max opt-in eligibility, original default math,
full-capacity admission, absolute-end/straddling fallback and split-decode's old
position bound. Historical reference tests explicitly retain BOTH7168 limits.
The product-constructor test recomputes32768 fresh tokens and matches the qualified
17-ID stream, then verifies bitwise original-graph fallback on identical KV at
32760/32768+16. Prefix immutability/frontiers pass.

The CPU exporter selects the largest normalized Current Marcus history ending on
a user turn that fits32768, renders the native generation prompt, and records:
32729 tokens,40 messages, token SHA-256
`6f9800353069ee12d99419651d6d3189e65997fa0d517cebf4005bb1f3212944`.
This is a complete native request, distinct from the fixed32768-token numerical
prefix. Actual `qwen run`, high/temp0/seed42,17 outputs/16 transitions, capacity32745:

- Prefill230847.402834ms /141.778tok/s;32720 packed rows plus9 scalar tails.
- Generation4054.618542ms; CLI4.193 emitted-tokens/s uses17 as numerator.
- External process235.527020s; token_limit finish, clean embedded sourceb50ffc19.
- Session1788231680B; aggregate required31656656896B, no new attention scratch.
- Output fingerprint `488b4f27680515715871df93994c2821efa153597bf603b82327079c3af42b94`.

This closes bounded delivery, not same-input17-ID reproduction for the distinct
CLI request or a cold-conditioned speedup. Numeric/greedy qualification does not
establish arbitrary sampled-output equivalence. Raw `long-attention-01*`, score,
`long-chunk-01*`, `fresh-{8k,32k}-01*`, `delivery32k-01*`, exporter and
`cli-long32k-01*` remain under `target/profiles/muse-live-prefix/`.

## Whole long-context split decode PASS

`a49678ec` reuses the whole-forward instrument for8192 and32768 prefixes of the
same adapted transcript. Admission includes the session-owned540672B partial
buffer, held in both arms. Optimized prefill primes8K; after its packet, rewind
excludes the generated suffix and extends the actual fixture through32K, checking
the retained8K prefix hash. No split-force hook is active during prefill/extension.

A uses original scalar decode (the shipped selection at those positions when
acquired); B forces only split attention. Each arm starts at the same frontier
and saved logits, then performs16 completed forwards plus17 greedy selections.
All16 returned logit vectors are retained, adding CPU oracle storage. B's new KV
rows are NaN-poisoned before its untimed oracle. All17 independently selected IDs
agree; all16 logits pass cosine>0.99999/RMS<0.002/abs<0.1, new KV passes
cosine>0.99999/RMS<0.002, and existing prefixes remain bitwise immutable.

| Prefix | Whole16-forward ABBA ms | Mean A -> B ms | Saved | A spread | Forwards/s A -> B |
| --- | --- | --- | ---: | ---: | --- |
| 8192 | 2063.071708 /1018.487917 /1017.192000 /2061.514750 | 2062.293229 ->1017.839959 | 50.645% | 0.07550% | 7.758 ->15.720 |
| 32768 | 4045.251125 /1086.632708 /1086.969959 /4045.167708 | 4045.209417 ->1086.801334 | 73.134% | 0.00206% | 3.955 ->14.722 |

Frozen32K primary >=35% mean/both-pair wall savings passes (73.138/73.129% pairs).
8K <=3% regression guard passes (50.632/50.658% saved); both A spreads pass5%.
Separate warm ABBA follows payload oracles; measured outputs are checked afterward,
with no hashes/numerical scans/profile encoders interposed between measured arms.
At32K worst logit cosine0.999999999623/RMS0.000031470/abs0.003263474; new KV cosine
0.999999991801/RMS0.000128053/reported maxdelta0.00390625. One273.99s packet.
This is controlled warm decode from optimized-prefill state, not a cold, prefill
or arbitrary sampled-output claim. Historical singleton-online KILL and earlier
split primitive HOLD remain unchanged; this is independent long-context evidence.

`00cf511b` widens the existing split opt-in to generated positions[1024,32784),
exactly covering the tested32K+16 forwards. No additional buffers or prefill changes.
CPU selector boundaries pass. A249.49s delivery regression uses the actual generated
API, passes original-reference numerics, then matches forced-pilot replay bitwise
in all16 logits and all active KV at8K/32K. First excluded position32784 matches
original fallback bitwise, including all active KV. No timing packet is rerun.

The same complete native32729-token CLI request is repeated because decode selection
changed:17 emitted tokens/16 transitions, identical stdout, token fingerprint and
token_limit finish. Generation1089.868ms is14.681 transitions/s (CLI15.598 emitted
tokens/s), versus4054.619ms before rollout. Prefill233353.236ms/140.255tok/s uses
the unchanged prefill graph. Process235.072657s versus235.527020s before: prefill
variation offsets most of the short16-transition benefit, so this establishes no
controlled end-to-end request-latency win. Clean embedded source00cf511b; memory
unchanged. Source/result review passes.

Raw `long-decode-01*`, `long-decode-score.json`, `long-decode-delivery-01*` and
`cli-long32k-split-01*` are retained beside the prior CLI attempt. The next real
product limit is output horizon: a32729-token prompt has55 eligible transitions
before the32784 fallback. Longer continuation/range qualification remains separate.
