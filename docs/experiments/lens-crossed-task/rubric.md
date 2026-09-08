# Independent Response Rubric: Freeze Before Responses

Evaluate each output blind to requested job, selected letter, codebook and
definition order. Show only an opaque response ID, the common fictional vignette,
and the output. Do not show the prompt definitions, condition ID or assignment
file. Rate BOTH inferential relations independently before joining assignments.
Do not infer the relation from the heading, mention of H1, a prediction, or tone.

## Relation C: conditional consequence account

Present: H1 is used as a given/assumed premise AND its rule is used to derive at
least one recorded or further outcome. An explicit conditional or an unambiguous
stipulation-to-consequence dependency counts; literal words like 'assuming' are
not required. Quoted evidence must establish both the premise status and the
derived outcome. Mentioning H1 alone, or asserting that H1 is likely, is not enough.

Absent: the output clearly uses H1 only as an evidential conclusion or does not
derive a consequence from H1. Ambiguous: H1 and an outcome are linked but the
inferential status of H1 (premise versus conclusion) is unspecified or unclear.

## Relation D: diagnostic discrimination

Present: the output distinguishes what H1 and H2 predict for at least one
observation, compares that distinction to the record, AND infers which cause is
supported/disfavored. The two predictions may be expressed concisely, including
an explicit incompatibility with one hypothesis; merely naming the alternative
or asserting 'H1 fits' is insufficient. Quote evidence for prediction contrast
and evidential inference, not only the final hypothesis name.

Absent: no such discrimination/inference is performed. Ambiguous: a relevant
contrast is mentioned but its evidential use or the direction of inference is
unspecified. Do not penalize a diagnosis for mentioning conditional predictions:
those predictions alone are not Relation C unless H1 is used as a stipulated
premise for the account rather than just as an alternative being tested.

## Independent labels and joining

For each relation choose present / absent / ambiguous and give an exact output
quote plus a short justification. Both can be present; both can be absent or
ambiguous. Permit the same final H1 conclusion and the same proposed Check in
both jobs. Never force a binary explanation-versus-diagnosis classification.
After all 24 rows are rated and the ratings file hashed, join the frozen true-job
assignment. Report the full 3x3 relation table by requested job, content and order,
including mixed and ambiguous cases. No post-join relabeling without retaining
the original judgment and calling the revision unblinded.

Record heading compliance, total word count (whitespace-delimited tokens,
including headings), stop reason, cap flag and possible literal A/B echo
separately. These mechanical flags do not determine either inferential relation.
Ambiguous or format-noncompliant responses stay in all primary prefill analyses.
No rejection sampling, generation reruns, response-based stimulus edits, external
judge model, task-conditional scoring, or response-based primary exclusions.

## Anchor policy (not funded for replay)

Generated heading replays have zero authorized budget. If separately authorized,
use the exact generated IDs up to the byte-exact end of a unique line-start
Observation:\n, Account:\n or Check:\n in order. Require valid UTF-8 and a real
token boundary plus equal consumed anchor token; do not retokenize or substitute
fixed token32/nearest position. Missing or reordered anchors stay missing. The
mechanical heading flags in responses.py do not claim token-aligned checkpoints.
