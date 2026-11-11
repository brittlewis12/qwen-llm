# GLM 48K Append Non-Inferiority: Packed Half Is Not Worse Than Sequential

Status: two bounded M5 packets, probe and report only. **No production
behavior changed.** GLM-5.3-Flash UD-IQ3_XXS, M5 Ultra, `iter` profile,
49,152-token prefix, teacher-forced appends of 8, 16, 32 and 64 real next
tokens.

## Cohort

The 48K prompt source set is uncylced and comes from the repository/witness
corpus:

- `code.txt`: frozen three-agent code corpus with five Rust source markers;
- `metal-code-corpus.txt`: `kernels/moe.metal` and `kernels/mat_vec.metal`;
- `scripts/profile/dflash_k0s.py`: Python source;
- `docs/PERF-LOG.md`: prose control.

The selected prompt token lengths are 150,273 / 188,853 / 113,046 / 798,175;
each has more than 49,217 tokens, so all prefixes and continuations use real
source tokens without cycling. `quality-v1.json` provides the established
prose/code/long strata and the Metal file witnesses, but its frozen items are
only 2,112 tokens and cannot supply a natural 48K prefix; full source text is
used here. Source and token SHA256 values are retained in each raw prompt row.

Arms per prompt:

- sequential 1-row Fast/Half decode, the qualified reference path;
- packed Fast/Half append, production arithmetic;
- packed `DenseF32` suffix (F32 dense projections, routed experts stay Half);
- packed all-F32 suffix.

The probe records per-prompt signed dNLL, both KL directions, top-1 agreement,
and route-set changes for every width. Positive dNLL means the candidate gives
the actual next token lower log probability. The cohort analysis uses equal
prompt weighting and a prompt-cluster bootstrap (20,000 resamples, central
90% CI and one-sided 95% upper bound).

## Sequential-relative result

The Exact 48K prefix attempt for `code.txt` did not reach a completed-prefix
event after **7m51s** in a 14-minute lease request. It was stopped before the
request expired to preserve the second short packet; no Exact logits or
absolute dNLL/KL versus Exact were produced.
For the requested *relative* decision, Exact cancels:

`dNLL(packed, Exact) − dNLL(sequential, Exact) = NLL(packed) − NLL(sequential)`.

The values below are that paired candidate-minus-sequential quantity on the
same tokens and same Fast/Half prefix. Absolute Exact-referenced quality remains
unmeasured for this 48K cohort.

| Width | Mean signed dNLL | 90% CI | 95% upper | Mean KL(seq ‖ packed) | Top-1 agreement |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | −0.02094 | [−0.04195, +0.00008] | +0.00008 | 1.40e-3 | 100% |
| 16 | −0.00563 | [−0.01145, +0.00019] | +0.00019 | 2.68e-3 | 97.9% |
| 32 | −0.00586 | [−0.01214, +0.00042] | +0.00042 | 2.60e-3 | 99.0% |
| 64 | −0.00499 | [−0.01165, +0.00167] | +0.00167 | 2.04e-3 | 99.0% |

All four code widths have a paired interval that includes zero. The 95% upper
bound on worsening is at most +0.00167 nats/token. This does not show a
systematic packed-Half penalty relative to sequential Fast/Half.

Per-prompt signed dNLL (packed arm minus sequential Fast/Half):

| Prompt / arm | 8 | 16 | 32 | 64 |
| --- | ---: | ---: | ---: | ---: |
| Rust corpus — Half | −0.06293 | −0.01719 | −0.01821 | −0.01749 |
| Rust corpus — DenseF32 | −0.01061 | +0.02604 | +0.01677 | +0.00128 |
| Rust corpus — all-F32 | +0.000005 | +0.000030 | +0.000111 | +0.000699 |
| Metal corpus — Half | −0.0000002 | +0.000016 | +0.000018 | +0.000031 |
| Metal corpus — DenseF32 | +0.000004 | +0.000001 | −0.000045 | −0.000048 |
| Metal corpus — all-F32 | −0.000001 | −0.000004 | −0.000002 | −0.000028 |
| Python — Half | +0.000117 | +0.000279 | +0.000617 | +0.002487 |
| Python — DenseF32 | −0.000088 | −0.000408 | −0.000061 | −0.000868 |
| Python — all-F32 | +0.000001 | +0.000008 | +0.000009 | −0.000054 |
| Prose — Half | +0.000460 | −0.010545 | −0.016696 | −0.011566 |
| Prose — DenseF32 | +0.000838 | −0.009491 | −0.023343 | −0.019849 |
| Prose — all-F32 | +0.000004 | +0.000026 | +0.001075 | +0.000557 |

The Python code prompt is the most positive packed-Half case at 64 rows
(+0.00249), while the Rust corpus favors packed Half. This prompt spread is
why the equal-prompt interval matters; the Rust outlier does not make the
cohort-wide result a general answer-quality qualification. The prose control
is one prompt, so it has per-prompt metrics but no document-level interval.

## F32 alternatives and decision

`DenseF32` is not a consistent narrow remedy: its code-cohort mean dNLL is
+0.00854 at 16 rows and +0.00556 at 32 rows, with intervals crossing zero and
upper bounds +0.0174 / +0.0112. Full F32 is closer to the sequential path
(code mean dNLL +1.1e-5 / +3.9e-5 at 16 / 32 rows; top-1 100%), at an observed
width-64 wall cost of about +13–16% across these prompts. The route-set
changes do not start at a single common block across prompts and widths; the
DenseF32 arm also leaves route-set changes.

| Width | DenseF32 mean dNLL (90% CI) | All-F32 mean dNLL (90% CI) |
| ---: | ---: | ---: |
| 8 | −0.00357 [−0.00711, −0.00003] | +0.0000017 [−0.0000004, +0.0000037] |
| 16 | +0.00854 [−0.00027, +0.01736] | +0.0000113 [+0.0000001, +0.0000224] |
| 32 | +0.00556 [−0.00006, +0.01117] | +0.0000393 [+0.0000018, +0.0000768] |
| 64 | +0.00012 [−0.00059, +0.00084] | +0.0002055 [−0.0000457, +0.0004568] |

**Decision: change nothing.** The packed-Half append trajectory differs from
sequential decode, but it is not systematically worse in signed teacher-forced
dNLL within the paired prompt CI. This follows the user's relative rule. A
router-boundary-only precision fix is not justified by this cohort, and there
is no absolute Exact-referenced quality claim because the Exact 48K prefix did
not complete within the bounded packet.

Raw files, incomplete Exact attempt, assembled input and analysis summary:
`metagross:~/qwen-runs/2026-11-11-glm-append-noninferiority/`. Cohort and
assembly manifest: `manifest.json`; summary generator:
`scripts/reference/glm53/append_noninferiority_analysis.py`.
