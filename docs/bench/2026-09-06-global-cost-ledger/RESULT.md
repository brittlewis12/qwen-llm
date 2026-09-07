# End-to-end cost re-ranking

CPU-only reconciliation of retained measurements, not a new benchmark or a
cross-model performance comparison. Raw script and complete 22-row ledger:
`target/profiles/global-cost-ledger/{ledger.py,ledger.json}`.

## Serving observations

Q4 sources: consumed-tail HTTP A1/A2, candidate disabled. Q8 sources: landed
single-chunk-VT policy B1/B2. Different packets and artifacts; do not pool them.
Times below are medians of two observations per lane, in milliseconds.

| Lane / request | First SSE | Full wall | Prefill | Prefill / wall | After first SSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Q4 fresh 8810 / 2 | 38654.435 | 38746.081 | 38557.500 | 99.51% | 91.646 |
| Q8 fresh 8810 / 2 | 37681.236 | 37792.664 | 37582.150 | 99.44% | 111.428 |
| Q4 fresh 19 / 3 | 735.834 | 833.338 | 709.550 | 85.15% | 97.504 |
| Q8 fresh 19 / 3 | 1076.188 | 1204.734 | 1059.150 | 87.92% | 128.546 |
| Q4 compact code / 128 | 1143.903 | 6753.632 | 1060.800 | 15.71% | 5609.729 |
| Q4 compact prose / 128 | 1043.967 | 6622.882 | 958.150 | 14.47% | 5578.914 |
| Q8 compact code / 128 | 326.112 | 8150.475 | 247.800 | 3.04% | 7824.364 |
| Q8 compact prose / 128 | 349.062 | 8187.976 | 266.050 | 3.25% | 7838.914 |

`restore_ms` starts before lookup/planning/allocation and encloses `alloc_ms`:
do not add them. Tokenize + restore + prefill + prompt capture leaves only about
1.6-2.1 ms to first SSE on these rows. This residual is not a separately measured
HTTP component. After-first-SSE includes generation, completed capture, cleanup
and protocol completion; it is not pure decode time.

The implications are causal ceilings, not projected speedups:

- After the Q8 win, deleting all compact-request prefill could save only about
  3% of those full requests. More warm-tail polishing has sharply reduced value.
- Fresh short serving still spends most of its wall in serial prefill. Source
  confirms the <=48 serving policy repeats token-major target forwards. Reusing
  existing packed execution is a different work unit, not a new attention kernel.
- Fresh long requests remain overwhelmingly prefill-bound. Removing setup or
  serialization cannot materially fix their measured wall. Large gains require
  substantial prefill work reduction or a demonstrably better work unit.
- Generation-heavy and speculative opportunities need their own charged target,
  draft, verification and fallback budgets. Warm output-heavy rows motivate that
  audit but do not prove a viable new mechanism or reopen prior failed controllers.

## Historical process-cold orientation, not current authority

Retained Q4 CLI 8840/64 controls at `f37b53a9`, from the prefill-lifetime packet:

| Requested lane | Load | Prefill | Generation | Request wall | Process wall |
| --- | ---: | ---: | ---: | ---: | ---: |
| No spec | 3154.957 | 38763.920 | 2693.797 | 41481.897 | 44905.147 |
| Spec | 2251.200 | 37944.390 | 2731.819 | 40724.930 | 43203.078 |

These are historical two-control medians with different conditioning and noisy
load costs, not a spec/no-spec speed comparison or storage-cold result. Prefill
accounts for roughly 86-88% of these process walls. Current process-cold short
and generation-heavy cells remain unpriced by this ledger.

Crucial scope correction: ordinary CLI `qwen/single_turn.rs` already calls packed
`prefill_span` for fresh prompts. The newly identified <=48 serial-policy issue
belongs to serving; it must not be sold as a disposable-process CLI optimization.

## Review and disposition

Independent review supports a cheap fresh-serving packed falsifier ahead of more
warm-cache machinery. It does not establish the largest global opportunity without
deployment frequencies. Preserve the project's fresh/cold BS=1 priority and all
closed-kernel, model-choice and residency gates. The accounting identifies where
work matters; only subsequent correctness and whole-request evidence can qualify
an implementation.
