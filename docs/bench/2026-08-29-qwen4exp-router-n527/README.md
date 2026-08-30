# Flash-Next Strict E8P32 Router At N=527

Decision: **KEEP** the existing strict-order router at exact N=527. The
qualified token set is now `{512, 527, 2048}`; every other packed width keeps
the generic F32 router.

## Scope And Isolation

- Baseline was the release binary from `896e82c3`. Candidate source differed
  only by admitting N=527 to the existing strict-router token set.
- Both arms explicitly enabled selected packed QSA, the two selected GQA4
  kernels, and the global strict router. N=2,048 therefore used strict routing
  in both arms; only the final N=527 selected command changed router kernels.
- Four fresh release-CLI processes ran in B-C-C-B order on Apple M4 Max. The
  frozen 2,578-token known-answer prompt planned `2048+3+527`, generated up to
  64 tokens greedily, and used aggregate prefill GPU time as the decision
  metric. Process wall time was descriptive only.
- The KEEP gate required exact component equality, the expected generated
  answer and EOS in every arm, positive savings in both balanced pairs, and at
  least 1% mean aggregate prefill-GPU saving.

## Correctness

The focused release differential
`packed_router_e8p32_strict_matches_generic_route_bits` passed at N=527. It
covered complete router logits, top-k IDs and weights, shared scale, route
counts and slots, and generic/candidate dispatch topology. The strict grid has
17 output tiles, including the final 15-row tail.

Every model arm emitted exactly
`FINAL_JSON: {"code":"amber-lattice-2049","record":"K-17"}`, generated 23
tokens over 22 transitions, and stopped on EOS.

## B-C-C-B Results

| Arm | N=527 router | Prefill GPU (ms) | Prefill wall (ms) |
|:--|:--|--:|--:|
| B1 | generic | 5,187.153625 | 31,322.5 |
| C1 | strict | 5,048.630875 | 18,585.2 |
| C2 | strict | 5,081.968750 | 17,477.4 |
| B2 | generic | 5,237.269209 | 15,545.5 |

Mean aggregate prefill GPU moves `5,212.211417 -> 5,065.299813 ms`, saving
`146.911605 ms` or 2.818604%. The two balanced savings are independently
positive at `138.522750` and `155.300459 ms`, so the candidate clears the 1%
command gate without relying on process-wall first-touch behavior.

## Promotion

- Admit exact N=527 alongside N=512 and N=2,048 only on the already-qualified
  Apple M4 Max, F32-router, `H=2560`, `E=512` geometry.
- Keep generic routing for every other width and unsupported geometry.
- Roll back the complete strict-router set with
  `QWEN4EXP_PACKED_ROUTER_E8P32_STRICT=0`.
