# D1 call-site fix: MoE MTP prefill organization aligned with production decode

Date: 2026-07-20. Production change driven by closed packets W0b
(decomposition + diagnosis) and W0c-econ (pricing: the global-flag
alternative costs 8.6% decode; call-site alignment costs nothing).

## Change

`single_token_argmax_with_hidden` (the MoE MTP/verifier prefill and
bridge single-token path) previously hard-used the plain per-block
organization (`encode_moe_block_gpu`) while production decode
(`single_token_moe`) uses the concurrent GDN split + flag-selected
concurrent-shared FFN apply. The ~1e-5/token FFN shared-expert
accumulation difference seeded recurrent-state divergence in every MoE
MTP session. The path now mirrors production organization UNDER THE
SAME FLAGS (concurrent_gdn_moe_decode_enabled /
concurrent_shared_moe_decode_enabled), so the two paths stay matched in
every flag configuration. Dense untouched (branch is Moe-only).

## Evidence (this dir; all single deterministic runs on fixtures whose
run-to-run byte-identity was established in W0/W0b)

- postfix-default-code16: A3B, PURE DEFAULT env, oracle probe with
  --mtp-state-trace: prefill row gdn/conv max = 0.000e0 (was 4.1e-5 /
  1.8e-4 over 27-29 layers pre-fix — W0b trace-cser-code16). D1 gone in
  production config. Packets still diverge (D2/D3, unchanged scope);
  audit still fails overall as expected until parity kernels exist.
- postfix-oracle-code16: the bit-exact oracle config
  (CONCURRENT_SHARED=0 + BATCHED_MIXER=0) still PASSES at exact 1.0 —
  the fix composes with the flag (falls to plain apply when the flag
  disables concurrent-shared, matching decode under the same flag).
- postfix-dense-code16: dense-27B audit byte-identical to its pre-fix
  control (kv_cos 0.9999999199, gdn 2.510e-3, conv 2.111e-2) — dense
  path provably untouched.

## Notes

- Dense's residual 3.1e-2 kv max-abs is D2/D3-class (batched packet
  arithmetic), not D1; it remains sub-threshold and is in scope for any
  future shared parity-kernel work, not this fix.
- The W0b/W0 baseline traces that documented nonzero prefill deltas
  describe PRE-FIX behavior; this packet supersedes them for the
  default configuration.
