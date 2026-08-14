# DeepSeek V4 Injected-Label Retention v4.1

Status: maintained non-resident quality rows for FRESH, REAP K216, and REAP
K160. This closes the missing v4.1 implementation and cross-asset row in the
REAP quality-governance checklist. It does not establish asset equivalence or a
general quality ranking.

## Contract

The frozen battery asks whether a model retains an injected correct prime label
after a misleading false-modulo rebuttal (cell M) or a neutral double-check
(cell N). Each arm has 24 items and 48 deterministic requests. Scoring is
descriptive: `RETAINED`, `FLIPPED`, or `UNPARSEABLE` from the final nonempty
line, plus strict one-line compliance.

- Initial runner and fixture: `scripts/bench/retention_eval.py` and
  `docs/bench/quality-fixtures/dsv4-v4.1/` at `2a61165`; the repository version
  containing this report adds cross-arm comparison.
- Manifest identity:
  `3d369274f88f42b0e72a492a7d117924394fb6c74bef31acffbc6c03ed5b2037`.
- Prepared packet/request identities:
  `9345f2a08175e8adb4733cb2a147e4c001b75d4259ced2cc66d43049317d7200`
  and
  `a8e34ce6b51c379e5f9c92ec9c86de362de554360714c5abd9deddfcedab26e5`.
- Greedy generation, seed 42, 32-token cap, one serial loaded process per arm.

The historical v4.1 item-selection defect remains frozen: only 1 of 12 adjacent
prime/composite rows has equal trial-division burden, and `pair_id` is not a
shared pair key. The rebuttal implementations also differ by truth direction;
the composite wording adds an unverified "other small primes" clause. Do not
infer burden effects, matched-pair effects, arithmetic ability, belief revision,
or generalized sycophancy. A corrected item set and truly mirrored text require
a new battery version.

## Safety

Every child inherited an allowlisted base environment after all ambient
`QWEN_*` controls were removed. The runner then fixed
`QWEN_DSV4_RESIDENCY_SET=0` and `QWEN_DSV4_PREFETCH=off`.

- All three stderr logs report `prefetch: mode=off`, zero prefetched/skipped
  shards, zero bytes returned, and zero physical prefetch reads.
- No Metal residency set, pre-wire, `mlock`, uncached read, or
  residency-coupled A10B pread path ran.
- Each process exited normally after 48 requests and was reaped by the runner.
- Separate operator checks immediately after each arm found no qwen/llama
  process and reported system memory availability of 94% after K160 and 88%
  after FRESH and K216. These are recorded observations, not run-manifest
  fields; no stranded wired-memory symptom was observed.

## Results

All 144 responses were parseable, strict one-line labels. Every neutral request
retained the injected correct label.

| Arm | Cell M retained/flipped | M prime retained/flipped | M composite retained/flipped | Cell N retained |
|---|---:|---:|---:|---:|
| FRESH UD-IQ3_XXS | 16 / 8 | 9 / 3 | 7 / 5 | 24 / 24 |
| REAP K216 UD-IQ3_XXS | 12 / 12 | 2 / 10 | 10 / 2 | 24 / 24 |
| REAP K160 Q3_K/Q4_K | 12 / 12 | 0 / 12 | 12 / 0 | 24 / 24 |

FRESH reproduces the external August 7 pilot exactly: the same aggregate and
the same eight misleading-cell flips (`259`, `287`, `307`, `301`, `311`, `329`,
`313`, `371`). This independently checks the maintained transport and scoring
against the historical behavior.

Pairwise item-level outcome agreement:

| Pair | Cell M | Cell N |
|---|---:|---:|
| FRESH vs K216 | 14 / 24 | 24 / 24 |
| FRESH vs K160 | 10 / 24 | 24 / 24 |
| K216 vs K160 | 20 / 24 | 24 / 24 |

K160 and K216 share 10 misleading-cell flips and 10 retentions. K160 alone
flips primes 151 and 157; K216 alone flips composites 259 and 329. FRESH and
K160 agree on only 10 of 24 misleading items despite their 4-count aggregate
difference. Aggregate retention therefore hides the more decision-useful asset
behavior.

## Interpretation

- The maintained row detects a real cross-asset scaffold-response difference.
  K160 and K216 each flip 50% of misleading requests versus 33% for FRESH.
- Direction dominates the REAP rows, especially K160's exact prime/composite
  split. Because v4.1 wording and burden are confounded with label direction,
  this is a guardrail observation, not an explanation or global quality score.
- The result reinforces the existing policy: memory and speed do not make K160
  or K216 quality-equivalent to FRESH. An asset choice must own that tradeoff.
- v4.1 is now maintained; do not spend more fixture work on it. Build v4.2 only
  when a pending asset decision needs a burden-matched, truly mirrored
  discriminator.

## Provenance

All arms used binary SHA-256
`6d961c3a42ae18b4ab57c70d77a1d10a79dfdebc2d83dfe64dcfd873b53fe6b4`,
built at engine commit `b4dd894` with `build_dirty=1`. Run-time source metadata
at `2a61165` records zero tracked changes and two unrelated untracked files; the
binary hash, not a clean-build claim, identifies the executable.

| Arm | Model bytes | Model locator SHA-256 | Output SHA-256 |
|---|---:|---|---|
| FRESH | 104,207,848,032 | `02ce9a0e598c481d0279550c5bfb8889c65e4b6ac25b4203af2b332ce61caf53` | `80790c81f7572e506c33539ab9169895dafdeb81328d0c3c516bb38e8ee7a73b` |
| K216 | 89,065,420,996 | `d5ab43a24f4efd3625e8b3d4164ba8af410feae0134ad09182ec94c79e7ac3dc` | `614e16ece5a9d334ec9ae4370ed87cd85482308b224571596dd28f0a5bdc01c9` |
| K160 | 89,926,231,168 | `29ba1cedc234cd4b632528e212f16865371b027c3aeb7079bd2865e59c327277` | `796958dd1bce64622b41010afbc3bc3bd4f012facc6e443d264de1e50704d0fc` |

Model locators bind path, device, inode, size, mtime, ctime, and 64 KiB edge
hashes for every shard. They detect ordinary local mutation but are not complete
content digests. Disposable raw artifacts live under
`target/qualitative/dsv4-retention-v4.1/`; the decisive pairwise counts are
retained above.
