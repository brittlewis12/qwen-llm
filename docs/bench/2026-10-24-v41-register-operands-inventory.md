# V4.1 Register-Operand Source And Evidence Inventory

## Closed Scope

The leaf is closed. The decisive matched-F16ACT result is 10.675500 ms for the
32x32 direct candidate versus 5.235042 ms for the incumbent. There is no product
kernel promotion, further register sweep or full-model qualification.

The reviewed landing surface is the maintained task diff from `6c4e0cd7` to the
current branch tip. The final measured implementation remains
`eb89e4d87391e0aec7a9f38169b79e2ec3f74562`; documentation cleanup does not rebuild
or relabel that measurement. The three result packets retain their exact source
and binary hashes, numerical contracts and technical ledgers:

- [Original F32-input cohorts](2026-10-24-v41-register-operands.md).
- [Corrected admission and one-SIMD continuation](2026-10-24-v41-register-operands-single-simd.md).
- [Final supported shape with matched F16ACT](2026-10-24-v41-register-operands-32x32-f16.md).

## Private Archive Convention

Archive label: `register-operands-private`. Every file name in this inventory is
relative to that private archive or its local mirror. Machine-specific locations
are deliberately not part of the tracked handoff.

Frozen source archives and raw build/observer records are immutable private
provenance. Historical copies may include original filesystem or execution
metadata, including documentation before this cleanup. Do not publish those
archives, raw logs or historical commits wholesale as sanitized documentation.
Review the cleaned maintained diff for landing. No measured commit, frozen
source archive, binary or raw observation is rewritten to alter its hash.

Each successful `frozen*/identity.json` binds the build source, compiler/SDK, AIR,
libraries and executable. The source archives below also preserve the failed
build attempts. All 11 successful identity manifests and all 13 source archives
are checked locally during closure; no remote access or new GPU run is needed.

## Source Archive Inventory

Each directory contains `source.tar`. Hashes are SHA256.

| Directory | Source Commit | source.tar SHA256 | Disposition |
| --- | --- | --- | --- |
| frozen1 | ce6a8e5f8edc141f03788af597c95d30e32a8950 | 352b305694ce40b426f271a97b47816809709407e2486064e7ef0d1a704cdd44 | Initial declaration compile failure |
| frozen2 | 03cac93f6f9acb8cfc4ced602c0fcea58c5fd6f1 | ddfe1b206ca6c81d4b7c87827764aea8b7780f0d3f0a25dc1b167ec319ca7f33 | First qualified IQ3 control |
| frozen3 | da60ce817c300c86613e247d387a7cd289014088 | fe65d361cbb27b60b7c5cf4d59851f958006f348305886dc063bd3a3f151c20e | K16 and resident controls |
| frozen4 | 49c516a31710ab70b290d08a943f2b05ebaf2983 | 6395f218968f4757851815c91f5aa148c97e9f4ed20050fbc12fd98cbd05b425 | Producer/broadcast controls |
| frozen5 | b4e254ed9bb67dda79196777f37a5f25aa955740 | e872259af24ad3ce5be25af6e080b405e9b14cc54a864e8c7f7f7cf0978e01a4 | Compiler-tier build; not a measured cohort |
| frozen6 | bd16b3e3099b754c775bae59128c03343005f6a4 | fcf4cf34f81005edb8225ed235e019ddabada0cdb7264637815baf885c15c04d | Final historical F32 cohorts |
| frozen7-domain-v2 | ad44071eb129dd010c3fb06c479f8d0a96d99976 | 0781406f76a5a91a9d0acee2135c5701420928ed5c076d8f7737ca15a6451948 | Corrected admission/lifetime parity |
| frozen8-single-simd | 7ffd6f04c537822b6ce71074e534fdc874611d95 | 0019ffb977fd3bc788e75cc22cde89aaf3bf40a040934f7ab4783cad6b0276ea | Unsupported both-cooperative M/N=64 |
| frozen9-single-simd | d0b998a349c26fecb5b8578cc186a6a1c6ef3163 | b34aa893a096d6329ae87edd32c28fc007f462f3037a76a880cba564bd346384 | Audit failure; not admitted to timing |
| frozen10-single-simd | 5fd2b3e3232114239f7abccaaca0451ca98e82fd | 2f5ef2f0debd92b0e7fce676facee5eabadd3c5f285af507e5ce9752a32454ec | Localized audit diagnostics; no timing |
| frozen11-single-simd | c64935da519a1ec84a87d8efe0839e0b7c769eac | 341ee770e914df8b7c53b254da11c1bb978857d17dc59ef03ac1027430c43cec | Qualified actual-input audits and timing |
| frozen12-butterfly | 9575bd2c649059f1842da7039e88a0d1b40f7ca6 | 6fb9e1a8634895c49ea6f9d3dffe5f4ee4e974cf7477f6ae5685d65ddd9c1f87 | Two-stage transpose, F32 inputs |
| frozen13-tile32-f16 | eb89e4d87391e0aec7a9f38169b79e2ec3f74562 | a3d312c9940aeb3ebef1315445b3bcc24a17d1db03f08404618c8bad01b137d6 | Final 32x32, matched F16ACT |

## Binary Inventory

The following successful build directories contain `probe` and
`private.metallib`. Hashes are SHA256. Failed builds 1 and 8 have no successful
probe/library identity; their source and compiler evidence remain separate.

| Directory | probe SHA256 | private.metallib SHA256 |
| --- | --- | --- |
| frozen2 | fee46998d18c21089bd8ae917078641cfb6a0b16ec01cd8aa2d459d09953bc54 | 3aedd6f38f55f4626cdff71414d7e4e46abe372b82189a19d559cbbc700febef |
| frozen3 | fac9c7013905244774a38ec8ec9d2ba20dca4729e54007e170380e9784248305 | 5cddb6a54643d7d793086015ed92c4c3b113891c27c5492db0e7fd2b56a3366c |
| frozen4 | 1e65edc0366e830c3334abb589d59ab7fcf49826907f673354ba9216c233af01 | be4985a8ad6e5d1ef73cb3459886a6f1c9542724ba4b2a86dae1675089e54dbe |
| frozen5 | 2b95a68562d55205aa11ecfe04e621d6c25dc2509bc0657a53dd77e9150f4965 | be4985a8ad6e5d1ef73cb3459886a6f1c9542724ba4b2a86dae1675089e54dbe |
| frozen6 | dada89307a13451949ff3cc5ab3a691f3dd61ad4878cbbae0b19b594177cff0d | be4985a8ad6e5d1ef73cb3459886a6f1c9542724ba4b2a86dae1675089e54dbe |
| frozen7-domain-v2 | baf77ed108c84bbc7e612202939792189fc7250dab5b273776fde266d23f8976 | be4985a8ad6e5d1ef73cb3459886a6f1c9542724ba4b2a86dae1675089e54dbe |
| frozen9-single-simd | 9713f3b76fb864173b16232f9dc118344f5c5c97b8afa6875c5e37ef41f1f70c | 7a24cd821070c563ce5e9cf67763b91fa5affcbb4f5ac11a541be74014069588 |
| frozen10-single-simd | 6c05da90238ee3f8176910accac4a8a59178af8d612cf92d27bfb19ddea17885 | 7a24cd821070c563ce5e9cf67763b91fa5affcbb4f5ac11a541be74014069588 |
| frozen11-single-simd | 6c05da90238ee3f8176910accac4a8a59178af8d612cf92d27bfb19ddea17885 | 4d8d7f526ba3c2cdf09a85acb2a31c71f958e2e7eb9390d49a73e9f1cf67b9f9 |
| frozen12-butterfly | abcc0c2128c7913dbca8332873c520925c397ac8f4609894d873196208abe152 | d26b0e19b5284aecf4a381723acfac2825afc8c1c0de64c7cf6c1f9328e9ec88 |
| frozen13-tile32-f16 | ec969f51d667210480e90ab8fc25ca94bcafead47bad6f63eb0bf3c33e9023e2 | 1bd26d4141abe4f30d8a8cf6b133980acd2440e5b1e2bc39c3c1e28960346fa0 |

All retained `baseline26.metallib` copies use the independent production-target
incumbent, SHA256
`e7a37d6471b2cfd978decffeb8443c9ca218c3b54349e46089ded1db74948e83`.
Each identity manifest additionally inventories AIR and, where present, the
compiled CPU contract test. Those files are not replaced during closure.

## Raw Evidence Inventory

Each run label below has `.stdout.jsonl`, `.stderr.log` and `.events.jsonl`
companions. Warmups, failures, diagnostics and all timed samples remain; the
private records are not copied verbatim into tracked documentation.

| Relative Labels | Scope |
| --- | --- |
| build1 through build6 | Original build chronology, including the first failure |
| parity2-* / timing2-* | Initial IQ3 experiment |
| parity3-* / timing3-* | Explicit K16 and resident arithmetic controls |
| parity4-* / timing4-* | Producer/broadcast alternative |
| parity6-* / timing6-* | Final historical F32 cohort: 203 output checks per dtype and 1,386 timed samples |
| domain-v2-build7 / domain-v2-parity7-iq3 | Corrected admission, bijective routing and drained fixture scopes |
| single-simd-build8 | Exact SDK M/N=64 rejection; no GPU timing |
| single-simd-build9 / single-simd-parity9-iq3 | Supported fallback with failed temporary-value audit |
| single-simd-build10 / single-simd-parity10-iq3 | Localized bit-level audit diagnostics; no GPU timing |
| single-simd-build11 / single-simd-parity11-iq3 / single-simd-timing11-iq3-c32 | Qualified actual-input audits: 62 output checks, 2,359,296 half checks, 63 timed samples |
| butterfly-build12 / butterfly-parity12-iq3 / butterfly-timing12-iq3-c32 | Two-stage transpose: 86 output checks, 3,538,944 half checks, 81 timed samples |
| tile32-f16-build13 / tile32-f16-parity13-iq3-c32 / tile32-f16-timing13-iq3-c32 | Matched F16ACT: 26 output checks, 655,360 half checks, six trait records, 36 timed samples |

Derived summaries and verification records:

- `final-report.json`, `handoff-check.json`, `verified-frozen-identities.json`.
- `continuation-v2-report.json`, `continuation-v2-handoff-check.json`,
  `continuation-v2-verified-identities.json`, `continuation-v2-immutable-originals.json`.
- `tile32-f16-report13.json`, `tile32-f16-handoff13.json`,
  `tile32-f16-verified13.json`, `tile32-f16-immutable-history13.json`.
- `MPPTensorOpsMatMul2d.h`, `MPPTensorOpsMatMul2dImpl.h`,
  `metal_cooperative_tensor`, `verified-mlx-nax.h`: retained API/source evidence.
- `frozen4-reflection.txt`: retained absence of usable reflection resource
  statistics; not an occupancy or spill measurement.

The final evidence still reports the same 5.235042/10.675500/11.400333 ms
incumbent/direct/retained medians. No result or admission ledger changes as part
of this documentation cleanup.

## Maintained-Source Handoff

Closure runs local CPU contract/property tests, Objective-C++ syntax checking,
the 30 unchanged observer tests, reporting test, Ruff, diff checks and a hygiene
audit of every Markdown file added by this task. No shader rebuild, GPU sweep,
remote operation or full-model run is part of closure.

After the cleanup commit, a fresh maintained-source archive, reviewed task patch,
private Git bundle and checksum inventory are created locally under the generic
handoff label `handoff-doc-hygiene`. They bind the cleaned branch tip separately
from the unchanged measured source archives. Their final hashes live in the
private handoff manifest to avoid a self-referential source-archive hash.
