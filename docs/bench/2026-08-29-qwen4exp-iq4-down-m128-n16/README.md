# Flash-Next Packed IQ4_NL M128xN16

Decision: **KEEP** and enable the retile by default only for the released
Flash-Next geometry at exact N=512 on Apple M4 Max. Roll back with
`QWEN4EXP_MOE_IQ4_DOWN_M128_N16=0`.

## Candidate

The natural N=512 route screen selected 43 IQ4_NL-down layers and measured
`S16=21,294`, `S32=15,377`, and
`R=S16/(2*S32)=0.692397737`, below the preregistered `0.75` code-entry gate.
The candidate keeps the same 14,090,240 launched threadgroups but reduces active
equal-FMA tiles by 30.76%. It trades 1.3848x weight/dequant traversal for 0.3462x
activation staging; the route ratio was an entry gate, not a speed forecast.

The new kernel changes only the packed IQ4_NL routed-down tile from
M64xN32xK32 to M128xN16xK32. Four SIMDgroups each own 32 output rows and all 16
slots. The 9,216-byte threadgroup allocation holds an 8,192-byte A tile and a
1,024-byte B tile, then reuses its first 8,192 bytes for F32 output. Expert
output remains materialized and the ordered weighted reduction remains separate.

## Correctness

- Candidate and incumbent outputs match as raw F32 bits over route counts
  `0/1/15/16/17/31/32/33/511/512`, output widths
  `64/65/127/128/129/2560`, and K widths `32/64/640`.
- Concentrated and dispersed buckets cover experts 0 and 511. Input weights,
  activations, counts, and slots remain byte-identical; every output slot is
  written and both output guards remain intact.
- Census gates require the exact M64xN32 and M128xN16 grids, 128 threads, and the
  candidate's 9,216-byte preflight. Scope and rollback tests reject every
  unsupported device, dtype, token width, or model geometry.
- The incumbent grouped IQ4_NL differential and packed common-MoE fallback tests
  pass. The focused default/rollback gates were rerun after promotion.

## Protocol

- Measured base: `799e773ba92f8de6ee61b0a91821611eebaf060e` plus candidate
  diff SHA-256
  `a46f8856f2bc5eec7f34be68f329fceaa39f699cf3d9e4dbc285386de53f68bd`.
- Measured executable SHA-256:
  `34af8cdfce128afcdd1659e5004cd6b0afd276a895bc4a61074e58ca3749de8d`;
  embedded metallib SHA-256:
  `d65856754594b3a0b9ee708e3aa0e4a2f6101f5c85ec9c30aa7617cf58a7dace`.
- Promotion: `a2bfeec4804c37176e12f3bea77e8b123f5aa054`; the measured
  opt-in became the exact-scope default with the same flag retained as rollback.
- Device: Apple M4 Max. Model: internal-SSD UD-Q3_K_XL from
  `unsloth/Qwen3.8-Flash-Next-GGUF` revision
  `8bdc666649440e9bdc97e16f3f75782c98478ff5`.
- Workload: tracked natural 512-token prompt, no special tokens, one generated
  token, exact 512-forward capacity, one packed command, and zero scalar tail.
- Order: separate-process A1-B1-B2-A2 with five seconds between processes. A set
  the candidate flag to `0`; B set it to `1`. Every process ran first, warm, and
  128-sample profiled commands.

## Results

| Arm | Down leaf (ms/layer) | Warm GPU (ms) | Profiled GPU (ms) |
|:---|---:|---:|---:|
| A1 baseline | 3.510209 | 992.264292 | 993.620875 |
| B1 candidate | 2.804000 | 967.123250 | 968.566375 |
| B2 candidate | 2.796375 | 967.146000 | 968.336917 |
| A2 baseline | 3.515500 | 992.046625 | 993.689500 |

Worst-case candidate leaf time is `2.804000 ms`, below both the fixed
`3.164099 ms` ceiling and 90% of the best baseline. The mean leaf moves
`3.512855 -> 2.800188 ms`, saving 20.29%. Mean warm/profiled command GPU moves
`992.155458 -> 967.134625 ms` / `993.655188 -> 968.451646 ms`, saving
2.52%/2.54%.

Crediting 43 IQ4_NL layers predicts `30.644681 ms`; the warm command realizes
`25.020833 ms`, or 81.65% conversion. GPU-equivalent throughput rises
`516.05 -> 529.40 tok/s`. Maximum within-arm drift is 0.2723%. Every observer
passes, raw timestamp coverage is 1.0, and all four generated stdout files share
SHA-256 `e530152ea80c3012dbfdb19a69e554de54aed4511184ae225fcec380233a46e9`.
Each process also retained the CLI's bitwise first/warm/profile endpoint-logit
check; this packet does not claim an unrecorded cross-arm full-logit digest.

The B-only activation marker proves candidate selection. The 43-layer count
comes from the recorded released dtype cohort plus exact source scope, not a
runtime 43-dispatch census.

## Shape Closure

The existing natural N=2,048 route census closes a tempting scope extension
without code: the same 43 layers have `S16=63,924`, `S32=37,264`, and
`R=0.857717905`; every layer fails individually. The repeated-token control is
worse at `R=0.961875`. Exact N=2,048 therefore remains on M64xN32.

Raw logs and hashes remain under
`target/profiles/qwen4exp-iq4-down-m128-n16-screen-20260829/`.

Adversarial design, source, disposition, and leverage review:
`01a04de7-d66e-73f1-b3de-14ba60527c89`.
