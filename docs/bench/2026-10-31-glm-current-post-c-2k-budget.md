# Current GLM post-C 2K coarse budget

Bounded measurement packet:
`glm5_next_metal::current_2k_budget::current_post_c_2048_budget`.
The frozen `ef87a265` canonical artifact passes the four-call M5 Ultra gate.
Independent receipt/timestamp recomputation accepts it as the current coarse
reference. The preparation
follow-up keeps frontend
layout/input/context preparation inside the front transaction and constructs
warm census/summary storage inside the fresh-session admission transaction.
Admission acquires the transaction before sampling memory signals and retains
it through the factory. No new global allocator policy is introduced.
This is a four-fresh-session diagnostic, not another kernel candidate, HTTP test,
quality verdict, exclusive-time decomposition or performance-win claim.

## Current Result

The exact ignored entry passes in 9.60 s. All four fresh sessions reach position
2048 with identical finite full-vocabulary F32-bit endpoints. Warm census verifies
1384 dispatches across 29 names, including the required C/Q5-down/Q6/router/Q8-wide
paths and geometry. There is no A staging or experimental kernel substitution.
Independent name-hash mapping finds 507 Q8 F32-wide calls and the exact required
41/1 gate/up pairs, 39 Q5 tensor downs, three Q6 generic downs and 42 routers.

| Case | Ordinary Call Wall ms |
| --- | ---: |
| Warm plus census, excluded from plain denominator | 3008.876958 |
| Plain 1 | 2001.652833 |
| Plain 2 | 2001.094417 |
| Profiled | 2020.813042 |

Plain 2 / plain 1 is 0.999721023 (-0.0279%). Profile / plain mean is
1.009713037 (+0.9713%, 19.439 ms). Sampler preparation is 0.049542 ms, outside
the call timer. These are one diagnostic process's calls, not a sustained-rate
campaign or a before/after optimization comparison.

The 492 ordered spans and 984 samples pass the exact resolution/pair checks.
Command GPU time is 2011.518875 ms; the scaled span sum is 1960.739201 ms.
Independent recalculation preserves 293 adjacent overlaps: the scaled interval
union is 1960.530625 ms, with 0.208576 ms overlap excess. Envelope minus union
is 50.988250 ms; it is not an independently identified overhead/idle interval.
Largest coarse stage totals:

| Parent Or Stage | Scaled ms |
| --- | ---: |
| Routed experts, combined | 856.841334 |
| MLA, combined despite the `mla_projection` label | 376.432251 |
| KDA recurrence | 189.704501 |
| KDA input projections | 184.103873 |
| KDA preparation | 80.191874 |
| Shared expert | 61.445750 |
| KDA output projection | 60.260792 |

This supports prioritizing routed arithmetic and auditing the combined MLA path,
with KDA recurrence/projection next. It does not identify an exclusive gate/up
fraction, an isolated MLA projection defect or a roofline-utilization number.
The span sum's difference from command time is not an idle-time measurement.

The subsequent [mixed-operand MLA gate](2026-10-31-glm-mla-mixed-operand-gate.md)
validates bounded QK/PV ownership and arithmetic. It does not measure the
CURRENT attention leaf or establish fused/native MLA savings. The separate
[current attention-only budget](2026-10-31-glm-current-attention-budget.md)
measures that unchanged dispatch at about 21.8 ms for the fixed synthetic 2K
shape; it funds one fused source gate, not a whole-MLA or application gain.

The actual cold gate requests 199265371264 Metal bytes and 1611860564 CPU bytes,
including the separate diagnostic policies. Each fresh gate requests 4146692096
Metal bytes with retained weights already live and the same conservative CPU
upper. Admission and construction pass; these are quotes, not peak measurements.

| Frozen Evidence | SHA256 |
| --- | --- |
| Canonical libtest | `3f2ac0a1c2221f10fc35ecad7e4e102cd4ca1d88d6647d90cd618096ad986642` |
| Build/source freeze receipt | `3c935f98f3b7487379894078e1c7b4ac84199382f906f31c420e1dec209b20b7` |
| Complete hardware report | `1d4926aecad4646aea13f78a77b05cd51cbad4b71199c85f2cda2d568363745a` |
| Independent recomputation | `aacabe75b9a6257c5b5bfc5d3cba322c926c36936fdf7a9d9d7e6376757f2bc8` |

Source is `ef87a265d150571443060dc2daf12734157ba503`, ordinary opt3/fat-LTO/
codegen1, default features with no diagnostic features, SDK 26.2, Metal
32023.864/4.0. Build and all 368 exact classified library CPU tests run unleased;
strict discovery matches 598 classifications and 106 ignored entries. Target
execution uses a fail-fast canonical inherited lease and an actual 180 s external
test timeout, with no wired bypass. The lease ends naturally; no server or
installed binary changes.

## Fixed request and path checks

- Actual six-shard stock UD-Q4 names/sizes and header-only primary SHA use the
  unchanged stock pin. Retained file descriptions are revalidated before
  construction and after execution. Weight payload hashes are not verified.
- Layout-only ordinary binding/coverage; no prepared artifact or tokenizer.
  The first frozen document with a 2048-token prefix supplies exactly those
  IDs, checked against the 154880 vocabulary. Its U32-LE SHA256 is
  `e90f6da88c2bb026273b74e56d7b1b07f1bc5266f38c5cfa7b7087c05fc83b2e`.
  Reports contain hashes, never input IDs, text, logits or artifact paths.
  Hash fields serialize fixed 32-byte SHA256 arrays; the expected binary SHA
  environment value uses the usual 64-character lowercase hexadecimal form.
- 45 blocks: 34 KDA, 11 MLA, three leading dense and 42 MoE; hidden 4096,
  expert FFN 2048, 288 experts, top eight. Capacity/rows are 2048 at position
  zero, below the 2052 sparse frontier. The replicated 456 mixed projection
  pins plus nine leading-dense Q8 pins are checked before the weight factory.
  The maintained binder/coverage checks all remaining roles. All serialized
  descriptors, including the unused MLA pair, enter an ordered digest.
- Required M5 unified Apple10/Metal4 device, 16 KiB host page and loaded tensor
  tier; Fast/Half and the ordinary workspace ledger are checked in each case.
  Both Q5/Q6 N64 switches and their resolved thresholds must be ON/1024;
  libtest's default threshold of 64 is not accepted.

There is deliberately **no complete pre-weight PSO manifest**. No primitive
selector is copied or extracted. Ordinary pipeline lookup, scratch, binding,
capacity guards and fallbacks remain in their existing production positions.
The first ordinary warm call uses the fixed census to verify actual C gate/up,
39 Q5 down and three Q6 down banks, Q8-wide and strict-router target paths and
their dispatch geometry. Missing target paths, target rollback, A staging or
experimental/probe dispatches invalidate the packet after the command drains;
plain/profile cases do not then run. This is post-warm path verification, not
pre-weight pipeline completeness.

## Resource phases

`W` is the ordinary device-priced retained-window plan, `S` the complete
ordinary session-buffer ledger, and `R = 536870912` bytes is the existing
command-buffer/argument-table/allocator-slack policy reserve. `R` appears once
in each phase, not once per check or per case. It is not an exact PSO cost or
a measured/formal bound on lazy framework/driver internals.

The added diagnostic policy is frozen separately in both domains:

```text
P_M = P_U = 1 MiB + 984 * 1 MiB = 1032847360 bytes each
visible timestamp payload = 984 * 8 = 7872 bytes
front Metal = R; CPU = K_cap
cold Metal = W + S + R + P_M; CPU = K_counted + P_U
fresh Metal = S + R + P_M; CPU = K_counted + P_U
```

Weights are already live at the fresh gate, so `W` is not requested again.
No opaque counter allocation receives live-allocation credit: the full `P_M`
and `P_U` remain uncredited envelopes even after sampler creation. The CPU
upper also conservatively retains its counted parser/loader/control envelope;
it is not a measurement of remaining malloc bytes or an assertion that all
loader temporaries remain live. Report phase names identify this distinction.
There is no retained warm full-logit vector, cache disabling or scratch credit.
Split admission preserves CPU-availability, combined physical/process and
Metal working-set guards. Transactions enclose fresh memory-signal sampling,
the new gates, frontend preparation and weight/sampler/session/census factories.
The warm summary uses preallocated row-index/count pairs rather than allocating
name storage after execution. Ordinary later gates remain enabled.

The scoped ordinary predecode traversal caps six shards, 1412 tensors, 512
aggregate metadata KVs, one million aggregate array elements, 16 MiB aggregate
string bytes, 1 MiB per string and 16 MiB header traversal per shard. Existing
64-byte tensor-name, 1024-byte key and rank 1..4 limits remain. Split-vector
counts are capped before construction. No alternate parser bypasses validation.

`K_cap` uses these cap counts before decode/context. `K_counted` uses the
observed header counts and compiled host layouts, not an arbitrary 128 MiB:

```text
V = sizeof(serde_json::Value); T = sizeof(TensorDesc); Z = sizeof(String)
BTree_node_upper = 11*(2*Z + V + 8) + 16*sizeof(usize)
header_parser_loader = 4 * (KV*BTree_node_upper
  + elements*2*(V+32) + tensors*2*(T+64+4*8+64) + string_bytes)
  + page-rounded traversed header bytes
  + 6*(65536 + sizeof(GgufShard) + 4096)
host_bindings = 1412*8*(sizeof(MetalTensor)+T+64+4*8+8*sizeof(usize))
```

The four header representations cover the locked gguf-rs `3a92c51` parser,
retained metadata, rebind/validation overlap and capacity growth; a full
11-entry BTree node per KV deliberately overprices node occupancy. Array
scalars include a 32-byte numeric-string upper. Host binding terms retain
window/entry vectors, map buckets, cloned names and normalized shapes.
Session-host storage is counted from ordinary buffer-spec list lengths and
compiled state/route/scratch layouts, with four layout/capacity envelopes.
Ordinary transient CPU demand includes the full-vocabulary readback upper,
input/status rows and tensor-view shapes. Packet terms separately price
16384 fixed census rows with 128-byte names, 256 summary entries, 492 labels,
984 raw timestamps, bounded fixture descriptors/IDs, hashing, paths and the
4 KiB writer. Tokenizer demand is exactly zero. Terms also include
page-rounded embedded fixture/source/metallib bytes and a bounded 64 KiB
streamed-environment control envelope (no collected environment vector).
The macOS environment walker borrows the native table and checks entry count,
bounded string length, key/value lengths, total bytes and UTF-8 before yielding
a pair. It does not use `std::env::vars_os`, which snapshots before iteration.
Its unsafe borrowing contract requires the native table and strings to remain
valid and immutable: run this sole exact test before Metal/model workers, with
no in-process environment mutation. This is not a general concurrent process
environment API. CPU fixtures check refusal before copying an offending pair.
There is no credit for those already-mapped evidence pages. Numerical terms are emitted
by the packet's counted helper and ordinary device-priced ledger on the actual
build; no artifact quote is claimed now. Counting embedded byte lengths does
not preload or resolve research libraries or pipelines.

## Collection and lifetime

The ordered plan is `2 + 4*45 + 5*34 + 11 + 3 + 3*42 = 492` spans and 984
timestamps. The new entry consumes a preallocated recorder; it never calls
the old allocating stage-profiled API. The existing five KDA boundaries and
packed body are reused. Fixed census diversion precedes legacy row/string/tag
allocation. Overflow remains invalid, never a truncated valid census or a
reason to silently stop stage scheduling.

Sampler creation requires stage-boundary support and a timestamp-only set
with one timestamp counter, matching device and sample count. Set/counter
identity uses Metal's `MTLCommonCounterSetTimestamp` and
`MTLCommonCounterTimestamp`, not an assumed shared display name. Resolution must
return exactly 7872 bytes before copying into caller storage. Checked pair
indices, sentinel/zero/reversed pairs and missing/order errors refuse after
drain. Legitimate overlapping pairs are retained; old command-endpoint scaling
is retained. Routed and MLA stages remain combined parents. Coarse sums are
not exclusive fractions of ordinary wall time.

The exact ordinary command is retained before encoding, with the sampler when
present. A separate submission bit arms immediately after commit. Precommit
error/unwind discards without committing or waiting, including an uncommitted
Enqueued status; status alone never authorizes a wait. The submitted guard
drains before referenced session/recorder resources can unwind. This path uses
ordinary `queue.commandBuffer()`, not an unretained-resources command. Recorder,
census and TLS scopes restore on error/unwind. One outer autorelease pool owns
the run.

Exactly warm+census, plain1, plain2 and profiled sessions check healthy
0-to-2048 state and identical finite full-vocabulary F32-bit LE SHA256.
Session/sampler allocation, sampler resolution, test checks, hashing and
reporting stay outside the call timer. The ordinary packed call's own
transients/status checks/wait/readback remain inside its unchanged body.
Sampler preparation, plain drift and profile/mean overhead are separate.

The private create-new regular/no-follow file uses mode 0600, a 4096-byte
streaming serializer buffer, at most 1024 records and one MiB total. Typed
records cannot receive raw model/input/error strings. Flush/sync and a complete
event plus successful test exit are mandatory. Refusal/incomplete output is
not an interpretable CURRENT packet.

## Review and future execution

Independent review clears the preparation-scope follow-up. The first canonical
M5 attempt at `d81ad503` then refuses `counter_layout` in 0.20 s, before weight
construction or any model submission. Its partial quote is not a CURRENT
profile. The sampler incorrectly assumed the set and counter share the literal
name `timestamp`; a device-free read of Metal's public constants returns set
`timestamp` and counter `GPUTimestamp`. The narrow correction uses the distinct
public identity symbols, retaining the single-counter requirement, duplicate-set
refusal and exact resolution-byte checks. CPU regression rejects the old
incorrect counter spelling. Independent review clears the identity correction;
the focused CPU regression passes. This first refusal is retained separately
from the later successful artifact.

The corrected `20bd6dde` artifact passes sampler creation and model warmup, then
refuses `current_target_path` in 22.32 s. Plain/profile calls do not run. Source
inspection identifies a second diagnostic-name error: incumbent Q6 down uses
the maintained generic name generator and ends in `_generic`, which the packet
omitted. The narrow follow-up corrects that pin and the Q4/Q5 generic-fallback
pins, crosschecks MoE and Q8 names against maintained CPU name generators, and
emits bounded hash-only kernel counts before target-count assertions. The
successful-census marker remains after every path check; partial observations
are not an accepted profile. No production selector or expected count is changed.
The missing observed summary prevents attributing that refusal solely to Q6;
all other runtime paths required the subsequent successful verification. Neither
refused attempt contributes ordinary timing samples or an accepted ranking.

Ten new exact CPU classifications and one ignored timing classification extend
the existing inventory; historical classifications/limits are unchanged.
CPU contracts cover topology/pins/frozen hashes, controls, split gates before
factories, bounded collectors/writer, strict resolve/pairs, fake submitted
resource lifetime, inactive hooks and preservation of the ordinary packed body
apart from test-only lifetime hooks. No GPU context is constructed by them.

Development verification passes 368 exact classified library CPU tests,
including all ten additions, plus 47 separate retained GLM binding/coverage,
ledger/planner and Metal admission/pricing/refusal regressions: 415 distinct
CPU tests, zero failed, ignored or skipped. Library discovery matches all 598
classifications and 106 ignored entries using the unchanged strict inventory
checks. This uses the ordinary non-LTO development libtest, not a timing
artifact. Workspace checking, formatting, environment-reference checking and
inventory self-tests also pass. After the narrow follow-up, all 368 exact
classified library CPU tests and strict discovery/ignore-state validation pass
again, including transaction ownership/release on success, refusal, error and
unwind, preallocated summary reuse and borrowed-environment refusal controls.
All 368 classified library CPU tests and strict discovery pass again in the
final `ef87a265` canonical artifact after both identity corrections.
Other CLI/oracle acceptance targets are not
rebuilt or executed for this measurement unit.

Future invocation requires the exact ignored test only, a reviewed canonical
release/source/features receipt and its SHA in
`GLM53_CURRENT_EXPECTED_BINARY_SHA256`, `GLM53_GGUF`, and a new private
`GLM53_CURRENT_BUDGET_REPORT`. The entire permitted `QWEN_*` set is:

```text
QWEN_METAL_LEASE_FD=<genuine canonical descriptor>
QWEN_METAL_LEASE_WAIT=0
QWEN_GLM53_TENSOR_GATE_UP=1
QWEN_GLM53_KDA_HALF_INPUTS=0
QWEN_GLM53_DENSE_Q8_WIDE=1
QWEN_GLM53_TENSOR_Q5_DOWN=1
QWEN_EXPERT_TENSOR_TILE=1
QWEN_MATMAT_Q5_K_N64=1
QWEN_MATMAT_Q6_K_N64=1
QWEN_MATMAT_Q5_K_N64_MIN_N=1024
QWEN_MATMAT_Q6_K_N64_MIN_N=1024
```

Other QWEN, MTL and DYLD intervention variables refuse; no wired/lease bypass
is accepted. Source review must precede any canonical build or M5 slot. Execution
also needs the parent's external actual-timeout/failfast boundary; synchronous
Metal waits cannot be interrupted by between-call checks. Actual
sampler capability/layout, warm target census, four endpoint digests, admission,
counter overhead and coarse costs are verified for the frozen run, not other
workloads or devices. The 985 MiB policy
per domain is not a formal driver proof; no frequency monitoring, model-quality
authorization or inference from a successful metadata quote is implied.
