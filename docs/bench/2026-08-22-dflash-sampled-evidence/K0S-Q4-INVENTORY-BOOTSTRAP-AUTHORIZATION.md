# K0-S Q4 Inventory Bootstrap Authorization - 2026-08-23

Status: prospective source-work authorization only, frozen after bridge-tooling
commit `0502b403349eee58da265b104f66045632790024` and tree
`2dba2990bd9d43758bae2f974edef5c6c235ab9c`. This document authorizes the
minimum model-free tooling amendment needed to make a later Q4 inventory plan
non-circular. It authorizes ordinary source-validation compilation, unit tests,
clippy, and the existing synthetic Metal parity test, all without real assets.
It authorizes no inventory executable build for later use, real-asset open or
stat, operational inventory/device query, real-model construction, real-model
command buffer, forward, K0-S acquisition, preparation over real artifacts, E0
acquisition, retry, replacement, K0-L, E1b, verifier, serving, or product work.
The existing mandatory synthetic parity test alone may query the physical Metal
device and dispatch its fixed synthetic workload under the lease; that grants no
operational device or model authority.

The K0-S semantic-tooling authorization, amendments, and acquisition-bridge
authorization remain controlling except where this document explicitly changes
the classification of pre-inventory predicates, introduces a non-writing
preparation-hash mode, and separates planner custody from the clean preparation
output checkout. It explicitly supersedes the requirements at bridge lines
111-120 and 340-341 that the final inventory-bound preparation spec itself exist
before inventory. Before inventory, the later plan instead freezes an exact
preparation-choice template; after inventory, only the inventory path/hash and
inventory-spec path/hash may be inserted into the final all-literal preparation
spec. Passing source tests grants no operational build or inventory authority.

## Problem Frame And Current No-Go

The reviewed bridge is promotable, but its exact artifact contracts cannot yet
be instantiated truthfully:

1. `qwen.dflash_k0s_inventory_spec` currently requires exact target and drafter
   byte counts, tokenizer-token digest and embedding facts, scalar-fixture file,
   and command-template file before inventory. The drafter byte count and full
   tokenizer identity are not present in committed evidence; obtaining them is
   the first real-asset observation that inventory itself is meant to perform.
2. The future executable, embedded metallib, compiler identity, and source-state
   digest cannot be frozen until one clean release build exists. Guessing them
   in a plan or rebuilding after seeing them would be circular.
3. Preparation requires the four exact output SHA-256 values on its first and
   only writing invocation, but the reducer has no authenticated non-writing
   render mode from which a prospective amendment can freeze those values.
4. A preparation spec cannot contain the Git commit/tree that contains that
   same spec. The existing synthetic path correctly keeps planning inputs
   outside the clean output checkout, but the operational role split is not yet
   an explicit protocol contract.

Therefore the current decision is **NO-GO for build, inventory, or preparation**.
Historical E0 or aggregate identities cannot fill missing raw/tokenizer facts,
and no `${...}` placeholder may stand in for a value that an exact command or
file claim requires.

## Chosen Pareto Approach

Use one future clean build, one future real-asset inventory acquisition, and one
future writing preparation invocation. Add only two model-free bridge
capabilities:

```text
source-only bootstrap amendment
  -> separately authorized clean build and build-identity observation
  -> all-literal inventory spec and preparation-choice template committed
     before first asset open
  -> exactly one real-asset inventory
  -> inventory-bound preparation spec in planner checkout P
  -> non-writing deterministic preparation-hash observation
  -> all-literal hash amendment
  -> exactly one writing preparation in clean output checkout Y
  -> acquisition remains separately closed
```

This is preferred to a metadata-discovery acquisition because it avoids an extra
discovery result and replacement boundary. It does not claim one total lifetime
read of each asset. The inventory command performs the one acquisition that
creates new identity facts. Hash observation and writing preparation later each
perform separately planned, full, read-only same-FD revalidation of the already
committed inventory-bound assets; later reduction revalidates them again. Those
passes may confirm custody but cannot update inventory facts or rescue a failed
inventory. The inventory spec freezes every fact knowable before inventory,
while the inventory artifact authenticates facts whose observation is
inventory's purpose. The reducer independently derives and checks every observed
fact available from retained file descriptors. Prompt IDs are instead checked
against the frozen predicate and authenticated tokenizer metadata; the reducer
does not claim an independent tokenizer implementation. The producer cannot
backfill its already authenticated spec.

Three immutable roles remove Git/hash self-reference:

- execution worktree **X**: clean, detached at the eventual bootstrap-tooling
  commit; owns the one release build and every executable/source path;
- planner checkout **P**: owns committed plans, inventory/spec, preparation
  spec, and hash-observation reports; it may advance without changing X or Y;
- output checkout **Y**: a separate clean checkout whose exact pre-preparation
  HEAD/tree is bound by the preparation spec and seal; before preparation it
  contains none of the four prepared outputs.

The preparation spec is authenticated by path, bytes, SHA-256, seal, and the
later prospective run plan. It does not claim to live in the Y commit/tree that
it names. The later plan authenticates the descendant Y commit containing the
four prepared outputs and seal; the seal continues to name only Y's
pre-preparation input commit/tree.

## Frozen Development Choices

These values are exposed development choices, never held-out or reserve data:

- run family: `k0s-q4-write-code-20260823`;
- target path:
  `/Users/tito/models/Qwen3.8-27B-Q4_K_M.gguf`;
- target expected raw bytes: `17106773984`;
- target authorization cap: `17106773984` bytes exactly;
- target expected raw SHA-256:
  `7b2aec3b9ababdfd75aa17552ee95607d866e44decf547f6f12fcef85cc89f1b`;
- prior target aggregate, retained only as a non-substitutable historical
  reference:
  `fb9bcd41434a7fc1c2ed3b2d19a4ceb6cd8adb9cc55b9363ff1042acc2b3e048`;
- drafter path:
  `/Users/tito/models/incoai-dflash2/Qwen3.8-27B-DFlash2-Q4_K_M.gguf`;
- drafter expected raw SHA-256:
  `18a380efc9b7ed8d88677fc895f5c11ae170653434ee378f7348f715c14d0594`;
- drafter authorization cap: `2147483648` bytes; exact bytes are deliberately
  `null` in the v2 predicate and become an inventory observation;
- prompt UTF-8 text `Write code`, bytes `577269746520636f6465`, and SHA-256
  `b365a7d68fd699d7938042031965dac164cac0696dab1fb9be239b62c31e2734`;
- expected prompt token IDs `[7734, 1970]`, `add_special=false`, with canonical
  i32le SHA-256
  `39400964f33473f82888817289bca3bba220e3e9f51856ce1e5f88dc8b411f2c`;
- initial carry token `364`, continuation carry token `264`, and mask token
  `248070`;
- request temperature `0.7`, exact f32 bits `0x3f333333`;
- fixed chains `fixed-slot-one:364:1,1,1,1,1,1,1` and
  `varied-slots:364:0,1,2,3,4,5,6`;
- target vocabulary `248320`, DFlash block `8`, active depths `7`, selector
  top-K `16`, selector rank `256`, hidden width `5120`, and Q4_K selector-hidden
  dispatch through exact kernel `kernel_mat_mat_q4_K_f32`;
- host predicate `macos`, `aarch64`, device name `Apple M4 Max`; the exact
  required family set is `apple9`, `mac2`, `common3`, and `metal3` with all-of
  semantics. Registry ID and the complete ordered `mtl-gpu-family-v1:...`
  string remain inventory observations;
- hard caps: inventory and each existing trace/sidecar remain `67108864` bytes;
  inventory spec, preparation-choice template, and preparation spec are at most
  `1048576` bytes each; claims/files exceeding those caps fail before allocation;
  the non-writing hash report is at most `65536` bytes; combined trace/sidecar
  remains `134217728` bytes. Later plans may tighten but never expand them;
- environment exactly `QWEN_METAL_LEASE_WAIT=1` for every future Metal-capable
  command. Any additional behavior-affecting `QWEN_*`, `MTL_*`, `METAL_*`,
  `GGML_METAL_*`, or relevant `DYLD_*` variable fails closed.

The two ignored target-policy variants remain reducer-only metadata over the
same captured selector input:

- A: top-K `1`, top-p bits `0x3f000000`, min-p bits `0x00000000`, no grammar or
  penalties;
- B: top-K `200`, top-p bits `0x3f800000`, min-p bits `0x3d4ccccd`, no grammar
  or penalties.

They authorize no target sampling, extra target/drafter forward, arm, dispatch,
or acquisition observation.

## Authorized Source Surface

Only these baseline files may change:

| Source | Baseline SHA-256 | Authorized purpose |
| --- | --- | --- |
| `crates/qwen-cli/src/dflash_k0s.rs` | `68e99a18ca6faaef8cba2192c8476335cc9d7244c1582be0f08704619b6b6eba` | separate inventory predicates from observed facts and validate both without model construction |
| `crates/qwen-cli/src/bench.rs` | `e61ca31e17b8d4b3b4dfbdab876f55c5e3e64b273a3b1108c75083250b0f5fcd` | update only hidden feature-gated bootstrap/inventory command parsing and tests if required |
| `scripts/profile/dflash_k0s.py` | `e58e725d80b1331bd006c1616f36d5f3534389207dc483aadf9e284b83885e3c` | validate amended inventory custody and add pure non-writing preparation-hash observation |

No Cargo manifest/lock, build script, loader, tokenizer implementation, model,
Metal runtime, kernel, sampler, generation, serve path, other script, or product
source may change. No dependency may be added. Every call site remains under the
non-default `dflash-k0s-diagnostics` feature.

Although they are not editable under this authorization, v2 exact source claims
must add `tokenizer_rs`, `gguf_rs`, `source_identity_rs`,
`workspace_cargo_toml`, `cargo_lock`, `qwen_cli_build_rs`, and
`qwen_llm_lib_rs` to the existing required roles. The eventual clean build's
Git source-state digest still commits the complete tree. The future build plan
authenticates compiler executable path/hash and `rustc --version --verbose` in a
separate bounded build report; current `build-info` authenticates commit/source
state but is not represented as authenticating compiler path/version.

The role-to-X-relative-path map is exact:

```text
metal_dflash_rs       crates/qwen-llm/src/metal_dflash.rs
bench_rs              crates/qwen-cli/src/bench.rs
dflash_k0s_rs         crates/qwen-cli/src/dflash_k0s.rs
qwen_llm_cargo_toml   crates/qwen-llm/Cargo.toml
qwen_cli_cargo_toml   crates/qwen-cli/Cargo.toml
metal_rs              crates/qwen-llm/src/metal.rs
metal_forward_rs      crates/qwen-llm/src/metal_forward.rs
dflash2_metal         kernels/dflash2.metal
mat_mat_mma8_metal    kernels/mat_mat_mma8.metal
mat_mat_q4_k_metal    kernels/mat_mat_q4_k.metal
build_rs              crates/qwen-llm/build.rs
tokenizer_rs          crates/qwen-llm/src/tokenizer.rs
gguf_rs               crates/qwen-llm/src/gguf.rs
source_identity_rs    crates/qwen-cli/source_identity.rs
workspace_cargo_toml  Cargo.toml
cargo_lock            Cargo.lock
qwen_cli_build_rs     crates/qwen-cli/build.rs
qwen_llm_lib_rs       crates/qwen-llm/src/lib.rs
```

Producer and reducer require this exact map, unique roles, unique canonical
paths/inodes, containment under X, and exact path-to-role equality. A valid hash
under the wrong role is invalid.

That later report uses schema `qwen.dflash_k0s_build_identity_report` v1,
authority
`development_k0s_build_identity_only_no_asset_model_forward_or_acquisition_authority`,
and cap `1048576` bytes. Its exact top-level order is
`schema,schema_version,authority,run_id,attempt_id,checkout,build_command,
build_root,executable,embedded_metallib,reducer,sources,
compiler,target,profile,features,build_info,environment`. `compiler` has exact
keys `path,bytes,sha256,version_verbose,version_verbose_sha256`.

- `checkout` uses exact keys `path,commit,tree,dirty`, with canonical X path,
  full Git OIDs, and `dirty=false`;
- `build_command` is the exact nonempty UTF-8 argv array from the later plan;
- `build_root` uses exact keys `path,bytes,max_bytes`, where `path` is canonical,
  both sizes are u64, and `bytes<=max_bytes`;
- `executable`, `embedded_metallib`, and `reducer` are complete `FileClaim`s
  with exact keys `path,bytes,sha256,max_bytes`;
- `sources` is the exact ordered array of `role` plus `FileClaim` objects from
  the role/path map above;
- `target` and `profile` are UTF-8 strings exactly
  `aarch64-apple-darwin` and `release`; `features` is exactly
  `["dflash-k0s-diagnostics"]`;
- `build_info` has exact keys `artifact,schema_version,build_commit,
  build_commit_short,build_dirty,build_source_state,stamp_source,stamp_error,
  runtime_commit,runtime_dirty,runtime_source_state,status,problems,overrides`.
  `artifact` is its complete `FileClaim`; optional values are explicit JSON
  nulls; status is `match`; both dirty values are false; problems/overrides are
  empty arrays; commit/source states agree with `checkout` and the executable;
- `environment` is a lexicographically ordered JSON object of UTF-8 string keys
  and values containing every allowed build-affecting override. The later plan
  freezes it exactly and separately requires all unlisted `RUSTC*`, `RUSTFLAGS`,
  `CARGO_*`, `SDKROOT`, and `MACOSX_DEPLOYMENT_TARGET` overrides absent.

The report does not claim its own hash. A later plan independently records its
path/bytes/SHA-256. Inventory-spec `build_report`, artifact `expected.build_report`,
and artifact `observed.build_report` are the identical complete report
`FileClaim`, not parsed report content. Producer and reducer independently open
that claim, parse the nested schema above, and require all `BuildClaim`
compiler fields and executable/metallib/source facts to equal the parsed report.

## Inventory Predicate Amendment

Introduce incompatible schemas `qwen.dflash_k0s_inventory_spec` v2 and
`qwen.dflash_k0s_inventory` v2. Every amended producer and validator rejects v1;
there is no dual interpretation or migration fallback. Replace only circular
pre-observation claims with strict predicates:

1. **Asset expectation** has exact ordered keys
   `role,path,expected_bytes,max_bytes,sha256`. `expected_bytes` is either a JSON
   integer or JSON `null`, never omitted. Target uses exact bytes and equal cap
   `17106773984`; drafter uses `null` and cap `2147483648`. Observed bytes come
   from the same retained descriptor used for the whole-file hash. The artifact
   emits a complete `FileClaim` for each asset.
2. **Tokenizer predicate** has exact ordered keys
   `vocab_size,token_embd_name,token_embd_rank,token_embd_hidden,
   token_embd_vocab_axis,allowed_token_embd_dtypes,require_token_metadata,
   metadata_identity_domain`. Values are `248320`, `token_embd.weight`, rank
   `2`, hidden dimension `5120`, vocabulary axis `1`, exact allowed dtype list
   `["Q4_K"]`, `true`, and
   `qwen.dflash_k0s.tokenizer_metadata.v1`. Inventory requires actual metadata
   for `general.architecture`, `tokenizer.ggml.model`, `tokenizer.ggml.pre`,
   `tokenizer.ggml.tokens`, `tokenizer.ggml.token_type`,
   `tokenizer.ggml.merges`, optional BOS/EOS IDs, and add-BOS/add-EOS flags. The
   old count/embedding fallback is forbidden.
3. **Prompt predicate** has exact ordered keys
   `utf8_hex,utf8_sha256,add_special,expected_token_ids,
   expected_token_ids_sha256_i32le`; values are the frozen choices above and
   `add_special=false`. Rust tokenization must equal those IDs exactly. Python
   independently validates the complete tokenizer metadata identity and fixed
   IDs/digest; it does not claim a second tokenizer implementation.
4. **Tokenizer observation** has exact ordered keys
   `vocab_size,token_embd_name,token_embd_shape,token_embd_dtype,token_count,
   model,pre,bos_token_id,eos_token_id,add_bos_token,add_eos_token,
   token_list_sha256,token_type_sha256,merges_sha256,
   metadata_identity_sha256`. Component domains are exactly
   `qwen.dflash_k0s.tokenizer.tokens.v1`,
   `qwen.dflash_k0s.tokenizer.token_type.v1`, and
   `qwen.dflash_k0s.tokenizer.merges.v1`. Each digest starts with its literal
   ASCII domain and a little-endian u64 item count. Tokens/merges then encode
   each ordered UTF-8 value as little-endian u64 byte length plus bytes; token
   types encode each ordered value as little-endian i64.
   `metadata_identity_sha256` starts with its frozen domain, then commits exact
   GGUF keys `general.architecture`, `tokenizer.ggml.model`,
   `tokenizer.ggml.pre`, the three component counts/digests,
   `tokenizer.ggml.bos_token_id`, `tokenizer.ggml.eos_token_id`,
   `tokenizer.ggml.add_bos_token`, and `tokenizer.ggml.add_eos_token` in this
   order. Every key is u64-length-prefixed UTF-8; strings are u64-length-prefixed
   UTF-8; counts are u64 LE; digests are 32 raw bytes; optional IDs use a
   one-byte presence tag then i64 LE; optional booleans use a one-byte presence
   tag then one byte `0` or `1`. Absence and an explicit default are distinct.
5. **Prompt observation** has exact ordered keys
   `utf8_hex,add_special,token_ids,token_ids_sha256_i32le,
   tokenizer_metadata_identity_sha256` and must equal its predicate plus the
   observed tokenizer identity.
6. **Host predicate** has exact ordered keys
   `os,arch,device_name,required_families,family_match`, with the frozen values
   above and `family_match="all"`. Device observation retains existing host keys
   and must contain every required family exactly once. No command queue,
   pipeline, allocation, or model object may be created.
7. **Mask predicate** has exact ordered keys
   `allowed_metadata_keys,expected_mask_token` and requires exactly one of
   `dflash-draft.dflash.mask_token_id` or
   `tokenizer.ggml.mask_token_id`, with value `248070`.
8. **Tensor predicates** remain exact for names, Q4_K selector-hidden dtype,
   geometry, orientation, and row domains. Offsets, byte ranges, and hashes are
   observed by bounded GGUF parsing and same-FD hashing.
9. **Build and source claims** remain exact. They can be materialized only after
   the separately authorized clean build; no inventory-spec placeholder or
   permissive mismatch is allowed. Compiler identity is copied only from the
   separately authenticated build report and is explicitly external to
   executable `build-info` attestation.
10. Remove scalar-fixture and acquisition-command-template *file* claims from
   inventory. Preserve their compiled/source contracts: the six-case scalar
   fixture digest is checked from the authenticated executable/library source,
   and an inline canonical command-template contract is schema-validated, while
   preparation deterministically emits the fixture and command. They are not
   asset facts and must not force pre-inventory external files.

The v2 inventory-spec top-level exact order is:

```text
schema,schema_version,run_id,inventory_max_bytes,checkout,build,build_report,
sources,executable,reducer,embedded_metallib,assets,tensor_requirements,
tokenizer_predicate,prompt_predicate,mask_predicate,parser_caps,
host_predicate,command,environment
```

The v2 inventory-artifact top-level exact order is:

```text
schema,schema_version,authority,inventory_spec_sha256,run_id,expected,
observed,command,environment
```

`expected` repeats, in spec order, every authenticated spec field after
`inventory_max_bytes`. `observed` has exact order
`checkout,build,build_report,sources,executable,reducer,embedded_metallib,
device,assets,gguf,tensors,tokenizer,prompt,mask_noise,parser_caps`. Unknown, duplicate,
missing, or legacy keys fail before semantic validation. The artifact retains
authority
`development_k0s_inventory_only_no_model_forward_or_semantic_authority`.

Inventory validation opens every claimed source/asset once, independently
parses both GGUFs, compares all observed claims to their predicates, and performs
final same-FD custody checks. It rejects unknown keys, missing facts,
substitutions, and any attempt to move an observed fact into a less restrictive
predicate.

Python `open_regular` must use lexical canonical-path checks, retain each
canonical parent directory FD, and open leaves directory-relatively with
`O_NOFOLLOW`. It requires regular files with `st_nlink==1`, retains parent and
leaf device/inode/size, mtime/ctime nanoseconds and hash, and revalidates the
parent FD, leaf FD, and canonical parent/leaf path identities at final custody.
The running reducer executable/script path must equal the authenticated reducer
claim; opening a different claimed copy is insufficient.

No bootstrap or inventory mode may construct `Model`, `MetalContext`,
`MetalModel`, `MetalDFlashHead`, `MetalSession`, or `MetalDFlashSession`; call a
forward/draft API; create a command queue/buffer/pipeline; observe logits,
hidden state, candidates, `z_t`, scores, generated/drafted/sampled or
forward-produced tokens, chains, model state, timing, or memory-performance
outcomes; or invoke Cargo/compiler/linker/build scripts. Reading fixed prompt
tokenization and deterministic mask/noise IDs is the only token observation in
scope.

## Pre-Inventory Preparation Choices

The later inventory plan must also commit
`qwen.dflash_k0s_preparation_choices` v1 before the first asset open. Its exact
top-level order is
`schema,schema_version,run_id,attempt_id,worktree_x,planner_p,control_y,outputs,
preparation_spec_path,fixture_content,acquisition_outputs,reduction_output,
continuation_carry_token,manifest_choices,transformation_sha256,
environment_allowlist,arm_order,selected_arm,parity_comparison_fields,
reducer_argv_template,failure_policy`.
It freezes every current preparation-spec choice except inventory/inventory-spec
paths and hashes. It contains no tokenizer, tensor, device, draft, state, parity,
or rendered-output observation.

After inventory is committed, the final v2 preparation spec is a deterministic
join of this byte-authenticated template and exactly four inventory-derived
values: inventory path/SHA-256 and inventory-spec path/SHA-256. Every other key
must compare byte-for-byte with the template. This is the only timing exception
to the prior bridge requirement; it cannot be used to choose chains, policy,
paths, host, selector predicate, arm order, reducer command, or failure handling
after inventory.

The final `qwen.dflash_k0s_preparation_spec` v2 exact top-level order is
`schema,schema_version,run_id,attempt_id,preparation_choices,inventory_path,
inventory_sha256,inventory_spec_path,inventory_spec_sha256,
preparation_spec_path,worktree_x,planner_p,control_y_input,outputs,
fixture_content,acquisition_outputs,reduction_output,continuation_carry_token,
manifest_choices,transformation_sha256,environment_allowlist,arm_order,
selected_arm,parity_comparison_fields,reducer_argv,failure_policy`.
`preparation_choices` is a complete `FileClaim` for the template. Both
observation and writing modes require independently supplied template path/hash,
open the same claimed file, and prove that every copied field is byte-identical.
`preparation_spec_path` is frozen by the template and self-identifies only its
path, never its own bytes/hash.

## Non-Writing Preparation-Hash Observation

Add a mutually exclusive reducer mode named
`--observe-preparation-hashes` (exact spelling may change only before tests are
frozen). It accepts the same authenticated inventory, inventory spec, and
preparation-choice template and preparation spec path/hash pairs as `--prepare`,
plus one exclusive report output with hard cap `65536`. It must:

- execute the exact same validation and deterministic render function as
  `--prepare`;
- render fixture, command, manifest, and seal bytes in memory only;
- require all seven future fixture/command/manifest/seal/trace/sidecar/reduction
  leaves absent before rendering and after report sync; create none of them;
- emit exactly one bounded report containing each rendered byte count/SHA-256,
  all four input identities, reducer identity, run/attempt IDs, X identity, and
  Y pre-preparation HEAD/tree/status;
- final-check every retained input FD and X/Y Git identity before and after
  writing the report;
- use exclusive creation, never overwrite, and leave any partial report terminal;
- contain no semantic/model output and grant no preparation or acquisition
  authority.

The report schema is `qwen.dflash_k0s_preparation_hash_observation` v1 with exact
top-level order
`schema,schema_version,authority,run_id,attempt_id,inventory,inventory_spec,
preparation_choices,preparation_spec,reducer,rendered,worktree_x,control_y,
report,environment`.
The four inputs and reducer use complete file claims; `report` has exact keys
`path,max_bytes` and cannot contain its own bytes/hash. `rendered` has
exact order `fixture,command,manifest,seal`, each with `bytes,sha256`;
`worktree_x` and `control_y` bind canonical path, HEAD, tree, status, common Git
directory and object-store identities. Authority is
`development_k0s_preparation_hashes_only_no_asset_discovery_model_forward_or_acquisition_authority`.

The writing `--prepare` mode remains unchanged in principle: it requires all four
independently supplied expected hashes, reserves exactly the four preparation
outputs relative to retained Y directory custody, preserves partials, and never
retries. Source tests compare both modes' rendered bytes byte-for-byte and prove
that observation creates only its report.

Expected rendered hashes may appear only in the P-resident observation report,
its later hash amendment, and the writing-preparation plan/argv. They must never
be inserted into the preparation spec, X, Y input commit/tree, fixture, command,
manifest, or seal.

## Planner And Output Custody

The preparation spec lives in P and binds an already clean, distinct Y. It must
freeze P input paths/hashes, X path/commit, Y path/current HEAD/tree, all seven
future output paths, arm order, selected `on-A`, parity fields, environment,
failure policy, and reducer argv. Y may contain tracked repository files but no
modified path or unrelated untracked file. The four preparation output leaves,
future trace, sidecar, and reduction leaves must all be absent, direct children
of Y, canonically distinct, and disjoint from every input.

The hash-observation report is a direct child of a prospectively frozen canonical
P root, never Y. Before observation, P must have an exact committed HEAD/tree,
clean status, shared repository/object-store identity with X and Y, and an absent
report leaf. The report path is canonically and by inode disjoint from every
input and all seven Y outputs. Observation retains a P directory FD and uses
directory-relative `O_EXCL|O_NOFOLLOW`; it rechecks P/X/Y HEAD/tree/status and
directory inode before and after sync. A collision or partial is terminal and is
not removed. The later P amendment authenticates the completed report's external
path/bytes/SHA-256. Committing that report or a hash amendment may advance P
only. Y stays at the exact input HEAD/tree until the one writing preparation
invocation.

## Required Source Tests

Before any operational build plan, this source phase may run ordinary
source-validation builds/tests only:

- amended inventory spec/artifact exact-key and duplicate-key rejection;
- synthetic target/drafter files proving asset expected-hash/cap versus observed
  byte count, including target exact-size and drafter size/cap failures;
- independently derived tokenizer/token-list/embedding facts, fixed prompt IDs,
  mask/noise hash, Q4 selector tensors, GGUF bounds, and adverse substitutions;
- source isolation proving inventory has no model/session/Metal queue/forward or
  build call path and default builds expose no hidden command;
- clean build/source/metallib claims remain mandatory and cannot be observed from
  or rewritten by inventory;
- non-writing preparation-hash observation is deterministic, creates one report
  only, and matches the writing renderer byte-for-byte;
- report collision, input mutation, X/Y drift, output preexistence, path/inode
  alias, mode mixing, placeholder mutation, and every rendered-hash mutation fail
  closed without retry;
- Python no-follow/link/timestamp/path custody, running-reducer identity, exact
  v2 source-role coverage, tokenizer metadata-domain vectors, and report schema;
- existing 28 library K0-S tests, 35 CLI tests, 121 reducer checks, strict
  modified-target clippy, default-feature isolation, and exact leased release
  synthetic Metal parity continue to pass.

Every Cargo or Metal-capable test command literally sets
`QWEN_METAL_LEASE_WAIT=1`. Synthetic tests open no real model asset.

The frozen validation commands are the existing reducer self-test, default and
feature checks, focused K0-S library/CLI tests, strict clippy on the three
modified targets, and exactly:

```sh
QWEN_METAL_LEASE_WAIT=1 QWEN_REQUIRE_METAL_TESTS=1 cargo test -p qwen-llm --release --features dflash-k0s-diagnostics k0s_synthetic_metal_diagnostic_parity_both_orders -- --nocapture --test-threads=1
```

No ignored real-model test may be enabled by this source phase.

## Stage Exit And Later Plans

This source amendment exits only when committed, independently adversarially
reviewed, and integrated. Source-validation artifacts have no operational
authority. It grants no authority to build or inspect future inventory
artifacts.

A separate **build plan** must then freeze the amended tooling commit/tree,
immutable X path, initially absent exclusive build root, exact offline locked
release command, compiler/toolchain, feature set, output cap, one build attempt,
and exact `build-info` observation. No future executable/metallib hash may appear
before that build. The build report, not `build-info` alone, binds compiler
executable path/hash and verbose version.

After build identities are committed, a separate **inventory plan** must freeze
the complete all-literal inventory spec, its SHA-256, exact command/output/caps,
P/Y state, decision rule, and replacement boundary before the first asset open.
It also freezes the complete preparation-choice template; only the later
inventory path/SHA-256 and inventory-spec path/SHA-256 may enter the
post-inventory preparation spec. No replacement is granted by default. Once
either asset is opened, every error,
signal, mismatch, or partial is terminal and requires incident review plus a new
attempt identity.

After inventory, materializing the exact preparation spec is a source-only join
over committed template/inventory bytes and opens no asset. Separate
hash-observation and writing-preparation plans authorize the first and second
post-inventory full read-only asset revalidations and freeze all-literal
commands/outcomes. Success
still grants no model execution. A later acquisition plan must authenticate the
descendant Y commit containing the prepared artifacts and seal, then separately
freeze the four-arm and reducer commands. No further development E0 acquisition
is allowed.

## Recommendation And Go/No-Go

Recommendation: **GO only for this three-file source amendment**, followed by an
adversarial review that tries to reintroduce a hidden asset preflight, spec
backfill, model/runtime construction, preparation write, Git self-reference, or
retry. Iterate NO-GO until the amended schemas classify every field as either a
strict preregistered predicate or an independently derived observation and the
non-writing renderer has exact byte parity with `--prepare`.

Current decisions:

- source amendment described here: **GO, prospectively, after review**;
- clean build: **NO-GO pending committed amendment and separate build plan**;
- target/drafter stat or open: **NO-GO**;
- inventory: **NO-GO pending exact post-build inventory plan**;
- preparation-hash observation over real inventory: **NO-GO pending inventory**;
- writing preparation: **NO-GO pending hash amendment**;
- K0-S acquisition/reduction, K0-L, E1b, verifier, serving, product: **NO-GO**.
