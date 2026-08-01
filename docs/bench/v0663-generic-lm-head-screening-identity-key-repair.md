# v0.663 Generic Lm-Head Screening Identity-Key Repair

Status: preregistration. v0.662 is sealed `INVALID`; no v0.663 implementation,
packet root, model observation, screening result, or authority exists.

## Inheritance And Question

Inherit `docs/bench/v0661-generic-lm-head-screening-oracle.md` and
`docs/bench/v0662-generic-lm-head-screening-parser-repair.md` in full. Preserve
the model, prompt, production capture path, prophecy firewall, exact arithmetic,
byte ledger, gates, decision precedence, parser rewind, readiness contract, and
narrow authority.

v0.662 proved the parser repair, then failed because the implementation compared
the spaced base-model value against the hyphenated basename key. v0.663 changes
only the frozen metadata identity binding:

```text
general.base_model.0.name = Qwen3.6 35B A3B
general.basename          = Qwen3.6-35B-A3B
```

Introduce one typed name extractor shared by production `validate_raw_gguf` and
the fixture. It must return exact `base_model_name` and `basename` fields from
the two keys above. Persist both fields in `ModelIdentityRecord`, require their
exact values during capture-manifest validation, and do not accept either key as
a fallback for the other.

Add one pre-model in-process fixture containing both deliberately different
string values. Parse it through the opened-file seam, call the shared extractor,
and archive its exact typed result. A fixture-only pair of independent `get_str`
calls is insufficient. The existing EOF-positioned, absent-diagnostic-path
repair test remains mandatory.

No changed model, path fallback, relaxed identity check, removed full-file or
head hash, alternate prompt, changed stop vector, adaptive screen, extra model,
or second acquisition is permitted.

## Static Reconciliation

Before this preregistration, reached v0.662 authentication and older static
evidence were reconciled across the full frozen identity and request tuple:

- v0.662 itself authenticates the exact model size/full SHA, architecture tuple,
  untied output selection and shape, zero MTP, and stop vector before the failed
  key check;
- the path-associated same-model v0.403 command binds
  `/Users/tito/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` and records architecture
  `qwen35moe`, both identity strings above, file type 15, 40 blocks, hidden
  2,048, EOS 248046, Q6_K `output.weight` `[2048,248320]`, and no
  `output.bias` entry;
- v0.602/v0.603 copied-loader inventories bind the absolute first tensor offset
  `10,990,048` and output extent `417,177,600` bytes;
- v0.655/v0.656 authenticate model size `22,134,528,992`, model SHA-256
  `ac0e2c1189e055faa36eff361580e79c5bd6f8e76bffb4ce547f167d53e31a61`,
  and complete output-head SHA-256
  `122386a599833ffec3a266e48dd8a90aedc65e24c31323ce2388f40d7cc7b30c`;
  and
- v0.657-v0.660 request evidence binds prompt bytes `1,891`, prompt SHA-256
  `e265de9742d1b22e566fc108ae26331ccf46166c6f071e73f211e0a1a7e8b474`,
  token count `419`, and prompt-token SHA-256
  `fb4bbb4dc66ca7d219099e2974e787ef976f80789cde3e48b8a905dceece1f9f`.

The v0.515 artifacts corroborate only the two metadata-key semantics; their MTP
or Q4_K_S profile constants have no authority here.

## Repair Gates

Before the sole acquisition:

- the shared typed extractor and dual-key fixture pass in the release
  acquisition self-test and a focused debug/release unit test;
- the EOF-positioned opened-file regression and path-identity/split rejection
  tests remain green;
- all oracle release tests pass unchanged apart from the added identity fixture;
- `cargo check --locked -p qwen-cli --bin qwen-bench`, formatting, and
  `git diff --check` pass; and
- independent review confirms that no mechanism input, arithmetic, ledger,
  disposition, or authority changed.

The packet schema becomes `qwen-lm-head-screening-oracle/v0663`. The fixed
artifact inventory and readiness/capture binding remain v0.662's. The initially
absent root is
`target/profiles/v0663-generic-lm-head-screening-oracle-a3b-p1`.

## Sole Acquisition

```sh
target/release/qwen-bench lm-head-screening-oracle \
  --model /Users/tito/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  --prompt-file docs/bench/tokenizer-prompts/current-reva-n8-interactive-qwen36.txt \
  --tokens 128 --capture-calls 0,1,7,31,63,127 \
  --packet-dir target/profiles/v0663-generic-lm-head-screening-oracle-a3b-p1 \
  --attest-no-other-user-gpu-workload
```

Require the same clean release identity, empty inherited `QWEN_*` controls,
host-validity gates, artifact contracts, and terminal authority. The v0.661 and
v0.662 roots remain untouched. An unsealed or `INVALID` v0.663 root is consumed
and requires a new preregistration.

Predecessor decision SHA-256:
`a457ae97a9f51ef62e2d01ff4c30f8f012421f6fab7c336cf44d536d6ac32c80`.
v0.662 post-acquisition review: `cx` session
`019fbf34-6270-72a3-b583-299d15505a88`.
