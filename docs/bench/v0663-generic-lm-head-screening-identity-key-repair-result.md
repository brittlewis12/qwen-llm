# v0.663 Generic Lm-Head Screening Identity-Key Repair Result

Status: sealed `INVALID` with `authority=[]`. The sole acquisition is consumed.
No prefill, generation, capture, screening, exact-survivor, byte-ledger, or
mechanism result exists.

## Terminal Record

- Root:
  `target/profiles/v0663-generic-lm-head-screening-oracle-a3b-p1/`.
- Clean build/runtime commit:
  `78da21d6719a118313c34f6433a1ef3f64bfcd22`.
- Executable SHA-256:
  `4ac50118eefcf172b2cbbd0555b8b83fba78e293af25f36d65caf6b6ff7846aa`.
- Readiness SHA-256:
  `40f9cbb474f10822c01ea80b9afda6caecc220344657c764e2bb314b5fad5059`.
- Self-tests SHA-256:
  `b39873acf072f344894a7bf775c08548ad6e596665f5cfbb24dd92d498b51944`.
- Artifact-manifest SHA-256:
  `a40ba99f440db3c3e053ba080c04cf205a46af897597b5c8f70ebd71532da882`.
- Decision SHA-256:
  `759750cfb5a8af2621e3c8c0dc9a8d120ad7a1df46ddeda223451f61e731dbcc`.

The manifest authenticates exactly `readiness.json` and `self-tests.json`.
The four preregistered payload directories are empty. Neutral mechanism flags
are `INVALID` defaults, not passes.

## What Cleared

Readiness and all eleven pre-model tests passed, including the 196-byte
EOF-positioned, absent-path fixture through the shared dual-name extractor.
The real path then completed:

1. host-before validation and complete model-file authentication;
2. both opened-GGUF parses and complete output-head hashes;
3. the dual metadata names and every frozen model identity check;
4. `Runtime::metal`, resident `MetalModel` loading, and loaded-GGUF
   reauthentication; and
5. tokenizer construction, vocabulary check, and frozen prompt byte/hash check.

This certifies both v0.662's parser repair and v0.663's identity-key repair.
Metal resources and resident weights were created, but no model forward or model
output occurred.

## Failure And Cause

The runner encoded the prompt, then failed its combined count-and-digest gate.
A post-acquisition diagnostic established that the deterministic stream contains
419 IDs and that the oracle compared a domain-separated digest against the
frozen production raw-i32le digest. Production computes:

```text
SHA256(token[0].to_le_bytes() || ... || token[n].to_le_bytes())
```

The oracle prepended `qwen-token-ids-i32le/v1\0`. A post-acquisition vocab-only
diagnostic on the same native tokenizer and prompt produced:

```text
token count:            419
production raw digest:  fb4bbb4dc66ca7d219099e2974e787ef976f80789cde3e48b8a905dceece1f9f
oracle domain digest:   28f8f41b73fee1823d2fc652f630fd796354860b8c48e4eb37dfc2faf323f931
```

The frozen token stream and constant are correct; the oracle selected the wrong
digest contract. `capture_request` was never entered, so no sequence, scratch,
prefill, sampler call, transition, logits, hidden state, capture manifest,
post-capture host check, metadata pass, exact validation, or analyzer exists.
There is no screening-mechanism inference.

## Decision

Do not rerun or modify v0.663. A separately preregistered v0.664 may centralize
the production raw-i32le token digest and use it for prompt identity and prompt
evidence only. Domain-separated generated-prefix and generated-output identities
remain unchanged. All model work, capture order, screening arithmetic, traffic
accounting, gates, dispositions, and authority stay frozen.

Post-acquisition review: `cx` session
`019fbf34-6270-72a3-b583-299d15505a88`.
