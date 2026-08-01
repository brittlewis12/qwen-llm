# v0.662 Generic Lm-Head Screening Parser Repair Result

Status: sealed `INVALID` with `authority=[]`. The sole acquisition is consumed.
No real capture, screening, exact-survivor, byte-ledger, or mechanism result
exists.

## Terminal Record

- Root:
  `target/profiles/v0662-generic-lm-head-screening-oracle-a3b-p1/`.
- Clean build/runtime commit:
  `70e999edd8d422bdb1078bcfec9d64225c6098f8`.
- Executable SHA-256:
  `5bac1520d6866c2332913962486ad77d37efa0c7808fe9b7d4d6edc11206f5ec`.
- Readiness SHA-256:
  `af528275cab915edbce852a414ed60cd632e9227ee1117e095dcf7949e9b150a`.
- Self-tests SHA-256:
  `77a6e76e4f120de9f937f43c9d239185f791c29df540d96feb2cd0b0452e8f4e`.
- Artifact-manifest SHA-256:
  `2f6a10c992e43f8a6d9877eec169e6a833cb433f3fede25f680913a4cb4761f0`.
- Decision SHA-256:
  `a457ae97a9f51ef62e2d01ff4c30f8f012421f6fab7c336cf44d536d6ac32c80`.

The manifest authenticates exactly `readiness.json` and `self-tests.json`.
The four preregistered payload directories are empty. The false mechanism flags
in `decision.json` are `INVALID` defaults, not observed passes.

## What Cleared

Typed readiness binds the exact command, clean source/build identity, release
executable, operator attestation, and v0.661 predecessor decision. All eleven
pre-model self-tests passed. The acquisition-time 68-byte fixture proved the
shared-EOF-cursor repair with an absent diagnostic path.

The real-model path then completed:

1. nofollow open, frozen size and complete model SHA-256;
2. the repaired descriptor-bound `GgufFile::from_opened_file` parse;
3. retained stamp, one-shard, mapped-length, and stop-token checks;
4. CPU `Model::from_gguf`, exact architecture equality, untied output, zero MTP,
   and `general.architecture == qwen35moe`.

This certifies the v0.662 parser repair on both the synthetic fixture and the
real 22.1 GB file.

## Failure And Scope

The next assertion incorrectly required:

```text
general.basename = Qwen3.6 35B A3B
```

Existing path-associated v0.403 metadata evidence records two distinct fields:

```text
general.base_model.0.name = Qwen3.6 35B A3B
general.basename          = Qwen3.6-35B-A3B
```

The preregistered spaced semantic value was transcribed from the first field,
but the implementation selected the second key. This is a fixture identity-key
error, not screening evidence.

The failure preceded `general.file_type`, `output.bias` absence, explicit Q6_K
dtype/shard/absolute-offset/extent checks, the complete output-head hash, the
independent analysis parse, `Runtime::metal`, Metal model loading, tokenization,
prefill, generation, capture, and analysis. CPU `Model::from_gguf` had already
selected the separate untied `output.weight` and validated its shape. No claim
about pruning, charged bytes, ideal-winner agreement, or production value is
permitted.

## Decision

Do not rerun or modify v0.662. A separately preregistered v0.663 may bind
`general.base_model.0.name` to the spaced value and `general.basename` to the
hyphenated value under a new schema/root. All mechanism inputs, arithmetic,
traffic accounting, gates, dispositions, and authority remain frozen.

Post-acquisition review: `cx` session
`019fbf34-6270-72a3-b583-299d15505a88`.
