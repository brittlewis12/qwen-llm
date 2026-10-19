# DeepSeek-V4 paper reference

- **Title:** DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence
- **Authors:** DeepSeek-AI
- **arXiv:** [2606.19348v1](https://arxiv.org/abs/2606.19348v1)

## Repository implementation notes

This repository implements a native Rust and Metal DeepSeek V4 family path
targeting the V4 Flash 0731 release. The implementation has separate model,
session, and snapshot components. The current serving behavior and limits are
documented in [SERVE](SERVE.md); performance evidence is indexed in
[PERF-LOG](PERF-LOG.md) and the dated packets under [bench](bench/).
