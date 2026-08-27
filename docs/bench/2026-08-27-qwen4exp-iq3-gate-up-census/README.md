# Flash-Next IQ3 Gate/Up Route Census

This directory records the clean, performance-ineligible route-count census
generated from checkpoint `69e51c9d2cf71a2e2872ae1e3053f09618c22cf7`.
The tracked-diff digest is the SHA-256 of an empty byte stream. The report binds
the exact test executable and embedded metallib; source metadata is explicitly
operator-declared, and local model files are identified by revalidated file
descriptor stamps rather than full-byte hashes.

The three workloads are individual controls, not population samples:

- N=18 canonical HELLO has 2,918 active standard expert instances and 16.57%
  active-panel lane occupancy at width 16.
- N=2,048 repeated `<|im_start|>` is a concentrated-route stress control with
  97.62% of standard-layer route mass in count 65+.
- N=2,048 natural technical text has 15,560 active standard expert instances.
  Its width-16 panels are 86.10% occupied, but only 63,924 of 2,818,048 panels
  are active before the ten-output-panel multiplier.

The census selects no kernel. It authorizes a separate no-work/count-band probe
to determine whether compact N16 active-panel descriptors merit a falsifier.
