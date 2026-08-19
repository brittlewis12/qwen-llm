//! `qwen serve` — Open Responses subset facade (S1).
//!
//! Contract: docs/SERVE.md. This module tree is layered so everything below
//! the HTTP loop is pure and unit-testable without a model:
//!
//! - [`items`]: wire types, request parsing, item-sequence validation
//!   (review defect 1: item-list grammar, not turn grammar), and the spec
//!   error envelope.
//! - [`render`]: validated transcript → prompt bytes for the generic Qwen
//!   family, satisfying the normative cases in
//!   `tests/fixtures/serve_render_fixtures_v1.json` (fixture-before-renderer).
//!
//! The serial HTTP/SSE loop lands in the next slice and consumes these.
#![allow(dead_code)] // consumed incrementally; the HTTP slice wires the rest

pub(crate) mod items;
pub(crate) mod render;
