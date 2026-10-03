//! Durable execution history, independent of sockets and model ownership.
//!
//! The CPU control plane accepts validated requests here. A CPU artifact writer
//! commits bounded batches; the model owner must not call disk-writing methods.
//! Recovery interrupts unfinished work instead of resubmitting inference.
//!
//! Layout: one immutable request, one append-only JSONL result, and one atomic
//! status/watermark snapshot per execution. Readers use committed watermarks;
//! no result body or full prompt history is retained in the in-memory index.
//! A bounded last-user-message excerpt is derived during acceptance/recovery.
//! Limits bound retained job count and logical file bytes, not filesystem block
//! allocation or a guarantee against ENOSPC. History is never auto-evicted.
//!
//! HTTP history is wired; native dispatch remains capability-gated. Execution
//! integration must accept before enqueueing, use bounded CPU artifact writing,
//! and map uncertain
//! storage failures to HTTP 500 (not a definitive pre-admission 503). A failed
//! status publication fences writes until restart; history/results retain the
//! last committed view while individual status requests report the failure.

mod preview;
pub(crate) mod state;
pub(crate) mod store;

#[cfg(test)]
mod tests;
