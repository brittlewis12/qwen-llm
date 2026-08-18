//! Shared tracing-subscriber install for the load-path examples.
//!
//! The library-side load path (`[metal-load-ledger]`, `[metal-gguf-*]`,
//! `[runtime-prefetch]`) now emits via `tracing::info!(target: "qwen_diag",
//! ...)` instead of `eprintln!`. Examples that exercise that path must
//! install a subscriber themselves — otherwise the events are dropped and
//! the example runs silently. This is *not* the CLI's custom
//! `DiagAwareFormat`: that formatter is bin-private to `qwen-cli` and its
//! byte-for-byte guarantee is only needed by the profile scripts. Here we
//! use the stock `Full` formatter with sane defaults so the examples stay
//! readable and pipe-safe.
//!
//! Consumed by `examples/load_spike.rs`, `examples/runtime_load_spike.rs`,
//! and `examples/first_byte_spike.rs` via `#[path = "support/diag_subscriber.rs"]`.
//!
//! # No `#[test]` here
//!
//! Do not add `#[cfg(test)] mod tests { … }` to this file. It is included
//! into example targets via `#[path]`; example binaries are built by
//! `cargo test` (to verify they compile) but their test harnesses are
//! never *executed*. Any `#[test]` fn here would silently never run — a
//! green suite that proves nothing. Keep test coverage for this
//! subscriber's behaviour in `crates/qwen-cli/src/tracing_init.rs`, which
//! is compiled as a real test target.

use std::io::IsTerminal;

/// Install a stock `tracing_subscriber::fmt` subscriber configured with:
///
/// * `writer = stderr` — never interleaves with the example's `println!`
///   report on stdout (this was the qwen-bench writer bug reused as
///   `tracing_subscriber::fmt::try_init()`'s stock default);
/// * `ansi` on iff stderr is a TTY and `TERM != "dumb"` and `NO_COLOR` is
///   unset/empty — so `cargo run --example foo 2> capture.log` does not
///   leak escape sequences into the captured file;
/// * `EnvFilter` from `RUST_LOG` (parsed lossily; `info` fallback) — so
///   developers can `RUST_LOG=qwen_diag=off` an example the same way they
///   would silence the CLI.
///
/// Uses `try_init` so a caller running the example under a harness that
/// pre-installs a subscriber does not panic. In normal `cargo run
/// --example` usage `try_init` succeeds because no other subscriber is
/// installed for the process.
pub fn install_example_diag_subscriber() {
    let ansi = std::io::stderr().is_terminal()
        && std::env::var_os("TERM").is_none_or(|term| term != "dumb")
        && std::env::var_os("NO_COLOR").is_none_or(|value| value.is_empty());
    let _ = tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_ansi(ansi)
        .with_env_filter(
            tracing_subscriber::EnvFilter::builder()
                .with_default_directive(tracing_subscriber::filter::LevelFilter::INFO.into())
                .from_env_lossy(),
        )
        .try_init();
}
