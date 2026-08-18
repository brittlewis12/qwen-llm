//! Process-wide tracing subscriber shared by every `qwen-cli` binary.
//!
//! # Design
//!
//! Historically the CLI produced two kinds of stderr output:
//!
//! * **Structured tracing events** (very few today — currently only a couple
//!   of `INFO`s in `qwen_llm::runtime`), formatted by
//!   `tracing_subscriber::fmt`'s default `Full` layout with ANSI colour when
//!   stderr looks like a terminal.
//! * **Bare diagnostic lines** emitted via `eprintln!` with prefixes such as
//!   `metal:`, `[metal-load]`, `[metal-load-ledger]`, `[metal-gguf-*]`,
//!   `[runtime-prefetch]`, and `stats:`. Roughly two dozen of the 81
//!   scripts under `scripts/profile/` (plus `scripts/bench/*_eval.py`)
//!   anchor `line.startswith(...)` or `re.fullmatch` against the exact
//!   line body. They ignored `RUST_LOG` completely.
//!
//! The `eprintln!` diagnostics are being migrated to
//! `tracing::info!(target: "qwen_diag", ...)`. To avoid breaking the
//! downstream scripts on the same release, this subscriber installs a
//! custom `FormatEvent` that emits `qwen_diag` events as *bare* message
//! bodies (no timestamp, level, or target prefix, no ANSI), while all other
//! events go through the default `Full` formatter unchanged. Users who want
//! to silence the diagnostics can now do `RUST_LOG=qwen_diag=off`, which
//! the previous `eprintln!` sites did not honour.
//!
//! ANSI colour is auto-disabled when stderr is not a TTY, when
//! `TERM=dumb`, or when `NO_COLOR` is set to a non-empty value, per the
//! conventions at <https://no-color.org>. This lets `qwen … 2>capture.log`
//! and `qwen … 2>&1 | tee` produce clean text without the caller having to
//! set `NO_COLOR=1` manually.
//!
//! `RUST_LOG` is parsed lossily (`from_env_lossy`) so a typo in a directive
//! logs a warning to stderr and falls back to the default filter rather
//! than silently disabling all diagnostics.

use std::io::IsTerminal;

use tracing::{Event, Subscriber};
use tracing_subscriber::EnvFilter;
use tracing_subscriber::filter::LevelFilter;
use tracing_subscriber::fmt::FmtContext;
use tracing_subscriber::fmt::format::{self, FormatEvent, FormatFields, Writer};
use tracing_subscriber::fmt::time::SystemTime;
use tracing_subscriber::registry::LookupSpan;

/// The tracing target used for bare-body diagnostic events. Emitters must
/// pass `target: "qwen_diag"` explicitly:
///
/// ```ignore
/// tracing::info!(
///     target: "qwen_diag",
///     "[metal-load-ledger] source={source} direct_copy={copied}",
/// );
/// ```
///
/// The literal `"qwen_diag"` is duplicated at emit sites rather than
/// referenced from this constant so that `qwen-llm` (the library that owns
/// most of the emit sites) does not need to depend on `qwen-cli`.
///
/// # Message-only contract
///
/// The renderer uses `DefaultFields` for the event body, which prints the
/// `message` field followed by any structured fields as ` key=value`. To
/// keep the on-wire byte stream stable for anchored downstream parsers,
/// emitters **must** pass a fully-formatted message and **must not** add
/// extra `key = value` fields. Encode every variable into the format
/// string itself:
///
/// ```ignore
/// tracing::info!(target: "qwen_diag", "[metal-load-ledger] source={source}");
/// // NOT: tracing::info!(target: "qwen_diag", source, "[metal-load-ledger]");
/// ```
pub const DIAG_TARGET: &str = "qwen_diag";

/// `FormatEvent` impl that emits events with `target = "qwen_diag"` as bare
/// bodies and delegates everything else to the default `Full` formatter.
struct DiagAwareFormat {
    default: format::Format<format::Full, SystemTime>,
}

impl<S, N> FormatEvent<S, N> for DiagAwareFormat
where
    S: Subscriber + for<'a> LookupSpan<'a>,
    N: for<'a> FormatFields<'a> + 'static,
{
    fn format_event(
        &self,
        ctx: &FmtContext<'_, S, N>,
        mut writer: Writer<'_>,
        event: &Event<'_>,
    ) -> std::fmt::Result {
        if event.metadata().target() == DIAG_TARGET {
            ctx.field_format().format_fields(writer.by_ref(), event)?;
            writeln!(writer)
        } else {
            self.default.format_event(ctx, writer, event)
        }
    }
}

/// Decide whether ANSI escapes are appropriate for stderr right now.
///
/// Returns `true` only when *all* of the following hold:
/// * stderr is an interactive terminal (`IsTerminal::is_terminal`);
/// * `TERM` is unset or is anything other than `dumb`;
/// * `NO_COLOR` is unset or empty (per <https://no-color.org>).
fn ansi_enabled_for_stderr() -> bool {
    if !std::io::stderr().is_terminal() {
        return false;
    }
    if let Some(term) = std::env::var_os("TERM")
        && term == "dumb"
    {
        return false;
    }
    if let Some(value) = std::env::var_os("NO_COLOR")
        && !value.is_empty()
    {
        return false;
    }
    true
}

/// Install the workspace's default tracing subscriber for the current
/// process. Call this exactly once, as early as possible in `main`.
///
/// The subscriber writes to `stderr`, uses ANSI colour only for
/// interactive terminals, and honours `RUST_LOG` (parsed lossily) with an
/// `info` fallback.
///
/// # Panics
///
/// Panics if a global default subscriber has already been installed for
/// the process (tracing's `try_init` failure). This is intentional: two
/// binaries in the same process would double-log.
pub fn install_default_subscriber() {
    let ansi = ansi_enabled_for_stderr();
    let default = format::Format::default().with_ansi(ansi);
    let filter = EnvFilter::builder()
        .with_default_directive(LevelFilter::INFO.into())
        .from_env_lossy();
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        // `.with_ansi(ansi)` on the *builder* is what propagates the
        // setting to `DefaultFields` (which styles field names / `=` /
        // non-message values on its own, independent of the event format
        // above). Without this call, a `qwen_diag` event that violates
        // the message-only contract would leak ANSI escapes into the
        // bare-body branch even on a redirected stderr. Belt-and-braces:
        // the event format above already got `with_ansi(ansi)` too.
        .with_ansi(ansi)
        .event_format(DiagAwareFormat { default })
        .with_env_filter(filter)
        .init();
}

#[cfg(test)]
mod tests {
    //! Byte-format guarantees for the `qwen_diag` render path.
    //!
    //! These tests are the actual contract that downstream Python parsers
    //! anchor against. If they break, so do the profile scripts under
    //! `scripts/profile/` and `scripts/bench/*_eval.py`.

    use std::io;
    use std::sync::{Arc, Mutex};

    use tracing::subscriber::with_default;
    use tracing_subscriber::EnvFilter;
    use tracing_subscriber::fmt::MakeWriter;
    use tracing_subscriber::fmt::format;

    use super::{DIAG_TARGET, DiagAwareFormat};

    /// `MakeWriter` that accumulates every write into a shared `Vec<u8>`.
    #[derive(Clone)]
    struct SharedBuffer(Arc<Mutex<Vec<u8>>>);

    impl SharedBuffer {
        fn new() -> Self {
            Self(Arc::new(Mutex::new(Vec::new())))
        }
        fn take(&self) -> String {
            let bytes = std::mem::take(&mut *self.0.lock().unwrap());
            String::from_utf8(bytes).expect("subscriber output must be UTF-8")
        }
    }

    impl<'a> MakeWriter<'a> for SharedBuffer {
        type Writer = SharedBufferWriter;
        fn make_writer(&'a self) -> Self::Writer {
            SharedBufferWriter(self.0.clone())
        }
    }

    struct SharedBufferWriter(Arc<Mutex<Vec<u8>>>);

    impl io::Write for SharedBufferWriter {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(buf);
            Ok(buf.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    fn subscribe<F: FnOnce()>(buffer: SharedBuffer, body: F) {
        // Mirror `install_default_subscriber` when stderr is a captured
        // (non-TTY) sink: ANSI off at both the event format *and* the
        // subscriber builder (which propagates to `DefaultFields`).
        let default = format::Format::default().with_ansi(false);
        let subscriber = tracing_subscriber::fmt()
            .with_writer(buffer)
            .with_ansi(false)
            .event_format(DiagAwareFormat { default })
            .with_env_filter(EnvFilter::new("info"))
            .finish();
        with_default(subscriber, body);
    }

    #[test]
    fn diag_target_renders_message_verbatim_with_trailing_newline() {
        assert_eq!(DIAG_TARGET, "qwen_diag");
        let buffer = SharedBuffer::new();
        subscribe(buffer.clone(), || {
            tracing::info!(
                target: "qwen_diag",
                "[metal-load-ledger] source=42/64 direct_copy=42/64 direct_view=0/0",
            );
        });
        assert_eq!(
            buffer.take(),
            "[metal-load-ledger] source=42/64 direct_copy=42/64 direct_view=0/0\n",
        );
    }

    #[test]
    fn suppression_line_const_matches_v0625_python_literal() {
        // Closes the Rust↔Python drift hole: if either
        // `AUTHENTICATED_DISPOSABLE_AUTO_A3B_PREAD_SUPPRESSION_LINE` in
        // qwen-llm or the `SUPPRESSION_LINE` literal in
        // `scripts/profile/v0625_a3b_prefetch_suppression_confirmation.py`
        // changes without the other, this test fails instead of the
        // profile script silently miscounting.
        //
        // v0625 is the only script with the full-line literal; the four
        // other consumers (v0651, v0652, v0626, v0628) anchor
        // `startswith("[runtime-prefetch]")` which the const's prefix
        // already covers.
        use qwen_llm::runtime::AUTHENTICATED_DISPOSABLE_AUTO_A3B_PREAD_SUPPRESSION_LINE as LINE;
        let script = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../scripts/profile/v0625_a3b_prefetch_suppression_confirmation.py");
        let text = std::fs::read_to_string(&script)
            .unwrap_or_else(|error| panic!("read {}: {error}", script.display()));
        // Collapse Python implicit string concatenation of the form
        // `"…" <whitespace> "…"` into a single logical string, so a
        // multi-line `SUPPRESSION_LINE = ("…" "…" "…")` literal is
        // searchable as a single substring.
        let normalized = collapse_python_adjacent_string_literals(&text);
        assert!(
            normalized.contains(LINE),
            "v0625 SUPPRESSION_LINE drifted from Rust const:\n  Rust: {LINE:?}\n  script: {}",
            script.display(),
        );
    }

    /// Collapse Python implicit string-literal concatenation
    /// (`"a" "b" "c"` -> `"abc"`) so a multi-line
    /// `SUPPRESSION_LINE = ("…" "…" "…")` becomes a single flat literal
    /// substring, searchable by the drift-protection test above.
    ///
    /// Does not handle escape sequences: the strings we anchor against
    /// contain none, and correct escape handling would balloon this
    /// helper beyond the value it provides.
    fn collapse_python_adjacent_string_literals(text: &str) -> String {
        let bytes = text.as_bytes();
        let mut out = String::with_capacity(text.len());
        let mut i = 0;
        while i < bytes.len() {
            if bytes[i] != b'"' {
                out.push(bytes[i] as char);
                i += 1;
                continue;
            }
            // Scan one or more adjacent `"…"` literals and join their
            // contents into a single logical string.
            let mut joined = String::new();
            loop {
                debug_assert!(bytes[i] == b'"');
                let content_start = i + 1;
                let mut end = content_start;
                while end < bytes.len() && bytes[end] != b'"' {
                    end += 1;
                }
                if end >= bytes.len() {
                    // Unterminated — emit an opening quote, the joined
                    // material so far, and the raw tail; then stop.
                    out.push('"');
                    out.push_str(&joined);
                    out.push_str(&text[content_start..]);
                    return out;
                }
                joined.push_str(&text[content_start..end]);
                i = end + 1;
                // Skip whitespace to look for an adjacent literal.
                let mut j = i;
                while j < bytes.len() && (bytes[j] as char).is_whitespace() {
                    j += 1;
                }
                if j < bytes.len() && bytes[j] == b'"' {
                    i = j;
                } else {
                    break;
                }
            }
            out.push('"');
            out.push_str(&joined);
            out.push('"');
        }
        out
    }

    #[test]
    fn collapse_python_adjacent_string_literals_joins_multiline_concat() {
        // Meta-test for the helper above — if this ever regresses, the
        // drift-protection test above becomes a false positive/negative.
        let src = "SUPPRESSION_LINE = (\n    \"foo \"\n    \"bar \"\n    \"baz\"\n)\n";
        let out = collapse_python_adjacent_string_literals(src);
        assert!(
            out.contains("\"foo bar baz\""),
            "expected joined literal in output: {out:?}",
        );
        // Standalone literals must survive unchanged.
        let standalone = "print(\"hello\")\n";
        assert_eq!(
            collapse_python_adjacent_string_literals(standalone),
            standalone,
        );
    }

    #[test]
    fn diag_target_field_bearing_event_stays_ansi_free_when_ansi_off() {
        // Doubles as the regression pin for the v4 ANSI-propagation fix
        // and the upgrade-alarm for `DefaultFields`.
        //
        // Regression: `DefaultFields::record_debug` styles field names /
        // `=` / non-message field values with italic + dim ANSI when the
        // *builder's* `is_ansi` is true. Without propagating
        // `.with_ansi(false)` to the builder on non-TTY stderr, a
        // contract-violating field-bearing event would leak
        // `msg \x1b[3mkey\x1b[0m\x1b[2m=\x1b[0m value\n` into a captured
        // file even though the event format was told ANSI-off. The
        // `subscribe` helper above mirrors the production fix (ANSI off
        // at every layer) — this test verifies that (a) no ESC bytes
        // reach the writer and (b) the exact byte shape is
        // `"msg key=value\n"`. That shape is not semver-guaranteed by
        // `tracing-subscriber`; if it changes, this test fires *before*
        // a profile script does.
        //
        // On a real TTY (ANSI on at both layers), `DefaultFields` does
        // legitimately style fields — that is not a leak, it is the
        // expected behaviour of an interactive terminal. Message-only
        // events on a TTY are still bare (see
        // `diag_target_stays_bare_when_ansi_is_enabled_globally`).
        let buffer = SharedBuffer::new();
        subscribe(buffer.clone(), || {
            tracing::info!(
                target: "qwen_diag",
                bytes = 3u64,
                "should-be-message-only",
            );
        });
        let rendered = buffer.take();
        assert!(
            !rendered.contains('\x1b'),
            "field-bearing qwen_diag event leaked ESC on non-TTY: {rendered:?}",
        );
        assert_eq!(rendered, "should-be-message-only bytes=3\n");
    }

    #[test]
    fn diag_target_stays_bare_when_ansi_is_enabled_globally() {
        // Mirrors the CLI subscriber's TTY branch: ANSI on at every
        // layer (event format, subscriber builder). Message-only events
        // must still render with no `0x1b` byte anywhere — this is the
        // guarantee that keeps interactive `qwen …` runs from polluting a
        // captured `2>&1 | tee` transcript with escape codes.
        let buffer = SharedBuffer::new();
        let default = format::Format::default().with_ansi(true);
        let subscriber = tracing_subscriber::fmt()
            .with_writer(buffer.clone())
            .with_ansi(true)
            .event_format(DiagAwareFormat { default })
            .with_env_filter(EnvFilter::new("info"))
            .finish();
        with_default(subscriber, || {
            tracing::info!(target: "qwen_diag", "[metal-load-ledger] source=1 direct_copy=1");
        });
        let bytes = buffer.take().into_bytes();
        assert!(
            !bytes.contains(&0x1b),
            "qwen_diag line contained ESC (0x1b): {:?}",
            String::from_utf8_lossy(&bytes),
        );
        assert_eq!(
            bytes,
            b"[metal-load-ledger] source=1 direct_copy=1\n".to_vec(),
        );
    }

    #[test]
    fn diag_target_stays_bare_inside_entered_span_with_fields() {
        // A well-meaning refactor that adds span context (fields, target
        // hierarchy, ancestor chain) to the bare branch would break every
        // anchored script. Pin the invariant: the rendered line depends
        // only on the event's own message.
        let buffer = SharedBuffer::new();
        subscribe(buffer.clone(), || {
            let span = tracing::info_span!("outer", request_id = 7u64, stage = "load",);
            let _guard = span.enter();
            tracing::info!(
                target: "qwen_diag",
                "[metal-gguf-owned] windows=1 window_bytes=1024",
            );
        });
        assert_eq!(
            buffer.take(),
            "[metal-gguf-owned] windows=1 window_bytes=1024\n",
        );
    }

    #[test]
    fn diag_target_emits_warn_body_without_level_prefix() {
        // Escalating a `qwen_diag` line to `warn!` must not inject a
        // `WARN` badge or any prefix — the point is bare bodies so that
        // profile-script anchors keep matching. Level survives only for
        // `RUST_LOG` filtering, not for human presentation.
        let buffer = SharedBuffer::new();
        subscribe(buffer.clone(), || {
            tracing::warn!(
                target: "qwen_diag",
                "metal: waiting for process lease /tmp/lease (owner)",
            );
        });
        assert_eq!(
            buffer.take(),
            "metal: waiting for process lease /tmp/lease (owner)\n",
        );
    }

    #[test]
    fn non_diag_target_uses_default_full_formatter() {
        // Plain `tracing::warn!` (module-path target) must go through the
        // default `Full` formatter, so operators see the timestamp, the
        // `WARN` badge, and the module path — this is why the poison-gate
        // bypass warning stayed off `qwen_diag`.
        let buffer = SharedBuffer::new();
        subscribe(buffer.clone(), || {
            tracing::warn!(target: "some_module", "poison gate bypassed");
        });
        let rendered = buffer.take();
        assert!(
            rendered.contains(" WARN "),
            "expected WARN badge in {rendered:?}"
        );
        assert!(
            rendered.contains("some_module"),
            "expected target in {rendered:?}",
        );
        assert!(
            rendered.contains("poison gate bypassed"),
            "expected message in {rendered:?}",
        );
        assert!(rendered.ends_with('\n'), "expected trailing newline");
    }

    #[test]
    fn diag_target_off_filters_out_diag_events() {
        // `RUST_LOG=qwen_diag=off` (or equivalent directive) must silence
        // every `qwen_diag` event without affecting other targets. This is
        // the property that makes the migration a net win over the old
        // `eprintln!` sites — those could not be silenced.
        let buffer = SharedBuffer::new();
        let default = format::Format::default().with_ansi(false);
        let subscriber = tracing_subscriber::fmt()
            .with_writer(buffer.clone())
            .event_format(DiagAwareFormat { default })
            .with_env_filter(EnvFilter::new("info,qwen_diag=off"))
            .finish();
        with_default(subscriber, || {
            tracing::info!(target: "qwen_diag", "should not appear");
            tracing::warn!(target: "other_target", "should appear");
        });
        let rendered = buffer.take();
        assert!(
            !rendered.contains("should not appear"),
            "diag leaked: {rendered:?}"
        );
        assert!(
            rendered.contains("should appear"),
            "other target missing: {rendered:?}"
        );
    }
}
