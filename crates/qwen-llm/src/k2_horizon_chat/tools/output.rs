//! Post-reasoning tool span buffering. UTF-8 assembly and reasoning partitioning
//! belong to the caller; raw or reasoning text must never be routed here.
use super::parse::TOOL_BLOCK_OPEN;
use super::*;
use std::borrow::Cow;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ToolOutputEnd {
    Stop,
    TokenLimit,
}

#[derive(Debug, PartialEq)]
pub struct ToolOutputFinish {
    pub visible_tail: String,
    pub calls: Vec<ToolCall>,
    pub incomplete_tool_block: bool,
}

pub struct ToolOutputStream<'a> {
    format: ToolCallFormat,
    definitions: Cow<'a, [Value]>,
    max_tool_bytes: usize,
    pending: String,
    block: Option<String>,
    failed: bool,
}

impl<'a> ToolOutputStream<'a> {
    /// The caller supplies a byte budget from request/output resource admission,
    /// not a hidden model-context or generated-token limit.
    pub fn new(format: ToolCallFormat, definitions: &'a [Value], max_tool_bytes: usize) -> Self {
        Self {
            format,
            definitions: Cow::Borrowed(definitions),
            max_tool_bytes,
            pending: String::new(),
            block: None,
            failed: false,
        }
    }

    /// Return ordinary visible text. Tool bytes never leave this stream as text
    /// or partial calls. Errors poison it; dropping it aborts without publication.
    pub fn push_visible(&mut self, text: &str) -> Result<String> {
        if self.failed {
            return Err(error("tool output stream is failed"));
        }
        if self.block.is_some() {
            self.append_block(text)?;
            return Ok(String::new());
        }
        let mut pending = std::mem::take(&mut self.pending);
        if !pending.is_empty() {
            let suffix = &TOOL_BLOCK_OPEN[pending.len()..];
            if text.starts_with(suffix) {
                self.begin_block(&pending, text)?;
                return Ok(String::new());
            }
            if suffix.starts_with(text) {
                pending.push_str(text);
                self.pending = pending;
                return Ok(String::new());
            }
        }
        if let Some(index) = text.find(TOOL_BLOCK_OPEN) {
            self.begin_block("", &text[index..])?;
            pending.push_str(&text[..index]);
            return Ok(pending);
        }
        let keep = (1..TOOL_BLOCK_OPEN.len())
            .rev()
            .find(|&n| text.ends_with(&TOOL_BLOCK_OPEN[..n]))
            .unwrap_or(0);
        let safe = text.len() - keep;
        self.pending.push_str(&text[safe..]);
        pending.push_str(&text[..safe]);
        Ok(pending)
    }
    fn begin_block(&mut self, prefix: &str, text: &str) -> Result<()> {
        let Some(bytes) = prefix
            .len()
            .checked_add(text.len())
            .filter(|&n| n <= self.max_tool_bytes)
        else {
            self.failed = true;
            return Err(error("tool output exceeds the admitted byte budget"));
        };
        let mut block = String::with_capacity(bytes);
        block.push_str(prefix);
        block.push_str(text);
        self.block = Some(block);
        Ok(())
    }
    fn append_block(&mut self, text: &str) -> Result<()> {
        let block = self.block.as_mut().unwrap();
        if block
            .len()
            .checked_add(text.len())
            .is_none_or(|n| n > self.max_tool_bytes)
        {
            self.failed = true;
            block.clear();
            return Err(error("tool output exceeds the admitted byte budget"));
        }
        block.push_str(text);
        Ok(())
    }

    /// Calls are published together only after the entire terminal span is known.
    /// A closed valid block can yield complete calls at a token-budget boundary;
    /// the caller must still mark the overall response as budget-incomplete.
    pub fn finish(self, end: ToolOutputEnd) -> Result<ToolOutputFinish> {
        if self.failed {
            return Err(error("tool output stream is failed"));
        }
        let Some(block) = self.block else {
            return Ok(ToolOutputFinish {
                visible_tail: self.pending,
                calls: Vec::new(),
                incomplete_tool_block: false,
            });
        };
        let parsed = match parse_tool_calls(&block, self.format, &self.definitions) {
            Err(error) if self.format == ToolCallFormat::XmlTyped => {
                // The released model can omit type labels despite the prompt.
                // Accept only a complete strict untyped block with unambiguous
                // schema-bound types, never a prefix or contradictory labels.
                match parse_tool_calls(&block, ToolCallFormat::Xml, &self.definitions) {
                    Ok(complete @ ParsedToolBlock::Complete(_)) => complete,
                    _ => return Err(error),
                }
            }
            other => other?,
        };
        match parsed {
            ParsedToolBlock::Complete(calls) => Ok(ToolOutputFinish {
                visible_tail: String::new(),
                calls,
                incomplete_tool_block: false,
            }),
            ParsedToolBlock::Incomplete if end == ToolOutputEnd::TokenLimit => {
                Ok(ToolOutputFinish {
                    visible_tail: String::new(),
                    calls: Vec::new(),
                    incomplete_tool_block: true,
                })
            }
            ParsedToolBlock::Incomplete => Err(error("stop before tool block closure")),
        }
    }
}

impl ToolOutputStream<'static> {
    pub fn owned(format: ToolCallFormat, definitions: Vec<Value>, max_tool_bytes: usize) -> Self {
        Self {
            format,
            definitions: Cow::Owned(definitions),
            max_tool_bytes,
            pending: String::new(),
            block: None,
            failed: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn typed_generation_tolerates_only_complete_unambiguous_untyped_blocks() {
        let defs = vec![
            serde_json::json!({"name":"f","parameters":{"properties":{"x":{"type":"string"}}}}),
        ];
        let calls = vec![ToolCall {
            name: "f".into(),
            arguments: serde_json::json!({"x":"7"}).as_object().unwrap().clone(),
        }];
        let block = render_tool_calls(&calls, ToolCallFormat::Xml, &defs).unwrap();
        assert!(parse_tool_calls(&block, ToolCallFormat::XmlTyped, &defs).is_err());
        for end in [ToolOutputEnd::Stop, ToolOutputEnd::TokenLimit] {
            let mut stream = ToolOutputStream::new(ToolCallFormat::XmlTyped, &defs, block.len());
            stream.push_visible(&block).unwrap();
            assert_eq!(stream.finish(end).unwrap().calls, calls);
        }
        let typed = render_tool_calls(&calls, ToolCallFormat::XmlTyped, &defs).unwrap();
        for bad in [
            format!("{block}trailing"),
            block.replace("</ifm|tool_calls>", ""),
            typed.replace("<ifm|arg_type>string", "<ifm|arg_type>integer"),
        ] {
            for end in [ToolOutputEnd::Stop, ToolOutputEnd::TokenLimit] {
                let mut stream = ToolOutputStream::new(ToolCallFormat::XmlTyped, &defs, bad.len());
                stream.push_visible(&bad).unwrap();
                assert!(stream.finish(end).is_err());
            }
        }
        let partial = "<ifm|tool_calls><ifm|tool_call>f\n<ifm|arg_key>x</ifm|arg_key><ifm|arg_ty";
        let mut stream = ToolOutputStream::new(ToolCallFormat::XmlTyped, &defs, partial.len());
        stream.push_visible(partial).unwrap();
        let result = stream.finish(ToolOutputEnd::TokenLimit).unwrap();
        assert!(result.calls.is_empty() && result.incomplete_tool_block);
        let ambiguous = vec![
            serde_json::json!({"name":"f","parameters":{"properties":{"x":{"type":["string","integer"]}}}}),
        ];
        let mut stream = ToolOutputStream::new(ToolCallFormat::XmlTyped, &ambiguous, block.len());
        stream.push_visible(&block).unwrap();
        assert!(stream.finish(ToolOutputEnd::Stop).is_err());
    }

    #[test]
    fn literal_prefixes_flush_and_initial_tool_budget_is_checked_before_copying() {
        for prefix in 1..TOOL_BLOCK_OPEN.len() {
            for suffix in (0u8..=127)
                .map(|b| char::from(b).to_string())
                .chain(["\u{1f389}".into()])
            {
                let expected = format!("{}{suffix}", &TOOL_BLOCK_OPEN[..prefix]);
                if expected == TOOL_BLOCK_OPEN {
                    continue;
                }
                let mut stream = ToolOutputStream::new(ToolCallFormat::Json, &[], 0);
                let a = stream.push_visible(&TOOL_BLOCK_OPEN[..prefix]).unwrap();
                let b = stream.push_visible(&suffix).unwrap();
                let result = stream.finish(ToolOutputEnd::Stop).unwrap();
                assert_eq!(a + &b + &result.visible_tail, expected);
                assert!(result.calls.is_empty());
            }
        }
        let text = "literal <ifm|tool_x> \u{1f389} <ifm|tool_";
        for cut in (0..=text.len()).filter(|&n| text.is_char_boundary(n)) {
            let mut stream = ToolOutputStream::new(ToolCallFormat::Json, &[], 0);
            let a = stream.push_visible(&text[..cut]).unwrap();
            let b = stream.push_visible(&text[cut..]).unwrap();
            let result = stream.finish(ToolOutputEnd::Stop).unwrap();
            assert_eq!(a + &b + &result.visible_tail, text);
            assert!(result.calls.is_empty());
        }
        let huge = format!("{TOOL_BLOCK_OPEN}{}", "x".repeat(1024 * 1024));
        let mut stream = ToolOutputStream::new(ToolCallFormat::Json, &[], 64);
        assert!(stream.push_visible(&huge).is_err());
        assert!(stream.block.is_none());
        assert_eq!(stream.pending.capacity(), 0);
    }

    #[test]
    fn tool_stream_publishes_only_complete_terminal_calls_for_every_text_split() {
        let defs = vec![
            serde_json::json!({"name":"f","parameters":{"properties":{"x":{"type":"string"}}}}),
        ];
        let calls = vec![ToolCall {
            name: "f".into(),
            arguments: serde_json::json!({"x":"hi \u{1f389} </ifm|tool_calls>"})
                .as_object()
                .unwrap()
                .clone(),
        }];
        let block = render_tool_calls(&calls, ToolCallFormat::Json, &defs).unwrap();
        let all = format!("Checking \u{2192} {block}");
        for split in (0..=all.len()).filter(|&n| all.is_char_boundary(n)) {
            for end in [ToolOutputEnd::Stop, ToolOutputEnd::TokenLimit] {
                let mut stream = ToolOutputStream::new(ToolCallFormat::Json, &defs, block.len());
                let a = stream.push_visible(&all[..split]).unwrap();
                let b = stream.push_visible(&all[split..]).unwrap();
                assert_eq!(a + &b, "Checking \u{2192} ");
                let result = stream.finish(end).unwrap();
                assert_eq!(result.calls, calls);
                assert!(!result.incomplete_tool_block);
                assert!(result.visible_tail.is_empty());
            }
        }
    }

    #[test]
    fn incomplete_malformed_aborted_and_over_budget_streams_never_publish_calls() {
        let defs = vec![serde_json::json!({"name":"f","parameters":{}})];
        let partial = "<ifm|tool_calls><ifm|tool_call>{\"name\":\"f\",\"arguments\":{";
        let mut stream = ToolOutputStream::new(ToolCallFormat::Json, &defs, partial.len());
        assert!(stream.push_visible(partial).unwrap().is_empty());
        let incomplete = stream.finish(ToolOutputEnd::TokenLimit).unwrap();
        assert!(incomplete.calls.is_empty() && incomplete.incomplete_tool_block);
        let mut stream = ToolOutputStream::new(ToolCallFormat::Json, &defs, partial.len());
        stream.push_visible(partial).unwrap();
        assert!(stream.finish(ToolOutputEnd::Stop).is_err());
        let mut stream = ToolOutputStream::new(ToolCallFormat::Json, &defs, partial.len() - 1);
        assert!(stream.push_visible(partial).is_err());
        assert!(stream.push_visible("").is_err());
        assert!(stream.finish(ToolOutputEnd::TokenLimit).is_err());
        let block = render_tool_calls(
            &[ToolCall {
                name: "f".into(),
                arguments: Map::new(),
            }],
            ToolCallFormat::Json,
            &defs,
        )
        .unwrap();
        let mut stream = ToolOutputStream::new(ToolCallFormat::Json, &defs, block.len());
        assert!(stream.push_visible(&block).unwrap().is_empty());
        drop(stream); // Abort has no publication path, even with a buffered complete call.
        let mut stream = ToolOutputStream::new(ToolCallFormat::Json, &defs, block.len() + 10);
        stream.push_visible(&format!("{block}answer")).unwrap();
        assert!(stream.finish(ToolOutputEnd::Stop).is_err());
    }
}
