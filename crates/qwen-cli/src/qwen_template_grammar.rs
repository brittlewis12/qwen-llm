//! Recognition, not evaluation: accept one complete supported template program.
//! Only whitespace between expression tokens is insignificant. Output text,
//! string spelling, control flow and Jinja whitespace-control remain exact.

const REFERENCE: &str =
    include_str!("../tests/fixtures/templates/qwen_xml_preserve_chat_template.jinja");
const MAX_BYTES: usize = 64 * 1024;
const MAX_TOKENS: usize = 16 * 1024;

pub(super) fn is_xml_preserve(source: &str) -> bool {
    tokens(source).is_some_and(|actual| Some(actual) == tokens(REFERENCE))
}

#[derive(Debug, PartialEq, Eq)]
enum Token<'a> {
    Text(&'a str),
    Syntax(&'a str),
}

fn tokens(source: &str) -> Option<Vec<Token<'_>>> {
    if source.len() > MAX_BYTES {
        return None;
    }
    let bytes = source.as_bytes();
    let mut out = Vec::new();
    let mut at = 0;
    while at < bytes.len() {
        let start = at;
        while at < bytes.len()
            && !source[at..].starts_with("{{")
            && !source[at..].starts_with("{%")
            && !source[at..].starts_with("{#")
        {
            at += source[at..].chars().next()?.len_utf8();
        }
        if start != at {
            out.push(Token::Text(&source[start..at]));
        }
        if at == bytes.len() {
            break;
        }
        // Comments and raw blocks are not part of this supported program.
        let close = match &source[at..at + 2] {
            "{{" => "}}",
            "{%" => "%}",
            _ => return None,
        };
        let start = at;
        at += 2;
        if matches!(bytes.get(at), Some(b'-' | b'+')) {
            at += 1;
        }
        out.push(Token::Syntax(&source[start..at]));
        loop {
            while bytes.get(at).is_some_and(u8::is_ascii_whitespace) {
                at += 1;
            }
            let start = at;
            let tail = source.get(at..)?;
            if tail.starts_with(close)
                || (matches!(bytes.get(at), Some(b'-' | b'+'))
                    && source.get(at + 1..)?.starts_with(close))
            {
                at += if tail.starts_with(close) { 2 } else { 3 };
                out.push(Token::Syntax(&source[start..at]));
                break;
            }
            match *bytes.get(at)? {
                quote @ (b'\'' | b'"') => {
                    at += 1;
                    loop {
                        match *bytes.get(at)? {
                            b'\\' => at += 2,
                            b if b == quote => {
                                at += 1;
                                break;
                            }
                            _ => at += 1,
                        }
                    }
                }
                b if b.is_ascii_alphanumeric() || b == b'_' => {
                    at += 1;
                    while bytes
                        .get(at)
                        .is_some_and(|b| b.is_ascii_alphanumeric() || *b == b'_')
                    {
                        at += 1;
                    }
                }
                _ => {
                    let pair = bytes.get(at..at + 2);
                    at += if matches!(pair, Some(b"==" | b"!=" | b"<=" | b">=" | b"//" | b"**")) {
                        2
                    } else {
                        source[at..].chars().next()?.len_utf8()
                    };
                }
            }
            out.push(Token::Syntax(source.get(start..at)?));
            if out.len() > MAX_TOKENS {
                return None;
            }
        }
    }
    (out.len() <= MAX_TOKENS).then_some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn complete_program_only_with_expression_spacing_normalized() {
        assert!(is_xml_preserve(REFERENCE));
        assert!(is_xml_preserve(
            &REFERENCE.replace("namespace(value=0)", "namespace( value = 0 )")
        ));
        for changed in [
            REFERENCE.replace("enable_thinking is false", "enable_thinking is true"),
            REFERENCE.replace("reasoning_content +", "content +"),
            REFERENCE.replace("| string", "| tojson"),
            REFERENCE.replacen("{%-", "{%", 1),
            REFERENCE.replace("<think>", "<thought>"),
            REFERENCE.replace("==", "= ="),
            format!("{REFERENCE}{{% set extra = 1 %}}"),
            format!("{REFERENCE}arbitrary output"),
            format!("{{# comment #}}{REFERENCE}"),
            REFERENCE[..REFERENCE.len() - 5].to_owned(),
            "x".repeat(MAX_BYTES + 1),
        ] {
            assert!(!is_xml_preserve(&changed));
        }
        for old in [
            include_str!("../tests/fixtures/templates/qwen35_chat_template.jinja"),
            include_str!("../tests/fixtures/templates/qwen36_a3b_chat_template.jinja"),
            include_str!("../tests/fixtures/templates/qwen38_27b_chat_template.jinja"),
        ] {
            assert!(!is_xml_preserve(old));
        }
    }
}
