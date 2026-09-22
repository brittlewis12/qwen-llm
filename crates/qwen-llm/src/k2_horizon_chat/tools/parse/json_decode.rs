//! Build containers directly: serde Value's arbitrary-precision visitor treats
//! a literal "$serde_json::private::Number" object key as an internal number tag.
//! Serde still owns scalar JSON syntax, escapes and numeric parsing.
use super::*;

pub(super) fn prefix(text: &str) -> ParseResult<(Value, usize)> {
    let mut parser = Json { text, at: 0 };
    let value = parser.value(0)?;
    Ok((value, parser.at))
}
pub(in crate::k2_horizon_chat::tools) fn complete(text: &str) -> Result<Value> {
    match prefix(text) {
        Ok((value, offset)) if text[offset..].trim_matches(json_space).is_empty() => Ok(value),
        Ok(_) => Err(error("trailing tool JSON content")),
        Err(Failure::Incomplete) => Err(error("incomplete tool JSON value")),
        Err(Failure::Malformed(e)) => Err(e),
    }
}
fn json_space(c: char) -> bool {
    matches!(c, ' ' | '\t' | '\r' | '\n')
}
struct Json<'a> {
    text: &'a str,
    at: usize,
}
impl Json<'_> {
    fn whitespace(&mut self) {
        while self
            .text
            .as_bytes()
            .get(self.at)
            .is_some_and(|&c| matches!(c, b' ' | b'\t' | b'\r' | b'\n'))
        {
            self.at += 1;
        }
    }
    fn take(&mut self, byte: u8) -> bool {
        self.whitespace();
        if self.text.as_bytes().get(self.at) == Some(&byte) {
            self.at += 1;
            true
        } else {
            false
        }
    }
    fn expect(&mut self, byte: u8) -> ParseResult<()> {
        if self.take(byte) {
            Ok(())
        } else if self.at == self.text.len() {
            Err(Failure::Incomplete)
        } else {
            Err(malformed("invalid tool JSON container syntax"))
        }
    }
    fn string(&mut self) -> ParseResult<String> {
        self.whitespace();
        let mut stream =
            serde_json::Deserializer::from_str(&self.text[self.at..]).into_iter::<String>();
        match stream.next() {
            Some(Ok(value)) => {
                self.at += stream.byte_offset();
                Ok(value)
            }
            Some(Err(e)) if e.is_eof() => Err(Failure::Incomplete),
            Some(Err(e)) => Err(malformed(format!("invalid tool JSON string: {e}"))),
            None => Err(Failure::Incomplete),
        }
    }
    fn value(&mut self, depth: usize) -> ParseResult<Value> {
        if depth >= 128 {
            return Err(malformed("tool JSON exceeds nesting safety limit"));
        }
        self.whitespace();
        match self.text.as_bytes().get(self.at).copied() {
            None => Err(Failure::Incomplete),
            Some(b'"') => self.string().map(Value::String),
            Some(b'{') => {
                self.at += 1;
                let mut map = Map::new();
                if self.take(b'}') {
                    return Ok(Value::Object(map));
                }
                loop {
                    let key = self.string()?;
                    self.expect(b':')?;
                    if map.contains_key(&key) {
                        return Err(malformed("duplicate tool JSON key"));
                    }
                    map.insert(key, self.value(depth + 1)?);
                    if self.take(b'}') {
                        return Ok(Value::Object(map));
                    }
                    self.expect(b',')?;
                }
            }
            Some(b'[') => {
                self.at += 1;
                let mut items = Vec::new();
                if self.take(b']') {
                    return Ok(Value::Array(items));
                }
                loop {
                    items.push(self.value(depth + 1)?);
                    if self.take(b']') {
                        return Ok(Value::Array(items));
                    }
                    self.expect(b',')?;
                }
            }
            Some(first @ (b't' | b'f' | b'n')) => {
                let (literal, value) = match first {
                    b't' => ("true", Value::Bool(true)),
                    b'f' => ("false", Value::Bool(false)),
                    _ => ("null", Value::Null),
                };
                let rest = &self.text[self.at..];
                if rest.starts_with(literal) {
                    self.at += literal.len();
                    Ok(value)
                } else if literal.starts_with(rest) {
                    Err(Failure::Incomplete)
                } else {
                    Err(malformed("invalid tool JSON literal"))
                }
            }
            Some(b'-' | b'0'..=b'9') => {
                let start = self.at;
                while self
                    .text
                    .as_bytes()
                    .get(self.at)
                    .is_some_and(|c| matches!(c, b'0'..=b'9' | b'-' | b'+' | b'.' | b'e' | b'E'))
                {
                    self.at += 1;
                }
                match serde_json::from_str::<serde_json::Number>(&self.text[start..self.at]) {
                    Ok(number) => Ok(Value::Number(number)),
                    Err(e) if e.is_eof() && self.at == self.text.len() => Err(Failure::Incomplete),
                    Err(e) => Err(malformed(format!("invalid tool JSON number: {e}"))),
                }
            }
            _ => Err(malformed("invalid tool JSON value")),
        }
    }
}
