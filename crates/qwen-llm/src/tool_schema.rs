//! Family-neutral tool-calling helpers shared by chat renderers and output
//! parsers (K2 Horizon, GLM-5.3-Flash).
//!
//! - [`python_json`]: HF chat templates' `tojson`, which is Python
//!   `json.dumps(ensure_ascii=False)`: `", "`/`": "` separators and Python's
//!   float repr (`1.0`, `1e+20`, `1.5e-07`). Compact serde JSON differs.
//! - Outer-type masks of a function's argument schema ([`argument_kinds`],
//!   [`value_kind`]): `type` (name or list), `enum`, `const`,
//!   `anyOf`/`oneOf`/`allOf` and local `$ref` (`#/$defs/`, `#/definitions/`,
//!   with the sibling overlay templates present). Ranges, patterns,
//!   required fields and nested validation remain the tool consumer's.
//! - [`decode_json_prefix`] / [`decode_json`]: JSON decoding that builds
//!   containers directly and refuses duplicate keys. serde's
//!   arbitrary-precision visitor would read a literal
//!   `"$serde_json::private::Number"` key as an internal number tag and
//!   silently collapse duplicates.

use serde_json::Value;

pub const STRING: u8 = 1;
pub const INTEGER: u8 = 2;
pub const NUMBER: u8 = 4;
pub const BOOL: u8 = 8;
pub const NULL: u8 = 16;
pub const ARRAY: u8 = 32;
pub const OBJECT: u8 = 64;
pub const ANY: u8 = 127;

/// Python `json.dumps(value, ensure_ascii=False)`. Refuses non-finite
/// floating-point values, which JSON cannot carry.
pub fn python_json(value: &Value) -> Result<String, String> {
    let mut out = String::new();
    write(value, &mut out)?;
    Ok(out)
}

fn write(value: &Value, out: &mut String) -> Result<(), String> {
    match value {
        Value::Array(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                write(item, out)?;
            }
            out.push(']');
        }
        Value::Object(map) => {
            out.push('{');
            for (i, (key, value)) in map.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                out.push_str(&serde_json::to_string(key).map_err(|e| e.to_string())?);
                out.push_str(": ");
                write(value, out)?;
            }
            out.push('}');
        }
        Value::Number(number) => {
            let text = number.to_string();
            if text.contains(['.', 'e', 'E']) {
                let value = number.as_f64().filter(|v| v.is_finite()).ok_or_else(|| {
                    "tool JSON requires finite binary64 floating-point values".to_owned()
                })?;
                let scientific = format!("{value:e}");
                let (mantissa, exponent) = scientific.split_once('e').unwrap();
                let exponent: i32 = exponent.parse().unwrap();
                if !(-4..16).contains(&exponent) {
                    out.push_str(mantissa);
                    out.push('e');
                    out.push(if exponent < 0 { '-' } else { '+' });
                    out.push_str(&format!("{:02}", exponent.abs()));
                } else {
                    let fixed = value.to_string();
                    out.push_str(&fixed);
                    if !fixed.contains('.') {
                        out.push_str(".0");
                    }
                }
            } else {
                out.push_str(&text);
            }
        }
        _ => out.push_str(&serde_json::to_string(value).map_err(|e| e.to_string())?),
    }
    Ok(())
}

// JSON Schema integer membership is mathematical, unlike a Python-style
// lexical type label. Do not round a fractional decimal through f64 to decide it.
fn integral(number: &serde_json::Number) -> bool {
    let text = number.to_string();
    let (mantissa, exponent) = text.split_once(['e', 'E']).unwrap_or((&text, "0"));
    if mantissa
        .bytes()
        .filter(u8::is_ascii_digit)
        .all(|b| b == b'0')
    {
        return true;
    }
    let Ok(exponent) = exponent.parse::<i64>() else {
        return !exponent.starts_with('-');
    };
    let fraction = mantissa
        .split_once('.')
        .map_or(0, |(_, fraction)| fraction.len());
    let power = i128::from(exponent) - fraction as i128;
    let zeros = mantissa
        .bytes()
        .rev()
        .filter(u8::is_ascii_digit)
        .take_while(|&b| b == b'0')
        .count();
    power >= 0 || zeros as i128 >= -power
}

/// The outer type of a JSON value as a single mask bit.
pub fn value_kind(value: &Value) -> u8 {
    match value {
        Value::String(_) => STRING,
        Value::Bool(_) => BOOL,
        Value::Null => NULL,
        Value::Array(_) => ARRAY,
        Value::Object(_) => OBJECT,
        Value::Number(number) => {
            if integral(number) {
                INTEGER
            } else {
                NUMBER
            }
        }
    }
}

/// A JSON Schema type name as a mask; unknown names admit anything.
pub fn type_name_kinds(name: &str) -> u8 {
    match name {
        "string" => STRING,
        "integer" => INTEGER,
        "number" => INTEGER | NUMBER,
        "boolean" => BOOL,
        "null" => NULL,
        "array" => ARRAY,
        "object" => OBJECT,
        _ => ANY,
    }
}

/// The schema a local `$ref` names: `#/$defs/<k>` in `$defs`,
/// `#/definitions/<k>` in `definitions` (each in its own namespace), with
/// JSON Pointer escapes (`~1`, `~0`) decoded. Deeper pointers are not
/// resolved.
pub fn ref_target<'a>(parameters: &'a Value, reference: &str) -> Option<&'a Value> {
    let (namespace, key) = if let Some(key) = reference.strip_prefix("#/$defs/") {
        ("$defs", key)
    } else {
        ("definitions", reference.strip_prefix("#/definitions/")?)
    };
    if key.contains('/') {
        return None;
    }
    let key = key.replace("~1", "/").replace("~0", "~");
    parameters.get(namespace)?.get(key.as_str())
}

/// K2 Horizon's template-presentation convention, which its renderer and
/// argument typing share and which must not change without K2's own
/// qualification: a `#/$defs/<k>` or `#/definitions/<k>` reference looks
/// `<k>` up in `$defs` when that object exists, else in `definitions` (not by
/// the reference's own namespace), with no pointer unescaping, merged with
/// `spec`'s sibling keys (the overlay the template presents). GLM uses the
/// namespace-correct [`ref_target`] instead.
pub fn resolve_ref(parameters: &Value, spec: &Value) -> Option<Value> {
    let reference = spec["$ref"].as_str()?;
    let key = reference
        .strip_prefix("#/$defs/")
        .or_else(|| reference.strip_prefix("#/definitions/"))?;
    let defs = if parameters["$defs"].is_object() {
        &parameters["$defs"]
    } else {
        &parameters["definitions"]
    };
    let mut merged = defs.get(key)?.as_object()?.clone();
    for (key, value) in spec.as_object()? {
        if key != "$ref" {
            merged.insert(key.clone(), value.clone());
        }
    }
    Some(Value::Object(merged))
}

/// The outer types `spec` admits, resolving references against `root`.
pub fn schema_kinds(spec: &Value, root: &Value, active: &mut Vec<String>) -> Result<u8, String> {
    if active.len() >= 128 {
        return Err("tool argument schema exceeds reference safety limit".into());
    }
    if spec == &Value::Bool(false) {
        return Ok(0);
    }
    if !spec.is_object() {
        return Ok(ANY);
    }
    let mut mask = ANY;
    if let Some(r) = spec["$ref"].as_str()
        && !active.iter().any(|v| v == r)
        && let Some(target) = resolve_ref(root, spec)
    {
        active.push(r.into());
        // Interpret the same sibling overlay the native template presents,
        // not a different JSON Schema $ref validation dialect.
        let resolved = schema_kinds(&target, root, active)?;
        active.pop();
        return Ok(resolved);
    }
    if let Some(name) = spec["type"].as_str() {
        mask &= type_name_kinds(name);
    } else if let Some(types) = spec["type"].as_array() {
        mask &= types
            .iter()
            .fold(0, |m, v| m | v.as_str().map_or(ANY, type_name_kinds));
    }
    if let Some(values) = spec["enum"].as_array() {
        mask &= values.iter().fold(0, |m, v| m | value_kind(v));
    }
    if let Some(value) = spec.get("const") {
        mask &= value_kind(value);
    }
    for key in ["anyOf", "oneOf", "allOf"] {
        if let Some(variants) = spec[key].as_array() {
            let mut combined = if key == "allOf" { ANY } else { 0 };
            for variant in variants {
                // Nesting is guarded independently of reference depth.
                active.push(String::new());
                let child = schema_kinds(variant, root, active)?;
                active.pop();
                if key == "allOf" {
                    combined &= child;
                } else {
                    combined |= child;
                }
            }
            mask &= combined;
        }
    }
    Ok(mask)
}

/// The outer types a function's `parameters` admit for argument `key`
/// (`parameters` itself may be a `$ref`). A missing schema admits anything.
pub fn argument_kinds(parameters: &Value, key: &str) -> Result<u8, String> {
    let presented = resolve_ref(parameters, parameters).unwrap_or_else(|| parameters.clone());
    schema_kinds(&presented["properties"][key], parameters, &mut Vec::new())
}

/// [`schema_kinds`] for interpreting generated arguments, where silently
/// admitting anything would mistype a value: a `$ref` must resolve
/// ([`ref_target`]) or the schema is refused, and a `$ref` with sibling
/// keywords admits the intersection of the target's and the siblings'
/// types (JSON Schema applies both), not an overlay. Malformed `type`,
/// `enum` or combinators are refused rather than read as absent.
pub fn strict_schema_kinds(spec: &Value, root: &Value) -> Result<u8, String> {
    strict_kinds(spec, root, &mut Walk::default())
}

/// Bound on schema objects visited while typing one tool definition (all
/// of its arguments together). Cycles are cut by the active-reference
/// lists, but shared targets reached along many paths (`allOf` members
/// referencing one definition, nested) would otherwise expand
/// exponentially within the depth limit.
const WALK_VISITS: usize = 1 << 18;

/// State of one strict typing pass: the references being resolved by the
/// current walk (cycle and depth guard; an empty entry marks combinator
/// nesting) and the visits, shared by every walk in the pass. The root
/// walk and each property walk keep separate reference lists: a definition
/// being expanded as the arguments object is not thereby a cycle when a
/// property's own type refers to it (a recursive `Node.child: Node`).
#[derive(Default)]
struct Walk {
    active: Vec<String>,
    visits: usize,
}

impl Walk {
    fn visit(&mut self) -> Result<(), String> {
        self.visits += 1;
        if self.active.len() >= 128 {
            return Err("tool argument schema exceeds reference safety limit".into());
        }
        if self.visits > WALK_VISITS {
            return Err(format!(
                "tool argument schema exceeds {WALK_VISITS} schema visits"
            ));
        }
        Ok(())
    }

    fn resolving(&self, reference: &str) -> bool {
        self.active.iter().any(|v| v == reference)
    }

    /// Run `walk` with a fresh reference list, sharing the visit budget.
    fn separately<T>(&mut self, walk: impl FnOnce(&mut Self) -> T) -> T {
        let outer = std::mem::take(&mut self.active);
        let result = walk(self);
        self.active = outer;
        result
    }
}

/// `object[name]` as an array, refusing any other present value.
fn array_keyword<'v>(
    object: &'v serde_json::Map<String, Value>,
    name: &str,
    what: &str,
) -> Result<Option<&'v Vec<Value>>, String> {
    match object.get(name) {
        None => Ok(None),
        Some(Value::Array(items)) => Ok(Some(items)),
        Some(_) => Err(format!("{what} {name} must be an array")),
    }
}

fn strict_kinds(spec: &Value, root: &Value, walk: &mut Walk) -> Result<u8, String> {
    walk.visit()?;
    if spec == &Value::Bool(false) {
        return Ok(0);
    }
    let Some(object) = spec.as_object() else {
        return Ok(ANY);
    };
    let what = "tool argument schema";
    let mut mask = ANY;
    if let Some(reference) = object.get("$ref") {
        let reference = reference
            .as_str()
            .ok_or_else(|| format!("{what} $ref must be a string"))?;
        let target = ref_target(root, reference)
            .ok_or_else(|| format!("{what} $ref {reference:?} is not a local definition"))?;
        if !walk.resolving(reference) {
            walk.active.push(reference.into());
            mask &= strict_kinds(target, root, walk)?;
            walk.active.pop();
        }
    }
    match object.get("type") {
        None => {}
        Some(Value::String(name)) => mask &= type_name_kinds(name),
        Some(Value::Array(names)) => {
            let mut kinds = 0;
            for name in names {
                let name = name
                    .as_str()
                    .ok_or_else(|| format!("{what} type entries must be strings"))?;
                kinds |= type_name_kinds(name);
            }
            mask &= kinds;
        }
        Some(_) => return Err(format!("{what} type must be a string or an array")),
    }
    if let Some(values) = array_keyword(object, "enum", what)? {
        mask &= values.iter().fold(0, |m, v| m | value_kind(v));
    }
    if let Some(value) = object.get("const") {
        mask &= value_kind(value);
    }
    for key in ["anyOf", "oneOf", "allOf"] {
        if let Some(variants) = array_keyword(object, key, what)? {
            let mut combined = if key == "allOf" { ANY } else { 0 };
            for variant in variants {
                walk.active.push(String::new());
                let child = strict_kinds(variant, root, walk)?;
                walk.active.pop();
                if key == "allOf" {
                    combined &= child;
                } else {
                    combined |= child;
                }
            }
            mask &= combined;
        }
    }
    Ok(mask)
}

/// The outer types a function's `parameters` admit for argument `key` under
/// [`strict_schema_kinds`]. Every object schema that applies to the
/// arguments object constrains `key`, and they intersect: `parameters`
/// itself, each target along its `$ref` chain (siblings apply alongside the
/// reference, as for properties), and each `allOf` member; `anyOf`/`oneOf`
/// members admit the union of what each allows. A schema in that walk that
/// does not name `key` in `properties` leaves it unconstrained, so a missing
/// schema admits anything. A non-string or unresolvable `$ref` anywhere in
/// the walk refuses the schema; a reference cycle adds nothing further.
pub fn strict_argument_kinds(parameters: &Value, key: &str) -> Result<u8, String> {
    object_argument_kinds(parameters, parameters, Some(key), &mut Walk::default())
}

/// [`strict_argument_kinds`] for one object schema in the root walk; `None`
/// types a name that no `properties` declares (no string stands in for it,
/// so it cannot collide with a declared name).
fn object_argument_kinds(
    schema: &Value,
    root: &Value,
    key: Option<&str>,
    walk: &mut Walk,
) -> Result<u8, String> {
    walk.visit()?;
    if schema == &Value::Bool(false) {
        return Ok(0);
    }
    let Some(object) = schema.as_object() else {
        return Ok(ANY);
    };
    let what = "tool parameters";
    let mut mask = ANY;
    if let Some(reference) = object.get("$ref") {
        let reference = reference
            .as_str()
            .ok_or_else(|| format!("{what} $ref must be a string"))?;
        let target = ref_target(root, reference)
            .ok_or_else(|| format!("{what} $ref {reference:?} is not a local definition"))?;
        if !walk.resolving(reference) {
            walk.active.push(reference.into());
            mask &= object_argument_kinds(target, root, key, walk)?;
            walk.active.pop();
        }
    }
    match object.get("properties") {
        None => {}
        Some(Value::Object(properties)) => {
            if let Some(property) = key.and_then(|key| properties.get(key)) {
                mask &= walk.separately(|walk| strict_kinds(property, root, walk))?;
            }
        }
        Some(_) => return Err(format!("{what} properties must be an object")),
    }
    for combinator in ["allOf", "anyOf", "oneOf"] {
        let Some(members) = array_keyword(object, combinator, what)? else {
            continue;
        };
        let mut combined = if combinator == "allOf" { ANY } else { 0 };
        for member in members {
            walk.active.push(String::new());
            let kinds = object_argument_kinds(member, root, key, walk)?;
            walk.active.pop();
            if combinator == "allOf" {
                combined &= kinds;
            } else {
                combined |= kinds;
            }
        }
        mask &= combined;
    }
    Ok(mask)
}

/// Every argument's outer types under one tool definition's `parameters`
/// ([`strict_argument_kinds`]), computed once when the definition arrives:
/// each name declared in any `properties` of the root walk, and one value
/// for every undeclared name. All walks share one visit budget, so a
/// definition's typing work is bounded as a whole.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ArgumentKinds {
    declared: std::collections::BTreeMap<String, u8>,
    undeclared: u8,
}

impl ArgumentKinds {
    /// No schema: every argument admits anything.
    pub fn any() -> Self {
        Self {
            declared: Default::default(),
            undeclared: ANY,
        }
    }

    pub fn of(&self, key: &str) -> u8 {
        self.declared.get(key).copied().unwrap_or(self.undeclared)
    }
}

/// [`ArgumentKinds`] for `parameters`, refusing a schema that could not
/// type some argument, so the refusal reaches the client before generation
/// rather than as a failed call after it.
pub fn strict_parameter_kinds(parameters: &Value) -> Result<ArgumentKinds, String> {
    let mut walk = Walk::default();
    let mut keys = std::collections::BTreeSet::new();
    collect_property_keys(
        parameters,
        parameters,
        &mut Vec::new(),
        &mut keys,
        &mut walk,
    )?;
    // Undeclared names: this walk also runs every structural check.
    let undeclared = object_argument_kinds(parameters, parameters, None, &mut walk)?;
    let mut declared = std::collections::BTreeMap::new();
    for key in keys {
        let kinds = object_argument_kinds(parameters, parameters, Some(key), &mut walk)?;
        declared.insert(key.to_owned(), kinds);
    }
    Ok(ArgumentKinds {
        declared,
        undeclared,
    })
}

/// Property names in the root walk of [`strict_argument_kinds`] (counted
/// against the pass's visits); malformed or unresolvable parts are skipped
/// here and reported by the typing walks.
fn collect_property_keys<'a>(
    schema: &'a Value,
    root: &'a Value,
    active: &mut Vec<&'a str>,
    keys: &mut std::collections::BTreeSet<&'a str>,
    walk: &mut Walk,
) -> Result<(), String> {
    let Some(object) = schema.as_object() else {
        return Ok(());
    };
    if active.len() >= 128 {
        return Ok(());
    }
    walk.visit()?;
    if let Some(reference) = object.get("$ref").and_then(Value::as_str)
        && !active.contains(&reference)
        && let Some(target) = ref_target(root, reference)
    {
        active.push(reference);
        collect_property_keys(target, root, active, keys, walk)?;
        active.pop();
    }
    if let Some(Value::Object(properties)) = object.get("properties") {
        keys.extend(properties.keys().map(String::as_str));
    }
    for combinator in ["allOf", "anyOf", "oneOf"] {
        if let Some(Value::Array(members)) = object.get(combinator) {
            for member in members {
                active.push("");
                collect_property_keys(member, root, active, keys, walk)?;
                active.pop();
            }
        }
    }
    Ok(())
}

/// Why [`decode_json_prefix`] stopped.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum JsonDecodeError {
    /// The text ends inside a value.
    Incomplete,
    Malformed(String),
}

/// One JSON value at the start of `text` and the bytes it used.
pub fn decode_json_prefix(text: &str) -> Result<(Value, usize), JsonDecodeError> {
    let mut parser = JsonParser { text, at: 0 };
    let value = parser.value(0)?;
    Ok((value, parser.at))
}

/// Exactly one JSON value (surrounding JSON whitespace allowed).
pub fn decode_json(text: &str) -> Result<Value, String> {
    match decode_json_prefix(text) {
        Ok((value, offset)) if text[offset..].trim_matches(json_space).is_empty() => Ok(value),
        Ok(_) => Err("trailing tool JSON content".into()),
        Err(JsonDecodeError::Incomplete) => Err("incomplete tool JSON value".into()),
        Err(JsonDecodeError::Malformed(e)) => Err(e),
    }
}

fn json_space(c: char) -> bool {
    matches!(c, ' ' | '\t' | '\r' | '\n')
}

fn malformed(message: impl Into<String>) -> JsonDecodeError {
    JsonDecodeError::Malformed(message.into())
}

struct JsonParser<'a> {
    text: &'a str,
    at: usize,
}

impl JsonParser<'_> {
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
    fn expect(&mut self, byte: u8) -> Result<(), JsonDecodeError> {
        if self.take(byte) {
            Ok(())
        } else if self.at == self.text.len() {
            Err(JsonDecodeError::Incomplete)
        } else {
            Err(malformed("invalid tool JSON container syntax"))
        }
    }
    fn string(&mut self) -> Result<String, JsonDecodeError> {
        self.whitespace();
        let mut stream =
            serde_json::Deserializer::from_str(&self.text[self.at..]).into_iter::<String>();
        match stream.next() {
            Some(Ok(value)) => {
                self.at += stream.byte_offset();
                Ok(value)
            }
            Some(Err(e)) if e.is_eof() => Err(JsonDecodeError::Incomplete),
            Some(Err(e)) => Err(malformed(format!("invalid tool JSON string: {e}"))),
            None => Err(JsonDecodeError::Incomplete),
        }
    }
    fn value(&mut self, depth: usize) -> Result<Value, JsonDecodeError> {
        if depth >= 128 {
            return Err(malformed("tool JSON exceeds nesting safety limit"));
        }
        self.whitespace();
        match self.text.as_bytes().get(self.at).copied() {
            None => Err(JsonDecodeError::Incomplete),
            Some(b'"') => self.string().map(Value::String),
            Some(b'{') => {
                self.at += 1;
                let mut map = serde_json::Map::new();
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
                    Err(JsonDecodeError::Incomplete)
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
                    Err(e) if e.is_eof() && self.at == self.text.len() => {
                        Err(JsonDecodeError::Incomplete)
                    }
                    Err(e) => Err(malformed(format!("invalid tool JSON number: {e}"))),
                }
            }
            _ => Err(malformed("invalid tool JSON value")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn refs_resolve_in_their_own_namespace_with_pointer_escapes() {
        let parameters = json!({
            "$defs": {"T": {"type": "integer"}, "a/b": {"type": "boolean"}},
            "definitions": {"T": {"type": "string"}},
            "properties": {
                "d": {"$ref": "#/definitions/T"},
                "s": {"$ref": "#/$defs/T"},
                "e": {"$ref": "#/$defs/a~1b"},
                "deep": {"$ref": "#/$defs/T/properties/x"},
                "remote": {"$ref": "https://example.com/t.json"},
                "sib": {"$ref": "#/$defs/T", "type": ["integer", "string"]}
            }
        });
        assert_eq!(strict_argument_kinds(&parameters, "d"), Ok(STRING));
        assert_eq!(strict_argument_kinds(&parameters, "s"), Ok(INTEGER));
        assert_eq!(strict_argument_kinds(&parameters, "e"), Ok(BOOL));
        assert!(strict_argument_kinds(&parameters, "deep").is_err());
        assert!(strict_argument_kinds(&parameters, "remote").is_err());
        // Siblings intersect with the target (integer), not override it.
        assert_eq!(strict_argument_kinds(&parameters, "sib"), Ok(INTEGER));
        assert_eq!(strict_argument_kinds(&parameters, "missing"), Ok(ANY));
        // K2's presentation convention is unchanged: `$defs` wins whenever it
        // exists, matching what K2's renderer shows the model.
        assert_eq!(
            resolve_ref(&parameters, &json!({"$ref": "#/definitions/T"})),
            Some(json!({"type": "integer"}))
        );
    }

    /// Definitions are checked when they arrive: every declared property
    /// (root, along `$ref` chains and in combinators) and an undeclared name
    /// must type, so a broken schema is refused before generation.
    #[test]
    fn parameters_are_checked_for_every_declared_property_up_front() {
        let kinds = strict_parameter_kinds(&json!({"type": "object", "properties": {
            "a": {"type": "string"}, "b": {"$ref": "#/$defs/B"}},
            "$defs": {"B": {"type": "integer"}}}))
        .unwrap();
        assert_eq!(kinds.of("a"), STRING);
        assert_eq!(kinds.of("b"), INTEGER);
        assert_eq!(kinds.of("undeclared"), ANY);
        // No sentinel name stands in for undeclared arguments: declaring
        // any name, however odd, constrains only that name.
        for odd in ["\u{0}undeclared", "", "undeclared"] {
            let kinds =
                strict_parameter_kinds(&json!({"properties": {odd: {"type": "integer"}}})).unwrap();
            assert_eq!(kinds.of(odd), INTEGER, "{odd:?}");
            assert_eq!(kinds.of("other"), ANY, "{odd:?}");
        }
        for bad in [
            // A property's reference that is never resolvable.
            json!({"properties": {"a": {"$ref": "#/$defs/Gone"}}}),
            // Reachable only through the root chain.
            json!({"$ref": "#/$defs/A", "$defs": {"A": {"properties": {
                "a": {"$ref": "https://example.com/a"}}}}}),
            // Reachable only through a root combinator.
            json!({"anyOf": [{"properties": {"a": {"$ref": 3}}}]}),
            // Root structure.
            json!({"$ref": "#/$defs/Gone"}),
            json!({"properties": "a"}),
            // Property-level keywords that are malformed, not absent.
            json!({"properties": {"x": {"allOf": {}}}}),
            json!({"properties": {"x": {"anyOf": "integer"}}}),
            json!({"properties": {"x": {"type": 5}}}),
            json!({"properties": {"x": {"type": ["integer", 5]}}}),
            json!({"properties": {"x": {"enum": "a"}}}),
            json!({"$ref": "#/$defs/A", "$defs": {"A": {"properties": {"x": {"oneOf": {}}}}}}),
        ] {
            assert!(strict_parameter_kinds(&bad).is_err(), "{bad}");
        }
    }

    /// A definition expanded as the arguments object is not a cycle when a
    /// property's own type refers to it: `Node.child: Node` admits objects
    /// only, so raw `abc` is never a string for it.
    #[test]
    fn a_recursive_property_keeps_its_type() {
        let parameters = json!({"$ref": "#/$defs/Node", "$defs": {"Node": {
            "type": "object",
            "properties": {"child": {"$ref": "#/$defs/Node"}, "n": {"type": "integer"}}}}});
        assert_eq!(strict_argument_kinds(&parameters, "child"), Ok(OBJECT));
        let kinds = strict_parameter_kinds(&parameters).unwrap();
        assert_eq!(kinds.of("child"), OBJECT);
        assert_eq!(kinds.of("n"), INTEGER);
        // Within one property walk, a true cycle still terminates.
        let cycle = json!({"properties": {"x": {"$ref": "#/$defs/A"}}, "$defs": {
            "A": {"$ref": "#/$defs/B", "type": ["integer", "string"]},
            "B": {"$ref": "#/$defs/A", "type": "integer"}}});
        assert_eq!(strict_argument_kinds(&cycle, "x"), Ok(INTEGER));
    }

    /// Shared targets reached along many paths would expand exponentially
    /// within the depth limit; the visit budget refuses such a schema
    /// promptly (cycles alone are cut by the active-reference list).
    #[test]
    fn shared_reference_fans_are_bounded() {
        let mut defs = serde_json::Map::new();
        for level in 0..40 {
            let next = format!("#/$defs/L{}", level + 1);
            defs.insert(
                format!("L{level}"),
                json!({"allOf": [{"$ref": next}, {"$ref": next}],
                    "properties": {"x": {"allOf": [{"$ref": next}, {"$ref": next}]}}}),
            );
        }
        defs.insert("L40".into(), json!({"type": "integer"}));
        let parameters = json!({"$ref": "#/$defs/L0", "$defs": defs});
        let started = std::time::Instant::now();
        let error = strict_argument_kinds(&parameters, "x").unwrap_err();
        assert!(error.contains("schema visits"), "{error}");
        assert!(strict_parameter_kinds(&parameters).is_err());
        assert!(started.elapsed() < std::time::Duration::from_secs(5));
    }

    /// The budget covers a definition as a whole: many properties that each
    /// stay under it alone, all referring to one branching definition, are
    /// refused together rather than each granted a fresh budget.
    #[test]
    fn one_budget_covers_every_property_of_a_definition() {
        let mut defs = serde_json::Map::new();
        for level in 0..9 {
            let next = format!("#/$defs/L{}", level + 1);
            defs.insert(
                format!("L{level}"),
                json!({"allOf": [{"$ref": next}, {"$ref": next}]}),
            );
        }
        defs.insert("L9".into(), json!({"type": "integer"}));
        let properties: serde_json::Map<String, Value> = (0..2_000)
            .map(|i| (format!("p{i}"), json!({"$ref": "#/$defs/L0"})))
            .collect();
        let parameters = json!({"properties": properties, "$defs": defs});
        // One property alone is cheap (about 2^10 visits) and types.
        assert_eq!(strict_argument_kinds(&parameters, "p0"), Ok(INTEGER));
        let started = std::time::Instant::now();
        let error = strict_parameter_kinds(&parameters).unwrap_err();
        assert!(error.contains("schema visits"), "{error}");
        assert!(started.elapsed() < std::time::Duration::from_secs(5));
    }

    /// A root `parameters.$ref` is one constraint among several, not a
    /// replacement for the root: siblings, chains, and root combinators all
    /// apply, and anything malformed or unresolvable refuses the schema.
    #[test]
    fn root_references_intersect_with_every_applicable_object_schema() {
        let kinds = |parameters: Value, key: &str| strict_argument_kinds(&parameters, key);
        // Root sibling properties apply alongside an unconstrained target:
        // `abc` must not pass as a string for an integer argument.
        assert_eq!(
            kinds(
                json!({"$ref": "#/$defs/Any", "$defs": {"Any": {"type": "object"}},
                    "properties": {"x": {"type": "integer"}}}),
                "x"
            ),
            Ok(INTEGER)
        );
        // Target and sibling constraints intersect.
        assert_eq!(
            kinds(
                json!({"$ref": "#/$defs/T",
                    "$defs": {"T": {"properties": {"x": {"type": ["integer", "string"]}}}},
                    "properties": {"x": {"type": ["string", "boolean"]}}}),
                "x"
            ),
            Ok(STRING)
        );
        // A chain resolves to its end; every link's properties apply.
        let chained = json!({"$ref": "#/$defs/A", "$defs": {
            "A": {"$ref": "#/$defs/B", "properties": {"x": {"type": ["integer", "null"]}}},
            "B": {"properties": {"x": {"type": "integer"}}}}});
        assert_eq!(kinds(chained, "x"), Ok(INTEGER));
        // A chain whose later link is missing, malformed or remote refuses.
        for bad in [
            json!({"$ref": "#/$defs/A", "$defs": {"A": {"$ref": "#/$defs/Missing"}}}),
            json!({"$ref": "#/$defs/A", "$defs": {"A": {"$ref": 7}}}),
            json!({"$ref": "#/$defs/A", "$defs": {"A": {"$ref": "https://example.com/a"}}}),
            json!({"$ref": "#/$defs/A/properties", "$defs": {"A": {}}}),
        ] {
            assert!(kinds(bad.clone(), "x").is_err(), "{bad}");
        }
        // A non-string root $ref is refused, not ignored.
        assert!(kinds(json!({"$ref": ["#/$defs/A"], "$defs": {"A": {}}}), "x").is_err());
        assert!(kinds(json!({"$ref": null}), "x").is_err());
        // Malformed root structure is refused rather than read as absent.
        assert!(kinds(json!({"properties": []}), "x").is_err());
        assert!(kinds(json!({"allOf": {}}), "x").is_err());
        // A cycle terminates and keeps the constraints it passed.
        let cycle = json!({"$ref": "#/$defs/A", "$defs": {
            "A": {"$ref": "#/$defs/B", "properties": {"x": {"type": "integer"}}},
            "B": {"$ref": "#/$defs/A"}}});
        assert_eq!(kinds(cycle, "x"), Ok(INTEGER));
        // Root allOf intersects; anyOf/oneOf unite (a member without the
        // property leaves it unconstrained).
        assert_eq!(
            kinds(
                json!({"allOf": [{"properties": {"x": {"type": ["integer", "string"]}}},
                    {"properties": {"x": {"type": "integer"}}}]}),
                "x"
            ),
            Ok(INTEGER)
        );
        assert_eq!(
            kinds(
                json!({"anyOf": [{"properties": {"x": {"type": "integer"}}},
                    {"properties": {"x": {"type": "boolean"}}}]}),
                "x"
            ),
            Ok(INTEGER | BOOL)
        );
        assert_eq!(
            kinds(
                json!({"oneOf": [{"properties": {"x": {"type": "integer"}}}, {}]}),
                "x"
            ),
            Ok(ANY)
        );
        // Plain schemas are unchanged.
        assert_eq!(
            kinds(json!({"properties": {"x": {"type": "number"}}}), "x"),
            Ok(NUMBER | INTEGER)
        );
        assert_eq!(kinds(json!({"type": "object"}), "x"), Ok(ANY));
    }

    #[test]
    fn the_decoder_keeps_containers_and_refuses_duplicates() {
        assert_eq!(
            decode_json(r#"{"x":{"$serde_json::private::Number":"7"}}"#).unwrap(),
            {
                let mut inner = serde_json::Map::new();
                inner.insert(
                    "$serde_json::private::Number".into(),
                    Value::String("7".into()),
                );
                let mut outer = serde_json::Map::new();
                outer.insert("x".into(), Value::Object(inner));
                Value::Object(outer)
            }
        );
        assert!(decode_json(r#"{"a":1,"a":2}"#).is_err());
        assert!(decode_json(r#"{"o":{"a":1,"a":2}}"#).is_err());
        assert_eq!(
            decode_json_prefix("[1, 2"),
            Err(JsonDecodeError::Incomplete)
        );
    }
}
