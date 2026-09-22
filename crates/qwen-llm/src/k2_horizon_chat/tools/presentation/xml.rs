use super::*;

impl Renderer<'_> {
    pub(super) fn xml_function(&mut self, function: &Value, out: &mut String) -> Result<()> {
        let parameters = self.resolve(&function["parameters"], true);
        out.push_str(&format!("\n<function name={}>", text(&function["name"])?));
        if truthy(&function["description"]) {
            out.push_str(&format!(
                "<description>{}</description>",
                text(&function["description"])?
            ));
        }
        out.push_str("<parameters>");
        if truthy(&parameters["properties"]) {
            for (name, spec) in object(&parameters["properties"])? {
                self.xml_param(name, spec, required(&parameters, name), out)?;
            }
        } else if parameters.is_object()
            && (truthy(&parameters["oneOf"])
                || truthy(&parameters["anyOf"])
                || parameters.get("items").is_some())
        {
            self.xml_children(&parameters, true, false, out)?;
        }
        out.push_str("</parameters>");
        if let Some(returns) = function.get("returns").or_else(|| function.get("response")) {
            self.xml_node("returns", returns, None, out)?;
        }
        out.push_str("</function>");
        Ok(())
    }
    fn xml_param(
        &mut self,
        name: &str,
        spec: &Value,
        required: bool,
        out: &mut String,
    ) -> Result<()> {
        self.nested(|this| {
            let spec = this.resolve(spec, false);
            out.push_str(&format!("<param name={name} type={}", compact(&spec)?));
            if required {
                out.push_str(" required=true");
            }
            value_attrs(&spec, out)?;
            attrs(&spec, false, out)?;
            if truthy(&spec["description"]) || has_children(&spec, false) {
                out.push('>');
                if truthy(&spec["description"]) {
                    out.push_str(text(&spec["description"])?);
                }
                this.xml_children(&spec, true, false, out)?;
                out.push_str("</param>");
            } else {
                out.push_str("/>");
            }
            Ok(())
        })
    }
    fn xml_node(
        &mut self,
        tag: &str,
        spec: &Value,
        pattern: Option<&str>,
        out: &mut String,
    ) -> Result<()> {
        self.nested(|this| {
            let spec = this.resolve(spec, false);
            out.push('<');
            out.push_str(tag);
            if let Some(pattern) = pattern {
                attr("pattern", &Value::String(pattern.into()), out)?;
            }
            if spec.is_object() {
                out.push_str(&format!(" type={}", compact(&spec)?));
                attrs(&spec, true, out)?;
                if has_children(&spec, true) {
                    out.push('>');
                    this.xml_children(&spec, true, true, out)?;
                    out.push_str(&format!("</{tag}>"));
                } else {
                    out.push_str("/>");
                }
            } else {
                out.push_str(&format!(">{}</{tag}>", value(&spec)?));
            }
            Ok(())
        })
    }
    fn xml_children(
        &mut self,
        spec: &Value,
        properties: bool,
        description: bool,
        out: &mut String,
    ) -> Result<()> {
        if description && let Some(description) = spec.get("description") {
            out.push_str(&format!(
                "<description>{}</description>",
                text(description)?
            ));
        }
        if properties && truthy(&spec["properties"]) {
            for (name, child) in object(&spec["properties"])? {
                self.xml_param(name, child, required(spec, name), out)?;
            }
        }
        if let Some(items) = spec.get("items") {
            self.xml_node("items", items, None, out)?;
        }
        for key in ["oneOf", "anyOf"] {
            if truthy(&spec[key]) {
                out.push_str(&format!("<{key}>"));
                for variant in array(&spec[key])? {
                    self.xml_node("variant", variant, None, out)?;
                }
                out.push_str(&format!("</{key}>"));
            }
        }
        if spec["additionalProperties"].is_object() {
            self.xml_node(
                "additionalProperties",
                &spec["additionalProperties"],
                None,
                out,
            )?;
        }
        if let Some(patterns) = spec["patternProperties"].as_object() {
            out.push_str("<patternProperties>");
            for (pattern, spec) in patterns {
                self.xml_node("patternProperty", spec, Some(pattern), out)?;
            }
            out.push_str("</patternProperties>");
        } else if let Some(patterns) = spec.get("patternProperties") {
            out.push_str(&format!(
                "<patternProperties>{}</patternProperties>",
                value(patterns)?
            ));
        }
        if let Some(returns) = spec.get("returns") {
            self.xml_node("returns", returns, None, out)?;
        }
        Ok(())
    }
}

fn value(v: &Value) -> Result<String> {
    v.as_str().map(collapse).map_or_else(|| python_repr(v), Ok)
}
fn quoted(v: &Value) -> Result<String> {
    let s = v
        .as_str()
        .map(str::to_owned)
        .map_or_else(|| python_repr(v), Ok)?;
    Ok(format!(
        "\"{}\"",
        s.replace('\\', "\\\\").replace('"', "\\\"")
    ))
}
fn attr(key: &str, v: &Value, out: &mut String) -> Result<()> {
    out.push_str(&format!(
        " {key}={}",
        if v == "" { "\"\"".into() } else { value(v)? }
    ));
    Ok(())
}
fn value_attrs(spec: &Value, out: &mut String) -> Result<()> {
    if truthy(&spec["enum"]) {
        out.push_str(&format!(
            " enum={}",
            array(&spec["enum"])?
                .iter()
                .map(quoted)
                .collect::<Result<Vec<_>>>()?
                .join("|")
        ));
    }
    if let Some(default) = spec.get("default") {
        out.push_str(&format!(
            " default={}",
            if default.is_string() {
                quoted(default)?
            } else {
                value(default)?
            }
        ));
    }
    Ok(())
}
fn attrs(spec: &Value, include_values: bool, out: &mut String) -> Result<()> {
    let Some(map) = spec.as_object() else {
        return Ok(());
    };
    if include_values {
        value_attrs(spec, out)?;
    }
    for key in ["additionalProperties", "patternProperties"] {
        if let Some(value) = spec.get(key).filter(|v| !v.is_object()) {
            attr(key, value, out)?;
        }
    }
    for (key, value) in map {
        if !STRUCTURAL.contains(&key.as_str()) {
            attr(key, value, out)?;
        }
    }
    Ok(())
}
fn has_children(spec: &Value, description: bool) -> bool {
    spec.is_object()
        && ((description && spec.get("description").is_some())
            || truthy(&spec["properties"])
            || spec.get("items").is_some()
            || truthy(&spec["oneOf"])
            || truthy(&spec["anyOf"])
            || spec["additionalProperties"].is_object()
            || spec["patternProperties"].is_object()
            || spec.get("returns").is_some())
}
