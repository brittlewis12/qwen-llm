use super::*;

impl Renderer<'_> {
    pub(super) fn markdown_function(&mut self, function: &Value, out: &mut String) -> Result<()> {
        let parameters = self.resolve(&function["parameters"], true);
        out.push_str(&format!("\n## {}", text(&function["name"])?));
        if truthy(&function["description"]) {
            out.push_str(&format!("\n{}", text(&function["description"])?));
        }
        out.push_str("\n\n**Parameters**");
        if truthy(&parameters["properties"]) {
            for (name, spec) in object(&parameters["properties"])? {
                self.markdown_param(name, spec, required(&parameters, name), "", out)?;
            }
        } else if parameters.is_object()
            && (truthy(&parameters["oneOf"])
                || truthy(&parameters["anyOf"])
                || parameters.get("items").is_some())
        {
            let parameters = self.resolve(&parameters, false);
            self.markdown_annotations(&parameters, "", true, true, out)?;
            self.markdown_structure(&parameters, "", false, true, out)?;
            extras(&parameters, "", true, out)?;
        } else {
            out.push_str("\n- None");
        }
        if let Some(returns) = function.get("returns").or_else(|| function.get("response")) {
            out.push_str("\n\n**Returns**");
            if returns.is_object() {
                out.push_str(&format!("\n- Return *({})*", markdown_type(returns)?));
                self.markdown_details(returns, "", true, out)?;
            } else {
                out.push_str(&format!("\n- {}", markdown_value(returns)?));
            }
        }
        Ok(())
    }
    fn markdown_param(
        &mut self,
        name: &str,
        spec: &Value,
        required: bool,
        indent: &str,
        out: &mut String,
    ) -> Result<()> {
        self.nested(|this| {
            let spec = this.resolve(spec, false);
            out.push_str(&format!(
                "\n{indent}- `{name}` *({}{})*",
                markdown_type(&spec)?,
                if required { ", required" } else { "" }
            ));
            if truthy(&spec["description"]) {
                out.push_str(&format!(
                    " - {}",
                    text(&spec["description"])?.replace('\n', &format!("\n{indent}  "))
                ));
            }
            if truthy(&spec["enum"]) {
                out.push_str(&format!(
                    "\n{indent}  - Allowed values: {}",
                    allowed(&spec["enum"])?
                ));
            }
            if let Some(default) = spec.get("default") {
                out.push_str(&format!("\n{indent}  - Default: {}", literal(default)?));
            }
            this.markdown_details(&spec, indent, false, out)
        })
    }
    fn markdown_details(
        &mut self,
        spec: &Value,
        indent: &str,
        include_values: bool,
        out: &mut String,
    ) -> Result<()> {
        self.nested(|this| {
            let spec = this.resolve(spec, false);
            if spec.is_object() {
                this.markdown_annotations(&spec, indent, include_values, false, out)?;
                this.markdown_structure(&spec, indent, true, false, out)?;
                extras(&spec, indent, false, out)?;
            } else if !spec.is_boolean() {
                out.push_str(&format!("\n{indent}  - Value: {}", literal(&spec)?));
            }
            Ok(())
        })
    }
    fn markdown_annotations(
        &mut self,
        spec: &Value,
        indent: &str,
        include_values: bool,
        metadata: bool,
        out: &mut String,
    ) -> Result<()> {
        let prefix = prefix(indent, metadata);
        if include_values {
            if let Some(description) = spec.get("description") {
                let continuation = if metadata {
                    "\n    ".into()
                } else {
                    format!("\n{indent}    ")
                };
                out.push_str(&format!(
                    "{prefix}Description: {}",
                    markdown_value(&Value::String(
                        text(description)?.replace('\n', &continuation)
                    ))?
                ));
            }
            if let Some(values) = spec.get("enum") {
                out.push_str(&format!("{prefix}Allowed values: {}", allowed(values)?));
            }
            if let Some(default) = spec.get("default") {
                out.push_str(&format!("{prefix}Default: {}", literal(default)?));
            }
        }
        if let Some(additional) = spec.get("additionalProperties") {
            if additional.is_object() {
                out.push_str(&format!(
                    "{prefix}Additional properties *({})*",
                    markdown_type(additional)?
                ));
                self.markdown_details(additional, &next_indent(indent, metadata), true, out)?;
            } else {
                out.push_str(&format!(
                    "{prefix}Additional properties: {}",
                    markdown_value(additional)?
                ));
            }
        }
        Ok(())
    }
    fn markdown_structure(
        &mut self,
        spec: &Value,
        indent: &str,
        include_properties: bool,
        metadata: bool,
        out: &mut String,
    ) -> Result<()> {
        let prefix = prefix(indent, metadata);
        let next = next_indent(indent, metadata);
        let variant_indent = format!("{next}  ");
        if include_properties && truthy(&spec["properties"]) {
            for (name, child) in object(&spec["properties"])? {
                self.markdown_param(name, child, required(spec, name), &next, out)?;
            }
        }
        if let Some(items) = spec.get("items") {
            if items.is_object() {
                out.push_str(&format!("{prefix}Items *({})*", markdown_type(items)?));
                self.markdown_details(items, &next, true, out)?;
            } else {
                out.push_str(&format!("{prefix}Items: {}", markdown_value(items)?));
            }
        }
        for key in ["oneOf", "anyOf"] {
            if truthy(&spec[key]) {
                out.push_str(&format!("{prefix}{key}:"));
                for (index, variant) in array(&spec[key])?.iter().enumerate() {
                    out.push_str(&format!(
                        "\n{variant_indent}- Variant {} *({})*",
                        index + 1,
                        markdown_type(variant)?
                    ));
                    self.markdown_details(variant, &variant_indent, true, out)?;
                }
            }
        }
        if let Some(patterns) = spec["patternProperties"].as_object() {
            out.push_str(&format!("{prefix}Pattern properties:"));
            for (pattern, child) in patterns {
                if child.is_object() {
                    out.push_str(&format!(
                        "\n{variant_indent}- `{pattern}` *({})*",
                        markdown_type(child)?
                    ));
                    self.markdown_details(child, &variant_indent, true, out)?;
                } else {
                    out.push_str(&format!(
                        "\n{variant_indent}- `{pattern}`: {}",
                        markdown_value(child)?
                    ));
                }
            }
        } else if let Some(patterns) = spec.get("patternProperties") {
            out.push_str(&format!(
                "{prefix}Pattern properties: {}",
                markdown_value(patterns)?
            ));
        }
        if let Some(returns) = spec.get("returns") {
            if returns.is_object() {
                out.push_str(&format!("{prefix}Returns *({})*", markdown_type(returns)?));
                self.markdown_details(returns, &next, true, out)?;
            } else {
                out.push_str(&format!("{prefix}Returns: {}", markdown_value(returns)?));
            }
        }
        Ok(())
    }
}

fn prefix(indent: &str, metadata: bool) -> String {
    if metadata {
        "\n- ".into()
    } else {
        format!("\n{indent}  - ")
    }
}
fn next_indent(indent: &str, metadata: bool) -> String {
    if metadata {
        "".into()
    } else {
        format!("{indent}  ")
    }
}
fn extras(spec: &Value, indent: &str, metadata: bool, out: &mut String) -> Result<()> {
    for (key, value) in object(spec)? {
        if !STRUCTURAL.contains(&key.as_str()) {
            out.push_str(&format!(
                "{}{key}: {}",
                prefix(indent, metadata),
                markdown_value(value)?
            ));
        }
    }
    Ok(())
}
