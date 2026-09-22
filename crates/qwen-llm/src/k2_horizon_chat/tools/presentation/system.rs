use super::*;

/// Render the pinned native system turn with tool definitions and call-format
/// instructions. This low-level encoder does not authorize a tool request.
pub fn render_tool_system(
    definitions: &[Value],
    system_content: &str,
    presentation: ToolPresentationFormat,
    calls: ToolCallFormat,
) -> Result<String> {
    let mut out = String::from(
        "<|ifm|im_start|>system\n# Tools\nYou may call one or more tools to assist with the user query.\n\nAvailable tools are:\n\n",
    );
    out.push_str(&render_tool_definitions(definitions, presentation)?);
    out.push_str("\n\nWhen calling tools, you MUST follow the tool-call format below:\n\n");
    out.push_str(match calls {
        ToolCallFormat::Json => JSON,
        ToolCallFormat::Xml => XML,
        ToolCallFormat::XmlTyped => XML_TYPED,
    });
    if !system_content.is_empty() {
        out.push_str("\n\n");
        out.push_str(system_content);
    }
    out.push_str("<|ifm|im_end|>");
    Ok(out)
}

const JSON: &str = "Wrap all tool calls in a single <ifm|tool_calls></ifm|tool_calls> block. For each call, emit one JSON object with the function name and arguments on the same line inside <ifm|tool_call></ifm|tool_call> tags:\n\n<ifm|tool_calls>\n<ifm|tool_call>{\"name\": <function-name>, \"arguments\": <args-json-object>}</ifm|tool_call>\n</ifm|tool_calls>";
const XML: &str = "Wrap all tool calls in a single <ifm|tool_calls></ifm|tool_calls> block. For each call, write the function name at the start of <ifm|tool_call>, followed by paired <ifm|arg_key> and <ifm|arg_value> tags for each argument:\n\n<ifm|tool_calls>\n<ifm|tool_call>$FUNCTION_NAME\n<ifm|arg_key>$PARAMETER_NAME</ifm|arg_key>\n<ifm|arg_value>$PARAMETER_VALUE</ifm|arg_value>\n...\n</ifm|tool_call>\n</ifm|tool_calls>\n\nString and scalar parameters should be written as plain text. Array and object parameters should be written as JSON literals.";
const XML_TYPED: &str = "Wrap all tool calls in a single <ifm|tool_calls></ifm|tool_calls> block. For each call, write the function name at the start of <ifm|tool_call>, followed by <ifm|arg_key>, <ifm|arg_type>, and <ifm|arg_value> tags for each argument:\n\n<ifm|tool_calls>\n<ifm|tool_call>$FUNCTION_NAME\n<ifm|arg_key>$PARAMETER_NAME</ifm|arg_key>\n<ifm|arg_type>$ARGUMENT_TYPE</ifm|arg_type>\n<ifm|arg_value>$PARAMETER_VALUE</ifm|arg_value>\n...\n</ifm|tool_call>\n</ifm|tool_calls>\n\nUse the parameter type shown in the tool definition. If that type contains anyOf or oneOf, use the actual argument value type instead. String and scalar parameters should be written as plain text. Array and object parameters should be written as JSON literals.";
