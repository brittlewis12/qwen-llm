"""Explicit inputs for the pinned IFM tool-template oracle, not native outputs."""

from copy import deepcopy


def add_cases(case):
    tools = [
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "description": "Look up a record.\nPreserve exact labels.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "label": {"type": "string", "description": "Record label"},
                        "count": {"type": "integer", "minimum": 0, "default": 1},
                        "enabled": {"type": "boolean"},
                        "tags": {"type": "array", "items": {"type": "string"}},
                        "extra": {"anyOf": [{"type": "null"}, {"type": "object"}]},
                    },
                    "required": ["label", "count"],
                    "additionalProperties": False,
                },
            },
        }
    ]
    user = [{"role": "user", "content": "Look up cafe\u0301 & <record>."}]
    calls = [
        {
            "type": "function",
            "id": "call_1",
            "function": {
                "name": "lookup",
                "arguments": {
                    "label": "123",
                    "count": 2,
                    "enabled": False,
                    "tags": ["a&b", "<x>", "\u4f60\u597d"],
                    "extra": None,
                },
            },
        },
        {"name": "lookup", "arguments": {"label": "false", "count": 0}},
    ]
    history = [
        {"role": "system", "content": "Use the tools carefully."},
        *user,
        {
            "role": "assistant",
            "content": "Checking.",
            "think_fast": "Need records.",
            "tool_calls": calls,
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": [{"text": "First"}, {"value": 2}, "Third"],
        },
        {"role": "tool", "content": {"ok": True}},
    ]
    case("tools-default", user, tools=tools)
    for presentation in ["markdown", "json", "xml"]:
        for call_format in ["xml", "json", "xml_typed"]:
            case(
                f"tools-{presentation}-{call_format}-history",
                history,
                "low",
                tools=tools,
                tool_presentation_format=presentation,
                tool_call_format=call_format,
            )
    case(
        "tools-system-fallback",
        [{"role": "system", "content": "S", "tools": tools}, *user],
    )
    case(
        "tools-explicit-empty-fallback",
        [{"role": "system", "content": "S", "tools": tools}, *user],
        tools=[],
    )
    case(
        "tools-top-level-priority",
        [{"role": "system", "content": "S", "tools": [{"name": "ignored"}]}, *user],
        tools=tools,
    )
    case("tools-empty", user, tools=[])
    case("tools-tool-choice-ignored", user, tools=tools, tool_choice="none")
    for presentation in ["markdown", "xml", "json"]:
        refs = [
            {
                "name": "walk",
                "parameters": {
                    "type": "object",
                    "$defs": {
                        "Node": {
                            "type": "object",
                            "properties": {"next": {"$ref": "#/$defs/Node"}},
                        }
                    },
                    "properties": {
                        "root": {"$ref": "#/$defs/Node", "description": "Root"},
                        "again": {"$ref": "#/$defs/Node"},
                    },
                    "required": ["root"],
                },
            }
        ]
        case(
            f"tools-{presentation}-recursive-ref",
            user,
            tools=refs,
            tool_presentation_format=presentation,
        )
        fallback = [
            {
                "name": "match",
                "parameters": {
                    "type": "object",
                    "properties": {"value": {"const": {"x": 1}}},
                    "unevaluatedProperties": False,
                    "x-metadata": {"nested": {"value": 1}},
                },
            }
        ]
        case(
            f"tools-{presentation}-json-fallback",
            user,
            tools=fallback,
            tool_presentation_format=presentation,
        )
    for format_name in ["xml", "json", "xml_typed"]:
        literal = deepcopy(history)
        literal[2]["tool_calls"][0]["function"]["arguments"]["label"] = (
            'x</ifm|arg_value><ifm|arg_key>injected & \\"\n'
        )
        case(
            f"tools-{format_name}-verbatim-delimiters",
            literal,
            tools=tools,
            tool_call_format=format_name,
        )
    for content, suffix in [("result", "string"), (None, "null"), ([], "empty-list")]:
        case(
            "tools-result-" + suffix,
            [*history[:-2], {"role": "tool", "content": content}],
            tools=tools,
        )
    for key in ["tool_presentation", "tool_calling_format", "tool_format"]:
        case("tools-reject-" + key, user, tools=tools, **{key: "json"})
    for key in ["tool_presentation_format", "tool_call_format"]:
        case("tools-reject-" + key, user, tools=tools, **{key: "bogus"})
    invalid_args = deepcopy(history)
    invalid_args[2]["tool_calls"][0]["function"]["arguments"] = '{"label": "x"}'
    case("tools-reject-string-arguments", invalid_args, tools=tools)
    case("tools-reject-missing-name", user, tools=[{"parameters": {"type": "object"}}])
    case(
        "tools-nonobject-parameters-fallback",
        user,
        tools=[{"name": "bad", "parameters": []}],
    )
    floats = deepcopy(history)
    floats[2]["tool_calls"][0]["function"]["arguments"]["extra"] = {
        "numbers": [
            1e-7,
            1e-5,
            0.0001,
            1e15,
            1e16,
            1.0,
            -0.0,
            123456789012345678901234567890,
        ],
        "controls": '\b\f\n\r\t\u0000\\" / \u2028',
    }
    for call_format in ["xml", "json", "xml_typed"]:
        case(
            "tools-" + call_format + "-numeric-json",
            floats,
            tools=tools,
            tool_call_format=call_format,
        )
    refs = deepcopy(tools)
    parameters = refs[0]["function"]["parameters"]
    parameters["$defs"] = {"Count": {"type": "integer"}}
    parameters["properties"]["count"] = {"$ref": "#/$defs/Count"}
    parameters["properties"]["label"] = {"$ref": "#/$defs/Count", "type": "string"}
    parameters["properties"]["tags"] = {
        "type": "array",
        "items": {"anyOf": [{"type": "string"}, {"type": "null"}]},
    }
    case(
        "tools-xml_typed-local-ref-history",
        history,
        tools=refs,
        tool_call_format="xml_typed",
    )
    duplicate = deepcopy(tools)
    duplicate[0]["function"]["parameters"]["properties"]["label"]["type"] = "integer"
    case(
        "tools-xml_typed-duplicate-definition",
        history,
        tools=tools + duplicate,
        tool_call_format="xml_typed",
    )
