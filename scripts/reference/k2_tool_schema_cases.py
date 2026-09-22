"""Schema presentation branches in the immutable IFM template."""


def add_schema_cases(case):
    user = [{"role": "user", "content": "Use the documented schema."}]
    schemas = {
        "annotations": {
            "name": "inspect",
            "description": "Line one.\nLine two & <three>.",
            "parameters": {
                "type": "object",
                "properties": {
                    "metadata": {
                        "type": "object",
                        "description": "A\nB",
                        "title": "T",
                        "examples": [{"a": " x\n y"}],
                        "properties": {
                            "flag": True,
                            "forbidden": False,
                            "empty": {
                                "type": "string",
                                "enum": ["", "a\nb", 2, None],
                                "default": "",
                            },
                        },
                        "patternProperties": {
                            "^x.*": {"type": "integer", "minimum": 0},
                            "other": False,
                        },
                        "additionalProperties": {
                            "type": "array",
                            "items": {"type": ["string", "null"]},
                        },
                        "returns": {"type": "number", "default": 1e-7},
                        "required": ["flag"],
                    },
                    "bounds": {
                        "type": "number",
                        "minimum": -1.5,
                        "maximum": 1e20,
                        "multipleOf": 0.1,
                    },
                },
                "required": ["metadata"],
            },
            "returns": {
                "type": "object",
                "description": "Return\nvalue",
                "properties": {"ok": {"type": "boolean", "default": True}},
            },
        },
        "root-variants": {
            "name": "choose",
            "parameters": {
                "description": "Select\none",
                "title": "Choices",
                "default": {},
                "enum": [None, "", 1],
                "anyOf": [
                    {"type": "object", "properties": {"x": {"type": "string"}}},
                    False,
                ],
                "items": [True, {"type": "number"}],
                "patternProperties": "literal pattern",
                "additionalProperties": False,
                "returns": [1, "two"],
            },
            "response": {"type": "string", "description": "legacy response"},
        },
        "reference-chain": {
            "name": "chain",
            "parameters": {
                "$ref": "#/$defs/A",
                "$defs": {
                    "A": {"$ref": "#/$defs/B", "description": "A"},
                    "B": {
                        "type": "object",
                        "properties": {"next": {"$ref": "#/$defs/A"}},
                    },
                },
                "properties": {
                    "first": {"$ref": "#/$defs/A", "title": "Sibling"},
                    "again": {"$ref": "#/$defs/A"},
                },
            },
        },
        "boolean-parameters": {"name": "any", "parameters": True, "returns": False},
        "container-fallback": {
            "name": "annotated",
            "parameters": {"type": "object", "properties": {}},
            "extra": {"must": "survive"},
        },
        "external-ref": {
            "name": "external",
            "parameters": {"$ref": "https://example.invalid/schema"},
        },
        "whitespace": {
            "name": "spaces",
            "parameters": {
                "type": "object",
                "properties": {
                    "s": {
                        "type": "string",
                        "title": " a\x1c b\u00a0c ",
                        "enum": ["'\\\"\n", {"value": " a\n b "}],
                        "default": {"x": " a\n b "},
                    }
                },
            },
        },
    }
    for name, tool in schemas.items():
        for presentation in ["markdown", "xml", "json"]:
            case(
                f"schema-{name}-{presentation}",
                user,
                tools=[tool],
                tool_presentation_format=presentation,
            )
    good = {
        "name": "ok",
        "parameters": {"type": "object", "properties": {"x": {"type": "string"}}},
    }
    bad = {"name": "bad", "parameters": {"required": ["missing"]}}
    for presentation in ["markdown", "xml", "json"]:
        case(
            f"schema-whole-set-fallback-{presentation}",
            user,
            tools=[good, schemas["container-fallback"]],
            tool_presentation_format=presentation,
        )
        case(
            f"schema-fallback-must-still-validate-{presentation}",
            user,
            tools=[schemas["container-fallback"], bad],
            tool_presentation_format=presentation,
        )
        case(
            f"schema-missing-required-{presentation}",
            user,
            tools=[bad],
            tool_presentation_format=presentation,
        )
        case(
            f"schema-string-required-{presentation}",
            user,
            tools=[{"name": "bad", "parameters": {"required": "x"}}],
            tool_presentation_format=presentation,
        )
        case(
            f"schema-undefined-required-{presentation}",
            user,
            tools=[
                {
                    "name": "bad",
                    "parameters": {"properties": {"x": {}}, "required": ["y"]},
                }
            ],
            tool_presentation_format=presentation,
        )
