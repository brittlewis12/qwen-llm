# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""A scripted Open Responses server for client-replay captures (map #13):
answers in `qwen serve`'s item and SSE shapes and logs every request body.

Each assistant step returns a reasoning item with distinct text, so a later
request shows exactly which past reasoning the client replays:

- a title request (input mentions "Generate a title"): a plain message;
- a request whose last input item is a user message: reasoning
  `R-call-<n>`, then one `function_call` to a tool the client offered
  (`list`, else `glob`, else `read`), with arguments for `--dir`;
- a request whose last input item is a `function_call_output`: reasoning
  `R-answer-<n>`, then the message `Done <n>.`.

  uv run scripts/reference/opencode_tool_mock.py --port 18737 \\
    --log requests.jsonl --dir /tmp/workdir
"""

import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from itertools import count


def envelope(output, status="completed"):
    return {
        "id": f"resp_{time.time_ns()}",
        "object": "response",
        "created_at": int(time.time()),
        "model": "mock-model",
        "status": status,
        "output": output,
        "usage": {
            "input_tokens": 10,
            "output_tokens": 3,
            "total_tokens": 13,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0},
        },
        "incomplete_details": None,
        "error": None,
    }


def reasoning_events(index, item_id, text):
    item = {
        "id": item_id,
        "type": "reasoning",
        "status": "completed",
        "summary": [],
        "content": [{"type": "reasoning_text", "text": text}],
    }
    part = {"item_id": item_id, "output_index": index, "content_index": 0}
    return item, [
        (
            "response.output_item.added",
            {
                "output_index": index,
                "item": {**item, "status": "in_progress", "content": []},
            },
        ),
        (
            "response.content_part.added",
            {**part, "part": {"type": "reasoning_text", "text": ""}},
        ),
        ("response.reasoning_text.delta", {**part, "delta": text}),
        ("response.reasoning.done", {**part, "text": text}),
        (
            "response.content_part.done",
            {**part, "part": {"type": "reasoning_text", "text": text}},
        ),
        ("response.output_item.done", {"output_index": index, "item": item}),
    ]


def message_events(index, item_id, text):
    item = {
        "id": item_id,
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }
    part = {"item_id": item_id, "output_index": index, "content_index": 0}
    return item, [
        (
            "response.output_item.added",
            {
                "output_index": index,
                "item": {**item, "status": "in_progress", "content": []},
            },
        ),
        (
            "response.content_part.added",
            {**part, "part": {"type": "output_text", "text": "", "annotations": []}},
        ),
        ("response.output_text.delta", {**part, "delta": text}),
        ("response.output_text.done", {**part, "text": text}),
        (
            "response.content_part.done",
            {**part, "part": {"type": "output_text", "text": text, "annotations": []}},
        ),
        ("response.output_item.done", {"output_index": index, "item": item}),
    ]


def call_events(index, item_id, call_id, name, arguments):
    item = {
        "id": item_id,
        "type": "function_call",
        "status": "completed",
        "call_id": call_id,
        "name": name,
        "arguments": arguments,
    }
    return item, [
        (
            "response.output_item.added",
            {
                "output_index": index,
                "item": {**item, "status": "in_progress", "arguments": ""},
            },
        ),
        (
            "response.function_call_arguments.delta",
            {
                "item_id": item_id,
                "output_index": index,
                "call_id": call_id,
                "delta": arguments,
            },
        ),
        (
            "response.function_call_arguments.done",
            {
                "item_id": item_id,
                "output_index": index,
                "call_id": call_id,
                "arguments": arguments,
            },
        ),
        ("response.output_item.done", {"output_index": index, "item": item}),
    ]


def tool_choice(tools, workdir):
    names = {tool.get("name") for tool in tools or []}
    for name, arguments in (
        ("list", {"path": workdir}),
        ("glob", {"pattern": "*", "path": workdir}),
        ("read", {"filePath": f"{workdir}/a.txt"}),
    ):
        if name in names:
            return name, json.dumps(arguments)
    raise SystemExit(f"no usable tool among {sorted(n for n in names if n)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--log", required=True)
    parser.add_argument("--dir", required=True)
    args = parser.parse_args()
    steps = count(1)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):
            body = json.loads(
                self.rfile.read(int(self.headers.get("content-length", 0))) or b"null"
            )
            with open(args.log, "a") as log:
                log.write(json.dumps({"body": body}) + "\n")
            items = body.get("input") or []
            text = json.dumps(items)
            n = next(steps)
            output, events = [], []
            if "Generate a title" in text:
                item, ev = message_events(0, f"msg_t{n}", "Listing files")
                output, events = [item], ev
            elif items and items[-1].get("type") == "function_call_output":
                rs, ev1 = reasoning_events(0, f"rs_a{n}", f"R-answer-{n}")
                msg, ev2 = message_events(1, f"msg_a{n}", f"Done {n}.")
                output, events = [rs, msg], ev1 + ev2
            else:
                name, arguments = tool_choice(body.get("tools"), args.dir)
                rs, ev1 = reasoning_events(0, f"rs_c{n}", f"R-call-{n}")
                fc, ev2 = call_events(1, f"fc_c{n}", f"call_c{n}", name, arguments)
                output, events = [rs, fc], ev1 + ev2
            events = (
                [("response.created", {"response": envelope([], "in_progress")})]
                + events
                + [("response.completed", {"response": envelope(output)})]
            )
            if body.get("stream"):
                data = (
                    "".join(
                        f"event: {t}\ndata: {json.dumps({'type': t, 'sequence_number': i + 1, **p})}\n\n"
                        for i, (t, p) in enumerate(events)
                    )
                    + "data: [DONE]\n\n"
                )
                payload, ctype = data.encode(), "text/event-stream"
            else:
                payload, ctype = (
                    json.dumps(envelope(output)).encode(),
                    "application/json",
                )
            self.send_response(200)
            self.send_header("content-type", ctype)
            self.send_header("content-length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
