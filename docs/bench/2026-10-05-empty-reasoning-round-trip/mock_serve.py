import json, sys, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
PORT, LOG, TEXT = int(sys.argv[1]), sys.argv[2], (sys.argv[3] if len(sys.argv) > 3 else "")
RS = {"id": "rs_e1", "type": "reasoning", "status": "completed", "summary": [], "content": [{"type": "reasoning_text", "text": TEXT}]}
MSG = {"id": "msg_e1", "type": "message", "role": "assistant", "status": "completed", "content": [{"type": "output_text", "text": "Hi.", "annotations": []}]}
def env(output, status="completed"):
    return {"id": "resp_1", "object": "response", "created_at": int(time.time()), "model": "mock-model", "status": status, "output": output,
            "usage": {"input_tokens": 10, "output_tokens": 3, "total_tokens": 13, "input_tokens_details": {"cached_tokens": 0}, "output_tokens_details": {"reasoning_tokens": 0}},
            "incomplete_details": None, "error": None}
class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("content-length", 0))) or b"null")
        open(LOG, "a").write(json.dumps({"body": body}) + "\n")
        ev = [("response.created", {"response": env([], "in_progress")}),
              ("response.output_item.added", {"output_index": 0, "item": {**RS, "status": "in_progress", "content": []}}),
              ("response.content_part.added", {"item_id": "rs_e1", "output_index": 0, "content_index": 0, "part": {"type": "reasoning_text", "text": ""}})]
        if TEXT:
            ev.append(("response.reasoning_text.delta", {"item_id": "rs_e1", "output_index": 0, "content_index": 0, "delta": TEXT}))
        ev += [("response.reasoning.done", {"item_id": "rs_e1", "output_index": 0, "content_index": 0, "text": TEXT}),
               ("response.content_part.done", {"item_id": "rs_e1", "output_index": 0, "content_index": 0, "part": {"type": "reasoning_text", "text": TEXT}}),
               ("response.output_item.done", {"output_index": 0, "item": RS}),
               ("response.output_item.added", {"output_index": 1, "item": {**MSG, "status": "in_progress", "content": []}}),
               ("response.content_part.added", {"item_id": "msg_e1", "output_index": 1, "content_index": 0, "part": {"type": "output_text", "text": "", "annotations": []}}),
               ("response.output_text.delta", {"item_id": "msg_e1", "output_index": 1, "content_index": 0, "delta": "Hi."}),
               ("response.output_text.done", {"item_id": "msg_e1", "output_index": 1, "content_index": 0, "text": "Hi."}),
               ("response.content_part.done", {"item_id": "msg_e1", "output_index": 1, "content_index": 0, "part": {"type": "output_text", "text": "Hi.", "annotations": []}}),
               ("response.output_item.done", {"output_index": 1, "item": MSG}),
               ("response.completed", {"response": env([RS, MSG])})]
        if body and body.get("stream"):
            data = "".join(f"event: {t}\ndata: {json.dumps({'type': t, 'sequence_number': i + 1, **p})}\n\n" for i, (t, p) in enumerate(ev)) + "data: [DONE]\n\n"
            payload, ctype = data.encode(), "text/event-stream"
        else:
            payload, ctype = json.dumps(env([RS, MSG])).encode(), "application/json"
        self.send_response(200); self.send_header("content-type", ctype); self.send_header("content-length", str(len(payload))); self.end_headers(); self.wfile.write(payload)
ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
