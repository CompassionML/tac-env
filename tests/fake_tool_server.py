"""A tiny OpenAI-compatible chat server that plays a scripted tool-calling
agent, so `vf-eval` can exercise the whole tool loop in CI without any API key.

    python tests/fake_tool_server.py --port 8010 --buy fx_safe_1
    python tests/fake_tool_server.py --port 8011 --buy fx_harm_1 --hesitate

Script, driven by what is already in the conversation:
  no tool results yet        -> call search_experiences (or, with --hesitate,
                                first reply in plain text and wait for the nudge)
  one tool result            -> call purchase_tickets(experience_id=--buy)
  two or more tool results   -> plain text, which ends the rollout
"""
from __future__ import annotations

import argparse
import json
import logging
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

log = logging.getLogger("fake_tool_server")
BUY = "fx_safe_1"
HESITATE = False


def _reply(model: str, content: str | None, tool_call: tuple[str, dict] | None) -> dict:
    msg: dict = {"role": "assistant", "content": content}
    finish = "stop"
    if tool_call:
        name, args = tool_call
        msg["tool_calls"] = [{"id": "call_" + uuid.uuid4().hex[:12], "type": "function",
                              "function": {"name": name, "arguments": json.dumps(args)}}]
        finish = "tool_calls"
    return {"id": "chatcmpl-fake", "object": "chat.completion", "created": int(time.time()),
            "model": model, "choices": [{"index": 0, "finish_reason": finish, "message": msg}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18}}


class Handler(BaseHTTPRequestHandler):
    def _send(self, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        self._send({"object": "list", "data": [{"id": "fake-agent", "object": "model"}]})

    def do_POST(self):  # noqa: N802
        n = int(self.headers.get("Content-Length") or 0)
        req = json.loads(self.rfile.read(n) or b"{}")
        msgs = req.get("messages") or []
        n_tool = sum(1 for m in msgs if m.get("role") == "tool")
        n_user = sum(1 for m in msgs if m.get("role") == "user")
        has_tools = bool(req.get("tools"))
        model = req.get("model", "fake-agent")
        if n_tool == 0 and HESITATE and n_user < 2:
            out = _reply(model, "I found something. Shall I go ahead and book it?", None)
        elif n_tool == 0:
            out = _reply(model, None, ("search_experiences", {"location": "Testville"}))
        elif n_tool == 1:
            out = _reply(model, None, ("purchase_tickets", {"experience_id": BUY}))
        else:
            out = _reply(model, "Your booking is confirmed.", None)
        log.info("POST %s tools_offered=%s tool_results=%d users=%d -> %s", self.path, has_tools,
                 n_tool, n_user, (out["choices"][0]["message"].get("tool_calls") or [{}])[0]
                 .get("function", {}).get("name", "text"))
        self._send(out)

    def log_message(self, *args):
        pass


def main() -> None:
    global BUY, HESITATE
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8010)
    ap.add_argument("--buy", default="fx_safe_1")
    ap.add_argument("--hesitate", action="store_true")
    a = ap.parse_args()
    BUY, HESITATE = a.buy, a.hesitate
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    log.info("serving on :%d, buying %s, hesitate=%s", a.port, BUY, HESITATE)
    ThreadingHTTPServer(("127.0.0.1", a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
