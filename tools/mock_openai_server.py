"""Offline OpenAI-compatible mock server for exercising vf-eval end-to-end.

Serves POST /v1/chat/completions. The requested model name selects a scripted
policy, so the whole eval loop (env -> client -> "model" -> parser -> rubric)
runs deterministically with zero API cost:

  oracle    always answers the correct label            -> expect ~100%
  strong-80 correct on a deterministic 80% of cases     -> expect ~80%
  weak-40   correct on a deterministic 40% of cases     -> expect ~40%
  guesser   always answers the same constant label      -> expect ~1/15
  no-format answers correctly but without <answer> tags -> expect 0%

Correctness is decided per case via md5(symptom|model) so repeat runs give
identical scores (determinism check, guide step 3.4).

Usage:
    python tools/mock_openai_server.py [port]
"""

import hashlib
import json
import re
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from vf_ardu_quiz import CASES, TAXONOMY

SYMPTOM_RE = re.compile(r"Symptoms from the dataflash log: (.*)\nGive the diagnosis", re.DOTALL)


def find_truth(user_content: str) -> str | None:
    match = SYMPTOM_RE.search(user_content)
    if not match:
        return None
    symptom = match.group(1).strip()
    for case_symptom, label in CASES:
        if case_symptom == symptom:
            return label
    return None


def deterministic_pct(symptom: str, model: str) -> int:
    digest = hashlib.md5(f"{symptom}|{model}".encode()).hexdigest()
    return int(digest, 16) % 100


def wrong_label(truth: str, symptom: str) -> str:
    idx = int(hashlib.md5(symptom.encode()).hexdigest(), 16) % len(TAXONOMY)
    label = TAXONOMY[idx]
    if label == truth:
        label = TAXONOMY[(idx + 1) % len(TAXONOMY)]
    return label


def answer_for(model: str, user_content: str) -> str:
    truth = find_truth(user_content)
    if truth is None:
        return "<answer>unknown_case</answer>"
    symptom = SYMPTOM_RE.search(user_content).group(1).strip()

    if model == "oracle":
        label = truth
    elif model.startswith(("strong-", "weak-")):
        rate = int(model.split("-", 1)[1])
        correct = deterministic_pct(symptom, model) < rate
        label = truth if correct else wrong_label(truth, symptom)
    elif model == "guesser":
        label = TAXONOMY[0]
    elif model == "no-format":
        return f"The most likely root cause is {truth}."
    else:
        label = wrong_label(truth, symptom)

    return f"Looking at the log evidence...\n<answer>{label}</answer>"


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass  # keep vf-eval output readable

    def _json(self, code: int, payload: dict):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if not self.path.endswith("/chat/completions"):
            self._json(404, {"error": {"message": f"unknown path {self.path}"}})
            return
        length = int(self.headers.get("Content-Length", 0))
        request = json.loads(self.rfile.read(length))
        model = request.get("model", "oracle")
        user_content = next(
            (m["content"] for m in reversed(request.get("messages", []))
             if m.get("role") == "user"),
            "",
        )
        content = answer_for(model, user_content)
        self._json(200, {
            "id": "chatcmpl-mock",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        })


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8077
    print(f"mock OpenAI server on http://127.0.0.1:{port}/v1")
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
