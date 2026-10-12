"""Actual model-serving client plus explicitly labelled engineering replay."""
from __future__ import annotations
import json
import os
import urllib.request
from urllib.parse import urlsplit


class ReplayPolicy:
    """Teacher replay only. It does not represent learned model performance."""
    def __init__(self, actions):
        self.actions = iter(actions)

    def generate(self, messages):
        return json.dumps(next(self.actions), ensure_ascii=False)


class HTTPPolicy:
    def __init__(self, base_url="http://127.0.0.1:8000/v1", model="tool-r1", max_tokens=512, temperature=0.0, seed=42, timeout=120.0):
        parsed = urlsplit(base_url)
        if parsed.scheme not in ("http", "https") or parsed.username or parsed.password:
            raise ValueError("Use an HTTP(S) base URL without embedded credentials")
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model, self.max_tokens, self.temperature = model, max_tokens, temperature
        self.seed, self.timeout, self.calls = seed, timeout, 0
        self.last_usage = {}
        self.last_finish_reason = None

    def generate(self, messages):
        payload = {"model": self.model, "messages": messages, "temperature": self.temperature,
                   "max_tokens": self.max_tokens, "seed": self.seed + self.calls,
                   "stream": False}
        headers = {"Content-Type": "application/json"}
        key = os.environ.get("TOOL_R1_API_KEY")
        if key:
            headers["Authorization"] = "Bearer " + key
        request = urllib.request.Request(self.url, json.dumps(payload).encode(), headers=headers)
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            raw = response.read(2_000_001)
        if len(raw) > 2_000_000:
            raise ValueError("Model server response exceeded size limit")
        result = json.loads(raw)
        content = result["choices"][0]["message"]["content"]
        if not isinstance(content, str):
            raise ValueError("Server returned no assistant text")
        self.last_usage = result.get("usage", {})
        self.last_finish_reason = result["choices"][0].get("finish_reason")
        self.calls += 1
        return content
