"""LLM-backed policy for the episode loop.

Targets any **OpenAI-compatible** chat endpoint, so the same code drives a local
Qwen served by Ollama or vLLM (free, no API key) as well as a hosted model. Only
the standard library is used for HTTP, so the benchmark adds no dependencies.

Tool calls are read from the native ``tool_calls`` field when the server emits
them, and otherwise recovered from a JSON object in the message content. That
fallback matters in practice: open models served through Ollama vary in how
reliably they emit structured tool calls, and a benchmark that scored parse
failures as debugging failures would measure the wrong thing.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

DEFAULT_BASE_URL = "http://localhost:11434/v1"   # Ollama's OpenAI-compatible port
DEFAULT_MODEL = "qwen3-coder:30b"

TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "read_artifact",
            "description": "Read a precomputed diagnostic artifact (JSON).",
            "parameters": {
                "type": "object",
                "properties": {"name": {"type": "string",
                                        "description": "Artifact name from the manifest."}},
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "view_code",
            "description": "View numbered source lines of the pipeline.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file": {"type": "string", "description": "Default 'pipeline.py'."},
                    "start": {"type": "integer", "description": "First line (1-indexed)."},
                    "end": {"type": "integer", "description": "Last line."},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "apply_patch",
            "description": (
                "Apply a unified diff to pipeline.py. The diff must contain a @@ hunk "
                "header, context lines starting with a space, removals with '-', and "
                "additions with '+'."
            ),
            "parameters": {
                "type": "object",
                "properties": {"diff": {"type": "string", "description": "Unified diff."}},
                "required": ["diff"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "execute_code",
            "description": "Run a CPU-only Python script (30s limit); returns stdout/stderr.",
            "parameters": {
                "type": "object",
                "properties": {"script": {"type": "string"}},
                "required": ["script"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit",
            "description": (
                "End the episode. Requires at least one successful apply_patch. "
                'Pass a JSON string: {"diagnosis": "...", "supporting_evidence": [...]}'
            ),
            "parameters": {
                "type": "object",
                "properties": {"diagnosis": {"type": "string"}},
                "required": ["diagnosis"],
            },
        },
    },
]

_VALID_TOOLS = {t["function"]["name"] for t in TOOL_SCHEMAS}

PATCH_HINT = """\

When you call apply_patch, the diff must apply to the current file contents. Example:

@@ -12,3 +12,3 @@
     "batch_size": 32,
-    "lr": 3e-6,
+    "lr": 3e-4,
     "epochs": 8,

View the exact lines with view_code first so your context lines match verbatim."""


class LLMError(RuntimeError):
    pass


def _post_json(url: str, payload: dict, api_key: str | None, timeout: float) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise LLMError(f"HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:500]}") from e
    except urllib.error.URLError as e:
        raise LLMError(f"cannot reach {url}: {e.reason}") from e


def _extract_json_object(text: str) -> dict | None:
    """Recover a {"tool": ..., "args": ...} object from free-form model output."""
    fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidates = list(fenced)
    depth, start = 0, None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                candidates.append(text[start:i + 1])
                start = None
    for cand in candidates:
        try:
            obj = json.loads(cand)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and ("tool" in obj or "name" in obj):
            return obj
    return None


def _normalise_action(obj: dict) -> dict | None:
    tool = obj.get("tool") or obj.get("name")
    if tool not in _VALID_TOOLS:
        return None
    args = obj.get("args") or obj.get("arguments") or obj.get("parameters") or {}
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = {}
    return {"tool": tool, "args": args if isinstance(args, dict) else {}}


@dataclass
class LLMPolicy:
    """Stateful policy: keeps the chat transcript across tool calls in an episode."""

    model: str = DEFAULT_MODEL
    base_url: str = DEFAULT_BASE_URL
    api_key: str | None = None
    temperature: float = 0.0
    timeout: float = 180.0
    use_native_tools: bool = True
    messages: list[dict] = field(default_factory=list, init=False)
    _last_seen: int = field(default=0, init=False)
    parse_failures: int = field(default=0, init=False)

    def reset(self) -> None:
        self.messages = []
        self._last_seen = 0
        self.parse_failures = 0

    def __call__(self, prompt: str, history: list[dict]) -> dict:
        if not self.messages:
            self.messages.append({
                "role": "system",
                "content": (
                    "You are an expert ML debugging agent. Investigate using the "
                    "provided tools, then fix the bug with apply_patch and finish "
                    "with submit. Call exactly one tool per turn." + PATCH_HINT
                ),
            })
            self.messages.append({"role": "user", "content": prompt})

        # Feed back any observations produced since the last model turn.
        for step in history[self._last_seen:]:
            obs = step["observation"]
            self.messages.append({
                "role": "user",
                "content": f"Result of {step['tool']}:\n{obs[:6000]}",
            })
        self._last_seen = len(history)

        action = self._request_action()
        if action is None:
            self.parse_failures += 1
            self.messages.append({
                "role": "user",
                "content": (
                    "That response contained no valid tool call. Reply with a single "
                    'JSON object only, e.g. {"tool": "view_code", "args": {"start": 1, '
                    '"end": 60}}. Valid tools: ' + ", ".join(sorted(_VALID_TOOLS))
                ),
            })
            # One retry; if it still fails, view code so the episode makes progress.
            action = self._request_action()
            if action is None:
                self.parse_failures += 1
                return {"tool": "view_code", "args": {"start": 1, "end": 80}}
        return action

    def _request_action(self) -> dict | None:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": self.messages,
            "temperature": self.temperature,
            "stream": False,
        }
        if self.use_native_tools:
            payload["tools"] = TOOL_SCHEMAS

        resp = _post_json(f"{self.base_url.rstrip('/')}/chat/completions",
                          payload, self.api_key, self.timeout)
        try:
            message = resp["choices"][0]["message"]
        except (KeyError, IndexError) as e:
            raise LLMError(f"unexpected response shape: {str(resp)[:400]}") from e
        self.messages.append({
            "role": "assistant",
            "content": message.get("content") or "",
        })

        # Preferred path: the server emitted a structured tool call.
        for call in message.get("tool_calls") or []:
            fn = call.get("function", {})
            action = _normalise_action({"name": fn.get("name"),
                                        "arguments": fn.get("arguments", {})})
            if action:
                return action

        # Fallback: recover a JSON action from the message body.
        content = message.get("content") or ""
        obj = _extract_json_object(content)
        if obj:
            return _normalise_action(obj)
        return None
