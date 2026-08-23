"""Tests for LLM tool-call parsing (no network: the HTTP layer is stubbed)."""

from __future__ import annotations

import json

import pytest

import silentml.agent.llm_policy as L
from silentml.agent.llm_policy import LLMPolicy, _extract_json_object, _normalise_action


def _reply(content: str = "", tool_calls=None) -> dict:
    msg: dict = {"content": content}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return {"choices": [{"message": msg}]}


def _stub(monkeypatch, responses):
    """Serve canned responses in order to the policy's HTTP call."""
    seq = list(responses)

    def fake_post(url, payload, api_key, timeout):
        return seq.pop(0) if seq else _reply("done")

    monkeypatch.setattr(L, "_post_json", fake_post)


# --- JSON recovery -----------------------------------------------------------
def test_extract_plain_json():
    obj = _extract_json_object('I will look. {"tool": "view_code", "args": {"start": 1}}')
    assert obj["tool"] == "view_code"


def test_extract_fenced_json():
    text = 'Reasoning...\n```json\n{"tool": "read_artifact", "args": {"name": "loss_curves"}}\n```'
    assert _extract_json_object(text)["tool"] == "read_artifact"


def test_extract_none_when_absent():
    assert _extract_json_object("I think the learning rate is wrong.") is None


def test_normalise_rejects_unknown_tool():
    assert _normalise_action({"tool": "rm_rf", "args": {}}) is None


def test_normalise_parses_string_arguments():
    action = _normalise_action({"name": "view_code", "arguments": '{"start": 3, "end": 9}'})
    assert action == {"tool": "view_code", "args": {"start": 3, "end": 9}}


# --- policy behaviour --------------------------------------------------------
def test_native_tool_call(monkeypatch):
    _stub(monkeypatch, [_reply(tool_calls=[
        {"function": {"name": "read_artifact",
                      "arguments": json.dumps({"name": "loss_curves"})}}
    ])])
    action = LLMPolicy()("prompt", [])
    assert action == {"tool": "read_artifact", "args": {"name": "loss_curves"}}


def test_content_fallback_when_no_tool_calls(monkeypatch):
    _stub(monkeypatch, [_reply('{"tool": "view_code", "args": {"start": 1, "end": 40}}')])
    action = LLMPolicy()("prompt", [])
    assert action["tool"] == "view_code"


def test_retry_then_default_on_unparseable(monkeypatch):
    _stub(monkeypatch, [_reply("no tool call here"), _reply("still prose")])
    policy = LLMPolicy()
    action = policy("prompt", [])
    # Falls back to a harmless read so the episode still progresses.
    assert action["tool"] == "view_code"
    assert policy.parse_failures == 2


def test_observations_are_fed_back(monkeypatch):
    _stub(monkeypatch, [_reply('{"tool": "submit", "args": {"diagnosis": "lr"}}')])
    policy = LLMPolicy()
    history = [{"tool": "read_artifact", "observation": "GRAD_STATS_HERE", "ok": True}]
    policy("prompt", history)
    assert any("GRAD_STATS_HERE" in m["content"] for m in policy.messages)


def test_transport_error_propagates(monkeypatch):
    def boom(url, payload, api_key, timeout):
        raise L.LLMError("connection refused")

    monkeypatch.setattr(L, "_post_json", boom)
    with pytest.raises(L.LLMError):
        LLMPolicy()("prompt", [])


# --- operator anchors (no episode/training needed) ---------------------------
def test_all_operator_anchors_apply_to_template():
    """Catches anchor drift immediately instead of after a long generation run."""
    from pathlib import Path

    from silentml.bugs.operators import operators_for

    src = (Path(__file__).resolve().parents[1]
           / "silentml" / "pipelines" / "transformer_text" / "pipeline.py"
           ).read_text(encoding="utf-8")
    for op in operators_for("transformer_text"):
        assert op.inject(src) != src, f"{op.id} anchor did not change the template"
