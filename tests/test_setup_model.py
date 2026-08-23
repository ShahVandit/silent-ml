"""Tests for model provisioning and endpoint verification (no network, no GPU)."""

from __future__ import annotations

import pytest

import silentml.agent.llm_policy as L
import silentml.setup_model as S


# --- model selection ---------------------------------------------------------
@pytest.mark.parametrize("vram,expected", [
    (80.0, "qwen3-coder:30b"),
    (24.0, "qwen3-coder:30b"),
    (23.9, "qwen2.5-coder:14b"),
    (12.0, "qwen2.5-coder:14b"),
    (6.0, "qwen2.5-coder:7b"),
    (2.0, "qwen2.5-coder:7b"),   # below every threshold -> smallest
    (None, "qwen2.5-coder:7b"),  # unknown VRAM -> smallest
])
def test_recommend_by_vram(vram, expected):
    assert S.recommend(vram).tag == expected


def test_models_ordered_largest_first():
    reqs = [m.min_free_vram_gb for m in S.QWEN_MODELS]
    assert reqs == sorted(reqs, reverse=True)


def test_pull_without_ollama_explains_install(monkeypatch):
    monkeypatch.setattr(S, "ollama_available", lambda: False)
    with pytest.raises(S.SetupError, match="ollama.com/install.sh"):
        S.pull("qwen3-coder:30b")


# --- endpoint verification ---------------------------------------------------
def _serve(monkeypatch, message, served="qwen3-coder:30b"):
    monkeypatch.setattr(S, "_get", lambda url, timeout=10.0:
                        {"data": [{"id": served}]})
    monkeypatch.setattr(L, "_post_json", lambda u, p, k, t:
                        {"choices": [{"message": message}]})


def test_native_tool_call_is_healthy(monkeypatch):
    _serve(monkeypatch, {"content": "", "tool_calls": [
        {"function": {"name": "read_artifact", "arguments": '{"name":"loss_curves"}'}}]})
    assert S.check_endpoint("http://x/v1", "qwen3-coder:30b") == []


def test_json_fallback_warns_but_works(monkeypatch):
    _serve(monkeypatch, {"content": '{"tool":"read_artifact","args":{"name":"x"}}'})
    (warning,) = S.check_endpoint("http://x/v1", "qwen3-coder:30b")
    assert "fallback" in warning


def test_no_parseable_tool_call_warns_loudly(monkeypatch):
    _serve(monkeypatch, {"content": "Sure, let me look at that."})
    (warning,) = S.check_endpoint("http://x/v1", "qwen3-coder:30b")
    assert "parse failures" in warning


def test_model_not_served_is_reported(monkeypatch):
    _serve(monkeypatch, {"content": "", "tool_calls": [
        {"function": {"name": "read_artifact", "arguments": "{}"}}]})
    warnings = S.check_endpoint("http://x/v1", "some-other-model")
    assert any("not in the endpoint's list" in w for w in warnings)


def test_unreachable_endpoint_is_fatal(monkeypatch):
    def boom(url, timeout=10.0):
        raise OSError("connection refused")

    monkeypatch.setattr(S, "_get", boom)
    with pytest.raises(S.SetupError, match="ollama serve"):
        S.check_endpoint("http://x/v1", "qwen3-coder:30b")
