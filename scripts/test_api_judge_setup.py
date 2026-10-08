"""API-only runs must not silently fall back to a large local reward model."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lab22 import judge as J


def test_api_only_rejects_local_provider(monkeypatch):
    assert hasattr(J, "require_api_judge"), "Missing API-only preflight"
    with pytest.raises(RuntimeError, match="API"):
        J.require_api_judge("rm", "")


def test_api_only_rejects_missing_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert hasattr(J, "require_api_judge"), "Missing API-only preflight"
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        J.require_api_judge("openai", "test-model")


def test_api_only_rejects_missing_model(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    assert hasattr(J, "require_api_judge"), "Missing API-only preflight"
    with pytest.raises(RuntimeError, match="JUDGE_MODEL"):
        J.require_api_judge("openai", "")


def test_api_only_accepts_configured_provider(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    assert hasattr(J, "require_api_judge"), "Missing API-only preflight"
    J.require_api_judge("openai", "test-model")


def test_api_sanity_counts_correct_and_position_inconsistent_pairs():
    assert hasattr(J, "api_sanity_accuracy"), "API judge should be checked on Vietnamese sanity pairs"
    # Contract of the external judge: each prompt has bad answer as SFT and good as DPO.
    replies = iter(['{"winner":"B"}', '{"winner":"A"}', '{"winner":"A"}', '{"winner":"A"}'])
    pairs = [("1+1?", "2", "3"), ("2+2?", "4", "5")]
    result = J.api_sanity_accuracy(lambda system, user: next(replies), pairs=pairs)
    assert result["n"] == 2
    assert result["accuracy"] == 0.5
    assert result["position_consistency"] == 0.5
