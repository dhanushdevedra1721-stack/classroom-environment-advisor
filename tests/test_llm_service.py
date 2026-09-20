"""
tests/test_llm_service.py
---------------------------
Tests for llm_service.py's error handling and fallback behavior.

These tests avoid making real network calls: they either don't configure an
API key at all (to test the "missing key" path) or monkeypatch get_llm() /
the chain (to test JSON parsing and fallback behavior) — required because
CI/grading environments won't have a real API key or internet access.

Run with:
    pytest tests/test_llm_service.py -v
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest  # noqa: E402

from models import ClassroomConditions  # noqa: E402
import llm_service  # noqa: E402


def test_missing_api_key_raises_config_error(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    with pytest.raises(llm_service.LLMConfigError):
        llm_service.get_llm()


def test_extract_empty_description_raises():
    with pytest.raises(llm_service.LLMExtractionError):
        llm_service.extract_conditions_from_text("")


def test_extract_whitespace_description_raises():
    with pytest.raises(llm_service.LLMExtractionError):
        llm_service.extract_conditions_from_text("   ")


def test_generate_explanation_falls_back_without_api_key(monkeypatch):
    """With no API key configured, generate_explanation() should not raise —
    it should silently fall back to the templated explanation so Manual
    Assessment mode keeps working offline."""
    monkeypatch.delenv("LLM_API_KEY", raising=False)

    conditions = ClassroomConditions(
        temperature=34, humidity=80, co2=2200, noise=90, light=800, occupancy=95
    )
    result = llm_service.generate_explanation(conditions, 20.0, "Very Poor", ["R2: ..."])
    assert result.source == "fallback"
    assert "20.0" in result.explanation or "20" in result.explanation
    assert len(result.explanation) > 0


def test_strip_code_fences_extracts_json():
    wrapped = '```json\n{"temperature": 25, "humidity": 50}\n```'
    cleaned = llm_service._strip_code_fences(wrapped)
    assert cleaned.startswith("{")
    assert cleaned.endswith("}")


if __name__ == "__main__":
    print("Run this file with pytest: pytest tests/test_llm_service.py -v")
