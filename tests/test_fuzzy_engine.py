"""
tests/test_fuzzy_engine.py
---------------------------
Unit tests for the fuzzy inference engine (fuzzy_engine.py) and the
Pydantic validation in models.py.

Run with:
    pytest tests/test_fuzzy_engine.py -v

These tests check *behavior/ranges*, not exact scores, since a fuzzy
system's precise output depends on the shape of the membership functions
and isn't meant to be pinned to one "magic number" (see project brief,
section 17).
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models import ClassroomConditions  # noqa: E402
from fuzzy_engine import run_fuzzy_inference  # noqa: E402


def test_comfortable_classroom_scores_high():
    """Test Case 1: comfortable conditions across the board -> high score."""
    conditions = ClassroomConditions(
        temperature=23, humidity=45, co2=450, noise=35, light=400, occupancy=30
    )
    result = run_fuzzy_inference(conditions)
    assert result.score >= 60, f"Expected a relatively high score, got {result.score}"
    assert result.category in ("Good", "Excellent")


def test_hot_noisy_crowded_classroom_scores_low():
    """Test Case 2: hot, high CO2, noisy, crowded -> low score."""
    conditions = ClassroomConditions(
        temperature=34, humidity=80, co2=2200, noise=90, light=800, occupancy=95
    )
    result = run_fuzzy_inference(conditions)
    assert result.score <= 45, f"Expected a relatively low score, got {result.score}"
    assert result.category in ("Very Poor", "Poor")


def test_moderate_classroom_scores_medium():
    """Test Case 3: middling conditions -> medium/good score."""
    conditions = ClassroomConditions(
        temperature=26, humidity=55, co2=900, noise=55, light=450, occupancy=55
    )
    result = run_fuzzy_inference(conditions)
    assert 35 <= result.score <= 85, f"Expected a medium/good score, got {result.score}"


def test_excellent_classroom_scores_very_high():
    """Test Case 4: near-ideal conditions -> high score, low occupancy, quiet, low CO2."""
    conditions = ClassroomConditions(
        temperature=22, humidity=45, co2=400, noise=30, light=450, occupancy=20
    )
    result = run_fuzzy_inference(conditions)
    assert result.score >= 65, f"Expected a high score, got {result.score}"
    assert result.category in ("Good", "Excellent")


def test_score_is_always_within_bounds():
    """Regardless of input, the score must stay within [0, 100]."""
    extremes = [
        (10, 0, 300, 20, 0, 0),
        (45, 100, 3000, 100, 1000, 100),
    ]
    for temperature, humidity, co2, noise, light, occupancy in extremes:
        conditions = ClassroomConditions(
            temperature=temperature, humidity=humidity, co2=co2,
            noise=noise, light=light, occupancy=occupancy,
        )
        result = run_fuzzy_inference(conditions)
        assert 0 <= result.score <= 100


def test_rule_strengths_are_populated():
    """At least one rule should fire for a typical classroom, and every
    reported strength should be a valid membership degree in [0, 1]."""
    conditions = ClassroomConditions(
        temperature=23, humidity=45, co2=450, noise=35, light=400, occupancy=30
    )
    result = run_fuzzy_inference(conditions)
    assert len(result.rule_strengths) > 0
    for strength in result.rule_strengths.values():
        assert 0 < strength <= 1


def test_out_of_range_inputs_are_clamped_not_rejected():
    """Values outside the declared range should be clamped by the Pydantic
    validator instead of raising, so a slightly-off LLM extraction doesn't
    crash the app."""
    conditions = ClassroomConditions(
        temperature=999, humidity=-50, co2=50, noise=5, light=-10, occupancy=500
    )
    assert conditions.temperature == 45
    assert conditions.humidity == 0
    assert conditions.co2 == 300
    assert conditions.noise == 20
    assert conditions.light == 0
    assert conditions.occupancy == 100


def test_missing_required_field_raises():
    """Pydantic should reject conditions missing a required field."""
    import pytest

    with pytest.raises(Exception):
        ClassroomConditions(temperature=23, humidity=45, co2=450, noise=35, light=400)  # missing occupancy


if __name__ == "__main__":
    # Allow running without pytest installed: `python tests/test_fuzzy_engine.py`
    tests = [
        test_comfortable_classroom_scores_high,
        test_hot_noisy_crowded_classroom_scores_low,
        test_moderate_classroom_scores_medium,
        test_excellent_classroom_scores_very_high,
        test_score_is_always_within_bounds,
        test_rule_strengths_are_populated,
        test_out_of_range_inputs_are_clamped_not_rejected,
    ]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"PASS: {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL: {t.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            print(f"ERROR: {t.__name__}: {e}")
    print(f"\n{passed}/{len(tests)} tests passed")
