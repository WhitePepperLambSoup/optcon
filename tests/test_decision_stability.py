"""Tests for conservative decision-stability margins."""

from __future__ import annotations

import pytest

from optcon import (
    ChoiceStability,
    DecisionStability,
    ErrorBudget,
    choice_stability,
    threshold_stability,
)


def test_threshold_stability_subtracts_component_error_budget():
    result = threshold_stability(
        1.5,
        1.0,
        numerical_error=0.1,
        reference_error=0.1,
        parameter_error=0.05,
    )

    assert isinstance(result, DecisionStability)
    assert result.signed_distance == pytest.approx(0.5)
    assert result.budget.total == pytest.approx(0.25)
    assert result.margin == pytest.approx(0.25)
    assert result.stable is True
    assert result.decision_passes is True


def test_at_most_threshold_uses_the_reverse_signed_distance():
    result = threshold_stability(0.8, 1.0, direction="at_most")

    assert result.direction == "at_most"
    assert result.signed_distance == pytest.approx(0.2)
    assert result.margin == pytest.approx(0.2)
    assert result.stable is True


@pytest.mark.parametrize(
    ("value", "boundary", "error_budget"),
    [(1.0, 1.0, 0.0), (1.01, 1.0, 0.02)],
)
def test_zero_or_negative_residual_margin_is_unstable(
    value: float, boundary: float, error_budget: float
):
    result = threshold_stability(value, boundary, reference_error=error_budget)

    assert result.margin <= 0.0
    assert result.stable is False


def test_choice_stability_reports_the_best_and_runner_up():
    result = choice_stability({"six-pair": 0.96, "seven-pair": 0.94}, error_budget=0.01)

    assert isinstance(result, ChoiceStability)
    assert result.selected == "six-pair"
    assert result.runner_up == "seven-pair"
    assert result.gap == pytest.approx(0.02)
    assert result.margin == pytest.approx(0.01)
    assert result.stable is True


def test_choice_stability_is_deterministic_for_ties_and_not_stable():
    result = choice_stability({"b": 1.0, "a": 1.0}, error_budget=0.0)

    assert (result.selected, result.runner_up) == ("a", "b")
    assert result.margin == pytest.approx(0.0)
    assert result.stable is False


def test_error_budget_rejects_negative_and_nonfinite_components():
    with pytest.raises(ValueError, match="non-negative"):
        ErrorBudget(reference=-1.0)
    with pytest.raises(ValueError, match="finite"):
        ErrorBudget(parameter=float("nan"))


@pytest.mark.parametrize("direction", ["sideways", "", "AT_LEAST"])
def test_threshold_stability_rejects_unknown_directions(direction: str):
    with pytest.raises(ValueError, match="direction"):
        threshold_stability(1.0, 0.5, direction=direction)


def test_choice_stability_requires_two_finite_candidates():
    with pytest.raises(ValueError, match="at least two"):
        choice_stability({"only": 1.0})
    with pytest.raises(ValueError, match="finite"):
        choice_stability({"a": 1.0, "b": float("inf")})
