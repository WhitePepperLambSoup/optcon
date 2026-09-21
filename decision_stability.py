"""Conservative margins for numerical decisions near a declared boundary."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass


def _finite_nonnegative(value: float, name: str) -> float:
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be finite")
    if converted < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return converted


def _finite(value: float, name: str) -> float:
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be finite")
    return converted


@dataclass(frozen=True)
class ErrorBudget:
    """Conservative additive error budget for one decision quantity."""

    numerical: float = 0.0
    reference: float = 0.0
    parameter: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "numerical", _finite_nonnegative(self.numerical, "numerical"))
        object.__setattr__(self, "reference", _finite_nonnegative(self.reference, "reference"))
        object.__setattr__(self, "parameter", _finite_nonnegative(self.parameter, "parameter"))

    @property
    def total(self) -> float:
        """Return the conservative sum of all declared error components."""
        return self.numerical + self.reference + self.parameter


@dataclass(frozen=True)
class DecisionStability:
    """Stability result for a scalar threshold decision."""

    decision_value: float
    boundary: float
    budget: ErrorBudget
    signed_distance: float
    margin: float
    stable: bool
    direction: str

    @property
    def decision_passes(self) -> bool:
        """Whether the unadjusted value is on the accepted side of the boundary."""
        return self.signed_distance >= 0.0

    @property
    def error_budget(self) -> float:
        """Return the total conservative error budget."""
        return self.budget.total


@dataclass(frozen=True)
class ChoiceStability:
    """Stability result for a two-best-candidate comparison."""

    selected: str
    runner_up: str
    gap: float
    error_budget: float
    margin: float
    stable: bool
    maximize: bool


def threshold_stability(
    value: float,
    boundary: float,
    *,
    numerical_error: float = 0.0,
    reference_error: float = 0.0,
    parameter_error: float = 0.0,
    direction: str = "at_least",
) -> DecisionStability:
    """Compute a conservative distance from a scalar decision boundary.

    ``direction="at_least"`` accepts values above the boundary, while
    ``direction="at_most"`` accepts values below it.  A result is stable only
    when the accepted-side distance remains strictly larger than the budget.
    """
    if direction not in {"at_least", "at_most"}:
        raise ValueError("direction must be 'at_least' or 'at_most'")
    decision_value = _finite(value, "value")
    threshold = _finite(boundary, "boundary")
    budget = ErrorBudget(
        numerical=numerical_error,
        reference=reference_error,
        parameter=parameter_error,
    )
    signed_distance = (
        decision_value - threshold
        if direction == "at_least"
        else threshold - decision_value
    )
    margin = signed_distance - budget.total
    return DecisionStability(
        decision_value=decision_value,
        boundary=threshold,
        budget=budget,
        signed_distance=signed_distance,
        margin=margin,
        stable=margin > 0.0,
        direction=direction,
    )


def choice_stability(
    values: Mapping[str, float],
    *,
    error_budget: float = 0.0,
    maximize: bool = True,
) -> ChoiceStability:
    """Measure whether the best candidate is separated from the runner-up."""
    if len(values) < 2:
        raise ValueError("choice stability requires at least two candidates")
    budget = _finite_nonnegative(error_budget, "error_budget")
    candidates = [(name, _finite(value, f"value for {name!r}")) for name, value in values.items()]
    candidates.sort(key=lambda item: (-item[1], item[0]) if maximize else (item[1], item[0]))
    selected, selected_value = candidates[0]
    runner_up, runner_value = candidates[1]
    gap = selected_value - runner_value if maximize else runner_value - selected_value
    margin = gap - budget
    return ChoiceStability(
        selected=selected,
        runner_up=runner_up,
        gap=gap,
        error_budget=budget,
        margin=margin,
        stable=margin > 0.0,
        maximize=maximize,
    )


__all__ = [
    "ChoiceStability",
    "DecisionStability",
    "ErrorBudget",
    "choice_stability",
    "threshold_stability",
]
