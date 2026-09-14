"""Dimensional quantities with amplitude/power provenance.

``amp_order`` records the exponent of the underlying *field* that a value
represents:

===== ===========================================================
value meaning
===== ===========================================================
``None`` untracked - algebra stays permissive (the escape hatch)
``0``    a pure ratio, e.g. an operator matrix or a count
``1``    a field amplitude, e.g. ``sqrt(transmission)``
``2``    a power-like quantity, e.g. ``transmission`` or ``|t|**2``
===== ===========================================================

Rules: multiplication adds orders, division subtracts, ``sqrt`` halves an even
order, and addition/comparison require equal orders.  Bare Python numbers are
treated as untracked, so ordinary numeric code keeps working.

Convention for bare numbers: in additive contexts a bare number inherits the
other operand's unit (``q(2, "mm") + 3`` is ``5 mm``); in multiplicative
contexts it is dimensionless (``q(2, "mm") * 3`` is ``6 mm``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

import numpy as np

from .errors import AmplitudeOrderError, DimensionError, UnitError
from .units import Unit
from .units import unit as _unit

RATIO = 0
AMPLITUDE = 1
POWER = 2

_DIMENSIONLESS = _unit("1")
_RADIAN = _unit("rad")

_ORDER_NAMES = {0: "ratio", 1: "amplitude", 2: "power"}


def _order_name(order: int | None) -> str:
    if order is None:
        return "untracked"
    return _ORDER_NAMES.get(order, f"amplitude^{order}")


@dataclass(frozen=True, eq=False)
class Quantity:
    """A number (or array) with a unit and an amplitude order."""

    value: Any
    unit: Unit
    amp_order: int | None = None

    # -- construction / conversion --------------------------------------
    def to(self, target: str | Unit) -> "Quantity":
        """Return this quantity expressed in ``target`` (dimension must match)."""
        target_unit = target if isinstance(target, Unit) else _unit(target)
        return Quantity(self.unit.to_value(self.value, target_unit), target_unit, self.amp_order)

    def to_value(self, target: str | Unit):
        """Return the raw magnitude in ``target`` units, dropping the unit."""
        target_unit = target if isinstance(target, Unit) else _unit(target)
        return self.unit.to_value(self.value, target_unit)

    @property
    def magnitude(self):
        """The raw number in this quantity's own unit."""
        return self.value

    @property
    def si_value(self):
        """The raw number in SI base units."""
        return self.value * self.unit.factor

    @property
    def is_dimensionless(self) -> bool:
        return self.unit.is_dimensionless

    def has_same_dimension_as(self, other: "Quantity") -> bool:
        return self.unit.dimension == other.unit.dimension

    # -- arithmetic ------------------------------------------------------
    def _add_sub(self, other, subtract: bool, context: str) -> "Quantity":
        if isinstance(other, Quantity):
            if self.unit.dimension != other.unit.dimension:
                raise DimensionError(
                    f"{context}: {self.unit} and {other.unit} have different dimensions"
                )
            order = _merge_orders(self.amp_order, other.amp_order, context, self, other)
            rhs = other.unit.to_value(other.value, self.unit)
            value = self.value - rhs if subtract else self.value + rhs
            return Quantity(value, self.unit, order)
        value = self.value - other if subtract else self.value + other
        return Quantity(value, self.unit, self.amp_order)

    def _mul_div(self, other, divide: bool, context: str) -> "Quantity":
        if isinstance(other, Quantity):
            order = _combine_orders(self.amp_order, other.amp_order, divide)
            unit = self.unit / other.unit if divide else self.unit * other.unit
            value = self.value / other.value if divide else self.value * other.value
            return Quantity(value, unit, order)
        value = self.value / other if divide else self.value * other
        return Quantity(value, self.unit, self.amp_order)

    def _raise_to(self, exponent, context: str = "power") -> "Quantity":
        if isinstance(exponent, Quantity):
            if not exponent.unit.is_dimensionless:
                raise DimensionError(
                    f"{context}: exponent must be dimensionless, got {exponent.unit}"
                )
            if exponent.amp_order not in (None, RATIO):
                raise AmplitudeOrderError(
                    f"{context}: exponent is {_order_name(exponent.amp_order)}, expected a ratio"
                )
            exponent = exponent.value
        if isinstance(exponent, (np.ndarray, list, tuple)):
            raise UnitError(f"{context}: exponent must be a scalar, got an array")
        order = _scale_order(self.amp_order, exponent, context)
        return Quantity(self.value**exponent, self.unit**exponent, order)

    def __add__(self, other):
        return self._add_sub(other, subtract=False, context="addition")

    def __radd__(self, other):
        return self._add_sub(other, subtract=False, context="addition")

    def __sub__(self, other):
        return self._add_sub(other, subtract=True, context="subtraction")

    def __rsub__(self, other):
        return (-self)._add_sub(other, subtract=False, context="subtraction")

    def __mul__(self, other):
        return self._mul_div(other, divide=False, context="multiplication")

    def __rmul__(self, other):
        return self._mul_div(other, divide=False, context="multiplication")

    def __truediv__(self, other):
        return self._mul_div(other, divide=True, context="division")

    def __rtruediv__(self, other):
        inverse = Quantity(1.0, _DIMENSIONLESS, RATIO) / self
        return inverse._mul_div(other, divide=False, context="division")

    def __pow__(self, exponent):
        return self._raise_to(exponent)

    def __rpow__(self, base):
        """A plain number raised to a dimensionless quantity.

        A dimensioned exponent has no meaning, so it is rejected rather than
        silently stripped.
        """
        if not self.unit.is_dimensionless:
            raise DimensionError(
                f"exponent must be dimensionless, got {self.unit}"
            )
        if self.amp_order not in (None, RATIO):
            raise AmplitudeOrderError(
                f"base ** quantity: exponent is {_order_name(self.amp_order)}, expected a ratio"
            )
        return Quantity(base**self.value, _DIMENSIONLESS, RATIO)

    def __neg__(self):
        return Quantity(-self.value, self.unit, self.amp_order)

    def __pos__(self):
        return self

    def __abs__(self):
        return Quantity(abs(self.value), self.unit, self.amp_order)

    def __matmul__(self, other):
        if isinstance(other, Quantity):
            return Quantity(
                self.value @ other.value,
                self.unit * other.unit,
                _combine_orders(self.amp_order, other.amp_order, divide=False),
            )
        return Quantity(self.value @ other, self.unit, self.amp_order)

    def __rmatmul__(self, other):
        return Quantity(other @ self.value, self.unit, self.amp_order)

    # -- comparisons -----------------------------------------------------
    def _compare(self, other, operation, context: str):
        if isinstance(other, Quantity):
            if self.unit.dimension != other.unit.dimension:
                raise DimensionError(
                    f"{context}: {self.unit} and {other.unit} have different dimensions"
                )
            _merge_orders(self.amp_order, other.amp_order, context, self, other)
            rhs = other.unit.to_value(other.value, self.unit)
        else:
            rhs = other
        return operation(self.value, rhs)

    def __eq__(self, other):
        if isinstance(other, (int, float, complex, np.ndarray, list, tuple)):
            return self.value == other
        if isinstance(other, Quantity):
            try:
                return self._compare(other, lambda a, b: a == b, "comparison")
            except (DimensionError, AmplitudeOrderError):
                return False
        return NotImplemented

    def __ne__(self, other):
        equal = self.__eq__(other)
        if equal is NotImplemented:
            return NotImplemented
        return np.logical_not(equal)

    def __lt__(self, other):
        return self._compare(other, lambda a, b: a < b, "comparison")

    def __le__(self, other):
        return self._compare(other, lambda a, b: a <= b, "comparison")

    def __gt__(self, other):
        return self._compare(other, lambda a, b: a > b, "comparison")

    def __ge__(self, other):
        return self._compare(other, lambda a, b: a >= b, "comparison")

    # -- numpy protocol --------------------------------------------------
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        name = ufunc.__name__
        if kwargs.get("out") is not None:
            raise UnitError(f"in-place ufunc {name!r} is not supported on quantities")
        if method == "reduce":
            if name in _REDUCTIONS:
                return Quantity(
                    getattr(np, name).reduce(self.value, **kwargs), self.unit, self.amp_order
                )
            if name == "logical_or":
                return ufunc.reduce(self.value, **kwargs)
            raise UnitError(f"reduction {name!r} is not supported on quantities")
        if method != "__call__":
            raise UnitError(f"ufunc method {method!r} is not supported on quantities")

        forward = _OPERATOR_UFUNCS.get(name)
        if forward is not None:
            left_method, right_method = forward
            if isinstance(inputs[0], Quantity):
                return getattr(inputs[0], left_method)(*inputs[1:])
            other = inputs[0]
            return getattr(inputs[1], right_method)(other)

        if name in _UNARY_PRESERVING:
            return Quantity(ufunc(self.value, **kwargs), self.unit, self.amp_order)
        if name == "square":
            return self._raise_to(2)
        if name == "sqrt":
            return _square_root(self)
        if name == "reciprocal":
            return self._raise_to(-1)
        if name in _REQUIRES_DIMENSIONLESS:
            self._assert_dimensionless(name)
            return Quantity(ufunc(self.value, **kwargs), _DIMENSIONLESS, RATIO)
        if name in _REQUIRES_ANGLE:
            radians = self.unit.to_value(self.value, _RADIAN)
            return Quantity(ufunc(radians, **kwargs), _DIMENSIONLESS, RATIO)
        if name in _ANGLE_CONVERSIONS:
            assumed, result = _ANGLE_CONVERSIONS[name]
            if not self.unit.dimension == _RADIAN.dimension:
                raise DimensionError(
                    f"{name}: requires an angle, got {self.unit}"
                )
            return Quantity(ufunc(self.unit.to_value(self.value, assumed), **kwargs), result, RATIO)
        if name in _RETURNS_ANGLE:
            self._assert_dimensionless(name)
            return Quantity(ufunc(self.value, **kwargs), _RADIAN, RATIO)
        if name in _COMPARISON_UFUNCS:
            left, right = inputs
            if isinstance(left, Quantity):
                return getattr(left, _COMPARISON_UFUNCS[name][0])(right)
            return getattr(right, _COMPARISON_UFUNCS[name][1])(left)
        if name in _TWO_QUANTITY_SAME_UNIT:
            left, right = inputs
            if not (isinstance(left, Quantity) and isinstance(right, Quantity)):
                raise UnitError(f"{name!r} requires two quantities")
            _require_same_dimension(left, right, name)
            rhs = (
                right.value
                if right.unit == left.unit
                else right.unit.to_value(right.value, left.unit)
            )
            return Quantity(ufunc(left.value, rhs, **kwargs), left.unit, left.amp_order)
        if name in _PREDICATES:
            return ufunc(self.unit.to_value(self.value, _DIMENSIONLESS), **kwargs)
        raise UnitError(
            f"numpy ufunc {name!r} has no defined meaning for dimensional quantities; "
            "drop the unit explicitly with .to_value() / .magnitude if you really mean it"
        )

    def _assert_dimensionless(self, name: str) -> None:
        if not self.unit.is_dimensionless:
            raise DimensionError(
                f"{name}: requires a dimensionless argument, got {self.unit}"
            )

    # -- reductions ------------------------------------------------------
    def sum(self, axis=None, **kwargs) -> "Quantity":
        return Quantity(np.sum(self.value, axis=axis, **kwargs), self.unit, self.amp_order)

    def mean(self, axis=None, **kwargs) -> "Quantity":
        return Quantity(np.mean(self.value, axis=axis, **kwargs), self.unit, self.amp_order)

    def min(self, axis=None, **kwargs) -> "Quantity":
        return Quantity(np.min(self.value, axis=axis, **kwargs), self.unit, self.amp_order)

    def max(self, axis=None, **kwargs) -> "Quantity":
        return Quantity(np.max(self.value, axis=axis, **kwargs), self.unit, self.amp_order)

    # -- display ---------------------------------------------------------
    def __str__(self) -> str:
        tag = "" if self.amp_order is None else f" [{_order_name(self.amp_order)}]"
        return f"{self.value} {self.unit}{tag}"

    def __repr__(self) -> str:
        tag = "" if self.amp_order is None else f", amp_order={self.amp_order}"
        return f"Q({self.value!r}, {str(self.unit)!r}{tag})"


def _merge_orders(a, b, context, left=None, right=None):
    """Additive rule: equal orders required; ``None`` is a wildcard."""
    if a is None:
        return b
    if b is None:
        return a
    if a != b:
        raise AmplitudeOrderError(
            f"{context}: cannot combine {_order_name(a)} with {_order_name(b)}; "
            "a field amplitude and a power ratio are different physical objects "
            "(use sqrt()/square() to move between them)"
        )
    return a


def _combine_orders(a, b, divide: bool) -> int | None:
    if a is None or b is None:
        return None
    return a - b if divide else a + b


def _scale_order(order, exponent, context: str) -> int | None:
    if order is None:
        return None
    scaled = Fraction(order) * Fraction(exponent).limit_denominator(1000)
    if scaled.denominator != 1:
        raise AmplitudeOrderError(
            f"{context}: raising {_order_name(order)} to the power {exponent} "
            f"would give {float(scaled):g} (a fractional amplitude order)"
        )
    return int(scaled)


def _square_root(quantity: Quantity) -> Quantity:
    if quantity.amp_order is not None and quantity.amp_order % 2 != 0:
        raise AmplitudeOrderError(
            f"sqrt(): operand is {_order_name(quantity.amp_order)} (order "
            f"{quantity.amp_order}); taking a square root requires an even order. "
            "A power ratio can be rooted, an amplitude ratio cannot - check that "
            "you are not applying sqrt twice."
        )
    return quantity._raise_to(Fraction(1, 2), context="sqrt")


def _require_same_dimension(left: Quantity, right: Quantity, context: str) -> None:
    if left.unit.dimension != right.unit.dimension:
        raise DimensionError(
            f"{context}: {left.unit} and {right.unit} have different dimensions"
        )


_OPERATOR_UFUNCS = {
    "add": ("__add__", "__radd__"),
    "subtract": ("__sub__", "__rsub__"),
    "multiply": ("__mul__", "__rmul__"),
    "divide": ("__truediv__", "__rtruediv__"),
    "true_divide": ("__truediv__", "__rtruediv__"),
    "power": ("__pow__", "__rpow__"),
    "float_power": ("__pow__", "__rpow__"),
    "matmul": ("__matmul__", "__rmatmul__"),
}

_UNARY_PRESERVING = frozenset(
    {"negative", "positive", "absolute", "fabs", "conjugate", "conj",
     "floor", "ceil", "trunc", "rint", "real", "imag"}
)

_REQUIRES_DIMENSIONLESS = frozenset(
    {"exp", "exp2", "expm1", "log", "log2", "log10", "log1p",
     "sinh", "cosh", "tanh", "arcsinh", "arccosh", "arctanh",
     "logaddexp", "logaddexp2"}
)

_REQUIRES_ANGLE = frozenset({"sin", "cos", "tan"})

_RETURNS_ANGLE = frozenset({"arcsin", "arccos", "arctan"})

_ANGLE_CONVERSIONS = {
    "radians": (_unit("deg"), _RADIAN),
    "degrees": (_RADIAN, _unit("deg")),
}

_TWO_QUANTITY_SAME_UNIT = frozenset(
    {"hypot", "arctan2", "maximum", "minimum", "fmax", "fmin", "copysign"}
)

_PREDICATES = frozenset({"isnan", "isinf", "isfinite", "signbit", "sign"})

_REDUCTIONS = frozenset({"add", "maximum", "minimum"})

_COMPARISON_UFUNCS = {
    "greater": ("__gt__", "__lt__"),
    "greater_equal": ("__ge__", "__le__"),
    "less": ("__lt__", "__gt__"),
    "less_equal": ("__le__", "__ge__"),
    "equal": ("__eq__", "__eq__"),
    "not_equal": ("__ne__", "__ne__"),
}


# ---------------------------------------------------------------------------
# Public constructors
# ---------------------------------------------------------------------------


def q(value, unit_spec: str | Unit | None = None, amp_order: int | None = None) -> Quantity:
    """Build an untracked quantity: ``q(300, "mm")``."""
    if isinstance(unit_spec, Unit):
        resolved = unit_spec
    elif unit_spec is None:
        resolved = _DIMENSIONLESS
    else:
        resolved = _unit(unit_spec)
    return Quantity(value, resolved, amp_order)


def dimensionless(value) -> Quantity:
    """A pure ratio (order 0): counts, operators, plain numbers."""
    return Quantity(value, _DIMENSIONLESS, RATIO)


def amplitude_ratio(value) -> Quantity:
    """A field amplitude ratio (order 1), e.g. ``t`` where ``T = |t|^2``."""
    return Quantity(value, _DIMENSIONLESS, AMPLITUDE)


def power_ratio(value) -> Quantity:
    """A power ratio (order 2), e.g. ``T`` from a transmission spec."""
    return Quantity(value, _DIMENSIONLESS, POWER)


def sqrt(value):
    """Square root with the amplitude-order rule applied."""
    if isinstance(value, Quantity):
        return _square_root(value)
    return math.sqrt(value) if isinstance(value, (int, float)) else np.sqrt(value)
