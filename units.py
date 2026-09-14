"""Dimensional algebra for physical quantities.

Two deliberate departures from SI:

1. **Angle is its own base dimension.**  SI calls the radian dimensionless,
   which makes ``mrad`` addable to a bare ratio.  In optical alignment code
   that is almost always a bug, so ``angle`` is tracked separately.
2. **Unit names are parsed, not enumerated.**  ``unit("mW/cm^2")`` works
   because prefixes, products, quotients and integer powers are composed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from fractions import Fraction
from functools import lru_cache
from typing import Iterator, Mapping

from .errors import DimensionError, UnitError

BASE_DIMENSIONS = (
    "length",
    "time",
    "mass",
    "current",
    "temperature",
    "amount",
    "luminous",
    "angle",
    "solid_angle",
    "information",
)


def _as_fraction(value: int | float | Fraction) -> Fraction:
    if isinstance(value, Fraction):
        return value
    if isinstance(value, int):
        return Fraction(value)
    return Fraction(value).limit_denominator(1000)


@dataclass(frozen=True)
class Dimension:
    """A dimension is a sparse vector of base-dimension exponents."""

    exponents: tuple[tuple[str, Fraction], ...] = ()

    @staticmethod
    def from_mapping(exponents: Mapping[str, int | float | Fraction]) -> "Dimension":
        items = []
        for name, power in exponents.items():
            if name not in BASE_DIMENSIONS:
                raise UnitError(f"unknown base dimension {name!r}")
            power = _as_fraction(power)
            if power != 0:
                items.append((name, power))
        return Dimension(tuple(sorted(items)))

    def as_dict(self) -> dict[str, Fraction]:
        return dict(self.exponents)

    @property
    def is_dimensionless(self) -> bool:
        return not self.exponents

    def __mul__(self, other: "Dimension") -> "Dimension":
        merged = self.as_dict()
        for name, power in other.exponents:
            merged[name] = merged.get(name, Fraction(0)) + power
        return Dimension.from_mapping(merged)

    def __truediv__(self, other: "Dimension") -> "Dimension":
        return self * (other**-1)

    def __pow__(self, power: int | float | Fraction) -> "Dimension":
        power = _as_fraction(power)
        return Dimension.from_mapping(
            {name: exponent * power for name, exponent in self.exponents}
        )

    def __str__(self) -> str:
        if not self.exponents:
            return "dimensionless"
        return " ".join(f"{name}^{power}" for name, power in self.exponents)

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"Dimension({self})"


@dataclass(frozen=True)
class Unit:
    """A scale factor into SI base units plus a dimension.

    ``value_in_SI = value * factor``.  Symbol is display-only and excluded
    from equality so that ``unit("mm") * unit("mm")`` compares equal to a
    freshly constructed ``mm^2``.
    """

    factor: float
    dimension: Dimension = Dimension()
    symbol: str | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        if not math.isfinite(self.factor) or self.factor == 0.0:
            raise UnitError(f"unit scale must be finite and non-zero, got {self.factor!r}")

    # -- algebra ---------------------------------------------------------
    def __mul__(self, other: "Unit") -> "Unit":
        if not isinstance(other, Unit):
            return NotImplemented
        return Unit(self.factor * other.factor, self.dimension * other.dimension)

    def __truediv__(self, other: "Unit") -> "Unit":
        if not isinstance(other, Unit):
            return NotImplemented
        return Unit(self.factor / other.factor, self.dimension / other.dimension)

    def __rtruediv__(self, other: float) -> "Unit":
        """``1 / mm`` is a perfectly good way to write a reciprocal unit."""
        return Unit(other / self.factor, Dimension() / self.dimension)

    def __pow__(self, power: int | float | Fraction) -> "Unit":
        power = _as_fraction(power)
        if power.denominator == 1:
            exponent = int(power)
            factor = self.factor**exponent
        else:
            if self.factor < 0:
                raise UnitError("cannot raise a negative unit scale to a fractional power")
            factor = self.factor ** float(power)
        return Unit(factor, self.dimension**power)

    # -- conversion ------------------------------------------------------
    @property
    def is_dimensionless(self) -> bool:
        return self.dimension.is_dimensionless

    def conversion_factor(self, target: "Unit") -> float:
        if self.dimension != target.dimension:
            raise DimensionError(
                f"cannot convert {self} to {target}: different dimensions"
            )
        return self.factor / target.factor

    def to_value(self, value, target: "Unit"):
        return value * self.conversion_factor(target)

    def is_equivalent_to(self, other: "Unit", rel_tol: float = 1e-12) -> bool:
        return self.dimension == other.dimension and math.isclose(
            self.factor, other.factor, rel_tol=rel_tol
        )

    def __str__(self) -> str:
        if self.symbol is not None:
            return self.symbol
        if self.dimension.is_dimensionless:
            return f"{self.factor:g}"
        return f"({self.factor:g}·{self.dimension})"


# ---------------------------------------------------------------------------
# Unit registry
# ---------------------------------------------------------------------------

_PREFIXES: dict[str, float] = {
    "da": 1e1,
    "Y": 1e24, "Z": 1e21, "E": 1e18, "P": 1e15, "T": 1e12, "G": 1e9,
    "M": 1e6, "k": 1e3, "h": 1e2,
    "d": 1e-1, "c": 1e-2, "m": 1e-3,
    "u": 1e-6, "µ": 1e-6, "μ": 1e-6,
    "n": 1e-9, "p": 1e-12, "f": 1e-15, "a": 1e-18, "z": 1e-21, "y": 1e-24,
}

_PREFIX_ORDER = tuple(sorted(_PREFIXES, key=len, reverse=True))

_BASE_SYMBOLS: dict[str, tuple[float, dict[str, int]]] = {
    "m": (1.0, {"length": 1}),
    "s": (1.0, {"time": 1}),
    "g": (1e-3, {"mass": 1}),
    "A": (1.0, {"current": 1}),
    "K": (1.0, {"temperature": 1}),
    "mol": (1.0, {"amount": 1}),
    "cd": (1.0, {"luminous": 1}),
    "rad": (1.0, {"angle": 1}),
    "deg": (math.pi / 180.0, {"angle": 1}),
    "sr": (1.0, {"solid_angle": 1}),
    "bit": (1.0, {"information": 1}),
    "byte": (8.0, {"information": 1}),
}

_DERIVED_SYMBOLS: dict[str, tuple[float, dict[str, int]]] = {
    "Hz": (1.0, {"time": -1}),
    "N": (1.0, {"mass": 1, "length": 1, "time": -2}),
    "J": (1.0, {"mass": 1, "length": 2, "time": -2}),
    "W": (1.0, {"mass": 1, "length": 2, "time": -3}),
    "Pa": (1.0, {"mass": 1, "length": -1, "time": -2}),
    "C": (1.0, {"current": 1, "time": 1}),
    "V": (1.0, {"mass": 1, "length": 2, "time": -3, "current": -1}),
    "ohm": (1.0, {"mass": 1, "length": 2, "time": -3, "current": -2}),
    "F": (-1.0, {"mass": -1, "length": -2, "time": 4, "current": 2}),
    "T": (1.0, {"mass": 1, "time": -2, "current": -1}),
    "eV": (1.602176634e-19, {"mass": 1, "length": 2, "time": -2}),
    "lm": (1.0, {"luminous": 1, "solid_angle": 1}),
    "lx": (1.0, {"luminous": 1, "solid_angle": -1, "length": -2}),
    "angstrom": (1e-10, {"length": 1}),
    "1": (1.0, {}),
    "%": (1e-2, {}),
    "ppm": (1e-6, {}),
}

_SYMBOLS: dict[str, tuple[float, dict[str, int]]] = {
    **_BASE_SYMBOLS,
    **_DERIVED_SYMBOLS,
}


def _make_unit(factor: float, exponents: dict[str, int], symbol: str | None) -> Unit:
    return Unit(factor, Dimension.from_mapping(exponents), symbol)


def _lookup(symbol: str) -> Unit:
    entry = _SYMBOLS.get(symbol)
    if entry is not None:
        factor, exponents = entry
        return _make_unit(factor, exponents, symbol)
    for prefix in _PREFIX_ORDER:
        if symbol.startswith(prefix) and len(symbol) > len(prefix):
            rest = symbol[len(prefix):]
            base = _SYMBOLS.get(rest)
            if base is not None:
                factor, exponents = base
                return _make_unit(_PREFIXES[prefix] * factor, exponents, symbol)
    raise UnitError(f"unknown unit {symbol!r}")


class _Parser:
    """Recursive descent over the unit grammar.

        expression := factor (('*' | '/' | '.') factor)*
        factor     := symbol ('^' | '**') integer
                    | '(' expression ')'

    Division applies to the whole following group, so ``W/(m^2*sr)`` means
    what it looks like.
    """

    _SYMBOL_CHARS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ%")
    _EXTRA_CHARS = set("µμÅ")

    def __init__(self, text: str) -> None:
        self.text = text
        self.position = 0

    def parse(self) -> Unit:
        result = self._expression()
        self._skip_space()
        if self.position != len(self.text):
            raise UnitError(
                f"unexpected {self.text[self.position]!r} in {self.text!r}"
            )
        return result

    def _skip_space(self) -> None:
        while self.position < len(self.text) and self.text[self.position].isspace():
            self.position += 1

    def _peek_operator(self) -> str | None:
        self._skip_space()
        if self.position >= len(self.text):
            return None
        char = self.text[self.position]
        if char in "*/·":
            return "/" if char == "/" else "*"
        return None

    def _expression(self) -> Unit:
        result = self._factor()
        while True:
            operator = self._peek_operator()
            if operator is None:
                return result
            self.position += 1
            operand = self._factor()
            result = result * operand if operator == "*" else result / operand

    def _factor(self) -> Unit:
        self._skip_space()
        if self.position >= len(self.text):
            raise UnitError(f"expression ends where a unit was expected: {self.text!r}")
        if self.text[self.position] == "(":
            self.position += 1
            inner = self._expression()
            self._skip_space()
            if self.position >= len(self.text) or self.text[self.position] != ")":
                raise UnitError(f"unbalanced parenthesis in {self.text!r}")
            self.position += 1
            return self._exponent(inner)
        return self._exponent(self._symbol())

    def _symbol(self) -> Unit:
        start = self.position
        while self.position < len(self.text):
            char = self.text[self.position]
            if char in self._SYMBOL_CHARS or char in self._EXTRA_CHARS or char.isdigit():
                self.position += 1
                continue
            break
        token = self.text[start : self.position]
        if not token:
            raise UnitError(
                f"expected a unit name at position {start} of {self.text!r}"
            )
        return _lookup(token)

    def _exponent(self, base: Unit) -> Unit:
        self._skip_space()
        if self.position >= len(self.text):
            return base
        if self.text[self.position] == "^":
            self.position += 1
        elif self.text.startswith("**", self.position):
            self.position += 2
        else:
            return base
        start = self.position
        if self.position < len(self.text) and self.text[self.position] in "+-":
            self.position += 1
        while self.position < len(self.text) and self.text[self.position].isdigit():
            self.position += 1
        digits = self.text[start : self.position]
        if not digits or digits in {"+", "-"}:
            raise UnitError(f"missing exponent in {self.text!r}")
        return base ** int(digits)


@lru_cache(maxsize=512)
def unit(expression: str) -> Unit:
    """Parse a unit expression such as ``"mm"``, ``"mW/cm^2"`` or ``"W/(m^2*sr)"``."""
    if not isinstance(expression, str):
        raise UnitError(f"unit expression must be a string, got {type(expression).__name__}")
    if not expression.strip():
        raise UnitError("empty unit expression")
    return _Parser(expression).parse()


def all_registered_symbols() -> Iterator[str]:
    """Yield every bare unit symbol known to the registry (no prefixes)."""
    return iter(sorted(_SYMBOLS))
