"""Jones calculus: polarisation states and the elements that act on them.

Conventions
-----------
* A state is a column vector ``(Ex, Ey)`` of complex field amplitudes, with
  overall phase arbitrary and the intensity equal to ``|Ex|^2 + |Ey|^2``.
* Operators act from the left, and a *rotated* element is
  ``R(theta) M R(-theta)``.
* A retardance ``delta`` delays the slow axis by ``delta`` relative to the
  fast axis; a half-wave plate is ``delta = pi`` and a quarter-wave plate
  ``delta = pi/2``.
* Stokes parameters are unnormalised, and ``S3 = 2 Im(Ex conj(Ey))`` so that a
  quarter-wave plate at 45 degrees turns horizontal linear light into
  ``S3 = +1``.

Because these are the same objects the :mod:`optcon.checks` contracts are
written for, polarisation is the cheapest place to see the type layer work: a
waveplate asserts unitary, a polariser asserts passive and fails unitary, and
a rotator asserts unitary and fails reciprocal.
"""

from __future__ import annotations

import cmath
import math
from typing import Any

import numpy as np

from .errors import DimensionError
from .quantity import AMPLITUDE, POWER, RATIO, Quantity
from .units import Dimension
from .units import unit as _unit

_DIMENSIONLESS = _unit("1")
_RADIAN = _unit("rad")
_ANGLE = _RADIAN.dimension


def _angle_of(value: Any, context: str) -> float:
    if isinstance(value, Quantity):
        if value.unit.dimension != _ANGLE:
            raise DimensionError(f"{context}: expected an angle, got {value.unit}")
        return float(value.to_value("rad"))
    raise TypeError(
        f"{context}: expected an angle with units, e.g. q(45, 'deg'), "
        f"got a bare {type(value).__name__}"
    )


def _state_of(value: Any, context: str) -> Quantity:
    if not isinstance(value, Quantity):
        raise TypeError(f"{context}: expected a dimensionless state vector")
    if value.unit.dimension != Dimension():
        raise DimensionError(
            f"{context}: a polarisation state is dimensionless, got {value.unit}"
        )
    return value


def _operator(value: Any, context: str) -> Quantity:
    if not isinstance(value, Quantity):
        raise TypeError(f"{context}: expected a dimensionless operator")
    if value.unit.dimension != Dimension():
        raise DimensionError(f"{context}: an operator is dimensionless, got {value.unit}")
    return value


# ---------------------------------------------------------------------------
# States
# ---------------------------------------------------------------------------


def jones_linear(angle: Any) -> Quantity:
    """Unit-amplitude linear polarisation at ``angle``."""
    theta = _angle_of(angle, "jones_linear")
    vector = np.array([math.cos(theta), math.sin(theta)], dtype=complex)
    return Quantity(vector, _DIMENSIONLESS, AMPLITUDE)


def jones_circular(handedness: str = "right") -> Quantity:
    """Unit-amplitude circular polarisation.

    ``"right"`` gives ``(1, -i)/sqrt(2)`` and hence ``S3 = +1``; ``"left"``
    gives the opposite.
    """
    if handedness not in {"right", "left"}:
        raise ValueError("handedness must be 'right' or 'left'")
    sign = -1.0j if handedness == "right" else +1.0j
    vector = np.array([1.0, sign], dtype=complex) / math.sqrt(2.0)
    return Quantity(vector, _DIMENSIONLESS, AMPLITUDE)


# ---------------------------------------------------------------------------
# Elements
# ---------------------------------------------------------------------------


def _rotated(core: np.ndarray, theta: float) -> np.ndarray:
    rotation = np.array(
        [[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]],
        dtype=complex,
    )
    return rotation @ core @ np.linalg.inv(rotation)


def polarizer(angle: Any) -> Quantity:
    """An ideal linear polariser; passive, idempotent, and not unitary."""
    theta = _angle_of(angle, "polarizer")
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    matrix = np.array(
        [[cos_t * cos_t, cos_t * sin_t], [cos_t * sin_t, sin_t * sin_t]], dtype=complex
    )
    return Quantity(matrix, _DIMENSIONLESS, RATIO)


def retarder(retardance: Any, angle: Any = None) -> Quantity:
    """A waveplate of retardance ``delta`` whose slow axis is at ``angle``."""
    delta = _angle_of(retardance, "retarder")
    theta = 0.0 if angle is None else _angle_of(angle, "retarder")
    phase = cmath.exp(1.0j * delta)
    core = np.array([[1.0, 0.0], [0.0, phase]], dtype=complex)
    return Quantity(_rotated(core, theta), _DIMENSIONLESS, RATIO)


def half_wave_plate(angle: Any) -> Quantity:
    """A half-wave plate: rotates linear polarisation by twice its angle."""
    return retarder(Quantity(math.pi, _RADIAN), angle)


def quarter_wave_plate(angle: Any) -> Quantity:
    """A quarter-wave plate: turns linear at 45 degrees into circular."""
    return retarder(Quantity(math.pi / 2.0, _RADIAN), angle)


def rotator(angle: Any) -> Quantity:
    """A polarisation rotator.  Unitary, but not reciprocal."""
    theta = _angle_of(angle, "rotator")
    matrix = np.array(
        [[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]],
        dtype=complex,
    )
    return Quantity(matrix, _DIMENSIONLESS, RATIO)


# ---------------------------------------------------------------------------
# Acting and measuring
# ---------------------------------------------------------------------------


def apply(element: Any, state: Any) -> Quantity:
    """Apply an operator to a state vector."""
    matrix = _operator(element, "apply").value
    vector = _state_of(state, "apply").value
    return Quantity(np.asarray(matrix) @ np.asarray(vector), _DIMENSIONLESS, AMPLITUDE)


def intensity(state: Any) -> Quantity:
    """``|Ex|^2 + |Ey|^2`` as a power ratio."""
    vector = np.asarray(_state_of(state, "intensity").value)
    return Quantity(float(np.sum(np.abs(vector) ** 2)), _DIMENSIONLESS, POWER)


def stokes(state: Any) -> dict[str, float]:
    """The four Stokes parameters, unnormalised, as plain floats."""
    vector = np.asarray(_state_of(state, "stokes").value)
    ex, ey = complex(vector[0]), complex(vector[1])
    cross = ex * np.conj(ey)
    return {
        "S0": float(abs(ex) ** 2 + abs(ey) ** 2),
        "S1": float(abs(ex) ** 2 - abs(ey) ** 2),
        "S2": float(2.0 * cross.real),
        "S3": float(2.0 * cross.imag),
    }


def degree_of_polarization(state: Any) -> float:
    """``sqrt(S1^2 + S2^2 + S3^2) / S0``; exactly one for a pure state."""
    parameters = stokes(state)
    total = parameters["S0"]
    if total == 0.0:
        return 0.0
    polarized = math.sqrt(
        parameters["S1"] ** 2 + parameters["S2"] ** 2 + parameters["S3"] ** 2
    )
    return float(polarized / total)


def extinction_ratio(first: Any, second: Any) -> float:
    """Best-to-worst transmission ratio of a polariser pair.

    The numerator is the best transmission achievable by rotating the analyser
    to match the polariser; the denominator is the transmission through the
    pair as actually configured.  A perfectly crossed ideal pair therefore
    returns ``inf``, and a one-degree misalignment returns a few thousand -
    the number an alignment budget actually cares about.
    """
    a = _operator(first, "extinction_ratio").value
    b = _operator(second, "extinction_ratio").value
    reference = jones_linear(Quantity(0.0, _RADIAN)).value
    after_polarizer = np.asarray(a) @ reference

    best = 0.0
    for angle in np.linspace(0.0, math.pi, 181):
        analyser = polarizer(Quantity(angle, _RADIAN)).value
        transmitted = float(np.sum(np.abs(np.asarray(analyser) @ after_polarizer) ** 2))
        best = max(best, transmitted)

    configured = float(np.sum(np.abs(np.asarray(b) @ after_polarizer) ** 2))
    # cos(pi/2) evaluates to ~6e-17 rather than 0, so an exactly crossed ideal
    # pair leaves ~1e-33 of floating-point dust.  Genuine physical leakage (a
    # one-degree misalignment) is ~3e-4, sixteen orders above this floor, so
    # anything below it is numerical zero rather than a real extinction.
    if configured <= 1e-20 * max(best, 1.0):
        return math.inf
    return float(best / configured)


__all__ = [
    "apply",
    "degree_of_polarization",
    "extinction_ratio",
    "half_wave_plate",
    "intensity",
    "jones_circular",
    "jones_linear",
    "polarizer",
    "quarter_wave_plate",
    "retarder",
    "rotator",
    "stokes",
]
