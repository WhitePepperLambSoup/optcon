"""Gaussian beams described by the complex q parameter.

The q parameter of a beam is ``q = z + i z_R``, where ``z`` is the distance
from the waist and ``z_R = pi w0^2 / lambda`` is the Rayleigh range.  It is a
*length*, and a complex one, so it is carried here as a complex
:class:`~optcon.quantity.Quantity` whose unit is a length; every function
checks that its arguments are lengths before doing anything with them.

The transformations are the standard ones from Kogelnik & Li (1970) and
Siegman, *Lasers* (1986), ch. 17:

* propagation through a system: ``q' = (A q + B) / (C q + D)``
* field radius: ``w = sqrt(-lambda / (pi Im(1/q)))``
* wavefront curvature: ``R = 1 / Re(1/q)``
* Gouy phase relative to the waist: ``psi = arctan(Re(q) / Im(q))``
"""

from __future__ import annotations

import cmath
import math
from typing import Any

from .elements import TransferMatrix
from .errors import DimensionError
from .quantity import Quantity
from .units import Unit
from .units import unit as _unit

_LENGTH = _unit("m").dimension
_ANGLE = _unit("rad").dimension


def _length_of(value: Any, context: str) -> tuple[float, Unit]:
    if isinstance(value, Quantity):
        if value.unit.dimension != _LENGTH:
            raise DimensionError(f"{context}: expected a length, got {value.unit}")
        return float(value.value), value.unit
    raise TypeError(
        f"{context}: expected a length with units, e.g. q(1, 'mm'), "
        f"got a bare {type(value).__name__}"
    )


def _wavelength_of(value: Any) -> tuple[float, Unit]:
    return _length_of(value, "wavelength")


def _beam_q(value: Any, context: str) -> Quantity:
    if isinstance(value, Quantity):
        if value.unit.dimension != _LENGTH:
            raise DimensionError(f"{context}: expected a length, got {value.unit}")
        return value
    raise TypeError(f"{context}: expected a q parameter as a Quantity")


# ---------------------------------------------------------------------------
# Beam parameters
# ---------------------------------------------------------------------------


def rayleigh_range(waist: Any, wavelength: Any) -> Quantity:
    """``z_R = pi w0^2 / lambda``."""
    w0, unit = _length_of(waist, "waist")
    lam, lam_unit = _wavelength_of(wavelength)
    lam = lam_unit.to_value(lam, unit)
    if w0 <= 0.0 or lam <= 0.0:
        raise ValueError("waist and wavelength must be positive")
    return Quantity(math.pi * w0**2 / lam, unit)


def divergence_half_angle(waist: Any, wavelength: Any) -> Quantity:
    """Far-field half angle ``theta = lambda / (pi w0)``, in radians."""
    w0, unit = _length_of(waist, "waist")
    lam, lam_unit = _wavelength_of(wavelength)
    lam = lam_unit.to_value(lam, unit)
    if w0 <= 0.0 or lam <= 0.0:
        raise ValueError("waist and wavelength must be positive")
    return Quantity(lam / (math.pi * w0), _unit("rad"))


def q_from_waist(waist: Any, wavelength: Any, distance: Any = None) -> Quantity:
    """The q parameter ``distance`` downstream of a waist.

    With no distance this returns ``i z_R``, the pure imaginary parameter at
    the waist itself.
    """
    z_r = rayleigh_range(waist, wavelength)
    offset = 0.0
    if distance is not None:
        value, unit = _length_of(distance, "distance")
        offset = z_r.unit.to_value(value, unit) if unit != z_r.unit else value
    return Quantity(complex(offset, z_r.value), z_r.unit)


def beam_radius(q_parameter: Any, wavelength: Any) -> Quantity:
    """Field radius ``w`` of the beam described by ``q``."""
    qp = _beam_q(q_parameter, "q_parameter")
    lam, lam_unit = _wavelength_of(wavelength)
    lam = lam_unit.to_value(lam, qp.unit)
    inverse_imaginary = (1.0 / qp.value).imag
    if inverse_imaginary >= 0.0:
        raise ValueError("this q parameter does not describe a physical beam")
    return Quantity(math.sqrt(-lam / (math.pi * inverse_imaginary)), qp.unit)


def radius_of_curvature(q_parameter: Any, wavelength: Any = None) -> Quantity:
    """Wavefront radius of curvature; infinite at the waist."""
    qp = _beam_q(q_parameter, "q_parameter")
    real_inverse = (1.0 / qp.value).real
    if abs(real_inverse) < 1e-300:
        return Quantity(math.inf, qp.unit)
    return Quantity(1.0 / real_inverse, qp.unit)


def gouy_phase(q_parameter: Any, wavelength: Any = None) -> Quantity:
    """Gouy phase relative to the waist, in radians."""
    qp = _beam_q(q_parameter, "q_parameter")
    return Quantity(math.atan2(qp.value.real, qp.value.imag), _unit("rad"))


def waist_of(q_parameter: Any, wavelength: Any = None) -> Quantity:
    """The waist radius that this q parameter corresponds to."""
    qp = _beam_q(q_parameter, "q_parameter")
    imaginary = qp.value.imag
    if imaginary <= 0.0:
        raise ValueError("this q parameter does not describe a physical beam")
    # Im(q) = z_R = pi w0^2 / lambda is invariant under propagation, so it is
    # recovered directly rather than by walking back to the waist.
    if wavelength is None:
        raise TypeError("waist_of needs the wavelength to convert Im(q) into a radius")
    lam, lam_unit = _wavelength_of(wavelength)
    lam = lam_unit.to_value(lam, qp.unit)
    return Quantity(math.sqrt(lam * imaginary / math.pi), qp.unit)


def position_of_waist(q_parameter: Any, wavelength: Any = None) -> Quantity:
    """Distance from the current plane back to the waist (negative upstream)."""
    qp = _beam_q(q_parameter, "q_parameter")
    return Quantity(-qp.value.real, qp.unit)


# ---------------------------------------------------------------------------
# Propagation
# ---------------------------------------------------------------------------


def propagate(q_parameter: Any, matrix: TransferMatrix) -> Quantity:
    """Apply an ABCD system to a beam: ``q' = (A q + B) / (C q + D)``."""
    qp = _beam_q(q_parameter, "q_parameter")
    if not isinstance(matrix, TransferMatrix):
        raise TypeError(f"propagate expects a TransferMatrix, got {type(matrix).__name__}")
    if matrix.unit.dimension != qp.unit.dimension:
        raise DimensionError(
            f"cannot propagate a beam in {qp.unit} through a system in {matrix.unit}"
        )
    local = qp.to(matrix.unit).value
    a, b = matrix.matrix[0]
    c, d = matrix.matrix[1]
    denominator = c * local + d
    if denominator == 0:
        raise ZeroDivisionError("the ABCD system maps this beam to infinity")
    return Quantity((a * local + b) / denominator, matrix.unit)


def self_consistent_mode(round_trip: TransferMatrix, wavelength: Any = None) -> Quantity:
    """The q parameter that maps onto itself after one ``round_trip``.

    Solves ``C q^2 + (D - A) q - B = 0`` and keeps the root with positive
    imaginary part, which is the physical (decaying) solution.  A stable
    cavity always has exactly one such root.
    """
    if not isinstance(round_trip, TransferMatrix):
        raise TypeError("self_consistent_mode expects a TransferMatrix")
    a, b = round_trip.matrix[0]
    c, d = round_trip.matrix[1]
    if c == 0.0:
        raise ValueError("this system is afocal: it has no self-consistent mode")
    discriminant = complex((a + d) ** 2 - 4.0)
    root = cmath.sqrt(discriminant)
    candidates = [(a - d + root) / (2.0 * c), (a - d - root) / (2.0 * c)]
    physical = [value for value in candidates if value.imag > 0.0]
    if not physical:
        raise ValueError(
            "this round trip has no stable mode (the cavity is outside the "
            "stability region)"
        )
    return Quantity(physical[0], round_trip.unit)


# ---------------------------------------------------------------------------
# Mode matching
# ---------------------------------------------------------------------------


def mode_overlap(first_waist: float, second_waist: float) -> float:
    """Power coupling between two coaxial Gaussian modes at the same plane."""
    if first_waist <= 0.0 or second_waist <= 0.0:
        raise ValueError("waist sizes must be positive")
    return (2.0 * first_waist * second_waist / (first_waist**2 + second_waist**2)) ** 2


def mode_matching_efficiency(
    first: Any, second: Any, wavelength: Any = None
) -> float:
    """Power coupled between two Gaussian modes given by their q parameters.

    ``eta = 4 Im(q1) Im(q2) / |conj(q1) - q2|^2``.  Both beams must be
    referenced to the same plane.
    """
    q1 = _beam_q(first, "first")
    q2 = _beam_q(second, "second")
    local = q2.to(q1.unit).value
    numerator = 4.0 * q1.value.imag * local.imag
    denominator = abs(q1.value.conjugate() - local) ** 2
    if denominator == 0.0:
        return 0.0
    return float(numerator / denominator)


def beam_diameter(q_parameter: Any, wavelength: Any) -> Quantity:
    """Convenience wrapper: ``2 w``."""
    radius = beam_radius(q_parameter, wavelength)
    return Quantity(2.0 * radius.value, radius.unit)


__all__ = [
    "beam_diameter",
    "beam_radius",
    "divergence_half_angle",
    "gouy_phase",
    "mode_matching_efficiency",
    "mode_overlap",
    "position_of_waist",
    "propagate",
    "q_from_waist",
    "radius_of_curvature",
    "rayleigh_range",
    "self_consistent_mode",
    "waist_of",
]
