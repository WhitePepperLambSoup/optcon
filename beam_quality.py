"""Beam quality: M squared and the beam parameter product (ISO 11146).

For an ideal Gaussian ``M^2 = 1``; a real beam of the same waist diverges
``M^2`` times faster, which folds into every propagation formula as an
effective Rayleigh range ``z_R / M^2`` and an effective wavelength
``M^2 lambda``.
"""

from __future__ import annotations

import math
from typing import Any

from .errors import DimensionError
from .quantity import Quantity
from .units import Unit
from .units import unit as _unit

_LENGTH = _unit("m").dimension


def _length_of(value: Any, context: str) -> tuple[float, Unit]:
    if isinstance(value, Quantity):
        if value.unit.dimension != _LENGTH:
            raise DimensionError(f"{context}: expected a length, got {value.unit}")
        return float(value.value), value.unit
    raise TypeError(
        f"{context}: expected a length with units, e.g. q(0.4, 'mm'), "
        f"got a bare {type(value).__name__}"
    )


def _checked_m_squared(value: float) -> float:
    squared = float(value)
    if squared < 1.0:
        raise ValueError(
            f"M squared must be at least one for a physical beam, got {squared}"
        )
    return squared


def beam_parameter_product(m_squared: float, wavelength: Any) -> Quantity:
    """``BPP = M^2 lambda / pi``, the waist-radius-times-half-angle invariant.

    Returned as a length times an angle.  Since this library keeps angle as
    its own dimension, that is expressible: the customary figure in
    ``mm mrad`` is this quantity converted with ``to_value("mm*mrad")``.
    """
    squared = _checked_m_squared(m_squared)
    lam, unit = _length_of(wavelength, "beam_parameter_product")
    return Quantity(squared * lam / math.pi, unit * _unit("rad"))


def m_squared_from_parameters(waist: Any, divergence: Any, wavelength: Any) -> float:
    """``M^2 = pi w0 theta / lambda`` from a measured waist and half angle."""
    w0, unit = _length_of(waist, "m_squared_from_parameters")
    theta = (
        float(divergence.to_value("rad"))
        if isinstance(divergence, Quantity)
        else float(divergence)
    )
    lam, lam_unit = _length_of(wavelength, "m_squared_from_parameters")
    lam = lam_unit.to_value(lam, unit)
    if w0 <= 0.0 or theta <= 0.0 or lam <= 0.0:
        raise ValueError("waist, divergence and wavelength must all be positive")
    return math.pi * w0 * theta / lam


def m_squared_from_widths(
    waist: Any, radius_at_distance: Any, distance: Any, wavelength: Any
) -> float:
    """``M^2`` from two measured widths and the separation, inverting the hyperbola."""
    w0, unit = _length_of(waist, "m_squared_from_widths")
    measured, measured_unit = _length_of(radius_at_distance, "m_squared_from_widths")
    measured = measured_unit.to_value(measured, unit)
    z, _ = _length_of(distance, "m_squared_from_widths")
    lam, lam_unit = _length_of(wavelength, "m_squared_from_widths")
    lam = lam_unit.to_value(lam, unit)
    if measured <= w0:
        raise ValueError(
            "the measured radius must exceed the waist radius, otherwise the "
            "beam is not expanding"
        )
    if z <= 0.0:
        raise ValueError("the separation must be positive")
    # w(z)^2 = w0^2 + (M^2 lambda z / (pi w0))^2
    numerator = math.sqrt(measured**2 - w0**2) * math.pi * w0
    return numerator / (lam * z)


def rayleigh_range_with_m_squared(waist: Any, wavelength: Any, m_squared: float) -> Quantity:
    """Effective Rayleigh range of a non-ideal beam, ``pi w0^2 / (M^2 lambda)``."""
    squared = _checked_m_squared(m_squared)
    w0, unit = _length_of(waist, "rayleigh_range_with_m_squared")
    lam, lam_unit = _length_of(wavelength, "rayleigh_range_with_m_squared")
    lam = lam_unit.to_value(lam, unit)
    return Quantity(math.pi * w0**2 / (squared * lam), unit)


def beam_radius_with_m_squared(
    waist: Any, wavelength: Any, distance: Any, m_squared: float
) -> Quantity:
    """``w(z)`` for a beam of the given ``M^2``."""
    squared = _checked_m_squared(m_squared)
    w0, unit = _length_of(waist, "beam_radius_with_m_squared")
    lam, lam_unit = _length_of(wavelength, "beam_radius_with_m_squared")
    lam = lam_unit.to_value(lam, unit)
    z, _ = _length_of(distance, "beam_radius_with_m_squared")
    return Quantity(w0 * math.sqrt(1.0 + (squared * lam * z / (math.pi * w0**2)) ** 2), unit)


def divergence_with_m_squared(waist: Any, wavelength: Any, m_squared: float) -> Quantity:
    """Half angle of a non-ideal beam, ``M^2 lambda / (pi w0)``."""
    squared = _checked_m_squared(m_squared)
    w0, unit = _length_of(waist, "divergence_with_m_squared")
    lam, lam_unit = _length_of(wavelength, "divergence_with_m_squared")
    lam = lam_unit.to_value(lam, unit)
    return Quantity(squared * lam / (math.pi * w0), _unit("rad"))


__all__ = [
    "beam_parameter_product",
    "beam_radius_with_m_squared",
    "divergence_with_m_squared",
    "m_squared_from_parameters",
    "m_squared_from_widths",
    "rayleigh_range_with_m_squared",
]
