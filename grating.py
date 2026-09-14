"""Diffraction gratings.

The grating equation ``m lambda = d (sin(theta_i) + sin(theta_m))`` follows
Hecht, *Optics*, ch. 10; the Littrow and blaze relations follow Palmer,
*Diffraction Grating Handbook*, ch. 2 and 4.  Angles are signed, so a
negative order is diffracted to the other side of the normal.
"""

from __future__ import annotations

import math
from typing import Any

from .errors import DimensionError
from .quantity import Quantity
from .units import Unit
from .units import unit as _unit

_LENGTH = _unit("m").dimension
_RADIAN = _unit("rad")
_ANGLE = _RADIAN.dimension


def _length_of(value: Any, context: str) -> tuple[float, Unit]:
    if isinstance(value, Quantity):
        if value.unit.dimension != _LENGTH:
            raise DimensionError(f"{context}: expected a length, got {value.unit}")
        return float(value.value), value.unit
    raise TypeError(
        f"{context}: expected a length with units, e.g. q(633, 'nm'), "
        f"got a bare {type(value).__name__}"
    )


def _angle_of(value: Any, context: str) -> float:
    if isinstance(value, Quantity):
        if value.unit.dimension != _ANGLE:
            raise DimensionError(f"{context}: expected an angle, got {value.unit}")
        return float(value.to_value("rad"))
    raise TypeError(f"{context}: expected an angle with units, e.g. q(20, 'deg')")


def period_from_lines_per_mm(lines_per_mm: float) -> Quantity:
    """Grating period ``d = 1 / (lines per mm)``."""
    density = float(lines_per_mm)
    if density <= 0.0:
        raise ValueError(f"line density must be positive, got {density}")
    return Quantity(1e6 / density, _unit("nm"))


def lines_per_mm_from_period(period: Any) -> float:
    """The inverse of :func:`period_from_lines_per_mm`."""
    value, unit = _length_of(period, "lines_per_mm_from_period")
    nanometres = unit.to_value(value, _unit("nm"))
    if nanometres <= 0.0:
        raise ValueError("the grating period must be positive")
    return 1e6 / nanometres


def grating_equation(
    wavelength: Any, period: Any, order: int = 1, incidence: Any = None
) -> Quantity:
    """Diffracted angle for one order, in radians."""
    lam, lam_unit = _length_of(wavelength, "grating_equation")
    d, d_unit = _length_of(period, "grating_equation")
    lam = lam_unit.to_value(lam, d_unit)
    theta_i = 0.0 if incidence is None else _angle_of(incidence, "grating_equation")
    sine_m = order * lam / d - math.sin(theta_i)
    if abs(sine_m) > 1.0:
        raise ValueError(
            f"no diffracted order: order {order} at {math.degrees(theta_i):.3f} deg "
            f"would need sin(theta) = {sine_m:.4f}, which exceeds one"
        )
    return Quantity(math.asin(sine_m), _RADIAN)


def angular_dispersion(
    wavelength: Any, period: Any, order: int = 1, incidence: Any = None
) -> Quantity:
    """``d(theta)/d(lambda)`` for the given order, in radians per nanometre."""
    lam, lam_unit = _length_of(wavelength, "angular_dispersion")
    d, d_unit = _length_of(period, "angular_dispersion")
    lam = lam_unit.to_value(lam, d_unit)
    theta_m = grating_equation(wavelength, period, order=order, incidence=incidence)
    return Quantity(order / (d * math.cos(theta_m.to_value("rad"))), _RADIAN / _unit("nm"))


def blaze_angle(wavelength: Any, period: Any, order: int = 1) -> Quantity:
    """Littrow angle: incidence and diffraction coincide, ``sin(theta) = m lambda / 2d``."""
    lam, lam_unit = _length_of(wavelength, "blaze_angle")
    d, d_unit = _length_of(period, "blaze_angle")
    lam = lam_unit.to_value(lam, d_unit)
    sine = order * lam / (2.0 * d)
    if abs(sine) > 1.0:
        raise ValueError(
            f"order {order} cannot be used in the Littrow configuration at this "
            f"wavelength: m lambda / 2d = {sine:.4f} exceeds one"
        )
    return Quantity(math.asin(sine), _RADIAN)


def resolving_power(order: int, illuminated_lines: float) -> float:
    """``R = m N = lambda / delta-lambda``."""
    lines = float(illuminated_lines)
    if lines <= 0.0:
        raise ValueError("the number of illuminated lines must be positive")
    return float(abs(order) * lines)


def wavelength_resolution(order: int, illuminated_lines: float, wavelength: Any) -> Quantity:
    """The smallest resolvable wavelength difference."""
    lam, unit = _length_of(wavelength, "wavelength_resolution")
    return Quantity(lam / resolving_power(order, illuminated_lines), unit)


def free_spectral_range_of_grating(wavelength: Any, order: int = 1) -> Quantity:
    """Spacing between successive orders: ``lambda / m``."""
    if order == 0:
        raise ValueError("order zero carries no dispersion")
    lam, unit = _length_of(wavelength, "free_spectral_range_of_grating")
    return Quantity(lam / abs(order), unit)


__all__ = [
    "angular_dispersion",
    "blaze_angle",
    "free_spectral_range_of_grating",
    "grating_equation",
    "lines_per_mm_from_period",
    "period_from_lines_per_mm",
    "resolving_power",
    "wavelength_resolution",
]
