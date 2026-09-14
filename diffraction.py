"""Diffraction limits: the Airy pattern, resolution, and the Strehl ratio.

Relations follow Born & Wolf, *Principles of Optics*, ch. 8, and Mahajan,
*Aberration Theory Made Simple*, for the Marechal approximation to the Strehl
ratio.

The Airy pattern is parameterised by ``u = pi r / (lambda f/#)``, the argument
of the Bessel functions, so ``u = 0`` is the axis and ``u = 3.8317`` (the
first zero of ``J1``) is the first dark ring.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.special import j0, j1

from .errors import DimensionError
from .quantity import Quantity
from .units import Unit
from .units import unit as _unit

_LENGTH = _unit("m").dimension
_RADIAN = _unit("rad")

#: first zero of J1, the radius of the Airy disc measured in u
FIRST_ZERO = 3.831705970207512


def _length_of(value: Any, context: str) -> tuple[float, Unit]:
    if isinstance(value, Quantity):
        if value.unit.dimension != _LENGTH:
            raise DimensionError(f"{context}: expected a length, got {value.unit}")
        return float(value.value), value.unit
    raise TypeError(
        f"{context}: expected a length with units, e.g. q(550, 'nm'), "
        f"got a bare {type(value).__name__}"
    )


def f_number(focal_length: Any, aperture_diameter: Any) -> float:
    """The ``f/#`` of a system."""
    focal, unit = _length_of(focal_length, "f_number")
    diameter, diameter_unit = _length_of(aperture_diameter, "f_number")
    diameter = diameter_unit.to_value(diameter, unit)
    if diameter <= 0.0:
        raise ValueError("the aperture diameter must be positive")
    return focal / diameter


def airy_radius(wavelength: Any, f_number_value: float) -> Quantity:
    """Radius of the first dark ring of the Airy disc, ``1.22 lambda f/#``."""
    lam, unit = _length_of(wavelength, "airy_radius")
    if f_number_value <= 0.0:
        raise ValueError("the f-number must be positive")
    return Quantity(1.22 * lam * float(f_number_value), unit)


def diffraction_limited_spot(wavelength: Any, f_number_value: float) -> Quantity:
    """Diameter of the Airy disc, ``2.44 lambda f/#``."""
    radius = airy_radius(wavelength, f_number_value)
    return Quantity(2.0 * radius.value, radius.unit)


def angular_resolution(wavelength: Any, aperture_diameter: Any) -> Quantity:
    """Rayleigh criterion ``1.22 lambda / D``, in radians."""
    lam, lam_unit = _length_of(wavelength, "angular_resolution")
    diameter, unit = _length_of(aperture_diameter, "angular_resolution")
    if diameter <= 0.0:
        raise ValueError("the aperture diameter must be positive")
    diameter_in_lam_unit = unit.to_value(diameter, lam_unit)
    return Quantity(1.22 * lam / diameter_in_lam_unit, _RADIAN)


def airy_intensity(u: float) -> float | np.ndarray:
    """Normalised Airy intensity ``(2 J1(u) / u)^2``, equal to one on axis.

    Accepts a scalar or an array, and returns the same shape.
    """
    argument = np.asarray(u, dtype=float)
    safe = np.where(argument == 0.0, 1.0, argument)
    ratio = np.where(argument == 0.0, 1.0, 2.0 * j1(safe) / safe)
    squared = ratio * ratio
    return float(squared) if squared.ndim == 0 else squared


def encircled_energy(u: float) -> float:
    """Fraction of the Airy pattern's power inside radius ``u``."""
    argument = float(u)
    if argument < 0.0:
        raise ValueError("the radius parameter u cannot be negative")
    return float(1.0 - j0(argument) ** 2 - j1(argument) ** 2)


def strehl_ratio(rms_wavefront: Any, wavelength: Any) -> float:
    """Marechal approximation ``exp(-(2 pi sigma / lambda)^2)``."""
    sigma, unit = _length_of(rms_wavefront, "strehl_ratio")
    lam, lam_unit = _length_of(wavelength, "strehl_ratio")
    lam = lam_unit.to_value(lam, unit)
    if lam <= 0.0:
        raise ValueError("the wavelength must be positive")
    if sigma < 0.0:
        raise ValueError("an RMS wavefront error cannot be negative")
    return float(math.exp(-((2.0 * math.pi * sigma / lam) ** 2)))


def rms_wavefront_from_strehl(strehl: float, wavelength: Any) -> Quantity:
    """Invert the Marechal approximation."""
    value = float(strehl)
    if not 0.0 < value <= 1.0:
        raise ValueError(f"a Strehl ratio must lie in (0, 1], got {value}")
    lam, unit = _length_of(wavelength, "rms_wavefront_from_strehl")
    return Quantity(lam * math.sqrt(-math.log(value)) / (2.0 * math.pi), unit)


def gaussian_far_field_half_angle(waist: Any, wavelength: Any) -> Quantity:
    """Divergence of a Gaussian mode, ``lambda / (pi w0)``."""
    w0, unit = _length_of(waist, "gaussian_far_field_half_angle")
    lam, lam_unit = _length_of(wavelength, "gaussian_far_field_half_angle")
    lam = lam_unit.to_value(lam, unit)
    if w0 <= 0.0 or lam <= 0.0:
        raise ValueError("waist and wavelength must be positive")
    return Quantity(lam / (math.pi * w0), _RADIAN)


__all__ = [
    "FIRST_ZERO",
    "airy_intensity",
    "airy_radius",
    "angular_resolution",
    "diffraction_limited_spot",
    "encircled_energy",
    "f_number",
    "gaussian_far_field_half_angle",
    "rms_wavefront_from_strehl",
    "strehl_ratio",
]
