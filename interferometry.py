"""Interferometry: fringe visibility, coherence, and the Fabry-Perot etalon.

Relations follow Hecht, *Optics*, ch. 9, and Born & Wolf, ch. 7.  The Airy
transmission of an etalon is written in terms of the phase detuning
``delta``, so ``delta = 2 pi m`` is a resonance and ``delta = pi`` is the
point of minimum transmission.
"""

from __future__ import annotations

import math
from typing import Any

from .errors import DimensionError
from .quantity import POWER, Quantity
from .units import Unit
from .units import unit as _unit

C_LIGHT = 299792458.0

_LENGTH = _unit("m").dimension
_FREQUENCY = _unit("Hz").dimension


def _length_of(value: Any, context: str) -> tuple[float, Unit]:
    if isinstance(value, Quantity):
        if value.unit.dimension != _LENGTH:
            raise DimensionError(f"{context}: expected a length, got {value.unit}")
        return float(value.value), value.unit
    raise TypeError(
        f"{context}: expected a length with units, e.g. q(633, 'nm'), "
        f"got a bare {type(value).__name__}"
    )


def _linewidth_in_hz(value: Any, context: str) -> float:
    if isinstance(value, Quantity):
        if value.unit.dimension != _FREQUENCY:
            raise DimensionError(f"{context}: expected a frequency, got {value.unit}")
        width = float(value.to_value("Hz"))
    else:
        width = float(value)
    if width <= 0.0:
        raise ValueError(f"{context}: the linewidth must be positive, got {width}")
    return width


def fringe_visibility(intensity_max: float, intensity_min: float) -> float:
    """``(Imax - Imin) / (Imax + Imin)``."""
    maximum = float(intensity_max)
    minimum = float(intensity_min)
    if maximum < 0.0 or minimum < 0.0:
        raise ValueError("intensities cannot be negative")
    if maximum < minimum:
        raise ValueError("the maximum intensity cannot be below the minimum")
    total = maximum + minimum
    if total == 0.0:
        return 0.0
    return (maximum - minimum) / total


def visibility_from_intensities(first: float, second: float) -> float:
    """Visibility of two-beam interference with unequal beam intensities."""
    a = float(first)
    b = float(second)
    if a < 0.0 or b < 0.0:
        raise ValueError("intensities cannot be negative")
    total = a + b
    if total == 0.0:
        return 0.0
    return 2.0 * math.sqrt(a * b) / total


def coherence_time(linewidth: Any) -> Quantity:
    """``tau_c = 1 / delta-nu``."""
    return Quantity(1.0 / _linewidth_in_hz(linewidth, "coherence_time"), _unit("s"))


def coherence_length(linewidth: Any) -> Quantity:
    """``L_c = c / delta-nu``."""
    return Quantity(C_LIGHT / _linewidth_in_hz(linewidth, "coherence_length"), _unit("m"))


def coherence_length_from_wavelength_spread(center_wavelength: Any, spread: Any) -> Quantity:
    """``L_c = lambda^2 / delta-lambda``, the wavelength-domain equivalent."""
    center, center_unit = _length_of(center_wavelength, "coherence_length_from_wavelength_spread")
    delta, delta_unit = _length_of(spread, "coherence_length_from_wavelength_spread")
    delta = delta_unit.to_value(delta, center_unit)
    if center <= 0.0 or delta <= 0.0:
        raise ValueError("the wavelength and its spread must both be positive")
    metres = center_unit.to_value(center, _unit("m"))
    spread_metres = center_unit.to_value(delta, _unit("m"))
    return Quantity(metres**2 / spread_metres, _unit("m"))


def etalon_transmission(detuning: float, reflectance: float) -> float:
    """Airy transmission ``1 / (1 + F sin^2(delta/2))``."""
    coefficient = _finesse_coefficient(reflectance)
    phase = 0.5 * float(detuning)
    return 1.0 / (1.0 + coefficient * math.sin(phase) ** 2)


def _finesse_coefficient(reflectance: float) -> float:
    value = float(reflectance)
    if not 0.0 <= value < 1.0:
        raise ValueError(f"reflectance must lie in [0, 1), got {value}")
    return 4.0 * value / (1.0 - value) ** 2


def peak_to_minimum_ratio(reflectance: float) -> float:
    """Contrast of an etalon, ``1 + F``."""
    return 1.0 + _finesse_coefficient(reflectance)


def two_beam_intensity(
    optical_path_difference: Any, wavelength: Any, first: float, second: float
) -> float:
    """Intensity of two-beam interference, ``I1 + I2 + 2 sqrt(I1 I2) cos(k dOPD)``."""
    difference, unit = _length_of(optical_path_difference, "two_beam_intensity")
    lam, lam_unit = _length_of(wavelength, "two_beam_intensity")
    lam = lam_unit.to_value(lam, unit)
    if lam <= 0.0:
        raise ValueError("the wavelength must be positive")
    a = float(first)
    b = float(second)
    if a < 0.0 or b < 0.0:
        raise ValueError("intensities cannot be negative")
    phase = 2.0 * math.pi * difference / lam
    return a + b + 2.0 * math.sqrt(a * b) * math.cos(phase)


def two_beam_contrast(
    optical_path_difference: Any, wavelength: Any, first: float, second: float
) -> Quantity:
    """Visibility of the fringes produced by two beams, as a power ratio."""
    fringes = visibility_from_intensities(first, second)
    del optical_path_difference, wavelength
    return Quantity(fringes, _unit("1"), POWER)


__all__ = [
    "C_LIGHT",
    "coherence_length",
    "coherence_length_from_wavelength_spread",
    "coherence_time",
    "etalon_transmission",
    "fringe_visibility",
    "peak_to_minimum_ratio",
    "two_beam_contrast",
    "two_beam_intensity",
    "visibility_from_intensities",
]
