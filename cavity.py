"""Two-mirror optical cavities.

Relations follow Siegman, *Lasers* (1986), ch. 19-21: the ``g`` parameters
``g_i = 1 - L/R_i``, the stability diagram, the free spectral range ``c/2L``,
the finesse ``pi sqrt(R)/(1 - R)``, and the ``g``-parameter expressions for
the waist and the spot sizes on the mirrors.

Stability is judged with the practical criterion ``0 <= g1 g2 < 1`` rather
than the inclusive textbook one.  The excluded point ``g1 g2 = 1`` is the
flat-flat (or concentric) corner of the diagram, where the mode size diverges;
a cavity there is geometrically "on the boundary" and useless in a laboratory.
"""

from __future__ import annotations

import math
from typing import Any

from .errors import DimensionError
from .quantity import Quantity
from .units import Unit
from .units import unit as _unit

C_LIGHT = 299792458.0

_LENGTH = _unit("m").dimension


def _length_of(value: Any, context: str) -> tuple[float, Unit]:
    if isinstance(value, Quantity):
        if value.unit.dimension != _LENGTH:
            raise DimensionError(f"{context}: expected a length, got {value.unit}")
        return float(value.value), value.unit
    raise TypeError(
        f"{context}: expected a length with units, e.g. q(300, 'mm'), "
        f"got a bare {type(value).__name__}"
    )


def _metres(value: Any, context: str) -> float:
    magnitude, unit = _length_of(value, context)
    return magnitude * unit.factor


def free_spectral_range(
    length: Any, refractive_index: float = 1.0, two_way: bool = True
) -> Quantity:
    """Mode spacing of a cavity, ``c / (n * path)``.

    ``two_way`` is the default because a linear two-mirror cavity accumulates
    ``2L`` of path per round trip; a ring cavity passes ``two_way=False``.
    """
    path = _metres(length, "free_spectral_range") * float(refractive_index)
    if path <= 0.0:
        raise ValueError("a cavity must have a positive optical length")
    if two_way:
        path *= 2.0
    return Quantity(C_LIGHT / path, _unit("Hz"))


def finesse(mirror_reflectance: float) -> float:
    """Finesse of a symmetric cavity with the given mirror reflectance."""
    reflectance = float(mirror_reflectance)
    if not 0.0 <= reflectance < 1.0:
        raise ValueError(
            f"mirror reflectance must lie in [0, 1), got {reflectance}"
        )
    return math.pi * math.sqrt(reflectance) / (1.0 - reflectance)


def linewidth(fsr: Quantity, cavity_finesse: float) -> Quantity:
    """Full width at half maximum of a resonance, ``FSR / F``."""
    if cavity_finesse <= 0.0:
        raise ValueError("finesse must be positive")
    return Quantity(fsr.to_value("Hz") / cavity_finesse, _unit("Hz"))


def quality_factor(frequency: Quantity, fsr: Quantity, cavity_finesse: float) -> float:
    """``Q = frequency / linewidth``."""
    return float(frequency.to_value("Hz") / linewidth(fsr, cavity_finesse).to_value("Hz"))


def photon_lifetime(fsr: Quantity, cavity_finesse: float) -> Quantity:
    """Energy decay time of the cavity, ``1 / (2 pi linewidth)``."""
    width = linewidth(fsr, cavity_finesse).to_value("Hz")
    return Quantity(1.0 / (2.0 * math.pi * width), _unit("s"))


def g_parameters(length: Any, radius_1: Any, radius_2: Any) -> tuple[float, float]:
    """The two stability parameters ``g_i = 1 - L/R_i``."""
    length_m = _metres(length, "g_parameters")
    first = _metres(radius_1, "g_parameters")
    second = _metres(radius_2, "g_parameters")
    return (
        1.0 - length_m / first,
        1.0 - length_m / second,
    )


def stability_product(length: Any, radius_1: Any, radius_2: Any) -> float:
    """``g1 * g2``; the cavity is usable when this lies in ``[0, 1)``."""
    g1, g2 = g_parameters(length, radius_1, radius_2)
    return float(g1 * g2)


def is_stable_cavity(length: Any, radius_1: Any, radius_2: Any) -> bool:
    """Whether the cavity has a confined, finite-size mode."""
    product = stability_product(length, radius_1, radius_2)
    return bool(0.0 <= product < 1.0)


def round_trip_gouy_phase(g1: float, g2: float) -> Quantity:
    """Gouy phase accumulated per round trip, ``arccos(sqrt(g1 g2))``."""
    product = float(g1) * float(g2)
    if not 0.0 <= product <= 1.0:
        raise ValueError(
            f"the cavity is not stable: g1*g2 = {product}, which lies outside [0, 1]"
        )
    return Quantity(math.acos(math.sqrt(product)), _unit("rad"))


def transverse_mode_spacing(fsr: Quantity, g1: float, g2: float) -> Quantity:
    """Frequency spacing between successive transverse modes."""
    phase = round_trip_gouy_phase(g1, g2).to_value("rad")
    return Quantity(fsr.to_value("Hz") * phase / math.pi, _unit("Hz"))


def waist_size(length: Any, radius_1: Any, radius_2: Any, wavelength: Any) -> Quantity:
    """Radius of the cavity eigenmode at its waist."""
    length_m = _metres(length, "waist_size")
    wavelength_m = _metres(wavelength, "waist_size")
    g1, g2 = g_parameters(length, radius_1, radius_2)
    product = g1 * g2
    if not 0.0 <= product < 1.0:
        raise ValueError(
            f"the cavity is not stable: g1*g2 = {product}; it has no confined mode"
        )
    numerator = product * (1.0 - product)
    denominator = g1 + g2 - 2.0 * product
    waist_squared = (
        wavelength_m * length_m / math.pi * math.sqrt(numerator) / denominator
    )
    return Quantity(math.sqrt(waist_squared), _unit("m"))


def spot_size_at_mirrors(
    length: Any, radius_1: Any, radius_2: Any, wavelength: Any
) -> tuple[Quantity, Quantity]:
    """The eigenmode radius on each mirror."""
    length_m = _metres(length, "spot_size_at_mirrors")
    wavelength_m = _metres(wavelength, "spot_size_at_mirrors")
    g1, g2 = g_parameters(length, radius_1, radius_2)
    product = g1 * g2
    if not 0.0 <= product < 1.0:
        raise ValueError(f"the cavity is not stable: g1*g2 = {product}")
    prefactor = wavelength_m * length_m / math.pi
    first = math.sqrt(prefactor * math.sqrt(g2 / (g1 * (1.0 - product))))
    second = math.sqrt(prefactor * math.sqrt(g1 / (g2 * (1.0 - product))))
    in_metres = _unit("m")
    return Quantity(first, in_metres), Quantity(second, in_metres)


def circulating_power(
    input_power: float, input_reflectance: float, round_trip_loss: float = 0.0
) -> float:
    """Steady-state circulating power on resonance, relative to the input.

    The buildup factor is ``1 / (1 - R(1 - loss))``.
    """
    reflectance = float(input_reflectance)
    loss = float(round_trip_loss)
    if not 0.0 <= reflectance < 1.0:
        raise ValueError("input reflectance must lie in [0, 1)")
    if not 0.0 <= loss < 1.0:
        raise ValueError("round trip loss must lie in [0, 1)")
    denominator = 1.0 - reflectance * (1.0 - loss)
    if denominator <= 0.0:
        raise ValueError("this cavity has no finite circulating power")
    return float(input_power) / denominator


__all__ = [
    "C_LIGHT",
    "circulating_power",
    "finesse",
    "free_spectral_range",
    "g_parameters",
    "is_stable_cavity",
    "linewidth",
    "photon_lifetime",
    "quality_factor",
    "round_trip_gouy_phase",
    "spot_size_at_mirrors",
    "stability_product",
    "transverse_mode_spacing",
    "waist_size",
]
