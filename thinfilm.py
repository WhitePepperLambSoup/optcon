"""Thin film design: quarter-wave stacks, anti-reflection coatings, Bragg mirrors.

The closed forms here follow Macleod, *Thin-Film Optical Filters*, ch. 3-5.
They are the formulas an engineer uses to *design* a coating; the
:mod:`optcon.engines` adapters are what you reach for to *verify* one with a
full solver.

The single-layer reflectance is the Airy sum for one film at normal
incidence, which reduces to the bare Fresnel result as the thickness goes to
zero and to zero reflectance when the film is a quarter wave of the geometric
mean index.
"""

from __future__ import annotations

import cmath
import math
from typing import Any

from .errors import DimensionError
from .quantity import POWER, Quantity
from .units import Unit
from .units import unit as _unit

_LENGTH = _unit("m").dimension


def _length_of(value: Any, context: str) -> tuple[float, Unit]:
    if isinstance(value, Quantity):
        if value.unit.dimension != _LENGTH:
            raise DimensionError(f"{context}: expected a length, got {value.unit}")
        return float(value.value), value.unit
    raise TypeError(
        f"{context}: expected a length with units, e.g. q(550, 'nm'), "
        f"got a bare {type(value).__name__}"
    )


def _index_of(value: Any, context: str) -> float:
    if isinstance(value, Quantity):
        if not value.unit.is_dimensionless:
            raise DimensionError(
                f"{context}: a refractive index must be dimensionless, got {value.unit}"
            )
        index = float(value.value)
    else:
        index = float(value)
    if index <= 0.0:
        raise ValueError(f"{context}: a refractive index must be positive, got {index}")
    return index


def optical_thickness(geometric_thickness: Any, refractive_index: float) -> Quantity:
    """``n t``: the thickness that matters for phase."""
    value, unit = _length_of(geometric_thickness, "optical_thickness")
    return Quantity(value * _index_of(refractive_index, "optical_thickness"), unit)


def anti_reflection_thickness(refractive_index: float, wavelength: Any) -> Quantity:
    """Quarter-wave thickness ``lambda / (4 n)``."""
    index = _index_of(refractive_index, "anti_reflection_thickness")
    lam, unit = _length_of(wavelength, "anti_reflection_thickness")
    return Quantity(lam / (4.0 * index), unit)


def ideal_anti_reflection_index(n_incident: Any, n_substrate: Any) -> float:
    """The single-layer index that nulls reflection: ``sqrt(n0 * ns)``."""
    first = _index_of(n_incident, "ideal_anti_reflection_index")
    second = _index_of(n_substrate, "ideal_anti_reflection_index")
    return math.sqrt(first * second)


def single_layer_reflectance(
    n_incident: Any,
    n_film: Any,
    n_substrate: Any,
    thickness: Any,
    wavelength: Any,
) -> Quantity:
    """Reflectance of one film on a substrate, at normal incidence."""
    n0 = _index_of(n_incident, "single_layer_reflectance")
    n1 = _index_of(n_film, "single_layer_reflectance")
    ns = _index_of(n_substrate, "single_layer_reflectance")
    thickness_value, unit = _length_of(thickness, "single_layer_reflectance")
    lam, lam_unit = _length_of(wavelength, "single_layer_reflectance")
    lam = lam_unit.to_value(lam, unit)
    if lam <= 0.0:
        raise ValueError("the wavelength must be positive")

    r01 = (n0 - n1) / (n0 + n1)
    r1s = (n1 - ns) / (n1 + ns)
    delta = 2.0 * math.pi * n1 * thickness_value / lam
    phase = cmath.exp(-2.0j * delta)
    amplitude = (r01 + r1s * phase) / (1.0 + r01 * r1s * phase)
    return Quantity(abs(amplitude) ** 2, _unit("1"), POWER)


def bragg_reflectance(
    n_incident: Any,
    n_high: Any,
    n_low: Any,
    n_substrate: Any,
    pairs: int,
) -> float:
    """Peak reflectance of a quarter-wave ``(HL)^N H`` stack.

    ``pairs`` is ``N``: the number of high/low pairs before the terminal
    high-index quarter-wave layer.  The incident medium is therefore followed
    by ``H, L, ..., H, L, H`` and then the substrate.
    """
    n0 = _index_of(n_incident, "bragg_reflectance")
    high = _index_of(n_high, "bragg_reflectance")
    low = _index_of(n_low, "bragg_reflectance")
    substrate = _index_of(n_substrate, "bragg_reflectance")
    if not isinstance(pairs, int) or pairs < 1:
        raise ValueError(f"a stack needs at least one pair, got {pairs!r}")
    if high <= low:
        raise ValueError(
            f"the high index must exceed the low index, got {high} and {low}"
        )
    effective = (high / low) ** (2 * pairs) * high**2 / substrate
    return float(((n0 - effective) / (n0 + effective)) ** 2)


def quarter_wave_stack_reflectance(
    n_high: Any, n_low: Any, n_substrate: Any, pairs: int, n_incident: Any = 1.0
) -> float:
    """Convenience wrapper for an air-incident ``(HL)^N H`` stack."""
    return bragg_reflectance(n_incident, n_high, n_low, n_substrate, pairs)


__all__ = [
    "anti_reflection_thickness",
    "bragg_reflectance",
    "ideal_anti_reflection_index",
    "optical_thickness",
    "quarter_wave_stack_reflectance",
    "single_layer_reflectance",
]
