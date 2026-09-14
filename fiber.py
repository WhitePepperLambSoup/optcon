"""Step-index optical fibre.

Relations follow Okamoto, *Fundamentals of Optical Waveguides*, ch. 3, and
Gloge (1971) for the normalised propagation constant:

* numerical aperture ``NA = sqrt(n_core^2 - n_clad^2)``
* normalised frequency ``V = 2 pi a NA / lambda``
* single mode when ``V < 2.4048``, the first zero of ``J_0``
* Marcuse's mode field diameter ``MFD = 2a (0.65 + 1.619/V^1.5 + 2.879/V^6)``
* Gloge's approximation ``b = (1.1428 - 0.996/V)^2`` for the fractional
  mode index, giving ``n_eff = n_clad + b (n_core - n_clad)``
"""

from __future__ import annotations

import math
from typing import Any

from .errors import DimensionError
from .quantity import Quantity
from .units import Unit
from .units import unit as _unit

#: first zero of the Bessel function J_0, the LP11 cutoff
V_CUTOFF = 2.404825557695773

_LENGTH = _unit("m").dimension


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


def _length_of(value: Any, context: str) -> tuple[float, Unit]:
    if isinstance(value, Quantity):
        if value.unit.dimension != _LENGTH:
            raise DimensionError(f"{context}: expected a length, got {value.unit}")
        return float(value.value), value.unit
    raise TypeError(
        f"{context}: expected a length with units, e.g. q(4.1, 'um'), "
        f"got a bare {type(value).__name__}"
    )


def numerical_aperture(n_core: Any, n_clad: Any) -> float:
    """Half-angle acceptance cone, ``sqrt(n_core^2 - n_clad^2)``."""
    core = _index_of(n_core, "numerical_aperture")
    clad = _index_of(n_clad, "numerical_aperture")
    if core <= clad:
        raise ValueError(
            f"light is guided only when n_core > n_clad, got {core} and {clad}"
        )
    return math.sqrt(core * core - clad * clad)


def acceptance_angle(n_core: Any, n_clad: Any) -> Quantity:
    """The largest external half angle that is still guided, in radians."""
    return Quantity(math.asin(numerical_aperture(n_core, n_clad)), _unit("rad"))


def v_number(core_radius: Any, na: float, wavelength: Any) -> float:
    """Normalised frequency ``V = 2 pi a NA / lambda``."""
    radius, unit = _length_of(core_radius, "v_number")
    lam, lam_unit = _length_of(wavelength, "v_number")
    lam = lam_unit.to_value(lam, unit)
    if radius <= 0.0 or lam <= 0.0:
        raise ValueError("core radius and wavelength must be positive")
    return 2.0 * math.pi * radius * float(na) / lam


def is_single_mode(v: float, cutoff: float = V_CUTOFF) -> bool:
    """Whether only the LP01 mode is guided."""
    return bool(v < cutoff)


def cutoff_wavelength(core_radius: Any, na: float, cutoff: float = V_CUTOFF) -> Quantity:
    """The wavelength above which the fibre guides only one mode."""
    radius, unit = _length_of(core_radius, "cutoff_wavelength")
    return Quantity(2.0 * math.pi * radius * float(na) / cutoff, unit)


def mode_field_diameter(core_radius: Any, v: float) -> Quantity:
    """Marcuse's approximation to the mode field diameter.

    Fitted for ``1.2 < V < 2.4``, the single-mode regime; it is monotonic
    there and approaches ``0.65 * 2a`` as ``V`` grows, so it should not be
    extrapolated to strongly multimode fibres.
    """
    radius, unit = _length_of(core_radius, "mode_field_diameter")
    if v <= 0.0:
        raise ValueError("V must be positive")
    shape = 0.65 + 1.619 / v**1.5 + 2.879 / v**6
    return Quantity(2.0 * radius * shape, unit)


def mode_field_radius(core_radius: Any, v: float) -> float:
    """Half the mode field diameter, in the core radius's own unit."""
    return mode_field_diameter(core_radius, v).value / 2.0


def propagation_constant(
    wavelength: Any, core_radius: Any, n_core: Any, n_clad: Any
) -> Quantity:
    """``beta`` of the fundamental mode, via Gloge's approximation."""
    core = _index_of(n_core, "propagation_constant")
    clad = _index_of(n_clad, "propagation_constant")
    na = numerical_aperture(core, clad)
    radius, unit = _length_of(core_radius, "propagation_constant")
    lam, lam_unit = _length_of(wavelength, "propagation_constant")
    lam = lam_unit.to_value(lam, unit)
    v = 2.0 * math.pi * radius * na / lam
    if v < 1.0:
        raise ValueError(
            f"Gloge's approximation is only valid for V > 1; this fibre has V = {v:.3f}"
        )
    fractional = (1.1428 - 0.9960 / v) ** 2
    effective_index = clad + fractional * (core - clad)
    k0 = 2.0 * math.pi / (lam * unit.factor)
    return Quantity(k0 * effective_index, _unit("1/m"))


def coupling_loss_db(efficiency: float) -> float:
    """Coupling efficiency expressed as a positive loss in decibels."""
    value = float(efficiency)
    if value < 0.0:
        raise ValueError("a coupling efficiency cannot be negative")
    if value == 0.0:
        return math.inf
    if value > 1.0:
        raise ValueError(f"a coupling efficiency cannot exceed one, got {value}")
    return -10.0 * math.log10(value)


__all__ = [
    "V_CUTOFF",
    "acceptance_angle",
    "coupling_loss_db",
    "cutoff_wavelength",
    "is_single_mode",
    "mode_field_diameter",
    "mode_field_radius",
    "numerical_aperture",
    "propagation_constant",
    "v_number",
]
