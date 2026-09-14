"""Second-harmonic generation: phase matching and effective interaction length.

Relations follow Boyd, *Nonlinear Optics*, ch. 2.  For a fundamental
wavelength ``lambda`` the mismatch between the driven polarisation and the
free second-harmonic wave is

    delta-k = k(2w) - 2 k(w) = (4 pi / lambda) [ n(2w) - n(w) ],

so the coherence length over which the two drift out of step is
``L_coh = pi / delta-k = lambda / (4 [n(2w) - n(w)])``.  Quasi-phase matching
compensates it by flipping the sign of the nonlinearity every ``2 L_coh``.

The effective interaction length of a crystal of length ``L`` is
``L_eff = sin(delta-k L / 2) / (delta-k / 2)``, which reduces to ``L`` when
the mismatch vanishes and passes through zero when ``delta-k L = 2 pi``.
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
        f"{context}: expected a length with units, e.g. q(1064, 'nm'), "
        f"got a bare {type(value).__name__}"
    )


def _index_of(value: Any, context: str, name: str) -> float:
    if isinstance(value, Quantity):
        if not value.unit.is_dimensionless:
            raise DimensionError(
                f"{context}: {name} must be dimensionless, got {value.unit}"
            )
        index = float(value.value)
    else:
        index = float(value)
    if index <= 0.0:
        raise ValueError(f"{context}: {name} must be positive, got {index}")
    return index


def phase_mismatch(
    fundamental_wavelength: Any, n_second_harmonic: Any, n_fundamental: Any
) -> Quantity:
    """``delta-k`` in inverse length units."""
    lam, unit = _length_of(fundamental_wavelength, "phase_mismatch")
    second = _index_of(n_second_harmonic, "phase_mismatch", "n_second_harmonic")
    first = _index_of(n_fundamental, "phase_mismatch", "n_fundamental")
    return Quantity(4.0 * math.pi * (second - first) / lam, 1.0 / unit)


def coherence_length(
    fundamental_wavelength: Any, n_second_harmonic: Any, n_fundamental: Any
) -> Quantity:
    """``lambda / (4 [n(2w) - n(w)])``; infinite under perfect matching."""
    mismatch = phase_mismatch(fundamental_wavelength, n_second_harmonic, n_fundamental)
    if mismatch.value == 0.0:
        return Quantity(math.inf, 1.0 / mismatch.unit)
    return Quantity(math.pi / mismatch.value, 1.0 / mismatch.unit)


def quasi_phase_matching_period(
    fundamental_wavelength: Any, n_second_harmonic: Any, n_fundamental: Any
) -> Quantity:
    """Poling period ``2 L_coh`` that restores the sign every coherence length."""
    length = coherence_length(fundamental_wavelength, n_second_harmonic, n_fundamental)
    if math.isinf(length.value):
        return Quantity(math.inf, length.unit)
    return Quantity(2.0 * length.value, length.unit)


def effective_interaction_length(crystal_length: Any, mismatch: Any) -> Quantity:
    """``sin(delta-k L / 2) / (delta-k / 2)``.

    The length over which a crystal of length ``L`` actually builds up
    second-harmonic amplitude; it is bounded by ``2 L_coh`` no matter how long
    the crystal is.
    """
    length, unit = _length_of(crystal_length, "effective_interaction_length")
    if isinstance(mismatch, Quantity):
        delta_k = float(mismatch.to_value(1.0 / unit))
    else:
        delta_k = float(mismatch)
    if length <= 0.0:
        raise ValueError("the crystal length must be positive")
    if delta_k == 0.0:
        return Quantity(length, unit)
    return Quantity(math.sin(delta_k * length / 2.0) / (delta_k / 2.0), unit)


__all__ = [
    "coherence_length",
    "effective_interaction_length",
    "phase_mismatch",
    "quasi_phase_matching_period",
]
