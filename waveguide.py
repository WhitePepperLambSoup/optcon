"""Guided modes of a symmetric slab waveguide.

The slab is the one dielectric waveguide whose modes follow from a
transcendental equation simple enough to solve exactly:

    even TE modes:  u tan(u) = w
    odd  TE modes: -u cot(u) = w
    u = k0 a sqrt(n_core^2 - n_eff^2),   w = k0 a sqrt(n_eff^2 - n_clad^2),
    u^2 + w^2 = V^2,   V = k0 a NA,   a = half thickness.

Conventions follow Okamoto, *Fundamentals of Optical Waveguides*, ch. 2.  The
roots are bracketed between the poles of ``tan`` and found by bisection, so
every mode returned can be substituted back into the relation that defined
it - which is what the test suite does.

The normalised frequency uses the **half** thickness; the single-mode
condition for a symmetric slab is therefore ``V < pi/2``.
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
        f"{context}: expected a length with units, e.g. q(500, 'nm'), "
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


def normalized_frequency(
    thickness: Any, n_core: Any, n_clad: Any, wavelength: Any
) -> float:
    """``V = k0 a NA``, with ``a`` the half thickness."""
    size, unit = _length_of(thickness, "normalized_frequency")
    core = _index_of(n_core, "normalized_frequency", "n_core")
    clad = _index_of(n_clad, "normalized_frequency", "n_clad")
    lam, lam_unit = _length_of(wavelength, "normalized_frequency")
    lam = lam_unit.to_value(lam, unit)
    if core <= clad:
        raise ValueError(
            f"light is guided only when n_core > n_clad, got {core} and {clad}"
        )
    if size <= 0.0 or lam <= 0.0:
        raise ValueError("the thickness and wavelength must be positive")
    return 2.0 * math.pi * (size / 2.0) * math.sqrt(core * core - clad * clad) / lam


def _residual(u: float, v: float, even: bool) -> float:
    w = math.sqrt(max(v * v - u * u, 0.0))
    if even:
        return u * math.tan(u) - w
    return -u / math.tan(u) - w


def _bisect(v: float, low: float, high: float, even: bool) -> float:
    for _ in range(200):
        middle = 0.5 * (low + high)
        if _residual(middle, v, even) > 0.0:
            high = middle
        else:
            low = middle
    return 0.5 * (low + high)


def _normalized_roots(v: float) -> list[float]:
    """``u`` for every guided mode, ascending - that is, fundamental first."""
    roots: list[float] = []
    half_pi = math.pi / 2.0
    band = 0
    while band * math.pi < v:
        even_low = band * math.pi
        even_high = even_low + half_pi
        odd_low = even_high
        odd_high = even_low + math.pi
        if even_low < v:
            top = min(even_high, v)
            if top > even_low and _residual(top - 1e-12, v, True) > 0.0:
                roots.append(_bisect(v, even_low + 1e-12, top - 1e-12, True))
        if odd_low < v:
            top = min(odd_high, v)
            if top > odd_low and _residual(top - 1e-12, v, False) > 0.0:
                roots.append(_bisect(v, odd_low + 1e-12, top - 1e-12, False))
        band += 1
    # the fundamental mode has the largest propagation constant and hence the
    # smallest u, so ascending u is descending effective index
    roots.sort()
    return roots


def effective_indices(
    thickness: Any, n_core: Any, n_clad: Any, wavelength: Any
) -> list[float]:
    """Effective indices of every guided TE mode, fundamental first."""
    core = _index_of(n_core, "effective_indices", "n_core")
    clad = _index_of(n_clad, "effective_indices", "n_clad")
    v = normalized_frequency(thickness, core, clad, wavelength)
    results = []
    for u in _normalized_roots(v):
        b = 1.0 - (u / v) ** 2  # normalised propagation constant
        results.append(math.sqrt(clad * clad + b * (core * core - clad * clad)))
    return results


def number_of_modes(v: float) -> int:
    """How many TE modes a symmetric slab with this ``V`` supports."""
    if v <= 0.0:
        return 0
    return int(math.floor(2.0 * v / math.pi)) + 1


def single_mode(thickness: Any, n_core: Any, n_clad: Any, wavelength: Any) -> bool:
    """Whether only the fundamental mode is guided, ``V < pi/2``."""
    return (
        number_of_modes(normalized_frequency(thickness, n_core, n_clad, wavelength)) == 1
    )


def confinement_factor(
    thickness: Any, n_core: Any, n_clad: Any, wavelength: Any, mode: int = 0
) -> float:
    """Fraction of a mode's power carried inside the core.

    The exact slab result is ``Gamma = w / (w + 1/w)`` in the normalised
    quantities ``u`` and ``w`` of that mode.
    """
    core = _index_of(n_core, "confinement_factor", "n_core")
    clad = _index_of(n_clad, "confinement_factor", "n_clad")
    v = normalized_frequency(thickness, core, clad, wavelength)
    roots = _normalized_roots(v)
    if not roots:
        raise ValueError(f"this slab guides no mode at V = {v:.4f}")
    if not 0 <= mode < len(roots):
        raise ValueError(
            f"mode {mode} does not exist; this slab guides {len(roots)} mode(s)"
        )
    u = roots[mode]
    w = math.sqrt(v * v - u * u)
    return float(w / (w + 1.0 / w))


__all__ = [
    "confinement_factor",
    "effective_indices",
    "normalized_frequency",
    "number_of_modes",
    "single_mode",
]
