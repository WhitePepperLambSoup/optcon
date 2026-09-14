"""Gauss-Hermite modes and the decomposition of a sampled field onto them.

The field layer has two representations: a sampled grid
(:class:`~optcon.propagation.Field`) and a modal basis, which is what this
module provides.  ``decompose`` and ``reconstruct`` are the only sanctioned
way to move between them, so a calculation cannot quietly treat one as the
other.

The modes are the standard Hermite-Gauss set

    u_m(x) = (2/pi)^(1/4) / sqrt(2^m m! w) H_m(sqrt(2) x / w) exp(-x^2 / w^2),

which is orthonormal with respect to the area measure, so the power in each
mode is the squared magnitude of its coefficient.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.special import eval_genlaguerre, eval_hermite

from .errors import DimensionError
from .propagation import Field, power
from .quantity import Quantity
from .units import Unit
from .units import unit as _unit

_LENGTH = _unit("m").dimension


def _waist_in(value: Any, target: Unit, context: str) -> float:
    if isinstance(value, Quantity):
        if value.unit.dimension != _LENGTH:
            raise DimensionError(f"{context}: expected a length, got {value.unit}")
        return float(value.to_value(target))
    raise TypeError(
        f"{context}: expected a waist with units, e.g. q(50, 'um'), "
        f"got a bare {type(value).__name__}"
    )


def hermite_gauss(
    x,
    y,
    m: int,
    n: int,
    waist: Any,
    *,
    radius_of_curvature: Any = None,
    wavelength: Any = None,
) -> np.ndarray:
    """The normalised ``HG_mn`` mode, sampled on coordinates ``x`` and ``y``.

    With no radius of curvature the mode is a waist: a flat wavefront at the
    plane where it is evaluated.  Give the wavefront radius (and the
    wavelength) to describe the same mode away from its waist, where the
    curved phase front matters: a diverging Gaussian is only a pure single
    mode once that phase is included.
    """
    if m < 0 or n < 0:
        raise ValueError(f"mode orders must be non-negative, got ({m}, {n})")
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    # The coordinates are in the unit the caller used for the waist.  Anything
    # else that enters the phase - the curvature radius and the wavelength -
    # is converted into that same unit here, because mixing three units inside
    # one exponential is exactly the mistake this library exists to prevent.
    if isinstance(waist, Quantity):
        target = waist.unit
        w = float(waist.value)
    else:
        target = None
        w = float(waist)
    profile = _one_dimension(x, m, w) * _one_dimension(y, n, w)
    if radius_of_curvature is None:
        return profile
    if wavelength is None:
        raise TypeError(
            "hermite_gauss needs the wavelength to apply a wavefront curvature"
        )
    radius = _in_unit(radius_of_curvature, target, "radius_of_curvature")
    lam = _in_unit(wavelength, target, "wavelength")
    if not math.isfinite(radius):
        return profile
    # Sign convention: the propagators in :mod:`optcon.propagation` use the
    # exp(+i k z) transfer function, under which a diverging beam acquires a
    # phase of +k r^2 / (2R).  That was measured, not assumed: fitting the
    # quadratic phase of a propagated Gaussian returns |R| = R_analytic with
    # this sign.
    wave_number = 2.0 * math.pi / lam
    phase = wave_number * (x**2 + y**2) / (2.0 * radius)
    return profile * np.exp(1.0j * phase)


def _in_unit(value: Any, target: Unit | None, name: str) -> float:
    if isinstance(value, Quantity):
        if target is None:
            return float(value.value)
        return float(value.to_value(target))
    return float(value)


def _one_dimension(coordinate: np.ndarray, order: int, waist: float) -> np.ndarray:
    normalisation = (2.0 / math.pi) ** 0.25 / math.sqrt(2**order * math.factorial(order) * waist)
    scaled = math.sqrt(2.0) * coordinate / waist
    return normalisation * eval_hermite(order, scaled) * np.exp(-(coordinate**2) / waist**2)


def laguerre_gauss(
    x,
    y,
    p: int,
    ell: int,
    waist: Any,
    *,
    radius_of_curvature: Any = None,
    wavelength: Any = None,
) -> np.ndarray:
    """The normalised ``LG_p^l`` mode: a vortex of charge ``l`` when ``l != 0``.

    The Hermite-Gauss set is separable in x and y; the Laguerre-Gauss set is
    separable in radius and azimuth, and its ``l != 0`` members carry orbital
    angular momentum and a dark core on axis.  Both span the same space, so a
    beam can be described in either - which is exactly why the basis has to be
    named when a field is decomposed.
    """
    if p < 0:
        raise ValueError(f"the radial index cannot be negative, got {p}")
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if isinstance(waist, Quantity):
        target: Unit | None = waist.unit
        w = float(waist.value)
    else:
        target, w = None, float(waist)
    if w <= 0.0:
        raise ValueError("the waist must be positive")

    radius = np.hypot(x, y)
    azimuth = np.arctan2(y, x)
    order = abs(ell)
    normalisation = (
        math.sqrt(2.0 / math.pi)
        / w
        * math.sqrt(math.factorial(p) / math.factorial(p + order))
    )
    scaled = 2.0 * radius**2 / w**2
    radial = (
        normalisation
        * (math.sqrt(2.0) * radius / w) ** order
        * eval_genlaguerre(p, order, scaled)
        * np.exp(-(radius**2) / w**2)
    )
    profile = radial * np.exp(1.0j * ell * azimuth)
    if radius_of_curvature is None:
        return profile
    if wavelength is None:
        raise TypeError(
            "laguerre_gauss needs the wavelength to apply a wavefront curvature"
        )
    curvature = _in_unit(radius_of_curvature, target, "radius_of_curvature")
    lam = _in_unit(wavelength, target, "wavelength")
    if not math.isfinite(curvature):
        return profile
    wave_number = 2.0 * math.pi / lam
    return profile * np.exp(1.0j * wave_number * (x**2 + y**2) / (2.0 * curvature))


def _basis_function(
    basis: str, x, y, first: int, second: int, waist: Quantity, **options
) -> np.ndarray:
    if basis == "hermite":
        return hermite_gauss(x, y, first, second, waist, **options)
    if basis == "laguerre":
        return laguerre_gauss(x, y, first, second, waist, **options)
    raise ValueError(f"basis must be 'hermite' or 'laguerre', got {basis!r}")


def decompose(
    field: Field,
    waist: Any,
    max_order: int = 3,
    *,
    radius_of_curvature: Any = None,
    basis: str = "hermite",
) -> dict[tuple[int, int], complex]:
    """Project ``field`` onto the Hermite-Gauss modes up to ``max_order``.

    Give ``radius_of_curvature`` when the field is not at a waist, otherwise
    a curved wavefront leaks into the higher-order modes.
    """
    if not isinstance(field, Field):
        raise TypeError("decompose expects a Field")
    if max_order < 0:
        raise ValueError("the maximum mode order cannot be negative")
    if basis not in {"hermite", "laguerre"}:
        raise ValueError(f"basis must be 'hermite' or 'laguerre', got {basis!r}")
    local_waist = _waist_in(waist, field.spacing.unit, "decompose")
    axis = field.coordinates
    quantum_waist = Quantity(local_waist, field.spacing.unit)

    if basis == "hermite" and max_order >= 0:
        # A Hermite-Gauss mode is u_m(x) u_n(y), and the optional wavefront
        # curvature is also separable, so the whole coefficient matrix is two
        # matrix products: C = dx^2 U^H F U.  That replaces one full 2-D mode
        # array per (m, n) with one 1-D profile per order.
        return _decompose_separable(
            field, axis, quantum_waist, max_order, radius_of_curvature
        )

    x, y = np.meshgrid(axis, axis)
    cell = field.spacing.value**2
    coefficients: dict[tuple[int, int], complex] = {}
    for first in range(max_order + 1):
        if basis == "laguerre":
            second_range = range(-max_order, max_order + 1)
        else:
            second_range = range(max_order + 1)
        for second in second_range:
            function = _basis_function(
                basis,
                x,
                y,
                first,
                second,
                quantum_waist,
                radius_of_curvature=radius_of_curvature,
                wavelength=field.wavelength,
            )
            coefficients[(first, second)] = complex(
                np.sum(np.conj(function) * field.amplitude) * cell
            )
    return coefficients


def _decompose_separable(
    field: Field,
    axis: np.ndarray,
    waist: Quantity,
    max_order: int,
    radius_of_curvature: Any,
) -> dict[tuple[int, int], complex]:
    """Hermite-Gauss decomposition by two matrix products."""
    columns = np.empty((max_order + 1, axis.size), dtype=complex)
    for order in range(max_order + 1):
        columns[order] = _one_dimension(axis, order, waist.value)
    if radius_of_curvature is not None:
        curvature = _in_unit(radius_of_curvature, waist.unit, "radius_of_curvature")
        lam = _in_unit(field.wavelength, waist.unit, "wavelength")
        if math.isfinite(curvature):
            wave_number = 2.0 * math.pi / lam
            columns *= np.exp(1.0j * wave_number * axis**2 / (2.0 * curvature))

    # The field array is indexed [y, x], and the mode is u_m(x) u_n(y), so the
    # coefficient C[m, n] = sum_{x,y} conj(u_m(x)) conj(u_n(y)) F[y, x].  With
    # U[i, m] = u_m(axis[i]) that is (U^H (F U*))^T.
    # V[i, m] = u_m(axis_i) phi(axis_i) is the un-conjugated profile; the
    # coefficient uses its conjugate on both sides, which only matters once a
    # wavefront curvature makes the profile complex.
    basis = columns.conj().T  # W = conj(V)
    coefficients = (basis.T @ (field.amplitude @ basis)).T
    coefficients = coefficients * field.spacing.value**2
    return {
        (first, second): complex(coefficients[first, second])
        for first in range(max_order + 1)
        for second in range(max_order + 1)
    }


def mode_power_fractions(
    coefficients: dict[tuple[int, int], complex], field: Field
) -> dict[tuple[int, int], float]:
    """Fraction of the field's power carried by each mode."""
    total = power(field).value
    if total <= 0.0:
        raise ValueError("this field carries no power, so it has no mode content")
    return {
        mode: float(abs(value) ** 2) / total for mode, value in coefficients.items()
    }


def mode_content(
    field: Field,
    waist: Any,
    max_order: int = 3,
    *,
    radius_of_curvature: Any = None,
) -> list[tuple[tuple[int, int], float]]:
    """Modes sorted by descending power fraction."""
    fractions = mode_power_fractions(
        decompose(field, waist, max_order, radius_of_curvature=radius_of_curvature),
        field,
    )
    return sorted(fractions.items(), key=lambda item: -item[1])


def reconstruct(
    coefficients: dict[tuple[int, int], complex],
    field: Field,
    waist: Any,
    *,
    radius_of_curvature: Any = None,
) -> Field:
    """Rebuild a field from coefficients, on the grid of ``field``.

    The waist has to be given because it is the basis the coefficients are
    expressed in, and a modal description without its basis is meaningless -
    which is the whole point of keeping the two representations apart.
    """
    if not isinstance(field, Field):
        raise TypeError("reconstruct expects the field whose grid should be reused")
    axis = field.coordinates
    local_waist = _waist_in(waist, field.spacing.unit, "reconstruct")
    # Rebuild through a small coefficient matrix instead of one full 2-D mode
    # array per coefficient: sum_mn C[m, n] u_m(x) u_n(y) is U C U^T.
    highest = max(max(first, second) for first, second in coefficients)
    columns = np.empty((highest + 1, axis.size), dtype=complex)
    for order in range(highest + 1):
        columns[order] = _one_dimension(axis, order, local_waist)
    if radius_of_curvature is not None:
        curvature = _in_unit(
            radius_of_curvature, field.spacing.unit, "radius_of_curvature"
        )
        lam = _in_unit(field.wavelength, field.spacing.unit, "wavelength")
        if math.isfinite(curvature):
            wave_number = 2.0 * math.pi / lam
            columns *= np.exp(1.0j * wave_number * axis**2 / (2.0 * curvature))
    matrix = np.zeros((highest + 1, highest + 1), dtype=complex)
    for (first, second), value in coefficients.items():
        matrix[first, second] = value
    # U is indexed [coordinate, order], and the field is indexed [y, x]
    profiles = columns.T
    total = profiles @ matrix @ profiles.T
    return Field(total, field.spacing, field.wavelength)


__all__ = [
    "decompose",
    "hermite_gauss",
    "laguerre_gauss",
    "mode_content",
    "mode_power_fractions",
    "reconstruct",
]
