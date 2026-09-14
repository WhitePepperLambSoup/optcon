"""Wavefront aberrations: Zernike modes, Seidel terms, and their RMS.

Conventions follow Noll (1976) for the index ordering and normalisation: the
modes are orthonormal on the unit disc with respect to the area measure
``dA / pi``, so the RMS wavefront error of a coefficient vector is the
quadrature sum of its coefficients.  That is the property every downstream
calculation depends on, and the test suite verifies it by numerical
integration rather than by assertion.

Seidel relations follow Mahajan, *Optical Imaging and Aberrations*, part 1.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

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
        f"{context}: expected a length with units, e.g. q(633, 'nm'), "
        f"got a bare {type(value).__name__}"
    )


def radial_polynomial(n: int, m: int, rho):
    """Zernike radial polynomial ``R_n^|m|(rho)``."""
    order = abs(int(m))
    if order > n:
        raise ValueError(
            f"the azimuthal order has greater magnitude than n: |m| = {order}, n = {n}"
        )
    if (n - order) % 2 != 0:
        raise ValueError(
            f"n = {n} and m = {m} must have the same parity for a Zernike mode"
        )
    rho = np.asarray(rho, dtype=float)
    total = 0.0
    for k in range((n - order) // 2 + 1):
        coefficient = (
            (-1) ** k
            * math.factorial(n - k)
            / (
                math.factorial(k)
                * math.factorial((n + order) // 2 - k)
                * math.factorial((n - order) // 2 - k)
            )
        )
        total = total + coefficient * rho ** (n - 2 * k)
    return total


def zernike(n: int, m: int, rho, theta):
    """The Noll-normalised Zernike mode ``Z_n^m``."""
    order = abs(int(m))
    normalisation = math.sqrt(n + 1.0) if order == 0 else math.sqrt(2.0 * (n + 1.0))
    radial = radial_polynomial(n, order, rho)
    angle = np.asarray(theta, dtype=float)
    if m >= 0:
        return normalisation * radial * np.cos(order * angle)
    return normalisation * radial * np.sin(order * angle)


def noll_to_nm(index: int) -> tuple[int, int]:
    """Convert a Noll index ``j`` to the pair ``(n, m)``."""
    if index < 1:
        raise ValueError(f"Noll indices start at one, got {index}")
    n = 0
    while index > (n + 1) * (n + 2) // 2:
        n += 1
    position = index - n * (n + 1) // 2  # 1-based position within the order
    # within an order the |m| values run 0,2,2,4,4,... for even n and
    # 1,1,3,3,... for odd n
    if n % 2 == 0:
        absolute = 2 * (position // 2)
    else:
        absolute = 2 * ((position - 1) // 2) + 1
    if absolute == 0:
        return n, 0
    # Noll puts the positive azimuthal order on even indices
    return n, absolute if index % 2 == 0 else -absolute


def zernike_modes(max_order: int) -> list[tuple[int, int]]:
    """Every ``(n, m)`` up to ``max_order``, in Noll order."""
    if max_order < 0:
        raise ValueError("the maximum order cannot be negative")
    total = (max_order + 1) * (max_order + 2) // 2
    return [noll_to_nm(index) for index in range(1, total + 1)]


def _coefficients(values: Sequence[float]) -> list[float]:
    return [float(value) for value in values]


def wavefront_rms(
    coefficients: Sequence[float],
    wavelength: Any = None,
    include_piston: bool = False,
) -> Quantity:
    """RMS wavefront error of a coefficient vector, in nanometres.

    Valid because the modes are orthonormal: the RMS is the quadrature sum of
    the coefficients.  Piston is excluded by default, since a uniform offset
    is not an aberration - which is also why the test suite confirms that the
    RMS of a bare ``rho^4`` term comes out as ``sqrt(1/5 - 1/9)``.
    """
    del wavelength
    values = _coefficients(coefficients)
    if not include_piston and values:
        values = values[1:]
    return Quantity(math.sqrt(sum(value * value for value in values)), _unit("nm"))


def wavefront_pv(
    coefficients: Sequence[float], wavelength: Any = None, samples: int = 400
) -> Quantity:
    """Peak-to-valley wavefront error, evaluated on a disc."""
    del wavelength
    axis = np.linspace(-1.0, 1.0, samples)
    x, y = np.meshgrid(axis, axis)
    rho = np.hypot(x, y)
    theta = np.arctan2(y, x)
    inside = rho <= 1.0
    total = np.zeros_like(rho)
    for index, value in enumerate(_coefficients(coefficients), start=1):
        if value == 0.0:
            continue
        n, m = noll_to_nm(index)
        total = total + value * zernike(n, m, rho, theta)
    inside_values = total[inside]
    return Quantity(float(inside_values.max() - inside_values.min()), _unit("nm"))


def strehl_from_zernikes(coefficients: Sequence[float], wavelength: Any) -> float:
    """Marechal estimate of the Strehl ratio from a coefficient vector."""
    rms = wavefront_rms(coefficients).value
    lam, _ = _length_of(wavelength, "strehl_from_zernikes")
    return float(math.exp(-((2.0 * math.pi * rms / lam) ** 2)))


def defocus_coefficient(peak_to_valley: float) -> float:
    """Zernike defocus coefficient for a given peak-to-valley wavefront.

    The defocus mode spans ``+/- sqrt(3)`` across the pupil, so a PV of ``W``
    corresponds to ``W / (2 sqrt 3)``.
    """
    return float(peak_to_valley) / (2.0 * math.sqrt(3.0))


def seidel_spherical_decomposition(
    amplitude: float, coefficients: int = 11
) -> list[float]:
    """Coefficients of ``amplitude * rho^4`` in Noll order.

    From ``rho^4 = Z11/(6 sqrt 5) + Z4/(2 sqrt 3) + Z1/3``: a pure Seidel
    spherical term is one part spherical aberration, one part defocus and one
    part piston once written in an orthonormal basis.
    """
    value = float(amplitude)
    result = [0.0] * coefficients
    result[0] = value / 3.0
    result[3] = value / (2.0 * math.sqrt(3.0))
    if coefficients >= 11:
        result[10] = value / (6.0 * math.sqrt(5.0))
    return result


__all__ = [
    "defocus_coefficient",
    "noll_to_nm",
    "radial_polynomial",
    "seidel_spherical_decomposition",
    "strehl_from_zernikes",
    "wavefront_pv",
    "wavefront_rms",
    "zernike",
    "zernike_modes",
]
