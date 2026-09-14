"""Fresnel reflection and transmission at a plane dielectric interface.

Amplitude coefficients follow Born & Wolf, *Principles of Optics*, sec. 1.5.2,
with the convention that ``r`` and ``t`` are the ratios of the tangential
electric field amplitudes.  With that convention ``1 + r_s = t_s`` holds at
every angle, while for p-polarisation the analogous statement carries the
index ratio, ``t_p = (n1/n2)(1 + r_p)``.

Reflectance and transmittance are **power** ratios, so they are returned with
amplitude order 2 and are guaranteed to sum to one for lossless dielectrics -
a guarantee the test suite checks at forty angles.
"""

from __future__ import annotations

import cmath
import math
from typing import Any

from .errors import DimensionError
from .quantity import AMPLITUDE, POWER, Quantity
from .units import unit as _unit

_RADIAN = _unit("rad")
_ANGLE = _RADIAN.dimension


def _angle_in_radians(angle: Any, context: str) -> float:
    if isinstance(angle, Quantity):
        if angle.unit.dimension != _ANGLE:
            raise DimensionError(f"{context}: expected an angle, got {angle.unit}")
        return float(angle.to_value("rad"))
    raise TypeError(
        f"{context}: expected an angle with units, e.g. q(30, 'deg'), "
        f"got a bare {type(angle).__name__}"
    )


def _index(value: Any, context: str) -> float:
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


def _refracted_sine(n_from: float, n_to: float, theta_i: float) -> float:
    return n_from * math.sin(theta_i) / n_to


def snell(n_from: Any, n_to: Any, angle: Any) -> Quantity:
    """Refraction angle at a plane interface, in radians."""
    first = _index(n_from, "snell")
    second = _index(n_to, "snell")
    theta_i = _angle_in_radians(angle, "snell")
    sine = _refracted_sine(first, second, theta_i)
    if abs(sine) > 1.0:
        raise ValueError(
            "total internal reflection: there is no transmitted angle for this ray"
        )
    return Quantity(math.asin(sine), _RADIAN)


def critical_angle(n_from: Any, n_to: Any) -> Quantity:
    """The angle beyond which total internal reflection occurs."""
    first = _index(n_from, "critical_angle")
    second = _index(n_to, "critical_angle")
    if first <= second:
        raise ValueError(
            "total internal reflection requires leaving an optically denser "
            f"medium, got n_from={first} and n_to={second}"
        )
    return Quantity(math.asin(second / first), _RADIAN)


def brewster_angle(n_from: Any, n_to: Any) -> Quantity:
    """The angle at which p-polarised light is not reflected."""
    first = _index(n_from, "brewster_angle")
    second = _index(n_to, "brewster_angle")
    return Quantity(math.atan2(second, first), _RADIAN)


def fresnel_coefficients(n_from: Any, n_to: Any, angle: Any) -> dict[str, complex]:
    """The four amplitude coefficients ``r_s``, ``r_p``, ``t_s``, ``t_p``.

    Complex in general: beyond the critical angle the transmitted wave becomes
    evanescent and the coefficients pick up a phase.
    """
    first = _index(n_from, "fresnel_coefficients")
    second = _index(n_to, "fresnel_coefficients")
    theta_i = _angle_in_radians(angle, "fresnel_coefficients")
    cos_i = math.cos(theta_i)
    sine_t = _refracted_sine(first, second, theta_i)
    cos_t = cmath.sqrt(1.0 - sine_t**2) if abs(sine_t) > 1.0 else math.sqrt(1.0 - sine_t**2)

    r_s = (first * cos_i - second * cos_t) / (first * cos_i + second * cos_t)
    r_p = (second * cos_i - first * cos_t) / (second * cos_i + first * cos_t)
    t_s = 2.0 * first * cos_i / (first * cos_i + second * cos_t)
    t_p = 2.0 * first * cos_i / (second * cos_i + first * cos_t)
    return {"r_s": complex(r_s), "r_p": complex(r_p), "t_s": complex(t_s), "t_p": complex(t_p)}


def _checked_polarization(polarization: str, context: str) -> str:
    if polarization not in {"s", "p", "unpolarized"}:
        raise ValueError(f"{context}: polarization must be 's', 'p' or 'unpolarized'")
    return polarization


def reflectance(n_from: Any, n_to: Any, angle: Any, polarization: str = "s") -> Quantity:
    """Fraction of incident power reflected, as a power ratio."""
    _checked_polarization(polarization, "reflectance")
    if polarization == "unpolarized":
        return Quantity(
            0.5
            * (
                reflectance(n_from, n_to, angle, "s").value
                + reflectance(n_from, n_to, angle, "p").value
            ),
            _unit("1"),
            POWER,
        )
    coefficients = fresnel_coefficients(n_from, n_to, angle)
    amplitude = coefficients["r_s"] if polarization == "s" else coefficients["r_p"]
    return Quantity(abs(amplitude) ** 2, _unit("1"), POWER)


def transmittance(n_from: Any, n_to: Any, angle: Any, polarization: str = "s") -> Quantity:
    """Fraction of incident power transmitted, as a power ratio.

    Includes the ``n2 cos(theta_t) / (n1 cos(theta_i))`` geometrical factor
    that makes ``R + T = 1`` exact for a lossless interface.
    """
    _checked_polarization(polarization, "transmittance")
    if polarization == "unpolarized":
        return Quantity(
            0.5
            * (
                transmittance(n_from, n_to, angle, "s").value
                + transmittance(n_from, n_to, angle, "p").value
            ),
            _unit("1"),
            POWER,
        )
    first = _index(n_from, "transmittance")
    second = _index(n_to, "transmittance")
    theta_i = _angle_in_radians(angle, "transmittance")
    cos_i = math.cos(theta_i)
    sine_t = _refracted_sine(first, second, theta_i)
    if abs(sine_t) > 1.0:
        return Quantity(0.0, _unit("1"), POWER)
    cos_t = math.sqrt(1.0 - sine_t**2)
    coefficients = fresnel_coefficients(first, second, angle)
    amplitude = coefficients["t_s"] if polarization == "s" else coefficients["t_p"]
    factor = (second * cos_t) / (first * cos_i)
    return Quantity(factor * abs(amplitude) ** 2, _unit("1"), POWER)


def unpolarized_reflectance(n_from: Any, n_to: Any, angle: Any) -> Quantity:
    """Mean of the s and p reflectances, for unpolarized light."""
    return reflectance(n_from, n_to, angle, polarization="unpolarized")


def amplitude_coefficient(
    n_from: Any, n_to: Any, angle: Any, polarization: str = "s"
) -> Quantity:
    """The field amplitude coefficient ``r`` or ``t``, as an amplitude ratio."""
    _checked_polarization(polarization, "amplitude_coefficient")
    if polarization == "unpolarized":
        raise ValueError(
            "unpolarized light has no single amplitude coefficient; use reflectance()"
        )
    coefficients = fresnel_coefficients(n_from, n_to, angle)
    key = "r_s" if polarization == "s" else "r_p"
    return Quantity(abs(coefficients[key]), _unit("1"), AMPLITUDE)


__all__ = [
    "amplitude_coefficient",
    "brewster_angle",
    "critical_angle",
    "fresnel_coefficients",
    "reflectance",
    "snell",
    "transmittance",
    "unpolarized_reflectance",
]
