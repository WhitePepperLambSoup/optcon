"""Radiometry and blackbody radiation.

Planck's law is given in its wavelength form,

    B_lambda(T) = 2 h c^2 / lambda^5 / (exp(h c / (lambda k T)) - 1),

in W m^-2 sr^-1 m^-1.  A Lambertian emitter has ``M = pi B``, and integrating
``M_lambda`` over all wavelengths returns ``sigma T^4``; the test suite checks
that integral numerically, so the Planck and Stefan-Boltzmann constants are
not independent claims.
"""

from __future__ import annotations

import math
from typing import Any

from .errors import DimensionError
from .quantity import Quantity
from .units import Unit
from .units import unit as _unit

PLANCK = 6.62607015e-34
C_LIGHT = 299792458.0
BOLTZMANN = 1.380649e-23
STEFAN_BOLTZMANN = 5.670374419e-8
WIEN_DISPLACEMENT = 2.897771955e-3

_LENGTH = _unit("m").dimension
_ANGLE = _unit("rad").dimension


def _length_of(value: Any, context: str) -> tuple[float, Unit]:
    if isinstance(value, Quantity):
        if value.unit.dimension != _LENGTH:
            raise DimensionError(f"{context}: expected a length, got {value.unit}")
        return float(value.value), value.unit
    raise TypeError(
        f"{context}: expected a length with units, e.g. q(10, 'um'), "
        f"got a bare {type(value).__name__}"
    )


def _wavelength_in_metres(value: Any, context: str) -> float:
    magnitude, unit = _length_of(value, context)
    metres = unit.to_value(magnitude, _unit("m"))
    if metres <= 0.0:
        raise ValueError(f"{context}: the wavelength must be positive")
    return metres


def _checked_temperature(temperature: float) -> float:
    value = float(temperature)
    if value <= 0.0:
        raise ValueError(f"the temperature must be positive, got {value}")
    return value


def planck_radiance(wavelength: Any, temperature: float) -> Quantity:
    """Spectral radiance of a blackbody, ``W m^-2 sr^-1 m^-1``."""
    lam = _wavelength_in_metres(wavelength, "planck_radiance")
    temp = _checked_temperature(temperature)
    exponent = PLANCK * C_LIGHT / (lam * BOLTZMANN * temp)
    if exponent > 700.0:
        return Quantity(0.0, _unit("W/(m^2*sr*m)"))
    spectral = 2.0 * PLANCK * C_LIGHT**2 / lam**5 / math.expm1(exponent)
    return Quantity(spectral, _unit("W/(m^2*sr*m)"))


def planck_exitance(wavelength: Any, temperature: float) -> Quantity:
    """Spectral exitance of a Lambertian blackbody, ``pi B_lambda``."""
    spectral = planck_radiance(wavelength, temperature).to_value("W/(m^2*sr*m)")
    return Quantity(math.pi * spectral, _unit("W/m^3"))


def stefan_boltzmann(temperature: float) -> Quantity:
    """Total exitance ``sigma T^4``."""
    temp = _checked_temperature(temperature)
    return Quantity(STEFAN_BOLTZMANN * temp**4, _unit("W/m^2"))


def wien_displacement(temperature: float) -> Quantity:
    """Wavelength of peak spectral radiance, ``b / T``."""
    temp = _checked_temperature(temperature)
    return Quantity(WIEN_DISPLACEMENT / temp, _unit("m"))


def solid_angle_cone(half_angle: Any) -> Quantity:
    """Solid angle of a cone, ``2 pi (1 - cos(theta))``."""
    if isinstance(half_angle, Quantity):
        if half_angle.unit.dimension != _ANGLE:
            raise DimensionError(
                f"solid_angle_cone: expected an angle, got {half_angle.unit}"
            )
        theta = float(half_angle.to_value("rad"))
    else:
        raise TypeError("solid_angle_cone: expected an angle, e.g. q(30, 'deg')")
    if not 0.0 <= theta <= math.pi:
        raise ValueError(f"a cone half angle must lie in [0, pi], got {theta}")
    return Quantity(2.0 * math.pi * (1.0 - math.cos(theta)), _unit("sr"))


def etendue(area: Any, solid_angle: Any) -> Quantity:
    """Throughput ``A Omega``; conserved by lossless optics."""
    if not isinstance(area, Quantity) or not isinstance(solid_angle, Quantity):
        raise TypeError("etendue expects two quantities, e.g. q(1, 'mm^2') and q(0.01, 'sr')")
    return Quantity(
        area.to_value("m^2") * solid_angle.to_value("sr"), _unit("m^2*sr")
    )


def irradiance(power: Quantity, area: Quantity) -> Quantity:
    """Power per unit area on a surface, ``W/m^2``."""
    if not isinstance(power, Quantity) or not isinstance(area, Quantity):
        raise TypeError("irradiance expects a power and an area as quantities")
    watts = power.to_value("W")
    square_metres = area.to_value("m^2")
    if square_metres <= 0.0:
        raise ValueError("the illuminated area must be positive")
    return Quantity(watts / square_metres, _unit("W/m^2"))


def radiance(power: Quantity, area: Quantity, solid_angle: Quantity) -> Quantity:
    """Power per unit area per unit solid angle, ``W m^-2 sr^-1``."""
    if not isinstance(solid_angle, Quantity):
        raise TypeError("radiance expects a solid angle as a quantity")
    steradians = solid_angle.to_value("sr")
    if steradians <= 0.0:
        raise ValueError("the solid angle must be positive")
    base = irradiance(power, area).to_value("W/m^2")
    return Quantity(base / steradians, _unit("W/(m^2*sr)"))


def luminous_flux(radiant_power: Quantity, luminous_efficacy: float) -> Quantity:
    """Photometric flux ``Phi = K P``, in lumens."""
    if not isinstance(radiant_power, Quantity):
        raise TypeError("luminous_flux expects a radiant power as a quantity")
    efficacy = float(luminous_efficacy)
    if efficacy < 0.0:
        raise ValueError("a luminous efficacy cannot be negative")
    return Quantity(radiant_power.to_value("W") * efficacy, _unit("lm"))


def radiance_invariance(first_etendue: Quantity, second_etendue: Quantity) -> float:
    """Ratio ``A2 Omega2 / (A1 Omega1)``; one for a lossless system.

    This is the brightness theorem: no passive optical system can increase
    radiance, so this ratio cannot exceed one.
    """
    if not isinstance(first_etendue, Quantity) or not isinstance(second_etendue, Quantity):
        raise TypeError("radiance_invariance expects two etendues")
    first = first_etendue.to_value("m^2*sr")
    second = second_etendue.to_value("m^2*sr")
    if first <= 0.0:
        raise ValueError("the first etendue must be positive")
    return float(second / first)


__all__ = [
    "BOLTZMANN",
    "C_LIGHT",
    "PLANCK",
    "STEFAN_BOLTZMANN",
    "WIEN_DISPLACEMENT",
    "etendue",
    "irradiance",
    "luminous_flux",
    "planck_exitance",
    "planck_radiance",
    "radiance",
    "radiance_invariance",
    "solid_angle_cone",
    "stefan_boltzmann",
    "wien_displacement",
]
