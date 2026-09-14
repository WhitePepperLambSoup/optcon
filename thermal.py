"""Thermal lensing and Gaussian aperture truncation.

The thermal lens of an end-pumped crystal follows from the radial temperature
gradient the pump creates.  In the usual thin-disk-free approximation the
focal length is

    f_th = 2 pi k w_p^2 / (P_abs * dn/dT),

with ``k`` the thermal conductivity, ``w_p`` the pump radius and ``P_abs`` the
absorbed pump power.  The sign convention is carried by ``dn/dT``: a positive
value gives a converging lens, and the caller must state the sign they mean
rather than having a zero slipped through.

The aperture results are exact for a Gaussian beam,

    transmission = 1 - exp(-2 a^2 / w^2),   clipping = exp(-2 a^2 / w^2),

which is the form used to budget mirror apertures and fibre mode matching.
"""

from __future__ import annotations

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
        f"{context}: expected a length with units, e.g. q(300, 'um'), "
        f"got a bare {type(value).__name__}"
    )


def _watts_of(value: Any, context: str) -> float:
    if isinstance(value, Quantity):
        return float(value.to_value("W"))
    raise TypeError(f"{context}: expected an optical power, e.g. q(10, 'W')")


def thermal_lens_focal_length(
    pump_power: Any,
    pump_radius: Any,
    thermal_conductivity: float,
    dn_dt: float,
    absorbed_fraction: float = 1.0,
) -> Quantity:
    """Focal length of the lens an absorbed pump creates.

    ``thermal_conductivity`` is in W/(m K) and ``dn_dt`` in 1/K.  A negative
    ``dn_dt`` is physical (it defocuses) but is rejected here, because a
    thermal lens whose sign is unstated is a lens whose sign will be wrong
    somewhere downstream: state the magnitude and flip the sign at the call
    site that means it.
    """
    power = _watts_of(pump_power, "thermal_lens_focal_length")
    radius, unit = _length_of(pump_radius, "thermal_lens_focal_length")
    conductivity = float(thermal_conductivity)
    coefficient = float(dn_dt)
    absorbed = float(absorbed_fraction)
    if not 0.0 < absorbed <= 1.0:
        raise ValueError(
            f"absorbed_fraction must lie in (0, 1], got {absorbed}"
        )
    if conductivity <= 0.0:
        raise ValueError(f"the thermal conductivity must be positive, got {conductivity}")
    if coefficient <= 0.0:
        raise ValueError(
            "dn_dt must be given as a positive magnitude; a defocusing medium "
            f"has a negative one and that sign belongs at the call site, got {coefficient}"
        )
    if power <= 0.0:
        return Quantity(math.inf, unit)
    focal_metres = (
        2.0
        * math.pi
        * conductivity
        * (radius * unit.factor) ** 2
        / (power * absorbed * coefficient)
    )
    return Quantity(focal_metres / unit.factor, unit)


def gaussian_aperture_transmission(
    aperture_radius: Any, waist: Any, wavelength: Any = None
) -> Quantity:
    """Fraction of a Gaussian beam's power that clears a circular aperture.

    ``1 - exp(-2 a^2 / w^2)``.  The wavelength is accepted for symmetry with
    the rest of the library and is not needed, since the result depends only
    on the ratio of aperture to beam radius.
    """
    del wavelength
    aperture, unit = _length_of(aperture_radius, "gaussian_aperture_transmission")
    beam, beam_unit = _length_of(waist, "gaussian_aperture_transmission")
    beam = beam_unit.to_value(beam, unit)
    if aperture <= 0.0 or beam <= 0.0:
        raise ValueError("the aperture radius and beam radius must be positive")
    clipped = math.exp(-2.0 * aperture**2 / beam**2)
    return Quantity(1.0 - clipped, _unit("1"), POWER)


def gaussian_clipping_loss(
    aperture_radius: Any, waist: Any, wavelength: Any = None
) -> Quantity:
    """Fraction of a Gaussian beam's power lost at a circular aperture."""
    transmitted = gaussian_aperture_transmission(
        aperture_radius, waist, wavelength
    ).to_value("1")
    return Quantity(1.0 - transmitted, _unit("1"), POWER)


__all__ = [
    "gaussian_aperture_transmission",
    "gaussian_clipping_loss",
    "thermal_lens_focal_length",
]
