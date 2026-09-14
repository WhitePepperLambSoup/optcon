"""Laser physics for a four-level gain medium.

Relations follow Svelto, *Principles of Lasers*, ch. 7.  The threshold pump
power is derived rather than quoted: with round-trip loss ``delta``, the
threshold inversion is ``N_th = delta / (2 sigma L)``, and the absorbed pump
power needed to sustain it in a mode volume ``V = A L`` is

    P_th = h nu_p V N_th / tau = I_sat A delta / 2,

which is exactly what this module returns.
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

_LENGTH = _unit("m").dimension
_TIME = _unit("s").dimension


def _length_of(value: Any, context: str) -> tuple[float, Unit]:
    if isinstance(value, Quantity):
        if value.unit.dimension != _LENGTH:
            raise DimensionError(f"{context}: expected a length, got {value.unit}")
        return float(value.value), value.unit
    raise TypeError(
        f"{context}: expected a length with units, e.g. q(0.3, 'm'), "
        f"got a bare {type(value).__name__}"
    )


def _duration_of(value: Any, context: str) -> float:
    if isinstance(value, Quantity):
        if value.unit.dimension != _TIME:
            raise DimensionError(f"{context}: expected a duration, got {value.unit}")
        seconds = float(value.to_value("s"))
        if seconds <= 0.0:
            raise ValueError(f"{context}: a lifetime must be positive")
        return seconds
    raise TypeError(f"{context}: expected a duration with units, e.g. q(230, 'us')")


def _round_trip_loss(
    length_metres: float,
    output_coupler_reflectance: float,
    internal_loss_per_m: float,
    other_losses: float,
) -> float:
    reflectance = float(output_coupler_reflectance)
    if not 0.0 < reflectance <= 1.0:
        raise ValueError(
            f"the output coupler reflectance must lie in (0, 1], got {reflectance}"
        )
    internal = float(internal_loss_per_m)
    if internal < 0.0:
        raise ValueError("an internal loss per metre cannot be negative")
    extra = float(other_losses)
    if extra < 0.0:
        raise ValueError("other losses cannot be negative")
    return 2.0 * internal * length_metres - math.log(reflectance) + extra


def saturation_intensity(wavelength: Any, cross_section: float, lifetime: float) -> Quantity:
    """``I_sat = h nu / (sigma tau)``."""
    lam_magnitude, unit = _length_of(wavelength, "saturation_intensity")
    lam_metres = unit.to_value(lam_magnitude, _unit("m"))
    sigma = float(cross_section)
    tau = float(lifetime)
    if sigma <= 0.0 or tau <= 0.0:
        raise ValueError("the cross section and lifetime must both be positive")
    return Quantity(PLANCK * C_LIGHT / lam_metres / (sigma * tau), _unit("W/m^2"))


def threshold_gain(
    length: Any,
    output_coupler_reflectance: float,
    internal_loss_per_m: float = 0.0,
    other_losses: float = 0.0,
) -> Quantity:
    """Small-signal gain per metre needed to reach threshold."""
    magnitude, unit = _length_of(length, "threshold_gain")
    metres = unit.to_value(magnitude, _unit("m"))
    if metres <= 0.0:
        raise ValueError("the gain medium length must be positive")
    loss = _round_trip_loss(metres, output_coupler_reflectance, internal_loss_per_m, other_losses)
    return Quantity(loss / (2.0 * metres), _unit("1/m"))


def photon_lifetime_cavity(
    length: Any,
    output_coupler_reflectance: float,
    internal_loss_per_m: float = 0.0,
    other_losses: float = 0.0,
) -> Quantity:
    """Cavity photon lifetime ``2L / (c delta)``."""
    magnitude, unit = _length_of(length, "photon_lifetime_cavity")
    metres = unit.to_value(magnitude, _unit("m"))
    if metres <= 0.0:
        raise ValueError("the cavity length must be positive")
    loss = _round_trip_loss(metres, output_coupler_reflectance, internal_loss_per_m, other_losses)
    if loss == 0.0:
        return Quantity(math.inf, _unit("s"))
    return Quantity(2.0 * metres / (C_LIGHT * loss), _unit("s"))


def slope_efficiency(
    pump_wavelength: Any,
    laser_wavelength: Any,
    mode_overlap: float = 1.0,
    quantum_efficiency: float = 1.0,
) -> float:
    """Fraction of the pump power above threshold that becomes output.

    Bounded by the quantum defect ``lambda_p / lambda_L``; mode overlap and
    quantum efficiency can only reduce it.
    """
    pump_magnitude, pump_unit = _length_of(pump_wavelength, "slope_efficiency")
    laser_magnitude, laser_unit = _length_of(laser_wavelength, "slope_efficiency")
    pump = pump_unit.to_value(pump_magnitude, _unit("m"))
    laser = laser_unit.to_value(laser_magnitude, _unit("m"))
    overlap = float(mode_overlap)
    eta = float(quantum_efficiency)
    for label, value in (("mode_overlap", overlap), ("quantum_efficiency", eta)):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{label} must lie in [0, 1], got {value}")
    ratio = pump / laser
    if ratio > 1.0:
        raise ValueError(
            "the slope efficiency cannot exceed one: the pump wavelength "
            f"({pump * 1e9:.1f} nm) is longer than the laser wavelength "
            f"({laser * 1e9:.1f} nm), which describes absorption rather than gain"
        )
    return ratio * overlap * eta


def threshold_pump_power(
    saturation_intensity_value: Quantity, mode_area: Quantity, round_trip_loss: float
) -> Quantity:
    """``P_th = I_sat A delta / 2`` for a four-level medium."""
    if not isinstance(saturation_intensity_value, Quantity):
        raise TypeError("threshold_pump_power expects a saturation intensity")
    if not isinstance(mode_area, Quantity):
        raise TypeError("threshold_pump_power expects a mode area")
    intensity = saturation_intensity_value.to_value("W/m^2")
    area = mode_area.to_value("m^2")
    loss = float(round_trip_loss)
    if loss < 0.0:
        raise ValueError("the round trip loss cannot be negative")
    return Quantity(intensity * area * loss / 2.0, _unit("W"))


def output_power(slope_efficiency_value: float, pump_power: float, threshold_power: float) -> float:
    """Output power above threshold; zero below it."""
    eta = float(slope_efficiency_value)
    pump = float(pump_power)
    threshold = float(threshold_power)
    if eta < 0.0:
        raise ValueError("the slope efficiency cannot be negative")
    if threshold < 0.0:
        raise ValueError("the threshold power cannot be negative")
    return eta * max(0.0, pump - threshold)


def relaxation_oscillation_frequency(
    photon_lifetime: Any, upper_state_lifetime: Any, pump_ratio: float
) -> Quantity:
    """Damping frequency of the spiking oscillations, ``(1/2pi) sqrt((r-1)/(tau_p tau_c))``."""
    tau_p = _duration_of(photon_lifetime, "relaxation_oscillation_frequency")
    tau_f = _duration_of(upper_state_lifetime, "relaxation_oscillation_frequency")
    ratio = float(pump_ratio)
    if ratio <= 1.0:
        raise ValueError(
            "the excitation ratio must exceed one for relaxation oscillations "
            f"to exist, got {ratio}"
        )
    return Quantity(
        math.sqrt((ratio - 1.0) / (tau_p * tau_f)) / (2.0 * math.pi), _unit("Hz")
    )


__all__ = [
    "output_power",
    "photon_lifetime_cavity",
    "relaxation_oscillation_frequency",
    "saturation_intensity",
    "slope_efficiency",
    "threshold_gain",
    "threshold_pump_power",
]
