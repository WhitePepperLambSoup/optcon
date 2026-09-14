"""Detector noise: photon energy, shot noise, signal to noise, NEP.

Relations follow Saleh & Teich, *Fundamentals of Photonics*, ch. 17.  The
ideal-detector results are exact, so every function here is checked against
closed-form Poisson statistics rather than against another implementation.

Currents are in amperes, bandwidths in hertz, powers in watts and integration
times in seconds; wavelengths cross the boundary as lengths.
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
ELECTRON_CHARGE = 1.602176634e-19

_LENGTH = _unit("m").dimension


def _length_of(value: Any, context: str) -> tuple[float, Unit]:
    if isinstance(value, Quantity):
        if value.unit.dimension != _LENGTH:
            raise DimensionError(f"{context}: expected a length, got {value.unit}")
        return float(value.value), value.unit
    raise TypeError(
        f"{context}: expected a length with units, e.g. q(1550, 'nm'), "
        f"got a bare {type(value).__name__}"
    )


def _wavelength_in_metres(value: Any, context: str) -> float:
    magnitude, unit = _length_of(value, context)
    metres = unit.to_value(magnitude, _unit("m"))
    if metres <= 0.0:
        raise ValueError(f"{context}: the wavelength must be positive")
    return metres


def _efficiency(value: Any, context: str) -> float:
    if isinstance(value, Quantity):
        if not value.unit.is_dimensionless:
            raise DimensionError(
                f"{context}: a quantum efficiency must be dimensionless, got {value.unit}"
            )
        efficiency = float(value.value)
    else:
        efficiency = float(value)
    if not 0.0 <= efficiency <= 1.0:
        raise ValueError(
            f"{context}: a quantum efficiency must lie in [0, 1], got {efficiency}"
        )
    return efficiency


def photon_energy(wavelength: Any) -> Quantity:
    """``h c / lambda``."""
    lam = _wavelength_in_metres(wavelength, "photon_energy")
    return Quantity(PLANCK * C_LIGHT / lam, _unit("J"))


def photons_per_second(optical_power: Quantity, wavelength: Any) -> Quantity:
    """Photon arrival rate for a given optical power."""
    if not isinstance(optical_power, Quantity):
        raise TypeError("photons_per_second expects an optical power, e.g. q(1, 'uW')")
    power_watts = float(optical_power.to_value("W"))
    if power_watts < 0.0:
        raise ValueError("an optical power cannot be negative")
    lam = _wavelength_in_metres(wavelength, "photons_per_second")
    return Quantity(power_watts * lam / (PLANCK * C_LIGHT), _unit("1/s"))


def responsivity(quantum_efficiency: Any, wavelength: Any) -> float:
    """Amperes per watt, ``eta q lambda / (h c)``."""
    eta = _efficiency(quantum_efficiency, "responsivity")
    lam = _wavelength_in_metres(wavelength, "responsivity")
    return eta * ELECTRON_CHARGE * lam / (PLANCK * C_LIGHT)


def quantum_efficiency_from_responsivity(current_per_watt: float, wavelength: Any) -> float:
    """The inverse of :func:`responsivity`."""
    lam = _wavelength_in_metres(wavelength, "quantum_efficiency_from_responsivity")
    value = float(current_per_watt)
    if value < 0.0:
        raise ValueError("a responsivity cannot be negative")
    return value * PLANCK * C_LIGHT / (ELECTRON_CHARGE * lam)


def shot_noise_current(
    photocurrent_a: float, bandwidth_hz: float, dark_current_a: float = 0.0
) -> Quantity:
    """RMS shot noise ``sqrt(2 q (I + I_dark) B)``."""
    current = abs(float(photocurrent_a)) + abs(float(dark_current_a))
    bandwidth = float(bandwidth_hz)
    if bandwidth <= 0.0:
        raise ValueError("the measurement bandwidth must be positive")
    return Quantity(math.sqrt(2.0 * ELECTRON_CHARGE * current * bandwidth), _unit("A"))


def dark_current_noise(
    photocurrent_a: float, dark_current_a: float, bandwidth_hz: float
) -> Quantity:
    """Shot noise with the dark current included; the two add in quadrature."""
    return shot_noise_current(photocurrent_a, bandwidth_hz, dark_current_a)


def signal_to_noise(
    optical_power: Quantity, wavelength: Any, integration_time_s: float
) -> float:
    """Shot-noise-limited SNR of an ideal detector.

    For Poisson statistics the SNR is the square root of the collected photon
    count, which is why it grows only as ``sqrt(t)``.
    """
    duration = float(integration_time_s)
    if duration <= 0.0:
        raise ValueError("the integration time must be positive")
    rate = photons_per_second(optical_power, wavelength).to_value("1/s")
    if rate <= 0.0:
        return 0.0
    return math.sqrt(rate * duration)


def noise_equivalent_power(
    noise_current_per_root_hz: float, bandwidth_hz: float, current_per_watt: float = 1.0
) -> Quantity:
    """Optical power that produces a signal equal to the noise."""
    bandwidth = float(bandwidth_hz)
    if bandwidth <= 0.0:
        raise ValueError("the measurement bandwidth must be positive")
    if current_per_watt == 0.0:
        raise ValueError("a zero responsivity has no finite noise equivalent power")
    return Quantity(
        abs(float(noise_current_per_root_hz)) * math.sqrt(bandwidth) / abs(current_per_watt),
        _unit("W"),
    )


__all__ = [
    "C_LIGHT",
    "ELECTRON_CHARGE",
    "PLANCK",
    "dark_current_noise",
    "noise_equivalent_power",
    "photon_energy",
    "photons_per_second",
    "quantum_efficiency_from_responsivity",
    "responsivity",
    "shot_noise_current",
    "signal_to_noise",
]
