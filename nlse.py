"""Generalized Nonlinear Schrodinger Equation (G-NLSE) Split-Step Fourier Solver.

Solves pulse propagation in nonlinear dispersive optical waveguides and fibers:

    \\frac{\\partial A}{\\partial z} = \\hat{D} A + \\hat{N}(A) A

following Agrawal, *Nonlinear Fiber Optics* (5th ed., 2013), ch. 2-3.

The linear dispersion operator \\hat{D} accounts for attenuation and arbitrary
orders of dispersion:

    \\tilde{D}(\\Omega) = -\\frac{\\alpha}{2} + i \\left( \\frac{\\beta_2}{2} \\Omega^2
        + \\frac{\\beta_3}{6} \\Omega^3 + \\frac{\\beta_4}{24} \\Omega^4 \\right)

The nonlinear operator \\hat{N}(A) includes Kerr self-phase modulation (SPM),
self-steepening (optical shock), and the Raman delayed response:

    \\hat{N}(A) = i \\gamma \\left( 1 + \\frac{i}{\\omega_0} \\frac{\\partial}{\\partial T} \\right)
                  \\left[ (1 - f_R) |A|^2 + f_R \\int_0^\\infty h_R(s) |A(T - s)|^2 ds \\right]

Numerical Scheme:
    Second-order symmetric Split-Step Fourier Method (SSFM), with global
    accuracy O(dz^2) and exact energy conservation in the lossless regime (alpha = 0).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.fft as _fft

from .errors import ContractViolation, DimensionError
from .quantity import Quantity
from .units import unit as _unit

C_LIGHT = 299792458.0

_LENGTH = _unit("m").dimension
_TIME = _unit("s").dimension
_POWER = _unit("W").dimension
_ENERGY = _unit("J").dimension
_DISPERSION_2 = _unit("s^2/m").dimension
_DISPERSION_3 = _unit("s^3/m").dimension
_DISPERSION_4 = _unit("s^4/m").dimension
_GAMMA_DIM = _unit("1/(W*m)").dimension
_ATTENUATION = _unit("1/m").dimension


def _length_of(value: Any, context: str) -> float:
    if isinstance(value, Quantity):
        if value.unit.dimension != _LENGTH:
            raise DimensionError(f"{context}: expected a length, got {value.unit}")
        return float(value.to_value("m"))
    raise TypeError(f"{context}: expected a Quantity with length units, got {type(value).__name__}")


def _time_of(value: Any, context: str) -> float:
    if isinstance(value, Quantity):
        if value.unit.dimension != _TIME:
            raise DimensionError(f"{context}: expected a time duration, got {value.unit}")
        return float(value.to_value("s"))
    raise TypeError(f"{context}: expected a Quantity with time units, got {type(value).__name__}")


def _power_of(value: Any, context: str) -> float:
    if isinstance(value, Quantity):
        if value.unit.dimension != _POWER:
            raise DimensionError(f"{context}: expected power, got {value.unit}")
        return float(value.to_value("W"))
    raise TypeError(f"{context}: expected a Quantity with power units, got {type(value).__name__}")


@dataclass(frozen=True)
class Pulse:
    """An optical pulse envelope sampled on a uniform temporal grid.

    The envelope amplitude is scaled such that |amplitude|^2 has physical units
    of instantaneous optical power (Watts). Total pulse energy is integral(|A|^2 dt).
    """

    amplitude: np.ndarray
    time_step: Quantity
    wavelength: Quantity

    def __post_init__(self) -> None:
        arr = np.asarray(self.amplitude, dtype=complex)
        if arr.ndim != 1:
            raise ValueError(f"Pulse amplitude must be a 1D array, got shape {arr.shape}")
        if len(arr) < 16:
            raise ValueError(f"Pulse grid must have at least 16 samples, got {len(arr)}")
        object.__setattr__(self, "amplitude", arr)
        _time_of(self.time_step, "Pulse.time_step")
        _length_of(self.wavelength, "Pulse.wavelength")

    @property
    def samples(self) -> int:
        return int(self.amplitude.shape[0])

    @property
    def time_step_s(self) -> float:
        return _time_of(self.time_step, "time_step")

    @property
    def wavelength_m(self) -> float:
        return _length_of(self.wavelength, "wavelength")

    @property
    def carrier_angular_frequency(self) -> float:
        """omega0 = 2 * pi * c / lambda."""
        return 2.0 * math.pi * C_LIGHT / self.wavelength_m

    @property
    def time_window(self) -> Quantity:
        """Physical time duration of the entire computational window."""
        return Quantity(self.samples * self.time_step.value, self.time_step.unit)

    @property
    def time_axis(self) -> np.ndarray:
        """Centred temporal coordinate axis in seconds."""
        dt = self.time_step_s
        return (np.arange(self.samples) - self.samples // 2) * dt

    @property
    def angular_frequencies(self) -> np.ndarray:
        """Angular frequency grid Omega = omega - omega0 in rad/s."""
        freqs = _fft.fftfreq(self.samples, d=self.time_step_s)
        return 2.0 * math.pi * freqs

    @property
    def peak_power(self) -> Quantity:
        """Peak instantaneous optical power in Watts."""
        p_max = float(np.max(np.abs(self.amplitude) ** 2))
        return Quantity(p_max, _unit("W"))

    @property
    def energy(self) -> Quantity:
        """Total pulse energy integral(|A|^2 dt) in Joules."""
        e_joules = float(np.sum(np.abs(self.amplitude) ** 2)) * self.time_step_s
        return Quantity(e_joules, _unit("J"))

    @property
    def fwhm_duration(self) -> Quantity:
        """Full-width at half-maximum (FWHM) of pulse intensity."""
        p = np.abs(self.amplitude) ** 2
        p_max = float(np.max(p))
        if p_max <= 0.0:
            return Quantity(0.0, _unit("s"))
        p_half = 0.5 * p_max
        above = np.where(p >= p_half)[0]
        if len(above) < 2:
            return Quantity(0.0, _unit("s"))

        # Sub-grid linear interpolation on leading and trailing edges
        i1 = int(above[0])
        if i1 > 0 and p[i1] != p[i1 - 1]:
            t1 = (i1 - 1) + (p_half - p[i1 - 1]) / (p[i1] - p[i1 - 1])
        else:
            t1 = float(i1)

        i2 = int(above[-1])
        if i2 < len(p) - 1 and p[i2 + 1] != p[i2]:
            t2 = i2 + (p_half - p[i2]) / (p[i2 + 1] - p[i2])
        else:
            t2 = float(i2)

        duration_s = float(t2 - t1) * self.time_step_s
        return Quantity(duration_s, _unit("s"))


@dataclass(frozen=True)
class FiberParameters:
    """Waveguide or optical fiber propagation parameters.

    Parameters
    ----------
    beta2 : Quantity
        Group velocity dispersion (GVD) parameter (e.g. q(-20, "ps^2/km")).
    gamma : Quantity
        Nonlinear Kerr parameter (e.g. q(1.5, "1/(W*km)")).
    beta3 : Quantity | None, default None
        Third-order dispersion (TOD) parameter (e.g. q(0.1, "ps^3/km")).
    beta4 : Quantity | None, default None
        Fourth-order dispersion parameter.
    loss : Quantity | None, default None
        Linear power attenuation coefficient alpha (e.g. in 1/m).
    raman_fraction : float, default 0.0
        Fractional contribution of the delayed Raman response f_R (0.18 for silica).
    self_steepening : bool, default False
        Whether to include the self-steepening shock term (1 / omega0).
    """

    beta2: Quantity
    gamma: Quantity
    beta3: Quantity | None = None
    beta4: Quantity | None = None
    loss: Quantity | None = None
    raman_fraction: float = 0.0
    self_steepening: bool = False

    def __post_init__(self) -> None:
        if self.beta2.unit.dimension != _DISPERSION_2:
            raise DimensionError(f"beta2 must have dimension time^2/length, got {self.beta2.unit}")
        if self.gamma.unit.dimension != _GAMMA_DIM:
            raise DimensionError(
                f"gamma must have dimension 1/(power*length), got {self.gamma.unit}"
            )
        if self.beta3 is not None and self.beta3.unit.dimension != _DISPERSION_3:
            raise DimensionError(f"beta3 must have dimension time^3/length, got {self.beta3.unit}")
        if self.beta4 is not None and self.beta4.unit.dimension != _DISPERSION_4:
            raise DimensionError(f"beta4 must have dimension time^4/length, got {self.beta4.unit}")
        if self.loss is not None and self.loss.unit.dimension != _ATTENUATION:
            raise DimensionError(f"loss must have dimension 1/length, got {self.loss.unit}")
        if not 0.0 <= self.raman_fraction <= 1.0:
            raise ValueError(f"raman_fraction must be in [0, 1], got {self.raman_fraction}")


def soliton_parameters(
    wavelength: Any,
    pulse_duration: Any,
    beta2: Any,
    gamma: Any,
) -> dict[str, Quantity]:
    """Compute fundamental soliton scales: L_D, L_NL, z_0, and peak power P_0.

    Parameters
    ----------
    wavelength : Quantity
        Carrier wavelength lambda.
    pulse_duration : Quantity
        Fundamental soliton parameter T_0 (where FWHM = 1.763 * T_0).
    beta2 : Quantity
        Anomalous GVD parameter (must be negative for bright solitons).
    gamma : Quantity
        Nonlinear Kerr parameter.

    Returns
    -------
    dict[str, Quantity]
        Calculated dispersion_length, nonlinear_length, soliton_period, and peak_power.
    """
    _length_of(wavelength, "soliton_parameters.wavelength")
    t0_s = _time_of(pulse_duration, "soliton_parameters.pulse_duration")
    b2_si = float(beta2.to_value("s^2/m"))
    gam_si = float(gamma.to_value("1/(W*m)"))

    if b2_si >= 0.0:
        raise ValueError(
            f"Bright solitons require anomalous dispersion (beta2 < 0), got {beta2}"
        )
    if gam_si <= 0.0:
        raise ValueError(f"gamma must be positive, got {gamma}")

    abs_b2 = abs(b2_si)
    l_d = t0_s**2 / abs_b2
    p0 = abs_b2 / (gam_si * t0_s**2)
    l_nl = 1.0 / (gam_si * p0)
    z0 = 0.5 * math.pi * l_d

    return {
        "dispersion_length": Quantity(l_d, _unit("m")),
        "nonlinear_length": Quantity(l_nl, _unit("m")),
        "soliton_period": Quantity(z0, _unit("m")),
        "peak_power": Quantity(p0, _unit("W")),
    }


def soliton_pulse(
    peak_power: Any,
    duration: Any,
    wavelength: Any,
    samples: int = 512,
    time_window: Any = None,
) -> Pulse:
    """Create a fundamental hyperbolic secant soliton pulse A(T) = sqrt(P0) * sech(T / T0)."""
    p0 = _power_of(peak_power, "soliton_pulse.peak_power")
    t0 = _time_of(duration, "soliton_pulse.duration")
    _length_of(wavelength, "soliton_pulse.wavelength")

    if time_window is None:
        t_win = 20.0 * t0
    else:
        t_win = _time_of(time_window, "soliton_pulse.time_window")

    dt = t_win / samples
    t_axis = (np.arange(samples) - samples // 2) * dt
    envelope = np.sqrt(p0) / np.cosh(t_axis / t0)

    return Pulse(
        amplitude=envelope.astype(complex),
        time_step=Quantity(dt, _unit("s")),
        wavelength=(
            wavelength
            if isinstance(wavelength, Quantity)
            else Quantity(wavelength, _unit("m"))
        ),
    )


def gaussian_pulse(
    peak_power: Any,
    fwhm_duration: Any,
    wavelength: Any,
    samples: int = 512,
    time_window: Any = None,
) -> Pulse:
    """Create a transform-limited Gaussian pulse A(T) = sqrt(P0) * exp(-T^2 / (2 T0^2))."""
    p0 = _power_of(peak_power, "gaussian_pulse.peak_power")
    fwhm = _time_of(fwhm_duration, "gaussian_pulse.fwhm_duration")
    _length_of(wavelength, "gaussian_pulse.wavelength")

    t0 = fwhm / (2.0 * math.sqrt(math.log(2.0)))
    if time_window is None:
        t_win = 10.0 * fwhm
    else:
        t_win = _time_of(time_window, "gaussian_pulse.time_window")

    dt = t_win / samples
    t_axis = (np.arange(samples) - samples // 2) * dt
    envelope = np.sqrt(p0) * np.exp(-0.5 * (t_axis / t0) ** 2)

    return Pulse(
        amplitude=envelope.astype(complex),
        time_step=Quantity(dt, _unit("s")),
        wavelength=(
            wavelength
            if isinstance(wavelength, Quantity)
            else Quantity(wavelength, _unit("m"))
        ),
    )


def _raman_response(samples: int, dt: float) -> np.ndarray:
    """Raman response function h_R(t) for fused silica."""
    tau1 = 12.2e-15
    tau2 = 32.0e-15
    t = np.arange(samples) * dt
    h_r = (tau1**2 + tau2**2) / (tau1 * tau2**2) * np.exp(-t / tau2) * np.sin(t / tau1)
    norm = np.sum(h_r) * dt
    if norm > 0.0:
        h_r = h_r / norm
    return h_r


def solve_nlse(
    pulse: Pulse,
    fiber: FiberParameters,
    distance: Any,
    steps: int = 100,
    check_energy: bool = True,
) -> tuple[Pulse, list[float]]:
    """Propagate pulse along distance using the symmetric Split-Step Fourier Method (SSFM).

    Parameters
    ----------
    pulse : Pulse
        Initial optical pulse envelope.
    fiber : FiberParameters
        Waveguide dispersion, loss, and nonlinear coefficients.
    distance : Quantity
        Total propagation distance z.
    steps : int, default 100
        Number of longitudinal z-steps.
    check_energy : bool, default True
        If True and loss is zero, validates energy conservation at every step.

    Returns
    -------
    pulse_out : Pulse
        Propagated optical pulse.
    energy_history : list[float]
        Pulse energy relative to initial energy E(z) / E(0) at each step.
    """
    total_z = _length_of(distance, "solve_nlse.distance")
    if total_z <= 0.0:
        raise ValueError(f"Propagation distance must be positive, got {distance}")
    if steps < 1:
        raise ValueError(f"Steps must be at least 1, got {steps}")

    dz = total_z / steps
    omega = pulse.angular_frequencies
    omega0 = pulse.carrier_angular_frequency

    # Convert fiber parameters to SI
    beta2_si = float(fiber.beta2.to_value("s^2/m"))
    gamma_si = float(fiber.gamma.to_value("1/(W*m)"))
    beta3_si = float(fiber.beta3.to_value("s^3/m")) if fiber.beta3 is not None else 0.0
    beta4_si = float(fiber.beta4.to_value("s^4/m")) if fiber.beta4 is not None else 0.0
    alpha_si = float(fiber.loss.to_value("1/m")) if fiber.loss is not None else 0.0

    # Dispersion operator D(Omega)
    dispersion = (
        0.5 * beta2_si * omega**2
        + (beta3_si / 6.0) * omega**3
        + (beta4_si / 24.0) * omega**4
    )
    d_half = np.exp((-0.5 * alpha_si + 1.0j * dispersion) * (0.5 * dz))

    # Raman response
    use_raman = fiber.raman_fraction > 0.0
    f_r = fiber.raman_fraction
    if use_raman:
        h_r = _raman_response(pulse.samples, pulse.time_step_s)
        h_r_fft = _fft.fft(h_r)

    # Initial field and energy
    a = pulse.amplitude.copy()
    e0 = float(np.sum(np.abs(a) ** 2))
    if e0 == 0.0:
        raise ValueError("Cannot propagate an empty pulse with zero energy")

    energy_history: list[float] = [1.0]

    for step in range(steps):
        # 1. First half-step dispersion
        a = _fft.ifft(d_half * _fft.fft(a))

        # 2. Full-step nonlinearity
        power = np.abs(a) ** 2
        if use_raman:
            # Convolution: IFFT(FFT(h_R) * FFT(|A|^2)) * dt
            conv_raman = _fft.ifft(h_r_fft * _fft.fft(power)).real * pulse.time_step_s
            n_term = (1.0 - f_r) * power + f_r * conv_raman
        else:
            n_term = power

        if fiber.self_steepening:
            # Shock derivative (1 + i/omega0 d/dt)
            a_mod = n_term * a
            a_mod_f = _fft.fft(a_mod)
            shock_factor = 1.0 + (omega / omega0)
            nl_field = 1.0j * gamma_si * _fft.ifft(shock_factor * a_mod_f)
            a = a + nl_field * dz
        else:
            nl_phase = gamma_si * n_term * dz
            a = a * np.exp(1.0j * nl_phase)

        # 3. Second half-step dispersion
        a = _fft.ifft(d_half * _fft.fft(a))

        # Energy tracking
        current_energy = float(np.sum(np.abs(a) ** 2))
        rel_energy = current_energy / e0
        energy_history.append(rel_energy)

        if check_energy and alpha_si == 0.0 and not fiber.self_steepening:
            # Lossless propagation strictly preserves L2 energy norm
            if abs(rel_energy - 1.0) > 1e-4:
                raise ContractViolation(
                    f"Energy conservation violated at step {step}: "
                    f"relative energy drift {abs(rel_energy - 1.0):.2e} exceeds tolerance 1e-4"
                )

    out_pulse = Pulse(
        amplitude=a,
        time_step=pulse.time_step,
        wavelength=pulse.wavelength,
    )
    return out_pulse, energy_history


__all__ = [
    "FiberParameters",
    "Pulse",
    "gaussian_pulse",
    "soliton_parameters",
    "soliton_pulse",
    "solve_nlse",
]
