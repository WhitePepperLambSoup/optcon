"""Tests for the Generalized Nonlinear Schrodinger Equation (G-NLSE) Split-Step Fourier Solver."""

import math

import numpy as np
import pytest

from optcon import q
from optcon.errors import DimensionError
from optcon.nlse import (
    FiberParameters,
    Pulse,
    gaussian_pulse,
    soliton_parameters,
    soliton_pulse,
    solve_nlse,
)


def test_pulse_creation_and_properties():
    pulse = gaussian_pulse(
        peak_power=q(100.0, "W"),
        fwhm_duration=q(200.0, "fs"),
        wavelength=q(1550.0, "nm"),
        samples=512,
    )
    assert pulse.samples == 512
    assert pulse.peak_power.to_value("W") == pytest.approx(100.0, rel=1e-3)
    assert pulse.fwhm_duration.to_value("fs") == pytest.approx(200.0, rel=1e-2)
    assert pulse.energy.to_value("pJ") > 0.0


def test_pulse_dimension_validation():
    # Pass length where time expected
    with pytest.raises(DimensionError):
        gaussian_pulse(
            peak_power=q(100.0, "W"),
            fwhm_duration=q(100.0, "um"),  # Invalid: length instead of time
            wavelength=q(1550.0, "nm"),
        )

    # Pass time where power expected
    with pytest.raises(DimensionError):
        gaussian_pulse(
            peak_power=q(100.0, "ps"),  # Invalid: time instead of power
            fwhm_duration=q(100.0, "fs"),
            wavelength=q(1550.0, "nm"),
        )


def test_soliton_parameters_and_balance():
    # For a fundamental soliton, dispersion length L_D equals nonlinear length L_NL
    scales = soliton_parameters(
        wavelength=q(1550.0, "nm"),
        pulse_duration=q(1.0, "ps"),  # T0 = 1 ps
        beta2=q(-20.0, "ps^2/km"),
        gamma=q(2.0, "1/(W*km)"),
    )
    l_d = scales["dispersion_length"].to_value("m")
    l_nl = scales["nonlinear_length"].to_value("m")
    z0 = scales["soliton_period"].to_value("m")
    p0 = scales["peak_power"].to_value("W")

    assert l_d == pytest.approx(50.0, rel=1e-6)
    assert l_nl == pytest.approx(50.0, rel=1e-6)
    assert l_d == pytest.approx(l_nl, rel=1e-9)  # Balance condition L_D == L_NL
    assert z0 == pytest.approx(0.5 * math.pi * 50.0, rel=1e-6)
    assert p0 == pytest.approx(10.0, rel=1e-6)

    # Normal dispersion raises ValueError for bright soliton
    with pytest.raises(ValueError, match="anomalous dispersion"):
        soliton_parameters(
            wavelength=q(1550.0, "nm"),
            pulse_duration=q(1.0, "ps"),
            beta2=q(20.0, "ps^2/km"),  # Normal dispersion
            gamma=q(2.0, "1/(W*km)"),
        )


def test_fundamental_soliton_propagation():
    # Propagate fundamental soliton over 1 full soliton period z0 = 78.5 km
    t0 = q(1.0, "ps")
    b2 = q(-20.0, "ps^2/km")
    gam = q(2.0, "1/(W*km)")
    lam = q(1550.0, "nm")

    scales = soliton_parameters(lam, t0, b2, gam)
    z0 = scales["soliton_period"]
    p0 = scales["peak_power"]

    pulse_in = soliton_pulse(
        peak_power=p0,
        duration=t0,
        wavelength=lam,
        samples=512,
        time_window=q(20.0, "ps"),
    )
    fiber = FiberParameters(beta2=b2, gamma=gam)

    pulse_out, energy_hist = solve_nlse(
        pulse_in, fiber, distance=z0, steps=100, check_energy=True
    )

    # 1. Energy conservation: lossless SSFM preserves pulse energy to machine precision
    e_in = pulse_in.energy.to_value("pJ")
    e_out = pulse_out.energy.to_value("pJ")
    assert abs(e_out - e_in) / e_in < 1e-11

    # 2. Intensity preservation: fundamental soliton shape is invariant
    intensity_in = np.abs(pulse_in.amplitude) ** 2
    intensity_out = np.abs(pulse_out.amplitude) ** 2
    max_diff = np.max(np.abs(intensity_out - intensity_in)) / np.max(intensity_in)
    assert max_diff < 1e-4

    # 3. Phase accumulation: theoretical phase shift after z0 is z0 / (2 * L_D) = pi / 4
    l_d = scales["dispersion_length"].to_value("m")
    z0_m = z0.to_value("m")
    expected_phase = z0_m / (2.0 * l_d)
    analytical_envelope = pulse_in.amplitude * np.exp(1.0j * expected_phase)
    complex_err = np.max(np.abs(pulse_out.amplitude - analytical_envelope)) / np.max(
        np.abs(pulse_in.amplitude)
    )
    assert complex_err < 1e-4


def test_pure_dispersion_gaussian_broadening():
    # Pure GVD without nonlinearity: gamma = 0
    t_fwhm = q(100.0, "fs")
    b2 = q(20.0, "ps^2/km")  # Normal GVD
    gam = q(0.0, "1/(W*km)")
    lam = q(1064.0, "nm")

    t0_s = t_fwhm.to_value("s") / (2.0 * math.sqrt(math.log(2.0)))
    b2_si = abs(b2.to_value("s^2/m"))
    l_d = t0_s**2 / b2_si

    pulse_in = gaussian_pulse(
        peak_power=q(50.0, "W"),
        fwhm_duration=t_fwhm,
        wavelength=lam,
        samples=512,
        time_window=q(2.0, "ps"),
    )
    fiber = FiberParameters(beta2=b2, gamma=gam)

    # Propagate by exactly 1 dispersion length L_D
    pulse_out, _ = solve_nlse(
        pulse_in, fiber, distance=q(l_d, "m"), steps=50, check_energy=True
    )

    # Analytical theory: at z = L_D, peak power drops by 1/sqrt(2) = 0.7071
    p_in = pulse_in.peak_power.to_value("W")
    p_out = pulse_out.peak_power.to_value("W")
    assert p_out == pytest.approx(p_in / math.sqrt(2.0), rel=1e-3)

    # Energy is strictly conserved
    assert pulse_out.energy.to_value("pJ") == pytest.approx(
        pulse_in.energy.to_value("pJ"), rel=1e-6
    )


def test_pure_spm_phase_modulation():
    # Pure Kerr nonlinearity: beta2 = 0
    pulse_in = gaussian_pulse(
        peak_power=q(10.0, "W"),
        fwhm_duration=q(500.0, "fs"),
        wavelength=q(1550.0, "nm"),
        samples=512,
    )
    fiber = FiberParameters(
        beta2=q(0.0, "ps^2/km"),
        gamma=q(5.0, "1/(W*km)"),
    )
    dist = q(1.0, "km")

    pulse_out, _ = solve_nlse(pulse_in, fiber, distance=dist, steps=50)

    # Temporal intensity is completely unchanged
    intensity_in = np.abs(pulse_in.amplitude) ** 2
    intensity_out = np.abs(pulse_out.amplitude) ** 2
    assert np.allclose(intensity_out, intensity_in, rtol=1e-6, atol=1e-6)

    # Maximum nonlinear phase shift phi_max = gamma * P0 * L = 5 * 10 * 1 = 50 rad
    phase_in = np.unwrap(np.angle(pulse_in.amplitude))
    phase_out = np.unwrap(np.angle(pulse_out.amplitude))
    delta_phi = phase_out - phase_in
    assert np.max(delta_phi) == pytest.approx(5.0e-3 * 10.0 * 1000.0, rel=1e-3)


def test_higher_order_dispersion_and_symmetry_breaking():
    # Third-order dispersion beta3 breaks temporal reflection symmetry
    pulse_in = gaussian_pulse(
        peak_power=q(20.0, "W"),
        fwhm_duration=q(50.0, "fs"),
        wavelength=q(1550.0, "nm"),
        samples=512,
    )
    fiber = FiberParameters(
        beta2=q(0.0, "ps^2/km"),
        gamma=q(0.0, "1/(W*km)"),
        beta3=q(0.5, "ps^3/km"),
    )
    pulse_out, _ = solve_nlse(pulse_in, fiber, distance=q(100.0, "m"), steps=40)

    # Symmetry breaking: pulse intensity is no longer symmetric around t=0
    p = np.abs(pulse_out.amplitude) ** 2
    left_half = p[: len(p) // 2]
    right_half = p[len(p) // 2 :][::-1]
    assert not np.allclose(left_half, right_half, rtol=1e-2)

    # Energy is still conserved
    assert pulse_out.energy.to_value("pJ") == pytest.approx(
        pulse_in.energy.to_value("pJ"), rel=1e-6
    )


def test_lossy_fiber_attenuation():
    # Linear attenuation: energy drops exponentially E(z) = E0 * exp(-alpha * z)
    pulse_in = gaussian_pulse(
        peak_power=q(1.0, "W"),
        fwhm_duration=q(1.0, "ps"),
        wavelength=q(1550.0, "nm"),
    )
    alpha = q(0.001, "1/m")  # 1e-3 / m
    fiber = FiberParameters(
        beta2=q(0.0, "ps^2/km"),
        gamma=q(0.0, "1/(W*km)"),
        loss=alpha,
    )
    dist = q(1000.0, "m")
    pulse_out, energy_hist = solve_nlse(
        pulse_in, fiber, distance=dist, steps=50, check_energy=False
    )

    # E(1000 m) / E0 = exp(-1e-3 * 1000) = exp(-1.0) = 0.36788
    expected_ratio = math.exp(-1.0)
    assert energy_hist[-1] == pytest.approx(expected_ratio, rel=1e-4)


def test_raman_self_frequency_redshift():
    # Raman delayed response causes spectral redshift for short pulses
    pulse_in = soliton_pulse(
        peak_power=q(100.0, "W"),
        duration=q(50.0, "fs"),
        wavelength=q(1550.0, "nm"),
        samples=512,
        time_window=q(2.0, "ps"),
    )
    fiber_no_raman = FiberParameters(
        beta2=q(-20.0, "ps^2/km"),
        gamma=q(5.0, "1/(W*km)"),
        raman_fraction=0.0,
    )
    fiber_raman = FiberParameters(
        beta2=q(-20.0, "ps^2/km"),
        gamma=q(5.0, "1/(W*km)"),
        raman_fraction=0.18,  # Standard silica Raman fraction
    )

    dist = q(50.0, "m")
    p_no_raman, _ = solve_nlse(pulse_in, fiber_no_raman, distance=dist, steps=50)
    p_raman, _ = solve_nlse(pulse_in, fiber_raman, distance=dist, steps=50)

    # Compute spectral centroid
    def spectral_centroid(p: Pulse) -> float:
        spec = np.abs(np.fft.fftshift(np.fft.fft(p.amplitude))) ** 2
        freqs = np.fft.fftshift(p.angular_frequencies)
        return float(np.sum(freqs * spec) / np.sum(spec))

    c_no_raman = spectral_centroid(p_no_raman)
    c_raman = spectral_centroid(p_raman)

    # Raman self-frequency shift: in envelope representation E = Re[A * exp(-i omega0 t)],
    # an optical redshift (omega < omega0) corresponds to positive envelope frequency shift
    assert c_raman > c_no_raman + 1e11
