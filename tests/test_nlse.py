"""Tests for the Generalized Nonlinear Schrodinger Equation (G-NLSE) Split-Step Fourier Solver."""

import math
from typing import Any, cast

import numpy as np
import pytest

from optcon import NLSEConservationTrace as PublicNLSEConservationTrace
from optcon import nlse as nlse_module
from optcon import q
from optcon.errors import DimensionError
from optcon.nlse import (
    FiberParameters,
    NLSEConservationTrace,
    Pulse,
    _self_steepening_multiplier,
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
    assert pulse.photon_number > 0.0


def test_conservation_trace_is_exported_from_the_public_package():
    assert PublicNLSEConservationTrace is NLSEConservationTrace


def test_photon_number_matches_energy_over_carrier_photon_energy_for_a_cw_envelope():
    pulse = Pulse(
        amplitude=np.ones(64, dtype=complex),
        time_step=q(2.0, "fs"),
        wavelength=q(1550.0, "nm"),
    )

    expected = pulse.energy.to_value("J") / (
        nlse_module._HBAR_J_S * pulse.carrier_angular_frequency
    )

    assert pulse.photon_number == pytest.approx(expected, rel=1e-12)


def test_pulse_dimension_validation():
    # Pass length where time expected
    with pytest.raises(DimensionError):
        gaussian_pulse(
            peak_power=q(100.0, "W"),
            fwhm_duration=q(100.0, "um"),  # Invalid: length instead of time
            wavelength=q(1550.0, "nm"),
        )


@pytest.mark.parametrize(
    ("time_step", "wavelength"),
    [(q(0.0, "fs"), q(1550.0, "nm")), (q(-1.0, "fs"), q(1550.0, "nm")), (q(1.0, "fs"), q(0.0, "nm"))],
)
def test_pulse_rejects_nonpositive_grid_parameters(time_step, wavelength):
    with pytest.raises(ValueError, match="strictly positive"):
        Pulse(np.ones(16), time_step, wavelength)


def test_pulse_rejects_nonfinite_amplitudes():
    amplitude = np.ones(16, dtype=complex)
    amplitude[3] = np.nan

    with pytest.raises(ValueError, match="finite"):
        Pulse(amplitude, q(1.0, "fs"), q(1550.0, "nm"))

    # Pass time where power expected
    with pytest.raises(DimensionError):
        gaussian_pulse(
            peak_power=q(100.0, "ps"),  # Invalid: time instead of power
            fwhm_duration=q(100.0, "fs"),
            wavelength=q(1550.0, "nm"),
        )


@pytest.mark.parametrize(
    "parameters",
    [
        {"beta2": q(float("nan"), "ps^2/km"), "gamma": q(2.0, "1/(W*km)")},
        {"beta2": q(-20.0, "ps^2/km"), "gamma": q(float("inf"), "1/(W*km)")},
        {
            "beta2": q(-20.0, "ps^2/km"),
            "gamma": q(2.0, "1/(W*km)"),
            "beta3": q(float("nan"), "ps^3/km"),
        },
        {
            "beta2": q(-20.0, "ps^2/km"),
            "gamma": q(2.0, "1/(W*km)"),
            "beta4": q(float("inf"), "ps^4/km"),
        },
        {
            "beta2": q(-20.0, "ps^2/km"),
            "gamma": q(2.0, "1/(W*km)"),
            "loss": q(float("nan"), "1/m"),
        },
    ],
)
def test_fiber_parameters_reject_nonfinite_coefficients(parameters):
    with pytest.raises(ValueError, match="finite"):
        FiberParameters(**parameters)


def test_fiber_parameters_reject_negative_attenuation():
    with pytest.raises(ValueError, match="loss must be non-negative"):
        FiberParameters(
            beta2=q(-20.0, "ps^2/km"),
            gamma=q(2.0, "1/(W*km)"),
            loss=q(-1.0, "1/m"),
        )


def test_soliton_parameters_reject_invalid_finite_inputs():
    with pytest.raises(ValueError, match="pulse_duration must be strictly positive"):
        soliton_parameters(
            q(1550.0, "nm"), q(0.0, "ps"), q(-20.0, "ps^2/km"), q(2.0, "1/(W*km)")
        )
    with pytest.raises(ValueError, match="gamma must be finite"):
        soliton_parameters(
            q(1550.0, "nm"),
            q(1.0, "ps"),
            q(-20.0, "ps^2/km"),
            q(float("nan"), "1/(W*km)"),
        )


@pytest.mark.parametrize("builder", [soliton_pulse, gaussian_pulse])
def test_pulse_builders_reject_fractional_sample_counts(builder):
    duration_name = "duration" if builder is soliton_pulse else "fwhm_duration"
    arguments = {
        "peak_power": q(1.0, "W"),
        duration_name: q(1.0, "ps"),
        "wavelength": q(1550.0, "nm"),
        "samples": 16.5,
    }
    with pytest.raises(ValueError, match="samples must be an integer"):
        builder(**arguments)


@pytest.mark.parametrize("builder", [soliton_pulse, gaussian_pulse])
def test_pulse_builders_reject_negative_peak_power(builder):
    duration_name = "duration" if builder is soliton_pulse else "fwhm_duration"
    arguments = {
        "peak_power": q(-1.0, "W"),
        duration_name: q(1.0, "ps"),
        "wavelength": q(1550.0, "nm"),
        "samples": 16,
    }
    with pytest.raises(ValueError, match="peak_power must be strictly positive"):
        builder(**arguments)


def test_nlse_helpers_reject_nonfinite_grid_parameters():
    with pytest.raises(ValueError, match="dt must be finite and strictly positive"):
        nlse_module._causal_raman_convolution(
            np.ones(2), np.ones(2), float("nan")
        )
    with pytest.raises(ValueError, match="carrier frequency must be finite and strictly positive"):
        _self_steepening_multiplier(np.ones(2), float("nan"))


def test_solve_nlse_rejects_invalid_distance_and_step_count():
    pulse = gaussian_pulse(q(1.0, "W"), q(1.0, "ps"), q(1550.0, "nm"), samples=16)
    fiber = FiberParameters(q(-20.0, "ps^2/km"), q(2.0, "1/(W*km)"))
    with pytest.raises(ValueError, match="distance must be finite"):
        solve_nlse(pulse, fiber, q(float("nan"), "m"), steps=1)
    with pytest.raises(ValueError, match="steps must be an integer"):
        solve_nlse(pulse, fiber, q(1.0, "m"), steps=cast(Any, 1.5))


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

    # 1. Energy conservation: the conservative SSFM case preserves pulse energy to numerical precision
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


def test_third_order_dispersion_uses_the_documented_fft_sign_convention():
    samples = 64
    pulse = Pulse(
        amplitude=np.exp(-0.5 * (np.linspace(-3.0, 3.0, samples) ** 2)).astype(complex),
        time_step=q(5.0, "fs"),
        wavelength=q(1550.0, "nm"),
    )
    beta3 = q(0.2, "ps^3/km")
    distance = q(25.0, "m")
    fiber = FiberParameters(
        beta2=q(0.0, "ps^2/km"),
        beta3=beta3,
        gamma=q(0.0, "1/(W*km)"),
    )

    propagated, _ = solve_nlse(pulse, fiber, distance=distance, steps=1)

    omega = pulse.angular_frequencies
    beta3_si = beta3.to_value("s^3/m")
    distance_m = distance.to_value("m")
    expected_spectrum = np.fft.fft(pulse.amplitude) * np.exp(
        -1.0j * beta3_si * omega**3 * distance_m / 6.0
    )
    expected = np.fft.ifft(expected_spectrum)

    assert propagated.amplitude == pytest.approx(expected, rel=1e-12, abs=1e-12)


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


def test_raman_convolution_is_causal_linear_and_matches_direct_sum():
    power = np.array([1.0, 2.0, 3.0, 4.0])
    response = np.array([0.0, 0.5, 0.25])
    dt = 0.2

    actual = nlse_module._causal_raman_convolution(power, response, dt)
    expected = np.convolve(power, response, mode="full")[: power.size] * dt

    assert np.allclose(actual, expected)

    # A response generated at the end of the finite window must not wrap into
    # the beginning as it would under circular convolution.
    late_impulse = np.array([0.0, 0.0, 0.0, 1.0])
    late_output = nlse_module._causal_raman_convolution(late_impulse, response, dt)
    assert np.allclose(late_output[:3], 0.0)


def test_self_steepening_fourier_multiplier_matches_derivative_convention():
    omega0 = 2.0e15
    omega = np.array([-0.5e15, 0.0, 0.5e15])
    assert _self_steepening_multiplier(omega, omega0) == pytest.approx(
        np.array([1.25, 1.0, 0.75])
    )


def test_self_steepening_substep_converges_with_longitudinal_refinement():
    pulse = gaussian_pulse(
        peak_power=q(10.0, "W"),
        fwhm_duration=q(200.0, "fs"),
        wavelength=q(1550.0, "nm"),
        samples=128,
        time_window=q(2.0, "ps"),
    )
    fiber = FiberParameters(
        beta2=q(0.0, "ps^2/km"),
        gamma=q(10.0, "1/(W*km)"),
        self_steepening=True,
    )
    distance = q(10.0, "m")
    coarse, _ = solve_nlse(pulse, fiber, distance, steps=8, check_energy=True)
    medium, _ = solve_nlse(pulse, fiber, distance, steps=16, check_energy=True)
    fine, _ = solve_nlse(pulse, fiber, distance, steps=32, check_energy=True)

    medium_error = np.linalg.norm(medium.amplitude - fine.amplitude)
    coarse_error = np.linalg.norm(coarse.amplitude - fine.amplitude)
    assert medium_error < 0.35 * coarse_error


def test_full_lossless_raman_and_self_steepening_track_photon_number():
    pulse = gaussian_pulse(
        peak_power=q(5.0, "W"),
        fwhm_duration=q(200.0, "fs"),
        wavelength=q(1550.0, "nm"),
        samples=128,
        time_window=q(2.0, "ps"),
    )
    fiber = FiberParameters(
        beta2=q(-20.0, "ps^2/km"),
        gamma=q(2.0, "1/(W*km)"),
        raman_fraction=0.18,
        self_steepening=True,
    )
    _, trace = solve_nlse(
        pulse, fiber, distance=q(1.0, "m"), steps=16, check_energy=True
    )

    assert isinstance(trace, NLSEConservationTrace)
    assert trace.conserved_quantity == "photon_number"
    assert max(abs(value - 1.0) for value in trace.invariant_history) < 1e-4
    assert len(trace.invariant_history) == len(trace)


def test_raman_without_self_steepening_uses_envelope_energy_as_the_invariant():
    pulse = gaussian_pulse(
        peak_power=q(5.0, "W"),
        fwhm_duration=q(200.0, "fs"),
        wavelength=q(1550.0, "nm"),
        samples=128,
        time_window=q(2.0, "ps"),
    )
    fiber = FiberParameters(
        beta2=q(-20.0, "ps^2/km"),
        gamma=q(2.0, "1/(W*km)"),
        raman_fraction=0.18,
    )

    _, trace = solve_nlse(
        pulse, fiber, distance=q(1.0, "m"), steps=16, check_energy=True
    )

    assert trace.conserved_quantity == "energy"
    assert trace.invariant_history == pytest.approx(list(trace))
