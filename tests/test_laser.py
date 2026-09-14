"""Laser physics: threshold gain, coupling, slope efficiency, relaxation dynamics.

Relations follow Svelto, *Principles of Lasers*, ch. 7, for the four-level
rate equations.  The threshold-pump result in particular is derived rather
than fitted: ``P_th = I_sat A delta / 2`` where ``delta`` is the round-trip
loss, which is what the tests check symbolically.
"""

import numpy as np
import pytest

from optcon import q
from optcon.laser import (
    output_power,
    photon_lifetime_cavity,
    relaxation_oscillation_frequency,
    saturation_intensity,
    slope_efficiency,
    threshold_gain,
    threshold_pump_power,
)

PLANCK = 6.62607015e-34
C_LIGHT = 299792458.0
LAMBDA_NM = 1064.0


def test_saturation_intensity_is_h_nu_over_sigma_tau():
    sigma = 1e-23
    tau = 230e-6
    expected = PLANCK * C_LIGHT / (1064e-9) / (sigma * tau)
    value = saturation_intensity(q(LAMBDA_NM, "nm"), sigma, tau)
    assert value.to_value("W/m^2") == pytest.approx(expected)


def test_threshold_gain_with_only_the_output_coupler():
    # 2 g_th L = -ln(R_out)
    value = threshold_gain(q(0.3, "m"), output_coupler_reflectance=0.99)
    assert value.to_value("1/m") == pytest.approx(-np.log(0.99) / (2.0 * 0.3))


def test_threshold_gain_vanishes_as_the_coupler_becomes_ideal():
    value = threshold_gain(q(0.3, "m"), output_coupler_reflectance=1.0)
    assert value.to_value("1/m") == pytest.approx(0.0)


def test_internal_losses_add_to_the_threshold():
    base = threshold_gain(q(0.3, "m"), output_coupler_reflectance=1.0)
    lossy = threshold_gain(q(0.3, "m"), output_coupler_reflectance=1.0, internal_loss_per_m=0.02)
    assert lossy.to_value("1/m") == pytest.approx(base.to_value("1/m") + 0.02)


def test_photon_lifetime_of_an_empty_cavity():
    # tau_c = 2L / (c * delta) with delta = -ln(R)
    lifetime = photon_lifetime_cavity(q(0.3, "m"), output_coupler_reflectance=0.99)
    expected = 2.0 * 0.3 / (C_LIGHT * (-np.log(0.99)))
    assert lifetime.to_value("s") == pytest.approx(expected)


def test_slope_efficiency_is_limited_by_the_quantum_defect():
    value = slope_efficiency(q(808.0, "nm"), q(1064.0, "nm"))
    assert value == pytest.approx(808.0 / 1064.0)
    # imperfect mode overlap or quantum efficiency can only reduce it
    worse = slope_efficiency(q(808.0, "nm"), q(1064.0, "nm"), mode_overlap=0.8)
    assert worse == pytest.approx(0.8 * 808.0 / 1064.0)
    assert worse < value


def test_a_slope_efficiency_above_one_is_rejected():
    with pytest.raises(ValueError, match="cannot exceed"):
        slope_efficiency(q(1064.0, "nm"), q(808.0, "nm"))


def test_threshold_pump_power_is_saturation_intensity_times_area_times_loss():
    intensity = q(1e7, "W/m^2")
    area = q(1e-8, "m^2")
    power = threshold_pump_power(intensity, area, round_trip_loss=0.05)
    assert power.to_value("W") == pytest.approx(1e7 * 1e-8 * 0.05 / 2.0)


def test_output_power_is_linear_above_threshold_and_zero_below():
    assert output_power(0.5, pump_power=10.0, threshold_power=8.0) == pytest.approx(1.0)
    assert output_power(0.5, pump_power=6.0, threshold_power=8.0) == pytest.approx(0.0)


def test_relaxation_oscillation_frequency_grows_as_the_root_of_the_pump_excess():
    tau_c = q(100e-9, "s")
    tau_f = q(230e-6, "s")
    at_two = relaxation_oscillation_frequency(tau_c, tau_f, pump_ratio=2.0)
    at_five = relaxation_oscillation_frequency(tau_c, tau_f, pump_ratio=5.0)
    assert at_two.to_value("Hz") == pytest.approx(
        np.sqrt(1.0 / (100e-9 * 230e-6)) / (2.0 * np.pi)
    )
    assert at_five.to_value("Hz") / at_two.to_value("Hz") == pytest.approx(np.sqrt(4.0 / 1.0))


def test_a_threshold_at_or_below_one_is_rejected():
    with pytest.raises(ValueError, match="excitation ratio"):
        relaxation_oscillation_frequency(q(1e-7, "s"), q(1e-4, "s"), pump_ratio=0.9)
