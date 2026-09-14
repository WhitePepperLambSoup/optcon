"""Detector noise: shot noise, signal to noise, noise equivalent power.

Relations follow Saleh & Teich, *Fundamentals of Photonics*, ch. 17.  The
checks are against closed-form Poisson statistics rather than another
library, because for an ideal detector the closed form is exact.
"""

import numpy as np
import pytest

from optcon import DimensionError, q
from optcon.noise import (
    dark_current_noise,
    noise_equivalent_power,
    photon_energy,
    photons_per_second,
    quantum_efficiency_from_responsivity,
    responsivity,
    shot_noise_current,
    signal_to_noise,
)

LAMBDA_NM = 1550.0
PLANCK = 6.62607015e-34
C_LIGHT = 299792458.0


def test_photon_energy_matches_hc_over_lambda():
    energy = photon_energy(q(LAMBDA_NM, "nm"))
    assert energy.to_value("J") == pytest.approx(PLANCK * C_LIGHT / 1550e-9)


def test_photon_energy_in_electronvolts():
    assert photon_energy(q(LAMBDA_NM, "nm")).to_value("eV") == pytest.approx(0.8, rel=1e-2)


def test_photons_per_second_from_optical_power():
    power = q(1e-6, "W")
    rate = photons_per_second(power, q(LAMBDA_NM, "nm"))
    assert rate.to_value("1/s") == pytest.approx(1e-6 / (PLANCK * C_LIGHT / 1550e-9))
    # a microwatt at 1550 nm delivers about 7.8 million photons per microsecond
    assert rate.to_value("1/s") * 1e-6 == pytest.approx(7.8e6, rel=2e-2)


def test_responsivity_of_a_typical_ingaas_detector():
    expected = 0.8 * 1.602176634e-19 * 1550e-9 / (PLANCK * C_LIGHT)
    assert responsivity(0.8, q(LAMBDA_NM, "nm")) == pytest.approx(expected)
    assert responsivity(0.8, q(LAMBDA_NM, "nm")) == pytest.approx(1.0, rel=5e-2)


def test_quantum_efficiency_is_the_inverse_of_responsivity():
    for efficiency in (0.3, 0.7, 0.95):
        recovered = quantum_efficiency_from_responsivity(
            responsivity(efficiency, q(LAMBDA_NM, "nm")), q(LAMBDA_NM, "nm")
        )
        assert recovered == pytest.approx(efficiency)


def test_shot_noise_follows_the_square_root_of_the_current():
    expected = np.sqrt(2.0 * 1.602176634e-19 * 1e-6 * 1e9)
    assert shot_noise_current(1e-6, bandwidth_hz=1e9).to_value("A") == pytest.approx(
        expected
    )


def test_shot_noise_scales_as_the_square_root_of_power():
    weak = shot_noise_current(1e-6, bandwidth_hz=1e9).to_value("A")
    strong = shot_noise_current(4e-6, bandwidth_hz=1e9).to_value("A")
    assert strong == pytest.approx(2.0 * weak)


def test_dark_current_adds_in_quadrature():
    total = dark_current_noise(1e-6, dark_current_a=1e-9, bandwidth_hz=1e9)
    expected = np.sqrt(2.0 * 1.602176634e-19 * (1e-6 + 1e-9) * 1e9)
    assert total.to_value("A") == pytest.approx(expected)


def test_signal_to_noise_is_the_square_root_of_the_photon_count():
    power = q(1e-9, "W")
    rate = photons_per_second(power, q(LAMBDA_NM, "nm")).to_value("1/s")
    assert signal_to_noise(power, q(LAMBDA_NM, "nm"), integration_time_s=1.0) == pytest.approx(
        np.sqrt(rate)
    )


def test_signal_to_noise_grows_as_the_square_root_of_integration_time():
    power, wavelength = q(1e-9, "W"), q(LAMBDA_NM, "nm")
    short = signal_to_noise(power, wavelength, integration_time_s=1.0)
    long = signal_to_noise(power, wavelength, integration_time_s=4.0)
    assert long == pytest.approx(2.0 * short)


def test_noise_equivalent_power_scales_with_the_bandwidth():
    assert noise_equivalent_power(1e-15, 1e9).to_value("W") == pytest.approx(
        1e-15 * np.sqrt(1e9)
    )


def test_a_dimensioned_quantum_efficiency_is_rejected():
    with pytest.raises((DimensionError, ValueError)):
        responsivity(q(0.8, "mm"), q(LAMBDA_NM, "nm"))
