"""Radiometry and blackbody radiation.

Relations follow Planck's law in its wavelength form and the standard
radiometric identities (Lambertian exitance, solid angle of a cone, radiance
invariance).  The strongest check here integrates Planck's law numerically
and recovers the Stefan-Boltzmann law, which ties two independently stated
constants together.
"""

import numpy as np
import pytest

from optcon import q
from optcon.radiometry import (
    etendue,
    irradiance,
    luminous_flux,
    planck_exitance,
    planck_radiance,
    radiance,
    solid_angle_cone,
    stefan_boltzmann,
    wien_displacement,
)

SIGMA = 5.670374419e-8
WIEN_B = 2.897771955e-3


def test_stefan_boltzmann_at_room_temperature():
    assert stefan_boltzmann(300.0).to_value("W/m^2") == pytest.approx(SIGMA * 300.0**4)


def test_stefan_boltzmann_scales_as_the_fourth_power():
    double = stefan_boltzmann(600.0).to_value("W/m^2")
    single = stefan_boltzmann(300.0).to_value("W/m^2")
    assert double == pytest.approx(16.0 * single)


def test_wien_displacement_law():
    peak = wien_displacement(300.0)
    assert peak.to_value("um") == pytest.approx(WIEN_B / 300.0 * 1e6)
    assert peak.to_value("um") == pytest.approx(9.66, rel=1e-2)


def test_the_planck_spectrum_actually_peaks_at_the_wien_wavelength():
    """Two statements of the same physics, checked against each other."""
    for temperature in (300.0, 1000.0, 5800.0):
        predicted = wien_displacement(temperature).to_value("m")
        grid = np.linspace(0.4 * predicted, 2.5 * predicted, 4000)
        values = [
            planck_radiance(q(float(wavelength), "m"), temperature).to_value(
                "W/(m^2*sr*m)"
            )
            for wavelength in grid
        ]
        assert grid[int(np.argmax(values))] == pytest.approx(predicted, rel=2e-3)


def test_integrating_planck_over_wavelength_recovers_stefan_boltzmann():
    """The pi B = M relation plus the Planck integral must give sigma T^4."""
    temperature = 1200.0
    peak = WIEN_B / temperature
    wavelengths = np.logspace(np.log10(peak / 40.0), np.log10(peak * 400.0), 20000)
    exitance = np.array(
        [
            planck_exitance(q(float(wavelength), "m"), temperature).to_value("W/m^3")
            for wavelength in wavelengths
        ]
    )
    total = np.trapezoid(exitance, wavelengths)
    assert total == pytest.approx(SIGMA * temperature**4, rel=1e-3)


def test_rayleigh_jeans_limit_at_long_wavelengths():
    temperature = 1000.0
    long_wavelength = q(1e-2, "m")
    exact = planck_radiance(long_wavelength, temperature).to_value("W/(m^2*sr*m)")
    # B ~ 2 c k T / lambda^4
    expected = 2.0 * 299792458.0 * 1.380649e-23 * temperature / (1e-2) ** 4
    assert exact == pytest.approx(expected, rel=1e-3)


def test_solid_angle_of_a_cone():
    assert solid_angle_cone(q(0.0, "deg")).to_value("sr") == pytest.approx(0.0)
    assert solid_angle_cone(q(90.0, "deg")).to_value("sr") == pytest.approx(2.0 * np.pi)
    half = np.deg2rad(30.0)
    assert solid_angle_cone(q(30.0, "deg")).to_value("sr") == pytest.approx(
        2.0 * np.pi * (1.0 - np.cos(half))
    )


def test_etendue_is_area_times_solid_angle():
    value = etendue(q(1e-6, "m^2"), q(0.01, "sr"))
    assert value.to_value("m^2*sr") == pytest.approx(1e-8)


def test_radiance_and_irradiance_differ_by_the_solid_angle():
    power = q(1.0, "W")
    area = q(1e-4, "m^2")
    angle = q(0.01, "sr")
    assert irradiance(power, area).to_value("W/m^2") == pytest.approx(1e4)
    assert radiance(power, area, angle).to_value("W/(m^2*sr)") == pytest.approx(1e6)


def test_a_lambertian_source_has_pi_between_radiance_and_exitance():
    # M = pi B for a Lambertian emitter
    temperature = 500.0
    wavelength = q(10e-6, "m")
    spectral_radiance = planck_radiance(wavelength, temperature).to_value("W/(m^2*sr*m)")
    spectral_exitance = planck_exitance(wavelength, temperature).to_value("W/m^3")
    assert spectral_exitance == pytest.approx(np.pi * spectral_radiance)


def test_luminous_flux_uses_the_luminous_efficacy():
    assert luminous_flux(q(1.0, "W"), 683.0).to_value("lm") == pytest.approx(683.0)


def test_a_negative_temperature_is_rejected():
    with pytest.raises(ValueError, match="positive"):
        stefan_boltzmann(-1.0)


def test_radiance_invariance_and_errors():
    from optcon import DimensionError
    from optcon.radiometry import radiance_invariance

    e1 = etendue(q(1e-6, "m^2"), q(0.01, "sr"))
    e2 = etendue(q(2e-6, "m^2"), q(0.005, "sr"))
    assert radiance_invariance(e1, e2) == pytest.approx(1.0)

    # Incompatible arguments to etendue
    with pytest.raises(TypeError):
        etendue(1.0, q(0.01, "sr"))

    # solid_angle_cone errors
    with pytest.raises(TypeError, match="expected an angle"):
        solid_angle_cone(0.5)

    with pytest.raises(DimensionError):
        solid_angle_cone(q(1.0, "m"))

    with pytest.raises(ValueError, match="must lie in"):
        solid_angle_cone(q(200.0, "deg"))

    # irradiance and radiance errors
    with pytest.raises(TypeError):
        irradiance(1.0, q(1e-4, "m^2"))  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="illuminated area must be positive"):
        irradiance(q(1.0, "W"), q(-1e-4, "m^2"))

    with pytest.raises(TypeError):
        radiance(q(1.0, "W"), q(1e-4, "m^2"), 0.01)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="solid angle must be positive"):
        radiance(q(1.0, "W"), q(1e-4, "m^2"), q(-0.01, "sr"))

    # luminous flux errors
    with pytest.raises(TypeError):
        luminous_flux(1.0, 683.0)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="cannot be negative"):
        luminous_flux(q(1.0, "W"), -10.0)

    # radiance invariance errors
    with pytest.raises(TypeError):
        radiance_invariance(1.0, e2)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="first etendue must be positive"):
        radiance_invariance(etendue(q(-1e-6, "m^2"), q(0.01, "sr")), e2)

