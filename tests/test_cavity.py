"""Two-mirror cavities: free spectral range, finesse, stability, mode sizes.

Every relation checked here is a textbook one, but the strongest test is the
last: the mode size predicted by the ``g``-parameter formulas must agree with
the eigenmode computed by solving the ABCD round trip and measuring its beam
radius.  Those two routes share no code, so agreement is evidence.
"""

import numpy as np
import pytest

from optcon import q
from optcon.cavity import (
    circulating_power,
    finesse,
    free_spectral_range,
    g_parameters,
    is_stable_cavity,
    linewidth,
    photon_lifetime,
    quality_factor,
    round_trip_gouy_phase,
    spot_size_at_mirrors,
    stability_product,
    transverse_mode_spacing,
    waist_size,
)
from optcon.elements import compose, curved_mirror, free_space
from optcon.gaussian import beam_radius, propagate, self_consistent_mode

C_LIGHT = 299792458.0
LENGTH_MM = 300.0
RADIUS_MM = 500.0
LAMBDA_MM = 1.064e-3


def test_free_spectral_range_of_a_two_mirror_cavity():
    fsr = free_spectral_range(q(LENGTH_MM, "mm"))
    assert fsr.to_value("Hz") == pytest.approx(C_LIGHT / (2.0 * LENGTH_MM * 1e-3))


def test_free_spectral_range_scales_with_the_refractive_index():
    in_glass = free_spectral_range(q(LENGTH_MM, "mm"), refractive_index=1.5)
    in_vacuum = free_spectral_range(q(LENGTH_MM, "mm"))
    assert in_glass.to_value("Hz") == pytest.approx(in_vacuum.to_value("Hz") / 1.5)


def test_finesse_matches_the_closed_form():
    for reflectance in (0.5, 0.9, 0.99, 0.999):
        expected = np.pi * np.sqrt(reflectance) / (1.0 - reflectance)
        assert finesse(reflectance) == pytest.approx(expected)


def test_linewidth_is_the_ratio_of_free_spectral_range_to_finesse():
    fsr = free_spectral_range(q(LENGTH_MM, "mm"))
    f = 300.0
    width = linewidth(fsr, f)
    assert width.to_value("Hz") == pytest.approx(fsr.to_value("Hz") / f)


def test_quality_factor_is_frequency_over_linewidth():
    fsr = free_spectral_range(q(LENGTH_MM, "mm"))
    f = 300.0
    width = linewidth(fsr, f).to_value("Hz")
    frequency = C_LIGHT / (LAMBDA_MM * 1e-3)
    assert quality_factor(q(frequency, "Hz"), fsr, f) == pytest.approx(frequency / width)


def test_g_parameters_and_stability_product():
    g1, g2 = g_parameters(q(LENGTH_MM, "mm"), q(RADIUS_MM, "mm"), q(RADIUS_MM, "mm"))
    expected = 1.0 - LENGTH_MM / RADIUS_MM
    assert g1 == pytest.approx(expected)
    assert g2 == pytest.approx(expected)
    assert stability_product(q(LENGTH_MM, "mm"), q(RADIUS_MM, "mm"), q(RADIUS_MM, "mm")) == pytest.approx(
        expected**2
    )
    assert is_stable_cavity(q(LENGTH_MM, "mm"), q(RADIUS_MM, "mm"), q(RADIUS_MM, "mm"))


def test_a_flat_flat_cavity_is_not_stable():
    assert not is_stable_cavity(
        q(LENGTH_MM, "mm"), q(np.inf, "mm"), q(np.inf, "mm")
    )


def test_round_trip_gouy_phase_of_a_confocal_cavity_is_half_a_wave():
    # R1 = R2 = L gives g = 0, so arccos(0) = pi/2
    g1, g2 = g_parameters(q(100.0, "mm"), q(100.0, "mm"), q(100.0, "mm"))
    assert round_trip_gouy_phase(g1, g2).to_value("rad") == pytest.approx(np.pi / 2.0)


def test_confocal_transverse_mode_spacing_is_half_the_free_spectral_range():
    fsr = free_spectral_range(q(100.0, "mm"))
    g1, g2 = g_parameters(q(100.0, "mm"), q(100.0, "mm"), q(100.0, "mm"))
    spacing = transverse_mode_spacing(fsr, g1, g2)
    assert spacing.to_value("Hz") == pytest.approx(fsr.to_value("Hz") / 2.0)


def test_waist_size_matches_the_g_parameter_formula():
    g1, g2 = g_parameters(q(LENGTH_MM, "mm"), q(RADIUS_MM, "mm"), q(RADIUS_MM, "mm"))
    length_mm = LENGTH_MM
    expected = np.sqrt(
        LAMBDA_MM * length_mm / np.pi * np.sqrt(g1 * g2 * (1 - g1 * g2)) / (g1 + g2 - 2 * g1 * g2)
    )
    assert waist_size(
        q(LENGTH_MM, "mm"), q(RADIUS_MM, "mm"), q(RADIUS_MM, "mm"), q(LAMBDA_MM, "mm")
    ).to_value("mm") == pytest.approx(expected)


def test_spot_sizes_agree_with_the_abcd_eigenmode():
    """Two independent routes to the same mode size on the mirrors."""
    length = q(LENGTH_MM, "mm")
    radius = q(RADIUS_MM, "mm")
    wavelength = q(LAMBDA_MM, "mm")

    on_mirror_1, on_mirror_2 = spot_size_at_mirrors(length, radius, radius, wavelength)

    round_trip = compose(
        free_space(length),
        curved_mirror(radius),
        free_space(length),
        curved_mirror(radius),
    )
    eigenmode = self_consistent_mode(round_trip, wavelength)
    # the round trip starts and ends at mirror 1, so its beam radius is w1;
    # after one free space plus a mirror the beam sits on mirror 2
    from_abcd_at_mirror_1 = beam_radius(eigenmode, wavelength).to_value("mm")
    at_mirror_2 = propagate(eigenmode, compose(free_space(length), curved_mirror(radius)))
    from_abcd_at_mirror_2 = beam_radius(at_mirror_2, wavelength).to_value("mm")

    assert from_abcd_at_mirror_1 == pytest.approx(on_mirror_1.to_value("mm"), rel=1e-9)
    assert from_abcd_at_mirror_2 == pytest.approx(on_mirror_2.to_value("mm"), rel=1e-9)


def test_circulating_power_buildup():
    # on resonance the internal power is 1/(1-R) times the input for a
    # symmetric cavity driven through one mirror
    assert circulating_power(1.0, input_reflectance=0.99) == pytest.approx(100.0, rel=1e-9)
    assert circulating_power(1.0, input_reflectance=0.0) == pytest.approx(1.0)


def test_photon_lifetime_scales_with_finesse():
    fsr = free_spectral_range(q(LENGTH_MM, "mm"))
    f = 300.0
    tau = photon_lifetime(fsr, f)
    # tau = 1 / (2 pi * linewidth)
    assert tau.to_value("s") == pytest.approx(
        1.0 / (2.0 * np.pi * linewidth(fsr, f).to_value("Hz"))
    )
