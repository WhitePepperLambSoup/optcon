"""Interferometry: fringe visibility, coherence, and the Fabry-Perot etalon.

Relations follow Hecht, *Optics*, ch. 9, and Born & Wolf, ch. 7.  The
coherence length is supplied in two forms - from a linewidth and from a
wavelength spread - which must agree with each other, and that agreement is
itself a test.
"""

import numpy as np
import pytest

from optcon import q
from optcon.interferometry import (
    coherence_length,
    coherence_length_from_wavelength_spread,
    coherence_time,
    etalon_transmission,
    fringe_visibility,
    peak_to_minimum_ratio,
    two_beam_intensity,
    visibility_from_intensities,
)

C_LIGHT = 299792458.0
LAMBDA_NM = 633.0


def test_visibility_of_ideal_fringes_is_one():
    assert fringe_visibility(1.0, 0.0) == pytest.approx(1.0)
    assert visibility_from_intensities(1.0, 1.0) == pytest.approx(1.0)


def test_visibility_vanishes_when_one_beam_dominates():
    assert fringe_visibility(1.0, 1.0) == pytest.approx(0.0)
    assert visibility_from_intensities(1.0, 0.0) == pytest.approx(0.0)


def test_visibility_of_unequal_beams_matches_the_closed_form():
    first, second = 4.0, 1.0
    expected = 2.0 * np.sqrt(first * second) / (first + second)
    assert visibility_from_intensities(first, second) == pytest.approx(expected)
    assert fringe_visibility(first + second + 2 * np.sqrt(first * second), first + second - 2 * np.sqrt(first * second)) == pytest.approx(expected)


def test_coherence_time_is_the_inverse_linewidth():
    linewidth = q(1e6, "Hz")
    assert coherence_time(linewidth).to_value("s") == pytest.approx(1e-6)


def test_coherence_length_is_c_over_linewidth():
    linewidth = q(1e6, "Hz")
    assert coherence_length(linewidth).to_value("m") == pytest.approx(C_LIGHT / 1e6)


def test_the_two_routes_to_coherence_length_agree():
    """lambda^2 / delta-lambda and c / delta-nu are the same statement."""
    linewidth = q(1e9, "Hz")
    from_linewidth = coherence_length(linewidth).to_value("m")
    delta_lambda = q(LAMBDA_NM * 1e-9 * 1e9 / C_LIGHT * LAMBDA_NM * 1e-9, "m")
    from_spread = coherence_length_from_wavelength_spread(
        q(LAMBDA_NM, "nm"), delta_lambda
    ).to_value("m")
    assert from_spread == pytest.approx(from_linewidth, rel=1e-9)


def test_etalon_transmission_is_one_on_resonance():
    for phase in (0.0, 2.0 * np.pi, -4.0 * np.pi):
        assert etalon_transmission(phase, reflectance=0.9) == pytest.approx(1.0)


def test_etalon_transmission_minimum_is_the_airy_value():
    reflectance = 0.9
    finesse_coefficient = 4.0 * reflectance / (1.0 - reflectance) ** 2
    minimum = etalon_transmission(np.pi, reflectance=reflectance)
    assert minimum == pytest.approx(1.0 / (1.0 + finesse_coefficient))


def test_peak_to_minimum_ratio_is_one_plus_the_finesse_coefficient():
    reflectance = 0.8
    expected = 1.0 + 4.0 * reflectance / (1.0 - reflectance) ** 2
    assert peak_to_minimum_ratio(reflectance) == pytest.approx(expected)


def test_two_beam_intensity_oscillates_with_the_path_difference():
    first, second = 1.0, 1.0
    wavelength = q(LAMBDA_NM, "nm")
    at_zero = two_beam_intensity(q(0.0, "nm"), wavelength, first, second)
    at_half_wave = two_beam_intensity(q(LAMBDA_NM / 2.0, "nm"), wavelength, first, second)
    at_full_wave = two_beam_intensity(q(LAMBDA_NM, "nm"), wavelength, first, second)
    assert at_zero == pytest.approx(4.0)
    assert at_half_wave == pytest.approx(0.0, abs=1e-18)
    assert at_full_wave == pytest.approx(4.0)


def test_a_negative_linewidth_is_rejected():
    with pytest.raises(ValueError, match="positive"):
        coherence_time(q(-1.0, "Hz"))
