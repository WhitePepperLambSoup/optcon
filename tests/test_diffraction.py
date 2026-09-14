"""Diffraction limits: the Airy pattern, resolution, Strehl ratio.

Relations follow Born & Wolf, *Principles of Optics*, ch. 8, and Mahajan,
*Aberration Theory Made Simple*, for the Maréchal approximation.
"""

import numpy as np
import pytest

from optcon import q
from optcon.diffraction import (
    airy_intensity,
    airy_radius,
    angular_resolution,
    encircled_energy,
    f_number,
    rms_wavefront_from_strehl,
    strehl_ratio,
)

LAMBDA_NM = 550.0
APERTURE_MM = 10.0


def test_f_number_is_focal_length_over_aperture():
    assert f_number(q(100.0, "mm"), q(10.0, "mm")) == pytest.approx(10.0)


def test_airy_radius_uses_the_1_22_factor():
    radius = airy_radius(q(LAMBDA_NM, "nm"), 4.0)
    assert radius.to_value("nm") == pytest.approx(1.22 * LAMBDA_NM * 4.0)


def test_angular_resolution_of_a_circular_aperture():
    angle = angular_resolution(q(LAMBDA_NM, "nm"), q(APERTURE_MM, "mm"))
    assert angle.to_value("rad") == pytest.approx(
        1.22 * LAMBDA_NM * 1e-9 / (APERTURE_MM * 1e-3)
    )


def test_airy_pattern_is_one_on_axis_and_zero_at_the_first_zero():
    assert airy_intensity(0.0) == pytest.approx(1.0)
    first_zero = 3.831705970207512  # first zero of J1(x)
    assert airy_intensity(first_zero) == pytest.approx(0.0, abs=1e-12)


def test_airy_pattern_has_the_characteristic_secondary_maximum():
    # the first ring peaks at about 1.75% of the central intensity, near
    # x = 5.136 (j2' first zero)
    peak = airy_intensity(5.1356)
    assert peak == pytest.approx(0.0175, rel=0.02)


def test_encircled_energy_limits():
    assert encircled_energy(0.0) == pytest.approx(0.0, abs=1e-12)
    # the tail approaches one only as fast as J0^2 + J1^2 decays, which is
    # ~2/(pi u): about 6e-4 at u = 1000, and still 6e-3 at u = 100
    assert encircled_energy(1000.0) == pytest.approx(1.0, abs=1e-3)
    assert encircled_energy(100.0) == pytest.approx(1.0, abs=1e-2)
    # 83.8% of the energy is inside the first dark ring
    assert encircled_energy(3.831705970207512) == pytest.approx(0.838, rel=2e-3)


def test_strehl_ratio_is_one_for_a_perfect_wavefront():
    assert strehl_ratio(q(0.0, "nm"), q(LAMBDA_NM, "nm")) == pytest.approx(1.0)


def test_marechal_approximation_matches_the_closed_form():
    rms = 30.0
    expected = np.exp(-((2.0 * np.pi * rms / LAMBDA_NM) ** 2))
    assert strehl_ratio(q(rms, "nm"), q(LAMBDA_NM, "nm")) == pytest.approx(expected)


def test_rms_wavefront_from_strehl_inverts_the_relation():
    for rms_nm in (10.0, 25.0, 50.0):
        strehl = strehl_ratio(q(rms_nm, "nm"), q(LAMBDA_NM, "nm"))
        recovered = rms_wavefront_from_strehl(strehl, q(LAMBDA_NM, "nm"))
        assert recovered.to_value("nm") == pytest.approx(rms_nm, rel=1e-12)


def test_seventy_one_percent_strehl_corresponds_to_a_quarter_wave():
    # the classical Maréchal criterion: Strehl > 0.8 at lambda/14 rms
    at_lambda_over_fourteen = strehl_ratio(
        q(LAMBDA_NM / 14.0, "nm"), q(LAMBDA_NM, "nm")
    )
    assert at_lambda_over_fourteen == pytest.approx(0.82, rel=0.02)


def test_diffraction_limited_spot_and_far_field_angle():
    from optcon.diffraction import (
        diffraction_limited_spot,
        gaussian_far_field_half_angle,
    )

    spot = diffraction_limited_spot(q(500.0, "nm"), 4.0)
    assert spot.to_value("um") == pytest.approx(2.44 * 0.5 * 4.0)

    div = gaussian_far_field_half_angle(q(1.0, "mm"), q(1064.0, "nm"))
    assert div.to_value("rad") == pytest.approx(1064e-9 / (np.pi * 1e-3))


def test_diffraction_errors_and_boundaries():
    from optcon import DimensionError

    # Bare number instead of length quantity
    with pytest.raises(TypeError, match="expected a length with units"):
        f_number(100.0, 10.0)

    # Incompatible dimension
    with pytest.raises(DimensionError):
        f_number(q(100.0, "s"), q(10.0, "mm"))

    # Negative diameter or f-number
    with pytest.raises(ValueError, match="aperture diameter must be positive"):
        f_number(q(100.0, "mm"), q(-10.0, "mm"))

    with pytest.raises(ValueError, match="f-number must be positive"):
        airy_radius(q(500.0, "nm"), -2.0)

    with pytest.raises(ValueError, match="cannot be negative"):
        encircled_energy(-1.0)

    with pytest.raises(ValueError, match="Strehl ratio must lie in"):
        rms_wavefront_from_strehl(1.5, q(500.0, "nm"))

    with pytest.raises(ValueError, match="Strehl ratio must lie in"):
        rms_wavefront_from_strehl(0.0, q(500.0, "nm"))

