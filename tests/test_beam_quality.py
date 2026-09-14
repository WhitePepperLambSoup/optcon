"""Beam quality: M squared and the beam parameter product.

An ideal Gaussian has ``M^2 = 1``; every real laser has ``M^2 > 1``.  The
relations used here are those of ISO 11146.
"""

import numpy as np
import pytest

from optcon import q
from optcon.beam_quality import (
    beam_parameter_product,
    beam_radius_with_m_squared,
    m_squared_from_parameters,
    m_squared_from_widths,
    rayleigh_range_with_m_squared,
)
from optcon.gaussian import rayleigh_range

WAIST_MM = 0.4
LAMBDA_MM = 1.064e-3


def test_an_ideal_gaussian_has_m_squared_one():
    divergence_rad = LAMBDA_MM / (np.pi * WAIST_MM)
    assert m_squared_from_parameters(
        q(WAIST_MM, "mm"), q(divergence_rad, "rad"), q(LAMBDA_MM, "mm")
    ) == pytest.approx(1.0)


def test_m_squared_from_two_measured_widths():
    # build the width from the M^2 = 2 hyperbola itself, then invert it; the
    # beam is only twice the ideal width in the far field, so the width has to
    # come from the formula rather than from a factor of two
    z = q(1000.0, "mm")
    rayleigh = np.pi * WAIST_MM**2 / LAMBDA_MM
    measured = WAIST_MM * np.sqrt(1.0 + (2.0 * z.value / rayleigh) ** 2)
    assert m_squared_from_widths(
        q(WAIST_MM, "mm"),
        q(float(measured), "mm"),
        z,
        q(LAMBDA_MM, "mm"),
    ) == pytest.approx(2.0, rel=1e-9)


def test_beam_parameter_product_is_m_squared_lambda_over_pi():
    bpp = beam_parameter_product(1.5, q(LAMBDA_MM, "mm"))
    # an angle is a dimension of its own here, so the product keeps it
    assert bpp.unit.dimension == (q(1.0, "mm") * q(1.0, "rad")).unit.dimension
    assert bpp.to_value("mm*rad") == pytest.approx(1.5 * LAMBDA_MM / np.pi)
    # which makes the customary mm mrad figure expressible
    assert bpp.to_value("mm*mrad") == pytest.approx(1.5 * LAMBDA_MM / np.pi * 1000.0)


def test_rayleigh_range_scales_inversely_with_m_squared():
    ideal = rayleigh_range(q(WAIST_MM, "mm"), q(LAMBDA_MM, "mm")).value
    non_ideal = rayleigh_range_with_m_squared(
        q(WAIST_MM, "mm"), q(LAMBDA_MM, "mm"), 2.0
    ).value
    assert non_ideal == pytest.approx(ideal / 2.0)


def test_beam_radius_with_m_squared_grows_faster_than_ideal():
    ideal = beam_radius_with_m_squared(
        q(WAIST_MM, "mm"), q(LAMBDA_MM, "mm"), q(2000.0, "mm"), 1.0
    )
    real = beam_radius_with_m_squared(
        q(WAIST_MM, "mm"), q(LAMBDA_MM, "mm"), q(2000.0, "mm"), 3.0
    )
    assert real.value > ideal.value


def test_a_sub_gaussian_m_squared_is_rejected():
    with pytest.raises(ValueError, match="at least one"):
        beam_parameter_product(0.5, q(LAMBDA_MM, "mm"))
