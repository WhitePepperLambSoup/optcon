"""Diffraction gratings: the grating equation, dispersion and resolution.

Relations follow Hecht, *Optics*, ch. 10, and Palmer, *Diffraction Grating
Handbook*, for the Littrow configuration and the blaze condition.
"""

import numpy as np
import pytest

from optcon import DimensionError, q
from optcon.grating import (
    angular_dispersion,
    blaze_angle,
    free_spectral_range_of_grating,
    grating_equation,
    lines_per_mm_from_period,
    period_from_lines_per_mm,
    resolving_power,
    wavelength_resolution,
)

LAMBDA_NM = 633.0
LINES_PER_MM = 1200.0


def test_period_and_line_density_are_reciprocal():
    period = period_from_lines_per_mm(LINES_PER_MM)
    assert period.to_value("nm") == pytest.approx(1e6 / LINES_PER_MM)
    assert lines_per_mm_from_period(period) == pytest.approx(LINES_PER_MM)


def test_first_order_angle_at_normal_incidence():
    period = period_from_lines_per_mm(LINES_PER_MM)
    angle = grating_equation(
        q(LAMBDA_NM, "nm"), period, order=1, incidence=q(0.0, "deg")
    )
    expected = np.arcsin(LAMBDA_NM / period.to_value("nm"))
    assert angle.to_value("rad") == pytest.approx(expected)


def test_the_grating_equation_is_satisfied_for_a_range_of_angles():
    period = period_from_lines_per_mm(LINES_PER_MM)
    # kept inside the range where order +1 still propagates for this period:
    # sin(theta_m) = lambda/d - sin(theta_i) must stay below one
    for incidence_deg in (0.0, 10.0, 25.0, -10.0):
        order = 1
        angle = grating_equation(
            q(LAMBDA_NM, "nm"), period, order=order, incidence=q(incidence_deg, "deg")
        )
        lhs = order * LAMBDA_NM
        rhs = period.to_value("nm") * (
            np.sin(np.deg2rad(incidence_deg)) + np.sin(angle.to_value("rad"))
        )
        assert lhs == pytest.approx(rhs)


def test_the_littrow_configuration_folds_the_equation_in_half():
    period = period_from_lines_per_mm(LINES_PER_MM)
    littrow = blaze_angle(q(LAMBDA_NM, "nm"), period, order=1)
    # in Littrow the incidence and diffraction angles coincide
    diffracted = grating_equation(
        q(LAMBDA_NM, "nm"), period, order=1, incidence=littrow
    )
    assert diffracted.to_value("rad") == pytest.approx(littrow.to_value("rad"))
    assert np.sin(littrow.to_value("rad")) == pytest.approx(
        LAMBDA_NM / (2.0 * period.to_value("nm"))
    )


def test_angular_dispersion_matches_the_closed_form():
    period = period_from_lines_per_mm(LINES_PER_MM)
    incidence = q(20.0, "deg")
    angle = grating_equation(q(LAMBDA_NM, "nm"), period, order=1, incidence=incidence)
    dispersion = angular_dispersion(
        q(LAMBDA_NM, "nm"), period, order=1, incidence=incidence
    )
    expected = 1.0 / (period.to_value("nm") * np.cos(angle.to_value("rad")))
    assert dispersion.to_value("rad/nm") == pytest.approx(expected)


def test_resolving_power_is_order_times_the_number_of_lines():
    assert resolving_power(order=2, illuminated_lines=50000.0) == pytest.approx(100000.0)
    # and equivalently lambda / delta-lambda, so the resolvable difference is
    # lambda / R
    resolving = resolving_power(order=1, illuminated_lines=10000.0)
    assert resolving == pytest.approx(10000.0)
    assert wavelength_resolution(1, 10000.0, q(LAMBDA_NM, "nm")).to_value(
        "nm"
    ) == pytest.approx(LAMBDA_NM / resolving)


def test_free_spectral_range_is_the_orders_apart_spacing():
    fsr = free_spectral_range_of_grating(q(LAMBDA_NM, "nm"), order=1)
    assert fsr.to_value("nm") == pytest.approx(LAMBDA_NM)


def test_an_order_that_cannot_exist_is_rejected():
    # a *fine* grating is the one that loses its orders: at 2000 lines/mm the
    # period is 500 nm, shorter than the 633 nm wavelength, so even the first
    # order would need sin(theta) > 1
    fine = period_from_lines_per_mm(2000.0)
    with pytest.raises(ValueError, match="no diffracted order"):
        grating_equation(q(LAMBDA_NM, "nm"), fine, order=1, incidence=q(0.0, "deg"))


def test_a_dimensioned_line_density_is_rejected():
    with pytest.raises((DimensionError, TypeError, ValueError)):
        # line density is a count per millimetre, so handing it a length is a
        # mistake the API should refuse rather than silently accept
        period_from_lines_per_mm(q(1200.0, "mm"))  # type: ignore[arg-type]
