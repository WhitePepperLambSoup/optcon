"""Gaussian beam propagation through the complex q parameter.

Everything here is checked against closed-form relations from Siegman,
*Lasers* (1986), ch. 17, so the tests are independent of the implementation:
``w(z)``, ``R(z)``, the Gouy phase, and the mode-overlap integral.
"""

import numpy as np
import pytest

from optcon import DimensionError, q
from optcon.elements import compose, free_space, thin_lens
from optcon.gaussian import (
    beam_radius,
    divergence_half_angle,
    gouy_phase,
    mode_matching_efficiency,
    mode_overlap,
    position_of_waist,
    q_from_waist,
    radius_of_curvature,
    rayleigh_range,
    self_consistent_mode,
    waist_of,
)
from optcon.gaussian import propagate as propagate_beam

WAIST_MM = 0.5
LAMBDA_MM = 1.064e-3
RAYLEIGH_MM = np.pi * WAIST_MM**2 / LAMBDA_MM


def test_rayleigh_range_matches_pi_w0_squared_over_lambda():
    assert rayleigh_range(q(WAIST_MM, "mm"), q(LAMBDA_MM, "mm")).value == pytest.approx(
        RAYLEIGH_MM
    )


def test_divergence_is_lambda_over_pi_w0():
    half_angle = divergence_half_angle(q(WAIST_MM, "mm"), q(LAMBDA_MM, "mm"))
    assert half_angle.to_value("rad") == pytest.approx(LAMBDA_MM / (np.pi * WAIST_MM))
    # equivalently, the far-field half angle is w0 / zR
    assert half_angle.to_value("rad") == pytest.approx(WAIST_MM / RAYLEIGH_MM)


def test_q_parameter_at_the_waist_is_purely_imaginary():
    qp = q_from_waist(q(WAIST_MM, "mm"), q(LAMBDA_MM, "mm"))
    assert qp.value.real == pytest.approx(0.0, abs=1e-12)
    assert qp.value.imag == pytest.approx(RAYLEIGH_MM)
    assert qp.unit.symbol == "mm"


def test_beam_radius_follows_the_hyperbolic_law():
    qp = q_from_waist(q(WAIST_MM, "mm"), q(LAMBDA_MM, "mm"))
    wavelength = q(LAMBDA_MM, "mm")
    for ratio in (0.0, 0.5, 1.0, 2.0, 5.0):
        distance = ratio * RAYLEIGH_MM
        propagated = propagate_beam(qp, free_space(q(distance, "mm")))
        expected = WAIST_MM * np.sqrt(1.0 + ratio**2)
        assert beam_radius(propagated, wavelength).to_value("mm") == pytest.approx(expected)


def test_radius_of_curvature_follows_the_closed_form():
    qp = q_from_waist(q(WAIST_MM, "mm"), q(LAMBDA_MM, "mm"))
    wavelength = q(LAMBDA_MM, "mm")
    # at the waist the wavefront is flat
    assert abs(radius_of_curvature(qp, wavelength).to_value("mm")) > 1e15
    for ratio in (0.5, 1.0, 3.0):
        distance = ratio * RAYLEIGH_MM
        propagated = propagate_beam(qp, free_space(q(distance, "mm")))
        expected = distance * (1.0 + 1.0 / ratio**2)
        assert radius_of_curvature(propagated, wavelength).to_value("mm") == pytest.approx(
            expected
        )


def test_gouy_phase_accumulates_from_zero_to_pi_over_two():
    qp = q_from_waist(q(WAIST_MM, "mm"), q(LAMBDA_MM, "mm"))
    wavelength = q(LAMBDA_MM, "mm")
    assert gouy_phase(qp, wavelength).to_value("rad") == pytest.approx(0.0)
    at_rayleigh = propagate_beam(qp, free_space(q(RAYLEIGH_MM, "mm")))
    assert gouy_phase(at_rayleigh, wavelength).to_value("rad") == pytest.approx(np.pi / 4)


def test_propagation_recovers_the_waist_size():
    qp = q_from_waist(q(WAIST_MM, "mm"), q(LAMBDA_MM, "mm"))
    wavelength = q(LAMBDA_MM, "mm")
    moved = propagate_beam(qp, free_space(q(10.0 * RAYLEIGH_MM, "mm")))
    assert waist_of(moved, wavelength).to_value("mm") == pytest.approx(WAIST_MM)
    assert position_of_waist(moved, wavelength).to_value("mm") == pytest.approx(
        -10.0 * RAYLEIGH_MM
    )


def test_a_thin_lens_focuses_a_collimated_beam_to_lambda_f_over_pi_w():
    collimated = q_from_waist(
        q(1.0, "mm"), q(LAMBDA_MM, "mm")
    )  # large waist -> nearly collimated at the lens
    focal_length_mm = 200.0
    lens = thin_lens(q(focal_length_mm, "mm"))
    focused = propagate_beam(collimated, lens)
    wavelength = q(LAMBDA_MM, "mm")
    expected_waist = LAMBDA_MM * focal_length_mm / (np.pi * 1.0)
    assert waist_of(focused, wavelength).to_value("mm") == pytest.approx(
        expected_waist, rel=5e-3
    )


def test_a_full_cavity_round_trip_has_a_self_consistent_mode():
    length = q(300.0, "mm")
    radius = q(500.0, "mm")
    from optcon.elements import curved_mirror

    round_trip = compose(
        free_space(length),
        curved_mirror(radius),
        free_space(length),
        curved_mirror(radius),
    )
    wavelength = q(LAMBDA_MM, "mm")
    mode = self_consistent_mode(round_trip, wavelength)
    # the defining property: the mode maps onto itself after one round trip
    after = propagate_beam(mode, round_trip)
    assert after.value == pytest.approx(mode.value, rel=1e-9)


def test_mode_overlap_reproduces_the_coaxial_coupling_formula():
    assert mode_overlap(1.0, 1.0) == pytest.approx(1.0)
    w1, w2 = 3.0, 5.0
    assert mode_overlap(w1, w2) == pytest.approx((2 * w1 * w2 / (w1**2 + w2**2)) ** 2)


def test_the_general_overlap_formula_agrees_with_the_simple_one():
    """Two routes to the same number: the q-parameter form and the w form."""
    wavelength = q(LAMBDA_MM, "mm")
    q1 = q_from_waist(q(3.0, "mm"), wavelength)
    q2 = q_from_waist(q(5.0, "mm"), wavelength)
    assert mode_matching_efficiency(q1, q2, wavelength) == pytest.approx(
        mode_overlap(3.0, 5.0), rel=1e-12
    )


def test_an_offset_waist_reduces_the_overlap():
    wavelength = q(LAMBDA_MM, "mm")
    q1 = q_from_waist(q(3.0, "mm"), wavelength)
    q2 = propagate_beam(q_from_waist(q(3.0, "mm"), wavelength), free_space(q(5.0, "mm")))
    assert mode_matching_efficiency(q1, q2, wavelength) < 1.0


def test_waist_and_wavelength_units_are_checked():
    with pytest.raises(DimensionError):
        rayleigh_range(q(1.0, "mm"), q(1.0, "s"))
