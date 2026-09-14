"""Jones calculus, with the invariant contracts applied to it.

This module is where the type layer earns its keep on polarisation: a
waveplate is unitary, a polariser is passive but not unitary, and a rotator is
unitary but not reciprocal.  Each of those statements is a checkable property
rather than a footnote.

Conventions follow Jones (1941) and Hecht, *Optics*, ch. 8: states are column
vectors ``(Ex, Ey)`` in an amplitude basis, operators act from the left, and a
retardance ``delta`` means the slow axis is delayed by ``delta``.
"""

import numpy as np
import pytest

from optcon import (
    ContractViolation,
    DimensionError,
    assert_passive,
    assert_reciprocal,
    assert_unitary,
    q,
)
from optcon.polarization import (
    apply,
    degree_of_polarization,
    extinction_ratio,
    half_wave_plate,
    intensity,
    jones_circular,
    jones_linear,
    polarizer,
    quarter_wave_plate,
    retarder,
    rotator,
    stokes,
)


def test_linear_state_is_a_unit_amplitude_vector():
    state = jones_linear(q(0.0, "deg"))
    assert state.value == pytest.approx(np.array([1.0, 0.0]))
    assert state.amp_order == 1


def test_circular_states_are_equal_weight_with_quarter_wave_phase():
    right = jones_circular("right")
    left = jones_circular("left")
    assert abs(right.value[0]) == pytest.approx(abs(right.value[1]))
    assert right.value[1].imag == pytest.approx(-right.value[0].real)
    assert left.value[1].imag == pytest.approx(+left.value[0].real)


def test_malus_law():
    for analyzer_deg in (0.0, 30.0, 45.0, 60.0, 90.0):
        analyzer = polarizer(q(analyzer_deg, "deg"))
        transmitted = apply(analyzer, jones_linear(q(0.0, "deg")))
        expected = np.cos(np.deg2rad(analyzer_deg)) ** 2
        assert intensity(transmitted).value == pytest.approx(expected, abs=1e-15)


def test_crossed_polarizers_extinguish():
    crossed = apply(
        polarizer(q(90.0, "deg")),
        apply(polarizer(q(0.0, "deg")), jones_linear(q(0.0, "deg"))),
    )
    assert intensity(crossed).value == pytest.approx(0.0, abs=1e-18)


def test_a_polarizer_is_passive_and_idempotent_but_not_unitary():
    matrix = polarizer(q(37.0, "deg"))
    assert_passive(matrix.value, name="polarizer")
    assert matrix.value @ matrix.value == pytest.approx(matrix.value)
    with pytest.raises(ContractViolation):
        assert_unitary(matrix.value, name="polarizer")


def test_waveplates_and_rotators_are_unitary():
    for element in (
        half_wave_plate(q(22.5, "deg")),
        quarter_wave_plate(q(45.0, "deg")),
        retarder(q(73.0, "deg"), q(11.0, "deg")),
        rotator(q(31.0, "deg")),
    ):
        assert_unitary(element.value, name=element.unit.symbol or "element")


def test_a_rotator_is_unitary_but_not_reciprocal():
    matrix = rotator(q(45.0, "deg")).value
    assert_unitary(matrix)
    with pytest.raises(ContractViolation):
        assert_reciprocal(matrix, name="rotator")


def test_a_half_wave_plate_at_45_degrees_rotates_linear_light_by_90():
    rotated = apply(half_wave_plate(q(45.0, "deg")), jones_linear(q(0.0, "deg")))
    assert abs(rotated.value[0]) == pytest.approx(0.0, abs=1e-12)
    assert abs(rotated.value[1]) == pytest.approx(1.0)


def test_a_quarter_wave_plate_at_45_degrees_makes_circular_light():
    circular = apply(quarter_wave_plate(q(45.0, "deg")), jones_linear(q(0.0, "deg")))
    assert stokes(circular)["S3"] == pytest.approx(1.0, abs=1e-12)
    assert degree_of_polarization(circular) == pytest.approx(1.0)


def test_a_rotator_rotates_linear_polarisation():
    rotated = apply(rotator(q(30.0, "deg")), jones_linear(q(10.0, "deg")))
    assert stokes(rotated)["S1"] == pytest.approx(np.cos(np.deg2rad(80.0)), abs=1e-12)
    assert stokes(rotated)["S2"] == pytest.approx(np.sin(np.deg2rad(80.0)), abs=1e-12)


def test_unitary_elements_conserve_intensity():
    state = jones_circular("right")
    for element in (half_wave_plate(q(17.0, "deg")), retarder(q(55.0, "deg"), q(80.0, "deg"))):
        assert intensity(apply(element, state)).value == pytest.approx(
            intensity(state).value, rel=1e-12
        )


def test_stokes_parameters_of_linear_and_circular_states():
    linear = stokes(jones_linear(q(30.0, "deg")))
    assert linear["S0"] == pytest.approx(1.0)
    assert linear["S1"] == pytest.approx(np.cos(np.deg2rad(60.0)), abs=1e-12)
    assert linear["S2"] == pytest.approx(np.sin(np.deg2rad(60.0)), abs=1e-12)
    assert linear["S3"] == pytest.approx(0.0, abs=1e-12)

    circular = stokes(jones_circular("left"))
    assert circular["S1"] == pytest.approx(0.0, abs=1e-12)
    assert circular["S2"] == pytest.approx(0.0, abs=1e-12)
    assert abs(circular["S3"]) == pytest.approx(1.0)


def test_degree_of_polarization_is_one_for_every_pure_state():
    for state in (jones_linear(q(0.0, "deg")), jones_circular("right")):
        assert degree_of_polarization(state) == pytest.approx(1.0)


def test_extinction_ratio_of_a_polarizer():
    # perfectly crossed ideal polarisers transmit nothing, so the ratio is
    # infinite; the helper reports inf rather than dividing by zero
    assert extinction_ratio(
        polarizer(q(0.0, "deg")), polarizer(q(90.0, "deg"))
    ) == pytest.approx(np.inf)
    # one degree of misalignment leaks cos^2(1) sin^2(1) ~ 3e-4, i.e. a ratio
    # of a few thousand - which is exactly the sensitivity an alignment
    # engineer wants quantified
    finite = extinction_ratio(polarizer(q(1.0, "deg")), polarizer(q(90.0, "deg")))
    assert 1e3 < finite < 1e4


def test_an_angle_without_units_is_rejected():
    with pytest.raises(TypeError, match="angle"):
        polarizer(30.0)


def test_a_dimensioned_state_vector_is_rejected():
    with pytest.raises(DimensionError):
        intensity(q(np.array([1.0, 0.0]), "mm"))
