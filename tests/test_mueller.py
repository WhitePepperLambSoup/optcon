"""Mueller calculus: the intensity-domain counterpart of Jones calculus.

Jones vectors describe fully polarised light; Stokes vectors and Mueller
matrices also describe partial polarisation and depolarising elements.  The
two must agree wherever both apply, and that agreement is the main test here:
every Mueller matrix derived from a Jones matrix must reproduce the Jones
result on an independent set of states.
"""

import numpy as np
import pytest

from optcon import q
from optcon.mueller import (
    apply_mueller,
    degree_of_polarization,
    mueller_depolarizer,
    mueller_from_jones,
    mueller_polarizer,
    mueller_retarder,
    mueller_rotator,
    stokes_vector,
)
from optcon.polarization import (
    half_wave_plate,
    jones_circular,
    jones_linear,
    polarizer,
    quarter_wave_plate,
    retarder,
    rotator,
)
from optcon.polarization import (
    stokes as jones_stokes,
)


def test_the_stokes_vector_agrees_with_the_jones_module():
    for state in (
        jones_linear(q(0.0, "deg")),
        jones_linear(q(37.0, "deg")),
        jones_circular("right"),
        jones_circular("left"),
    ):
        mine = stokes_vector(state)
        theirs = jones_stokes(state)
        assert mine[0] == pytest.approx(theirs["S0"])
        assert mine[1] == pytest.approx(theirs["S1"])
        assert mine[2] == pytest.approx(theirs["S2"])
        assert mine[3] == pytest.approx(theirs["S3"])


def test_a_mueller_matrix_reproduces_the_jones_result():
    """The cross-check that ties the two calculi together."""
    elements = (
        polarizer(q(30.0, "deg")),
        half_wave_plate(q(22.5, "deg")),
        quarter_wave_plate(q(45.0, "deg")),
        retarder(q(73.0, "deg"), q(11.0, "deg")),
        rotator(q(31.0, "deg")),
    )
    probes = (
        jones_linear(q(0.0, "deg")),
        jones_linear(q(55.0, "deg")),
        jones_circular("right"),
    )
    for element in elements:
        matrix = mueller_from_jones(element.value)
        for probe in probes:
            from_jones = stokes_vector(element.value @ probe.value)
            from_mueller = apply_mueller(matrix, stokes_vector(probe))
            assert from_mueller == pytest.approx(from_jones, abs=1e-12)


def test_an_ideal_polarizer_halves_unpolarized_light():
    unpolarized = np.array([1.0, 0.0, 0.0, 0.0])
    transmitted = apply_mueller(mueller_polarizer(q(0.0, "deg")), unpolarized)
    assert transmitted[0] == pytest.approx(0.5)
    assert transmitted[1] == pytest.approx(0.5)
    assert transmitted[2] == pytest.approx(0.0, abs=1e-15)


def test_malus_law_through_two_polarizers():
    for angle_deg in (0.0, 30.0, 45.0, 90.0):
        first = mueller_polarizer(q(0.0, "deg"))
        second = mueller_polarizer(q(angle_deg, "deg"))
        state = apply_mueller(second, apply_mueller(first, np.array([1.0, 0.0, 0.0, 0.0])))
        # unpolarized input loses half at the first polariser, then cos^2 at
        # the second
        assert state[0] == pytest.approx(0.5 * np.cos(np.deg2rad(angle_deg)) ** 2)


def test_lossless_elements_preserve_the_total_intensity():
    for matrix in (
        mueller_retarder(q(90.0, "deg"), q(45.0, "deg")),
        mueller_rotator(q(20.0, "deg")),
        mueller_from_jones(half_wave_plate(q(30.0, "deg")).value),
    ):
        for state in (
            np.array([1.0, 1.0, 0.0, 0.0]),
            np.array([1.0, 0.0, 1.0, 0.0]),
            np.array([1.0, 0.0, 0.0, 1.0]),
        ):
            out = apply_mueller(matrix, state)
            assert out[0] == pytest.approx(state[0], rel=1e-12)


def test_a_retarder_rotates_the_stokes_vector_on_the_poincare_sphere():
    # a quarter-wave plate at 45 degrees takes horizontal to circular
    state = np.array([1.0, 1.0, 0.0, 0.0])
    out = apply_mueller(mueller_retarder(q(90.0, "deg"), q(45.0, "deg")), state)
    assert out[1] == pytest.approx(0.0, abs=1e-12)
    assert abs(out[3]) == pytest.approx(1.0, abs=1e-12)


def test_a_depolarizer_reduces_the_degree_of_polarization():
    fully_polarized = np.array([1.0, 1.0, 0.0, 0.0])
    assert degree_of_polarization(fully_polarized) == pytest.approx(1.0)
    partially = apply_mueller(mueller_depolarizer(0.4), fully_polarized)
    assert 0.0 < degree_of_polarization(partially) < 1.0
    # the argument is the fraction of polarization *retained*
    assert degree_of_polarization(partially) == pytest.approx(0.4)


def test_a_depolarizer_leaves_unpolarized_light_unpolarized():
    state = np.array([1.0, 0.0, 0.0, 0.0])
    out = apply_mueller(mueller_depolarizer(0.7), state)
    assert degree_of_polarization(out) == pytest.approx(0.0)
    assert out[0] == pytest.approx(1.0)


def test_the_mueller_matrix_of_a_polarizer_is_idempotent():
    matrix = mueller_polarizer(q(25.0, "deg"))
    assert matrix @ matrix == pytest.approx(matrix, abs=1e-12)


def test_an_unknown_jones_shape_is_rejected():
    with pytest.raises(ValueError, match="2x2"):
        mueller_from_jones(np.eye(3))
