"""Physical invariant contracts.

These are properties that a numerical optical model either has or does not
have, and that current libraries leave entirely to the author's memory.
Test data is a small self-contained Jones calculus, written here so the
library stays independent of any external project code.
"""

import numpy as np
import pytest

from optcon import ContractViolation, DimensionError, dimensionless, q
from optcon.checks import (
    assert_passive,
    assert_reciprocal,
    assert_unitary,
    gain_above_unity,
    is_passive,
    is_reciprocal,
    is_unitary,
    singular_values,
)


def rotation(degrees: float) -> np.ndarray:
    theta = np.deg2rad(degrees)
    return np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]], dtype=complex
    )


def half_wave_plate(degrees: float) -> np.ndarray:
    theta = np.deg2rad(degrees)
    return np.array(
        [[np.cos(2 * theta), np.sin(2 * theta)], [np.sin(2 * theta), -np.cos(2 * theta)]],
        dtype=complex,
    )


def test_half_wave_plate_is_unitary():
    assert is_unitary(half_wave_plate(22.5))


def test_half_wave_plate_is_reciprocal():
    assert is_reciprocal(half_wave_plate(22.5))


def test_faraday_rotator_is_unitary_but_not_reciprocal():
    rotator = rotation(45.0)
    assert is_unitary(rotator)
    assert not is_reciprocal(rotator)
    with pytest.raises(ContractViolation, match="declared reciprocal"):
        assert_reciprocal(rotator, name="Faraday rotator")


def test_lossy_element_is_passive_but_not_unitary():
    lossy = np.diag([0.7, 0.7]).astype(complex)
    assert is_passive(lossy)
    assert not is_unitary(lossy)


def test_amplifying_element_is_rejected_by_assert_passive():
    amplifier = np.diag([1.0, 1.5]).astype(complex)
    assert not is_passive(amplifier)
    with pytest.raises(ContractViolation):
        assert_passive(amplifier, name="gain element")


def test_assert_unitary_names_the_violated_contract():
    with pytest.raises(ContractViolation, match="unitary"):
        assert_unitary(np.diag([1.0, 1.5]).astype(complex), name="bad waveplate")


def test_assert_reciprocal_rejects_a_non_symmetric_matrix():
    with pytest.raises(ContractViolation, match="reciprocal"):
        assert_reciprocal(rotation(45.0), name="faraday rotator")


def test_singular_values_of_a_lossy_element():
    lossy = np.diag([0.7, 0.7]).astype(complex)
    assert list(singular_values(lossy)) == pytest.approx([0.7, 0.7])


def test_gain_above_unity_is_zero_for_a_passive_element():
    assert gain_above_unity(np.diag([0.7, 0.7]).astype(complex)) == pytest.approx(0.0)


def test_gain_above_unity_is_positive_for_an_amplifier():
    assert gain_above_unity(np.diag([1.0, 1.5]).astype(complex)) == pytest.approx(0.5)


def test_unitary_check_accepts_dimensionless_quantities():
    assert is_unitary(dimensionless(half_wave_plate(22.5)))


def test_dimensioned_matrix_is_rejected():
    with pytest.raises(DimensionError):
        is_unitary(q(np.eye(2), "mm"))
