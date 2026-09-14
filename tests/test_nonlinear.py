"""Nonlinear optics: second-harmonic phase matching and effective length.

The relations are the standard ones from Boyd, *Nonlinear Optics*, ch. 2.
The coherence length, the phase mismatch and the effective interaction
length are three statements of the same fact, so the tests check them against
each other rather than against a table.
"""

import numpy as np
import pytest

from optcon import DimensionError, q
from optcon.nonlinear import (
    coherence_length,
    effective_interaction_length,
    phase_mismatch,
    quasi_phase_matching_period,
)

LAMBDA_NM = 1064.0
DISPERSION = 0.01  # n(2w) - n(w) for a typical birefringently matched crystal


def test_phase_mismatch_is_four_pi_over_lambda_times_the_index_difference():
    delta_k = phase_mismatch(q(LAMBDA_NM, "nm"), 1.51, 1.50)
    assert delta_k.to_value("1/mm") == pytest.approx(
        4.0 * np.pi * DISPERSION / (LAMBDA_NM * 1e-6)
    )


def test_coherence_length_is_lambda_over_four_delta_n():
    length = coherence_length(q(LAMBDA_NM, "nm"), 1.51, 1.50)
    assert length.to_value("um") == pytest.approx(
        LAMBDA_NM * 1e-3 / (4.0 * DISPERSION)
    )


def test_the_coherence_length_inverts_the_phase_mismatch():
    """pi / delta-k must reproduce the coherence length exactly."""
    delta_k = phase_mismatch(q(LAMBDA_NM, "nm"), 1.51, 1.50)
    length = coherence_length(q(LAMBDA_NM, "nm"), 1.51, 1.50)
    assert np.pi / delta_k.to_value("1/mm") == pytest.approx(length.to_value("mm"))


def test_quasi_phase_matching_period_is_twice_the_coherence_length():
    period = quasi_phase_matching_period(q(LAMBDA_NM, "nm"), 1.51, 1.50)
    length = coherence_length(q(LAMBDA_NM, "nm"), 1.51, 1.50)
    assert period.to_value("um") == pytest.approx(2.0 * length.to_value("um"))
    # for this case the poling period is the familiar few tens of microns
    assert period.to_value("um") == pytest.approx(53.2, rel=1e-2)


def test_perfect_phase_matching_has_no_coherence_limit():
    assert phase_mismatch(q(LAMBDA_NM, "nm"), 1.5, 1.5).to_value("1/mm") == pytest.approx(0.0)
    assert np.isinf(coherence_length(q(LAMBDA_NM, "nm"), 1.5, 1.5).to_value("mm"))
    assert np.isinf(quasi_phase_matching_period(q(LAMBDA_NM, "nm"), 1.5, 1.5).to_value("mm"))


def test_the_effective_length_equals_the_crystal_length_when_matched():
    value = effective_interaction_length(q(10.0, "mm"), q(0.0, "1/mm"))
    assert value.to_value("mm") == pytest.approx(10.0)


def test_the_effective_length_vanishes_after_one_full_phase_slip():
    """delta-k L = 2 pi, i.e. a crystal two coherence lengths long."""
    delta_k = phase_mismatch(q(LAMBDA_NM, "nm"), 1.51, 1.50).to_value("1/mm")
    two_lengths = 2.0 * np.pi / delta_k
    value = effective_interaction_length(q(two_lengths, "mm"), q(delta_k, "1/mm"))
    assert value.to_value("mm") == pytest.approx(0.0, abs=1e-9)


def test_the_effective_length_is_bounded_by_the_coherence_length():
    delta_k = phase_mismatch(q(LAMBDA_NM, "nm"), 1.51, 1.50).to_value("1/mm")
    coherence = np.pi / delta_k
    for multiple in (0.5, 1.0, 5.0, 50.0):
        value = effective_interaction_length(
            q(multiple * 2.0 * coherence, "mm"), q(delta_k, "1/mm")
        ).to_value("mm")
        assert abs(value) <= 2.0 * coherence + 1e-12


def test_a_dimensioned_index_is_rejected():
    with pytest.raises((DimensionError, ValueError)):
        coherence_length(q(LAMBDA_NM, "nm"), q(1.51, "mm"), 1.50)
