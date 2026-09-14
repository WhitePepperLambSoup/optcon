"""Step-index optical fibre: V number, numerical aperture, mode field, coupling.

The relations are the standard ones from Okamoto, *Fundamentals of Optical
Waveguides*, ch. 3, with Marcuse's closed-form approximation for the mode
field diameter.  The coupling helper reuses the Gaussian mode-overlap result
from :mod:`optcon.gaussian`, which is how these two modules compose.
"""

import numpy as np
import pytest

from optcon import DimensionError, q
from optcon.fiber import (
    acceptance_angle,
    coupling_loss_db,
    cutoff_wavelength,
    is_single_mode,
    mode_field_diameter,
    numerical_aperture,
    propagation_constant,
    v_number,
)
from optcon.gaussian import mode_overlap

CORE_RADIUS_UM = 4.1
N_CORE = 1.4500
N_CLAD = 1.4468
WAVELENGTH_NM = 1550.0


def test_numerical_aperture_from_indices():
    na = numerical_aperture(N_CORE, N_CLAD)
    assert na == pytest.approx(np.sqrt(N_CORE**2 - N_CLAD**2))


def test_acceptance_angle_is_the_arcsine_of_the_numerical_aperture():
    na = numerical_aperture(N_CORE, N_CLAD)
    assert acceptance_angle(N_CORE, N_CLAD).to_value("deg") == pytest.approx(
        np.rad2deg(np.arcsin(na))
    )


def test_v_number_of_a_standard_single_mode_fibre():
    na = numerical_aperture(N_CORE, N_CLAD)
    v = v_number(q(CORE_RADIUS_UM, "um"), na, q(WAVELENGTH_NM, "nm"))
    # both lengths in metres, which is what the helper must do internally
    expected = (
        2.0 * np.pi * (CORE_RADIUS_UM * 1e-6) * na / (WAVELENGTH_NM * 1e-9)
    )
    assert v == pytest.approx(expected)
    # these indices give NA ~ 0.096 and hence V ~ 1.60: single mode, with the
    # cutoff a comfortable margin away
    assert v == pytest.approx(1.6002, rel=1e-3)
    assert is_single_mode(v)


def test_a_large_v_number_is_multimode():
    assert not is_single_mode(10.0)


def test_cutoff_wavelength_matches_the_v_number_condition():
    na = numerical_aperture(N_CORE, N_CLAD)
    cutoff = cutoff_wavelength(q(CORE_RADIUS_UM, "um"), na)
    v_at_cutoff = v_number(q(CORE_RADIUS_UM, "um"), na, cutoff)
    assert v_at_cutoff == pytest.approx(2.404825557695773, rel=1e-12)


def test_mode_field_diameter_uses_marcuses_approximation():
    na = numerical_aperture(N_CORE, N_CLAD)
    v = v_number(q(CORE_RADIUS_UM, "um"), na, q(WAVELENGTH_NM, "nm"))
    mfd = mode_field_diameter(q(CORE_RADIUS_UM, "um"), v)
    expected = 2.0 * CORE_RADIUS_UM * (0.65 + 1.619 / v**1.5 + 2.879 / v**6)
    assert mfd.to_value("um") == pytest.approx(expected)
    # the mode field is wider than the core for a weakly guiding fibre
    assert mfd.to_value("um") > 2.0 * CORE_RADIUS_UM


def test_mode_field_grows_as_the_fibre_is_operated_closer_to_cutoff():
    """Marcuse's fit is valid for 1.2 < V < 2.4, and is monotonic there."""
    near_cutoff = mode_field_diameter(q(CORE_RADIUS_UM, "um"), 1.3).to_value("um")
    well_guided = mode_field_diameter(q(CORE_RADIUS_UM, "um"), 2.4).to_value("um")
    assert near_cutoff > well_guided > 2.0 * CORE_RADIUS_UM


def test_coupling_between_two_gaussian_modes_reuses_the_overlap_result():
    first = mode_field_diameter(q(CORE_RADIUS_UM, "um"), 2.2).to_value("um") / 2.0
    second = mode_field_diameter(q(CORE_RADIUS_UM, "um"), 2.4).to_value("um") / 2.0
    assert mode_overlap(first, second) == pytest.approx(
        (2.0 * first * second / (first**2 + second**2)) ** 2
    )


def test_coupling_loss_in_decibels():
    assert coupling_loss_db(1.0) == pytest.approx(0.0)
    assert coupling_loss_db(0.5) == pytest.approx(3.010299956639812)
    assert coupling_loss_db(0.0) == pytest.approx(np.inf)


def test_propagation_constant_of_the_fundamental_mode():
    # well inside the single-mode regime the mode index sits between n_clad
    # and n_core
    beta = propagation_constant(
        q(WAVELENGTH_NM, "nm"), q(CORE_RADIUS_UM, "um"), N_CORE, N_CLAD
    )
    k0 = 2.0 * np.pi / (WAVELENGTH_NM * 1e-9)
    assert k0 * N_CLAD < beta.to_value("1/m") < k0 * N_CORE


def test_a_dimensioned_index_is_rejected():
    with pytest.raises((DimensionError, ValueError)):
        numerical_aperture(q(1.45, "mm"), N_CLAD)
