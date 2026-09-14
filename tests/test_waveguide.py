"""Guided modes of a symmetric slab waveguide.

The slab is the one waveguide with a closed-form dispersion relation, which
makes it the right place to check that a mode solver is solving the right
equation: every returned mode is substituted back into the relation that
defines it.
"""

import numpy as np
import pytest

from optcon import DimensionError, q
from optcon.waveguide import (
    confinement_factor,
    effective_indices,
    normalized_frequency,
    number_of_modes,
    single_mode,
)

N_CORE = 3.48
N_CLAD = 1.44
THICKNESS_NM = 500.0
WAVELENGTH_NM = 1550.0


def test_normalized_frequency_is_the_usual_v_number():
    v = normalized_frequency(
        q(THICKNESS_NM, "nm"), N_CORE, N_CLAD, q(WAVELENGTH_NM, "nm")
    )
    half_thickness_nm = THICKNESS_NM / 2.0
    na = np.sqrt(N_CORE**2 - N_CLAD**2)
    assert v == pytest.approx(2.0 * np.pi * half_thickness_nm / WAVELENGTH_NM * na)


def test_effective_indices_lie_between_the_two_refractive_indices():
    modes = effective_indices(
        q(THICKNESS_NM, "nm"), N_CORE, N_CLAD, q(WAVELENGTH_NM, "nm")
    )
    assert modes
    for value in modes:
        assert N_CLAD < value < N_CORE
    assert modes == sorted(modes, reverse=True)


def test_every_mode_satisfies_the_dispersion_relation():
    """Substitute each root back into the equation that defined it."""
    wavelength_nm = WAVELENGTH_NM
    half_thickness_nm = THICKNESS_NM / 2.0
    k0 = 2.0 * np.pi / wavelength_nm
    v = normalized_frequency(
        q(THICKNESS_NM, "nm"), N_CORE, N_CLAD, q(wavelength_nm, "nm")
    )
    modes = effective_indices(
        q(THICKNESS_NM, "nm"), N_CORE, N_CLAD, q(wavelength_nm, "nm")
    )
    for index, n_eff in enumerate(modes):
        kappa = k0 * np.sqrt(N_CORE**2 - n_eff**2)
        gamma = k0 * np.sqrt(n_eff**2 - N_CLAD**2)
        u = kappa * half_thickness_nm
        w = gamma * half_thickness_nm
        even = index % 2 == 0
        residual = u * np.tan(u) - w if even else -u / np.tan(u) - w
        assert abs(residual) < 1e-6, (index, n_eff, residual)
        assert u**2 + w**2 == pytest.approx(v**2, rel=1e-9)


def test_a_thin_slab_guides_only_one_mode():
    modes = effective_indices(q(100.0, "nm"), N_CORE, N_CLAD, q(WAVELENGTH_NM, "nm"))
    assert len(modes) == 1
    assert single_mode(q(100.0, "nm"), N_CORE, N_CLAD, q(WAVELENGTH_NM, "nm"))


def test_a_thick_slab_guides_many_modes():
    modes = effective_indices(q(2000.0, "nm"), N_CORE, N_CLAD, q(WAVELENGTH_NM, "nm"))
    assert len(modes) > 4
    v = normalized_frequency(
        q(2000.0, "nm"), N_CORE, N_CLAD, q(WAVELENGTH_NM, "nm")
    )
    assert number_of_modes(v) == len(modes)


def test_the_mode_count_grows_by_one_at_each_cutoff():
    """Every additional mode appears when V crosses another multiple of pi/2."""
    for multiple in (1, 2, 3):
        just_below = number_of_modes(multiple * np.pi / 2.0 - 1e-6)
        just_above = number_of_modes(multiple * np.pi / 2.0 + 1e-6)
        assert just_above == just_below + 1


def test_confinement_grows_with_the_normalized_frequency():
    thin = confinement_factor(q(400.0, "nm"), N_CORE, N_CLAD, q(WAVELENGTH_NM, "nm"))
    thick = confinement_factor(q(1200.0, "nm"), N_CORE, N_CLAD, q(WAVELENGTH_NM, "nm"))
    assert 0.0 < thin < thick < 1.0


def test_the_fundamental_mode_never_reaches_cutoff():
    """A symmetric slab guides TE0 for any V above zero."""
    for thickness in (50.0, 20.0, 10.0):
        modes = effective_indices(
            q(thickness, "nm"), N_CORE, N_CLAD, q(WAVELENGTH_NM, "nm")
        )
        assert len(modes) >= 1
        assert modes[0] > N_CLAD


def test_a_dimensioned_refractive_index_is_rejected():
    with pytest.raises((DimensionError, ValueError)):
        effective_indices(
            q(THICKNESS_NM, "nm"), q(3.48, "mm"), N_CLAD, q(WAVELENGTH_NM, "nm")
        )


def test_a_core_denser_than_its_cladding_is_required():
    with pytest.raises(ValueError, match="n_core"):
        effective_indices(q(THICKNESS_NM, "nm"), 1.4, 1.5, q(WAVELENGTH_NM, "nm"))
