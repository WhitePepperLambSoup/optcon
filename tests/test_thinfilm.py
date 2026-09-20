"""Thin film design: quarter-wave stacks, anti-reflection coatings, Bragg mirrors.

These are the design formulas an engineer reaches for before running a full
solver, so they are checked both against the closed forms (Macleod,
/Thin-Film Optical Filters/, ch. 3 and 5) and - for the single layer - against
the ``tmm_core`` engine in this workspace, which solves the same problem by an
entirely different route.
"""

import numpy as np
import pytest

from optcon import q
from optcon.engines import available_engines
from optcon.thinfilm import (
    anti_reflection_thickness,
    bragg_reflectance,
    ideal_anti_reflection_index,
    optical_thickness,
    quarter_wave_stack_reflectance,
    single_layer_reflectance,
)

N_AIR = 1.0
N_GLASS = 1.5
LAMBDA_NM = 550.0


def test_optical_thickness_is_the_index_times_the_geometric_thickness():
    assert optical_thickness(q(100.0, "nm"), 2.0).to_value("nm") == pytest.approx(200.0)


def test_quarter_wave_thickness_for_an_anti_reflection_coating():
    # a quarter wave at 550 nm in a film of index 1.22 is 112.7 nm thick
    thickness = anti_reflection_thickness(1.22, q(LAMBDA_NM, "nm"))
    assert thickness.to_value("nm") == pytest.approx(LAMBDA_NM / (4.0 * 1.22))


def test_ideal_single_layer_index_is_the_geometric_mean():
    assert ideal_anti_reflection_index(N_AIR, N_GLASS) == pytest.approx(
        np.sqrt(N_AIR * N_GLASS)
    )


def test_a_perfect_quarter_wave_coating_nullifies_reflection():
    ideal = ideal_anti_reflection_index(N_AIR, N_GLASS)
    thickness = anti_reflection_thickness(ideal, q(LAMBDA_NM, "nm"))
    value = single_layer_reflectance(
        N_AIR, ideal, N_GLASS, thickness, q(LAMBDA_NM, "nm")
    )
    assert value.to_value("1") == pytest.approx(0.0, abs=1e-15)


def test_a_bare_glass_surface_reflects_about_four_percent():
    # no film: the Fresnel result at normal incidence
    value = single_layer_reflectance(
        N_AIR, N_AIR, N_GLASS, q(0.0, "nm"), q(LAMBDA_NM, "nm")
    )
    assert value.to_value("1") == pytest.approx(((N_AIR - N_GLASS) / (N_AIR + N_GLASS)) ** 2)


def test_single_layer_reflectance_is_a_power_ratio():
    value = single_layer_reflectance(
        N_AIR, 1.38, N_GLASS, q(100.0, "nm"), q(LAMBDA_NM, "nm")
    )
    assert value.amp_order == 2
    assert 0.0 <= value.to_value("1") <= 1.0


def test_bragg_mirror_reflectance_grows_with_the_number_of_pairs():
    reflectances = [
        quarter_wave_stack_reflectance(2.3, 1.45, N_GLASS, pairs) for pairs in (1, 4, 10, 20)
    ]
    assert reflectances == sorted(reflectances)
    assert reflectances[-1] > 0.99


def test_bragg_reflectance_matches_the_closed_form():
    n_high, n_low, pairs = 2.3, 1.45, 8
    expected_ratio = (n_high / n_low) ** (2 * pairs) * n_high**2 / N_GLASS
    expected = ((N_AIR - expected_ratio) / (N_AIR + expected_ratio)) ** 2
    assert bragg_reflectance(N_AIR, n_high, n_low, N_GLASS, pairs) == pytest.approx(expected)


def test_bragg_reflectance_documents_its_terminal_high_index_layer():
    assert "(HL)^N H" in (bragg_reflectance.__doc__ or "")


def test_a_graded_index_stack_beats_a_single_layer_of_the_same_material():
    single = single_layer_reflectance(
        N_AIR, 2.3, N_GLASS, anti_reflection_thickness(2.3, q(LAMBDA_NM, "nm")), q(LAMBDA_NM, "nm")
    ).to_value("1")
    stack = quarter_wave_stack_reflectance(2.3, 1.45, N_GLASS, 20)
    assert stack > single


@pytest.mark.skipif("tmm_core" not in available_engines(), reason="tmm_core unavailable")
def test_single_layer_agrees_with_the_tmm_engine():
    import math

    import tmm_core

    n_film = 1.38
    thickness = 100.0
    for wavelength_nm in (450.0, 550.0, 650.0):
        mine = single_layer_reflectance(
            N_AIR, n_film, N_GLASS, q(thickness, "nm"), q(wavelength_nm, "nm")
        ).to_value("1")
        theirs = tmm_core.coh_tmm(
            "s",
            [N_AIR, n_film, N_GLASS],
            [math.inf, thickness, math.inf],
            0.0,
            wavelength_nm,
        )["R"]
        assert mine == pytest.approx(float(theirs), rel=1e-12)
