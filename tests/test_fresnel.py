"""Fresnel reflection and transmission at a plane interface.

The checks are the ones a textbook would demand: Snell's law, the normal
incidence limit, Brewster's angle, total internal reflection, and - the
strongest of them - energy conservation, ``R + T = 1`` at every angle for a
lossless dielectric.

The results are additionally compared against the ``tmm_core`` engine that
ships in this workspace, which computes the same interface coefficients by a
completely different route.
"""

import numpy as np
import pytest

from optcon import ContractViolation, DimensionError, q
from optcon.engines import available_engines
from optcon.fresnel import (
    brewster_angle,
    critical_angle,
    fresnel_coefficients,
    reflectance,
    snell,
    transmittance,
    unpolarized_reflectance,
)

N_GLASS = 1.5
N_AIR = 1.0


def test_snell_law_holds():
    for incidence_deg in (0.0, 15.0, 30.0, 45.0, 60.0, 80.0):
        theta_i = np.deg2rad(incidence_deg)
        theta_t = snell(N_AIR, N_GLASS, q(incidence_deg, "deg")).to_value("rad")
        assert N_AIR * np.sin(theta_i) == pytest.approx(N_GLASS * np.sin(theta_t))


def test_normal_incidence_matches_the_familiar_limit():
    expected = ((N_AIR - N_GLASS) / (N_AIR + N_GLASS)) ** 2
    for polarization in ("s", "p", "unpolarized"):
        value = reflectance(N_AIR, N_GLASS, q(0.0, "deg"), polarization=polarization)
        assert value.to_value("1") == pytest.approx(expected)


def test_brewster_angle_is_arctan_of_the_index_ratio():
    expected = np.arctan(N_GLASS / N_AIR)
    assert brewster_angle(N_AIR, N_GLASS).to_value("rad") == pytest.approx(expected)


def test_p_polarization_vanishes_at_brewster():
    theta_b = brewster_angle(N_AIR, N_GLASS)
    value = reflectance(N_AIR, N_GLASS, theta_b, polarization="p")
    assert value.to_value("1") == pytest.approx(0.0, abs=1e-15)
    # ... while s-polarization does not
    assert reflectance(N_AIR, N_GLASS, theta_b, polarization="s").to_value("1") > 0.1


def test_energy_is_conserved_at_every_angle():
    for incidence_deg in np.linspace(0.0, 89.0, 40):
        for polarization in ("s", "p"):
            angle = q(float(incidence_deg), "deg")
            reflected = reflectance(N_AIR, N_GLASS, angle, polarization=polarization)
            transmitted = transmittance(N_AIR, N_GLASS, angle, polarization=polarization)
            assert reflected.value + transmitted.value == pytest.approx(1.0, abs=1e-12)


def test_transmittance_is_a_power_ratio_not_an_amplitude():
    value = transmittance(N_AIR, N_GLASS, q(45.0, "deg"), polarization="s")
    assert value.amp_order == 2


def test_total_internal_reflection_beyond_the_critical_angle():
    theta_c = critical_angle(N_GLASS, N_AIR)
    assert theta_c.to_value("deg") == pytest.approx(np.rad2deg(np.arcsin(N_AIR / N_GLASS)))
    for polarization in ("s", "p"):
        beyond = q(theta_c.to_value("deg") + 5.0, "deg")
        reflected = reflectance(N_GLASS, N_AIR, beyond, polarization=polarization)
        assert reflected.value == pytest.approx(1.0, abs=1e-12)
        assert transmittance(N_GLASS, N_AIR, beyond, polarization=polarization).value == pytest.approx(
            0.0, abs=1e-12
        )


def test_no_critical_angle_when_leaving_the_denser_medium_is_impossible():
    with pytest.raises(ValueError, match="total internal reflection"):
        critical_angle(N_AIR, N_GLASS)


def test_unpolarized_reflectance_is_the_mean_of_the_two():
    angle = q(60.0, "deg")
    s = reflectance(N_AIR, N_GLASS, angle, polarization="s").value
    p = reflectance(N_AIR, N_GLASS, angle, polarization="p").value
    assert unpolarized_reflectance(N_AIR, N_GLASS, angle).value == pytest.approx((s + p) / 2.0)


def test_fresnel_coefficients_satisfy_the_interface_relations():
    coefficients = fresnel_coefficients(N_AIR, N_GLASS, q(30.0, "deg"))
    assert set(coefficients) == {"r_s", "r_p", "t_s", "t_p"}
    # the tangential field is continuous: 1 + r = t for s at every angle, and
    # for p only once the geometric cos-factor asymmetry is removed
    for angle_deg in (0.0, 30.0, 60.0):
        c = fresnel_coefficients(N_AIR, N_GLASS, q(angle_deg, "deg"))
        assert 1.0 + c["r_s"] == pytest.approx(c["t_s"])
        assert (N_AIR / N_GLASS) * (1.0 + c["r_p"]) == pytest.approx(c["t_p"])
    assert coefficients["r_s"].real < 0.0  # air to glass flips the phase of s


def test_a_dimensioned_refractive_index_is_rejected():
    with pytest.raises((DimensionError, ValueError)):
        reflectance(q(1.0, "mm"), N_GLASS, q(30.0, "deg"))


@pytest.mark.skipif("tmm_core" not in available_engines(), reason="tmm_core unavailable")
def test_agrees_with_the_tmm_core_engine():
    """The same interface, computed by an independent library."""
    import tmm_core

    for incidence_deg in (0.0, 20.0, 45.0, 70.0):
        theta_i = np.deg2rad(incidence_deg)
        theta_t = np.arcsin(N_AIR * np.sin(theta_i) / N_GLASS)
        for polarization, engine_letter in (("s", "s"), ("p", "p")):
            mine = reflectance(
                N_AIR, N_GLASS, q(incidence_deg, "deg"), polarization=polarization
            ).value
            theirs = float(
                tmm_core.interface_R(engine_letter, N_AIR, N_GLASS, theta_i, theta_t)
            )
            assert mine == pytest.approx(theirs, rel=1e-12), (incidence_deg, polarization)


@pytest.mark.skipif("tmm_core" not in available_engines(), reason="tmm_core unavailable")
def test_transmittance_also_agrees_with_the_engine():
    import tmm_core

    for incidence_deg in (0.0, 35.0, 65.0):
        theta_i = np.deg2rad(incidence_deg)
        theta_t = np.arcsin(N_AIR * np.sin(theta_i) / N_GLASS)
        for polarization in ("s", "p"):
            mine = transmittance(
                N_AIR, N_GLASS, q(incidence_deg, "deg"), polarization=polarization
            ).value
            theirs = float(
                tmm_core.interface_T(polarization, N_AIR, N_GLASS, theta_i, theta_t)
            )
            assert mine == pytest.approx(theirs, rel=1e-12), (incidence_deg, polarization)


def test_a_contract_violation_is_raised_for_a_non_physical_index():
    with pytest.raises((ValueError, ContractViolation)):
        reflectance(-1.0, N_GLASS, q(30.0, "deg"))
