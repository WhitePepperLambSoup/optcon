"""A scalar field on a sampled grid, and FFT propagation of it.

This is the first piece of the field-representation layer: a ``Field`` knows
the pitch of the grid it lives on, so a sampled field carries its
discretisation with it instead of leaving it in the caller's head.

The propagator is checked three ways: against the analytic Gaussian beam it
is launched with, against itself by propagating back, and against the
LightPipes engine that ships in this workspace.
"""

import numpy as np
import pytest

from optcon import q
from optcon.engines import available_engines
from optcon.propagation import (
    Field,
    gaussian_field,
    intensity,
    power,
    propagate,
    second_moment_radius,
)

WAIST_UM = 50.0
LAMBDA_NM = 633.0
EXTENT_UM = 1600.0
SAMPLES = 256
# The Rayleigh range, and the validity limit of the transfer-function method
# z <= extent^2 / (N lambda), which is 15.8 mm for this grid.
RAYLEIGH_MM = np.pi * (WAIST_UM * 1e-3) ** 2 / (LAMBDA_NM * 1e-6)


def _field() -> Field:
    return gaussian_field(
        waist=q(WAIST_UM, "um"),
        wavelength=q(LAMBDA_NM, "nm"),
        samples=SAMPLES,
        extent=q(EXTENT_UM, "um"),
    )


def analytic_radius(distance_mm: float) -> float:
    return WAIST_UM * np.sqrt(1.0 + (distance_mm / RAYLEIGH_MM) ** 2)


def test_a_gaussian_field_has_the_waist_it_was_given():
    assert second_moment_radius(_field()).to_value("um") == pytest.approx(
        WAIST_UM, rel=2e-3
    )


def test_the_grid_carries_its_spacing_and_shape():
    field = _field()
    assert field.spacing.to_value("um") == pytest.approx(EXTENT_UM / SAMPLES)
    assert field.amplitude.shape == (SAMPLES, SAMPLES)
    assert field.wavelength.to_value("nm") == pytest.approx(LAMBDA_NM)


def test_an_inconsistent_field_is_rejected():
    with pytest.raises(ValueError, match="two-dimensional"):
        Field(np.zeros((4, 4, 4), dtype=complex), q(1.0, "um"), q(1.0, "nm"))


def test_intensity_is_non_negative_and_matches_the_field_shape():
    field = _field()
    values = intensity(field)
    assert values.shape == field.amplitude.shape
    assert np.all(values >= 0.0)


def test_power_is_measured_in_grid_area_units():
    field = _field()
    total = power(field)
    assert total.unit.dimension == (q(1.0, "um") ** 2).unit.dimension
    # the field is not renormalised, so integrating exp(-2 r^2 / w0^2) gives
    # pi w0^2 / 2
    assert total.to_value("um^2") == pytest.approx(np.pi * WAIST_UM**2 / 2.0, rel=2e-3)


def test_zero_distance_leaves_the_field_alone():
    field = _field()
    unchanged = propagate(field, q(0.0, "um"))
    assert np.allclose(unchanged.amplitude, field.amplitude)


def test_propagation_conserves_power():
    field = _field()
    moved = propagate(field, q(RAYLEIGH_MM, "mm"))
    assert power(moved).to_value("um^2") == pytest.approx(
        power(field).to_value("um^2"), rel=1e-9
    )


def test_the_beam_radius_follows_the_analytic_hyperbola():
    field = _field()
    for distance_mm in (RAYLEIGH_MM / 2.0, RAYLEIGH_MM, 1.5 * RAYLEIGH_MM):
        moved = propagate(field, q(distance_mm, "mm"))
        measured = second_moment_radius(moved).to_value("um")
        assert measured == pytest.approx(analytic_radius(distance_mm), rel=5e-3), distance_mm


def test_propagating_back_recovers_the_original_beam():
    field = _field()
    distance = q(0.8 * RAYLEIGH_MM, "mm")
    there_and_back = propagate(propagate(field, distance), -distance)
    assert second_moment_radius(there_and_back).to_value("um") == pytest.approx(
        WAIST_UM, rel=5e-3
    )


def test_the_paraxial_and_exact_transfer_functions_agree_at_small_angles():
    field = _field()
    distance = q(0.5 * RAYLEIGH_MM, "mm")
    exact = propagate(field, distance, method="angular_spectrum")
    paraxial = propagate(field, distance, method="fresnel")
    assert second_moment_radius(exact).to_value("um") == pytest.approx(
        second_moment_radius(paraxial).to_value("um"), rel=1e-3
    )


def test_an_unknown_method_is_rejected():
    with pytest.raises(ValueError, match="method"):
        propagate(_field(), q(1.0, "mm"), method="ray_trace")


@pytest.mark.skipif(
    "lightpipes_forvard" not in available_engines("beam"), reason="LightPipes unavailable"
)
def test_agrees_with_the_lightpipes_spectral_propagator():
    """Two independent implementations of the same FFT propagation."""
    import LightPipes

    distance_mm = RAYLEIGH_MM
    mine = second_moment_radius(propagate(_field(), q(distance_mm, "mm"))).to_value("um")

    # LightPipes works in whatever single unit you hand it, so size,
    # wavelength and distance must all be millimetres here
    start = LightPipes.Begin(EXTENT_UM * 1e-3, LAMBDA_NM * 1e-6, SAMPLES)
    start = LightPipes.GaussBeam(start, WAIST_UM * 1e-3)
    start = LightPipes.Forvard(start, distance_mm)
    theirs = LightPipes.D4sigma(start)[0] / 2.0 * 1e3  # mm -> um

    assert mine == pytest.approx(theirs, rel=1e-2)
