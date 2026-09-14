"""Vector fields: a spatial profile that also carries a polarisation state.

``Field`` is one complex array; ``VectorField`` is two, one per transverse
component.  Keeping them separate types means a scalar calculation cannot
quietly drop polarisation, and the conversion between them has to be asked
for.

In the paraxial, weakly guiding limit each transverse component propagates
independently, which is what makes the propagation of a vector field the
propagation of its two scalar parts.
"""

import numpy as np
import pytest

from optcon import q
from optcon.propagation import Field, gaussian_field, second_moment_radius
from optcon.vector_fields import (
    VectorField,
    analyzer_transmission,
    from_scalar,
    intensity_map,
    polarization_map,
    propagate_vector,
    stokes_map,
)

WAIST_UM = 50.0
LAMBDA_NM = 633.0
EXTENT_UM = 1600.0
SAMPLES = 128


def _linear_field(angle_deg: float = 0.0) -> VectorField:
    scalar = gaussian_field(
        waist=q(WAIST_UM, "um"),
        wavelength=q(LAMBDA_NM, "nm"),
        samples=SAMPLES,
        extent=q(EXTENT_UM, "um"),
    )
    angle = np.deg2rad(angle_deg)
    return from_scalar(scalar, angle=angle)


def test_a_scalar_field_converts_to_a_linear_vector_field():
    field = _linear_field(0.0)
    assert field.ex.shape == (SAMPLES, SAMPLES)
    assert np.allclose(field.ey, 0.0)
    ratio = np.abs(field.ey).max() / np.abs(field.ex).max()
    assert ratio == 0.0


def test_a_forty_five_degree_field_has_equal_components():
    field = _linear_field(45.0)
    peak = np.abs(field.ex).max()
    assert np.abs(field.ey).max() == pytest.approx(peak)


def test_the_intensity_map_is_the_sum_of_the_component_intensities():
    field = _linear_field(30.0)
    total = intensity_map(field)
    separate = np.abs(field.ex) ** 2 + np.abs(field.ey) ** 2
    assert total == pytest.approx(separate)


def test_a_linear_vector_field_reduces_to_its_scalar_part():
    field = _linear_field(0.0)
    scalar_equivalent = Field(field.ex, field.spacing, field.wavelength)
    assert second_moment_radius(scalar_equivalent).to_value("um") == pytest.approx(
        WAIST_UM, rel=2e-3
    )
    total = float(np.sum(intensity_map(field)))
    assert total == pytest.approx(float(np.sum(np.abs(field.ex) ** 2)))


def test_the_stokes_map_reports_a_pure_linear_state_everywhere():
    field = _linear_field(0.0)
    parameters = stokes_map(field)
    peak = np.unravel_index(np.argmax(parameters["S0"]), parameters["S0"].shape)
    assert parameters["S0"][peak] > 0.0
    assert parameters["S1"][peak] / parameters["S0"][peak] == pytest.approx(1.0)
    assert parameters["S2"][peak] / parameters["S0"][peak] == pytest.approx(0.0, abs=1e-9)
    assert parameters["S3"][peak] / parameters["S0"][peak] == pytest.approx(0.0, abs=1e-9)


def test_a_pure_state_satisfies_the_stokes_identity():
    field = _linear_field(22.0)
    parameters = stokes_map(field)
    mask = parameters["S0"] > 1e-6 * parameters["S0"].max()
    left = (
        parameters["S1"][mask] ** 2
        + parameters["S2"][mask] ** 2
        + parameters["S3"][mask] ** 2
    )
    right = parameters["S0"][mask] ** 2
    assert np.allclose(left, right, rtol=1e-9)


def test_polarization_map_reports_the_azimuth_of_a_linear_state():
    field = _linear_field(30.0)
    angle = polarization_map(field)
    centre = angle[SAMPLES // 2, SAMPLES // 2]
    assert np.rad2deg(centre) == pytest.approx(30.0, abs=1e-6)


def test_an_analyzer_transmits_malus_law_for_every_pixel():
    field = _linear_field(30.0)
    for analyzer_deg in (0.0, 30.0, 60.0, 90.0):
        transmitted = analyzer_transmission(field, q(analyzer_deg, "deg"))
        expected = np.cos(np.deg2rad(analyzer_deg - 30.0)) ** 2
        assert float(np.sum(transmitted)) == pytest.approx(
            float(np.sum(intensity_map(field))) * expected, rel=1e-9
        )


def test_propagation_keeps_a_linear_state_linear():
    field = _linear_field(40.0)
    moved = propagate_vector(field, q(5.0, "mm"))
    angle = polarization_map(moved)
    centre = angle[SAMPLES // 2, SAMPLES // 2]
    assert np.rad2deg(centre) == pytest.approx(40.0, abs=1e-6)


def test_propagation_carries_both_components_and_widens_the_beam():
    field = _linear_field(0.0)
    before = second_moment_radius(
        Field(field.ex, field.spacing, field.wavelength)
    ).to_value("um")
    moved = propagate_vector(field, q(5.0, "mm"))
    after = second_moment_radius(
        Field(moved.ex, moved.spacing, moved.wavelength)
    ).to_value("um")
    assert after > before


def test_a_radial_field_stays_radial_through_propagation():
    """A radially polarised beam is a vector field that is not separable."""
    axis = (np.arange(SAMPLES) - SAMPLES // 2) * (EXTENT_UM / SAMPLES)
    x, y = np.meshgrid(axis, axis)
    radius = np.hypot(x, y)
    profile = np.exp(-(radius**2) / WAIST_UM**2)
    with np.errstate(invalid="ignore", divide="ignore"):
        ex = np.where(radius > 0, profile * x / np.where(radius > 0, radius, 1), 0.0)
        ey = np.where(radius > 0, profile * y / np.where(radius > 0, radius, 1), 0.0)
    field = VectorField(
        ex.astype(complex), ey.astype(complex), q(EXTENT_UM / SAMPLES, "um"), q(LAMBDA_NM, "nm")
    )
    assert np.max(np.abs(field.ey)) > 0.1
    moved = propagate_vector(field, q(2.0, "mm"))
    # the polarisation azimuth at a fixed off-axis point must be unchanged
    # (a point off both axes, where both components are clearly non-zero -
    #  on the x axis a radial field has no y component and atan2 degenerates)
    row, column = SAMPLES // 2 - 3, SAMPLES // 2 + 3
    before = np.arctan2(field.ey[row, column].real, field.ex[row, column].real)
    after = np.arctan2(moved.ey[row, column].real, moved.ex[row, column].real)
    # a linear azimuth is defined modulo pi: propagation carries an overall
    # phase, and flipping both components does not change the state
    difference = (after - before) % np.pi
    assert min(difference, np.pi - difference) == pytest.approx(0.0, abs=1e-3)
    assert np.abs(field.ex[row, column]) > 0.1
    assert np.abs(field.ey[row, column]) > 0.1


def test_mismatched_component_shapes_are_rejected():
    with pytest.raises(ValueError, match="same shape"):
        VectorField(
            np.zeros((4, 4), dtype=complex),
            np.zeros((4, 5), dtype=complex),
            q(1.0, "um"),
            q(1.0, "nm"),
        )
