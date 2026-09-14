"""A user-facing adapter for paraxial focal length, across engines.

The engine adapters so far cover thin films, Mie scattering and beam
propagation.  This one covers ray tracing: the same singlet, described once,
computed by optcon's own ABCD chain and by optiland's paraxial solver.
"""

import numpy as np
import pytest

from optcon import q
from optcon.engines import available_engines, compare_across_engines
from optcon.paraxial import thick_lens_focal_length

RADIUS_MM = 100.0
THICKNESS_MM = 5.0
INDEX = 1.517


def test_the_reference_engine_matches_the_thick_lens_formula():
    value = thick_lens_focal_length(
        radius=q(RADIUS_MM, "mm"),
        thickness=q(THICKNESS_MM, "mm"),
        refractive_index=INDEX,
        engine="optcon_paraxial",
    )
    n = INDEX
    r1, r2, d = RADIUS_MM, -RADIUS_MM, THICKNESS_MM
    expected_inverse = (n - 1.0) * (
        1.0 / r1 - 1.0 / r2 + (n - 1.0) * d / (n * r1 * r2)
    )
    assert value.to_value("mm") == pytest.approx(1.0 / expected_inverse)


def test_a_symmetric_lens_has_a_unit_scale_convention():
    # flint glass focuses harder than crown: a higher index shortens f
    crown = thick_lens_focal_length(
        radius=q(RADIUS_MM, "mm"),
        thickness=q(THICKNESS_MM, "mm"),
        refractive_index=1.517,
        engine="optcon_paraxial",
    ).to_value("mm")
    flint = thick_lens_focal_length(
        radius=q(RADIUS_MM, "mm"),
        thickness=q(THICKNESS_MM, "mm"),
        refractive_index=1.62,
        engine="optcon_paraxial",
    ).to_value("mm")
    assert flint < crown


@pytest.mark.skipif("optiland" not in available_engines(), reason="optiland unavailable")
def test_optiland_agrees_with_the_abcd_chain():
    reference = thick_lens_focal_length(
        radius=q(RADIUS_MM, "mm"),
        thickness=q(THICKNESS_MM, "mm"),
        refractive_index=INDEX,
        engine="optcon_paraxial",
    )
    engine = thick_lens_focal_length(
        radius=q(RADIUS_MM, "mm"),
        thickness=q(THICKNESS_MM, "mm"),
        refractive_index=INDEX,
        engine="optiland",
    )
    assert engine.to_value("mm") == pytest.approx(reference.to_value("mm"), rel=1e-9)


@pytest.mark.skipif("optiland" not in available_engines(), reason="optiland unavailable")
def test_the_differential_harness_can_be_pointed_at_the_ray_tracers():
    report = compare_across_engines(
        lambda engine: {
            "f_mm": thick_lens_focal_length(
                radius=q(RADIUS_MM, "mm"),
                thickness=q(THICKNESS_MM, "mm"),
                refractive_index=INDEX,
                engine=engine,
            ).to_value("mm")
        },
        engines=["optcon_paraxial", "optiland"],
        rtol=1e-9,
    )
    assert report["agree"] is True


def test_an_unknown_engine_is_reported_with_the_available_ones():
    with pytest.raises(KeyError):
        thick_lens_focal_length(
            radius=q(RADIUS_MM, "mm"),
            thickness=q(THICKNESS_MM, "mm"),
            refractive_index=INDEX,
            engine="not_an_engine",
        )


def test_a_negative_radius_focal_length_is_reported_with_its_sign():
    value = thick_lens_focal_length(
        radius=q(-RADIUS_MM, "mm"),
        thickness=q(THICKNESS_MM, "mm"),
        refractive_index=INDEX,
        engine="optcon_paraxial",
    )
    assert np.isfinite(value.to_value("mm"))
