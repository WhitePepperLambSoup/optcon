"""Ray-transfer (ABCD) matrices for common optical elements.

An ABCD matrix mixes lengths and angles, so it is not a homogeneous
dimensionless object: the ``B`` entry carries a length and the ``C`` entry its
reciprocal.  ``TransferMatrix`` therefore carries the unit in which the
off-diagonal entries are expressed, which is what lets a chain of elements
given in millimetres and metres compose without anyone converting by hand.
"""

import numpy as np
import pytest

from optcon import DimensionError, q, unit
from optcon.elements import (
    TransferMatrix,
    compose,
    curved_mirror,
    dielectric_interface,
    effective_focal_length,
    flat_mirror,
    free_space,
    is_marginally_stable,
    is_stable,
    refracting_surface,
    stability,
    thin_lens,
)


def test_free_space_matrix_has_a_length_in_the_b_entry():
    space = free_space(q(100.0, "mm"))
    assert space.matrix[0, 0] == pytest.approx(1.0)
    assert space.matrix[0, 1] == pytest.approx(100.0)
    assert space.matrix[1, 0] == pytest.approx(0.0)
    assert space.matrix[1, 1] == pytest.approx(1.0)
    assert space.unit.symbol == "mm"


def test_thin_lens_matrix_inverts_the_focal_length():
    lens = thin_lens(q(50.0, "mm"))
    assert lens.matrix[1, 0] == pytest.approx(-1.0 / 50.0)
    assert lens.unit.symbol == "mm"


def test_a_bare_number_is_rejected_where_a_length_is_required():
    with pytest.raises(TypeError, match="units"):
        free_space(100.0)


def test_a_chain_written_in_two_units_has_the_right_physics():
    chain = compose(free_space(q(100.0, "mm")), thin_lens(q(0.1, "m")))
    # the left operand of A @ B supplies the display unit ...
    assert chain.unit.symbol == "m"
    # ... and the 0.1 m focal length survives the mixed-unit chain
    assert effective_focal_length(chain).to_value("mm") == pytest.approx(100.0)
    # re-expressing in millimetres moves B by 1000 and C by 1/1000
    in_mm = chain.to(unit("mm"))
    assert in_mm.matrix[1, 0] == pytest.approx(-1.0 / 100.0)
    assert in_mm.to(unit("m")).matrix == pytest.approx(chain.matrix)


def test_incompatible_units_are_rejected():
    with pytest.raises(DimensionError):
        compose(free_space(q(100.0, "mm")), thin_lens(q(1.0, "s")))


def test_determinant_is_unity_for_same_medium_systems():
    chain = compose(
        free_space(q(200.0, "mm")),
        thin_lens(q(75.0, "mm")),
        free_space(q(50.0, "mm")),
    )
    assert chain.determinant == pytest.approx(1.0, rel=1e-12)


def test_dielectric_interface_carries_the_index_ratio():
    interface = dielectric_interface(q(1.0, "1"), q(1.5, "1"))
    assert interface.matrix[1, 1] == pytest.approx(1.0 / 1.5)
    assert interface.determinant == pytest.approx(1.0 / 1.5)


def test_compose_applies_elements_in_beam_order():
    space = free_space(q(100.0, "mm"))
    lens = thin_lens(q(50.0, "mm"))
    assert compose(space, lens).matrix == pytest.approx((lens @ space).matrix)
    # order matters: lens-then-space is a different system
    assert compose(space, lens).matrix != pytest.approx(compose(lens, space).matrix)


def test_two_lens_effective_focal_length_matches_the_textbook_formula():
    f1 = q(100.0, "mm")
    f2 = q(50.0, "mm")
    separation = q(40.0, "mm")
    system = compose(
        thin_lens(f1), free_space(separation), thin_lens(f2)
    )
    expected = 1.0 / (1.0 / f1.value + 1.0 / f2.value - separation.value / (f1.value * f2.value))
    assert effective_focal_length(system).value == pytest.approx(expected, rel=1e-10)


def test_a_single_thin_lens_is_its_own_effective_focal_length():
    assert effective_focal_length(thin_lens(q(80.0, "mm"))).value == pytest.approx(80.0)


def test_flat_mirror_is_the_identity():
    assert flat_mirror().matrix == pytest.approx(np.eye(2))


def test_stability_of_a_symmetric_two_mirror_cavity():
    length = q(300.0, "mm")
    mirror_radius = q(500.0, "mm")
    arm = compose(
        free_space(length),
        curved_mirror(mirror_radius),
        free_space(length),
        curved_mirror(mirror_radius),
    )
    # g = 1 - L/R = 1 - 0.6 = 0.4, so g1*g2 = 0.16 -> stable
    assert is_stable(arm)
    assert abs(stability(arm)) < 1.0


def test_a_confocal_boundary_cavity_is_on_the_edge_of_stability():
    # a symmetric confocal cavity has R = L, hence g = 0 and (A + D)/2 = -1
    length = q(100.0, "mm")
    radius = q(100.0, "mm")
    arm = compose(
        free_space(length),
        curved_mirror(radius),
        free_space(length),
        curved_mirror(radius),
    )
    assert stability(arm) == pytest.approx(-1.0, rel=1e-12)
    assert is_stable(arm, atol=1e-12)


def test_cavity_stability_equals_two_g1_g2_minus_one():
    """The textbook identity (A + D)/2 = 2 g1 g2 - 1, checked over a sweep."""
    for length_mm, radius_mm in [(100.0, 300.0), (250.0, 400.0), (300.0, 500.0), (120.0, 120.0)]:
        arm = compose(
            free_space(q(length_mm, "mm")),
            curved_mirror(q(radius_mm, "mm")),
            free_space(q(length_mm, "mm")),
            curved_mirror(q(radius_mm, "mm")),
        )
        g = 1.0 - length_mm / radius_mm
        assert stability(arm) == pytest.approx(2.0 * g * g - 1.0, rel=1e-12)


def test_a_flat_flat_cavity_sits_on_the_degenerate_end_of_the_boundary():
    arm = compose(
        free_space(q(100.0, "mm")),
        flat_mirror(),
        free_space(q(100.0, "mm")),
        flat_mirror(),
    )
    # g1*g2 = 1, so (A + D)/2 = +1: the mode is not confined in practice
    assert stability(arm) == pytest.approx(1.0, rel=1e-12)
    assert is_marginally_stable(arm)


def test_raising_a_matrix_to_a_power_repeats_the_round_trip():
    single = compose(free_space(q(100.0, "mm")), flat_mirror())
    assert (single**2).matrix == pytest.approx((single @ single).matrix)
    assert (single**2).unit.symbol == "mm"


def test_transfer_matrix_rejects_a_non_square_array():
    with pytest.raises(ValueError, match="2x2"):
        TransferMatrix(np.zeros((3, 3)), free_space(q(1.0, "mm")).unit)


def test_a_refracting_surface_has_the_surface_power():
    # Phi = (n2 - n1) / R, so C = -Phi / n2
    surface = refracting_surface(1.0, 1.5, q(100.0, "mm"))
    assert surface.matrix[1, 0] == pytest.approx(-(1.5 - 1.0) / (1.5 * 100.0))
    assert surface.unit.symbol == "mm"


def test_a_flat_refracting_surface_reduces_to_the_interface_matrix():
    surface = refracting_surface(1.0, 1.5, q(np.inf, "mm"))
    assert surface.matrix == pytest.approx(dielectric_interface(1.0, 1.5).matrix)


def test_a_single_refracting_surface_has_the_textbook_focal_length():
    # f = n2 R / (n2 - n1)
    surface = refracting_surface(1.0, 1.5, q(100.0, "mm"))
    assert effective_focal_length(surface).to_value("mm") == pytest.approx(
        1.5 * 100.0 / 0.5
    )


def test_a_thick_biconvex_lens_focal_length_from_the_thick_lens_formula():
    """The lensmaker equation for a thick lens, assembled from primitives."""
    n_lens = 1.5
    radius_1 = q(100.0, "mm")
    radius_2 = q(-100.0, "mm")
    thickness = q(5.0, "mm")
    lens = compose(
        refracting_surface(1.0, n_lens, radius_1),
        free_space(thickness),
        refracting_surface(n_lens, 1.0, radius_2),
    )
    # 1/f = (n-1)[1/R1 - 1/R2 + (n-1)d / (n R1 R2)]
    expected_inverse = (n_lens - 1.0) * (
        1.0 / radius_1.value
        - 1.0 / radius_2.value
        + (n_lens - 1.0) * thickness.value / (n_lens * radius_1.value * radius_2.value)
    )
    assert effective_focal_length(lens).to_value("mm") == pytest.approx(
        1.0 / expected_inverse
    )


def test_a_thick_lens_has_unit_determinant_between_the_same_media():
    lens = compose(
        refracting_surface(1.0, 1.5, q(100.0, "mm")),
        free_space(q(5.0, "mm")),
        refracting_surface(1.5, 1.0, q(-100.0, "mm")),
    )
    assert lens.determinant == pytest.approx(1.0, rel=1e-12)


def test_stability_status_and_afocal_systems():
    from optcon.elements import (
        chief_ray_trace,
        magnification,
        stability_status,
    )

    # Afocal system (e.g. pure free space has C = 0)
    fs = free_space(q(100.0, "mm"))
    with pytest.raises(ValueError, match="afocal"):
        effective_focal_length(fs)

    # Magnification
    # B = 0 for 2f-2f imaging or identity
    identity_system = compose(thin_lens(q(50.0, "mm")), thin_lens(q(-50.0, "mm")))
    assert magnification(identity_system) == pytest.approx(1.0)
    # For a general system with B != 0, magnification raises
    with pytest.raises(ValueError, match="magnification is only defined"):
        magnification(fs)

    # Stability status reporting
    stable_cav = compose(free_space(q(100.0, "mm")), curved_mirror(q(300.0, "mm")))
    assert stability_status(stable_cav) == "stable"

    # Chief ray trace
    y_out, theta_out = chief_ray_trace(fs, 1.0, q(0.01, "rad"))
    assert theta_out == pytest.approx(0.01)
    assert y_out == pytest.approx(1.0 + 100.0 * 0.01)

