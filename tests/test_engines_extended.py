"""Cross-checks against the heavier engines now that their dependencies exist.

Installing numba, autograd, astropy and vtk brought a second wave of the
surveyed libraries under test: a ray tracer, an FDFD solver with an adjoint,
and a photonic-circuit package.  Each check below compares an external
implementation against something optcon states independently - a closed-form
relation, or a different library's version of the same predicate.
"""

import numpy as np
import pytest

from optcon import q
from optcon.checks import check_gradient, finite_difference_gradient, is_unitary
from optcon.elements import (
    TransferMatrix,
    compose,
    effective_focal_length,
    free_space,
    refracting_surface,
)
from optcon.engines import (
    ENGINE_SPECS,
    airy_psf_radius,
    available_engines,
    compare_across_engines,
)


def test_the_second_wave_of_engines_is_registered_with_conventions():
    for name in ("optiland", "rayoptics", "poppy", "prysm", "neuroptica", "ceviche"):
        spec = ENGINE_SPECS[name]
        assert spec.domain
        assert spec.length_unit
        assert spec.angle_unit


def test_the_newly_installable_engines_actually_import():
    available = available_engines()
    missing = [name for name in ("optiland", "rayoptics", "poppy", "prysm", "neuroptica", "ceviche") if name not in available]
    if missing:
        pytest.skip(f"Extended optional engines not installed in this environment: {missing}")
    for name in ("optiland", "rayoptics", "poppy", "prysm", "neuroptica", "ceviche"):
        assert name in available


@pytest.mark.skipif("neuroptica" not in available_engines(), reason="neuroptica unavailable")
def test_two_libraries_agree_on_what_unitary_means():
    """neuroptica ships is_unitary; optcon states it as a contract."""
    from neuroptica.utils import is_unitary as their_is_unitary

    rng = np.random.default_rng(7)
    cases = []
    # genuinely unitary: a random complex matrix from a QR decomposition
    for _ in range(4):
        square = rng.standard_normal((4, 4)) + 1.0j * rng.standard_normal((4, 4))
        orthogonal, _ = np.linalg.qr(square)
        cases.append(orthogonal)
    # deliberately not unitary
    cases.append(np.diag([1.0, 1.4]).astype(complex))
    cases.append(np.zeros((3, 3), dtype=complex))
    cases.append(np.array([[1.0, 1.0], [0.0, 1.0]], dtype=complex))
    cases.append(np.eye(5) * 0.99)

    for index, matrix in enumerate(cases):
        assert bool(their_is_unitary(matrix)) == is_unitary(matrix), index


@pytest.mark.skipif("optiland" not in available_engines(), reason="optiland unavailable")
def test_a_thick_lens_matches_the_optiland_paraxial_solver():
    """The ABCD primitives against a ray tracer's own paraxial calculation."""
    from optiland.optic import Optic

    radius_mm = 100.0
    thickness_mm = 5.0
    lens = Optic()
    add_surf = lens.surfaces.add if hasattr(lens, "surfaces") and hasattr(lens.surfaces, "add") else lens.add_surface
    add_surf(index=0, radius=np.inf, thickness=100.0)
    add_surf(
        index=1, radius=radius_mm, thickness=thickness_mm, material="N-BK7", is_stop=True
    )
    add_surf(index=2, radius=-radius_mm, thickness=50.0, material="air")
    add_surf(index=3)
    lens.wavelengths.add(0.55, is_primary=True)

    glass = float(np.asarray(lens.surfaces[1].material_post.n(0.55)).ravel()[0])
    theirs = float(lens.paraxial.f2())

    mine = effective_focal_length(
        compose(
            refracting_surface(1.0, glass, q(radius_mm, "mm")),
            free_space(q(thickness_mm, "mm")),
            refracting_surface(glass, 1.0, q(-radius_mm, "mm")),
        )
    ).to_value("mm")

    assert mine == pytest.approx(theirs, rel=1e-6)


@pytest.mark.skipif("ceviche" not in available_engines(), reason="ceviche unavailable")
def test_ceviche_adjoint_matches_finite_differences():
    """An external solver's gradient, checked by optcon's verifier."""
    import autograd.numpy as anp
    from ceviche import fdfd_ez, jacobian

    nx = ny = 30
    omega = 2.0 * np.pi * 200e12
    spacing = 1e-6
    npml = [10, 10]
    source = np.zeros((nx, ny), dtype=complex)
    source[15, 15] = 1.0
    probe = (10, 20)

    base = np.full((nx, ny), 2.0)
    design = (slice(12, 18), slice(12, 18))
    flat_design = np.arange(nx * ny).reshape(nx, ny)[design].ravel()
    scatter = np.zeros((nx * ny, flat_design.size))
    scatter[flat_design, np.arange(flat_design.size)] = 1.0
    keep = np.ones(nx * ny)
    keep[flat_design] = 0.0
    base_flat = base.ravel()

    def objective(vector):
        flat = base_flat * keep + scatter @ vector
        eps = anp.reshape(flat, (nx, ny))
        simulation = fdfd_ez(omega, spacing, eps, npml)
        _, _, ez = simulation.solve(source)
        return anp.abs(ez[probe]) ** 2

    start = base[design].ravel().copy()
    adjoint = np.asarray(jacobian(objective, mode="reverse")(start), dtype=float).ravel()
    numerical = finite_difference_gradient(objective, start, eps=1e-3)

    scale = max(np.max(np.abs(numerical)), 1e-30)
    assert np.max(np.abs(adjoint - numerical)) / scale < 1e-3

    report = check_gradient(
        objective,
        lambda vector: np.asarray(
            jacobian(objective, mode="reverse")(vector), dtype=float
        ).ravel(),
        start,
        eps=1e-3,
        rtol=1e-2,
    )
    assert report["ok"] is True
    assert report["max_rel_error"] < 1e-2


def test_the_transfer_matrix_still_refuses_a_dimensioned_index():
    from optcon import DimensionError

    with pytest.raises((DimensionError, ValueError)):
        refracting_surface(q(1.0, "mm"), 1.5, q(100.0, "mm"))


def test_a_transfer_matrix_needs_a_length_for_its_off_diagonal():
    matrix = TransferMatrix(np.eye(2), q(1.0, "mm").unit)
    assert matrix.carries_length is False


def test_the_airy_reference_matches_the_closed_form():
    value = airy_psf_radius(
        aperture_diameter=q(0.1, "m"),
        wavelength=q(550.0, "nm"),
        engine="optcon_diffraction",
    )
    assert value.to_value("rad") == pytest.approx(
        1.22 * 550e-9 / 0.1
    )


@pytest.mark.skipif("poppy" not in available_engines(), reason="poppy unavailable")
def test_poppy_reproduces_the_airy_radius():
    """A wavefront-propagation package against the closed form."""
    reference = airy_psf_radius(
        aperture_diameter=q(0.1, "m"),
        wavelength=q(550.0, "nm"),
        engine="optcon_diffraction",
    ).to_value("rad")
    measured = airy_psf_radius(
        aperture_diameter=q(0.1, "m"),
        wavelength=q(550.0, "nm"),
        engine="poppy",
    ).to_value("rad")
    assert measured == pytest.approx(reference, rel=2e-2)


@pytest.mark.skipif("poppy" not in available_engines(), reason="poppy unavailable")
def test_the_differential_harness_covers_the_psf_family():
    report = compare_across_engines(
        lambda engine: {
            "theta_rad": airy_psf_radius(
                aperture_diameter=q(0.1, "m"),
                wavelength=q(550.0, "nm"),
                engine=engine,
            ).to_value("rad")
        },
        engines=["optcon_diffraction", "poppy"],
        rtol=2e-2,
    )
    assert report["agree"] is True
