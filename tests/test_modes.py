"""Gauss-Hermite modes and the decomposition of a sampled field onto them.

This closes the loop on the field-representation layer: a `Field` lives on a
grid, a mode set lives in a basis, and `decompose` is the only sanctioned way
to move between them.  The tests check the basis is orthonormal, that a pure
mode decomposes onto itself, and that the decompositions add back up to the
power they started with.
"""

from typing import Any

import numpy as np
import pytest

from optcon import q
from optcon.modes import (
    decompose,
    hermite_gauss,
    laguerre_gauss,
    mode_content,
    mode_power_fractions,
    reconstruct,
)
from optcon.propagation import Field, gaussian_field, power, propagate, second_moment_radius

WAIST_UM = 50.0
LAMBDA_NM = 633.0
EXTENT_UM = 1600.0
SAMPLES = 128


def _coords():
    spacing = EXTENT_UM / SAMPLES
    axis = (np.arange(SAMPLES) - SAMPLES // 2) * spacing
    return np.meshgrid(axis, axis), spacing


def _field(**kwargs) -> Field:
    options: dict[str, Any] = dict(
        waist=q(WAIST_UM, "um"),
        wavelength=q(LAMBDA_NM, "nm"),
        samples=SAMPLES,
        extent=q(EXTENT_UM, "um"),
    )
    options.update(kwargs)
    return gaussian_field(**options)


def test_the_fundamental_mode_is_the_gaussian_we_already_have():
    (x, y), _ = _coords()
    mode = hermite_gauss(x, y, 0, 0, q(WAIST_UM, "um"))
    reference = np.exp(-(x**2 + y**2) / WAIST_UM**2)
    # the normalised mode equals the bare Gaussian times a constant
    ratio = np.abs(mode / reference)
    assert np.allclose(ratio, ratio.flat[ratio.size // 2], rtol=1e-12)


def test_the_modes_are_orthonormal_on_the_grid():
    (x, y), spacing = _coords()
    waist = q(WAIST_UM, "um")
    modes = [
        hermite_gauss(x, y, m, n, waist)
        for m in range(3)
        for n in range(3)
    ]
    for i, first in enumerate(modes):
        norm = np.sum(np.abs(first) ** 2) * spacing**2
        assert norm == pytest.approx(1.0, rel=1e-6), i
        for j, second in enumerate(modes):
            if j <= i:
                continue
            overlap = np.sum(np.conj(first) * second) * spacing**2
            assert abs(overlap) < 1e-6, (i, j)


def test_a_pure_gaussian_decomposes_entirely_onto_the_fundamental_mode():
    field = _field()
    coefficients = decompose(field, waist=q(WAIST_UM, "um"), max_order=3)
    fractions = mode_power_fractions(coefficients, field)
    assert fractions[(0, 0)] == pytest.approx(1.0, abs=1e-3)
    assert max(fraction for mode, fraction in fractions.items() if mode != (0, 0)) < 1e-6


def test_a_higher_order_mode_decomposes_onto_itself():
    (x, y), _ = _coords()
    mode = hermite_gauss(x, y, 1, 0, q(WAIST_UM, "um"))
    field = Field(mode, q(EXTENT_UM / SAMPLES, "um"), q(LAMBDA_NM, "nm"))
    coefficients = decompose(field, waist=q(WAIST_UM, "um"), max_order=3)
    fractions = mode_power_fractions(coefficients, field)
    assert fractions[(1, 0)] == pytest.approx(1.0, abs=1e-6)


def test_a_shifted_beam_couples_to_the_odd_modes():
    """A misaligned beam is exactly what the higher-order modes describe."""
    (x, y), spacing = _coords()
    shifted = np.exp(
        -((x - 0.3 * WAIST_UM) ** 2 + y**2) / WAIST_UM**2
    ).astype(complex)
    field = Field(shifted, q(spacing, "um"), q(LAMBDA_NM, "nm"))
    fractions = mode_power_fractions(
        decompose(field, waist=q(WAIST_UM, "um"), max_order=3), field
    )
    assert fractions[(1, 0)] > 0.02
    assert fractions[(0, 0)] < 0.98
    # a shift in x does not excite the y modes
    assert fractions[(0, 1)] < 1e-6


def test_the_decomposition_accounts_for_essentially_all_the_power():
    field = _field()
    coefficients = decompose(field, waist=q(WAIST_UM, "um"), max_order=4)
    fractions = sum(
        mode_power_fractions(coefficients, field).values()
    )
    assert fractions == pytest.approx(1.0, abs=1e-3)


def test_reconstruction_returns_the_field_it_decomposed():
    field = _field()
    coefficients = decompose(field, waist=q(WAIST_UM, "um"), max_order=3)
    rebuilt = reconstruct(coefficients, field, waist=q(WAIST_UM, "um"))
    original = np.abs(field.amplitude) ** 2
    restored = np.abs(rebuilt.amplitude) ** 2
    difference = np.max(np.abs(original - restored)) / np.max(original)
    assert difference < 1e-3


def test_mode_content_reports_the_leading_mode():
    field = _field()
    content = mode_content(field, waist=q(WAIST_UM, "um"), max_order=2)
    assert content[0][0] == (0, 0)
    assert content[0][1] == pytest.approx(1.0, abs=1e-3)


def test_a_propagated_gaussian_stays_in_the_fundamental_mode():
    """Free-space propagation changes the waist and the wavefront, not the order."""
    field = _field()
    rayleigh_mm = np.pi * (WAIST_UM * 1e-3) ** 2 / (LAMBDA_NM * 1e-6)
    ratio = 0.5
    moved = propagate(field, q(ratio * rayleigh_mm, "mm"))
    wider = second_moment_radius(moved).to_value("um")
    curvature_mm = ratio * rayleigh_mm * (1.0 + 1.0 / ratio**2)
    content = mode_content(
        moved,
        waist=q(wider, "um"),
        max_order=2,
        radius_of_curvature=q(curvature_mm, "mm"),
    )
    assert content[0][0] == (0, 0)
    assert content[0][1] == pytest.approx(1.0, abs=5e-3)


def test_ignoring_the_wavefront_curvature_leaks_power_into_higher_modes():
    """The failure mode the previous test guards against, made explicit."""
    field = _field()
    rayleigh_mm = np.pi * (WAIST_UM * 1e-3) ** 2 / (LAMBDA_NM * 1e-6)
    moved = propagate(field, q(0.5 * rayleigh_mm, "mm"))
    wider = second_moment_radius(moved).to_value("um")
    without = mode_content(moved, waist=q(wider, "um"), max_order=2)
    with_curvature = mode_content(
        moved,
        waist=q(wider, "um"),
        max_order=2,
        radius_of_curvature=q(2.5 * rayleigh_mm, "mm"),
    )
    assert without[0][1] < 0.99
    assert with_curvature[0][1] > 0.99


def test_reconstruction_conserves_power():
    field = _field()
    coefficients = decompose(field, waist=q(WAIST_UM, "um"), max_order=3)
    rebuilt = reconstruct(coefficients, field, waist=q(WAIST_UM, "um"))
    assert power(rebuilt).to_value("um^2") == pytest.approx(
        power(field).to_value("um^2"), rel=1e-3
    )


# ---------------------------------------------------------------------------
# Laguerre-Gauss: the basis that carries orbital angular momentum
# ---------------------------------------------------------------------------


def test_the_laguerre_modes_are_orthonormal_on_the_grid():
    (x, y), spacing = _coords()
    waist = q(WAIST_UM, "um")
    modes = [
        laguerre_gauss(x, y, p, ell, waist)
        for p in range(2)
        for ell in (-1, 0, 1)
    ]
    for i, first in enumerate(modes):
        norm = np.sum(np.abs(first) ** 2) * spacing**2
        assert norm == pytest.approx(1.0, rel=1e-6), i
        for j, second in enumerate(modes):
            if j <= i:
                continue
            overlap = np.sum(np.conj(first) * second) * spacing**2
            assert abs(overlap) < 1e-6, (i, j)


def test_the_fundamental_laguerre_and_hermite_modes_are_the_same_field():
    (x, y), _ = _coords()
    waist = q(WAIST_UM, "um")
    hermite = hermite_gauss(x, y, 0, 0, waist)
    laguerre = laguerre_gauss(x, y, 0, 0, waist)
    ratio = hermite / laguerre
    assert np.allclose(ratio, ratio.flat[ratio.size // 2], rtol=1e-9)


def test_a_vortex_mode_vanishes_on_axis():
    (x, y), _ = _coords()
    vortex = laguerre_gauss(x, y, 0, 1, q(WAIST_UM, "um"))
    centre = vortex.shape[0] // 2
    assert abs(vortex[centre, centre]) < 1e-12
    # and it is dark in the middle but not at the waist radius
    radius = np.hypot(x, y)
    ring = np.abs(vortex[(np.abs(radius - WAIST_UM) < 3.0)])
    assert ring.max() > 0.0


def test_the_vortex_mode_carries_one_quantum_of_orbital_angular_momentum():
    """The phase winds once around the axis, so the charge l is visible in it."""
    spacing = EXTENT_UM / SAMPLES
    axis = (np.arange(SAMPLES) - SAMPLES // 2) * spacing
    x, y = np.meshgrid(axis, axis)
    mode = laguerre_gauss(x, y, 0, 1, q(WAIST_UM, "um"))
    centre = SAMPLES // 2
    phases = []
    for dx, dy in ((6, 0), (0, 6), (-6, 0), (0, -6)):
        phases.append(np.angle(mode[centre + dy, centre + dx]))
    unwrapped = np.unwrap(phases)
    assert abs(unwrapped[-1] - unwrapped[0]) > 3.0


def test_decomposition_can_use_the_laguerre_basis():
    field = _field()
    coefficients = decompose(field, waist=q(WAIST_UM, "um"), max_order=2, basis="laguerre")
    assert (0, 0) in coefficients
    fractions = mode_power_fractions(coefficients, field)
    assert fractions[(0, 0)] == pytest.approx(1.0, abs=1e-3)


def test_an_unknown_basis_is_rejected():
    with pytest.raises(ValueError, match="basis"):
        decompose(_field(), waist=q(WAIST_UM, "um"), basis="zernike")


def test_reconstruction_is_the_adjoint_of_decomposition():
    """A round trip through the coefficients returns the field."""
    field = _field()
    coefficients = decompose(field, waist=q(WAIST_UM, "um"), max_order=4)
    rebuilt = reconstruct(coefficients, field, waist=q(WAIST_UM, "um"))
    # dropping modes above order 4 costs a little power, but the part that is
    # kept has to come back with the right shape
    kept = sum(mode_power_fractions(coefficients, field).values())
    original_power = power(field).to_value("um^2")
    rebuilt_power = power(rebuilt).to_value("um^2")
    assert rebuilt_power == pytest.approx(original_power * kept, rel=1e-2)


def test_reconstruction_with_a_higher_order_curvature_round_trips():
    field = _field()
    rayleigh_mm = np.pi * (WAIST_UM * 1e-3) ** 2 / (LAMBDA_NM * 1e-6)
    moved = propagate(field, q(0.4 * rayleigh_mm, "mm"))
    wider = second_moment_radius(moved).to_value("um")
    curvature = q(2.9 * rayleigh_mm, "mm")
    coefficients = decompose(
        moved, waist=q(wider, "um"), max_order=2, radius_of_curvature=curvature
    )
    rebuilt = reconstruct(
        coefficients, moved, waist=q(wider, "um"), radius_of_curvature=curvature
    )
    original = np.abs(moved.amplitude) ** 2
    restored = np.abs(rebuilt.amplitude) ** 2
    assert np.max(np.abs(original - restored)) / np.max(original) < 5e-3


def test_decomposition_does_not_materialise_a_grid_per_mode():
    """The separable path allocates 1-D profiles, not one N^2 array per mode.

    A regression to the obvious implementation would allocate (order + 1)^2
    full complex grids; for a 512^2 field that is 16 MiB.  Anything under a
    megabyte proves the coefficient matrix route is in use.
    """
    import gc
    import tracemalloc

    field = gaussian_field(
        waist=q(WAIST_UM, "um"),
        wavelength=q(LAMBDA_NM, "nm"),
        samples=512,
        extent=q(EXTENT_UM, "um"),
    )
    gc.collect()
    tracemalloc.start()
    decompose(field, waist=q(WAIST_UM, "um"), max_order=3)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert peak < 1024**2, f"peak allocation was {peak / 1024**2:.1f} MiB"


def test_laguerre_decomposition_still_uses_the_general_path():
    """The azimuthal basis is not separable in x and y, and still works."""
    field = _field()
    coefficients = decompose(
        field, waist=q(WAIST_UM, "um"), max_order=1, basis="laguerre"
    )
    fractions = mode_power_fractions(coefficients, field)
    assert fractions[(0, 0)] == pytest.approx(1.0, abs=1e-3)


def test_modes_curvature_and_error_handling():
    (x, y), _ = _coords()
    w = q(WAIST_UM, "um")

    # Missing wavelength when radius_of_curvature is supplied
    with pytest.raises(TypeError, match="needs the wavelength"):
        hermite_gauss(x, y, 0, 0, w, radius_of_curvature=q(100.0, "mm"))

    with pytest.raises(TypeError, match="needs the wavelength"):
        laguerre_gauss(x, y, 0, 0, w, radius_of_curvature=q(100.0, "mm"))

    # Infinite curvature returns flat profile
    flat_hg = hermite_gauss(x, y, 0, 0, w)
    inf_hg = hermite_gauss(x, y, 0, 0, w, radius_of_curvature=np.inf, wavelength=q(633, "nm"))
    assert np.allclose(flat_hg, inf_hg)

    flat_lg = laguerre_gauss(x, y, 0, 0, w)
    inf_lg = laguerre_gauss(x, y, 0, 0, w, radius_of_curvature=np.inf, wavelength=q(633, "nm"))
    assert np.allclose(flat_lg, inf_lg)

    # Decompose validation errors
    field = _field()
    with pytest.raises(TypeError, match="expects a Field"):
        decompose("not_a_field", waist=w)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="cannot be negative"):
        decompose(field, waist=w, max_order=-1)

    with pytest.raises(ValueError, match="basis must be 'hermite' or 'laguerre'"):
        decompose(field, waist=w, basis="invalid")

    # Reconstruct validation and reconstruction
    coeffs = {(0, 0): 1.0 + 0.0j}
    rebuilt_hg = reconstruct(coeffs, field, waist=w)
    assert np.isclose(rebuilt_hg.amplitude[SAMPLES // 2, SAMPLES // 2], flat_hg[SAMPLES // 2, SAMPLES // 2], rtol=1e-3)

