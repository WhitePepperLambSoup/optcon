"""Tests for the Fox-Li cavity diffraction integral operator."""

import numpy as np
import pytest

from optcon import q
from optcon.checks import is_passive, is_reciprocal
from optcon.errors import DimensionError
from optcon.fox_li import (
    FoxLiResonator,
    fox_li_operator,
    fox_li_power_iteration,
    fresnel_number,
    solve_fox_li_modes,
)


def test_fox_li_fresnel_number():
    # a = 1 mm, lambda = 1000 nm, L = 1 m -> N_F = (1e-3)^2 / (1e-6 * 1) = 1.0
    res = FoxLiResonator(
        wavelength=q(1000.0, "nm"),
        length=q(1.0, "m"),
        aperture=q(1.0, "mm"),
        g1=1.0,
        g2=1.0,
    )
    assert fresnel_number(res) == pytest.approx(1.0, rel=1e-9)

    # a = 2 mm -> N_F = 4.0
    res4 = FoxLiResonator(
        wavelength=q(1000.0, "nm"),
        length=q(1.0, "m"),
        aperture=q(2.0, "mm"),
    )
    assert fresnel_number(res4) == pytest.approx(4.0, rel=1e-9)


def test_fox_li_dimension_validation():
    # Length passed where angle expected
    with pytest.raises(DimensionError):
        FoxLiResonator(
            wavelength=q(1064.0, "nm"),
            length=q(100.0, "mm"),
            aperture=q(1.0, "mm"),
            tilt=q(10.0, "mm"),  # Invalid: tilt must be an angle
        )

    # Angle passed where length expected
    with pytest.raises(DimensionError):
        FoxLiResonator(
            wavelength=q(1064.0, "nm"),
            length=q(0.5, "rad"),  # Invalid: length must be a length
            aperture=q(1.0, "mm"),
        )


def test_fox_li_passivity_and_reciprocity_contracts():
    res = FoxLiResonator(
        wavelength=q(1064.0, "nm"),
        length=q(500.0, "mm"),
        aperture=q(1.5, "mm"),
        g1=0.8,
        g2=0.8,
        geometry="strip",
    )
    k_sym, coords, weights = fox_li_operator(res, num_points=64)

    # Invariant contracts
    assert is_passive(k_sym, rtol=1e-4, atol=1e-4)
    assert is_reciprocal(k_sym, rtol=1e-4, atol=1e-4)

    # Operator is contractive: sigma_max <= 1
    sv = np.linalg.svd(k_sym, compute_uv=False)
    assert sv[0] <= 1.0 + 1e-4


def test_fox_li_eigenmodes_monotonic_loss():
    res = FoxLiResonator(
        wavelength=q(1064.0, "nm"),
        length=q(200.0, "mm"),
        aperture=q(1.0, "mm"),
        g1=0.9,
        g2=0.9,
        geometry="strip",
    )
    modes = solve_fox_li_modes(res, num_modes=4, num_points=64)
    assert len(modes) == 4

    # Monotonically increasing diffraction loss: loss(TEM0) < loss(TEM1) < loss(TEM2)
    losses = [m.loss for m in modes]
    assert losses[0] < losses[1] < losses[2] < losses[3]

    # Every mode loss is strictly in [0, 1)
    for m in modes:
        assert 0.0 <= m.loss < 1.0
        assert 0.0 <= m.round_trip_loss < 1.0

    # Symmetry check: TEM0 is even, TEM1 is odd
    u0 = modes[0].amplitude
    u1 = modes[1].amplitude
    # For even mode: u(x) == u(-x)
    assert np.allclose(u0, u0[::-1], rtol=1e-2, atol=1e-2)
    # For odd mode: u(x) == -u(-x)
    assert np.allclose(u1, -u1[::-1], rtol=1e-2, atol=1e-2)


def test_fox_li_power_iteration_matches_eigensolver():
    # In the diffraction-discriminating regime (N_F ~ 1.5 - 2.0),
    # the fundamental mode separates cleanly and power iteration converges quickly.
    res = FoxLiResonator(
        wavelength=q(1064.0, "nm"),
        length=q(500.0, "mm"),
        aperture=q(1.0, "mm"),  # N_F = 1.0^2 / (0.5 * 1.064) = 1.88
        g1=1.0,
        g2=1.0,
        geometry="strip",
    )
    eigenmodes = solve_fox_li_modes(res, num_modes=1, num_points=64)
    dominant_mode, history = fox_li_power_iteration(
        res, max_iter=200, tol=1e-7, num_points=64
    )

    # Power iteration should match the analytical eigensolver eigenvalue
    assert dominant_mode.loss == pytest.approx(eigenmodes[0].loss, rel=1e-4)
    assert dominant_mode.phase_shift == pytest.approx(
        eigenmodes[0].phase_shift, abs=1e-3
    )

    # Amplitudes should match up to a global phase
    weights = fox_li_operator(res, num_points=64)[2]
    int_overlap = abs(
        np.sum(weights * np.conj(dominant_mode.amplitude) * eigenmodes[0].amplitude)
    )
    assert int_overlap == pytest.approx(1.0, rel=1e-2)


def test_fox_li_confocal_low_diffraction_loss():
    # Confocal cavity g1 = g2 = 0 has well-known ultra-low diffraction loss
    res = FoxLiResonator(
        wavelength=q(1000.0, "nm"),
        length=q(1.0, "m"),
        aperture=q(1.0, "mm"),
        g1=0.0,
        g2=0.0,
        geometry="strip",
    )
    assert fresnel_number(res) == pytest.approx(1.0, rel=1e-9)

    modes = solve_fox_li_modes(res, num_modes=1, num_points=64)
    # Slepian & Pollak: for N_F = 1 confocal, loss is under 0.1% per transit
    assert modes[0].loss < 0.005


def test_fox_li_flat_flat_loss_scaling():
    # Flat-flat strip resonator (g=1): loss decreases as N_F increases
    res1 = FoxLiResonator(
        wavelength=q(1000.0, "nm"),
        length=q(1.0, "m"),
        aperture=q(1.0, "mm"),
        g1=1.0,
        g2=1.0,
    )
    res2 = FoxLiResonator(
        wavelength=q(1000.0, "nm"),
        length=q(1.0, "m"),
        aperture=q(1.41421356, "mm"),  # N_F = 2.0
        g1=1.0,
        g2=1.0,
    )

    mode1 = solve_fox_li_modes(res1, num_modes=1, num_points=64)[0]
    mode2 = solve_fox_li_modes(res2, num_modes=1, num_points=64)[0]

    # Fox & Li (1961): N_F=1 loss ~ 8%, N_F=2 loss ~ 3-4%
    assert mode1.loss > mode2.loss
    assert 0.06 < mode1.loss < 0.11
    assert 0.02 < mode2.loss < 0.06


def test_fox_li_circular_geometry():
    res = FoxLiResonator(
        wavelength=q(1064.0, "nm"),
        length=q(250.0, "mm"),
        aperture=q(1.0, "mm"),
        g1=0.8,
        g2=0.8,
        geometry="circular",
    )
    modes_m0 = solve_fox_li_modes(
        res, num_modes=2, num_points=64, azimuthal_order=0
    )
    modes_m1 = solve_fox_li_modes(
        res, num_modes=2, num_points=64, azimuthal_order=1
    )

    # Passivity
    assert modes_m0[0].loss >= 0.0
    assert modes_m1[0].loss >= 0.0

    # Higher azimuthal mode m=1 has higher diffraction loss than fundamental m=0
    assert modes_m0[0].loss < modes_m1[0].loss

    # Normalization check: int 2 pi r |u|^2 dr = 1
    _, _, weights = fox_li_operator(res, num_points=64, azimuthal_order=0)
    norm = np.sum(weights * np.abs(modes_m0[0].amplitude) ** 2)
    assert norm == pytest.approx(1.0, rel=1e-3)


def test_fox_li_angular_tilt_increases_loss():
    res_aligned = FoxLiResonator(
        wavelength=q(1064.0, "nm"),
        length=q(500.0, "mm"),
        aperture=q(1.5, "mm"),
        g1=0.9,
        g2=0.9,
        tilt=None,
    )
    res_tilted = FoxLiResonator(
        wavelength=q(1064.0, "nm"),
        length=q(500.0, "mm"),
        aperture=q(1.5, "mm"),
        g1=0.9,
        g2=0.9,
        tilt=q(200.0, "urad"),  # 200 microradians
    )

    mode_aligned = solve_fox_li_modes(
        res_aligned, num_modes=1, num_points=64, check_contracts=True
    )[0]
    # For tilted resonator, reciprocity is broken, but passivity still holds
    mode_tilted = solve_fox_li_modes(
        res_tilted, num_modes=1, num_points=64, check_contracts=True
    )[0]

    # Tilt increases diffraction clipping loss
    assert mode_tilted.loss > mode_aligned.loss

    # Tilt displaces the mode centroid: int x |u|^2 dx != 0
    coords = mode_tilted.coordinates
    weights = fox_li_operator(res_tilted, num_points=64)[2]
    intensity_tilted = np.abs(mode_tilted.amplitude) ** 2
    centroid = np.sum(coords * weights * intensity_tilted)
    assert abs(centroid) > 1e-6  # Displaced from origin
