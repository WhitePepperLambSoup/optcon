"""Wavefront aberrations: Zernike modes, Seidel terms, and their RMS.

Conventions follow Noll (1976) for the index ordering and the normalisation,
and Mahajan, *Optical Imaging and Aberrations*, for the Seidel relations.  The
Zernike modes are orthonormal on the unit disc, which the tests verify by
numerical integration rather than by trusting the formula.
"""

import numpy as np
import pytest

from optcon import q
from optcon.aberrations import (
    defocus_coefficient,
    noll_to_nm,
    radial_polynomial,
    seidel_spherical_decomposition,
    strehl_from_zernikes,
    wavefront_pv,
    wavefront_rms,
    zernike,
    zernike_modes,
)
from optcon.diffraction import strehl_ratio


def test_noll_index_mapping_matches_the_standard_table():
    expected = {
        1: (0, 0),
        2: (1, 1),
        3: (1, -1),
        4: (2, 0),
        5: (2, -2),
        6: (2, 2),
        7: (3, -1),
        8: (3, 1),
        9: (3, -3),
        10: (3, 3),
        11: (4, 0),
    }
    for index, pair in expected.items():
        assert noll_to_nm(index) == pair


def test_piston_and_tilt_have_the_expected_shape():
    assert zernike(0, 0, 0.3, 0.0) == pytest.approx(1.0)
    # tilt is 2 rho cos(theta)
    rho, theta = 0.5, np.deg2rad(60.0)
    assert zernike(1, 1, rho, theta) == pytest.approx(2.0 * rho * np.cos(theta))


def test_defocus_is_the_familiar_quadratic():
    for rho in (0.0, 0.25, 0.5, 1.0):
        assert zernike(2, 0, rho, 0.0) == pytest.approx(np.sqrt(3.0) * (2.0 * rho**2 - 1.0))


def test_spherical_aberration_mode_matches_its_closed_form():
    for rho in (0.0, 0.3, 0.7, 1.0):
        expected = np.sqrt(5.0) * (6.0 * rho**4 - 6.0 * rho**2 + 1.0)
        assert zernike(4, 0, rho, 0.0) == pytest.approx(expected)


def test_radial_polynomial_rejects_impossible_orders():
    with pytest.raises(ValueError, match="same parity"):
        radial_polynomial(3, 0, 0.5)
    with pytest.raises(ValueError, match="magnitude"):
        radial_polynomial(2, 3, 0.5)


def _disc_grid(samples: int = 400):
    axis = np.linspace(-1.0, 1.0, samples)
    x, y = np.meshgrid(axis, axis)
    rho = np.hypot(x, y)
    theta = np.arctan2(y, x)
    inside = rho <= 1.0
    return rho, theta, inside, (axis[1] - axis[0]) ** 2


def test_the_modes_are_orthonormal_on_the_disc():
    """The definition of 'RMS wavefront' depends on this being true."""
    rho, theta, inside, cell = _disc_grid()
    modes = [(n, m) for n, m in zernike_modes(4)]
    values = [np.where(inside, zernike(n, m, rho, theta), 0.0) for n, m in modes]
    for i, first in enumerate(values):
        norm = np.sum(first**2) * cell / np.pi
        assert norm == pytest.approx(1.0, abs=5e-3), modes[i]
        for j, second in enumerate(values):
            if j <= i:
                continue
            overlap = np.sum(first * second) * cell / np.pi
            assert abs(overlap) < 5e-3, (modes[i], modes[j])


def test_wavefront_rms_is_the_quadrature_sum_of_coefficients():
    coefficients = [0.0, 0.1, -0.2, 0.05]
    expected = np.sqrt(sum(value**2 for value in coefficients))
    assert wavefront_rms(coefficients, wavelength=q(633.0, "nm")).to_value("nm") == pytest.approx(
        expected
    )


def test_wavefront_pv_of_a_single_defocus_mode():
    # defocus ranges over [-sqrt(3), +sqrt(3)], so PV = 2 sqrt(3) * coefficient
    value = wavefront_pv([0.0, 0.0, 0.0, 0.1], wavelength=q(633.0, "nm"))
    # PV is found by sampling the disc, and a square grid never lands exactly
    # on rho = 1, so the extreme is reached to about a part in 1e4
    assert value.to_value("nm") == pytest.approx(2.0 * np.sqrt(3.0) * 0.1, rel=1e-3)


def test_strehl_from_zernikes_agrees_with_the_marechal_formula():
    """Two modules, one number: the Zernike RMS and the wavefront RMS."""
    wavelength = q(633.0, "nm")
    coefficients = [0.0, 0.0, 0.0, 20.0, 0.0, 0.0, 0.0, 0.0]
    rms_nm = np.sqrt(sum(value**2 for value in coefficients))
    from_zernikes = strehl_from_zernikes(coefficients, wavelength)
    from_formula = strehl_ratio(q(rms_nm, "nm"), wavelength)
    assert from_zernikes == pytest.approx(from_formula)


def test_a_perfect_wavefront_has_unit_strehl():
    assert strehl_from_zernikes([0.0] * 11, q(633.0, "nm")) == pytest.approx(1.0)


def test_seidel_spherical_aberration_decomposes_onto_zernikes():
    """rho^4 = Z11/(6 sqrt 5) + Z4/(2 sqrt 3) + Z1/3, checked numerically."""
    wavelength = q(633.0, "nm")
    amplitude = 1.0
    coefficients = seidel_spherical_decomposition(amplitude)
    rho, theta, inside, _ = _disc_grid(120)
    reconstructed = np.zeros_like(rho)
    for index, value in enumerate(coefficients, start=1):
        n, m = noll_to_nm(index)
        reconstructed += value * zernike(n, m, rho, theta)
    error = np.max(np.abs(reconstructed[inside] - amplitude * rho[inside] ** 4))
    assert error < 2e-3
    # and the RMS of the pure rho^4 term is what the coefficients say
    assert wavefront_rms(coefficients, wavelength).to_value("nm") == pytest.approx(
        amplitude * np.sqrt(1.0 / 5.0 - 1.0 / 9.0), rel=5e-3
    )


def test_defocus_coefficient_scales_with_the_wavefront_error():
    # the mode spans -sqrt(3) to +sqrt(3), so a PV of W is W / (2 sqrt 3)
    assert defocus_coefficient(0.5) == pytest.approx(0.5 / (2.0 * np.sqrt(3.0)))
