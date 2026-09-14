"""Modulation transfer function: the imaging metric, two ways.

The diffraction MTF of a circular aperture has a closed form.  The same
quantity can be obtained by Fourier transforming the Airy point spread
function.  Those two routes share no algebra, so requiring them to agree is
the strongest available check on either.
"""

import numpy as np
import pytest

from optcon import q
from optcon.mtf import (
    circular_aperture_mtf,
    cutoff_frequency,
    mtf_from_psf,
)

LAMBDA_NM = 550.0
F_NUMBER = 4.0


def test_cutoff_frequency_is_one_over_lambda_f_number():
    value = cutoff_frequency(F_NUMBER, q(LAMBDA_NM, "nm"))
    assert value.to_value("1/mm") == pytest.approx(1.0 / (550e-6 * F_NUMBER))


def test_the_diffraction_mtf_starts_at_one_and_ends_at_zero():
    assert circular_aperture_mtf(0.0, F_NUMBER, q(LAMBDA_NM, "nm")) == pytest.approx(1.0)
    cutoff = cutoff_frequency(F_NUMBER, q(LAMBDA_NM, "nm")).to_value("1/mm")
    assert circular_aperture_mtf(cutoff, F_NUMBER, q(LAMBDA_NM, "nm")) == pytest.approx(
        0.0, abs=1e-12
    )
    assert circular_aperture_mtf(cutoff * 1.5, F_NUMBER, q(LAMBDA_NM, "nm")) == pytest.approx(
        0.0
    )


def test_the_diffraction_mtf_decreases_monotonically():
    cutoff = cutoff_frequency(F_NUMBER, q(LAMBDA_NM, "nm")).to_value("1/mm")
    values = [
        circular_aperture_mtf(fraction * cutoff, F_NUMBER, q(LAMBDA_NM, "nm"))
        for fraction in np.linspace(0.0, 1.0, 25)
    ]
    assert values == sorted(values, reverse=True)


def test_the_mtf_matches_its_closed_form_at_a_known_point():
    cutoff = cutoff_frequency(F_NUMBER, q(LAMBDA_NM, "nm")).to_value("1/mm")
    half = 0.5 * cutoff
    expected = (2.0 / np.pi) * (np.arccos(0.5) - 0.5 * np.sqrt(1.0 - 0.25))
    assert circular_aperture_mtf(half, F_NUMBER, q(LAMBDA_NM, "nm")) == pytest.approx(
        expected
    )


def test_mtf_from_psf_is_normalised_to_one_at_zero_frequency():
    psf = np.exp(-((np.arange(64)[None, :] - 32) ** 2) / 32.0) * np.ones((64, 1))
    frequencies, values = mtf_from_psf(psf, q(1.0, "um"))
    centre = 32
    assert values[centre, centre] == pytest.approx(1.0)
    assert values.max() == pytest.approx(1.0)
    assert frequencies.to_value("1/um").max() > 0.0


def test_fourier_transforming_the_airy_pattern_reproduces_the_closed_form():
    """Two independent routes to the same curve, required to agree."""
    from optcon.diffraction import airy_intensity

    samples = 1024
    # the first Airy zero sits at 1.22 lambda f/# = 2.7 um, so the grid has to
    # be fine enough to put several samples inside the central lobe
    extent_mm = 1024 * 2.5e-4
    spacing = extent_mm / samples
    axis = (np.arange(samples) - samples // 2) * spacing
    x, y = np.meshgrid(axis, axis)
    # u = pi r / (lambda f/#) is the Airy argument, so invert it for radius
    u = np.pi * np.hypot(x, y) / (550e-6 * F_NUMBER)
    psf = airy_intensity(u)
    frequencies, values = mtf_from_psf(psf, q(spacing, "mm"))
    frequencies_1_per_mm = np.asarray(frequencies.to_value("1/mm"))
    centre = samples // 2
    for fraction in (0.1, 0.3, 0.5, 0.7):
        cutoff = 1.0 / (550e-6 * F_NUMBER)
        index = int(np.argmin(np.abs(frequencies_1_per_mm[centre:] - fraction * cutoff)))
        measured = float(values[centre, centre + index])
        expected = circular_aperture_mtf(
            fraction * cutoff, F_NUMBER, q(LAMBDA_NM, "nm")
        )
        assert measured == pytest.approx(expected, abs=5e-3), fraction


def test_a_gaussian_psf_gives_a_gaussian_mtf():
    """The Fourier transform of a Gaussian is a Gaussian."""
    samples = 512
    spacing = 0.5
    axis = (np.arange(samples) - samples // 2) * spacing
    sigma = 4.0
    x, y = np.meshgrid(axis, axis)
    psf = np.exp(-(x**2 + y**2) / (2.0 * sigma**2))
    frequencies, values = mtf_from_psf(psf, q(spacing, "um"))
    frequencies_1_per_um = np.asarray(frequencies.to_value("1/um"))
    centre = samples // 2
    # the transform of exp(-r^2/2s^2) is exp(-2 pi^2 s^2 f^2)
    for offset in (5, 10, 20):
        f = abs(frequencies_1_per_um[centre + offset])
        expected = np.exp(-2.0 * np.pi**2 * sigma**2 * f**2)
        assert float(values[centre, centre + offset]) == pytest.approx(expected, rel=5e-3)
