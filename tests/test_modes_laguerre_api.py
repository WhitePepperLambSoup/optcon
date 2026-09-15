"""Regression tests for the complete Laguerre-Gauss modal API."""

import numpy as np
import pytest

from optcon import q
from optcon.modes import decompose, laguerre_gauss, mode_content, reconstruct
from optcon.propagation import Field


def _grid(samples: int = 128, extent_um: float = 1600.0):
    spacing = extent_um / samples
    axis = (np.arange(samples) - samples // 2) * spacing
    return np.meshgrid(axis, axis), spacing


def test_laguerre_mode_content_and_reconstruction_round_trip():
    (x, y), spacing = _grid()
    wavelength = q(633.0, "nm")
    waist = q(50.0, "um")
    amplitude = laguerre_gauss(x, y, 1, -1, waist)
    field = Field(amplitude, q(spacing, "um"), wavelength)

    coefficients = decompose(field, waist=waist, max_order=2, basis="laguerre")
    content = mode_content(field, waist=waist, max_order=2, basis="laguerre")
    rebuilt = reconstruct(
        coefficients,
        field,
        waist=waist,
        basis="laguerre",
    )

    assert content[0][0] == (1, -1)
    assert content[0][1] == pytest.approx(1.0, abs=2e-3)
    assert np.max(np.abs(rebuilt.amplitude - amplitude)) < 2e-5


def test_laguerre_reconstruction_preserves_negative_azimuthal_charge():
    (x, y), spacing = _grid()
    waist = q(50.0, "um")
    field = Field(
        laguerre_gauss(x, y, 0, -2, waist),
        q(spacing, "um"),
        q(633.0, "nm"),
    )

    rebuilt = reconstruct({(0, -2): 1.0 + 0.0j}, field, waist, basis="laguerre")
    assert np.max(np.abs(rebuilt.amplitude - field.amplitude)) < 2e-5


def test_reconstruct_rejects_malformed_laguerre_indices():
    (x, y), spacing = _grid(samples=64)
    field = Field(
        laguerre_gauss(x, y, 0, 0, q(50.0, "um")),
        q(spacing, "um"),
        q(633.0, "nm"),
    )

    with pytest.raises(ValueError, match="radial index"):
        reconstruct({(-1, 0): 1.0 + 0.0j}, field, q(50.0, "um"), basis="laguerre")

    with pytest.raises(ValueError, match="basis must"):
        reconstruct({(0, 0): 1.0 + 0.0j}, field, q(50.0, "um"), basis="zernike")
