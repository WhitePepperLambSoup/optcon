"""Modulation transfer function of an imaging system.

Two routes are provided, and they are meant to agree:

* :func:`circular_aperture_mtf` is the closed form for an unaberrated
  circular pupil,

      MTF(nu) = (2/pi) [ arccos(nu) - nu sqrt(1 - nu^2) ],   nu = f / f_c,

  with cutoff ``f_c = 1/(lambda f/#)`` (Williams & Becklund, *Introduction to
  the Optical Transfer Function*).
* :func:`mtf_from_psf` takes the Fourier transform of any point spread
  function, which is the definition when the system is not a perfect pupil.

Frequencies come back as a quantity per unit length, so a PSF sampled in
micrometres yields line pairs per micrometre without any hand conversion.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy import fft as _fft

from .errors import DimensionError
from .quantity import Quantity
from .units import Unit
from .units import unit as _unit

_LENGTH = _unit("m").dimension


def _length_of(value: Any, context: str) -> tuple[float, Unit]:
    if isinstance(value, Quantity):
        if value.unit.dimension != _LENGTH:
            raise DimensionError(f"{context}: expected a length, got {value.unit}")
        return float(value.value), value.unit
    raise TypeError(
        f"{context}: expected a length with units, e.g. q(550, 'nm'), "
        f"got a bare {type(value).__name__}"
    )


def cutoff_frequency(f_number_value: float, wavelength: Any) -> Quantity:
    """Highest spatial frequency a circular pupil transmits, ``1/(lambda f/#)``."""
    lam, unit = _length_of(wavelength, "cutoff_frequency")
    if f_number_value <= 0.0:
        raise ValueError("the f-number must be positive")
    return Quantity(1.0 / (lam * float(f_number_value)), 1.0 / unit)


def circular_aperture_mtf(
    frequency: Any, f_number_value: float, wavelength: Any
) -> float:
    """Diffraction MTF of an unaberrated circular pupil.

    ``frequency`` may be a quantity or a bare number, and a bare number is
    read as **cycles per millimetre** - the unit MTF curves are quoted in,
    rather than the inverse of whatever unit the wavelength happened to use.
    """
    cutoff = cutoff_frequency(f_number_value, wavelength)
    cutoff_per_mm = cutoff.to_value("1/mm")
    if isinstance(frequency, Quantity):
        value = abs(float(frequency.to_value("1/mm")))
    else:
        value = abs(float(frequency))
    normalised = value / cutoff_per_mm
    if normalised >= 1.0:
        return 0.0
    return float(
        (2.0 / math.pi) * (math.acos(normalised) - normalised * math.sqrt(1.0 - normalised**2))
    )


def mtf_from_psf(psf, spacing: Any) -> tuple[Quantity, np.ndarray]:
    """Modulation transfer function from a sampled point spread function.

    Returns the frequency axis (as a quantity, in inverse length) and the MTF
    normalised to one at zero frequency.

    The input must already be an **intensity**: real and non-negative.  A
    complex field is rejected rather than silently squared, because squaring
    an intensity halves the apparent width in the frequency domain.
    """
    values = np.asarray(psf)
    if values.ndim != 2:
        raise ValueError(f"a point spread function must be 2-D, got shape {values.shape}")
    if np.iscomplexobj(values):
        raise TypeError(
            "mtf_from_psf expects an intensity array; pass abs(field)**2 for a "
            "complex field"
        )
    pitch, unit = _length_of(spacing, "mtf_from_psf")
    if pitch <= 0.0:
        raise ValueError("the sample spacing must be positive")
    intensity_values = np.asarray(values, dtype=float)
    if np.any(intensity_values < 0.0):
        raise ValueError("an intensity distribution cannot be negative")
    total = float(np.sum(intensity_values))
    if total <= 0.0:
        raise ValueError("this point spread function carries no energy")
    # fftshift(fft2(ifftshift(x))) equals fftshift(fft2(x)) times an
    # alternating sign, so the input does not have to be copied just to move
    # its origin to the corner for the FFT
    transform = _fft.fftshift(_fft.fft2(intensity_values))
    indices = np.arange(values.shape[0])
    alternating = np.where(((indices[:, None] + indices[None, :]) % 2) == 0, 1.0, -1.0)
    transform = transform * alternating
    magnitude = np.abs(transform) / total
    frequencies = _fft.fftshift(_fft.fftfreq(values.shape[0], d=pitch))
    return Quantity(frequencies, 1.0 / unit), magnitude


__all__ = ["circular_aperture_mtf", "cutoff_frequency", "mtf_from_psf"]
