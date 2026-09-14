"""A scalar field on a sampled grid, and FFT propagation of it.

The field representation is the first piece of a layer this library treats as
first class: a :class:`Field` carries the pitch of the grid it lives on and
the wavelength it is written for, so the discretisation travels with the data
instead of living in the caller's head.  Mixing a sampled field with a modal
description - or propagating on a grid whose sampling cannot support the
distance - is then something the code can notice.

Two propagators are provided, both as transfer functions on the angular
spectrum:

``angular_spectrum``
    ``H = exp(i k z sqrt(1 - (lambda f)^2))``, exact within scalar theory and
    including evanescent decay.
``fresnel``
    ``H = exp(i k z) exp(-i pi lambda z f^2)``, the paraxial limit.

Both are evaluated on a fixed grid, which is valid while
``z <= extent^2 / (N lambda)``; beyond that the transfer function is
undersampled and a scaled or two-step method is required.  ``max_propagation``
reports the limit for a given field so the caller can check before trusting a
result.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
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
        f"{context}: expected a length with units, e.g. q(50, 'um'), "
        f"got a bare {type(value).__name__}"
    )


@dataclass(frozen=True)
class Field:
    """A complex scalar field sampled on a square, uniformly spaced grid.

    ``amplitude`` carries arbitrary amplitude units, so ``power`` comes out in
    units of area.  The grid is square and centred on the origin.
    """

    amplitude: np.ndarray
    spacing: Quantity
    wavelength: Quantity

    def __post_init__(self) -> None:
        array = np.asarray(self.amplitude, dtype=complex)
        if array.ndim != 2 or array.shape[0] != array.shape[1]:
            raise ValueError(
                f"a field must be a two-dimensional square array, got shape {array.shape}"
            )
        object.__setattr__(self, "amplitude", array)

    @property
    def samples(self) -> int:
        return int(self.amplitude.shape[0])

    @property
    def extent(self) -> Quantity:
        """Physical width of the grid."""
        return Quantity(self.samples * self.spacing.value, self.spacing.unit)

    @property
    def coordinates(self) -> np.ndarray:
        """The centred coordinate axis, in the grid's own unit."""
        axis = (np.arange(self.samples) - self.samples // 2) * self.spacing.value
        return axis

    def amplitude_order(self) -> int:
        """Field amplitudes are order 1, which is what makes intensity order 2."""
        return 1


def gaussian_field(waist: Any, wavelength: Any, samples: int, extent: Any) -> Field:
    """A TEM00 Gaussian beam of the given 1/e^2 radius, sampled on a grid."""
    w0, w0_unit = _length_of(waist, "gaussian_field")
    lam, lam_unit = _length_of(wavelength, "gaussian_field")
    lam = lam_unit.to_value(lam, w0_unit)
    width, _ = _length_of(extent, "gaussian_field")
    if w0 <= 0.0 or lam <= 0.0 or width <= 0.0:
        raise ValueError("the waist, wavelength and extent must all be positive")
    if samples < 8:
        raise ValueError(f"a field needs a usable grid, got {samples} samples")
    spacing = width / samples
    axis = (np.arange(samples) - samples // 2) * spacing
    # separable, so one 1-D exponential per axis rather than an N^2 array
    profile = np.exp(-(axis**2) / w0**2)
    amplitude = np.outer(profile, profile).astype(complex)
    return Field(
        amplitude,
        Quantity(spacing, w0_unit),
        Quantity(lam, w0_unit),
    )


def intensity(field: Field) -> np.ndarray:
    """``|u|^2``, the quantity that becomes a power ratio once normalised."""
    if not isinstance(field, Field):
        raise TypeError("intensity expects a Field")
    return np.abs(field.amplitude) ** 2


def power(field: Field) -> Quantity:
    """Total power, in units of the grid cell area."""
    total = float(np.sum(intensity(field))) * field.spacing.value**2
    return Quantity(total, field.spacing.unit**2)


def second_moment_radius(field: Field) -> Quantity:
    """The D4-sigma radius, ``2 sqrt(<x^2>)``, taken from the x marginal.

    For a Gaussian this is exactly the 1/e^2 radius, which is what makes it
    comparable with the closed-form ``w(z)``.
    """
    if not isinstance(field, Field):
        raise TypeError("second_moment_radius expects a Field")
    values = intensity(field)
    total = float(np.sum(values))
    if total <= 0.0:
        raise ValueError("this field carries no power, so it has no radius")
    axis = field.coordinates
    marginal = np.sum(values, axis=1)
    centroid = float(np.sum(marginal * axis) / np.sum(marginal))
    variance = float(np.sum(marginal * (axis - centroid) ** 2) / np.sum(marginal))
    return Quantity(2.0 * np.sqrt(variance), field.spacing.unit)


def max_propagation(field: Field) -> Quantity:
    """Distance beyond which the transfer function is undersampled."""
    extent = field.extent.value
    wavelength = field.wavelength.value
    limit = extent**2 / (field.samples * wavelength)
    return Quantity(limit, field.spacing.unit)


def _transfer_function(field: Field, distance: float, method: str) -> np.ndarray:
    frequencies = _frequency_axis(field.samples, field.spacing.value)
    wavelength = field.wavelength.value
    if method == "fresnel":
        # exp(-i pi lambda z (fx^2 + fy^2)) is separable, so it costs two 1-D
        # exponentials instead of one N^2 array of them
        tilt = np.exp(-1.0j * np.pi * wavelength * distance * frequencies**2)
        return np.exp(2.0j * np.pi * distance / wavelength) * np.outer(tilt, tilt)
    # angular spectrum: kz = sqrt(1/lambda^2 - fx^2 - fy^2), imaginary beyond
    # the light line where the wave is evanescent and decays instead of
    # accumulating phase
    squared = frequencies**2
    argument = 1.0 / wavelength**2 - (squared[None, :] + squared[:, None])
    magnitude = np.sqrt(np.abs(argument))
    propagating = argument > 0.0
    direction = 1.0j * (1.0 if distance >= 0.0 else -1.0)
    exponent = np.where(propagating, direction, -1.0) * (
        2.0 * np.pi * abs(distance) * magnitude
    )
    return np.exp(exponent)


@lru_cache(maxsize=64)
def _frequency_axis(samples: int, spacing: float) -> np.ndarray:
    """Spatial frequency axis, cached: it depends only on the grid."""
    return _fft.fftfreq(samples, d=spacing)


def propagate(field: Field, distance: Any, method: str = "angular_spectrum") -> Field:
    """Propagate ``field`` by ``distance`` along the optical axis."""
    if not isinstance(field, Field):
        raise TypeError("propagate expects a Field")
    if method not in {"angular_spectrum", "fresnel"}:
        raise ValueError(
            f"method must be 'angular_spectrum' or 'fresnel', got {method!r}"
        )
    magnitude, unit = _length_of(distance, "propagate")
    local = unit.to_value(magnitude, field.spacing.unit)
    if local == 0.0:
        return field
    # scipy.fft is the same pocketfft algorithm behind numpy's, but several
    # times faster for these sizes on this build, and it can overwrite
    # temporaries that numpy's wrapper copies
    spectrum = _fft.fft2(field.amplitude)
    propagated = _fft.ifft2(spectrum * _transfer_function(field, local, method))
    return Field(propagated, field.spacing, field.wavelength)


__all__ = [
    "Field",
    "gaussian_field",
    "intensity",
    "max_propagation",
    "power",
    "propagate",
    "second_moment_radius",
]
