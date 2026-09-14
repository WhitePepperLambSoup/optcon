"""Vector fields: a spatial profile that also carries a polarisation state.

A scalar field has one complex array; a vector field has two, one per
transverse component.  They are separate types on purpose: a scalar
calculation cannot quietly drop polarisation, and the conversion has to be
asked for with :func:`from_scalar`.

In the paraxial and weakly guiding limit the two transverse components
propagate independently, so :func:`propagate_vector` is simply
:func:`optcon.propagation.propagate` applied to each - which is also the
assumption under which the polarisation state is preserved during
propagation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .propagation import Field
from .propagation import propagate as propagate_scalar
from .quantity import Quantity
from .units import unit as _unit

_DIMENSIONLESS = _unit("1")


@dataclass(frozen=True)
class VectorField:
    """Two transverse components on a shared square grid."""

    ex: np.ndarray
    ey: np.ndarray
    spacing: Quantity
    wavelength: Quantity

    def __post_init__(self) -> None:
        first = np.asarray(self.ex, dtype=complex)
        second = np.asarray(self.ey, dtype=complex)
        if first.shape != second.shape:
            raise ValueError(
                f"both components must have the same shape, got {first.shape} "
                f"and {second.shape}"
            )
        if first.ndim != 2 or first.shape[0] != first.shape[1]:
            raise ValueError(
                f"a vector field needs square 2-D components, got {first.shape}"
            )
        object.__setattr__(self, "ex", first)
        object.__setattr__(self, "ey", second)

    @property
    def samples(self) -> int:
        return int(self.ex.shape[0])

    @property
    def coordinates(self) -> np.ndarray:
        return (np.arange(self.samples) - self.samples // 2) * self.spacing.value

    def component(self, which: str) -> Field:
        """One component as an ordinary scalar field."""
        if which == "x":
            return Field(self.ex, self.spacing, self.wavelength)
        if which == "y":
            return Field(self.ey, self.spacing, self.wavelength)
        raise ValueError(f"component must be 'x' or 'y', got {which!r}")


def from_scalar(field: Field, angle: Any = 0.0) -> VectorField:
    """Give a scalar field a uniform linear polarisation at ``angle``.

    ``angle`` is an unnormalised angle in radians, matching the amplitude
    convention used throughout the polarisation module: the total intensity is
    preserved, since ``cos^2 + sin^2 = 1``.
    """
    if not isinstance(field, Field):
        raise TypeError("from_scalar expects a Field")
    if isinstance(angle, Quantity):
        angle = float(angle.to_value("rad"))
    theta = float(angle)
    return VectorField(
        field.amplitude * np.cos(theta),
        field.amplitude * np.sin(theta),
        field.spacing,
        field.wavelength,
    )


def intensity_map(field: VectorField) -> np.ndarray:
    """``|Ex|^2 + |Ey|^2`` at every point."""
    if not isinstance(field, VectorField):
        raise TypeError("intensity_map expects a VectorField")
    return np.abs(field.ex) ** 2 + np.abs(field.ey) ** 2


def stokes_map(field: VectorField, mask_threshold: float = 0.0) -> dict[str, np.ndarray]:
    """Stokes parameters at every point, with the same convention as the
    polarisation module: ``S3 = 2 Im(Ex conj(Ey))``."""
    if not isinstance(field, VectorField):
        raise TypeError("stokes_map expects a VectorField")
    ex, ey = field.ex, field.ey
    cross = ex * np.conj(ey)
    parameters = {
        "S0": np.abs(ex) ** 2 + np.abs(ey) ** 2,
        "S1": np.abs(ex) ** 2 - np.abs(ey) ** 2,
        "S2": 2.0 * cross.real,
        "S3": 2.0 * cross.imag,
    }
    if mask_threshold:
        parameters["mask"] = parameters["S0"] > mask_threshold
    return parameters


def polarization_map(field: VectorField) -> np.ndarray:
    """Azimuth of the local linear polarisation, in radians.

    Meaningful where the state is linear; for a general state use
    :func:`stokes_map`, which carries the full information.
    """
    return np.arctan2(field.ey.real, field.ex.real)


def analyzer_transmission(field: VectorField, analyzer_angle: Any) -> np.ndarray:
    """Intensity transmitted by a linear analyzer at ``analyzer_angle``."""
    theta = (
        float(analyzer_angle.to_value("rad"))
        if isinstance(analyzer_angle, Quantity)
        else float(analyzer_angle)
    )
    projection = field.ex * np.cos(theta) + field.ey * np.sin(theta)
    return np.abs(projection) ** 2


def propagate_vector(
    field: VectorField, distance: Any, method: str = "angular_spectrum"
) -> VectorField:
    """Propagate both components; the polarisation state is carried along."""
    if not isinstance(field, VectorField):
        raise TypeError("propagate_vector expects a VectorField")
    moved_x = propagate_scalar(
        Field(field.ex, field.spacing, field.wavelength), distance, method
    )
    moved_y = propagate_scalar(
        Field(field.ey, field.spacing, field.wavelength), distance, method
    )
    return VectorField(moved_x.amplitude, moved_y.amplitude, field.spacing, field.wavelength)


__all__ = [
    "VectorField",
    "analyzer_transmission",
    "from_scalar",
    "intensity_map",
    "polarization_map",
    "propagate_vector",
    "stokes_map",
]
