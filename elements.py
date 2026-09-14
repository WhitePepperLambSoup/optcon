"""Ray-transfer (ABCD) matrices for the elements an optical bench is built from.

An ABCD matrix mixes lengths and angles, so it is not a homogeneous object:
its ``B`` entry carries a length and its ``C`` entry the reciprocal.  Rather
than pretend otherwise, :class:`TransferMatrix` carries the unit in which the
off-diagonal entries are expressed, and refuses to compose matrices whose
units are not compatible.  That is what lets a chain written partly in
millimetres and partly in metres compose without a hand conversion, and it is
what stops a focal length in seconds from ever reaching a matrix.

Conventions follow Gerrard & Burch, *Introduction to Matrix Methods in
Optics* (1975), and Siegman, *Lasers* (1986), ch. 15: the ray vector is
``(y, theta)`` with ``theta`` the paraxial angle, and matrices act on it from
the left in beam order.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .errors import DimensionError
from .quantity import Quantity
from .units import Unit
from .units import unit as _unit


def _length_unit_of(value: Any, context: str) -> tuple[float, Unit]:
    """Return ``(magnitude, unit)`` for a length, refusing bare numbers."""
    if isinstance(value, Quantity):
        if value.unit.dimension != _unit("m").dimension:
            raise DimensionError(f"{context}: expected a length, got {value.unit}")
        return float(value.value), value.unit
    raise TypeError(
        f"{context}: expected a length with units, e.g. q(100, 'mm'), "
        f"got a bare {type(value).__name__}"
    )


@dataclass(frozen=True)
class TransferMatrix:
    """A 2x2 paraxial transfer matrix with the unit of its off-diagonal terms."""

    matrix: np.ndarray
    unit: Unit
    name: str = field(default="", compare=False)

    def __post_init__(self) -> None:
        array = np.asarray(self.matrix, dtype=float)
        if array.shape != (2, 2):
            raise ValueError(f"a transfer matrix must be 2x2, got shape {array.shape}")
        object.__setattr__(self, "matrix", array)

    # -- composition -----------------------------------------------------
    @property
    def carries_length(self) -> bool:
        """Whether this matrix actually contains a length.

        A purely diagonal matrix (identity, magnification, dielectric
        interface) has ``B = C = 0`` and is therefore expressible in any
        length unit; one with a non-zero ``B`` or ``C`` is not.
        """
        return bool(self.matrix[0, 1] != 0.0 or self.matrix[1, 0] != 0.0)

    def __matmul__(self, other: TransferMatrix) -> TransferMatrix:
        """``self @ other`` applies ``other`` first, as in matrix algebra."""
        if not isinstance(other, TransferMatrix):
            return NotImplemented
        if self.carries_length and other.carries_length:
            if self.unit.dimension != other.unit.dimension:
                raise DimensionError(
                    f"cannot compose {other.unit} with {self.unit}: different dimensions"
                )
            unit = self.unit
            other = other.to(unit)
        elif self.carries_length:
            # a unit-free element adopts the unit of the one that has a length
            unit = self.unit
            other = TransferMatrix(other.matrix, unit, other.name)
        elif other.carries_length:
            unit = other.unit
            self = TransferMatrix(self.matrix, unit, self.name)
        else:
            unit = self.unit
        return TransferMatrix(
            self.matrix @ other.matrix,
            unit,
            name=f"{self.name}@{other.name}",
        )

    def __pow__(self, repeats: int) -> TransferMatrix:
        if not isinstance(repeats, int) or repeats < 1:
            raise ValueError("a transfer matrix can only be raised to a positive integer")
        return TransferMatrix(np.linalg.matrix_power(self.matrix, repeats), self.unit, self.name)

    def to(self, target: Unit) -> TransferMatrix:
        """Re-express the matrix in another length unit.

        ``B`` scales with the unit and ``C`` inversely, which is exactly why
        an ABCD matrix without a unit is an under-specified object.
        """
        factor = self.unit.conversion_factor(target)
        ruler = np.array([[1.0, 0.0], [0.0, 1.0 / factor]])
        return TransferMatrix(ruler @ self.matrix @ np.linalg.inv(ruler), target, self.name)

    # -- properties ------------------------------------------------------
    @property
    def determinant(self) -> float:
        """``AD - BC``; exactly one for a system that starts and ends in the same medium."""
        return float(np.linalg.det(self.matrix))

    @property
    def is_identity(self) -> bool:
        return bool(np.allclose(self.matrix, np.eye(2)))

    def __str__(self) -> str:
        rows = "  ".join(f"[{row[0]:12.6g} {row[1]:12.6g}]" for row in self.matrix)
        label = f" {self.name}" if self.name else ""
        return f"TransferMatrix{label} in {self.unit}\n  {rows}"


def free_space(distance: Any) -> TransferMatrix:
    """Propagation over ``distance``: ``[[1, d], [0, 1]]``."""
    value, unit = _length_unit_of(distance, "free_space")
    return TransferMatrix(np.array([[1.0, value], [0.0, 1.0]]), unit, "free_space")


def thin_lens(focal_length: Any) -> TransferMatrix:
    """An ideal thin lens: ``[[1, 0], [-1/f, 1]]``.

    A negative focal length describes a diverging lens.
    """
    value, unit = _length_unit_of(focal_length, "thin_lens")
    if value == 0.0:
        raise ValueError("a thin lens needs a finite focal length")
    return TransferMatrix(np.array([[1.0, 0.0], [-1.0 / value, 1.0]]), unit, "thin_lens")


def curved_mirror(radius_of_curvature: Any) -> TransferMatrix:
    """A spherical mirror: ``[[1, 0], [-2/R, 1]]`` in the reflective convention.

    ``R`` is positive for a concave mirror facing the beam.
    """
    value, unit = _length_unit_of(radius_of_curvature, "curved_mirror")
    if value == 0.0:
        raise ValueError("a curved mirror needs a finite radius of curvature")
    return TransferMatrix(
        np.array([[1.0, 0.0], [-2.0 / value, 1.0]]), unit, "curved_mirror"
    )


def flat_mirror() -> TransferMatrix:
    """A plane mirror: the identity in the paraxial approximation."""
    return TransferMatrix(np.eye(2), _unit("1"), "flat_mirror")


def dielectric_interface(refractive_index_from: Any, refractive_index_to: Any) -> TransferMatrix:
    """Refraction at a plane interface: ``[[1, 0], [0, n1/n2]]``.

    Both indices are dimensionless ratios, so the matrix carries no length
    unit and its determinant is ``n1/n2`` rather than one.
    """
    n_from = (
        float(refractive_index_from.value)
        if isinstance(refractive_index_from, Quantity)
        else float(refractive_index_from)
    )
    n_to = (
        float(refractive_index_to.value)
        if isinstance(refractive_index_to, Quantity)
        else float(refractive_index_to)
    )
    if n_from <= 0.0 or n_to <= 0.0:
        raise ValueError("refractive indices must be positive")
    return TransferMatrix(
        np.array([[1.0, 0.0], [0.0, n_from / n_to]]), _unit("1"), "interface"
    )


def refracting_surface(
    refractive_index_from: Any, refractive_index_to: Any, radius_of_curvature: Any
) -> TransferMatrix:
    """Refraction at a **curved** interface: ``[[1, 0], [-Phi/n2, n1/n2]]``.

    The surface power is ``Phi = (n2 - n1)/R``, with ``R`` positive when the
    centre of curvature lies downstream.  An infinite radius reproduces
    :func:`dielectric_interface`.
    """
    n_from = _index_value(refractive_index_from, "refracting_surface")
    n_to = _index_value(refractive_index_to, "refracting_surface")
    radius, unit = _length_unit_of(radius_of_curvature, "refracting_surface")
    if radius == 0.0:
        raise ValueError("a refracting surface needs a finite radius or infinity")
    power = (n_to - n_from) / radius
    # Snell in the paraxial limit, with the ray angle measured in each medium:
    #   n' theta' = n theta - y Phi   ->   theta' = (n/n') theta - y Phi/n'
    return TransferMatrix(
        np.array([[1.0, 0.0], [-power / n_to, n_from / n_to]]),
        unit,
        "refracting_surface",
    )


def _index_value(value: Any, context: str) -> float:
    if isinstance(value, Quantity):
        if not value.unit.is_dimensionless:
            raise DimensionError(
                f"{context}: a refractive index must be dimensionless, got {value.unit}"
            )
        index = float(value.value)
    else:
        index = float(value)
    if index <= 0.0:
        raise ValueError(f"{context}: a refractive index must be positive, got {index}")
    return index


def compose(*elements: TransferMatrix) -> TransferMatrix:
    """Multiply elements **in beam order**.

    ``compose(space, lens)`` is the system in which the beam crosses the space
    and then meets the lens, i.e. ``lens @ space`` in matrix terms.
    """
    if not elements:
        raise ValueError("compose needs at least one element")
    result = elements[0]
    for element in elements[1:]:
        if not isinstance(element, TransferMatrix):
            raise TypeError(f"compose expects TransferMatrix, got {type(element).__name__}")
        result = element @ result
    return result


def stability(matrix: TransferMatrix) -> float:
    """Half the trace, ``(A + D)/2``.

    A round trip is stable when this lies in ``[-1, 1]``; it is exactly the
    ``g1*g2`` invariant of a two-mirror cavity.
    """
    return float(0.5 * (matrix.matrix[0, 0] + matrix.matrix[1, 1]))


def is_stable(matrix: TransferMatrix, atol: float = 1e-9) -> bool:
    """Whether a round-trip matrix is stable, i.e. ``|(A + D)/2| <= 1``.

    The boundary is inclusive here, which is the usual textbook criterion.
    The two ends of the boundary are not physically equivalent though: for a
    two-mirror cavity ``(A + D)/2 = 2 g1 g2 - 1``, so ``-1`` is the confocal
    end (finite, well-behaved mode) while ``+1`` is the degenerate flat-flat
    end.  Use :func:`stability_status` when that distinction matters.
    """
    return abs(stability(matrix)) <= 1.0 + atol


def is_marginally_stable(matrix: TransferMatrix, atol: float = 1e-9) -> bool:
    """Whether a round trip sits exactly on the stability boundary."""
    return abs(abs(stability(matrix)) - 1.0) <= atol


def stability_status(matrix: TransferMatrix, atol: float = 1e-9) -> str:
    """``"stable"``, ``"marginal"`` or ``"unstable"``, for reporting."""
    value = stability(matrix)
    if abs(value) > 1.0 + atol:
        return "unstable"
    if abs(value) >= 1.0 - atol:
        return "marginal"
    return "stable"


def effective_focal_length(matrix: TransferMatrix) -> Quantity:
    """``-1/C``: where a collimated input comes to a focus.

    Raises for an afocal system, whose ``C`` is zero and which therefore has
    no focal length.
    """
    c = float(matrix.matrix[1, 0])
    if c == 0.0:
        raise ValueError("this system is afocal: C = 0, so it has no focal length")
    return Quantity(-1.0 / c, matrix.unit)


def magnification(matrix: TransferMatrix) -> float:
    """Transverse magnification ``A`` of a system that images (``B = 0``)."""
    if matrix.matrix[0, 1] != 0.0:
        raise ValueError("magnification is only defined for an imaging system with B = 0")
    return float(matrix.matrix[0, 0])


def chief_ray_trace(matrix: TransferMatrix, height: Any, angle: Any) -> tuple[float, float]:
    """Trace ``(y, theta)`` through ``matrix``, with ``theta`` in radians."""
    theta = (
        float(angle.to_value("rad")) if isinstance(angle, Quantity) else float(angle)
    )
    vector = matrix.matrix @ np.array([float(height), theta])
    return float(vector[0]), float(vector[1])


__all__ = [
    "TransferMatrix",
    "chief_ray_trace",
    "compose",
    "curved_mirror",
    "dielectric_interface",
    "effective_focal_length",
    "flat_mirror",
    "free_space",
    "is_marginally_stable",
    "is_stable",
    "magnification",
    "refracting_surface",
    "stability",
    "stability_status",
    "thin_lens",
]
