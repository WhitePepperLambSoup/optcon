"""Mueller calculus: the intensity-domain counterpart of Jones calculus.

A Jones vector describes fully polarised light and loses any depolarisation;
a Stokes vector and a 4x4 Mueller matrix keep it.  Both calculi are provided
because both are needed: coherent multilayer design wants Jones, and anything
involving scattering, rough surfaces or depolarising optics wants Mueller.

The two are tied together by :func:`mueller_from_jones`, built so that it
agrees with :mod:`optcon.polarization` by construction: the matrix is
determined by acting with the Jones operator on four independent probe states
and solving for the linear map between their Stokes vectors.  The test suite
then checks it on *other* states, which is what makes the agreement evidence
rather than tautology.

The Stokes convention is the one used throughout optcon:
``S3 = 2 Im(Ex conj(Ey))``, so a quarter-wave plate at 45 degrees turns
horizontal linear light into ``S3 = +1``.
"""

from __future__ import annotations

import numpy as np

from .errors import DimensionError
from .quantity import Quantity
from .units import unit as _unit

_ANGLE = _unit("rad").dimension


def _angle_of(value, context: str) -> float:
    if isinstance(value, Quantity):
        if value.unit.dimension != _ANGLE:
            raise DimensionError(f"{context}: expected an angle, got {value.unit}")
        return float(value.to_value("rad"))
    raise TypeError(
        f"{context}: expected an angle with units, e.g. q(45, 'deg'), "
        f"got a bare {type(value).__name__}"
    )


def stokes_vector(state) -> np.ndarray:
    """Stokes vector of a Jones state, as a plain 4-vector."""
    if isinstance(state, Quantity):
        state = state.value
    vector = np.asarray(state, dtype=complex).ravel()
    if vector.size != 2:
        raise ValueError(f"a polarisation state needs two components, got {vector.size}")
    ex, ey = complex(vector[0]), complex(vector[1])
    cross = ex * np.conj(ey)
    return np.array(
        [
            abs(ex) ** 2 + abs(ey) ** 2,
            abs(ex) ** 2 - abs(ey) ** 2,
            2.0 * cross.real,
            2.0 * cross.imag,
        ]
    )


def mueller_from_jones(jones) -> np.ndarray:
    """The Mueller matrix of a non-depolarising element described by Jones."""
    if isinstance(jones, Quantity):
        jones = jones.value
    matrix = np.asarray(jones, dtype=complex)
    if matrix.shape != (2, 2):
        raise ValueError(f"a Jones operator must be 2x2, got shape {matrix.shape}")
    probes = [
        np.array([1.0, 0.0]),
        np.array([0.0, 1.0]),
        np.array([1.0, 1.0]) / np.sqrt(2.0),
        np.array([1.0, 1.0j]) / np.sqrt(2.0),
    ]
    inputs = np.column_stack([stokes_vector(probe) for probe in probes])
    outputs = np.column_stack([stokes_vector(matrix @ probe) for probe in probes])
    return outputs @ np.linalg.inv(inputs)


def mueller_polarizer(angle) -> np.ndarray:
    """An ideal linear polariser."""
    from .polarization import polarizer

    return mueller_from_jones(polarizer(angle).value)


def mueller_retarder(retardance, angle=None) -> np.ndarray:
    """An ideal waveplate: lossless, so ``S0`` is untouched."""
    from .polarization import retarder

    return mueller_from_jones(retarder(retardance, angle).value)


def mueller_rotator(angle) -> np.ndarray:
    """A polarisation rotator."""
    from .polarization import rotator

    return mueller_from_jones(rotator(angle).value)


def mueller_depolarizer(polarization_retained: float) -> np.ndarray:
    """Partial depolarisation that keeps ``S0`` and shrinks the rest.

    ``polarization_retained = 1`` changes nothing; ``0`` leaves only
    unpolarized light.
    """
    retained = float(polarization_retained)
    if not 0.0 <= retained <= 1.0:
        raise ValueError(
            f"polarization_retained must lie in [0, 1], got {retained}"
        )
    return np.diag([1.0, retained, retained, retained])


def apply_mueller(matrix, stokes) -> np.ndarray:
    """``S' = M S``."""
    operator = np.asarray(matrix, dtype=float)
    if operator.shape != (4, 4):
        raise ValueError(f"a Mueller matrix must be 4x4, got shape {operator.shape}")
    vector = np.asarray(stokes, dtype=float).ravel()
    if vector.size != 4:
        raise ValueError(f"a Stokes vector needs four components, got {vector.size}")
    return operator @ vector


def degree_of_polarization(stokes) -> float:
    """``sqrt(S1^2 + S2^2 + S3^2) / S0``."""
    vector = np.asarray(stokes, dtype=float).ravel()
    if vector.size != 4:
        raise ValueError(f"a Stokes vector needs four components, got {vector.size}")
    total = vector[0]
    if total <= 0.0:
        return 0.0
    polarized = float(np.sqrt(np.sum(vector[1:] ** 2)))
    return min(1.0, polarized / total)


__all__ = [
    "apply_mueller",
    "degree_of_polarization",
    "mueller_depolarizer",
    "mueller_from_jones",
    "mueller_polarizer",
    "mueller_retarder",
    "mueller_rotator",
    "stokes_vector",
]
